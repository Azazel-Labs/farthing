//! Reading `KHR_mesh_quantization`, `EXT_meshopt_compression` and `KHR_texture_transform`
//! (gltfpack's output, and farthing's own compact exports). The `gltf` crate rejects the
//! first two, reads positions, normals and tangents only as floats, and ignores texture
//! transforms, so such a document is rewritten into plain glTF first: compressed views decoded,
//! quantized attributes converted to floats, texture transforms baked into the texture
//! coordinates, and every view moved into one buffer.
//!
//! A transform is baked when every texture of a material that reads a UV set has the same one
//! and it has no rotation (gltfpack writes only offset and scale). Others are ignored, as
//! before.

use super::meshopt;
use crate::{Error, Result};
use serde_json::{json, Value};
use std::collections::HashMap;

const QUANTIZATION: &str = "KHR_mesh_quantization";
const MESHOPT: &str = "EXT_meshopt_compression";
const TRANSFORM: &str = "KHR_texture_transform";
const HANDLED: [&str; 3] = [QUANTIZATION, MESHOPT, TRANSFORM];

/// A texture transform without rotation: offset x, y, then scale x, y.
type UvTransform = [f32; 4];

fn invalid(what: impl std::fmt::Display) -> Error {
    Error::InvalidInput(format!("glTF: {what}"))
}

fn uint(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|n| n as usize)
}

/// For a document using any of the extensions: the document rewritten without them, and its
/// one buffer. `None` for any other document. `fetch` returns a buffer URI's contents.
pub(crate) fn normalize(
    bytes: &[u8],
    fetch: &mut dyn FnMut(&str) -> Result<Vec<u8>>,
) -> Result<Option<(gltf::Document, Vec<u8>)>> {
    let (json_bytes, blob) = if bytes.starts_with(b"glTF") {
        let glb = gltf::Glb::from_slice(bytes).map_err(invalid)?;
        (glb.json, glb.bin)
    } else {
        (bytes.into(), None)
    };
    let mentions = |ext: &str| json_bytes.windows(ext.len()).any(|w| w == ext.as_bytes());
    if !HANDLED.iter().any(|e| mentions(e)) {
        return Ok(None);
    }
    let mut root: Value = serde_json::from_slice(&json_bytes).map_err(invalid)?;
    let uses = |root: &Value, ext: &str| {
        root.get("extensionsUsed")
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(|e| e == ext))
    };
    if !HANDLED.iter().any(|e| uses(&root, e)) {
        return Ok(None);
    }

    // Source buffers; fallback and placeholder buffers have no contents.
    let mut sources: Vec<Option<Vec<u8>>> = Vec::new();
    for (i, b) in root["buffers"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let fallback =
            b.pointer("/extensions/EXT_meshopt_compression/fallback") == Some(&json!(true));
        sources.push(match b.get("uri").and_then(Value::as_str) {
            _ if fallback => None,
            Some(uri) => Some(fetch(uri)?),
            None if i == 0 => blob.as_deref().map(<[u8]>::to_vec),
            None => None,
        });
    }
    let source = |i: usize, offset: usize, len: usize| -> Result<&[u8]> {
        sources
            .get(i)
            .and_then(Option::as_deref)
            .and_then(|b| b.get(offset..offset + len))
            .ok_or_else(|| {
                invalid(format!(
                    "buffer {i} has no bytes {offset}..{}",
                    offset + len
                ))
            })
    };

    // Every view into one buffer, decoding compressed ones.
    let mut out: Vec<u8> = Vec::new();
    let views = root["bufferViews"].as_array().cloned().unwrap_or_default();
    let mut new_views = Vec::with_capacity(views.len());
    for v in &views {
        let bytes = match v.pointer("/extensions/EXT_meshopt_compression") {
            Some(ext) => {
                let field = |k: &str| {
                    uint(ext, k).ok_or_else(|| invalid(format!("{MESHOPT} view without {k}")))
                };
                let (count, stride) = (field("count")?, field("byteStride")?);
                let src = source(
                    field("buffer")?,
                    uint(ext, "byteOffset").unwrap_or(0),
                    field("byteLength")?,
                )?;
                match ext.get("mode").and_then(Value::as_str) {
                    Some("ATTRIBUTES") => {
                        let mut data = meshopt::decode_vertex_buffer(src, count, stride)?;
                        let filter = ext.get("filter").and_then(Value::as_str).unwrap_or("NONE");
                        meshopt::apply_filter(filter, &mut data, stride)?;
                        data
                    }
                    Some(mode @ ("TRIANGLES" | "INDICES")) => {
                        let idx = if mode == "TRIANGLES" {
                            meshopt::decode_index_buffer(src, count)?
                        } else {
                            meshopt::decode_index_sequence(src, count)?
                        };
                        match stride {
                            2 => idx.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect(),
                            4 => idx.iter().flat_map(|i| i.to_le_bytes()).collect(),
                            _ => return Err(invalid(format!("{MESHOPT} index stride {stride}"))),
                        }
                    }
                    m => return Err(invalid(format!("{MESHOPT} mode {m:?}"))),
                }
            }
            None => {
                let len = uint(v, "byteLength")
                    .ok_or_else(|| invalid("bufferView without byteLength"))?;
                let buffer =
                    uint(v, "buffer").ok_or_else(|| invalid("bufferView without buffer"))?;
                source(buffer, uint(v, "byteOffset").unwrap_or(0), len)?.to_vec()
            }
        };
        out.resize(out.len().div_ceil(4) * 4, 0);
        let mut nv = json!({ "buffer": 0, "byteOffset": out.len(), "byteLength": bytes.len() });
        for k in ["byteStride", "target", "name"] {
            if let Some(x) = v.get(k) {
                nv[k] = x.clone();
            }
        }
        out.extend_from_slice(&bytes);
        new_views.push(nv);
    }
    root["bufferViews"] = Value::Array(new_views);

    // Each material's bakeable texture transforms, by UV set; removed from the material.
    let transforms: Vec<HashMap<u64, UvTransform>> = root
        .get_mut("materials")
        .and_then(Value::as_array_mut)
        .map(|ms| ms.iter_mut().map(take_transforms).collect())
        .unwrap_or_default();

    // Quantized positions, normals and tangents (and texture coordinates the crate reads
    // wrongly or not at all, or that a transform applies to) become float accessors.
    let mut converted: HashMap<(usize, Option<[u32; 4]>), usize> = HashMap::new();
    let mut meshes = root
        .get_mut("meshes")
        .map(Value::take)
        .unwrap_or(Value::Null);
    for prim in meshes.as_array_mut().into_iter().flatten().flat_map(|m| {
        m.get_mut("primitives")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
    }) {
        let material = prim.get("material").and_then(Value::as_u64);
        let uv_transforms = material.and_then(|m| transforms.get(m as usize));
        // Attribute sets, and whether each holds morph deltas (which take no offset).
        let mut sets: Vec<(&mut Value, bool)> = Vec::new();
        for (k, v) in prim.as_object_mut().into_iter().flatten() {
            match k.as_str() {
                "attributes" => sets.push((v, false)),
                "targets" => sets.extend(v.as_array_mut().into_iter().flatten().map(|t| (t, true))),
                _ => {}
            }
        }
        for (set, delta) in sets {
            for (name, index) in set.as_object_mut().into_iter().flatten() {
                let uv_set = name
                    .strip_prefix("TEXCOORD_")
                    .and_then(|n| n.parse::<u64>().ok());
                let float =
                    ["POSITION", "NORMAL", "TANGENT"].contains(&name.as_str()) || uv_set.is_some();
                let Some(a) = index.as_u64().map(|a| a as usize).filter(|_| float) else {
                    continue;
                };
                let transform =
                    uv_set
                        .and_then(|k| uv_transforms?.get(&k).copied())
                        .map(|[ox, oy, sx, sy]| {
                            if delta {
                                [0.0, 0.0, sx, sy]
                            } else {
                                [ox, oy, sx, sy]
                            }
                        });
                let key = (a, transform.map(|t| t.map(f32::to_bits)));
                let new = match converted.get(&key) {
                    Some(&n) => n,
                    None => {
                        match to_float(&mut root, &mut out, a, name == "POSITION", transform)? {
                            Some(n) => {
                                converted.insert(key, n);
                                n
                            }
                            None => continue,
                        }
                    }
                };
                *index = json!(new);
            }
        }
    }
    if !meshes.is_null() {
        root["meshes"] = meshes;
    }

    for v in root["bufferViews"].as_array_mut().into_iter().flatten() {
        if let Some(e) = v.get_mut("extensions").and_then(Value::as_object_mut) {
            e.remove(MESHOPT);
        }
    }
    // Transforms that could not be baked are ignored, as the crate ignores them elsewhere.
    let transforms_left = root
        .get("materials")
        .is_some_and(|m| m.to_string().contains(TRANSFORM));
    for key in ["extensionsUsed", "extensionsRequired"] {
        let keep_transform = key == "extensionsUsed" && transforms_left;
        if let Some(a) = root.get_mut(key).and_then(Value::as_array_mut) {
            a.retain(|e| e != QUANTIZATION && e != MESHOPT && (keep_transform || e != TRANSFORM));
        }
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    root["buffers"] = json!([{ "byteLength": out.len() }]);
    let doc = gltf::Document::from_json(serde_json::from_value(root).map_err(invalid)?)
        .map_err(invalid)?;
    Ok(Some((doc, out)))
}

/// The texture transforms of `material` that can be baked into its UV sets, removed from its
/// textures. A UV set qualifies when every texture reading it has the same transform, without
/// rotation.
fn take_transforms(material: &mut Value) -> HashMap<u64, UvTransform> {
    const SLOTS: [&str; 5] = [
        "/pbrMetallicRoughness/baseColorTexture",
        "/pbrMetallicRoughness/metallicRoughnessTexture",
        "/normalTexture",
        "/occlusionTexture",
        "/emissiveTexture",
    ];
    // Per UV set: its transform, or `None` once textures disagree or one rotates.
    let mut by_set: HashMap<u64, Option<UvTransform>> = HashMap::new();
    for slot in SLOTS {
        let Some(info) = material.pointer(slot) else {
            continue;
        };
        let ext = info.pointer("/extensions/KHR_texture_transform");
        let set = ext
            .and_then(|e| e.get("texCoord"))
            .or_else(|| info.get("texCoord"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let pair = |k: &str, default: f32| -> [f32; 2] {
            let v = ext.and_then(|e| e.get(k));
            let c = |i: usize| {
                v.and_then(|v| v.get(i)?.as_f64())
                    .map_or(default, |x| x as f32)
            };
            [c(0), c(1)]
        };
        let rotation = ext.and_then(|e| e.get("rotation")?.as_f64()).unwrap_or(0.0);
        let [ox, oy] = pair("offset", 0.0);
        let [sx, sy] = pair("scale", 1.0);
        let t = (rotation == 0.0).then_some([ox, oy, sx, sy]);
        by_set
            .entry(set)
            .and_modify(|e| {
                if *e != t {
                    *e = None;
                }
            })
            .or_insert(t);
    }
    let bake: HashMap<u64, UvTransform> = by_set
        .into_iter()
        .filter_map(|(set, t)| Some((set, t?)))
        .filter(|(_, t)| *t != [0.0, 0.0, 1.0, 1.0])
        .collect();
    for slot in SLOTS {
        let Some(info) = material.pointer_mut(slot) else {
            continue;
        };
        let ext = info.pointer("/extensions/KHR_texture_transform");
        let set = ext
            .and_then(|e| e.get("texCoord"))
            .or_else(|| info.get("texCoord"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if ext.is_some() && bake.contains_key(&set) {
            info["texCoord"] = json!(set);
            if let Some(e) = info.get_mut("extensions").and_then(Value::as_object_mut) {
                e.remove(TRANSFORM);
            }
        }
    }
    bake
}

/// Accessor `a` as a new float accessor (its data appended to `out`), with `transform` (offset,
/// scale) applied to 2D values; `None` when it already holds floats and needs no transform.
fn to_float(
    root: &mut Value,
    out: &mut Vec<u8>,
    a: usize,
    minmax: bool,
    transform: Option<UvTransform>,
) -> Result<Option<usize>> {
    let acc = root["accessors"]
        .get(a)
        .cloned()
        .ok_or_else(|| invalid(format!("no accessor {a}")))?;
    let component = uint(&acc, "componentType").unwrap_or(0);
    if component == 5126 && transform.is_none() {
        return Ok(None);
    }
    if acc.get("sparse").is_some() {
        return Err(invalid("sparse quantized accessors are not supported"));
    }
    let comps = match acc.get("type").and_then(Value::as_str) {
        Some("SCALAR") => 1,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        t => return Err(invalid(format!("accessor type {t:?}"))),
    };
    let size = match component {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        c => return Err(invalid(format!("component type {c}"))),
    };
    let count = uint(&acc, "count").unwrap_or(0);
    let normalized = acc.get("normalized") == Some(&json!(true));
    let mut floats = vec![0f32; count * comps];
    if let Some(view) = uint(&acc, "bufferView") {
        let v = &root["bufferViews"][view];
        let stride = uint(v, "byteStride").unwrap_or(comps * size);
        let base = uint(v, "byteOffset").unwrap_or(0) + uint(&acc, "byteOffset").unwrap_or(0);
        let end = uint(v, "byteOffset").unwrap_or(0) + uint(v, "byteLength").unwrap_or(0);
        if count > 0 && base + (count - 1) * stride + comps * size > end.min(out.len()) {
            return Err(invalid(format!("accessor {a} overruns its view")));
        }
        for (i, f) in floats.iter_mut().enumerate() {
            let at = base + (i / comps) * stride + (i % comps) * size;
            let b = &out[at..at + size];
            *f = match (component, normalized) {
                (5120, false) => b[0] as i8 as f32,
                (5120, true) => (b[0] as i8 as f32 / 127.0).max(-1.0),
                (5121, false) => b[0] as f32,
                (5121, true) => b[0] as f32 / 255.0,
                (5122, false) => i16::from_le_bytes([b[0], b[1]]) as f32,
                (5122, true) => (i16::from_le_bytes([b[0], b[1]]) as f32 / 32767.0).max(-1.0),
                (5123, false) => u16::from_le_bytes([b[0], b[1]]) as f32,
                (5123, true) => u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0,
                (5126, _) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            };
        }
    }
    if let (Some([ox, oy, sx, sy]), 2) = (transform, comps) {
        for uv in floats.chunks_exact_mut(2) {
            uv[0] = uv[0] * sx + ox;
            uv[1] = uv[1] * sy + oy;
        }
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    let views = root["bufferViews"]
        .as_array_mut()
        .ok_or_else(|| invalid("no bufferViews"))?;
    views.push(json!({ "buffer": 0, "byteOffset": out.len(), "byteLength": floats.len() * 4, "target": 34962 }));
    out.extend(floats.iter().flat_map(|f| f.to_le_bytes()));
    let mut new = json!({ "bufferView": views.len() - 1, "componentType": 5126, "count": count, "type": acc["type"] });
    if minmax && count > 0 {
        let mut min = vec![f32::INFINITY; comps];
        let mut max = vec![f32::NEG_INFINITY; comps];
        for e in floats.chunks_exact(comps) {
            for c in 0..comps {
                min[c] = min[c].min(e[c]);
                max[c] = max[c].max(e[c]);
            }
        }
        new["min"] = json!(min);
        new["max"] = json!(max);
    }
    let accessors = root["accessors"]
        .as_array_mut()
        .ok_or_else(|| invalid("no accessors"))?;
    accessors.push(new);
    Ok(Some(accessors.len() - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quad drawn twice: with a material whose transforms bake (u16 normalized UVs on set 0,
    /// float UVs on set 1), and with one whose rotated transform cannot.
    #[test]
    fn texture_transforms_bake_into_uvs() {
        let uv = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut bin = Vec::new();
        for [u, v] in uv {
            bin.extend([u, v, 0.0f32].iter().flat_map(|f| f.to_le_bytes()));
        }
        for [u, v] in uv {
            bin.extend(
                [u, v]
                    .map(|f| (f * 65535.0) as u16)
                    .iter()
                    .flat_map(|q| q.to_le_bytes()),
            );
        }
        for [u, v] in uv {
            bin.extend([u, v].iter().flat_map(|f| f.to_le_bytes()));
        }
        let prim = |material: u32| json!({ "attributes": { "POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2 }, "material": material });
        let transform = json!({ "offset": [0.5, 0.25], "scale": [2.0, 4.0] });
        let doc = json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": [TRANSFORM, QUANTIZATION],
            "extensionsRequired": [TRANSFORM, QUANTIZATION],
            "buffers": [{ "uri": "b.bin", "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 48 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 16 },
                { "buffer": 0, "byteOffset": 64, "byteLength": 32 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, 1, 0] },
                { "bufferView": 1, "componentType": 5123, "normalized": true, "count": 4, "type": "VEC2" },
                { "bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2" }
            ],
            "images": [{ "uri": "t.png" }],
            "textures": [{ "source": 0 }],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorTexture": { "index": 0, "extensions": { TRANSFORM: transform } }
                    },
                    "normalTexture": { "index": 0, "extensions": { TRANSFORM: transform } },
                    "emissiveTexture": {
                        "index": 0,
                        "extensions": { TRANSFORM: { "offset": [1.0, 0.0], "texCoord": 1 } }
                    }
                },
                {
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {
                            "index": 0,
                            "extensions": { TRANSFORM: { "offset": [0.5, 0.0], "rotation": 0.3 } }
                        }
                    }
                }
            ],
            "meshes": [{ "primitives": [prim(0), prim(1)] }]
        });
        let bytes = serde_json::to_vec(&doc).unwrap();
        let (doc, buf) = normalize(&bytes, &mut |uri| {
            assert_eq!(uri, "b.bin");
            Ok(bin.clone())
        })
        .unwrap()
        .expect("the document uses the extensions");

        // The rotated transform is left (and ignored); nothing requires the extension.
        assert_eq!(doc.extensions_used().collect::<Vec<_>>(), [TRANSFORM]);
        assert_eq!(doc.extensions_required().count(), 0);

        let mesh = doc.meshes().next().unwrap();
        let read = |p: &gltf::Primitive, set: u32| -> Vec<[f32; 2]> {
            let r = p.reader(|_| Some(&buf[..]));
            r.read_tex_coords(set).unwrap().into_f32().collect()
        };
        let near = |a: Vec<[f32; 2]>, b: Vec<[f32; 2]>| {
            let ok = a
                .iter()
                .zip(&b)
                .all(|(p, q)| (p[0] - q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4);
            assert!(ok, "{a:?} vs {b:?}");
        };
        let prims: Vec<_> = mesh.primitives().collect();
        near(
            read(&prims[0], 0),
            uv.map(|[u, v]| [u * 2.0 + 0.5, v * 4.0 + 0.25]).to_vec(),
        );
        near(read(&prims[0], 1), uv.map(|[u, v]| [u + 1.0, v]).to_vec());
        near(read(&prims[1], 0), uv.to_vec());
        near(read(&prims[1], 1), uv.to_vec());

        let m = doc.materials().next().unwrap();
        assert_eq!(m.emissive_texture().unwrap().tex_coord(), 1);
        assert_eq!(
            m.pbr_metallic_roughness()
                .base_color_texture()
                .unwrap()
                .tex_coord(),
            0
        );
    }
}
