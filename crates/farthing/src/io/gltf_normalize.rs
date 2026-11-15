//! Reading `KHR_mesh_quantization` and `EXT_meshopt_compression` (gltfpack's output, and
//! farthing's own compact exports). The `gltf` crate rejects both, and reads positions,
//! normals and tangents only as floats, so such a document is rewritten into plain glTF first:
//! compressed views decoded, quantized attributes converted to floats, and every view moved
//! into one buffer.

use super::meshopt;
use crate::{Error, Result};
use serde_json::{json, Value};
use std::collections::HashMap;

const QUANTIZATION: &str = "KHR_mesh_quantization";
const MESHOPT: &str = "EXT_meshopt_compression";

fn invalid(what: impl std::fmt::Display) -> Error {
    Error::InvalidInput(format!("glTF: {what}"))
}

fn uint(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|n| n as usize)
}

/// For a document using either extension: the document rewritten without them, and its one
/// buffer. `None` for any other document. `fetch` returns a buffer URI's contents.
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
    if !mentions(QUANTIZATION) && !mentions(MESHOPT) {
        return Ok(None);
    }
    let mut root: Value = serde_json::from_slice(&json_bytes).map_err(invalid)?;
    let uses = |root: &Value, ext: &str| {
        root.get("extensionsUsed")
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(|e| e == ext))
    };
    if !uses(&root, QUANTIZATION) && !uses(&root, MESHOPT) {
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

    // Quantized positions, normals and tangents (and texture coordinates the crate reads
    // wrongly or not at all) become float accessors.
    let mut converted: HashMap<usize, usize> = HashMap::new();
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
        let mut sets: Vec<&mut Value> = Vec::new();
        for (k, v) in prim.as_object_mut().into_iter().flatten() {
            match k.as_str() {
                "attributes" => sets.push(v),
                "targets" => sets.extend(v.as_array_mut().into_iter().flatten()),
                _ => {}
            }
        }
        for set in sets {
            for (name, index) in set.as_object_mut().into_iter().flatten() {
                let float = ["POSITION", "NORMAL", "TANGENT"].contains(&name.as_str())
                    || name.starts_with("TEXCOORD_");
                let Some(a) = index.as_u64().map(|a| a as usize).filter(|_| float) else {
                    continue;
                };
                let new = match converted.get(&a) {
                    Some(&n) => n,
                    None => match to_float(&mut root, &mut out, a, name == "POSITION")? {
                        Some(n) => {
                            converted.insert(a, n);
                            n
                        }
                        None => continue,
                    },
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
    for key in ["extensionsUsed", "extensionsRequired"] {
        if let Some(a) = root.get_mut(key).and_then(Value::as_array_mut) {
            a.retain(|e| e != QUANTIZATION && e != MESHOPT);
        }
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    root["buffers"] = json!([{ "byteLength": out.len() }]);
    let doc = gltf::Document::from_json(serde_json::from_value(root).map_err(invalid)?)
        .map_err(invalid)?;
    Ok(Some((doc, out)))
}

/// Accessor `a` as a new float accessor (its data appended to `out`), or `None` when it already
/// holds floats.
fn to_float(root: &mut Value, out: &mut Vec<u8>, a: usize, minmax: bool) -> Result<Option<usize>> {
    let acc = root["accessors"]
        .get(a)
        .cloned()
        .ok_or_else(|| invalid(format!("no accessor {a}")))?;
    let component = uint(&acc, "componentType").unwrap_or(0);
    if component == 5126 {
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
        5125 => 4,
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
                _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            };
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
