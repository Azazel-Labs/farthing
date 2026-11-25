use crate::anim::{Animation, Curve, Interpolation, Track};
use crate::host::{MaterialClass, MaterialInfo};
use crate::image::Texture;
use crate::{
    Error, Instance, Joint, MaterialId, MaterialSlot, Mesh, MorphTarget, Result, Scene, Skeleton,
    Skin, Transform,
};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

/// The glTF description of a material — for hosts that want to build their own material
/// from it. farthing itself only uses the slot name.
#[derive(Clone, Debug)]
pub struct GltfMaterial {
    pub name: String,
    pub info: MaterialInfo,
    pub base_color_factor: Vec4,
    /// Base color texture (sRGB; [`Texture::sample_bilinear`] returns linear RGB) and its UV
    /// set. Textures used by several materials share their texels.
    pub base_color_texture: Option<(Texture, u32)>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    /// Metallic-roughness texture as stored (linear data: G = roughness, B = metalness) and
    /// its UV set.
    pub metallic_roughness_texture: Option<(Texture, u32)>,
    /// Tangent-space normal map as stored (linear data, xyz in 0..1), its UV set and scale.
    pub normal_texture: Option<(Texture, u32, f32)>,
}

/// Loads a glTF / GLB file. Each glTF mesh becomes one [`Mesh`] (primitives merged, material
/// ids = slot index); each node with a mesh becomes an [`Instance`] with its world transform.
///
/// Each skin becomes a [`Skeleton`]: its joints' rest transforms come from their nodes, and
/// the animation channels that target its joints become its [`Skeleton::animations`] (one
/// [`Animation`] per glTF animation that moves any of them). Nodes between two joints that
/// aren't joints themselves are folded into the child joint, curves included (exactly, when
/// they don't scale unevenly). Channels on other nodes and morph target weights are not read.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<(Scene, Vec<GltfMaterial>)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let dir = path.parent().unwrap_or(Path::new(""));
    load_gltf_with(&bytes, |uri| std::fs::read(dir.join(uri)).ok())
}

fn transform_is_identity(t: &Transform) -> bool {
    t.translation.abs_diff_eq(Vec3::ZERO, 1e-6)
        && t.rotation.abs_diff_eq(Quat::IDENTITY, 1e-6)
        && t.scale.abs_diff_eq(Vec3::ONE, 1e-6)
}

/// Loads a GLB, or a `.gltf` whose buffers and images are embedded (`data:` URIs), from memory.
/// For `.gltf` files that reference external files, use [`load_gltf_with`].
pub fn load_gltf_slice(bytes: &[u8]) -> Result<(Scene, Vec<GltfMaterial>)> {
    load_gltf_with(bytes, |_| None)
}

/// Loads a GLB or `.gltf` from memory; `resolve` returns the contents of each external file
/// the document references, by its (percent-decoded) relative URI — e.g. from a zip archive or
/// files dropped together in a browser. Works without a file system (WebAssembly).
/// PNG and JPEG images are supported.
pub fn load_gltf_with(
    bytes: &[u8],
    mut resolve: impl FnMut(&str) -> Option<Vec<u8>>,
) -> Result<(Scene, Vec<GltfMaterial>)> {
    let mut fetch = |uri: &str| -> Result<Vec<u8>> {
        if let Some(rest) = uri.strip_prefix("data:") {
            let (_, b64) = rest.split_once(";base64,").ok_or_else(|| {
                Error::InvalidInput("glTF: only base64 data URIs are supported".into())
            })?;
            return base64_decode(b64)
                .ok_or_else(|| Error::InvalidInput("glTF: invalid base64 data URI".into()));
        }
        let name = percent_decode(uri);
        resolve(&name).ok_or_else(|| Error::InvalidInput(format!("glTF: missing file `{name}`")))
    };
    // Compressed or quantized documents are rewritten into plain glTF (a GLB) first.
    let plain = decode_encodings(bytes, &mut fetch)?;
    let (json, blob) = split_glb(plain.as_deref().unwrap_or(bytes))?;
    let (document, uv_transforms) = parse_baking_transforms(&json)?;
    let mut buffers = Vec::new();
    for b in document.buffers() {
        let mut data = match b.source() {
            gltf::buffer::Source::Bin => blob
                .as_deref()
                .map(<[u8]>::to_vec)
                .ok_or_else(|| Error::InvalidInput("glTF: missing GLB binary chunk".into()))?,
            gltf::buffer::Source::Uri(uri) => fetch(uri)?,
        };
        if data.len() < b.length() {
            return Err(Error::InvalidInput(format!(
                "glTF: buffer {} is shorter than declared",
                b.index()
            )));
        }
        while data.len() % 4 != 0 {
            data.push(0);
        }
        buffers.push(gltf::buffer::Data(data));
    }
    // Images are decoded on demand (only the material textures farthing reads). Embedded
    // ones are borrowed from their buffer, not copied.
    let mut external: Vec<Option<Vec<u8>>> = Vec::new();
    for img in document.images() {
        external.push(match img.source() {
            gltf::image::Source::View { .. } => None,
            gltf::image::Source::Uri { uri, .. } => fetch(uri).ok(),
        });
    }
    let encoded = |i: usize| -> Option<&[u8]> {
        match document.images().nth(i)?.source() {
            gltf::image::Source::View { view, .. } => {
                buffers[view.buffer().index()].get(view.offset()..view.offset() + view.length())
            }
            gltf::image::Source::Uri { .. } => external.get(i)?.as_deref(),
        }
    };
    build(&document, &buffers, &uv_transforms, &|i, srgb| {
        let rgba = image::load_from_memory(encoded(i)?).ok()?.into_rgba8();
        Texture::from_rgba8(rgba.width(), rgba.height(), rgba.as_raw(), srgb).ok()
    })
}

fn invalid(what: impl std::fmt::Display) -> Error {
    Error::InvalidInput(format!("glTF: {what}"))
}

/// The extensions the `gltf` crate rejects (compression, and attributes stored as integers),
/// which polysmelt decodes into plain glTF.
const ENCODINGS: [&str; 4] = [
    "EXT_meshopt_compression",
    "KHR_meshopt_compression",
    "KHR_draco_mesh_compression",
    "KHR_mesh_quantization",
];
const TRANSFORM: &str = "KHR_texture_transform";

/// A texture transform without rotation: offset x, y, then scale x, y.
type UvTransform = [f32; 4];

/// A GLB's JSON and binary chunks, or a `.gltf`'s JSON.
fn split_glb(bytes: &[u8]) -> Result<(Cow<'_, [u8]>, Option<Cow<'_, [u8]>>)> {
    if bytes.starts_with(b"glTF") {
        let glb = gltf::Glb::from_slice(bytes).map_err(invalid)?;
        Ok((glb.json, glb.bin))
    } else {
        Ok((Cow::Borrowed(bytes), None))
    }
}

fn extensions_used(root: &Value) -> impl Iterator<Item = &str> {
    root.get("extensionsUsed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
}

/// For a document using meshopt, Draco or quantization: the document rewritten by polysmelt as
/// a plain GLB, its buffers and images taken in. `None` for any other document.
fn decode_encodings(
    bytes: &[u8],
    fetch: &mut dyn FnMut(&str) -> Result<Vec<u8>>,
) -> Result<Option<Vec<u8>>> {
    let (json, _) = split_glb(bytes)?;
    let mentions = |ext: &str| json.windows(ext.len()).any(|w| w == ext.as_bytes());
    if !ENCODINGS.iter().any(|e| mentions(e)) {
        return Ok(None);
    }
    let root: Value = serde_json::from_slice(&json).map_err(invalid)?;
    if !extensions_used(&root).any(|e| ENCODINGS.contains(&e)) {
        return Ok(None);
    }
    // polysmelt reads the files the document names from memory, by percent-decoded name.
    let mut files = polysmelt::assets::Memory::new();
    for key in ["buffers", "images"] {
        for item in root
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(uri) = item.get("uri").and_then(Value::as_str)
                && !uri.starts_with("data:")
                && let Ok(file) = fetch(uri)
            {
                files.insert(&percent_decode(uri), file);
            }
        }
    }
    let name = if bytes.starts_with(b"glTF") {
        "model.glb"
    } else {
        "model.gltf"
    };
    // Everything farthing reads: the meshes and what shapes them, and the skins' animations.
    let keep = polysmelt::document::Keep::MESHES.with_animations(true);
    let mut doc = polysmelt::Loader::new()
        .keep(keep)
        .read_with(bytes.to_vec(), name, &files)
        .map_err(invalid)?;
    doc.decode().map_err(invalid)?;
    Ok(Some(doc.to_glb().map_err(invalid)?))
}

/// The document, and per material the texture transforms baked into its UV sets (see
/// [`take_transforms`]). The `gltf` crate ignores texture transforms, which gltfpack writes
/// for quantized texture coordinates.
fn parse_baking_transforms(
    json: &[u8],
) -> Result<(gltf::Document, Vec<HashMap<u32, UvTransform>>)> {
    if !json
        .windows(TRANSFORM.len())
        .any(|w| w == TRANSFORM.as_bytes())
    {
        let root = gltf::json::Root::from_slice(json).map_err(invalid)?;
        return Ok((
            gltf::Document::from_json(root).map_err(invalid)?,
            Vec::new(),
        ));
    }
    let mut root: Value = serde_json::from_slice(json).map_err(invalid)?;
    let bake = root
        .get_mut("materials")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .map(take_transforms)
        .collect();
    // Transforms that could not be baked are ignored, as the crate ignores them elsewhere.
    let left = root
        .get("materials")
        .is_some_and(|m| m.to_string().contains(TRANSFORM));
    for key in ["extensionsUsed", "extensionsRequired"] {
        let keep = key == "extensionsUsed" && left;
        if let Some(a) = root.get_mut(key).and_then(Value::as_array_mut) {
            a.retain(|e| keep || e != TRANSFORM);
        }
    }
    let root = serde_json::from_value(root).map_err(invalid)?;
    Ok((gltf::Document::from_json(root).map_err(invalid)?, bake))
}

/// The texture transforms of `material` that can be baked into its UV sets, removed from its
/// textures. A UV set qualifies when every texture reading it has the same transform, without
/// rotation (gltfpack writes only offset and scale).
fn take_transforms(material: &mut Value) -> HashMap<u32, UvTransform> {
    const SLOTS: [&str; 5] = [
        "/pbrMetallicRoughness/baseColorTexture",
        "/pbrMetallicRoughness/metallicRoughnessTexture",
        "/normalTexture",
        "/occlusionTexture",
        "/emissiveTexture",
    ];
    let set_of = |info: &Value| {
        info.pointer("/extensions/KHR_texture_transform/texCoord")
            .or_else(|| info.get("texCoord"))
            .and_then(Value::as_u64)
            .map_or(0, |s| s as u32)
    };
    // Per UV set: its transform, or `None` once textures disagree or one rotates.
    let mut by_set: HashMap<u32, Option<UvTransform>> = HashMap::new();
    for slot in SLOTS {
        let Some(info) = material.pointer(slot) else {
            continue;
        };
        let ext = info.pointer("/extensions/KHR_texture_transform");
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
            .entry(set_of(info))
            .and_modify(|e| {
                if *e != t {
                    *e = None;
                }
            })
            .or_insert(t);
    }
    let bake: HashMap<u32, UvTransform> = by_set
        .into_iter()
        .filter_map(|(set, t)| Some((set, t?)))
        .filter(|(_, t)| *t != [0.0, 0.0, 1.0, 1.0])
        .collect();
    for slot in SLOTS {
        let Some(info) = material.pointer_mut(slot) else {
            continue;
        };
        let set = set_of(info);
        if info.pointer("/extensions/KHR_texture_transform").is_some() && bake.contains_key(&set) {
            info["texCoord"] = json!(set);
            if let Some(e) = info.get_mut("extensions").and_then(Value::as_object_mut) {
                e.remove(TRANSFORM);
            }
        }
    }
    bake
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2]))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let digits: Vec<u8> = s
        .bytes()
        .filter(|c| !c.is_ascii_whitespace() && *c != b'=')
        .collect();
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut acc = 0u32;
        for (k, &c) in chunk.iter().enumerate() {
            acc |= val(c)? << (18 - 6 * k);
        }
        let n = match chunk.len() {
            4 => 3,
            3 => 2,
            2 => 1,
            _ => return None,
        };
        out.extend_from_slice(&acc.to_be_bytes()[1..1 + n]);
    }
    Some(out)
}

fn build(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    // Per material, the texture transforms to bake into its UV sets.
    uv_transforms: &[HashMap<u32, UvTransform>],
    // Image index, sRGB → decoded texture.
    decode: &dyn Fn(usize, bool) -> Option<Texture>,
) -> Result<(Scene, Vec<GltfMaterial>)> {
    // Each image decodes once, however many materials use it.
    let cache = std::cell::RefCell::new(HashMap::<(usize, bool), Option<Texture>>::new());
    let image = |i: usize, srgb: bool| -> Option<Texture> {
        if let Some(t) = cache.borrow().get(&(i, srgb)) {
            return t.clone();
        }
        let t = decode(i, srgb);
        cache.borrow_mut().insert((i, srgb), t.clone());
        t
    };
    // Material slots: one per glTF material, plus a default slot for primitives without one.
    let mut materials: Vec<MaterialSlot> = doc
        .materials()
        .enumerate()
        .map(|(i, m)| MaterialSlot {
            name: m
                .name()
                .map_or_else(|| format!("material_{i}"), str::to_owned),
            id: MaterialId(i as u32),
        })
        .collect();
    let mut gltf_materials: Vec<GltfMaterial> = doc
        .materials()
        .enumerate()
        .map(|(i, m)| {
            let class = match m.alpha_mode() {
                gltf::material::AlphaMode::Opaque => MaterialClass::Opaque,
                gltf::material::AlphaMode::Mask => MaterialClass::Masked {
                    cutoff: m.alpha_cutoff().unwrap_or(0.5),
                },
                gltf::material::AlphaMode::Blend => MaterialClass::Blended,
            };
            let pbr = m.pbr_metallic_roughness();
            let tex = pbr
                .base_color_texture()
                .and_then(|t| Some((image(t.texture().source().index(), true)?, t.tex_coord())));
            let mr = pbr
                .metallic_roughness_texture()
                .and_then(|t| Some((image(t.texture().source().index(), false)?, t.tex_coord())));
            let normal = m.normal_texture().and_then(|t| {
                Some((
                    image(t.texture().source().index(), false)?,
                    t.tex_coord(),
                    t.scale(),
                ))
            });
            GltfMaterial {
                name: materials[i].name.clone(),
                info: MaterialInfo {
                    class,
                    two_sided: m.double_sided(),
                },
                base_color_factor: Vec4::from_array(pbr.base_color_factor()),
                base_color_texture: tex,
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                metallic_roughness_texture: mr,
                normal_texture: normal,
            }
        })
        .collect();
    let default_slot = materials.len() as u32;
    let mut needs_default = false;

    // Meshes.
    let mut meshes = Vec::new();
    for gm in doc.meshes() {
        let mut mesh = Mesh::default();
        for prim in gm.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(pos) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<Vec3> = pos.map(Vec3::from_array).collect();
            let n = positions.len();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..n as u32).collect(),
            };
            let slot = prim.material().index().map_or_else(
                || {
                    needs_default = true;
                    default_slot
                },
                |i| i as u32,
            );
            let mut part = Mesh::new(positions, indices);
            part.material_ids.iter_mut().for_each(|m| *m = slot);
            part.normals = reader
                .read_normals()
                .map(|it| it.map(Vec3::from_array).collect());
            part.tangents = reader
                .read_tangents()
                .map(|it| it.map(Vec4::from_array).collect());
            for set in 0..2 {
                let Some(tc) = reader.read_tex_coords(set) else {
                    break;
                };
                let mut uvs: Vec<Vec2> = tc.into_f32().map(Vec2::from_array).collect();
                let transform = prim
                    .material()
                    .index()
                    .and_then(|m| uv_transforms.get(m)?.get(&set));
                if let Some(&[ox, oy, sx, sy]) = transform {
                    uvs.iter_mut()
                        .for_each(|uv| *uv = *uv * Vec2::new(sx, sy) + Vec2::new(ox, oy));
                }
                part.uvs.push(uvs);
            }
            part.colors = reader
                .read_colors(0)
                .map(|c| c.into_rgba_f32().map(Vec4::from_array).collect());
            if let (Some(j), Some(w)) = (reader.read_joints(0), reader.read_weights(0)) {
                part.skin = Some(Skin {
                    joints: j.into_u16().collect(),
                    weights: w.into_f32().collect(),
                });
            }
            let names: Vec<String> = gm
                .extras()
                .as_ref()
                .and_then(|e| serde_json::from_str::<serde_json::Value>(e.get()).ok())
                .and_then(|v| v.get("targetNames").cloned())
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();
            for (ti, (p, nrm, _)) in reader.read_morph_targets().enumerate() {
                let deltas: Vec<Vec3> = p.map_or_else(
                    || vec![Vec3::ZERO; n],
                    |it| it.map(Vec3::from_array).collect(),
                );
                part.morph_targets.push(MorphTarget {
                    name: names
                        .get(ti)
                        .cloned()
                        .unwrap_or_else(|| format!("target_{ti}")),
                    position_deltas: deltas,
                    normal_deltas: nrm.map(|it| it.map(Vec3::from_array).collect()),
                });
            }
            part.validate()?;
            mesh.append(&part, 0);
        }
        meshes.push(mesh);
    }
    if needs_default {
        materials.push(MaterialSlot {
            name: "default".into(),
            id: MaterialId(default_slot),
        });
        gltf_materials.push(GltfMaterial {
            name: "default".into(),
            info: MaterialInfo::default(),
            base_color_factor: Vec4::ONE,
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
        });
    }

    // Node hierarchy → world transforms.
    let mut parent: HashMap<usize, usize> = HashMap::new();
    for node in doc.nodes() {
        for c in node.children() {
            parent.insert(c.index(), node.index());
        }
    }
    let local: Vec<Mat4> = doc
        .nodes()
        .map(|n| Mat4::from_cols_array_2d(&n.transform().matrix()))
        .collect();
    let world = |mut i: usize| {
        let mut m = local[i];
        let mut guard = 0;
        while let Some(&p) = parent.get(&i) {
            m = local[p] * m;
            i = p;
            guard += 1;
            if guard > 1024 {
                break;
            }
        }
        m
    };

    // Skins → skeletons.
    let mut skeletons = Vec::new();
    for skin in doc.skins() {
        let joint_nodes: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        let ibms: Vec<Mat4> = skin
            .reader(|b| Some(&buffers[b.index()]))
            .read_inverse_bind_matrices()
            .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_nodes.len()]);
        let node_to_joint: HashMap<usize, usize> = joint_nodes
            .iter()
            .enumerate()
            .map(|(j, &n)| (n, j))
            .collect();
        // Per joint: what lies between it and its parent joint (the skeleton's space, for a
        // root), when that's more than the node's own parent link.
        let mut offsets: Vec<Option<Transform>> = Vec::with_capacity(joint_nodes.len());
        let joints = joint_nodes
            .iter()
            .enumerate()
            .map(|(j, &n)| {
                // Nearest ancestor that is also a joint of this skin.
                let mut p = parent.get(&n).copied();
                while let Some(pn) = p {
                    if node_to_joint.contains_key(&pn) {
                        break;
                    }
                    p = parent.get(&pn).copied();
                }
                let direct = parent.get(&n).copied();
                let offset = match (p, direct) {
                    (Some(pj), Some(d)) if pj == d => None,
                    (_, None) => None,
                    (Some(pj), Some(d)) => Some(world(pj).inverse() * world(d)),
                    (None, Some(d)) => Some(world(d)),
                }
                .map(Transform::from_matrix)
                .filter(|o| !transform_is_identity(o));
                offsets.push(offset);
                let node = doc.nodes().nth(n);
                let rest = node.as_ref().map_or(Transform::IDENTITY, |nd| {
                    let (t, r, s) = nd.transform().decomposed();
                    let own = Transform {
                        translation: Vec3::from(t),
                        rotation: Quat::from_array(r).normalize(),
                        scale: Vec3::from(s),
                    };
                    match offset {
                        Some(o) => Transform::from_matrix(o.matrix() * own.matrix()),
                        None => own,
                    }
                });
                Joint {
                    name: node
                        .and_then(|nd| nd.name().map(str::to_owned))
                        .unwrap_or_else(|| format!("joint_{j}")),
                    parent: p.and_then(|pn| node_to_joint.get(&pn).copied()),
                    inverse_bind: ibms.get(j).copied().unwrap_or(Mat4::IDENTITY),
                    rest,
                }
            })
            .collect();
        let animations = doc
            .animations()
            .enumerate()
            .filter_map(|(ai, anim)| {
                let mut tracks: Vec<Track> = Vec::new();
                for channel in anim.channels() {
                    let Some(&j) = node_to_joint.get(&channel.target().node().index()) else {
                        continue;
                    };
                    let reader = channel.reader(|b| Some(&buffers[b.index()]));
                    let Some(times) = reader.read_inputs().map(|t| t.collect::<Vec<f32>>()) else {
                        continue;
                    };
                    let interpolation = match channel.sampler().interpolation() {
                        gltf::animation::Interpolation::Step => Interpolation::Step,
                        gltf::animation::Interpolation::Linear => Interpolation::Linear,
                        gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                    };
                    let k = match tracks.iter().position(|t| t.joint == j) {
                        Some(k) => k,
                        None => {
                            tracks.push(Track::new(j));
                            tracks.len() - 1
                        }
                    };
                    let curve = |values| Curve {
                        interpolation,
                        times: times.clone(),
                        values,
                    };
                    use gltf::animation::util::ReadOutputs;
                    match reader.read_outputs() {
                        Some(ReadOutputs::Translations(v)) => {
                            tracks[k].translation = Some(curve(v.map(Vec3::from).collect()))
                        }
                        Some(ReadOutputs::Rotations(v)) => {
                            let values = v.into_f32().map(Quat::from_array).collect();
                            tracks[k].rotation = Some(Curve {
                                interpolation,
                                times: times.clone(),
                                values,
                            })
                        }
                        Some(ReadOutputs::Scales(v)) => {
                            tracks[k].scale = Some(curve(v.map(Vec3::from).collect()))
                        }
                        _ => {}
                    }
                }
                // Drop malformed curves rather than the whole clip.
                for t in &mut tracks {
                    t.translation = t.translation.take().filter(Curve::is_valid);
                    t.rotation = t.rotation.take().filter(Curve::is_valid);
                    t.scale = t.scale.take().filter(Curve::is_valid);
                    if let Some(o) = offsets[t.joint] {
                        t.premultiply(o);
                    }
                }
                tracks.retain(|t| {
                    t.translation.is_some() || t.rotation.is_some() || t.scale.is_some()
                });
                tracks.sort_by_key(|t| t.joint);
                (!tracks.is_empty()).then(|| Animation {
                    name: anim
                        .name()
                        .map_or_else(|| format!("animation_{ai}"), str::to_owned),
                    tracks,
                })
            })
            .collect();
        skeletons.push(Skeleton {
            name: skin.name().unwrap_or("skin").to_owned(),
            joints,
            animations,
        });
    }

    // Instances: only nodes reachable from the default scene (or all scenes).
    let mut instances = Vec::new();
    let roots: Vec<gltf::Node> = match doc.default_scene().or_else(|| doc.scenes().next()) {
        Some(s) => s.nodes().collect(),
        None => doc
            .nodes()
            .filter(|n| !parent.contains_key(&n.index()))
            .collect(),
    };
    let mut stack = roots;
    while let Some(node) = stack.pop() {
        if let Some(m) = node.mesh() {
            instances.push(Instance {
                name: node.name().unwrap_or("node").to_owned(),
                mesh: m.index(),
                transform: world(node.index()),
                skeleton: node.skin().map(|s| s.index()),
            });
        }
        stack.extend(node.children());
    }
    instances.reverse();

    Ok((
        Scene {
            meshes,
            materials,
            instances,
            skeletons,
        },
        gltf_materials,
    ))
}
