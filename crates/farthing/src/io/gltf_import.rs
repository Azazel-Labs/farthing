use glam::{Mat4, Vec2, Vec3, Vec4};
use crate::host::{MaterialClass, MaterialInfo};
use crate::image::Image;
use crate::{
    Error, Instance, Joint, MaterialId, MaterialSlot, Mesh, MorphTarget, Result, Scene, Skeleton,
    Skin,
};
use std::collections::HashMap;
use std::path::Path;

/// The glTF description of a material — for hosts that want to build their own material
/// from it. farthing itself only uses the slot name.
#[derive(Clone, Debug)]
pub struct GltfMaterial {
    pub name: String,
    pub info: MaterialInfo,
    pub base_color_factor: Vec4,
    /// Decoded base color texture (linear RGB, sRGB-decoded) and its UV set.
    pub base_color_texture: Option<(Image, u32)>,
}

/// Loads a glTF / GLB file. Each glTF mesh becomes one [`Mesh`] (primitives merged, material
/// ids = slot index); each node with a mesh becomes an [`Instance`] with its world transform.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<(Scene, Vec<GltfMaterial>)> {
    let (doc, buffers, images) =
        gltf::import(path.as_ref()).map_err(|e| Error::InvalidInput(format!("glTF: {e}")))?;
    build(&doc, &buffers, &|i| images.get(i).map(decode_image))
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
    let gltf::Gltf { document, blob } =
        gltf::Gltf::from_slice(bytes).map_err(|e| Error::InvalidInput(format!("glTF: {e}")))?;
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
    let mut buffers = Vec::new();
    for b in document.buffers() {
        let mut data = match b.source() {
            gltf::buffer::Source::Bin => blob
                .clone()
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
    // Images are decoded on demand (only base color textures are read).
    let mut encoded: Vec<Option<Vec<u8>>> = Vec::new();
    for img in document.images() {
        encoded.push(match img.source() {
            gltf::image::Source::View { view, .. } => {
                let buf = &buffers[view.buffer().index()];
                buf.get(view.offset()..view.offset() + view.length())
                    .map(<[u8]>::to_vec)
            }
            gltf::image::Source::Uri { uri, .. } => fetch(uri).ok(),
        });
    }
    build(&document, &buffers, &|i| {
        let bytes = encoded.get(i)?.as_ref()?;
        let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
        Some(Image::from_rgba8(rgba.width(), rgba.height(), rgba.as_raw(), true))
    })
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
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
    image: &dyn Fn(usize) -> Option<Image>,
) -> Result<(Scene, Vec<GltfMaterial>)> {
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
                .and_then(|t| Some((image(t.texture().source().index())?, t.tex_coord())));
            GltfMaterial {
                name: materials[i].name.clone(),
                info: MaterialInfo {
                    class,
                    two_sided: m.double_sided(),
                },
                base_color_factor: Vec4::from_array(pbr.base_color_factor()),
                base_color_texture: tex,
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
                match reader.read_tex_coords(set) {
                    Some(tc) => part.uvs.push(tc.into_f32().map(Vec2::from_array).collect()),
                    None => break,
                }
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
                Joint {
                    name: doc
                        .nodes()
                        .nth(n)
                        .and_then(|nd| nd.name().map(str::to_owned))
                        .unwrap_or_else(|| format!("joint_{j}")),
                    parent: p.and_then(|pn| node_to_joint.get(&pn).copied()),
                    inverse_bind: ibms.get(j).copied().unwrap_or(Mat4::IDENTITY),
                }
            })
            .collect();
        skeletons.push(Skeleton {
            name: skin.name().unwrap_or("skin").to_owned(),
            joints,
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

fn decode_image(img: &gltf::image::Data) -> Image {
    use gltf::image::Format;
    let px = &img.pixels;
    let rgba: Vec<u8> = match img.format {
        Format::R8G8B8A8 => px.clone(),
        Format::R8G8B8 => px
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        Format::R8G8 => px
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[1], 0, 255])
            .collect(),
        Format::R8 => px.iter().flat_map(|&c| [c, c, c, 255]).collect(),
        // 16-bit and float formats: keep the high byte, good enough for previews.
        Format::R16G16B16A16 => px
            .chunks_exact(8)
            .flat_map(|c| [c[1], c[3], c[5], c[7]])
            .collect(),
        Format::R16G16B16 => px
            .chunks_exact(6)
            .flat_map(|c| [c[1], c[3], c[5], 255])
            .collect(),
        _ => vec![255; (img.width * img.height * 4) as usize],
    };
    Image::from_rgba8(img.width, img.height, &rgba, true)
}
