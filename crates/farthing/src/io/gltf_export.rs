use crate::image::Texture;
use crate::{Error, Result, Scene};
use glam::Mat4;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// Optional textures to embed per material slot, so outputs open in any glTF viewer.
#[derive(Clone, Debug, Default)]
pub struct ExportTextures {
    /// slot index → base color, sampled with UV set 0. Written as sRGB (converted if the
    /// texture holds linear bytes); build one from a bake with [`Texture::from_image`].
    pub base_color: BTreeMap<u32, Texture>,
    /// slot index → tangent-space normal map stored as glTF does (xyz · 0.5 + 0.5; see
    /// [`Texture::from_signed_normals`]).
    pub normal: BTreeMap<u32, Texture>,
    /// slot index → alpha cutoff, for alpha-tested outputs (impostors, cards).
    pub alpha_cutoff: BTreeMap<u32, f32>,
    /// slot index → base color factor (linear RGBA; multiplies the base color texture).
    /// Default white.
    pub base_color_factor: BTreeMap<u32, glam::Vec4>,
    /// slot index → glTF metallic-roughness texture (linear data: G = roughness,
    /// B = metalness), sampled with UV set 0.
    pub metallic_roughness: BTreeMap<u32, Texture>,
    /// slot index → (metallic, roughness) factors. Default (0, 0.8) without a texture,
    /// (1, 1) with one.
    pub metallic_roughness_factor: BTreeMap<u32, glam::Vec2>,
}

struct Bin {
    data: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Bin {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.data.len().is_multiple_of(4) {
            self.data.push(0);
        }
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        let mut v = json!({ "buffer": 0, "byteOffset": offset, "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }
    fn accessor_f32(&mut self, data: &[f32], comps: usize, kind: &str, minmax: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(34962));
        let count = data.len() / comps;
        let mut a =
            json!({ "bufferView": view, "componentType": 5126, "count": count, "type": kind });
        if minmax && count > 0 {
            let mut min = vec![f32::INFINITY; comps];
            let mut max = vec![f32::NEG_INFINITY; comps];
            for c in data.chunks_exact(comps) {
                for i in 0..comps {
                    min[i] = min[i].min(c[i]);
                    max[i] = max[i].max(c[i]);
                }
            }
            a["min"] = json!(min);
            a["max"] = json!(max);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }
    fn accessor_u16x4(&mut self, data: &[[u16; 4]]) -> usize {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|j| j.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let view = self.view(&bytes, Some(34962));
        self.accessors.push(json!({ "bufferView": view, "componentType": 5123, "count": data.len(), "type": "VEC4" }));
        self.accessors.len() - 1
    }
    fn accessor_indices(&mut self, data: &[u32]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(34963));
        self.accessors.push(json!({ "bufferView": view, "componentType": 5125, "count": data.len(), "type": "SCALAR" }));
        self.accessors.len() - 1
    }
    fn accessor_mat4(&mut self, mats: &[Mat4]) -> usize {
        let bytes: Vec<u8> = mats
            .iter()
            .flat_map(|m| m.to_cols_array().into_iter().flat_map(f32::to_le_bytes))
            .collect();
        let view = self.view(&bytes, None);
        self.accessors.push(json!({ "bufferView": view, "componentType": 5126, "count": mats.len(), "type": "MAT4" }));
        self.accessors.len() - 1
    }
}

fn png_bytes(tex: &Texture, srgb: bool) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(
            &tex.rgba8_as(srgb),
            tex.width,
            tex.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| Error::Image(e.to_string()))?;
    Ok(out)
}

/// Writes a binary glTF. Each mesh becomes one glTF mesh with one primitive per material
/// slot used (vertex accessors shared). Instances become nodes; skeletons become joint nodes.
pub fn save_glb(scene: &Scene, path: impl AsRef<Path>, textures: &ExportTextures) -> Result<()> {
    std::fs::write(path.as_ref(), glb_bytes(scene, textures)?)?;
    Ok(())
}

/// [`save_glb`] into memory: the bytes of a binary glTF.
pub fn glb_bytes(scene: &Scene, textures: &ExportTextures) -> Result<Vec<u8>> {
    let mut bin = Bin {
        data: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
    };
    let mut images = Vec::new();
    let mut gl_textures = Vec::new();

    let materials: Vec<Value> = scene
        .materials
        .iter()
        .enumerate()
        .map(|(slot, m)| -> Result<Value> {
            let factor = textures
                .base_color_factor
                .get(&(slot as u32))
                .copied()
                .unwrap_or(glam::Vec4::ONE);
            let mr_tex = textures.metallic_roughness.get(&(slot as u32));
            let mr = textures
                .metallic_roughness_factor
                .get(&(slot as u32))
                .copied()
                .unwrap_or(if mr_tex.is_some() { glam::Vec2::ONE } else { glam::Vec2::new(0.0, 0.8) });
            let mut mat = json!({
                "name": m.name,
                "pbrMetallicRoughness": { "baseColorFactor": factor.to_array(), "metallicFactor": mr.x, "roughnessFactor": mr.y }
            });
            if let Some(img) = mr_tex {
                let view = bin.view(&png_bytes(img, false)?, None);
                images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
                gl_textures.push(json!({ "source": images.len() - 1 }));
                mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = json!({ "index": gl_textures.len() - 1 });
            }
            if let Some(img) = textures.base_color.get(&(slot as u32)) {
                let view = bin.view(&png_bytes(img, true)?, None);
                images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
                gl_textures.push(json!({ "source": images.len() - 1 }));
                mat["pbrMetallicRoughness"]["baseColorTexture"] = json!({ "index": gl_textures.len() - 1 });
            }
            if let Some(img) = textures.normal.get(&(slot as u32)) {
                let view = bin.view(&png_bytes(img, false)?, None);
                images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
                gl_textures.push(json!({ "source": images.len() - 1 }));
                mat["normalTexture"] = json!({ "index": gl_textures.len() - 1 });
            }
            if let Some(c) = textures.alpha_cutoff.get(&(slot as u32)) {
                mat["alphaMode"] = json!("MASK");
                mat["alphaCutoff"] = json!(c);
                mat["doubleSided"] = json!(true);
            }
            Ok(mat)
        })
        .collect::<Result<_>>()?;

    let mut meshes = Vec::new();
    for mesh in &scene.meshes {
        mesh.validate()?;
        let flat3 = |v: &[glam::Vec3]| v.iter().flat_map(|p| p.to_array()).collect::<Vec<f32>>();
        let mut attrs = serde_json::Map::new();
        attrs.insert(
            "POSITION".into(),
            json!(bin.accessor_f32(&flat3(&mesh.positions), 3, "VEC3", true)),
        );
        if let Some(n) = &mesh.normals {
            attrs.insert(
                "NORMAL".into(),
                json!(bin.accessor_f32(&flat3(n), 3, "VEC3", false)),
            );
        }
        if let Some(t) = &mesh.tangents {
            let d: Vec<f32> = t.iter().flat_map(|v| v.to_array()).collect();
            attrs.insert(
                "TANGENT".into(),
                json!(bin.accessor_f32(&d, 4, "VEC4", false)),
            );
        }
        for (i, uv) in mesh.uvs.iter().enumerate() {
            let d: Vec<f32> = uv.iter().flat_map(|v| v.to_array()).collect();
            attrs.insert(
                format!("TEXCOORD_{i}"),
                json!(bin.accessor_f32(&d, 2, "VEC2", false)),
            );
        }
        if let Some(c) = &mesh.colors {
            let d: Vec<f32> = c.iter().flat_map(|v| v.to_array()).collect();
            attrs.insert(
                "COLOR_0".into(),
                json!(bin.accessor_f32(&d, 4, "VEC4", false)),
            );
        }
        if let Some(s) = &mesh.skin {
            attrs.insert("JOINTS_0".into(), json!(bin.accessor_u16x4(&s.joints)));
            let d: Vec<f32> = s.weights.iter().flatten().copied().collect();
            attrs.insert(
                "WEIGHTS_0".into(),
                json!(bin.accessor_f32(&d, 4, "VEC4", false)),
            );
        }
        let targets: Vec<Value> = mesh
            .morph_targets
            .iter()
            .map(|t| json!({ "POSITION": bin.accessor_f32(&flat3(&t.position_deltas), 3, "VEC3", true) }))
            .collect();
        // One primitive per material slot.
        let mut by_slot: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for t in 0..mesh.triangle_count() {
            by_slot
                .entry(mesh.material_ids[t])
                .or_default()
                .extend_from_slice(&mesh.triangle(t));
        }
        let prims: Vec<Value> = by_slot
            .into_iter()
            .map(|(slot, idx)| {
                let mut p = json!({ "attributes": attrs.clone(), "indices": bin.accessor_indices(&idx), "mode": 4 });
                if (slot as usize) < materials.len() {
                    p["material"] = json!(slot);
                }
                if !targets.is_empty() {
                    p["targets"] = json!(targets);
                }
                p
            })
            .collect();
        let mut m = json!({ "primitives": prims });
        if !mesh.morph_targets.is_empty() {
            m["extras"] = json!({ "targetNames": mesh.morph_targets.iter().map(|t| t.name.clone()).collect::<Vec<_>>() });
        }
        meshes.push(m);
    }

    // Nodes: skeleton joints first, then instances.
    let mut nodes: Vec<Value> = Vec::new();
    let mut scene_roots: Vec<usize> = Vec::new();
    let mut skins = Vec::new();
    for sk in &scene.skeletons {
        let base = nodes.len();
        let worlds: Vec<Mat4> = (0..sk.joints.len()).map(|j| sk.bind_world(j)).collect();
        for (j, joint) in sk.joints.iter().enumerate() {
            let local = match joint.parent {
                Some(p) => worlds[p].inverse() * worlds[j],
                None => worlds[j],
            };
            nodes.push(json!({ "name": joint.name, "matrix": local.to_cols_array() }));
        }
        for (j, joint) in sk.joints.iter().enumerate() {
            match joint.parent {
                Some(p) => {
                    let children = nodes[base + p]
                        .get("children")
                        .cloned()
                        .unwrap_or(json!([]));
                    let mut c: Vec<usize> = serde_json::from_value(children).unwrap_or_default();
                    c.push(base + j);
                    nodes[base + p]["children"] = json!(c);
                }
                None => scene_roots.push(base + j),
            }
        }
        let ibm = bin.accessor_mat4(&sk.joints.iter().map(|j| j.inverse_bind).collect::<Vec<_>>());
        skins.push(json!({ "name": sk.name, "joints": (base..base + sk.joints.len()).collect::<Vec<_>>(), "inverseBindMatrices": ibm }));
    }
    for inst in &scene.instances {
        let mut n = json!({ "name": inst.name, "mesh": inst.mesh });
        if inst.transform != Mat4::IDENTITY {
            n["matrix"] = json!(inst.transform.to_cols_array());
        }
        if let Some(s) = inst.skeleton {
            n["skin"] = json!(s);
        }
        nodes.push(n);
        scene_roots.push(nodes.len() - 1);
    }

    while !bin.data.len().is_multiple_of(4) {
        bin.data.push(0);
    }
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": "farthing" },
        "scene": 0,
        "scenes": [{ "nodes": scene_roots }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": bin.accessors,
        "bufferViews": bin.views,
        "buffers": [{ "byteLength": bin.data.len() }],
    });
    if !skins.is_empty() {
        doc["skins"] = json!(skins);
    }
    if !images.is_empty() {
        doc["images"] = json!(images);
        doc["textures"] = json!(gl_textures);
        doc["samplers"] = json!([{ "magFilter": 9729, "minFilter": 9987 }]);
        for t in doc["textures"].as_array_mut().unwrap() {
            t["sampler"] = json!(0);
        }
    }

    let mut json_bytes =
        serde_json::to_vec(&doc).map_err(|e| Error::InvalidInput(e.to_string()))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.data.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin.data.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    out.extend_from_slice(&bin.data);
    Ok(out)
}
