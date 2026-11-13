//! glTF and OBJ import and export (feature `io`, on by default).
//!
//! Load and save [`Scene`]s as glTF 2.0 (`.gltf` / `.glb`) and Wavefront OBJ.
//!
//! Material handling follows the farthing boundary: scenes only carry named material slots.
//! [`GltfMaterial`] exposes the glTF material description as a *host-side convenience*, so a
//! simple engine (like the PoC engine) can build its own materials from a glTF file.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

mod gltf_export;
mod gltf_import;
mod obj;
mod obj_quads;

pub use gltf_export::{glb_bytes, save_glb, ExportTextures};
pub use gltf_import::{load_gltf, load_gltf_slice, load_gltf_with, GltfMaterial};
pub use obj::{load_obj, save_obj};
pub use obj_quads::{load_obj_quads, save_obj_quads};

use crate::{Error, Result, Scene};
use std::path::Path;

/// Loads `.glb`, `.gltf` or `.obj` by extension.
pub fn load(path: impl AsRef<Path>) -> Result<Scene> {
    let path = path.as_ref();
    match ext(path).as_str() {
        "glb" | "gltf" => Ok(load_gltf(path)?.0),
        "obj" => load_obj(path),
        other => Err(Error::InvalidInput(format!(
            "unsupported input format `{other}` (use .glb, .gltf or .obj)"
        ))),
    }
}

/// Saves `.glb` or `.obj` by extension.
pub fn save(scene: &Scene, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    match ext(path).as_str() {
        "glb" => save_glb(scene, path, &ExportTextures::default()),
        "obj" => save_obj(scene, path),
        other => Err(Error::InvalidInput(format!(
            "unsupported output format `{other}` (use .glb or .obj)"
        ))),
    }
}

fn ext(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glam::{Mat4, Vec2, Vec3, Vec4};
    use crate::{Instance, Joint, Mesh, MorphTarget, Skeleton, Skin};

    fn sample_scene() -> Scene {
        let mut m = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)],
            vec![0, 1, 2, 2, 1, 3],
        );
        m.material_ids = vec![0, 1];
        m.normals = Some(vec![Vec3::Z; 4]);
        m.uvs = vec![vec![Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::ONE]];
        m.colors = Some(vec![Vec4::ONE; 4]);
        m.skin = Some(Skin {
            joints: vec![[0, 1, 0, 0]; 4],
            weights: vec![[0.5, 0.5, 0.0, 0.0]; 4],
        });
        m.morph_targets = vec![MorphTarget {
            name: "up".into(),
            position_deltas: vec![Vec3::Z; 4],
            normal_deltas: None,
        }];
        let mut s = Scene::from_mesh(m);
        s.materials[1].name = "second".into();
        s.skeletons.push(Skeleton {
            name: "rig".into(),
            joints: vec![
                Joint {
                    name: "root".into(),
                    parent: None,
                    inverse_bind: Mat4::IDENTITY,
                },
                Joint {
                    name: "tip".into(),
                    parent: Some(0),
                    inverse_bind: Mat4::from_translation(-Vec3::Y),
                },
            ],
        });
        s.instances[0] = Instance {
            name: "quad".into(),
            mesh: 0,
            transform: Mat4::from_translation(Vec3::X),
            skeleton: Some(0),
        };
        s
    }

    #[test]
    fn glb_roundtrip() {
        let s = sample_scene();
        let dir = std::env::temp_dir().join("farthing_io_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rt.glb");
        save(&s, &p).unwrap();
        let (back, mats) = load_gltf(&p).unwrap();
        assert_eq!(back.triangle_count(), 2);
        assert_eq!(mats.len(), 2);
        assert_eq!(back.materials[1].name, "second");
        let m = &back.meshes[back.instances[0].mesh];
        m.validate().unwrap();
        assert!(m.skin.is_some() && m.normals.is_some() && m.colors.is_some());
        assert_eq!(m.morph_targets.len(), 1);
        assert_eq!(back.skeletons[0].joints.len(), 2);
        assert_eq!(back.skeletons[0].joints[1].parent, Some(0));
        let ib = back.skeletons[0].joints[1].inverse_bind;
        assert!((ib.w_axis.truncate() + Vec3::Y).length() < 1e-5);
        assert!((back.bounds().min.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn glb_roundtrip_in_memory() {
        let s = sample_scene();
        let bytes = glb_bytes(&s, &ExportTextures::default()).unwrap();
        let (back, mats) = load_gltf_slice(&bytes).unwrap();
        assert_eq!(back.triangle_count(), 2);
        assert_eq!(mats.len(), 2);
        assert_eq!(back.skeletons[0].joints.len(), 2);
    }

    /// A `.gltf` whose buffer is a separate file, resolved by name (as from a zip archive),
    /// and the same buffer embedded as a base64 data URI.
    #[test]
    fn gltf_external_and_data_uri_buffers() {
        let glb = glb_bytes(&sample_scene(), &ExportTextures::default()).unwrap();
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let mut doc: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
        let bin = glb[20 + json_len + 8..].to_vec();

        doc["buffers"][0]["uri"] = "my%20mesh.bin".into();
        let text = serde_json::to_vec(&doc).unwrap();
        let (back, _) = load_gltf_with(&text, |name| (name == "my mesh.bin").then(|| bin.clone()))
            .unwrap();
        assert_eq!(back.triangle_count(), 2);
        assert!(load_gltf_slice(&text).is_err(), "external file without a resolver");

        let b64 = {
            const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut s = String::new();
            for c in bin.chunks(3) {
                let n = (c[0] as u32) << 16
                    | (*c.get(1).unwrap_or(&0) as u32) << 8
                    | *c.get(2).unwrap_or(&0) as u32;
                for k in 0..4 {
                    s.push(if k <= c.len() { T[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
                }
            }
            s
        };
        doc["buffers"][0]["uri"] = format!("data:application/octet-stream;base64,{b64}").into();
        let (back, _) = load_gltf_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert_eq!(back.triangle_count(), 2);
    }

    #[test]
    fn obj_roundtrip() {
        let s = sample_scene();
        let p = std::env::temp_dir().join("farthing_io_rt.obj");
        save(&s, &p).unwrap();
        let back = load(&p).unwrap();
        assert_eq!(back.triangle_count(), 2);
        back.meshes[0].validate().unwrap();
    }
}
