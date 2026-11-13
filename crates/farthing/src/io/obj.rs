use glam::{Vec2, Vec3};
use crate::{Error, MaterialId, MaterialSlot, Mesh, Result, Scene};
use std::fmt::Write as _;
use std::path::Path;

/// Loads an OBJ as one mesh; each OBJ object/material becomes a material slot.
pub fn load_obj(path: impl AsRef<Path>) -> Result<Scene> {
    let opts = tobj::LoadOptions {
        single_index: true,
        triangulate: true,
        ignore_points: true,
        ignore_lines: true,
    };
    let (models, mats) = tobj::load_obj(path.as_ref(), &opts)
        .map_err(|e| Error::InvalidInput(format!("OBJ: {e}")))?;
    let mats = mats.unwrap_or_default();
    let mut mesh = Mesh::default();
    let mut slots: Vec<MaterialSlot> = mats
        .iter()
        .enumerate()
        .map(|(i, m)| MaterialSlot {
            name: m.name.clone(),
            id: MaterialId(i as u32),
        })
        .collect();
    let mut default_slot = None;
    for model in &models {
        let m = &model.mesh;
        let positions: Vec<Vec3> = m
            .positions
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect();
        let mut part = Mesh::new(positions, m.indices.clone());
        let slot = match m.material_id {
            Some(i) => i as u32,
            None => *default_slot.get_or_insert_with(|| {
                slots.push(MaterialSlot {
                    name: "default".into(),
                    id: MaterialId(slots.len() as u32),
                });
                slots.len() as u32 - 1
            }),
        };
        part.material_ids.iter_mut().for_each(|s| *s = slot);
        if m.normals.len() == m.positions.len() {
            part.normals = Some(
                m.normals
                    .chunks_exact(3)
                    .map(|c| Vec3::new(c[0], c[1], c[2]))
                    .collect(),
            );
        }
        if m.texcoords.len() / 2 == m.positions.len() / 3 && !m.texcoords.is_empty() {
            // OBJ V points up; glTF / farthing V points down.
            part.uvs.push(
                m.texcoords
                    .chunks_exact(2)
                    .map(|c| Vec2::new(c[0], 1.0 - c[1]))
                    .collect(),
            );
        }
        part.validate()?;
        mesh.append(&part, 0);
    }
    if slots.is_empty() {
        slots.push(MaterialSlot {
            name: "default".into(),
            id: MaterialId(0),
        });
    }
    let mut scene = Scene::from_mesh(mesh);
    scene.materials = slots;
    Ok(scene)
}

/// Writes all instances (world space) into one OBJ with `usemtl` groups.
pub fn save_obj(scene: &Scene, path: impl AsRef<Path>) -> Result<()> {
    let (mesh, _) = scene.flatten();
    let mut s = String::with_capacity(mesh.vertex_count() * 64);
    s.push_str("# farthing\n");
    for p in &mesh.positions {
        let _ = writeln!(s, "v {} {} {}", p.x, p.y, p.z);
    }
    let has_uv = !mesh.uvs.is_empty();
    if has_uv {
        for t in &mesh.uvs[0] {
            let _ = writeln!(s, "vt {} {}", t.x, 1.0 - t.y);
        }
    }
    let has_n = mesh.normals.is_some();
    if let Some(ns) = &mesh.normals {
        for n in ns {
            let _ = writeln!(s, "vn {} {} {}", n.x, n.y, n.z);
        }
    }
    let mut current = u32::MAX;
    for t in 0..mesh.triangle_count() {
        let slot = mesh.material_ids[t];
        if slot != current {
            current = slot;
            let name = scene
                .materials
                .get(slot as usize)
                .map_or("default", |m| m.name.as_str());
            let _ = writeln!(s, "usemtl {name}");
        }
        s.push('f');
        for v in mesh.triangle(t) {
            let i = v + 1;
            let _ = match (has_uv, has_n) {
                (true, true) => write!(s, " {i}/{i}/{i}"),
                (true, false) => write!(s, " {i}/{i}"),
                (false, true) => write!(s, " {i}//{i}"),
                (false, false) => write!(s, " {i}"),
            };
        }
        s.push('\n');
    }
    std::fs::write(path.as_ref(), s)?;
    Ok(())
}
