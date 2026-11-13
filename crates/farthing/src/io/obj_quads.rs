use crate::quad::QuadMesh;
use crate::{Error, Result};
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

/// Loads an OBJ whose faces are quads, keeping them as quads (positions + UV set 0).
/// Vertices are keyed by (position index, uv index), so UV seams become duplicated vertices.
pub fn load_obj_quads(path: impl AsRef<Path>) -> Result<QuadMesh> {
    let text = std::fs::read_to_string(path.as_ref())?;
    let (mut v, mut vt) = (Vec::new(), Vec::new());
    let mut key: HashMap<(usize, Option<usize>), u32> = HashMap::new();
    let mut m = QuadMesh::default();
    let mut uvs: Vec<Vec2> = Vec::new();
    let mut other_faces = 0usize;
    let idx = |s: &str, len: usize| -> Option<usize> {
        let i: i64 = s.parse().ok()?;
        Some(if i < 0 {
            (len as i64 + i) as usize
        } else {
            i as usize - 1
        })
    };
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).filter_map(|x| x.parse().ok()).collect();
                if c.len() == 3 {
                    v.push(Vec3::new(c[0], c[1], c[2]));
                }
            }
            Some("vt") => {
                let c: Vec<f32> = it.take(2).filter_map(|x| x.parse().ok()).collect();
                if c.len() == 2 {
                    vt.push(Vec2::new(c[0], 1.0 - c[1]));
                }
            }
            Some("f") => {
                let corners: Vec<&str> = it.collect();
                if corners.len() != 4 {
                    other_faces += 1;
                    continue;
                }
                let mut q = [0u32; 4];
                for (k, c) in corners.iter().enumerate() {
                    let mut parts = c.split('/');
                    let pi = parts
                        .next()
                        .and_then(|s| idx(s, v.len()))
                        .ok_or_else(|| Error::InvalidInput(format!("bad face corner `{c}`")))?;
                    let ti = parts
                        .next()
                        .filter(|s| !s.is_empty())
                        .and_then(|s| idx(s, vt.len()));
                    q[k] = *key.entry((pi, ti)).or_insert_with(|| {
                        m.positions.push(v[pi]);
                        uvs.push(ti.and_then(|t| vt.get(t).copied()).unwrap_or(Vec2::ZERO));
                        m.positions.len() as u32 - 1
                    });
                }
                m.quads.push(q);
            }
            _ => {}
        }
    }
    if other_faces > 0 {
        return Err(Error::InvalidInput(format!(
            "{other_faces} non-quad faces; quad reduction needs a pure-quad OBJ"
        )));
    }
    if !vt.is_empty() {
        m.uvs = Some(uvs);
    }
    m.validate()?;
    Ok(m)
}

/// Writes a quad OBJ.
pub fn save_obj_quads(m: &QuadMesh, path: impl AsRef<Path>) -> Result<()> {
    let mut s = String::from("# farthing quads\n");
    for p in &m.positions {
        let _ = writeln!(s, "v {} {} {}", p.x, p.y, p.z);
    }
    if let Some(u) = &m.uvs {
        for t in u {
            let _ = writeln!(s, "vt {} {}", t.x, 1.0 - t.y);
        }
    }
    for q in &m.quads {
        let c = q.map(|i| i + 1);
        let _ = if m.uvs.is_some() {
            writeln!(
                s,
                "f {0}/{0} {1}/{1} {2}/{2} {3}/{3}",
                c[0], c[1], c[2], c[3]
            )
        } else {
            writeln!(s, "f {} {} {} {}", c[0], c[1], c[2], c[3])
        };
    }
    std::fs::write(path.as_ref(), s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_obj_roundtrip() {
        let m = QuadMesh {
            positions: vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::Y,
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
            ],
            uvs: Some(vec![
                Vec2::ZERO,
                Vec2::X,
                Vec2::ONE,
                Vec2::Y,
                Vec2::X,
                Vec2::ONE,
            ]),
            quads: vec![[0, 1, 2, 3], [1, 4, 5, 2]],
        };
        let p = std::env::temp_dir().join("farthing_quads.obj");
        save_obj_quads(&m, &p).unwrap();
        let back = load_obj_quads(&p).unwrap();
        assert_eq!(back.quads.len(), 2);
        assert_eq!(back.positions.len(), 6);
    }
}
