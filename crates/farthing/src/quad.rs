//! Pure-quad meshes, for quad-aware reduction.

use crate::mesh::Mesh;
use crate::{Error, Result};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// A mesh made only of quads (counter-clockwise), with per-vertex attributes.
/// Vertices are shared between quads; a UV seam is represented as duplicated vertices
/// (and therefore behaves like a border for quad-ring traversal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuadMesh {
    pub positions: Vec<Vec3>,
    pub uvs: Option<Vec<Vec2>>,
    pub quads: Vec<[u32; 4]>,
}

impl QuadMesh {
    pub fn quad_count(&self) -> usize {
        self.quads.len()
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        if self.uvs.as_ref().is_some_and(|u| u.len() != n) {
            return Err(Error::InvalidMesh(
                "uv count differs from vertex count".into(),
            ));
        }
        for q in &self.quads {
            if q.iter().any(|&i| i as usize >= n) {
                return Err(Error::InvalidMesh("quad index out of range".into()));
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    if q[i] == q[j] {
                        return Err(Error::InvalidMesh(format!("degenerate quad {q:?}")));
                    }
                }
            }
        }
        Ok(())
    }

    /// Triangulates along the shorter diagonal.
    pub fn to_mesh(&self) -> Mesh {
        let mut idx = Vec::with_capacity(self.quads.len() * 6);
        for q in &self.quads {
            let p = q.map(|i| self.positions[i as usize]);
            if (p[0] - p[2]).length_squared() <= (p[1] - p[3]).length_squared() {
                idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            } else {
                idx.extend_from_slice(&[q[0], q[1], q[3], q[1], q[2], q[3]]);
            }
        }
        let mut m = Mesh::new(self.positions.clone(), idx);
        if let Some(u) = &self.uvs {
            m.uvs = vec![u.clone()];
        }
        m
    }

    /// Pairs triangles that share an edge into quads (greedy, best-shaped pairs first).
    /// Fails if more than `max_unpaired_fraction` of the triangles stay unpaired.
    pub fn from_triangles(mesh: &Mesh, max_unpaired_fraction: f32) -> Result<(QuadMesh, usize)> {
        mesh.validate()?;
        let tc = mesh.triangle_count();
        let mut edge_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for t in 0..tc {
            let tri = mesh.triangle(t);
            for i in 0..3 {
                let (a, b) = (tri[i], tri[(i + 1) % 3]);
                edge_tris.entry((a.min(b), a.max(b))).or_default().push(t);
            }
        }
        // Score every candidate pair: right angles and coplanarity are good.
        let mut pairs: Vec<(f32, usize, usize, [u32; 4])> = Vec::new();
        for (&(a, b), ts) in &edge_tris {
            if ts.len() != 2 {
                continue;
            }
            let (t0, t1) = (ts[0], ts[1]);
            let tri0 = mesh.triangle(t0);
            let tri1 = mesh.triangle(t1);
            let c0 = *tri0.iter().find(|&&v| v != a && v != b).unwrap();
            let c1 = *tri1.iter().find(|&&v| v != a && v != b).unwrap();
            // Quad winding follows t0: rotate t0 so the shared edge is its last edge (x -> y),
            // then the quad is [c0, x, c1, y].
            let i = tri0.iter().position(|&v| v == c0).unwrap();
            let (x, y) = (tri0[(i + 1) % 3], tri0[(i + 2) % 3]);
            let quad = [c0, x, c1, y];
            let p = quad.map(|v| mesh.positions[v as usize]);
            let n0 = (p[1] - p[0]).cross(p[3] - p[0]).normalize_or_zero();
            let n1 = (p[3] - p[2]).cross(p[1] - p[2]).normalize_or_zero();
            let planar = n0.dot(n1);
            if planar < 0.5 {
                continue;
            }
            let mut angle_err = 0.0;
            for k in 0..4 {
                let e0 = (p[(k + 1) % 4] - p[k]).normalize_or_zero();
                let e1 = (p[(k + 3) % 4] - p[k]).normalize_or_zero();
                angle_err += e0.dot(e1).abs();
            }
            pairs.push((angle_err + (1.0 - planar) * 4.0, t0, t1, quad));
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used = vec![false; tc];
        let mut quads = Vec::new();
        for (_, t0, t1, q) in pairs {
            if !used[t0] && !used[t1] {
                used[t0] = true;
                used[t1] = true;
                quads.push(q);
            }
        }
        let unpaired = used.iter().filter(|u| !**u).count();
        if unpaired as f32 > max_unpaired_fraction * tc as f32 {
            return Err(Error::InvalidMesh(format!(
                "{unpaired} of {tc} triangles could not be paired into quads"
            )));
        }
        let qm = QuadMesh {
            positions: mesh.positions.clone(),
            uvs: mesh.uvs.first().cloned(),
            quads,
        };
        Ok((qm, unpaired))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_pairs_back_into_quads() {
        let n = 6u32;
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.push(Vec3::new(x as f32, 0.0, y as f32));
            }
        }
        let mut quads = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                quads.push([a, a + n + 1, a + n + 2, a + 1]);
            }
        }
        let qm = QuadMesh {
            positions: pos,
            uvs: None,
            quads,
        };
        qm.validate().unwrap();
        let tri = qm.to_mesh();
        let (back, unpaired) = QuadMesh::from_triangles(&tri, 0.0).unwrap();
        assert_eq!(unpaired, 0);
        assert_eq!(back.quad_count(), 36);
        back.validate().unwrap();
        // Same orientation as the triangles.
        let t = back.to_mesh();
        for k in 0..t.triangle_count() {
            let [a, b, c] = t.triangle_positions(k);
            assert!((b - a).cross(c - a).y > 0.0, "pairing flipped the winding");
        }
    }
}
