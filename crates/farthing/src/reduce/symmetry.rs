//! Mirror-symmetry detection for symmetry-aware reduction.
//!
//! Candidate planes pass through the bounding-box center with normals along X, Y and Z (the
//! common authoring case). A plane is accepted when at least 95% of positions have a mirror
//! partner within 1e-4 of the bounding-box diagonal.

use crate::Mesh;
use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Symmetry {
    #[default]
    Off,
    /// Detect an axis-aligned mirror plane; fall back to no symmetry if none is found.
    Auto,
    /// Use this plane (n.xyz unit normal, w = d with n·p + d = 0).
    Plane(Vec4),
}

const MATCH_FRACTION: f32 = 0.95;

struct PointGrid {
    cell: f32,
    map: HashMap<[i64; 3], Vec<u32>>,
}

impl PointGrid {
    fn new(points: impl Iterator<Item = (u32, Vec3)>, cell: f32) -> PointGrid {
        let mut map: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, p) in points {
            map.entry(Self::key(p, cell)).or_default().push(i);
        }
        PointGrid { cell, map }
    }
    fn key(p: Vec3, cell: f32) -> [i64; 3] {
        [
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        ]
    }
    /// Closest stored point within `tol` of `p`.
    fn find(&self, p: Vec3, tol: f32, pts: &[Vec3]) -> Option<u32> {
        let k = Self::key(p, self.cell);
        let mut best: Option<(f32, u32)> = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) = self.map.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        for &i in list {
                            let d = (pts[i as usize] - p).length();
                            if d <= tol && best.is_none_or(|b| d < b.0) {
                                best = Some((d, i));
                            }
                        }
                    }
                }
            }
        }
        best.map(|b| b.1)
    }
}

fn reflect(p: Vec3, plane: Vec4) -> Vec3 {
    let n = plane.truncate();
    p - n * 2.0 * (n.dot(p) + plane.w)
}

fn match_fraction(points: &[Vec3], plane: Vec4, tol: f32) -> f32 {
    let grid = PointGrid::new(points.iter().enumerate().map(|(i, p)| (i as u32, *p)), tol);
    let matched = points
        .iter()
        .filter(|p| grid.find(reflect(**p, plane), tol, points).is_some())
        .count();
    matched as f32 / points.len().max(1) as f32
}

fn unique_positions(mesh: &Mesh) -> Vec<Vec3> {
    let mut seen = std::collections::HashSet::new();
    mesh.positions
        .iter()
        .filter(|p| seen.insert([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]))
        .copied()
        .collect()
}

/// Finds an axis-aligned mirror plane. Returns the plane and the fraction of matched positions.
pub fn detect_symmetry(mesh: &Mesh) -> Option<(Vec4, f32)> {
    let pts = unique_positions(mesh);
    if pts.len() < 4 {
        return None;
    }
    let b = mesh.bounds();
    let tol = (b.diagonal() * 1e-4).max(1e-7);
    let c = b.center();
    [Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .map(|n| {
            let plane = n.extend(-n.dot(c));
            (plane, match_fraction(&pts, plane, tol))
        })
        .filter(|&(_, f)| f >= MATCH_FRACTION)
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

pub(crate) fn resolve_plane(mesh: &Mesh, s: Symmetry) -> Option<Vec4> {
    match s {
        Symmetry::Off => None,
        Symmetry::Auto => detect_symmetry(mesh).map(|p| p.0),
        Symmetry::Plane(p) => Some(p.truncate().normalize().extend(p.w / p.truncate().length())),
    }
}

/// For each welded position id: its mirror partner, itself if on the plane, `u32::MAX` if none.
pub(crate) fn mirror_map(
    positions: &[Vec3],
    vpos: &[u32],
    np: usize,
    plane: Vec4,
    diag: f32,
) -> Vec<u32> {
    let tol = (diag * 1e-4).max(1e-7);
    let mut pts = vec![Vec3::ZERO; np];
    for (v, &p) in vpos.iter().enumerate() {
        pts[p as usize] = positions[v];
    }
    let grid = PointGrid::new(pts.iter().enumerate().map(|(i, p)| (i as u32, *p)), tol);
    (0..np)
        .map(|i| {
            let p = pts[i];
            if (plane.truncate().dot(p) + plane.w).abs() <= tol {
                return i as u32;
            }
            grid.find(reflect(p, plane), tol, &pts).unwrap_or(u32::MAX)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::{simplify, SimplifyOptions};

    /// A symmetric (about x = 0) bumpy surface with an asymmetric triangulation.
    fn symmetric_patch(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let fx = x as f32 / n as f32 * 2.0 - 1.0;
                let fz = y as f32 / n as f32 * 2.0 - 1.0;
                let h = 0.3 * (3.0 * fx).cos() * (2.0 * fz).sin() + 0.2 * fx * fx;
                pos.push(Vec3::new(fx, h, fz));
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                // Same diagonal everywhere: not mirror-symmetric as a triangulation.
                idx.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        Mesh::new(pos, idx)
    }

    #[test]
    fn detects_x_plane() {
        let m = symmetric_patch(20);
        let (plane, f) = detect_symmetry(&m).expect("symmetric");
        assert!(
            (plane.truncate() - Vec3::X).length() < 1e-6 && plane.w.abs() < 1e-5,
            "{plane:?}"
        );
        assert!(f > 0.99);
    }

    #[test]
    fn symmetric_reduction_stays_symmetric() {
        let m = symmetric_patch(24);
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_ratio: Some(0.2),
                symmetry: Symmetry::Auto,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.symmetry_plane.is_some());
        let pts = unique_positions(&r.mesh);
        let frac = match_fraction(&pts, Vec4::new(1.0, 0.0, 0.0, 0.0), 1e-4);
        assert!(
            frac > 0.97,
            "only {frac} of output positions have a mirror partner"
        );
        // Without symmetry the same reduction is visibly asymmetric.
        let plain = simplify(&m, &SimplifyOptions::ratio(0.2)).unwrap();
        let pf = match_fraction(
            &unique_positions(&plain.mesh),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            1e-4,
        );
        assert!(frac > pf, "symmetric {frac} vs plain {pf}");
    }

    #[test]
    fn asymmetric_mesh_has_no_plane() {
        let mut m = symmetric_patch(10);
        for (i, p) in m.positions.iter_mut().enumerate() {
            p.y += (i as f32 * 0.37).sin() * 0.1;
        }
        assert!(detect_symmetry(&m).is_none());
    }
}
