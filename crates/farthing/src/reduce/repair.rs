//! Mesh repair: welding, degenerate and duplicate triangle removal, small-part removal.

use crate::Mesh;
use crate::fxhash::{HashMap, HashSet};
use crate::math::triangle_area;

/// Merges vertices whose positions fall in the same `epsilon` grid cell (or match exactly when
/// `epsilon == 0`) *and* whose other attributes are bit-identical. Returns the number of
/// vertices removed.
pub fn weld(mesh: &mut Mesh, epsilon: f32) -> usize {
    let n = mesh.vertex_count();
    // Every vertex's key, laid end to end in one buffer: allocating one per vertex and hashing it
    // with SipHash was most of the time a weld took.
    let stride = 3 * if epsilon > 0.0 { 2 } else { 1 }
        + mesh.normals.as_ref().map_or(0, |_| 3)
        + mesh.tangents.as_ref().map_or(0, |_| 4)
        + mesh.uvs.len() * 2
        + mesh.colors.as_ref().map_or(0, |_| 4)
        + mesh.skin.as_ref().map_or(0, |_| 8)
        + mesh.morph_targets.len() * 3;
    let mut keys = Vec::with_capacity(n * stride);
    for v in 0..n {
        let p = mesh.positions[v].to_array();
        if epsilon > 0.0 {
            for x in p {
                let cell = (x / epsilon).round() as i64;
                keys.extend([cell as u32, (cell >> 32) as u32]);
            }
        } else {
            keys.extend(p.map(f32::to_bits));
        }
        if let Some(n) = &mesh.normals {
            keys.extend(n[v].to_array().map(f32::to_bits));
        }
        if let Some(t) = &mesh.tangents {
            keys.extend(t[v].to_array().map(f32::to_bits));
        }
        for s in &mesh.uvs {
            keys.extend(s[v].to_array().map(f32::to_bits));
        }
        if let Some(c) = &mesh.colors {
            keys.extend(c[v].to_array().map(f32::to_bits));
        }
        if let Some(s) = &mesh.skin {
            keys.extend(s.joints[v].map(u32::from));
            keys.extend(s.weights[v].map(f32::to_bits));
        }
        for m in &mesh.morph_targets {
            keys.extend(m.position_deltas[v].to_array().map(f32::to_bits));
        }
    }
    let mut map: HashMap<&[u32], u32> = HashMap::with_capacity_and_hasher(n, Default::default());
    let mut order = Vec::new();
    let remap: Vec<u32> = keys
        .chunks_exact(stride)
        .enumerate()
        .map(|(v, key)| {
            *map.entry(key).or_insert_with(|| {
                order.push(v as u32);
                order.len() as u32 - 1
            })
        })
        .collect();
    let mut out = mesh.select_vertices(&order);
    out.indices = mesh.indices.iter().map(|&i| remap[i as usize]).collect();
    out.material_ids = std::mem::take(&mut mesh.material_ids);
    *mesh = out;
    n - mesh.vertex_count()
}

/// Strips per-wedge surface attributes (UVs, normals, tangents, colors) and welds by position,
/// so reduction is no longer constrained by UV seams or hard edges. Use for far LODs that will
/// get fresh UVs and a re-bake (`uv::parameterize` + `bake`). Skin weights
/// and morph targets are kept.
pub fn geometry_only(mesh: &Mesh) -> Mesh {
    let mut m = mesh.clone();
    m.uvs.clear();
    m.normals = None;
    m.tangents = None;
    m.colors = None;
    weld(&mut m, 0.0);
    m
}

/// Removes triangles with repeated vertices or area below `min_area`. Returns how many were removed.
pub fn remove_degenerate_triangles(mesh: &mut Mesh, min_area: f32) -> usize {
    let before = mesh.triangle_count();
    let mut idx = Vec::with_capacity(mesh.indices.len());
    let mut mats = Vec::with_capacity(mesh.material_ids.len());
    for t in 0..before {
        let [a, b, c] = mesh.triangle(t);
        let [pa, pb, pc] = mesh.triangle_positions(t);
        if a != b && b != c && a != c && triangle_area(pa, pb, pc) > min_area {
            idx.extend_from_slice(&[a, b, c]);
            mats.push(mesh.material_ids[t]);
        }
    }
    mesh.indices = idx;
    mesh.material_ids = mats;
    before - mesh.triangle_count()
}

/// Removes triangles that use the same three positions as an earlier triangle (any winding).
pub fn remove_duplicate_triangles(mesh: &mut Mesh) -> usize {
    let before = mesh.triangle_count();
    let key = |p: glam::Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let mut seen: HashSet<[[u32; 3]; 3]> = HashSet::default();
    let mut idx = Vec::new();
    let mut mats = Vec::new();
    for t in 0..before {
        let mut k = mesh.triangle_positions(t).map(key);
        k.sort();
        if seen.insert(k) {
            idx.extend_from_slice(&mesh.triangle(t));
            mats.push(mesh.material_ids[t]);
        }
    }
    mesh.indices = idx;
    mesh.material_ids = mats;
    before - mesh.triangle_count()
}

/// Removes connected components (by shared positions) whose bounding-box diagonal is smaller
/// than `min_diagonal`. Returns the number of triangles removed.
pub fn remove_small_parts(mesh: &mut Mesh, min_diagonal: f32) -> usize {
    let before = mesh.triangle_count();
    let key = |p: glam::Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let mut pid: HashMap<[u32; 3], usize> = HashMap::default();
    let vp: Vec<usize> = mesh
        .positions
        .iter()
        .map(|p| {
            let n = pid.len();
            *pid.entry(key(*p)).or_insert(n)
        })
        .collect();
    let mut parent: Vec<usize> = (0..pid.len()).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for t in 0..before {
        let [a, b, c] = mesh.triangle(t).map(|v| vp[v as usize]);
        let (ra, rb, rc) = (
            find(&mut parent, a),
            find(&mut parent, b),
            find(&mut parent, c),
        );
        parent[rb] = ra;
        let rc2 = find(&mut parent, rc);
        parent[rc2] = ra;
    }
    let mut bounds: HashMap<usize, crate::Aabb> = HashMap::default();
    for (v, p) in mesh.positions.iter().enumerate() {
        let r = find(&mut parent, vp[v]);
        bounds.entry(r).or_default().extend(*p);
    }
    let mut idx = Vec::new();
    let mut mats = Vec::new();
    for t in 0..before {
        let tri = mesh.triangle(t);
        let r = find(&mut parent, vp[tri[0] as usize]);
        if bounds[&r].diagonal() >= min_diagonal {
            idx.extend_from_slice(&tri);
            mats.push(mesh.material_ids[t]);
        }
    }
    mesh.indices = idx;
    mesh.material_ids = mats;
    before - mesh.triangle_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn weld_merges_identical() {
        let mut m = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::X, Vec3::Y, Vec3::ONE],
            vec![0, 1, 2, 3, 5, 4],
        );
        assert_eq!(weld(&mut m, 0.0), 2);
        assert_eq!(m.vertex_count(), 4);
        m.validate().unwrap();
    }

    #[test]
    fn degenerates_and_duplicates() {
        let mut m = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::X * 2.0],
            vec![0, 1, 2, 0, 1, 3, 2, 1, 0, 0, 0, 1],
        );
        assert_eq!(remove_degenerate_triangles(&mut m, 1e-9), 2);
        assert_eq!(remove_duplicate_triangles(&mut m), 1);
        assert_eq!(m.triangle_count(), 1);
    }

    #[test]
    fn small_parts() {
        let mut m = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::X * 10.0,
                Vec3::Y * 10.0,
                Vec3::splat(20.0),
                Vec3::splat(20.0) + Vec3::X * 0.01,
                Vec3::splat(20.0) + Vec3::Y * 0.01,
            ],
            vec![0, 1, 2, 3, 4, 5],
        );
        assert_eq!(remove_small_parts(&mut m, 1.0), 1);
    }
}
