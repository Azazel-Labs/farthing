//! GPU-friendly ordering: vertex-cache optimization (Forsyth's linear-speed algorithm),
//! overdraw ordering (Sander, Nehab and Barczak 2007), vertex-fetch ordering, and ACMR
//! analysis.

use crate::Mesh;
use glam::Vec3;

/// Average cache miss ratio (misses per triangle) for a FIFO cache of `cache_size`.
/// 0.5 is the theoretical best for large regular meshes; 3.0 is the worst.
pub fn acmr(indices: &[u32], cache_size: usize) -> f32 {
    let mut cache: std::collections::VecDeque<u32> =
        std::collections::VecDeque::with_capacity(cache_size);
    let mut misses = 0usize;
    for &i in indices {
        if !cache.contains(&i) {
            misses += 1;
            if cache.len() == cache_size {
                cache.pop_front();
            }
            cache.push_back(i);
        }
    }
    misses as f32 / (indices.len() / 3).max(1) as f32
}

const CACHE: usize = 32;

fn vertex_score(cache_pos: Option<usize>, valence: u32) -> f32 {
    if valence == 0 {
        return -1.0;
    }
    let mut score = match cache_pos {
        Some(p) if p < 3 => 0.75,
        Some(p) => {
            let s = 1.0 - (p - 3) as f32 / (CACHE - 3) as f32;
            s.powf(1.5)
        }
        None => 0.0,
    };
    score += 2.0 * (valence as f32).powf(-0.5);
    score
}

/// Reorders triangles for the post-transform vertex cache (Forsyth 2006). Returns new indices.
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let tri_count = indices.len() / 3;
    let mut valence = vec![0u32; vertex_count];
    for &i in indices {
        valence[i as usize] += 1;
    }
    // Vertex → triangles adjacency (CSR).
    let mut offsets = vec![0usize; vertex_count + 1];
    for v in 0..vertex_count {
        offsets[v + 1] = offsets[v] + valence[v] as usize;
    }
    let mut fill = offsets.clone();
    let mut adj = vec![0u32; indices.len()];
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        for &v in tri {
            adj[fill[v as usize]] = t as u32;
            fill[v as usize] += 1;
        }
    }
    let mut live = valence.clone();
    let mut cache_pos: Vec<Option<usize>> = vec![None; vertex_count];
    let mut vscore: Vec<f32> = (0..vertex_count)
        .map(|v| vertex_score(None, live[v]))
        .collect();
    let mut emitted = vec![false; tri_count];
    let tri_score = |t: usize, vs: &[f32]| {
        indices[3 * t..3 * t + 3]
            .iter()
            .map(|&v| vs[v as usize])
            .sum::<f32>()
    };
    let mut tscore: Vec<f32> = (0..tri_count).map(|t| tri_score(t, &vscore)).collect();
    let mut cache: Vec<u32> = Vec::with_capacity(CACHE + 3);
    let mut out = Vec::with_capacity(indices.len());
    let mut scan = 0usize;
    for _ in 0..tri_count {
        // Best triangle among those touching the cache; else next unemitted by scan.
        let mut best: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for &v in &cache {
            for &t in &adj[offsets[v as usize]..offsets[v as usize + 1]] {
                let t = t as usize;
                if !emitted[t] && tscore[t] > best_score {
                    best_score = tscore[t];
                    best = Some(t);
                }
            }
        }
        let t = match best {
            Some(t) => t,
            None => {
                while emitted[scan] {
                    scan += 1;
                }
                scan
            }
        };
        emitted[t] = true;
        let tri = [indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]];
        out.extend_from_slice(&tri);
        for &v in &tri {
            live[v as usize] -= 1;
            if let Some(p) = cache.iter().position(|&c| c == v) {
                cache.remove(p);
            }
            cache.insert(0, v);
        }
        let evicted: Vec<u32> = if cache.len() > CACHE {
            cache.split_off(CACHE)
        } else {
            Vec::new()
        };
        for v in evicted {
            cache_pos[v as usize] = None;
            vscore[v as usize] = vertex_score(None, live[v as usize]);
        }
        for (p, &v) in cache.iter().enumerate() {
            cache_pos[v as usize] = Some(p);
            vscore[v as usize] = vertex_score(Some(p), live[v as usize]);
        }
        for &v in &cache {
            for &tt in &adj[offsets[v as usize]..offsets[v as usize + 1]] {
                let tt = tt as usize;
                if !emitted[tt] {
                    tscore[tt] = tri_score(tt, &vscore);
                }
            }
        }
    }
    out
}

/// The cache [`optimize_overdraw`] measures clusters against.
const OVERDRAW_CACHE: u32 = 16;

/// Marks `a`, `b` and `c` used at `timestamp` and returns how many missed a FIFO cache of
/// [`OVERDRAW_CACHE`] (a vertex hits when it was used within the last that many misses).
fn update_cache(tri: &[u32], stamps: &mut [u32], timestamp: &mut u32) -> u32 {
    let mut misses = 0;
    for &v in tri {
        if timestamp.wrapping_sub(stamps[v as usize]) > OVERDRAW_CACHE {
            stamps[v as usize] = *timestamp;
            *timestamp = timestamp.wrapping_add(1);
            misses += 1;
        }
    }
    misses
}

/// Reorders cache-optimized triangles to draw outward-facing patches first, so the depth test
/// rejects more of what is behind them, while keeping each patch's cache efficiency within
/// `threshold` (1.05 = 5% worse ACMR) of its own.
///
/// `indices` is cut where all three vertices of a triangle miss the cache (a new strip), each
/// part is cut again wherever its running ACMR reaches `threshold` times its own, and the
/// patches are sorted by how far their area-weighted centroid lies along their average normal
/// from the mesh's centroid, outermost first.
pub fn optimize_overdraw(indices: &[u32], positions: &[Vec3], threshold: f32) -> Vec<u32> {
    let faces = indices.len() / 3;
    if faces == 0 {
        return indices.to_vec();
    }
    let tri = |i: usize| &indices[3 * i..3 * i + 3];
    let mut stamps = vec![0u32; positions.len()];
    // Hard boundaries: triangles whose three vertices all miss.
    let mut timestamp = OVERDRAW_CACHE + 1;
    let mut hard = Vec::new();
    for i in 0..faces {
        if update_cache(tri(i), &mut stamps, &mut timestamp) == 3 || i == 0 {
            hard.push(i);
        }
    }
    // Soft boundaries within each: cut as soon as the running ACMR is good enough, dropping
    // the last cut so the incomplete tail joins the cluster before it.
    stamps.fill(0);
    let mut timestamp = 0u32;
    let mut clusters = Vec::new();
    for (k, &start) in hard.iter().enumerate() {
        let end = hard.get(k + 1).copied().unwrap_or(faces);
        timestamp = timestamp.wrapping_add(OVERDRAW_CACHE + 1);
        let misses: u32 = (start..end)
            .map(|i| update_cache(tri(i), &mut stamps, &mut timestamp))
            .sum();
        let limit = threshold * misses as f32 / (end - start) as f32;
        clusters.push(start);
        timestamp = timestamp.wrapping_add(OVERDRAW_CACHE + 1);
        let (mut run_misses, mut run_faces) = (0u32, 0u32);
        for i in start..end {
            run_misses += update_cache(tri(i), &mut stamps, &mut timestamp);
            run_faces += 1;
            if run_misses as f32 / run_faces as f32 <= limit {
                clusters.push(i + 1);
                timestamp = timestamp.wrapping_add(OVERDRAW_CACHE + 1);
                (run_misses, run_faces) = (0, 0);
            }
        }
        if clusters.last() != Some(&start) {
            clusters.pop();
        }
    }
    // Sort key: the cluster centroid's offset from the mesh centroid along its normal.
    let p = |v: u32| positions[v as usize];
    let centroid = indices.iter().map(|&v| p(v)).sum::<Vec3>() / indices.len() as f32;
    let range = |k: usize| clusters[k]..clusters.get(k + 1).copied().unwrap_or(faces);
    let key: Vec<f32> = (0..clusters.len())
        .map(|k| {
            let (mut area, mut center, mut normal) = (0.0, Vec3::ZERO, Vec3::ZERO);
            for i in range(k) {
                let [a, b, c] = [p(tri(i)[0]), p(tri(i)[1]), p(tri(i)[2])];
                let n = (b - a).cross(c - a);
                let ar = n.length();
                center += (a + b + c) * (ar / 3.0);
                normal += n;
                area += ar;
            }
            let center = if area > 0.0 { center / area } else { center };
            (center - centroid).dot(normal.normalize_or_zero())
        })
        .collect();
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    order
        .into_iter()
        .flat_map(|k| range(k).flat_map(|i| tri(i).iter().copied()))
        .collect()
}

/// Reorders triangles for the vertex cache and then for overdraw (each patch within 5% of its
/// cache efficiency), then vertices by first use (vertex fetch). Material ids follow their
/// triangles.
pub fn optimize_mesh(mesh: &mut Mesh) {
    // Optimize per material run so material ids stay attached to triangles.
    let tri_count = mesh.triangle_count();
    let mut order: Vec<usize> = (0..tri_count).collect();
    order.sort_by_key(|&t| mesh.material_ids[t]);
    let mut new_idx = Vec::with_capacity(mesh.indices.len());
    let mut new_mats = Vec::with_capacity(tri_count);
    let mut start = 0;
    while start < order.len() {
        let mat = mesh.material_ids[order[start]];
        let end = order[start..]
            .iter()
            .position(|&t| mesh.material_ids[t] != mat)
            .map_or(order.len(), |p| start + p);
        let run: Vec<u32> = order[start..end]
            .iter()
            .flat_map(|&t| mesh.triangle(t))
            .collect();
        let opt = optimize_vertex_cache(&run, mesh.vertex_count());
        let opt = optimize_overdraw(&opt, &mesh.positions, 1.05);
        new_mats.extend(std::iter::repeat_n(mat, opt.len() / 3));
        new_idx.extend(opt);
        start = end;
    }
    mesh.indices = new_idx;
    mesh.material_ids = new_mats;
    mesh.compact();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.push(Vec3::new(x as f32, 0.0, y as f32));
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                idx.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        // Shuffle triangles deterministically to make the input cache-hostile.
        let mut tris: Vec<[u32; 3]> = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let mut r = crate::rng::Pcg32::new(5);
        for i in (1..tris.len()).rev() {
            let j = r.below(i as u32 + 1) as usize;
            tris.swap(i, j);
        }
        Mesh::new(pos, tris.concat())
    }

    #[test]
    fn forsyth_improves_acmr() {
        let mut m = grid(40);
        let before = acmr(&m.indices, 16);
        optimize_mesh(&mut m);
        let after = acmr(&m.indices, 16);
        assert!(after < before * 0.5, "before {before} after {after}");
        assert!(after < 0.9, "after {after}");
        m.validate().unwrap();
    }

    /// Two nested shells: the outer one should come first, and the reorder must permute
    /// triangles without losing cache efficiency.
    #[test]
    fn overdraw_draws_the_outside_first() {
        let mut inner = crate::mesh::tests::cube();
        let mut outer = crate::mesh::tests::cube();
        outer.transform(glam::Mat4::from_scale(Vec3::splat(3.0)));
        inner.append(&outer, 0);
        let n = inner.vertex_count();
        let cached = optimize_vertex_cache(&inner.indices, n);
        let out = optimize_overdraw(&cached, &inner.positions, 1.05);
        let tris = |v: &[u32]| {
            let mut t: Vec<[u32; 3]> = v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            t.sort();
            t
        };
        assert_eq!(tris(&out), tris(&cached));
        let first = out[0] as usize;
        assert!(
            inner.positions[first].abs().max_element() > 1.0,
            "{:?}",
            inner.positions[first]
        );

        let g = grid(30);
        let cached = optimize_vertex_cache(&g.indices, g.vertex_count());
        let out = optimize_overdraw(&cached, &g.positions, 1.05);
        assert_eq!(tris(&out), tris(&cached));
        let (a, b) = (acmr(&cached, 16), acmr(&out, 16));
        assert!(b <= a * 1.25, "{a} -> {b}");
    }
}
