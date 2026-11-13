//! GPU-friendly ordering: vertex-cache optimization (Forsyth's linear-speed algorithm),
//! vertex-fetch ordering, and ACMR / ATVR analysis.

use crate::Mesh;

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

/// Reorders triangles for the vertex cache, then vertices by first use (vertex fetch).
/// Material ids follow their triangles.
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
    use glam::Vec3;

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
}
