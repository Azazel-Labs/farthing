//! `diag_target file ratio...`: output triangles and error per ratio (aggressive-target diagnosis).
use farthing::reduce::{simplify, SimplifyOptions};
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut m, _) = farthing::io::load(&a[0]).unwrap().flatten();
    if m.normals.is_none() {
        m.compute_normals(60.0);
    }
    println!("input tris {} verts {}", m.triangle_count(), m.vertex_count());
    stats(&m);
    for r in &a[1..] {
        let r: f32 = r.parse().unwrap();
        let mut v = serde_json::to_value(SimplifyOptions::default()).unwrap();
        if let Ok(s) = std::env::var("LF_OPTS") {
            let over: serde_json::Value = serde_json::from_str(&s).unwrap();
            for (k, x) in over.as_object().unwrap() {
                v[k] = x.clone();
            }
        }
        let mut o: SimplifyOptions = serde_json::from_value(v).unwrap();
        o.target_triangles = Some(((m.triangle_count() as f32 * r) as usize).max(8));
        let t = std::time::Instant::now();
        let res = simplify(&m, &o).unwrap();
        if std::env::var("LF_WORST").is_ok() {
            worst(&m, &res.mesh);
        }
        println!("ratio {r}: target {:?} got {} rel err {:.4} part err {:.4} collapses {} {:.0} ms", o.target_triangles, res.mesh.triangle_count(), res.relative_error, res.error, res.collapses, t.elapsed().as_secs_f64()*1e3);
    }
}

#[allow(dead_code)]
pub fn stats(m: &farthing::Mesh) {
    use std::collections::HashMap;
    let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
    let pid: Vec<u32> = m.positions.iter().map(|p| { let n = ids.len() as u32; *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n) }).collect();
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut degenerate = 0;
    let mut seen_faces = std::collections::HashSet::new();
    let mut twins = 0;
    for t in 0..m.triangle_count() {
        let v = m.triangle(t).map(|i| pid[i as usize]);
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] { degenerate += 1; continue; }
        let mut k = v; k.sort_unstable();
        if !seen_faces.insert(k) { twins += 1; continue; }
        for k in 0..3 { let (a, b) = (v[k], v[(k + 1) % 3]); *edges.entry((a.min(b), a.max(b))).or_default() += 1; }
    }
    let border = edges.values().filter(|&&c| c == 1).count();
    let mut jdeg = vec![0u32; ids.len()];
    let mut bdeg = vec![0u32; ids.len()];
    let mut emax: HashMap<u32, usize> = HashMap::new();
    for (&(a, b), &c) in &edges {
        if c > 2 { jdeg[a as usize] += 1; jdeg[b as usize] += 1; *emax.entry(c).or_default() += 1; }
        if c == 1 { bdeg[a as usize] += 1; bdeg[b as usize] += 1; }
    }
    let mut hist: std::collections::BTreeMap<(u32, bool), usize> = Default::default();
    for v in 0..ids.len() { if jdeg[v] > 0 { *hist.entry((jdeg[v], bdeg[v] > 0)).or_default() += 1; } }
    println!("twin faces {twins}");
    println!("junction vertices by (junction degree, on border): {hist:?}; edge multiplicity {emax:?}");
    let nm = edges.values().filter(|&&c| c > 2).count();
    // parts by union find
    let mut par: Vec<u32> = (0..ids.len() as u32).collect();
    fn f(p: &mut [u32], x: u32) -> u32 { let mut r = x; while p[r as usize] != r { r = p[r as usize]; } p[x as usize] = r; r }
    for t in 0..m.triangle_count() { let v = m.triangle(t).map(|i| pid[i as usize]); let a = f(&mut par, v[0]); let b = f(&mut par, v[1]); par[a as usize] = b; let b = f(&mut par, v[1]); let c = f(&mut par, v[2]); par[b as usize] = c; }
    let mut cnt: HashMap<u32, usize> = HashMap::new();
    for t in 0..m.triangle_count() { let r = f(&mut par, pid[m.triangle(t)[0] as usize]); *cnt.entry(r).or_default() += 1; }
    let mut sizes: Vec<usize> = cnt.values().copied().collect(); sizes.sort_unstable();
    println!("positions {} edges {} border {} nonmanifold {} degenerate {} parts {} (tiny ≤12 tris: {}, median {} max {})", ids.len(), edges.len(), border, nm, degenerate, sizes.len(), sizes.iter().filter(|&&s| s <= 12).count(), sizes[sizes.len()/2], sizes.last().unwrap());
}

fn worst(input: &farthing::Mesh, out: &farthing::Mesh) {
    use glam::Vec3;
    use farthing::bvh::Bvh;
    let bvh = Bvh::build((0..input.triangle_count()).map(|t| input.triangle_positions(t)).collect());
    let b = input.bounds();
    let diag = b.extent().length();
    let mut d: Vec<(f32, usize)> = (0..out.vertex_count())
        .map(|v| (bvh.closest_point(out.positions[v], diag * 4.0).map(|c| c.distance).unwrap_or(diag), v))
        .collect();
    d.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("  bounds {:?} diag {diag:.3}", b);
    for &(e, v) in d.iter().take(3) {
        println!("  out v{v} at {:?}: {:.2}% diag", out.positions[v], 100.0 * e / diag);
    }
    let ob = Bvh::build((0..out.triangle_count()).map(|t| out.triangle_positions(t)).collect());
    let mut back: Vec<(f32, usize)> = (0..input.vertex_count())
        .map(|v| (ob.closest_point(input.positions[v], diag * 4.0).map(|c| c.distance).unwrap_or(diag), v))
        .collect();
    back.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Which input part (by connected positions) the worst ones belong to.
    for &(e, v) in back.iter().step_by(back.len() / 400 + 1).take(0) { let _ = (e, v); }
    let mut shown = 0;
    let mut last = Vec3::splat(f32::MAX);
    for &(e, v) in &back {
        let p = input.positions[v];
        if (p - last).length() < diag * 0.05 { continue; }
        last = p;
        println!("  input v{v} at {p:?}: {:.2}% diag from output", 100.0 * e / diag);
        shown += 1;
        if shown == 8 { break; }
    }
}
