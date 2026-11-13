//! Head-to-head benchmark: farthing vs meshoptimizer, baby_shark and xatlas.
//!
//! `cargo run --release -- reduce <assets...>` / `cargo run --release -- uv <assets...>`
//! Results are printed as a table and written to `results-<mode>.json`.
//! The xatlas UV rows need `--features xatlas` (and libclang).

use glam::{Vec2, Vec3};
use farthing::bvh::Bvh;
use farthing::metrics::surface_distance;
use farthing::rng::Pcg32;
use farthing::Mesh;
use farthing::reduce::{simplify, SimplifyOptions};
use serde_json::json;
use std::time::Instant;

fn load(path: &str) -> Mesh {
    let scene = farthing::io::load(path).expect("load");
    let (mut m, _) = scene.flatten();
    if m.normals.is_none() {
        m.compute_normals(60.0);
    }
    if std::env::var("STRIP_UV").is_ok() {
        m.uvs.clear();
        m.tangents = None;
    }
    if std::env::var("WELD").is_ok() {
        farthing::reduce::repair::weld(&mut m, 0.0);
    }
    m
}

/// Area-weighted samples with their triangle and barycentrics.
fn samples(mesh: &Mesh, n: usize, seed: u64) -> Vec<(usize, Vec3)> {
    let mut cdf = Vec::with_capacity(mesh.triangle_count());
    let mut total = 0.0f64;
    for t in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle_positions(t);
        total += 0.5 * (b - a).cross(c - a).length() as f64;
        cdf.push(total);
    }
    let mut rng = Pcg32::new(seed);
    (0..n)
        .map(|_| {
            let x = rng.f32() as f64 * total;
            let t = cdf.partition_point(|&c| c < x).min(mesh.triangle_count() - 1);
            let (mut u, mut v) = (rng.f32(), rng.f32());
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            (t, Vec3::new(1.0 - u - v, u, v))
        })
        .collect()
}

fn interp<T: Copy + std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>>(v: &[T], tri: [u32; 3], b: Vec3) -> T {
    v[tri[0] as usize] * b.x + v[tri[1] as usize] * b.y + v[tri[2] as usize] * b.z
}

/// Attribute fidelity: for points on the original, compare UV and normal against the closest
/// point on the reduced mesh. Returns (uv median, uv p90, normal mean degrees).
fn attribute_error(orig: &Mesh, red: &Mesh) -> (f32, f32, f32) {
    let bvh = Bvh::build((0..red.triangle_count()).map(|t| red.triangle_positions(t)).collect());
    let mut uv_err = Vec::new();
    let mut n_err = Vec::new();
    for (t, b) in samples(orig, 20000, 7) {
        let tri = orig.triangle(t);
        let p = interp(&orig.positions, tri, b);
        let Some(c) = bvh.closest_point(p, f32::INFINITY) else { continue };
        let rt = red.triangle(c.triangle as usize);
        if !orig.uvs.is_empty() && !red.uvs.is_empty() {
            let a: Vec2 = interp(&orig.uvs[0], tri, b);
            let r: Vec2 = interp(&red.uvs[0], rt, c.barycentrics);
            uv_err.push(a.distance(r));
        }
        if let (Some(on), Some(rn)) = (&orig.normals, &red.normals) {
            let a = interp(on, tri, b).normalize_or_zero();
            let r = interp(rn, rt, c.barycentrics).normalize_or_zero();
            let e = a.dot(r).clamp(-1.0, 1.0).acos().to_degrees();
            if std::env::var("DUMP_N").is_ok() && e > 3.0 {
                let g = (orig.triangle_positions(t)[1] - orig.triangle_positions(t)[0]).cross(orig.triangle_positions(t)[2] - orig.triangle_positions(t)[0]).normalize_or_zero();
                eprintln!("ERR {:.1} p {:.3} {:.3} {:.3} n {:.2} {:.2} {:.2} face {:.2} {:.2} {:.2} red {:.2} {:.2} {:.2} dist {:.5}", e, p.x, p.y, p.z, a.x, a.y, a.z, g.x, g.y, g.z, r.x, r.y, r.z, c.distance);
            }
            n_err.push(e);
        }
    }
    if let (Some(os), Some(rs)) = (&orig.skin, &red.skin) {
        // skin fidelity: L1/2 distance between interpolated joint-weight distributions
        let mut tot = 0.0f64;
        let mut cnt = 0;
        for (t, b) in samples(orig, 5000, 9) {
            let tri = orig.triangle(t);
            let p = interp(&orig.positions, tri, b);
            let Some(c) = bvh.closest_point(p, f32::INFINITY) else { continue };
            let rt = red.triangle(c.triangle as usize);
            let mut wa: std::collections::HashMap<u16, f32> = Default::default();
            for (k, &v) in tri.iter().enumerate() { for j in 0..4 { *wa.entry(os.joints[v as usize][j]).or_default() += os.weights[v as usize][j] * b[k]; } }
            for (k, &v) in rt.iter().enumerate() { for j in 0..4 { *wa.entry(rs.joints[v as usize][j]).or_default() -= rs.weights[v as usize][j] * c.barycentrics[k]; } }
            tot += wa.values().map(|x| x.abs() as f64).sum::<f64>() * 0.5;
            cnt += 1;
        }
        eprintln!("  skin weight error {:.4}", tot / cnt.max(1) as f64);
    }
    uv_err.sort_by(f32::total_cmp);
    if std::env::var("DEBUG_N").is_ok() {
        let mut v = n_err.clone();
        v.sort_by(f32::total_cmp);
        let q = |x: f32| v[((v.len() - 1) as f32 * x) as usize];
        let big = v.iter().filter(|&&x| x > 90.0).count() as f32 / v.len() as f32;
        eprintln!("  normal err p50 {:.2} p90 {:.2} p99 {:.2} frac>90deg {:.4}", q(0.5), q(0.9), q(0.99), big);
    }
    let pct = |v: &[f32], q: f32| if v.is_empty() { f32::NAN } else { v[((v.len() - 1) as f32 * q) as usize] };
    let nm = if n_err.is_empty() { f32::NAN } else { n_err.iter().sum::<f32>() / n_err.len() as f32 };
    (pct(&uv_err, 0.5), pct(&uv_err, 0.9), nm)
}

fn with_indices(m: &Mesh, idx: Vec<u32>) -> Mesh {
    let mut r = m.clone();
    r.material_ids = vec![0; idx.len() / 3];
    r.indices = idx;
    r.compact();
    r
}

fn meshopt_reduce(m: &Mesh, target: usize, attrs: bool, permissive: bool) -> Mesh {
    let pos: Vec<[f32; 3]> = m.positions.iter().map(|p| p.to_array()).collect();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 12) };
    let adapter = meshopt::VertexDataAdapter::new(bytes, 12, 0).unwrap();
    let mut flags = meshopt::SimplifyOptions::None;
    if permissive {
        flags |= meshopt::SimplifyOptions::Permissive;
    }
    let idx = if attrs {
        let n = m.normals.clone().unwrap_or_else(|| vec![Vec3::Z; m.vertex_count()]);
        let uv = m.uvs.first().cloned().unwrap_or_else(|| vec![Vec2::ZERO; m.vertex_count()]);
        let mut a = Vec::with_capacity(m.vertex_count() * 5);
        for i in 0..m.vertex_count() {
            a.extend_from_slice(&[n[i].x, n[i].y, n[i].z, uv[i].x, uv[i].y]);
        }
        // gltfpack-style weights: normals 0.5, UVs 1.0.
        let nwt: f32 = std::env::var("MO_NW").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
        let uwt: f32 = std::env::var("MO_UW").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let w = [nwt, nwt, nwt, uwt, uwt];
        let locks: Vec<bool> = vec![false; m.vertex_count()];
        meshopt::simplify_with_attributes_and_locks(&m.indices, &adapter, &a, &w, 20, &locks, target * 3, 1.0, flags, None)
    } else {
        meshopt::simplify(&m.indices, &adapter, target * 3, 1.0, flags, None)
    };
    with_indices(m, idx)
}

fn baby_shark_reduce(m: &Mesh, target: usize) -> Mesh {
    use baby_shark::decimation::{ConstantErrorDecimationCriteria, EdgeDecimator};
    use baby_shark::mesh::corner_table::CornerTableF;
    // Geometry only: weld by position.
    let mut map = std::collections::HashMap::new();
    let mut verts = Vec::new();
    let mut remap = Vec::with_capacity(m.vertex_count());
    for p in &m.positions {
        let k = p.to_array().map(f32::to_bits);
        let id = *map.entry(k).or_insert_with(|| {
            verts.push(nalgebra::Vector3::new(p.x, p.y, p.z));
            verts.len() - 1
        });
        remap.push(id);
    }
    let faces: Vec<usize> = m.indices.iter().map(|&i| remap[i as usize]).collect();
    let mut ct = CornerTableF::from_vertex_and_face_slices(&verts, &faces);
    let mut dec = EdgeDecimator::new()
        .decimation_criteria(ConstantErrorDecimationCriteria::new(f32::MAX))
        .min_faces_count(Some(target));
    dec.decimate(&mut ct);
    let mut vid = std::collections::HashMap::new();
    let mut pos = Vec::new();
    let mut idx = Vec::new();
    for f in ct.faces() {
        let (a, b, c) = ct.face_vertices(f);
        for v in [a, b, c] {
            let n = *vid.entry(v).or_insert_with(|| {
                let p = ct.vertex_position(v);
                pos.push(Vec3::new(p.x, p.y, p.z));
                pos.len() as u32 - 1
            });
            idx.push(n);
        }
    }
    Mesh::new(pos, idx)
}

/// Default reducer options with `LF_OPTS` (a JSON object of field overrides) applied.
fn lf_opts_env() -> SimplifyOptions {
    let mut v = serde_json::to_value(SimplifyOptions::default()).unwrap();
    if let Ok(s) = std::env::var("LF_OPTS") {
        let over: serde_json::Value = serde_json::from_str(&s).unwrap();
        for (k, x) in over.as_object().unwrap() {
            v[k] = x.clone();
        }
    }
    serde_json::from_value(v).unwrap()
}

fn reduce_mode(assets: &[String]) {
    let mut rows = Vec::new();
    let mut summary: std::collections::BTreeMap<String, Vec<(f32, f32, f32, f32, f64, bool)>> = Default::default();
    let mut vis_summary: std::collections::BTreeMap<String, Vec<f32>> = Default::default();
    println!(
        "{:<16} {:>6} {:<22} {:>8} {:>8} {:>9} {:>8} {:>9} {:>9} {:>7} {:>8}",
        "asset", "ratio", "method", "tris", "hausd%", "mean%", "vis hd%", "uv p50", "uv p90", "n deg", "ms"
    );
    for path in assets {
        let m = load(path);
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        let ratios: Vec<f32> = std::env::var("RD_RATIOS")
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![0.5, 0.2, 0.05, 0.01]);
        let only: Option<Vec<String>> = std::env::var("RD_METHODS").ok().map(|v| v.split(',').map(String::from).collect());
        for ratio in ratios {
            let target = ((m.triangle_count() as f32 * ratio) as usize).max(8);
            let methods: Vec<(&str, Box<dyn Fn() -> Mesh>)> = vec![
                ("farthing", Box::new(|| simplify(&m, &SimplifyOptions { target_triangles: Some(target), ..lf_opts_env() }).unwrap().mesh)),
                ("farthing fast", Box::new(|| simplify(&m, &SimplifyOptions { target_triangles: Some(target), quality: farthing::reduce::Quality::Fast, ..lf_opts_env() }).unwrap().mesh)),
                ("meshopt", Box::new(|| meshopt_reduce(&m, target, false, false))),
                ("meshopt attr", Box::new(|| meshopt_reduce(&m, target, true, false))),
                ("meshopt attr perm", Box::new(|| meshopt_reduce(&m, target, true, true))),
                ("baby_shark (geom)", Box::new(|| baby_shark_reduce(&m, target))),
            ];
            for (label, f) in methods {
                if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == label)) {
                    continue;
                }
                let t = Instant::now();
                let r = f();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let d = surface_distance(&m, &r, 40000);
                let vd = farthing::metrics::visible_surface_distance(&m, &r, 40000);
                let (u50, u90, nd) = attribute_error(&m, &r);
                println!(
                    "{:<16} {:>6} {:<22} {:>8} {:>8.3} {:>9.4} {:>8.3} {:>9.5} {:>9.5} {:>7.2} {:>8.1}",
                    name, ratio, label, r.triangle_count(), d.relative_hausdorff * 100.0, d.relative_mean * 100.0, vd.relative_hausdorff * 100.0, u50, u90, nd, ms
                );
                vis_summary.entry(label.to_string()).or_insert_with(Vec::new).push(vd.relative_hausdorff);
                summary.entry(label.to_string()).or_insert_with(Vec::new).push((d.relative_hausdorff, d.relative_mean, u90, nd, ms, (r.triangle_count() as f32) <= target as f32 * 1.05 + 2.0));
                rows.push(json!({"asset": name, "ratio": ratio, "method": label, "tris": r.triangle_count(),
                    "hausdorff": d.relative_hausdorff, "mean": d.relative_mean, "uv_p50": u50, "uv_p90": u90, "normal_deg": nd, "ms": ms}));
            }
        }
    }
    println!("\nSUMMARY (geometric means over all assets x ratios; lower is better)");
    println!("{:<20} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}", "method", "hausd%", "mean%", "vis hd%", "uv p90", "n deg", "ms", "reached");
    let mut summ = Vec::new();
    for (label, v) in &summary {
        let gm = |f: &dyn Fn(&(f32, f32, f32, f32, f64, bool)) -> f64| (v.iter().map(|x| f(x).max(1e-7).ln()).sum::<f64>() / v.len() as f64).exp();
        let (h, mm, u, n, t) = (gm(&|x| x.0 as f64) * 100.0, gm(&|x| x.1 as f64) * 100.0, gm(&|x| x.2 as f64), gm(&|x| x.3 as f64), gm(&|x| x.4));
        let reached = v.iter().filter(|x| x.5).count() as f32 / v.len() as f32;
        let (h, mm, u, n) = if label.contains("baby") { (h, mm, f64::NAN, f64::NAN) } else { (h, mm, u, n) };
        let vh = vis_summary.get(label).map_or(f64::NAN, |v| (v.iter().map(|x| (*x as f64).max(1e-7).ln()).sum::<f64>() / v.len() as f64).exp() * 100.0);
        println!("{:<20} {:>9.4} {:>9.5} {:>9.4} {:>9.5} {:>9.3} {:>8.2} {:>7.0}%", label, h, mm, vh, u, n, t, reached * 100.0);
        summ.push(json!({"method": label, "hausdorff_pct": h, "mean_pct": mm, "uv_p90": u, "normal_deg": n, "ms": t, "reached_target": reached}));
    }
    std::fs::write("results-reduce.json", serde_json::to_string_pretty(&json!({"rows": rows, "summary": summ})).unwrap()).unwrap();
}

/// Chart/UV quality on a common footing: L2 stretch (Sander et al.), normalized so that 1.0 is
/// isometric up to global scale; utilization = UV area / atlas area; seam length relative to
/// the mesh's total edge length.
struct UvStats {
    charts: usize,
    stretch: f32,
    utilization: f32,
    seam_ratio: f32,
}

fn uv_stats(pos: &[Vec3], uv: &[Vec2], idx: &[u32], charts: usize) -> UvStats {
    let (mut a3, mut a2, mut l2) = (0.0f64, 0.0f64, 0.0f64);
    for t in idx.chunks(3) {
        let [p0, p1, p2] = [pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]];
        let [q0, q1, q2] = [uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]];
        let area3 = 0.5 * (p1 - p0).cross(p2 - p0).length() as f64;
        let area2 = 0.5 * ((q1 - q0).perp_dot(q2 - q0)) as f64;
        if area3 <= 0.0 || area2.abs() < 1e-20 {
            continue;
        }
        let ss = (p0 * (q1.y - q2.y) + p1 * (q2.y - q0.y) + p2 * (q0.y - q1.y)) / (2.0 * area2 as f32);
        let st = (p0 * (q2.x - q1.x) + p1 * (q0.x - q2.x) + p2 * (q1.x - q0.x)) / (2.0 * area2 as f32);
        let l2t = ((ss.length_squared() + st.length_squared()) as f64 / 2.0).sqrt();
        l2 += l2t * l2t * area3;
        a3 += area3;
        a2 += area2.abs();
    }
    // Normalized L2: sqrt(sum A3 L2^2 / sum A3) * sqrt(sum A2 / sum A3); 1.0 is optimal.
    let stretch = ((l2 / a3).sqrt() * (a2 / a3).sqrt()) as f32;
    // Seams: edges whose two UV-sides differ, by 3D length.
    let mut edges: std::collections::HashMap<([u32; 3], [u32; 3]), Vec<(Vec2, Vec2)>> = Default::default();
    let key = |p: Vec3| p.to_array().map(f32::to_bits);
    let mut total_len = 0.0f64;
    for t in idx.chunks(3) {
        for e in 0..3 {
            let (i, j) = (t[e] as usize, t[(e + 1) % 3] as usize);
            let (ki, kj) = (key(pos[i]), key(pos[j]));
            let (k, uvs) = if ki < kj { ((ki, kj), (uv[i], uv[j])) } else { ((kj, ki), (uv[j], uv[i])) };
            edges.entry(k).or_default().push(uvs);
        }
    }
    let mut seam_len = 0.0f64;
    for ((a, b), sides) in &edges {
        let len = (Vec3::from_array(a.map(f32::from_bits)) - Vec3::from_array(b.map(f32::from_bits))).length() as f64;
        total_len += len;
        if sides.len() == 1 || sides.len() == 2 && (sides[0].0.distance(sides[1].0) > 1e-6 || sides[0].1.distance(sides[1].1) > 1e-6) {
            seam_len += len;
        }
    }
    let (lo, hi) = idx.iter().fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(lo, hi), &i| (lo.min(uv[i as usize]), hi.max(uv[i as usize])));
    let side = (hi - lo).max_element() as f64;
    UvStats { charts, stretch, utilization: (a2 / (side * side)) as f32, seam_ratio: (seam_len / total_len) as f32 }
}

fn uv_mode(assets: &[String]) {
    use farthing::uv::{parameterize, ParamOptions};
    let mut rows = Vec::new();
    println!("{:<16} {:<12} {:>7} {:>9} {:>8} {:>8} {:>9}", "asset", "method", "charts", "L2str", "util", "seam%", "ms");
    for path in assets {
        let mut m = load(path);
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        m.uvs.clear();
        // farthing
        let t = Instant::now();
        let r = parameterize(&m, &ParamOptions { atlas_size: 1024, padding: 2, ..Default::default() }).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let s = uv_stats(&r.mesh.positions, &r.mesh.uvs[0], &r.mesh.indices, r.charts);
        // UV_DUMP=<dir>: also write each atlas (UVs + indices) as JSON, for figures.
        let dump = |label: &str, uv: &[Vec2], idx: &[u32], s: &UvStats| {
            if let Ok(dir) = std::env::var("UV_DUMP") {
                let v = json!({"asset": name, "method": label, "charts": s.charts, "stretch": s.stretch,
                    "utilization": s.utilization, "seam_ratio": s.seam_ratio,
                    "uv": uv.iter().flat_map(|p| p.to_array()).collect::<Vec<f32>>(), "indices": idx});
                std::fs::write(format!("{dir}/{name}_{label}.json"), v.to_string()).unwrap();
            }
        };
        dump("farthing", &r.mesh.uvs[0], &r.mesh.indices, &s);
        let mut emit = |label: &str, s: UvStats, ms: f64| {
            println!(
                "{:<16} {:<12} {:>7} {:>9.4} {:>8.3} {:>8.2} {:>9.0}",
                name, label, s.charts, s.stretch, s.utilization, s.seam_ratio * 100.0, ms
            );
            rows.push(json!({"asset": name, "method": label, "charts": s.charts, "stretch": s.stretch,
                "utilization": s.utilization, "seam_ratio": s.seam_ratio, "ms": ms}));
        };
        emit("farthing", s, ms);
        // xatlas (feature `xatlas`; its bindings need libclang)
        #[cfg(feature = "xatlas")]
        {
            let t = Instant::now();
            let pos: Vec<f32> = m.positions.iter().flat_map(|p| p.to_array()).collect();
            let nrm: Vec<f32> = m.normals.as_ref().unwrap().iter().flat_map(|p| p.to_array()).collect();
            let decl = xatlas_rs_v2::MeshDecl {
                vertex_position_data: xatlas_rs_v2::MeshData::Contiguous(&pos),
                vertex_normal_data: Some(xatlas_rs_v2::MeshData::Contiguous(&nrm)),
                vertex_uv_data: None,
                face_ignore_data: None,
                face_material_data: None,
                face_vertex_count: None,
                index_data: Some(xatlas_rs_v2::IndexData::U32(&m.indices)),
                face_count: m.triangle_count() as u32,
                epsilon: 1.192_092_9e-7,
            };
            let mut atlas = xatlas_rs_v2::Xatlas::new();
            atlas.add_mesh(&decl).expect("xatlas add_mesh");
            atlas.generate(&xatlas_rs_v2::ChartOptions::default(), &xatlas_rs_v2::PackOptions { resolution: 1024, padding: 2, ..Default::default() });
            let (w, h) = (atlas.width() as f32, atlas.height() as f32);
            let charts = atlas.chart_count() as usize;
            let meshes = atlas.meshes();
            let xm = &meshes[0];
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let xp: Vec<Vec3> = xm.vertex_array.iter().map(|v| m.positions[v.xref as usize]).collect();
            let xu: Vec<Vec2> = xm.vertex_array.iter().map(|v| Vec2::new(v.uv[0] / w, v.uv[1] / h)).collect();
            let s = uv_stats(&xp, &xu, xm.index_array, charts);
            dump("xatlas", &xu, xm.index_array, &s);
            emit("xatlas", s, ms);

            // xatlas at its best: more chart-growing iterations and the largest texel density that
            // still fits one 1024² atlas (farthing searches the same way). With default options
            // xatlas picks its own density, which can overflow the requested resolution (a 1319²
            // atlas on DamagedHelmet), so the row above is rescaled. Brute-force and bilinear
            // packing are off by default: neither raised utilization (XATLAS_BRUTE=1 /
            // XATLAS_BILINEAR=1 to try them; brute force takes ~90 s per asset). XATLAS_PAD sets
            // the padding, to match farthing's measured chart spacing.
            let pad: u32 = std::env::var("XATLAS_PAD").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            let t = Instant::now();
            let mut atlas = xatlas_rs_v2::Xatlas::new();
            atlas.add_mesh(&decl).expect("xatlas add_mesh");
            let env = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            let (iters, brute) = (env("XATLAS_ITERS", 4), env("XATLAS_BRUTE", 0) == 1);
            atlas.compute_charts(&xatlas_rs_v2::ChartOptions { max_iterations: iters, ..Default::default() });
            let popts = |tpu: f32| xatlas_rs_v2::PackOptions {
                resolution: 1024, padding: pad, brute_force: brute, texels_per_unit: tpu, bilinear: env("XATLAS_BILINEAR", 0) == 1, ..Default::default()
            };
            atlas.pack_charts(&popts(0.0));
            let fits = |atlas: &mut xatlas_rs_v2::Xatlas, tpu: f32| { atlas.pack_charts(&popts(tpu)); atlas.atlas_count() == 1 };
            // A verified lower bound, then grow to a failing upper bound, then bisect.
            let mut lo = atlas.texels_per_unit();
            while !fits(&mut atlas, lo) { lo *= 0.9; }
            let mut hi = lo * 1.1;
            while fits(&mut atlas, hi) { lo = hi; hi *= 1.1; }
            while hi / lo > 1.003 {
                let mid = 0.5 * (lo + hi);
                if fits(&mut atlas, mid) { lo = mid } else { hi = mid }
            }
            // Packing is not monotonic in density: also try a range above the bisection result.
            for k in 1..=25 {
                let tpu = lo * (1.0 + 0.01 * k as f32);
                if fits(&mut atlas, tpu) { lo = tpu; }
            }
            assert!(fits(&mut atlas, lo), "xatlas: final density does not fit one atlas");
            let (w, h) = (atlas.width() as f32, atlas.height() as f32);
            let charts = atlas.chart_count() as usize;
            let meshes = atlas.meshes();
            let xm = &meshes[0];
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let xp: Vec<Vec3> = xm.vertex_array.iter().map(|v| m.positions[v.xref as usize]).collect();
            let xu: Vec<Vec2> = xm.vertex_array.iter().map(|v| Vec2::new(v.uv[0] / w, v.uv[1] / h)).collect();
            let s = uv_stats(&xp, &xu, xm.index_array, charts);
            dump("xatlas_best", &xu, xm.index_array, &s);
            emit("xatlas_best", s, ms);
        }
    }
    std::fs::write("results-uv.json", serde_json::to_string_pretty(&rows).unwrap()).unwrap();
}

/// Fraction of edges not shared by exactly two triangles (0 = closed 2-manifold).
fn open_edge_fraction(m: &Mesh) -> f32 {
    let mut e: Vec<([u32; 3], [u32; 3])> = Vec::new();
    for t in 0..m.triangle_count() {
        let p = m.triangle_positions(t).map(|v| v.to_array().map(f32::to_bits));
        for k in 0..3 {
            let (a, b) = (p[k], p[(k + 1) % 3]);
            e.push(if a < b { (a, b) } else { (b, a) });
        }
    }
    e.sort_unstable();
    let (mut bad, mut total, mut i) = (0usize, 0usize, 0usize);
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j] == e[i] {
            j += 1;
        }
        total += 1;
        if j - i != 2 {
            bad += 1;
        }
        i = j;
    }
    bad as f32 / total.max(1) as f32
}

/// `remesh <assets...>`: voxel proxies at the same voxel size and final triangle budget.
/// farthing: `remesh` (surface nets + projection + reduction). baby_shark: voxel remesher (dual
/// contouring and marching cubes) followed by its edge decimator to the same budget.
fn remesh_mode(assets: &[String]) {
    let res: usize = std::env::var("RM_RES").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let budget = 3000usize;
    println!("{:<16} {:<22} {:>7} {:>8} {:>9} {:>8} {:>8}", "asset", "method", "tris", "hausd%", "mean%", "open%", "ms");
    let mut rows = Vec::new();
    for path in assets {
        let scene = farthing::io::load(path).expect("load");
        let (m, _) = scene.flatten();
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        let ext = m.bounds().max - m.bounds().min;
        let voxel = ext.max_element() / res as f32;
        let mut emit = |label: &str, r: &Mesh, ms: f64| {
            let d = surface_distance(&m, r, 40000);
            if std::env::var("HD_DBG").is_ok() { eprintln!("  {label}: fwd {:.3}% bwd {:.3}%", d.forward_max / d.diagonal * 100.0, d.backward_max / d.diagonal * 100.0); }
            let open = open_edge_fraction(r);
            println!("{:<16} {:<22} {:>7} {:>8.3} {:>9.4} {:>8.2} {:>8.0}", name, label, r.triangle_count(), d.relative_hausdorff * 100.0, d.relative_mean * 100.0, open * 100.0, ms);
            rows.push(json!({"asset": name, "method": label, "tris": r.triangle_count(), "hausdorff": d.relative_hausdorff, "mean": d.relative_mean, "open_edges": open, "ms": ms}));
        };
        let t = Instant::now();
        let mut ro: farthing::voxel::RemeshOptions = farthing::voxel::RemeshOptions { resolution: res, target_triangles: Some(budget), ..Default::default() };
        if let Ok(v) = std::env::var("RM_CLOSE") { ro.close_gaps = v.parse().unwrap(); }
        if let Ok(v) = std::env::var("RM_SMOOTH") { ro.smooth_iterations = v.parse().unwrap(); }
        if std::env::var("RM_NOSIMP").is_ok() { ro.target_triangles = None; }
        if std::env::var("RM_NOPROJ").is_ok() { ro.project_to_surface = false; }
        if std::env::var("RM_NOSUB").is_ok() { ro.subvoxel = false; }
        let lf = farthing::voxel::remesh(&scene, &ro);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        match lf {
            Ok(r) => emit("farthing", &r.mesh, ms),
            Err(e) => println!("{name} farthing failed: {e}"),
        }
        if std::env::var("RM_LFONLY").is_ok() { continue; }
        let mut bs_tris = usize::MAX;
        for (label, method) in [("baby_shark DC+decimate", baby_shark::remeshing::voxel::MeshingMethod::FeaturePreserving), ("baby_shark MC+decimate", baby_shark::remeshing::voxel::MeshingMethod::Manifold)] {
            let t = Instant::now();
            let soup: baby_shark::mesh::polygon_soup::data_structure::PolygonSoup<f32> = baby_shark::mesh::polygon_soup::data_structure::PolygonSoup::from_triangles_soup(
                (0..m.triangle_count()).flat_map(|t| m.triangle_positions(t)).map(|p| nalgebra::Vector3::new(p.x, p.y, p.z)),
            );
            let mut remesher = baby_shark::remeshing::voxel::VoxelRemesher::default().with_voxel_size(voxel).with_meshing_method(method);
            let Some(out) = remesher.remesh(&soup) else {
                println!("{name} {label} failed");
                continue;
            };
            let mut pos = Vec::new();
            let mut idx = Vec::new();
            for v in out.vertices() {
                idx.push(pos.len() as u32);
                pos.push(Vec3::new(v.x, v.y, v.z));
            }
            let mut raw = Mesh::new(pos, idx);
            farthing::reduce::repair::weld(&mut raw, 0.0);
            let Ok(reduced) = std::panic::catch_unwind(|| baby_shark_reduce(&raw, budget)) else {
                println!("{name} {label}: decimator panicked");
                continue;
            };
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            bs_tris = bs_tris.min(reduced.triangle_count());
            emit(label, &reduced, ms);
        }
        // baby_shark's decimator stops short of the budget: also compare at its count.
        if bs_tris < budget {
            let t = Instant::now();
            ro.target_triangles = Some(bs_tris);
            if let Ok(r) = farthing::voxel::remesh(&scene, &ro) {
                emit("farthing (matched)", &r.mesh, t.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    std::fs::write("results-remesh.json", serde_json::to_string_pretty(&rows).unwrap()).unwrap();
}

/// `export <asset.glb> <ratio> <outdir>`: per mesh, reduces with meshoptimizer (attribute-aware,
/// strict and permissive), then with farthing to exactly the triangle count meshoptimizer reached,
/// and writes each variant as a glb with the original material slots (for image-space scoring).
fn export_mode(args: &[String]) {
    let (path, ratio, out) = (&args[0], args[1].parse::<f32>().unwrap(), std::path::Path::new(&args[2]));
    std::fs::create_dir_all(out).unwrap();
    let (scene, _) = farthing::io::load_gltf(path).expect("load");
    let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
    for (label, perm) in [("attr", false), ("perm", true)] {
        let mut mo = scene.clone();
        let mut lf = scene.clone();
        for (i, m) in scene.meshes.iter().enumerate() {
            let mut m = m.clone();
            if m.normals.is_none() {
                m.compute_normals(60.0);
            }
            let target = ((m.triangle_count() as f32 * ratio) as usize).max(4);
            let r = meshopt_reduce(&m, target, true, perm);
            let matched = r.triangle_count();
            mo.meshes[i] = r;
            lf.meshes[i] = simplify(&m, &lf_opts(matched)).unwrap().mesh;
        }
        let tex = farthing::io::ExportTextures::default();
        farthing::io::save_glb(&mo, out.join(format!("{name}_{ratio}_mo_{label}.glb")), &tex).unwrap();
        farthing::io::save_glb(&lf, out.join(format!("{name}_{ratio}_lf_{label}.glb")), &tex).unwrap();
        println!("{name} {ratio} {label}: meshopt {} tris, farthing {} tris", mo.triangle_count(), lf.triangle_count());
    }
    // Sanity check of the round trip: the original written the same way.
    farthing::io::save_glb(&scene, out.join(format!("{name}_orig_roundtrip.glb")), &farthing::io::ExportTextures::default()).unwrap();
}

fn lf_opts(target: usize) -> SimplifyOptions {
    let mut v = serde_json::to_value(SimplifyOptions::default()).unwrap();
    if let Ok(e) = std::env::var("LF_OPTS") {
        let patch: serde_json::Value = serde_json::from_str(&e).expect("LF_OPTS json");
        for (k, x) in patch.as_object().unwrap() {
            v[k] = x.clone();
        }
    }
    let mut o: SimplifyOptions = serde_json::from_value(v).unwrap();
    o.target_triangles = Some(target);
    o
}

/// Quick A/B: farthing with LF_OPTS vs meshopt, geometry + attribute metrics.
fn exp_mode(assets: &[String]) {
    let ratios: Vec<f32> = std::env::var("RATIOS").map(|r| r.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0.5, 0.2]);
    let mo = std::env::var("MO").is_ok();
    let mut g = [[0.0f64; 5]; 2];
    let mut n = 0;
    let acc = |g: &mut [f64; 5], h: f32, m: f32, u: f32, nd: f32, ms: f64| {
        for (i, x) in [h as f64, m as f64, (u as f64).max(1e-6), (nd as f64).max(1e-3), ms].iter().enumerate() {
            g[i] += x.ln();
        }
    };
    for path in assets {
        let m = load(path);
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        for &ratio in &ratios {
            let target = ((m.triangle_count() as f32 * ratio) as usize).max(8);
            let t = Instant::now();
            let r = simplify(&m, &lf_opts(target)).unwrap().mesh;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let d = surface_distance(&m, &r, 40000);
            let (u50, u90, nd) = attribute_error(&m, &r);
            if std::env::var("HD_WHERE").is_ok() {
                let bvh = Bvh::build((0..r.triangle_count()).map(|t| r.triangle_positions(t)).collect());
                // seam / border classification of original vertices by position
                let mut by_pos: std::collections::HashMap<[u32; 3], Vec<u32>> = Default::default();
                for (i, p) in m.positions.iter().enumerate() { by_pos.entry(p.to_array().map(f32::to_bits)).or_default().push(i as u32); }
                let mut ec: std::collections::HashMap<([u32;3],[u32;3]), u32> = Default::default();
                for t in 0..m.triangle_count() { let tr = m.triangle(t); for e in 0..3 { let (a,b) = (m.positions[tr[e] as usize].to_array().map(f32::to_bits), m.positions[tr[(e+1)%3] as usize].to_array().map(f32::to_bits)); *ec.entry(if a<b {(a,b)} else {(b,a)}).or_default() += 1; } }
                let mut border_pos = std::collections::HashSet::new();
                for (&(a,b), &c) in &ec { if c == 1 { border_pos.insert(a); border_pos.insert(b); } }
                let mut worst: Vec<(f32, Vec3, usize)> = samples(&m, 40000, 3).into_iter().map(|(t, b)| { let p = interp(&m.positions, m.triangle(t), b); (bvh.closest_point(p, f32::INFINITY).unwrap().distance, p, t) }).collect();
                worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                for (d, p, t) in worst.iter().take(5) {
                    let tr = m.triangle(*t);
                    let seam = tr.iter().filter(|&&v| by_pos[&m.positions[v as usize].to_array().map(f32::to_bits)].len() > 1).count();
                    let bord = tr.iter().filter(|&&v| border_pos.contains(&m.positions[v as usize].to_array().map(f32::to_bits))).count();
                    let area = m.triangle_positions(*t); let area = (area[1]-area[0]).cross(area[2]-area[0]).length()*0.5;
                    eprintln!("  worst {:.3}% at {:?} tri {} seamverts {} borderverts {} area/avg {:.2}", d / d.max(1e-9) * 0.0 + d / m.bounds().diagonal() * 100.0, p, t, seam, bord, area / (m.surface_area() / m.triangle_count() as f32));
                }
            }
            if std::env::var("HD_DBG").is_ok() {
                eprintln!("  fwd {:.3}% bwd {:.3}%", d.forward_max / d.diagonal * 100.0, d.backward_max / d.diagonal * 100.0);
            }
            acc(&mut g[0], d.relative_hausdorff, d.relative_mean, u90, nd, ms);
            n += 1;
            print!("{:<14} {:>5} lf {:>6} h {:>7.3} m {:>7.4} uv {:>8.5}/{:>7.4} n {:>6.2} {:>6.1}ms", name, ratio, r.triangle_count(), d.relative_hausdorff*100.0, d.relative_mean*100.0, u50, u90, nd, ms);
            if mo {
                let t = Instant::now();
                let r = meshopt_reduce(&m, target, true, std::env::var("MOPERM").is_ok());
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let d = surface_distance(&m, &r, 40000);
                let (u50, u90, nd) = attribute_error(&m, &r);
                acc(&mut g[1], d.relative_hausdorff, d.relative_mean, u90, nd, ms);
                print!(" | mo {:>6} h {:>7.3} m {:>7.4} uv {:>8.5}/{:>7.4} n {:>6.2} {:>6.1}ms", r.triangle_count(), d.relative_hausdorff*100.0, d.relative_mean*100.0, u50, u90, nd, ms);
            }
            println!();
        }
    }
    for (label, g) in [("lf", g[0]), ("mo", g[1])] {
        if label == "mo" && !mo { continue; }
        let e = |i: usize| (g[i] / n as f64).exp();
        println!("GEOMEAN {label}: hausd {:.4}%  mean {:.5}%  uv90 {:.5}  normal {:.3}deg  time {:.2}ms", e(0) * 100.0, e(1) * 100.0, e(2), e(3), e(4));
    }
}

mod pack_bench;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("reduce") => reduce_mode(&args[1..]),
        Some("uv") => uv_mode(&args[1..]),
        Some("exp") => exp_mode(&args[1..]),
        Some("parts") => parts_mode(&args[1..]),
        Some("export") => export_mode(&args[1..]),
        Some("remesh") => remesh_mode(&args[1..]),
        Some("rmdiag") => rmdiag_mode(&args[1..]),
        Some("pack") => pack_bench::pack_mode(&args[1..]),
        Some("nrm") => {
            for path in &args[1..] {
                let m = load(path);
                let mut c = m.clone();
                c.compute_normals(180.0);
                // angle between stored normals and smooth geometric normals, per vertex
                let (on, gn) = (m.normals.as_ref().unwrap(), c.normals.as_ref().unwrap());
                let mut a: Vec<f32> = (0..m.vertex_count()).map(|i| on[i].normalize_or_zero().dot(gn[i]).clamp(-1.0, 1.0).acos().to_degrees()).collect();
                a.sort_by(f32::total_cmp);
                let q = |x: f32| a[((a.len() - 1) as f32 * x) as usize];
                println!("{path}: stored-vs-geometric normal deg p50 {:.2} p90 {:.2} p99 {:.2}", q(0.5), q(0.9), q(0.99));
            }
        }
        _ => eprintln!("usage: bench-compare reduce|uv <assets...>"),
    }
}

/// `parts <asset> <ratio>`: per input part (connected by position), triangles kept and the
/// one-sided distance from the part to each LOD (farthing, meshopt attr perm).
fn parts_mode(args: &[String]) {
    let m = load(&args[0]);
    let ratio: f32 = args[1].parse().unwrap();
    let target = ((m.triangle_count() as f32 * ratio) as usize).max(8);
    let lf = simplify(&m, &lf_opts(target)).unwrap().mesh;
    let mo = meshopt_reduce(&m, target, true, true);
    // parts by welded position
    let mut ids: std::collections::HashMap<[u32; 3], usize> = Default::default();
    let pid: Vec<usize> = m.positions.iter().map(|p| { let n = ids.len(); *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n) }).collect();
    let mut par: Vec<usize> = (0..ids.len()).collect();
    fn f(p: &mut Vec<usize>, x: usize) -> usize { let mut r = x; while p[r] != r { r = p[r]; } r }
    for t in 0..m.triangle_count() { let [a, b, c] = m.triangle(t).map(|v| pid[v as usize]); let (x, y, z) = (f(&mut par, a), f(&mut par, b), f(&mut par, c)); par[x] = z; let y2 = f(&mut par, y); par[y2] = z; }
    let mut parts: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for t in 0..m.triangle_count() { parts.entry(f(&mut par, pid[m.triangle(t)[0] as usize])).or_default().push(t); }
    let diag = (m.bounds().max - m.bounds().min).length();
    let bl = farthing::bvh::Bvh::build((0..lf.triangle_count()).map(|t| lf.triangle_positions(t)).collect());
    let bm = farthing::bvh::Bvh::build((0..mo.triangle_count()).map(|t| mo.triangle_positions(t)).collect());
    let mut rows: Vec<(usize, f32, f32, f32, f32, f32)> = Vec::new();
    for ts in parts.values() {
        let mut area = 0.0; let (mut hl, mut hm, mut ml, mut mm) = (0.0f32, 0.0f32, 0.0f32, 0.0f32); let mut n = 0.0;
        for &t in ts { let [a, b, c] = m.triangle_positions(t); area += (b - a).cross(c - a).length() * 0.5;
            for w in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.34, 0.33, 0.33]] { let p = a * w[0] + b * w[1] + c * w[2];
                let dl = bl.closest_point(p, diag * 2.0).map_or(diag, |c| c.distance); let dm = bm.closest_point(p, diag * 2.0).map_or(diag, |c| c.distance);
                hl = hl.max(dl); hm = hm.max(dm); ml += dl; mm += dm; n += 1.0; } }
        rows.push((ts.len(), area, hl / diag * 100.0, hm / diag * 100.0, ml / n / diag * 100.0, mm / n / diag * 100.0));
    }
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("lf {} tris, mo {} tris, {} parts", lf.triangle_count(), mo.triangle_count(), rows.len());
    println!("{:>7} {:>10} {:>8} {:>8} {:>8} {:>8}", "tris", "area", "h lf%", "h mo%", "m lf%", "m mo%");
    for r in rows.iter().take(25) { println!("{:>7} {:>10.4} {:>8.3} {:>8.3} {:>8.4} {:>8.4}", r.0, r.1, r.2, r.3, r.4, r.5); }
}

/// `rmdiag <asset>`: where the unreduced voxel proxy is farthest from the source surface.
fn rmdiag_mode(args: &[String]) {
    use farthing::bvh::Bvh;
    let res: usize = std::env::var("RM_RES").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let scene = farthing::io::load(&args[0]).expect("load");
    let (m, _) = scene.flatten();
    let ext = m.bounds().max - m.bounds().min;
    let cell = ext.max_element() / res as f32;
    let mut ro = farthing::voxel::RemeshOptions { resolution: res, target_triangles: None, ..Default::default() };
    if let Ok(v) = std::env::var("RM_CLOSE") { ro.close_gaps = v.parse().unwrap(); }
    if std::env::var("RM_NOPROJ").is_ok() { ro.project_to_surface = false; ro.smooth_iterations = 0; }
    if let Ok(v) = std::env::var("RM_TGT") { ro.target_triangles = Some(v.parse().unwrap()); }
    if std::env::var("RM_NOSUB").is_ok() { ro.subvoxel = false; }
    let r = if std::env::var("RM_BS").is_ok() {
        let soup: baby_shark::mesh::polygon_soup::data_structure::PolygonSoup<f32> = baby_shark::mesh::polygon_soup::data_structure::PolygonSoup::from_triangles_soup(
            (0..m.triangle_count()).flat_map(|t| m.triangle_positions(t)).map(|p| nalgebra::Vector3::new(p.x, p.y, p.z)),
        );
        let method = if std::env::var("RM_BS").unwrap() == "dc" { baby_shark::remeshing::voxel::MeshingMethod::FeaturePreserving } else { baby_shark::remeshing::voxel::MeshingMethod::Manifold };
        let mut remesher = baby_shark::remeshing::voxel::VoxelRemesher::default().with_voxel_size(cell).with_meshing_method(method);
        let out = remesher.remesh(&soup).unwrap();
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        for v in out.vertices() { idx.push(pos.len() as u32); pos.push(Vec3::new(v.x, v.y, v.z)); }
        let mut raw = Mesh::new(pos, idx);
        farthing::reduce::repair::weld(&mut raw, 0.0);
        raw
    } else {
        farthing::voxel::remesh(&scene, &ro).unwrap().mesh
    };
    {
        // orientation vs ray winding: probe ±0.3 cell along each face normal
        let srcb = Bvh::build((0..m.triangle_count()).map(|t| m.triangle_positions(t)).collect());
        let dirs = [Vec3::new(0.577,0.611,0.541).normalize(), Vec3::new(-0.70,0.12,-0.70).normalize(), Vec3::new(0.23,-0.94,0.247).normalize()];
        let enclosed = |p: Vec3| dirs.iter().any(|&d| srcb.ray_winding(p, d) > 0);
        let (mut ok, mut inv, mut amb) = (0, 0, 0);
        for t in (0..m.triangle_count()).step_by(7) {
            let tr = m.triangle_positions(t);
            let c = (tr[0]+tr[1]+tr[2])/3.0; let n = (tr[1]-tr[0]).cross(tr[2]-tr[0]).normalize_or_zero();
            let (o, i) = (enclosed(c + n*0.3*cell), enclosed(c - n*0.3*cell));
            match (o, i) { (false, true) => ok += 1, (true, false) => inv += 1, _ => amb += 1 }
        }
        println!("orientation: consistent {ok} inverted {inv} ambiguous {amb}");
    }
    let d = surface_distance(&m, &r, 40000);
    println!("fwd {:.3}% bwd {:.3}% mean {:.4}%", d.forward_max / d.diagonal * 100.0, d.backward_max / d.diagonal * 100.0, d.relative_mean * 100.0);
    {
        let mut w = r.clone();
        farthing::reduce::repair::weld(&mut w, 0.0);
        // connected components by shared vertices
        let nv = w.vertex_count();
        let mut parent: Vec<usize> = (0..nv).collect();
        fn f(p: &mut [usize], mut x: usize) -> usize { while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x }
        for t in 0..w.triangle_count() { let [a, b, c] = w.triangle(t); let (ra, rb, rc) = (f(&mut parent, a as usize), f(&mut parent, b as usize), f(&mut parent, c as usize)); parent[rb] = ra; let rc2 = f(&mut parent, rc); parent[rc2] = f(&mut parent, ra); }
        let mut sizes = std::collections::HashMap::new();
        for t in 0..w.triangle_count() { let r0 = f(&mut parent, w.triangle(t)[0] as usize); *sizes.entry(r0).or_insert(0usize) += 1; }
        let mut v: Vec<usize> = sizes.values().copied().collect(); v.sort_unstable_by(|a, b| b.cmp(a));
        println!("proxy tris {} components {} sizes {:?}", w.triangle_count(), v.len(), &v[..v.len().min(12)]);
    }
    let bvh = Bvh::build((0..r.triangle_count()).map(|t| r.triangle_positions(t)).collect());
    let src = Bvh::build((0..m.triangle_count()).map(|t| m.triangle_positions(t)).collect());
    let mut rng = farthing::rng::Pcg32::new(3);
    let mut worst: Vec<(f32, Vec3, bool, usize)> = Vec::new();
    for t in 0..m.triangle_count() {
        let [a, b, c] = m.triangle_positions(t);
        for _ in 0..4 {
            let (mut u, mut v) = (rng.f32(), rng.f32());
            if u + v > 1.0 { u = 1.0 - u; v = 1.0 - v; }
            let p = a + (b - a) * u + (c - a) * v;
            let h = bvh.closest_point(p, f32::INFINITY).unwrap();
            let tri = r.triangle_positions(h.triangle as usize);
            let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
            let outside = n.dot(p - h.point) > 0.0;
            worst.push((h.distance / cell, p, outside, t));
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    {
        // proxy -> source worst points
        let mut wb: Vec<(f32, Vec3)> = Vec::new();
        let mut rng2 = farthing::rng::Pcg32::new(5);
        for t in 0..r.triangle_count() {
            let [a, b, c] = r.triangle_positions(t);
            let (mut u, mut v) = (rng2.f32(), rng2.f32());
            if u + v > 1.0 { u = 1.0 - u; v = 1.0 - v; }
            let p = a + (b - a) * u + (c - a) * v;
            let h = src.closest_point(p, f32::INFINITY).unwrap();
            let tri = m.triangle_positions(h.triangle as usize);
            let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
            wb.push((if n.dot(p - h.point) > 0.0 { 1.0 } else { -1.0 } * h.distance / cell, p));
        }
        wb.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
        let far = wb.iter().filter(|w| w.0.abs() > 0.5).count();
        let far_in = wb.iter().filter(|w| w.0 < -0.5).count();
        println!("proxy samples >0.5 cell from source: {far} (of which inside source {far_in})");
        for w in wb.iter().take(5) { println!("   bwd {:.2} cells at {:?}", w.0, w.1); }
        // source triangles near the worst proxy point
        let p0 = wb[0].1;
        let mut near: Vec<(f32, Vec3, Vec3)> = (0..m.triangle_count()).filter_map(|t| { let tr = m.triangle_positions(t); let c = (tr[0]+tr[1]+tr[2])/3.0; let d = (c-p0).length(); (d < 3.0*cell).then(|| (d/cell, c, (tr[1]-tr[0]).cross(tr[2]-tr[0]).normalize_or_zero())) }).collect();
        near.sort_by(|a,b| a.0.total_cmp(&b.0));
        for n in near.iter().take(8) { println!("     src tri {:.2} cells  c {:?} n {:?}", n.0, n.1, n.2); }
        println!("     winding at p0: {:?}", [Vec3::new(0.577,0.611,0.541), Vec3::new(-0.70,0.12,-0.70), Vec3::new(0.23,-0.94,0.247), Vec3::Y, Vec3::NEG_Y].map(|d| src.ray_winding(p0, d.normalize())));
    }
    let mut shown: Vec<Vec3> = Vec::new();
    let n_out = worst.iter().filter(|w| w.0 > 0.5 && w.2).count();
    let n_in = worst.iter().filter(|w| w.0 > 0.5 && !w.2).count();
    println!("cell {cell:.5}  samples >0.5 cell: outside proxy {n_out}, inside proxy {n_in}");
    for &(d, p, out, t) in &worst {
        if shown.iter().any(|q| (*q - p).length() < 3.0 * cell) { continue; }
        shown.push(p);
        // Local thickness: distance from p along -normal to the next source surface.
        let [a, b, c] = m.triangle_positions(t);
        let nn = (b - a).cross(c - a).normalize_or_zero();
        let back = src.intersect(p - nn * 1e-4 * cell, -nn, 0.0, 100.0 * cell).map_or(f32::INFINITY, |h| h.t / cell);
        let front = src.intersect(p + nn * 1e-4 * cell, nn, 0.0, 100.0 * cell).map_or(f32::INFINITY, |h| h.t / cell);
        println!("{d:6.2} cells  at {:?}  {}  thickness(back) {:.2}  gap(front) {:.2}", p, if out { "OUTSIDE" } else { "inside " }, back, front);
        if shown.len() >= 15 { break; }
    }
}
