//! `clusters <assets...>`: cluster-LOD DAGs, farthing vs the meshoptimizer clusterlod scheme
//! (build_meshlets → partition_clusters → simplify each group to half with its border locked
//! → re-cluster; the structure of meshoptimizer's demo/clusterlod.h).
//!
//! For each DAG the camera is put at 4 radii and the screen-error threshold is swept; each cut
//! gets its triangle count and its *measured* surface distance to the source (sampled, in
//! pixels at a 1080p, 60° view). The table gives the triangles each method needs to stay under
//! a measured error, so a method that under-reports its error doesn't win by drawing less.

use farthing::cluster::{build_cluster_dag, Cluster, ClusterDag, ClusterOptions, Cone, LodView, Sphere};
use farthing::metrics::surface_distance;
use farthing::Mesh;
use glam::Vec3;
use serde_json::json;
use std::time::Instant;

const MAX_T: usize = 128;
const MAX_V: usize = 255;

fn weld(m: &Mesh) -> Vec<u32> {
    let mut ids: std::collections::HashMap<[u32; 3], u32> = Default::default();
    m.positions
        .iter()
        .map(|p| {
            let n = ids.len() as u32;
            *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n)
        })
        .collect()
}

fn sphere_of(m: &Mesh, idx: &[u32]) -> Sphere {
    let pts: Vec<Vec3> = idx.iter().map(|&v| m.positions[v as usize]).collect();
    Sphere::from_points(&pts)
}

fn mo_cluster(m: &Mesh, idx: Vec<u32>, level: u32, err: f32, lod: Option<Sphere>) -> Cluster {
    let b = sphere_of(m, &idx);
    Cluster {
        material_ids: vec![0; idx.len() / 3],
        indices: idx,
        level,
        bounds: b,
        cone: Cone::default(),
        lod_error: err,
        lod_bounds: lod.unwrap_or(b),
        parent_error: f32::INFINITY,
        parent_bounds: b,
        group: u32::MAX,
        parent_group: u32::MAX,
    }
}

/// The clusterlod.h scheme with meshoptimizer's own primitives.
fn meshopt_dag(m: &Mesh, attrs: bool) -> ClusterDag {
    let pos: Vec<[f32; 3]> = m.positions.iter().map(|p| p.to_array()).collect();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 12) };
    let adapter = meshopt::VertexDataAdapter::new(bytes, 12, 0).unwrap();
    let pid = weld(m);
    let npos = pid.iter().map(|&p| p as usize + 1).max().unwrap_or(0);
    let n = m.normals.clone().unwrap_or_else(|| vec![Vec3::Z; m.vertex_count()]);
    let uv = m.uvs.first().cloned().unwrap_or_else(|| vec![glam::Vec2::ZERO; m.vertex_count()]);
    let mut a = Vec::with_capacity(m.vertex_count() * 5);
    for i in 0..m.vertex_count() {
        a.extend_from_slice(&[n[i].x, n[i].y, n[i].z, uv[i].x, uv[i].y]);
    }
    let w = [0.5f32, 0.5, 0.5, 1.0, 1.0];

    let split = |idx: &[u32]| -> Vec<Vec<u32>> {
        let ml = meshopt::build_meshlets(idx, &adapter, MAX_V, MAX_T, 0.0);
        ml.iter()
            .map(|l| l.triangles.iter().map(|&t| l.vertices[t as usize]).collect())
            .collect()
    };
    let mut dag = ClusterDag::default();
    for c in split(&m.indices) {
        dag.clusters.push(mo_cluster(m, c, 0, 0.0, None));
    }
    let mut current: Vec<u32> = (0..dag.clusters.len() as u32).collect();
    let mut level = 0;
    while current.len() > 1 {
        let mut cidx = Vec::new();
        let mut counts = Vec::new();
        for &c in &current {
            cidx.extend_from_slice(&dag.clusters[c as usize].indices);
            counts.push(dag.clusters[c as usize].indices.len() as u32);
        }
        let mut part = vec![0u32; current.len()];
        let np = meshopt::partition_clusters_with_positions(&mut part, &cidx, &counts, &adapter, 16);
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); np];
        for (i, &p) in part.iter().enumerate() {
            groups[p as usize].push(current[i]);
        }
        let mut owner = vec![u32::MAX; npos];
        let mut border = vec![false; npos];
        for (g, mem) in groups.iter().enumerate() {
            for &c in mem {
                for &v in &dag.clusters[c as usize].indices {
                    let p = pid[v as usize] as usize;
                    if owner[p] == u32::MAX {
                        owner[p] = g as u32;
                    } else if owner[p] != g as u32 {
                        border[p] = true;
                    }
                }
            }
        }
        let locks: Vec<bool> = (0..m.vertex_count()).map(|v| border[pid[v] as usize]).collect();
        let mut next = Vec::new();
        let mut progress = false;
        for mem in &groups {
            let idx: Vec<u32> = mem.iter().flat_map(|&c| dag.clusters[c as usize].indices.clone()).collect();
            if mem.len() < 2 {
                next.extend_from_slice(mem);
                continue;
            }
            let target = (idx.len() / 3 / 2) * 3;
            let mut err = 0.0f32;
            let flags = meshopt::SimplifyOptions::Sparse | meshopt::SimplifyOptions::ErrorAbsolute;
            let out = if attrs {
                // clusterlod.h's attribute path: seams may move (Permissive), as farthing's do.
                let flags = flags | meshopt::SimplifyOptions::Permissive;
                meshopt::simplify_with_attributes_and_locks(&idx, &adapter, &a, &w, 20, &locks, target, f32::MAX, flags, Some(&mut err))
            } else {
                meshopt::simplify_with_locks(&idx, &adapter, &locks, target, f32::MAX, flags, Some(&mut err))
            };
            if out.len() as f32 > 0.85 * idx.len() as f32 {
                next.extend_from_slice(mem);
                continue;
            }
            progress = true;
            let gi = dag.groups.len() as u32;
            let bounds = Sphere::enclose(&mem.iter().map(|&c| dag.clusters[c as usize].lod_bounds).collect::<Vec<_>>());
            let error = mem.iter().map(|&c| dag.clusters[c as usize].lod_error).fold(err, f32::max);
            for &c in mem {
                let c = &mut dag.clusters[c as usize];
                c.parent_error = error;
                c.parent_bounds = bounds;
                c.parent_group = gi;
            }
            let mut produced = Vec::new();
            for c in split(&out) {
                let mut cl = mo_cluster(m, c, level + 1, error, Some(bounds));
                cl.group = gi;
                produced.push(dag.clusters.len() as u32);
                next.push(dag.clusters.len() as u32);
                dag.clusters.push(cl);
            }
            dag.groups.push(farthing::cluster::ClusterGroup { level, children: mem.clone(), clusters: produced, error, bounds });
        }
        if !progress {
            break;
        }
        current = next;
        level += 1;
    }
    dag
}

pub fn cluster_mode(assets: &[String]) {
    let mut rows = Vec::new();
    let targets = [0.25f32, 0.5, 1.0, 2.0];
    println!("{:<18} {:<14} {:>7} {:>8} {:>8} {:>6} {:>6} {:>8} | tris for measured mean error ≤ px {:?}", "asset", "method", "tris", "build ms", "peak MB", "levels", "roots", "fill", targets);
    for path in assets {
        let m = crate::load(path);
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        let b = m.bounds();
        let r = b.radius();
        let eye = b.center() + Vec3::new(0.3, 0.5, 0.81).normalize() * 4.0 * r;
        let ppr = 1080.0 / (2.0 * (30f32.to_radians()).tan());
        let px = ppr / (3.0 * r); // pixels per world unit at the nearest point of the bounds
        let methods: Vec<(&str, Box<dyn Fn() -> ClusterDag>)> = vec![
            ("farthing", Box::new(|| build_cluster_dag(&m, &ClusterOptions::default()).unwrap())),
            ("farthing fast", Box::new(|| build_cluster_dag(&m, &ClusterOptions::fast()).unwrap())),
            ("meshopt", Box::new(|| meshopt_dag(&m, false))),
            ("meshopt attr", Box::new(|| meshopt_dag(&m, true))),
        ];
        for (label, build) in &methods {
            if std::env::var("ONLY").is_ok_and(|o| o != *label) {
                continue;
            }
            // REPEAT=n: best of n builds (timing noise); the peak heap is the first build's.
            let reps: usize = std::env::var("REPEAT").ok().and_then(|r| r.parse().ok()).unwrap_or(1).max(1);
            let base = crate::reset_peak();
            let t0 = Instant::now();
            let mut dag = build();
            let mut ms = t0.elapsed().as_secs_f64() * 1000.0;
            let peak_mb = (crate::peak() - base) as f64 / 1048576.0;
            for _ in 1..reps {
                drop(dag);
                let t0 = Instant::now();
                dag = build();
                ms = ms.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let l0 = dag.clusters.iter().filter(|c| c.level == 0).count();
            let fill = m.triangle_count() as f32 / (l0 * MAX_T) as f32;
            // Sweep thresholds: (tris, measured mean px, measured max px).
            let mut pts: Vec<(usize, f32, f32)> = Vec::new();
            let mut thr = if std::env::var("SPEED_ONLY").is_ok() { 1e9 } else { 0.02f32 };
            while thr < 200.0 {
                let sel = dag.select(&LodView { eye, pixels_per_radian: ppr, threshold: thr });
                let cut = dag.to_mesh(&m, &sel);
                let d = surface_distance(&m, &cut, 40_000);
                pts.push((cut.triangle_count(), d.mean * px, d.hausdorff * px));
                thr *= 1.5;
            }
            let need = |e: f32| pts.iter().filter(|p| p.1 <= e).map(|p| p.0).min();
            let cells: Vec<String> = targets.iter().map(|&e| need(e).map_or("-".into(), |t| t.to_string())).collect();
            println!(
                "{:<18} {:<14} {:>7} {:>8.0} {:>8.1} {:>6} {:>6} {:>8.3} | {}",
                name, label, m.triangle_count(), ms, peak_mb, dag.levels(), dag.roots().count(), fill, cells.join(" / ")
            );
            rows.push(json!({"asset": name, "method": label, "tris": m.triangle_count(), "ms": ms, "peak_mb": peak_mb,
                "threads": rayon_threads(), "levels": dag.levels(),
                "roots": dag.roots().count(), "root_tris": dag.roots().map(|c| dag.clusters[c].triangle_count()).sum::<usize>(),
                "fill": fill, "curve": pts, "targets": targets, "need": targets.iter().map(|&e| need(e)).collect::<Vec<_>>()}));
        }
    }
    if let Ok(out) = std::env::var("JSONL") {
        let s: String = rows.iter().map(|r| r.to_string() + "\n").collect();
        std::fs::write(out, s).unwrap();
    }
}

fn rayon_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    })
}
