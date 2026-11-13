//! Reducer timing + golden hashes: `cargo run --release -p farthing --example bench_assets -- [-n RUNS] files...`
//! Prints min-of-N wall time per (asset, ratio) and an FNV hash of the output mesh so layout and
//! SIMD refactors can be checked for identical behaviour.
use farthing::Mesh;
use farthing::reduce::{simplify, SimplifyOptions};
use std::time::Instant;

fn hash(m: &Mesh) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |x: u32| {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    };
    for &i in &m.indices {
        eat(i);
    }
    for p in &m.positions {
        p.to_array().iter().for_each(|v| eat(v.to_bits()));
    }
    h
}

fn sphere(seg: u32) -> Mesh {
    use glam::{Vec2, Vec3};
    let rings = seg / 2;
    let (mut pos, mut uv) = (Vec::new(), Vec::new());
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let th = v * std::f32::consts::PI;
        for s in 0..=seg {
            let u = s as f32 / seg as f32;
            let ph = if s == seg {
                0.0
            } else {
                u * std::f32::consts::TAU
            };
            let bump = 1.0 + 0.05 * (7.0 * th).sin() * (5.0 * ph).cos();
            let p = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()) * bump;
            pos.push(if r == 0 {
                Vec3::Y
            } else if r == rings {
                -Vec3::Y
            } else {
                p
            });
            uv.push(Vec2::new(u, v));
        }
    }
    let mut idx = Vec::new();
    for r in 0..rings {
        for s in 0..seg {
            let a = r * (seg + 1) + s;
            let b = a + seg + 1;
            for t in [[a, a + 1, b], [a + 1, b + 1, b]] {
                let [p0, p1, p2] = t.map(|i| pos[i as usize]);
                if (p1 - p0).cross(p2 - p0).length_squared() > 1e-14 {
                    idx.extend_from_slice(&t);
                }
            }
        }
    }
    let mut m = Mesh::new(pos, idx);
    m.uvs = vec![uv];
    m.compute_normals(180.0);
    m
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut runs = 5;
    if args.first().is_some_and(|a| a == "-n") {
        runs = args[1].parse().unwrap();
        args.drain(..2);
    }
    let no_sphere = args.first().is_some_and(|a| a == "--no-sphere");
    if no_sphere {
        args.remove(0);
    }
    let mut meshes: Vec<(String, Mesh)> = if no_sphere {
        Vec::new()
    } else {
        vec![("sphere512".into(), sphere(512))]
    };
    for a in &args {
        let (m, _) = farthing::io::load(a).expect("load").flatten();
        let mut m = m;
        if m.normals.is_none() {
            m.compute_normals(60.0);
        }
        meshes.push((
            std::path::Path::new(a)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into(),
            m,
        ));
    }
    // Run inside the pool so profilers see the whole reduction on one call stack.
    rayon::ThreadPoolBuilder::new().build().unwrap().install(|| run(&meshes, runs));
}

fn run(meshes: &[(String, Mesh)], runs: usize) {
    let mut total = 0.0;
    for (name, m) in meshes {
        for ratio in [0.25f32, 0.05] {
            let mut best = f64::INFINITY;
            let mut h = 0;
            for _ in 0..runs {
                let t = Instant::now();
                let mut o = SimplifyOptions::ratio(ratio);
                if let Ok(j) = std::env::var("LF_OPTS") {
                    // JSON object of option overrides, e.g. {"remove_parts":false}.
                    let mut v = serde_json::to_value(&o).unwrap();
                    let over: serde_json::Value = serde_json::from_str(&j).unwrap();
                    for (k, x) in over.as_object().unwrap() {
                        v[k] = x.clone();
                    }
                    o = serde_json::from_value(v).unwrap();
                }
                let r = simplify(m, &o).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
                h = hash(&r.mesh);
            }
            total += best;
            println!(
                "{name:<14} {ratio:>5} {:>8} tris  {:>9.2} ms  {:>8.0}k tris/s  hash {h:016x}",
                m.triangle_count(),
                best * 1e3,
                m.triangle_count() as f64 / best / 1e3
            );
        }
    }
    println!("TOTAL {:.1} ms", total * 1e3);
}
