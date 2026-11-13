//! Reduction throughput benchmark: `cargo run --release -p farthing --example bench_reduce [segments]`
use glam::{Vec2, Vec3};
use farthing::Mesh;
use farthing::reduce::{simplify, SeamMode, SimplifyOptions};
use std::time::Instant;

/// Noisy UV sphere with seams, normals and UVs.
fn sphere(seg: u32) -> Mesh {
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
    let seg: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let m = sphere(seg);
    println!(
        "input: {} triangles, {} vertices",
        m.triangle_count(),
        m.vertex_count()
    );
    let strict = std::env::args().nth(2).is_some_and(|a| a == "strict");
    for ratio in [0.5, 0.1] {
        let t = Instant::now();
        let mut o = SimplifyOptions::ratio(ratio);
        if strict {
            o.seams = SeamMode::Strict;
        }
        let r = simplify(&m, &o).unwrap();
        let s = t.elapsed().as_secs_f64();
        let d = farthing::metrics::surface_distance(&m, &r.mesh, 20000);
        println!(
            "ratio {ratio}: {} tris in {:.2}s ({:.0}k input tris/s), rel. Hausdorff {:.4}, mean {:.5}",
            r.mesh.triangle_count(), s, m.triangle_count() as f64 / s / 1000.0, d.relative_hausdorff, d.relative_mean
        );
    }
}
