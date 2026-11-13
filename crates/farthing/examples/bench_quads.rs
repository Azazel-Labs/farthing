//! Quad reduction timing and error on bumpy quad spheres of growing size.
use glam::Vec3;
use farthing::quad::QuadMesh;
use farthing::reduce::{reduce_quads, QuadReduceOptions};
use std::collections::HashMap;
use std::time::Instant;

fn sphere(n: u32) -> QuadMesh {
    let mut m = QuadMesh::default();
    let mut index: HashMap<[i32; 3], u32> = HashMap::new();
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    for (nrm, u, v) in faces {
        let mut vid = |i: u32, j: u32, m: &mut QuadMesh| {
            let p =
                nrm + u * (2.0 * i as f32 / n as f32 - 1.0) + v * (2.0 * j as f32 / n as f32 - 1.0);
            let key = [
                (p.x * 100000.0).round() as i32,
                (p.y * 100000.0).round() as i32,
                (p.z * 100000.0).round() as i32,
            ];
            *index.entry(key).or_insert_with(|| {
                let d = p.normalize();
                let bump =
                    1.0 + 0.08 * (d.x * 7.0).sin() * (d.y * 5.0).cos() + 0.03 * (d.z * 13.0).sin();
                m.positions.push(d * bump);
                m.positions.len() as u32 - 1
            })
        };
        for j in 0..n {
            for i in 0..n {
                let a = vid(i, j, &mut m);
                let b = vid(i + 1, j, &mut m);
                let c = vid(i + 1, j + 1, &mut m);
                let d = vid(i, j + 1, &mut m);
                m.quads.push([a, b, c, d]);
            }
        }
    }
    m
}

fn main() {
    let sizes: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    for n in if sizes.is_empty() {
        vec![20, 40, 60]
    } else {
        sizes
    } {
        let m = sphere(n);
        let t = Instant::now();
        let r = reduce_quads(
            &m,
            &QuadReduceOptions {
                target_ratio: Some(0.1),
                ..Default::default()
            },
        )
        .unwrap();
        let s = t.elapsed().as_secs_f64();
        let d = farthing::metrics::surface_distance(&m.to_mesh(), &r.mesh.to_mesh(), 20000);
        println!(
            "n {n:>4}  quads {:>7} -> {:>6}  {:>8.1} ms  hausdorff {:.3}%  mean {:.4}%",
            m.quad_count(),
            r.mesh.quad_count(),
            s * 1000.0,
            d.relative_hausdorff * 100.0,
            d.relative_mean * 100.0
        );
    }
}
