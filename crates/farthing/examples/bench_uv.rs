//! UV timing + output hash: `cargo run --release -p farthing --example bench_uv -- [-n RUNS] files...`
use farthing::uv::{parameterize, ParamOptions};
use std::time::Instant;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut runs = 3;
    if args.first().is_some_and(|a| a == "-n") {
        runs = args[1].parse().unwrap();
        args.drain(..2);
    }
    let mut total = 0.0;
    for a in &args {
        let (mut m, _) = farthing::io::load(a).expect("load").flatten();
        if m.normals.is_none() {
            m.compute_normals(60.0);
        }
        m.uvs.clear();
        let mut best = f64::INFINITY;
        let mut line = String::new();
        for _ in 0..runs {
            let t = Instant::now();
            let r = parameterize(
                &m,
                &ParamOptions {
                    atlas_size: 1024,
                    padding: 2,
                    ..Default::default()
                },
            )
            .unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            let mut h = 0xcbf29ce484222325u64;
            for uv in &r.mesh.uvs[0] {
                for v in uv.to_array() {
                    h ^= v.to_bits() as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            line = format!(
                "charts {:>4} stretch {:.4} util {:.3} hash {h:016x}",
                r.charts,
                r.mean_stretch + 1.0,
                r.utilization
            );
        }
        total += best;
        let name = std::path::Path::new(a)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        println!("{name:<14} {:>8.1} ms  {line}", best * 1e3);
    }
    println!("TOTAL {:.1} ms", total * 1e3);
}
