//! Voxel proxy timing: `bench_remesh [-n runs] <assets...>` (resolution 128, 3000 triangles).
//! Prints the best of `runs` per asset and an output hash, so runs at different thread counts
//! can be checked for identical output.
use farthing::voxel::{remesh, RemeshOptions};
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
        let scene = farthing::io::load(a).expect("load");
        let name = std::path::Path::new(a)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let opts = RemeshOptions {
            resolution: 128,
            target_triangles: Some(3000),
            ..Default::default()
        };
        let (mut best, mut h) = (f64::INFINITY, 0u64);
        for _ in 0..runs {
            let t = Instant::now();
            let r = remesh(&scene, &opts).expect("remesh");
            best = best.min(t.elapsed().as_secs_f64());
            h = r
                .mesh
                .positions
                .iter()
                .flat_map(|p| p.to_array())
                .fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
                    (h ^ x.to_bits() as u64).wrapping_mul(0x0100_0000_01b3)
                });
            h = r
                .mesh
                .indices
                .iter()
                .fold(h, |h, &i| (h ^ i as u64).wrapping_mul(0x0100_0000_01b3));
        }
        total += best;
        println!("{name:<18} {:>9.1} ms  hash {h:016x}", best * 1e3);
    }
    println!("TOTAL {:.1} ms", total * 1e3);
}
