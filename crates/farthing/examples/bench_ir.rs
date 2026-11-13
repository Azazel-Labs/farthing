//! One pass of each case, for instruction counting under callgrind.
use farthing::reduce::{simplify, SimplifyOptions};
fn main() {
    let mut meshes = Vec::new();
    for a in std::env::args().skip(1) {
        let (mut m, _) = farthing::io::load(&a).expect("load").flatten();
        if m.normals.is_none() {
            m.compute_normals(60.0);
        }
        meshes.push(m);
    }
    for m in &meshes {
        for ratio in [0.25f32, 0.05] {
            let r = simplify(m, &SimplifyOptions::ratio(ratio)).unwrap();
            println!("{}", r.mesh.triangle_count());
        }
    }
}
