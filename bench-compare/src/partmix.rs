//! `partmix <asset> <ratio> <outdir>`: the default and fast reducers on a one-mesh asset, the
//! triangles each keeps per connected part, and mixes: the default result with the parts in
//! `MIX_PARTS` (comma-separated part ids, or groups joined by `+`) taken from the fast result,
//! each written as `<name>_mix<i>.glb` for scoring.

use farthing::Mesh;
use farthing::reduce::{Quality, SimplifyOptions, SimplifyResult, simplify};

fn part_ids(m: &Mesh) -> (Vec<usize>, usize) {
    let mut ids: std::collections::HashMap<[u32; 3], usize> = Default::default();
    let pid: Vec<usize> = m
        .positions
        .iter()
        .map(|p| {
            let n = ids.len();
            *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n)
        })
        .collect();
    let mut par: Vec<usize> = (0..ids.len()).collect();
    fn f(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        r
    }
    for t in 0..m.triangle_count() {
        let [a, b, c] = m.triangle(t).map(|v| pid[v as usize]);
        let (x, z) = (f(&mut par, a), f(&mut par, c));
        par[x] = z;
        let (y, z) = (f(&mut par, b), f(&mut par, c));
        par[y] = z;
    }
    let roots: Vec<usize> = (0..m.vertex_count()).map(|v| f(&mut par, pid[v])).collect();
    let mut u = roots.clone();
    u.sort_unstable();
    u.dedup();
    (
        roots.iter().map(|r| u.binary_search(r).unwrap()).collect(),
        u.len(),
    )
}

pub fn partmix_mode(args: &[String], opts: impl Fn(usize) -> SimplifyOptions) {
    let (path, ratio, out) = (
        &args[0],
        args[1].parse::<f64>().unwrap(),
        std::path::Path::new(&args[2]),
    );
    std::fs::create_dir_all(out).unwrap();
    let (scene, _) = farthing::io::load_gltf(path).expect("load");
    let name = std::path::Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert_eq!(scene.meshes.len(), 1);
    let mut m = scene.meshes[0].clone();
    if m.normals.is_none() {
        m.compute_normals(60.0);
    }
    let target = ((m.triangle_count() as f64 * ratio).round() as usize).max(4);
    let (vpart, np) = part_ids(&m);
    if let Ok(f) = std::env::var("VPART_OUT") {
        let s: String = m
            .positions
            .iter()
            .zip(&vpart)
            .map(|(p, k)| format!("{} {} {} {k}\n", p.x, p.y, p.z))
            .collect();
        std::fs::write(f, s).unwrap();
    }
    // IMP_PARTS=<k+k+..:w>: vertex importance w on those parts (default reducer only).
    let importance = std::env::var("IMP_PARTS").ok().map(|v| {
        let (ks, w) = v.split_once(':').unwrap();
        let ks: Vec<usize> = ks.split('+').map(|x| x.parse().unwrap()).collect();
        let w: f32 = w.parse().unwrap();
        vpart
            .iter()
            .map(|k| if ks.contains(k) { w } else { 1.0 })
            .collect::<Vec<f32>>()
    });
    let run = |q: Quality| {
        simplify(
            &m,
            &SimplifyOptions {
                quality: q,
                vertex_importance: if q == Quality::Best {
                    importance.clone()
                } else {
                    None
                },
                ..opts(target)
            },
        )
        .unwrap()
    };
    // UV0_PARTS=<k+k..>: the default reducer sees those parts with flat UVs (no UV cost there);
    // their UVs are put back after.
    let best = match std::env::var("UV0_PARTS") {
        Ok(v) => {
            let ks: Vec<usize> = v.split('+').map(|x| x.parse().unwrap()).collect();
            let mut flat = m.clone();
            for set in &mut flat.uvs {
                for (u, k) in set.iter_mut().zip(&vpart) {
                    if ks.contains(k) {
                        *u = glam::Vec2::ZERO;
                    }
                }
            }
            let mut r = simplify(
                &flat,
                &SimplifyOptions {
                    vertex_importance: importance.clone(),
                    ..opts(target)
                },
            )
            .unwrap();
            r.mesh.uvs = m
                .uvs
                .iter()
                .map(|set| r.source_vertex.iter().map(|&v| set[v as usize]).collect())
                .collect();
            r
        }
        Err(_) => run(Quality::Best),
    };
    // ALT_OPTS=<json>: the second result is the default reducer with these options instead.
    let fast = match std::env::var("ALT_OPTS") {
        Ok(j) => {
            let mut v = serde_json::to_value(opts(target)).unwrap();
            let over: serde_json::Value = serde_json::from_str(&j).unwrap();
            for (k, x) in over.as_object().unwrap() {
                v[k] = x.clone();
            }
            let mut o: SimplifyOptions = serde_json::from_value(v).unwrap();
            o.target_triangles = Some(target);
            simplify(&m, &o).unwrap()
        }
        Err(_) => run(Quality::Fast),
    };
    // Per output triangle: source vertices and the part.
    let tris = |r: &SimplifyResult| -> Vec<([u32; 3], usize)> {
        (0..r.mesh.triangle_count())
            .map(|t| {
                let s = r.mesh.triangle(t).map(|v| r.source_vertex[v as usize]);
                (s, vpart[s[0] as usize])
            })
            .collect()
    };
    let (bt, ft) = (tris(&best), tris(&fast));
    let mut src = vec![0usize; np];
    let mut lo = vec![glam::Vec3::splat(f32::MAX); np];
    let mut hi = vec![glam::Vec3::splat(f32::MIN); np];
    for t in 0..m.triangle_count() {
        let k = vpart[m.triangle(t)[0] as usize];
        src[k] += 1;
        for p in m.triangle_positions(t) {
            lo[k] = lo[k].min(p);
            hi[k] = hi[k].max(p);
        }
    }
    let count = |ts: &[([u32; 3], usize)]| {
        let mut c = vec![0usize; np];
        for &(_, k) in ts {
            c[k] += 1;
        }
        c
    };
    let (bc, fc) = (count(&bt), count(&ft));
    // Border edges per part (by welded position pairs).
    let mut border = vec![0usize; np];
    {
        let mut ids: std::collections::HashMap<[u32; 3], u32> = Default::default();
        let pid: Vec<u32> = m
            .positions
            .iter()
            .map(|p| {
                let n = ids.len() as u32;
                *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n)
            })
            .collect();
        let mut e: std::collections::HashMap<(u32, u32), (usize, usize)> = Default::default();
        for t in 0..m.triangle_count() {
            let tri = m.triangle(t);
            for i in 0..3 {
                let (a, b) = (pid[tri[i] as usize], pid[tri[(i + 1) % 3] as usize]);
                e.entry((a.min(b), a.max(b)))
                    .or_insert((0, vpart[tri[0] as usize]))
                    .0 += 1;
            }
        }
        for (_, (n, k)) in e {
            if n == 1 {
                border[k] += 1;
            }
        }
    }
    println!("part src best fast center size border");
    for k in 0..np {
        let c = (lo[k] + hi[k]) * 0.5;
        println!(
            "{k} {} {} {} {:.3},{:.3},{:.3} {:.3} {}",
            src[k],
            bc[k],
            fc[k],
            c.x,
            c.y,
            c.z,
            (hi[k] - lo[k]).length(),
            border[k]
        );
    }
    let mixes: Vec<Vec<usize>> = std::env::var("MIX_PARTS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|g| g.split('+').map(|x| x.parse().unwrap()).collect())
                .collect()
        })
        .unwrap_or_default();
    let write = |file: String, from_fast: &[usize]| {
        let mut mesh = m.clone();
        mesh.indices.clear();
        mesh.material_ids.clear();
        let mut put = |r: &SimplifyResult, ts: &[([u32; 3], usize)], fast: bool| {
            for (t, &(s, k)) in ts.iter().enumerate() {
                if from_fast.contains(&k) != fast {
                    continue;
                }
                let o = r.mesh.triangle(t);
                for i in 0..3 {
                    mesh.positions[s[i] as usize] = r.mesh.positions[o[i] as usize];
                }
                mesh.indices.extend_from_slice(&s);
                mesh.material_ids.push(r.mesh.material_ids[t]);
            }
        };
        put(&best, &bt, false);
        put(&fast, &ft, true);
        mesh.compact();
        let mut s = scene.clone();
        s.meshes[0] = mesh;
        farthing::io::save_glb(&s, out.join(file), &farthing::io::ExportTextures::default())
            .unwrap();
    };
    // DUMP_PART=<k,..>: those parts' triangles (source, best, fast) as OBJ with UVs.
    if let Ok(v) = std::env::var("DUMP_PART") {
        let ks: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect();
        let dump = |file: String, mesh: &Mesh, tri_part: &dyn Fn(usize) -> usize| {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(out.join(file)).unwrap());
            for p in &mesh.positions {
                writeln!(f, "v {} {} {}", p.x, p.y, p.z).unwrap();
            }
            if let Some(uv) = mesh.uvs.first() {
                for t in uv {
                    writeln!(f, "vt {} {}", t.x, t.y).unwrap();
                }
            }
            for t in 0..mesh.triangle_count() {
                if ks.contains(&tri_part(t)) {
                    let [a, b, c] = mesh.triangle(t).map(|v| v + 1);
                    writeln!(f, "f {a}/{a} {b}/{b} {c}/{c}").unwrap();
                }
            }
        };
        dump(format!("{name}_src.obj"), &m, &|t| {
            vpart[m.triangle(t)[0] as usize]
        });
        dump(format!("{name}_best.obj"), &best.mesh, &|t| bt[t].1);
        dump(format!("{name}_fast.obj"), &fast.mesh, &|t| ft[t].1);
    }
    write(format!("{name}_mixbest.glb"), &[]);
    write(format!("{name}_mixfast.glb"), &(0..np).collect::<Vec<_>>());
    for (i, g) in mixes.iter().enumerate() {
        let tb: usize = g.iter().map(|&k| bc[k]).sum();
        let tf: usize = g.iter().map(|&k| fc[k]).sum();
        println!("mix{i} {g:?}: best {tb} fast {tf}");
        write(format!("{name}_mix{i}.glb"), g);
    }
}

/// `uvdens <assets..>`: per mesh, UV area over surface area (unit-diagonal frame).
pub fn uvdens_mode(args: &[String]) {
    for path in args {
        let (scene, _) = farthing::io::load_gltf(path).expect("load");
        let diag = scene
            .meshes
            .iter()
            .filter(|m| m.triangle_count() > 0)
            .map(|m| m.bounds().diagonal())
            .fold(0.0f32, f32::max);
        for (i, m) in scene.meshes.iter().enumerate() {
            if m.triangle_count() == 0 || m.uvs.is_empty() {
                continue;
            }
            let d = m.bounds().diagonal().max(1e-12);
            let (mut a3, mut a2) = (0.0f64, 0.0f64);
            for t in 0..m.triangle_count() {
                let [p0, p1, p2] = m.triangle_positions(t);
                a3 += ((p1 - p0).cross(p2 - p0).length() * 0.5 / (d * d)) as f64;
                let [u0, u1, u2] = m.triangle(t).map(|v| m.uvs[0][v as usize]);
                a2 += (((u1 - u0).perp_dot(u2 - u0)).abs() * 0.5) as f64;
            }
            println!(
                "{} mesh {i}: tris {} surf {:.3} uv {:.3} uv/surf {:.3} (diag {:.3} of {:.3})",
                path.rsplit('/').next().unwrap(),
                m.triangle_count(),
                a3,
                a2,
                a2 / a3,
                d,
                diag
            );
        }
    }
}
