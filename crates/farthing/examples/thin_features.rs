//! What reduction does to long, thin things (docs/thin-features-plan.md, phase 0):
//! `cargo run --release -p farthing --features io --example thin_features -- [--png DIR] [--keep-parts] [files...]`
//!
//! For each model (glTF files, plus three procedural ones: a lamp with a chain, a cable and a
//! beaded string; a railing; a grille), finds thin parts (median shape diameter small next to
//! their length) and assemblies (four or more similar small parts next to each other, like chain
//! links), builds a LOD chain halving the triangles each level, and measures how much of the
//! thin features' coverage each level loses: from orthographic views, at full resolution and as
//! seen where the level would switch in (its deviation one pixel).
use farthing::Mesh;
use farthing::bvh::Bvh;
use farthing::host::View;
use farthing::image::Image;
use farthing::metrics::surface_distance;
use farthing::reduce::{SimplifyOptions, lod_chain};
use glam::{Mat3, Vec3, Vec4};
use std::f32::consts::{PI, TAU};

/// Side of the measuring renders, in pixels.
const RES: u32 = 768;
/// Triangle ratio between levels.
const STEP: f32 = 0.5;
const LEVELS: i32 = 7;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut png = None;
    let mut opts = SimplifyOptions::default();
    loop {
        match args.first().map(String::as_str) {
            Some("--png") => {
                png = Some(std::path::PathBuf::from(&args[1]));
                args.drain(..2);
            }
            Some("--keep-parts") => {
                opts.remove_parts = false;
                args.remove(0);
            }
            _ => break,
        }
    }
    let mut models: Vec<(String, Mesh)> = vec![
        ("proc_lamp".into(), proc_lamp()),
        ("proc_railing".into(), proc_railing()),
        ("proc_grille".into(), proc_grille()),
    ];
    for a in &args {
        let (mut m, _) = farthing::io::load(a).expect("load").flatten();
        if m.normals.is_none() {
            m.compute_normals(60.0);
        }
        let name = std::path::Path::new(a)
            .file_stem()
            .unwrap()
            .to_string_lossy();
        models.push((name.into(), m));
    }
    for (name, m) in &models {
        measure(name, m, &opts, png.as_deref());
    }
}

// ---------------------------------------------------------------- thin features

struct Part {
    tris: Vec<u32>,
    lo: Vec3,
    hi: Vec3,
    /// Extents along the principal axes, largest first.
    ext: [f32; 3],
    /// Median shape diameter (inward ray length) and the length that area implies for a tube.
    diameter: f32,
    length: f32,
}

fn parts(m: &Mesh, bvh: &Bvh) -> Vec<Part> {
    let mut ids = std::collections::HashMap::new();
    let pid: Vec<usize> = m
        .positions
        .iter()
        .map(|p| {
            let n = ids.len();
            *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n)
        })
        .collect();
    let mut uf = Uf::new(ids.len());
    for t in 0..m.triangle_count() {
        let [a, b, c] = m.triangle(t).map(|v| pid[v as usize]);
        uf.join(a, b);
        uf.join(b, c);
    }
    let mut by_root = std::collections::BTreeMap::<usize, Vec<u32>>::new();
    for t in 0..m.triangle_count() {
        let r = uf.find(pid[m.triangle(t)[0] as usize]);
        by_root.entry(r).or_default().push(t as u32);
    }
    by_root
        .into_values()
        .map(|tris| {
            let corners = |t: u32| m.triangle(t as usize).map(|v| m.positions[v as usize]);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let (mut area, mut mean) = (0.0, Vec3::ZERO);
            for &t in &tris {
                let [a, b, c] = corners(t);
                let w = 0.5 * (b - a).cross(c - a).length();
                area += w;
                mean += w * (a + b + c) / 3.0;
                for p in [a, b, c] {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            mean /= area.max(1e-20);
            let mut cov = Mat3::ZERO;
            for &t in &tris {
                for p in corners(t) {
                    let d = p - mean;
                    cov += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
                }
            }
            let axes = principal_axes(cov);
            let mut ext = [0f32; 3];
            for (k, ax) in axes.iter().enumerate() {
                let (mut a, mut b) = (f32::MAX, f32::MIN);
                for &t in &tris {
                    for p in corners(t) {
                        a = a.min(p.dot(*ax));
                        b = b.max(p.dot(*ax));
                    }
                }
                ext[k] = b - a;
            }
            ext.sort_by(|a, b| b.total_cmp(a));
            // Shape diameter: inward rays from up to 64 triangle centers.
            let stride = (tris.len() / 64).max(1);
            let mut ds: Vec<f32> = tris
                .iter()
                .step_by(stride)
                .filter_map(|&t| {
                    let [a, b, c] = corners(t);
                    let n = (b - a).cross(c - a).normalize_or_zero();
                    let o = (a + b + c) / 3.0;
                    bvh.intersect(o - n * 1e-5 * ext[0], -n, 0.0, 2.0 * ext[0])
                        .map(|h| h.t)
                })
                .collect();
            ds.sort_by(f32::total_cmp);
            let diameter = ds.get(ds.len() / 2).copied().unwrap_or(ext[2]);
            let length = area / (PI * diameter.max(1e-9));
            Part {
                tris,
                lo,
                hi,
                ext,
                diameter,
                length,
            }
        })
        .collect()
}

/// A thin feature: one long thin part, or an assembly of small similar parts.
struct Feature {
    kind: &'static str,
    parts: usize,
    tris: Vec<u32>,
    /// Extents of the whole feature (box), and the width of its strands.
    size: f32,
    width: f32,
}

fn features(m: &Mesh, parts: &[Part], diag: f32) -> Vec<Feature> {
    let mut out = Vec::new();
    let total = m.triangle_count();
    // A tube's area over its girth gives about its length, which its box can hold; a panel's
    // gives far more (its area over its thickness), so panels don't count.
    let slender = |p: &Part| {
        p.length >= 8.0 * p.diameter
            && p.length <= 1.5 * (p.ext[0] + p.ext[1] + p.ext[2])
            && p.diameter < 0.03 * diag
            && p.tris.len() < total / 2
    };
    // Assemblies: small or slender parts within a quarter of their size of a similar part.
    let small: Vec<usize> = (0..parts.len())
        .filter(|&i| parts[i].ext[0] < 0.08 * diag || slender(&parts[i]))
        .collect();
    let mut uf = Uf::new(parts.len());
    for (k, &i) in small.iter().enumerate() {
        for &j in &small[k + 1..] {
            let (a, b) = (&parts[i], &parts[j]);
            let (sa, sb) = (a.ext[0], b.ext[0]);
            if sa.max(sb) > 3.0 * sa.min(sb) {
                continue;
            }
            let gap = (a.lo - b.hi).max(b.lo - a.hi).max(Vec3::ZERO).length();
            if gap <= 0.25 * sa.min(sb) {
                uf.join(i, j);
            }
        }
    }
    let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for &i in &small {
        groups.entry(uf.find(i)).or_default().push(i);
    }
    let mut in_assembly = vec![false; parts.len()];
    for g in groups.into_values().filter(|g| g.len() >= 4) {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let mut widths: Vec<f32> = Vec::new();
        let mut tris = Vec::new();
        for &i in &g {
            in_assembly[i] = true;
            lo = lo.min(parts[i].lo);
            hi = hi.max(parts[i].hi);
            widths.push(parts[i].ext[1]);
            tris.extend_from_slice(&parts[i].tris);
        }
        widths.sort_by(f32::total_cmp);
        let strands = g.iter().filter(|&&i| slender(&parts[i])).count();
        out.push(Feature {
            kind: if 2 * strands > g.len() {
                "lattice"
            } else {
                "assembly"
            },
            parts: g.len(),
            tris,
            size: (hi - lo).length(),
            width: widths[widths.len() / 2],
        });
    }
    for (i, p) in parts.iter().enumerate() {
        if !in_assembly[i] && slender(p) {
            out.push(Feature {
                kind: "strand",
                parts: 1,
                tris: p.tris.clone(),
                size: (p.hi - p.lo).length(),
                width: p.diameter,
            });
        }
    }
    out
}

// ---------------------------------------------------------------- measuring

fn views(center: Vec3, radius: f32, res: u32) -> Vec<View> {
    let dirs = [
        Vec3::Z,
        -Vec3::Z,
        Vec3::X,
        -Vec3::X,
        Vec3::new(1.0, -0.6, 1.0),
        Vec3::new(-1.0, -0.6, 1.0),
        Vec3::new(1.0, 0.6, -1.0),
        Vec3::new(-1.0, 0.6, -1.0),
        Vec3::new(0.3, -1.0, 0.2),
        Vec3::new(0.2, 1.0, -0.3),
    ];
    dirs.iter()
        .map(|&d| View::orthographic(center, d, Vec3::Y, radius, radius, 2.0 * radius, res, res))
        .collect()
}

/// Triangle hit per pixel.
fn render(bvh: &Bvh, v: &View) -> Vec<Option<u32>> {
    use rayon::prelude::*;
    (0..v.width * v.height)
        .into_par_iter()
        .map(|i| {
            let (o, d) = v.ray(i % v.width, i / v.width);
            bvh.intersect(o, d, 0.0, f32::INFINITY).map(|h| h.triangle)
        })
        .collect()
}

fn tris_of(m: &Mesh) -> Vec<[Vec3; 3]> {
    (0..m.triangle_count())
        .map(|t| m.triangle(t).map(|v| m.positions[v as usize]))
        .collect()
}

/// Coverage lost from the original's thin pixels, in blocks of `b`×`b` pixels: per block, the
/// thin coverage minus what the level still covers there (never below zero), over all blocks.
fn thin_loss(orig: &[Option<u32>], thin: &[bool], level: &[Option<u32>], b: u32) -> (f64, f64) {
    let n = RES / b;
    let (mut lost, mut had) = (0.0, 0.0);
    for by in 0..n {
        for bx in 0..n {
            let (mut c_thin, mut c_o, mut c_l) = (0u32, 0u32, 0u32);
            for y in by * b..(by + 1) * b {
                for x in bx * b..(bx + 1) * b {
                    let i = (y * RES + x) as usize;
                    if let Some(t) = orig[i] {
                        c_o += 1;
                        c_thin += u32::from(thin[t as usize]);
                    }
                    c_l += u32::from(level[i].is_some());
                }
            }
            had += c_thin as f64;
            lost += (c_o.saturating_sub(c_l)).min(c_thin) as f64;
        }
    }
    (lost, had)
}

/// Same for all of the original's coverage.
fn all_loss(orig: &[Option<u32>], level: &[Option<u32>], b: u32) -> (f64, f64) {
    let all = vec![true; 1 + orig.iter().flatten().max().copied().unwrap_or(0) as usize];
    thin_loss(orig, &all, level, b)
}

/// Coverage the level adds where the original had none, at `b`-pixel blocks (an inflated
/// level hides loss by spilling into the next block).
fn all_added(orig: &[Option<u32>], level: &[Option<u32>], b: u32) -> f64 {
    let n = RES / b;
    let mut added = 0.0;
    for by in 0..n {
        for bx in 0..n {
            let (mut c_o, mut c_l) = (0u32, 0u32);
            for y in by * b..(by + 1) * b {
                for x in bx * b..(bx + 1) * b {
                    let i = (y * RES + x) as usize;
                    c_o += u32::from(orig[i].is_some());
                    c_l += u32::from(level[i].is_some());
                }
            }
            added += c_l.saturating_sub(c_o) as f64;
        }
    }
    added
}

fn measure(name: &str, m: &Mesh, opts: &SimplifyOptions, png: Option<&std::path::Path>) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &m.positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let diag = (hi - lo).length();
    let (center, radius) = ((lo + hi) * 0.5, 0.5 * diag);
    let bvh = Bvh::build(tris_of(m));
    let ps = parts(m, &bvh);
    let fs = features(m, &ps, diag);
    println!(
        "\n## {name}: {} triangles, {} parts, diagonal {diag:.3}",
        m.triangle_count(),
        ps.len()
    );
    let mut thin = vec![false; m.triangle_count()];
    for f in &fs {
        println!(
            "  {:<8} {:>4} parts {:>7} tris  size {:>6.1}% diag  width {:>6.3}% diag",
            f.kind,
            f.parts,
            f.tris.len(),
            100.0 * f.size / diag,
            100.0 * f.width / diag
        );
        for &t in &f.tris {
            thin[t as usize] = true;
        }
    }
    if fs.is_empty() {
        println!("  no thin features found");
        return;
    }
    let vs = views(center, radius, RES);
    let orig: Vec<Vec<Option<u32>>> = vs.iter().map(|v| render(&bvh, v)).collect();
    let ratios: Vec<f32> = (1..=LEVELS).map(|k| STEP.powi(k)).collect();
    let chain = lod_chain(m, &ratios, opts).expect("lod_chain");
    println!(
        "  {:>5} {:>8} {:>9} {:>9} {:>11} {:>11} {:>11} {:>11}",
        "level", "tris", "deviation", "px wide", "thin lost", "thin lost", "all lost", "all added"
    );
    println!(
        "  {:>5} {:>8} {:>9} {:>9} {:>11} {:>11} {:>11} {:>11}",
        "", "", "% diag", "at switch", "full res", "at switch", "at switch", "at switch"
    );
    let mut strip = png.map(|_| Image::new(RES * (LEVELS as u32 + 1), RES, Vec4::ONE));
    for (k, l) in chain.iter().enumerate() {
        let dev = if k == 0 {
            0.0
        } else {
            surface_distance(m, &l.mesh, 20_000).hausdorff
        };
        let lb = Bvh::build(tris_of(&l.mesh));
        // Seen where the level switches in, the model is diagonal / deviation pixels wide.
        let px = if dev > 0.0 { diag / dev } else { RES as f32 };
        let b = ((RES as f32 / px).round() as u32).clamp(1, RES / 8);
        let b = (1..=b).rev().find(|d| RES.is_multiple_of(*d)).unwrap();
        let (mut full, mut at, mut had, mut all_at, mut all_had) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut added = 0.0;
        for (v, o) in vs.iter().zip(&orig) {
            let r = render(&lb, v);
            let (lf, h) = thin_loss(o, &thin, &r, 1);
            let (la, _) = thin_loss(o, &thin, &r, b);
            let (aa, ah) = all_loss(o, &r, b);
            all_at += aa;
            added += all_added(o, &r, b);
            all_had += ah;
            full += lf;
            at += la;
            had += h;
            if let (Some(img), true) = (strip.as_mut(), std::ptr::eq(v, &vs[0])) {
                for y in 0..RES {
                    for x in 0..RES {
                        let i = (y * RES + x) as usize;
                        let c = match (o[i], r[i]) {
                            (Some(t), None) if thin[t as usize] => Vec4::new(0.9, 0.1, 0.1, 1.0),
                            (Some(_), None) => Vec4::new(0.95, 0.6, 0.1, 1.0),
                            (None, Some(_)) => Vec4::new(0.1, 0.4, 0.9, 1.0),
                            (Some(t), Some(_)) if thin[t as usize] => Vec4::new(0.1, 0.1, 0.1, 1.0),
                            (Some(_), Some(_)) => Vec4::new(0.6, 0.6, 0.6, 1.0),
                            (None, None) => Vec4::ONE,
                        };
                        img.set(k as u32 * RES + x, y, c);
                    }
                }
            }
        }
        println!(
            "  {k:>5} {:>8} {:>9.3} {:>9.0} {:>10.1}% {:>10.1}% {:>10.1}% {:>10.1}%",
            l.triangles,
            100.0 * dev / diag,
            px,
            100.0 * full / had.max(1.0),
            100.0 * at / had.max(1.0),
            100.0 * all_at / all_had.max(1.0),
            100.0 * added / all_had.max(1.0)
        );
    }
    if let (Some(img), Some(dir)) = (strip, png) {
        std::fs::create_dir_all(dir).unwrap();
        img.save_png(dir.join(format!("{name}.png")), true).unwrap();
    }
}

// ---------------------------------------------------------------- procedural assets

fn push(m: &mut Mesh, pos: Vec<Vec3>, idx: Vec<u32>) {
    let mut part = Mesh::new(pos, idx);
    part.compute_normals(60.0);
    if m.positions.is_empty() {
        *m = part;
    } else {
        m.append(&part, 0);
    }
}

fn empty() -> Mesh {
    Mesh::new(Vec::new(), Vec::new())
}

/// Closed grid of `u`×`v` quads from a parametric surface periodic in both directions.
fn torus_grid(u: u32, v: u32, f: impl Fn(f32, f32) -> Vec3) -> (Vec<Vec3>, Vec<u32>) {
    let pos = (0..v)
        .flat_map(|j| (0..u).map(move |i| (i, j)))
        .map(|(i, j)| f(i as f32 / u as f32 * TAU, j as f32 / v as f32 * TAU))
        .collect();
    let mut idx = Vec::new();
    for j in 0..v {
        for i in 0..u {
            let at = |i: u32, j: u32| (j % v) * u + (i % u);
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            idx.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    (pos, idx)
}

fn sphere(m: &mut Mesh, c: Vec3, r: f32, seg: u32) {
    let rings = seg / 2;
    let mut pos = vec![c + Vec3::Y * r];
    for j in 1..rings {
        let th = j as f32 / rings as f32 * PI;
        for i in 0..seg {
            let ph = i as f32 / seg as f32 * TAU;
            pos.push(c + r * Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()));
        }
    }
    pos.push(c - Vec3::Y * r);
    let last = pos.len() as u32 - 1;
    let ring = |j: u32, i: u32| 1 + (j - 1) * seg + i % seg;
    let mut idx = Vec::new();
    for i in 0..seg {
        idx.extend_from_slice(&[0, ring(1, i + 1), ring(1, i)]);
        idx.extend_from_slice(&[last, ring(rings - 1, i), ring(rings - 1, i + 1)]);
    }
    for j in 1..rings - 1 {
        for i in 0..seg {
            let (a, b, c2, d) = (
                ring(j, i),
                ring(j, i + 1),
                ring(j + 1, i + 1),
                ring(j + 1, i),
            );
            idx.extend_from_slice(&[a, b, c2, a, c2, d]);
        }
    }
    push(m, pos, idx);
}

fn cuboid(m: &mut Mesh, lo: Vec3, hi: Vec3) {
    let p = |i: u32| {
        Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    };
    let pos = (0..8).map(p).collect();
    let idx = vec![
        0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2,
        1, 3, 7, 1, 7, 5,
    ];
    push(m, pos, idx);
}

/// 12-sided closed tube along a polyline (caps from the end centers).
fn tube(m: &mut Mesh, path: &[Vec3], r: f32) {
    let sides = 12;
    let mut pos = Vec::new();
    let mut n_prev = Vec3::X;
    for (k, &p) in path.iter().enumerate() {
        let t = (path[(k + 1).min(path.len() - 1)] - path[k.saturating_sub(1)]).normalize();
        let n = (n_prev - t * n_prev.dot(t)).normalize_or(t.any_orthonormal_vector());
        n_prev = n;
        let b = t.cross(n);
        for i in 0..sides {
            let a = i as f32 / sides as f32 * TAU;
            pos.push(p + r * (n * a.cos() + b * a.sin()));
        }
    }
    let rings = path.len() as u32;
    let mut idx = Vec::new();
    for j in 0..rings - 1 {
        for i in 0..sides {
            let at = |i: u32, j: u32| j * sides + i % sides;
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            idx.extend_from_slice(&[a, d, c, a, c, b]);
        }
    }
    let (c0, c1) = (pos.len() as u32, pos.len() as u32 + 1);
    pos.push(path[0]);
    pos.push(*path.last().unwrap());
    for i in 0..sides {
        idx.extend_from_slice(&[c0, i, (i + 1) % sides]);
        let l = (rings - 1) * sides;
        idx.extend_from_slice(&[c1, l + (i + 1) % sides, l + i]);
    }
    push(m, pos, idx);
}

/// A lamp: a sphere hanging from a chain of 40 torus links, a cable sagging to the side and a
/// string of beads.
fn proc_lamp() -> Mesh {
    let mut m = empty();
    sphere(&mut m, Vec3::ZERO, 0.3, 48);
    // Links: elongated tori, 0.06 long, wire 0.008 thick, each turned 90° from the last.
    let (pitch, wire) = (0.044, 0.004);
    for k in 0..40 {
        let y = 0.3 + 0.03 + k as f32 * pitch;
        let rot = if k % 2 == 0 { 0.0 } else { PI / 2.0 };
        let (pos, idx) = torus_grid(16, 8, |u, v| {
            let core = Vec3::new(0.014 * u.cos(), 0.026 * u.sin(), 0.0);
            let out = Vec3::new(u.cos(), u.sin(), 0.0);
            let p = core + wire * (out * v.cos() + Vec3::Z * v.sin());
            let (s, c) = rot.sin_cos();
            Vec3::new(p.x * c - p.z * s, p.y + y, p.x * s + p.z * c)
        });
        push(&mut m, pos, idx);
    }
    // Cable: sags from the lamp to a point off to the side.
    let path: Vec<Vec3> = (0..=100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Vec3::new(0.25 + 1.2 * t, 0.1 + 1.0 * t - 0.8 * (PI * t).sin(), 0.0)
        })
        .collect();
    tube(&mut m, &path, 0.008);
    // Beads: 30 spheres just overlapping, hanging from the bottom.
    for k in 0..30 {
        sphere(
            &mut m,
            Vec3::new(0.0, -0.32 - k as f32 * 0.0235, 0.0),
            0.012,
            12,
        );
    }
    m
}

/// A plinth with a railing: 12 posts and two rails.
fn proc_railing() -> Mesh {
    let mut m = empty();
    cuboid(
        &mut m,
        Vec3::new(-1.2, -0.4, -0.3),
        Vec3::new(1.2, 0.0, 0.3),
    );
    for k in 0..12 {
        let x = -1.1 + k as f32 * 0.2;
        cuboid(
            &mut m,
            Vec3::new(x - 0.01, 0.0, -0.01),
            Vec3::new(x + 0.01, 0.8, 0.01),
        );
    }
    for y in [0.4, 0.8] {
        let path: Vec<Vec3> = (0..=40)
            .map(|i| Vec3::new(-1.15 + i as f32 * 0.0575, y, 0.0))
            .collect();
        tube(&mut m, &path, 0.012);
    }
    m
}

/// A box with a grille of 24 + 24 bars over its front.
fn proc_grille() -> Mesh {
    let mut m = empty();
    cuboid(
        &mut m,
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, 0.4),
    );
    for k in 0..24 {
        let s = -0.46 + k as f32 * 0.04;
        cuboid(
            &mut m,
            Vec3::new(s - 0.004, -0.48, 0.44),
            Vec3::new(s + 0.004, 0.48, 0.448),
        );
        cuboid(
            &mut m,
            Vec3::new(-0.48, s - 0.004, 0.452),
            Vec3::new(0.48, s + 0.004, 0.46),
        );
    }
    m
}

// ---------------------------------------------------------------- helpers

struct Uf(Vec<usize>);

impl Uf {
    fn new(n: usize) -> Uf {
        Uf((0..n).collect())
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        self.0[a] = b;
    }
}

/// Eigenvectors of a symmetric 3×3 matrix (Jacobi rotations).
fn principal_axes(mut a: Mat3) -> [Vec3; 3] {
    let mut v = Mat3::IDENTITY;
    for _ in 0..24 {
        let m = a.to_cols_array_2d();
        let (mut p, mut q, mut big) = (0, 1, 0f32);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if m[j][i].abs() > big {
                (p, q, big) = (i, j, m[j][i].abs());
            }
        }
        if big < 1e-12 {
            break;
        }
        let theta = 0.5 * (m[q][q] - m[p][p]) / m[q][p];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
        let mut r = Mat3::IDENTITY.to_cols_array_2d();
        r[p][p] = c;
        r[q][q] = c;
        r[q][p] = s;
        r[p][q] = -s;
        let r = Mat3::from_cols_array_2d(&r);
        a = r.transpose() * a * r;
        v *= r;
    }
    [v.x_axis, v.y_axis, v.z_axis]
}
