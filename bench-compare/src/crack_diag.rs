//! `rtcrack <asset> [ratios...]`: do meshes reduced one at a time come apart where their
//! open borders coincide in the source?
//!
//! Source border edges are edges used by one triangle after welding equal positions within
//! a mesh. One is *shared* when both its endpoints coincide in world space with border
//! vertices of one other part (a part is an instance of a mesh). After every mesh is reduced
//! on its own with the realtime reducer, each reduced border vertex and border-edge midpoint
//! is measured against the source border of its own part (drift) and, when the nearest
//! source border edge is shared, against the reduced border of the neighboring part (crack).
//! Distances are % of the scene's bounding-box diagonal.

use farthing::Mesh;
use farthing::reduce::{Quality, SimplifyOptions, simplify};
use glam::Vec3;
use rayon::prelude::*;
use std::collections::HashMap;

fn bits(p: Vec3) -> [u32; 3] {
    (p + Vec3::ZERO).to_array().map(f32::to_bits)
}

/// Border edges of a mesh in its own space: used by one triangle, positions welded.
fn border_edges(m: &Mesh) -> Vec<[Vec3; 2]> {
    let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
    let mut pos: Vec<Vec3> = Vec::new();
    let pid: Vec<u32> = m
        .positions
        .iter()
        .map(|&p| {
            *ids.entry(bits(p)).or_insert_with(|| {
                pos.push(p);
                pos.len() as u32 - 1
            })
        })
        .collect();
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for t in 0..m.triangle_count() {
        let tr = m.triangle(t);
        let c = [
            pid[tr[0] as usize],
            pid[tr[1] as usize],
            pid[tr[2] as usize],
        ];
        if c[0] == c[1] || c[1] == c[2] || c[0] == c[2] {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (c[k], c[(k + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut out: Vec<(u32, u32)> = edges
        .into_iter()
        .filter(|&(_, n)| n == 1)
        .map(|(e, _)| e)
        .collect();
    out.sort_unstable();
    out.into_iter()
        .map(|(a, b)| [pos[a as usize], pos[b as usize]])
        .collect()
}

fn seg_dist2(p: Vec3, s: &[Vec3; 2]) -> f32 {
    let d = s[1] - s[0];
    let l = d.length_squared();
    let t = if l > 0.0 {
        ((p - s[0]).dot(d) / l).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (s[0] + d * t - p).length_squared()
}

/// Segments in a uniform grid, for nearest-segment queries.
struct SegGrid<'a> {
    segs: &'a [[Vec3; 2]],
    min: Vec3,
    cell: f32,
    dims: [i32; 3],
    cells: HashMap<[i32; 3], Vec<u32>>,
}

impl<'a> SegGrid<'a> {
    fn new(segs: &'a [[Vec3; 2]]) -> Self {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let mut len = 0.0;
        for s in segs {
            lo = lo.min(s[0]).min(s[1]);
            hi = hi.max(s[0]).max(s[1]);
            len += (s[1] - s[0]).length();
        }
        let ext = (hi - lo).max_element().max(1e-20);
        // Cells about two mean segment lengths wide, at most 256 along the longest axis.
        let cell = (2.0 * len / segs.len().max(1) as f32)
            .max(ext / 256.0)
            .max(1e-20);
        let dims = ((hi - lo) / cell).to_array().map(|x| x as i32 + 1);
        let mut cells: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, s) in segs.iter().enumerate() {
            let a = ((s[0].min(s[1]) - lo) / cell).to_array().map(|x| x as i32);
            let b = ((s[0].max(s[1]) - lo) / cell).to_array().map(|x| x as i32);
            for x in a[0]..=b[0] {
                for y in a[1]..=b[1] {
                    for z in a[2]..=b[2] {
                        cells.entry([x, y, z]).or_default().push(i as u32);
                    }
                }
            }
        }
        SegGrid {
            segs,
            min: lo,
            cell,
            dims,
            cells,
        }
    }

    /// Nearest segment to `p`: (distance, index). Exact: rings of cells are searched until
    /// the best found is nearer than anything a further ring could hold.
    fn nearest(&self, p: Vec3) -> Option<(f32, u32)> {
        if self.segs.is_empty() {
            return None;
        }
        let c = ((p - self.min) / self.cell)
            .to_array()
            .map(|x| x.floor() as i32);
        // Rings below this one cannot touch the grid.
        let start = (0..3)
            .map(|k| (-c[k]).max(c[k] - (self.dims[k] - 1)).max(0))
            .max()
            .unwrap();
        let last = (0..3)
            .map(|k| c[k].max(self.dims[k] - 1 - c[k]))
            .max()
            .unwrap()
            .max(start);
        let mut best = (f32::INFINITY, u32::MAX);
        for r in start..=last {
            for x in (c[0] - r).max(0)..=(c[0] + r).min(self.dims[0] - 1) {
                for y in (c[1] - r).max(0)..=(c[1] + r).min(self.dims[1] - 1) {
                    for z in (c[2] - r).max(0)..=(c[2] + r).min(self.dims[2] - 1) {
                        let ring = (x - c[0]).abs().max((y - c[1]).abs()).max((z - c[2]).abs());
                        if ring != r {
                            continue;
                        }
                        if let Some(v) = self.cells.get(&[x, y, z]) {
                            for &i in v {
                                let d = seg_dist2(p, &self.segs[i as usize]);
                                if d < best.0 || (d == best.0 && i < best.1) {
                                    best = (d, i);
                                }
                            }
                        }
                    }
                }
            }
            // Everything in rings beyond r is at least r cells away.
            let reach = r as f32 * self.cell;
            if best.0 <= reach * reach {
                break;
            }
        }
        Some((best.0.sqrt(), best.1))
    }
}

struct Stat(Vec<f32>);

impl Stat {
    /// "n mean p95 max" in % of `diag`.
    fn line(&mut self, diag: f32) -> String {
        if self.0.is_empty() {
            return format!("{:>7} {:>8} {:>8} {:>8}", 0, "-", "-", "-");
        }
        self.0.sort_by(f32::total_cmp);
        let n = self.0.len();
        let mean = self.0.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let p95 = self.0[((n as f64 * 0.95) as usize).min(n - 1)];
        let k = 100.0 / diag;
        format!(
            "{n:>7} {:>8.4} {:>8.4} {:>8.4}",
            mean as f32 * k,
            p95 * k,
            self.0[n - 1] * k
        )
    }
}

pub fn rtcrack_mode(args: &[String]) {
    let path = &args[0];
    let ratios: Vec<f64> = if args.len() > 1 {
        args[1..].iter().map(|r| r.parse().unwrap()).collect()
    } else {
        vec![0.1, 0.03]
    };
    let (scene, _) = farthing::io::load_gltf(path).expect("load");
    let name = std::path::Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let b = scene.bounds();
    let diag = (b.max - b.min).length();
    let eps = diag
        * std::env::var("CRACK_EPS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1e-5);
    let parts = &scene.instances;
    let world = |part: usize, segs: &[[Vec3; 2]]| -> Vec<[Vec3; 2]> {
        let t = parts[part].transform;
        segs.iter()
            .map(|s| s.map(|p| t.transform_point3(p)))
            .collect()
    };

    // Source borders per part, in world space.
    let local: Vec<Vec<[Vec3; 2]>> = scene.meshes.iter().map(border_edges).collect();
    let src: Vec<Vec<[Vec3; 2]>> = (0..parts.len())
        .map(|p| world(p, &local[parts[p].mesh]))
        .collect();

    // Border vertices of every part in a grid of eps cells.
    let key = |p: Vec3| (p / eps).to_array().map(|x| x.floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<(u32, Vec3)>> = HashMap::new();
    for (pi, segs) in src.iter().enumerate() {
        let mut seen: std::collections::HashSet<[u32; 3]> = Default::default();
        for s in segs {
            for &p in s {
                if seen.insert(bits(p)) {
                    grid.entry(key(p)).or_default().push((pi as u32, p));
                }
            }
        }
    }
    let near = |pi: usize, p: Vec3| -> Vec<u32> {
        let k = key(p);
        let mut out = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(v) = grid.get(&[k[0] + x, k[1] + y, k[2] + z]) {
                        for &(o, q) in v {
                            if o as usize != pi && (q - p).length() <= eps && !out.contains(&o) {
                                out.push(o);
                            }
                        }
                    }
                }
            }
        }
        out
    };
    // Per source border edge: the other parts whose border vertices both its ends touch.
    let shared: Vec<Vec<Vec<u32>>> = src
        .par_iter()
        .enumerate()
        .map(|(pi, segs)| {
            segs.iter()
                .map(|s| {
                    let a = near(pi, s[0]);
                    let b = near(pi, s[1]);
                    a.into_iter().filter(|o| b.contains(o)).collect()
                })
                .collect()
        })
        .collect();
    let (mut len_all, mut len_shared) = (0f64, 0f64);
    let (mut n_all, mut n_shared) = (0usize, 0usize);
    let mut parts_sharing = 0;
    for (segs, sh) in src.iter().zip(&shared) {
        let mut any = false;
        for (s, o) in segs.iter().zip(sh) {
            let l = (s[1] - s[0]).length() as f64;
            len_all += l;
            n_all += 1;
            if !o.is_empty() {
                len_shared += l;
                n_shared += 1;
                any = true;
            }
        }
        parts_sharing += any as usize;
    }
    println!(
        "BORDER {name}: {} meshes, {} parts, {} tris, diag {diag:.4}; border edges {n_all}, shared {n_shared}; border length {:.2} diags, shared {:.2} diags = {:.1}% of border length; {parts_sharing} parts have a shared border",
        scene.meshes.len(),
        parts.len(),
        scene.triangle_count(),
        len_all / diag as f64,
        len_shared / diag as f64,
        100.0 * len_shared / len_all.max(1e-30),
    );
    if parts.len() < 2 {
        return;
    }
    let src_grid: Vec<SegGrid> = src.iter().map(|s| SegGrid::new(s)).collect();

    let input = scene.triangle_count();
    for &ratio in &ratios {
        let total = (input as f64 * ratio).round() as usize;
        let mut kept = 0;
        let reduced: Vec<Vec<[Vec3; 2]>> = scene
            .meshes
            .iter()
            .map(|m| {
                let mut m = m.clone();
                if m.normals.is_none() {
                    m.compute_normals(60.0);
                }
                let share = m.triangle_count() as f64 / input.max(1) as f64;
                let target = ((total as f64 * share).round() as usize).max(4);
                let opts = SimplifyOptions {
                    // CRACK_QUALITY=best or fast: that reducer instead, for comparison.
                    quality: match std::env::var("CRACK_QUALITY").as_deref() {
                        Ok("best") => Quality::Best,
                        Ok("fast") => Quality::Fast,
                        _ => Quality::Realtime,
                    },
                    ..crate::lf_opts(target)
                };
                let r = simplify(&m, &opts).unwrap().mesh;
                kept += r.triangle_count();
                border_edges(&r)
            })
            .collect();
        let red: Vec<Vec<[Vec3; 2]>> = (0..parts.len())
            .map(|p| world(p, &reduced[parts[p].mesh]))
            .collect();
        let red_grid: Vec<SegGrid> = red.iter().map(|s| SegGrid::new(s)).collect();

        // Per sample: (drift, crack if near a shared border and the neighbor kept a border,
        // near a shared border at all).
        let rows: Vec<(f32, Option<f32>, bool, bool)> = (0..parts.len())
            .into_par_iter()
            .flat_map_iter(|pi| {
                let mut seen: std::collections::HashSet<[u32; 3]> = Default::default();
                let mut samples: Vec<(Vec3, bool)> = Vec::new();
                for s in &red[pi] {
                    for &p in s {
                        if seen.insert(bits(p)) {
                            samples.push((p, true));
                        }
                    }
                    samples.push(((s[0] + s[1]) * 0.5, false));
                }
                let mut out = Vec::with_capacity(samples.len());
                for (p, vertex) in samples {
                    let Some((drift, e)) = src_grid[pi].nearest(p) else {
                        continue;
                    };
                    let nb = &shared[pi][e as usize];
                    let crack = nb
                        .iter()
                        .filter_map(|&o| red_grid[o as usize].nearest(p).map(|x| x.0))
                        .min_by(f32::total_cmp);
                    out.push((drift, crack, !nb.is_empty(), vertex));
                }
                out
            })
            .collect();
        let mut all = Stat(rows.iter().map(|r| r.0).collect());
        let mut sdrift = Stat(rows.iter().filter(|r| r.2).map(|r| r.0).collect());
        let mut crack = Stat(rows.iter().filter_map(|r| r.1).collect());
        let lost = rows.iter().filter(|r| r.2 && r.1.is_none()).count();
        let over = |s: &Stat, pct: f32| {
            100.0 * s.0.iter().filter(|&&x| x > diag * pct / 100.0).count() as f32
                / s.0.len().max(1) as f32
        };
        let (o1, o2) = (over(&crack, 0.05), over(&crack, 0.2));
        // Vertices alone: a border vertex that stands on a source border vertex drifts 0.
        let mut vall = Stat(rows.iter().filter(|r| r.3).map(|r| r.0).collect());
        let mut vdrift = Stat(rows.iter().filter(|r| r.2 && r.3).map(|r| r.0).collect());
        let mut vcrack = Stat(rows.iter().filter(|r| r.3).filter_map(|r| r.1).collect());
        let tight = |s: &Stat| 100.0 - over(s, 0.001);
        let (t0, t1, t2, t3) = (tight(&vall), tight(&vdrift), tight(&vcrack), tight(&crack));
        println!(
            "VERTS {name} {ratio} | all drift {} on-border {t0:.1}% | shared drift {} on-border {t1:.1}% | crack {} closed {t2:.1}% | closed, with midpoints {t3:.1}%",
            vall.line(diag),
            vdrift.line(diag),
            vcrack.line(diag),
        );
        println!(
            "CRACK {name} {ratio} kept {kept} | all drift {} | shared drift {} | crack {} | crack>0.05% {o1:.1}% >0.2% {o2:.1}% | no neighbor border {lost}",
            all.line(diag),
            sdrift.line(diag),
            crack.line(diag),
        );
    }
}
