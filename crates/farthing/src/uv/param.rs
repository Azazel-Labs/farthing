//! Chart segmentation + per-chart flattening (LSCM or planar) + packing.

use crate::uv::lscm::{layout_overlaps, lscm};
use snugly::{pack, rotate_point, ChartMask, HoleFill, MaskPlacement, PackOptions};
use glam::{Vec2, Vec3};
use crate::math::{orthonormal_basis, triangle_area};
use crate::{Error, Mesh, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Small-key hasher (multiply-rotate, as in rustc's FxHash): the maps here hold vertex, face and
/// corner ids, where SipHash's DoS resistance only costs time. Output never depends on map
/// iteration order (it is sorted wherever it matters), so the hasher does not change results.
#[derive(Default, Clone, Copy)]
struct Fx(u64);
impl std::hash::Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}
type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;
type HashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<Fx>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamMethod {
    /// Planar projection per chart (charts limited to `max_chart_angle_deg`).
    Planar,
    /// Least squares conformal maps on larger charts; charts that fold or overlap fall back
    /// to planar sub-charts.
    #[default]
    Lscm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamOptions {
    pub method: ParamMethod,
    /// Max angle between a face normal and its chart's normal. Planar: ~60°. LSCM: ~85°.
    pub max_chart_angle_deg: f32,
    pub atlas_size: u32,
    /// Gutter in pixels around each chart.
    pub padding: u32,
    /// Fixed texel density; None = as large as fits in the atlas.
    pub texels_per_unit: Option<f32>,
    /// Also pack with free intervals per column, which nests charts under the overhangs of
    /// others, and keep the lower atlas. More texel area for the same atlas (DamagedHelmet
    /// 0.610 → 0.697 utilization, Lantern 0.744 → 0.792) for about 2.5× the packing time on
    /// atlases with hundreds of charts. Off: the skyline packer alone.
    #[serde(default = "default_true")]
    pub fill_holes: bool,
    /// Split charts that fill less than 45% of their bounding rectangle (star-like unwraps)
    /// where the pieces' rectangles are at least 15% smaller in total. Trades a little seam
    /// length for packing density.
    #[serde(default = "default_true")]
    pub split_sprawling: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ParamOptions {
    fn default() -> Self {
        ParamOptions {
            method: ParamMethod::Lscm,
            max_chart_angle_deg: 85.0,
            atlas_size: 1024,
            padding: 4,
            texels_per_unit: None,
            fill_holes: true,
            split_sprawling: true,
        }
    }
}

impl ParamOptions {
    pub fn planar() -> Self {
        ParamOptions {
            method: ParamMethod::Planar,
            max_chart_angle_deg: 60.0,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParamResult {
    /// Input mesh with vertices split along chart boundaries and new UVs in set 0
    /// (other UV sets are dropped).
    pub mesh: Mesh,
    pub charts: usize,
    /// Charts flattened with LSCM (the rest are planar).
    pub lscm_charts: usize,
    /// Texels per world unit used.
    pub texels_per_unit: f32,
    /// Fraction of the atlas covered by chart rectangles.
    pub utilization: f32,
    /// Area-weighted mean of max(σ1/σ2) − 1 over triangles (0 = no stretch).
    pub mean_stretch: f32,
    /// For each output vertex, the input vertex it was copied from.
    pub source_vertex: Vec<u32>,
}

struct Chart {
    faces: Vec<usize>,
    /// 2D coordinates per (face, corner), world-unit scale.
    uv: HashMap<(usize, usize), Vec2>,
    min: Vec2,
    max: Vec2,
    lscm: bool,
}

struct Topo {
    normals: Vec<Vec3>,
    areas: Vec<f32>,
    adj: Vec<Vec<usize>>,
}

fn topology(mesh: &Mesh) -> Topo {
    let tc = mesh.triangle_count();
    let normals = (0..tc)
        .map(|t| {
            let [a, b, c] = mesh.triangle_positions(t);
            (b - a).cross(c - a).normalize_or_zero()
        })
        .collect();
    let areas = (0..tc)
        .map(|t| {
            let [a, b, c] = mesh.triangle_positions(t);
            triangle_area(a, b, c)
        })
        .collect();
    let key = |p: Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let mut pid: HashMap<[u32; 3], u32> = HashMap::default();
    let vp: Vec<u32> = mesh
        .positions
        .iter()
        .map(|p| {
            let n = pid.len() as u32;
            *pid.entry(key(*p)).or_insert(n)
        })
        .collect();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::default();
    for t in 0..tc {
        let p = mesh.triangle(t).map(|v| vp[v as usize]);
        for i in 0..3 {
            let (a, b) = (p[i], p[(i + 1) % 3]);
            edge_faces
                .entry(if a < b { (a, b) } else { (b, a) })
                .or_default()
                .push(t);
        }
    }
    let mut adj = vec![Vec::new(); tc];
    for faces in edge_faces.values() {
        if faces.len() == 2 {
            adj[faces[0]].push(faces[1]);
            adj[faces[1]].push(faces[0]);
        }
    }
    // HashMap iteration order is random: sort so chart growing is reproducible.
    for a in &mut adj {
        a.sort_unstable();
    }
    Topo {
        normals,
        areas,
        adj,
    }
}

/// Region growing by normal cone within `allowed` faces, largest faces first.
fn grow(topo: &Topo, allowed: &[usize], max_angle_deg: f32) -> Vec<Vec<usize>> {
    let cos_max = max_angle_deg.to_radians().cos();
    let in_set: HashSet<usize> = allowed.iter().copied().collect();
    let mut assigned: HashMap<usize, bool> = HashMap::default();
    let mut seeds = allowed.to_vec();
    seeds.sort_by(|&a, &b| topo.areas[b].total_cmp(&topo.areas[a]));
    let mut charts = Vec::new();
    for &seed in &seeds {
        if assigned.contains_key(&seed) {
            continue;
        }
        let mut n_acc = topo.normals[seed] * topo.areas[seed].max(1e-12);
        let mut faces = vec![seed];
        assigned.insert(seed, true);
        let mut queue = VecDeque::from([seed]);
        while let Some(f) = queue.pop_front() {
            for &g in &topo.adj[f] {
                if assigned.contains_key(&g) || !in_set.contains(&g) {
                    continue;
                }
                let cn = n_acc.normalize_or_zero();
                if topo.normals[g].dot(cn) >= cos_max
                    && topo.normals[g].dot(topo.normals[seed]) >= cos_max.min(0.5) * 0.5
                {
                    assigned.insert(g, true);
                    n_acc += topo.normals[g] * topo.areas[g].max(1e-12);
                    faces.push(g);
                    queue.push_back(g);
                }
            }
        }
        charts.push(faces);
    }
    charts
}

fn finalize(faces: Vec<usize>, mut uv: HashMap<(usize, usize), Vec2>, lscm: bool) -> Chart {
    // Rotate to the minimum-area bounding rectangle (rotating calipers over the convex hull):
    // tighter footprints pack better than principal-axis alignment.
    let mut keys: Vec<(usize, usize)> = uv.keys().copied().collect();
    keys.sort_unstable();
    let pts: Vec<Vec2> = keys.iter().map(|k| uv[k]).collect();
    let hull = convex_hull(&pts);
    let mut best = (f32::INFINITY, 0.0f32, 1.0f32); // (area, sin, cos)
    for i in 0..hull.len() {
        let e = hull[(i + 1) % hull.len()] - hull[i];
        let len = e.length();
        if len < 1e-12 {
            continue;
        }
        let (c, s) = (e.x / len, e.y / len);
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for p in &hull {
            let r = Vec2::new(p.x * c + p.y * s, -p.x * s + p.y * c);
            lo = lo.min(r);
            hi = hi.max(r);
        }
        let area = (hi - lo).x * (hi - lo).y;
        if area < best.0 - 1e-12 * area.abs() {
            best = (area, s, c);
        }
    }
    let (_, s, c) = best;
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for q in uv.values_mut() {
        let d = *q;
        *q = Vec2::new(d.x * c + d.y * s, -d.x * s + d.y * c);
        min = min.min(*q);
        max = max.max(*q);
    }
    // Landscape orientation (packing tries the other rotations anyway).
    Chart {
        faces,
        uv,
        min,
        max,
        lscm,
    }
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, deterministic.
fn convex_hull(pts: &[Vec2]) -> Vec<Vec2> {
    let mut p: Vec<Vec2> = pts.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut h: Vec<Vec2> = Vec::with_capacity(2 * p.len());
    for &q in &p {
        while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
            h.pop();
        }
        h.push(q);
    }
    let lower = h.len() + 1;
    for &q in p.iter().rev().skip(1) {
        while h.len() >= lower && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
            h.pop();
        }
        h.push(q);
    }
    h.pop();
    h
}

fn planar_chart(mesh: &Mesh, topo: &Topo, faces: Vec<usize>) -> Chart {
    let n_acc: Vec3 = faces
        .iter()
        .map(|&f| topo.normals[f] * topo.areas[f].max(1e-12))
        .sum();
    let cn = n_acc.normalize_or_zero();
    let cn = if cn == Vec3::ZERO {
        topo.normals[faces[0]]
    } else {
        cn
    };
    let (t, b) = orthonormal_basis(if cn == Vec3::ZERO { Vec3::Y } else { cn });
    // Shared local vertices so the projected chart can be refined as one piece.
    let key = |p: Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let mut local: HashMap<[u32; 3], u32> = HashMap::default();
    let mut pos = Vec::new();
    let mut tris = Vec::with_capacity(faces.len());
    for &f in &faces {
        tris.push(mesh.triangle_positions(f).map(|p| {
            *local.entry(key(p)).or_insert_with(|| {
                pos.push(p);
                pos.len() as u32 - 1
            })
        }));
    }
    let mut uvs: Vec<Vec2> = pos.iter().map(|p| Vec2::new(p.dot(t), p.dot(b))).collect();
    let flipped = tris.iter().any(|t| {
        let q = t.map(|i| uvs[i as usize]);
        (q[1] - q[0]).perp_dot(q[2] - q[0]) <= 0.0
    });
    crate::uv::lscm::normalize_area(&pos, &tris, &mut uvs);
    if !flipped && tris.len() >= 2 {
        let before = uvs.clone();
        if !crate::uv::lscm::arap_refine(&pos, &tris, &mut uvs, 10) || layout_overlaps(&uvs, &tris, 256)
        {
            uvs = before;
        }
    }
    let mut uv = HashMap::default();
    for (fi, &f) in faces.iter().enumerate() {
        for c in 0..3 {
            uv.insert((f, c), uvs[tris[fi][c] as usize]);
        }
    }
    finalize(faces, uv, false)
}

fn lscm_chart(mesh: &Mesh, faces: &[usize]) -> Option<Chart> {
    // Local vertex indexing by position (so the chart is connected through shared positions).
    let key = |p: Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let mut local: HashMap<[u32; 3], u32> = HashMap::default();
    let mut pos = Vec::new();
    let mut tris = Vec::with_capacity(faces.len());
    for &f in faces {
        let t = mesh.triangle_positions(f).map(|p| {
            *local.entry(key(p)).or_insert_with(|| {
                pos.push(p);
                pos.len() as u32 - 1
            })
        });
        tris.push(t);
    }
    let mut uvs = lscm(&pos, &tris)?;
    // LSCM is conformal but not area-preserving: fix the chart's scale, then pull towards
    // isometry with ARAP, which often also untangles LSCM's boundary overlaps.
    crate::uv::lscm::normalize_area(&pos, &tris, &mut uvs);
    let before = uvs.clone();
    let refined = crate::uv::lscm::arap_refine(&pos, &tris, &mut uvs, 10);
    if !refined || layout_overlaps(&uvs, &tris, 256) {
        uvs = before;
        if layout_overlaps(&uvs, &tris, 256) {
            return None;
        }
    }
    let mut uv = HashMap::default();
    for (fi, &f) in faces.iter().enumerate() {
        for c in 0..3 {
            uv.insert((f, c), uvs[tris[fi][c] as usize]);
        }
    }
    Some(finalize(faces.to_vec(), uv, true))
}

/// Atlas-wide normalized L2 stretch (Sander et al. 2001) of `mesh.uvs[0]`: 1.0 means every
/// triangle is mapped isometrically at one common texel density.
pub fn atlas_l2_stretch(mesh: &Mesh) -> f32 {
    let (mut a3, mut a2, mut l2) = (0.0f64, 0.0f64, 0.0f64);
    let uv = &mesh.uvs[0];
    for t in 0..mesh.triangle_count() {
        let p = mesh.triangle_positions(t);
        let q = mesh.triangle(t).map(|v| uv[v as usize]);
        let area3 = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length() as f64;
        let area2 = 0.5 * (q[1] - q[0]).perp_dot(q[2] - q[0]) as f64;
        if area3 <= 1e-30 || area2.abs() < 1e-30 {
            continue;
        }
        let ss = (p[0] * (q[1].y - q[2].y) + p[1] * (q[2].y - q[0].y) + p[2] * (q[0].y - q[1].y))
            / (2.0 * area2 as f32);
        let st = (p[0] * (q[2].x - q[1].x) + p[1] * (q[0].x - q[2].x) + p[2] * (q[1].x - q[0].x))
            / (2.0 * area2 as f32);
        l2 += (ss.length_squared() + st.length_squared()) as f64 / 2.0 * area3;
        a3 += area3;
        a2 += area2.abs();
    }
    if a3 <= 0.0 {
        1.0
    } else {
        ((l2 / a3).sqrt() * (a2 / a3).sqrt()) as f32
    }
}

/// Normalized L2 stretch of a chart (Sander et al. 2001): 1.0 is isometric up to scale.
fn chart_l2(mesh: &Mesh, chart: &Chart) -> f32 {
    let (mut a3, mut a2, mut l2) = (0.0f64, 0.0f64, 0.0f64);
    for &f in &chart.faces {
        let p = mesh.triangle_positions(f);
        let q = [0, 1, 2].map(|c| chart.uv[&(f, c)]);
        let area3 = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length() as f64;
        let area2 = 0.5 * (q[1] - q[0]).perp_dot(q[2] - q[0]) as f64;
        if area3 <= 1e-30 || area2.abs() < 1e-30 {
            continue;
        }
        let ss = (p[0] * (q[1].y - q[2].y) + p[1] * (q[2].y - q[0].y) + p[2] * (q[0].y - q[1].y))
            / (2.0 * area2 as f32);
        let st = (p[0] * (q[2].x - q[1].x) + p[1] * (q[0].x - q[2].x) + p[2] * (q[1].x - q[0].x))
            / (2.0 * area2 as f32);
        let t = (ss.length_squared() + st.length_squared()) as f64 / 2.0;
        l2 += t * area3;
        a3 += area3;
        a2 += area2.abs();
    }
    if a3 <= 0.0 {
        1.0
    } else {
        ((l2 / a3).sqrt() * (a2 / a3).sqrt()) as f32
    }
}

/// LSCM+ARAP for a face group; splits it with a tighter normal cone when it cannot be
/// flattened without overlap or with too much residual stretch.
fn lscm_or_split(
    mesh: &Mesh,
    topo: &Topo,
    faces: Vec<usize>,
    angle: f32,
    depth: u32,
) -> Vec<Chart> {
    const MAX_STRETCH: f32 = 1.03;
    const MAX_DEPTH: u32 = 4;
    let tighter = angle * 0.7;
    let split = |faces: &[usize]| -> Option<Vec<Chart>> {
        let subs = grow(topo, faces, tighter);
        (subs.len() > 1).then(|| {
            subs.into_iter()
                .flat_map(|f| lscm_or_split(mesh, topo, f, tighter, depth + 1))
                .collect()
        })
    };
    if faces.len() >= 2 {
        match lscm_chart(mesh, &faces) {
            Some(c) => {
                if depth >= MAX_DEPTH || faces.len() < 8 || chart_l2(mesh, &c) <= MAX_STRETCH {
                    return vec![c];
                }
                return split(&faces).unwrap_or_else(|| vec![c]);
            }
            // Overlapping layout: split and retry LSCM on the parts before giving up to planar.
            None if depth < MAX_DEPTH => {
                if let Some(parts) = split(&faces) {
                    return parts;
                }
            }
            None => {}
        }
    }
    grow(topo, &faces, angle.min(60.0))
        .into_iter()
        .map(|sub| planar_chart(mesh, topo, sub))
        .collect()
}

/// Merges neighbouring charts while the union still flattens without overlap and within the
/// stretch limit. Pairs are ranked by shared seam length relative to the smaller perimeter;
/// each pass evaluates a matching of pairs in parallel, so the result does not depend on the
/// thread count. Fewer charts means shorter seams and less packing padding.
fn merge_charts(mesh: &Mesh, topo: &Topo, charts: Vec<Chart>) -> Vec<Chart> {
    // Chosen on the benchmark: 1.005 kept lower stretch than xatlas on 8/8 assets while
    // cutting chart counts below xatlas's on 7/8 (1.0 merged little; 1.01+ lost stretch).
    let max_stretch: f32 = 1.005;
    const MIN_SCORE: f32 = 0.08;
    let nf = topo.areas.len();
    let mut slots: Vec<Option<Chart>> = charts.into_iter().map(Some).collect();
    let mut owner = vec![u32::MAX; nf];
    for (c, ch) in slots.iter().enumerate() {
        for &f in &ch.as_ref().unwrap().faces {
            owner[f] = c as u32;
        }
    }
    // Shared edge length of adjacent faces (positions matched exactly).
    let edge_len = |f: usize, g: usize| -> f32 {
        let (a, b) = (mesh.triangle_positions(f), mesh.triangle_positions(g));
        let shared: Vec<Vec3> = a.iter().copied().filter(|p| b.contains(p)).collect();
        if shared.len() >= 2 {
            shared[0].distance(shared[1])
        } else {
            0.0
        }
    };
    // A merge must keep the union's rectangle fill at least that of its worse part, or 0.7
    // (chosen on the benchmark: looser floors merged ragged charts that packed worse).
    let (fill_keep, fill_floor) = (1.0f32, 0.7f32);
    let fill = |c: &Chart| -> f32 {
        let a: f32 = c
            .faces
            .iter()
            .map(|&f| {
                let q = [0, 1, 2].map(|k| c.uv[&(f, k)]);
                0.5 * (q[1] - q[0]).perp_dot(q[2] - q[0]).abs()
            })
            .sum();
        let e = (c.max - c.min).max(Vec2::splat(1e-12));
        a / (e.x * e.y)
    };
    let mut failed: HashSet<(u32, u32)> = HashSet::default();
    for _pass in 0..12 {
        // Perimeters and shared seams from the current ownership.
        let mut perim = vec![0.0f32; slots.len()];
        let mut shared: Vec<((u32, u32), f32)> = Vec::new();
        for f in 0..nf {
            let cf = owner[f];
            let [p0, p1, p2] = mesh.triangle_positions(f);
            let full = p0.distance(p1) + p1.distance(p2) + p2.distance(p0);
            let mut inner = 0.0;
            for &g in &topo.adj[f] {
                let l = edge_len(f, g);
                if owner[g] == cf {
                    inner += l;
                } else if f < g {
                    let (a, b) = (cf.min(owner[g]), cf.max(owner[g]));
                    shared.push(((a, b), l));
                }
            }
            perim[cf as usize] += full - inner;
        }
        shared.sort_unstable_by_key(|x| x.0);
        let mut pairs: Vec<(f32, u32, u32)> = Vec::new();
        let mut i = 0;
        while i < shared.len() {
            let mut j = i;
            let mut len = 0.0;
            while j < shared.len() && shared[j].0 == shared[i].0 {
                len += shared[j].1;
                j += 1;
            }
            let (a, b) = shared[i].0;
            let score = len / perim[a as usize].min(perim[b as usize]).max(1e-12);
            if score >= MIN_SCORE && !failed.contains(&(a, b)) {
                pairs.push((score, a, b));
            }
            i = j;
        }
        pairs.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2))));
        let mut used = vec![false; slots.len()];
        let mut batch = Vec::new();
        for &(_, a, b) in &pairs {
            if !used[a as usize] && !used[b as usize] {
                used[a as usize] = true;
                used[b as usize] = true;
                batch.push((a, b));
            }
        }
        if batch.is_empty() {
            break;
        }
        let results: Vec<Option<Chart>> = batch
            .par_iter()
            .map(|&(a, b)| {
                let mut faces: Vec<usize> = slots[a as usize]
                    .as_ref()
                    .unwrap()
                    .faces
                    .iter()
                    .chain(&slots[b as usize].as_ref().unwrap().faces)
                    .copied()
                    .collect();
                faces.sort_unstable();
                lscm_chart(mesh, &faces).filter(|c| {
                    chart_l2(mesh, c) <= max_stretch && {
                        // Packability: the union must fill its bounding rectangle about as well
                        // as the worse of its parts (ragged unions pack badly).
                        let fa = fill(slots[a as usize].as_ref().unwrap());
                        let fb = fill(slots[b as usize].as_ref().unwrap());
                        fill(c) >= (fa.min(fb) * fill_keep).min(fill_floor)
                    }
                })
            })
            .collect();
        let mut merged_any = false;
        for (&(a, b), r) in batch.iter().zip(results) {
            match r {
                Some(c) => {
                    for &f in &slots[b as usize].as_ref().unwrap().faces {
                        owner[f] = a;
                    }
                    slots[a as usize] = Some(c);
                    slots[b as usize] = None;
                    failed.retain(|&(x, y)| x != a && y != a && x != b && y != b);
                    merged_any = true;
                }
                None => {
                    failed.insert((a, b));
                }
            }
        }
        if !merged_any && batch.len() == pairs.len() {
            break;
        }
    }
    slots.into_iter().flatten().collect()
}

fn chart_area(c: &Chart) -> f32 {
    c.faces
        .iter()
        .map(|&f| {
            let q = [0, 1, 2].map(|k| c.uv[&(f, k)]);
            0.5 * (q[1] - q[0]).perp_dot(q[2] - q[0]).abs()
        })
        .sum()
}

fn rect_area(c: &Chart) -> f32 {
    let e = c.max - c.min;
    e.x * e.y
}

/// Charts that fill less than this share of their bounding rectangle are split if that pays.
const SPRAWL_FILL: f32 = 0.45;

/// Splits a chart that fills little of its bounding rectangle (star-like unwraps of bodies and
/// limbs) into pieces with a smaller total rectangle area: two-means on face centroids in UV,
/// then connected components. Pieces keep their UVs (a subset of an embedding is still one) and
/// are realigned to their own minimum-area rectangles. Kept only if the rectangles shrink by at
/// least 15%; pieces split again while that holds (depth ≤ 3).
fn split_sprawling(topo: &Topo, c: Chart, depth: u32) -> Vec<Chart> {
    let area = chart_area(&c);
    let rect = rect_area(&c);
    if depth >= 3 || c.faces.len() < 16 || rect <= 0.0 || area / rect >= SPRAWL_FILL {
        return vec![c];
    }
    let cent: Vec<(usize, Vec2, f32)> = c
        .faces
        .iter()
        .map(|&f| {
            let q = [0, 1, 2].map(|k| c.uv[&(f, k)]);
            let a = 0.5 * (q[1] - q[0]).perp_dot(q[2] - q[0]).abs();
            (f, (q[0] + q[1] + q[2]) / 3.0, a.max(1e-12))
        })
        .collect();
    // Seeds at the two ends of the rectangle's long side.
    let long_x = (c.max - c.min).x >= (c.max - c.min).y;
    let key = |p: Vec2| if long_x { p.x } else { p.y };
    let (mut a, mut b) = (cent[0].1, cent[0].1);
    for &(_, p, _) in &cent {
        if key(p) < key(a) {
            a = p;
        }
        if key(p) > key(b) {
            b = p;
        }
    }
    let mut side = vec![false; cent.len()];
    for _ in 0..8 {
        for (i, &(_, p, _)) in cent.iter().enumerate() {
            side[i] = (p - b).length_squared() < (p - a).length_squared();
        }
        let (mut sa, mut sb, mut wa, mut wb) = (Vec2::ZERO, Vec2::ZERO, 0.0, 0.0);
        for (i, &(_, p, w)) in cent.iter().enumerate() {
            if side[i] {
                sb += p * w;
                wb += w;
            } else {
                sa += p * w;
                wa += w;
            }
        }
        if wa == 0.0 || wb == 0.0 {
            return vec![c];
        }
        (a, b) = (sa / wa, sb / wb);
    }
    // Connected components within each side (faces adjacent in the mesh and in this chart).
    let index: HashMap<usize, usize> = cent.iter().enumerate().map(|(i, x)| (x.0, i)).collect();
    let mut comp = vec![usize::MAX; cent.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for start in 0..cent.len() {
        if comp[start] != usize::MAX {
            continue;
        }
        let id = groups.len();
        comp[start] = id;
        let mut stack = vec![start];
        let mut faces = Vec::new();
        while let Some(i) = stack.pop() {
            faces.push(cent[i].0);
            for g in &topo.adj[cent[i].0] {
                if let Some(&j) = index.get(g) {
                    if comp[j] == usize::MAX && side[j] == side[i] {
                        comp[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        faces.sort_unstable();
        groups.push(faces);
    }
    if groups.len() < 2 || groups.len() > 4 {
        return vec![c];
    }
    let pieces: Vec<Chart> = groups
        .into_iter()
        .map(|faces| {
            let uv = faces
                .iter()
                .flat_map(|&f| (0..3).map(move |k| (f, k)))
                .map(|k| (k, c.uv[&k]))
                .collect();
            finalize(faces, uv, c.lscm)
        })
        .collect();
    let split: f32 = pieces.iter().map(rect_area).sum();
    if split > 0.85 * rect {
        return vec![c];
    }
    pieces
        .into_iter()
        .flat_map(|p| split_sprawling(topo, p, depth + 1))
        .collect()
}

/// Generates a fresh UV atlas for `mesh`.
pub fn parameterize(mesh: &Mesh, opts: &ParamOptions) -> Result<ParamResult> {
    mesh.validate()?;
    let tri_count = mesh.triangle_count();
    if tri_count == 0 {
        return Err(Error::InvalidMesh("no triangles to parameterize".into()));
    }
    let topo = topology(mesh);
    let all: Vec<usize> = (0..tri_count).collect();
    let mut charts: Vec<Chart> = Vec::new();
    match opts.method {
        ParamMethod::Planar => {
            for faces in grow(&topo, &all, opts.max_chart_angle_deg) {
                charts.push(planar_chart(mesh, &topo, faces));
            }
        }
        ParamMethod::Lscm => {
            let groups = grow(&topo, &all, opts.max_chart_angle_deg);
            let built: Vec<Vec<Chart>> = groups
                .into_par_iter()
                .map(|faces| lscm_or_split(mesh, &topo, faces, opts.max_chart_angle_deg, 0))
                .collect();
            let merged = merge_charts(mesh, &topo, built.into_iter().flatten().collect());
            if opts.split_sprawling {
                charts.extend(merged.into_iter().flat_map(|c| split_sprawling(&topo, c, 0)));
            } else {
                charts.extend(merged);
            }
        }
    }

    // Find the largest texel density whose chart footprints pack into the atlas.
    let atlas = opts.atlas_size;
    let pad = opts.padding;
    let uv_area: f32 = charts
        .iter()
        .map(|c| {
            c.faces
                .iter()
                .map(|&f| {
                    let q = [0, 1, 2].map(|k| c.uv[&(f, k)]);
                    0.5 * (q[1] - q[0]).perp_dot(q[2] - q[0]).abs()
                })
                .sum::<f32>()
        })
        .sum();
    // Packs at `density` into the atlas width with unbounded height; returns the used height.
    let try_pack = |density: f32| -> Option<(Vec<MaskPlacement>, Vec<(u32, u32)>, u32)> {
        let masks: Vec<ChartMask> = charts
            .par_iter()
            .map(|c| {
                let e = (c.max - c.min) * density;
                let (w, h) = (
                    e.x.ceil() as u32 + 2 * pad + 1,
                    e.y.ceil() as u32 + 2 * pad + 1,
                );
                let tris: Vec<[Vec2; 3]> = c
                    .faces
                    .iter()
                    .map(|&f| {
                        [0, 1, 2]
                            .map(|k| (c.uv[&(f, k)] - c.min) * density + Vec2::splat(pad as f32))
                    })
                    .collect();
                ChartMask::from_triangles(&tris, w, h, pad)
            })
            .collect();
        let frames = masks.iter().map(|m| (m.w, m.h)).collect();
        let popts = PackOptions {
            fill_holes: if opts.fill_holes {
                HoleFill::Always
            } else {
                HoleFill::Never
            },
            ..PackOptions::width(atlas)
        };
        let best = pack(&masks, &popts).ok().map(|p| (p.placements, p.height));
        best.map(|(p, used)| (p, frames, used))
    };
    let (density, placements, frames) = if let Some(d) = opts.texels_per_unit {
        match try_pack(d) {
            Some((p, f, used)) if used <= atlas => (d, p, f),
            _ => {
                return Err(Error::InvalidInput(format!(
                    "charts do not fit a {atlas}px atlas at {d} texels/unit"
                )))
            }
        }
    } else {
        // Height scales with density²: jump towards the density that fills the atlas exactly,
        // then keep the densest packing that fits.
        let mut d = (0.7 * (atlas as f32).powi(2) / uv_area.max(1e-12)).sqrt();
        let mut best: Option<(f32, Vec<MaskPlacement>, Vec<(u32, u32)>)> = None;
        let mut fail = f32::INFINITY;
        for _ in 0..10 {
            let Some((p, f, used)) = try_pack(d) else {
                d *= 0.8; // a chart is wider than the atlas at this density
                fail = fail.min(d / 0.8);
                continue;
            };
            let fits = used <= atlas;
            if fits && best.as_ref().is_none_or(|b| d > b.0) {
                best = Some((d, p, f));
            } else if !fits {
                fail = fail.min(d);
            }
            let lo = best.as_ref().map_or(0.0, |b| b.0);
            if fail.is_finite() && fail / lo.max(1e-12) < 1.006 {
                break;
            }
            let mut next = d
                * (atlas as f32 / used.max(1) as f32).sqrt().clamp(0.7, 1.4)
                * if fits { 0.995 } else { 0.985 };
            if next <= lo || next >= fail {
                next = 0.5 * (lo + fail.min(lo * 1.1 + 1e-6));
            }
            d = next;
        }
        let (d, p, f) = best.ok_or_else(|| Error::InvalidInput("could not pack charts".into()))?;
        (d, p, f)
    };

    let mut vmap: HashMap<(usize, u32), u32> = HashMap::default();
    let mut source_vertex = Vec::new();
    let mut new_uv = Vec::new();
    let mut indices = vec![0u32; mesh.indices.len()];
    for (ci, chart) in charts.iter().enumerate() {
        let pl = placements[ci];
        for &f in &chart.faces {
            let tri = mesh.triangle(f);
            for c in 0..3 {
                let v = tri[c];
                let id = *vmap.entry((ci, v)).or_insert_with(|| {
                    let local = (chart.uv[&(f, c)] - chart.min) * density + Vec2::splat(pad as f32);
                    let (fw, fh) = frames[ci];
                    let px = rotate_point(local, fw, fh, pl.rotation);
                    new_uv.push((Vec2::new(pl.x as f32, pl.y as f32) + px) / atlas as f32);
                    source_vertex.push(v);
                    source_vertex.len() as u32 - 1
                });
                indices[3 * f + c] = id;
            }
        }
    }
    let mut out = mesh.select_vertices(&source_vertex);
    out.uvs = vec![new_uv];
    out.tangents = None;
    out.indices = indices;
    out.material_ids = mesh.material_ids.clone();
    let used = uv_area * density * density;
    let mean_stretch = atlas_l2_stretch(&out) - 1.0;
    Ok(ParamResult {
        mesh: out,
        charts: charts.len(),
        lscm_charts: charts.iter().filter(|c| c.lscm).count(),
        texels_per_unit: density,
        utilization: used / (atlas as f32 * atlas as f32),
        mean_stretch,
        source_vertex,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::signed_area_2d;

    fn cube() -> Mesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for (n, u, v) in [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y, Vec3::X),
        ] {
            let b = positions.len() as u32;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                positions.push((n + u * su + v * sv) * 0.5);
            }
            indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        Mesh::new(positions, indices)
    }

    fn sphere(seg: u32, rings: u32) -> Mesh {
        let mut pos = Vec::new();
        for r in 0..=rings {
            let th = r as f32 / rings as f32 * std::f32::consts::PI;
            for s in 0..seg {
                let ph = s as f32 / seg as f32 * std::f32::consts::TAU;
                pos.push(if r == 0 {
                    Vec3::Y
                } else if r == rings {
                    -Vec3::Y
                } else {
                    Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin())
                });
            }
        }
        let mut idx = Vec::new();
        for r in 0..rings {
            for s in 0..seg {
                let (a, b) = (r * seg + s, r * seg + (s + 1) % seg);
                let (c, d) = (a + seg, b + seg);
                for t in [[a, b, c], [b, d, c]] {
                    let [p0, p1, p2] = t.map(|i| pos[i as usize]);
                    if (p1 - p0).cross(p2 - p0).length_squared() > 1e-12 {
                        idx.extend_from_slice(&t);
                    }
                }
            }
        }
        let mut m = Mesh::new(pos, idx);
        // Make outward-facing (winding check).
        let [a, b, c] = m.triangle_positions(0);
        if (b - a).cross(c - a).dot(a + b + c) < 0.0 {
            for t in m.indices.chunks_exact_mut(3) {
                t.swap(1, 2);
            }
        }
        m
    }

    fn assert_valid_atlas(r: &ParamResult) {
        r.mesh.validate().unwrap();
        let uv = &r.mesh.uvs[0];
        assert!(uv
            .iter()
            .all(|t| (0.0..=1.0).contains(&t.x) && (0.0..=1.0).contains(&t.y)));
        let tris: Vec<[u32; 3]> = (0..r.mesh.triangle_count())
            .map(|t| r.mesh.triangle(t))
            .collect();
        assert!(
            !crate::uv::lscm::layout_overlaps(uv, &tris, 256),
            "charts overlap"
        );
        for t in &tris {
            assert!(
                signed_area_2d(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]).abs()
                    > 1e-9
            );
        }
    }

    #[test]
    fn same_atlas_for_any_thread_count() {
        // A bumpy grid of cubes: LSCM charts, merging, both packers.
        let mut m = cube();
        for k in 1..6 {
            let mut c = cube();
            for p in &mut c.positions {
                *p = *p * (0.5 + 0.1 * k as f32) + Vec3::new(k as f32 * 1.3, (k % 2) as f32, 0.0);
            }
            m.append(&c, 0);
        }
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| parameterize(&m, &ParamOptions::default()).unwrap())
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.mesh.uvs, b.mesh.uvs);
        assert_eq!(a.mesh.indices, b.mesh.indices);
    }

    #[test]
    fn cube_gets_six_charts_without_overlap() {
        let r = parameterize(
            &cube(),
            &ParamOptions {
                atlas_size: 256,
                ..ParamOptions::planar()
            },
        )
        .unwrap();
        assert_eq!(r.charts, 6);
        assert_valid_atlas(&r);
        assert!(r.utilization > 0.3, "utilization {}", r.utilization);
        assert!(r.mean_stretch < 1e-3);
    }

    #[test]
    fn parameterization_is_deterministic() {
        let s = sphere(32, 16);
        let a = parameterize(&s, &ParamOptions::default()).unwrap();
        let b = parameterize(&s, &ParamOptions::default()).unwrap();
        assert_eq!(a.mesh.uvs, b.mesh.uvs);
        assert_eq!(a.mesh.indices, b.mesh.indices);
    }

    #[test]
    fn sphere_atlas_is_near_isometric() {
        // Charts are split until their residual stretch is low.
        let s = sphere(48, 24);
        let planar = parameterize(&s, &ParamOptions::planar()).unwrap();
        let lscm = parameterize(&s, &ParamOptions::default()).unwrap();
        assert_valid_atlas(&planar);
        assert_valid_atlas(&lscm);
        assert!(lscm.lscm_charts > 0);
        assert!(
            lscm.charts <= planar.charts * 2,
            "lscm {} vs planar {}",
            lscm.charts,
            planar.charts
        );
        // Both paths finish with ARAP, so both end up close to isometric.
        assert!(
            lscm.mean_stretch < 0.02 && planar.mean_stretch < 0.02,
            "lscm stretch {} planar {}",
            lscm.mean_stretch,
            planar.mean_stretch
        );
        assert!(lscm.utilization > 0.5, "utilization {}", lscm.utilization);
    }
}
