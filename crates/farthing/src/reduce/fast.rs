//! Fast reduction (`Quality::Fast`): a lean half-edge collapse loop in passes, built for speed
//! on one thread.
//!
//! Garland–Heckbert plane quadrics in f32 over positions normalized to the unit bounding
//! box (face planes, border planes, and planes along sharp creases beside narrow faces), plus
//! a cheap attribute term: per wedge of p, its area × the squared distance between the q
//! wedge it maps to and its own attributes carried along their surface gradient (normals,
//! UVs, colors; see `St::wcost`). Each pass rebuilds a flat
//! position→triangle index, costs every edge in its cheaper allowed direction, sorts the
//! candidates once, and collapses them in cost order, at most one collapse per neighborhood
//! per pass, up to a cost limit. There is no candidate heap and no incremental re-ranking:
//! everything is linear passes over flat arrays. Border vertices slide along their border and
//! junction vertices along their non-manifold junction (duplicated double-sided faces count
//! once); locked vertices stay, and collapses that flip or degenerate a face or break the link
//! condition are skipped. When collapses stall above the target, whole parts go by the full
//! reducer's rule (visible removal error, target floor, visible-part protection). Output is
//! deterministic (single-threaded; hash sets are only looked up, never iterated).
//!
//! A collapse that takes a vertex off a UV seam carries the wedges it would tear: a wedge of p
//! that no triangle joins to q, and that lies on another UV island than the wedges that are
//! joined, moves to q's position with its own attributes, so its triangles stay on their
//! island and the seam moves with the vertex. Mapped onto a wedge of q instead (the only
//! choice when output vertices must be exact copies of input vertices), its triangles span
//! two islands and show whatever lies between them in the texture, over an area that grows
//! with every later collapse. See `CARRY_SLIDE` and `St::unshared`.

use crate::reduce::simplify::{SimplifyOptions, SimplifyResult};
use crate::{Mesh, Result};
use glam::Vec3;

/// Largest extra cost of a deviating normal, on a material of gloss 1 (see
/// [`SimplifyOptions::material_gloss`]).
const GLOSS_NORMAL_GAIN: f32 = 30.0;

/// Per vertex, the factor its normal is scaled by in the attribute distance: the root of
/// 1 + gain × the glossiest material on its triangles. None when no material is glossy.
fn vertex_gloss(mesh: &Mesh, opts: &SimplifyOptions) -> Option<Vec<f32>> {
    if !opts.material_gloss.iter().any(|&g| g > 0.0) {
        return None;
    }
    let gain = GLOSS_NORMAL_GAIN;
    let mut g = vec![0f32; mesh.vertex_count()];
    for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
        let m = mesh.material_ids.get(t).copied().unwrap_or(0) as usize;
        let w = opts
            .material_gloss
            .get(m)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        for &v in tri {
            g[v as usize] = g[v as usize].max(w);
        }
    }
    for x in &mut g {
        *x = (1.0 + gain * *x).sqrt();
    }
    Some(g)
}

/// Symmetric 4×4 quadric: a00 a01 a02 a03 a11 a12 a13 a22 a23 a33.
#[derive(Clone, Copy, Default)]
struct Q([f32; 10]);

impl Q {
    fn plane(n: Vec3, d: f32, w: f32) -> Q {
        let (a, b, c) = (n.x, n.y, n.z);
        Q([
            a * a * w,
            a * b * w,
            a * c * w,
            a * d * w,
            b * b * w,
            b * c * w,
            b * d * w,
            c * c * w,
            c * d * w,
            d * d * w,
        ])
    }
    #[inline]
    fn add(&mut self, o: &Q) {
        for i in 0..10 {
            self.0[i] += o.0[i];
        }
    }
    #[inline]
    fn eval(&self, v: Vec3) -> f32 {
        let a = &self.0;
        let (x, y, z) = (v.x, v.y, v.z);
        let e = a[0] * x * x
            + 2.0 * a[1] * x * y
            + 2.0 * a[2] * x * z
            + 2.0 * a[3] * x
            + a[4] * y * y
            + 2.0 * a[5] * y * z
            + 2.0 * a[6] * y
            + a[7] * z * z
            + 2.0 * a[8] * z
            + a[9];
        e.max(0.0)
    }
}

/// Attribute term scales relative to the default option weights, tuned on the benchmark
/// corpus (8 assets × 4 ratios): positions are in the unit-diagonal frame, so attribute
/// differences must be scaled far down to stay comparable to squared distances.
const NORMAL_SCALE: f32 = 1e-5;
const UV_SCALE: f32 = 1e-3;
const COLOR_SCALE: f32 = 1e-3;
/// A carried wedge (see the module docs) keeps its UVs while its corner moves, so its texture
/// slides over the surface about as far as the joined wedges' UVs move. That squared distance
/// times this is its attribute cost. The slide moves an island's edge, which shows at once,
/// where the same UV change inside an island mostly follows the surface: at 1 and 4 the
/// corpus's smooth single-island assets lose up to 0.6 dB to seams that wander.
const CARRY_SLIDE: f32 = 16.0;
/// Squared UV distance (unscaled) under which two wedges of a vertex are on the same island.
const SAME_UV2: f32 = 1e-10;
/// Weight of the constraint planes along sharp creases beside narrow faces, relative to the
/// edge length² × (1 − cos dihedral) / 2. A thin panel (a windshield, a body panel with a
/// folded rim) has its outline on such a crease: the faces across are a narrow strip whose
/// planes barely price a vertex sliding inward in the panel's plane, so without these the
/// outline sank in and the panel's edge showed a gap. Gentle creases get none (the corpus's
/// smooth assets lose up to 1 dB when strips of a cylinder or bevel are pinned), nor do
/// creases between wide faces, whose own planes already hold the edge.
const CREASE_WEIGHT: f32 = 2.0;
/// Crease value, (1 − cos dihedral) / 2, above which a crease is sharp: 30°.
const CREASE_SHARP: f32 = 0.067;
/// A face is narrow across an edge when its area is under this × the edge's length² (a right
/// isosceles triangle across its hypotenuse is 0.25, an equilateral one 0.43).
const CREASE_NARROW: f32 = 0.3;
/// Pass size: candidates up to `PASS_LIMIT` × the cost of the m-th cheapest, m = what is still
/// needed but at most `PASS_SHARE` of all edges.
const PASS_SHARE: f32 = 0.5;
const PASS_LIMIT: f32 = 2.0;

/// Near the target (m under 1/8 of the candidates), a pass that makes under 1/`PASS_STUCK`
/// of its m collapses (the cheap ones block each other) lifts the limit for the next pass,
/// which then finishes, instead of re-costing every edge for a handful at a time.
const PASS_STUCK: usize = 4;

pub(crate) fn simplify_fast(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: Option<usize>,
) -> Result<SimplifyResult> {
    let nv = mesh.vertex_count();
    let nt = mesh.triangle_count();
    let bounds = mesh.bounds();
    let diag = bounds.diagonal().max(1e-12);
    let inv = 1.0 / diag;

    // Weld positions by exact bits.
    let key = |i: usize| mesh.positions[i].to_array().map(f32::to_bits);
    let mut order: Vec<u32> = (0..nv as u32).collect();
    order.sort_unstable_by_key(|&i| (key(i as usize), i));
    let mut vpos = vec![0u32; nv];
    let mut pos: Vec<Vec3> = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        if k == 0 || key(order[k - 1] as usize) != key(i as usize) {
            pos.push((mesh.positions[i as usize] - bounds.min) * inv);
        }
        vpos[i as usize] = pos.len() as u32 - 1;
    }
    let np = pos.len();

    let tris: Vec<[u32; 3]> = (0..nt).map(|t| mesh.triangle(t)).collect();
    let tp: Vec<[u32; 3]> = tris.iter().map(|t| t.map(|v| vpos[v as usize])).collect();
    let alive: Vec<bool> = tp
        .iter()
        .map(|p| p[0] != p[1] && p[1] != p[2] && p[0] != p[2])
        .collect();
    let mut alive_tris = alive.iter().filter(|&&a| a).count();

    // Edges by position pair: borders (one triangle) and non-manifold (more than two). Faces
    // over the same three positions (double-sided twins) count once.
    let twin = {
        let mut key: Vec<([u32; 3], u32)> = tp
            .iter()
            .enumerate()
            .map(|(t, p)| {
                let mut k = *p;
                k.sort_unstable();
                (k, t as u32)
            })
            .collect();
        key.sort_unstable();
        let mut twin = vec![false; nt];
        for w in key.windows(2) {
            if w[0].0 == w[1].0 {
                twin[w[1].1 as usize] = true;
            }
        }
        twin
    };
    let mut edges: Vec<(u64, u32)> = Vec::with_capacity(nt * 3);
    for (t, p) in tp.iter().enumerate() {
        if alive[t] && !twin[t] {
            for i in 0..3 {
                let (a, b) = (p[i].min(p[(i + 1) % 3]), p[i].max(p[(i + 1) % 3]));
                edges.push((((a as u64) << 32) | b as u64, (t * 3 + i) as u32));
            }
        }
    }
    edges.sort_unstable();
    let mut quad = vec![Q::default(); np];
    let face = |t: usize| {
        let p = tp[t];
        let (a, b, c) = (pos[p[0] as usize], pos[p[1] as usize], pos[p[2] as usize]);
        let n = (b - a).cross(c - a);
        (n.normalize_or_zero(), n.length() * 0.5)
    };
    let mut locked = vec![false; np];
    let mut border_deg = vec![0u8; np];
    let mut border_edges: Vec<u64> = Vec::new();
    // Junction (non-manifold) edges and how many touch each position.
    let mut junction_edges: Vec<u64> = Vec::new();
    let mut jdeg = vec![0u8; np];
    let mut k = 0;
    while k < edges.len() {
        let mut j = k + 1;
        while j < edges.len() && edges[j].0 == edges[k].0 {
            j += 1;
        }
        let key = edges[k].0;
        let (a, b) = ((key >> 32) as usize, key as u32 as usize);
        match j - k {
            1 => {
                border_deg[a] = border_deg[a].saturating_add(1);
                border_deg[b] = border_deg[b].saturating_add(1);
                border_edges.push(key);
            }
            2 => {
                // A sharp crease beside a narrow face: plane through the edge, perpendicular
                // to the face, as on an open border (see CREASE_WEIGHT).
                let (t0, t1) = (edges[k].1 as usize / 3, edges[k + 1].1 as usize / 3);
                let (n0, a0) = face(t0);
                let (n1, a1) = face(t1);
                let crease = (1.0 - n0.dot(n1).clamp(-1.0, 1.0)) * 0.5;
                if crease > CREASE_SHARP && a0 > 0.0 && a1 > 0.0 {
                    let (x0, x1) = (pos[a], pos[b]);
                    let dir = x1 - x0;
                    let len2 = dir.length_squared();
                    for (nn, across) in [(n0, a1), (n1, a0)] {
                        let bn = dir.cross(nn).normalize_or_zero();
                        if across < CREASE_NARROW * len2 && bn != Vec3::ZERO {
                            let cq = Q::plane(bn, -bn.dot(x0), len2 * CREASE_WEIGHT * crease);
                            quad[a].add(&cq);
                            quad[b].add(&cq);
                        }
                    }
                }
            }
            _ => {
                jdeg[a] = jdeg[a].saturating_add(1);
                jdeg[b] = jdeg[b].saturating_add(1);
                junction_edges.push(key);
            }
        }
        k = j;
    }
    for p in 0..np {
        if border_deg[p] > 2
            || (opts.lock_border && border_deg[p] > 0)
            || (jdeg[p] > 0 && border_deg[p] > 0)
        {
            locked[p] = true;
        }
    }
    if let Some(l) = &opts.locked_vertices {
        for (v, &is) in l.iter().enumerate().take(nv) {
            if is {
                locked[vpos[v] as usize] = true;
            }
        }
    }
    let border_edges: std::collections::HashSet<u64> = border_edges.into_iter().collect();
    let is_border_edge = |a: u32, b: u32| border_edges.contains(&ekey(a, b));

    // Quadrics: area-weighted face planes plus border constraint planes.
    let mut total_area = 0.0f32;
    let mut warea = vec![0.0f32; nv];
    for t in 0..nt {
        if !alive[t] {
            continue;
        }
        let p = tp[t];
        let (a, b, c) = (pos[p[0] as usize], pos[p[1] as usize], pos[p[2] as usize]);
        let n = (b - a).cross(c - a);
        let area = n.length() * 0.5;
        if area <= 0.0 {
            continue;
        }
        total_area += area;
        let nn = n / (2.0 * area);
        let q = Q::plane(nn, -nn.dot(a), area);
        for &pi in &p {
            quad[pi as usize].add(&q);
        }
        for &w in &tris[t] {
            warea[w as usize] += area / 3.0;
        }
        for i in 0..3 {
            let (e0, e1) = (p[i], p[(i + 1) % 3]);
            if border_deg[e0 as usize] > 0 && border_deg[e1 as usize] > 0 && is_border_edge(e0, e1)
            {
                let (x0, x1) = (pos[e0 as usize], pos[e1 as usize]);
                let dir = x1 - x0;
                let bn = dir.cross(nn).normalize_or_zero();
                if bn != Vec3::ZERO {
                    let bq = Q::plane(bn, -bn.dot(x0), dir.length_squared() * opts.border_weight);
                    quad[e0 as usize].add(&bq);
                    quad[e1 as usize].add(&bq);
                }
            }
        }
    }
    let avg_area = (total_area / alive_tris.max(1) as f32).max(1e-30);
    // Junction vertices move only along junction edges, like border vertices along borders
    // (after the border planes: the faces meeting at a junction already pin it).
    let mut border_edges = border_edges;
    for p in 0..np {
        if jdeg[p] > 0 && !locked[p] {
            border_deg[p] = jdeg[p];
        }
    }
    border_edges.extend(junction_edges);

    // Attributes per wedge, pre-scaled.
    let nw = (opts.normal_weight / 5e-5).max(0.0).sqrt() * NORMAL_SCALE.sqrt();
    let uw = (opts.uv_weight / 0.5).max(0.0).sqrt() * UV_SCALE.sqrt();
    let cw = (opts.color_weight / 0.5).max(0.0).sqrt() * COLOR_SCALE.sqrt();
    let mut dim = 0;
    if mesh.normals.is_some() && nw > 0.0 {
        dim += 3;
    }
    if uw > 0.0 {
        dim += 2 * mesh.uvs.len();
    }
    if mesh.colors.is_some() && cw > 0.0 {
        dim += 4;
    }
    let gloss = vertex_gloss(mesh, opts);
    let mut attr = Vec::with_capacity(nv * dim);
    if dim > 0 {
        for v in 0..nv {
            if let (Some(ns), true) = (&mesh.normals, nw > 0.0) {
                let g = gloss.as_ref().map_or(1.0, |g| g[v]);
                attr.extend((ns[v] * (nw * g)).to_array());
            }
            if uw > 0.0 {
                for set in &mesh.uvs {
                    attr.extend((set[v] * uw).to_array());
                }
            }
            if let (Some(cs), true) = (&mesh.colors, cw > 0.0) {
                attr.extend((cs[v] * cw).to_array());
            }
        }
    }

    let max_err2 = opts.max_error.map(|e| (e * inv) * (e * inv));
    let nd = if mesh.normals.is_some() && nw > 0.0 {
        3
    } else {
        0
    };
    let uv_dims = if uw > 0.0 { 2 * mesh.uvs.len() } else { 0 };
    // Per wedge, one block (see `St::wd`): its area, its attributes, and per attribute dim its
    // gradient over the surface, the area-weighted mean of the gradients over the wedge's
    // faces (see `St::wcost`).
    let stride = 1 + 4 * dim;
    let wd = {
        let mut wd = vec![0f32; nv * stride];
        for v in 0..nv {
            let b = &mut wd[v * stride..(v + 1) * stride];
            b[0] = warea[v];
            b[1..1 + dim].copy_from_slice(&attr[v * dim..(v + 1) * dim]);
        }
        let mut gw = vec![0f32; nv];
        for t in 0..nt {
            if !alive[t] {
                continue;
            }
            let p = tp[t];
            let x = p.map(|v| pos[v as usize]);
            let n = (x[1] - x[0]).cross(x[2] - x[0]);
            let a2 = n.length_squared();
            if a2 <= 0.0 {
                continue;
            }
            // Gradients of the barycentric basis: n × (opposite edge) / |n|².
            let gphi = [
                n.cross(x[2] - x[1]) / a2,
                n.cross(x[0] - x[2]) / a2,
                n.cross(x[1] - x[0]) / a2,
            ];
            let area = a2.sqrt() * 0.5;
            let w = tris[t];
            for k in 0..dim {
                let g = gphi[0] * attr[w[0] as usize * dim + k]
                    + gphi[1] * attr[w[1] as usize * dim + k]
                    + gphi[2] * attr[w[2] as usize * dim + k];
                let g = g * area;
                for &c in &w {
                    let o = c as usize * stride + 1 + dim + k;
                    wd[o] += g.x;
                    wd[o + dim] += g.y;
                    wd[o + 2 * dim] += g.z;
                }
            }
            for &c in &w {
                gw[c as usize] += area;
            }
        }
        for v in 0..nv {
            if gw[v] > 0.0 {
                let s = 1.0 / gw[v];
                for g in &mut wd[v * stride + 1 + dim..(v + 1) * stride] {
                    *g *= s;
                }
            }
        }
        wd
    };
    let mut st = St {
        uvs: nd..nd + uv_dims,
        same_uv2: SAME_UV2 * uw * uw,
        carry: uv_dims > 0 && !opts.exact_vertices,
        pos,
        tris,
        tp,
        alive,
        quad,
        wd,
        stride,
        dim,
        avg_area,
        locked,
        border_deg,
        border_edges,
        start: vec![0u32; np + 1],
        list: Vec::new(),
        corners: Vec::new(),
        single: Vec::new(),
    };
    // Position of each wedge: its own until it is carried.
    let mut wpos = vpos.clone();
    let mut wremap: Vec<u32> = (0..nv as u32).collect();
    let mut premap: Vec<u32> = (0..np as u32).collect();
    let mut pos_alive = vec![true; np];
    let mut worst2 = 0.0f32;
    let mut part_err2 = 0.0f32;
    let mut collapses = 0usize;
    let mut mark = vec![0u32; np];
    let mut epoch = 0u32;
    let mut used = vec![0u32; np];
    let mut pass = 0u32;
    let mut cands: Vec<(u32, u32, u32, f32)> = Vec::new(); // (cost bits, p, q, mean dist²)
    let mut edges: Vec<Edge> = Vec::new();
    let mut bits: Vec<u32> = Vec::new();
    let mut map: Vec<(u32, u32)> = Vec::new();
    let mut stall = 0;
    // Collapses are stuck above the target: take a part before the next pass.
    let mut stuck = false;
    type Analyzed = Option<Option<crate::reduce::parts::Parts>>;
    let analyze = || {
        crate::reduce::parts::Parts::analyze(
            mesh,
            opts.locked_vertices.as_deref(),
            &opts.transparent_materials,
        )
    };

    // The passes. Parts are analyzed on first need, or, with `ahead`, by the caller while the
    // passes run: then the passes stop where they first need them (returning false).
    let mut passes = |parts: &mut Analyzed, ahead: bool| -> bool {
        loop {
            if stuck {
                // Same rule as the full reducer: cheapest visible removal error first, the floor,
                // and visible-part protection.
                let Some(t) = target else { break };
                if ahead {
                    return false;
                }
                stuck = false;
                let parts = parts.get_or_insert_with(analyze);
                let Some(parts) = parts.as_mut() else { break };
                let limit = if opts.part_protection > 0.0 {
                    (opts.part_protection * diag).max(0.25 * worst2.sqrt() * diag)
                } else {
                    f32::INFINITY
                };
                let floor = t.saturating_sub(alive_tris - t);
                let Some((n, e)) = take_part(
                    &mut st,
                    parts,
                    limit,
                    opts.max_error,
                    alive_tris,
                    floor,
                    &mut pos_alive,
                ) else {
                    break;
                };
                alive_tris -= n;
                part_err2 = part_err2.max((e / diag) * (e / diag));
                stall = 0;
            }
            if target.is_some_and(|t| alive_tris <= t) {
                break;
            }
            // Near an aggressive target, parts whose removal moves the surface less than a quarter of
            // the deviation collapses have accepted go (the full reducer's rule).
            let near = target.filter(|&t| opts.remove_parts && alive_tris <= 4 * t && 10 * t <= nt);
            if near.is_some() && ahead {
                return false;
            }
            if let Some(t) = near
                && let Some(parts) = parts.get_or_insert_with(analyze).as_mut()
            {
                let thr = 0.25 * worst2.sqrt() * diag;
                while let Some((n, e)) = take_part(
                    &mut st,
                    parts,
                    thr,
                    opts.max_error,
                    alive_tris,
                    t,
                    &mut pos_alive,
                ) {
                    alive_tris -= n;
                    part_err2 = part_err2.max((e / diag) * (e / diag));
                }
            }
            pass += 1;
            st.rebuild_stars(nt);
            // Every edge with an allowed direction, its geometric costs first (cheap). Large
            // meshes gather them on all threads, in the same order.
            edges.clear();
            if np < PAR_MIN {
                for p in 0..np as u32 {
                    st.edges_at(p, &pos_alive, &mut edges);
                }
            } else {
                use rayon::prelude::*;
                edges.par_extend(
                    (0..np as u32)
                        .into_par_iter()
                        .with_min_len(PAR_MIN / 4)
                        .fold(Vec::new, |mut out, p| {
                            st.edges_at(p, &pos_alive, &mut out);
                            out
                        })
                        .flatten_iter(),
                );
            }
            if edges.is_empty() {
                break;
            }
            let needed = target.map_or(edges.len(), |t| {
                alive_tris.saturating_sub(t).div_ceil(2).max(1)
            });
            // Pass cost limit (see PASS_SHARE, PASS_LIMIT): passes stay close to cost order. It
            // depends on the m-th cheapest full cost, and a candidate's full cost is at least its
            // geometric one, so only the edges whose geometric cost is under the limit get their
            // attribute cost (see `Edge`).
            let m = needed
                .min((edges.len() as f32 * PASS_SHARE).ceil() as usize)
                .clamp(1, edges.len());
            cands.clear();
            let mut costed = 0;
            let limit = if stall > 0 {
                cost_edges(
                    &st,
                    &mut edges,
                    &mut costed,
                    f32::INFINITY,
                    &mut cands,
                    &mut map,
                );
                f32::INFINITY
            } else {
                // The m-th smallest of `bits` as a cost.
                let nth = |bits: &mut Vec<u32>| f32::from_bits(*bits.select_nth_unstable(m - 1).1);
                // The m-th cheapest geometric cost is a lower bound on the m-th cheapest full
                // cost, so twice it is under the limit.
                bits.clear();
                bits.extend(edges.iter().map(|e| e.low.to_bits()));
                let mut upto = nth(&mut bits) * PASS_LIMIT + 1e-12;
                let mth = loop {
                    cost_edges(&st, &mut edges, &mut costed, upto, &mut cands, &mut map);
                    // The edges not costed yet cost more than `upto`, so with m costed candidates
                    // at or under it, the m-th cheapest is among them.
                    let under = cands.iter().filter(|c| c.0 <= upto.to_bits()).count();
                    bits.clear();
                    bits.extend(cands.iter().map(|c| c.0));
                    if under >= m || costed == edges.len() {
                        break nth(&mut bits);
                    }
                    // Otherwise the m-th cheapest is at least the m-th of the known costs and the
                    // lower bounds left, again within the limit.
                    bits.extend(edges[costed..].iter().map(|e| e.low.to_bits()));
                    let next = nth(&mut bits) * PASS_LIMIT + 1e-12;
                    upto = if next > upto { next } else { f32::INFINITY };
                };
                let limit = mth * PASS_LIMIT + 1e-12;
                if limit > upto {
                    cost_edges(&st, &mut edges, &mut costed, limit, &mut cands, &mut map);
                }
                limit
            };
            // Keys are distinct (one candidate per edge), so any sort gives the same order.
            if cands.len() < PAR_MIN {
                cands.sort_unstable_by_key(|c| (c.0, c.1, c.2));
            } else {
                use rayon::prelude::*;
                cands.par_sort_unstable_by_key(|c| (c.0, c.1, c.2));
            }
            let mut done = 0;
            for &(cb, p, q, g2) in &cands {
                if target.is_some_and(|t| alive_tris <= t) {
                    break;
                }
                if f32::from_bits(cb) > limit {
                    break;
                }
                if max_err2.is_some_and(|m| g2 > m)
                    || used[p as usize] == pass
                    || used[q as usize] == pass
                    || !pos_alive[p as usize]
                    || !pos_alive[q as usize]
                {
                    continue;
                }
                if !st.link_ok(p, q, &mut mark, &mut epoch) || !st.no_flip(p, q, opts.min_face_dot)
                {
                    continue;
                }
                st.wedge_map(p, q, &mut map);
                for &(pw, qw) in &map {
                    if pw == qw {
                        wpos[pw as usize] = q; // carried
                    } else {
                        wremap[pw as usize] = qw;
                    }
                }
                alive_tris -= st.collapse(p, q, &map);
                pos_alive[p as usize] = false;
                premap[p as usize] = q;
                worst2 = worst2.max(g2);
                collapses += 1;
                done += 1;
                // q's triangle list is now stale and p is gone: neither takes part again this pass.
                used[p as usize] = pass;
                used[q as usize] = pass;
            }
            if done == 0 {
                stall += 1;
                if stall > 1 {
                    // Collapses are stuck above the target (closed parts that cannot shrink
                    // further): delete the smallest parts by area (at the top of the loop) and go
                    // on.
                    match target {
                        Some(t) if alive_tris > t && opts.remove_parts => stuck = true,
                        _ => break,
                    }
                }
            } else if done * PASS_STUCK < m && m * 8 < edges.len() {
                stall = 1;
            } else {
                stall = 0;
            }
        }
        true
    };
    // Aggressive targets always need the parts: analyze them alongside the first passes.
    if opts.remove_parts && target.is_some_and(|t| 10 * t <= nt) {
        let (analyzed, done) = rayon::join(analyze, || passes(&mut None, true));
        if !done {
            passes(&mut Some(analyzed), false);
        }
    } else {
        passes(&mut None, false);
    }
    let St { tris, alive, .. } = st;

    // Output.
    let resolve = |mut v: u32| {
        while wremap[v as usize] != v {
            v = wremap[v as usize];
        }
        v
    };
    let mut out = mesh.clone();
    out.indices.clear();
    out.material_ids.clear();
    for t in 0..nt {
        if alive[t] {
            out.indices.extend(tris[t].iter().map(|&v| resolve(v)));
            out.material_ids
                .push(mesh.material_ids.get(t).copied().unwrap_or(0));
        }
    }
    // Carried wedges sit where they were carried to, with what belongs to that position.
    let mut at = vec![u32::MAX; np];
    for v in 0..nv {
        if at[vpos[v] as usize] == u32::MAX {
            at[vpos[v] as usize] = v as u32;
        }
    }
    for v in 0..nv {
        if wpos[v] != vpos[v] {
            let r = at[wpos[v] as usize] as usize;
            out.positions[v] = mesh.positions[r];
            if let Some(skin) = &mut out.skin {
                skin.joints[v] = skin.joints[r];
                skin.weights[v] = skin.weights[r];
            }
            for target in &mut out.morph_targets {
                target.position_deltas[v] = target.position_deltas[r];
            }
        }
    }
    let remap = out.compact();
    let resolve_pos = |mut p: u32| {
        while premap[p as usize] != p {
            p = premap[p as usize];
        }
        p
    };
    let mut pos_out = vec![u32::MAX; np];
    for v in 0..nv {
        if remap[v] != u32::MAX && pos_out[wpos[v] as usize] == u32::MAX {
            pos_out[wpos[v] as usize] = remap[v];
        }
    }
    let vertex_remap = (0..nv as u32)
        .map(|v| {
            let direct = remap[resolve(v) as usize];
            if direct != u32::MAX {
                direct
            } else {
                pos_out[resolve_pos(vpos[v as usize]) as usize]
            }
        })
        .collect();
    let mut source_vertex = vec![0u32; out.vertex_count()];
    for (old, &new) in remap.iter().enumerate() {
        if new != u32::MAX {
            source_vertex[new as usize] = old as u32;
        }
    }
    Ok(SimplifyResult {
        mesh: out,
        error: worst2.max(part_err2).sqrt() * diag,
        relative_error: worst2.max(part_err2).sqrt(),
        vertex_remap,
        collapses,
        symmetry_plane: None,
        source_vertex,
    })
}

/// An edge in its cheaper allowed direction by geometry, a → b. Its candidate (the cheaper
/// full cost over both directions) costs at least `low`.
struct Edge {
    /// `g`, with -0.0 as 0.0.
    low: f32,
    a: u32,
    b: u32,
    /// Geometric cost of a → b and of b → a (infinite when not allowed).
    g: f32,
    g_back: f32,
}

/// Costs the edges after `edges[..*costed]` whose geometric cost is at most `upto`, moving them
/// up to the costed part, and adds their candidates: (cost bits, p, q, mean distance²) for the
/// cheaper allowed direction by geometry plus attributes.
fn cost_edges(
    st: &St,
    edges: &mut [Edge],
    costed: &mut usize,
    upto: f32,
    cands: &mut Vec<(u32, u32, u32, f32)>,
    map: &mut Vec<(u32, u32)>,
) {
    let first = *costed;
    for k in first..edges.len() {
        if edges[k].low <= upto {
            edges.swap(*costed, k);
            *costed += 1;
        }
    }
    let new = &edges[first..*costed];
    // Candidates are sorted by (cost, p, q) later, all distinct, so their order here is free.
    if new.len() < PAR_MIN {
        cands.extend(new.iter().filter_map(|e| st.candidate(e, map)));
    } else {
        use rayon::prelude::*;
        cands.par_extend(
            new.par_iter()
                .with_min_len(PAR_MIN / 4)
                .map_init(Vec::new, |map, e| st.candidate(e, map))
                .flatten(),
        );
    }
}

/// Below this many positions (edges to cost), a pass gathers edges (costs them) on one
/// thread.
const PAR_MIN: usize = 16384;

/// Working state of the fast loop.
struct St {
    pos: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    tp: Vec<[u32; 3]>,
    alive: Vec<bool>,
    quad: Vec<Q>,
    /// Per wedge, `stride` floats side by side (one cache-friendly read in `wcost`): the
    /// surface area the wedge stands for (grows as wedges merge), its `dim` pre-scaled
    /// attributes, then their surface gradients: the x components of all `dim`, the y
    /// components, the z components.
    wd: Vec<f32>,
    stride: usize,
    dim: usize,
    /// The attribute dims that hold the UV sets.
    uvs: std::ops::Range<usize>,
    /// [`SAME_UV2`] in the scaled units of `attr`.
    same_uv2: f32,
    /// Whether wedges torn off their island are carried (see the module docs).
    carry: bool,
    avg_area: f32,
    locked: Vec<bool>,
    border_deg: Vec<u8>,
    /// Open-border edges by position pair, kept current as border vertices slide.
    border_edges: std::collections::HashSet<u64>,
    /// Position → alive triangles (CSR), rebuilt each pass.
    start: Vec<u32>,
    list: Vec<u32>,
    /// Per `list` entry: p's wedge and the other two corners (position, wedge), as of the
    /// start of the pass. Later in the pass only entries naming a collapsed position go stale,
    /// and those are never read (that position and its target are used up for the pass).
    corners: Vec<[u32; 5]>,
    /// Per position: its only wedge, or u32::MAX if a seam passes through it (start of pass).
    single: Vec<u32>,
}

impl St {
    fn rebuild_stars(&mut self, nt: usize) {
        let np = self.pos.len();
        self.start.iter_mut().for_each(|s| *s = 0);
        for t in 0..nt {
            if self.alive[t] {
                for &p in &self.tp[t] {
                    self.start[p as usize + 1] += 1;
                }
            }
        }
        for i in 0..np {
            self.start[i + 1] += self.start[i];
        }
        self.list.clear();
        self.list.resize(self.start[np] as usize, 0);
        self.corners.clear();
        self.corners.resize(self.start[np] as usize, [0; 5]);
        self.single.clear();
        self.single.resize(np, u32::MAX - 1); // MAX-1: no triangle yet
        let mut fill = self.start.clone();
        for t in 0..nt {
            if self.alive[t] {
                let (tp, tw) = (self.tp[t], self.tris[t]);
                for i in 0..3 {
                    let p = tp[i] as usize;
                    let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                    self.list[fill[p] as usize] = t as u32;
                    self.corners[fill[p] as usize] = [tw[i], tp[j], tw[j], tp[k], tw[k]];
                    fill[p] += 1;
                    self.single[p] = match self.single[p] {
                        x if x == u32::MAX - 1 => tw[i],
                        x if x == tw[i] => x,
                        _ => u32::MAX,
                    };
                }
            }
        }
    }

    /// Adds the edges from p to its higher neighbors (see [`Edge`]), in the order its star
    /// first meets them.
    fn edges_at(&self, p: u32, pos_alive: &[bool], out: &mut Vec<Edge>) {
        if !pos_alive[p as usize] {
            return;
        }
        let mut nbr = [0u32; 32];
        let mut n = 0;
        let mut more: Vec<u32> = Vec::new();
        for &t in self.star(p) {
            for &x in &self.tp[t as usize] {
                if x > p && !nbr[..n].contains(&x) && !more.contains(&x) {
                    if n < nbr.len() {
                        nbr[n] = x;
                        n += 1;
                    } else {
                        more.push(x);
                    }
                }
            }
        }
        for &q in nbr[..n].iter().chain(&more) {
            // Both directions share the summed quadric (addition commutes exactly).
            let mut qq = self.quad[p as usize];
            qq.add(&self.quad[q as usize]);
            let mut dirs = [(p, q, f32::INFINITY), (q, p, f32::INFINITY)];
            for d in &mut dirs {
                if self.can(d.0, d.1) {
                    d.2 = qq.eval(self.pos[d.1 as usize]) / self.avg_area;
                }
            }
            if dirs[1].2 < dirs[0].2 {
                dirs.swap(0, 1);
            }
            let (a, b, g) = dirs[0];
            if g.is_finite() {
                out.push(Edge {
                    // Never -0.0, so that its bits order like the value.
                    low: if g > 0.0 { g } else { 0.0 },
                    a,
                    b,
                    g,
                    g_back: dirs[1].2,
                });
            }
        }
    }

    /// An edge's candidate: (cost bits, p, q, mean distance²) for its cheaper allowed
    /// direction by geometry plus attributes. `map` is scratch space.
    fn candidate(&self, e: &Edge, map: &mut Vec<(u32, u32)>) -> Option<(u32, u32, u32, f32)> {
        // The attribute term only where it can still win.
        let mut best: Option<(f32, u32, u32)> = None;
        for (a, b, g) in [(e.a, e.b, e.g), (e.b, e.a, e.g_back)] {
            if !g.is_finite() || best.is_some_and(|x| g >= x.0) {
                continue;
            }
            let Some(ac) = self.attr_cost(a, b, map) else {
                continue;
            };
            let c = g + ac;
            if best.is_none_or(|x| c < x.0) {
                best = Some((c, a, b));
            }
        }
        // Every edge has a triangle on it at the start of a pass, so a candidate.
        debug_assert!(best.is_some());
        let (c, a, b) = best?;
        let mut qq = self.quad[a as usize];
        qq.add(&self.quad[b as usize]);
        // Mean squared distance, for error reporting and max_error.
        let w = (qq.0[0] + qq.0[4] + qq.0[7]).max(1e-30);
        Some((c.to_bits(), a, b, qq.eval(self.pos[b as usize]) / w))
    }

    /// p's triangles as of the start of the pass, dead ones included (callers skip them). For
    /// any position that has not received a collapse this pass, the alive ones are exactly its
    /// current triangles: half-edge collapses rename corners but never add a triangle to a
    /// vertex other than the target.
    #[inline]
    fn star(&self, p: u32) -> &[u32] {
        &self.list[self.start[p as usize] as usize..self.start[p as usize + 1] as usize]
    }

    #[inline]
    fn live(&self, p: u32) -> impl Iterator<Item = usize> + '_ {
        self.star(p)
            .iter()
            .map(|&t| t as usize)
            .filter(|&t| self.alive[t])
    }

    fn is_border_edge(&self, a: u32, b: u32) -> bool {
        self.border_edges.contains(&ekey(a, b))
    }

    /// Whether p may move onto q: not locked, and border vertices slide along their border.
    fn can(&self, p: u32, q: u32) -> bool {
        !self.locked[p as usize] && (self.border_deg[p as usize] == 0 || self.is_border_edge(p, q))
    }

    /// Move from p to q.
    #[inline]
    fn dx(&self, p: u32, q: u32) -> Vec3 {
        self.pos[q as usize] - self.pos[p as usize]
    }

    /// Attribute cost of wedge a, of p, taking wedge b's attributes when p moves `dx` onto q:
    /// a's area × the squared difference between b's attributes and a's carried along its
    /// surface gradient (pre-scaled channels). Where the attributes change linearly over the
    /// surface (UVs inside an island, normals on an evenly curved panel) that is near zero
    /// however far p moves; what remains is what the larger triangles cannot interpolate: a
    /// seam, a fold, a rim whose normal leans away from the faces beside it.
    #[inline]
    fn wcost(&self, a: u32, b: u32, dx: Vec3) -> f32 {
        if self.dim == 0 || a == b {
            return 0.0;
        }
        // Fixed sizes for the usual channel sets (normal and one or two UV sets, with or
        // without colors), so the loops unroll.
        match self.dim {
            3 => self.wcost_n::<3>(a, b, dx),
            5 => self.wcost_n::<5>(a, b, dx),
            7 => self.wcost_n::<7>(a, b, dx),
            9 => self.wcost_n::<9>(a, b, dx),
            d => self.wcost_dyn(d, a, b, dx),
        }
    }

    /// [`wcost`](Self::wcost) for `D` attribute dims.
    #[inline(always)]
    fn wcost_n<const D: usize>(&self, a: u32, b: u32, dx: Vec3) -> f32 {
        let s = self.stride;
        let (a, b) = (a as usize * s, b as usize * s);
        let blk = &self.wd[a..a + s];
        let x: &[f32; D] = blk[1..1 + D].try_into().unwrap();
        let gx: &[f32; D] = blk[1 + D..1 + 2 * D].try_into().unwrap();
        let gy: &[f32; D] = blk[1 + 2 * D..1 + 3 * D].try_into().unwrap();
        let gz: &[f32; D] = blk[1 + 3 * D..1 + 4 * D].try_into().unwrap();
        let y: &[f32; D] = self.wd[b + 1..b + 1 + D].try_into().unwrap();
        let mut e = [0f32; D];
        for k in 0..D {
            e[k] = y[k] - x[k] - (gx[k] * dx.x + gy[k] * dx.y + gz[k] * dx.z);
        }
        let mut e2 = 0.0;
        for e in e {
            e2 += e * e;
        }
        blk[0] * e2
    }

    /// [`wcost`](Self::wcost) for any number of attribute dims.
    fn wcost_dyn(&self, d: usize, a: u32, b: u32, dx: Vec3) -> f32 {
        let s = self.stride;
        let (a, b) = (a as usize * s, b as usize * s);
        let blk = &self.wd[a..a + s];
        let (x, g) = blk[1..].split_at(d);
        let (gx, g) = g.split_at(d);
        let (gy, gz) = g.split_at(d);
        let y = &self.wd[b + 1..b + 1 + d];
        let mut e2 = 0.0;
        for k in 0..d {
            let e = y[k] - x[k] - (gx[k] * dx.x + gy[k] * dx.y + gz[k] * dx.z);
            e2 += e * e;
        }
        blk[0] * e2
    }

    /// Area wedge `w` stands for.
    #[inline]
    fn warea(&self, w: u32) -> f32 {
        self.wd[w as usize * self.stride]
    }

    /// The pre-scaled attributes of wedge `w`.
    #[inline]
    fn attr(&self, w: u32) -> &[f32] {
        let o = w as usize * self.stride + 1;
        &self.wd[o..o + self.dim]
    }

    /// Wedge map p→q: wedges of p in triangles shared with q map to q's wedge there; the others
    /// as [`unshared`](Self::unshared) says, a wedge mapped to itself being carried. Returns
    /// the attribute cost (None if p and q share no triangle).
    fn wedge_map(&self, p: u32, q: u32, out: &mut Vec<(u32, u32)>) -> Option<f32> {
        out.clear();
        let dx = self.dx(p, q);
        let (s, e) = (
            self.start[p as usize] as usize,
            self.start[p as usize + 1] as usize,
        );
        let mut single = true;
        let first_pw = self.corners.get(s).map_or(u32::MAX, |c| c[0]);
        for k in s..e {
            if !self.alive[self.list[k] as usize] {
                continue;
            }
            let [pw, pa, aw, pb, bw] = self.corners[k];
            single &= pw == first_pw;
            let qw = if pa == q {
                aw
            } else if pb == q {
                bw
            } else {
                continue;
            };
            if !out.iter().any(|&(a, _)| a == pw) {
                out.push((pw, qw));
            }
        }
        if out.is_empty() {
            return None;
        }
        let mut cost: f32 = out.iter().map(|&(pw, qw)| self.wcost(pw, qw, dx)).sum();
        if !single {
            let joined = out.len();
            for k in s..e {
                if !self.alive[self.list[k] as usize] {
                    continue;
                }
                let pw = self.corners[k][0];
                if out.iter().any(|&(a, _)| a == pw) {
                    continue;
                }
                let (c, to) = self.unshared(pw, &out[..joined], dx);
                cost += c;
                out.push((pw, to));
            }
        }
        Some(cost / self.avg_area)
    }

    /// Where wedge `pw` of p goes when no triangle joins it to q, given the wedges that are
    /// joined (p's wedge, q's wedge): its area-weighted attribute cost and the wedge it
    /// becomes. With the UVs of a joined wedge it is on that wedge's island (split from it by
    /// a normal or a color) and takes the nearest q wedge of that island. Otherwise the
    /// collapse leaves its seam behind: it is carried (becomes itself, see the module docs)
    /// at the cost of its texture sliding, or, where it cannot be, takes the nearest q wedge.
    fn unshared(&self, pw: u32, joined: &[(u32, u32)], dx: Vec3) -> (f32, u32) {
        let mut on = (f32::INFINITY, u32::MAX);
        let mut off = (f32::INFINITY, u32::MAX);
        let mut slide = 0.0f32;
        for &(sw, qw) in joined {
            let d = self.wcost(pw, qw, dx);
            let best = if self.uv_dist(pw, sw) <= self.same_uv2 {
                &mut on
            } else {
                &mut off
            };
            if d < best.0 || (d == best.0 && qw < best.1) {
                *best = (d, qw);
            }
            slide = slide.max(self.uv_dist(sw, qw));
        }
        let w = self.warea(pw);
        if on.1 != u32::MAX {
            on
        } else if self.carry {
            (w * CARRY_SLIDE * slide, pw)
        } else {
            off
        }
    }

    /// Squared distance between the (pre-scaled) UVs of wedges a and b, over all UV sets.
    fn uv_dist(&self, a: u32, b: u32) -> f32 {
        if a == b {
            return 0.0;
        }
        let (x, y) = (self.attr(a), self.attr(b));
        x[self.uvs.clone()]
            .iter()
            .zip(&y[self.uvs.clone()])
            .map(|(p, q)| (p - q) * (p - q))
            .sum()
    }

    /// Attribute cost of p→q (as [`wedge_map`](Self::wedge_map) returns) without building the
    /// map when p has a single wedge.
    fn attr_cost(&self, p: u32, q: u32, scratch: &mut Vec<(u32, u32)>) -> Option<f32> {
        let pw = self.single[p as usize];
        if pw >= u32::MAX - 1 {
            return self
                .attr_cost_seam(p, q)
                .or_else(|| self.wedge_map(p, q, scratch));
        }
        let (s, e) = (
            self.start[p as usize] as usize,
            self.start[p as usize + 1] as usize,
        );
        for k in s..e {
            let [_, pa, aw, pb, bw] = self.corners[k];
            let qw = if pa == q {
                aw
            } else if pb == q {
                bw
            } else {
                continue;
            };
            if !self.alive[self.list[k] as usize] {
                continue;
            }
            return Some(self.wcost(pw, qw, self.dx(p, q)) / self.avg_area);
        }
        None
    }

    /// [`attr_cost`](Self::attr_cost) for a seam vertex, on the stack: None also when p has
    /// too many wedges for the fixed arrays (the caller falls back to the general map).
    fn attr_cost_seam(&self, p: u32, q: u32) -> Option<f32> {
        const N: usize = 8;
        let dx = self.dx(p, q);
        let mut pws = [0u32; N];
        let mut np = 0;
        let mut maps = [(0u32, 0u32); N];
        let mut nm = 0;
        let (s, e) = (
            self.start[p as usize] as usize,
            self.start[p as usize + 1] as usize,
        );
        for k in s..e {
            if !self.alive[self.list[k] as usize] {
                continue;
            }
            let [pw, pa, aw, pb, bw] = self.corners[k];
            if !pws[..np].contains(&pw) {
                if np == N {
                    return None;
                }
                pws[np] = pw;
                np += 1;
            }
            let qw = if pa == q {
                aw
            } else if pb == q {
                bw
            } else {
                continue;
            };
            if !maps[..nm].iter().any(|m| m.0 == pw) {
                if nm == N {
                    return None;
                }
                maps[nm] = (pw, qw);
                nm += 1;
            }
        }
        if nm == 0 {
            return None;
        }
        let mut cost = 0.0;
        for &pw in &pws[..np] {
            cost += match maps[..nm].iter().find(|m| m.0 == pw) {
                Some(&(_, qw)) => self.wcost(pw, qw, dx),
                None => self.unshared(pw, &maps[..nm], dx).0,
            };
        }
        Some(cost / self.avg_area)
    }

    /// Link condition: p and q may share only the third vertices of their common faces.
    fn link_ok(&self, p: u32, q: u32, mark: &mut [u32], epoch: &mut u32) -> bool {
        *epoch += 1;
        let e1 = *epoch;
        for t in self.live(p) {
            for &x in &self.tp[t] {
                mark[x as usize] = e1;
            }
        }
        *epoch += 1;
        let e2 = *epoch;
        let mut common = 0;
        for t in self.live(q) {
            for &x in &self.tp[t] {
                if x != p && x != q && mark[x as usize] == e1 {
                    mark[x as usize] = e2;
                    common += 1;
                }
            }
        }
        let shared = self.live(p).filter(|&t| self.tp[t].contains(&q)).count();
        common <= shared
    }

    /// No face around p flips or degenerates when p moves to q.
    fn no_flip(&self, p: u32, q: u32, min_dot: f32) -> bool {
        let qp = self.pos[q as usize];
        self.live(p).all(|t| {
            let s = self.tp[t];
            if s.contains(&q) {
                return true;
            }
            let v = s.map(|x| self.pos[x as usize]);
            let n0 = (v[1] - v[0]).cross(v[2] - v[0]);
            let w = s.map(|x| if x == p { qp } else { self.pos[x as usize] });
            let n1 = (w[1] - w[0]).cross(w[2] - w[0]);
            let l1 = n1.length();
            l1 > 1e-12 && n0.dot(n1) >= min_dot * n0.length() * l1
        })
    }

    /// Moves p onto q. Returns triangles removed.
    fn collapse(&mut self, p: u32, q: u32, map: &[(u32, u32)]) -> usize {
        for &(pw, qw) in map {
            if pw != qw {
                let a = self.warea(pw);
                self.wd[qw as usize * self.stride] += a;
            }
        }
        let mut removed = 0;
        let (s, e) = (
            self.start[p as usize] as usize,
            self.start[p as usize + 1] as usize,
        );
        for k in s..e {
            let t = self.list[k] as usize;
            if !self.alive[t] {
                continue;
            }
            if self.tp[t].contains(&q) {
                if self.alive[t] {
                    self.alive[t] = false;
                    removed += 1;
                }
            } else {
                for i in 0..3 {
                    if self.tp[t][i] == p {
                        self.tp[t][i] = q;
                        let w = self.tris[t][i];
                        self.tris[t][i] = map.iter().find(|m| m.0 == w).map_or(w, |m| m.1);
                    }
                }
            }
        }
        let qp = self.quad[p as usize];
        self.quad[q as usize].add(&qp);
        if self.border_deg[p as usize] > 0 {
            // p's border edges (p, r) become (q, r); the edge (p, q) disappears.
            let (s, e) = (
                self.start[p as usize] as usize,
                self.start[p as usize + 1] as usize,
            );
            let mut moved = Vec::new();
            for k in s..e {
                let t = self.list[k] as usize;
                // Dead now (shared with q) or alive: both still list p's border neighbors.
                for &r in &self.tp[t] {
                    if r != p && self.border_edges.remove(&ekey(p, r)) && r != q {
                        moved.push(r);
                    }
                }
            }
            for r in moved {
                self.border_edges.insert(ekey(q, r));
            }
            self.border_deg[q as usize] = self.border_deg[q as usize].max(1);
        }
        removed
    }
}

/// Takes the cheapest removable part (see `Parts::take`) and deletes its live triangles.
/// Returns (triangles removed, removal error in world units).
fn take_part(
    st: &mut St,
    parts: &mut crate::reduce::parts::Parts,
    threshold: f32,
    max_error: Option<f32>,
    alive: usize,
    floor: usize,
    pos_alive: &mut [bool],
) -> Option<(usize, f32)> {
    if alive <= floor {
        return None;
    }
    let alive_ref = &st.alive;
    let live = |ts: &[u32]| ts.iter().filter(|&&x| alive_ref[x as usize]).count();
    let (k, e) = parts.take(threshold, max_error, live, alive, floor)?;
    let mut n = 0;
    for &x in parts.triangles(k) {
        let x = x as usize;
        if st.alive[x] {
            st.alive[x] = false;
            n += 1;
            for p in st.tp[x] {
                pos_alive[p as usize] = false;
            }
        }
    }
    Some((n, e))
}

#[inline]
fn ekey(a: u32, b: u32) -> u64 {
    ((a.min(b) as u64) << 32) | a.max(b) as u64
}

#[cfg(test)]
mod tests {
    use crate::Mesh;
    use crate::reduce::simplify::{Quality, SimplifyOptions, simplify};

    fn opts(ratio: f32) -> SimplifyOptions {
        SimplifyOptions {
            target_ratio: Some(ratio),
            quality: Quality::Fast,
            ..Default::default()
        }
    }

    fn sphere() -> Mesh {
        let mut m = crate::reduce::simplify::tests::uv_sphere(64, 32);
        m.compute_normals(60.0);
        m
    }

    /// Edges by position pair with their face counts.
    fn edge_counts(m: &Mesh) -> std::collections::HashMap<(u32, u32), u32> {
        let key = |p: glam::Vec3| p.to_array().map(f32::to_bits);
        let mut ids = std::collections::HashMap::new();
        let vp: Vec<u32> = m
            .positions
            .iter()
            .map(|p| {
                let n = ids.len() as u32;
                *ids.entry(key(*p)).or_insert(n)
            })
            .collect();
        let mut e = std::collections::HashMap::new();
        for t in 0..m.triangle_count() {
            let tri = m.triangle(t).map(|v| vp[v as usize]);
            for i in 0..3 {
                let (a, b) = (tri[i], tri[(i + 1) % 3]);
                *e.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        e
    }

    #[test]
    fn reaches_targets_and_stays_closed() {
        let m = sphere();
        assert!(edge_counts(&m).values().all(|&c| c == 2));
        for (ratio, mean) in [(0.5, 0.002), (0.1, 0.01), (0.02, 0.03)] {
            let r = simplify(&m, &opts(ratio)).unwrap();
            r.mesh.validate().unwrap();
            let target = (m.triangle_count() as f32 * ratio) as usize;
            assert!(
                r.mesh.triangle_count() <= target,
                "{ratio}: {}",
                r.mesh.triangle_count()
            );
            assert!(
                edge_counts(&r.mesh).values().all(|&c| c == 2),
                "{ratio}: not closed and manifold"
            );
            assert!(
                r.vertex_remap
                    .iter()
                    .all(|&v| (v as usize) < r.mesh.vertex_count())
            );
            let d = crate::metrics::surface_distance(&m, &r.mesh, 4000);
            assert!(d.relative_mean < mean, "{ratio}: {d:?}");
        }
    }

    #[test]
    fn deterministic_and_close_to_best() {
        let m = sphere();
        let (a, b) = (
            simplify(&m, &opts(0.1)).unwrap(),
            simplify(&m, &opts(0.1)).unwrap(),
        );
        assert_eq!(a.mesh, b.mesh);
        let best = simplify(&m, &SimplifyOptions::ratio(0.1)).unwrap();
        let df = crate::metrics::surface_distance(&m, &a.mesh, 8000);
        let db = crate::metrics::surface_distance(&m, &best.mesh, 8000);
        assert!(
            df.relative_mean < db.relative_mean * 2.0,
            "{df:?} vs {db:?}"
        );
    }

    /// A cube of n × n quads per face, each face its own UV island (a sixth of the texture's
    /// width) with its own vertices.
    fn island_cube(n: u32) -> Mesh {
        let (mut pos, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new());
        for f in 0..6u32 {
            let axis = (f / 2) as usize;
            let base = pos.len() as u32;
            for j in 0..=n {
                for i in 0..=n {
                    let (a, b) = (i as f32 / n as f32, j as f32 / n as f32);
                    let mut p = [0.0f32; 3];
                    p[axis] = if f % 2 == 0 { -0.5 } else { 0.5 };
                    p[(axis + 1) % 3] = a - 0.5;
                    p[(axis + 2) % 3] = b - 0.5;
                    pos.push(glam::Vec3::from(p));
                    uv.push(glam::Vec2::new((f as f32 + 0.1 + 0.8 * a) / 6.0, b));
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let v = base + j * (n + 1) + i;
                    let (b, c, d) = (v + 1, v + n + 2, v + n + 1);
                    if f % 2 == 1 {
                        idx.extend_from_slice(&[v, b, c, v, c, d]);
                    } else {
                        idx.extend_from_slice(&[v, c, b, v, d, c]);
                    }
                }
            }
        }
        let mut m = Mesh::new(pos, idx);
        m.uvs = vec![uv];
        m.compute_normals(30.0);
        m
    }

    #[test]
    fn collapses_off_a_seam_keep_triangles_on_their_islands() {
        // Fewer triangles than the cube's 12 can only be had by moving its corners off their
        // seams. Every triangle must still lie on one face's island, and no vertex anywhere
        // but at an input position.
        let m = island_cube(4);
        assert!(edge_counts(&m).values().all(|&c| c == 2));
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(8),
                target_ratio: None,
                ..opts(1.0)
            },
        )
        .unwrap();
        r.mesh.validate().unwrap();
        assert!(r.mesh.triangle_count() < 12, "{}", r.mesh.triangle_count());
        let island = |v: u32| (r.mesh.uvs[0][v as usize].x * 6.0).floor() as u32;
        for t in 0..r.mesh.triangle_count() {
            let [a, b, c] = r.mesh.triangle(t).map(island);
            assert!(a == b && b == c, "triangle {t} spans islands {a} {b} {c}");
        }
        let key = |p: glam::Vec3| p.to_array().map(f32::to_bits);
        let input: std::collections::HashSet<_> = m.positions.iter().map(|&p| key(p)).collect();
        assert!(r.mesh.positions.iter().all(|&p| input.contains(&key(p))));
        assert!(
            r.vertex_remap
                .iter()
                .all(|&v| (v as usize) < r.mesh.vertex_count())
        );
        // Attributes other than the position are still the source vertex's.
        for (v, &s) in r.source_vertex.iter().enumerate() {
            assert_eq!(r.mesh.uvs[0][v], m.uvs[0][s as usize]);
        }
    }

    #[test]
    fn exact_vertices_keep_output_vertices_exact_copies() {
        // Every output vertex is its source vertex unchanged, as callers that rebuild
        // triangles from `source_vertex` need.
        let m = island_cube(4);
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(8),
                target_ratio: None,
                exact_vertices: true,
                ..opts(1.0)
            },
        )
        .unwrap();
        r.mesh.validate().unwrap();
        for (v, &s) in r.source_vertex.iter().enumerate() {
            assert_eq!(r.mesh.positions[v], m.positions[s as usize]);
            assert_eq!(r.mesh.uvs[0][v], m.uvs[0][s as usize]);
        }
    }

    #[test]
    fn open_borders_stay_in_place() {
        // A flat grid: every collapse must keep the outline (border vertices slide along it).
        let n = 20u32;
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.push(glam::Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let i = y * (n + 1) + x;
                idx.extend_from_slice(&[i, i + 1, i + n + 2, i, i + n + 2, i + n + 1]);
            }
        }
        let m = Mesh::new(pos, idx);
        let r = simplify(&m, &opts(0.05)).unwrap();
        let area: f32 = (0..r.mesh.triangle_count())
            .map(|t| {
                let [a, b, c] = r.mesh.triangle_positions(t);
                (b - a).cross(c - a).length() * 0.5
            })
            .sum();
        assert!(
            (area - (n * n) as f32).abs() < 1e-3,
            "outline moved: area {area}"
        );
        assert!(r.mesh.triangle_count() <= 40);
    }
}
