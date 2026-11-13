//! Fast reduction (`Quality::Fast`): a lean half-edge collapse loop in passes, built for speed
//! on one thread.
//!
//! Garland–Heckbert plane quadrics in f32 over positions normalized to the unit bounding
//! box, plus a cheap attribute term: the area-weighted squared distance between each of p's
//! wedges and the q wedge it maps to (normals, UVs, colors). Each pass rebuilds a flat
//! position→triangle index, costs every edge in its cheaper allowed direction, sorts the
//! candidates once, and collapses them in cost order, at most one collapse per neighbourhood
//! per pass, up to a cost limit. There is no candidate heap and no incremental re-ranking:
//! everything is linear passes over flat arrays. Border vertices slide along their border and
//! junction vertices along their non-manifold junction (duplicated double-sided faces count
//! once); locked vertices stay, and collapses that flip or degenerate a face or break the link
//! condition are skipped. When collapses stall above the target, whole parts go by the full
//! reducer's rule (visible removal error, target floor, visible-part protection). Output is
//! deterministic (single-threaded; hash sets are only looked up, never iterated).

use crate::reduce::simplify::{SimplifyOptions, SimplifyResult};
use crate::{Mesh, Result};
use glam::Vec3;

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

/// Attribute scale relative to the default option weights (tuned on the benchmark corpus).
/// Attribute term scales relative to the default option weights, tuned on the benchmark
/// corpus (8 assets × 4 ratios): positions are in the unit-diagonal frame, so attribute
/// differences must be scaled far down to stay comparable to squared distances.
const NORMAL_SCALE: f32 = 1e-5;
const UV_SCALE: f32 = 1e-3;
const COLOR_SCALE: f32 = 1e-3;
/// Pass size: candidates up to `PASS_LIMIT` × the cost of the m-th cheapest, m = what is still
/// needed but at most `PASS_SHARE` of all edges.
const PASS_SHARE: f32 = 0.5;
const PASS_LIMIT: f32 = 2.0;

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
    let mut edges: Vec<u64> = Vec::with_capacity(nt * 3);
    for (t, p) in tp.iter().enumerate() {
        if alive[t] && !twin[t] {
            for i in 0..3 {
                let (a, b) = (p[i].min(p[(i + 1) % 3]), p[i].max(p[(i + 1) % 3]));
                edges.push(((a as u64) << 32) | b as u64);
            }
        }
    }
    edges.sort_unstable();
    let mut locked = vec![false; np];
    let mut border_deg = vec![0u8; np];
    let mut border_edges: Vec<u64> = Vec::new();
    // Junction (non-manifold) edges and how many touch each position.
    let mut junction_edges: Vec<u64> = Vec::new();
    let mut jdeg = vec![0u8; np];
    let mut k = 0;
    while k < edges.len() {
        let mut j = k + 1;
        while j < edges.len() && edges[j] == edges[k] {
            j += 1;
        }
        let (a, b) = ((edges[k] >> 32) as usize, edges[k] as u32 as usize);
        match j - k {
            1 => {
                border_deg[a] = border_deg[a].saturating_add(1);
                border_deg[b] = border_deg[b].saturating_add(1);
                border_edges.push(edges[k]);
            }
            2 => {}
            _ => {
                jdeg[a] = jdeg[a].saturating_add(1);
                jdeg[b] = jdeg[b].saturating_add(1);
                junction_edges.push(edges[k]);
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
    let mut quad = vec![Q::default(); np];
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
    let mut attr = Vec::with_capacity(nv * dim);
    if dim > 0 {
        for v in 0..nv {
            if let (Some(ns), true) = (&mesh.normals, nw > 0.0) {
                attr.extend((ns[v] * nw).to_array());
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
    let mut st = St {
        pos,
        tris,
        tp,
        alive,
        quad,
        warea,
        attr,
        dim,
        nd: if mesh.normals.is_some() && nw > 0.0 {
            3
        } else {
            0
        },
        avg_area,
        locked,
        border_deg,
        border_edges,
        start: vec![0u32; np + 1],
        list: Vec::new(),
        corners: Vec::new(),
        single: Vec::new(),
    };
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
    let mut map: Vec<(u32, u32)> = Vec::new();
    let mut nbr: Vec<u32> = Vec::new();
    let mut stall = 0;
    let mut parts: Option<Option<crate::reduce::parts::Parts>> = None;
    let analyze = || {
        crate::reduce::parts::Parts::analyze(
            mesh,
            opts.locked_vertices.as_deref(),
            &opts.transparent_materials,
        )
    };

    loop {
        if target.is_some_and(|t| alive_tris <= t) {
            break;
        }
        // Near an aggressive target, parts whose removal moves the surface less than a quarter of
        // the deviation collapses have accepted go (the full reducer's rule); analysed on first
        // need.
        if let Some(t) =
            target.filter(|&t| opts.remove_parts && alive_tris <= 4 * t && 10 * t <= nt)
        {
            if let Some(parts) = parts.get_or_insert_with(analyze).as_mut() {
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
        }
        pass += 1;
        st.rebuild_stars(nt);
        // Cost every edge in its cheaper allowed direction.
        cands.clear();
        for p in 0..np as u32 {
            if !pos_alive[p as usize] || st.star(p).is_empty() {
                continue;
            }
            epoch += 1;
            nbr.clear();
            for &t in st.star(p) {
                for &x in &st.tp[t as usize] {
                    if x > p && mark[x as usize] != epoch {
                        mark[x as usize] = epoch;
                        nbr.push(x);
                    }
                }
            }
            for &q in &nbr {
                // Geometry first (cheap); the attribute term only where it can still win.
                let geo = |a: u32, b: u32| {
                    let mut qq = st.quad[a as usize];
                    qq.add(&st.quad[b as usize]);
                    qq.eval(st.pos[b as usize]) / st.avg_area
                };
                let mut dirs = [(p, q, f32::INFINITY), (q, p, f32::INFINITY)];
                for d in &mut dirs {
                    if st.can(d.0, d.1) {
                        d.2 = geo(d.0, d.1);
                    }
                }
                if dirs[1].2 < dirs[0].2 {
                    dirs.swap(0, 1);
                }
                let mut best: Option<(f32, u32, u32)> = None;
                for &(a, b, g) in &dirs {
                    if !g.is_finite() || best.is_some_and(|x| g >= x.0) {
                        continue;
                    }
                    let Some(ac) = st.attr_cost(a, b, &mut map) else {
                        continue;
                    };
                    let c = g + ac;
                    if best.is_none_or(|x| c < x.0) {
                        best = Some((c, a, b));
                    }
                }
                if let Some((c, a, b)) = best {
                    let mut qq = st.quad[a as usize];
                    qq.add(&st.quad[b as usize]);
                    // Mean squared distance, for error reporting and max_error.
                    let w = (qq.0[0] + qq.0[4] + qq.0[7]).max(1e-30);
                    cands.push((c.to_bits(), a, b, qq.eval(st.pos[b as usize]) / w));
                }
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort_unstable_by_key(|c| (c.0, c.1, c.2));
        let needed = target.map_or(cands.len(), |t| {
            alive_tris.saturating_sub(t).div_ceil(2).max(1)
        });
        // Pass cost limit (see PASS_SHARE, PASS_LIMIT): passes stay close to cost order.
        let m = needed
            .min((cands.len() as f32 * PASS_SHARE).ceil() as usize)
            .clamp(1, cands.len());
        let limit = if stall > 0 {
            f32::INFINITY
        } else {
            f32::from_bits(cands[m - 1].0) * PASS_LIMIT + 1e-12
        };
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
            if !st.link_ok(p, q, &mut mark, &mut epoch) || !st.no_flip(p, q, opts.min_face_dot) {
                continue;
            }
            st.wedge_map(p, q, &mut map);
            for &(pw, qw) in &map {
                wremap[pw as usize] = qw;
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
                // further): delete the smallest parts by area and go on.
                match target {
                    Some(t) if alive_tris > t && opts.remove_parts => {
                        // Same rule as the full reducer: cheapest visible removal error first,
                        // the floor, and visible-part protection.
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
                    _ => break,
                }
            }
        } else {
            stall = 0;
        }
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
    let remap = out.compact();
    let resolve_pos = |mut p: u32| {
        while premap[p as usize] != p {
            p = premap[p as usize];
        }
        p
    };
    let mut pos_out = vec![u32::MAX; np];
    for v in 0..nv {
        if remap[v] != u32::MAX && pos_out[vpos[v] as usize] == u32::MAX {
            pos_out[vpos[v] as usize] = remap[v];
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

/// Working state of the fast loop.
struct St {
    pos: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    tp: Vec<[u32; 3]>,
    alive: Vec<bool>,
    quad: Vec<Q>,
    /// Surface area each wedge stands for (grows as wedges merge).
    warea: Vec<f32>,
    attr: Vec<f32>,
    dim: usize,
    /// The first `nd` attribute dims are the normal.
    nd: usize,
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

    /// Squared attribute distance between wedges a and b (pre-scaled channels).
    fn adist(&self, a: u32, b: u32) -> f32 {
        if self.dim == 0 || a == b {
            return 0.0;
        }
        let d = self.dim;
        let (x, y) = (
            &self.attr[a as usize * d..(a as usize + 1) * d],
            &self.attr[b as usize * d..(b as usize + 1) * d],
        );
        let sq = |r: std::ops::Range<usize>| -> f32 {
            x[r.clone()]
                .iter()
                .zip(&y[r])
                .map(|(p, q)| (p - q) * (p - q))
                .sum()
        };
        sq(0..self.nd) + sq(self.nd..d)
    }

    /// Wedge map p→q: wedges of p in triangles shared with q map to q's wedge there; others to
    /// the shared q wedge with the nearest attributes. Returns the attribute cost (None if p
    /// and q share no triangle).
    fn wedge_map(&self, p: u32, q: u32, out: &mut Vec<(u32, u32)>) -> Option<f32> {
        out.clear();
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
        if !single {
            let shared = out.len();
            for k in s..e {
                if !self.alive[self.list[k] as usize] {
                    continue;
                }
                let pw = self.corners[k][0];
                if out.iter().any(|&(a, _)| a == pw) {
                    continue;
                }
                let best = out[..shared]
                    .iter()
                    .map(|&(_, qw)| (self.adist(pw, qw), qw))
                    .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                    .unwrap();
                out.push((pw, best.1));
            }
        }
        Some(
            out.iter()
                .map(|&(pw, qw)| self.warea[pw as usize] * self.adist(pw, qw))
                .sum::<f32>()
                / self.avg_area,
        )
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
            return Some(self.warea[pw as usize] * self.adist(pw, qw) / self.avg_area);
        }
        None
    }

    /// [`attr_cost`](Self::attr_cost) for a seam vertex, on the stack: None also when p has
    /// too many wedges for the fixed arrays (the caller falls back to the general map).
    fn attr_cost_seam(&self, p: u32, q: u32) -> Option<f32> {
        const N: usize = 8;
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
            let d = match maps[..nm].iter().find(|m| m.0 == pw) {
                Some(&(_, qw)) => self.adist(pw, qw),
                None => maps[..nm]
                    .iter()
                    .map(|&(_, qw)| self.adist(pw, qw))
                    .fold(f32::INFINITY, f32::min),
            };
            cost += self.warea[pw as usize] * d;
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
            let a = self.warea[pw as usize];
            self.warea[qw as usize] += a;
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
                // Dead now (shared with q) or alive: both still list p's border neighbours.
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
    use crate::reduce::simplify::{simplify, Quality, SimplifyOptions};
    use crate::Mesh;

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
            assert!(r
                .vertex_remap
                .iter()
                .all(|&v| (v as usize) < r.mesh.vertex_count()));
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
