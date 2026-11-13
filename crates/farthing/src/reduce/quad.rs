//! Quad-aware reduction by edge-ring (quad strip) removal.
//!
//! In a pure-quad mesh every edge belongs to exactly one *edge ring*: start at an edge, step
//! across the quad to its opposite edge, continue into the next quad. Collapsing every edge of
//! a ring ("rungs") removes the whole strip of quads between them and leaves a pure-quad mesh
//! with its edge flow intact. Rings are removed cheapest-first, where a ring's cost is the
//! largest quadric error of its rung collapses (Garland–Heckbert quadrics built from the quad
//! faces). Related work: Daniels et al. 2008, Tarini et al. 2010 (quad simplification).

use crate::quad::QuadMesh;
use crate::reduce::quadric::Quadric;
use crate::{Error, Result};
use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QuadReduceOptions {
    pub target_ratio: Option<f32>,
    pub target_quads: Option<usize>,
    /// Max estimated deviation per rung collapse (world units).
    pub max_error: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct QuadReduceResult {
    pub mesh: QuadMesh,
    pub rings_removed: usize,
    /// Largest estimated deviation of any rung collapse, world units.
    pub error: f32,
}

#[inline]
fn ekey(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

struct Ring {
    quads: Vec<usize>,
    rungs: Vec<(u32, u32)>,
}

/// Finds all edge rings. Rings that cross themselves or share a rung vertex are dropped, and
/// edges of more than two quads are never rungs.
///
/// Linear time: edges are numbered by a parallel sort of the quads' sides, so stepping across
/// a quad is two array lookups, and each edge is walked once.
fn rings(qm: &QuadMesh) -> Vec<Ring> {
    let nq = qm.quads.len();
    // Sides as (edge key, quad·4 + side), sorted: equal keys are adjacent, ordered by quad.
    // Built without a global sort: each vertex a lists its sides to vertices b > a, sorted
    // locally; vertex blocks are generated in parallel and joined in order.
    let nv = qm.positions.len();
    let mut vstart = vec![0u32; nv + 1];
    for quad in &qm.quads {
        for &v in quad {
            vstart[v as usize + 1] += 1;
        }
    }
    for i in 0..nv {
        vstart[i + 1] += vstart[i];
    }
    let mut fill = vstart.clone();
    let mut vcorner = vec![0u32; vstart[nv] as usize];
    for (qi, quad) in qm.quads.iter().enumerate() {
        for (k, &v) in quad.iter().enumerate() {
            vcorner[fill[v as usize] as usize] = (qi * 4 + k) as u32;
            fill[v as usize] += 1;
        }
    }
    let blocks: Vec<Vec<(u64, u32)>> = (0..nv.div_ceil(1024))
        .into_par_iter()
        .map(|blk| {
            let mut out = Vec::new();
            for a in blk * 1024..((blk + 1) * 1024).min(nv) {
                let at = out.len();
                for &c in &vcorner[vstart[a] as usize..vstart[a + 1] as usize] {
                    let (qi, k) = ((c / 4) as usize, (c % 4) as usize);
                    let quad = &qm.quads[qi];
                    // Side k runs k → k+1; side k+3 runs k+3 → k.
                    for (b, side) in [(quad[(k + 1) % 4], k), (quad[(k + 3) % 4], (k + 3) % 4)] {
                        if b as usize > a {
                            out.push((((a as u64) << 32) | b as u64, (qi * 4 + side) as u32));
                        }
                    }
                }
                out[at..].sort_unstable();
            }
            out
        })
        .collect();
    let sides: Vec<(u64, u32)> = blocks.concat();
    // Edge ids in ascending key order; `first[id]..first[id + 1]` are its sides. Per side:
    // its edge and the side across it (BORDER / MULTI for one / more than two quads), packed
    // together so a walk step touches one cache line.
    const BORDER: u32 = u32::MAX - 1;
    const MULTI: u32 = u32::MAX;
    let mut first: Vec<u32> = Vec::with_capacity(sides.len() / 2 + 1);
    let mut info = vec![(0u32, BORDER); nq * 4];
    let mut i = 0;
    while i < sides.len() {
        let mut j = i + 1;
        while j < sides.len() && sides[j].0 == sides[i].0 {
            j += 1;
        }
        let id = first.len() as u32;
        first.push(i as u32);
        for k in i..j {
            let across = match j - i {
                1 => BORDER,
                2 => sides[i + j - 1 - k].1,
                _ => MULTI,
            };
            info[sides[k].1 as usize] = (id, across);
        }
        i = j;
    }
    let ne = first.len();
    first.push(sides.len() as u32);
    let key_of = |id: u32| {
        let k = sides[first[id as usize] as usize].0;
        ((k >> 32) as u32, k as u32)
    };
    let quads_of = |id: u32| &sides[first[id as usize] as usize..first[id as usize + 1] as usize];
    let side_edge = |s: usize| info[s].0;
    // Rings are the classes of rung-able edges (one or two quads) linked to the opposite side
    // of each quad. Union-find gives the classes; a class whose ring would cross itself (both
    // side pairs of a quad in it) is dropped; every other class is walked from its smallest
    // edge, all classes in parallel. Classes do not depend on any visiting order, so neither
    // do the rings.
    let rungable = |id: u32| quads_of(id).len() <= 2;
    let mut parent: Vec<u32> = (0..ne as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let g = parent[parent[x as usize] as usize];
            parent[x as usize] = g;
            x = g;
        }
        x
    }
    for qi in 0..nq {
        for k in 0..2 {
            let (x, y) = (side_edge(qi * 4 + k), side_edge(qi * 4 + k + 2));
            if rungable(x) && rungable(y) {
                let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
                // The smaller root wins, so each root is its class's smallest edge.
                if rx < ry {
                    parent[ry as usize] = rx;
                } else if ry < rx {
                    parent[rx as usize] = ry;
                }
            }
        }
    }
    for x in 0..ne as u32 {
        let r = find(&mut parent, x);
        parent[x as usize] = r;
    }
    let mut crossed = vec![false; ne];
    for qi in 0..nq {
        let (x, y) = (side_edge(qi * 4), side_edge(qi * 4 + 1));
        if rungable(x) && rungable(y) && parent[x as usize] == parent[y as usize] {
            crossed[parent[x as usize] as usize] = true;
        }
    }
    // Walk from edge `start` into side (q, e). Returns (quads, rungs after start, closed).
    let walk = |start: u32, mut q: usize, mut e: usize| {
        let mut quads = Vec::new();
        let mut rungs = Vec::new();
        let mut closed = false;
        loop {
            quads.push(q);
            let (opp, across) = info[q * 4 + (e + 2) % 4];
            if opp == start {
                closed = true;
                break;
            }
            if across == MULTI {
                break; // not a rung: the ring ends at this quad
            }
            rungs.push(opp);
            if across == BORDER {
                break;
            }
            q = (across / 4) as usize;
            e = (across % 4) as usize;
        }
        (quads, rungs, closed)
    };
    let heads: Vec<u32> = (0..ne as u32)
        .filter(|&x| parent[x as usize] == x && rungable(x) && !crossed[x as usize])
        .collect();
    let found: Vec<(Vec<usize>, Vec<u32>)> = heads
        .par_iter()
        .with_min_len(8)
        .map(|&start| {
            let list = quads_of(start);
            let side = |i: usize| ((list[i].1 / 4) as usize, (list[i].1 % 4) as usize);
            let (q0, e0) = side(0);
            let (fq, fr, closed) = walk(start, q0, e0);
            if closed || list.len() < 2 {
                let mut r = vec![start];
                r.extend(fr);
                (fq, r)
            } else {
                let (q1, e1) = side(1);
                let (bq, br, _) = walk(start, q1, e1);
                let mut quads: Vec<usize> = bq.into_iter().rev().collect();
                quads.extend(fq);
                let mut rungs: Vec<u32> = br.into_iter().rev().collect();
                rungs.push(start);
                rungs.extend(fr);
                (quads, rungs)
            }
        })
        .collect();
    // Rings that share a rung vertex are dropped (checked in parallel).
    found
        .into_par_iter()
        .with_min_len(16)
        .filter_map(|(quads, rungs)| {
            let rungs: Vec<(u32, u32)> = rungs.into_iter().map(key_of).collect();
            let mut qs = quads.clone();
            qs.sort_unstable();
            qs.dedup();
            let mut vs: Vec<u32> = rungs.iter().flat_map(|&(a, b)| [a, b]).collect();
            let n = vs.len();
            vs.sort_unstable();
            vs.dedup();
            (qs.len() == quads.len() && vs.len() == n && !quads.is_empty())
                .then_some(Ring { quads, rungs })
        })
        .collect()
}

fn quadrics(qm: &QuadMesh) -> Vec<Quadric> {
    let mut q = vec![Quadric::default(); qm.positions.len()];
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for quad in &qm.quads {
        for e in 0..4 {
            *edge_count
                .entry(ekey(quad[e], quad[(e + 1) % 4]))
                .or_insert(0) += 1;
        }
    }
    for quad in &qm.quads {
        let p = quad.map(|i| qm.positions[i as usize].as_dvec3());
        let qn = (p[2] - p[0]).cross(p[3] - p[1]).normalize_or_zero();
        for tri in [[0, 1, 2], [0, 2, 3]] {
            let (a, b, c) = (p[tri[0]], p[tri[1]], p[tri[2]]);
            let n = (b - a).cross(c - a);
            let area = n.length() * 0.5;
            if area <= 1e-30 {
                continue;
            }
            let nn = n / (2.0 * area);
            let pl = Quadric::plane(nn, -nn.dot(a), area);
            for &k in &tri {
                q[quad[k] as usize].add(&pl);
            }
        }
        // Border constraint planes keep open boundaries in place.
        for e in 0..4 {
            let (i0, i1) = (quad[e], quad[(e + 1) % 4]);
            if edge_count[&ekey(i0, i1)] == 1 {
                let (x0, x1) = (p[e], p[(e + 1) % 4]);
                let dir = x1 - x0;
                let bn = dir.cross(qn).normalize_or_zero();
                if bn != glam::DVec3::ZERO {
                    let bq = Quadric::plane(bn, -bn.dot(x0), dir.length_squared() * 10.0);
                    q[i0 as usize].add(&bq);
                    q[i1 as usize].add(&bq);
                }
            }
        }
    }
    q
}

/// Best merge point for rung (a, b): t ∈ {0, 0.5, 1}, returning (t, mean error²).
fn rung_merge(q: &[Quadric], pos: &[Vec3], a: u32, b: u32) -> (f32, f64) {
    let mut qq = q[a as usize];
    qq.add(&q[b as usize]);
    let (pa, pb) = (pos[a as usize].as_dvec3(), pos[b as usize].as_dvec3());
    [0.0f64, 0.5, 1.0]
        .into_iter()
        .map(|t| (t as f32, qq.mean_error(pa.lerp(pb, t))))
        .min_by(|x, y| {
            x.1.total_cmp(&y.1)
                .then(((x.0 - 0.5).abs()).total_cmp(&(y.0 - 0.5).abs()))
        })
        .unwrap()
}

/// Removes edge rings until the target is met. Input must be a valid pure-quad mesh.
pub fn reduce_quads(input: &QuadMesh, opts: &QuadReduceOptions) -> Result<QuadReduceResult> {
    input.validate()?;
    let n0 = input.quad_count();
    let target = match (opts.target_ratio, opts.target_quads) {
        (Some(r), Some(t)) => ((n0 as f32 * r) as usize).min(t),
        (Some(r), None) => (n0 as f32 * r) as usize,
        (None, Some(t)) => t,
        (None, None) if opts.max_error.is_some() => 0,
        (None, None) => n0 / 2,
    };
    let mut qm = input.clone();
    let mut quad_q = quadrics(&qm);
    let mut removed = 0;
    let mut worst = 0.0f64;
    // Passes: find all rings once, then remove many in cost order, as long as their strips
    // (with every quad touching them) are vertex-disjoint, so their collapses cannot interact
    // and each can be checked on its own. One rebuild per pass instead of per ring keeps this
    // near-linear. A pass takes at most a quarter of the rings, so later rings are costed
    // again on the reduced mesh, close to strict cheapest-first order.
    while qm.quad_count() > target {
        let mut cands: Vec<(f64, Ring)> = rings(&qm)
            .into_par_iter()
            .with_min_len(64)
            .map(|r| {
                let cost = r
                    .rungs
                    .iter()
                    .map(|&(a, b)| rung_merge(&quad_q, &qm.positions, a, b).1)
                    .fold(0.0, f64::max);
                (cost, r)
            })
            .collect();
        cands.par_sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.quads.len().cmp(&b.1.quads.len()))
                .then(a.1.rungs.cmp(&b.1.rungs))
        });
        let cap = cands.len().div_ceil(4).max(1);
        // Quads around each vertex (CSR, ascending).
        let nv = qm.positions.len();
        let mut vstart = vec![0u32; nv + 1];
        for quad in &qm.quads {
            for &v in quad {
                vstart[v as usize + 1] += 1;
            }
        }
        for i in 0..nv {
            vstart[i + 1] += vstart[i];
        }
        let mut fill = vstart.clone();
        let mut vq = vec![0u32; vstart[nv] as usize];
        for (qi, quad) in qm.quads.iter().enumerate() {
            for &v in quad {
                vq[fill[v as usize] as usize] = qi as u32;
                fill[v as usize] += 1;
            }
        }
        let vquads = |v: u32| &vq[vstart[v as usize] as usize..vstart[v as usize + 1] as usize];
        // Never take a ring dearer than the cap-th cheapest: rings blocked this pass by a
        // neighbour wait for the next pass instead of letting dearer ones through.
        let limit = cands.get(cap - 1).map_or(f64::INFINITY, |c| c.0) * (1.0 + 1e-4) + 1e-30;
        let open = cands
            .iter()
            .take_while(|(cost, _)| {
                *cost <= limit && !opts.max_error.is_some_and(|m| cost.sqrt() > m as f64)
            })
            .count();
        cands.truncate(open);
        // Rings are taken serially in cost order. Validity depends only on the mesh as it
        // stands at the start of the pass (nothing changes until the pass is applied), so it is
        // checked in parallel for blocks of the next rings not yet blocked by a neighbour; the
        // block size only decides how much checking is wasted, never the result.
        let hood_of = |ring: &Ring| {
            // Neighbourhood: every quad touching a rung vertex.
            let mut hood: Vec<u32> = ring
                .rungs
                .iter()
                .flat_map(|&(a, b)| [a, b])
                .flat_map(|v| vquads(v).iter().copied())
                .collect();
            hood.sort_unstable();
            hood.dedup();
            hood
        };
        let block = rayon::current_num_threads() * 8;
        let mut claimed = vec![false; nv];
        let touches = |hood: &[u32], claimed: &[bool]| {
            hood.iter()
                .any(|&qi| qm.quads[qi as usize].iter().any(|&v| claimed[v as usize]))
        };
        let mut chosen: Vec<(Ring, Vec<f32>)> = Vec::new();
        let mut left = qm.quad_count();
        let mut cands = cands.into_iter().peekable();
        'pass: while cands.peek().is_some() {
            // The next block of rings: neighbourhoods in parallel, then those already blocked
            // are dropped, then the rest are checked in parallel.
            let next: Vec<(f64, Ring)> = cands.by_ref().take(block).collect();
            let hoods: Vec<Vec<u32>> = next
                .par_iter()
                .with_min_len(4)
                .map(|(_, ring)| hood_of(ring))
                .collect();
            let batch: Vec<(f64, Ring, Vec<u32>)> = next
                .into_iter()
                .zip(hoods)
                .filter(|((_, ring), hood)| ring.quads.len() < left && !touches(hood, &claimed))
                .map(|((cost, ring), hood)| (cost, ring, hood))
                .collect();
            let checked: Vec<Option<Vec<f32>>> = {
                let (qm, quad_q) = (&qm, &quad_q);
                batch
                    .par_iter()
                    .with_min_len(4)
                    .map(|(_, ring, hood)| ring_valid(qm, quad_q, ring, hood))
                    .collect()
            };
            for ((cost, ring, hood), ts) in batch.into_iter().zip(checked) {
                if left <= target || chosen.len() >= cap {
                    break 'pass;
                }
                if ring.quads.len() >= left || touches(&hood, &claimed) {
                    continue;
                }
                let Some(ts) = ts else { continue };
                for &qi in &hood {
                    for &v in &qm.quads[qi as usize] {
                        claimed[v as usize] = true;
                    }
                }
                worst = worst.max(cost);
                left -= ring.quads.len();
                chosen.push((ring, ts));
            }
        }
        if chosen.is_empty() {
            break;
        }
        // Apply every chosen ring at once.
        let mut remap: Vec<u32> = (0..qm.positions.len() as u32).collect();
        let mut drop = vec![false; qm.quads.len()];
        let old_pos = qm.positions.clone();
        let old_uv = qm.uvs.clone();
        for (ring, ts) in &chosen {
            for (&(a, b), &t) in ring.rungs.iter().zip(ts) {
                qm.positions[a as usize] = old_pos[a as usize].lerp(old_pos[b as usize], t);
                if let (Some(u), Some(src)) = (qm.uvs.as_mut(), old_uv.as_ref()) {
                    u[a as usize] = src[a as usize].lerp(src[b as usize], t);
                }
                remap[b as usize] = a;
                let qb = quad_q[b as usize];
                quad_q[a as usize].add(&qb);
            }
            for &qi in &ring.quads {
                drop[qi] = true;
            }
        }
        qm.quads = qm
            .quads
            .iter()
            .enumerate()
            .filter(|&(qi, _)| !drop[qi])
            .map(|(_, q)| q.map(|v| remap[v as usize]))
            .collect();
        removed += chosen.len();
    }
    // Compact unused vertices.
    let mut remap = vec![u32::MAX; qm.positions.len()];
    let mut pos = Vec::new();
    let mut uvs = qm.uvs.as_ref().map(|_| Vec::new());
    for q in &mut qm.quads {
        for v in q.iter_mut() {
            if remap[*v as usize] == u32::MAX {
                remap[*v as usize] = pos.len() as u32;
                pos.push(qm.positions[*v as usize]);
                if let (Some(out), Some(src)) = (uvs.as_mut(), qm.uvs.as_ref()) {
                    out.push(src[*v as usize]);
                }
            }
            *v = remap[*v as usize];
        }
    }
    qm.positions = pos;
    qm.uvs = uvs;
    qm.validate()
        .map_err(|e| Error::InvalidMesh(format!("quad reduction produced an invalid mesh: {e}")))?;
    Ok(QuadReduceResult {
        mesh: qm,
        rings_removed: removed,
        error: worst.sqrt() as f32,
    })
}

/// Per-thread scratch for [`ring_valid`]: vertex and quad arrays stamped per call, so a
/// check costs time in the size of its neighbourhood, not of the mesh.
#[derive(Default)]
struct Scratch {
    stamp: u32,
    /// Vertex → stamp when it is merged away (`target` valid) or a merge target (`moved`).
    gone: Vec<u32>,
    target: Vec<u32>,
    kept: Vec<u32>,
    moved: Vec<Vec3>,
    ring_quad: Vec<u32>,
    edges: Vec<u64>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Whether collapsing every rung of `ring` keeps the quads in `hood` (all quads touching a
/// rung vertex) valid: no degenerate or flipped quad, no edge shared by more than two quads.
/// Returns each rung's merge parameter. Only quads touching rung vertices can change, so the
/// check is local.
fn ring_valid(qm: &QuadMesh, q: &[Quadric], ring: &Ring, hood: &[u32]) -> Option<Vec<f32>> {
    SCRATCH.with(|s| ring_valid_in(qm, q, ring, hood, &mut s.borrow_mut()))
}

fn ring_valid_in(
    qm: &QuadMesh,
    q: &[Quadric],
    ring: &Ring,
    hood: &[u32],
    s: &mut Scratch,
) -> Option<Vec<f32>> {
    let (nv, nq) = (qm.positions.len(), qm.quads.len());
    if s.gone.len() < nv {
        s.gone.resize(nv, 0);
        s.target.resize(nv, 0);
        s.kept.resize(nv, 0);
        s.moved.resize(nv, Vec3::ZERO);
    }
    if s.ring_quad.len() < nq {
        s.ring_quad.resize(nq, 0);
    }
    s.stamp = s.stamp.wrapping_add(1);
    if s.stamp == 0 {
        s.gone.fill(0);
        s.kept.fill(0);
        s.ring_quad.fill(0);
        s.stamp = 1;
    }
    let st = s.stamp;
    let mut ts = Vec::with_capacity(ring.rungs.len());
    for &(a, b) in &ring.rungs {
        let (t, _) = rung_merge(q, &qm.positions, a, b);
        s.moved[a as usize] = qm.positions[a as usize].lerp(qm.positions[b as usize], t);
        s.kept[a as usize] = st;
        s.gone[b as usize] = st;
        s.target[b as usize] = a;
        ts.push(t);
    }
    for &qi in &ring.quads {
        s.ring_quad[qi] = st;
    }
    let remap = |v: u32| {
        if s.gone[v as usize] == st {
            s.target[v as usize]
        } else {
            v
        }
    };
    let pos_of = |v: u32| {
        if s.kept[v as usize] == st {
            s.moved[v as usize]
        } else {
            qm.positions[v as usize]
        }
    };
    let mut edges = std::mem::take(&mut s.edges);
    edges.clear();
    let mut ok = true;
    for &qi in hood {
        if s.ring_quad[qi as usize] == st {
            continue;
        }
        let quad = qm.quads[qi as usize];
        let nq = quad.map(remap);
        if (0..4).any(|i| (i + 1..4).any(|j| nq[i] == nq[j])) {
            ok = false;
            break;
        }
        let n_old = qnormal(&qm.positions, &quad);
        let pv = nq.map(pos_of);
        let n_new = (pv[2] - pv[0]).cross(pv[3] - pv[1]);
        if n_old.dot(n_new) < 0.2 * n_old.length() * n_new.length()
            || n_new.length_squared() < 1e-20
        {
            ok = false;
            break;
        }
        for e in 0..4 {
            let (x, y) = (nq[e], nq[(e + 1) % 4]);
            if s.kept[x as usize] == st || s.kept[y as usize] == st {
                let (a, b) = ekey(x, y);
                edges.push(((a as u64) << 32) | b as u64);
            }
        }
    }
    if ok {
        // No edge at a merged vertex may end up shared by more than two quads.
        edges.sort_unstable();
        ok = !edges.windows(3).any(|w| w[0] == w[2]);
    }
    s.edges = edges;
    ok.then_some(ts)
}

fn qnormal(p: &[Vec3], q: &[u32; 4]) -> Vec3 {
    let v = q.map(|i| p[i as usize]);
    (v[2] - v[0]).cross(v[3] - v[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::surface_distance;

    /// Open cylinder made of quads: `around` segments, `along` rings of quads.
    fn cylinder(around: u32, along: u32) -> QuadMesh {
        let mut pos = Vec::new();
        for j in 0..=along {
            for i in 0..around {
                let a = i as f32 / around as f32 * std::f32::consts::TAU;
                pos.push(Vec3::new(a.cos(), j as f32 / along as f32 * 4.0, a.sin()));
            }
        }
        let mut quads = Vec::new();
        for j in 0..along {
            for i in 0..around {
                let a = j * around + i;
                let b = j * around + (i + 1) % around;
                quads.push([a, a + around, b + around, b]);
            }
        }
        QuadMesh {
            positions: pos,
            uvs: None,
            quads,
        }
    }

    /// Cube with each face subdivided n × n.
    fn subdivided_cube(n: u32) -> QuadMesh {
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
                let p = nrm
                    + u * (2.0 * i as f32 / n as f32 - 1.0)
                    + v * (2.0 * j as f32 / n as f32 - 1.0);
                let key = [
                    (p.x * 1000.0).round() as i32,
                    (p.y * 1000.0).round() as i32,
                    (p.z * 1000.0).round() as i32,
                ];
                *index.entry(key).or_insert_with(|| {
                    m.positions.push(p);
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

    #[test]
    fn cylinder_keeps_its_cross_section() {
        let c = cylinder(24, 32);
        let r = reduce_quads(
            &c,
            &QuadReduceOptions {
                target_ratio: Some(0.25),
                ..Default::default()
            },
        )
        .unwrap();
        r.mesh.validate().unwrap();
        assert!(r.mesh.quad_count() <= 24 * 8, "{}", r.mesh.quad_count());
        // Axial rings are free on a straight cylinder; the 24-gon cross-section must survive.
        let bottom = r.mesh.positions.iter().filter(|p| p.y.abs() < 1e-4).count();
        assert_eq!(bottom, 24, "cross-section lost vertices");
        assert!(r.error < 1e-3);
    }

    #[test]
    fn cube_stays_all_quads_and_close() {
        let c = subdivided_cube(8);
        assert_eq!(c.quad_count(), 384);
        let r = reduce_quads(
            &c,
            &QuadReduceOptions {
                target_ratio: Some(0.1),
                ..Default::default()
            },
        )
        .unwrap();
        r.mesh.validate().unwrap();
        assert!(r.mesh.quad_count() <= 39);
        let d = surface_distance(&c.to_mesh(), &r.mesh.to_mesh(), 4000);
        assert!(d.relative_hausdorff < 0.01, "{d:?}");
    }

    #[test]
    fn same_result_for_any_thread_count() {
        let c = subdivided_cube(40);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    reduce_quads(
                        &c,
                        &QuadReduceOptions {
                            target_ratio: Some(0.1),
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.mesh.quads, b.mesh.quads);
        assert_eq!(a.mesh.positions, b.mesh.positions);
        assert_eq!(a.rings_removed, b.rings_removed);
    }

    #[test]
    fn rings_partition_edges_of_a_grid() {
        let c = cylinder(8, 4);
        let rs = rings(&c);
        // Axial edges form 4 closed loops of 8 quads; circumferential edges form 8 open strips
        // of 4 quads.
        let closed = rs.iter().filter(|r| r.quads.len() == 8).count();
        let open = rs.iter().filter(|r| r.quads.len() == 4).count();
        assert_eq!(closed, 4);
        assert_eq!(open, 8);
    }
}
