//! Realtime reduction (`Quality::Realtime`): adaptive quadric clustering followed by a short
//! collapse tail, for reductions that must finish in a frame.
//!
//! Edge collapse spends nearly all its time on the first 90% of collapses, which hardly
//! matter: the visible error is decided by the last ones. So the first stage gets most of the
//! way in linear time, and only the tail collapses edge by edge.
//!
//! 1. **Octree refinement over a Morton order.** Vertices are radix-sorted by a 30-bit Morton
//!    code, so every octree cell is a contiguous range. Each vertex carries its plane quadric
//!    (area-weighted face planes, plus border planes found without an edge table: the sums of
//!    a vertex's next and previous corner ids, and of their squares, cancel around a closed fan
//!    and give the two border neighbors of an open one). Cells are split top-down, worst first,
//!    by the quadric error at their area-weighted centroid plus a normal-cone term that splits
//!    cells holding opposite sheets. Flat regions stay coarse and curved ones refine, like
//!    greedy collapse and unlike a uniform grid. The split stops at about one cluster per
//!    target triangle (about twice the target in triangles).
//! 2. **Clusters as vertices.** Each cluster keeps the summed quadric of the original surface
//!    and is represented by its member vertex of least error (vertices are never invented, so
//!    attributes stay exact). Members joined by an edge inside the cluster share a wedge:
//!    UV seams and hard edges survive clustering as separate wedges at one position.
//! 3. **Collapse tail.** Lazy greedy half-edge collapse from about twice the target down to
//!    it, with the inherited quadrics, a wedge distance attribute term, border sliding, link
//!    and flip checks. Each live cluster has at most one heap entry, keyed by a geometry-only
//!    lower bound and checked exactly when it surfaces, so most neighbors a collapse touches
//!    are never costed again.
//! 4. **Finish.** Each remaining free cluster moves along its surface normal to the minimum
//!    of its summed quadric (never sideways, which would slide the texture under attributes
//!    copied from one member), and each output normal is the area-weighted mean over every
//!    input vertex merged into its wedge.
//!
//! Every pass before the tail runs in parallel over a fixed number of chunks of the Morton
//! order, set by the vertex count alone. Wedges, seams and clusters all lie inside one octree
//! cell, and cells are contiguous rank ranges, so each chunk finishes its clusters without
//! seeing another's; the shared union-find keeps every set's smallest rank as its root. The
//! result is the same for any thread count, and linear in the input apart from the tail,
//! whose size depends only on the target.

use crate::reduce::simplify::{SimplifyOptions, SimplifyResult};
use crate::{Mesh, Result};
use glam::Vec3;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// Octree depth (Morton bits per axis).
const DEPTH: u32 = 10;
/// Stage-one clusters per target triangle.
const CLUSTERS_PER_TRI: f32 = 0.55;
/// Weight of the normal-cone term (opposite sheets in one cell) against quadric error.
const NORMAL_KAPPA: f64 = 0.125;
/// Share of `max_error` stage one may use, leaving the tail room to collapse.
const STAGE1_ERROR_SHARE: f32 = 0.5;
/// Attribute term scales, as in the fast reducer (positions in the unit-diagonal frame).
const NORMAL_SCALE: f32 = 1e-5;
const UV_SCALE: f32 = 1e-3;
const COLOR_SCALE: f32 = 1e-3;
/// The position solve after the tail: its pull toward the cluster's current position, as a
/// share of the quadric's mean curvature term, and how far a cluster may move, as a share of
/// its shortest edge.
const RELAX_RIDGE: f64 = 1e-3;
const RELAX_REACH: f32 = 0.25;
/// Bit of a sort key where the 30-bit Morton code starts; below it, a position hash and
/// the vertex index share the rest.
const CODE_SHIFT: u32 = 34;
/// Weight of the distance to a cluster's centroid in choosing its representative: enough to
/// settle ties (flat regions, where every member's error is zero, and would otherwise all pick
/// their cells' low corners and fold), too little to override real error.
const CENTROID_PULL: f64 = 1e-5;
/// Fine cells per wanted cluster: the resolution refinement starts from.
const FINE_PER_CLUSTER: usize = 4;

/// Vertices per parallel chunk, and the most chunks; items per task for parallel maps.
const CHUNK_VERTS: usize = 4096;
const MAX_CHUNKS: usize = 128;
const PAR_MIN: usize = 4096;
/// Refinement sums: quadric (10), area, area-weighted normal (3), area-weighted position (3).
const NS: usize = 17;
type Q = [f64; 10];

#[inline]
fn plane(n: Vec3, d: f32, w: f32) -> [f32; 10] {
    let (a, b, c) = (n.x, n.y, n.z);
    [
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
    ]
}

#[inline]
fn qeval(a: &[f64], x: f64, y: f64, z: f64) -> f64 {
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

#[inline]
fn qeval_v(a: &[f64], p: Vec3) -> f64 {
    qeval(a, p.x as f64, p.y as f64, p.z as f64)
}

/// Spreads the low 10 bits of `x` three apart.
#[inline]
fn spread(x: u32) -> u32 {
    let mut x = x & 0x3ff;
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// Sorts keys (code above `CODE_SHIFT`, tie-breaking bits below) in
/// parallel: blocks count their top digits and write their keys straight to place (relaxed
/// atomic stores, plain stores on common targets), then each bucket is sorted on its own.
/// The order is the same however it is split.
fn sort_by_code(keys: Vec<u64>) -> Vec<u64> {
    // One scatter by the top 10 bits of the code, then each bucket (a few hundred keys,
    // in cache) sorted on its own. Keys are distinct, so the order is the full sort's.
    let n = keys.len();
    let blk = (n / (4 * rayon::current_num_threads())).max(PAR_MIN);
    let digit = |k: u64| (k >> (CODE_SHIFT + 3 * DEPTH - 10)) as usize & 1023;
    let mut hist: Vec<[u32; 1024]> = keys
        .par_chunks(blk)
        .map(|b| {
            let mut h = [0u32; 1024];
            for &k in b {
                h[digit(k)] += 1;
            }
            h
        })
        .collect();
    let mut start = [0u32; 1025];
    let mut sum = 0u32;
    for d in 0..1024 {
        start[d] = sum;
        for h in hist.iter_mut() {
            let c = h[d];
            h[d] = sum;
            sum += c;
        }
    }
    start[1024] = sum;
    let dst: Vec<AtomicU64> = (0..n)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|_| AtomicU64::new(0))
        .collect();
    keys.par_chunks(blk)
        .zip(hist.par_iter_mut())
        .for_each(|(b, h)| {
            for &k in b {
                let d = digit(k);
                dst[h[d] as usize].store(k, Relaxed);
                h[d] += 1;
            }
        });
    drop(keys);
    let mut out: Vec<u64> = dst.into_iter().map(AtomicU64::into_inner).collect();
    let mut buckets: Vec<&mut [u64]> = Vec::with_capacity(1024);
    let mut rest = &mut out[..];
    for d in 0..1024 {
        let (b, r) = rest.split_at_mut((start[d + 1] - start[d]) as usize);
        if b.len() > 1 {
            buckets.push(b);
        }
        rest = r;
    }
    buckets
        .into_par_iter()
        .with_min_len(8)
        .for_each(|b| b.sort_unstable());
    out
}

/// A refinement node: a range of fine cells, the leading Morton digits all its codes share
/// (`DEPTH`: one finest cell), and its sums.
struct Node {
    fs: u32,
    fe: u32,
    level: u32,
    st: [f64; NS],
}

/// A fine cell: a range of the Morton order and its sums.
struct FCell {
    s: u32,
    e: u32,
    st: [f32; NS],
}

/// A vertex's face sums: area-weighted normal, area, and the corner-id sums that find open
/// fans (wrapping; exact for the values they are read for).
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct VSum {
    n: [f32; 3],
    area: f32,
    s1: i32,
    s2: i64,
}

/// The two border neighbors of an open fan v, a1, ..., ak: Σnext − Σprev = a1 − ak and
/// Σnext² − Σprev² = a1² − ak², so a1 + ak = (a1² − ak²) / (a1 − ak).
fn fan_ends(v: usize, s: &VSum, nv: usize) -> Option<[usize; 2]> {
    let d1 = s.s1 as i64;
    if d1 == 0 {
        return None;
    }
    // A float quotient, checked exactly: the true sum is under 2^33.
    // Rounded by truncation (`round` is a library call without SSE4.1).
    let q = s.s2 as f64 / d1 as f64;
    let sum = (q + 0.5f64.copysign(q)) as i64;
    if d1.wrapping_mul(sum) != s.s2 {
        return None;
    }
    let x = d1.wrapping_add(sum);
    if x & 1 != 0 {
        return None;
    }
    let (a1, ak) = (x / 2, sum.wrapping_sub(x / 2));
    let ok = |o: i64| (0..nv as i64).contains(&o) && o != v as i64;
    (ok(a1) && ok(ak)).then_some([a1 as usize, ak as usize])
}

/// A vertex's quadric from its face sums, moment-matched: the fan's face-normal second
/// moment M = Σ Aᵢ nᵢnᵢᵀ has trace A, and with mean normal length |Σ Aᵢnᵢ| = A·cos θ for a
/// cone of normals, M ≈ (A cos²θ) uuᵀ + (A sin²θ / 2)(I − uuᵀ). Flat fans give their exact
/// plane; curved ones also resist tangential motion, as their faces would. Open fans add the
/// border planes of `borders`, its end edges that are borders (`u32::MAX`: not one).
fn vertex_quadric(v: usize, s: &VSum, upos: &[Vec3], bw: f32, borders: [u32; 2]) -> [f32; 10] {
    // In terms of n = Σ Aᵢnᵢ itself (u = n / |n|), so one reciprocal serves every term.
    let n = Vec3::from_array(s.n);
    let (a, n2) = (s.area, n.length_squared());
    if !(a > 0.0 && n2 > 0.0) {
        return [0.0; 10];
    }
    let p = upos[v];
    let inv = 1.0 / (a * n2);
    let inv_n2 = a * inv;
    let c2 = (n2 * n2 * inv).min(a);
    let beta = 0.5 * (a - c2);
    let k = (c2 - beta) * inv_n2;
    let mp = n * (k * n.dot(p)) + p * beta;
    let mut q = [
        k * n.x * n.x + beta,
        k * n.x * n.y,
        k * n.x * n.z,
        -mp.x,
        k * n.y * n.y + beta,
        k * n.y * n.z,
        -mp.y,
        k * n.z * n.z + beta,
        -mp.z,
        p.dot(mp),
    ];
    if bw > 0.0 && borders != [u32::MAX; 2] {
        // Planes through the border edges, perpendicular to the surface, weighted by
        // |e × u|² = |e|² sin²θ: the edge length squared for edges in the tangent plane.
        let w = bw * inv_n2;
        for o in borders {
            if o == u32::MAX {
                continue;
            }
            let m = (upos[o as usize] - p).cross(n);
            let b = plane(m, -m.dot(p), w);
            for i in 0..10 {
                q[i] += b[i];
            }
        }
    }
    q
}

/// Adds a vertex to refinement sums: quadric, area, area-weighted normal and position.
#[inline]
fn vstats(st: &mut [f32; NS], q: &[f32; 10], s: &VSum, p: Vec3) {
    for i in 0..10 {
        st[i] += q[i];
    }
    st[10] += s.area;
    st[11] += s.n[0];
    st[12] += s.n[1];
    st[13] += s.n[2];
    st[14] += s.area * p.x;
    st[15] += s.area * p.y;
    st[16] += s.area * p.z;
}

impl VSum {
    /// Adds one face's share.
    #[inline]
    fn add(&mut self, d: &VSum) {
        for i in 0..3 {
            self.n[i] += d.n[i];
        }
        self.area += d.area;
        self.s1 = self.s1.wrapping_add(d.s1);
        self.s2 = self.s2.wrapping_add(d.s2);
    }
}

/// The root of v in a union-find shared between threads. Links only ever point at a smaller
/// vertex, so chains end, and halving them (to any ancestor) is safe without a swap.
fn afind(p: &[AtomicU32], mut v: u32) -> u32 {
    loop {
        let q = p[v as usize].load(Relaxed);
        if q == v {
            return v;
        }
        let g = p[q as usize].load(Relaxed);
        if g != q {
            p[v as usize].store(g, Relaxed);
        }
        v = g;
    }
}

/// Joins the sets of a and b under the smaller root. Every set ends up rooted at its
/// smallest member whatever order the joins run in, so the result is deterministic.
fn aunion(p: &[AtomicU32], mut a: u32, mut b: u32) {
    loop {
        (a, b) = (afind(p, a), afind(p, b));
        if a == b {
            return;
        }
        let (lo, hi) = (a.min(b), a.max(b));
        if p[hi as usize]
            .compare_exchange(hi, lo, Relaxed, Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

/// Splits a slice at the given ascending end offsets (the last is its length).
fn split_ends<T>(mut s: &mut [T], ends: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    let mut at = 0;
    for e in ends {
        let (a, b) = std::mem::take(&mut s).split_at_mut(e - at);
        out.push(a);
        s = b;
        at = e;
    }
    out
}

/// One chunk's share of the per-rank arrays: whole cells, so every set it joins is its own.
struct Chunk<'a> {
    r0: usize,
    c0: usize,
    parent: &'a mut [u32],
    cid: &'a mut [u32],
    wbest: &'a mut [(f32, u32)],
    warea: &'a mut [f32],
    cell_st: &'a mut [[f64; NS]],
}

/// A chunk's clusters, numbered from zero in rank order.
struct ChunkOut {
    cq: Vec<Q>,
    best: Vec<(f64, u32)>,
    clocked: Vec<bool>,
    worst1: f64,
}

fn find(parent: &mut [u32], mut v: u32) -> u32 {
    while parent[v as usize] != v {
        let p = parent[parent[v as usize] as usize];
        parent[v as usize] = p;
        v = p;
    }
    v
}

/// Points every vertex straight at its root; needs parent[x] <= x, which union keeps.
fn flatten(parent: &mut [u32]) {
    for v in 0..parent.len() {
        parent[v] = parent[parent[v] as usize];
    }
}

/// Joins the sets of a and b under the smaller root.
fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb) as usize] = ra.min(rb);
    }
}

/// Open-addressing map from a key to a slot index, for the small stage-two tables.
struct KeyMap {
    keys: Vec<u128>,
    vals: Vec<u32>,
    mask: usize,
}

impl KeyMap {
    fn new(n: usize) -> KeyMap {
        let cap = (n * 2).next_power_of_two().max(16);
        KeyMap {
            keys: vec![u128::MAX; cap],
            vals: vec![0; cap],
            mask: cap - 1,
        }
    }
    #[inline]
    fn slot(&self, k: u128) -> usize {
        let h = (k as u64 ^ ((k >> 64) as u64).rotate_left(29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut i = (h >> 32) as usize & self.mask;
        while self.keys[i] != u128::MAX && self.keys[i] != k {
            i = (i + 1) & self.mask;
        }
        i
    }
    /// The value for `k`, inserting `v` if absent; returns (value, was it present).
    fn get_or_insert(&mut self, k: u128, v: u32) -> (u32, bool) {
        let i = self.slot(k);
        if self.keys[i] == k {
            (self.vals[i], true)
        } else {
            self.keys[i] = k;
            self.vals[i] = v;
            (v, false)
        }
    }
}

pub(crate) fn simplify_realtime(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: Option<usize>,
) -> Result<SimplifyResult> {
    let nv = mesh.vertex_count();
    let nt = mesh.triangle_count();
    let prof = std::env::var_os("FARTHING_RT_PROFILE").is_some();
    let mut clock = std::time::Instant::now();
    let mut lap = |what: &str| {
        if prof {
            eprintln!(
                "  {what:<10} {:7.2} ms",
                clock.elapsed().as_secs_f64() * 1e3
            );
            clock = std::time::Instant::now();
        }
    };
    if prof {
        eprintln!("  {nv} vertices, {nt} triangles");
    }
    let bounds = mesh.bounds();
    let diag = bounds.diagonal().max(1e-12);
    let inv = 1.0 / diag;
    let center = (bounds.min + bounds.max) * 0.5;
    let ext = (bounds.max - bounds.min).max_element().max(1e-12);

    // Morton order: keys of code, position hash and vertex, sorted.
    let index_bits = (usize::BITS - nv.saturating_sub(1).leading_zeros()).max(1);
    let hash_bits = CODE_SHIFT - index_bits;
    let cell = (1u32 << DEPTH) as f32 / ext;
    let top = ((1u32 << DEPTH) - 1) as f32;
    let keys: Vec<u64> = (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|v| {
            let g = ((mesh.positions[v] - bounds.min) * cell).clamp(Vec3::ZERO, Vec3::splat(top));
            let c = (spread(g.x as u32) << 2) | (spread(g.y as u32) << 1) | spread(g.z as u32);
            // A hash of the exact position between code and index: vertices at one position
            // sort next to each other (with the odd stranger whose hash collides).
            let q = mesh.positions[v] + Vec3::ZERO;
            let h = (q.x.to_bits() ^ q.y.to_bits().rotate_left(11) ^ q.z.to_bits().rotate_left(22))
                .wrapping_mul(0x9E37_79B1);
            let h = (h as u64 >> 1) >> (31 - hash_bits);
            ((c as u64) << CODE_SHIFT) | (h << index_bits) | v as u64
        })
        .collect();
    let keys = sort_by_code(keys);
    let code = |r: usize| (keys[r] >> CODE_SHIFT) as u32;
    let vert = |r: usize| (keys[r] & ((1 << index_bits) - 1)) as u32;

    // From here on vertices are Morton ranks: positions (unit-diagonal frame, centered) are
    // gathered once, triangle corners map through `rank`, and every per-vertex pass after
    // reads its arrays in order.
    let arank: Vec<AtomicU32> = (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|_| AtomicU32::new(0))
        .collect();
    let spos: Vec<Vec3> = (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|r| {
            let v = vert(r) as usize;
            arank[v].store(r as u32, Relaxed);
            (mesh.positions[v] - center) * inv
        })
        .collect();
    let rank: Vec<u32> = arank.into_iter().map(AtomicU32::into_inner).collect();
    lap("morton");

    // The parallel passes run over a fixed number of chunks of the Morton order, set by the
    // vertex count alone, so the result never depends on the thread count. Triangles go to
    // the chunk of their first corner in rank order, kept in index order within it.
    let csize = nv.div_ceil((nv / CHUNK_VERTS).clamp(1, MAX_CHUNKS));
    let nch = nv.div_ceil(csize);
    let tblock = nt.div_ceil(nch).max(1);
    let routed: Vec<Vec<Vec<([u32; 3], u32)>>> = (0..nt.div_ceil(tblock))
        .into_par_iter()
        .map(|b| {
            let mut out = vec![Vec::new(); nch];
            for t in b * tblock..((b + 1) * tblock).min(nt) {
                let tr = mesh.triangle(t).map(|v| rank[v as usize]);
                out[tr[0].min(tr[1]).min(tr[2]) as usize / csize].push((tr, t as u32));
            }
            out
        })
        .collect();
    let chunk_tris = |k: usize| routed.iter().flat_map(move |b| b[k].iter());

    // Face sums per vertex. A chunk owns its vertices' sums; corners past its end are handed
    // on and added after, in chunk order.
    let mut vs = vec![VSum::default(); nv];
    let parts: Vec<(f64, usize, Vec<(u32, VSum)>)> = vs
        .par_chunks_mut(csize)
        .enumerate()
        .map(|(k, own)| {
            let lo = (k * csize) as u32;
            let (mut total, mut live, mut spill) = (0f64, 0usize, Vec::new());
            for &([a, b, c], _) in chunk_tris(k) {
                if a == b || b == c || a == c {
                    continue;
                }
                let (pa, pb, pc) = (spos[a as usize], spos[b as usize], spos[c as usize]);
                let mut h = (pb - pa).cross(pc - pa) * 0.5;
                let mut area = h.length();
                if area > 0.0 {
                    total += area as f64;
                    live += 1;
                } else {
                    (h, area) = (Vec3::ZERO, 0.0);
                }
                for (v, nx, pv) in [(a, b, c), (b, c, a), (c, a, b)] {
                    let d = VSum {
                        n: h.to_array(),
                        area,
                        s1: (nx as i32).wrapping_sub(pv as i32),
                        s2: (nx as i64 * nx as i64).wrapping_sub(pv as i64 * pv as i64),
                    };
                    match own.get_mut((v - lo) as usize) {
                        Some(s) => s.add(&d),
                        None => spill.push((v, d)),
                    }
                }
            }
            (total, live, spill)
        })
        .collect();
    let (mut total_area, mut live) = (0f64, 0usize);
    for (t, l, spill) in parts {
        total_area += t;
        live += l;
        for (v, d) in spill {
            vs[v as usize].add(&d);
        }
    }
    if live == 0 {
        return Ok(unchanged(mesh));
    }
    let avg_area = (total_area / live as f64).max(1e-30);
    // Border edges, worked out once per vertex: an open fan's end edge is a border only if
    // the surface stops there. Where another vertex at the same position, facing the same
    // way, has an end edge to the same place, the two fans meet along it: a seam (split UVs or
    // normals) or unwelded faces, where border planes would pin vertices for no reason.
    // Vertices at one position are neighbors in rank (see `sort_by_code`). (The back
    // of a double-sided sheet is there too, facing the other way, and doesn't count.)
    const NONE: [u32; 2] = [u32::MAX; 2];
    let bw = opts.border_weight;
    // Each vertex's position, as the first rank at it; end edges are then compared by these
    // alone, reading only a vertex's rank neighbors.
    let first_at: Vec<u32> = if bw > 0.0 {
        (0..nv)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .map(|r| {
                let mut q = r;
                while q > 0 && spos[q - 1] == spos[r] {
                    q -= 1;
                }
                q as u32
            })
            .collect()
    } else {
        Vec::new()
    };
    let ends: Vec<[u32; 2]> = (0..first_at.len())
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|r| fan_ends(r, &vs[r], nv).map_or(NONE, |e| e.map(|x| first_at[x])))
        .collect();
    drop(first_at);
    let borders: Vec<[u32; 2]> = (0..ends.len())
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|r| {
            let mut mine = ends[r];
            if mine == NONE {
                return NONE;
            }
            let p = spos[r];
            let n = Vec3::from_array(vs[r].n);
            let n2 = n.length_squared();
            let meet = |mut mine: [u32; 2], q: usize| {
                if (ends[q][0] != mine[0] && ends[q][0] != mine[1])
                    && (ends[q][1] != mine[0] && ends[q][1] != mine[1])
                {
                    return mine;
                }
                // Facing within about 154° of each other (cos > −0.9), without the root.
                let m = Vec3::from_array(vs[q].n);
                let d = n.dot(m);
                if d < 0.0 && d * d >= 0.81 * n2 * m.length_squared() {
                    return mine;
                }
                for o in &mut mine {
                    if ends[q].contains(o) {
                        *o = u32::MAX;
                    }
                }
                mine
            };
            let mut q = r;
            while q > 0 && spos[q - 1] == p && mine != NONE {
                q -= 1;
                mine = meet(mine, q);
            }
            let mut q = r + 1;
            while q < nv && spos[q] == p && mine != NONE {
                mine = meet(mine, q);
                q += 1;
            }
            mine
        })
        .collect();
    drop(ends);
    // Importance scales a vertex's error, so clusters stay finer and collapse later there.
    let importance = opts.vertex_importance.as_deref();
    let vq = |r: usize| {
        let mut q = vertex_quadric(
            r,
            &vs[r],
            &spos,
            bw,
            borders.get(r).copied().unwrap_or(NONE),
        );
        if let Some(w) = importance.and_then(|i| i.get(vert(r) as usize)) {
            for x in &mut q {
                *x *= w.max(0.0);
            }
        }
        q
    };
    lap("faces");

    // Fine cells: the coarsest level with about FINE_PER_CLUSTER cells per wanted cluster,
    // summed in one ordered pass. Refinement works on these and only goes back to vertices
    // for a fine cell that must split further.
    let k_target = target.map(|t| ((t as f32 * CLUSTERS_PER_TRI).ceil() as usize).max(4));
    let fine_level = {
        const NL: usize = DEPTH as usize + 1;
        let first_split = (1..nv)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .fold(
                || [0usize; NL],
                |mut h, r| {
                    let x = code(r) ^ code(r - 1);
                    if x != 0 {
                        h[((3 * DEPTH - 1 - (31 - x.leading_zeros())) / 3) as usize] += 1;
                    }
                    h
                },
            )
            .reduce(
                || [0usize; NL],
                |mut a, b| {
                    for i in 0..NL {
                        a[i] += b[i];
                    }
                    a
                },
            );
        let want = k_target.unwrap_or(nv / 8).saturating_mul(FINE_PER_CLUSTER);
        let mut cells = 1;
        let mut level = DEPTH;
        for (l, &n) in first_split.iter().enumerate() {
            if cells >= want {
                level = l as u32;
                break;
            }
            cells += n;
        }
        level
    };
    let mut fcells: Vec<FCell> = {
        let shift = 3 * (DEPTH - fine_level);
        let mut starts: Vec<u32> = vec![0];
        starts.par_extend(
            (1..nv)
                .into_par_iter()
                .with_min_len(PAR_MIN)
                .filter(|&r| (code(r) ^ code(r - 1)) >> shift != 0)
                .map(|r| r as u32),
        );
        starts.push(nv as u32);
        starts
            .windows(2)
            .map(|w| FCell {
                s: w[0],
                e: w[1],
                st: [0.0; NS],
            })
            .collect()
    };
    fcells.par_iter_mut().with_min_len(16).for_each(|f| {
        for r in f.s as usize..f.e as usize {
            vstats(&mut f.st, &vq(r), &vs[r], spos[r]);
        }
    });
    lap("fine");

    // Refinement: split the worst cell until there are enough.
    let cube_to_unit = (ext * inv) as f64;
    let vlevel = |s: usize, e: usize| -> u32 {
        let x = code(s) ^ code(e - 1);
        if x == 0 {
            DEPTH
        } else {
            (3 * DEPTH - 1 - (31 - x.leading_zeros())) / 3
        }
    };
    let err = |st: &[f64; NS], level: u32| -> (f64, f64) {
        let area = st[10];
        if area <= 0.0 {
            return (0.0, 0.0);
        }
        let e = qeval(st, st[14] / area, st[15] / area, st[16] / area);
        let nlen = (st[11] * st[11] + st[12] * st[12] + st[13] * st[13]).sqrt();
        let h = cube_to_unit * 0.5f64.powi(level as i32);
        let en = (area - nlen).max(0.0) * h * h * NORMAL_KAPPA;
        (e + en, e / area)
    };
    let mut nodes: Vec<Node> = Vec::with_capacity(k_target.unwrap_or(1024) * 2);
    {
        let mut st = [0f64; NS];
        for f in &fcells {
            for i in 0..NS {
                st[i] += f.st[i] as f64;
            }
        }
        nodes.push(Node {
            fs: 0,
            fe: fcells.len() as u32,
            level: vlevel(0, nv),
            st,
        });
    }
    let split = |nodes: &mut Vec<Node>, fcells: &mut Vec<FCell>, i: usize, out: &mut Vec<u32>| {
        out.clear();
        let (mut fs, mut fe, level) = (nodes[i].fs as usize, nodes[i].fe as usize, nodes[i].level);
        let shift = 3 * (DEPTH - 1 - level);
        if fe - fs == 1 {
            // One fine cell: break it into finer ones from its vertices.
            let (s, e) = (fcells[fs].s as usize, fcells[fs].e as usize);
            fs = fcells.len();
            let mut r = s;
            while r < e {
                let d = (code(r) >> shift) & 7;
                let mut f = FCell {
                    s: r as u32,
                    e: 0,
                    st: [0.0; NS],
                };
                while r < e && (code(r) >> shift) & 7 == d {
                    vstats(&mut f.st, &vq(r), &vs[r], spos[r]);
                    r += 1;
                }
                f.e = r as u32;
                fcells.push(f);
            }
            fe = fcells.len();
        }
        let mut k = fs;
        while k < fe {
            let d = (code(fcells[k].s as usize) >> shift) & 7;
            let start = k;
            let mut st = [0f64; NS];
            while k < fe && (code(fcells[k].s as usize) >> shift) & 7 == d {
                for (a, b) in st.iter_mut().zip(&fcells[k].st) {
                    *a += *b as f64;
                }
                k += 1;
            }
            out.push(nodes.len() as u32);
            nodes.push(Node {
                fs: start as u32,
                fe: k as u32,
                level: vlevel(fcells[start].s as usize, fcells[k - 1].e as usize),
                st,
            });
        }
    };
    let max1 = opts
        .max_error
        .map(|e| (e * inv * STAGE1_ERROR_SHARE) as f64)
        .map(|e| e * e);
    let mut leaves: Vec<u32> = Vec::new();
    let mut kids = Vec::new();
    let mut heap: BinaryHeap<(u64, std::cmp::Reverse<u32>)> = BinaryHeap::new();
    if nodes[0].level == DEPTH {
        leaves.push(0);
    } else {
        heap.push((
            err(&nodes[0].st, nodes[0].level).0.to_bits(),
            std::cmp::Reverse(0),
        ));
    }
    let mut count = 1usize;
    if let Some(k) = k_target {
        while count < k {
            let Some((_, std::cmp::Reverse(i))) = heap.pop() else {
                break;
            };
            split(&mut nodes, &mut fcells, i as usize, &mut kids);
            count += kids.len() - 1;
            for &c in &kids {
                let n = &nodes[c as usize];
                if n.level == DEPTH {
                    leaves.push(c);
                } else {
                    heap.push((err(&n.st, n.level).0.to_bits(), std::cmp::Reverse(c)));
                }
            }
        }
    }
    let mut rest: Vec<u32> = heap.into_iter().map(|x| x.1.0).collect();
    if let Some(m) = max1 {
        // Within the error budget everywhere (stage one's share of it).
        while let Some(i) = rest.pop() {
            let n = &nodes[i as usize];
            if n.level == DEPTH || err(&n.st, n.level).1 <= m {
                leaves.push(i);
                continue;
            }
            split(&mut nodes, &mut fcells, i as usize, &mut kids);
            rest.extend_from_slice(&kids);
        }
    } else {
        leaves.append(&mut rest);
    }
    // Cells in Morton order, as rank ranges, grouped into chunks of whole cells.
    let mut cells: Vec<(u32, u32, u32)> = leaves
        .iter()
        .map(|&i| {
            let n = &nodes[i as usize];
            (fcells[n.fs as usize].s, fcells[n.fe as usize - 1].e, i)
        })
        .collect();
    cells.sort_unstable_by_key(|c| c.0);
    let ncell = cells.len();
    let mut cellr = vec![0u32; nv];
    let mut cell_st: Vec<[f64; NS]> = Vec::with_capacity(ncell);
    let mut chunks: Vec<[usize; 4]> = Vec::new();
    let mut c0 = 0;
    for (c, &(s, e, i)) in cells.iter().enumerate() {
        cellr[s as usize..e as usize].fill(c as u32);
        cell_st.push(nodes[i as usize].st);
        if e as usize >= (chunks.len() + 1) * csize || c + 1 == ncell {
            chunks.push([cells[c0].0 as usize, e as usize, c0, c + 1]);
            c0 = c + 1;
        }
    }
    drop(cells);
    drop(nodes);
    drop(leaves);
    drop(fcells);
    lap("octree");

    // Wedges: members joined by an edge inside their cell. A triangle with all three corners
    // in different cells has them in different clusters too (two corners in one cell share
    // an edge, hence a cluster), so the kept set is final here. Two edges of a triangle
    // connect all three corners.
    let aparent: Vec<AtomicU32> = (0..nv as u32).map(AtomicU32::new).collect();
    let kept: Vec<Vec<([u32; 3], u32)>> = (0..nch)
        .into_par_iter()
        .map(|k| {
            let mut kept = Vec::new();
            for &(r, t) in chunk_tris(k) {
                let c = r.map(|r| cellr[r as usize]);
                if c[0] == c[1] {
                    aunion(&aparent, r[0], r[1]);
                } else if c[0] != c[2] && c[1] != c[2] {
                    kept.push((r, t));
                    continue;
                }
                if c[1] == c[2] {
                    aunion(&aparent, r[1], r[2]);
                } else if c[0] == c[2] {
                    aunion(&aparent, r[0], r[2]);
                }
            }
            kept
        })
        .collect();
    drop(routed);
    let kept = kept.concat();
    let mut parent: Vec<u32> = aparent.into_iter().map(AtomicU32::into_inner).collect();
    lap("wpass");

    // Every set from here on (edges inside a cell, equal positions, clusters) lies inside
    // one cell, so each chunk finishes its clusters on its own.
    let locked_v: Option<Vec<bool>> = opts.locked_vertices.as_ref().map(|l| {
        (0..nv)
            .map(|r| l.get(vert(r) as usize).copied().unwrap_or(false))
            .collect()
    });
    let mut cid = vec![0u32; nv];
    let mut wbest = vec![(f32::INFINITY, u32::MAX); nv];
    let mut warea_v = vec![0f32; nv];
    let rends = || chunks.iter().map(|c| c[1]);
    let jobs: Vec<Chunk> = split_ends(&mut parent, rends())
        .into_iter()
        .zip(split_ends(&mut cid, rends()))
        .zip(split_ends(&mut wbest, rends()))
        .zip(split_ends(&mut warea_v, rends()))
        .zip(split_ends(&mut cell_st, chunks.iter().map(|c| c[3])))
        .zip(&chunks)
        .map(|(((((parent, cid), wbest), warea), cell_st), ch)| Chunk {
            r0: ch[0],
            c0: ch[2],
            parent,
            cid,
            wbest,
            warea,
            cell_st,
        })
        .collect();
    let lock_border = opts.lock_border;
    let outs: Vec<ChunkOut> = jobs
        .into_par_iter()
        .map(|j| {
            let (r0, c0, n) = (j.r0, j.c0, j.parent.len());
            let base = r0 as u32;
            // Wedge roots (each set's smallest rank), then clusters in local ids.
            for i in 0..n {
                j.parent[i] = j.parent[(j.parent[i] - base) as usize];
            }
            let sp = j.cid;
            for i in 0..n {
                sp[i] = j.parent[i] - base;
            }
            // Clusters: wedges of a cell that share a position (a seam) are one cluster;
            // separate surfaces passing through the same cell stay apart. Equal positions
            // share a Morton code, and runs of one code are short.
            let mut run: Vec<u32> = Vec::new();
            let mut i = 0;
            while i < n {
                let mut e = i + 1;
                while e < n && code(r0 + e) == code(r0 + i) {
                    e += 1;
                }
                if e - i > 1 {
                    let pos = |i: u32| spos[r0 + i as usize];
                    if e - i <= 8 {
                        for a in i as u32..e as u32 {
                            for b in a + 1..e as u32 {
                                if pos(a) == pos(b) {
                                    union(sp, a, b);
                                }
                            }
                        }
                    } else {
                        run.clear();
                        run.extend(i as u32..e as u32);
                        let key = |i: u32| pos(i).to_array().map(f32::to_bits);
                        run.sort_unstable_by_key(|&i| key(i));
                        for w in run.windows(2) {
                            if key(w[0]) == key(w[1]) {
                                union(sp, w[0], w[1]);
                            }
                        }
                    }
                }
                i = e;
            }
            flatten(sp);
            // Numbered in rank order, with their cells and member counts.
            let ncl = j.cell_st.len();
            let mut ccell: Vec<u32> = Vec::with_capacity(ncl + ncl / 2);
            let mut cnum: Vec<u32> = Vec::with_capacity(ncl + ncl / 2);
            let mut per_cell = vec![0u8; ncl];
            for i in 0..n {
                let root = sp[i] as usize;
                let c = if root == i {
                    let k = cellr[r0 + i] as usize - c0;
                    ccell.push(k as u32);
                    cnum.push(0);
                    per_cell[k] = per_cell[k].saturating_add(1);
                    ccell.len() as u32 - 1
                } else {
                    sp[root]
                };
                sp[i] = c;
                cnum[c as usize] += 1;
            }
            // A cluster alone in its cell takes the cell's sums. In a shared cell the largest
            // cluster takes the cell's sums less the others', so only the smaller ones sum
            // their members.
            let nc = ccell.len();
            let mut big = vec![u32::MAX; ncl];
            for c in 0..nc {
                let k = ccell[c] as usize;
                if per_cell[k] > 1 && (big[k] == u32::MAX || cnum[c] > cnum[big[k] as usize]) {
                    big[k] = c as u32;
                }
            }
            let mut cq: Vec<Q> = vec![[0.0; 10]; nc];
            let mut carea: Vec<f64> = vec![0.0; nc];
            let mut cmom: Vec<[f64; 3]> = vec![[0.0; 3]; nc];
            for i in 0..n {
                let k = cellr[r0 + i] as usize - c0;
                let c = sp[i];
                if per_cell[k] > 1 && big[k] != c {
                    let c = c as usize;
                    for (a, b) in cq[c].iter_mut().zip(vq(r0 + i)) {
                        *a += b as f64;
                    }
                    let a = vs[r0 + i].area;
                    carea[c] += a as f64;
                    let p = spos[r0 + i] * a;
                    for (m, x) in cmom[c].iter_mut().zip(p.to_array()) {
                        *m += x as f64;
                    }
                }
            }
            for c in 0..nc {
                let k = ccell[c] as usize;
                if per_cell[k] > 1 && big[k] != c as u32 {
                    let st = &mut j.cell_st[k];
                    for i in 0..10 {
                        st[i] -= cq[c][i];
                    }
                    st[10] -= carea[c];
                    for i in 0..3 {
                        st[14 + i] -= cmom[c][i];
                    }
                }
            }
            for c in 0..nc {
                let k = ccell[c] as usize;
                if per_cell[k] == 1 || big[k] == c as u32 {
                    let st = &j.cell_st[k];
                    cq[c] = st[..10].try_into().unwrap();
                    carea[c] = st[10].max(0.0);
                    cmom[c] = [st[14], st[15], st[16]];
                }
            }
            // The member of least error represents its cluster, near-ties going to the member
            // nearest the area-weighted centroid (in flat regions every member's error is zero).
            let cen: Vec<[f64; 3]> = (0..nc)
                .map(|c| {
                    let a = carea[c].max(1e-30);
                    cmom[c].map(|m| m / a)
                })
                .collect();
            let mut best = vec![(f64::INFINITY, 0u32); nc];
            let mut score = vec![f64::INFINITY; nc];
            let mut clocked = vec![false; nc];
            for i in 0..n {
                let (r, c) = (r0 + i, sp[i] as usize);
                clocked[c] |=
                    locked_v.as_ref().is_some_and(|l| l[r]) || (lock_border && vs[r].s1 != 0);
                let p = spos[r];
                let e = qeval_v(&cq[c], p);
                let d2 = (p.x as f64 - cen[c][0]).powi(2)
                    + (p.y as f64 - cen[c][1]).powi(2)
                    + (p.z as f64 - cen[c][2]).powi(2);
                let sc = e + CENTROID_PULL * carea[c] * d2;
                if sc < score[c] {
                    score[c] = sc;
                    best[c] = (e, r as u32);
                }
            }
            let mut worst1 = 0f64;
            for c in 0..nc {
                if carea[c] > 0.0 && cnum[c] > 1 {
                    worst1 = worst1.max(best[c].0 / carea[c]);
                }
            }
            // Each wedge's attributes come from its member nearest the cluster's position.
            for i in 0..n {
                let r = r0 + i;
                let root = (j.parent[i] - base) as usize;
                let d = (spos[r] - spos[best[sp[i] as usize].1 as usize]).length_squared();
                if d < j.wbest[root].0 {
                    j.wbest[root] = (d, r as u32);
                }
                j.warea[root] += vs[r].area / 3.0;
            }
            // Wedges that meet at a position with the same UVs and colors differ in their
            // normals alone (a hard edge, or unwelded faces): the texture runs on across them,
            // so they take their attributes from one member, the nearest to the cluster's
            // position among them all. Left apart, the wedge without a member at that position
            // would shift the texture by the distance to its own nearest. Normals stay each
            // wedge's own (its mean, set at output).
            // (A class's root is its smallest wedge root, kept in `class` only for wedges
            // that joined one; most chunks of a welded mesh never allocate it.)
            {
                let mut class: Vec<u32> = Vec::new();
                fn find(c: &mut [u32], mut x: u32) -> u32 {
                    while c[x as usize] != x {
                        c[x as usize] = c[c[x as usize] as usize];
                        x = c[x as usize];
                    }
                    x
                }
                let same = |a: usize, b: usize| {
                    mesh.uvs.iter().all(|u| u[a] == u[b])
                        && mesh.colors.as_ref().is_none_or(|c| c[a] == c[b])
                };
                for i in 1..n {
                    if spos[r0 + i - 1] != spos[r0 + i] {
                        continue;
                    }
                    let mut k = i;
                    while k > 0 && spos[r0 + k - 1] == spos[r0 + i] {
                        k -= 1;
                        if j.parent[i] == j.parent[k] {
                            break;
                        }
                        if same(vert(r0 + i) as usize, vert(r0 + k) as usize) {
                            if class.is_empty() {
                                class = (0..n as u32).collect();
                            }
                            let (a, b) = (
                                find(&mut class, j.parent[i] - base),
                                find(&mut class, j.parent[k] - base),
                            );
                            class[a.max(b) as usize] = a.min(b);
                            break;
                        }
                    }
                }
                if !class.is_empty() {
                    // The nearest member of each class, gathered at its root, then handed
                    // back to every wedge of the class.
                    for i in 0..n {
                        if class[i] != i as u32 {
                            let c = find(&mut class, i as u32) as usize;
                            if j.wbest[i] < j.wbest[c] {
                                j.wbest[c] = j.wbest[i];
                            }
                        }
                    }
                    for i in 0..n {
                        if class[i] != i as u32 {
                            j.wbest[i] = j.wbest[class[i] as usize];
                        }
                    }
                }
            }
            ChunkOut {
                cq,
                best,
                clocked,
                worst1,
            }
        })
        .collect();
    drop(cellr);
    drop(cell_st);
    drop(locked_v);
    // Global cluster ids: chunk offsets, in rank order.
    let mut off = Vec::with_capacity(outs.len());
    let mut nc = 0usize;
    for o in &outs {
        off.push(nc as u32);
        nc += o.cq.len();
    }
    split_ends(&mut cid, rends())
        .into_par_iter()
        .zip(off)
        .for_each(|(s, o)| s.iter_mut().for_each(|c| *c += o));
    let mut cq: Vec<Q> = Vec::with_capacity(nc);
    let mut cpos: Vec<Vec3> = Vec::with_capacity(nc);
    let mut rep: Vec<u32> = Vec::with_capacity(nc);
    let mut clocked: Vec<bool> = Vec::with_capacity(nc);
    let mut worst1 = 0f64;
    for o in outs {
        cq.extend(o.cq);
        cpos.extend(o.best.iter().map(|b| spos[b.1 as usize]));
        rep.extend(o.best.iter().map(|b| vert(b.1 as usize)));
        clocked.extend(o.clocked);
        worst1 = worst1.max(o.worst1);
    }
    lap("clusters");

    // Stage-two triangles over wedges; same-orientation duplicates go, opposite ones are
    // kept as double-sided twins.
    let nw_est = kept.len() * 3 / 2 + 16;
    let mut wid = vec![u32::MAX; nv];
    let mut wsrc: Vec<u32> = Vec::with_capacity(nw_est);
    let mut wc: Vec<u32> = Vec::with_capacity(nw_est);
    let mut warea: Vec<f32> = Vec::with_capacity(nw_est);
    let mut tw: Vec<[u32; 3]> = Vec::with_capacity(kept.len());
    let mut tc: Vec<[u32; 3]> = Vec::with_capacity(kept.len());
    let mut tmat: Vec<u32> = Vec::with_capacity(kept.len());
    let mut twin: Vec<bool> = Vec::with_capacity(kept.len());
    let mut seen = KeyMap::new(kept.len());
    for &(tri, t) in &kept {
        let c = tri.map(|r| cid[r as usize]);
        let mut sorted = c;
        sorted.sort_unstable();
        let key = ((sorted[0] as u128) << 64) | ((sorted[1] as u128) << 32) | sorted[2] as u128;
        // Rotation-invariant orientation: the cyclic order with the smallest id first.
        let r = (0..3).min_by_key(|&i| c[i]).unwrap();
        let fwd = c[(r + 1) % 3] == sorted[1];
        let (slot, present) = seen.get_or_insert(key, tw.len() as u32 * 2 + fwd as u32);
        let is_twin = if present {
            if (slot & 1 == 1) == fwd {
                continue;
            }
            true
        } else {
            false
        };
        let w = tri.map(|r| {
            let root = parent[r as usize] as usize;
            if wid[root] == u32::MAX {
                wid[root] = wsrc.len() as u32;
                wsrc.push(vert(wbest[root].1 as usize));
                wc.push(cid[r as usize]);
                warea.push(warea_v[root]);
            }
            wid[root]
        });
        tw.push(w);
        tc.push(c);
        tmat.push(mesh.material_ids.get(t as usize).copied().unwrap_or(0));
        twin.push(is_twin);
    }
    drop(kept);
    lap("tris");

    // Attributes per wedge, pre-scaled as in the fast reducer.
    let nwt = (opts.normal_weight / 5e-5).max(0.0).sqrt() * NORMAL_SCALE.sqrt();
    let uwt = (opts.uv_weight / 0.5).max(0.0).sqrt() * UV_SCALE.sqrt();
    let cwt = (opts.color_weight / 0.5).max(0.0).sqrt() * COLOR_SCALE.sqrt();
    let nd = if mesh.normals.is_some() && nwt > 0.0 {
        3
    } else {
        0
    };
    let mut dim = nd;
    if uwt > 0.0 {
        dim += 2 * mesh.uvs.len();
    }
    if mesh.colors.is_some() && cwt > 0.0 {
        dim += 4;
    }
    let mut attr = vec![0f32; wsrc.len() * dim];
    if dim > 0 {
        attr.par_chunks_mut(dim)
            .zip(&wsrc)
            .with_min_len(1024)
            .for_each(|(a, &v)| {
                let v = v as usize;
                let mut k = 0;
                let mut put = |x: &[f32]| {
                    a[k..k + x.len()].copy_from_slice(x);
                    k += x.len();
                };
                if let (Some(ns), true) = (&mesh.normals, nd > 0) {
                    put(&(ns[v] * nwt).to_array());
                }
                if uwt > 0.0 {
                    for set in &mesh.uvs {
                        put(&(set[v] * uwt).to_array());
                    }
                }
                if let (Some(cs), true) = (&mesh.colors, cwt > 0.0) {
                    put(&(cs[v] * cwt).to_array());
                }
            });
    }

    let alive_count = tw.len();
    let mut st = Tail {
        cpos,
        cq,
        tw,
        tc,
        alive: vec![true; alive_count],
        twin,
        warea,
        attr,
        dim,
        avg_area,
        locked: clocked,
        side: vec![[u32::MAX; 2]; nc],
        sdeg: vec![0; nc],
        soff: vec![0; nc],
        slen: vec![0; nc],
        scap: vec![0; nc],
        pool: Vec::new(),
    };
    st.classify_edges();
    lap("attrs");
    let max_err2 = opts.max_error.map(|e| (e * inv) * (e * inv));
    let (collapses, worst2, wremap, cremap) = st.run(target, max_err2, opts.min_face_dot);
    let moved = st.relax(&cremap, opts.min_face_dot);
    lap("tail");

    // Output: one vertex per wedge still used, at its cluster's representative position.
    let nw = wsrc.len();
    let mut wout = vec![u32::MAX; nw];
    let mut srcs: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();
    let mut cluster_out = vec![u32::MAX; nc];
    for t in 0..st.tw.len() {
        if !st.alive[t] {
            continue;
        }
        for (k, &w) in st.tw[t].iter().enumerate() {
            if wout[w as usize] == u32::MAX {
                wout[w as usize] = srcs.len() as u32;
                srcs.push(w);
            }
            indices.push(wout[w as usize]);
            let c = st.tc[t][k] as usize;
            if cluster_out[c] == u32::MAX {
                cluster_out[c] = wout[w as usize];
            }
        }
        material_ids.push(tmat[t]);
    }
    let source_vertex: Vec<u32> = srcs.iter().map(|&w| wsrc[w as usize]).collect();
    let mut out = mesh.select_vertices(&source_vertex);
    for (o, &w) in srcs.iter().enumerate() {
        let c = wc[w as usize] as usize;
        out.positions[o] = if moved[c] {
            st.cpos[c] * diag + center
        } else {
            mesh.positions[rep[c] as usize]
        };
    }
    out.indices = indices;
    out.material_ids = material_ids;

    // Remap chains resolved to their ends once, then every input vertex in parallel.
    let ends = |chain: &mut [u32]| {
        for x in 0..chain.len() {
            let mut y = chain[x];
            while chain[y as usize] != y {
                y = chain[y as usize];
            }
            chain[x] = y;
        }
    };
    let (mut wremap, mut cremap) = (wremap, cremap);
    ends(&mut wremap);
    ends(&mut cremap);
    let vertex_remap = (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|v| {
            let r = rank[v] as usize;
            let root = parent[r] as usize;
            if wid[root] != u32::MAX {
                let o = wout[wremap[wid[root] as usize] as usize];
                if o != u32::MAX {
                    return o;
                }
            }
            let o = cluster_out[cremap[cid[r] as usize] as usize];
            if o != u32::MAX { o } else { 0 }
        })
        .collect();
    // Each output vertex stands for every input vertex merged into its wedge: its normal is
    // their area-weighted mean, not the one member's the other attributes are copied from.
    if let (Some(src), Some(dst)) = (&mesh.normals, &mut out.normals) {
        // Summed over fixed spans of the Morton order and then span by span, so the result
        // doesn't depend on the thread count.
        let span = nv.div_ceil(8).max(PAR_MIN);
        let n_out = dst.len();
        let parts: Vec<Vec<Vec3>> = (0..nv.div_ceil(span))
            .into_par_iter()
            .map(|k| {
                let mut sum = vec![Vec3::ZERO; n_out];
                for r in k * span..((k + 1) * span).min(nv) {
                    let root = parent[r] as usize;
                    if wid[root] != u32::MAX {
                        let o = wout[wremap[wid[root] as usize] as usize];
                        if o != u32::MAX {
                            sum[o as usize] += src[vert(r) as usize] * vs[r].area;
                        }
                    }
                }
                sum
            })
            .collect();
        let mut parts = parts.into_iter();
        let mut sum = parts.next().unwrap_or_default();
        for part in parts {
            for (a, b) in sum.iter_mut().zip(part) {
                *a += b;
            }
        }
        for (n, s) in dst.iter_mut().zip(sum) {
            let l = s.length();
            if l > 1e-20 {
                *n = s / l;
            }
        }
    }
    let rel = worst1.max(worst2 as f64).sqrt() as f32;
    lap("output");
    Ok(SimplifyResult {
        mesh: out,
        error: rel * diag,
        relative_error: rel,
        vertex_remap,
        collapses: collapses + (nv - nc),
        symmetry_plane: None,
        source_vertex,
    })
}

fn unchanged(mesh: &Mesh) -> SimplifyResult {
    let n = mesh.vertex_count() as u32;
    SimplifyResult {
        mesh: mesh.clone(),
        error: 0.0,
        relative_error: 0.0,
        vertex_remap: (0..n).collect(),
        collapses: 0,
        symmetry_plane: None,
        source_vertex: (0..n).collect(),
    }
}

/// The collapse tail over clusters (positions) and wedges, in exact greedy order: a heap of
/// each cluster's cheapest collapse, kept current as neighborhoods change.
struct Tail {
    cpos: Vec<Vec3>,
    cq: Vec<Q>,
    /// Triangles by wedge and by cluster.
    tw: Vec<[u32; 3]>,
    tc: Vec<[u32; 3]>,
    alive: Vec<bool>,
    /// Opposite-facing copy of another triangle (a double-sided sheet): left out of edge counts.
    twin: Vec<bool>,
    warea: Vec<f32>,
    attr: Vec<f32>,
    dim: usize,
    avg_area: f64,
    locked: Vec<bool>,
    /// The two neighbors along open-border or non-manifold edges a constrained cluster may
    /// slide to; `sdeg` counts such edges (0: unconstrained).
    side: Vec<[u32; 2]>,
    sdeg: Vec<u8>,
    /// Cluster → triangles, as growable runs in one pool.
    soff: Vec<u32>,
    slen: Vec<u32>,
    scap: Vec<u32>,
    pool: Vec<u32>,
}

impl Tail {
    fn classify_edges(&mut self) {
        let nc = self.cpos.len();
        // Triangle lists.
        let mut count = vec![0u32; nc];
        for c in &self.tc {
            for &x in c {
                count[x as usize] += 1;
            }
        }
        let mut off = 0u32;
        for c in 0..nc {
            self.soff[c] = off;
            self.scap[c] = count[c] + 2;
            off += count[c] + 2;
        }
        self.pool = vec![0; off as usize];
        for (t, c) in self.tc.iter().enumerate() {
            for &x in c {
                let x = x as usize;
                self.pool[(self.soff[x] + self.slen[x]) as usize] = t as u32;
                self.slen[x] += 1;
            }
        }
        // Open-border edges (one face) and junctions (more than two), as in the fast reducer:
        // border clusters slide along their border, junction clusters along their junction,
        // and clusters on both, or on more than two such edges, stay put. Each cluster
        // counts its own edges from its triangles (twins left out), in parallel.
        let this = &*self;
        let class: Vec<([u32; 2], u8, bool)> = (0..nc as u32)
            .into_par_iter()
            .with_min_len(256)
            .map_init(Vec::new, |nb: &mut Vec<u32>, a| {
                nb.clear();
                for &t in this.star(a) {
                    if !this.twin[t as usize] {
                        nb.extend(this.tc[t as usize].iter().filter(|&&x| x != a));
                    }
                }
                nb.sort_unstable();
                let (mut side, mut bdeg) = ([u32::MAX; 2], 0u8);
                let (mut jside, mut jdeg) = ([u32::MAX; 2], 0u8);
                for run in nb.chunk_by(|x, y| x == y) {
                    let (s, d) = match run.len() {
                        1 => (&mut side, &mut bdeg),
                        2 => continue,
                        _ => (&mut jside, &mut jdeg),
                    };
                    if *d < 2 {
                        s[*d as usize] = run[0];
                    }
                    *d = d.saturating_add(1);
                }
                let mut lock = bdeg > 0 && jdeg > 0;
                if !lock && jdeg > 0 {
                    bdeg = jdeg;
                    side = jside;
                }
                lock |= bdeg > 2;
                (side, bdeg, lock)
            })
            .collect();
        for (a, (side, sdeg, lock)) in class.into_iter().enumerate() {
            self.side[a] = side;
            self.sdeg[a] = sdeg;
            self.locked[a] |= lock;
        }
    }

    #[inline]
    fn star(&self, p: u32) -> &[u32] {
        let o = self.soff[p as usize] as usize;
        &self.pool[o..o + self.slen[p as usize] as usize]
    }

    /// Moves each remaining free cluster to the minimum of its summed quadric (it has stood
    /// at one of its member vertices so far), where that stays inside its neighborhood and
    /// turns no triangle. Returns which clusters moved.
    fn relax(&mut self, cremap: &[u32], min_dot: f32) -> Vec<bool> {
        let nc = self.cpos.len();
        let mut moved = vec![false; nc];
        let mut star: Vec<([Vec3; 3], Vec3, usize)> = Vec::new();
        for c in 0..nc {
            if cremap[c] != c as u32 || self.locked[c] || self.sdeg[c] != 0 {
                continue;
            }
            let p = self.cpos[c];
            let q = &self.cq[c];
            let l = RELAX_RIDGE * (q[0] + q[4] + q[7]) / 3.0;
            if l <= 0.0 {
                continue;
            }
            let a = [
                [q[0] + l, q[1], q[2]],
                [q[1], q[4] + l, q[5]],
                [q[2], q[5], q[7] + l],
            ];
            let b = [
                l * p.x as f64 - q[3],
                l * p.y as f64 - q[6],
                l * p.z as f64 - q[8],
            ];
            let det3 = |m: &[[f64; 3]; 3]| {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            };
            let d = det3(&a);
            if d.abs() < 1e-300 {
                continue;
            }
            let mut x = [0f64; 3];
            for (k, xk) in x.iter_mut().enumerate() {
                let mut m = a;
                for (row, bi) in m.iter_mut().zip(b) {
                    row[k] = bi;
                }
                *xk = det3(&m) / d;
            }
            let np = Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32);
            if !np.is_finite() {
                continue;
            }
            // Along the surface normal only: a sideways move would slide the texture, since
            // the attributes stay those of the member vertex. The star is walked once: each
            // triangle's corners, its normal, and which corner is this cluster's.
            star.clear();
            let mut n = Vec3::ZERO;
            for t in self.live(c as u32) {
                let v = self.tc[t].map(|x| self.cpos[x as usize]);
                let n0 = (v[1] - v[0]).cross(v[2] - v[0]);
                n += n0;
                let i = self.tc[t].iter().position(|&x| x == c as u32).unwrap();
                star.push((v, n0, i));
            }
            let n = n.normalize_or_zero();
            let np = p + n * (np - p).dot(n);
            // Within a share of the shortest edge to a neighbor, and no triangle turned.
            let mut short2 = f32::INFINITY;
            let mut ok = true;
            for &(v, n0, i) in &star {
                for k in 1..3 {
                    short2 = short2.min((v[(i + k) % 3] - p).length_squared());
                }
                let mut w = v;
                w[i] = np;
                let n1 = (w[1] - w[0]).cross(w[2] - w[0]);
                let l1 = n1.length();
                if !(l1 > 1e-12 && n0.dot(n1) >= min_dot.max(0.5) * n0.length() * l1) {
                    ok = false;
                    break;
                }
            }
            if !ok || (np - p).length_squared() > RELAX_REACH * RELAX_REACH * short2 {
                continue;
            }
            self.cpos[c] = np;
            moved[c] = true;
        }
        moved
    }

    fn live(&self, p: u32) -> impl Iterator<Item = usize> + '_ {
        self.star(p)
            .iter()
            .map(|&t| t as usize)
            .filter(|&t| self.alive[t])
    }

    fn adist(&self, a: u32, b: u32) -> f32 {
        if self.dim == 0 || a == b {
            return 0.0;
        }
        let d = self.dim;
        let (x, y) = (
            &self.attr[a as usize * d..(a as usize + 1) * d],
            &self.attr[b as usize * d..(b as usize + 1) * d],
        );
        x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum()
    }

    /// p's live triangles with p's corner in each, gathered once per turn.
    fn gather(&self, p: u32, loc: &mut Vec<Corner>) {
        loc.clear();
        for t in self.live(p) {
            let c = self.tc[t];
            let i = if c[0] == p {
                0
            } else if c[1] == p {
                1
            } else {
                2
            };
            loc.push(Corner {
                c,
                w: self.tw[t],
                i,
            });
        }
    }

    /// Wedge map p→q (wedges of p in triangles shared with q take q's wedge there, the rest
    /// the nearest of those) and its attribute cost, from p's gathered triangles; None when p
    /// and q share no triangle.
    fn wedge_map(&self, loc: &[Corner], q: u32, out: &mut Vec<(u32, u32)>) -> Option<f32> {
        out.clear();
        let mut pws: [u32; 8] = [u32::MAX; 8];
        let mut np = 0usize;
        for k in loc {
            let pw = k.w[k.i as usize];
            if !pws[..np].contains(&pw) {
                if np == 8 {
                    return None;
                }
                pws[np] = pw;
                np += 1;
            }
            if let Some(j) = (0..3).find(|&j| k.c[j] == q)
                && !out.iter().any(|&(a, _)| a == pw)
            {
                out.push((pw, k.w[j]));
            }
        }
        if out.is_empty() {
            return None;
        }
        let shared = out.len();
        for &pw in &pws[..np] {
            if out[..shared].iter().any(|&(a, _)| a == pw) {
                continue;
            }
            let best = out[..shared]
                .iter()
                .map(|&(_, qw)| (self.adist(pw, qw), qw))
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                .unwrap();
            out.push((pw, best.1));
        }
        Some(
            out.iter()
                .map(|&(pw, qw)| self.warea[pw as usize] * self.adist(pw, qw))
                .sum::<f32>()
                / self.avg_area as f32,
        )
    }

    /// Whether p has anywhere to go at all: not locked, and if constrained, a side to slide to.
    #[inline]
    fn movable(&self, p: u32) -> bool {
        !self.locked[p as usize]
            && (self.sdeg[p as usize] == 0 || self.side[p as usize] != [u32::MAX; 2])
    }

    #[inline]
    fn can(&self, p: u32, q: u32) -> bool {
        !self.locked[p as usize]
            && (self.sdeg[p as usize] == 0 || self.side[p as usize].contains(&q))
    }

    fn link_ok(&self, loc: &[Corner], p: u32, q: u32, s: &mut Scratch) -> bool {
        s.epoch += 1;
        let e1 = s.epoch;
        let mut shared = 0;
        for k in loc {
            shared += k.c.contains(&q) as usize;
            for &x in &k.c {
                s.mark[x as usize] = e1;
            }
        }
        s.epoch += 1;
        let e2 = s.epoch;
        let mut common = 0;
        for t in self.live(q) {
            for &x in &self.tc[t] {
                if x != p && x != q && s.mark[x as usize] == e1 {
                    s.mark[x as usize] = e2;
                    common += 1;
                }
            }
        }
        common <= shared
    }

    fn no_flip(&self, loc: &[Corner], q: u32, min_dot: f32) -> bool {
        let qp = self.cpos[q as usize];
        loc.iter().all(|k| {
            if k.c.contains(&q) {
                return true;
            }
            let v = k.c.map(|x| self.cpos[x as usize]);
            let n0 = (v[1] - v[0]).cross(v[2] - v[0]);
            let mut w = v;
            w[k.i as usize] = qp;
            let n1 = (w[1] - w[0]).cross(w[2] - w[0]);
            let l1 = n1.length();
            l1 > 1e-12 && n0.dot(n1) >= min_dot * n0.length() * l1
        })
    }

    /// p's cheapest allowed collapse: (cost, target, mean squared distance), from p's
    /// triangles gathered in `s.loc`. With `exact` false, a lower bound on the cost from
    /// geometry alone (the heap's first key).
    fn best(
        &self,
        p: u32,
        max_err2: Option<f32>,
        exact: bool,
        s: &mut Scratch,
    ) -> Option<(f32, u32, f32)> {
        if self.locked[p as usize] {
            return None;
        }
        s.epoch += 1;
        s.nbr.clear();
        for k in &s.loc {
            for &x in &k.c {
                if x != p && s.mark[x as usize] != s.epoch {
                    s.mark[x as usize] = s.epoch;
                    s.nbr.push(x);
                }
            }
        }
        let qp = &self.cq[p as usize];
        s.cand.clear();
        for &q in s.nbr.iter() {
            if !self.can(p, q) {
                continue;
            }
            let mut qq = self.cq[q as usize];
            for (a, b) in qq.iter_mut().zip(qp) {
                *a += b;
            }
            let raw = qeval_v(&qq, self.cpos[q as usize]);
            let g2 = (raw / (qq[0] + qq[4] + qq[7]).max(1e-30)) as f32;
            if max_err2.is_some_and(|m| g2 > m) {
                continue;
            }
            s.cand.push(((raw / self.avg_area) as f32, q, g2));
        }
        if !exact {
            return s.cand.iter().copied().min_by(|a, b| a.0.total_cmp(&b.0));
        }
        // Cheapest geometry first; the attribute cost only adds, so stop once geometry alone
        // costs more than the best found.
        s.cand
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        // A cluster with one wedge (most are) maps it onto q's wedge in their first shared
        // triangle: the same cost as `wedge_map`'s, without scanning the star per candidate.
        let single = s
            .loc
            .first()
            .map(|k| k.w[k.i as usize])
            .filter(|&w| s.loc.iter().all(|k| k.w[k.i as usize] == w));
        let mut best: Option<(f32, u32, f32)> = None;
        for &(g, q, g2) in s.cand.iter() {
            if best.is_some_and(|b| g >= b.0) {
                break;
            }
            let ac = match single {
                Some(pw) => s
                    .loc
                    .iter()
                    .find_map(|k| (0..3).find(|&j| k.c[j] == q).map(|j| k.w[j]))
                    .map(|qw| self.warea[pw as usize] * self.adist(pw, qw) / self.avg_area as f32),
                None => self.wedge_map(&s.loc, q, &mut s.map),
            };
            let Some(ac) = ac else {
                continue;
            };
            let c = g + ac;
            if best.is_none_or(|b| c < b.0) {
                best = Some((c, q, g2));
            }
        }
        best
    }

    /// Moves p onto q; returns the triangles removed.
    fn collapse(&mut self, p: u32, q: u32, map: &[(u32, u32)]) -> usize {
        for &(pw, qw) in map {
            let a = self.warea[pw as usize];
            self.warea[qw as usize] += a;
        }
        let mut removed = 0;
        let (po, pl) = (
            self.soff[p as usize] as usize,
            self.slen[p as usize] as usize,
        );
        for k in po..po + pl {
            let t = self.pool[k] as usize;
            if !self.alive[t] {
                continue;
            }
            if self.tc[t].contains(&q) {
                self.alive[t] = false;
                removed += 1;
            } else {
                for i in 0..3 {
                    if self.tc[t][i] == p {
                        self.tc[t][i] = q;
                        let w = self.tw[t][i];
                        self.tw[t][i] = map.iter().find(|m| m.0 == w).map_or(w, |m| m.1);
                    }
                }
            }
        }
        // q's list: its own live triangles, then p's.
        let (qo, ql) = (
            self.soff[q as usize] as usize,
            self.slen[q as usize] as usize,
        );
        let mut n = 0;
        for k in qo..qo + ql {
            let t = self.pool[k];
            if self.alive[t as usize] {
                self.pool[qo + n] = t;
                n += 1;
            }
        }
        let extra = (po..po + pl)
            .filter(|&k| self.alive[self.pool[k] as usize])
            .count();
        let (mut qo, cap) = (qo, self.scap[q as usize] as usize);
        if n + extra > cap {
            let new = self.pool.len();
            let ncap = (n + extra) * 2;
            self.pool.resize(new + ncap, 0);
            self.pool.copy_within(qo..qo + n, new);
            qo = new;
            self.soff[q as usize] = new as u32;
            self.scap[q as usize] = ncap as u32;
        }
        for k in po..po + pl {
            let t = self.pool[k];
            if self.alive[t as usize] {
                self.pool[qo + n] = t;
                n += 1;
            }
        }
        self.slen[q as usize] = n as u32;
        self.slen[p as usize] = 0;
        let qq = self.cq[p as usize];
        for (a, b) in self.cq[q as usize].iter_mut().zip(qq) {
            *a += b;
        }
        // Sliding along a constraint: p's other constraint neighbor now meets q.
        if self.sdeg[p as usize] == 2 {
            let [a, b] = self.side[p as usize];
            let r = if a == q { b } else { a };
            if r != q {
                for (x, from, to) in [(q, p, r), (r, p, q)] {
                    for s in &mut self.side[x as usize] {
                        if *s == from {
                            *s = to;
                        }
                    }
                }
            }
        }
        removed
    }

    /// Collapses in greedy order down to `target`; returns (collapses, worst mean squared
    /// distance, wedge remap chain, cluster remap chain).
    fn run(
        &mut self,
        target: Option<usize>,
        max_err2: Option<f32>,
        min_dot: f32,
    ) -> (usize, f32, Vec<u32>, Vec<u32>) {
        use std::cmp::Reverse;
        let nc = self.cpos.len();
        let nw = self.warea.len();
        let mut wremap: Vec<u32> = (0..nw as u32).collect();
        let mut cremap: Vec<u32> = (0..nc as u32).collect();
        let mut alive_tris = self.alive.len();
        let mut s = Scratch::new(nc);
        // Lazy greedy: each live cluster has at most one heap entry, whose cost is checked
        // again when it surfaces (collapses mostly raise neighbors' costs). Clusters with no
        // allowed collapse wait out of the heap until a collapse changes their neighborhood.
        let mut queued = vec![false; nc];
        let mut gone = vec![false; nc];
        let mut map = Vec::new();
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(nc);
        // First costs exact, in parallel (they only read the state), so a cluster's first
        // turn at the top mostly holds. (Lower bounds here and exact costs on demand were
        // measured slower: that moves parallel work into the serial loop.)
        //
        // A cluster's exact best is kept with the collapse count it was found at, and stands
        // while no collapse since has changed what it reads (`dirty`): its triangles, its
        // neighbors' quadrics, its wedges' areas and its constraint sides.
        let mut cache: Vec<Option<(f32, u32, f32)>> = (0..nc as u32)
            .into_par_iter()
            .with_min_len(64)
            .map_init(
                || Scratch::new(nc),
                |s, p| {
                    if !self.movable(p) {
                        return None;
                    }
                    self.gather(p, &mut s.loc);
                    self.best(p, max_err2, true, s)
                },
            )
            .collect();
        let mut cache_at = vec![0u32; nc];
        let mut dirty = vec![0u32; nc];
        let mut done = 0u32;
        for (p, c) in cache.iter().enumerate() {
            if let Some(c) = c {
                queued[p] = true;
                heap.push(Reverse((c.0.to_bits(), p as u32)));
            }
        }
        let mut worst2 = 0f32;
        let mut collapses = 0;
        let mut touched: Vec<u32> = Vec::new();
        while let Some(Reverse((c0, p))) = heap.pop() {
            if target.is_some_and(|t| alive_tris <= t) {
                break;
            }
            let pu = p as usize;
            queued[pu] = false;
            if gone[pu] {
                continue;
            }
            let fresh = cache_at[pu] < dirty[pu];
            if fresh {
                self.gather(p, &mut s.loc);
                cache[pu] = self.best(p, max_err2, true, &mut s);
                cache_at[pu] = done;
            }
            let Some((c, q, g2)) = cache[pu] else {
                continue;
            };
            if c.to_bits() > c0 && heap.peek().is_some_and(|Reverse((n, _))| c.to_bits() > *n) {
                queued[pu] = true;
                heap.push(Reverse((c.to_bits(), p)));
                continue;
            }
            if !fresh {
                self.gather(p, &mut s.loc);
            }
            let loc = std::mem::take(&mut s.loc);
            let ok = self.link_ok(&loc, p, q, &mut s)
                && self.no_flip(&loc, q, min_dot)
                && self.wedge_map(&loc, q, &mut map).is_some();
            s.loc = loc;
            if !ok {
                continue;
            }
            for &(pw, qw) in &map {
                wremap[pw as usize] = qw;
            }
            // What changes: every cluster on p's triangles (their triangles do), every one on
            // q's after (q's quadric does), and p's other constraint neighbor (its side does).
            done += 1;
            for k in &s.loc {
                for &x in &k.c {
                    dirty[x as usize] = done;
                }
            }
            for &x in &self.side[pu] {
                if x != u32::MAX {
                    dirty[x as usize] = done;
                }
            }
            alive_tris -= self.collapse(p, q, &map);
            gone[p as usize] = true;
            cremap[p as usize] = q;
            worst2 = worst2.max(g2);
            collapses += 1;
            // Waiting clusters around q may have an allowed collapse now.
            s.epoch += 1;
            touched.clear();
            s.mark[q as usize] = s.epoch;
            touched.push(q);
            for t in self.live(q) {
                for &x in &self.tc[t] {
                    if s.mark[x as usize] != s.epoch {
                        s.mark[x as usize] = s.epoch;
                        touched.push(x);
                    }
                }
            }
            for &r in &touched {
                dirty[r as usize] = done;
                if queued[r as usize] || !self.movable(r) {
                    continue;
                }
                self.gather(r, &mut s.loc);
                if let Some((c, _, _)) = self.best(r, max_err2, false, &mut s) {
                    queued[r as usize] = true;
                    heap.push(Reverse((c.to_bits(), r)));
                }
            }
        }
        (collapses, worst2, wremap, cremap)
    }
}

/// One of a cluster's live triangles: its clusters and wedges, and which corner is the
/// cluster's.
#[derive(Clone, Copy)]
struct Corner {
    c: [u32; 3],
    w: [u32; 3],
    i: u8,
}

/// Per-thread working space for the tail's neighborhood queries.
struct Scratch {
    mark: Vec<u32>,
    epoch: u32,
    nbr: Vec<u32>,
    cand: Vec<(f32, u32, f32)>,
    map: Vec<(u32, u32)>,
    loc: Vec<Corner>,
}

impl Scratch {
    fn new(n: usize) -> Scratch {
        Scratch {
            mark: vec![0; n],
            epoch: 0,
            nbr: Vec::new(),
            cand: Vec::new(),
            map: Vec::new(),
            loc: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Mesh;
    use crate::reduce::simplify::{Quality, SimplifyOptions, simplify, tests::uv_sphere};

    fn opts(ratio: f32) -> SimplifyOptions {
        SimplifyOptions {
            target_ratio: Some(ratio),
            quality: Quality::Realtime,
            ..Default::default()
        }
    }

    fn sphere() -> Mesh {
        let mut m = uv_sphere(128, 64);
        m.compute_normals(60.0);
        m
    }

    #[test]
    fn reaches_targets_with_valid_output() {
        let m = sphere();
        for (ratio, mean) in [(0.25, 0.004), (0.05, 0.01), (0.01, 0.04)] {
            let r = simplify(&m, &opts(ratio)).unwrap();
            r.mesh.validate().unwrap();
            let target = (m.triangle_count() as f32 * ratio) as usize;
            let n = r.mesh.triangle_count();
            assert!(n <= target && n > target / 2, "{ratio}: {n} vs {target}");
            assert_eq!(r.vertex_remap.len(), m.vertex_count());
            assert!(
                r.vertex_remap
                    .iter()
                    .all(|&v| (v as usize) < r.mesh.vertex_count())
            );
            let d = crate::metrics::surface_distance(&m, &r.mesh, 4000);
            assert!(d.relative_mean < mean, "{ratio}: {d:?}");
            assert!(
                d.relative_hausdorff <= r.relative_error * 4.0 + 0.01,
                "{ratio}: {d:?}"
            );
        }
    }

    #[test]
    fn same_result_for_any_thread_count() {
        let m = sphere();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| simplify(&m, &opts(0.05)).unwrap())
        };
        let a = run(1);
        for t in [2, 3, 8] {
            let b = run(t);
            assert_eq!(a.mesh, b.mesh, "{t} threads");
            assert_eq!(a.vertex_remap, b.vertex_remap, "{t} threads");
        }
    }

    #[test]
    fn uv_seam_survives() {
        // Clusters across the seam keep a wedge per side, so no triangle's UVs wrap around
        // (a wrapped triangle would show the whole texture squeezed into it).
        let m = sphere();
        let r = simplify(&m, &opts(0.05)).unwrap();
        let (o, uv) = (&r.mesh, &r.mesh.uvs[0]);
        for t in o.indices.chunks(3) {
            let u = [
                uv[t[0] as usize].x,
                uv[t[1] as usize].x,
                uv[t[2] as usize].x,
            ];
            let span =
                u.iter().fold(0f32, |a, &x| a.max(x)) - u.iter().fold(1f32, |a, &x| a.min(x));
            assert!(span < 0.5, "triangle {t:?} wraps the seam: u = {u:?}");
        }
        let sides = (0..o.vertex_count())
            .filter(|&v| (0..v).any(|w| o.positions[w] == o.positions[v] && uv[w] != uv[v]))
            .count();
        assert!(sides >= 4, "seam gone: {sides} split vertices left");
    }

    #[test]
    fn max_error_bounds_the_result() {
        let m = sphere();
        let r = simplify(
            &m,
            &SimplifyOptions {
                max_error: Some(0.01),
                quality: Quality::Realtime,
                ..Default::default()
            },
        )
        .unwrap();
        r.mesh.validate().unwrap();
        assert!(r.error <= 0.01, "{}", r.error);
        assert!(r.mesh.triangle_count() < m.triangle_count() / 4);
        let d = crate::metrics::surface_distance(&m, &r.mesh, 4000);
        assert!(d.hausdorff < 0.06, "{d:?}");
    }

    #[test]
    fn flat_grid_keeps_its_outline() {
        let n = 40u32;
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
        let r = simplify(&m, &opts(0.02)).unwrap();
        r.mesh.validate().unwrap();
        let area = r.mesh.surface_area();
        assert!(
            (area - (n * n) as f32).abs() < 0.1,
            "outline moved: area {area}"
        );
        assert!(r.mesh.triangle_count() <= 64, "{}", r.mesh.triangle_count());
    }
}
