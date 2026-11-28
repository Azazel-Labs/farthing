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
//!    a vertex's next and previous corners, counted by position, and of their squares, cancel
//!    around a closed fan and give the two border neighbors of an open one). Cells are split top-down, worst first,
//!    by the quadric error at their area-weighted centroid plus a normal-cone term that splits
//!    cells holding opposite sheets. Flat regions stay coarse and curved ones refine, like
//!    greedy collapse and unlike a uniform grid. The split stops at about one cluster per
//!    target triangle (about twice the target in triangles). A leaf cell holding a tube or
//!    strut narrower than itself, which would fall to one point and vanish, is instead cut
//!    into sectors around the part's axis when that removes enough error for its clusters.
//!    A leaf cell at an open border whose border vertices turn away from the rest (a sheet's
//!    edge turned down, such as glass modeled with its thickness) is cut in two, so the
//!    turned-down strip keeps its width.
//! 2. **Clusters as vertices.** Each cluster keeps the summed quadric of the original surface
//!    and is represented by its member vertex of least error (vertices are never invented, so
//!    attributes stay exact). Members joined by an edge inside the cluster share a wedge:
//!    UV seams and hard edges survive clustering as separate wedges at one position.
//! 3. **Collapse tail.** Lazy greedy half-edge collapse from about twice the target down to
//!    it, with the inherited quadrics, a wedge distance attribute term, border sliding, link
//!    and flip checks. Each live cluster has at most one heap entry, keyed by a geometry-only
//!    lower bound and checked exactly when it surfaces, so most neighbors a collapse touches
//!    are never costed again.
//! 4. **Finish.** Each remaining unlocked cluster moves along its surface normal to the minimum
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
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// Octree depth (Morton bits per axis).
const DEPTH: u32 = 10;
/// Stage-one clusters per target triangle.
const CLUSTERS_PER_TRI: f32 = 0.55;
/// Extra clusters at small targets: the budget is scaled by 1 + this / (target + 8).
const SMALL_TARGET: f32 = 24.0;
/// A mesh of at most `RETRY_MAX_TRIS` triangles that ends below `RETRY_BELOW` of its target
/// is reduced again with more clusters, up to `RETRIES` times. At 0.75, a small open mesh
/// whose flange cuts lift the first pass to 76-81% of target kept that count and lost the
/// border samples its neighbors share: CarConcept's seats at 3% widened the scene's mean
/// crack from 0.87% to 1.20% of the diagonal.
const RETRY_MAX_TRIS: usize = 20_000;
const RETRY_BELOW: f32 = 0.8;
const RETRIES: usize = 2;
/// Cost factor of a collapse that removes a folded triangle, so the tail takes folds out
/// before anything else.
const FOLD_BIAS: f32 = 1e-3;
/// A wedge with no partner across a collapsed edge is kept, not merged into the nearest
/// wedge, when its normal differs from that one's by more than this (squared difference of
/// unit normals; 0.25 is about 29 degrees): a hard edge.
const HARD_EDGE: f32 = 0.25;
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
/// The least move the position solve makes, as a share of the shortest edge, squared.
const RELAX_LEAST: f32 = 1e-10;
/// Bit of a sort key where the 30-bit Morton code starts; below it, a position hash and
/// the vertex index share the rest.
const CODE_SHIFT: u32 = 34;
/// Weight of the distance to a cluster's centroid in choosing its representative: enough to
/// settle ties (flat regions, where every member's error is zero, and would otherwise all pick
/// their cells' low corners and fold), too little to override real error.
const CENTROID_PULL: f64 = 1e-5;
/// Fine cells per wanted cluster: the resolution refinement starts from.
const FINE_PER_CLUSTER: usize = 4;
/// Thin parts (see `verify` in `realtime_pass`): a cell is a candidate when its normals
/// cancel by at least `THIN_CANCEL` of its area, its surface lies within `THIN_DEPTH` cell
/// sizes (RMS) of the centroid's planes, and its normal spread (the normal moment's
/// principal 2x2 minors against its trace squared, 1/4 for a tube) is at least
/// `THIN_SPREAD`. A candidate with at least `THIN_MIN_VERTS` vertices is cut into
/// `THIN_SECTORS` cells around its axis (three; `sector_of` assumes it) when the error that
/// removes is at least `THIN_GAIN` of what an octree split at the refinement threshold
/// removes per cluster.
const THIN_CANCEL: f64 = 0.3;
const THIN_DEPTH: f64 = 0.5;
const THIN_SPREAD: f64 = 0.1;
const THIN_MIN_VERTS: u32 = 4;
const THIN_SECTORS: usize = 3;
const THIN_GAIN: f64 = 0.05;
/// No thin splits on a mesh with open borders whose cluster budget is below this.
const THIN_OPEN_MIN_TARGET: usize = 300;
/// A cell is cut at an open border with a crease beside it (see `cheap_flange`) when the
/// normals of its border positions and of the rest meet at a cosine below `FLANGE_COS`, and
/// the rest's normals keep at least `FLANGE_REST` of their area.
const FLANGE_COS: f64 = 0.5;
const FLANGE_REST: f64 = 0.7;

/// Vertices per parallel chunk, and the most chunks; items per task for parallel maps.
const CHUNK_VERTS: usize = 4096;
const MAX_CHUNKS: usize = 128;
const PAR_MIN: usize = 4096;
/// Refinement sums: quadric (10), area, area-weighted normal (3), area-weighted position (3),
/// and over vertices at an open-border position, area-weighted normal (3) and area.
const NS: usize = 21;
type Q = [f64; 10];
/// A thin cell's axes: its two cross-section directions and its centroid.
type Axes = ([f64; 3], [f64; 3], [f64; 3]);
/// A thin cell's split: its axes and the refinement sums of its sectors.
type Thin = (Axes, [[f32; NS]; THIN_SECTORS]);

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

/// The leading eigenvector of a symmetric 3x3 matrix (xx, xy, xz, yy, yz, zz) and its
/// eigenvalue, by power iteration from the column of the largest diagonal entry.
fn lead_eig(m: &[f64; 6]) -> Option<([f64; 3], f64)> {
    let mul = |d: [f64; 3]| {
        [
            m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
            m[1] * d[0] + m[3] * d[1] + m[4] * d[2],
            m[2] * d[0] + m[4] * d[1] + m[5] * d[2],
        ]
    };
    let mut d = if m[0] >= m[3] && m[0] >= m[5] {
        [m[0], m[1], m[2]]
    } else if m[3] >= m[5] {
        [m[1], m[3], m[4]]
    } else {
        [m[2], m[4], m[5]]
    };
    let mut l = 0.0;
    for _ in 0..10 {
        let x = mul(d);
        l = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        if l.is_nan() || l <= 0.0 {
            return None;
        }
        d = x.map(|v| v / l);
    }
    Some((d, l))
}

/// Spreads the low 10 bits of `x` three apart.
const fn spread_bits(x: u32) -> u32 {
    let mut x = x & 0x3ff;
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// `spread_bits` of every 10-bit value.
static SPREAD: [u32; 1024] = {
    let mut t = [0u32; 1024];
    let mut i = 0;
    while i < 1024 {
        t[i] = spread_bits(i as u32);
        i += 1;
    }
    t
};

/// Spreads the low 10 bits of `x` three apart.
#[inline]
fn spread(x: u32) -> u32 {
    SPREAD[(x & 0x3ff) as usize]
}

/// Sorts keys (code above `CODE_SHIFT`, tie-breaking bits below) in
/// parallel: blocks count their top digits and write their keys straight to place (relaxed
/// atomic stores, plain stores on common targets), then each bucket is sorted on its own.
/// The order is the same however it is split. Sorted into `out`; `hist` is working space.
fn sort_by_code(keys: &[u64], hist: &mut Vec<[u32; 1024]>, out: &mut Vec<u64>) {
    // One scatter by the top 10 bits of the code, then each bucket (a few hundred keys,
    // in cache) sorted on its own. Keys are distinct, so the order is the full sort's.
    let n = keys.len();
    let blk = (n / (4 * rayon::current_num_threads())).max(PAR_MIN);
    let digit = |k: u64| (k >> (CODE_SHIFT + 3 * DEPTH - 10)) as usize & 1023;
    keys.par_chunks(blk)
        .map(|b| {
            let mut h = [0u32; 1024];
            for &k in b {
                h[digit(k)] += 1;
            }
            h
        })
        .collect_into_vec(hist);
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
    let mut dst: Vec<AtomicU64> = recycle(std::mem::take(out), AtomicU64::new);
    (0..n)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|_| AtomicU64::new(0))
        .collect_into_vec(&mut dst);
    keys.par_chunks(blk)
        .zip(hist.par_iter_mut())
        .for_each(|(b, h)| {
            for &k in b {
                let d = digit(k);
                dst[h[d] as usize].store(k, Relaxed);
                h[d] += 1;
            }
        });
    *out = dst.into_iter().map(AtomicU64::into_inner).collect();
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
}

/// A refinement node: a range of fine cells, the leading Morton digits all its codes share
/// (`DEPTH`: one finest cell), and its sums.
struct Node {
    fs: u32,
    fe: u32,
    level: u32,
    st: [f64; NS],
}

/// A fine cell: a range of the Morton order, the codes at its two ends, and its sums.
struct FCell {
    s: u32,
    e: u32,
    c0: u32,
    c1: u32,
    st: [f32; NS],
}

/// A vertex's face sums: area-weighted normal, area, and the corner sums that find open fans
/// (wrapping; exact for the values they are read for). Corners count by position (the first
/// rank at it), so the copies of a vertex at one position sum to the fan around it.
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct VSum {
    n: [f32; 3],
    area: f32,
    s1: i32,
    /// Σ mix(next) − Σ mix(prev): a check on the ends `s1` and `s2` give. They are an open
    /// fan's only when the fan is one arc; faces forming two arcs around a vertex (common on
    /// unwelded meshes) pass the other checks with two unrelated vertices about half the time.
    s3: u32,
    s2: i64,
}

const _: () = assert!(std::mem::size_of::<VSum>() == 32);

/// A non-linear mix of a vertex id for the fan check sum `VSum::s3`.
#[inline]
fn fan_mix(x: u32) -> u32 {
    let x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x ^ (x >> 13)
}

/// The two border neighbors of an open fan v, a1, ..., ak: Σnext − Σprev = a1 − ak and
/// Σnext² − Σprev² = a1² − ak², so a1 + ak = (a1² − ak²) / (a1 − ak); checked against
/// Σmix(next) − Σmix(prev) = mix(a1) − mix(ak).
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
    (ok(a1) && ok(ak) && fan_mix(a1 as u32).wrapping_sub(fan_mix(ak as u32)) == s.s3)
        .then_some([a1 as usize, ak as usize])
}

/// A vertex's quadric from its face sums, moment-matched: the fan's face-normal second
/// moment M = Σ Aᵢ nᵢnᵢᵀ has trace A, and with mean normal length |Σ Aᵢnᵢ| = A·cos θ for a
/// cone of normals, M ≈ (A cos²θ) uuᵀ + (A sin²θ / 2)(I − uuᵀ). Flat fans give their exact
/// plane; curved ones also resist tangential motion, as their faces would. Open fans add the
/// border planes of `borders`, its end edges that are borders (`u32::MAX`: not one).
#[inline]
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

/// Adds a vertex to refinement sums: quadric, area, area-weighted normal and position, and
/// its normal and area again when it stands at an open-border position (`border`).
#[inline]
fn vstats(st: &mut [f32; NS], q: &[f32; 10], s: &VSum, p: Vec3, border: bool) {
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
    if border {
        st[17] += s.n[0];
        st[18] += s.n[1];
        st[19] += s.n[2];
        st[20] += s.area;
    }
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
        self.s3 = self.s3.wrapping_add(d.s3);
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
    cbord: Vec<bool>,
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

/// A stage-two triangle's sorted cluster ids packed into one key: a u64 while the ids fit in
/// 21 bits each (any target under a few million triangles), a u128 past that.
trait TriKey:
    Copy
    + Eq
    + From<bool>
    + std::ops::Shl<u32, Output = Self>
    + std::ops::Shr<u32, Output = Self>
    + std::ops::BitOr<Output = Self>
{
    const EMPTY: Self;
    fn mix(self) -> u64;
}

impl TriKey for u64 {
    const EMPTY: u64 = u64::MAX;
    #[inline]
    fn mix(self) -> u64 {
        self
    }
}

impl TriKey for u128 {
    const EMPTY: u128 = u128::MAX;
    #[inline]
    fn mix(self) -> u64 {
        self as u64 ^ ((self >> 64) as u64).rotate_left(29)
    }
}

/// Open-addressing set of stage-two triangles, each stored as its key shifted up one with the
/// orientation it was first seen in as the low bit: one word per slot. A kept triangle's
/// three clusters differ, so no slot of a key is all ones, the empty slot.
struct TriSet<K> {
    slots: Vec<K>,
    shift: u32,
}

impl<K: TriKey> TriSet<K> {
    /// Room for `n` triangles, in `slots`' memory.
    fn new(n: usize, slots: Vec<K>) -> TriSet<K> {
        let cap = (n * 2).next_power_of_two().max(16);
        TriSet {
            slots: filled(slots, cap, K::EMPTY),
            shift: 64 - cap.trailing_zeros(),
        }
    }
    /// The orientation `k` was first seen in, or None (recording `fwd`) if it is new.
    #[inline]
    fn first(&mut self, k: K, fwd: bool) -> Option<bool> {
        let mask = self.slots.len() - 1;
        let mut i = (k.mix().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        loop {
            let s = self.slots[i];
            if s == K::EMPTY {
                self.slots[i] = (k << 1) | K::from(fwd);
                return None;
            }
            if s >> 1 == k {
                return Some(s != k << 1);
            }
            i = (i + 1) & mask;
        }
    }
}

enum Seen {
    Small(TriSet<u64>),
    Large(TriSet<u128>),
}

/// What a stage-two triangle needs, gathered in parallel (see `realtime_pass`).
struct Pre {
    key: u128,
    c: [u32; 3],
    roots: [u32; 3],
    facing: Vec3,
    mat: u32,
    fwd: bool,
    fold: bool,
}

/// Memory a thread's scratch keeps whatever the calls after need, and how much more than a
/// call used it may hold past that before it is freed.
const RETAIN_ALWAYS: usize = 64 << 20;
const RETAIN_RATIO: usize = 4;

/// The size of a buffer: bytes held and bytes in use (inner buffers included).
trait Held {
    const NESTED: bool = false;
    fn held(&self) -> (usize, usize) {
        (0, 0)
    }
}

macro_rules! plain_held {
    ($($t:ty),* $(,)?) => { $(impl Held for $t {})* };
}
plain_held!(
    u8,
    u32,
    u64,
    u128,
    f32,
    bool,
    Vec3,
    [u32; 2],
    [u32; 3],
    [u32; 1024],
    (f32, u32),
    (u32, u32),
    ([u32; 3], u32),
    (u32, VSum),
    [f64; NS],
    Q,
    VSum,
    FCell,
    Node,
    Pre,
    Option<Thin>,
    (u32, u32, u32),
    UvGrad,
);

impl<T: Held> Held for Vec<T> {
    const NESTED: bool = true;
    fn held(&self) -> (usize, usize) {
        let s = std::mem::size_of::<T>();
        let (mut cap, mut len) = (self.capacity() * s, self.len() * s);
        if T::NESTED {
            for x in self {
                let (c, l) = x.held();
                cap += c;
                len += l;
            }
        }
        (cap, len)
    }
}

impl<T: Ord> Held for BinaryHeap<T> {
    fn held(&self) -> (usize, usize) {
        let s = std::mem::size_of::<T>();
        (self.capacity() * s, self.len() * s)
    }
}

impl<K, V> Held for std::collections::HashMap<K, V> {
    fn held(&self) -> (usize, usize) {
        let s = std::mem::size_of::<(K, V)>() + 1;
        (self.capacity() * s, self.len() * s)
    }
}

/// The working buffers of a realtime call, each named for the array it holds. A call takes
/// the calling thread's set, sizes every buffer it uses afresh (cleared and refilled, so
/// nothing of an earlier call is read), and puts the set back, so a thread that reduces again
/// writes to memory it already has instead of new pages from the system.
macro_rules! scratch {
    ($($f:ident: $t:ty,)*) => {
        #[derive(Default)]
        struct RealtimeScratch {
            $($f: $t,)*
        }

        impl RealtimeScratch {
            /// Bytes held and bytes in use, over every buffer.
            fn bytes(&self) -> (usize, usize) {
                let mut b = (0, 0);
                $(
                    let (c, l) = self.$f.held();
                    b = (b.0 + c, b.1 + l);
                )*
                b
            }

            /// The buffers that hold no memory.
            #[cfg(test)]
            fn empty(&self) -> Vec<&'static str> {
                let mut v = Vec::new();
                $(
                    if self.$f.held().0 == 0 {
                        v.push(stringify!($f));
                    }
                )*
                v
            }
        }
    };
}

scratch! {
    keys_in: Vec<u64>,
    keys: Vec<u64>,
    hist: Vec<[u32; 1024]>,
    rank: Vec<u32>,
    spos: Vec<Vec3>,
    routed: Vec<Vec<Vec<([u32; 3], u32)>>>,
    first_at: Vec<u32>,
    vs: Vec<VSum>,
    spill: Vec<Vec<(u32, VSum)>>,
    borders: Vec<[u32; 2]>,
    bflag: Vec<bool>,
    starts: Vec<u32>,
    fcells: Vec<FCell>,
    nodes: Vec<Node>,
    leaves: Vec<u32>,
    kids: Vec<u32>,
    heap: BinaryHeap<(u64, std::cmp::Reverse<u32>)>,
    tstate: Vec<u8>,
    fl: Vec<bool>,
    tinfo: std::collections::HashMap<u32, Thin>,
    pending: Vec<u32>,
    found: Vec<Option<Thin>>,
    rest: Vec<u32>,
    cells: Vec<(u32, u32, u32)>,
    cellr: Vec<u32>,
    cell_st: Vec<[f64; NS]>,
    parent: Vec<u32>,
    kept: Vec<Vec<([u32; 3], u32)>>,
    cid: Vec<u32>,
    wbest: Vec<(f32, u32)>,
    warea_v: Vec<f32>,
    cq: Vec<Q>,
    cpos: Vec<Vec3>,
    rep: Vec<u32>,
    clocked: Vec<bool>,
    cbord: Vec<bool>,
    pre: Vec<Vec<Pre>>,
    wid: Vec<u32>,
    wsrc: Vec<u32>,
    wc: Vec<u32>,
    warea: Vec<f32>,
    tw: Vec<[u32; 3]>,
    tc: Vec<[u32; 3]>,
    tmat: Vec<u32>,
    twin: Vec<bool>,
    seen_small: Vec<u64>,
    seen_large: Vec<u128>,
    fold: Vec<bool>,
    tn: Vec<Vec3>,
    attr: Vec<f32>,
    alive: Vec<bool>,
    side: Vec<[u32; 2]>,
    sdeg: Vec<u8>,
    soff: Vec<u32>,
    slen: Vec<u32>,
    scap: Vec<u32>,
    star_pool: Vec<u32>,
    wout: Vec<u32>,
    srcs: Vec<u32>,
    cluster_out: Vec<u32>,
    normal_parts: Vec<Vec<Vec3>>,
    uv: UvCarry,
}

thread_local! {
    /// The calling thread's realtime buffers between calls.
    static SCRATCH: std::cell::Cell<RealtimeScratch> = std::cell::Cell::new(RealtimeScratch::default());
}

/// Frees the working memory [`Quality::Realtime`](crate::reduce::Quality::Realtime) keeps on
/// the calling thread between calls (see there). Other threads keep theirs; a program that
/// reduced on a rayon pool can free every worker's with `rayon::broadcast(|_|
/// release_realtime_scratch())`.
pub fn release_realtime_scratch() {
    let _ = SCRATCH.try_with(|s| drop(s.take()));
}

/// An emptied buffer with `v`'s memory, for another element type of the same size and
/// alignment (an atomic and its plain value): collected in place, so nothing is allocated.
fn recycle<T, U>(mut v: Vec<T>, f: impl FnMut(T) -> U) -> Vec<U> {
    v.clear();
    v.into_iter().map(f).collect()
}

/// `v` emptied, with room for `n` items.
fn reuse<T>(mut v: Vec<T>, n: usize) -> Vec<T> {
    v.clear();
    v.reserve(n);
    v
}

/// `v` holding `n` copies of `x`.
fn filled<T: Clone>(mut v: Vec<T>, n: usize, x: T) -> Vec<T> {
    v.clear();
    v.resize(n, x);
    v
}

pub(crate) fn simplify_realtime(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: Option<usize>,
) -> Result<SimplifyResult> {
    // The thread's buffers are taken for the call, not borrowed: a call that rayon starts on
    // this thread while it waits (a host reducing meshes in parallel) finds none and makes its
    // own, and whichever call ends last leaves its set.
    let mut s = SCRATCH.try_with(|s| s.take()).unwrap_or_default();
    let r = reduce_with(mesh, opts, target, &mut s);
    let (held, used) = s.bytes();
    // A call that needed far less than one before it frees the excess (all of it) once
    // there is enough to matter, so one huge mesh doesn't pin its memory for good.
    if held <= RETAIN_ALWAYS || held <= used.saturating_mul(RETAIN_RATIO) {
        let _ = SCRATCH.try_with(|c| c.set(s));
    }
    r
}

fn reduce_with(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: Option<usize>,
    s: &mut RealtimeScratch,
) -> Result<SimplifyResult> {
    // The cluster budget assumes about two triangles per cluster, as on a closed surface. An
    // open sheet has about one (a disc of k vertices has k - 2 triangles), so stage one can
    // land under the target, where the tail has nothing to add. A small mesh that comes out
    // short goes again with the budget scaled by the shortfall; a large one is rarely short
    // by much and is not worth a second pass.
    let mut per_tri = CLUSTERS_PER_TRI;
    let mut r = realtime_pass(mesh, opts, target, per_tri, s)?;
    let Some(t) = target else {
        return Ok(r);
    };
    for _ in 0..RETRIES {
        let got = r.mesh.triangle_count();
        if mesh.triangle_count() > RETRY_MAX_TRIS
            || got as f32 >= t as f32 * RETRY_BELOW
            || got >= mesh.triangle_count()
        {
            break;
        }
        per_tri *= (1.2 * t as f32 / got.max(1) as f32).min(4.0);
        let again = realtime_pass(mesh, opts, target, per_tri, s)?;
        if again.mesh.triangle_count() <= got {
            break;
        }
        r = again;
    }
    Ok(r)
}

/// Input triangles from which the UV carry works out its sums on the pool, beside the collapse
/// tail: below this, handing them out costs more than the work (a scene of many small meshes
/// pays it on every one).
const UV_BESIDE_TRIS: usize = 65536;

/// One source vertex's sums for one UV set over its input triangles: twice the area-weighted
/// gradients of u and v, and the UV box of those triangles.
#[derive(Clone, Copy)]
struct UvGrad {
    gu: Vec3,
    gv: Vec3,
    lo: Vec2,
    hi: Vec2,
}

impl UvGrad {
    const EMPTY: UvGrad = UvGrad {
        gu: Vec3::ZERO,
        gv: Vec3::ZERO,
        lo: Vec2::INFINITY,
        hi: Vec2::NEG_INFINITY,
    };
}

/// Moves each output vertex's UVs with it. A vertex takes its attributes from one input
/// vertex (its wedge's source) but stands at its cluster's position, which can lie well away
/// from that source (across a merged layer or at the end of a long cluster), so the texture
/// there would slide. Each UV set is carried along its surface gradient at the source (the
/// area-weighted mean over the source's input triangles) by the move, and held to the UV box
/// of those triangles so it never runs off the end of a chart. Vertices at their source's
/// position keep its UVs as they are.
///
/// The sums at a source don't depend on where it ends up, only the move does: a large mesh
/// works them out for every wedge's source on the pool while the collapse tail runs, a small
/// one for the sources that moved, after. Either way a source's triangles are added up in
/// index order, so the result is the same.
#[derive(Default)]
struct UvCarry {
    /// A bit per input vertex whose sums are wanted.
    mark: Vec<u64>,
    /// Each marked vertex's slot. A slot is read only where its bit is set, which is only
    /// where this call wrote it, so the array is never cleared.
    slot: Vec<u32>,
    /// Slots in use.
    used: usize,
    /// Per slot: twice the area of its triangles, and a [`UvGrad`] per UV set.
    area: Vec<f32>,
    grad: Vec<UvGrad>,
}

impl Held for UvCarry {
    fn held(&self) -> (usize, usize) {
        let parts = [
            self.mark.held(),
            self.slot.held(),
            self.area.held(),
            self.grad.held(),
        ];
        parts
            .iter()
            .fold((0, 0), |(c, l), (pc, pl)| (c + pc, l + pl))
    }
}

/// Twice the area, the unit normal and the edges opposite each corner of input triangle `t`,
/// or None when it has no area.
#[inline]
fn uv_tri(mesh: &Mesh, t: &[u32]) -> Option<(f32, Vec3, [Vec3; 3])> {
    let x = [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]);
    let n = (x[1] - x[0]).cross(x[2] - x[0]);
    let a = n.length();
    if a <= 0.0 {
        return None;
    }
    Some((a, n / a, [x[2] - x[1], x[0] - x[2], x[1] - x[0]]))
}

/// Adds one triangle to a source's sums for one UV set: a corner's hat function has gradient
/// n × (opposite edge) / |n|², so over the triangle the area-weighted gradient of u is
/// n̂ × Σ uᵢ eᵢ / 2.
#[inline]
fn uv_add(g: &mut UvGrad, n: Vec3, e: &[Vec3; 3], u: [Vec2; 3]) {
    for w in u {
        (g.lo, g.hi) = (g.lo.min(w), g.hi.max(w));
    }
    g.gu += n.cross(e[0] * u[0].x + e[1] * u[1].x + e[2] * u[2].x);
    g.gv += n.cross(e[0] * u[0].y + e[1] * u[1].y + e[2] * u[2].y);
}

impl UvCarry {
    /// Marks each distinct vertex of `sources` and gives it a slot.
    fn mark(&mut self, nv: usize, sources: impl Iterator<Item = u32>) {
        self.mark.clear();
        self.mark.resize(nv.div_ceil(64), 0);
        if self.slot.len() < nv {
            self.slot.resize(nv, 0);
        }
        let mut used = 0usize;
        for v in sources {
            let (v, bit) = (v as usize, 1u64 << (v % 64));
            if self.mark[v / 64] & bit == 0 {
                self.mark[v / 64] |= bit;
                self.slot[v] = used as u32;
                used += 1;
            }
        }
        self.used = used;
    }

    /// The marked vertices' sums in one pass over the triangles in index order.
    fn sums_serial(&mut self, mesh: &Mesh) {
        let ns = mesh.uvs.len();
        let (mark, slot) = (&self.mark, &self.slot);
        self.area.clear();
        self.area.resize(self.used, 0.0);
        self.grad.clear();
        self.grad.resize(self.used * ns, UvGrad::EMPTY);
        let is = |v: u32| mark[v as usize / 64] >> (v % 64) & 1 != 0;
        for t in mesh.indices.chunks_exact(3) {
            if !(is(t[0]) || is(t[1]) || is(t[2])) {
                continue;
            }
            let Some((a, n, e)) = uv_tri(mesh, t) else {
                continue;
            };
            for &v in t {
                if !is(v) {
                    continue;
                }
                let k = slot[v as usize] as usize;
                for (g, set) in self.grad[k * ns..(k + 1) * ns].iter_mut().zip(&mesh.uvs) {
                    uv_add(g, n, &e, [t[0], t[1], t[2]].map(|i| set[i as usize]));
                }
                self.area[k] += a;
            }
        }
    }

    /// Each output vertex that stands away from its source moved by its source's gradients
    /// (every such source must be marked and summed).
    fn apply(&self, mesh: &Mesh, source_vertex: &[u32], out: &mut Mesh) {
        let ns = mesh.uvs.len();
        for (o, &v) in source_vertex.iter().enumerate() {
            let v = v as usize;
            let d = out.positions[o] - mesh.positions[v];
            if d == Vec3::ZERO || self.mark[v / 64] >> (v % 64) & 1 == 0 {
                continue;
            }
            let k = self.slot[v] as usize;
            let area = self.area[k];
            if area > 0.0 {
                for ((g, set), uvs) in self.grad[k * ns..(k + 1) * ns]
                    .iter()
                    .zip(&mesh.uvs)
                    .zip(out.uvs.iter_mut())
                {
                    let step = Vec2::new((g.gu / area).dot(d), (g.gv / area).dot(d));
                    uvs[o] = (set[v] + step).clamp(g.lo, g.hi);
                }
            }
        }
    }
}

/// One realtime reduction with `per_tri` stage-one clusters per target triangle.
fn realtime_pass(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: Option<usize>,
    per_tri: f32,
    s: &mut RealtimeScratch,
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
    (0..nv)
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
        .collect_into_vec(&mut s.keys_in);
    sort_by_code(&s.keys_in, &mut s.hist, &mut s.keys);
    let keys = &s.keys[..];
    let code = |r: usize| (keys[r] >> CODE_SHIFT) as u32;
    let vert = |r: usize| (keys[r] & ((1 << index_bits) - 1)) as u32;

    // From here on vertices are Morton ranks: positions (unit-diagonal frame, centered) are
    // gathered once, triangle corners map through `rank`, and every per-vertex pass after
    // reads its arrays in order.
    let mut arank: Vec<AtomicU32> = recycle(std::mem::take(&mut s.rank), AtomicU32::new);
    (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|_| AtomicU32::new(0))
        .collect_into_vec(&mut arank);
    (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|r| {
            let v = vert(r) as usize;
            arank[v].store(r as u32, Relaxed);
            (mesh.positions[v] - center) * inv
        })
        .collect_into_vec(&mut s.spos);
    s.rank = arank.into_iter().map(AtomicU32::into_inner).collect();
    let (spos, rank) = (&s.spos[..], &s.rank[..]);
    lap("morton");

    // The parallel passes run over a fixed number of chunks of the Morton order, set by the
    // vertex count alone, so the result never depends on the thread count. Triangles go to
    // the chunk of their first corner in rank order, kept in index order within it.
    let csize = nv.div_ceil((nv / CHUNK_VERTS).clamp(1, MAX_CHUNKS));
    let nch = nv.div_ceil(csize);
    let tblock = nt.div_ceil(nch).max(1);
    // (Block and chunk lists past this call's counts are kept for later calls, unread.)
    let nblk = nt.div_ceil(tblock);
    if s.routed.len() < nblk {
        s.routed.resize_with(nblk, Vec::new);
    }
    s.routed[..nblk]
        .par_iter_mut()
        .enumerate()
        .for_each(|(b, out)| {
            if out.len() < nch {
                out.resize_with(nch, Vec::new);
            }
            out[..nch].iter_mut().for_each(Vec::clear);
            for t in b * tblock..((b + 1) * tblock).min(nt) {
                let tr = mesh.triangle(t).map(|v| rank[v as usize]);
                out[tr[0].min(tr[1]).min(tr[2]) as usize / csize].push((tr, t as u32));
            }
        });
    let routed = &s.routed[..nblk];
    let chunk_tris = |k: usize| routed.iter().flat_map(move |b| b[k].iter());

    // Each vertex's position, as the first rank at it (vertices at one position are
    // neighbors in rank, see `sort_by_code`). Fan sums count neighbors by these, so the copies
    // of a vertex split by a seam or unwelded faces sum to one fan.
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
        .collect_into_vec(&mut s.first_at);
    let first_at = &s.first_at[..];

    // Face sums per vertex. A chunk owns its vertices' sums; corners past its end are handed
    // on and added after, in chunk order.
    // Filled in parallel, so its pages are first touched by every thread.
    let vs = &mut s.vs;
    (0..nv)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|_| VSum::default())
        .collect_into_vec(vs);
    if s.spill.len() < nch {
        s.spill.resize_with(nch, Vec::new);
    }
    let parts: Vec<(f64, usize)> = vs
        .par_chunks_mut(csize)
        .zip(s.spill[..nch].par_iter_mut())
        .enumerate()
        .map(|(k, (own, spill))| {
            let lo = (k * csize) as u32;
            let (mut total, mut live) = (0f64, 0usize);
            spill.clear();
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
                    let (nx, pv) = (first_at[nx as usize], first_at[pv as usize]);
                    let d = VSum {
                        n: h.to_array(),
                        area,
                        s1: (nx as i32).wrapping_sub(pv as i32),
                        s2: (nx as i64 * nx as i64).wrapping_sub(pv as i64 * pv as i64),
                        s3: fan_mix(nx).wrapping_sub(fan_mix(pv)),
                    };
                    match own.get_mut((v - lo) as usize) {
                        Some(s) => s.add(&d),
                        None => spill.push((v, d)),
                    }
                }
            }
            (total, live)
        })
        .collect();
    let (mut total_area, mut live) = (0f64, 0usize);
    for ((t, l), spill) in parts.into_iter().zip(&s.spill[..nch]) {
        total_area += t;
        live += l;
        for (v, d) in spill {
            vs[*v as usize].add(d);
        }
    }
    if live == 0 {
        return Ok(unchanged(mesh));
    }
    let vs = &s.vs[..];
    let avg_area = (total_area / live as f64).max(1e-30);
    // Border edges, worked out per position: the fan sums count neighbors by position (see
    // `first_at`), so the copies of a vertex at one position facing one way (split by a seam,
    // or unwelded faces) add up to the fan of the surface around that position, closed unless
    // the surface stops there. The first such copy holds the position's border planes. (The
    // back of a double-sided sheet is there too, facing the other way, and is summed apart.)
    const NONE: [u32; 2] = [u32::MAX; 2];
    let bw = opts.border_weight;
    if bw > 0.0 {
        let holder = |r: usize| {
            let (p, f) = (spos[r], first_at[r] as usize);
            let n = Vec3::from_array(vs[r].n);
            let n2 = n.length_squared();
            // Facing within about 154° of each other (cos > −0.9), without the root.
            let facing = |q: usize| {
                let m = Vec3::from_array(vs[q].n);
                let d = n.dot(m);
                !(d < 0.0 && d * d >= 0.81 * n2 * m.length_squared())
            };
            if (f..r).any(facing) {
                return NONE;
            }
            let mut sum = vs[r];
            let mut q = r + 1;
            while q < nv && spos[q] == p {
                if facing(q) {
                    sum.add(&vs[q]);
                }
                q += 1;
            }
            match fan_ends(f, &sum, nv) {
                Some([a, b]) if a != b => [a as u32, b as u32],
                _ => NONE,
            }
        };
        // Whether each vertex stands at an open-border position: a copy there facing its way
        // holds border planes (the back of a double-sided sheet faces the other way and is not
        // on that border). The copies split by a seam fall on the same side of a flange cut.
        // Both worked out per position run, a run straddling a chunk's end in both chunks.
        // Every entry is written below, so the buffers are only sized here.
        let (mut borders, mut bflag) =
            (std::mem::take(&mut s.borders), std::mem::take(&mut s.bflag));
        borders.resize(nv, NONE);
        bflag.resize(nv, false);
        borders
            .par_chunks_mut(PAR_MIN)
            .zip(bflag.par_chunks_mut(PAR_MIN))
            .enumerate()
            .for_each(|(k, (ob, of))| {
                let (lo, hi) = (k * PAR_MIN, k * PAR_MIN + ob.len());
                let mut run: Vec<[u32; 2]> = Vec::new();
                let mut r = lo;
                while r < hi {
                    let (p, f) = (spos[r], first_at[r] as usize);
                    let mut e = r + 1;
                    while e < nv && spos[e] == p {
                        e += 1;
                    }
                    if e - f == 1 {
                        let b = holder(r);
                        (ob[r - lo], of[r - lo]) = (b, b != NONE);
                        r = e;
                        continue;
                    }
                    run.clear();
                    run.extend((f..e).map(holder));
                    let any = run.iter().any(|b| *b != NONE);
                    for x in r..e.min(hi) {
                        ob[x - lo] = run[x - f];
                        of[x - lo] = any && {
                            let n = Vec3::from_array(vs[x].n);
                            let n2 = n.length_squared();
                            (f..e).any(|q| {
                                if run[q - f] == NONE {
                                    return false;
                                }
                                let m = Vec3::from_array(vs[q].n);
                                let d = n.dot(m);
                                !(d < 0.0 && d * d >= 0.81 * n2 * m.length_squared())
                            })
                        };
                    }
                    r = e;
                }
            });
        (s.borders, s.bflag) = (borders, bflag);
    } else {
        s.borders.clear();
        s.bflag.clear();
    }
    let borders = &s.borders[..];
    let bflag = &s.bflag[..];
    let at_border = |r: usize| bflag.get(r).copied().unwrap_or(false);
    // Importance scales a vertex's error, so clusters stay finer and collapse later there.
    let importance = opts.vertex_importance.as_deref();
    let vq = |r: usize| {
        let mut q = vertex_quadric(r, &vs[r], spos, bw, borders.get(r).copied().unwrap_or(NONE));
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
    //
    // The budget assumes a closed surface's two triangles per cluster. A cluster on an open
    // border has one fewer, and at a small target most clusters are on a border (a disc of
    // four clusters is two triangles), so small targets get more clusters per triangle; the
    // extra fades out by a few hundred triangles.
    let k_target = target.map(|t| {
        let small = 1.0 + SMALL_TARGET / (t as f32 + 8.0);
        ((t as f32 * per_tri * small).ceil() as usize).max(4)
    });
    let fine_level = {
        const NL: usize = DEPTH as usize + 1;
        // Counted over blocks with a plain loop each (integer sums, so in any order).
        let first_split = (0..nv.div_ceil(PAR_MIN))
            .into_par_iter()
            .map(|b| {
                let mut h = [0usize; NL];
                for r in (b * PAR_MIN).max(1)..((b + 1) * PAR_MIN).min(nv) {
                    let x = code(r) ^ code(r - 1);
                    if x != 0 {
                        h[((3 * DEPTH - 1 - (31 - x.leading_zeros())) / 3) as usize] += 1;
                    }
                }
                h
            })
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
        let starts = &mut s.starts;
        starts.clear();
        starts.push(0);
        starts.par_extend(
            (1..nv)
                .into_par_iter()
                .with_min_len(PAR_MIN)
                .filter(|&r| (code(r) ^ code(r - 1)) >> shift != 0)
                .map(|r| r as u32),
        );
        starts.push(nv as u32);
        let mut fcells = reuse(std::mem::take(&mut s.fcells), starts.len());
        fcells.extend(starts.windows(2).map(|w| FCell {
            s: w[0],
            e: w[1],
            c0: 0,
            c1: 0,
            st: [0.0; NS],
        }));
        fcells
    };
    fcells.par_iter_mut().with_min_len(16).for_each(|f| {
        (f.c0, f.c1) = (code(f.s as usize), code(f.e as usize - 1));
        for r in f.s as usize..f.e as usize {
            vstats(&mut f.st, &vq(r), &vs[r], spos[r], at_border(r));
        }
    });
    lap("fine");

    // Refinement: split the worst cell until there are enough.
    let cube_to_unit = (ext * inv) as f64;
    // The level of a range whose first and last codes are these.
    let vlevel = |first: u32, last: u32| -> u32 {
        let x = first ^ last;
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
    // A cell that may hold a thin part, from its sums alone: its normals partly cancel, its
    // surface lies close to the centroid's planes for its size, and its normals spread over
    // more than one direction (a tube or strut, not a folded plate). Such a cell is counted
    // as `THIN_SECTORS` clusters while it is a leaf.
    // A small target on a mesh with open borders: there the clusters a split takes from
    // other cells move the border's clusters, and a mesh beside it sharing the border
    // comes apart from it.
    let small_open =
        k_target.is_some_and(|k| k < THIN_OPEN_MIN_TARGET) && borders.iter().any(|b| *b != NONE);
    let cheap_thin = |st: &[f64; NS], level: u32| -> bool {
        let area = st[10];
        if area <= 0.0 || small_open {
            return false;
        }
        let nlen = (st[11] * st[11] + st[12] * st[12] + st[13] * st[13]).sqrt();
        if nlen > (1.0 - THIN_CANCEL) * area {
            return false;
        }
        let e = qeval(st, st[14] / area, st[15] / area, st[16] / area);
        let h = cube_to_unit * 0.5f64.powi(level as i32);
        if e >= area * (THIN_DEPTH * h).powi(2) {
            return false;
        }
        // The sum of the normal moment's principal 2x2 minors against its trace squared: 0
        // for a plate, 1/4 for a tube, 1/3 for a ball.
        let (a, b, c, d, f, g) = (st[0], st[1], st[2], st[4], st[5], st[7]);
        let t = a + d + g;
        let s2 = a * d - b * b + a * g - c * c + d * g - f * f;
        s2 >= THIN_SPREAD * t * t
    };
    // A cell holding an open border with a crease beside it (a sheet's edge turned down, as
    // on glass modeled with its thickness): the normals at its border positions turn from
    // those of the rest by more than `FLANGE_COS`, and the rest is one surface (its normals
    // cancel by less than `FLANGE_REST`; the open end of a tube, whose border normals point
    // all ways, is not a flange). One cluster would press the turned-down strip flat into
    // the sheet and lose its shading, so such a cell is cut in two, border positions and the
    // rest, and counted as two clusters while it is a leaf.
    let cheap_flange = |st: &[f64; NS]| -> bool {
        let (ab, ar) = (st[20], st[10] - st[20]);
        if !(ab > 0.0 && ar > 1e-6 * st[10]) {
            return false;
        }
        let nb = [st[17], st[18], st[19]];
        let nr = [st[11] - nb[0], st[12] - nb[1], st[13] - nb[2]];
        let d = nb[0] * nr[0] + nb[1] * nr[1] + nb[2] * nr[2];
        let lb = nb[0] * nb[0] + nb[1] * nb[1] + nb[2] * nb[2];
        let lr = nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2];
        lr >= FLANGE_REST * FLANGE_REST * ar * ar && d < FLANGE_COS * (lb * lr).sqrt()
    };
    // The sector of a thin cell a position falls in, by its angle around the part's axis.
    let sector_of = |p: Vec3, t: &Axes| -> usize {
        let (d, d2, cen) = t;
        let q = [
            p.x as f64 - cen[0],
            p.y as f64 - cen[1],
            p.z as f64 - cen[2],
        ];
        let o1 = q[0] * d[0] + q[1] * d[1] + q[2] * d[2];
        let o2 = q[0] * d2[0] + q[1] * d2[1] + q[2] * d2[2];
        // Three sectors of 120 degrees: within 60 degrees of d, then the side of d2.
        let r3 = 3f64.sqrt();
        if o1 > 0.0 && o2.abs() * (1.0 / r3) < o1 {
            1
        } else if o2 >= 0.0 {
            2
        } else {
            0
        }
    };
    // The full test for a cell `cheap_thin` passed, over its vertices: the part's
    // cross-section (the two directions its normals lie along) and whether cutting the cell
    // into sectors around the axis is worth its extra clusters, against `thr`, what one more
    // octree split removes at the current refinement threshold.
    let verify = |s: u32, e: u32, n: &Node, thr: f64| -> Option<Thin> {
        let st = &n.st;
        // The split removes at most the cell's whole error.
        let before = err(st, n.level).0;
        let need = THIN_GAIN * thr * (THIN_SECTORS - 1) as f64;
        if e - s < THIN_MIN_VERTS || before < need || !cheap_thin(st, n.level) {
            return None;
        }
        // Not on an open border: its clusters keep their members' positions there, so a mesh
        // beside it sharing the border keeps meeting it, which a different cut would undo.
        if borders
            .get(s as usize..e as usize)
            .is_some_and(|b| b.iter().any(|b| *b != NONE))
        {
            return None;
        }
        let mut m = [0f64; 6];
        for r in s as usize..e as usize {
            let v = &vs[r];
            if v.area <= 0.0 {
                continue;
            }
            let (x, y, z) = (v.n[0] as f64, v.n[1] as f64, v.n[2] as f64);
            let ia = 1.0 / v.area as f64;
            m[0] += x * x * ia;
            m[1] += x * y * ia;
            m[2] += x * z * ia;
            m[3] += y * y * ia;
            m[4] += y * z * ia;
            m[5] += z * z * ia;
        }
        let (d, l1) = lead_eig(&m)?;
        // The second direction, from the moment with the first taken out, made orthogonal
        // to it exactly.
        let mut m2 = m;
        m2[0] -= l1 * d[0] * d[0];
        m2[1] -= l1 * d[0] * d[1];
        m2[2] -= l1 * d[0] * d[2];
        m2[3] -= l1 * d[1] * d[1];
        m2[4] -= l1 * d[1] * d[2];
        m2[5] -= l1 * d[2] * d[2];
        let (d2, _) = lead_eig(&m2)?;
        let k = d2[0] * d[0] + d2[1] * d[1] + d2[2] * d[2];
        let d2 = [d2[0] - k * d[0], d2[1] - k * d[1], d2[2] - k * d[2]];
        let l2 = (d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]).sqrt();
        if l2.is_nan() || l2 <= 0.0 {
            return None;
        }
        let area = st[10];
        let t = (
            d,
            d2.map(|v| v / l2),
            [st[14] / area, st[15] / area, st[16] / area],
        );
        let mut parts = [[0f32; NS]; THIN_SECTORS];
        for r in s as usize..e as usize {
            vstats(
                &mut parts[sector_of(spos[r], &t)],
                &vq(r),
                &vs[r],
                spos[r],
                at_border(r),
            );
        }
        let after: f64 = parts
            .iter()
            .map(|p| err(&p.map(|x| x as f64), n.level).0)
            .sum();
        (before - after >= need).then_some((t, parts))
    };
    let mut nodes: Vec<Node> = reuse(std::mem::take(&mut s.nodes), k_target.unwrap_or(1024) * 2);
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
            level: vlevel(code(0), code(nv - 1)),
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
                    c0: code(r),
                    c1: 0,
                    st: [0.0; NS],
                };
                while r < e && (code(r) >> shift) & 7 == d {
                    vstats(&mut f.st, &vq(r), &vs[r], spos[r], at_border(r));
                    r += 1;
                }
                (f.e, f.c1) = (r as u32, code(r - 1));
                fcells.push(f);
            }
            fe = fcells.len();
        }
        let mut k = fs;
        while k < fe {
            let d = (fcells[k].c0 >> shift) & 7;
            let start = k;
            let mut st = [0f64; NS];
            while k < fe && (fcells[k].c0 >> shift) & 7 == d {
                for (a, b) in st.iter_mut().zip(&fcells[k].st) {
                    *a += *b as f64;
                }
                k += 1;
            }
            out.push(nodes.len() as u32);
            nodes.push(Node {
                fs: start as u32,
                fe: k as u32,
                level: vlevel(fcells[start].c0, fcells[k - 1].c1),
                st,
            });
        }
    };
    let max1 = opts
        .max_error
        .map(|e| (e * inv * STAGE1_ERROR_SHARE) as f64)
        .map(|e| e * e);
    let mut leaves: Vec<u32> = reuse(std::mem::take(&mut s.leaves), 0);
    let mut kids = std::mem::take(&mut s.kids);
    let mut heap = std::mem::take(&mut s.heap);
    heap.clear();
    if nodes[0].level == DEPTH {
        leaves.push(0);
    } else {
        heap.push((
            err(&nodes[0].st, nodes[0].level).0.to_bits(),
            std::cmp::Reverse(0),
        ));
    }
    // Thin state per node: 0 none, 1 counted as thin by its sums, 2 verified thin, 3 not
    // thin after all. A node counted as thin is counted as `THIN_SECTORS` clusters while it
    // is a leaf, so the budget stays the same; once it is reached the counted leaves are
    // verified, and the budget freed by those that fail goes back to refinement.
    let extra = THIN_SECTORS - 1;
    let reserve = |n: &Node| cheap_thin(&n.st, n.level) as u8;
    let mut tstate: Vec<u8> = filled(std::mem::take(&mut s.tstate), 1, reserve(&nodes[0]));
    let mut tinfo = std::mem::take(&mut s.tinfo);
    tinfo.clear();
    let (mut pending, mut found) = (std::mem::take(&mut s.pending), std::mem::take(&mut s.found));
    // Flange cells (see `cheap_flange`) count one extra cluster while they are leaves.
    let mut fl: Vec<bool> = filled(std::mem::take(&mut s.fl), 1, cheap_flange(&nodes[0].st));
    let mut count = 1 + tstate[0] as usize * extra + fl[0] as usize;
    if let Some(k) = k_target {
        loop {
            while count < k {
                let Some((_, std::cmp::Reverse(i))) = heap.pop() else {
                    break;
                };
                if matches!(tstate[i as usize], 1 | 2) {
                    count -= extra;
                }
                tstate[i as usize] = 0;
                count -= fl[i as usize] as usize;
                split(&mut nodes, &mut fcells, i as usize, &mut kids);
                count += kids.len() - 1;
                for &c in &kids {
                    let n = &nodes[c as usize];
                    let t = reserve(n);
                    tstate.push(t);
                    let f = cheap_flange(&n.st);
                    fl.push(f);
                    count += t as usize * extra + f as usize;
                    if n.level == DEPTH {
                        leaves.push(c);
                    } else {
                        heap.push((err(&n.st, n.level).0.to_bits(), std::cmp::Reverse(c)));
                    }
                }
            }
            pending.clear();
            pending.extend(
                heap.iter()
                    .map(|x| x.1.0)
                    .chain(leaves.iter().copied())
                    .filter(|&i| tstate[i as usize] == 1),
            );
            if pending.is_empty() {
                break;
            }
            let thr = heap.peek().map_or(0.0, |x| f64::from_bits(x.0));
            let one = |&i: &u32| {
                let n = &nodes[i as usize];
                verify(fcells[n.fs as usize].s, fcells[n.fe as usize - 1].e, n, thr)
            };
            // In parallel only when there is enough work to pay for waking the pool.
            let work: usize = pending
                .iter()
                .map(|&i| {
                    let n = &nodes[i as usize];
                    (fcells[n.fe as usize - 1].e - fcells[n.fs as usize].s) as usize
                })
                .sum();
            if work < PAR_MIN * 4 {
                found.clear();
                found.extend(pending.iter().map(one));
            } else {
                pending.par_iter().map(one).collect_into_vec(&mut found);
            }
            let mut freed = false;
            for (&i, &f) in pending.iter().zip(&found) {
                if let Some(t) = f {
                    tstate[i as usize] = 2;
                    tinfo.insert(i, t);
                } else {
                    tstate[i as usize] = 3;
                    count -= extra;
                    freed = true;
                }
            }
            if !freed {
                break;
            }
        }
    }
    let mut rest = reuse(std::mem::take(&mut s.rest), heap.len());
    rest.extend(heap.drain().map(|x| x.1.0));
    if let Some(m) = max1 {
        // Within the error budget everywhere (stage one's share of it).
        while let Some(i) = rest.pop() {
            let n = &nodes[i as usize];
            if n.level == DEPTH || err(&n.st, n.level).1 <= m {
                leaves.push(i);
                continue;
            }
            split(&mut nodes, &mut fcells, i as usize, &mut kids);
            tstate.resize(nodes.len(), 0);
            for &c in &kids {
                tstate[c as usize] = reserve(&nodes[c as usize]);
            }
            rest.extend_from_slice(&kids);
        }
    } else {
        leaves.append(&mut rest);
    }
    // Cells in Morton order, as rank ranges, grouped into chunks of whole cells.
    let mut cells: Vec<(u32, u32, u32)> = reuse(std::mem::take(&mut s.cells), leaves.len());
    cells.extend(leaves.iter().map(|&i| {
        let n = &nodes[i as usize];
        (fcells[n.fs as usize].s, fcells[n.fe as usize - 1].e, i)
    }));
    cells.sort_unstable_by_key(|c| c.0);
    // Thin parts: a leaf cell holding a tube or strut narrower than the cell would fall to
    // one point and the part would vanish. Such a cell is cut into sectors around the part's
    // axis, each its own cell, so the part keeps a cross-section.
    let ncell = cells.len() + tinfo.len() * extra;
    let mut cellr = filled(std::mem::take(&mut s.cellr), nv, 0u32);
    let mut cell_st: Vec<[f64; NS]> = reuse(std::mem::take(&mut s.cell_st), ncell);
    let mut chunks: Vec<[usize; 4]> = Vec::new();
    let (mut r0, mut c0) = (0usize, 0usize);
    let (mut nsplit, mut nflange) = (0, 0);
    // Flange cells' two parts, worked out ahead (in parallel when there is enough work):
    // border positions summed over their vertices, the rest as the cell less those. A cell
    // that turns out thin below drops its parts.
    let fcand: Vec<usize> = (0..cells.len())
        .filter(|&k| {
            let i = cells[k].2 as usize;
            tstate[i] != 2 && cheap_flange(&nodes[i].st)
        })
        .collect();
    let fpart = |&k: &usize| {
        let (s, e, i) = cells[k];
        let mut b = [0f32; NS];
        for r in s as usize..e as usize {
            if at_border(r) {
                vstats(&mut b, &vq(r), &vs[r], spos[r], true);
            }
        }
        let b = b.map(|x| x as f64);
        let all = &nodes[i as usize].st;
        [b, std::array::from_fn(|j| all[j] - b[j])]
    };
    let fwork: usize = fcand
        .iter()
        .map(|&k| (cells[k].1 - cells[k].0) as usize)
        .sum();
    let fparts: Vec<[[f64; NS]; 2]> = if fwork < PAR_MIN * 4 {
        fcand.iter().map(fpart).collect()
    } else {
        fcand.par_iter().map(fpart).collect()
    };
    let mut fnext = fcand.iter().zip(fparts).peekable();
    for (k, &(s, e, i)) in cells.iter().enumerate() {
        let flange = match fnext.peek() {
            Some(&(&j, _)) if j == k => fnext.next().map(|x| x.1),
            _ => None,
        };
        let c = cell_st.len();
        let thin = match tstate[i as usize] {
            2 => tinfo.get(&i).copied(),
            1 => verify(s, e, &nodes[i as usize], 0.0),
            _ => None,
        };
        if let Some((t, parts)) = &thin {
            nsplit += 1;
            for r in s as usize..e as usize {
                cellr[r] = (c + sector_of(spos[r], t)) as u32;
            }
            cell_st.extend(parts.iter().map(|p| p.map(|x| x as f64)));
        } else if let Some(parts) = flange {
            // Border positions first, then the rest.
            nflange += 1;
            for r in s as usize..e as usize {
                cellr[r] = (c + !at_border(r) as usize) as u32;
            }
            cell_st.extend(parts);
        } else {
            cellr[s as usize..e as usize].fill(c as u32);
            cell_st.push(nodes[i as usize].st);
        }
        if e as usize >= (chunks.len() + 1) * csize || k + 1 == cells.len() {
            chunks.push([r0, e as usize, c0, cell_st.len()]);
            (r0, c0) = (e as usize, cell_st.len());
        }
    }
    if prof {
        eprintln!(
            "  thin cells {nsplit}, flange cells {nflange} of {}",
            cells.len()
        );
    }
    (s.cells, s.nodes, s.leaves, s.fcells) = (cells, nodes, leaves, fcells);
    (s.kids, s.heap, s.tstate, s.tinfo, s.fl) = (kids, heap, tstate, tinfo, fl);
    (s.pending, s.found, s.rest) = (pending, found, rest);
    lap("octree");

    // Wedges: members joined by an edge inside their cell. A triangle with all three corners
    // in different cells has them in different clusters too (two corners in one cell share
    // an edge, hence a cluster), so the kept set is final here. Two edges of a triangle
    // connect all three corners.
    let mut aparent: Vec<AtomicU32> = recycle(std::mem::take(&mut s.parent), AtomicU32::new);
    aparent.extend((0..nv as u32).map(AtomicU32::new));
    if s.kept.len() < nch {
        s.kept.resize_with(nch, Vec::new);
    }
    s.kept[..nch]
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, kept)| {
            kept.clear();
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
        });
    let kept = &s.kept[..nch];
    let mut parent: Vec<u32> = aparent.into_iter().map(AtomicU32::into_inner).collect();
    lap("wpass");

    // Every set from here on (edges inside a cell, equal positions, clusters) lies inside
    // one cell, so each chunk finishes its clusters on its own.
    let locked_v: Option<Vec<bool>> = opts.locked_vertices.as_ref().map(|l| {
        (0..nv)
            .map(|r| l.get(vert(r) as usize).copied().unwrap_or(false))
            .collect()
    });
    let mut cid = filled(std::mem::take(&mut s.cid), nv, 0u32);
    let mut wbest = filled(std::mem::take(&mut s.wbest), nv, (f32::INFINITY, u32::MAX));
    let mut warea_v = filled(std::mem::take(&mut s.warea_v), nv, 0f32);
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
            // share a Morton code, and runs of one code are short; most are one position.
            let mut run: Vec<u32> = Vec::new();
            let mut i = 0;
            while i < n {
                let mut e = i + 1;
                while e < n && code(r0 + e) == code(r0 + i) {
                    e += 1;
                }
                if e - i > 1 {
                    let pos = |i: u32| spos[r0 + i as usize];
                    let bits = |i: u32| pos(i).to_array().map(f32::to_bits);
                    let (a, b) = (i as u32, e as u32);
                    if !pos(a).is_nan() && (a + 1..b).all(|x| bits(x) == bits(a)) {
                        // One position (equal however compared).
                        for x in a + 1..b {
                            union(sp, a, x);
                        }
                    } else if e - i <= 8 {
                        for a in a..b {
                            for b in a + 1..b {
                                if pos(a) == pos(b) {
                                    union(sp, a, b);
                                }
                            }
                        }
                    } else {
                        run.clear();
                        run.extend(a..b);
                        run.sort_unstable_by_key(|&i| bits(i));
                        for w in run.windows(2) {
                            if bits(w[0]) == bits(w[1]) {
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
            let mut cbord = vec![false; nc];
            for i in 0..n {
                let (r, c) = (r0 + i, sp[i] as usize);
                clocked[c] |=
                    locked_v.as_ref().is_some_and(|l| l[r]) || (lock_border && vs[r].s1 != 0);
                cbord[c] |= borders.get(r).is_some_and(|b| *b != NONE);
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
                cbord,
                worst1,
            }
        })
        .collect();
    (s.cellr, s.cell_st) = (cellr, cell_st);
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
    let mut cq: Vec<Q> = reuse(std::mem::take(&mut s.cq), nc);
    let mut cpos: Vec<Vec3> = reuse(std::mem::take(&mut s.cpos), nc);
    let mut rep: Vec<u32> = reuse(std::mem::take(&mut s.rep), nc);
    let mut clocked: Vec<bool> = reuse(std::mem::take(&mut s.clocked), nc);
    let mut cbord: Vec<bool> = reuse(std::mem::take(&mut s.cbord), nc);
    let mut worst1 = 0f64;
    for o in outs {
        cq.extend(o.cq);
        cpos.extend(o.best.iter().map(|b| spos[b.1 as usize]));
        rep.extend(o.best.iter().map(|b| vert(b.1 as usize)));
        clocked.extend(o.clocked);
        cbord.extend(o.cbord);
        worst1 = worst1.max(o.worst1);
    }
    lap("clusters");

    // Stage-two triangles over wedges; same-orientation duplicates go, opposite ones are
    // kept as double-sided twins. What each triangle needs is gathered in parallel, chunk by
    // chunk; duplicates and wedge numbers are settled after, in order.
    let small_keys = nc < 1 << 21;
    let key_bits = if small_keys { 21 } else { 32 };
    if s.pre.len() < nch {
        s.pre.resize_with(nch, Vec::new);
    }
    s.pre[..nch]
        .par_iter_mut()
        .zip(kept.par_iter())
        .for_each(|(pre, ch)| {
            pre.clear();
            pre.extend(ch.iter().map(|&(tri, t)| {
                let c = tri.map(|r| cid[r as usize]);
                let mut sorted = c;
                sorted.sort_unstable();
                // Rotation-invariant orientation: the cyclic order with the smallest id
                // first.
                let r = (0..3).min_by_key(|&i| c[i]).unwrap();
                // Clusters stand at member vertices chosen one by one, so three of them
                // can form a triangle facing the other way from the one it stands for: a
                // fold. The tail takes those out first.
                let o = tri.map(|r| spos[r as usize]);
                let v = c.map(|x| cpos[x as usize]);
                let facing = (o[1] - o[0]).cross(o[2] - o[0]);
                Pre {
                    key: ((sorted[0] as u128) << (2 * key_bits))
                        | ((sorted[1] as u128) << key_bits)
                        | sorted[2] as u128,
                    c,
                    roots: tri.map(|r| parent[r as usize]),
                    facing,
                    mat: mesh.material_ids.get(t as usize).copied().unwrap_or(0),
                    fwd: c[(r + 1) % 3] == sorted[1],
                    fold: facing.dot((v[1] - v[0]).cross(v[2] - v[0])) < 0.0,
                }
            }));
        });
    let pre = &s.pre[..nch];
    let nkept: usize = pre.iter().map(Vec::len).sum();
    let nw_est = nkept * 3 / 2 + 16;
    let mut wid = filled(std::mem::take(&mut s.wid), nv, u32::MAX);
    let mut wsrc: Vec<u32> = reuse(std::mem::take(&mut s.wsrc), nw_est);
    let mut wc: Vec<u32> = reuse(std::mem::take(&mut s.wc), nw_est);
    let mut warea: Vec<f32> = reuse(std::mem::take(&mut s.warea), nw_est);
    let mut tw: Vec<[u32; 3]> = reuse(std::mem::take(&mut s.tw), nkept);
    let mut tc: Vec<[u32; 3]> = reuse(std::mem::take(&mut s.tc), nkept);
    let mut tmat: Vec<u32> = reuse(std::mem::take(&mut s.tmat), nkept);
    let mut twin: Vec<bool> = reuse(std::mem::take(&mut s.twin), nkept);
    let mut seen = if small_keys {
        Seen::Small(TriSet::new(nkept, std::mem::take(&mut s.seen_small)))
    } else {
        Seen::Large(TriSet::new(nkept, std::mem::take(&mut s.seen_large)))
    };
    let mut fold: Vec<bool> = reuse(std::mem::take(&mut s.fold), nkept);
    let mut tn: Vec<Vec3> = reuse(std::mem::take(&mut s.tn), nkept);
    for p in pre.iter().flatten() {
        let first = match &mut seen {
            Seen::Small(s) => s.first(p.key as u64, p.fwd),
            Seen::Large(s) => s.first(p.key, p.fwd),
        };
        // A duplicate facing the first one's way goes; one facing the other way is its twin.
        let is_twin = match first {
            Some(f) if f == p.fwd => continue,
            Some(_) => true,
            None => false,
        };
        let mut w = [0u32; 3];
        for k in 0..3 {
            let root = p.roots[k] as usize;
            if wid[root] == u32::MAX {
                wid[root] = wsrc.len() as u32;
                wsrc.push(vert(wbest[root].1 as usize));
                wc.push(p.c[k]);
                warea.push(warea_v[root]);
            }
            w[k] = wid[root];
        }
        fold.push(p.fold);
        tn.push(p.facing);
        tw.push(w);
        tc.push(p.c);
        tmat.push(p.mat);
        twin.push(is_twin);
    }
    match seen {
        Seen::Small(t) => s.seen_small = t.slots,
        Seen::Large(t) => s.seen_large = t.slots,
    }
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
    let mut attr = filled(std::mem::take(&mut s.attr), wsrc.len() * dim, 0f32);
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
        alive: filled(std::mem::take(&mut s.alive), alive_count, true),
        twin,
        fold,
        tn,
        warea,
        attr,
        dim,
        nd2: if nd > 0 { nwt * nwt } else { 0.0 },
        avg_area,
        locked: clocked,
        side: filled(std::mem::take(&mut s.side), nc, [u32::MAX; 2]),
        sdeg: filled(std::mem::take(&mut s.sdeg), nc, 0),
        cbord,
        soff: filled(std::mem::take(&mut s.soff), nc, 0),
        slen: filled(std::mem::take(&mut s.slen), nc, 0),
        scap: filled(std::mem::take(&mut s.scap), nc, 0),
        pool: std::mem::take(&mut s.star_pool),
    };
    st.classify_edges();
    lap("attrs");
    let max_err2 = opts.max_error.map(|e| (e * inv) * (e * inv));
    let first = st.first_costs(max_err2);
    let tail = || {
        let (collapses, worst2, wremap, cremap) =
            st.run(first, target, max_err2, opts.min_face_dot);
        let moved = st.relax(&cremap, opts.min_face_dot);
        (collapses, worst2, wremap, cremap, moved)
    };
    // A large mesh's UV sums at every wedge's source, on another thread beside the serial tail
    // (after the tail's parallel first pass, which would otherwise wait on them).
    let uv_beside = !mesh.uvs.is_empty() && nt >= UV_BESIDE_TRIS;
    let (collapses, worst2, wremap, cremap, moved) = if uv_beside {
        let uv = &mut s.uv;
        rayon::join(tail, || {
            uv.mark(nv, wsrc.iter().copied());
            uv.sums_serial(mesh);
        })
        .0
    } else {
        tail()
    };
    lap("tail");

    // Output: one vertex per wedge still used, at its cluster's representative position.
    let nw = wsrc.len();
    let mut wout = filled(std::mem::take(&mut s.wout), nw, u32::MAX);
    let mut srcs: Vec<u32> = reuse(std::mem::take(&mut s.srcs), 0);
    let mut indices: Vec<u32> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();
    let mut cluster_out = filled(std::mem::take(&mut s.cluster_out), nc, u32::MAX);
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
        // A wedge kept across a hard edge stands where its cluster ended up.
        let mut c = wc[w as usize] as usize;
        while cremap[c] as usize != c {
            c = cremap[c] as usize;
        }
        out.positions[o] = if moved[c] {
            st.cpos[c] * diag + center
        } else {
            mesh.positions[rep[c] as usize]
        };
    }
    if !mesh.uvs.is_empty() {
        if !uv_beside {
            let moved = source_vertex
                .iter()
                .enumerate()
                .filter(|&(o, &v)| out.positions[o] != mesh.positions[v as usize])
                .map(|(_, &v)| v);
            s.uv.mark(nv, moved);
            s.uv.sums_serial(mesh);
        }
        s.uv.apply(mesh, &source_vertex, &mut out);
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
        let nparts = nv.div_ceil(span);
        if s.normal_parts.len() < nparts {
            s.normal_parts.resize_with(nparts, Vec::new);
        }
        let parts = &mut s.normal_parts[..nparts];
        parts.par_iter_mut().enumerate().for_each(|(k, sum)| {
            sum.clear();
            sum.resize(n_out, Vec3::ZERO);
            for r in k * span..((k + 1) * span).min(nv) {
                let root = parent[r] as usize;
                if wid[root] != u32::MAX {
                    let o = wout[wremap[wid[root] as usize] as usize];
                    if o != u32::MAX {
                        sum[o as usize] += src[vert(r) as usize] * vs[r].area;
                    }
                }
            }
        });
        if let Some((sum, rest)) = parts.split_first_mut() {
            for part in rest {
                for (a, b) in sum.iter_mut().zip(part.iter()) {
                    *a += *b;
                }
            }
            for (n, s) in dst.iter_mut().zip(sum.iter()) {
                let l = s.length();
                if l > 1e-20 {
                    *n = *s / l;
                }
            }
        }
    }
    // Every buffer taken from the scratch goes back for the next call.
    let Tail {
        cpos,
        cq,
        tw,
        tc,
        alive,
        twin,
        fold,
        tn,
        warea,
        attr,
        locked,
        side,
        sdeg,
        cbord,
        soff,
        slen,
        scap,
        pool,
        ..
    } = st;
    (s.cpos, s.cq, s.tw, s.tc, s.alive, s.twin, s.fold, s.tn) =
        (cpos, cq, tw, tc, alive, twin, fold, tn);
    (s.warea, s.attr, s.clocked, s.side, s.sdeg, s.cbord) =
        (warea, attr, locked, side, sdeg, cbord);
    (s.soff, s.slen, s.scap, s.star_pool) = (soff, slen, scap, pool);
    (s.parent, s.cid, s.wbest, s.warea_v, s.rep, s.wid) = (parent, cid, wbest, warea_v, rep, wid);
    (s.wsrc, s.wc, s.tmat, s.wout, s.srcs, s.cluster_out) =
        (wsrc, wc, tmat, wout, srcs, cluster_out);
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
    /// Faces the other way from the surface it stands for (see where it is set).
    fold: Vec<bool>,
    /// The normal (unnormalized) of the original triangle each one stands for: the side it
    /// must keep facing.
    tn: Vec<Vec3>,
    warea: Vec<f32>,
    attr: Vec<f32>,
    dim: usize,
    /// Squared scale of the normals, a wedge's first three attributes (0: no normals).
    nd2: f32,
    avg_area: f64,
    locked: Vec<bool>,
    /// The two neighbors along open-border or non-manifold edges a constrained cluster may
    /// slide to; `sdeg` counts such edges (0: unconstrained).
    side: Vec<[u32; 2]>,
    sdeg: Vec<u8>,
    /// Holds an input vertex on an open border: its position stays an input position, so
    /// where a separately reduced neighbor picks the same vertex the two stay sealed.
    cbord: Vec<bool>,
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
        self.pool.clear();
        self.pool.resize(off as usize, 0);
        for (t, c) in self.tc.iter().enumerate() {
            for &x in c {
                let x = x as usize;
                self.pool[(self.soff[x] + self.slen[x]) as usize] = t as u32;
                self.slen[x] += 1;
            }
        }
        // Open-border edges (one face) and junctions (more than two): border clusters slide
        // along their border, junction clusters along their junction, and clusters on more
        // than two such edges stay put. A cluster on both slides along its border: after
        // clustering most junctions are thin parts pressed flat, and locking every cluster
        // where one meets a border left the tail the good clusters to take instead. Each
        // cluster counts its own edges from its triangles (twins left out), in parallel.
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
                if bdeg == 0 && jdeg > 0 {
                    bdeg = jdeg;
                    side = jside;
                }
                (side, bdeg, bdeg > 2)
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

    /// Moves each remaining cluster that is not locked to the minimum of its summed quadric
    /// (it has stood at one of its member vertices so far), where that stays inside its
    /// neighborhood and turns no triangle. Clusters that slide along a junction move too: the
    /// move is along the surface normal, which their planes contain. Clusters holding an open
    /// border vertex stay put. Returns which clusters moved.
    fn relax(&mut self, cremap: &[u32], min_dot: f32) -> Vec<bool> {
        let nc = self.cpos.len();
        let mut moved = vec![false; nc];
        let mut star: Vec<([Vec3; 3], Vec3, usize)> = Vec::new();
        for c in 0..nc {
            // A cluster on an open border keeps its input position: moved along the normal it
            // leaves the border line, and a mesh reduced on its own beside it, sharing that
            // border, would come apart from it.
            if cremap[c] != c as u32 || self.locked[c] || self.cbord[c] {
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
            // A move too small to see is not made, so a cluster already in place keeps its
            // member's exact position (a flat sheet's outline, a border shared with a neighbor).
            let d2 = (np - p).length_squared();
            if !ok || d2 > RELAX_REACH * RELAX_REACH * short2 || d2 <= RELAX_LEAST * short2 {
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
                fold: self.fold[t],
                t: t as u32,
            });
        }
    }

    /// Wedge map p→q (wedges of p in triangles shared with q take q's wedge there, the rest
    /// the nearest of those) and its attribute cost, from p's gathered triangles; None when p
    /// and q share no triangle. A wedge of the rest whose nearest lies across a hard edge
    /// maps to itself: it moves to q as it is, at the same cost, so the faces on its side
    /// keep their own normal.
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
        let mut cost = out
            .iter()
            .map(|&(pw, qw)| self.warea[pw as usize] * self.adist(pw, qw))
            .sum::<f32>();
        for &pw in &pws[..np] {
            if out[..shared].iter().any(|&(a, _)| a == pw) {
                continue;
            }
            let best = out[..shared]
                .iter()
                .map(|&(_, qw)| (self.adist(pw, qw), qw))
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                .unwrap();
            cost += self.warea[pw as usize] * best.0;
            let d = self.dim;
            let (x, y) = (pw as usize * d, best.1 as usize * d);
            let hard = self.nd2 > 0.0
                && (0..3)
                    .map(|k| (self.attr[x + k] - self.attr[y + k]).powi(2))
                    .sum::<f32>()
                    > HARD_EDGE * self.nd2;
            out.push((pw, if hard { pw } else { best.1 }));
        }
        Some(cost / self.avg_area as f32)
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
            // Not turned against where it was, and not turned against the surface it stands
            // for either: a triangle of no area (three clusters in a line) has no "was", and
            // would otherwise be free to come out facing either way.
            let facing = self.tn[k.t as usize];
            l1 > 1e-12
                && n0.dot(n1) >= min_dot * n0.length() * l1
                && !(n1.dot(facing) < 0.0 && n0.dot(facing) >= 0.0)
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
        // A star has a handful of neighbors: a short scan, with no marks over all clusters
        // (which every job of the parallel first pass would have to allocate).
        s.nbr.clear();
        for k in &s.loc {
            for &x in &k.c {
                if x != p && !s.nbr.contains(&x) {
                    s.nbr.push(x);
                }
            }
        }
        let qp = &self.cq[p as usize];
        // Most stars hold no fold; then no candidate needs checking for one.
        let any_fold = s.loc.iter().any(|k| k.fold);
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
            // A collapse that takes a fold out goes ahead of every ordinary one.
            let fixes = any_fold && s.loc.iter().any(|k| k.fold && k.c.contains(&q));
            let g = (raw / self.avg_area) as f32;
            s.cand
                .push((if fixes { g * FOLD_BIAS } else { g }, q, g2, fixes));
        }
        if !exact {
            return s
                .cand
                .iter()
                .map(|c| (c.0, c.1, c.2))
                .min_by(|a, b| a.0.total_cmp(&b.0));
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
        for &(g, q, g2, fixes) in s.cand.iter() {
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
            let c = g + if fixes { ac * FOLD_BIAS } else { ac };
            if best.is_none_or(|b| c < b.0) {
                best = Some((c, q, g2));
            }
        }
        best
    }

    /// Moves p onto q; returns the triangles removed.
    fn collapse(&mut self, p: u32, q: u32, map: &[(u32, u32)]) -> usize {
        for &(pw, qw) in map {
            if pw != qw {
                let a = self.warea[pw as usize];
                self.warea[qw as usize] += a;
            }
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

    /// Each cluster's first exact best, in parallel (they only read the state), so a
    /// cluster's first turn at the top mostly holds. (Lower bounds here and exact costs on
    /// demand were measured slower: that moves parallel work into the serial loop.)
    fn first_costs(&self, max_err2: Option<f32>) -> Vec<Option<(f32, u32, f32)>> {
        (0..self.cpos.len() as u32)
            .into_par_iter()
            .with_min_len(64)
            .map_init(
                || Scratch::new(0),
                |s, p| {
                    if !self.movable(p) {
                        return None;
                    }
                    self.gather(p, &mut s.loc);
                    self.best(p, max_err2, true, s)
                },
            )
            .collect()
    }

    /// Collapses in greedy order down to `target`, from the first costs of
    /// [`first_costs`](Self::first_costs); returns (collapses, worst mean squared distance,
    /// wedge remap chain, cluster remap chain).
    fn run(
        &mut self,
        mut cache: Vec<Option<(f32, u32, f32)>>,
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
        // A cluster's exact best is kept with the collapse count it was found at, and stands
        // while no collapse since has changed what it reads (`dirty`): its triangles, its
        // neighbors' quadrics, its wedges' areas and its constraint sides.
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
    fold: bool,
    t: u32,
}

/// Per-thread working space for the tail's neighborhood queries.
struct Scratch {
    mark: Vec<u32>,
    epoch: u32,
    nbr: Vec<u32>,
    cand: Vec<(f32, u32, f32, bool)>,
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

    #[test]
    fn triangle_set_keeps_first_orientation() {
        fn check<K: super::TriKey>(key: impl Fn(u64) -> K) {
            let mut s = super::TriSet::new(40, Vec::new());
            for i in 0..40u64 {
                assert_eq!(s.first(key(i), i % 3 == 0), None);
            }
            for i in 0..40u64 {
                assert_eq!(s.first(key(i), true), Some(i % 3 == 0));
            }
        }
        // Keys as packed for the largest cluster ids of each width.
        check(|i| (((1u64 << 21) - 3) << 42) | (i << 21) | 7);
        check(|i| ((u32::MAX as u128 - 2) << 64) | ((i as u128) << 32) | 7);
    }

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

    fn grid(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32, y as f32);
                pos.push(glam::Vec3::new(fx, fy, (fx * 0.3).sin() * (fy * 0.2).cos()));
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let i = y * (n + 1) + x;
                idx.extend_from_slice(&[i, i + 1, i + n + 2, i, i + n + 2, i + n + 1]);
            }
        }
        Mesh::new(pos, idx)
    }

    #[test]
    fn reused_buffers_match_fresh_ones() {
        // A thread keeps its buffers between calls; every call must come out as on a thread
        // that never ran one, whatever ran before it (bigger or smaller meshes, other options).
        let small = {
            let mut m = uv_sphere(24, 12);
            m.compute_normals(60.0);
            m
        };
        let no_borders = SimplifyOptions {
            border_weight: 0.0,
            ..opts(0.05)
        };
        let max_error = SimplifyOptions {
            max_error: Some(0.02),
            quality: Quality::Realtime,
            ..Default::default()
        };
        let cases = [
            (sphere(), opts(0.05)),
            (small.clone(), opts(0.1)),
            (grid(60), opts(0.02)),
            (sphere(), no_borders.clone()),
            (grid(12), opts(0.2)),
            (small, max_error),
            (grid(60), no_borders),
            (sphere(), opts(0.3)),
        ];
        let on_new_thread = |m: &Mesh, o: &SimplifyOptions| {
            std::thread::scope(|sc| sc.spawn(|| simplify(m, o).unwrap()).join().unwrap())
        };
        let fresh: Vec<_> = cases.iter().map(|(m, o)| on_new_thread(m, o)).collect();
        std::thread::scope(|sc| {
            sc.spawn(|| {
                let order: Vec<usize> = (0..cases.len()).chain((0..cases.len()).rev()).collect();
                for i in order {
                    let ((m, o), f) = (&cases[i], &fresh[i]);
                    let r = simplify(m, o).unwrap();
                    assert_eq!(r.mesh, f.mesh, "case {i}");
                    assert_eq!(r.vertex_remap, f.vertex_remap, "case {i}");
                    assert_eq!(r.source_vertex, f.source_vertex, "case {i}");
                    assert_eq!(r.error.to_bits(), f.error.to_bits(), "case {i}");
                    assert_eq!(r.collapses, f.collapses, "case {i}");
                }
                // Every buffer went back to the thread (these four are only used by meshes
                // with thin parts or millions of clusters).
                let s = super::SCRATCH.take();
                let missing: Vec<_> = s
                    .empty()
                    .into_iter()
                    .filter(|f| !["seen_large", "tinfo", "pending", "found"].contains(f))
                    .collect();
                assert!(missing.is_empty(), "not handed back: {missing:?}");
                super::SCRATCH.set(s);
                super::release_realtime_scratch();
                assert_eq!(super::SCRATCH.take().bytes().0, 0);
            })
            .join()
            .unwrap()
        });
    }

    #[test]
    fn recycled_buffers_keep_their_memory() {
        let v: Vec<u32> = vec![7; 1000];
        let p = v.as_ptr() as usize;
        let a = super::recycle(v, std::sync::atomic::AtomicU32::new);
        assert!(a.is_empty() && a.capacity() >= 1000 && a.as_ptr() as usize == p);
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
        // The outline holds exactly: the bounds are the grid's, and the area is too, but for
        // at most one fold. (Known gap: a triangle of two border clusters and an interior
        // one can come out of stage one facing down, and the tail cannot always take it out,
        // because every collapse that would is degenerate on the flat border. It shows as
        // twice its own small area here.)
        let b = r.mesh.bounds();
        assert_eq!(
            (b.min, b.max),
            (glam::Vec3::ZERO, glam::Vec3::new(n as f32, n as f32, 0.0))
        );
        let area = r.mesh.surface_area();
        assert!(
            (area - (n * n) as f32).abs() <= 0.005 * (n * n) as f32,
            "outline moved: area {area}"
        );
        assert!(r.mesh.triangle_count() <= 64, "{}", r.mesh.triangle_count());
    }

    #[test]
    fn turned_down_edge_keeps_its_strip() {
        // A flat sheet whose outline turns down into a narrow strip (glass modeled with its
        // thickness): one cluster per border cell would press the strip into the sheet.
        let (n, w) = (40u32, 0.2f32);
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
        // The outline counterclockwise from above, and a copy of it lowered by w.
        let mut ring: Vec<u32> = (0..n).collect();
        ring.extend((0..n).map(|y| y * (n + 1) + n));
        ring.extend((1..=n).rev().map(|x| n * (n + 1) + x));
        ring.extend((1..=n).rev().map(|y| y * (n + 1)));
        let low: Vec<u32> = ring
            .iter()
            .map(|&v| {
                pos.push(pos[v as usize] - glam::Vec3::Z * w);
                pos.len() as u32 - 1
            })
            .collect();
        for k in 0..ring.len() {
            let j = (k + 1) % ring.len();
            let (a, b, c, d) = (ring[k], ring[j], low[j], low[k]);
            idx.extend_from_slice(&[a, d, c, a, c, b]);
        }
        let mut m = Mesh::new(pos, idx);
        m.compute_normals(60.0);
        let strip = |m: &Mesh| -> f32 {
            m.indices
                .chunks(3)
                .map(|t| {
                    let p = [0, 1, 2].map(|k| m.positions[t[k] as usize]);
                    let c = (p[1] - p[0]).cross(p[2] - p[0]);
                    if c.z.abs() < 0.5 * c.length() {
                        c.length() * 0.5
                    } else {
                        0.0
                    }
                })
                .sum()
        };
        let r = simplify(&m, &opts(0.03)).unwrap();
        r.mesh.validate().unwrap();
        let (before, after) = (strip(&m), strip(&r.mesh));
        assert!(after > 0.8 * before, "strip area {before} -> {after}");
    }
}
