//! Half-edge collapse simplification (v2).
//!
//! * Positions are welded internally; each input vertex is a *wedge* of its position.
//!   Collapsing position P onto neighbour Q remaps every wedge of P to a wedge of Q found
//!   through the triangles P and Q share. In [`SeamMode::Strict`] a wedge without a partner
//!   rejects the collapse (seams never move off their line); in [`SeamMode::Permissive`] it maps
//!   to Q's wedge with the cheapest attribute error, and the attribute quadrics price the
//!   distortion.
//! * Geometric error: area-weighted plane quadrics per position (Garland & Heckbert 1997) plus
//!   border constraint planes.
//! * Attribute error: accumulated attribute quadrics per wedge (after Hoppe 1999, in the
//!   compact form that folds every attribute's quadratic term into one shared quadric plus a
//!   4-float gradient per attribute). Normals, UV sets and colors are attributes; skin weights
//!   and morph deltas are priced by deviation at the removed vertex.
//! * Collapses are half-edge: Q keeps its position and attributes, so no interpolation is
//!   needed and every removed vertex has a surviving geomorph target.
//! * Optional mirror symmetry: collapses are mirrored across a detected plane.

use crate::reduce::kernels;
use crate::reduce::quadric::Quadric;
use crate::reduce::symmetry::{mirror_map, Symmetry};
use crate::{Error, Mesh, Result, Scene};
use glam::DVec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

/// When to stop if several targets are set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopWhen {
    /// Stop as soon as any target is met.
    #[default]
    Any,
    /// Keep going until every target is met.
    All,
}

/// How collapses are ordered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollapseOrder {
    /// Rounds of non-conflicting collapses: candidates are ranked in parallel, then the
    /// cheapest are executed in cost order (each re-checked first). No priority queue.
    #[default]
    Batched,
    /// One collapse at a time from an exact priority queue (the classic greedy order).
    Greedy,
}

/// How UV seams and hard edges constrain collapses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeamMode {
    /// Seams may be collapsed across when the attribute-aware cost says it is the cheapest
    /// collapse (default): reaches any triangle target and measured best overall.
    #[default]
    Permissive,
    /// Seam vertices only slide along their seam: UV islands keep their exact boundaries, but
    /// reduction can stop short of a triangle target.
    Strict,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimplifyOptions {
    /// Fraction of the input triangle count to keep (0..1).
    pub target_ratio: Option<f32>,
    /// Absolute triangle count to reach.
    pub target_triangles: Option<usize>,
    /// Max estimated geometric deviation in world units.
    pub max_error: Option<f32>,
    pub stop: StopWhen,
    pub seams: SeamMode,
    pub symmetry: Symmetry,
    /// Never move open-border vertices.
    pub lock_border: bool,
    /// Attribute weights (relative to geometric error in unit-diagonal space).
    pub normal_weight: f32,
    pub uv_weight: f32,
    pub color_weight: f32,
    pub skin_weight: f32,
    pub morph_weight: f32,
    /// Weight of the constraint planes along open borders.
    pub border_weight: f32,
    /// How collapse error is normalized by the surface area a vertex represents: 0 ranks by
    /// area-weighted error (the default; best mean, UV and normal fidelity), 1 by mean distance
    /// (classic QEM; protects tiny features at the cost of overall fidelity).
    pub error_normalization: f32,
    /// Meshes with at least this many triangles are reduced cluster-parallel: fixed-size spatial
    /// clusters are reduced concurrently with their borders locked, then one exact pass cleans
    /// up the borders. The cluster layout depends only on the mesh, so output is identical on
    /// any number of threads. 0 disables.
    pub cluster_threshold: usize,
    /// Collapse ordering (batched rounds or exact greedy).
    pub collapse_order: CollapseOrder,
    /// Reject collapses that rotate a face normal so that cos(angle) drops below this.
    pub min_face_dot: f32,
    /// Per-vertex importance (≥ 0, default 1). Collapse cost is multiplied by it.
    #[serde(skip)]
    pub vertex_importance: Option<Vec<f32>>,
    /// Per-vertex lock flags (modular seams, user locks).
    #[serde(skip)]
    pub locked_vertices: Option<Vec<bool>>,
    /// Delete whole connected parts (batched order, with a triangle target): a part goes once
    /// removing it moves the surface (to same-facing geometry) less than a quarter of the
    /// deviation the collapses already accepted, and, if collapses stall above the target
    /// (closed parts at their minimal triangulation, small locked open parts), the cheapest
    /// parts go until it is met. Never exceeds `max_error`, never removes a part with a locked
    /// vertex or holding half the remaining triangles. Off with symmetry. Removed vertices'
    /// `vertex_remap` points at the nearest remaining vertex.
    #[serde(default = "default_true")]
    pub remove_parts: bool,
    /// Extra weight on geometric error near likely silhouettes: each face's plane quadric is
    /// scaled by 1 + w × (mean over its edges of dihedral angle / π, 1 on open borders), the
    /// fraction of view directions for which an edge is on the silhouette. 0 disables; the
    /// default 0.5 scored best in rendered comparisons (higher keeps improving geometry only).
    #[serde(default = "default_silhouette_weight")]
    pub silhouette_weight: f32,
    /// [`Quality::Best`] (default) or [`Quality::Fast`].
    #[serde(default)]
    pub quality: Quality,
    /// Material ids that don't hide what is behind them (glass, visors, water). Part removal
    /// weighs a part's removal error by how much of it can be seen from outside; triangles with
    /// these materials let those visibility rays through, so a watch movement under its
    /// crystal still counts as visible.
    #[serde(default)]
    pub transparent_materials: Vec<u32>,
    /// Visible-part protection. When collapses stall above the triangle target, the cheapest
    /// part is deleted to go on, but never one whose visible removal error exceeds the larger
    /// of this fraction of the bounding-box diagonal and the usual removal threshold (a quarter
    /// of the deviation collapses have accepted): the reduction stops above the target instead
    /// of deleting a prominent part. 0 disables (stalls delete parts of any error).
    #[serde(default = "default_part_protection")]
    pub part_protection: f32,
}

fn default_part_protection() -> f32 {
    0.05
}

/// Speed/quality trade-off of [`simplify`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quality {
    /// The full reducer: attribute quadrics with gradients, part removal, silhouette weight,
    /// batched rounds with exact re-ranking, parallel and cluster-parallel.
    #[default]
    Best,
    /// A lean single-threaded loop in passes (f32 quadrics, wedge-distance attribute cost),
    /// several times faster. Uses the target, error, weight, border and lock options; ignores
    /// symmetry, part removal, silhouette weight, importance and skinning/morph weights.
    Fast,
}

fn default_silhouette_weight() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            target_ratio: None,
            target_triangles: None,
            max_error: None,
            stop: StopWhen::Any,
            seams: SeamMode::Permissive,
            symmetry: Symmetry::Off,
            lock_border: false,
            // Normals deviate at first order on real assets (baked/smoothed normals), so a tiny
            // weight is enough to stop shading drift without fighting the geometric error.
            normal_weight: 5e-5,
            uv_weight: 0.5,
            color_weight: 0.5,
            // Joint-weight differences are O(1) where geometric errors are O(1e-3): keep small.
            skin_weight: 0.003,
            morph_weight: 1.0,
            border_weight: 10.0,
            error_normalization: 0.0,
            cluster_threshold: 120_000,
            collapse_order: CollapseOrder::Batched,
            min_face_dot: 0.2,
            vertex_importance: None,
            locked_vertices: None,
            remove_parts: true,
            transparent_materials: Vec::new(),
            part_protection: default_part_protection(),
            silhouette_weight: 0.5,
            quality: Quality::Best,
        }
    }
}

impl SimplifyOptions {
    pub fn ratio(r: f32) -> Self {
        SimplifyOptions {
            target_ratio: Some(r),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimplifyResult {
    pub mesh: Mesh,
    /// Largest estimated deviation of any collapse performed, in world units.
    pub error: f32,
    /// `error` divided by the input bounding-box diagonal.
    pub relative_error: f32,
    /// For each input vertex, the output vertex it collapsed into (geomorph target).
    pub vertex_remap: Vec<u32>,
    pub collapses: usize,
    /// The symmetry plane used, if any (n.xyz, d with n·p + d = 0).
    pub symmetry_plane: Option<glam::Vec4>,
    /// For each output vertex, the input vertex it is an exact copy of (half-edge collapse),
    /// so callers can carry custom per-vertex attributes across.
    pub source_vertex: Vec<u32>,
}

/// Converts an on-screen size target into a max deviation: an object whose bounding box
/// diagonal spans `screen_size_px` pixels may deviate by `pixel_tolerance` pixels.
pub fn screen_size_to_max_error(diagonal: f32, screen_size_px: f32, pixel_tolerance: f32) -> f32 {
    diagonal / screen_size_px.max(1.0) * pixel_tolerance
}

/// Positions with more wedges than this are never collapse sources in permissive mode.
const MAX_SOURCE_WEDGES: usize = 8;

/// Least UV jump (texture units) a seam tear is priced at (see `State::tear_error`).
const TEAR_UV: f64 = 1.0;
/// Wedges whose UVs are closer than this have the same UV (a normal-only seam).
const TEAR_SAME_UV: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Validity {
    Ok,
    Locked,
    Border,
    Mirror,
    Link,
    Degenerate,
    Flip,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    cost: f64,
    geom: f64,
    q: u32,
    /// Validity was checked when this candidate was ranked.
    checked: bool,
}

/// Indexed binary min-heap: one entry per position, keys updated in place (no stale entries).
struct IndexedHeap {
    heap: Vec<u32>,
    slot: Vec<u32>,
    key: Vec<f64>,
}

impl IndexedHeap {
    fn new(n: usize) -> Self {
        IndexedHeap {
            heap: Vec::with_capacity(n),
            slot: vec![u32::MAX; n],
            key: vec![f64::INFINITY; n],
        }
    }
    #[inline(always)]
    fn less(&self, a: u32, b: u32) -> bool {
        let (ka, kb) = (self.key[a as usize], self.key[b as usize]);
        ka < kb || (ka == kb && a < b)
    }
    #[inline(always)]
    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.slot[self.heap[i] as usize] = i as u32;
        self.slot[self.heap[j] as usize] = j as u32;
    }
    #[inline(always)]
    fn up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.less(self.heap[i], self.heap[p]) {
                self.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }
    #[inline(always)]
    fn down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < n && self.less(self.heap[l], self.heap[m]) {
                m = l;
            }
            if r < n && self.less(self.heap[r], self.heap[m]) {
                m = r;
            }
            if m == i {
                break;
            }
            self.swap(i, m);
            i = m;
        }
    }
    #[inline(always)]
    fn set(&mut self, x: u32, k: f64) {
        self.key[x as usize] = k;
        match self.slot[x as usize] {
            u32::MAX => {
                self.heap.push(x);
                let i = self.heap.len() - 1;
                self.slot[x as usize] = i as u32;
                self.up(i);
            }
            i => {
                self.up(i as usize);
                let i = self.slot[x as usize] as usize;
                self.down(i);
            }
        }
    }
    #[inline(always)]
    fn remove(&mut self, x: u32) {
        let i = self.slot[x as usize];
        if i == u32::MAX {
            return;
        }
        let i = i as usize;
        let last = self.heap.len() - 1;
        if i != last {
            self.swap(i, last);
        }
        self.heap.pop();
        self.slot[x as usize] = u32::MAX;
        if i < self.heap.len() {
            let moved = self.heap[i];
            self.up(i);
            let j = self.slot[moved as usize] as usize;
            self.down(j);
        }
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<u32> {
        let top = *self.heap.first()?;
        self.remove(top);
        Some(top)
    }
}

#[derive(Default)]
struct Scratch {
    mark: Vec<u32>,
    epoch: u32,
    nbr_a: Vec<u32>,
    nbr_b: Vec<u32>,
    map: Vec<(u32, u32)>,
    cands: Vec<(f64, f64, u32)>,
    corners: Vec<Corner>,
    map2: Vec<(u32, u32)>,
    corners2: Vec<Corner>,
}

/// Per-position triangle lists in one arena: contiguous, no per-vertex allocations. A list that
/// outgrows its slot moves to the end of the arena with doubled capacity; only alive triangles
/// are kept, in a stable order (iteration order decides ties, so it must not change).
struct Adjacency {
    start: Vec<u32>,
    len: Vec<u32>,
    cap: Vec<u32>,
    data: Vec<u32>,
}

impl Adjacency {
    fn new(lists: &[u32], counts: &[u32]) -> Adjacency {
        let n = counts.len();
        let (mut start, mut cap) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut total = 0u32;
        for &c in counts {
            start.push(total);
            let k = c + c / 2 + 2;
            cap.push(k);
            total += k;
        }
        let mut data = vec![u32::MAX; total as usize];
        let mut len = vec![0u32; n];
        // `lists` is (position, triangle) pairs in insertion order.
        for pair in lists.chunks_exact(2) {
            let (p, t) = (pair[0] as usize, pair[1]);
            data[(start[p] + len[p]) as usize] = t;
            len[p] += 1;
        }
        Adjacency {
            start,
            len,
            cap,
            data,
        }
    }
    #[inline]
    fn get(&self, p: u32) -> &[u32] {
        let (s, l) = (
            self.start[p as usize] as usize,
            self.len[p as usize] as usize,
        );
        &self.data[s..s + l]
    }
    fn push(&mut self, p: u32, t: u32) {
        let p = p as usize;
        if self.len[p] == self.cap[p] {
            let new_cap = (self.cap[p] * 2).max(8);
            let (s, l) = (self.start[p] as usize, self.len[p] as usize);
            let at = self.data.len();
            self.data.extend_from_within(s..s + l);
            self.data.resize(at + new_cap as usize, u32::MAX);
            self.start[p] = at as u32;
            self.cap[p] = new_cap;
        }
        let i = (self.start[p] + self.len[p]) as usize;
        self.data[i] = t;
        self.len[p] += 1;
    }
    /// Order-preserving removal of `t` from p's list.
    fn remove(&mut self, p: u32, t: u32) {
        let (s, l) = (
            self.start[p as usize] as usize,
            self.len[p as usize] as usize,
        );
        let list = &mut self.data[s..s + l];
        if let Some(i) = list.iter().position(|&x| x == t) {
            list.copy_within(i + 1.., i);
            self.len[p as usize] -= 1;
        }
    }
    fn clear(&mut self, p: u32) {
        self.len[p as usize] = 0;
    }
}

/// The writes of one collapse (see `State::plan_collapse`).
struct CollapsePlan {
    p: u32,
    q: u32,
    /// Wedge renames (p's wedge → q's wedge).
    map: Vec<(u32, u32)>,
    /// q's wedges whose attribute rows change, and the new rows, flat.
    row_ids: Vec<u32>,
    rows: Vec<f64>,
    /// Triangles that die (they contain both p and q).
    die: Vec<u32>,
    /// Triangles that move to q: (triangle, new wedges, new positions).
    moved: Vec<(u32, [u32; 3], [u32; 3])>,
    /// q's quadric after absorbing p's.
    quadric: Quadric,
}

/// One of p's corners: p's wedge, the other two wedges, and their positions.
#[derive(Clone, Copy)]
struct Corner {
    pw: u32,
    a: u32,
    b: u32,
    pa: u32,
    pb: u32,
}

/// Attribute layout: which mesh channels feed the attribute quadrics, with their scale.
struct AttrLayout {
    dim: usize,
    /// The first `nd` dims are the normal (scaled by region size in the cost).
    nd: usize,
    /// Dims `nd..nd + nuv` are the UV sets.
    nuv: usize,
    /// The UV weight (values hold UVs · √weight).
    uv_weight: f64,
    /// Per wedge, `dim` weighted attribute values.
    values: Vec<f32>,
}

impl AttrLayout {
    fn new(mesh: &Mesh, o: &SimplifyOptions) -> AttrLayout {
        let mut dim = 0;
        let nw = (o.normal_weight.max(0.0)).sqrt() * 0.5;
        let uw = o.uv_weight.max(0.0).sqrt();
        let cw = (o.color_weight.max(0.0)).sqrt() * 0.5;
        if mesh.normals.is_some() && nw > 0.0 {
            dim += 3;
        }
        let nd = dim;
        if uw > 0.0 {
            dim += 2 * mesh.uvs.len();
        }
        let nuv = dim - nd;
        if mesh.colors.is_some() && cw > 0.0 {
            dim += 4;
        }
        let n = mesh.vertex_count();
        let mut values = Vec::with_capacity(n * dim);
        for v in 0..n {
            if let (Some(ns), true) = (&mesh.normals, nw > 0.0) {
                values.extend((ns[v] * nw).to_array());
            }
            if uw > 0.0 {
                for set in &mesh.uvs {
                    values.extend((set[v] * uw).to_array());
                }
            }
            if let (Some(cs), true) = (&mesh.colors, cw > 0.0) {
                values.extend((cs[v] * cw).to_array());
            }
        }
        AttrLayout {
            dim,
            nd,
            nuv,
            uv_weight: (uw * uw) as f64,
            values,
        }
    }
    #[inline]
    fn get(&self, w: u32) -> &[f32] {
        &self.values[w as usize * self.dim..(w as usize + 1) * self.dim]
    }
}

struct State<'a> {
    /// Whole parts that may be deleted (see [`SimplifyOptions::remove_parts`]).
    parts: Option<crate::reduce::parts::Parts>,
    /// Squared (unit-frame) deviation already accepted before this pass.
    prior_geom2: f64,
    /// Largest removal error (world units) of a part deleted in this pass.
    part_error: f32,
    mesh: &'a Mesh,
    opts: &'a SimplifyOptions,
    pos: Vec<DVec3>,
    vpos: Vec<u32>,
    tris: Vec<[u32; 3]>,
    /// Position ids per triangle (kept in step with `tris`; saves a `vpos` lookup per corner).
    tpos: Vec<[u32; 3]>,
    tri_alive: Vec<bool>,
    adj: Adjacency,
    pos_alive: Vec<bool>,
    locked: Vec<bool>,
    border: Vec<bool>,
    /// Border neighbours per position (open-border edges).
    bnbr: Vec<Vec<u32>>,
    quadric: Vec<Quadric>,
    attrs: AttrLayout,
    /// Per wedge: shared attribute quadric (10) + per-attribute gradient (4 × dim) + weight (1).
    aq: Vec<f64>,
    aq_stride: usize,
    importance: Vec<f32>,
    /// Current best collapse per position (geometric error², target).
    best: Vec<(f64, u32)>,
    vremap: Vec<u32>,
    premap: Vec<u32>,
    /// An input vertex at each position (for the output position of slid wedges).
    pos_src: Vec<u32>,
    /// Wedges that slid onto another position, keeping their attributes (see
    /// [`State::map_from_corners`]); their output position is that of their final position.
    slid: Vec<u32>,
    inv_scale: f64,
    mirror: Option<Vec<u32>>,
    avg_area: f64,
    alpha: f64,
    /// Heap key is a lower bound only; re-rank before collapsing.
    dirty: Vec<bool>,
    strict: bool,
    /// Set once collapses stall: vertices with many wedges become collapse sources too.
    relaxed: bool,
    /// Per-thread scratch buffers (marker array sized for this mesh), so `State` can be shared
    /// by parallel evaluations.
    scratch: thread_local::ThreadLocal<RefCell<Scratch>>,
}

#[inline]
fn ekey(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

impl<'a> State<'a> {
    /// This thread's scratch buffers (marker array sized for this mesh). Per-thread scratch
    /// keeps `State` shareable, so independent evaluations can run in parallel.
    #[inline]
    fn scratch(&self) -> &RefCell<Scratch> {
        self.scratch.get_or(|| {
            RefCell::new(Scratch {
                mark: vec![0; self.pos.len()],
                ..Default::default()
            })
        })
    }

    /// `frame` overrides the normalization frame (min corner, diagonal) so that pieces of one
    /// mesh (clusters) weigh geometry against attributes exactly like the whole mesh would.
    fn new(mesh: &'a Mesh, opts: &'a SimplifyOptions, frame: Option<(DVec3, f64)>) -> State<'a> {
        let bounds = mesh.bounds();
        let (min, diag) =
            frame.unwrap_or((bounds.min.as_dvec3(), bounds.diagonal().max(1e-12) as f64));
        // Weld by exact position with a parallel sort (ids in first-occurrence order).
        let key = |i: usize| {
            let p = mesh.positions[i];
            [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
        };
        let mut order: Vec<([u32; 3], u32)> = (0..mesh.vertex_count())
            .map(|i| (key(i), i as u32))
            .collect();
        order.par_sort_unstable();
        let mut group_of = vec![0u32; mesh.vertex_count()];
        let mut first: Vec<u32> = Vec::new();
        for (k, &(ki, i)) in order.iter().enumerate() {
            if k == 0 || order[k - 1].0 != ki {
                first.push(i); // smallest index of the group (sorted by (key, i))
            }
            group_of[i as usize] = first.len() as u32 - 1;
        }
        let mut rank: Vec<u32> = (0..first.len() as u32).collect();
        rank.sort_unstable_by_key(|&g| first[g as usize]);
        let mut gid = vec![0u32; first.len()];
        for (new, &g) in rank.iter().enumerate() {
            gid[g as usize] = new as u32;
        }
        let pos: Vec<DVec3> = rank
            .iter()
            .map(|&g| (mesh.positions[first[g as usize] as usize].as_dvec3() - min) / diag)
            .collect();
        let vpos: Vec<u32> = group_of.iter().map(|&g| gid[g as usize]).collect();
        let pos_src: Vec<u32> = rank.iter().map(|&g| first[g as usize]).collect();
        let np = pos.len();
        let tris: Vec<[u32; 3]> = (0..mesh.triangle_count())
            .map(|t| mesh.triangle(t))
            .collect();
        let mut tri_alive = vec![true; tris.len()];
        let mut pairs: Vec<u32> = Vec::with_capacity(tris.len() * 6);
        let mut counts = vec![0u32; np];
        let tpos: Vec<[u32; 3]> = tris
            .iter()
            .map(|tri| tri.map(|v| vpos[v as usize]))
            .collect();
        // Faces over the same three positions (double-sided geometry, duplicated shells) count
        // once when classifying edges, so twins don't make every shared edge a junction.
        let twin = {
            let mut key: Vec<([u32; 3], u32)> = tpos
                .iter()
                .enumerate()
                .map(|(t, p)| {
                    let mut k = *p;
                    k.sort_unstable();
                    (k, t as u32)
                })
                .collect();
            key.par_sort_unstable();
            let mut twin = vec![false; tris.len()];
            for w in key.windows(2) {
                if w[0].0 == w[1].0 {
                    twin[w[1].1 as usize] = true;
                }
            }
            twin
        };
        let mut edges: Vec<u64> = Vec::with_capacity(tris.len() * 3);
        for (t, p) in tpos.iter().enumerate() {
            if p[0] == p[1] || p[1] == p[2] || p[0] == p[2] {
                tri_alive[t] = false;
                continue;
            }
            for i in 0..3 {
                pairs.extend_from_slice(&[p[i], t as u32]);
                counts[p[i] as usize] += 1;
                if !twin[t] {
                    let (a, b) = ekey(p[i], p[(i + 1) % 3]);
                    edges.push(((a as u64) << 32) | b as u64);
                }
            }
        }
        edges.par_sort_unstable();
        let mut locked = vec![false; np];
        let mut border = vec![false; np];
        let mut bnbr: Vec<Vec<u32>> = vec![Vec::new(); np];
        let mut border_degree = vec![0u32; np];
        // Non-manifold junctions (edges shared by three or more triangles): neighbours along them.
        let mut jnbr: Vec<Vec<u32>> = vec![Vec::new(); np];
        let mut k = 0;
        while k < edges.len() {
            let mut j = k + 1;
            while j < edges.len() && edges[j] == edges[k] {
                j += 1;
            }
            let (a, b) = ((edges[k] >> 32) as u32, edges[k] as u32);
            let c = j - k;
            if c == 1 {
                bnbr[a as usize].push(b);
                bnbr[b as usize].push(a);
                border[a as usize] = true;
                border[b as usize] = true;
                border_degree[a as usize] += 1;
                border_degree[b as usize] += 1;
            } else if c > 2 {
                jnbr[a as usize].push(b);
                jnbr[b as usize].push(a);
            }
            k = j;
        }

        for p in 0..np {
            if border_degree[p] > 2 || (opts.lock_border && border[p]) {
                locked[p] = true;
            }
            // A junction vertex (not on an open border) may only collapse along the junction,
            // like a border vertex along its border: inside a junction curve it slides; at an end
            // or a branch point it merges into a junction neighbour, which shortens or merges
            // junction curves (the link condition still guards topology). On a border too: put.
            if !jnbr[p].is_empty() && border[p] {
                locked[p] = true;
            }
        }
        if let Some(l) = &opts.locked_vertices {
            for (v, &is) in l.iter().enumerate() {
                if is && v < vpos.len() {
                    locked[vpos[v] as usize] = true;
                }
            }
        }
        let mut importance = vec![1.0f32; np];
        if let Some(imp) = &opts.vertex_importance {
            let mut seen = vec![false; np];
            for (v, &w) in imp.iter().enumerate().take(vpos.len()) {
                let p = vpos[v] as usize;
                importance[p] = if seen[p] { importance[p].max(w) } else { w };
                seen[p] = true;
            }
        }
        let attrs = AttrLayout::new(mesh, opts);
        let dim = attrs.dim;
        // Per wedge: normal-group quadric (12, padded), other-group quadric (12), gradients
        // (4·dim), weight (+3 pad): every block starts on a 4-lane boundary.
        let aq_stride = 24 + 4 * dim + 4;
        let nd = attrs.nd;
        let mut aq = vec![
            0.0f64;
            if dim > 0 {
                mesh.vertex_count() * aq_stride
            } else {
                0
            }
        ];
        let mut quadric = vec![Quadric::default(); np];
        // Expected-silhouette weight per face: an edge is on the silhouette for a fraction of
        // view directions equal to its dihedral angle over π (1 on open borders), and moving a
        // silhouette moves the image most. Each face's plane quadric is scaled by
        // 1 + silhouette_weight × (mean of that fraction over its edges).
        let sil_beta = opts.silhouette_weight as f64;
        let adj = Adjacency::new(&pairs, &counts);
        let sil: Vec<f64> = if sil_beta > 0.0 {
            let fnorm: Vec<DVec3> = tpos
                .par_iter()
                .with_min_len(1024)
                .map(|p| {
                    let p = p.map(|v| pos[v as usize]);
                    (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero()
                })
                .collect();
            // Faces across each edge, found through the position-to-triangle lists.
            let (adj, tpos, tri_alive) = (&adj, &tpos, &tri_alive);
            (0..tris.len())
                .into_par_iter()
                .with_min_len(1024)
                .map(|t| {
                    if !tri_alive[t] {
                        return 1.0;
                    }
                    let p = tpos[t];
                    let mut acc = 0.0;
                    for i in 0..3 {
                        let (a, b) = (p[i], p[(i + 1) % 3]);
                        let (mut n, mut other) = (0usize, 0u32);
                        for &g in adj.get(a) {
                            if tpos[g as usize].contains(&b) {
                                n += 1;
                                if g as usize != t {
                                    other = g;
                                }
                            }
                        }
                        acc += if n == 2 {
                            fnorm[t].dot(fnorm[other as usize]).clamp(-1.0, 1.0).acos()
                                / std::f64::consts::PI
                        } else {
                            1.0
                        };
                    }
                    1.0 + sil_beta * acc / 3.0
                })
                .collect()
        } else {
            Vec::new()
        };
        struct Face {
            p: [u32; 3],
            plane: Quadric,
            border: [Option<Quadric>; 3],
            /// Normal- and other-group attribute quadrics; the gradients go to the chunk's
            /// gradient buffer (`dim` per face).
            attr: Option<([f64; 12], [f64; 12])>,
            area: f64,
        }
        let face = |t: usize, grads: &mut [(DVec3, f64)]| -> Option<Face> {
            if !tri_alive[t] {
                return None;
            }
            let tri = &tris[t];
            let p = tri.map(|v| vpos[v as usize]);
            let (a, b, c) = (pos[p[0] as usize], pos[p[1] as usize], pos[p[2] as usize]);
            let e1 = b - a;
            let e2 = c - a;
            let n = e1.cross(e2);
            let area = n.length() * 0.5;
            if area <= 1e-30 {
                return None;
            }
            let nn = n / (2.0 * area);
            let plane = Quadric::plane(nn, -nn.dot(a), area * sil.get(t).copied().unwrap_or(1.0));
            let mut border = [None; 3];
            for (i, slot) in border.iter_mut().enumerate() {
                let (e0, e1v) = (p[i], p[(i + 1) % 3]);
                if bnbr[e0 as usize].contains(&e1v) {
                    let (x0, x1) = (pos[e0 as usize], pos[e1v as usize]);
                    let dir = x1 - x0;
                    let bn = dir.cross(nn).normalize_or_zero();
                    if bn != DVec3::ZERO {
                        *slot = Some(Quadric::constraint(
                            bn,
                            -bn.dot(x0),
                            dir.length_squared() * opts.border_weight as f64,
                        ));
                    }
                }
            }
            let mut attr = None;
            if dim > 0 {
                // Attribute gradients over the face: s(p) = g·p + d with g in the face plane.
                // Rows [e1; e2; n] · g = [Δ1, Δ2, 0].
                let m = glam::DMat3::from_cols(e1, e2, nn).transpose();
                if m.determinant().abs() >= 1e-30 {
                    let inv = m.inverse();
                    let w = tri.map(|v| attrs.get(v));
                    let (mut sqn, mut sqo) = (Quadric::default(), Quadric::default());
                    for j in 0..dim {
                        let (s0, s1, s2) = (w[0][j] as f64, w[1][j] as f64, w[2][j] as f64);
                        let g = inv * DVec3::new(s1 - s0, s2 - s0, 0.0);
                        let d = s0 - g.dot(a);
                        if j < nd { &mut sqn } else { &mut sqo }.add(&Quadric::plane(g, d, area));
                        grads[j] = (g, d);
                    }
                    attr = Some((sqn.coeffs(), sqo.coeffs()));
                }
            }
            Some(Face {
                p,
                plane,
                border,
                attr,
                area,
            })
        };
        // Triangles of each wedge, ascending (the order the sums are taken in).
        let (wstart, wtris) = if dim > 0 {
            let nv = mesh.vertex_count();
            let mut wstart = vec![0u32; nv + 1];
            for (t, tri) in tris.iter().enumerate() {
                if tri_alive[t] {
                    for &w in tri {
                        wstart[w as usize + 1] += 1;
                    }
                }
            }
            for i in 0..nv {
                wstart[i + 1] += wstart[i];
            }
            let mut fill = wstart.clone();
            let mut wtris = vec![0u32; wstart[nv] as usize];
            for (t, tri) in tris.iter().enumerate() {
                if tri_alive[t] {
                    for &w in tri {
                        wtris[fill[w as usize] as usize] = t as u32;
                        fill[w as usize] += 1;
                    }
                }
            }
            (wstart, wtris)
        } else {
            (Vec::new(), Vec::new())
        };
        // Faces are computed in parallel a chunk at a time; then every position and wedge adds
        // the chunk's faces it belongs to, in triangle order, also in parallel. Each sum is
        // taken in the same order as a serial scatter, so results match for any thread count.
        const FACE_CHUNK: usize = 65_536;
        let mut faces: Vec<Option<Face>> = Vec::with_capacity(FACE_CHUNK.min(tris.len()));
        let gd = dim.max(1);
        let mut fgrads = vec![(DVec3::ZERO, 0.0); FACE_CHUNK.min(tris.len()) * gd];
        let mut pcur = vec![0u32; np];
        let mut wcur: Vec<u32> = if dim > 0 {
            wstart[..wstart.len() - 1].to_vec()
        } else {
            Vec::new()
        };
        for c0 in (0..tris.len()).step_by(FACE_CHUNK) {
            let c1 = (c0 + FACE_CHUNK).min(tris.len());
            faces.clear();
            faces.resize_with(c1 - c0, || None);
            faces
                .par_iter_mut()
                .zip(fgrads[..(c1 - c0) * gd].par_chunks_mut(gd))
                .enumerate()
                .with_min_len(256)
                .for_each(|(i, (f, g))| *f = face(c0 + i, g));
            let faces = &faces;
            let fgrads = &fgrads;
            let adj = &adj;
            quadric
                .par_iter_mut()
                .zip(pcur.par_iter_mut())
                .enumerate()
                .with_min_len(1024)
                .for_each(|(p, (q, cur))| {
                    let list = adj.get(p as u32);
                    while let Some(&t) = list.get(*cur as usize) {
                        if t as usize >= c1 {
                            break;
                        }
                        *cur += 1;
                        let Some(f) = &faces[t as usize - c0] else {
                            continue;
                        };
                        q.add(&f.plane);
                        for (i, bq) in f.border.iter().enumerate() {
                            if let Some(bq) = bq {
                                if f.p[i] as usize == p || f.p[(i + 1) % 3] as usize == p {
                                    q.add(bq);
                                }
                            }
                        }
                    }
                });
            if dim > 0 {
                let (wstart, wtris) = (&wstart, &wtris);
                aq.par_chunks_mut(aq_stride)
                    .zip(wcur.par_iter_mut())
                    .enumerate()
                    .with_min_len(512)
                    .for_each(|(w, (row, cur))| {
                        let end = wstart[w + 1];
                        while *cur < end {
                            let t = wtris[*cur as usize] as usize;
                            if t >= c1 {
                                break;
                            }
                            *cur += 1;
                            let Some(f) = &faces[t - c0] else { continue };
                            let Some((cn, co)) = &f.attr else {
                                continue;
                            };
                            let grads = &fgrads[(t - c0) * dim..(t - c0 + 1) * dim];
                            let area = f.area;
                            for k in 0..12 {
                                row[k] += cn[k];
                                row[12 + k] += co[k];
                            }
                            for (j, &(g, d)) in grads.iter().enumerate() {
                                let gi = 24 + 4 * j;
                                row[gi] += area * g.x;
                                row[gi + 1] += area * g.y;
                                row[gi + 2] += area * g.z;
                                row[gi + 3] += area * d;
                            }
                            row[24 + 4 * dim] += area;
                        }
                    });
            }
        }
        let avg_area =
            (quadric.iter().map(|q| q.w).sum::<f64>() / 3.0 / tris.len().max(1) as f64).max(1e-30);
        // Junction vertices slide along their junction like border vertices (after the border
        // constraint planes are built: the faces meeting at a junction already pin it).
        for p in 0..np {
            if !jnbr[p].is_empty() && !locked[p] {
                border[p] = true;
                bnbr[p] = std::mem::take(&mut jnbr[p]);
            }
        }
        let mirror = match opts.symmetry {
            Symmetry::Off => None,
            _ => crate::reduce::symmetry::resolve_plane(mesh, opts.symmetry)
                .map(|plane| mirror_map(&mesh.positions, &vpos, np, plane, bounds.diagonal())),
        };
        State {
            parts: None,
            prior_geom2: 0.0,
            part_error: 0.0,
            mesh,
            opts,
            pos,
            vpos,
            tris,
            tpos,
            tri_alive,
            adj,
            pos_alive: vec![true; np],
            locked,
            border,
            bnbr,
            quadric,
            attrs,
            aq,
            aq_stride,
            importance,
            best: vec![(0.0, u32::MAX); np],
            vremap: (0..mesh.vertex_count() as u32).collect(),
            pos_src,
            slid: Vec::new(),
            premap: (0..np as u32).collect(),
            inv_scale: diag,
            mirror,
            avg_area,
            dirty: vec![false; np],
            strict: opts.seams == SeamMode::Strict,
            relaxed: false,
            scratch: thread_local::ThreadLocal::new(),
            alpha: opts.error_normalization.clamp(0.0, 1.0) as f64,
        }
    }

    #[inline(always)]
    fn neighbors_into(&self, p: u32, out: &mut Vec<u32>, mark: &mut [u32], epoch: u32) {
        out.clear();
        for &t in self.adj.get(p) {
            for &q in &self.tpos[t as usize] {
                if q != p && mark[q as usize] != epoch {
                    mark[q as usize] = epoch;
                    out.push(q);
                }
            }
        }
    }

    /// Positions adjacent to p (this thread's marker array).
    fn ring(&self, p: u32, out: &mut Vec<u32>) {
        let mut s = self.scratch().borrow_mut();
        let s = &mut *s;
        s.epoch = s.epoch.wrapping_add(1);
        let e = s.epoch;
        self.neighbors_into(p, out, &mut s.mark, e);
    }

    #[cfg(test)]
    fn neighbors(&self, p: u32) -> Vec<u32> {
        let mut s = self.scratch().borrow_mut();
        s.epoch = s.epoch.wrapping_add(1);
        let epoch = s.epoch;
        let mut out = Vec::new();
        self.neighbors_into(p, &mut out, &mut s.mark, epoch);
        out
    }

    /// Attribute quadric error of wedge `pw` evaluated at position `q` with attributes of wedge
    /// `qw`. Returns (normal-group error, other error, weight).
    #[cfg(test)]
    fn attr_eval(&self, pw: u32, q: DVec3, qw: u32) -> (f64, f64, f64) {
        self.attr_eval_m(pw, &kernels::monomials(q), &kernels::homogeneous(q), qw)
    }

    /// [`Self::attr_eval`] with the point's monomials and homogeneous coordinates precomputed
    /// (shared by every wedge and the geometric quadric of one candidate).
    #[inline(always)]
    fn attr_eval_m(&self, pw: u32, m: &[f64; 12], h: &[f64; 4], qw: u32) -> (f64, f64, f64) {
        let dim = self.attrs.dim;
        if dim == 0 {
            return (0.0, 0.0, 0.0);
        }
        let nd = self.attrs.nd;
        let base = pw as usize * self.aq_stride;
        let a = &self.aq[base..base + 28 + 4 * dim];
        let (quads, rest) = a.split_at(24);
        let (grads, tail) = rest.split_at(4 * dim);
        let w = tail[0];
        let s = &self.attrs.values[qw as usize * dim..qw as usize * dim + dim];
        let term = |g: &[f64], sj: f32| {
            let sj = sj as f64;
            -2.0 * sj * kernels::dot4(g, h) + sj * sj * w
        };
        let (gn, go) = grads.split_at(4 * nd);
        let (sn, so) = s.split_at(nd);
        let mut en = 0.0;
        if nd > 0 {
            en = kernels::dot12(&quads[..12], m);
            for (g, &sj) in gn.chunks_exact(4).zip(sn) {
                en += term(g, sj);
            }
        }
        let mut eo = 0.0;
        if nd < dim {
            eo = kernels::dot12(&quads[12..24], m);
            for (g, &sj) in go.chunks_exact(4).zip(so) {
                eo += term(g, sj);
            }
        }
        (en.max(0.0), eo.max(0.0), w)
    }

    /// P's corners in alive triangles, rotated so p's wedge comes first.
    #[inline(always)]
    fn p_corners(&self, p: u32, out: &mut Vec<Corner>) {
        out.clear();
        for &t in self.adj.get(p) {
            let [a, b, c] = self.tris[t as usize];
            let [pa, pb, pc] = self.tpos[t as usize];
            out.push(if pa == p {
                Corner {
                    pw: a,
                    a: b,
                    b: c,
                    pa: pb,
                    pb: pc,
                }
            } else if pb == p {
                Corner {
                    pw: b,
                    a: c,
                    b: a,
                    pa: pc,
                    pb: pa,
                }
            } else {
                Corner {
                    pw: c,
                    a,
                    b,
                    pa,
                    pb,
                }
            });
        }
    }

    /// Wedge mapping from precomputed corners (see [`Self::map_wedges`]). `None` if the
    /// collapse is not possible; otherwise the extra attribute error of the seams it tears
    /// (see [`Self::tear_error`]), to add in [`Self::cost`].
    fn map_from_corners(
        &self,
        corners: &[Corner],
        q: u32,
        out: &mut Vec<(u32, u32)>,
    ) -> Option<f64> {
        out.clear();
        // Fast path: p has a single wedge (no seam through it) and seams may move. The general
        // rule then maps it to q's wedge in the first triangle the two share.
        if !self.strict {
            if let Some(first) = corners.first() {
                if corners.iter().all(|c| c.pw == first.pw) {
                    for c in corners {
                        let qw = if c.pa == q {
                            c.a
                        } else if c.pb == q {
                            c.b
                        } else {
                            continue;
                        };
                        out.push((first.pw, qw));
                        return Some(0.0);
                    }
                    return None;
                }
            }
        }
        let mut shared = 0;
        for c in corners {
            let pw = c.pw;
            let qw = if c.pa == q {
                c.a
            } else if c.pb == q {
                c.b
            } else {
                continue;
            };
            shared += 1;
            match out.iter().find(|(x, _)| *x == pw) {
                Some(&(_, existing)) if existing != qw => {
                    if self.strict {
                        return None;
                    }
                }
                Some(_) => {}
                None => out.push((pw, qw)),
            }
        }
        if shared == 0 {
            return None;
        }
        let mut tear = 0.0;
        let mut mh = None;
        // Candidates for p's unpartnered wedges: every wedge of q, not only those in the
        // triangles p and q share. When both lie on a seam and the edge is a chord through one
        // island, p's wedge on the other island belongs with q's wedge there; restricting it
        // to the shared triangles' wedges dragged it across the atlas.
        let mut q_wedges: Vec<u32> = Vec::new();
        for c in corners {
            let pw = c.pw;
            if out.iter().any(|(x, _)| *x == pw) {
                continue;
            }
            if self.strict {
                return None;
            }
            let (m, h) = *mh.get_or_insert_with(|| {
                let qp = self.pos[q as usize];
                (kernels::monomials(qp), kernels::homogeneous(qp))
            });
            if q_wedges.is_empty() {
                for &t in self.adj.get(q) {
                    let tp = self.tpos[t as usize];
                    if let Some(k) = tp.iter().position(|&x| x == q) {
                        let w = self.tris[t as usize][k];
                        if !q_wedges.contains(&w) {
                            q_wedges.push(w);
                        }
                    }
                }
            }
            // Or the wedge slides onto q keeping its own attributes: its triangles stay on
            // their UV island and only stretch, where any wedge of q would move them to
            // another island (q has no wedge of pw's island, or it would be partnered).
            let (en, eo, _) = self.attr_eval_m(pw, &m, &h, pw);
            let mut best: Option<(f64, u32, f64)> = Some((en + eo, pw, 0.0));
            for &qw in &q_wedges {
                let (en, eo, w) = self.attr_eval_m(pw, &m, &h, qw);
                let t = self.tear_error(pw, qw, eo, w);
                let e = en + eo + t;
                if best.is_none_or(|b| e < b.0) {
                    best = Some((e, qw, t));
                }
            }
            match best {
                Some((_, qw, t)) => {
                    out.push((pw, qw));
                    tear += t;
                }
                None => return None,
            }
        }
        Some(tear)
    }

    /// Extra error for remapping wedge `pw` onto `qw`, a wedge of a vertex it shares no
    /// triangle with, beyond `eo` (its plain attribute error; `w` its area). If their UVs
    /// differ the remap moves `pw`'s triangles off their UV island, and what they show is then
    /// wrong however short the jump (islands are often thin strips, and a jump of a few
    /// hundredths lands in the gutter). So the UV part of the error is at least that of a
    /// [`TEAR_UV`] jump. Wedges split only by normals (same UVs) cost nothing extra.
    #[inline]
    fn tear_error(&self, pw: u32, qw: u32, eo: f64, w: f64) -> f64 {
        let a = &self.attrs;
        if a.nuv == 0 {
            return 0.0;
        }
        let (va, vb) = (a.get(pw), a.get(qw));
        let d2: f64 = (a.nd..a.nd + a.nuv)
            .map(|i| ((va[i] - vb[i]) as f64).powi(2))
            .sum();
        if d2 <= a.uv_weight * TEAR_SAME_UV * TEAR_SAME_UV {
            return 0.0;
        }
        (w * a.uv_weight * TEAR_UV * TEAR_UV - eo).max(0.0)
    }

    /// Wedge mapping for collapsing p onto q into `out`: pairs (p's wedge, q's wedge it merges
    /// into), or (w, w) for a wedge that slides onto q as it is. False if the collapse would
    /// tear a seam (strict mode) or p and q share no triangle.
    fn map_wedges(&self, p: u32, q: u32, out: &mut Vec<(u32, u32)>) -> bool {
        let mut corners = std::mem::take(&mut self.scratch().borrow_mut().corners2);
        self.p_corners(p, &mut corners);
        let ok = self.map_from_corners(&corners, q, out).is_some();
        self.scratch().borrow_mut().corners2 = corners;
        ok
    }

    fn skin_morph_cost(&self, p: u32, q: u32, map: &[(u32, u32)]) -> f64 {
        let m = self.mesh;
        if m.skin.is_none() && m.morph_targets.is_empty() {
            return 0.0;
        }
        let o = self.opts;
        let pp = self.pos[p as usize].as_vec3();
        let qp = self.pos[q as usize].as_vec3();
        let mut worst = 0.0f64;
        for &t in self.adj.get(p) {
            let tri = self.tris[t as usize];
            let tp = self.tpos[t as usize];
            if tp.contains(&q) {
                continue;
            }
            let Some(k) = tp.iter().position(|&x| x == p) else {
                continue;
            };
            let pw = tri[k];
            let nt = tri.map(|v| {
                if v == pw {
                    map.iter().find(|(a, _)| *a == v).map_or(v, |&(_, b)| b)
                } else {
                    v
                }
            });
            let pts = tp.map(|x| {
                if x == p {
                    qp
                } else {
                    self.pos[x as usize].as_vec3()
                }
            });
            let (_, w) = crate::bvh::closest_on_triangle(pp, pts[0], pts[1], pts[2]);
            let (a, tri) = (pw as usize, nt.map(|v| v as usize));
            let mut c = 0.0f64;
            if let Some(s) = &m.skin {
                let mut diff = 0.0f32;
                for k in 0..4 {
                    let (j, wa) = (s.joints[a][k], s.weights[a][k]);
                    if wa <= 0.0 {
                        continue;
                    }
                    let blended: f32 = (0..3)
                        .map(|i| {
                            (0..4)
                                .filter(|&kk| s.joints[tri[i]][kk] == j)
                                .map(|kk| s.weights[tri[i]][kk])
                                .sum::<f32>()
                                * w[i]
                        })
                        .sum();
                    diff += (wa - blended).abs();
                }
                c += o.skin_weight as f64 * (diff as f64 * 0.5).powi(2);
            }
            for mt in &m.morph_targets {
                let d = &mt.position_deltas;
                let di = d[tri[0]] * w.x + d[tri[1]] * w.y + d[tri[2]] * w.z;
                let e = (di - d[a]).length() as f64 / self.inv_scale;
                c += o.morph_weight as f64 * e * e;
            }
            worst = worst.max(c);
        }
        worst
    }

    /// Cost of collapsing p onto q with wedge map `map`: (cost, geometric error²).
    fn cost(&self, p: u32, q: u32, map: &[(u32, u32)], tear: f64) -> (f64, f64) {
        let qp = self.pos[q as usize];
        let (m, h) = (kernels::monomials(qp), kernels::homogeneous(qp));
        let qd = &self.quadric[p as usize];
        let raw = qd.eval_m(&m);
        // Error normalization: area^alpha. alpha = 1 is the classic mean distance, which dilutes
        // as a vertex absorbs surface; alpha = 0 keeps the area-weighted sum.
        let alpha = self.alpha;
        let norm = |w: f64| {
            if w <= 0.0 || alpha == 0.0 {
                self.avg_area
            } else if alpha == 1.0 {
                w
            } else {
                (w / self.avg_area).powf(alpha) * self.avg_area
            }
        };
        let geom2 = raw / norm(qd.w);
        let (mut en, mut eo, mut w) = (0.0, 0.0, 0.0);
        for &(pw, qw) in map {
            let (n, o, wi) = self.attr_eval_m(pw, &m, &h, qw);
            // A normal deviation over a region of area A shades like a bump of height ~θ·√A:
            // weight it by region size so normals keep mattering as triangles grow.
            en += n * (wi / self.avg_area);
            eo += o;
            w += wi;
        }
        let attr2 = if w > 0.0 {
            (en + eo + tear) / norm(w)
        } else {
            0.0
        } + self.skin_morph_cost(p, q, map);
        let imp =
            ((self.importance[p as usize] + self.importance[q as usize]) * 0.5).max(1e-3) as f64;
        // Second value: mean squared distance, for max_error and reporting.
        let mean2 = if qd.w > 0.0 { raw / qd.w } else { 0.0 };
        ((geom2 + attr2).sqrt() * imp, mean2)
    }

    /// Topological and geometric validity of collapsing p onto q.
    fn valid(&self, p: u32, q: u32) -> bool {
        self.validity(p, q) == Validity::Ok
    }

    /// Like [`State::valid`], naming the first rule a collapse breaks.
    fn validity(&self, p: u32, q: u32) -> Validity {
        if self.locked[p as usize] || !self.pos_alive[q as usize] || !self.pos_alive[p as usize] {
            return Validity::Locked;
        }
        if self.border[p as usize] && !self.bnbr[p as usize].contains(&q) {
            return Validity::Border;
        }
        if let Some(m) = &self.mirror {
            let (mp, mq) = (m[p as usize], m[q as usize]);
            if mp == p && mq != q {
                return Validity::Mirror; // vertices on the plane stay on the plane
            }
            if mp == q {
                return Validity::Mirror; // edge crosses the plane
            }
        }
        // Link condition via marker arrays.
        let mut s = self.scratch().borrow_mut();
        let s = &mut *s;
        s.epoch = s.epoch.wrapping_add(1);
        let e1 = s.epoch;
        self.neighbors_into(p, &mut s.nbr_a, &mut s.mark, e1);
        s.epoch = s.epoch.wrapping_add(1);
        let e2 = s.epoch;
        // Re-mark p's neighbours with e2 then count q's neighbours that carry e2.
        for &x in &s.nbr_a {
            s.mark[x as usize] = e2;
        }
        let mut common = 0;
        for &t in self.adj.get(q) {
            for &x in &self.tpos[t as usize] {
                if x != q && s.mark[x as usize] == e2 {
                    s.mark[x as usize] = 0; // count once
                    common += 1;
                }
            }
        }
        let shared = self
            .adj
            .get(p)
            .iter()
            .filter(|&&t| self.tpos[t as usize].contains(&q))
            .count();
        if common > shared {
            return Validity::Link;
        }
        // Flips and degeneracy.
        let qp = self.pos[q as usize];
        for &t in self.adj.get(p) {
            let ps = self.tpos[t as usize];
            if ps.contains(&q) {
                continue;
            }
            let pts = ps.map(|x| self.pos[x as usize]);
            let n0 = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
            let moved = ps.map(|x| if x == p { qp } else { self.pos[x as usize] });
            let n1 = (moved[1] - moved[0]).cross(moved[2] - moved[0]);
            let (l0, l1) = (n0.length(), n1.length());
            if l1 < 1e-14 {
                return Validity::Degenerate;
            }
            if l0 > 1e-20 && n0.dot(n1) / (l0 * l1) < self.opts.min_face_dot as f64 {
                return Validity::Flip;
            }
        }
        Validity::Ok
    }

    /// Debug (`LF_STALL`): why each live position has no collapse left.
    fn stall_report(&self, alive_tris: usize) {
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        let mut mark = vec![0u32; self.pos.len()];
        let mut nb = Vec::new();
        let mut map = Vec::new();
        for p in 0..self.pos.len() as u32 {
            if !self.pos_alive[p as usize] || self.adj.get(p).is_empty() {
                continue;
            }
            let mut corners = Vec::new();
            self.p_corners(p, &mut corners);
            let why = if self.locked[p as usize] {
                "locked".to_string()
            } else if !self.strict
                && !self.relaxed
                && Self::distinct_wedges(&corners) > MAX_SOURCE_WEDGES
            {
                "wedge-limit".to_string()
            } else {
                self.neighbors_into(p, &mut nb, &mut mark, p + 1);
                let mut reasons: Vec<String> = nb
                    .iter()
                    .map(|&q| {
                        if !self.map_wedges(p, q, &mut map) {
                            "wedges".into()
                        } else {
                            format!("{:?}", self.validity(p, q))
                        }
                    })
                    .collect();
                reasons.sort();
                reasons.dedup();
                reasons.join("+")
            };
            *counts.entry(why).or_default() += 1;
        }
        eprintln!("stall at {alive_tris} tris: {counts:?}");
    }

    /// Full evaluation (cost + validity).
    fn evaluate(&self, p: u32, q: u32) -> Option<(f64, f64)> {
        let mut map = std::mem::take(&mut self.scratch().borrow_mut().map);
        let mut corners = std::mem::take(&mut self.scratch().borrow_mut().corners2);
        self.p_corners(p, &mut corners);
        let tear = self.map_from_corners(&corners, q, &mut map);
        self.scratch().borrow_mut().corners2 = corners;
        let r = match tear {
            Some(t) if self.valid(p, q) => Some(self.cost(p, q, &map, t)),
            _ => None,
        };
        self.scratch().borrow_mut().map = map;
        r
    }

    /// Largest removal error (world units) a stalled reduction may still delete a part for
    /// (see [`SimplifyOptions::part_protection`]).
    fn stall_part_limit(&self, worst_geom2: f64) -> f32 {
        if self.opts.part_protection <= 0.0 {
            return f32::INFINITY;
        }
        (self.opts.part_protection * self.inv_scale as f32).max(self.part_threshold(worst_geom2))
    }

    /// Removal error below which a part goes as soon as it qualifies.
    fn part_threshold(&self, worst_geom2: f64) -> f32 {
        let accepted = (worst_geom2.max(self.prior_geom2)).sqrt() * self.inv_scale;
        (accepted * 0.25) as f32
    }

    /// Deletes whole parts whose removal error is at most `threshold` (world units), cheapest
    /// first, never taking the live triangle count below `floor`. Returns the triangles
    /// removed.
    fn delete_parts(
        &mut self,
        threshold: f32,
        max_parts: usize,
        alive: usize,
        floor: usize,
        cand: &mut [Option<Candidate>],
    ) -> usize {
        let Some(mut parts) = self.parts.take() else {
            return 0;
        };
        let mut removed = 0;
        for _ in 0..max_parts {
            let tri_alive = &self.tri_alive;
            let live = |ts: &[u32]| ts.iter().filter(|&&t| tri_alive[t as usize]).count();
            if alive - removed <= floor {
                break;
            }
            let Some((k, e)) =
                parts.take(threshold, self.opts.max_error, live, alive - removed, floor)
            else {
                break;
            };
            for &t in parts.triangles(k) {
                let t = t as usize;
                if self.tri_alive[t] {
                    self.tri_alive[t] = false;
                    removed += 1;
                    for p in self.tpos[t] {
                        self.pos_alive[p as usize] = false;
                        cand[p as usize] = None;
                    }
                }
            }
            self.part_error = self.part_error.max(e);
        }
        self.parts = Some(parts);
        removed
    }

    /// Cheapest valid collapse of p: costs first, validity only for the cheapest candidates.
    fn best_candidate(&self, p: u32) -> Option<Candidate> {
        if !self.pos_alive[p as usize] || self.locked[p as usize] {
            return None;
        }
        let (mut nbrs, mut cands, mut map, mut corners) = {
            let mut s = self.scratch().borrow_mut();
            (
                std::mem::take(&mut s.nbr_b),
                std::mem::take(&mut s.cands),
                std::mem::take(&mut s.map),
                std::mem::take(&mut s.corners),
            )
        };
        self.p_corners(p, &mut corners);
        // UV singularities (pole fans: one wedge per triangle) stay as collapse targets only:
        // mapping their many wedges on every re-rank is quadratic and rarely pays off.
        if !self.strict && !self.relaxed && Self::distinct_wedges(&corners) > MAX_SOURCE_WEDGES {
            let mut s = self.scratch().borrow_mut();
            s.nbr_b = nbrs;
            s.cands = cands;
            s.map = map;
            s.corners = corners;
            return None;
        }
        {
            let mut s = self.scratch().borrow_mut();
            let s = &mut *s;
            s.epoch = s.epoch.wrapping_add(1);
            let e = s.epoch;
            self.neighbors_into(p, &mut nbrs, &mut s.mark, e);
        }
        cands.clear();
        for &q in &nbrs {
            if self.border[p as usize] && !self.bnbr[p as usize].contains(&q) {
                continue;
            }
            if let Some(tear) = self.map_from_corners(&corners, q, &mut map) {
                let (c, g) = self.cost(p, q, &map, tear);
                cands.push((c, g, q));
            }
        }
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best = None;
        for &(c, g, q) in &cands {
            if self.valid(p, q) {
                best = Some(Candidate {
                    cost: c,
                    geom: g,
                    q,
                    checked: true,
                });
                break;
            }
        }
        let mut s = self.scratch().borrow_mut();
        s.nbr_b = nbrs;
        s.cands = cands;
        s.map = map;
        s.corners = corners;
        best
    }

    #[inline(always)]
    fn distinct_wedges(corners: &[Corner]) -> usize {
        let mut seen: [u32; MAX_SOURCE_WEDGES + 1] = [u32::MAX; MAX_SOURCE_WEDGES + 1];
        let mut n = 0;
        for &Corner { pw: w, .. } in corners {
            if !seen[..n].contains(&w) {
                if n == MAX_SOURCE_WEDGES + 1 {
                    return n;
                }
                seen[n] = w;
                n += 1;
            }
        }
        n
    }

    /// Collapses `p` onto `q` in place (the serial path; `plan_collapse` + `apply_collapse`
    /// produce the same state).
    fn collapse(&mut self, p: u32, q: u32) -> usize {
        let mut map = std::mem::take(&mut self.scratch().borrow_mut().map2);
        let ok = self.map_wedges(p, q, &mut map);
        debug_assert!(ok);
        for &(a, b) in &map {
            if a == b {
                self.vpos[a as usize] = q;
                self.slid.push(a);
                continue;
            }
            self.vremap[a as usize] = b;
            if self.attrs.dim > 0 {
                let (sa, sb) = (a as usize * self.aq_stride, b as usize * self.aq_stride);
                for k in 0..self.aq_stride {
                    self.aq[sb + k] += self.aq[sa + k];
                }
            }
        }
        let mut removed = 0;
        // p's list: triangles shared with q die (and leave q's and the third vertex's lists);
        // the rest move to q, appended in p's order after q's surviving triangles.
        let (s0, l0) = (
            self.adj.start[p as usize] as usize,
            self.adj.len[p as usize] as usize,
        );
        let mut moved = std::mem::take(&mut self.scratch().borrow_mut().nbr_b);
        moved.clear();
        for i in s0..s0 + l0 {
            let t = self.adj.data[i];
            let tp = self.tpos[t as usize];
            if tp.contains(&q) {
                self.tri_alive[t as usize] = false;
                removed += 1;
                self.adj.remove(q, t);
                for &r in &tp {
                    if r != p && r != q {
                        self.adj.remove(r, t);
                    }
                }
            } else {
                let tri = &mut self.tris[t as usize];
                let tpm = &mut self.tpos[t as usize];
                for k in 0..3 {
                    if tpm[k] == p {
                        tpm[k] = q;
                        let v = tri[k];
                        tri[k] = map.iter().find(|(a, _)| *a == v).map_or(v, |&(_, b)| b);
                    }
                }
                moved.push(t);
            }
        }
        self.adj.clear(p);
        for &t in &moved {
            self.adj.push(q, t);
        }
        self.scratch().borrow_mut().nbr_b = moved;
        self.pos_alive[p as usize] = false;
        self.premap[p as usize] = q;
        let qp = self.quadric[p as usize];
        self.quadric[q as usize].add(&qp);
        self.scratch().borrow_mut().map2 = map;
        if self.border[p as usize] {
            let pn = std::mem::take(&mut self.bnbr[p as usize]);
            self.bnbr[q as usize].retain(|&x| x != p);
            for x in pn {
                if x == q {
                    continue;
                }
                let list = &mut self.bnbr[x as usize];
                list.retain(|&y| y != p);
                if !list.contains(&q) {
                    list.push(q);
                }
                if !self.bnbr[q as usize].contains(&x) {
                    self.bnbr[q as usize].push(x);
                }
            }
        }
        removed
    }

    /// Everything collapsing `p` onto `q` will write, computed from a shared borrow, so the
    /// plans of non-interacting collapses can be made in parallel. Sums are taken in the same
    /// order as writing them in place would, so applying a plan is bit-identical.
    fn plan_collapse(&self, p: u32, q: u32) -> CollapsePlan {
        let mut map = Vec::new();
        let ok = self.map_wedges(p, q, &mut map);
        debug_assert!(ok);
        // Attribute rows of q's wedges after absorbing p's (pairs in order, as `+=` would),
        // stored flat: row k is rows[k * stride..], for wedge row_ids[k].
        let stride = self.aq_stride;
        let mut row_ids: Vec<u32> = Vec::new();
        let mut rows: Vec<f64> = Vec::new();
        if self.attrs.dim > 0 {
            for &(a, b) in &map {
                if a == b {
                    continue;
                }
                let sa = a as usize * stride;
                let k = match row_ids.iter().position(|&r| r == b) {
                    Some(k) => k,
                    None => {
                        let sb = b as usize * stride;
                        rows.extend_from_slice(&self.aq[sb..sb + stride]);
                        row_ids.push(b);
                        row_ids.len() - 1
                    }
                };
                for (v, x) in rows[k * stride..(k + 1) * stride]
                    .iter_mut()
                    .zip(&self.aq[sa..sa + stride])
                {
                    *v += *x;
                }
            }
        }
        // p's list: triangles shared with q die; the rest move to q with p's corners renamed.
        let mut die = Vec::new();
        let mut moved = Vec::new();
        for &t in self.adj.get(p) {
            let tp = self.tpos[t as usize];
            if tp.contains(&q) {
                die.push(t);
            } else {
                let mut tri = self.tris[t as usize];
                let mut tpm = tp;
                for k in 0..3 {
                    if tpm[k] == p {
                        tpm[k] = q;
                        let v = tri[k];
                        tri[k] = map.iter().find(|(a, _)| *a == v).map_or(v, |&(_, b)| b);
                    }
                }
                moved.push((t, tri, tpm));
            }
        }
        let mut quadric = self.quadric[q as usize];
        quadric.add(&self.quadric[p as usize]);
        CollapsePlan {
            p,
            q,
            map,
            row_ids,
            rows,
            die,
            moved,
            quadric,
        }
    }

    /// Writes a plan from [`plan_collapse`](Self::plan_collapse). Returns triangles removed.
    fn apply_collapse(&mut self, plan: &CollapsePlan) -> usize {
        let (p, q) = (plan.p, plan.q);
        for &(a, b) in &plan.map {
            if a == b {
                self.vpos[a as usize] = q;
                self.slid.push(a);
            } else {
                self.vremap[a as usize] = b;
            }
        }
        let stride = self.aq_stride;
        for (k, &b) in plan.row_ids.iter().enumerate() {
            let sb = b as usize * stride;
            self.aq[sb..sb + stride].copy_from_slice(&plan.rows[k * stride..(k + 1) * stride]);
        }
        for &t in &plan.die {
            let tp = self.tpos[t as usize];
            self.tri_alive[t as usize] = false;
            self.adj.remove(q, t);
            for &r in &tp {
                if r != p && r != q {
                    self.adj.remove(r, t);
                }
            }
        }
        for &(t, tri, tpm) in &plan.moved {
            self.tris[t as usize] = tri;
            self.tpos[t as usize] = tpm;
        }
        self.adj.clear(p);
        for &(t, _, _) in &plan.moved {
            self.adj.push(q, t);
        }
        self.pos_alive[p as usize] = false;
        self.premap[p as usize] = q;
        self.quadric[q as usize] = plan.quadric;
        if self.border[p as usize] {
            let pn = std::mem::take(&mut self.bnbr[p as usize]);
            self.bnbr[q as usize].retain(|&x| x != p);
            for x in pn {
                if x == q {
                    continue;
                }
                let list = &mut self.bnbr[x as usize];
                list.retain(|&y| y != p);
                if !list.contains(&q) {
                    list.push(q);
                }
                if !self.bnbr[q as usize].contains(&x) {
                    self.bnbr[q as usize].push(x);
                }
            }
        }
        plan.die.len()
    }

    /// After collapsing `p` onto `q`: q's quadric and one-ring changed, so q is fully re-ranked.
    /// A neighbour's other candidates keep their cost (half-edge collapse moves nothing), so it
    /// only needs the new or changed candidate x→q scored, unless its cached target died or was q.
    /// Validity is re-checked lazily when an entry reaches the top of the heap.
    fn refresh(&mut self, p: u32, q: u32, heap: &mut IndexedHeap) {
        // q's collapse costs only grow when it absorbs p (sums of PSD quadrics), so its old key
        // stays a valid lower bound: defer the re-rank until q reaches the top of the heap.
        let (_, qt) = self.best[q as usize];
        if self.alpha == 0.0
            && qt != u32::MAX
            && qt != p
            && self.pos_alive[qt as usize]
            && heap.slot[q as usize] != u32::MAX
        {
            self.dirty[q as usize] = true;
        } else {
            self.requeue(q, heap);
        }
        let mut touched = std::mem::take(&mut self.scratch().borrow_mut().nbr_a);
        {
            let mut s = self.scratch().borrow_mut();
            let s = &mut *s;
            s.epoch = s.epoch.wrapping_add(1);
            let e = s.epoch;
            self.neighbors_into(q, &mut touched, &mut s.mark, e);
        }
        let (mut map, mut corners) = {
            let mut s = self.scratch().borrow_mut();
            (std::mem::take(&mut s.map2), std::mem::take(&mut s.corners2))
        };
        for &x in &touched {
            let (_, target) = self.best[x as usize];
            if target == u32::MAX
                || target == p
                || !self.pos_alive[target as usize]
                || self.locked[x as usize]
            {
                self.requeue(x, heap);
                continue;
            }
            if self.border[x as usize] && !self.bnbr[x as usize].contains(&q) {
                if target == q {
                    self.requeue(x, heap);
                }
                continue;
            }
            self.p_corners(x, &mut corners);
            if !self.strict && !self.relaxed && Self::distinct_wedges(&corners) > MAX_SOURCE_WEDGES
            {
                continue;
            }
            if let Some(tear) = self.map_from_corners(&corners, q, &mut map) {
                let (c, g) = self.cost(x, q, &map, tear);
                let old = heap.key[x as usize];
                if c < old || target == q && c <= old * 1.0001 + 1e-12 {
                    self.best[x as usize] = (g, q);
                    heap.set(x, c);
                } else if target == q {
                    self.requeue(x, heap); // x→q got dearer: another candidate may win now
                }
            } else if target == q {
                self.requeue(x, heap);
            }
        }
        {
            let mut s = self.scratch().borrow_mut();
            s.map2 = map;
            s.corners2 = corners;
        }
        self.scratch().borrow_mut().nbr_a = touched;
    }

    /// Recomputes x's best collapse and updates its heap entry.
    fn requeue(&mut self, x: u32, heap: &mut IndexedHeap) {
        self.dirty[x as usize] = false;
        match self.best_candidate(x) {
            Some(c) => {
                self.best[x as usize] = (c.geom, c.q);
                heap.set(x, c.cost);
            }
            None => {
                self.best[x as usize] = (0.0, u32::MAX);
                heap.remove(x);
            }
        }
    }
}

/// The reduction loop: builds the queue and collapses until a stop condition.
fn run_loop(st: &mut State, target: Option<usize>) -> (usize, f64, usize) {
    let opts = st.opts;
    let max_err_norm = opts.max_error.map(|e| e as f64 / st.inv_scale);
    let mut alive_tris = st.tri_alive.iter().filter(|&&a| a).count();
    let mut heap = IndexedHeap::new(st.pos.len());
    // Initial ranking is independent per position: evaluate in parallel, insert in order
    // (identical to the sequential result on any number of threads).
    let first: Vec<Option<Candidate>> = {
        let shared: &State = st;
        (0..shared.pos.len() as u32)
            .into_par_iter()
            .with_min_len(512)
            .map(|p| shared.best_candidate(p))
            .collect()
    };
    for (p, c) in first.into_iter().enumerate() {
        if let Some(c) = c {
            st.best[p] = (c.geom, c.q);
            heap.set(p as u32, c.cost);
        }
    }
    let mut worst_geom2 = 0.0f64;
    let mut collapses = 0;
    while let Some(p) = heap.pop() {
        let (geom_c, q) = st.best[p as usize];
        let cost_c = heap.key[p as usize];
        if !st.pos_alive[p as usize] || q == u32::MAX {
            continue;
        }
        if st.dirty[p as usize] {
            st.requeue(p, &mut heap);
            continue;
        }
        // Triangle target: stop. Max error: a per-collapse constraint (attribute costs mean
        // the cheapest candidate is not always the geometrically smallest one).
        let tri_met = target.is_some_and(|t| alive_tris <= t);
        if tri_met && (opts.stop == StopWhen::Any || max_err_norm.is_none()) {
            break;
        }
        let over = max_err_norm.is_some_and(|m| geom_c.sqrt() > m);
        let enforce = match opts.stop {
            StopWhen::Any => true,
            StopWhen::All => tri_met || target.is_none(), // All: hit the triangle target first
        };
        if over && enforce {
            continue; // dropped; a neighbourhood refresh may queue p again later
        }
        let partner = st.mirror.as_ref().and_then(|m| {
            let (mp, mq) = (m[p as usize], m[q as usize]);
            (mp != p && mp != u32::MAX && mq != u32::MAX).then_some((mp, mq))
        });
        match st.evaluate(p, q) {
            Some((cost, geom)) if cost <= cost_c * 1.0001 + 1e-12 => {
                if let Some((mp, mq)) = partner {
                    if st.pos_alive[mp as usize] && st.evaluate(mp, mq).is_none() {
                        continue; // mirrored collapse invalid: skip to stay symmetric
                    }
                }
                worst_geom2 = worst_geom2.max(geom);
                alive_tris -= st.collapse(p, q);
                collapses += 1;
                heap.remove(p);
                st.refresh(p, q, &mut heap);
                if let Some((mp, mq)) = partner {
                    if st.pos_alive[mp as usize] && st.pos_alive[mq as usize] {
                        if let Some((_, g2)) = st.evaluate(mp, mq) {
                            worst_geom2 = worst_geom2.max(g2);
                            alive_tris -= st.collapse(mp, mq);
                            collapses += 1;
                            heap.remove(mp);
                            st.refresh(mp, mq, &mut heap);
                        }
                    }
                }
            }
            _ => st.requeue(p, &mut heap),
        }
    }
    (alive_tris, worst_geom2, collapses)
}

fn run_dispatch(st: &mut State, target: Option<usize>) -> (usize, f64, usize) {
    // Mirrored collapses must happen pairwise in order: symmetric runs stay greedy.
    if st.opts.collapse_order == CollapseOrder::Batched && st.mirror.is_none() {
        run_batched(st, target)
    } else {
        run_loop(st, target)
    }
}

/// Share of the current candidates a round may consider (cheapest first). Chosen on the
/// benchmark: 1/8 matched or beat the exact greedy order on every quality metric, while smaller
/// rounds cost time without gaining quality and larger ones lost Hausdorff.
fn batch_fraction() -> f64 {
    0.125
}

/// Batched reduction (see [`CollapseOrder::Batched`]).
fn run_batched(st: &mut State, target: Option<usize>) -> (usize, f64, usize) {
    let opts = st.opts;
    let np = st.pos.len();
    let max_err_norm = opts.max_error.map(|e| e as f64 / st.inv_scale);
    let mut alive_tris = st.tri_alive.iter().filter(|&&a| a).count();
    let mut cand: Vec<Option<Candidate>> = {
        let shared: &State = st;
        (0..np as u32)
            .into_par_iter()
            .with_min_len(512)
            .map(|p| shared.best_candidate(p))
            .collect()
    };
    let mut full = vec![false; np];
    let mut full_list: Vec<u32> = Vec::new();
    let mut inc: Vec<(u32, u32)> = Vec::new();
    let mut stamp = vec![0u32; np];
    // Last round in which each position's one-ring changed, and the round before which each
    // candidate's validity was last established. A candidate is fresh only if neither endpoint
    // changed since: an older check can be stale even when nothing moved this round.
    let mut touched = vec![0u32; np];
    let mut checked_at = vec![1u32; np];
    let reserve: Vec<AtomicU32> = (0..np).map(|_| AtomicU32::new(u32::MAX)).collect();
    // Sub-round in which each position last lay in a committed edge's reservation set.
    let mut mark = vec![0u32; np];
    let mut sub = 0u32;
    let mut ring = Vec::new();
    let (mut worst_geom2, mut collapses) = (0.0f64, 0usize);
    let frac = batch_fraction();
    // Stall escalation: 1 = open the round to every candidate, 2 = re-rank everything, 3 = stop.
    let mut stall = 0u32;
    let cmp = |a: &(f64, u32), b: &(f64, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    let mut round = 0u32;
    loop {
        let tri_met = target.is_some_and(|t| alive_tris <= t);
        if tri_met && (opts.stop == StopWhen::Any || max_err_norm.is_none()) {
            break;
        }
        round += 1;
        let mut list: Vec<(f64, u32)> = {
            let alive = &st.pos_alive;
            cand.par_iter()
                .with_min_len(4096)
                .enumerate()
                .filter_map(|(p, c)| c.as_ref().filter(|_| alive[p]).map(|c| (c.cost, p as u32)))
                .collect()
        };
        if list.is_empty() {
            // Nothing left to collapse: only part deletion can still serve the target.
            if let Some(t) = target.filter(|&t| st.parts.is_some() && alive_tris > t) {
                let floor = t.saturating_sub(alive_tris - t);
                let limit = st.stall_part_limit(worst_geom2);
                let n = st.delete_parts(limit, 1, alive_tris, floor, &mut cand);
                if n > 0 {
                    alive_tris -= n;
                    continue;
                }
            }
            if std::env::var_os("LF_STALL").is_some() {
                st.stall_report(alive_tris);
            }
            break;
        }
        // Round limit: the cost of the m-th cheapest candidate, where m is the smaller of
        // what is still needed and a fixed share of all candidates.
        let needed = target
            .map_or(list.len(), |t| alive_tris.saturating_sub(t).div_ceil(2))
            .max(1);
        let m = if stall > 0 {
            list.len()
        } else {
            needed
                .min((list.len() as f64 * frac).ceil() as usize)
                .clamp(1, list.len())
        };
        list.select_nth_unstable_by(m - 1, cmp);
        list.truncate(m);
        list.sort_unstable_by(cmp);
        let limit = list[m - 1].0;
        let mut executed = 0;
        // Evaluate the round's stale candidates in parallel against the state at the start of
        // the round. The serial loop below uses a precomputed result only while neither
        // endpoint has been touched this round, the same condition under which the "fresh"
        // shortcut reuses ranking-time results, so the outcome is identical to evaluating in
        // the loop, for any thread count.
        let pre: Vec<Option<Option<(f64, f64)>>> = {
            let shared: &State = st;
            let (cand_ref, touched_ref, checked_ref) = (&cand, &touched, &checked_at);
            list.par_iter()
                .with_min_len(32)
                .map(|&(_, p)| {
                    let c = cand_ref[p as usize]?;
                    let q = c.q;
                    if !shared.pos_alive[p as usize] || !shared.pos_alive[q as usize] {
                        return None;
                    }
                    let at = checked_ref[p as usize];
                    if c.checked && touched_ref[p as usize] < at && touched_ref[q as usize] < at {
                        return None; // fresh: no evaluation needed
                    }
                    Some(shared.evaluate(p, q))
                })
                .collect()
        };
        // Parallel commits by deterministic reservations (Blelloch et al. 2012): each pending
        // candidate claims its edge's one-ring with `fetch_min` of its position in the list;
        // a candidate that holds every claim cannot interact with any earlier pending one, so
        // all such winners are decided at once: plans in parallel from a shared borrow, writes
        // in list order. Losers retry. The result is the serial loop's, for any thread count.
        // Near the triangle target the rest of the round runs serially, so it stops at the
        // same collapse.
        let mut rest: std::collections::VecDeque<usize> = (0..list.len()).collect();
        let mut set_cache: Vec<Option<(u32, Vec<u32>)>> = vec![None; list.len()];
        let enforce_par = opts.stop == StopWhen::Any || target.is_none();
        // Every collapse removes at most p's triangles, and stars never grow within a round
        // (a vertex that receives a collapse is used up): if the whole round could reach the
        // target, it runs serially so it stops at the same collapse.
        let can_reach = target.is_some_and(|t| {
            let bound: usize = list.iter().map(|&(_, p)| st.adj.get(p).len()).sum();
            alive_tris <= t + bound
        });
        let reserve_on = rayon::current_num_threads() >= par_commit_threads();
        while reserve_on && !can_reach && rest.len() >= PAR_COMMIT_MIN {
            // The front of the list: the next live candidates (dead or used-up ones are
            // decided by state already and dropped).
            let mut front: Vec<usize> = Vec::with_capacity(COMMIT_PREFIX);
            {
                let shared: &State = st;
                while front.len() < COMMIT_PREFIX {
                    let Some(li) = rest.pop_front() else { break };
                    let p = list[li].1;
                    let Some(c) = cand[p as usize] else { continue };
                    let over = max_err_norm.is_some_and(|m| c.geom.sqrt() > m);
                    if shared.pos_alive[p as usize]
                        && shared.pos_alive[c.q as usize]
                        && stamp[p as usize] != round
                        && stamp[c.q as usize] != round
                        && !(over && enforce_par)
                    {
                        front.push(li);
                    }
                }
            }
            if front.len() < COMMIT_PREFIX {
                for &li in front.iter().rev() {
                    rest.push_front(li);
                }
                break;
            }
            let pending = front;
            let prefix = pending.len();
            #[cfg(test)]
            RESERVATION_SUBROUNDS.fetch_add(1, AtomicOrdering::Relaxed);
            let shared: &State = st;
            let (cand_ref, list_ref) = (&cand, &list);
            // A loser's set is reused unless a commit since touched its edge (any change to
            // either one-ring marks that endpoint: it lay in the committing edge's set).
            let (mark_ref, cache_ref) = (&mark, &set_cache);
            let sets: Vec<Vec<u32>> = pending[..prefix]
                .par_iter()
                .with_min_len(16)
                .map(|&li| {
                    let p = list_ref[li].1;
                    let q = cand_ref[p as usize].unwrap().q;
                    if let Some((at, set)) = &cache_ref[li] {
                        if mark_ref[p as usize] <= *at && mark_ref[q as usize] <= *at {
                            return set.clone();
                        }
                    }
                    let (mut a, mut b) = (Vec::new(), Vec::new());
                    shared.ring(p, &mut a);
                    shared.ring(q, &mut b);
                    a.push(p);
                    a.push(q);
                    a.extend(b);
                    a
                })
                .collect();
            sub += 1;
            let reserve_ref = &reserve;
            sets.par_iter()
                .zip(pending[..prefix].par_iter())
                .for_each(|(set, &li)| {
                    for &v in set {
                        reserve_ref[v as usize].fetch_min(li as u32, AtomicOrdering::Relaxed);
                    }
                });
            let won: Vec<bool> = sets
                .par_iter()
                .zip(pending[..prefix].par_iter())
                .map(|(set, &li)| {
                    set.iter().all(|&v| {
                        reserve_ref[v as usize].load(AtomicOrdering::Relaxed) == li as u32
                    })
                })
                .collect();
            sets.par_iter().for_each(|set| {
                for &v in set {
                    reserve_ref[v as usize].store(u32::MAX, AtomicOrdering::Relaxed);
                }
            });
            let winners: Vec<usize> = pending[..prefix]
                .iter()
                .zip(&won)
                .filter(|(_, &w)| w)
                .map(|(&li, _)| li)
                .collect();
            let (touched_ref, checked_ref, pre_ref) = (&touched, &checked_at, &pre);
            let plans: Vec<Option<(f64, CollapsePlan)>> = winners
                .par_iter()
                .with_min_len(4)
                .map(|&li| {
                    let p = list_ref[li].1;
                    let c = cand_ref[p as usize].unwrap();
                    let q = c.q;
                    let at = checked_ref[p as usize];
                    let fresh =
                        c.checked && touched_ref[p as usize] < at && touched_ref[q as usize] < at;
                    let eval = if fresh {
                        Some((c.cost, c.geom))
                    } else if touched_ref[p as usize] != round && touched_ref[q as usize] != round {
                        match pre_ref[li] {
                            Some(e) => e,
                            None => shared.evaluate(p, q),
                        }
                    } else {
                        shared.evaluate(p, q)
                    };
                    match eval {
                        Some((cost, geom)) if cost <= limit * 1.0001 + 1e-12 => {
                            Some((geom, shared.plan_collapse(p, q)))
                        }
                        _ => None,
                    }
                })
                .collect();
            let mut done: Vec<(u32, u32)> = Vec::new();
            for (&li, plan) in winners.iter().zip(plans) {
                let p = list[li].1;
                match plan {
                    Some((geom, plan)) => {
                        let q = plan.q;
                        worst_geom2 = worst_geom2.max(geom);
                        alive_tris -= st.apply_collapse(&plan);
                        collapses += 1;
                        executed += 1;
                        stamp[p as usize] = round;
                        stamp[q as usize] = round;
                        cand[p as usize] = None;
                        if !full[q as usize] {
                            full[q as usize] = true;
                            full_list.push(q);
                        }
                        done.push((p, q));
                    }
                    None => {
                        if !full[p as usize] {
                            full[p as usize] = true;
                            full_list.push(p);
                        }
                    }
                }
            }
            // One-rings after the writes (disjoint between winners), then the bookkeeping.
            let rings: Vec<Vec<u32>> = {
                let shared: &State = st;
                done.par_iter()
                    .with_min_len(16)
                    .map(|&(_, q)| {
                        let mut r = Vec::new();
                        shared.ring(q, &mut r);
                        r
                    })
                    .collect()
            };
            for (&(p, q), r) in done.iter().zip(&rings) {
                touched[q as usize] = round;
                for &x in r {
                    touched[x as usize] = round;
                    let needs_full = cand[x as usize].is_none_or(|cx| cx.q == p || cx.q == q);
                    if needs_full {
                        if !full[x as usize] {
                            full[x as usize] = true;
                            full_list.push(x);
                        }
                    } else {
                        inc.push((x, q));
                    }
                }
            }
            // Committed edges mark their whole set; losers keep theirs for the next sub-round.
            for ((set, &li), &w) in sets.into_iter().zip(&pending).zip(&won) {
                if w {
                    for &v in &set {
                        mark[v as usize] = sub;
                    }
                    set_cache[li] = None;
                } else {
                    set_cache[li] = Some((sub, set));
                }
            }
            // Losers go back to the front of the list, in order.
            for (&li, &w) in pending.iter().zip(&won).rev() {
                if !w {
                    rest.push_front(li);
                }
            }
            // Too few winners to pay for a sub-round: finish the round serially.
            if winners.len() < COMMIT_PREFIX / 8 {
                break;
            }
        }
        for &li in &rest {
            let p = list[li].1;
            let tri_met = target.is_some_and(|t| alive_tris <= t);
            if tri_met && (opts.stop == StopWhen::Any || max_err_norm.is_none()) {
                break;
            }
            let Some(c) = cand[p as usize] else { continue };
            let q = c.q;
            // Each position takes part in at most one collapse per round.
            if !st.pos_alive[p as usize]
                || !st.pos_alive[q as usize]
                || stamp[p as usize] == round
                || stamp[q as usize] == round
            {
                continue;
            }
            let over = max_err_norm.is_some_and(|m| c.geom.sqrt() > m);
            let enforce = match opts.stop {
                StopWhen::Any => true,
                StopWhen::All => tri_met || target.is_none(),
            };
            if over && enforce {
                continue;
            }
            // Neither endpoint's neighbourhood changed since validity was checked at ranking:
            // cost, wedge map and validity are all still exact.
            let at = checked_at[p as usize];
            let fresh = c.checked && touched[p as usize] < at && touched[q as usize] < at;
            let eval = if fresh {
                Some((c.cost, c.geom))
            } else if touched[p as usize] != round && touched[q as usize] != round {
                match pre[li] {
                    Some(e) => e,
                    None => st.evaluate(p, q),
                }
            } else {
                st.evaluate(p, q)
            };
            match eval {
                Some((cost, geom)) if cost <= limit * 1.0001 + 1e-12 => {
                    worst_geom2 = worst_geom2.max(geom);
                    alive_tris -= st.collapse(p, q);
                    collapses += 1;
                    executed += 1;
                    stamp[p as usize] = round;
                    stamp[q as usize] = round;
                    cand[p as usize] = None;
                    if !full[q as usize] {
                        full[q as usize] = true;
                        full_list.push(q);
                    }
                    st.ring(q, &mut ring);
                    touched[q as usize] = round;
                    for &x in &ring {
                        touched[x as usize] = round;
                        let needs_full = cand[x as usize].is_none_or(|cx| cx.q == p || cx.q == q);
                        if needs_full {
                            if !full[x as usize] {
                                full[x as usize] = true;
                                full_list.push(x);
                            }
                        } else {
                            inc.push((x, q));
                        }
                    }
                }
                _ => {
                    // Stale (now invalid or dearer): re-rank next round.
                    if !full[p as usize] {
                        full[p as usize] = true;
                        full_list.push(p);
                    }
                }
            }
        }
        // Parts whose removal moves the surface less than a quarter of the deviation already
        // accepted go now: their triangles are better spent elsewhere.
        if st.parts.is_some() {
            let thr = st.part_threshold(worst_geom2);
            let floor = target.unwrap_or(0);
            let n = st.delete_parts(thr, usize::MAX, alive_tris, floor, &mut cand);
            alive_tris -= n;
            if n > 0 {
                executed += 1;
            }
        }
        if executed == 0 {
            stall += 1;
            // Collapses are stuck above the target: delete the cheapest part and go on.
            if let Some(t) = target.filter(|&t| stall >= 3 && st.parts.is_some() && alive_tris > t)
            {
                let floor = t.saturating_sub(alive_tris - t);
                let limit = st.stall_part_limit(worst_geom2);
                let n = st.delete_parts(limit, 1, alive_tris, floor, &mut cand);
                if n > 0 {
                    alive_tris -= n;
                    stall = 0;
                    continue;
                }
            }
            match stall {
                1 => {}
                2 => {
                    // Re-rank every live position from scratch before giving up, now letting
                    // many-wedge vertices collapse too.
                    st.relaxed = true;
                    for p in 0..np as u32 {
                        if st.pos_alive[p as usize] && !full[p as usize] {
                            full[p as usize] = true;
                            full_list.push(p);
                        }
                    }
                }
                _ => {
                    if std::env::var_os("LF_STALL").is_some() {
                        st.stall_report(alive_tris);
                    }
                    break;
                }
            }
        } else {
            stall = 0;
        }
        // Re-rank changed positions in parallel (deterministic: applied in list order).
        full_list.sort_unstable();
        let ranked: Vec<(u32, Option<Candidate>)> = {
            let shared: &State = st;
            full_list
                .par_iter()
                .with_min_len(64)
                .map(|&x| (x, shared.best_candidate(x)))
                .collect()
        };
        for (x, c) in ranked {
            cand[x as usize] = c;
            checked_at[x as usize] = round + 1;
            full[x as usize] = false;
        }
        full_list.clear();
        // Incremental: only the candidate x→q changed for these.
        inc.sort_unstable();
        inc.dedup();
        let updates: Vec<(u32, Option<(f64, f64)>)> = {
            let shared: &State = st;
            let cand_ref = &cand;
            inc.par_iter()
                .with_min_len(64)
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(corners, map), &(x, q)| {
                        if !shared.pos_alive[x as usize]
                            || !shared.pos_alive[q as usize]
                            || cand_ref[x as usize].is_none()
                        {
                            return (x, None);
                        }
                        if shared.border[x as usize] && !shared.bnbr[x as usize].contains(&q) {
                            return (x, None);
                        }
                        shared.p_corners(x, corners);
                        if !shared.strict
                            && !shared.relaxed
                            && State::distinct_wedges(corners) > MAX_SOURCE_WEDGES
                        {
                            return (x, None);
                        }
                        if let Some(tear) = shared.map_from_corners(corners, q, map) {
                            (x, Some(shared.cost(x, q, map, tear)))
                        } else {
                            (x, None)
                        }
                    },
                )
                .collect()
        };
        for ((x, u), &(_, q)) in updates.into_iter().zip(inc.iter()) {
            if let (Some((c, g)), Some(old)) = (u, cand[x as usize]) {
                if c < old.cost {
                    cand[x as usize] = Some(Candidate {
                        cost: c,
                        geom: g,
                        q,
                        checked: false,
                    });
                }
            }
        }
        inc.clear();
    }
    (alive_tris, worst_geom2, collapses)
}

#[cfg(test)]
static RESERVATION_SUBROUNDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Thread count from which rounds commit by reservations: [`PAR_COMMIT_THREADS`], or the
/// `FARTHING_PAR_COMMIT_THREADS` environment variable (for tuning on a given machine; the
/// output is the same either way, only the speed changes).
fn par_commit_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FARTHING_PAR_COMMIT_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PAR_COMMIT_THREADS)
    })
}

/// Rounds with at least this many candidates commit by reservations (see `run_batched`).
const PAR_COMMIT_MIN: usize = 256;

/// Reservations cost about 13% more total work than the serial loop (claims, plans, retries)
/// and make the reducer about 91% parallel instead of 87%. Measured on 2 cores they are 9%
/// slower; projected, they break even near 5 threads. Below this many threads the round runs
/// serially. The output is the same either way.
const PAR_COMMIT_THREADS: usize = 8;

/// Candidates a reservation sub-round considers (the front of the round's pending list).
const COMMIT_PREFIX: usize = 256;

/// Triangles per cluster in cluster-parallel reduction.
const CLUSTER_TRIS: usize = 32_768;

fn spread10(v: u32) -> u64 {
    let mut x = (v & 0x3ff) as u64;
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    (x | (x << 2)) & 0x0924_9249
}

/// Cluster-parallel reduction (see [`SimplifyOptions::cluster_threshold`]).
fn simplify_clustered(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: usize,
) -> Result<SimplifyResult> {
    let tc = mesh.triangle_count();
    let vc = mesh.vertex_count();
    // 1. Spatial clusters: triangles in Morton order of their centroids, cut into equal runs.
    let b = mesh.bounds();
    let frame = (b.min.as_dvec3(), b.diagonal().max(1e-12) as f64);
    let ext = (b.max - b.min).max(glam::Vec3::splat(1e-12));
    let code = |t: usize| {
        let [p0, p1, p2] = mesh.triangle_positions(t);
        let c = (((p0 + p1 + p2) / 3.0 - b.min) / ext * 1023.0)
            .clamp(glam::Vec3::ZERO, glam::Vec3::splat(1023.0));
        spread10(c.x as u32) | (spread10(c.y as u32) << 1) | (spread10(c.z as u32) << 2)
    };
    let mut order: Vec<u32> = (0..tc as u32).collect();
    order.par_sort_unstable_by_key(|&t| (code(t as usize), t));
    let k = tc.div_ceil(CLUSTER_TRIS);
    let chunk = tc.div_ceil(k);
    // 2. Positions used by more than one cluster are locked during the parallel phase.
    let key = |v: usize| mesh.positions[v].to_array().map(f32::to_bits);
    let mut vorder: Vec<u32> = (0..vc as u32).collect();
    vorder.par_sort_unstable_by_key(|&v| (key(v as usize), v));
    let mut group = vec![0u32; vc];
    let mut ng = 0u32;
    for (i, &v) in vorder.iter().enumerate() {
        if i > 0 && key(vorder[i - 1] as usize) != key(v as usize) {
            ng += 1;
        }
        group[v as usize] = ng;
    }
    let mut gcl = vec![u32::MAX; ng as usize + 1];
    let mut gborder = vec![false; ng as usize + 1];
    for (i, &t) in order.iter().enumerate() {
        let c = (i / chunk) as u32;
        for v in mesh.triangle(t as usize) {
            let g = group[v as usize] as usize;
            if gcl[g] == u32::MAX {
                gcl[g] = c;
            } else if gcl[g] != c {
                gborder[g] = true;
            }
        }
    }
    let ratio = target as f64 / tc as f64;
    // 3. Reduce every cluster interior in parallel (deterministic: results are joined in order).
    type Part = (Vec<u32>, Vec<u32>, f32, f32, usize, Vec<(u32, u32)>);
    let parts: Vec<Result<Part>> = (0..k)
        .into_par_iter()
        .map(|c| {
            let tris = &order[c * chunk..((c + 1) * chunk).min(tc)];
            let mut local = vec![u32::MAX; vc];
            let mut verts: Vec<u32> = Vec::new();
            let mut idx = Vec::with_capacity(tris.len() * 3);
            let mut mats = Vec::with_capacity(tris.len());
            for &t in tris {
                for v in mesh.triangle(t as usize) {
                    if local[v as usize] == u32::MAX {
                        local[v as usize] = verts.len() as u32;
                        verts.push(v);
                    }
                    idx.push(local[v as usize]);
                }
                mats.push(mesh.material_ids[t as usize]);
            }
            let mut sub = mesh.select_vertices(&verts);
            sub.indices = idx;
            sub.material_ids = mats;
            let locked: Vec<bool> = verts
                .iter()
                .map(|&v| {
                    gborder[group[v as usize] as usize]
                        || opts
                            .locked_vertices
                            .as_ref()
                            .is_some_and(|l| l.get(v as usize).copied().unwrap_or(false))
                })
                .collect();
            let importance = opts.vertex_importance.as_ref().map(|imp| {
                verts
                    .iter()
                    .map(|&v| imp.get(v as usize).copied().unwrap_or(1.0))
                    .collect()
            });
            // Interior triangles go to the final density; the band touching locked border
            // vertices stays for the exact pass (counting it in the budget would over-reduce
            // the interior to pay for triangles it cannot remove).
            let band = sub
                .indices
                .chunks(3)
                .filter(|t| t.iter().any(|&v| locked[v as usize]))
                .count();
            let interior = sub.triangle_count() - band;
            let sub_opts = SimplifyOptions {
                target_ratio: None,
                target_triangles: Some((interior as f64 * ratio).round() as usize + band),
                locked_vertices: Some(locked),
                vertex_importance: importance,
                cluster_threshold: 0,
                ..opts.clone()
            };
            let r = simplify_frame(&sub, &sub_opts, Some(frame), PartsMode::Cluster)?;
            let out_idx = r
                .mesh
                .indices
                .iter()
                .map(|&i| verts[r.source_vertex[i as usize] as usize])
                .collect();
            // Vertices only used by degenerate triangles have no target: keep them pointing at
            // themselves (they are dropped by the final compaction).
            let remap = (0..verts.len())
                .map(|i| match r.vertex_remap[i] {
                    u32::MAX => (verts[i], verts[i]),
                    o => (verts[i], verts[r.source_vertex[o as usize] as usize]),
                })
                .collect();
            Ok((
                out_idx,
                r.mesh.material_ids,
                r.error,
                r.relative_error,
                r.collapses,
                remap,
            ))
        })
        .collect();
    // 4. Join, then one exact pass over the whole mesh to reduce the locked border bands.
    let mut merged = mesh.clone();
    merged.indices.clear();
    merged.material_ids.clear();
    let mut stage_remap: Vec<u32> = (0..vc as u32).collect();
    let (mut err, mut collapses) = (0.0f32, 0usize);
    let mut prior_geom2 = 0.0f64;
    for part in parts {
        let (idx, mats, e, rel, n, remap) = part?;
        prior_geom2 = prior_geom2.max((rel as f64).powi(2));
        merged.indices.extend_from_slice(&idx);
        merged.material_ids.extend_from_slice(&mats);
        for (from, to) in remap {
            stage_remap[from as usize] = to;
        }
        err = err.max(e);
        collapses += n;
    }
    let cmap = merged.compact();
    let mut inv = vec![0u32; merged.vertex_count()];
    for (orig, &c) in cmap.iter().enumerate() {
        if c != u32::MAX {
            inv[c as usize] = orig as u32;
        }
    }
    let map_vec = |v: &Option<Vec<bool>>| {
        v.as_ref().map(|l| {
            inv.iter()
                .map(|&o| l.get(o as usize).copied().unwrap_or(false))
                .collect()
        })
    };
    let final_opts = SimplifyOptions {
        target_ratio: None,
        target_triangles: Some(target),
        locked_vertices: map_vec(&opts.locked_vertices),
        vertex_importance: opts.vertex_importance.as_ref().map(|imp| {
            inv.iter()
                .map(|&o| imp.get(o as usize).copied().unwrap_or(1.0))
                .collect()
        }),
        cluster_threshold: 0,
        ..opts.clone()
    };
    let r = simplify_frame(
        &merged,
        &final_opts,
        Some(frame),
        PartsMode::Final(prior_geom2),
    )?;
    let source_vertex = r.source_vertex.iter().map(|&c| inv[c as usize]).collect();
    let vertex_remap = (0..vc)
        .map(|v| {
            let s = stage_remap[v];
            match cmap[s as usize] {
                u32::MAX => r.vertex_remap.first().copied().unwrap_or(0),
                c => r.vertex_remap[c as usize],
            }
        })
        .collect();
    Ok(SimplifyResult {
        mesh: r.mesh,
        error: err.max(r.error),
        relative_error: err.max(r.error) / frame.1 as f32,
        vertex_remap,
        collapses: collapses + r.collapses,
        symmetry_plane: None,
        source_vertex,
    })
}

/// Simplifies `mesh`. With no target set, keeps 50% of the triangles.
pub fn simplify(mesh: &Mesh, opts: &SimplifyOptions) -> Result<SimplifyResult> {
    simplify_top(mesh, opts)
}

fn simplify_top(mesh: &Mesh, opts: &SimplifyOptions) -> Result<SimplifyResult> {
    if opts.quality == Quality::Fast {
        mesh.validate()?;
        if let Some(r) = opts.target_ratio {
            if !(0.0..=1.0).contains(&r) {
                return Err(Error::InvalidInput(format!(
                    "target_ratio {r} must be within 0..=1"
                )));
            }
        }
        let target = resolve_target(mesh.triangle_count(), opts);
        let mut r = crate::reduce::fast::simplify_fast(mesh, opts, target)?;
        // Vertices of deleted parts map onto the remaining surface.
        crate::reduce::parts::remap_orphans(mesh, &mut r);
        return Ok(r);
    }
    let mut r = simplify_frame(mesh, opts, None, PartsMode::Top)?;
    // Vertices of removed parts map onto the remaining surface.
    crate::reduce::parts::remap_orphans(mesh, &mut r);
    Ok(r)
}

/// Whether a reduction may delete whole parts (see [`SimplifyOptions::remove_parts`]).
#[derive(Clone, Copy, PartialEq, Debug)]
enum PartsMode {
    /// The caller's own reduction.
    Top,
    /// A cluster of a cluster-parallel reduction: parts are left to the final pass.
    Cluster,
    /// The final pass of a cluster-parallel reduction, after clusters already accepted
    /// deviations up to this (unit-frame) squared error.
    Final(f64),
}

/// The triangle target `opts` asks for (none when only `max_error` is set).
fn resolve_target(input_tris: usize, opts: &SimplifyOptions) -> Option<usize> {
    match (opts.target_ratio, opts.target_triangles) {
        (Some(r), Some(t)) => Some(((input_tris as f32 * r) as usize).min(t)),
        (Some(r), None) => Some((input_tris as f32 * r) as usize),
        (None, Some(t)) => Some(t),
        (None, None) => None,
    }
    .or(opts.max_error.is_none().then_some(input_tris / 2))
}

fn simplify_frame(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    frame: Option<(DVec3, f64)>,
    parts_mode: PartsMode,
) -> Result<SimplifyResult> {
    mesh.validate()?;
    if let Some(r) = opts.target_ratio {
        if !(0.0..=1.0).contains(&r) {
            return Err(Error::InvalidInput(format!(
                "target_ratio {r} must be within 0..=1"
            )));
        }
    }
    let input_tris = mesh.triangle_count();
    let target = resolve_target(input_tris, opts);
    if let Some(t) = target {
        if opts.cluster_threshold > 0
            && input_tris >= opts.cluster_threshold.max(2 * CLUSTER_TRIS)
            && opts.symmetry == Symmetry::Off
            && t < input_tris
        {
            return simplify_clustered(mesh, opts, t);
        }
    }
    let wants_parts = opts.remove_parts
        && opts.symmetry == Symmetry::Off
        && opts.collapse_order == CollapseOrder::Batched
        && target.is_some_and(|t| t < input_tris)
        && parts_mode != PartsMode::Cluster;
    let prior = match parts_mode {
        PartsMode::Final(g) => g,
        _ => 0.0,
    };
    // The part analysis runs alongside building the reducer state.
    let (parts, mut st) = rayon::join(
        || {
            wants_parts
                .then(|| {
                    crate::reduce::parts::Parts::analyze(
                        mesh,
                        opts.locked_vertices.as_deref(),
                        &opts.transparent_materials,
                    )
                })
                .flatten()
        },
        || State::new(mesh, opts, frame),
    );
    st.parts = parts;
    st.prior_geom2 = prior;
    let (alive_tris, worst_geom2, collapses) = run_dispatch(&mut st, target);

    // Assemble output.
    let resolve = |mut v: u32| {
        let mut guard = 0;
        while st.vremap[v as usize] != v && guard < 1_000_000 {
            v = st.vremap[v as usize];
            guard += 1;
        }
        v
    };
    let mut out_indices = Vec::with_capacity(alive_tris * 3);
    let mut out_mats = Vec::with_capacity(alive_tris);
    for (t, tri) in st.tris.iter().enumerate() {
        if st.tri_alive[t] {
            out_indices.extend(tri.iter().map(|&v| resolve(v)));
            out_mats.push(mesh.material_ids[t]);
        }
    }
    let mut out = mesh.clone();
    out.indices = out_indices;
    out.material_ids = out_mats;
    let remap = out.compact();
    for &v in &st.slid {
        let new = remap[v as usize];
        if new != u32::MAX {
            let src = st.pos_src[st.vpos[v as usize] as usize];
            out.positions[new as usize] = mesh.positions[src as usize];
        }
    }
    let resolve_pos = |mut p: u32| {
        let mut guard = 0;
        while st.premap[p as usize] != p && guard < 1_000_000 {
            p = st.premap[p as usize];
            guard += 1;
        }
        p
    };
    let mut pos_out = vec![u32::MAX; st.pos.len()];
    for v in 0..mesh.vertex_count() {
        if remap[v] != u32::MAX {
            let p = st.vpos[v] as usize;
            if pos_out[p] == u32::MAX {
                pos_out[p] = remap[v];
            }
        }
    }
    let vertex_remap = (0..mesh.vertex_count() as u32)
        .map(|v| {
            let direct = remap[resolve(v) as usize];
            if direct != u32::MAX {
                direct
            } else {
                pos_out[resolve_pos(st.vpos[v as usize]) as usize]
            }
        })
        .collect();
    let symmetry_plane = st
        .mirror
        .as_ref()
        .and_then(|_| crate::reduce::symmetry::resolve_plane(mesh, opts.symmetry));
    let mut source_vertex = vec![0u32; out.vertex_count()];
    for (old, &new) in remap.iter().enumerate() {
        if new != u32::MAX {
            source_vertex[new as usize] = old as u32;
        }
    }
    Ok(SimplifyResult {
        mesh: out,
        error: ((worst_geom2.sqrt() * st.inv_scale) as f32).max(st.part_error),
        relative_error: worst_geom2.sqrt() as f32,
        vertex_remap,
        collapses,
        symmetry_plane,
        source_vertex,
    })
}

/// Simplifies every mesh of a scene in parallel.
pub fn simplify_scene(
    scene: &Scene,
    opts: &SimplifyOptions,
) -> Result<(Scene, Vec<SimplifyResult>)> {
    let results: Vec<SimplifyResult> = scene
        .meshes
        .par_iter()
        .map(|m| simplify(m, opts))
        .collect::<Result<_>>()?;
    let mut out = scene.clone();
    for (m, r) in out.meshes.iter_mut().zip(&results) {
        *m = r.mesh.clone();
    }
    Ok((out, results))
}

/// One level of a LOD chain.
#[derive(Clone, Debug)]
pub struct LodLevel {
    pub mesh: Mesh,
    /// Accumulated estimated deviation from LOD0 (world units).
    pub error: f32,
    pub triangles: usize,
}

/// Builds a LOD chain. `ratios` are relative to the *input* triangle count (e.g. 0.5, 0.25,
/// 0.125). Each level is simplified from the previous one; errors accumulate.
pub fn lod_chain(mesh: &Mesh, ratios: &[f32], base: &SimplifyOptions) -> Result<Vec<LodLevel>> {
    let input = mesh.triangle_count();
    let mut levels = vec![LodLevel {
        mesh: mesh.clone(),
        error: 0.0,
        triangles: input,
    }];
    for &r in ratios {
        let prev = levels.last().unwrap();
        let target = ((input as f32 * r) as usize).max(1);
        let opts = SimplifyOptions {
            target_ratio: None,
            target_triangles: Some(target),
            vertex_importance: None,
            locked_vertices: None,
            ..base.clone()
        };
        let res = simplify(&prev.mesh, &opts)?;
        let error = prev.error + res.error;
        let triangles = res.mesh.triangle_count();
        levels.push(LodLevel {
            mesh: res.mesh,
            error,
            triangles,
        });
    }
    Ok(levels)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::metrics::surface_distance;
    use glam::{Vec2, Vec3};

    /// UV sphere with a seam (duplicated vertices along u = 0/1) and poles.
    pub fn uv_sphere(seg: u32, rings: u32) -> Mesh {
        let mut pos = Vec::new();
        let mut uv = Vec::new();
        let mut nrm = Vec::new();
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let th = v * std::f32::consts::PI;
            for s in 0..=seg {
                let u = s as f32 / seg as f32;
                let ph = if s == seg {
                    0.0
                } else {
                    u * std::f32::consts::TAU
                };
                let p = if r == 0 {
                    Vec3::Y
                } else if r == rings {
                    -Vec3::Y
                } else {
                    Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin())
                };
                pos.push(p);
                nrm.push(p);
                uv.push(Vec2::new(u, v));
            }
        }
        let mut idx = Vec::new();
        for r in 0..rings {
            for s in 0..seg {
                let a = r * (seg + 1) + s;
                let b = a + seg + 1;
                idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        let mut m = Mesh::new(pos, idx);
        let mut keep_i = Vec::new();
        for t in 0..m.triangle_count() {
            let [a, b, c] = m.triangle_positions(t);
            if (b - a).cross(c - a).length() > 1e-9 {
                keep_i.extend_from_slice(&m.triangle(t));
            }
        }
        m.material_ids = vec![0; keep_i.len() / 3];
        m.indices = keep_i;
        m.normals = Some(nrm);
        m.uvs = vec![uv];
        m
    }

    fn grid(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.push(Vec3::new(x as f32, 0.0, y as f32));
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                idx.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        Mesh::new(pos, idx)
    }

    #[test]
    fn flat_grid_collapses_to_few_triangles_without_error() {
        let g = grid(20);
        let r = simplify(
            &g,
            &SimplifyOptions {
                target_triangles: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.mesh.triangle_count() <= 8,
            "got {}",
            r.mesh.triangle_count()
        );
        assert!(r.error < 1e-3);
        assert!(
            (r.mesh.surface_area() - 400.0).abs() < 1e-2,
            "area {}",
            r.mesh.surface_area()
        );
        r.mesh.validate().unwrap();
    }

    #[test]
    fn sphere_ratio_and_quality() {
        let s = uv_sphere(48, 24);
        let r = simplify(&s, &SimplifyOptions::ratio(0.25)).unwrap();
        let n = r.mesh.triangle_count();
        let target = s.triangle_count() / 4;
        assert!(n <= target && n > target / 2, "{n} vs {target}");
        let d = surface_distance(&s, &r.mesh, 4000);
        assert!(d.relative_hausdorff < 0.03, "{d:?}");
        r.mesh.validate().unwrap();
    }

    #[test]
    fn seam_is_preserved() {
        let s = uv_sphere(32, 16);
        let r = simplify(
            &s,
            &SimplifyOptions {
                target_ratio: Some(0.2),
                seams: SeamMode::Strict,
                ..Default::default()
            },
        )
        .unwrap();
        let m = &r.mesh;
        let uv = &m.uvs[0];
        let mut u0 = 0;
        for v in 0..m.vertex_count() {
            // Poles are UV singularities (one wedge per fan triangle): not a seam pair.
            if uv[v].x == 0.0 && m.positions[v].y.abs() < 0.999 {
                u0 += 1;
                let p = m.positions[v];
                let others: Vec<_> = (0..m.vertex_count())
                    .filter(|&w| w != v && (m.positions[w] - p).length() < 1e-6)
                    .map(|w| uv[w])
                    .collect();
                assert!(
                    (0..m.vertex_count())
                        .any(|w| uv[w].x == 1.0 && (m.positions[w] - p).length() < 1e-6),
                    "seam vertex lost its partner at {p:?} uv {:?}, co-located wedges {others:?}",
                    uv[v]
                );
            }
        }
        assert!(u0 >= 2);
    }

    #[test]
    fn permissive_seams_reduce_further() {
        // A strip with a hard UV seam across the middle: strict cannot remove seam rows.
        let s = uv_sphere(32, 16);
        let strict = simplify(
            &s,
            &SimplifyOptions {
                target_triangles: Some(20),
                seams: SeamMode::Strict,
                ..Default::default()
            },
        )
        .unwrap();
        let perm = simplify(
            &s,
            &SimplifyOptions {
                target_triangles: Some(20),
                seams: SeamMode::Permissive,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(perm.mesh.triangle_count() <= strict.mesh.triangle_count());
        perm.mesh.validate().unwrap();
    }

    #[test]
    fn permissive_seams_reach_target_where_strict_stalls() {
        // Four UV islands meeting at the centre of a grid: the four-wedge centre vertex is
        // locked under strict seams.
        let g = grid(12);
        let mut pos = Vec::new();
        let mut uv = Vec::new();
        let mut idx = Vec::new();
        for t in 0..g.triangle_count() {
            let tri = g.triangle_positions(t);
            let c = (tri[0] + tri[1] + tri[2]) / 3.0;
            let island = Vec2::new(
                if c.x < 6.0 { 0.0 } else { 10.0 },
                if c.z < 6.0 { 0.0 } else { 10.0 },
            );
            for p in tri {
                idx.push(pos.len() as u32);
                pos.push(p);
                uv.push(island + Vec2::new(p.x, p.z) / 12.0);
            }
        }
        let mut m = Mesh::new(pos, idx);
        m.uvs = vec![uv];
        crate::reduce::repair::weld(&mut m, 0.0);
        let strict = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(2),
                seams: SeamMode::Strict,
                ..Default::default()
            },
        )
        .unwrap();
        let auto = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(2),
                seams: SeamMode::Permissive,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            strict.mesh.triangle_count() > 2,
            "test premise: strict stalls ({})",
            strict.mesh.triangle_count()
        );
        assert!(
            auto.mesh.triangle_count() < strict.mesh.triangle_count(),
            "auto {} vs strict {}",
            auto.mesh.triangle_count(),
            strict.mesh.triangle_count()
        );
        auto.mesh.validate().unwrap();
    }

    /// `m` cut into UV islands 10 apart in UV: `island` picks a triangle's island from its
    /// centroid, and each island maps positions to UVs within a unit square.
    fn with_islands(m: &Mesh, island: impl Fn(Vec3) -> u32, local: impl Fn(Vec3) -> Vec2) -> Mesh {
        let mut pos = Vec::new();
        let mut uv = Vec::new();
        let mut idx = Vec::new();
        for t in 0..m.triangle_count() {
            let tri = m.triangle_positions(t);
            let k = island((tri[0] + tri[1] + tri[2]) / 3.0);
            let off = Vec2::new(10.0 * (k % 4) as f32, 10.0 * (k / 4) as f32);
            for p in tri {
                idx.push(pos.len() as u32);
                pos.push(p);
                uv.push(off + local(p));
            }
        }
        let mut out = Mesh::new(pos, idx);
        out.uvs = vec![uv];
        crate::reduce::repair::weld(&mut out, 0.0);
        out
    }

    #[test]
    fn permissive_seams_keep_triangles_on_one_uv_island() {
        // UV islands meeting at T-junctions, so many seam wedges have no wedge of their island
        // at the position they collapse onto. Such a wedge must slide (keeping its UVs), never
        // take another island's wedge: a triangle spanning two islands samples whatever lies
        // between them in the texture (the Lantern's brace showed the atlas gutter).
        let grid = with_islands(
            &grid(12),
            |c| u32::from(c.x >= 5.0) + u32::from(c.x >= 5.0 && c.z >= 4.0),
            |p| Vec2::new(p.x, p.z) / 12.0,
        );
        let sphere = with_islands(
            &uv_sphere(32, 16),
            |c| u32::from(c.x > 0.3) + 2 * u32::from(c.y > -0.2) + 4 * u32::from(c.z > 0.1),
            |p| Vec2::new(p.x + p.z, p.y + p.z) * 0.25 + 0.5,
        );
        for (name, m) in [("grid", &grid), ("sphere", &sphere)] {
            for target in [120, 60, 30, 16, 8] {
                let r = simplify(
                    m,
                    &SimplifyOptions {
                        target_triangles: Some(target),
                        seams: SeamMode::Permissive,
                        ..Default::default()
                    },
                )
                .unwrap();
                r.mesh.validate().unwrap();
                for t in 0..r.mesh.triangle_count() {
                    let [a, b, c] = r.mesh.triangle(t).map(|v| r.mesh.uvs[0][v as usize]);
                    let span = a.max(b).max(c) - a.min(b).min(c);
                    assert!(
                        span.max_element() < 2.0,
                        "{name} target {target}: triangle {t} spans UV islands: {a} {b} {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn attribute_quadrics_protect_uv_discontinuity() {
        // Flat grid whose UVs fold sharply along x = 10: the fold should survive reduction.
        let mut g = grid(20);
        g.uvs = vec![g
            .positions
            .iter()
            .map(|p| {
                Vec2::new(
                    if p.x <= 10.0 {
                        p.x / 10.0
                    } else {
                        1.0 + (p.x - 10.0) * 2.0
                    },
                    p.z / 20.0,
                )
            })
            .collect()];
        let r = simplify(
            &g,
            &SimplifyOptions {
                target_triangles: Some(16),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.mesh.positions.iter().any(|p| (p.x - 10.0).abs() < 1e-5),
            "UV kink column was collapsed away"
        );
    }

    #[test]
    fn reduction_is_deterministic() {
        let s = uv_sphere(48, 24);
        let a = simplify(&s, &SimplifyOptions::ratio(0.2)).unwrap();
        let b = simplify(&s, &SimplifyOptions::ratio(0.2)).unwrap();
        assert_eq!(a.mesh, b.mesh);
    }

    #[test]
    fn max_error_stops_early() {
        let s = uv_sphere(48, 24);
        let r = simplify(
            &s,
            &SimplifyOptions {
                max_error: Some(0.002),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.error <= 0.002 + 1e-6);
        assert!(r.mesh.triangle_count() < s.triangle_count());
        assert!(r.mesh.triangle_count() > s.triangle_count() / 10);
    }

    #[test]
    fn importance_protects_region() {
        let s = uv_sphere(48, 24);
        let imp: Vec<f32> = s
            .positions
            .iter()
            .map(|p| if p.y > 0.0 { 50.0 } else { 1.0 })
            .collect();
        let r = simplify(
            &s,
            &SimplifyOptions {
                target_ratio: Some(0.3),
                vertex_importance: Some(imp),
                ..Default::default()
            },
        )
        .unwrap();
        let top = r.mesh.positions.iter().filter(|p| p.y > 0.05).count();
        let bottom = r.mesh.positions.iter().filter(|p| p.y < -0.05).count();
        assert!(top > bottom * 2, "top {top} bottom {bottom}");
    }

    #[test]
    fn locked_border() {
        let g = grid(10);
        let r = simplify(
            &g,
            &SimplifyOptions {
                target_triangles: Some(2),
                lock_border: true,
                ..Default::default()
            },
        )
        .unwrap();
        let border = r
            .mesh
            .positions
            .iter()
            .filter(|p| p.x == 0.0 || p.x == 10.0 || p.z == 0.0 || p.z == 10.0)
            .count();
        assert_eq!(border, 40);
    }

    #[test]
    fn lod_chain_is_monotonic() {
        let s = uv_sphere(40, 20);
        let chain = lod_chain(&s, &[0.5, 0.25, 0.1], &SimplifyOptions::default()).unwrap();
        assert_eq!(chain.len(), 4);
        for w in chain.windows(2) {
            assert!(w[1].triangles < w[0].triangles);
            assert!(w[1].error >= w[0].error);
        }
    }

    #[test]
    fn geomorph_remap_targets_exist() {
        let s = uv_sphere(24, 12);
        let r = simplify(&s, &SimplifyOptions::ratio(0.3)).unwrap();
        let mut used = vec![false; s.vertex_count()];
        for &i in &s.indices {
            used[i as usize] = true;
        }
        for (v, &t) in r.vertex_remap.iter().enumerate() {
            if used[v] {
                assert!((t as usize) < r.mesh.vertex_count());
            }
        }
    }

    #[test]
    fn scene_reduction_in_parallel() {
        let mut s = Scene::from_mesh(uv_sphere(24, 12));
        s.meshes.push(grid(10));
        let (out, res) = simplify_scene(&s, &SimplifyOptions::ratio(0.5)).unwrap();
        assert_eq!(res.len(), 2);
        assert!(out.meshes[0].triangle_count() < s.meshes[0].triangle_count());
    }
}

#[cfg(test)]
mod attr_quadric_check {
    use super::*;
    use glam::Vec3;

    /// The compact attribute quadric must equal the brute-force sum over p's faces of
    /// area · |linear_model_f(q) − s(q)|², normalized by area.
    #[test]
    fn compact_form_matches_brute_force() {
        let n = 9;
        let mut pos = Vec::new();
        let mut nrm = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32);
                let z = 0.1 * (3.0 * x).sin() * (2.0 * y).cos();
                pos.push(Vec3::new(x, y, z));
                nrm.push(Vec3::new((5.0 * x).sin(), (4.0 * y).cos(), 1.0).normalize());
            }
        }
        let mut idx = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = (j * n + i) as u32;
                idx.extend_from_slice(&[
                    a,
                    a + 1,
                    a + n as u32,
                    a + 1,
                    a + n as u32 + 1,
                    a + n as u32,
                ]);
            }
        }
        let mut m = Mesh::new(pos, idx);
        m.normals = Some(nrm);
        let opts = SimplifyOptions {
            normal_weight: 1.0,
            uv_weight: 0.0,
            ..Default::default()
        };
        let st = State::new(&m, &opts, None);
        let p = (4 * n + 4) as u32;
        let nw = 0.5f64; // sqrt(1.0) * 0.5
        for q in st.neighbors(p) {
            let mut map = Vec::new();
            assert!(st.map_wedges(p, q, &mut map));
            let (mut e, mut w) = (0.0, 0.0);
            for &(pw, qw) in &map {
                let (en, eo, wi) = st.attr_eval(pw, st.pos[q as usize], qw);
                e += en + eo;
                w += wi;
            }
            // brute force
            let (mut be, mut bw) = (0.0f64, 0.0f64);
            let sq: Vec<f64> = m.normals.as_ref().unwrap()[q as usize]
                .to_array()
                .iter()
                .map(|&v| v as f64 * nw)
                .collect();
            for (t, tri) in st.tris.iter().enumerate() {
                if !tri.contains(&p) || !st.tri_alive[t] {
                    continue;
                }
                let x = tri.map(|v| st.pos[v as usize]);
                let nn = (x[1] - x[0]).cross(x[2] - x[0]);
                let area = nn.length() * 0.5;
                let nn = nn.normalize();
                let mm = glam::DMat3::from_cols(x[1] - x[0], x[2] - x[0], nn)
                    .transpose()
                    .inverse();
                for j in 0..3 {
                    let s: Vec<f64> = tri
                        .iter()
                        .map(|&v| m.normals.as_ref().unwrap()[v as usize][j] as f64 * nw)
                        .collect();
                    let g = mm * DVec3::new(s[1] - s[0], s[2] - s[0], 0.0);
                    let model = s[0] + g.dot(st.pos[q as usize] - x[0]);
                    be += area * (model - sq[j]).powi(2);
                }
                bw += area;
            }
            let (a, b) = (e / w, be / bw);
            assert!(
                (a - b).abs() <= 1e-9 + 1e-6 * b.abs(),
                "q {q}: compact {a} vs brute {b}"
            );
        }
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;

    #[test]
    fn batched_is_deterministic_and_reaches_targets() {
        let s = tests::uv_sphere(64, 32);
        for ratio in [0.5, 0.1, 0.02] {
            let o = SimplifyOptions::ratio(ratio);
            let a = simplify(&s, &o).unwrap();
            let b = simplify(&s, &o).unwrap();
            assert_eq!(a.mesh, b.mesh);
            let target = (s.triangle_count() as f32 * ratio) as usize;
            assert!(
                a.mesh.triangle_count() <= target,
                "{} > {target}",
                a.mesh.triangle_count()
            );
            a.mesh.validate().unwrap();
        }
    }

    #[test]
    fn batched_quality_matches_greedy() {
        let s = tests::uv_sphere(96, 48);
        let o = SimplifyOptions::ratio(0.1);
        let b = simplify(&s, &o).unwrap();
        let g = simplify(
            &s,
            &SimplifyOptions {
                collapse_order: CollapseOrder::Greedy,
                ..o.clone()
            },
        )
        .unwrap();
        let db = crate::metrics::surface_distance(&s, &b.mesh, 20000);
        let dg = crate::metrics::surface_distance(&s, &g.mesh, 20000);
        assert!(
            db.relative_mean <= dg.relative_mean * 1.15,
            "batched {} greedy {}",
            db.relative_mean,
            dg.relative_mean
        );
    }
}

#[cfg(test)]
mod part_tests {
    use super::*;
    use glam::Vec3;

    /// A closed torus (genus 1: collapses cannot take it below about 14 triangles).
    pub(super) fn torus(seg: u32, ring: u32) -> Mesh {
        let mut pos = Vec::new();
        for i in 0..seg {
            for j in 0..ring {
                let (a, b) = (
                    i as f32 / seg as f32 * std::f32::consts::TAU,
                    j as f32 / ring as f32 * std::f32::consts::TAU,
                );
                let r = 1.0 + 0.4 * b.cos();
                pos.push(Vec3::new(r * a.cos(), 0.4 * b.sin(), r * a.sin()));
            }
        }
        let id = |i: u32, j: u32| (i % seg) * ring + (j % ring);
        let mut idx = Vec::new();
        for i in 0..seg {
            for j in 0..ring {
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                idx.extend_from_slice(&[a, c, b, a, d, c]);
            }
        }
        Mesh::new(pos, idx)
    }

    /// A big sphere with `n` small tori sitting on its surface, plus (optionally) one small
    /// torus far away from everything.
    fn scene(n: usize, with_far: bool) -> Mesh {
        let mut m = tests::uv_sphere(48, 24);
        let small = torus(12, 6);
        let add = |c: Vec3, r: f32, m: &mut Mesh| {
            let mut s = small.clone();
            for p in &mut s.positions {
                *p = c + *p * r;
            }
            m.append(&s, 0);
        };
        for i in 0..n {
            let a = i as f32 * 2.399;
            let c = Vec3::new(a.cos(), ((i as f32 / n as f32) - 0.5) * 1.6, a.sin()).normalize();
            add(c * 0.99, 0.04, &mut m);
        }
        if with_far {
            add(Vec3::new(3.0, 0.0, 0.0), 0.04, &mut m);
        }
        m
    }

    #[test]
    fn same_result_for_any_thread_count() {
        // Parts, attributes (normals), borders and many rounds: every parallel stage,
        // including commits by reservations (8+ threads).
        let mut m = scene(30, true);
        m.append(&tests::uv_sphere(160, 80), 0);
        m.compute_normals(60.0);
        let opts = SimplifyOptions {
            target_ratio: Some(0.1),
            ..Default::default()
        };
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| simplify(&m, &opts).unwrap())
        };
        let a = run(1);
        let before = RESERVATION_SUBROUNDS.load(std::sync::atomic::Ordering::Relaxed);
        let b = run(PAR_COMMIT_THREADS);
        assert!(
            RESERVATION_SUBROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before,
            "reservation path not exercised"
        );
        assert_eq!(a.mesh.indices, b.mesh.indices);
        assert_eq!(a.mesh.positions, b.mesh.positions);
        assert_eq!(a.mesh.normals, b.mesh.normals);
        assert_eq!(a.vertex_remap, b.vertex_remap);
        assert_eq!(a.error.to_bits(), b.error.to_bits());
    }

    fn moved(mut m: Mesh, c: Vec3, r: f32) -> Mesh {
        for p in &mut m.positions {
            *p = c + *p * r;
        }
        m
    }

    #[test]
    fn hidden_parts_cost_nothing_unless_seen_through_glass() {
        // A torus inside a closed sphere and one outside it.
        let mut m = tests::uv_sphere(32, 16);
        let first_inner = m.triangle_count();
        m.append(&moved(torus(12, 6), Vec3::ZERO, 0.3), 1);
        m.append(&moved(torus(12, 6), Vec3::new(3.0, 0.0, 0.0), 0.3), 1);
        let take_free = |transparent: &[u32]| {
            let mut parts = crate::reduce::parts::Parts::analyze(&m, None, transparent).unwrap();
            parts.take(1e-6, None, |t| t.len(), usize::MAX / 4, 0)
        };
        // Only the enclosed torus is free to remove…
        let (k, e) = take_free(&[]).expect("enclosed part is free");
        assert_eq!(e, 0.0);
        assert_eq!(k, crate::reduce::parts::parts_of(&m).0[first_inner]);
        // …and not when the sphere is glass.
        assert!(take_free(&[0]).is_none());
    }

    /// Three fins (grids) meeting along the x axis: a non-manifold junction line.
    fn fins(n: u32) -> Mesh {
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        for i in 0..=n {
            pos.push(Vec3::new(i as f32 / n as f32, 0.0, 0.0));
        }
        for f in 0..3 {
            let a = f as f32 * std::f32::consts::TAU / 3.0;
            let dir = Vec3::new(0.0, a.cos(), a.sin());
            let base = pos.len() as u32;
            for j in 1..=4 {
                for i in 0..=n {
                    pos.push(Vec3::new(i as f32 / n as f32, 0.0, 0.0) + dir * (j as f32 * 0.1));
                }
            }
            let at = |i: u32, j: u32| {
                if j == 0 {
                    i
                } else {
                    base + (j - 1) * (n + 1) + i
                }
            };
            for j in 0..4 {
                for i in 0..n {
                    let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                    idx.extend_from_slice(&[a, b, c, a, c, d]);
                }
            }
        }
        Mesh::new(pos, idx)
    }

    #[test]
    fn junction_vertices_slide_along_the_junction() {
        let n = 20;
        let m = fins(n);
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(30),
                ..Default::default()
            },
        )
        .unwrap();
        let on_line = |m: &Mesh| {
            m.positions
                .iter()
                .filter(|p| p.y.abs() < 1e-6 && p.z.abs() < 1e-6)
                .count()
        };
        assert!(
            on_line(&r.mesh) < (n + 1) as usize / 2,
            "{} junction vertices left",
            on_line(&r.mesh)
        );
        // The junction keeps its ends and the fins keep their extent.
        let d = crate::metrics::surface_distance(&m, &r.mesh, 4000);
        assert!(d.relative_hausdorff < 0.01, "{d:?}");
    }

    #[test]
    fn double_sided_twins_reduce_like_one_sheet() {
        // An open grid plus its reversed twin: every edge is shared by the two copies.
        let n = 16u32;
        let mut pos = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
                pos.push(Vec3::new(x, y, 0.1 * (3.0 * x).sin() * (2.0 * y).cos()));
            }
        }
        let mut idx = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                let (b, c, d) = (a + 1, a + n + 2, a + n + 1);
                idx.extend_from_slice(&[a, b, c, a, c, d]);
                idx.extend_from_slice(&[a, c, b, a, d, c]);
            }
        }
        let m = Mesh::new(pos, idx);
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_ratio: Some(0.1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.mesh.triangle_count() <= m.triangle_count() / 10 + 2,
            "{}",
            r.mesh.triangle_count()
        );
        // Its open border is still a border: corners and edges stay in place.
        let d = crate::metrics::surface_distance(&m, &r.mesh, 4000);
        assert!(d.relative_hausdorff < 0.02, "{d:?}");
    }

    #[test]
    fn protection_keeps_prominent_parts_when_stalled() {
        // Three tori far apart: past their minimal triangulations only deletion helps, and
        // deleting one removes a whole visible object.
        let mut m = Mesh::default();
        for k in 0..3 {
            m.append(
                &moved(torus(16, 8), Vec3::new(10.0 * k as f32, 0.0, 0.0), 1.0),
                0,
            );
        }
        let run = |protect: f32| {
            simplify(
                &m,
                &SimplifyOptions {
                    target_triangles: Some(20),
                    part_protection: protect,
                    ..Default::default()
                },
            )
            .unwrap()
            .mesh
        };
        let kept = run(0.05);
        assert_eq!(crate::reduce::parts::parts_of(&kept).1, 3);
        let cut = run(0.0);
        assert!(crate::reduce::parts::parts_of(&cut).1 < 3);
    }

    #[test]
    fn stalled_parts_are_removed_to_reach_the_target() {
        let m = scene(40, false);
        let target = 300;
        let opts = SimplifyOptions {
            target_triangles: Some(target),
            ..Default::default()
        };
        let r = simplify(&m, &opts).unwrap();
        assert!(
            r.mesh.triangle_count() <= target,
            "{}",
            r.mesh.triangle_count()
        );
        let kept = simplify(
            &m,
            &SimplifyOptions {
                remove_parts: false,
                ..opts.clone()
            },
        )
        .unwrap();
        assert!(
            kept.mesh.triangle_count() > target,
            "parts alone should stall: {} parts {}",
            kept.mesh.triangle_count(),
            crate::reduce::parts::parts_of(&kept.mesh).1
        );
        // Every input vertex maps to a real output vertex.
        assert!(r
            .vertex_remap
            .iter()
            .all(|&v| (v as usize) < r.mesh.vertex_count()));
        r.mesh.validate().unwrap();
        // Same output on a second run.
        assert_eq!(simplify(&m, &opts).unwrap().mesh, r.mesh);
    }

    #[test]
    fn far_part_survives_and_locked_parts_stay() {
        let m = scene(40, true);
        let far_vertex = m.vertex_count() - 1;
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(300),
                ..Default::default()
            },
        )
        .unwrap();
        let far = |mesh: &Mesh| mesh.positions.iter().any(|p| p.x > 2.5);
        assert!(
            far(&r.mesh),
            "the isolated part is the most expensive to remove"
        );
        // Locking one vertex of each small part keeps all of them.
        let mut locked = vec![false; m.vertex_count()];
        let small_verts = torus(12, 6).vertex_count();
        let base = tests::uv_sphere(48, 24).vertex_count();
        for i in 0..41 {
            locked[base + i * small_verts] = true;
        }
        let r = simplify(
            &m,
            &SimplifyOptions {
                target_triangles: Some(300),
                locked_vertices: Some(locked),
                ..Default::default()
            },
        )
        .unwrap();
        // Every locked part is still there (each locked vertex survives as an exact copy).
        let kept: std::collections::HashSet<u32> = r.source_vertex.iter().copied().collect();
        for i in 0..41 {
            assert!(
                kept.contains(&((base + i * small_verts) as u32)),
                "part {i} removed"
            );
        }
        let _ = far_vertex;
    }

    #[test]
    fn duplicated_shell_loses_exactly_one_copy() {
        // Two identical spheres: each is free to remove while the other exists, but not both.
        let mut m = tests::uv_sphere(32, 16);
        let copy = m.clone();
        m.append(&copy, 0);
        let r = simplify(&m, &SimplifyOptions::ratio(0.25)).unwrap();
        assert_eq!(crate::reduce::parts::parts_of(&r.mesh).1, 1);
        let d = crate::metrics::surface_distance(&copy, &r.mesh, 5000);
        assert!(d.relative_hausdorff < 0.05, "{d:?}");
    }
}
