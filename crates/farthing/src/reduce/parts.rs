//! Deleting whole connected parts during reduction.
//!
//! Edge collapses never change topology: a closed part stops at its minimal triangulation
//! (4 triangles for a sphere-like part, about 14 for a ring), and a small open part stops when
//! its border locks it. An asset made of many parts therefore spends budget on small parts it
//! cannot simplify further, and at aggressive targets stalls well above the requested count.
//!
//! The removal error of a part is the largest distance from its visible surface samples to the
//! rest of the mesh that faces the same way (capped by the distance to any surface plus the
//! part's size), so decals, parts lying on a bigger surface, duplicated shells and parts hidden
//! inside others are cheap while a lone part in open space is expensive. The batched reducer deletes a
//! part as soon as its removal error drops below a quarter of the deviation the collapses have
//! already accepted, and, when collapses stall above the target, deletes the cheapest part and
//! carries on.

use crate::Mesh;
use crate::bvh::Bvh;
use crate::reduce::SimplifyResult;
use glam::Vec3;
use rayon::prelude::*;

/// Parts of the original input (connected by shared positions) with what removing each costs.
pub(crate) struct Parts {
    /// Input triangles of each part.
    part_tris: Vec<Vec<u32>>,
    /// (point, unit face normal) samples per part.
    samples: Vec<Vec<(Vec3, Vec3)>>,
    /// Unit face normals per part, in that part's BVH triangle order.
    normals: Vec<Vec<Vec3>>,
    boxes: Vec<(Vec3, Vec3)>,
    bvhs: Vec<Bvh>,
    /// All input triangles, for visibility rays, and the part of each.
    all: Bvh,
    paths: crate::bvh::Paths,
    tri_part: Vec<u32>,
    /// Per input triangle: blocks visibility rays (not a transparent material).
    occluder: Vec<bool>,
    reach: f32,
    /// Parts already removed (errors are measured against the others).
    pub removed: Vec<bool>,
    /// Parts not yet removed and not locked, keyed by a lower bound on their removal error
    /// (exact when last refined; removal only makes it grow).
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(Key, usize)>>,
    /// Removal count at which `err[k]` was last computed exactly.
    exact_at: Vec<Option<usize>>,
    removals: usize,
    /// Per part, what the visibility tests of its samples found (see [`Seen`]).
    seen: Vec<Seen>,
}

/// What earlier visibility tests found for one part's samples. Removing parts only frees
/// rays, so a sample once visible stays visible, and a ray blocked by a part that is still
/// kept is still blocked: a later test only casts the rays whose blocking part has gone.
/// The answers are the ones a fresh test would give.
#[derive(Default)]
struct Seen {
    /// Per sample.
    state: Vec<State>,
    /// Per direction of [`vis_dirs`], the triangle that last blocked a ray of this part's
    /// samples, tried first for the next sample (neighboring samples are usually blocked by
    /// the same triangle). Only a guess: any value gives the same answers.
    last: Vec<u32>,
}

/// What the visibility test of one sample found.
#[derive(Clone, Default)]
enum State {
    #[default]
    Unseen,
    Visible,
    /// Hidden: per outward direction it tests, the part that blocked that ray.
    Hidden(Box<[u32]>),
}

impl Parts {
    /// None for single-part meshes and when every part has a locked vertex.
    pub fn analyze(input: &Mesh, locked: Option<&[bool]>, transparent: &[u32]) -> Option<Parts> {
        let (tri_part, np) = parts_of(input);
        if np < 2 {
            return None;
        }
        let mut vpart = vec![usize::MAX; input.vertex_count()];
        let mut ids: Vec<Vec<u32>> = vec![Vec::new(); np];
        let mut samples: Vec<Vec<(Vec3, Vec3)>> = vec![Vec::new(); np];
        let mut normals: Vec<Vec<Vec3>> = vec![Vec::new(); np];
        let mut part_tris: Vec<Vec<[Vec3; 3]>> = vec![Vec::new(); np];
        let mut pinned = vec![false; np];
        for t in 0..input.triangle_count() {
            let k = tri_part[t];
            let tri = input.triangle_positions(t);
            let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
            for (i, v) in input.triangle(t).into_iter().enumerate() {
                if vpart[v as usize] == usize::MAX {
                    vpart[v as usize] = k;
                    samples[k].push((tri[i], n));
                    if locked.is_some_and(|l| l.get(v as usize) == Some(&true)) {
                        pinned[k] = true;
                    }
                }
            }
            samples[k].push(((tri[0] + tri[1] + tri[2]) / 3.0, n));
            part_tris[k].push(tri);
            normals[k].push(n);
            ids[k].push(t as u32);
        }
        // Parts with a locked vertex are never removed; with none left, there is nothing to
        // analyze (an empty queue behaves exactly like no parts at all).
        if pinned.iter().all(|&p| p) {
            return None;
        }
        // Samples per part: vertices and triangle centroids, at most about 256.
        for s in &mut samples {
            let step = s.len().div_ceil(256).max(1);
            if step > 1 {
                *s = s.iter().step_by(step).copied().collect();
            }
        }
        let boxes = part_tris
            .iter()
            .map(|ts| {
                ts.iter().flatten().fold(
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(lo, hi), &p| (lo.min(p), hi.max(p)),
                )
            })
            .collect();
        // The trees side by side (each is the same for any thread count).
        let (bvhs, all) = rayon::join(
            || part_tris.into_par_iter().map(Bvh::build).collect(),
            || {
                Bvh::build(
                    (0..input.triangle_count())
                        .map(|t| input.triangle_positions(t))
                        .collect(),
                )
            },
        );
        let mut parts = Parts {
            part_tris: ids,
            samples,
            normals,
            boxes,
            bvhs,
            paths: all.paths(),
            all,
            tri_part: tri_part.iter().map(|&k| k as u32).collect(),
            occluder: input
                .material_ids
                .iter()
                .map(|m| !transparent.contains(m))
                .collect(),
            reach: input.bounds().extent().length() * 2.0 + 1.0,
            removed: vec![false; np],
            heap: Default::default(),
            exact_at: vec![None; np],
            removals: 0,
            seen: Vec::new(),
        };
        let mut seen: Vec<Seen> = parts
            .samples
            .iter()
            .map(|s| Seen {
                state: vec![State::Unseen; s.len()],
                last: vec![u32::MAX; vis_dirs().len()],
            })
            .collect();
        let coarse: Vec<f32> = {
            let p = &parts;
            // Pinned parts never enter the queue: their errors are not needed.
            seen.par_iter_mut()
                .enumerate()
                .map(|(k, s)| {
                    if pinned[k] {
                        0.0
                    } else {
                        p.error_of(k, true, s)
                    }
                })
                .collect()
        };
        parts.seen = seen;
        parts.heap = (0..np)
            .filter(|&k| !pinned[k])
            .map(|k| std::cmp::Reverse((Key(coarse[k]), k)))
            .collect();
        Some(parts)
    }

    /// Removal error of part k: the largest distance from its visible samples (a ray escapes the
    /// other kept parts in some outward direction) to the parts still kept, counting only
    /// surface that faces the same way (within 60°; removing the outer of two back-to-back
    /// shells would otherwise look free while exposing inverted normals), but at most the
    /// distance to any kept surface plus the part's own diagonal. A part enclosed by
    /// others (a watch movement inside its case) costs nothing. One BVH per part, the others
    /// visited in order of box distance, so a query only touches parts whose box can still
    /// beat the best distance so far. Both the visible set and the distances only grow as parts
    /// go, so a value computed earlier is a lower bound.
    fn error_of(&self, k: usize, coarse: bool, seen: &mut Seen) -> f32 {
        self.distance_of(k, coarse, seen)
    }

    /// Whether a ray from sample (p, n) of part k escapes the parts still kept in one of the
    /// outward directions, given what its last test found (`state`, updated). Removing parts
    /// only frees rays, so a visible sample stays visible (see [`Seen`]).
    fn sample_visible(
        &self,
        k: usize,
        (p, n): (Vec3, Vec3),
        state: &mut State,
        last: &mut [u32],
    ) -> bool {
        let eps = self.reach * 1e-6;
        let o = p + n * eps;
        let blocks = |t: u32| {
            let j = self.tri_part[t as usize] as usize;
            j != k && !self.removed[j] && self.occluder[t as usize]
        };
        // The part whose triangle blocks the ray in direction `d` (index `di`), if any: the
        // triangle that blocked this direction last if the full search is sure to find a hit
        // (see `Bvh::reaches`), else what the search finds.
        let mut blocker = |di: usize, d: Vec3| {
            let t = last[di];
            if t != u32::MAX && blocks(t) && self.all.reaches(&self.paths, o, d, eps, self.reach, t)
            {
                return Some(self.tri_part[t as usize]);
            }
            let t = self
                .all
                .first_occluder_where(o, d, eps, self.reach, blocks)?;
            last[di] = t;
            Some(self.tri_part[t as usize])
        };
        let dirs = vis_dirs()
            .iter()
            .enumerate()
            .filter(|(_, d)| d.dot(n) > 0.2);
        match state {
            State::Visible => true,
            State::Unseen => {
                let mut found = Vec::new();
                for (di, &d) in dirs {
                    let Some(j) = blocker(di, d) else {
                        *state = State::Visible;
                        return true;
                    };
                    found.push(j);
                }
                *state = State::Hidden(found.into_boxed_slice());
                false
            }
            State::Hidden(found) => {
                for (slot, (di, &d)) in found.iter_mut().zip(dirs) {
                    if !self.removed[*slot as usize] {
                        continue;
                    }
                    let Some(j) = blocker(di, d) else {
                        *state = State::Visible;
                        return true;
                    };
                    *slot = j;
                }
                false
            }
        }
    }

    /// Largest distance from part k's samples to the parts still kept (see `error_of`).
    fn distance_of(&self, k: usize, coarse: bool, seen: &mut Seen) -> f32 {
        let np = self.bvhs.len();
        let box_gap = |a: (Vec3, Vec3), b: (Vec3, Vec3)| -> f32 {
            (a.0 - b.1).max(b.0 - a.1).max(Vec3::ZERO).length()
        };
        let mut others: Vec<(f32, usize)> = (0..np)
            .filter(|&j| j != k && !self.removed[j] && !self.bvhs[j].is_empty())
            .map(|j| (box_gap(self.boxes[k], self.boxes[j]), j))
            .collect();
        others.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let size = (self.boxes[k].1 - self.boxes[k].0).length();
        if coarse {
            // Every 8th sample, a lower bound on the full error (parts run side by side).
            let mut worst = 0.0f32;
            for (&s, state) in self.samples[k].iter().zip(&mut seen.state).step_by(8) {
                worst = worst.max(self.sample_error(k, &others, size, s, state, &mut seen.last));
                if worst >= self.reach {
                    break;
                }
            }
            worst
        } else {
            // The largest of the samples' errors, whichever thread finds it.
            let last = &seen.last;
            self.samples[k]
                .par_iter()
                .zip(seen.state.par_iter_mut())
                .map_init(
                    || last.clone(),
                    |last, (&s, state)| self.sample_error(k, &others, size, s, state, last),
                )
                .reduce(|| 0.0, f32::max)
        }
    }

    /// Removal error at one sample of part k (see `error_of`); 0 if no ray from it escapes.
    /// `others` lists the kept parts by box distance, `size` is the part's diagonal.
    fn sample_error(
        &self,
        k: usize,
        others: &[(f32, usize)],
        size: f32,
        (p, n): (Vec3, Vec3),
        state: &mut State,
        last: &mut [u32],
    ) -> f32 {
        // Samples hidden behind other parts (outside view from every direction) don't count.
        if !self.sample_visible(k, (p, n), state, last) {
            return 0.0;
        }
        let point_gap =
            |p: Vec3, b: (Vec3, Vec3)| -> f32 { (b.0 - p).max(p - b.1).max(Vec3::ZERO).length() };
        // Distance to any kept surface, plus the part's own size, bounds what removing it
        // can change: back-to-back shells (near, but large) stay protected by the facing
        // rule, while a small part's sides don't need same-facing surface nearby.
        let mut any = self.reach;
        for &(gap, j) in others {
            if gap >= any {
                break;
            }
            if point_gap(p, self.boxes[j]) >= any {
                continue;
            }
            if let Some(c) = self.bvhs[j].closest_point(p, any) {
                any = any.min(c.distance);
            }
        }
        let mut best = (any + size).min(self.reach);
        for &(gap, j) in others {
            if gap >= best {
                break;
            }
            if point_gap(p, self.boxes[j]) >= best {
                continue;
            }
            let nj = &self.normals[j];
            if let Some(c) =
                self.bvhs[j].closest_point_where(p, best, |t| nj[t as usize].dot(n) > 0.5)
            {
                best = best.min(c.distance);
            }
        }
        best
    }

    /// Input triangles of part k.
    pub fn triangles(&self, k: usize) -> &[u32] {
        &self.part_tris[k]
    }

    /// Takes the part with the smallest removal error, if that error is at most `threshold`
    /// and within `max_error`, marking it removed. Parts with a locked vertex, with no live
    /// triangles, or with half or more of the `alive` triangles are passed over. Errors are
    /// measured against the parts still kept; removal only increases them, so the stored lower
    /// bounds are refined lazily and the queue stays exact (two coincident shells cannot both
    /// go). `live` counts the live triangles among a part's input triangles. Parts whose removal
    /// would leave fewer than `floor` live triangles are passed over too. Returns the part and
    /// its error.
    pub fn take(
        &mut self,
        threshold: f32,
        max_error: Option<f32>,
        live: impl Fn(&[u32]) -> usize,
        alive: usize,
        floor: usize,
    ) -> Option<(usize, f32)> {
        let mut passed = Vec::new();
        let mut taken = None;
        while let Some(std::cmp::Reverse((Key(e), k))) = self.heap.pop() {
            if e > threshold || max_error.is_some_and(|m| e > m) {
                self.heap.push(std::cmp::Reverse((Key(e), k)));
                break;
            }
            let n = live(&self.part_tris[k]);
            if n == 0 {
                continue; // gone for good
            }
            let now = if self.exact_at[k] == Some(self.removals) {
                e
            } else {
                let mut seen = std::mem::take(&mut self.seen[k]);
                let now = self.error_of(k, false, &mut seen);
                self.seen[k] = seen;
                now
            };
            self.exact_at[k] = Some(self.removals);
            if now != e {
                // Coarse keys are estimates (visibility from fewer samples): requeue the exact
                // value and pop again.
                self.heap.push(std::cmp::Reverse((Key(now), k)));
                continue;
            }
            if 2 * n >= alive || alive - n < floor {
                passed.push(std::cmp::Reverse((Key(e), k)));
                continue;
            }
            self.removed[k] = true;
            self.removals += 1;
            taken = Some((k, e));
            break;
        }
        self.heap.extend(passed);
        taken
    }
}

/// 64 directions spread evenly over the sphere (Fibonacci lattice).
fn vis_dirs() -> &'static [Vec3; 64] {
    static DIRS: std::sync::OnceLock<[Vec3; 64]> = std::sync::OnceLock::new();
    DIRS.get_or_init(|| {
        let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
        std::array::from_fn(|i| {
            let z = 1.0 - (i as f32 + 0.5) * 2.0 / 64.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let a = golden * i as f32;
            // Slightly irrational tilt so no direction runs exactly along an axis.
            Vec3::new(r * a.cos(), r * a.sin(), z).normalize() + Vec3::new(1e-3, 2e-3, 3e-3)
        })
        .map(|d: Vec3| d.normalize())
    })
}

/// A totally ordered f32 for the removal queue.
#[derive(Clone, Copy)]
struct Key(f32);
impl PartialEq for Key {
    fn eq(&self, o: &Self) -> bool {
        self.0.total_cmp(&o.0).is_eq()
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Part index of every triangle (connected through shared positions) and the part count.
pub(crate) fn parts_of(m: &Mesh) -> (Vec<usize>, usize) {
    let mut ids: std::collections::HashMap<[u32; 3], u32> = Default::default();
    let pos_id: Vec<u32> = m
        .positions
        .iter()
        .map(|p| {
            let n = ids.len() as u32;
            *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n)
        })
        .collect();
    let mut parent: Vec<u32> = (0..ids.len() as u32).collect();
    fn find(parent: &mut [u32], x: u32) -> u32 {
        let mut r = x;
        while parent[r as usize] != r {
            r = parent[r as usize];
        }
        let mut y = x;
        while parent[y as usize] != r {
            let z = parent[y as usize];
            parent[y as usize] = r;
            y = z;
        }
        r
    }
    let nt = m.triangle_count();
    for t in 0..nt {
        let [a, b, c] = m.triangle(t).map(|v| pos_id[v as usize]);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[ra as usize] = rb;
        let (rb, rc) = (find(&mut parent, b), find(&mut parent, c));
        parent[rb as usize] = rc;
    }
    let roots: Vec<u32> = (0..nt)
        .map(|t| find(&mut parent, pos_id[m.triangle(t)[0] as usize]))
        .collect();
    let mut ids: Vec<u32> = roots.clone();
    ids.sort_unstable();
    ids.dedup();
    (
        roots
            .iter()
            .map(|r| ids.binary_search(r).unwrap())
            .collect(),
        ids.len(),
    )
}

/// Points every input vertex that no longer has an output vertex at the nearest corner of the
/// closest output triangle (so geomorphs collapse removed parts onto the remaining surface).
pub(crate) fn remap_orphans(input: &Mesh, result: &mut SimplifyResult) {
    let m = &result.mesh;
    if m.triangle_count() == 0
        || !result
            .vertex_remap
            .iter()
            .any(|&r| r as usize >= m.vertex_count())
    {
        return;
    }
    let bvh = Bvh::build(
        (0..m.triangle_count())
            .map(|t| m.triangle_positions(t))
            .collect(),
    );
    let reach = m
        .bounds()
        .extent()
        .length()
        .max(input.bounds().extent().length())
        * 4.0
        + 1.0;
    result
        .vertex_remap
        .par_iter_mut()
        .enumerate()
        .with_min_len(256)
        .for_each(|(v, r)| {
            if (*r as usize) < m.vertex_count() {
                return;
            }
            let p = input.positions[v];
            if let Some(c) = bvh.closest_point(p, reach) {
                let tri = m.triangle(c.triangle as usize);
                *r = *tri
                    .iter()
                    .min_by(|&&a, &&b| {
                        (m.positions[a as usize] - p)
                            .length_squared()
                            .total_cmp(&(m.positions[b as usize] - p).length_squared())
                    })
                    .unwrap();
            }
        });
}
