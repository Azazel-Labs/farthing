//! Binned-SAH bounding volume hierarchy over triangles.
//!
//! Supports nearest-hit ray casts (Möller–Trumbore), any-hit occlusion tests and
//! closest-point queries (Ericson, *Real-Time Collision Detection*, §5.1.5).

use crate::math::Aabb;
use glam::Vec3;

#[derive(Clone, Copy, Debug)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: first index into `order`; inner: index of the left child (right = left + 1).
    first: u32,
    /// Leaf triangle count; 0 for inner nodes.
    count: u32,
}

/// A ray hit. Barycentrics of the hit point are `(1 - u - v, u, v)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub triangle: u32,
    pub u: f32,
    pub v: f32,
}

impl Hit {
    pub fn barycentrics(&self) -> Vec3 {
        Vec3::new(1.0 - self.u - self.v, self.u, self.v)
    }
}

/// Result of a closest-point query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Closest {
    pub point: Vec3,
    pub triangle: u32,
    pub barycentrics: Vec3,
    pub distance: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
    pub triangles: Vec<[Vec3; 3]>,
}

const LEAF_SIZE: usize = 4;
const BINS: usize = 12;

impl Bvh {
    pub fn build(triangles: Vec<[Vec3; 3]>) -> Bvh {
        use rayon::prelude::*;
        let n = triangles.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        if n == 0 {
            return Bvh {
                nodes: Vec::new(),
                order,
                triangles,
            };
        }
        let centroids: Vec<Vec3> = triangles
            .par_iter()
            .with_min_len(4096)
            .map(|t| (t[0] + t[1] + t[2]) / 3.0)
            .collect();
        let tri_bounds: Vec<Aabb> = triangles
            .par_iter()
            .with_min_len(4096)
            .map(|t| Aabb::from_points(t.iter()))
            .collect();
        let ctx = Build {
            centroids: &centroids,
            bounds: &tri_bounds,
        };
        let nodes = ctx.subtree(&mut order, 0);
        Bvh {
            nodes,
            order,
            triangles,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::EMPTY, |n| Aabb {
            min: n.min,
            max: n.max,
        })
    }

    /// Nearest hit with `t` in `[t_min, t_max]`. Hits both faces.
    pub fn intersect(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> Option<Hit> {
        self.traverse(origin, dir, t_min, t_max, false, |_| true)
    }

    /// True if anything blocks the segment.
    pub fn occluded(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> bool {
        self.traverse(origin, dir, t_min, t_max, true, |_| true)
            .is_some()
    }

    /// True if a triangle accepted by `keep` (given its index in the input list) blocks the
    /// segment.
    pub fn occluded_where(
        &self,
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        t_max: f32,
        keep: impl Fn(u32) -> bool,
    ) -> bool {
        self.traverse(origin, dir, t_min, t_max, true, keep)
            .is_some()
    }

    /// Winding number of `origin` along the ray `dir`: over every triangle the ray crosses
    /// ahead, +1 where it leaves (direction along the face normal) and −1 where it enters.
    /// For closed, consistently oriented shells this is how many shells enclose the origin,
    /// nested and overlapping ones included.
    pub fn ray_winding(&self, origin: Vec3, dir: Vec3) -> i32 {
        if self.nodes.is_empty() {
            return 0;
        }
        let inv = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        let mut w = 0;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(top) = stack.pop() {
            let node = &self.nodes[top as usize];
            if slab(origin, inv, node.min, node.max, 0.0, f32::INFINITY).is_none() {
                continue;
            }
            if node.count > 0 {
                for &ti in &self.order[node.first as usize..(node.first + node.count) as usize] {
                    let tri = &self.triangles[ti as usize];
                    if let Some((t, _, _)) = ray_triangle(origin, dir, tri) {
                        if t > 0.0 {
                            let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
                            w += if n.dot(dir) > 0.0 { 1 } else { -1 };
                        }
                    }
                }
            } else {
                stack.push(node.first);
                stack.push(node.first + 1);
            }
        }
        w
    }

    fn traverse(
        &self,
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        mut t_max: f32,
        any: bool,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        let mut best: Option<Hit> = None;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(top) = stack.pop() {
            let node = &self.nodes[top as usize];
            if slab(origin, inv, node.min, node.max, t_min, t_max).is_none() {
                continue;
            }
            if node.count > 0 {
                for &ti in &self.order[node.first as usize..(node.first + node.count) as usize] {
                    if !keep(ti) {
                        continue;
                    }
                    if let Some((t, u, v)) = ray_triangle(origin, dir, &self.triangles[ti as usize])
                    {
                        if t >= t_min && t <= t_max {
                            t_max = t;
                            best = Some(Hit {
                                t,
                                triangle: ti,
                                u,
                                v,
                            });
                            if any {
                                return best;
                            }
                        }
                    }
                }
            } else {
                let (l, r) = (node.first, node.first + 1);
                let dl = slab(
                    origin,
                    inv,
                    self.nodes[l as usize].min,
                    self.nodes[l as usize].max,
                    t_min,
                    t_max,
                );
                let dr = slab(
                    origin,
                    inv,
                    self.nodes[r as usize].min,
                    self.nodes[r as usize].max,
                    t_min,
                    t_max,
                );
                // Push the farther child first so the nearer one is popped first.
                match (dl, dr) {
                    (Some(a), Some(b)) => {
                        if a <= b {
                            stack.push(r);
                            stack.push(l);
                        } else {
                            stack.push(l);
                            stack.push(r);
                        }
                    }
                    (Some(_), None) => stack.push(l),
                    (None, Some(_)) => stack.push(r),
                    _ => {}
                }
            }
        }
        best
    }

    /// Closest point on any triangle within `max_distance`.
    pub fn closest_point(&self, p: Vec3, max_distance: f32) -> Option<Closest> {
        self.closest_point_where(p, max_distance, |_| true)
    }

    /// Closest point on any triangle `keep` accepts (by triangle index), within `max_distance`.
    pub fn closest_point_where(
        &self,
        p: Vec3,
        max_distance: f32,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Closest> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best_d2 = max_distance * max_distance;
        let mut best: Option<Closest> = None;
        let mut stack: Vec<(u32, f32)> =
            vec![(0, box_dist2(p, self.nodes[0].min, self.nodes[0].max))];
        while let Some((ni, d2)) = stack.pop() {
            if d2 > best_d2 {
                continue;
            }
            let node = &self.nodes[ni as usize];
            if node.count > 0 {
                for &ti in &self.order[node.first as usize..(node.first + node.count) as usize] {
                    if !keep(ti) {
                        continue;
                    }
                    let tri = &self.triangles[ti as usize];
                    let (q, bary) = closest_on_triangle(p, tri[0], tri[1], tri[2]);
                    let dd = (q - p).length_squared();
                    if dd <= best_d2 {
                        best_d2 = dd;
                        best = Some(Closest {
                            point: q,
                            triangle: ti,
                            barycentrics: bary,
                            distance: dd.sqrt(),
                        });
                    }
                }
            } else {
                let (l, r) = (node.first, node.first + 1);
                let dl = box_dist2(p, self.nodes[l as usize].min, self.nodes[l as usize].max);
                let dr = box_dist2(p, self.nodes[r as usize].min, self.nodes[r as usize].max);
                if dl < dr {
                    stack.push((r, dr));
                    stack.push((l, dl));
                } else {
                    stack.push((l, dl));
                    stack.push((r, dr));
                }
            }
        }
        best
    }
}

#[inline]
fn slab(o: Vec3, inv: Vec3, min: Vec3, max: Vec3, t_min: f32, t_max: f32) -> Option<f32> {
    let t0 = (min - o) * inv;
    let t1 = (max - o) * inv;
    let tn = t0.min(t1);
    let tf = t0.max(t1);
    let enter = tn.max_element().max(t_min);
    let exit = tf.min_element().min(t_max);
    // NaN (0 * inf) compares false, which conservatively keeps the box.
    if enter <= exit || enter.is_nan() || exit.is_nan() {
        Some(enter)
    } else {
        None
    }
}

#[inline]
fn box_dist2(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    let d = (min - p).max(Vec3::ZERO).max(p - max);
    d.length_squared()
}

/// Möller–Trumbore, two-sided. Returns (t, u, v).
#[inline]
pub fn ray_triangle(o: Vec3, d: Vec3, tri: &[Vec3; 3]) -> Option<(f32, f32, f32)> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - tri[0];
    let u = s.dot(p) * inv;
    if !(-1e-7..=1.0 + 1e-7).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < -1e-7 || u + v > 1.0 + 1e-7 {
        return None;
    }
    Some((e2.dot(q) * inv, u, v))
}

/// Closest point on triangle (a, b, c) to p, with barycentrics (Ericson 5.1.5).
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Vec3::X);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Vec3::Y);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Vec3::new(1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Vec3::Z);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Vec3::new(1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Vec3::new(0.0, 1.0 - w, w));
    }
    let denom = 1.0 / (va + vb + vc);
    if !denom.is_finite() {
        return (a, Vec3::X); // degenerate triangle
    }
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Vec3::new(1.0 - v - w, v, w))
}

/// Inputs shared by every part of a build.
struct Build<'a> {
    centroids: &'a [Vec3],
    bounds: &'a [Aabb],
}

/// Below this many triangles a subtree is built on one thread.
const PAR_BUILD: usize = 8192;

impl Build<'_> {
    /// Bounds of `order` and, unless it should stay a leaf, the split: `order` is partitioned
    /// so its first `mid` triangles go left.
    fn split(&self, order: &mut [u32]) -> (Aabb, Option<usize>) {
        let count = order.len();
        let mut bounds = Aabb::EMPTY;
        let mut cb = Aabb::EMPTY;
        for &t in order.iter() {
            bounds = bounds.union(&self.bounds[t as usize]);
            cb.extend(self.centroids[t as usize]);
        }
        if count <= LEAF_SIZE {
            return (bounds, None);
        }
        let axis = cb.largest_axis();
        let (lo, hi) = (cb.min[axis], cb.max[axis]);
        if hi - lo < 1e-12 {
            return (bounds, None); // all centroids coincide: keep as leaf
        }
        // Bin triangles by centroid.
        let scale = BINS as f32 / (hi - lo);
        let bin_of = |c: Vec3| (((c[axis] - lo) * scale) as usize).min(BINS - 1);
        let mut bin_bounds = [Aabb::EMPTY; BINS];
        let mut bin_count = [0usize; BINS];
        for &t in order.iter() {
            let b = bin_of(self.centroids[t as usize]);
            bin_count[b] += 1;
            bin_bounds[b] = bin_bounds[b].union(&self.bounds[t as usize]);
        }
        // Sweep for the cheapest split plane.
        let mut left_area = [0.0f32; BINS];
        let mut left_n = [0usize; BINS];
        let (mut acc, mut accn) = (Aabb::EMPTY, 0usize);
        for i in 0..BINS {
            acc = acc.union(&bin_bounds[i]);
            accn += bin_count[i];
            left_area[i] = if acc.is_empty() {
                0.0
            } else {
                acc.surface_area()
            };
            left_n[i] = accn;
        }
        let (mut best_cost, mut best_split) = (f32::INFINITY, 0usize);
        let (mut racc, mut raccn) = (Aabb::EMPTY, 0usize);
        for i in (1..BINS).rev() {
            racc = racc.union(&bin_bounds[i]);
            raccn += bin_count[i];
            let ra = if racc.is_empty() {
                0.0
            } else {
                racc.surface_area()
            };
            let cost = left_area[i - 1] * left_n[i - 1] as f32 + ra * raccn as f32;
            if cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }
        let leaf_cost = bounds.surface_area() * count as f32;
        if best_cost >= leaf_cost && count <= 16 {
            return (bounds, None);
        }
        // Partition in place.
        let mut mid = 0;
        for i in 0..count {
            if bin_of(self.centroids[order[i] as usize]) < best_split {
                order.swap(i, mid);
                mid += 1;
            }
        }
        if mid == 0 || mid == count {
            mid = count / 2; // degenerate split: fall back to median by index
        }
        (bounds, Some(mid))
    }

    /// Nodes of the subtree over `order` (which starts at `base` in the full order), root
    /// first. Large subtrees build their halves in parallel; the tree is the same for any
    /// thread count.
    fn subtree(&self, order: &mut [u32], base: u32) -> Vec<Node> {
        if order.len() < PAR_BUILD {
            return self.subtree_serial(order, base);
        }
        let (bounds, split) = self.split(order);
        let leaf = Node {
            min: bounds.min,
            max: bounds.max,
            first: base,
            count: order.len() as u32,
        };
        let Some(mid) = split else { return vec![leaf] };
        let (lo, hi) = order.split_at_mut(mid);
        let (l, r) = rayon::join(
            || self.subtree(lo, base),
            || self.subtree(hi, base + mid as u32),
        );
        // Layout: root, both children, then the left and the right descendants.
        let nl = l.len() as u32;
        let map_l = |i: u32| if i == 0 { 1 } else { 2 + i };
        let map_r = |i: u32| if i == 0 { 2 } else { 1 + nl + i };
        let mut out = Vec::with_capacity(1 + l.len() + r.len());
        out.push(Node {
            first: 1,
            count: 0,
            ..leaf
        });
        let fix = |n: &Node, map: &dyn Fn(u32) -> u32| {
            if n.count == 0 {
                Node {
                    first: map(n.first),
                    ..*n
                }
            } else {
                *n
            }
        };
        out.push(fix(&l[0], &map_l));
        out.push(fix(&r[0], &map_r));
        out.extend(l[1..].iter().map(|n| fix(n, &map_l)));
        out.extend(r[1..].iter().map(|n| fix(n, &map_r)));
        out
    }

    fn subtree_serial(&self, order: &mut [u32], base: u32) -> Vec<Node> {
        let mut nodes = Vec::with_capacity(2 * order.len() / LEAF_SIZE + 1);
        nodes.push(Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            first: 0,
            count: order.len() as u32,
        });
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            let (first, count) = (nodes[ni].first as usize, nodes[ni].count as usize);
            let (bounds, split) = self.split(&mut order[first..first + count]);
            nodes[ni].min = bounds.min;
            nodes[ni].max = bounds.max;
            let Some(mid) = split else { continue };
            let left = nodes.len();
            nodes.push(Node {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                first: first as u32,
                count: mid as u32,
            });
            nodes.push(Node {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                first: (first + mid) as u32,
                count: (count - mid) as u32,
            });
            nodes[ni].first = left as u32;
            nodes[ni].count = 0;
            stack.push(left);
            stack.push(left + 1);
        }
        // Leaves index the full order.
        for n in &mut nodes {
            if n.count > 0 {
                n.first += base;
            }
        }
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg32;

    fn random_soup(n: usize, seed: u64) -> Vec<[Vec3; 3]> {
        let mut r = Pcg32::new(seed);
        (0..n)
            .map(|_| {
                let c = Vec3::new(r.f32(), r.f32(), r.f32()) * 10.0;
                [
                    c,
                    c + Vec3::new(r.f32(), r.f32(), r.f32()),
                    c + Vec3::new(r.f32(), r.f32(), r.f32()),
                ]
            })
            .collect()
    }

    #[test]
    fn ray_matches_brute_force() {
        let tris = random_soup(2000, 1);
        let bvh = Bvh::build(tris.clone());
        let mut r = Pcg32::new(7);
        for _ in 0..300 {
            let o = Vec3::new(r.f32(), r.f32(), r.f32()) * 12.0 - 1.0;
            let d = Vec3::new(r.f32() - 0.5, r.f32() - 0.5, r.f32() - 0.5).normalize();
            let brute = tris
                .iter()
                .filter_map(|t| ray_triangle(o, d, t).map(|h| h.0))
                .filter(|&t| t >= 0.0)
                .fold(f32::INFINITY, f32::min);
            let got = bvh
                .intersect(o, d, 0.0, f32::INFINITY)
                .map_or(f32::INFINITY, |h| h.t);
            assert!(
                (brute - got).abs() < 1e-3 || (brute.is_infinite() && got.is_infinite()),
                "{brute} vs {got}"
            );
        }
    }

    #[test]
    fn closest_matches_brute_force() {
        let tris = random_soup(1000, 3);
        let bvh = Bvh::build(tris.clone());
        let mut r = Pcg32::new(9);
        for _ in 0..200 {
            let p = Vec3::new(r.f32(), r.f32(), r.f32()) * 12.0 - 1.0;
            let brute = tris
                .iter()
                .map(|t| (closest_on_triangle(p, t[0], t[1], t[2]).0 - p).length())
                .fold(f32::INFINITY, f32::min);
            let got = bvh.closest_point(p, f32::INFINITY).unwrap().distance;
            assert!((brute - got).abs() < 1e-4);
        }
    }

    #[test]
    fn parallel_build_matches_brute_force_and_any_thread_count() {
        // Large enough that the top of the tree is built in parallel.
        let tris = random_soup(40_000, 5);
        let build_in = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| Bvh::build(tris.clone()))
        };
        let bvh = build_in(4);
        let one = build_in(1);
        assert_eq!(bvh.order, one.order);
        assert_eq!(bvh.nodes.len(), one.nodes.len());
        for (a, b) in bvh.nodes.iter().zip(&one.nodes) {
            assert_eq!(
                (a.min, a.max, a.first, a.count),
                (b.min, b.max, b.first, b.count)
            );
        }
        let mut r = Pcg32::new(11);
        for _ in 0..100 {
            let p = Vec3::new(r.f32(), r.f32(), r.f32()) * 12.0 - 1.0;
            let brute = tris
                .iter()
                .map(|t| (closest_on_triangle(p, t[0], t[1], t[2]).0 - p).length())
                .fold(f32::INFINITY, f32::min);
            let got = bvh.closest_point(p, f32::INFINITY).unwrap().distance;
            assert!((brute - got).abs() < 1e-4);
            let d = Vec3::new(r.f32() - 0.5, r.f32() - 0.5, r.f32() - 0.5).normalize();
            let brute = tris
                .iter()
                .filter_map(|t| ray_triangle(p, d, t).map(|h| h.0))
                .filter(|&t| t >= 0.0)
                .fold(f32::INFINITY, f32::min);
            let got = bvh
                .intersect(p, d, 0.0, f32::INFINITY)
                .map_or(f32::INFINITY, |h| h.t);
            assert!((brute - got).abs() < 1e-3 || (brute.is_infinite() && got.is_infinite()));
        }
    }

    #[test]
    fn closest_point_barycentrics_reconstruct() {
        let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::Y);
        for p in [
            Vec3::new(0.2, 0.2, 1.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(-1.0, 0.5, 0.3),
        ] {
            let (q, w) = closest_on_triangle(p, a, b, c);
            assert!((a * w.x + b * w.y + c * w.z - q).length() < 1e-5);
        }
    }
}
