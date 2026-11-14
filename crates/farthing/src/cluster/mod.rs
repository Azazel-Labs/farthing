//! Cluster LOD: a DAG of small triangle clusters for per-cluster detail selection at run time
//! (the scheme Nanite made popular, Karis 2021).
//!
//! [`build_cluster_dag`] splits a mesh into clusters of at most `max_triangles` triangles and
//! `max_vertices` vertices, then repeats:
//!
//! 1. group neighbouring clusters (by shared border, `group_size` per group);
//! 2. reduce each group to half its triangles with [`crate::reduce::simplify`], its border with
//!    other groups locked, so neighbours still meet exactly;
//! 3. split the reduced group into new, coarser clusters.
//!
//! Groups whose border is locked at one level are interior at the next, so no edge stays
//! locked for long. Every output cluster indexes the *source* mesh's vertices (the reducer
//! only collapses, it never invents vertices), so all levels share one vertex buffer.
//!
//! Each cluster carries its own bounds and error and its parent group's. Both only grow
//! towards the roots (a group's sphere encloses its children's, and its error is at least
//! theirs), so the test in [`ClusterDag::select`] — draw a cluster when its own error is
//! acceptable and its parent's is not — picks a crack-free cut for any camera, and a GPU can
//! run it per cluster with no traversal. A normal cone per cluster supports back-face culling.

#![forbid(unsafe_code)]

use crate::reduce::{simplify, SimplifyOptions};
use crate::{Error, Mesh, Result};
use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Range;

pub mod gpu;
pub use gpu::{decode_gpu, encode_gpu, GpuClusterDag, GpuOptions};

/// A bounding sphere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    /// A small enclosing sphere of `points` (the smaller of Ritter's sphere and the box sphere).
    pub fn from_points(points: &[Vec3]) -> Sphere {
        let Some(&first) = points.first() else {
            return Sphere::default();
        };
        let far = |from: Vec3| {
            points.iter().copied().fold(from, |a, p| {
                if p.distance_squared(from) > a.distance_squared(from) {
                    p
                } else {
                    a
                }
            })
        };
        let x = far(first);
        let y = far(x);
        let mut c = (x + y) * 0.5;
        let mut r = x.distance(y) * 0.5;
        for &p in points {
            let d = p.distance(c);
            if d > r {
                let nr = (r + d) * 0.5;
                c += (p - c) * ((nr - r) / d);
                r = nr;
            }
        }
        let b = crate::Aabb::from_points(points.iter());
        let bc = b.center();
        let br = points.iter().map(|p| p.distance(bc)).fold(0.0f32, f32::max);
        let s = if br < r {
            Sphere {
                center: bc,
                radius: br,
            }
        } else {
            Sphere {
                center: c,
                radius: r,
            }
        };
        // Float slack so every point tests inside.
        Sphere {
            radius: s.radius * (1.0 + 1e-6) + 1e-7,
            ..s
        }
    }

    /// A small sphere enclosing all of `spheres`.
    pub fn enclose(spheres: &[Sphere]) -> Sphere {
        let Some(start) = spheres
            .iter()
            .copied()
            .reduce(|a, b| if b.radius > a.radius { b } else { a })
        else {
            return Sphere::default();
        };
        let (mut c, mut r) = (start.center, start.radius);
        for _ in 0..2 {
            for s in spheres {
                let d = s.center.distance(c);
                if d + s.radius > r {
                    let nr = (r + d + s.radius) * 0.5;
                    if d > 0.0 {
                        c += (s.center - c) * ((nr - r) / d);
                    }
                    r = nr;
                }
            }
        }
        let r = spheres
            .iter()
            .map(|s| s.center.distance(c) + s.radius)
            .fold(r, f32::max);
        Sphere {
            center: c,
            radius: r * (1.0 + 1e-6) + 1e-7,
        }
    }

    /// Whether `other` lies inside (with a relative tolerance).
    pub fn contains(&self, other: &Sphere) -> bool {
        self.center.distance(other.center) + other.radius <= self.radius * (1.0 + 1e-4) + 1e-6
    }
}

/// A normal cone for back-face culling: the whole cluster faces away from a camera at `eye`
/// when `dot(normalize(apex - eye), axis) >= cutoff`. `cutoff > 1` means never culled (the
/// normals spread too far).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cone {
    pub apex: Vec3,
    pub axis: Vec3,
    pub cutoff: f32,
}

impl Cone {
    /// Whether every triangle of the cluster faces away from `eye`.
    pub fn backfacing(&self, eye: Vec3) -> bool {
        self.cutoff <= 1.0 && (self.apex - eye).normalize_or_zero().dot(self.axis) >= self.cutoff
    }
}

/// One cluster of the DAG.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cluster {
    /// Three indices per triangle into the source mesh's vertices.
    pub indices: Vec<u32>,
    /// Material slot per triangle.
    pub material_ids: Vec<u32>,
    /// 0 for the source triangles, +1 per reduction.
    pub level: u32,
    /// Culling bounds of this cluster's triangles.
    pub bounds: Sphere,
    pub cone: Cone,
    /// Error (world units) and bounds of the group that produced this cluster; 0 and the
    /// cluster's own bounds at level 0.
    pub lod_error: f32,
    pub lod_bounds: Sphere,
    /// Error and bounds of the group that reduced this cluster further; infinite error for
    /// roots.
    pub parent_error: f32,
    pub parent_bounds: Sphere,
    /// Index of the producing group (`u32::MAX` at level 0) and of the consuming group
    /// (`u32::MAX` for roots).
    pub group: u32,
    pub parent_group: u32,
}

impl Cluster {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The meshlet layout: the cluster's unique vertices (source indices) and three local
    /// `u8` indices per triangle. Requires `max_vertices <= 256`.
    pub fn meshlet(&self) -> (Vec<u32>, Vec<u8>) {
        let mut verts: Vec<u32> = Vec::new();
        let local = self
            .indices
            .iter()
            .map(|&v| match verts.iter().position(|&w| w == v) {
                Some(i) => i as u8,
                None => {
                    verts.push(v);
                    (verts.len() - 1) as u8
                }
            })
            .collect();
        (verts, local)
    }
}

/// A group: clusters reduced together, and the clusters the reduction produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterGroup {
    /// Level of the clusters it consumed.
    pub level: u32,
    pub children: Vec<u32>,
    pub clusters: Vec<u32>,
    pub error: f32,
    pub bounds: Sphere,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterOptions {
    /// Triangles per cluster (Nanite uses 128).
    pub max_triangles: usize,
    /// Unique vertices per cluster (255 fits `u8` local indices; mesh-shader pipelines often
    /// want 64 with 124 triangles).
    pub max_vertices: usize,
    /// Clusters reduced together. Larger groups lock fewer edges but cost more per level.
    pub group_size: usize,
    /// A group whose reduction keeps more than this fraction of its triangles is not
    /// reduced; its clusters move on to the next level to be grouped differently.
    pub min_reduction: f32,
    /// Reduction options for each group (targets and locks are set per group).
    pub simplify: SimplifyOptions,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        ClusterOptions {
            max_triangles: 128,
            max_vertices: 255,
            group_size: 8,
            min_reduction: 0.85,
            simplify: SimplifyOptions {
                // A group's reduction should be exact to its target; the parts it deletes are
                // the ones whose removal costs least, which is what a LOD wants.
                ..SimplifyOptions::default()
            },
        }
    }
}

impl ClusterOptions {
    /// About twice as fast to build, with [`crate::reduce::Quality::Fast`] group reductions:
    /// geometric error stays as low (in our benchmarks often lower), attributes (normals, UVs)
    /// are kept less faithfully.
    pub fn fast() -> Self {
        let mut o = ClusterOptions::default();
        o.simplify.quality = crate::reduce::Quality::Fast;
        o
    }
}

/// The cluster DAG of one mesh.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterDag {
    pub clusters: Vec<Cluster>,
    pub groups: Vec<ClusterGroup>,
}

/// What [`ClusterDag::select`] needs to know about the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodView {
    pub eye: Vec3,
    /// Projection scale: viewport height in pixels / (2 tan(vertical fov / 2)).
    pub pixels_per_radian: f32,
    /// Largest acceptable error on screen, in pixels.
    pub threshold: f32,
}

impl LodView {
    /// Projected size of `error` at `bounds`' nearest point, in pixels.
    pub fn projected(&self, error: f32, bounds: &Sphere) -> f32 {
        if error <= 0.0 {
            return 0.0;
        }
        if !error.is_finite() {
            return f32::INFINITY;
        }
        let d = bounds.center.distance(self.eye) - bounds.radius;
        if d <= 0.0 {
            f32::INFINITY
        } else {
            error * self.pixels_per_radian / d
        }
    }
}

impl ClusterDag {
    pub fn levels(&self) -> u32 {
        self.clusters.iter().map(|c| c.level + 1).max().unwrap_or(0)
    }

    /// Triangles per level (all clusters of each level).
    pub fn level_triangles(&self) -> Vec<usize> {
        let mut t = vec![0; self.levels() as usize];
        for c in &self.clusters {
            t[c.level as usize] += c.triangle_count();
        }
        t
    }

    pub fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.clusters.len()).filter(|&c| self.clusters[c].parent_group == u32::MAX)
    }

    /// Clusters to draw: those whose own error is acceptable and whose parent's is not.
    pub fn select(&self, view: &LodView) -> Vec<u32> {
        (0..self.clusters.len() as u32)
            .filter(|&i| {
                let c = &self.clusters[i as usize];
                view.projected(c.lod_error, &c.lod_bounds) <= view.threshold
                    && view.projected(c.parent_error, &c.parent_bounds) > view.threshold
            })
            .collect()
    }

    /// The selected clusters as one mesh over the source's vertices (unused ones dropped).
    pub fn to_mesh(&self, source: &Mesh, selection: &[u32]) -> Mesh {
        let mut m = source.select_vertices(&(0..source.vertex_count() as u32).collect::<Vec<_>>());
        for &c in selection {
            let c = &self.clusters[c as usize];
            m.indices.extend_from_slice(&c.indices);
            m.material_ids.extend_from_slice(&c.material_ids);
        }
        m.compact();
        m
    }
}

/// Builds the cluster DAG of `mesh`.
pub fn build_cluster_dag(mesh: &Mesh, opts: &ClusterOptions) -> Result<ClusterDag> {
    mesh.validate()?;
    if opts.max_triangles == 0 || opts.max_vertices < 3 || opts.group_size < 2 {
        return Err(Error::InvalidInput(
            "max_triangles ≥ 1, max_vertices ≥ 3 and group_size ≥ 2 required".into(),
        ));
    }
    let pos_id = weld(mesh);
    let npos = pos_id.iter().map(|&p| p as usize + 1).max().unwrap_or(0);
    // Triangles degenerate by position are dropped (they'd never render and confuse growth).
    let (tris, mats): (Vec<[u32; 3]>, Vec<u32>) = (0..mesh.triangle_count())
        .into_par_iter()
        .filter_map(|t| {
            let tri = mesh.triangle(t);
            let p = tri.map(|v| pos_id[v as usize]);
            (p[0] != p[1] && p[1] != p[2] && p[0] != p[2]).then(|| (tri, mesh.material_ids[t]))
        })
        .unzip();
    let ctx = Ctx {
        mesh,
        pos_id: &pos_id,
        opts,
    };

    let mut dag = ClusterDag::default();
    let all: Vec<u32> = (0..tris.len() as u32).collect();
    dag.clusters = ctx
        .make_clusters(&tris, &all)
        .into_par_iter()
        .map(|cl| {
            let t: Vec<[u32; 3]> = cl.iter().map(|&i| tris[i as usize]).collect();
            let m: Vec<u32> = cl.iter().map(|&i| mats[i as usize]).collect();
            ctx.cluster(&t, m, 0, u32::MAX, 0.0, None)
        })
        .collect();
    let mut current: Vec<u32> = (0..dag.clusters.len() as u32).collect();
    let mut level = 0u32;
    while current.len() > 1 {
        let groups = ctx.group_clusters(&dag, &current, npos);
        // Positions shared by clusters of different groups stay put.
        let mut pos_group = vec![u32::MAX; npos];
        let mut border = vec![false; npos];
        for (g, members) in groups.iter().enumerate() {
            for &c in members {
                for &v in &dag.clusters[c as usize].indices {
                    let p = pos_id[v as usize] as usize;
                    if pos_group[p] == u32::MAX {
                        pos_group[p] = g as u32;
                    } else if pos_group[p] != g as u32 {
                        border[p] = true;
                    }
                }
            }
        }
        // Reduce each group and re-cluster its output, all in parallel; ids are assigned after.
        let dag_ref = &dag;
        let built: Vec<Result<Option<(f32, Sphere, Vec<Cluster>)>>> = groups
            .par_iter()
            .map(|members| {
                if members.len() < 2 && current.len() > 1 && groups.len() > 1 {
                    // A lone cluster gains nothing from reduction on its own border.
                    return Ok(None);
                }
                let Some((t, m, err)) = ctx.reduce_group(dag_ref, members, &border)? else {
                    return Ok(None);
                };
                let child_bounds: Vec<Sphere> = members
                    .iter()
                    .map(|&c| dag_ref.clusters[c as usize].lod_bounds)
                    .collect();
                let bounds = Sphere::enclose(&child_bounds);
                let error = members
                    .iter()
                    .map(|&c| dag_ref.clusters[c as usize].lod_error)
                    .fold(err, f32::max);
                let idx: Vec<u32> = (0..t.len() as u32).collect();
                let clusters = ctx
                    .make_clusters(&t, &idx)
                    .into_iter()
                    .map(|cl| {
                        let ct: Vec<[u32; 3]> = cl.iter().map(|&i| t[i as usize]).collect();
                        let cm: Vec<u32> = cl.iter().map(|&i| m[i as usize]).collect();
                        ctx.cluster(&ct, cm, level + 1, u32::MAX, error, Some(bounds))
                    })
                    .collect();
                Ok(Some((error, bounds, clusters)))
            })
            .collect();
        let mut next = Vec::new();
        let mut progress = false;
        for (members, r) in groups.iter().zip(built) {
            let Some((error, bounds, clusters)) = r? else {
                next.extend_from_slice(members);
                continue;
            };
            progress = true;
            let gi = dag.groups.len() as u32;
            for &c in members {
                let c = &mut dag.clusters[c as usize];
                c.parent_error = error;
                c.parent_bounds = bounds;
                c.parent_group = gi;
            }
            let mut produced = Vec::with_capacity(clusters.len());
            for mut c in clusters {
                c.group = gi;
                produced.push(dag.clusters.len() as u32);
                next.push(dag.clusters.len() as u32);
                dag.clusters.push(c);
            }
            dag.groups.push(ClusterGroup {
                level,
                children: members.clone(),
                clusters: produced,
                error,
                bounds,
            });
        }
        if !progress {
            break;
        }
        current = next;
        level += 1;
    }
    Ok(dag)
}

struct Ctx<'a> {
    mesh: &'a Mesh,
    pos_id: &'a [u32],
    opts: &'a ClusterOptions,
}

impl Ctx<'_> {
    fn cluster(
        &self,
        tris: &[[u32; 3]],
        material_ids: Vec<u32>,
        level: u32,
        group: u32,
        lod_error: f32,
        lod_bounds: Option<Sphere>,
    ) -> Cluster {
        let pts: Vec<Vec3> = tris
            .iter()
            .flatten()
            .map(|&v| self.mesh.positions[v as usize])
            .collect();
        let bounds = Sphere::from_points(&pts);
        Cluster {
            indices: tris.iter().flatten().copied().collect(),
            material_ids,
            level,
            bounds,
            cone: cone(&pts, bounds.center),
            lod_error,
            lod_bounds: lod_bounds.unwrap_or(bounds),
            parent_error: f32::INFINITY,
            parent_bounds: bounds,
            group,
            parent_group: u32::MAX,
        }
    }

    fn centroid(&self, t: &[u32; 3]) -> Vec3 {
        t.iter()
            .map(|&v| self.mesh.positions[v as usize])
            .sum::<Vec3>()
            / 3.0
    }

    /// Splits triangles `sel` of `tris` into clusters: spatial blocks of 16 clusters' worth,
    /// each grown greedily (in parallel; output order is fixed).
    fn make_clusters(&self, tris: &[[u32; 3]], sel: &[u32]) -> Vec<Vec<u32>> {
        let cent: Vec<Vec3> = sel
            .iter()
            .map(|&i| self.centroid(&tris[i as usize]))
            .collect();
        let mut order: Vec<u32> = (0..sel.len() as u32).collect();
        let mut blocks = Vec::new();
        bisect(
            &mut order,
            &cent,
            0..sel.len(),
            self.opts.max_triangles * 16,
            &mut blocks,
        );
        blocks
            .into_par_iter()
            .map(|r| {
                let local: Vec<u32> = order[r].iter().map(|&i| sel[i as usize]).collect();
                self.grow_block(tris, &local)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }

    fn grow_block(&self, tris: &[[u32; 3]], block: &[u32]) -> Vec<Vec<u32>> {
        let (max_t, max_v) = (self.opts.max_triangles, self.opts.max_vertices);
        let m = block.len();
        let lt: Vec<[u32; 3]> = block.iter().map(|&i| tris[i as usize]).collect();
        // Block-local wedge and position numbering: ids are ranks among the distinct keys,
        // from one sort of (key, corner) pairs.
        let local_ids = |key: &dyn Fn(u32) -> u32| {
            let mut all: Vec<u64> = lt
                .iter()
                .flatten()
                .enumerate()
                .map(|(i, &v)| (u64::from(key(v)) << 32) | i as u64)
                .collect();
            all.sort_unstable();
            let mut ids = vec![[0u32; 3]; m];
            let (mut n, mut prev) = (0u32, u64::MAX);
            for &k in &all {
                if k >> 32 != prev {
                    prev = k >> 32;
                    n += 1;
                }
                let i = (k & 0xFFFF_FFFF) as usize;
                ids[i / 3][i % 3] = n - 1;
            }
            (ids, n as usize)
        };
        let (tv, nv) = local_ids(&|v| v);
        let (tp, np) = local_ids(&|v| self.pos_id[v as usize]);
        let mut start = vec![0u32; np + 1];
        for t in &tp {
            for &p in t {
                start[p as usize + 1] += 1;
            }
        }
        for i in 0..np {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut adj = vec![0u32; start[np] as usize];
        for (t, ps) in tp.iter().enumerate() {
            for &p in ps {
                adj[fill[p as usize] as usize] = t as u32;
                fill[p as usize] += 1;
            }
        }
        let cent: Vec<Vec3> = lt.iter().map(|t| self.centroid(t)).collect();
        let (normal, area): (Vec<Vec3>, Vec<f32>) = lt
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|v| self.mesh.positions[v as usize]);
                let n = (b - a).cross(c - a);
                (n.normalize_or_zero(), n.length() * 0.5)
            })
            .unzip();
        let mean_area = area.iter().sum::<f32>() / m.max(1) as f32;
        let radius = (max_t as f32 * mean_area / std::f32::consts::PI)
            .sqrt()
            .max(1e-12);
        let b = crate::Aabb::from_points(cent.iter());
        let axis = b.largest_axis();
        let mut sweep: Vec<u32> = (0..m as u32).collect();
        sweep.sort_unstable_by(|&a, &b| {
            cent[a as usize][axis]
                .total_cmp(&cent[b as usize][axis])
                .then(a.cmp(&b))
        });

        let mut assigned = vec![false; m];
        let mut vstamp = vec![u32::MAX; nv];
        let mut cstamp = vec![u32::MAX; m];
        let mut out = Vec::new();
        let mut cursor = 0;
        let mut taken = 0;
        loop {
            // Drop assigned triangles from the sweep once they are the majority, so the
            // nearest-loose-triangle walk below stays short (order is kept; everything before
            // the cursor is assigned).
            if taken * 2 > sweep.len() {
                sweep.retain(|&t| !assigned[t as usize]);
                (cursor, taken) = (0, 0);
            }
            while cursor < sweep.len() && assigned[sweep[cursor] as usize] {
                cursor += 1;
            }
            if cursor == sweep.len() {
                break;
            }
            let cid = out.len() as u32;
            let mut cur: Vec<u32> = Vec::with_capacity(max_t);
            let mut nverts = 0usize;
            let mut csum = Vec3::ZERO;
            let mut nsum = Vec3::ZERO;
            let mut cands: Vec<u32> = Vec::new();
            let mut next = Some(sweep[cursor]);
            while let Some(t) = next {
                let tu = t as usize;
                assigned[tu] = true;
                taken += 1;
                cur.push(t);
                for &v in &tv[tu] {
                    if vstamp[v as usize] != cid {
                        vstamp[v as usize] = cid;
                        nverts += 1;
                    }
                }
                csum += cent[tu];
                nsum += normal[tu];
                for &p in &tp[tu] {
                    for &t2 in &adj[start[p as usize] as usize..start[p as usize + 1] as usize] {
                        if !assigned[t2 as usize] && cstamp[t2 as usize] != cid {
                            cstamp[t2 as usize] = cid;
                            cands.push(t2);
                        }
                    }
                }
                if cur.len() >= max_t {
                    break;
                }
                let center = csum / cur.len() as f32;
                let ndir = nsum.normalize_or_zero();
                let new_verts = |t: usize, vstamp: &[u32]| {
                    tv[t].iter().filter(|&&v| vstamp[v as usize] != cid).count()
                };
                let mut best: Option<(f32, u32)> = None;
                let mut i = 0;
                while i < cands.len() {
                    let c = cands[i] as usize;
                    if assigned[c] {
                        cands.swap_remove(i);
                        continue;
                    }
                    i += 1;
                    let nvc = new_verts(c, &vstamp);
                    if nverts + nvc > max_v {
                        continue;
                    }
                    // The other two terms are non-negative (up to rounding): a candidate whose
                    // new-vertex count alone exceeds the best cost cannot win or tie.
                    if best.is_some_and(|(bc, _)| nvc as f32 > bc + 0.01) {
                        continue;
                    }
                    let cost = nvc as f32
                        + cent[c].distance(center) / radius
                        + 0.5 * (1.0 - normal[c].dot(ndir));
                    if best.is_none_or(|(bc, bt)| cost < bc || (cost == bc && (c as u32) < bt)) {
                        best = Some((cost, c as u32));
                    }
                }
                next = best.map(|b| b.1);
                if next.is_none() && nverts + 3 <= max_v {
                    // Nothing connected fits: take the nearest loose triangle of the block
                    // (foliage, debris; the block is spatially compact), so disconnected
                    // pieces still fill clusters.
                    // Walk the sweep order outwards from the center (sorted by the axis
                    // coordinate, so the squared axis gap bounds the distance from below),
                    // keeping the lowest index among the nearest.
                    let mut bd = f32::INFINITY;
                    let mut bt = u32::MAX;
                    let ca = center[axis];
                    let mid =
                        sweep.partition_point(|&t| cent[t as usize][axis].total_cmp(&ca).is_lt());
                    let mut visit = |t: u32| {
                        let gap = cent[t as usize][axis] - ca;
                        if gap * gap > bd {
                            return false;
                        }
                        if !assigned[t as usize] {
                            let d = cent[t as usize].distance_squared(center);
                            if d < bd || (d == bd && t < bt) {
                                bd = d;
                                bt = t;
                            }
                        }
                        true
                    };
                    for &t in &sweep[mid..] {
                        if !visit(t) {
                            break;
                        }
                    }
                    for &t in sweep[..mid].iter().rev() {
                        if !visit(t) {
                            break;
                        }
                    }
                    if bt != u32::MAX {
                        next = Some(bt);
                    }
                }
            }
            out.push(cur.iter().map(|&t| block[t as usize]).collect());
        }
        out
    }

    /// Partitions `current` clusters into groups of about `group_size` that share borders.
    fn group_clusters(&self, dag: &ClusterDag, current: &[u32], npos: usize) -> Vec<Vec<u32>> {
        let n = current.len();
        let gs = self.opts.group_size;
        // Unique positions per cluster, then the clusters at each position (sorted by index).
        let cps: Vec<Vec<u32>> = current
            .par_iter()
            .map(|&c| {
                let mut ps: Vec<u32> = dag.clusters[c as usize]
                    .indices
                    .iter()
                    .map(|&v| self.pos_id[v as usize])
                    .collect();
                ps.sort_unstable();
                ps.dedup();
                ps
            })
            .collect();
        let mut pstart = vec![0u32; npos + 1];
        for ps in &cps {
            for &p in ps {
                pstart[p as usize + 1] += 1;
            }
        }
        for i in 0..npos {
            pstart[i + 1] += pstart[i];
        }
        let mut fill = pstart.clone();
        let mut pcl = vec![0u32; pstart[npos] as usize];
        for (ci, ps) in cps.iter().enumerate() {
            for &p in ps {
                pcl[fill[p as usize] as usize] = ci as u32;
                fill[p as usize] += 1;
            }
        }
        drop(fill);
        // Neighbours with shared-position counts. A position joins at most its first 16
        // clusters (a vertex shared by hundreds of clusters says little about adjacency).
        let lists: Vec<Vec<(u32, u32)>> = (0..n as u32)
            .into_par_iter()
            .map(|ci| {
                let mut v: Vec<u32> = Vec::new();
                for &p in &cps[ci as usize] {
                    let run = &pcl[pstart[p as usize] as usize..pstart[p as usize + 1] as usize];
                    let run = &run[..run.len().min(16)];
                    if run.len() > 1 && run.contains(&ci) {
                        v.extend(run.iter().copied().filter(|&o| o != ci));
                    }
                }
                v.sort_unstable();
                let mut out: Vec<(u32, u32)> = Vec::new();
                for o in v {
                    match out.last_mut() {
                        Some(l) if l.0 == o => l.1 += 1,
                        _ => out.push((o, 1)),
                    }
                }
                out
            })
            .collect();
        drop((cps, pstart, pcl));
        let mut start = vec![0u32; n + 1];
        let mut nbr: Vec<(u32, u32)> = Vec::with_capacity(lists.iter().map(Vec::len).sum());
        for (c, l) in lists.iter().enumerate() {
            nbr.extend_from_slice(l);
            start[c + 1] = nbr.len() as u32;
        }

        let center: Vec<Vec3> = current
            .iter()
            .map(|&c| dag.clusters[c as usize].bounds.center)
            .collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut blocks = Vec::new();
        bisect(&mut order, &center, 0..n, gs * 16, &mut blocks);
        let mut block_of = vec![0u32; n];
        for (bi, r) in blocks.iter().enumerate() {
            for &c in &order[r.clone()] {
                block_of[c as usize] = bi as u32;
            }
        }
        let groups: Vec<Vec<Vec<u32>>> = blocks
            .par_iter()
            .enumerate()
            .map(|(bi, r)| {
                let members = &order[r.clone()];
                let b = crate::Aabb::from_points(members.iter().map(|&c| &center[c as usize]));
                let axis = b.largest_axis();
                let mut sweep = members.to_vec();
                sweep.sort_unstable_by(|&a, &b| {
                    center[a as usize][axis]
                        .total_cmp(&center[b as usize][axis])
                        .then(a.cmp(&b))
                });
                let mut taken = std::collections::HashSet::new();
                let mut weight: std::collections::HashMap<u32, u32> = Default::default();
                let mut out = Vec::new();
                for &seed in &sweep {
                    if taken.contains(&seed) {
                        continue;
                    }
                    let mut g = vec![seed];
                    taken.insert(seed);
                    weight.clear();
                    let mut csum = center[seed as usize];
                    let add_nbrs = |c: u32, weight: &mut std::collections::HashMap<u32, u32>| {
                        for &(o, w) in
                            &nbr[start[c as usize] as usize..start[c as usize + 1] as usize]
                        {
                            if block_of[o as usize] == bi as u32 {
                                *weight.entry(o).or_insert(0) += w;
                            }
                        }
                    };
                    add_nbrs(seed, &mut weight);
                    while g.len() < gs {
                        let gc = csum / g.len() as f32;
                        let mut best: Option<(u32, f32, u32)> = None;
                        for (&o, &w) in &weight {
                            if taken.contains(&o) {
                                continue;
                            }
                            let d = center[o as usize].distance_squared(gc);
                            let better = match best {
                                None => true,
                                Some((bw, bd, bo)) => {
                                    w > bw || (w == bw && (d < bd || (d == bd && o < bo)))
                                }
                            };
                            if better {
                                best = Some((w, d, o));
                            }
                        }
                        let pick = match best {
                            Some(b) => b.2,
                            None => {
                                // No neighbour left: the nearest free cluster of the block.
                                let near =
                                    sweep.iter().copied().filter(|c| !taken.contains(c)).min_by(
                                        |&a, &b| {
                                            center[a as usize]
                                                .distance_squared(gc)
                                                .total_cmp(&center[b as usize].distance_squared(gc))
                                                .then(a.cmp(&b))
                                        },
                                    );
                                match near {
                                    Some(c) => c,
                                    None => break,
                                }
                            }
                        };
                        taken.insert(pick);
                        g.push(pick);
                        csum += center[pick as usize];
                        add_nbrs(pick, &mut weight);
                    }
                    out.push(g.into_iter().map(|c| current[c as usize]).collect());
                }
                out
            })
            .collect();
        groups.into_iter().flatten().collect()
    }

    /// Reduces one group to half its triangles with its border locked. `None` when the
    /// reduction falls short of `min_reduction`.
    fn reduce_group(
        &self,
        dag: &ClusterDag,
        members: &[u32],
        border: &[bool],
    ) -> Result<Option<(Vec<[u32; 3]>, Vec<u32>, f32)>> {
        let mut idx: Vec<u32> = Vec::new();
        let mut mats: Vec<u32> = Vec::new();
        for &c in members {
            idx.extend_from_slice(&dag.clusters[c as usize].indices);
            mats.extend_from_slice(&dag.clusters[c as usize].material_ids);
        }
        let mut verts = idx.clone();
        verts.sort_unstable();
        verts.dedup();
        let mut sub = self.mesh.select_vertices(&verts);
        sub.indices = idx
            .iter()
            .map(|v| verts.binary_search(v).unwrap() as u32)
            .collect();
        sub.material_ids = mats;
        let tc = sub.triangle_count();
        let locked: Vec<bool> = verts
            .iter()
            .map(|&v| border[self.pos_id[v as usize] as usize])
            .collect();
        let o = SimplifyOptions {
            target_ratio: None,
            target_triangles: Some(tc / 2),
            locked_vertices: Some(locked),
            cluster_threshold: 0,
            ..self.opts.simplify.clone()
        };
        let r = simplify(&sub, &o)?;
        let out_tc = r.mesh.triangle_count();
        if out_tc as f32 > self.opts.min_reduction * tc as f32 {
            return Ok(None);
        }
        let tris = (0..out_tc)
            .map(|t| {
                r.mesh
                    .triangle(t)
                    .map(|i| verts[r.source_vertex[i as usize] as usize])
            })
            .collect();
        Ok(Some((tris, r.mesh.material_ids, r.error)))
    }
}

/// Splits `order[range]` at the median of `center` along the widest axis until each part
/// has at most `leaf` items. Ties break by index, so the result is deterministic.
fn bisect(
    order: &mut [u32],
    center: &[Vec3],
    range: Range<usize>,
    leaf: usize,
    out: &mut Vec<Range<usize>>,
) {
    let start = range.start;
    out.extend(bisect_slice(&mut order[range], center, start, leaf.max(1)));
}

/// [`bisect`] on a slice that starts at `start`; large halves split in parallel (the result
/// is the same on any thread count: each half only touches its own items).
fn bisect_slice(s: &mut [u32], center: &[Vec3], start: usize, leaf: usize) -> Vec<Range<usize>> {
    if s.len() <= leaf {
        return if s.is_empty() {
            Vec::new()
        } else {
            std::iter::once(start..start + s.len()).collect()
        };
    }
    let b = crate::Aabb::from_points(s.iter().map(|&i| &center[i as usize]));
    let axis = b.largest_axis();
    // Split at a multiple of the leaf size so the leaves come out full.
    let leaves = s.len().div_ceil(leaf);
    let mid = (leaves / 2) * leaf;
    s.select_nth_unstable_by(mid, |&a, &b| {
        center[a as usize][axis]
            .total_cmp(&center[b as usize][axis])
            .then(a.cmp(&b))
    });
    let big = s.len() > 16_384;
    let (l, r) = s.split_at_mut(mid);
    let (mut a, b) = if big {
        rayon::join(
            || bisect_slice(l, center, start, leaf),
            || bisect_slice(r, center, start + mid, leaf),
        )
    } else {
        (
            bisect_slice(l, center, start, leaf),
            bisect_slice(r, center, start + mid, leaf),
        )
    };
    a.extend(b);
    a
}

/// Position-welded id per vertex (bitwise-equal positions share an id).
fn weld(mesh: &Mesh) -> Vec<u32> {
    let key = |v: u32| mesh.positions[v as usize].to_array().map(f32::to_bits);
    let mut order: Vec<u32> = (0..mesh.vertex_count() as u32).collect();
    order.par_sort_unstable_by_key(|&v| (key(v), v));
    let mut id = vec![0u32; mesh.vertex_count()];
    let mut n = 0u32;
    for (i, &v) in order.iter().enumerate() {
        if i > 0 && key(order[i - 1]) != key(v) {
            n += 1;
        }
        id[v as usize] = n;
    }
    id
}

/// Normal cone of a triangle soup (three points per triangle) around `center`.
fn cone(pts: &[Vec3], center: Vec3) -> Cone {
    let never = Cone {
        apex: center,
        axis: Vec3::Z,
        cutoff: 2.0,
    };
    let normals: Vec<(Vec3, Vec3)> = pts
        .chunks_exact(3)
        .filter_map(|t| {
            let n = (t[1] - t[0]).cross(t[2] - t[0]).try_normalize()?;
            Some((n, t[0]))
        })
        .collect();
    let Some(axis) = normals.iter().map(|n| n.0).sum::<Vec3>().try_normalize() else {
        return never;
    };
    let mindp = normals.iter().map(|n| n.0.dot(axis)).fold(1.0f32, f32::min);
    if mindp <= 0.1 {
        return Cone { axis, ..never };
    }
    // Move the apex back along the axis until it is behind every triangle's plane.
    let maxt = normals
        .iter()
        .map(|&(n, p)| (center - p).dot(n) / n.dot(axis))
        .fold(0.0f32, f32::max);
    Cone {
        apex: center - axis * maxt,
        axis,
        cutoff: (1.0 - mindp * mindp).max(0.0).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A bumpy closed sphere with shared vertices (a projected subdivided cube).
    pub(super) fn blob(n: i32) -> Mesh {
        let mut ids: HashMap<[i32; 3], u32> = HashMap::new();
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        let mut vid = |q: [i32; 3], pos: &mut Vec<Vec3>| {
            *ids.entry(q).or_insert_with(|| {
                let p = Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32) / n as f32 * 2.0 - 1.0;
                let d = p.normalize();
                let bump = 1.0 + 0.08 * (5.0 * d.x).sin() * (4.0 * d.y).cos() * (3.0 * d.z).sin();
                pos.push(d * bump);
                (pos.len() - 1) as u32
            })
        };
        for axis in 0..3 {
            for side in [0, n] {
                for i in 0..n {
                    for j in 0..n {
                        let q = |a: i32, b: i32| {
                            let mut v = [0; 3];
                            v[axis] = side;
                            v[(axis + 1) % 3] = a;
                            v[(axis + 2) % 3] = b;
                            v
                        };
                        let c = [q(i, j), q(i + 1, j), q(i + 1, j + 1), q(i, j + 1)]
                            .map(|v| vid(v, &mut pos));
                        let quad = if side == 0 {
                            [c[0], c[2], c[1], c[0], c[3], c[2]]
                        } else {
                            [c[0], c[1], c[2], c[0], c[2], c[3]]
                        };
                        idx.extend_from_slice(&quad);
                    }
                }
            }
        }
        let mut m = Mesh::new(pos, idx);
        m.compute_normals(180.0);
        m
    }

    pub(super) fn edge_counts(m: &Mesh) -> HashMap<(u32, u32), u32> {
        let mut e = HashMap::new();
        for t in 0..m.triangle_count() {
            let [a, b, c] = m.triangle(t);
            for (x, y) in [(a, b), (b, c), (c, a)] {
                *e.entry((x.min(y), x.max(y))).or_insert(0) += 1;
            }
        }
        e
    }

    #[test]
    fn dag_is_bounded_monotone_and_crack_free() {
        let m = blob(40); // 9,600 triangles
        let opts = ClusterOptions::default();
        let dag = build_cluster_dag(&m, &opts).unwrap();
        let lt = dag.level_triangles();
        assert_eq!(lt[0], m.triangle_count());
        assert!(dag.levels() >= 5, "levels {lt:?}");
        assert!(*lt.last().unwrap() < 400, "coarsest level {lt:?}");
        for c in &dag.clusters {
            assert!(c.triangle_count() <= opts.max_triangles);
            assert!(c.meshlet().0.len() <= opts.max_vertices);
            if c.parent_group != u32::MAX {
                assert!(c.parent_error >= c.lod_error);
                assert!(c.parent_bounds.contains(&c.lod_bounds));
            }
            for p in c.indices.iter().map(|&v| m.positions[v as usize]) {
                assert!(c.bounds.center.distance(p) <= c.bounds.radius);
            }
        }
        // Level-0 cluster fill.
        let l0: Vec<_> = dag.clusters.iter().filter(|c| c.level == 0).collect();
        let fill = m.triangle_count() as f32 / (l0.len() * opts.max_triangles) as f32;
        assert!(fill > 0.85, "fill {fill}");
        // Any cut is a closed surface with each source triangle area covered once.
        let area = m.surface_area();
        let mut seen_levels = std::collections::BTreeSet::new();
        for dist in [1.5f32, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 400.0] {
            for dir in [Vec3::X, Vec3::new(0.3, 0.8, -0.5).normalize()] {
                let view = LodView {
                    eye: dir * dist,
                    pixels_per_radian: 1000.0,
                    threshold: 1.0,
                };
                let sel = dag.select(&view);
                assert!(!sel.is_empty());
                let cut = dag.to_mesh(&m, &sel);
                assert!(
                    edge_counts(&cut).values().all(|&k| k == 2),
                    "crack at {dist}"
                );
                let a = cut.surface_area();
                assert!((a - area).abs() < 0.1 * area, "area {a} vs {area}");
                seen_levels.extend(sel.iter().map(|&c| dag.clusters[c as usize].level));
            }
        }
        assert!(seen_levels.len() >= 4, "levels used {seen_levels:?}");
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let m = blob(24);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| build_cluster_dag(&m, &ClusterOptions::default()).unwrap())
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn cones_never_cull_a_front_face() {
        let m = blob(24);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let mut rng = crate::rng::Pcg32::new(7);
        let mut culled = 0;
        for _ in 0..200 {
            let eye = Vec3::new(rng.f32() - 0.5, rng.f32() - 0.5, rng.f32() - 0.5).normalize()
                * (1.5 + 6.0 * rng.f32());
            for c in &dag.clusters {
                if !c.cone.backfacing(eye) {
                    continue;
                }
                culled += 1;
                for t in c.indices.chunks_exact(3) {
                    let [a, b, d] = [t[0], t[1], t[2]].map(|v| m.positions[v as usize]);
                    let n = (b - a).cross(d - a);
                    assert!((a - eye).dot(n) >= -1e-5 * n.length(), "front face culled");
                }
            }
        }
        assert!(culled > 0);
    }

    #[test]
    fn loose_triangles_fill_clusters() {
        // 2,000 separate leaf quads in a cloud: clusters should still fill up.
        let mut rng = crate::rng::Pcg32::new(3);
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        for _ in 0..2000 {
            let c = Vec3::new(rng.f32(), rng.f32(), rng.f32()) * 10.0;
            let base = pos.len() as u32;
            pos.extend([
                c,
                c + Vec3::X * 0.2,
                c + Vec3::new(0.2, 0.2, 0.0),
                c + Vec3::Y * 0.2,
            ]);
            idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let m = Mesh::new(pos, idx);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let l0 = dag.clusters.iter().filter(|c| c.level == 0).count();
        assert!(l0 <= 4000 / 128 + 8, "{l0} clusters");
    }
}
