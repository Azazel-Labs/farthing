//! Voxel remeshing and occluder generation.
//!
//! * [`remesh`] — watertight proxy generation: voxelize, classify inside/outside by flood fill
//!   (with gap closing kept only for the cavities behind leaks, and clipping planes), make the solid
//!   manifold, extract with surface nets, project back onto the source (fold-safe), smooth what
//!   could not be projected, then reduce.
//! * [`occluder`] — conservative occluder meshes: the same solid, eroded so the result sits
//!   inside the original, reduced to a few hundred triangles and verified.
//! * [`box_occluders`] — greedy inscribed axis-aligned boxes for engines with box occluders.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

pub mod grid;
mod sdf;

use crate::bvh::Bvh;
use crate::metrics::sample_surface;
use crate::reduce::{simplify, SimplifyOptions};
use crate::{Aabb, Error, Mesh, Result, Scene};
use glam::{Vec3, Vec4};
use grid::Grid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RemeshMode {
    /// Remesh the outer shell (default): everything not reachable from outside is solid.
    Outside,
    /// Remesh the inside of an enclosure: the region reachable from `seed` is air.
    Inside { seed: Vec3 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemeshOptions {
    /// Cells along the longest axis.
    pub resolution: usize,
    /// Seal gaps up to this many cells (morphological closing).
    pub close_gaps: usize,
    /// Outside mode: seal gaps per pocket. Closing seals leaks but also fills real crevices;
    /// of the cells it adds, only connected regions deeper than `close_gaps + 1` cells (the
    /// cavities behind leaks) are kept, so crevices stay open.
    #[serde(default = "default_true")]
    pub adaptive_gaps: bool,
    pub mode: RemeshMode,
    /// Planes (n.xyz, d): cells with n·p + d > 0 are forced empty (e.g. below the ground).
    pub clip_planes: Vec<Vec4>,
    pub smooth_iterations: usize,
    /// Snap vertices to the nearest source surface within 1.5 cells.
    pub project_to_surface: bool,
    pub target_triangles: Option<usize>,
    pub max_error: Option<f32>,
    pub hard_angle_deg: f32,
    /// Boundary cells take the sign of the source's signed distance at their centre instead
    /// of being solid whenever the surface touches them, except where that would lose a sheet
    /// thinner than a cell. The shell then sits on the surface instead of up to a cell outside
    /// it, and slots narrower than a cell stay open. Needs a consistently oriented source.
    #[serde(default = "default_true")]
    pub subvoxel: bool,
    /// Reducer settings for the final reduction (targets come from this struct). None: the
    /// defaults. Worst-case distance on thin parts improves with
    /// `error_normalization: 0.5` (dragon 1.40% → 1.24% Hausdorff) at some cost in mean error.
    #[serde(default)]
    pub simplify: Option<SimplifyOptions>,
}

impl Default for RemeshOptions {
    fn default() -> Self {
        RemeshOptions {
            resolution: 96,
            close_gaps: 1,
            adaptive_gaps: true,
            mode: RemeshMode::Outside,
            clip_planes: Vec::new(),
            smooth_iterations: 2,
            project_to_surface: true,
            target_triangles: Some(2000),
            max_error: None,
            hard_angle_deg: 60.0,
            subvoxel: true,
            simplify: None,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug)]
pub struct RemeshResult {
    pub mesh: Mesh,
    pub voxels: [usize; 3],
    pub cell_size: f32,
    pub extracted_triangles: usize,
}

struct Solid {
    grid: Grid,
    solid: Vec<bool>,
    /// Cells the source surface touches.
    surface: Vec<bool>,
    bvh: Bvh,
}

fn classify(
    scene: &Scene,
    resolution: usize,
    close_gaps: usize,
    mode: RemeshMode,
    clip: &[Vec4],
    pad: usize,
) -> Result<Solid> {
    let (tris, _) = scene.world_triangles();
    if tris.is_empty() {
        return Err(Error::InvalidInput("scene has no triangles".into()));
    }
    if !(4..=1024).contains(&resolution) {
        return Err(Error::InvalidInput(format!(
            "resolution {resolution} must be within 4..=1024"
        )));
    }
    let bounds = Aabb::from_points(tris.iter().flatten());
    let grid = Grid::around(bounds, resolution, pad.max(2 + close_gaps));
    let surface = grid.voxelize(&tris);
    let mut blocked = grid.dilate(&surface, close_gaps);
    let mut forced_empty = vec![false; grid.len()];
    if !clip.is_empty() {
        for i in 0..grid.len() {
            let [x, y, z] = grid.coords(i);
            let c = grid.center(x, y, z);
            if clip.iter().any(|p| p.truncate().dot(c) + p.w > 0.0) {
                blocked[i] = false;
                forced_empty[i] = true;
            }
        }
    }
    let passable: Vec<bool> = blocked.iter().map(|b| !b).collect();
    let bvh = Bvh::build(tris);
    let solid = match mode {
        RemeshMode::Outside => {
            let seeds: Vec<usize> = grid
                .border_cells()
                .into_iter()
                .chain((0..grid.len()).filter(|&i| forced_empty[i]))
                .collect();
            let outside = grid.flood(&passable, seeds);
            let solid: Vec<bool> = outside.iter().map(|o| !o).collect();
            grid.erode(&solid, close_gaps)
        }
        RemeshMode::Inside { seed } => {
            let c = grid
                .cell_of(seed)
                .ok_or_else(|| Error::InvalidInput("inside seed is outside the grid".into()))?;
            let air = grid.flood(&passable, [grid.idx(c[0], c[1], c[2])]);
            let air = grid.dilate(&air, close_gaps);
            air.iter().map(|a| !a).collect()
        }
    };
    let mut solid = solid;
    for i in 0..grid.len() {
        if forced_empty[i] {
            solid[i] = false;
        }
    }
    Ok(Solid {
        grid,
        solid,
        surface,
        bvh,
    })
}

fn smooth(mesh: &mut Mesh, iterations: usize, lambda: f32) {
    smooth_masked(mesh, iterations, lambda, None);
}

/// Laplacian smoothing; with `movable`, only those vertices move.
fn smooth_masked(mesh: &mut Mesh, iterations: usize, lambda: f32, movable: Option<&[bool]>) {
    let n = mesh.vertex_count();
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in 0..mesh.triangle_count() {
        let tri = mesh.triangle(t);
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            if !nbrs[a as usize].contains(&b) {
                nbrs[a as usize].push(b);
            }
            if !nbrs[b as usize].contains(&a) {
                nbrs[b as usize].push(a);
            }
        }
    }
    for _ in 0..iterations {
        let src = mesh.positions.clone();
        for v in 0..n {
            if nbrs[v].is_empty() || movable.is_some_and(|m| !m[v]) {
                continue;
            }
            let avg = nbrs[v].iter().map(|&u| src[u as usize]).sum::<Vec3>() / nbrs[v].len() as f32;
            mesh.positions[v] = src[v].lerp(avg, lambda);
        }
    }
}

/// Per-pocket gap sealing. `open` is the solid without closing, `sealed` with closing by
/// `close` cells (same grid). Closing adds cells of two kinds: thin fills of crevices narrower
/// than about `2 × close` cells, and whole cavities that the unsealed flood reached through a
/// leak. Each connected region of added cells is kept only if it is deep: some cell lies more
/// than `close + 1` steps from every cell that was not added (a crevice fill is at most about
/// `close` thick). The rest reverts to air, so crevices stay open where the input has no leak.
fn seal_pockets(grid: &Grid, open: &[bool], sealed: &[bool], close: usize) -> Vec<bool> {
    let n = grid.len();
    let added: Vec<bool> = (0..n).map(|i| sealed[i] && !open[i]).collect();
    if !added.iter().any(|&a| a) {
        return open.to_vec();
    }
    let [nx, ny, nz] = grid.dims;
    let nbrs = |i: usize| -> [Option<usize>; 6] {
        let [x, y, z] = grid.coords(i);
        [
            (x > 0).then(|| i - 1),
            (x + 1 < nx).then(|| i + 1),
            (y > 0).then(|| i - nx),
            (y + 1 < ny).then(|| i + nx),
            (z > 0).then(|| i - nx * ny),
            (z + 1 < nz).then(|| i + nx * ny),
        ]
    };
    // Depth of every added cell: BFS steps from the nearest non-added cell.
    let mut depth = vec![u32::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    for i in 0..n {
        if added[i] && nbrs(i).iter().any(|j| j.is_some_and(|j| !added[j])) {
            depth[i] = 1;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for j in nbrs(i).into_iter().flatten() {
            if added[j] && depth[j] == u32::MAX {
                depth[j] = depth[i] + 1;
                queue.push_back(j);
            }
        }
    }
    // Connected regions of added cells; keep the deep ones.
    let mut out = open.to_vec();
    let mut seen = vec![false; n];
    let mut region = Vec::new();
    for s in 0..n {
        if !added[s] || seen[s] {
            continue;
        }
        region.clear();
        seen[s] = true;
        queue.push_back(s);
        let mut deepest = 0;
        while let Some(i) = queue.pop_front() {
            region.push(i);
            deepest = deepest.max(depth[i]);
            for j in nbrs(i).into_iter().flatten() {
                if added[j] && !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        if deepest as usize > close + 1 {
            for &i in &region {
                out[i] = true;
            }
        }
    }
    out
}

/// Builds a watertight proxy of the whole scene (world space).
pub fn remesh(scene: &Scene, opts: &RemeshOptions) -> Result<RemeshResult> {
    let pad = 2 + opts.close_gaps;
    let classify_with = |close: usize| {
        classify(
            scene,
            opts.resolution,
            close,
            opts.mode,
            &opts.clip_planes,
            pad,
        )
    };
    let s = if opts.adaptive_gaps && opts.close_gaps > 0 && opts.mode == RemeshMode::Outside {
        // Same grid for both: closing seals leaks but also fills real crevices. Keep only the
        // cavities it seals, not the crevices it fills.
        let (open, sealed) = rayon::join(|| classify_with(0), || classify_with(opts.close_gaps));
        let (open, mut sealed) = (open?, sealed?);
        sealed.solid = seal_pockets(&sealed.grid, &open.solid, &sealed.solid, opts.close_gaps);
        sealed
    } else {
        classify_with(opts.close_gaps)?
    };
    let mut solid = s.solid.clone();
    if opts.subvoxel {
        let signed = sdf::Signed::new(&s.bvh);
        sdf::relabel_boundary(&s.grid, &mut solid, &s.surface, &signed);
    }
    s.grid.manifoldize(&mut solid);
    let mut mesh = s.grid.surface_nets(&solid);
    if mesh.triangle_count() == 0 {
        return Err(Error::InvalidInput(
            "remesh produced no surface (is the input closed off by clipping planes?)".into(),
        ));
    }
    let extracted = mesh.triangle_count();
    // Without projection, smooth the voxel staircase; with it, smooth only the vertices the
    // projection had to leave in place (smoothing first rounds off the corners the projection would otherwise recover).
    if !opts.project_to_surface {
        smooth(&mut mesh, opts.smooth_iterations, 0.5);
    }
    if opts.project_to_surface {
        let maxd = s.grid.cell * 1.5;
        let before = mesh.positions.clone();
        for p in &mut mesh.positions {
            if let Some(c) = s.bvh.closest_point(*p, maxd) {
                *p = c.point;
            }
        }
        // Several proxy vertices can land on the same source point (a convex corner or vertex
        // of the source, or both sides of a thin sheet); identical positions would weld into
        // non-manifold edges. Back each of them off the point towards where it came from, by a
        // sliver of a cell, and keep the voxel position for any that still coincide.
        let key = |p: Vec3| p.to_array().map(f32::to_bits);
        let mut first: HashMap<[u32; 3], usize> = HashMap::with_capacity(mesh.positions.len());
        let mut clash = vec![false; mesh.positions.len()];
        for (v, p) in mesh.positions.iter().enumerate() {
            if let Some(&w) = first.get(&key(*p)) {
                clash[v] = true;
                clash[w] = true;
            } else {
                first.insert(key(*p), v);
            }
        }
        let back_off = s.grid.cell * 0.02;
        for (v, &c) in clash.iter().enumerate() {
            if c {
                let p = mesh.positions[v];
                mesh.positions[v] = p + (before[v] - p).normalize_or_zero() * back_off;
            }
        }
        first.clear();
        let mut still = vec![false; mesh.positions.len()];
        for (v, p) in mesh.positions.iter().enumerate() {
            if let Some(&w) = first.get(&key(*p)) {
                still[v] = true;
                still[w] = true;
            } else {
                first.insert(key(*p), v);
            }
        }
        for (v, &c) in still.iter().enumerate() {
            if c {
                mesh.positions[v] = before[v];
            }
        }
        // Projection must not fold or flatten faces (both sides of a thin shell would land on
        // the same source sheet): undo it around any face it collapses or turns, rather than
        // deleting faces and opening holes.
        for _ in 0..4 {
            let mut reverted = false;
            for t in 0..mesh.triangle_count() {
                let tri = mesh.triangle(t).map(|v| v as usize);
                let n0 = (before[tri[1]] - before[tri[0]]).cross(before[tri[2]] - before[tri[0]]);
                let [a, b, c] = tri.map(|v| mesh.positions[v]);
                let n1 = (b - a).cross(c - a);
                let (l0, l1) = (n0.length(), n1.length());
                if l1 < l0 * 0.05 || n0.dot(n1) < 0.3 * l0 * l1 {
                    for v in tri {
                        if mesh.positions[v] != before[v] {
                            mesh.positions[v] = before[v];
                            reverted = true;
                        }
                    }
                }
            }
            if !reverted {
                break;
            }
        }
        let movable: Vec<bool> = mesh
            .positions
            .iter()
            .zip(&before)
            .map(|(p, b)| p == b)
            .collect();
        smooth_masked(&mut mesh, opts.smooth_iterations, 0.5, Some(&movable));
    }
    let mut mesh = if opts.target_triangles.is_some() || opts.max_error.is_some() {
        let so = SimplifyOptions {
            target_triangles: opts.target_triangles,
            max_error: opts.max_error,
            ..opts.simplify.clone().unwrap_or_default()
        };
        simplify(&mesh, &so)?.mesh
    } else {
        mesh
    };
    mesh.compute_normals(opts.hard_angle_deg);
    Ok(RemeshResult {
        mesh,
        voxels: s.grid.dims,
        cell_size: s.grid.cell,
        extracted_triangles: extracted,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccluderOptions {
    pub resolution: usize,
    pub close_gaps: usize,
    /// Cells to erode before extraction (≥ 1 keeps the occluder inside the source).
    pub erode: usize,
    pub target_triangles: usize,
    /// Accept at most this fraction of occluder sample points outside the source solid;
    /// erosion is increased (up to 4) until met.
    pub max_outside_fraction: f32,
}

impl Default for OccluderOptions {
    fn default() -> Self {
        OccluderOptions {
            resolution: 64,
            close_gaps: 1,
            erode: 1,
            target_triangles: 300,
            max_outside_fraction: 0.01,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OccluderResult {
    pub mesh: Mesh,
    /// Fraction of sample points on the occluder that fall outside the source solid.
    pub outside_fraction: f32,
    pub erode_used: usize,
}

/// Generates a conservative occluder mesh for the scene.
pub fn occluder(scene: &Scene, opts: &OccluderOptions) -> Result<OccluderResult> {
    let s = classify(
        scene,
        opts.resolution,
        opts.close_gaps,
        RemeshMode::Outside,
        &[],
        0,
    )?;
    let mut erode = opts.erode.max(1);
    let mut budget = opts.target_triangles;
    loop {
        let core = s.grid.erode(&s.solid, erode);
        if !core.iter().any(|&c| c) {
            return Err(Error::InvalidInput(format!(
                "nothing left after eroding {erode} cells; raise the resolution"
            )));
        }
        let mut mesh = s.grid.surface_nets(&core);
        smooth(&mut mesh, 2, 0.5); // shrinks slightly: stays conservative
        let so = SimplifyOptions {
            target_triangles: Some(budget),
            normal_weight: 0.0,
            uv_weight: 0.0,
            // Occluders must stay inside the source; silhouette emphasis pushed 0.85% of
            // samples outside (0.05% without it).
            silhouette_weight: 0.0,
            ..Default::default()
        };
        let mut mesh = simplify(&mesh, &so)?.mesh;
        // Pull vertices that reduction pushed outside the solid back in along -normal.
        let inside = |p: Vec3| {
            s.grid
                .cell_of(p)
                .is_some_and(|c| s.solid[s.grid.idx(c[0], c[1], c[2])])
        };
        mesh.normals = None;
        mesh.compute_normals(180.0);
        let normals = mesh.normals.take().unwrap();
        for (p, n) in mesh.positions.iter_mut().zip(&normals) {
            let mut steps = 0;
            while !inside(*p) && steps < 8 {
                *p -= *n * s.grid.cell * 0.5;
                steps += 1;
            }
        }
        let pts = sample_surface(&mesh, 4000, 3);
        let outside = pts
            .iter()
            .filter(|p| {
                s.grid
                    .cell_of(**p)
                    .is_none_or(|c| !s.solid[s.grid.idx(c[0], c[1], c[2])])
            })
            .count();
        let frac = outside as f32 / pts.len().max(1) as f32;
        let exhausted = erode >= 4 && budget >= opts.target_triangles * 8;
        if frac <= opts.max_outside_fraction || exhausted {
            mesh.compute_normals(180.0);
            return Ok(OccluderResult {
                mesh,
                outside_fraction: frac,
                erode_used: erode,
            });
        }
        // First erode more; then spend more triangles so big faces stop bridging concavities.
        if erode < 4 {
            erode += 1;
        } else {
            budget *= 2;
        }
    }
}

/// Greedy largest inscribed boxes in the solid (world-space AABBs), largest first.
pub fn box_occluders(scene: &Scene, resolution: usize, max_boxes: usize) -> Result<Vec<Aabb>> {
    let s = classify(scene, resolution, 1, RemeshMode::Outside, &[], 0)?;
    let g = &s.grid;
    let core = g.erode(&s.solid, 1);
    let mut covered = vec![false; g.len()];
    let mut boxes: Vec<([usize; 3], [usize; 3], usize)> = Vec::new();
    let ok = |lo: [usize; 3], hi: [usize; 3], covered: &[bool]| {
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let i = g.idx(x, y, z);
                    if !core[i] || covered[i] {
                        return false;
                    }
                }
            }
        }
        true
    };
    for _ in 0..max_boxes {
        let mut best: Option<([usize; 3], [usize; 3], usize)> = None;
        for i in (0..g.len()).filter(|&i| core[i] && !covered[i]).step_by(3) {
            let start = g.coords(i);
            let (lo, mut hi) = (start, start);
            for axis in [0, 1, 2, 0, 1, 2] {
                while hi[axis] + 1 < g.dims[axis] {
                    let mut t = hi;
                    t[axis] += 1;
                    let mut slab_lo = lo;
                    slab_lo[axis] = t[axis];
                    if ok(slab_lo, t, &covered) {
                        hi = t;
                    } else {
                        break;
                    }
                }
            }
            let vol = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
            if best.is_none_or(|b| vol > b.2) {
                best = Some((lo, hi, vol));
            }
        }
        let Some((lo, hi, vol)) = best else { break };
        if vol < 8 {
            break;
        }
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    covered[g.idx(x, y, z)] = true;
                }
            }
        }
        boxes.push((lo, hi, vol));
    }
    Ok(boxes
        .into_iter()
        .map(|(lo, hi, _)| Aabb {
            min: g.origin + Vec3::new(lo[0] as f32, lo[1] as f32, lo[2] as f32) * g.cell,
            max: g.origin
                + Vec3::new(hi[0] as f32 + 1.0, hi[1] as f32 + 1.0, hi[2] as f32 + 1.0) * g.cell,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::surface_distance;
    use crate::Instance;
    use glam::Mat4;

    fn cube(size: f32) -> Mesh {
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
                positions.push((n + u * su + v * sv) * 0.5 * size);
            }
            indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        Mesh::new(positions, indices)
    }

    /// Two overlapping cubes plus one fully hidden inside: kitbash-style input.
    fn kitbash() -> Scene {
        let mut s = Scene::from_mesh(cube(1.0));
        s.instances.push(Instance {
            name: "b".into(),
            mesh: 0,
            transform: Mat4::from_translation(Vec3::new(0.6, 0.3, 0.0)),
            skeleton: None,
        });
        s.instances.push(Instance {
            name: "hidden".into(),
            mesh: 0,
            transform: Mat4::from_scale(Vec3::splat(0.3)),
            skeleton: None,
        });
        s
    }

    #[test]
    fn remesh_kitbash_is_closed_and_close() {
        let s = kitbash();
        let r = remesh(
            &s,
            &RemeshOptions {
                resolution: 48,
                target_triangles: Some(400),
                ..Default::default()
            },
        )
        .unwrap();
        let m = &r.mesh;
        m.validate().unwrap();
        assert!(m.triangle_count() <= 400);
        // Watertight by position: every position edge used exactly twice.
        let key = |p: Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        let mut ids = std::collections::HashMap::new();
        let vp: Vec<usize> = m
            .positions
            .iter()
            .map(|p| {
                let n = ids.len();
                *ids.entry(key(*p)).or_insert(n)
            })
            .collect();
        let mut edges = std::collections::HashMap::new();
        for t in 0..m.triangle_count() {
            let tri = m.triangle(t).map(|v| vp[v as usize]);
            for i in 0..3 {
                let (a, b) = (tri[i], tri[(i + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2), "not watertight");
        // Close to the visible surface of the union (hidden cube removed).
        let (flat, _) = s.flatten();
        let mut outer = flat.clone();
        outer.indices.truncate(24 * 3);
        outer.material_ids.truncate(24);
        let d = surface_distance(m, &outer, 3000);
        assert!(d.relative_mean < 0.02, "{d:?}");
    }

    /// Random kitbashes of rotated boxes: the proxy must stay watertight by position through
    /// reduction. (Seeds 20..28 hit a stale-validity collapse in the batched reducer that made
    /// one edge non-manifold, before it was fixed.)
    #[test]
    fn random_kitbash_proxies_stay_watertight() {
        for seed in 20..28u32 {
            let mut rng = seed.wrapping_mul(2654435761).wrapping_add(12345);
            let mut r = || {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                (rng % 10000) as f32 / 10000.0
            };
            let mut s = Scene::from_mesh(cube(1.0));
            for k in 0..12 {
                let rot =
                    glam::Quat::from_euler(glam::EulerRot::XYZ, r() * 6.0, r() * 6.0, r() * 6.0);
                let sc = Vec3::new(0.2 + r(), 0.1 + r() * 0.5, 0.2 + r());
                let t = Vec3::new(r() * 2.0 - 1.0, r() * 2.0 - 1.0, r() * 2.0 - 1.0);
                s.instances.push(Instance {
                    name: format!("c{k}"),
                    transform: Mat4::from_scale_rotation_translation(sc, rot, t),
                    ..s.instances[0].clone()
                });
            }
            let o = RemeshOptions {
                resolution: 96,
                target_triangles: Some(400),
                ..Default::default()
            };
            let m = remesh(&s, &o).unwrap().mesh;
            let key = |v: u32| m.positions[v as usize].to_array().map(f32::to_bits);
            let mut e: HashMap<([u32; 3], [u32; 3]), u32> = HashMap::new();
            for t in 0..m.triangle_count() {
                let k = m.triangle(t).map(key);
                for i in 0..3 {
                    let (a, b) = (k[i], k[(i + 1) % 3]);
                    *e.entry(if a < b { (a, b) } else { (b, a) }).or_default() += 1;
                }
            }
            let bad = e.values().filter(|&&c| c != 2).count();
            assert_eq!(bad, 0, "seed {seed}: {bad} non-manifold or open edges");
        }
    }

    /// Two blocks with a slot a third of a cell wide between them, and a lone open sheet
    /// (two triangles) off to the side. With the signed boundary the slot's floor is reached
    /// and the shell sits on the surface; the sheet survives; the proxy stays watertight.
    #[test]
    fn subvoxel_boundary_opens_slots_and_keeps_sheets() {
        let res = 40usize;
        let cell = 4.0 / res as f32;
        let mut m = cube(1.0);
        let mut b = cube(1.0);
        for p in &mut b.positions {
            p.x += 1.0 + cell / 3.0;
        }
        m.append(&b, 0);
        let sheet = Mesh::new(
            vec![
                Vec3::new(-0.5, 1.2, -0.5),
                Vec3::new(1.8, 1.2, -0.5),
                Vec3::new(1.8, 1.2, 0.5),
                Vec3::new(-0.5, 1.2, 0.5),
            ],
            vec![0, 2, 1, 0, 3, 2],
        );
        let mut all = m.clone();
        all.append(&sheet, 0);
        let scene = Scene::from_mesh(all);
        let run = |subvoxel: bool| {
            remesh(
                &scene,
                &RemeshOptions {
                    resolution: res,
                    target_triangles: None,
                    subvoxel,
                    ..Default::default()
                },
            )
            .unwrap()
            .mesh
        };
        let (on, off) = (run(true), run(false));
        let dist = |proxy: &Mesh, src: &Mesh| {
            let bvh = Bvh::build(
                (0..proxy.triangle_count())
                    .map(|t| proxy.triangle_positions(t))
                    .collect(),
            );
            let pts = sample_surface(src, 4000, 7);
            let ds: Vec<f32> = pts
                .iter()
                .map(|&p| bvh.closest_point(p, f32::INFINITY).unwrap().distance / cell)
                .collect();
            (
                ds.iter().copied().fold(0.0, f32::max),
                ds.iter().sum::<f32>() / ds.len() as f32,
            )
        };
        let (on_max, on_mean) = dist(&on, &m);
        let (off_max, off_mean) = dist(&off, &m);
        assert!(on_max < off_max * 0.7, "slot: {on_max} vs {off_max} cells");
        assert!(on_mean < off_mean, "{on_mean} vs {off_mean}");
        // The sheet is still there.
        let (sheet_max, _) = dist(&on, &sheet);
        assert!(sheet_max < 1.0, "sheet lost: {sheet_max} cells");
        // Watertight by position.
        let key = |p: Vec3| p.to_array().map(f32::to_bits);
        let mut ids = std::collections::HashMap::new();
        let vp: Vec<usize> = on
            .positions
            .iter()
            .map(|p| {
                let n = ids.len();
                *ids.entry(key(*p)).or_insert(n)
            })
            .collect();
        let mut edges = std::collections::HashMap::new();
        for t in 0..on.triangle_count() {
            let tri = on.triangle(t).map(|v| vp[v as usize]);
            for i in 0..3 {
                let (a, b) = (tri[i], tri[(i + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2), "not watertight");
    }

    #[test]
    fn pocket_sealing_keeps_cavities_and_drops_crevices() {
        let g = Grid::around(
            Aabb {
                min: Vec3::ZERO,
                max: Vec3::splat(20.0),
            },
            20,
            0,
        );
        let inside = |x: usize, y: usize, z: usize, lo: usize, hi: usize| {
            (lo..hi).contains(&x) && (lo..hi).contains(&y) && (lo..hi).contains(&z)
        };
        let mut sealed = vec![false; g.len()];
        let mut open = vec![false; g.len()];
        for i in 0..g.len() {
            let [x, y, z] = g.coords(i);
            sealed[i] = inside(x, y, z, 3, 17);
            // A hollow box: the cavity 5..15 leaks through a 1-cell hole in the +x wall, and a
            // 1-cell-wide slot is cut into the -y face (not through the 2-cell wall).
            let cavity = inside(x, y, z, 5, 15);
            let hole = (15..17).contains(&x) && y == 10 && z == 10;
            let slot = x == 10 && y == 3 && (6..14).contains(&z);
            open[i] = sealed[i] && !cavity && !hole && !slot;
        }
        let out = seal_pockets(&g, &open, &sealed, 1);
        let at = |x, y, z| out[g.idx(x, y, z)];
        assert!(at(10, 10, 10), "cavity behind the leak is sealed");
        assert!(at(16, 10, 10), "the leak itself is sealed");
        assert!(!at(10, 3, 10), "the slot stays open");
    }

    #[test]
    fn clip_plane_removes_bottom() {
        let s = Scene::from_mesh(cube(1.0));
        let r = remesh(
            &s,
            &RemeshOptions {
                resolution: 32,
                clip_planes: vec![Vec4::new(0.0, -1.0, 0.0, 0.0)],
                target_triangles: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            r.mesh.positions.iter().all(|p| p.y >= -0.1),
            "geometry below the clip plane"
        );
    }

    #[test]
    fn occluder_is_inside() {
        let s = kitbash();
        let r = occluder(
            &s,
            &OccluderOptions {
                resolution: 48,
                target_triangles: 100,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.mesh.triangle_count() <= 100);
        assert!(r.outside_fraction <= 0.01, "outside {}", r.outside_fraction);
        // Strictly inside the union's bounds.
        let b = s.bounds();
        assert!(r
            .mesh
            .positions
            .iter()
            .all(|p| p.cmpge(b.min).all() && p.cmple(b.max).all()));
    }

    #[test]
    fn boxes_fit_inside() {
        let s = Scene::from_mesh(cube(1.0));
        let boxes = box_occluders(&s, 24, 4).unwrap();
        assert!(!boxes.is_empty());
        let vol = boxes[0].extent().x * boxes[0].extent().y * boxes[0].extent().z;
        assert!(vol > 0.3 && vol <= 1.0, "volume {vol}");
    }
}
