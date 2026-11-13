//! Sub-voxel signed field for the boundary cells of a voxel solid.
//!
//! Conservative voxelization makes every cell the surface touches solid, so the extracted
//! shell sits up to a cell outside the source everywhere, and slots narrower than a cell fill
//! to their mouths. Here each touched cell instead takes the sign of the source's signed
//! distance at its centre, which puts the shell on the surface (as a signed-distance contourer
//! would). Three guards keep that safe on real assets:
//!
//! * **Sign.** Angle-weighted pseudo-normals at the closest point (Bærentzen & Aanæs,
//!   *Signed Distance Computation Using the Angle Weighted Pseudonormal*, IEEE TVCG 2005).
//!   A centre found outside is carved only if no winding ray finds it enclosed: inside a
//!   body, the closest face may belong to a nested or overlapping piece that faces outwards.
//!   A ray leaving through a hole of an open source counts 0 or −1, never "enclosed".
//! * **No lost surface.** Every surface sample (half-cell spacing) must keep a solid cell
//!   centre behind it, within a cell, that lies inside a volume: an interior cell, or a
//!   boundary cell that every winding ray finds enclosed. Otherwise its cell stays solid. A
//!   sheet thinner than a cell has no such centre, so it survives, where a pure
//!   signed-distance contourer drops it.
//! * **Lone sheets keep their slab.** A sample with no volume on either side (an open sheet
//!   on its own) keeps every boundary cell around it, so the slab does not thin into a
//!   staircase that falls apart into islands.

use crate::voxel::grid::Grid;
use glam::Vec3;
use crate::bvh::Bvh;
use rayon::prelude::*;
use std::collections::HashMap;

/// Pseudo-normals of a triangle soup, welded by exact position.
pub(crate) struct Signed<'a> {
    bvh: &'a Bvh,
    /// Per triangle corner: welded vertex id.
    corner: Vec<u32>,
    vertex_n: Vec<Vec3>,
    /// Per triangle edge k (corners k, k+1): pseudo-normal (sum of the incident face normals).
    edge_n: Vec<[Vec3; 3]>,
    face_n: Vec<Vec3>,
}

impl<'a> Signed<'a> {
    pub(crate) fn new(bvh: &'a Bvh) -> Signed<'a> {
        let tris = &bvh.triangles;
        let key = |p: Vec3| p.to_array().map(f32::to_bits);
        let mut ids: HashMap<[u32; 3], u32> = HashMap::with_capacity(tris.len());
        let mut corner = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            for p in t {
                let n = ids.len() as u32;
                corner.push(*ids.entry(key(*p)).or_insert(n));
            }
        }
        let face_n: Vec<Vec3> = tris
            .par_iter()
            .map(|t| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero())
            .collect();
        let mut vertex_n = vec![Vec3::ZERO; ids.len()];
        let mut edges: HashMap<(u32, u32), Vec3> = HashMap::with_capacity(tris.len() * 2);
        for (f, t) in tris.iter().enumerate() {
            let n = face_n[f];
            for k in 0..3 {
                let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                let angle = (b - a).normalize_or_zero().dot((c - a).normalize_or_zero());
                vertex_n[corner[3 * f + k] as usize] += n * angle.clamp(-1.0, 1.0).acos();
                let (i, j) = (corner[3 * f + k], corner[3 * f + (k + 1) % 3]);
                *edges.entry((i.min(j), i.max(j))).or_insert(Vec3::ZERO) += n;
            }
        }
        let edge_n = (0..tris.len())
            .map(|f| {
                [0, 1, 2].map(|k| {
                    let (i, j) = (corner[3 * f + k], corner[3 * f + (k + 1) % 3]);
                    edges[&(i.min(j), i.max(j))]
                })
            })
            .collect();
        Signed {
            bvh,
            corner,
            vertex_n,
            edge_n,
            face_n,
        }
    }

    /// Signed distance to the source (negative inside) if it is within `max`.
    pub(crate) fn distance(&self, p: Vec3, max: f32) -> Option<f32> {
        let h = self.bvh.closest_point(p, max)?;
        let f = h.triangle as usize;
        let w = h.barycentrics;
        const EPS: f32 = 1e-5;
        let zero = [w.x <= EPS, w.y <= EPS, w.z <= EPS];
        let n = match zero.iter().filter(|&&z| z).count() {
            0 => self.face_n[f],
            1 => {
                // On the edge opposite the zero weight: corners (k+1, k+2) = edge k+1.
                let k = zero.iter().position(|&z| z).unwrap();
                self.edge_n[f][(k + 1) % 3]
            }
            _ => {
                let k = zero.iter().position(|&z| !z).unwrap_or(0);
                self.vertex_n[self.corner[3 * f + k] as usize]
            }
        };
        let d = p - h.point;
        Some(if d.dot(n) < 0.0 {
            -h.distance
        } else {
            h.distance
        })
    }
}

/// Ray directions for winding checks: irrational-ish, away from the axes and from each other.
const WINDING_DIRS: [Vec3; 3] = [
    Vec3::new(0.577_350_3, 0.611_201_6, 0.541_400_1),
    Vec3::new(-0.701_793_8, 0.120_391_6, -0.702_186_7),
    Vec3::new(0.231_224_6, -0.940_996_3, 0.247_002_9),
];

/// Relabels the boundary cells of `solid` (those in `surface`, the cells the source touches)
/// by the sign at their centre, with the guards in the module docs. Returns how many cells
/// became air.
pub(crate) fn relabel_boundary(
    grid: &Grid,
    solid: &mut [bool],
    surface: &[bool],
    signed: &Signed,
) -> usize {
    let n = grid.len();
    let cell = grid.cell;
    let [nx, ny, nz] = grid.dims;
    // Per boundary cell that is solid now: (centre inside, centre inside a volume for sure).
    // None: no surface within 1.5 cells.
    let inside: Vec<Option<(bool, bool)>> = (0..n)
        .into_par_iter()
        .with_min_len(1024)
        .map(|i| {
            if !(solid[i] && surface[i]) {
                return None;
            }
            let [x, y, z] = grid.coords(i);
            let c = grid.center(x, y, z);
            let d = signed.distance(c, 1.5 * cell)?;
            let w = WINDING_DIRS.map(|dir| signed.bvh.ray_winding(c, dir));
            // Outside by the closest face: confirmed only if no ray finds the centre
            // enclosed (a nested or overlapping piece can face outwards inside the body; a
            // ray leaving through a hole in an open source counts 0 or −1, never enclosed).
            // Inside for sure only if every ray agrees: behind a lone open sheet, rays that
            // pass its edge see 0.
            let enclosed_any = w.iter().any(|&k| k > 0);
            let enclosed_all = w.iter().all(|&k| k > 0);
            Some((d < 0.0 || enclosed_any, d < 0.0 && enclosed_all))
        })
        .collect();
    // A cell centre supports a surface sample if it is solid and inside a volume: interior
    // cells (inside by the flood), or boundary cells inside for sure.
    let centre_in = |j: usize| solid[j] && inside[j].is_none_or(|(_, sure)| sure);
    // Surface samples no farther apart than half a cell; a sample is supported if a cell
    // centre within a cell of it, behind it, is inside.
    let tris = &signed.bvh.triangles;
    let unsupported: Vec<u32> = tris
        .par_iter()
        .enumerate()
        .flat_map_iter(|(f, t)| {
            let nrm = signed.face_n[f];
            let longest = (t[1] - t[0])
                .length()
                .max((t[2] - t[1]).length())
                .max((t[0] - t[2]).length());
            let k = ((longest / (0.5 * cell)).ceil() as usize).clamp(1, 256);
            let t = *t;
            let mut out = Vec::new();
            if nrm == Vec3::ZERO {
                return out.into_iter();
            }
            for a in 0..=k {
                for b in 0..=k - a {
                    let (u, v) = (a as f32 / k as f32, b as f32 / k as f32);
                    let p = t[0] + (t[1] - t[0]) * u + (t[2] - t[0]) * v;
                    let Some([x, y, z]) = grid.cell_of(p) else { continue };
                    let (mut ok, mut any_in) = (false, false);
                    'search: for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let (cx, cy, cz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                                if cx < 0
                                    || cy < 0
                                    || cz < 0
                                    || cx >= nx as i32
                                    || cy >= ny as i32
                                    || cz >= nz as i32
                                {
                                    continue;
                                }
                                let (cx, cy, cz) = (cx as usize, cy as usize, cz as usize);
                                let j = grid.idx(cx, cy, cz);
                                let off = grid.center(cx, cy, cz) - p;
                                // An interior cell (solid, untouched by the surface) or a
                                // centre inside for sure means a volume is near: not a lone
                                // sheet.
                                if solid[j] && (!surface[j] || centre_in(j)) {
                                    any_in = true;
                                }
                                if centre_in(j) && off.dot(nrm) < 0.0 && off.length() <= cell {
                                    ok = true;
                                    break 'search;
                                }
                            }
                        }
                    }
                    if !ok {
                        // High bit: no volume on either side (a lone sheet).
                        let sheet = if any_in { 0 } else { 1 << 31 };
                        out.push(grid.idx(x, y, z) as u32 | sheet);
                    }
                }
            }
            out.into_iter()
        })
        .collect();
    // An unsupported sample keeps its cell. On a lone sheet (no volume on either side) it
    // keeps every boundary cell around it, so the sheet keeps its whole conservative slab:
    // cells it only grazes contain no sample of their own.
    let mut keep = vec![false; n];
    for &c in &unsupported {
        let cell_i = (c & !(1 << 31)) as usize;
        keep[cell_i] = true;
        if c & (1 << 31) == 0 {
            continue;
        }
        let [x, y, z] = grid.coords(cell_i);
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (a, b, cz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    if a < 0 || b < 0 || cz < 0 || a >= nx as i32 || b >= ny as i32 || cz >= nz as i32 {
                        continue;
                    }
                    let j = grid.idx(a as usize, b as usize, cz as usize);
                    if surface[j] {
                        keep[j] = true;
                    }
                }
            }
        }
    }
    let carvable: Vec<bool> = (0..n)
        .map(|i| inside[i].is_some_and(|(inn, _)| !inn) && !keep[i])
        .collect();
    let mut carved = 0;
    for i in 0..n {
        if carvable[i] {
            solid[i] = false;
            carved += 1;
        }
    }
    carved
}
