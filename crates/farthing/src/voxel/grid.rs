//! Dense voxel grids: triangle voxelization, flood fill, morphology, surface nets.

use crate::{Aabb, Mesh};
use glam::Vec3;
use std::collections::VecDeque;

/// A dense grid of cubic cells. Cell (x, y, z) covers `origin + [x, x+1) * cell` etc.
#[derive(Clone, Debug)]
pub struct Grid {
    pub dims: [usize; 3],
    pub origin: Vec3,
    pub cell: f32,
}

impl Grid {
    /// Grid covering `bounds` with `resolution` cells along the longest axis plus `pad` empty
    /// cells on every side.
    pub fn around(bounds: Aabb, resolution: usize, pad: usize) -> Grid {
        let ext = bounds.extent().max(Vec3::splat(1e-6));
        let cell = ext.max_element() / resolution.max(1) as f32;
        let dims = [0, 1, 2].map(|a| ((ext[a] / cell).ceil() as usize).max(1) + 2 * pad);
        Grid {
            dims,
            origin: bounds.min - Vec3::splat(pad as f32 * cell),
            cell,
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }
    #[inline]
    pub fn coords(&self, i: usize) -> [usize; 3] {
        let x = i % self.dims[0];
        let y = (i / self.dims[0]) % self.dims[1];
        let z = i / (self.dims[0] * self.dims[1]);
        [x, y, z]
    }
    #[inline]
    pub fn center(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.origin + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * self.cell
    }
    /// Cell containing point `p`, if inside the grid.
    pub fn cell_of(&self, p: Vec3) -> Option<[usize; 3]> {
        let f = (p - self.origin) / self.cell;
        if f.min_element() < 0.0 {
            return None;
        }
        let c = [f.x as usize, f.y as usize, f.z as usize];
        (c[0] < self.dims[0] && c[1] < self.dims[1] && c[2] < self.dims[2]).then_some(c)
    }

    /// Marks every cell a triangle touches (exact triangle/box overlap test).
    pub fn voxelize(&self, tris: &[[Vec3; 3]]) -> Vec<bool> {
        let mut out = vec![false; self.len()];
        let h = Vec3::splat(self.cell * 0.5 * 1.0001);
        for t in tris {
            let lo = ((t[0].min(t[1]).min(t[2]) - self.origin) / self.cell)
                .floor()
                .max(Vec3::ZERO);
            let hi = ((t[0].max(t[1]).max(t[2]) - self.origin) / self.cell).floor();
            let hi = [hi.x as usize, hi.y as usize, hi.z as usize];
            for z in lo.z as usize..=hi[2].min(self.dims[2] - 1) {
                for y in lo.y as usize..=hi[1].min(self.dims[1] - 1) {
                    for x in lo.x as usize..=hi[0].min(self.dims[0] - 1) {
                        let i = self.idx(x, y, z);
                        if !out[i] && tri_box_overlap(self.center(x, y, z), h, t) {
                            out[i] = true;
                        }
                    }
                }
            }
        }
        out
    }

    /// 6-connected flood fill through cells where `passable` is true, from `seeds`.
    pub fn flood(&self, passable: &[bool], seeds: impl IntoIterator<Item = usize>) -> Vec<bool> {
        let mut reached = vec![false; self.len()];
        let mut q = VecDeque::new();
        for s in seeds {
            if passable[s] && !reached[s] {
                reached[s] = true;
                q.push_back(s);
            }
        }
        while let Some(i) = q.pop_front() {
            let [x, y, z] = self.coords(i);
            let mut visit = |j: usize| {
                if passable[j] && !reached[j] {
                    reached[j] = true;
                    q.push_back(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < self.dims[0] {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - self.dims[0]);
            }
            if y + 1 < self.dims[1] {
                visit(i + self.dims[0]);
            }
            if z > 0 {
                visit(i - self.dims[0] * self.dims[1]);
            }
            if z + 1 < self.dims[2] {
                visit(i + self.dims[0] * self.dims[1]);
            }
        }
        reached
    }

    /// Indices of all boundary cells of the grid.
    pub fn border_cells(&self) -> Vec<usize> {
        let [nx, ny, nz] = self.dims;
        (0..self.len())
            .filter(|&i| {
                let [x, y, z] = self.coords(i);
                x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1
            })
            .collect()
    }

    /// 6-neighbourhood dilation, `iterations` times.
    pub fn dilate(&self, mask: &[bool], iterations: usize) -> Vec<bool> {
        let mut m = mask.to_vec();
        for _ in 0..iterations {
            let src = m.clone();
            for i in 0..self.len() {
                if !src[i] && self.neighbors(i).any(|j| src[j]) {
                    m[i] = true;
                }
            }
        }
        m
    }

    /// 6-neighbourhood erosion (grid boundary counts as empty).
    pub fn erode(&self, mask: &[bool], iterations: usize) -> Vec<bool> {
        let mut m = mask.to_vec();
        for _ in 0..iterations {
            let src = m.clone();
            for i in 0..self.len() {
                if src[i] && (self.neighbors(i).count() < 6 || self.neighbors(i).any(|j| !src[j])) {
                    m[i] = false;
                }
            }
        }
        m
    }

    fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let [x, y, z] = self.coords(i);
        let (sx, sy) = (1, self.dims[0]);
        let sz = self.dims[0] * self.dims[1];
        [
            (x > 0).then(|| i - sx),
            (x + 1 < self.dims[0]).then(|| i + sx),
            (y > 0).then(|| i - sy),
            (y + 1 < self.dims[1]).then(|| i + sy),
            (z > 0).then(|| i - sz),
            (z + 1 < self.dims[2]).then(|| i + sz),
        ]
        .into_iter()
        .flatten()
    }

    /// Surface nets over the binary field `solid` (samples at cell centers). Produces a closed,
    /// outward-facing triangle mesh. Requires at least one empty cell of padding at the border.
    /// Fills voxels until the solid has no ambiguous 2×2×2 configurations: no two cells of the
    /// same kind (solid or empty) touching only along an edge or a corner while the cells between
    /// them are of the other kind. Surface nets on such a solid are a closed 2-manifold. Only adds
    /// solid, so the result encloses the input. Returns the number of cells filled.
    pub fn manifoldize(&self, solid: &mut [bool]) -> usize {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return 0;
        }
        let mut filled = 0;
        for _ in 0..16 {
            let mut changed = 0;
            for z in 0..nz - 1 {
                for y in 0..ny - 1 {
                    for x in 0..nx - 1 {
                        let id = |dx: usize, dy: usize, dz: usize| self.idx(x + dx, y + dy, z + dz);
                        let c: [usize; 8] = [
                            id(0, 0, 0),
                            id(1, 0, 0),
                            id(0, 1, 0),
                            id(1, 1, 0),
                            id(0, 0, 1),
                            id(1, 0, 1),
                            id(0, 1, 1),
                            id(1, 1, 1),
                        ];
                        let v = c.map(|i| solid[i]);
                        // Face diagonals on the six faces of the cube (corner index pairs).
                        const FACES: [[usize; 4]; 6] = [
                            [0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [0, 1, 4, 5],
                            [2, 3, 6, 7],
                            [0, 2, 4, 6],
                            [1, 3, 5, 7],
                        ];
                        let mut fix = None;
                        for f in FACES {
                            // f = [a, b, c, d] laid out as a b / c d: diagonals (a, d) and (b, c).
                            let (a, b, cc, d) = (v[f[0]], v[f[1]], v[f[2]], v[f[3]]);
                            if a == d && b == cc && a != b {
                                // Fill the first empty corner of the face.
                                fix = Some(if !a { f[0] } else { f[1] });
                                break;
                            }
                        }
                        if fix.is_none() {
                            // Corner diagonals: two opposite corners of one kind, the other six
                            // of the other kind.
                            for (p, q) in [(0, 7), (1, 6), (2, 5), (3, 4)] {
                                if v[p] == v[q]
                                    && (0..8).filter(|&k| k != p && k != q).all(|k| v[k] != v[p])
                                {
                                    fix = Some(if !v[p] {
                                        p
                                    } else {
                                        (0..8).find(|&k| k != p && k != q).unwrap()
                                    });
                                    break;
                                }
                            }
                        }
                        if let Some(k) = fix {
                            solid[c[k]] = true;
                            changed += 1;
                        }
                    }
                }
            }
            filled += changed;
            if changed == 0 {
                break;
            }
        }
        filled
    }

    pub fn surface_nets(&self, solid: &[bool]) -> Mesh {
        let [nx, ny, nz] = self.dims;
        let s = |x: usize, y: usize, z: usize| solid[self.idx(x, y, z)];
        // One vertex per dual cube (min corner sample (x, y, z)) with mixed samples.
        let cube_idx = |x: usize, y: usize, z: usize| (z * (ny - 1) + y) * (nx - 1) + x;
        let mut vert_of = vec![u32::MAX; (nx - 1) * (ny - 1) * (nz - 1)];
        let mut positions = Vec::new();
        let corners = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ];
        let edges = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for z in 0..nz - 1 {
            for y in 0..ny - 1 {
                for x in 0..nx - 1 {
                    let v: [bool; 8] = corners.map(|(dx, dy, dz)| s(x + dx, y + dy, z + dz));
                    if v.iter().all(|&b| b) || v.iter().all(|&b| !b) {
                        continue;
                    }
                    let mut acc = Vec3::ZERO;
                    let mut n = 0.0;
                    for (a, b) in edges {
                        if v[a] != v[b] {
                            let pa = Vec3::new(
                                corners[a].0 as f32,
                                corners[a].1 as f32,
                                corners[a].2 as f32,
                            );
                            let pb = Vec3::new(
                                corners[b].0 as f32,
                                corners[b].1 as f32,
                                corners[b].2 as f32,
                            );
                            acc += (pa + pb) * 0.5;
                            n += 1.0;
                        }
                    }
                    let local = acc / n;
                    vert_of[cube_idx(x, y, z)] = positions.len() as u32;
                    positions.push(self.center(x, y, z) + local * self.cell);
                }
            }
        }
        let mut indices = Vec::new();
        let mut quad = |q: [u32; 4], flip: bool, positions: &Vec<Vec3>| {
            if q.contains(&u32::MAX) {
                return;
            }
            let q = if flip { [q[0], q[3], q[2], q[1]] } else { q };
            // Split along the shorter diagonal.
            let d02 = (positions[q[0] as usize] - positions[q[2] as usize]).length_squared();
            let d13 = (positions[q[1] as usize] - positions[q[3] as usize]).length_squared();
            if d02 <= d13 {
                indices.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            } else {
                indices.extend_from_slice(&[q[0], q[1], q[3], q[1], q[2], q[3]]);
            }
        };
        // A grid edge yields a quad when all four dual cubes around it exist, i.e. the edge is
        // not on the outer layer of samples along the two other axes. (Edges leaving sample 0
        // matter: a surface voxel can touch layer 1 when geometry lies on the bounds.)
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let here = s(x, y, z);
                    let (iy, iz, ix) = (
                        y >= 1 && y < ny - 1,
                        z >= 1 && z < nz - 1,
                        x >= 1 && x < nx - 1,
                    );
                    // Edge along +X to (x+1, y, z).
                    if x + 1 < nx && iy && iz && here != s(x + 1, y, z) {
                        let q = [
                            cube_idx(x, y - 1, z - 1),
                            cube_idx(x, y, z - 1),
                            cube_idx(x, y, z),
                            cube_idx(x, y - 1, z),
                        ]
                        .map(|c| vert_of[c]);
                        quad(q, !here, &positions);
                    }
                    if y + 1 < ny && ix && iz && here != s(x, y + 1, z) {
                        let q = [
                            cube_idx(x - 1, y, z - 1),
                            cube_idx(x - 1, y, z),
                            cube_idx(x, y, z),
                            cube_idx(x, y, z - 1),
                        ]
                        .map(|c| vert_of[c]);
                        quad(q, !here, &positions);
                    }
                    if z + 1 < nz && ix && iy && here != s(x, y, z + 1) {
                        let q = [
                            cube_idx(x - 1, y - 1, z),
                            cube_idx(x, y - 1, z),
                            cube_idx(x, y, z),
                            cube_idx(x - 1, y, z),
                        ]
                        .map(|c| vert_of[c]);
                        quad(q, !here, &positions);
                    }
                }
            }
        }
        Mesh::new(positions, indices)
    }
}

/// Triangle / axis-aligned box overlap via the separating axis theorem (Akenine-Möller 2001).
pub fn tri_box_overlap(center: Vec3, half: Vec3, tri: &[Vec3; 3]) -> bool {
    let v = [tri[0] - center, tri[1] - center, tri[2] - center];
    let e = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];
    // 9 edge cross-product axes.
    for ei in &e {
        for axis in [
            Vec3::new(0.0, -ei.z, ei.y),
            Vec3::new(ei.z, 0.0, -ei.x),
            Vec3::new(-ei.y, ei.x, 0.0),
        ] {
            let p = [axis.dot(v[0]), axis.dot(v[1]), axis.dot(v[2])];
            let r = half.x * axis.x.abs() + half.y * axis.y.abs() + half.z * axis.z.abs();
            let (mn, mx) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
            if mn > r || mx < -r {
                return false;
            }
        }
    }
    // Box face normals.
    for a in 0..3 {
        let (mn, mx) = (
            v[0][a].min(v[1][a]).min(v[2][a]),
            v[0][a].max(v[1][a]).max(v[2][a]),
        );
        if mn > half[a] || mx < -half[a] {
            return false;
        }
    }
    // Triangle plane.
    let n = e[0].cross(e[1]);
    let d = n.dot(v[0]);
    let r = half.x * n.x.abs() + half.y * n.y.abs() + half.z * n.z.abs();
    d.abs() <= r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_basics() {
        let t = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(tri_box_overlap(Vec3::ZERO, Vec3::splat(0.1), &t));
        assert!(!tri_box_overlap(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::splat(0.1),
            &t
        ));
        assert!(!tri_box_overlap(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::splat(0.1),
            &t
        ));
    }

    #[test]
    fn nets_of_a_solid_box_is_closed_and_outward() {
        let g = Grid {
            dims: [6, 6, 6],
            origin: Vec3::ZERO,
            cell: 1.0,
        };
        let mut solid = vec![false; g.len()];
        for z in 2..4 {
            for y in 2..4 {
                for x in 2..4 {
                    solid[g.idx(x, y, z)] = true;
                }
            }
        }
        let m = g.surface_nets(&solid);
        m.validate().unwrap();
        assert!(m.triangle_count() > 0);
        // Closed: every undirected edge used exactly twice.
        let mut edges = std::collections::HashMap::new();
        for t in 0..m.triangle_count() {
            let tri = m.triangle(t);
            for i in 0..3 {
                let (a, b) = (tri[i], tri[(i + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2));
        // Outward: signed volume positive.
        let vol: f32 = (0..m.triangle_count())
            .map(|t| {
                let [a, b, c] = m.triangle_positions(t);
                a.dot(b.cross(c)) / 6.0
            })
            .sum();
        assert!(vol > 0.0, "volume {vol}");
    }

    #[test]
    fn flood_and_morphology() {
        let g = Grid {
            dims: [5, 5, 5],
            origin: Vec3::ZERO,
            cell: 1.0,
        };
        let mut wall = vec![false; g.len()];
        // Hollow shell around the center cell.
        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    if (x, y, z) != (2, 2, 2) {
                        wall[g.idx(x, y, z)] = true;
                    }
                }
            }
        }
        let passable: Vec<bool> = wall.iter().map(|w| !w).collect();
        let outside = g.flood(&passable, g.border_cells());
        assert!(!outside[g.idx(2, 2, 2)]);
        let solid: Vec<bool> = outside.iter().map(|o| !o).collect();
        assert_eq!(solid.iter().filter(|&&s| s).count(), 27);
        let e = g.erode(&solid, 1);
        assert_eq!(e.iter().filter(|&&s| s).count(), 1);
        assert_eq!(g.dilate(&e, 1).iter().filter(|&&s| s).count(), 7);
    }
}
