//! Indexed triangle meshes with per-vertex ("wedge") attributes.

use crate::math::{triangle_area, Aabb};
use crate::{Error, Result};
use glam::{Vec2, Vec3, Vec4};
use std::collections::HashMap;

/// Up to four joint influences per vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skin {
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}

/// A morph target (blend shape): per-vertex deltas.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MorphTarget {
    pub name: String,
    pub position_deltas: Vec<Vec3>,
    pub normal_deltas: Option<Vec<Vec3>>,
}

/// Indexed triangle mesh.
///
/// Every vertex is a *wedge*: a unique combination of position and attributes. Two vertices
/// may share a position (UV seams, hard edges). Processors that care about topology weld by
/// position internally and keep the wedges intact.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Option<Vec<Vec3>>,
    /// MikkTSpace-style tangents (xyz + handedness sign in w).
    pub tangents: Option<Vec<Vec4>>,
    /// UV sets; `uvs[0]` is the primary set.
    pub uvs: Vec<Vec<Vec2>>,
    pub colors: Option<Vec<Vec4>>,
    pub skin: Option<Skin>,
    pub morph_targets: Vec<MorphTarget>,
    /// Three indices per triangle.
    pub indices: Vec<u32>,
    /// One material slot per triangle (index into [`crate::Scene::materials`]).
    pub material_ids: Vec<u32>,
}

impl Mesh {
    pub fn new(positions: Vec<Vec3>, indices: Vec<u32>) -> Self {
        let tris = indices.len() / 3;
        Mesh {
            positions,
            indices,
            material_ids: vec![0; tris],
            ..Default::default()
        }
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
    #[inline]
    pub fn triangle(&self, t: usize) -> [u32; 3] {
        [
            self.indices[3 * t],
            self.indices[3 * t + 1],
            self.indices[3 * t + 2],
        ]
    }
    #[inline]
    pub fn triangle_positions(&self, t: usize) -> [Vec3; 3] {
        let [a, b, c] = self.triangle(t);
        [
            self.positions[a as usize],
            self.positions[b as usize],
            self.positions[c as usize],
        ]
    }
    pub fn uv(&self, set: usize, v: u32) -> Vec2 {
        self.uvs
            .get(set)
            .map(|s| s[v as usize])
            .unwrap_or(Vec2::ZERO)
    }

    pub fn bounds(&self) -> Aabb {
        Aabb::from_points(self.positions.iter())
    }

    pub fn surface_area(&self) -> f32 {
        (0..self.triangle_count())
            .map(|t| {
                let [a, b, c] = self.triangle_positions(t);
                triangle_area(a, b, c)
            })
            .sum()
    }

    /// Checks that all attribute arrays agree in length and indices are in range.
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        let bad = |what: &str| {
            Err(Error::InvalidMesh(format!(
                "{what} length does not match vertex count {n}"
            )))
        };
        if !self.indices.len().is_multiple_of(3) {
            return Err(Error::InvalidMesh(
                "index count is not a multiple of 3".into(),
            ));
        }
        if self.material_ids.len() != self.triangle_count() {
            return Err(Error::InvalidMesh(format!(
                "material_ids has {} entries for {} triangles",
                self.material_ids.len(),
                self.triangle_count()
            )));
        }
        if self.normals.as_ref().is_some_and(|a| a.len() != n) {
            return bad("normals");
        }
        if self.tangents.as_ref().is_some_and(|a| a.len() != n) {
            return bad("tangents");
        }
        if self.colors.as_ref().is_some_and(|a| a.len() != n) {
            return bad("colors");
        }
        if self.uvs.iter().any(|a| a.len() != n) {
            return bad("uvs");
        }
        if let Some(s) = &self.skin {
            if s.joints.len() != n || s.weights.len() != n {
                return bad("skin");
            }
        }
        if self
            .morph_targets
            .iter()
            .any(|m| m.position_deltas.len() != n)
        {
            return bad("morph target");
        }
        if let Some(i) = self.indices.iter().find(|&&i| i as usize >= n) {
            return Err(Error::InvalidMesh(format!(
                "index {i} out of range ({n} vertices)"
            )));
        }
        if self.positions.iter().any(|p| !p.is_finite()) {
            return Err(Error::InvalidMesh("non-finite position".into()));
        }
        Ok(())
    }

    /// A mesh with no triangles and the same attribute layout, holding the vertices in
    /// `order` (old vertex indices) in that order.
    pub fn select_vertices(&self, order: &[u32]) -> Mesh {
        let pick3 = |a: &Vec<Vec3>| order.iter().map(|&v| a[v as usize]).collect::<Vec<_>>();
        Mesh {
            positions: pick3(&self.positions),
            normals: self.normals.as_ref().map(pick3),
            tangents: self
                .tangents
                .as_ref()
                .map(|a| order.iter().map(|&v| a[v as usize]).collect()),
            uvs: self
                .uvs
                .iter()
                .map(|s| order.iter().map(|&v| s[v as usize]).collect())
                .collect(),
            colors: self
                .colors
                .as_ref()
                .map(|a| order.iter().map(|&v| a[v as usize]).collect()),
            skin: self.skin.as_ref().map(|s| Skin {
                joints: order.iter().map(|&v| s.joints[v as usize]).collect(),
                weights: order.iter().map(|&v| s.weights[v as usize]).collect(),
            }),
            morph_targets: self
                .morph_targets
                .iter()
                .map(|m| MorphTarget {
                    name: m.name.clone(),
                    position_deltas: pick3(&m.position_deltas),
                    normal_deltas: m.normal_deltas.as_ref().map(pick3),
                })
                .collect(),
            indices: Vec::new(),
            material_ids: Vec::new(),
        }
    }

    /// Removes vertices no triangle references. Returns the old→new remap (`u32::MAX` = removed).
    pub fn compact(&mut self) -> Vec<u32> {
        let mut remap = vec![u32::MAX; self.vertex_count()];
        let mut order = Vec::new();
        for &i in &self.indices {
            if remap[i as usize] == u32::MAX {
                remap[i as usize] = order.len() as u32;
                order.push(i);
            }
        }
        let mut out = self.select_vertices(&order);
        out.indices = self.indices.iter().map(|&i| remap[i as usize]).collect();
        out.material_ids = std::mem::take(&mut self.material_ids);
        *self = out;
        remap
    }

    /// Appends `other` (all vertices and triangles), offsetting material ids by `material_offset`.
    /// Attribute sets missing on one side are filled with defaults.
    pub fn append(&mut self, other: &Mesh, material_offset: u32) {
        let base = self.vertex_count() as u32;
        let n_self = self.vertex_count();
        let n_other = other.vertex_count();
        fn merge<T: Clone>(
            dst: &mut Option<Vec<T>>,
            src: &Option<Vec<T>>,
            n_self: usize,
            n_other: usize,
            default: T,
        ) {
            match (dst.as_mut(), src) {
                (Some(d), Some(s)) => d.extend_from_slice(s),
                (Some(d), None) => d.extend(std::iter::repeat_n(default, n_other)),
                (None, Some(s)) => {
                    let mut d = vec![default; n_self];
                    d.extend_from_slice(s);
                    *dst = Some(d);
                }
                (None, None) => {}
            }
        }
        merge(&mut self.normals, &other.normals, n_self, n_other, Vec3::Y);
        merge(
            &mut self.tangents,
            &other.tangents,
            n_self,
            n_other,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        );
        merge(&mut self.colors, &other.colors, n_self, n_other, Vec4::ONE);
        let sets = self.uvs.len().max(other.uvs.len());
        while self.uvs.len() < sets {
            self.uvs.push(vec![Vec2::ZERO; n_self]);
        }
        for s in 0..sets {
            match other.uvs.get(s) {
                Some(src) => self.uvs[s].extend_from_slice(src),
                None => self.uvs[s].extend(std::iter::repeat_n(Vec2::ZERO, n_other)),
            }
        }
        match (&mut self.skin, &other.skin) {
            (Some(d), Some(s)) => {
                d.joints.extend_from_slice(&s.joints);
                d.weights.extend_from_slice(&s.weights);
            }
            (Some(d), None) => {
                d.joints.extend(std::iter::repeat_n([0; 4], n_other));
                d.weights
                    .extend(std::iter::repeat_n([1.0, 0.0, 0.0, 0.0], n_other));
            }
            (None, Some(s)) => {
                let mut d = Skin {
                    joints: vec![[0; 4]; n_self],
                    weights: vec![[1.0, 0.0, 0.0, 0.0]; n_self],
                };
                d.joints.extend_from_slice(&s.joints);
                d.weights.extend_from_slice(&s.weights);
                self.skin = Some(d);
            }
            (None, None) => {}
        }
        // Morph targets only survive when both sides agree on the set.
        if self.morph_targets.len() == other.morph_targets.len() && n_self > 0 {
            for (d, s) in self.morph_targets.iter_mut().zip(&other.morph_targets) {
                d.position_deltas.extend_from_slice(&s.position_deltas);
                d.normal_deltas = None;
            }
        } else if n_self == 0 {
            self.morph_targets = other.morph_targets.clone();
        } else {
            self.morph_targets.clear();
        }
        self.positions.extend_from_slice(&other.positions);
        self.indices.extend(other.indices.iter().map(|&i| i + base));
        self.material_ids
            .extend(other.material_ids.iter().map(|&m| m + material_offset));
    }

    /// Area-weighted vertex normals, with vertices that share a position averaged together
    /// unless the angle between their faces exceeds `hard_angle_deg` — then the mesh is split.
    /// Overwrites `normals`. Returns the number of vertices added by splitting.
    pub fn compute_normals(&mut self, hard_angle_deg: f32) -> usize {
        let cos_hard = hard_angle_deg.to_radians().cos();
        let tri_count = self.triangle_count();
        let face_n: Vec<Vec3> = (0..tri_count)
            .map(|t| {
                let [a, b, c] = self.triangle_positions(t);
                (b - a).cross(c - a) // area weighted
            })
            .collect();
        // Group corners by exact position.
        let mut pos_corners: HashMap<[u32; 3], Vec<usize>> = HashMap::new();
        for (corner, &v) in self.indices.iter().enumerate() {
            let p = self.positions[v as usize];
            pos_corners
                .entry([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                .or_default()
                .push(corner);
        }
        // Per-corner smoothed normal.
        let mut corner_n = vec![Vec3::ZERO; self.indices.len()];
        for corners in pos_corners.values() {
            for &c in corners {
                let fc = face_n[c / 3].normalize_or_zero();
                let mut acc = Vec3::ZERO;
                for &o in corners {
                    let fo = face_n[o / 3];
                    if fc.dot(fo.normalize_or_zero()) >= cos_hard {
                        acc += fo;
                    }
                }
                corner_n[c] = acc.normalize_or_zero();
            }
        }
        // Re-index: a vertex keeps one normal; corners with a different normal get a split copy.
        let mut normals: Vec<Option<Vec3>> = vec![None; self.vertex_count()];
        let mut split_map: HashMap<(u32, [u32; 3]), u32> = HashMap::new();
        let mut extra: Vec<u32> = Vec::new();
        let mut new_indices = self.indices.clone();
        let base = self.vertex_count() as u32;
        for (c, &v) in self.indices.iter().enumerate() {
            let n = corner_n[c];
            match normals[v as usize] {
                None => normals[v as usize] = Some(n),
                Some(existing) if existing.dot(n) > 0.9999 => {}
                Some(_) => {
                    let key = (v, [n.x.to_bits(), n.y.to_bits(), n.z.to_bits()]);
                    let id = *split_map.entry(key).or_insert_with(|| {
                        extra.push(v);
                        base + extra.len() as u32 - 1
                    });
                    new_indices[c] = id;
                }
            }
        }
        let added = extra.len();
        if added > 0 {
            let copies = self.select_vertices(&extra);
            let mut ids = std::mem::take(&mut self.material_ids);
            self.append(&copies, 0);
            ids.truncate(tri_count);
            self.material_ids = ids;
            self.indices = new_indices.clone();
        }
        let mut out = vec![Vec3::Y; self.vertex_count()];
        for (c, &v) in new_indices.iter().enumerate() {
            out[v as usize] = corner_n[c];
        }
        self.normals = Some(out);
        added
    }

    /// Transforms positions, normals and tangents in place.
    pub fn transform(&mut self, m: glam::Mat4) {
        let nm = glam::Mat3::from_mat4(m).inverse().transpose();
        for p in &mut self.positions {
            *p = m.transform_point3(*p);
        }
        if let Some(ns) = &mut self.normals {
            for n in ns {
                *n = (nm * *n).normalize_or_zero();
            }
        }
        if let Some(ts) = &mut self.tangents {
            let m3 = glam::Mat3::from_mat4(m);
            for t in ts {
                let v = (m3 * t.truncate()).normalize_or_zero();
                *t = v.extend(t.w);
            }
        }
        if m.determinant() < 0.0 {
            for t in self.indices.chunks_exact_mut(3) {
                t.swap(1, 2);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Unit cube with 24 vertices (hard edges), 12 triangles.
    pub fn cube() -> Mesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        let faces = [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y, Vec3::X),
        ];
        for (n, u, v) in faces {
            let b = positions.len() as u32;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                positions.push((n + u * su + v * sv) * 0.5);
            }
            indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        Mesh::new(positions, indices)
    }

    #[test]
    fn cube_is_valid_and_outward() {
        let m = cube();
        m.validate().unwrap();
        assert!((m.surface_area() - 6.0).abs() < 1e-5);
        for t in 0..m.triangle_count() {
            let [a, b, c] = m.triangle_positions(t);
            let n = (b - a).cross(c - a);
            assert!(n.dot((a + b + c) / 3.0) > 0.0, "triangle {t} faces inward");
        }
    }

    #[test]
    fn compact_removes_unused() {
        let mut m = cube();
        m.indices.truncate(6);
        m.material_ids.truncate(2);
        m.compact();
        assert_eq!(m.vertex_count(), 4);
        m.validate().unwrap();
    }

    #[test]
    fn normals_hard_and_smooth() {
        let mut m = cube();
        assert_eq!(m.compute_normals(30.0), 0);
        let n = m.normals.as_ref().unwrap();
        assert!((n[0] - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn append_merges_attributes() {
        let mut a = cube();
        let mut b = cube();
        b.uvs.push(vec![Vec2::ONE; b.vertex_count()]);
        a.append(&b, 1);
        a.validate().unwrap();
        assert_eq!(a.triangle_count(), 24);
        assert_eq!(a.material_ids[23], 1);
        assert_eq!(a.uvs[0][0], Vec2::ZERO);
    }
}
