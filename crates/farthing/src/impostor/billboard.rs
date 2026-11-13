//! Billboard clouds (Décoret, Durand, Sillion & Dorsey 2003): approximate geometry with a few
//! textured planes chosen greedily by how much surface area each plane can represent within
//! a distance tolerance.

use crate::host::{
    channels, BakedMaterialDesc, BakedMaterialKind, ChannelImage, ChannelImages, MaterialClass,
    RenderHost, View, ViewRenderRequest,
};
use crate::image::Image;
use crate::math::{fibonacci_directions, orthonormal_basis, triangle_area};
use crate::uv::pack_rects;
use crate::{Error, Instance, MaterialId, Mesh, Result, Scene};
use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BillboardOptions {
    pub max_planes: usize,
    /// Distance tolerance relative to the bounding-box diagonal.
    pub epsilon: f32,
    /// Candidate plane orientations (hemisphere directions).
    pub direction_samples: usize,
    pub atlas_size: u32,
}

impl Default for BillboardOptions {
    fn default() -> Self {
        BillboardOptions {
            max_planes: 24,
            epsilon: 0.03,
            direction_samples: 96,
            atlas_size: 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BillboardPlane {
    pub normal: Vec3,
    pub offset: f32,
    pub faces: Vec<u32>,
    /// Plane-space basis and rectangle covering the projected faces.
    pub axis_u: Vec3,
    pub axis_v: Vec3,
    pub min: Vec2,
    pub max: Vec2,
}

impl BillboardPlane {
    pub fn corners(&self) -> [Vec3; 4] {
        let o = self.normal * self.offset;
        let at = |u: f32, v: f32| o + self.axis_u * u + self.axis_v * v;
        [
            at(self.min.x, self.min.y),
            at(self.max.x, self.min.y),
            at(self.max.x, self.max.y),
            at(self.min.x, self.max.y),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct BillboardCloud {
    pub planes: Vec<BillboardPlane>,
    /// Faces covered within tolerance (the rest were snapped to their nearest plane).
    pub covered_area_fraction: f32,
}

/// Chooses planes for `mesh` (world space).
pub fn billboard_cloud(mesh: &Mesh, opts: &BillboardOptions) -> Result<BillboardCloud> {
    let tc = mesh.triangle_count();
    if tc == 0 {
        return Err(Error::InvalidMesh("no triangles".into()));
    }
    let eps = mesh.bounds().diagonal() * opts.epsilon;
    let dirs = fibonacci_directions(opts.direction_samples.max(6), true);
    let tris: Vec<[Vec3; 3]> = (0..tc).map(|t| mesh.triangle_positions(t)).collect();
    let normals: Vec<Vec3> = tris
        .iter()
        .map(|t| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero())
        .collect();
    let areas: Vec<f32> = tris
        .iter()
        .map(|t| triangle_area(t[0], t[1], t[2]))
        .collect();
    let total_area: f32 = areas.iter().sum();
    let mut assigned = vec![false; tc];
    let mut planes: Vec<BillboardPlane> = Vec::new();
    let mut covered = 0.0f32;
    while planes.len() < opts.max_planes && assigned.iter().any(|a| !a) {
        // Vote: for each direction, bins of width eps over plane offsets.
        let mut best: Option<(f32, Vec3, f32)> = None;
        // Candidates: sampled directions plus the normals of the largest unassigned faces.
        let mut cand = dirs.clone();
        let mut big: Vec<usize> = (0..tc).filter(|&t| !assigned[t]).collect();
        big.sort_by(|&a, &b| areas[b].total_cmp(&areas[a]));
        for &t in big.iter().take(64) {
            let n = normals[t];
            let n = if n.y < 0.0 || (n.y == 0.0 && (n.x < 0.0 || (n.x == 0.0 && n.z < 0.0))) {
                -n
            } else {
                n
            };
            if n != Vec3::ZERO && !cand.iter().any(|c| c.dot(n) > 0.9999) {
                cand.push(n);
            }
        }
        for &d in &cand {
            let mut bins: BTreeMap<i64, f32> = BTreeMap::new();
            for t in (0..tc).filter(|&t| !assigned[t]) {
                let p = tris[t].map(|v| v.dot(d));
                let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
                if hi - lo > 2.0 * eps {
                    continue;
                }
                let score = areas[t] * normals[t].dot(d).abs();
                let (b0, b1) = (
                    ((hi - eps) / eps).floor() as i64,
                    ((lo + eps) / eps).floor() as i64,
                );
                for b in b0..=b1 {
                    *bins.entry(b).or_default() += score;
                }
            }
            for (&b, &s) in &bins {
                if best.is_none_or(|x| s > x.0) {
                    best = Some((s, d, (b as f32 + 0.5) * eps));
                }
            }
        }
        let Some((score, d, off)) = best else { break };
        if score <= 0.0 {
            break;
        }
        let faces: Vec<u32> = (0..tc)
            .filter(|&t| !assigned[t] && tris[t].iter().all(|v| (v.dot(d) - off).abs() <= eps))
            .map(|t| t as u32)
            .collect();
        if faces.is_empty() {
            break;
        }
        for &f in &faces {
            assigned[f as usize] = true;
            covered += areas[f as usize];
        }
        let (u, v) = orthonormal_basis(d);
        planes.push(BillboardPlane {
            normal: d,
            offset: off,
            faces,
            axis_u: u,
            axis_v: v,
            min: Vec2::ZERO,
            max: Vec2::ZERO,
        });
    }
    if planes.is_empty() {
        return Err(Error::InvalidInput(
            "no billboard plane could be fitted".into(),
        ));
    }
    // Snap leftovers to the plane with the smallest max distance.
    for t in (0..tc).filter(|&t| !assigned[t]) {
        let best = planes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    i,
                    tris[t]
                        .iter()
                        .map(|v| (v.dot(p.normal) - p.offset).abs())
                        .fold(0.0f32, f32::max),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        planes[best].faces.push(t as u32);
    }
    for p in &mut planes {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for &f in &p.faces {
            for v in &tris[f as usize] {
                let q = Vec2::new(v.dot(p.axis_u), v.dot(p.axis_v));
                min = min.min(q);
                max = max.max(q);
            }
        }
        p.min = min;
        p.max = max;
    }
    Ok(BillboardCloud {
        planes,
        covered_area_fraction: covered / total_area.max(1e-12),
    })
}

/// Fits planes to the whole scene, has the host render each plane's faces onto it, packs
/// the cards into one atlas and returns (card mesh with UVs, atlas, material).
pub fn build_billboard_cloud(
    host: &mut dyn RenderHost,
    scene: &Scene,
    opts: &BillboardOptions,
) -> Result<(Mesh, ChannelImages, BillboardCloud, MaterialId)> {
    let (flat, _) = scene.flatten();
    let cloud = billboard_cloud(&flat, opts)?;
    let atlas = opts.atlas_size;
    let total: f32 = cloud
        .planes
        .iter()
        .map(|p| (p.max - p.min).x * (p.max - p.min).y)
        .sum();
    let mut density = (0.7 * (atlas * atlas) as f32 / total.max(1e-12)).sqrt();
    let mut placed = None;
    for _ in 0..60 {
        let sizes: Vec<(u32, u32)> = cloud
            .planes
            .iter()
            .map(|p| {
                let e = (p.max - p.min) * density;
                (
                    (e.x.ceil() as u32).clamp(2, atlas),
                    (e.y.ceil() as u32).clamp(2, atlas),
                )
            })
            .collect();
        if let Some(pl) = pack_rects(&sizes, atlas, atlas, 2, false) {
            placed = Some(pl);
            break;
        }
        density *= 0.92;
    }
    let placed = placed.ok_or_else(|| Error::InvalidInput("cards do not fit the atlas".into()))?;

    let mut color = Image::new(atlas, atlas, Vec4::ZERO);
    let mut normal = Image::new(atlas, atlas, Vec4::ZERO);
    let mut cards = Mesh::default();
    for (p, pl) in cloud.planes.iter().zip(&placed) {
        // Sub-scene: only this plane's faces, materials preserved.
        let mut sub = Mesh {
            indices: Vec::new(),
            material_ids: Vec::new(),
            ..flat.clone()
        };
        for &f in &p.faces {
            sub.indices.extend_from_slice(&flat.triangle(f as usize));
            sub.material_ids.push(flat.material_ids[f as usize]);
        }
        sub.compact();
        let sub_scene = Scene {
            meshes: vec![sub],
            materials: scene.materials.clone(),
            instances: vec![Instance {
                name: "card".into(),
                mesh: 0,
                transform: glam::Mat4::IDENTITY,
                skeleton: None,
            }],
            skeletons: Vec::new(),
        };
        let mid = (p.min + p.max) * 0.5;
        let half = (p.max - p.min) * 0.5;
        let center = p.normal * p.offset + p.axis_u * mid.x + p.axis_v * mid.y;
        let depth = scene.bounds().diagonal().max(1e-3);
        let view = View::orthographic(
            center,
            -p.normal,
            p.axis_v,
            half.x.max(1e-6),
            half.y.max(1e-6),
            depth,
            pl.w,
            pl.h,
        );
        let imgs = host.render_views(&ViewRenderRequest {
            scene: &sub_scene,
            views: vec![view],
            channels: vec![channels::BASE_COLOR.into(), channels::NORMAL.into()],
            lit: false,
        })?;
        let img = imgs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Host("no image returned".into()))?;
        let cover = |mut i: Image| {
            for (px, d) in i.pixels.iter_mut().zip(&img.depth.pixels) {
                px.w = if d.w > 0.0 { 1.0 } else { 0.0 };
            }
            i
        };
        if let Some(c) = img.channels.get(channels::BASE_COLOR) {
            color.blit(&cover(c.image.clone()), pl.x, pl.y);
        }
        if let Some(n) = img.channels.get(channels::NORMAL) {
            normal.blit(&cover(n.image.clone()), pl.x, pl.y);
        }
        // Card quad. The view's right vector is axis_v × (−normal)... derive it from the view.
        let right = view.view.inverse().x_axis.truncate();
        let up = view.view.inverse().y_axis.truncate();
        let base = cards.vertex_count() as u32;
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let mut q = Mesh::new(
            corners
                .iter()
                .map(|&(sx, sy)| center + right * half.x * sx + up * half.y * sy)
                .collect(),
            vec![0, 1, 2, 0, 2, 3],
        );
        q.normals = Some(vec![p.normal; 4]);
        q.uvs = vec![corners
            .iter()
            .map(|&(sx, sy)| {
                Vec2::new(
                    pl.x as f32 + (sx * 0.5 + 0.5) * pl.w as f32,
                    pl.y as f32 + (0.5 - sy * 0.5) * pl.h as f32,
                ) / atlas as f32
            })
            .collect()];
        let _ = base;
        cards.append(&q, 0);
    }
    let mut textures = ChannelImages::new();
    let color_cov = color.dilate(2);
    for (p, c) in color.pixels.iter_mut().zip(&color_cov) {
        if !c {
            p.w = 0.0;
        }
    }
    normal.dilate(2);
    textures.insert(
        channels::BASE_COLOR.into(),
        ChannelImage {
            image: color,
            encoding: crate::host::ChannelEncoding::Color,
        },
    );
    textures.insert(
        channels::NORMAL.into(),
        ChannelImage {
            image: normal,
            encoding: crate::host::ChannelEncoding::NormalWorld,
        },
    );
    let material = host.create_material(&BakedMaterialDesc {
        name: "billboard_cloud".into(),
        kind: BakedMaterialKind::Cards,
        class: MaterialClass::Masked { cutoff: 0.5 },
        textures: textures.clone(),
        parameters: BTreeMap::new(),
    })?;
    Ok((cards, textures, cloud, material))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossed_quads_need_two_planes() {
        // Two perpendicular quads, like a cheap tree.
        let mut m = Mesh::new(
            vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 2.0, 0.0),
                Vec3::new(-1.0, 2.0, 0.0),
                Vec3::new(0.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 2.0, 1.0),
                Vec3::new(0.0, 2.0, -1.0),
            ],
            vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        );
        m.material_ids = vec![0; 4];
        let c = billboard_cloud(
            &m,
            &BillboardOptions {
                max_planes: 4,
                epsilon: 0.01,
                direction_samples: 200,
                atlas_size: 256,
            },
        )
        .unwrap();
        assert_eq!(
            c.planes.len(),
            2,
            "{:?}",
            c.planes.iter().map(|p| p.normal).collect::<Vec<_>>()
        );
        assert!(c.covered_area_fraction > 0.99);
        for p in &c.planes {
            assert_eq!(p.faces.len(), 2);
        }
    }
}
