//! Camera-sampled visibility: cull hidden triangles and weight the visible ones.
//!
//! Renders triangle-ID buffers from sampled cameras through [`RenderHost::render_ids`]
//! (farthing's CPU ray caster by default; hosts can render with real alpha masks), then:
//!
//! * [`cull_invisible`] — delete triangles no camera saw, growing the kept set by a few rings
//!   so sub-pixel misses don't open cracks;
//! * [`VisibilityResult::vertex_importance`] — feed visibility into reduction importance;
//! * [`VisibilityResult::triangle_weights`] — feed visibility into texel budgets.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

use crate::host::{IdRenderRequest, RenderHost, View};
use crate::math::fibonacci_directions;
use crate::scene::TriRef;
use crate::{Instance, Mesh, Result, Scene};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CameraSet {
    /// Cameras on a sphere around the bounds, looking inward.
    Sphere { count: usize, distance_factor: f32 },
    /// Upper hemisphere only (objects never seen from below).
    Hemisphere { count: usize, distance_factor: f32 },
    /// Cube captures (6 × 90° views) from each point — e.g. along a play path.
    Points(Vec<Vec3>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisibilityOptions {
    pub cameras: CameraSet,
    /// Pixels per side of each view.
    pub resolution: u32,
}

impl Default for VisibilityOptions {
    fn default() -> Self {
        VisibilityOptions {
            cameras: CameraSet::Sphere {
                count: 64,
                distance_factor: 2.5,
            },
            resolution: 128,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VisibilityResult {
    /// Per instance, per triangle: number of views that saw it.
    pub views_seen: Vec<Vec<u32>>,
    /// Per instance, per triangle: total pixels covered across all views.
    pub pixels: Vec<Vec<u32>>,
    pub view_count: usize,
}

/// Camera views for a camera set around `scene`.
pub fn camera_views(scene: &Scene, set: &CameraSet, resolution: u32) -> Vec<View> {
    let b = scene.bounds();
    let c = b.center();
    let r = b.radius().max(1e-6);
    let ring = |dirs: Vec<Vec3>, k: f32| {
        let fov = 2.0 * (1.0 / k).clamp(0.0, 1.0).asin() * 1.1;
        dirs.into_iter()
            .map(|d| {
                View::perspective(
                    c + d * r * k,
                    c,
                    fov,
                    resolution,
                    resolution,
                    r * 0.01,
                    r * (k + 2.0),
                )
            })
            .collect::<Vec<_>>()
    };
    match set {
        CameraSet::Sphere {
            count,
            distance_factor,
        } => ring(
            fibonacci_directions(*count, false),
            distance_factor.max(1.05),
        ),
        CameraSet::Hemisphere {
            count,
            distance_factor,
        } => ring(
            fibonacci_directions(*count, true),
            distance_factor.max(1.05),
        ),
        CameraSet::Points(points) => points
            .iter()
            .flat_map(|&p| {
                [
                    Vec3::X,
                    Vec3::NEG_X,
                    Vec3::Y,
                    Vec3::NEG_Y,
                    Vec3::Z,
                    Vec3::NEG_Z,
                ]
                .map(|d| {
                    View::perspective(
                        p,
                        p + d,
                        std::f32::consts::FRAC_PI_2,
                        resolution,
                        resolution,
                        r * 0.001,
                        r * 10.0,
                    )
                })
            })
            .collect(),
    }
}

/// Computes visibility of every triangle of every instance.
pub fn compute_visibility(
    host: &mut dyn RenderHost,
    scene: &Scene,
    opts: &VisibilityOptions,
) -> Result<VisibilityResult> {
    let views = camera_views(scene, &opts.cameras, opts.resolution);
    let ids = host.render_ids(&IdRenderRequest {
        scene,
        views: views.clone(),
    })?;
    let mut views_seen: Vec<Vec<u32>> = scene
        .instances
        .iter()
        .map(|i| vec![0; scene.meshes[i.mesh].triangle_count()])
        .collect();
    let mut pixels = views_seen.clone();
    for img in &ids {
        let mut seen_here: HashSet<TriRef> = HashSet::new();
        for id in img.ids.iter().flatten() {
            pixels[id.instance as usize][id.triangle as usize] += 1;
            seen_here.insert(*id);
        }
        for id in seen_here {
            views_seen[id.instance as usize][id.triangle as usize] += 1;
        }
    }
    Ok(VisibilityResult {
        views_seen,
        pixels,
        view_count: views.len(),
    })
}

impl VisibilityResult {
    pub fn visible_fraction(&self) -> f32 {
        let total: usize = self.views_seen.iter().map(Vec::len).sum();
        let seen: usize = self.views_seen.iter().flatten().filter(|&&v| v > 0).count();
        seen as f32 / total.max(1) as f32
    }

    /// Per-triangle weight in 0..1 in [`Scene::flatten`] order: share of views that saw it.
    pub fn triangle_weights(&self) -> Vec<f32> {
        let n = self.view_count.max(1) as f32;
        self.views_seen
            .iter()
            .flatten()
            .map(|&v| (v as f32 / n).sqrt())
            .collect()
    }

    /// Per-vertex importance for reducing `instance`'s mesh: `floor + (1 - floor) × weight`,
    /// where weight is the max over the vertex's triangles.
    pub fn vertex_importance(&self, scene: &Scene, instance: usize, floor: f32) -> Vec<f32> {
        let mesh = &scene.meshes[scene.instances[instance].mesh];
        let n = self.view_count.max(1) as f32;
        let mut imp = vec![0.0f32; mesh.vertex_count()];
        for t in 0..mesh.triangle_count() {
            let w = (self.views_seen[instance][t] as f32 / n).sqrt();
            for v in mesh.triangle(t) {
                imp[v as usize] = imp[v as usize].max(w);
            }
        }
        imp.into_iter().map(|w| floor + (1.0 - floor) * w).collect()
    }
}

/// Removes triangles no view saw, keeping `grow_rings` rings of neighbours around visible
/// ones. Every instance gets its own mesh in the output. Returns the scene and the number of
/// triangles removed.
pub fn cull_invisible(scene: &Scene, vis: &VisibilityResult, grow_rings: usize) -> (Scene, usize) {
    let mut out = Scene {
        materials: scene.materials.clone(),
        skeletons: scene.skeletons.clone(),
        ..Default::default()
    };
    let mut removed = 0;
    for (ii, inst) in scene.instances.iter().enumerate() {
        let mesh = &scene.meshes[inst.mesh];
        let tc = mesh.triangle_count();
        let mut keep: Vec<bool> = vis.views_seen[ii].iter().map(|&v| v > 0).collect();
        if grow_rings > 0 {
            let key = |p: Vec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
            let mut pos_tris: HashMap<[u32; 3], Vec<usize>> = HashMap::new();
            for t in 0..tc {
                for p in mesh.triangle_positions(t) {
                    pos_tris.entry(key(p)).or_default().push(t);
                }
            }
            for _ in 0..grow_rings {
                let snapshot = keep.clone();
                for t in (0..tc).filter(|&t| snapshot[t]) {
                    for p in mesh.triangle_positions(t) {
                        for &n in &pos_tris[&key(p)] {
                            keep[n] = true;
                        }
                    }
                }
            }
        }
        let mut m = Mesh {
            indices: Vec::new(),
            material_ids: Vec::new(),
            ..mesh.clone()
        };
        for t in 0..tc {
            if keep[t] {
                m.indices.extend_from_slice(&mesh.triangle(t));
                m.material_ids.push(mesh.material_ids[t]);
            } else {
                removed += 1;
            }
        }
        m.compact();
        out.meshes.push(m);
        out.instances.push(Instance {
            mesh: out.meshes.len() - 1,
            ..inst.clone()
        });
    }
    (out, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::GeometryOnlyHost;
    use glam::Mat4;

    fn cube() -> Mesh {
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
                positions.push((n + u * su + v * sv) * 0.5);
            }
            indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        Mesh::new(positions, indices)
    }

    #[test]
    fn hidden_inner_cube_is_culled() {
        let mut s = Scene::from_mesh(cube());
        s.instances.push(Instance {
            name: "inner".into(),
            mesh: 0,
            transform: Mat4::from_scale(Vec3::splat(0.5)),
            skeleton: None,
        });
        let vis = compute_visibility(
            &mut GeometryOnlyHost,
            &s,
            &VisibilityOptions {
                resolution: 48,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            vis.views_seen[0].iter().all(|&v| v > 0),
            "outer cube fully visible"
        );
        assert!(
            vis.views_seen[1].iter().all(|&v| v == 0),
            "inner cube hidden"
        );
        let (culled, removed) = cull_invisible(&s, &vis, 0);
        assert_eq!(removed, 12);
        assert_eq!(culled.triangle_count(), 12);
    }

    #[test]
    fn hemisphere_never_sees_bottom() {
        let s = Scene::from_mesh(cube());
        let vis = compute_visibility(
            &mut GeometryOnlyHost,
            &s,
            &VisibilityOptions {
                cameras: CameraSet::Hemisphere {
                    count: 32,
                    distance_factor: 3.0,
                },
                resolution: 48,
            },
        )
        .unwrap();
        let mesh = &s.meshes[0];
        for t in 0..mesh.triangle_count() {
            let [a, b, c] = mesh.triangle_positions(t);
            if (a + b + c).y / 3.0 < -0.49 {
                assert_eq!(vis.views_seen[0][t], 0);
            }
        }
        let imp = vis.vertex_importance(&s, 0, 0.1);
        assert!(imp.iter().any(|&w| (w - 0.1).abs() < 1e-6));
    }
}
