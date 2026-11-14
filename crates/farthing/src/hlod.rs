//! Hierarchical LOD proxies: many objects in, one mesh with one baked material out.
//!
//! The usual distant-LOD bake of an engine's HLOD system. A cell of a level (buildings, props,
//! rocks, anything) is replaced at distance by a single draw:
//!
//! 1. **Hidden-surface removal.** Triangle-ID buffers are rendered from sampled cameras
//!    ([`crate::visibility`]; the host's renderer when it has one, so alpha masks count) and
//!    every triangle no camera saw is deleted: faces buried in the ground or inside other
//!    objects, the undersides of things standing on the floor, interiors behind closed walls.
//!    Visibility is decided by what the cameras see, not by tests along surface normals.
//! 2. **Merge.** Every instance goes to world space in one mesh, attributes are dropped and
//!    positions welded, so nothing constrains the reduction but the geometry.
//! 3. **Reduce** to a triangle budget or an error bound (whole small parts go first when they
//!    are what costs least), or **remesh** through a voxel grid first, which also fuses
//!    interpenetrating objects into one watertight shell.
//! 4. **Unwrap and bake.** New UVs, tangents, and base color / normal / ORM (or any channels)
//!    cast from the culled source through the host into one atlas.

use crate::bake::{bake_with_host, compute_tangents, BakeOptions, BakeOutput};
use crate::host::RenderHost;
use crate::reduce::{repair, simplify, SeamMode, SimplifyOptions};
use crate::uv::{parameterize, ParamOptions};
use crate::visibility::{compute_visibility, cull_invisible, VisibilityOptions, VisibilityResult};
use crate::voxel::{remesh, RemeshOptions};
use crate::{Error, Mesh, Result, Scene};
use glam::Mat4;
use serde::{Deserialize, Serialize};

/// How the merged surface is brought down to its budget.
#[allow(clippy::large_enum_variant)] // options, built once
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HlodGeometry {
    /// Edge-collapse reduction of the merged surface (keeps silhouettes and thin parts).
    Reduce,
    /// Voxel remesh first, then reduction: one closed shell, gaps up to `close_gaps` cells
    /// sealed, interpenetrating objects fused. Best for clutter seen from far away.
    Remesh(RemeshOptions),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HlodOptions {
    /// Hidden-surface removal (None: keep everything).
    pub visibility: Option<VisibilityOptions>,
    /// Rings of neighbours kept around seen triangles, so sub-pixel misses don't open holes.
    pub grow_rings: usize,
    /// Connected parts smaller than this many visibility pixels are kept or dropped whole
    /// (kept when any camera saw them): culling them triangle by triangle would leave shards
    /// of what a few pixels happened to hit, which no reduction can clean up.
    pub small_part_pixels: f32,
    pub geometry: HlodGeometry,
    pub target_triangles: Option<usize>,
    /// Largest deviation in world units (see [`crate::reduce::screen_size_to_max_error`]).
    pub max_error: Option<f32>,
    /// Reducer settings (targets come from this struct).
    pub simplify: SimplifyOptions,
    /// UV layout of the proxy (None: no UVs; a bake needs them).
    pub uv: Option<ParamOptions>,
    /// Material bake through the host into UV set 0 (None: geometry and UVs only, e.g. to
    /// bake in the engine).
    pub bake: Option<BakeOptions>,
    /// Hard-edge angle for the proxy's normals.
    pub hard_angle_deg: f32,
}

impl Default for HlodOptions {
    fn default() -> Self {
        HlodOptions {
            visibility: Some(VisibilityOptions {
                resolution: 256,
                ..VisibilityOptions::default()
            }),
            grow_rings: 1,
            small_part_pixels: 8.0,
            geometry: HlodGeometry::Reduce,
            target_triangles: Some(2000),
            max_error: None,
            simplify: SimplifyOptions {
                seams: SeamMode::Permissive,
                remove_parts: true,
                ..SimplifyOptions::default()
            },
            uv: Some(ParamOptions {
                atlas_size: 1024,
                padding: 2,
                ..ParamOptions::default()
            }),
            bake: Some(BakeOptions {
                width: 1024,
                height: 1024,
                gutter: 4,
                channels: vec![
                    crate::host::channels::BASE_COLOR.into(),
                    crate::host::channels::NORMAL.into(),
                    crate::host::channels::ORM.into(),
                ],
                ..BakeOptions::default()
            }),
            hard_angle_deg: 60.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HlodResult {
    /// The proxy in world space (one material slot; UV set 0 and tangents when unwrapped).
    pub mesh: Mesh,
    /// The baked material, when [`HlodOptions::bake`] is set.
    pub bake: Option<BakeOutput>,
    pub input_triangles: usize,
    /// Triangles deleted as never seen.
    pub hidden_triangles: usize,
    /// Largest estimated deviation of the reduction (world units; the voxel step adds up to
    /// about a cell on top).
    pub error: f32,
}

/// Builds one proxy for everything in `scene`.
pub fn build_hlod(
    host: &mut dyn RenderHost,
    scene: &Scene,
    opts: &HlodOptions,
) -> Result<HlodResult> {
    let input_triangles = scene.triangle_count();
    if input_triangles == 0 {
        return Err(Error::InvalidInput("HLOD of an empty scene".into()));
    }
    // 1. Hidden-surface removal.
    let (visible, hidden_triangles) = match &opts.visibility {
        Some(v) => {
            let mut vis = compute_visibility(host, scene, v)?;
            // Views frame the bounding sphere: about 2.5 radii across `resolution` pixels.
            let pixel = 2.5 * scene.bounds().radius() / v.resolution.max(1) as f32;
            keep_small_parts_whole(scene, &mut vis, opts.small_part_pixels * pixel);
            cull_invisible(scene, &vis, opts.grow_rings)
        }
        None => (scene.clone(), 0),
    };
    if visible.triangle_count() == 0 {
        return Err(Error::InvalidInput("no camera saw any triangle".into()));
    }
    // 2. One world-space mesh without attributes.
    let mut geo = repair::geometry_only(&visible.flatten().0);
    geo.skin = None;
    geo.morph_targets.clear();
    geo.material_ids.iter_mut().for_each(|m| *m = 0);
    repair::remove_degenerate_triangles(&mut geo, 0.0);
    geo.compact();

    // 3. Reduce (optionally after a voxel remesh).
    let simplify_opts = SimplifyOptions {
        target_ratio: None,
        target_triangles: opts.target_triangles,
        max_error: opts.max_error,
        ..opts.simplify.clone()
    };
    let (mut mesh, error) = match &opts.geometry {
        HlodGeometry::Reduce => {
            if opts
                .target_triangles
                .is_some_and(|t| t >= geo.triangle_count())
            {
                (geo, 0.0)
            } else {
                let r = simplify(&geo, &simplify_opts)?;
                (r.mesh, r.error)
            }
        }
        HlodGeometry::Remesh(ro) => {
            let r = remesh(
                &Scene::from_mesh(geo),
                &RemeshOptions {
                    target_triangles: opts.target_triangles,
                    max_error: opts.max_error,
                    simplify: Some(simplify_opts),
                    ..ro.clone()
                },
            )?;
            (r.mesh, r.cell_size)
        }
    };
    mesh.uvs.clear();
    mesh.tangents = None;
    mesh.compute_normals(opts.hard_angle_deg);

    // 4. Unwrap and bake from the culled source (hidden surfaces can't bleed into the atlas).
    if let Some(uv) = &opts.uv {
        let atlas = opts
            .bake
            .as_ref()
            .map_or(uv.atlas_size, |b| b.width.max(b.height));
        mesh = parameterize(
            &mesh,
            &ParamOptions {
                atlas_size: atlas,
                ..uv.clone()
            },
        )?
        .mesh;
        compute_tangents(&mut mesh, 0)?;
    }
    let bake = match &opts.bake {
        Some(_) if opts.uv.is_none() => {
            return Err(Error::InvalidInput(
                "an HLOD bake needs `uv` options".into(),
            ))
        }
        Some(b) => Some(bake_with_host(
            host,
            &mesh,
            Mat4::IDENTITY,
            &visible,
            &BakeOptions {
                uv_set: 0,
                ..b.clone()
            },
        )?),
        None => None,
    };
    Ok(HlodResult {
        mesh,
        bake,
        input_triangles,
        hidden_triangles,
        error,
    })
}

/// Marks every triangle of a connected part (by welded position) whose world-space diagonal is
/// below `max_diagonal` as seen when any of its triangles is, and as unseen otherwise.
fn keep_small_parts_whole(scene: &Scene, vis: &mut VisibilityResult, max_diagonal: f32) {
    use rayon::prelude::*;
    vis.views_seen
        .par_iter_mut()
        .zip(scene.instances.par_iter())
        .for_each(|(seen, inst)| {
            let mesh = &scene.meshes[inst.mesh];
            let (part, n) = crate::reduce::parts_of(mesh);
            let mut b = vec![crate::Aabb::EMPTY; n];
            let mut any = vec![false; n];
            for t in 0..mesh.triangle_count() {
                let c = part[t];
                for p in mesh.triangle_positions(t) {
                    b[c].extend(inst.transform.transform_point3(p));
                }
                any[c] |= seen[t] > 0;
            }
            for (t, s) in seen.iter_mut().enumerate() {
                let c = part[t];
                if (b[c].max - b[c].min).length() < max_diagonal {
                    *s = any[c] as u32;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::GeometryOnlyHost;
    use crate::visibility::CameraSet;
    use crate::{Instance, MaterialId, MaterialSlot};
    use glam::Vec3;

    /// Axis-aligned box, 2 triangles per face, outward winding, `n` subdivisions per edge.
    fn cube(n: usize) -> Mesh {
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        let faces = [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y, Vec3::X),
        ];
        for (nrm, u, v) in faces {
            let base = pos.len() as u32;
            for j in 0..=n {
                for i in 0..=n {
                    let s = i as f32 / n as f32 * 2.0 - 1.0;
                    let t = j as f32 / n as f32 * 2.0 - 1.0;
                    pos.push(nrm + u * s + v * t);
                }
            }
            let w = n as u32 + 1;
            for j in 0..n as u32 {
                for i in 0..n as u32 {
                    let a = base + j * w + i;
                    idx.extend_from_slice(&[a, a + 1, a + w + 1, a, a + w + 1, a + w]);
                }
            }
        }
        Mesh::new(pos, idx)
    }

    /// A ground slab, a crate standing on it, a box buried inside the crate and a row of
    /// small pebbles (separate instances, as an engine would hand them over).
    fn level() -> Scene {
        let mut s = Scene {
            meshes: vec![cube(4)],
            materials: vec![MaterialSlot {
                name: "m".into(),
                id: MaterialId(0),
            }],
            ..Default::default()
        };
        let mut add = |name: &str, t: Mat4| {
            s.instances.push(Instance {
                name: name.into(),
                mesh: 0,
                transform: t,
                skeleton: None,
            })
        };
        add(
            "ground",
            Mat4::from_scale_rotation_translation(
                Vec3::new(4.0, 0.1, 4.0),
                Default::default(),
                Vec3::new(0.0, -0.1, 0.0),
            ),
        );
        add("crate", Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)));
        add(
            "hidden",
            Mat4::from_scale_rotation_translation(
                Vec3::splat(0.5),
                Default::default(),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        );
        for i in 0..5 {
            let x = -3.0 + i as f32 * 1.5;
            add(
                "pebble",
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(0.1),
                    Default::default(),
                    Vec3::new(x, 0.1, 3.0),
                ),
            );
        }
        s
    }

    #[test]
    fn hidden_surfaces_go_and_the_budget_holds() {
        let s = level();
        let opts = HlodOptions {
            visibility: Some(VisibilityOptions {
                cameras: CameraSet::Hemisphere {
                    count: 48,
                    distance_factor: 2.5,
                },
                resolution: 128,
            }),
            grow_rings: 0,
            target_triangles: Some(200),
            bake: None,
            ..HlodOptions::default()
        };
        let r = build_hlod(&mut GeometryOnlyHost, &s, &opts).unwrap();
        let per_box = s.meshes[0].triangle_count();
        // The buried box (a whole cube) and at least the crate's floor face and the ground's
        // underside never show.
        assert!(
            r.hidden_triangles >= per_box + 2 * per_box / 6,
            "hidden {}",
            r.hidden_triangles
        );
        assert!(
            r.mesh.triangle_count() <= 200,
            "{}",
            r.mesh.triangle_count()
        );
        // Nothing of the proxy is inside the crate, where only the buried box was.
        let b = crate::Aabb::from_points(r.mesh.positions.iter());
        assert!(
            b.max.y > 1.9 && b.min.y < 0.01,
            "{b:?} hidden {}",
            r.hidden_triangles
        );
        for p in &r.mesh.positions {
            let local = *p - Vec3::new(0.0, 1.0, 0.0);
            assert!(
                local.abs().max_element() > 0.9,
                "a vertex inside the crate: {p}"
            );
        }
    }

    #[test]
    fn remesh_fuses_into_one_shell() {
        let s = level();
        let opts = HlodOptions {
            geometry: HlodGeometry::Remesh(RemeshOptions {
                resolution: 48,
                ..RemeshOptions::default()
            }),
            target_triangles: Some(300),
            bake: None,
            ..HlodOptions::default()
        };
        let r = build_hlod(&mut GeometryOnlyHost, &s, &opts).unwrap();
        assert!(r.mesh.triangle_count() <= 300 && r.mesh.triangle_count() > 20);
    }

    /// A host whose "material" colors by world position.
    struct PositionHost;
    impl RenderHost for PositionHost {
        fn material_info(&self, _m: MaterialId) -> crate::host::MaterialInfo {
            Default::default()
        }
        fn bake_surface(
            &mut self,
            req: &crate::host::SurfaceBakeRequest,
        ) -> Result<crate::host::ChannelImages> {
            let mut out = crate::host::ChannelImages::new();
            for c in &req.channels {
                let mut img = crate::image::Image::new(req.width, req.height, glam::Vec4::ZERO);
                for (i, s) in req.samples.iter().enumerate() {
                    if let Some(s) = s {
                        img.pixels[i] = (s.position * 0.1 + 0.5)
                            .clamp(Vec3::ZERO, Vec3::ONE)
                            .extend(1.0);
                    }
                }
                out.insert(
                    c.clone(),
                    crate::host::ChannelImage {
                        image: img,
                        encoding: crate::host::ChannelEncoding::Color,
                    },
                );
            }
            Ok(out)
        }
        fn render_views(
            &mut self,
            _r: &crate::host::ViewRenderRequest,
        ) -> Result<Vec<crate::host::ViewImages>> {
            Err(Error::Unsupported("render_views"))
        }
        fn create_material(&mut self, _d: &crate::host::BakedMaterialDesc) -> Result<MaterialId> {
            Ok(MaterialId(1))
        }
    }

    #[test]
    fn bakes_one_atlas_from_the_visible_source() {
        let opts = HlodOptions {
            target_triangles: Some(300),
            bake: Some(BakeOptions {
                width: 128,
                height: 128,
                supersample: 1,
                gutter: 2,
                channels: vec![crate::host::channels::BASE_COLOR.into()],
                ..BakeOptions::default()
            }),
            ..HlodOptions::default()
        };
        let r = build_hlod(&mut PositionHost, &level(), &opts).unwrap();
        let b = r.bake.unwrap();
        assert_eq!(r.mesh.uvs.len(), 1);
        // Coverage is over the whole atlas: nearly every texel inside a chart finds source.
        let uv = &r.mesh.uvs[0];
        let chart_area: f32 = (0..r.mesh.triangle_count())
            .map(|t| {
                let [a, b, c] = r.mesh.triangle(t).map(|v| uv[v as usize]);
                (b - a).perp_dot(c - a).abs() * 0.5
            })
            .sum();
        assert!(
            b.coverage > 0.9 * chart_area,
            "coverage {} of {chart_area}",
            b.coverage
        );
        assert!(r.mesh.tangents.is_some());
    }
}
