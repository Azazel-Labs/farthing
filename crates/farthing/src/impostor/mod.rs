//! Impostors: octahedral and flipbook captures, billboard clouds and surfel proxies.
//!
//! farthing plans every capture (directions, camera matrices, atlas layout); the host renders
//! the views with its real materials ([`RenderHost::render_views`]); farthing assembles,
//! dilates and emits runtime parameters.
//!
//! * Octahedral / hemi-octahedral (Brucks 2018): frames on an octahedral grid, 3-frame blend
//!   at runtime with per-frame virtual-plane reprojection and depth parallax. Reference
//!   shader: [`REFERENCE_SHADER`]; the atlas carries a [`HEIGHT`] channel.
//! * Flipbook: frames around the up axis.
//! * Single view: one orthographic capture onto one quad.
//! * Billboard clouds (Décoret et al. 2003): greedy plane fitting, faces rendered onto a few
//!   textured cards.
//! * Surfel proxies ([`surfels`]): blue-noise oriented discs carrying normals, for relightable
//!   distant LODs of foliage-like geometry. Progressive order (every prefix is an even
//!   covering, for continuous LOD) and a compact streaming format. [`footprint`] measures the
//!   patch of surface each surfel stands for (anisotropic shape, coverage, mean normal and
//!   its spread) and averages host attributes over it. [`splats`] turns a cloud into 3D
//!   Gaussians shaped by those footprints and writes the standard 3D Gaussian Splatting `.ply`.
//! * Layered impostors ([`build_layered_impostor`]): a second, depth-peeled layer per frame for
//!   concave props.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

mod billboard;
pub mod footprint;
pub mod splats;
pub mod surfels;

pub use billboard::{
    billboard_cloud, build_billboard_cloud, BillboardCloud, BillboardOptions, BillboardPlane,
};

use crate::host::{
    channels, BakedMaterialDesc, BakedMaterialKind, ChannelImage, ChannelImages, MaterialClass,
    RenderHost, View, ViewImages, ViewRenderRequest,
};
use crate::image::Image;
use crate::{Aabb, Error, MaterialId, Mesh, Result, Scene};
use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Reference WGSL runtime shader for octahedral impostors.
pub const REFERENCE_SHADER: &str = include_str!("../../shaders/octahedral_impostor.wgsl");

#[inline]
fn sign_nz(v: f32) -> f32 {
    if v >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Full octahedral encoding, y-up: unit direction → [0, 1]².
pub fn oct_encode(d: Vec3) -> Vec2 {
    let o = d / (d.x.abs() + d.y.abs() + d.z.abs()).max(1e-20);
    let p = if o.y < 0.0 {
        Vec2::new(
            (1.0 - o.z.abs()) * sign_nz(o.x),
            (1.0 - o.x.abs()) * sign_nz(o.z),
        )
    } else {
        Vec2::new(o.x, o.z)
    };
    p * 0.5 + 0.5
}

/// Inverse of [`oct_encode`].
pub fn oct_decode(uv: Vec2) -> Vec3 {
    let p = uv * 2.0 - 1.0;
    let y = 1.0 - p.x.abs() - p.y.abs();
    let (x, z) = if y < 0.0 {
        (
            (1.0 - p.y.abs()) * sign_nz(p.x),
            (1.0 - p.x.abs()) * sign_nz(p.y),
        )
    } else {
        (p.x, p.y)
    };
    Vec3::new(x, y, z).normalize()
}

/// Hemi-octahedral encoding (upper hemisphere, y ≥ 0) onto the whole square.
pub fn hemi_oct_encode(d: Vec3) -> Vec2 {
    let o = d / (d.x.abs() + d.y.abs() + d.z.abs()).max(1e-20);
    Vec2::new(o.x + o.z, o.z - o.x) * 0.5 + 0.5
}

/// Inverse of [`hemi_oct_encode`].
pub fn hemi_oct_decode(uv: Vec2) -> Vec3 {
    let a = uv * 2.0 - 1.0;
    let x = (a.x - a.y) * 0.5;
    let z = (a.x + a.y) * 0.5;
    Vec3::new(x, (1.0 - x.abs() - z.abs()).max(0.0), z).normalize()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImpostorKind {
    Octahedral {
        frames_per_side: u32,
        hemisphere: bool,
    },
    Flipbook {
        frames: u32,
        elevation_deg: f32,
    },
    SingleView {
        direction: Vec3,
    },
}

/// Everything needed to capture and assemble an impostor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapturePlan {
    pub kind: ImpostorKind,
    pub center: Vec3,
    pub radius: f32,
    pub frame_size: u32,
    pub cols: u32,
    pub rows: u32,
    /// Direction from the center toward each camera, row-major over (cols × rows).
    pub directions: Vec<Vec3>,
}

impl CapturePlan {
    pub fn views(&self) -> Vec<View> {
        self.directions
            .iter()
            .map(|&d| {
                View::orthographic(
                    self.center,
                    -d,
                    Vec3::Y,
                    self.radius,
                    self.radius,
                    self.radius * 2.0,
                    self.frame_size,
                    self.frame_size,
                )
            })
            .collect()
    }
    pub fn atlas_size(&self) -> (u32, u32) {
        (self.cols * self.frame_size, self.rows * self.frame_size)
    }
}

/// Plans the captures for `kind` around `bounds`.
pub fn plan(bounds: Aabb, kind: &ImpostorKind, frame_size: u32) -> Result<CapturePlan> {
    let center = bounds.center();
    let radius = bounds.radius().max(1e-6);
    let (cols, rows, directions) = match kind {
        ImpostorKind::Octahedral {
            frames_per_side,
            hemisphere,
        } => {
            let n = *frames_per_side;
            if n < 2 {
                return Err(Error::InvalidInput(
                    "octahedral impostors need at least 2 frames per side".into(),
                ));
            }
            let mut dirs = Vec::with_capacity((n * n) as usize);
            for j in 0..n {
                for i in 0..n {
                    let uv = Vec2::new(i as f32, j as f32) / (n - 1) as f32;
                    dirs.push(if *hemisphere {
                        hemi_oct_decode(uv)
                    } else {
                        oct_decode(uv)
                    });
                }
            }
            (n, n, dirs)
        }
        ImpostorKind::Flipbook {
            frames,
            elevation_deg,
        } => {
            let e = elevation_deg.to_radians();
            let dirs: Vec<Vec3> = (0..*frames)
                .map(|k| {
                    let th = k as f32 / *frames as f32 * std::f32::consts::TAU;
                    Vec3::new(th.cos() * e.cos(), e.sin(), th.sin() * e.cos())
                })
                .collect();
            let cols = (*frames as f32).sqrt().ceil() as u32;
            let rows = frames.div_ceil(cols);
            (cols, rows, dirs)
        }
        ImpostorKind::SingleView { direction } => (1, 1, vec![direction.normalize()]),
    };
    Ok(CapturePlan {
        kind: kind.clone(),
        center,
        radius,
        frame_size,
        cols,
        rows,
        directions,
    })
}

/// Lays captured views into atlases (one per channel), dilating each frame by `gutter`.
pub fn assemble(plan: &CapturePlan, views: &[ViewImages], gutter: u32) -> Result<ChannelImages> {
    if views.len() != plan.directions.len() {
        return Err(Error::Host(format!(
            "expected {} views, got {}",
            plan.directions.len(),
            views.len()
        )));
    }
    let (w, h) = plan.atlas_size();
    let mut out = ChannelImages::new();
    for (name, first) in &views[0].channels {
        let mut atlas = Image::new(w, h, Vec4::ZERO);
        for (k, v) in views.iter().enumerate() {
            let mut frame = v
                .channels
                .get(name)
                .ok_or_else(|| Error::Host(format!("view {k} lacks channel `{name}`")))?
                .image
                .clone();
            // Coverage from depth alpha (fractional for supersampled captures: the reference
            // shader alpha-tests at 0.5, so the silhouette lands where half a texel is covered).
            for (p, d) in frame.pixels.iter_mut().zip(&v.depth.pixels) {
                p.w = d.w.clamp(0.0, 1.0);
            }
            let covered = frame.dilate(gutter);
            // Dilated texels keep alpha 0 so alpha testing still cuts the silhouette.
            for (p, c) in frame.pixels.iter_mut().zip(&covered) {
                if !c {
                    p.w = 0.0;
                }
            }
            let (col, row) = (k as u32 % plan.cols, k as u32 / plan.cols);
            atlas.blit(&frame, col * plan.frame_size, row * plan.frame_size);
        }
        out.insert(
            name.clone(),
            ChannelImage {
                image: atlas,
                encoding: first.encoding,
            },
        );
    }
    // Height above each frame's capture plane (in units of the radius), from linear depth: the
    // orthographic capture camera sits 2r in front of the center, so h = (2r − depth) / r.
    let mut height = Image::new(w, h, Vec4::ZERO);
    for (k, v) in views.iter().enumerate() {
        let mut frame = Image::new(plan.frame_size, plan.frame_size, Vec4::ZERO);
        for (p, d) in frame.pixels.iter_mut().zip(&v.depth.pixels) {
            if d.w > 0.0 {
                let hh = ((2.0 * plan.radius - d.x) / plan.radius).clamp(-1.0, 1.0);
                *p = Vec4::new(hh, hh, hh, 1.0);
            }
        }
        frame.dilate(gutter);
        let (col, row) = (k as u32 % plan.cols, k as u32 / plan.cols);
        height.blit(&frame, col * plan.frame_size, row * plan.frame_size);
    }
    out.insert(
        HEIGHT.into(),
        ChannelImage {
            image: height,
            encoding: crate::host::ChannelEncoding::Data,
        },
    );
    Ok(out)
}

/// Atlas channel with each texel's height above its frame's capture plane, in units of the
/// impostor radius (−1..1). Engines pack it next to the normal (the reference shader reads it
/// from the normal atlas alpha) for per-frame parallax refinement.
pub const HEIGHT: &str = "height";

/// Runtime parameters for an impostor (serialize next to the atlas).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpostorParams {
    pub kind: ImpostorKind,
    pub center: Vec3,
    pub radius: f32,
    pub frame_size: u32,
    pub cols: u32,
    pub rows: u32,
    /// Depth layers captured per frame (1, or 2 for [`build_layered_impostor`]).
    #[serde(default = "one")]
    pub layers: u32,
}

fn one() -> u32 {
    1
}

/// Atlas channel name of `channel` for depth layer `layer` (0 = front): the second layer's
/// atlases are `base_color.layer2`, `normal.layer2` and `height.layer2`.
pub fn layer_channel(channel: &str, layer: u32) -> String {
    if layer == 0 {
        channel.to_string()
    } else {
        format!("{channel}.layer{}", layer + 1)
    }
}

/// Depth gap (in units of the impostor radius) below which a surface behind the front layer is
/// treated as the same surface when peeling the second layer.
pub const LAYER_BIAS: f32 = 0.01;

#[derive(Clone, Debug)]
pub struct ImpostorAsset {
    pub atlas: ChannelImages,
    /// Camera-facing quad: 4 corners in [-1, 1]² (xy), expanded in the vertex shader.
    pub quad: Mesh,
    pub params: ImpostorParams,
    pub material: Option<MaterialId>,
}

/// Box-filters a capture rendered at `factor`× the frame size down to the frame size: each
/// channel averages its covered subpixels (coverage from the depth alpha), and the depth image
/// keeps the covered fraction in alpha. Normal channels are averaged as encoded; the reference
/// shader renormalizes.
pub fn downsample_view(v: &ViewImages, factor: u32) -> ViewImages {
    if factor <= 1 {
        return v.clone();
    }
    let mask = |img: &Image| -> Image {
        let mut m = img.clone();
        for (p, d) in m.pixels.iter_mut().zip(&v.depth.pixels) {
            p.w = if d.w > 0.0 { 1.0 } else { 0.0 };
        }
        m
    };
    let mut channels = ChannelImages::new();
    for (name, ch) in &v.channels {
        channels.insert(
            name.clone(),
            ChannelImage {
                image: mask(&ch.image).downsample_coverage(factor),
                encoding: ch.encoding,
            },
        );
    }
    ViewImages {
        channels,
        depth: mask(&v.depth).downsample_coverage(factor),
    }
}

/// Frame size (pixels per frame side) for an impostor whose bounding sphere spans
/// `screen_diameter_px` pixels at the distance where the impostor takes over: one atlas texel per
/// screen pixel there, rounded up to a multiple of 4 and clamped to 8..=1024. Closer than that
/// distance the impostor magnifies; switch to geometry instead.
pub fn frame_size_for_screen(screen_diameter_px: f32) -> u32 {
    let px = screen_diameter_px.max(1.0).ceil() as u32;
    (px.div_ceil(4) * 4).clamp(8, 1024)
}

/// Plans, has the host capture, and assembles an impostor for `scene`. Frames are captured at
/// twice the frame size and box-filtered down (anti-aliased silhouettes and texels); see
/// [`build_impostor_supersampled`] to choose the factor.
pub fn build_impostor(
    host: &mut dyn RenderHost,
    scene: &Scene,
    kind: &ImpostorKind,
    frame_size: u32,
    create_material: bool,
) -> Result<ImpostorAsset> {
    build_impostor_supersampled(host, scene, kind, frame_size, create_material, 2)
}

/// [`build_impostor`] with an explicit supersampling factor (1 = no anti-aliasing).
pub fn build_impostor_supersampled(
    host: &mut dyn RenderHost,
    scene: &Scene,
    kind: &ImpostorKind,
    frame_size: u32,
    create_material: bool,
    supersample: u32,
) -> Result<ImpostorAsset> {
    build(
        host,
        scene,
        kind,
        frame_size,
        create_material,
        supersample,
        1,
        None,
    )
}

/// Two-layer impostor for concave props. Each frame also stores the nearest front-facing
/// surface behind the first one (depth peeling through
/// [`RenderHost::render_views_behind`]): the surface a slightly different view direction
/// uncovers, which a single layer can only stretch over. The second layer's atlases are named
/// by [`layer_channel`]; the reference shader marches both. Twice the atlas memory. Fails with
/// [`Error::Unsupported`] if the host can't peel.
pub fn build_layered_impostor(
    host: &mut dyn RenderHost,
    scene: &Scene,
    kind: &ImpostorKind,
    frame_size: u32,
    create_material: bool,
) -> Result<ImpostorAsset> {
    build_layered_impostor_with(
        host,
        scene,
        kind,
        frame_size,
        create_material,
        &LayerOptions {
            half_resolution: false,
            auto: false,
            ..Default::default()
        },
    )
}

/// How [`build_layered_impostor_with`] uses the second layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerOptions {
    /// Store the second layer at half the frame size (a quarter of its texels): the second
    /// layer fills disoccluded surface, which is seen at a glancing angle and rarely needs full
    /// resolution. Needs an even frame size; ignored otherwise.
    pub half_resolution: bool,
    /// Keep the second layer only where it pays: if [`layer_benefit`] is below
    /// `auto_threshold`, it is dropped and a single-layer impostor is returned. Hosts that
    /// can't peel also get a single-layer impostor instead of an error.
    pub auto: bool,
    /// See `auto`. Default 0.25.
    pub auto_threshold: f32,
}

impl Default for LayerOptions {
    fn default() -> Self {
        LayerOptions {
            half_resolution: true,
            auto: true,
            auto_threshold: 0.25,
        }
    }
}

/// How much a captured second layer can add, from the atlases alone: the share of first-layer
/// texels with a second-layer surface behind them, times their mean depth gap (in radii).
/// Cavities score high (a vase: 0.51), self-occluding but convex-ish shapes low (a torus knot:
/// 0.14, foliage: 0.03). None without a second layer at the first layer's resolution.
pub fn layer_benefit(atlas: &ChannelImages) -> Option<f32> {
    let c0 = &atlas.get(channels::BASE_COLOR)?.image;
    let c1 = &atlas.get(&layer_channel(channels::BASE_COLOR, 1))?.image;
    let h0 = &atlas.get(HEIGHT)?.image;
    let h1 = &atlas.get(&layer_channel(HEIGHT, 1))?.image;
    if (c0.width, c0.height) != (c1.width, c1.height) {
        return None;
    }
    let (mut n0, mut n1, mut gap) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..c0.pixels.len() {
        if c0.pixels[i].w >= 0.5 {
            n0 += 1.0;
            if c1.pixels[i].w >= 0.5 {
                n1 += 1.0;
                gap += (h0.pixels[i].x - h1.pixels[i].x).max(0.0) as f64;
            }
        }
    }
    if n0 == 0.0 || n1 == 0.0 {
        return Some(0.0);
    }
    Some((n1 / n0 * gap / n1) as f32)
}

/// Layered impostor with options: half-resolution second layer, and automatic fallback to one
/// layer where the second wouldn't pay ([`LayerOptions`]).
pub fn build_layered_impostor_with(
    host: &mut dyn RenderHost,
    scene: &Scene,
    kind: &ImpostorKind,
    frame_size: u32,
    create_material: bool,
    opts: &LayerOptions,
) -> Result<ImpostorAsset> {
    match build(
        host,
        scene,
        kind,
        frame_size,
        create_material,
        2,
        2,
        Some(opts),
    ) {
        Err(Error::Unsupported(_)) if opts.auto => {
            build(host, scene, kind, frame_size, create_material, 2, 1, None)
        }
        r => r,
    }
}

fn build(
    host: &mut dyn RenderHost,
    scene: &Scene,
    kind: &ImpostorKind,
    frame_size: u32,
    create_material: bool,
    supersample: u32,
    layers: u32,
    layer_opts: Option<&LayerOptions>,
) -> Result<ImpostorAsset> {
    let ss = supersample.max(1);
    let mut layers = layers;
    let p = plan(scene.bounds(), kind, frame_size)?;
    let capture = CapturePlan {
        frame_size: frame_size * ss,
        ..p.clone()
    };
    let req = ViewRenderRequest {
        scene,
        views: capture.views(),
        channels: vec![channels::BASE_COLOR.into(), channels::NORMAL.into()],
        lit: false,
    };
    let full = host.render_views(&req)?;
    let images: Vec<ViewImages> = full.iter().map(|v| downsample_view(v, ss)).collect();
    let mut atlas = assemble(&p, &images, 4)?;
    if layers >= 2 {
        let behind: Vec<Image> = full.iter().map(|v| v.depth.clone()).collect();
        let back_full = host.render_views_behind(&req, &behind, LAYER_BIAS * p.radius)?;
        let back: Vec<ViewImages> = back_full.iter().map(|v| downsample_view(v, ss)).collect();
        let mut back_atlas = ChannelImages::new();
        for (name, ch) in assemble(&p, &back, 4)? {
            back_atlas.insert(layer_channel(&name, 1), ch);
        }
        let mut with_back = atlas.clone();
        with_back.extend(back_atlas.clone());
        let keep = match layer_opts {
            Some(o) if o.auto => layer_benefit(&with_back).unwrap_or(0.0) >= o.auto_threshold,
            _ => true,
        };
        if !keep {
            layers = 1;
        } else if layer_opts.is_some_and(|o| o.half_resolution)
            && frame_size.is_multiple_of(2)
            && frame_size >= 8
        {
            // Half resolution: the same captures box-filtered twice as far, same layout
            // (normalized atlas coordinates match the first layer's).
            let half = CapturePlan {
                frame_size: frame_size / 2,
                ..p.clone()
            };
            let small: Vec<ViewImages> = back_full
                .iter()
                .map(|v| downsample_view(v, 2 * ss))
                .collect();
            for (name, ch) in assemble(&half, &small, 2)? {
                atlas.insert(layer_channel(&name, 1), ch);
            }
        } else {
            atlas = with_back;
        }
    }
    let mut quad = Mesh::new(
        vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ],
        vec![0, 1, 2, 0, 2, 3],
    );
    quad.uvs = vec![vec![
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
    ]];
    let params = ImpostorParams {
        kind: kind.clone(),
        center: p.center,
        radius: p.radius,
        frame_size,
        cols: p.cols,
        rows: p.rows,
        layers: layers.clamp(1, 2),
    };
    let material = if create_material {
        let kind = match kind {
            ImpostorKind::Octahedral {
                frames_per_side,
                hemisphere,
            } => BakedMaterialKind::OctahedralImpostor {
                frames_per_side: *frames_per_side,
                hemisphere: *hemisphere,
            },
            ImpostorKind::Flipbook { frames, .. } => {
                BakedMaterialKind::Flipbook { frames: *frames }
            }
            ImpostorKind::SingleView { .. } => BakedMaterialKind::Cards,
        };
        Some(host.create_material(&BakedMaterialDesc {
            name: "impostor".into(),
            kind,
            class: MaterialClass::Masked { cutoff: 0.5 },
            textures: atlas.clone(),
            parameters: BTreeMap::from([
                ("radius".into(), p.radius),
                ("layers".into(), params.layers as f32),
            ]),
        })?)
    } else {
        None
    };
    Ok(ImpostorAsset {
        atlas,
        quad,
        params,
        material,
    })
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::host::{
        ChannelEncoding, MaterialInfo, SurfaceBakeRequest, ViewImages, ViewRenderRequest,
    };

    /// Covers every pixel at depth 1 on the first layer; peeling covers the left half at
    /// depth 3 (and checks it was asked behind the first layer's depth).
    struct Peeler {
        peel: bool,
    }

    fn images(v: &View, depth: f32, left_only: bool) -> ViewImages {
        let mut d = Image::new(v.width, v.height, Vec4::ZERO);
        for y in 0..v.height {
            for x in 0..v.width {
                if !left_only || x < v.width / 2 {
                    d.pixels[(y * v.width + x) as usize] = Vec4::new(depth, 0.0, 0.0, 1.0);
                }
            }
        }
        let mut channels = ChannelImages::new();
        channels.insert(
            channels::BASE_COLOR.into(),
            ChannelImage {
                image: Image::new(v.width, v.height, Vec4::new(depth / 4.0, 0.0, 0.0, 1.0)),
                encoding: ChannelEncoding::Color,
            },
        );
        ViewImages { channels, depth: d }
    }

    impl RenderHost for Peeler {
        fn material_info(&self, _m: MaterialId) -> MaterialInfo {
            MaterialInfo::default()
        }
        fn bake_surface(&mut self, _r: &SurfaceBakeRequest) -> Result<ChannelImages> {
            Err(Error::Unsupported("bake_surface"))
        }
        fn render_views(&mut self, req: &ViewRenderRequest) -> Result<Vec<ViewImages>> {
            Ok(req.views.iter().map(|v| images(v, 1.0, false)).collect())
        }
        fn render_views_behind(
            &mut self,
            req: &ViewRenderRequest,
            behind: &[Image],
            bias: f32,
        ) -> Result<Vec<ViewImages>> {
            if !self.peel {
                return Err(Error::Unsupported("render_views_behind"));
            }
            assert_eq!(behind.len(), req.views.len());
            assert!(bias > 0.0);
            assert_eq!(
                behind[0].width, req.views[0].width,
                "peel at capture resolution"
            );
            assert_eq!(behind[0].pixels[0].x, 1.0);
            Ok(req.views.iter().map(|v| images(v, 3.0, true)).collect())
        }
        fn create_material(&mut self, _d: &BakedMaterialDesc) -> Result<MaterialId> {
            Ok(MaterialId(0))
        }
    }

    #[test]
    fn layered_impostor_stores_the_peeled_layer() {
        let scene = Scene::from_mesh(Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            vec![0, 1, 2, 0, 2, 3],
        ));
        let kind = ImpostorKind::Octahedral {
            frames_per_side: 4,
            hemisphere: false,
        };
        let imp =
            build_layered_impostor(&mut Peeler { peel: true }, &scene, &kind, 16, false).unwrap();
        assert_eq!(imp.params.layers, 2);
        let front = &imp.atlas[channels::BASE_COLOR].image;
        let back = &imp.atlas[&layer_channel(channels::BASE_COLOR, 1)].image;
        assert_eq!((front.width, front.height), (back.width, back.height));
        assert!(imp.atlas.contains_key(&layer_channel(HEIGHT, 1)));
        // Back layer: covered on the left of each frame only, and deeper (lower height).
        let (l, r) = (back.get(2, 8), back.get(13, 8));
        assert!(l.w > 0.5 && r.w < 0.5, "{l:?} {r:?}");
        let h = &imp.atlas[HEIGHT].image;
        let hb = &imp.atlas[&layer_channel(HEIGHT, 1)].image;
        assert!(hb.get(2, 8).x < h.get(2, 8).x);
        // Hosts that can't peel make layered impostors fail; single-layer ones still work.
        let err = build_layered_impostor(&mut Peeler { peel: false }, &scene, &kind, 16, false);
        assert!(matches!(err, Err(Error::Unsupported(_))));
        let one = build_impostor(&mut Peeler { peel: false }, &scene, &kind, 16, false).unwrap();
        assert_eq!(one.params.layers, 1);
        assert!(!one
            .atlas
            .contains_key(&layer_channel(channels::BASE_COLOR, 1)));
        // Half of each frame has a surface behind it: the benefit is that share × the gap.
        let b = layer_benefit(&imp.atlas).unwrap();
        assert!(b > 0.25, "{b}");

        // Half resolution: the second layer's atlases are half the size in each direction.
        let half = build_layered_impostor_with(
            &mut Peeler { peel: true },
            &scene,
            &kind,
            16,
            false,
            &LayerOptions::default(),
        )
        .unwrap();
        assert_eq!(half.params.layers, 2);
        let back = &half.atlas[&layer_channel(channels::BASE_COLOR, 1)].image;
        assert_eq!(
            (back.width, back.height),
            (front.width / 2, front.height / 2)
        );
        // Auto: below the threshold the second layer is dropped; without peeling, one layer.
        let strict = LayerOptions {
            auto_threshold: 100.0,
            ..Default::default()
        };
        let dropped = build_layered_impostor_with(
            &mut Peeler { peel: true },
            &scene,
            &kind,
            16,
            false,
            &strict,
        )
        .unwrap();
        assert_eq!(dropped.params.layers, 1);
        assert!(!dropped.atlas.contains_key(&layer_channel(HEIGHT, 1)));
        let fallback = build_layered_impostor_with(
            &mut Peeler { peel: false },
            &scene,
            &kind,
            16,
            false,
            &LayerOptions::default(),
        )
        .unwrap();
        assert_eq!(fallback.params.layers, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame_size_follows_screen_size() {
        assert_eq!(super::frame_size_for_screen(3.0), 8);
        assert_eq!(super::frame_size_for_screen(97.0), 100);
        assert_eq!(super::frame_size_for_screen(5000.0), 1024);
    }

    use super::*;
    use crate::math::fibonacci_directions;

    #[test]
    fn oct_roundtrip() {
        for d in fibonacci_directions(500, false) {
            let back = oct_decode(oct_encode(d));
            assert!((back - d).length() < 1e-4, "{d:?} -> {back:?}");
        }
    }

    #[test]
    fn hemi_oct_roundtrip() {
        for d in fibonacci_directions(300, true) {
            let back = hemi_oct_decode(hemi_oct_encode(d));
            assert!((back - d).length() < 1e-4, "{d:?} -> {back:?}");
        }
        assert!((hemi_oct_encode(Vec3::Y) - Vec2::splat(0.5)).length() < 1e-6);
    }

    #[test]
    fn octahedral_plan_covers_sphere() {
        let b = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        // Odd frame counts put a frame exactly on each pole.
        let p = plan(
            b,
            &ImpostorKind::Octahedral {
                frames_per_side: 9,
                hemisphere: false,
            },
            64,
        )
        .unwrap();
        assert_eq!(p.directions.len(), 81);
        assert!(
            p.directions.iter().any(|d| d.y > 0.99) && p.directions.iter().any(|d| d.y < -0.99)
        );
        let views = p.views();
        // Every view sees the center in the middle of its frame.
        for v in &views {
            let c = v.view_proj().project_point3(p.center);
            assert!(
                c.x.abs() < 1e-4 && c.y.abs() < 1e-4 && (0.0..=1.0).contains(&c.z),
                "{c:?}"
            );
        }
        let hemi = plan(
            b,
            &ImpostorKind::Octahedral {
                frames_per_side: 8,
                hemisphere: true,
            },
            64,
        )
        .unwrap();
        assert!(hemi.directions.iter().any(|d| d.y > 0.95));
        assert!(hemi.directions.iter().all(|d| d.y >= -1e-6));
    }

    #[test]
    fn flipbook_layout() {
        let b = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let p = plan(
            b,
            &ImpostorKind::Flipbook {
                frames: 12,
                elevation_deg: 10.0,
            },
            32,
        )
        .unwrap();
        assert_eq!((p.cols, p.rows), (4, 3));
        assert_eq!(p.atlas_size(), (128, 96));
    }

    #[test]
    fn shader_contains_matching_functions() {
        assert!(
            REFERENCE_SHADER.contains("fn oct_encode")
                && REFERENCE_SHADER.contains("fn hemi_oct_encode")
        );
    }
}
