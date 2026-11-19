//! The demo pipeline, independent of the browser so it can be tested natively.
//!
//! * [`Demo::load`]: glTF / GLB from dropped files or a zip (external buffers and textures
//!   resolved by relative path).
//! * [`Demo::run`]: a LOD chain. By default ([`TextureMode::Recut`]) the whole scene's skin
//!   is cut afresh and resampled into one atlas at the artist's resolution, and every level
//!   is a reduction of that LOD0 drawn with the same atlas. Otherwise near levels are reduced
//!   per mesh and keep the original materials, or ([`TextureMode::Shared`]) are reduced from
//!   one copy of the whole scene whose atlas holds the original materials; far levels (from
//!   `bake_from`) are geometry-only proxies of the whole scene with fresh UVs and base color,
//!   normal and roughness/metalness maps baked from the original. Every finished level and
//!   baked texture is reported through an [`Event`] as soon as it exists.
//! * The other example modes ([`DemoMode`]: option examples, cluster DAG, HLOD tree, splats,
//!   GPU crowd) live in [`crate::modes`]; [`Demo::run`] dispatches on [`LodOptions::mode`].

use crate::cook::{CookOptions, Cooked, CookedInfo, TextureCompression, bare, cook};
use crate::modes::Extra;
use farthing::bake::{BakeOptions, bake_rows_with_host, compute_tangents};
use farthing::host::{
    BakedMaterialDesc, ChannelEncoding, ChannelImage, ChannelImages, IdImage, IdRenderRequest,
    MaterialClass, MaterialInfo, RenderHost, SurfaceBakeRequest, ViewImages, ViewRenderRequest,
    channels,
};
use farthing::image::{Downsample, DownsampleOptions, Image, Texture, linear_to_srgb};
use farthing::io::{
    ExportTextures, GlbOptions, GltfMaterial, MODEL_EXTENSIONS, glb_bytes_with, load_gltf_with,
    load_model,
};
use farthing::metrics::{SurfaceDistance, surface_distance};
use farthing::reduce::{Quality, SeamMode, SimplifyOptions, Symmetry, repair, simplify};
use farthing::uv::{ParamOptions, SourceTexels, parameterize, texel_density};
use farthing::{Error, MaterialId, Mesh, Scene};
use glam::{Mat4, Vec2, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{Cursor, Read, Write};
use std::sync::Arc;

/// Surface samples per mesh when measuring a level against LOD0.
pub(crate) const MEASURE_SAMPLES: usize = 20_000;

/// Pixels per radian of the reference view the switch distances are given for: 1080 pixels
/// high, 60 degree vertical field of view.
pub(crate) fn reference_ppr() -> f32 {
    1080.0 / (2.0 * 30f32.to_radians().tan())
}

/// Distance from the scene's bounding-box center beyond which a deviation of `error` projects
/// to at most one pixel in a view with `ppr` pixels per radian, wherever on the model it is:
/// half the diagonal more than the error alone needs, since the near side of the model is that
/// much closer. Measured from the center like `HlodNode::switch_distance`, and like THREE.LOD
/// on the page. Zero when there is no error.
pub(crate) fn switch_distance(error: f32, diagonal: f32, ppr: f32) -> f32 {
    if error > 0.0 {
        error * ppr + 0.5 * diagonal
    } else {
        0.0
    }
}

/// Which example the page runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DemoMode {
    /// Classic LOD chain: reduced levels, then baked proxies.
    #[default]
    Chain,
    /// One reduction target under several named [`SimplifyOptions`] configurations.
    Presets,
    /// A cluster DAG per mesh with a view-dependent cut ([`farthing::cluster`]).
    Clusters,
    /// An HLOD tree over the scene, optionally tiled into a grid of copies
    /// ([`farthing::hlod::build_hlod_tree`]).
    Hlod,
    /// Progressive surfel cloud turned into Gaussian splats ([`farthing::impostor`]).
    Splats,
    /// A skinned model's levels for a GPU crowd: pose palette, bone LODs, vertex animation
    /// ([`farthing::anim`], [`farthing::bones`]).
    Crowd,
}

/// Largest shared atlas side: a 2048² bake holds a few hundred megabytes at its peak (the
/// decoded source textures plus about 30 bytes a texel), and 4096² would need four times
/// that, more than a browser gives WebAssembly.
pub const MAX_SHARED_SIZE: u32 = 2048;

/// Which textures the chain's near levels use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureMode {
    /// Each mesh reduced on its own, keeping the original UVs and materials.
    Original,
    /// The whole scene laid out once in one atlas (the artist's UV islands repacked, or with
    /// [`LodOptions::shared_keep_layout`] off, charts cut afresh; mip-safe 8 × 8 blocks), the
    /// original materials baked into it at the artist's texel density, and every near level
    /// reduced from that mesh.
    Shared,
    /// The whole scene's skin cut afresh (fresh charts, mip-safe blocks) and every texel of the
    /// original materials resampled into one atlas at the artist's texel density. LOD0 keeps
    /// every triangle; each further level is reduced from the one before and draws with the
    /// same atlas, its tangent-space normal map included, so far levels shade flatter than a
    /// bake for them would. Nothing is baked per level: `bake_from` and `shared_keep_layout`
    /// do not apply, and the chain has `levels` levels after LOD0.
    #[default]
    Recut,
}

/// How a chain's reduced levels are spaced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelSpacing {
    /// Each level keeps [`LodOptions::step`] of the triangles, compounded; the chain stops at
    /// the first level past [`LodOptions::max_error`].
    #[default]
    Triangles,
    /// Each level is reduced until it deviates a set amount: the last reaches
    /// [`LodOptions::max_error`], and each one before deviates [`LodOptions::error_ratio`]
    /// times less. Screen-space error shrinks with distance, so the switch distances grow by
    /// the same ratio. Triangle counts follow from the model.
    Error,
}

/// How the levels after the first atlas level (a recut's LOD0, a shared chain's LOD1, the
/// first proxy) size their textures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureShrink {
    /// Every atlas level draws with the first one's atlas; proxies keep the proxy atlas size.
    #[default]
    Keep,
    /// Each level's textures are half the side of the level before's, as a texture LOD bias of
    /// one per level would leave them. Shared and recut atlases are downsampled with the chosen
    /// filter (their charts sit in mip-safe blocks); proxies are baked at that size.
    Halve,
    /// Each level's textures keep a texel per pixel where the level first shows (its switch
    /// distance, or the closest view if that is farther), in power-of-two steps. Shared and
    /// recut atlases are downsampled with the chosen filter; proxies are baked at that size.
    Fit,
}

/// Options the page exposes. Unknown or missing fields take the defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct LodOptions {
    /// Levels after LOD0 (1–8): the most there are, with [`max_error`](Self::max_error).
    pub levels: usize,
    /// Stop at the first level that measures more than this deviation, as a fraction of the
    /// scene's bounding-box diagonal: that level and the ones after it are left out. The first
    /// level (a recut's LOD0) is always kept. 0 = always build [`levels`](Self::levels). The page
    /// uses 2%: past that, levels look plainly wrong.
    pub max_error: f32,
    /// Triangle ratio between consecutive levels.
    pub step: f32,
    /// Whether levels are spaced by triangle count or by deviation.
    pub spacing: LevelSpacing,
    /// [`LevelSpacing::Error`]: how many times as far each level may deviate as the one before.
    pub error_ratio: f32,
    pub fast: bool,
    pub strict_seams: bool,
    pub symmetry: bool,
    pub lock_border: bool,
    /// First level (1-based) that becomes a baked proxy; 0 = none. Not with [`TextureMode::Recut`].
    pub bake_from: usize,
    /// Baked texture size (square).
    pub atlas: u32,
    /// Bake samples per texel along each axis (1–4).
    pub supersample: u32,
    /// Planar charts instead of LSCM for baked levels and the shared atlas.
    pub planar_uv: bool,
    pub textures: TextureMode,
    /// Shared or recut atlas: largest side (at most [`MAX_SHARED_SIZE`]). The atlas keeps the artist's
    /// texel density when that fits, and shrinks every chart by the same factor when it does not.
    pub shared_size: u32,
    /// Closest distance the scene is ever seen from, in its own units (glTF: meters); 0 = no
    /// viewing budget. With one, nothing is kept that would not show there: the first level is
    /// reduced until it deviates about a pixel at that distance, the shared atlas is capped at
    /// a texel per pixel, and each baked atlas is sized for the distance its level appears at.
    pub view_distance: f32,
    /// Vertical field of view of that view, degrees. Switch distances are given for it.
    pub view_fov: f32,
    /// Screen height of that view, pixels.
    pub view_height: u32,
    /// Shared atlas: keep the artist's UV islands (moved, not reshaped) so the bake copies the
    /// original texels one to one, and store texels that several islands reuse (mirrored
    /// halves, repeated parts) once. Off, or when a viewing budget caps the atlas below the
    /// artist's resolution: unwrap the scene afresh.
    pub shared_keep_layout: bool,
    /// Recut: the deviation LOD0 may have, as a fraction of the scene's bounding-box diagonal;
    /// 0 = every triangle. LOD0 is reduced until it measures about that, and the atlas is
    /// capped at a texel per that distance, since texels finer than the geometry keeps are
    /// wasted. With a viewing budget too, the larger of the two deviations applies.
    pub lod0_error: f32,
    /// How later levels' textures shrink. Not with [`TextureMode::Original`], whose reduced
    /// levels draw with the original materials.
    pub texture_shrink: TextureShrink,
    /// The downsample filter: `nearest`, `box`, `triangle`, `kaiser`, `lanczos` or `blur1`–`blur5`.
    pub texture_filter: String,
    /// Sharpening after a downsample, 0–10.
    pub texture_sharpen: u8,
    /// Cook the atlas and proxy textures with mipple for a platform: GPU block formats with
    /// mips, in KTX2 files with Zstd. The viewer then draws the decoded blocks.
    pub texture_compression: TextureCompression,
    /// mipple's effort, 0–9.
    pub compression_effort: u8,
    /// mipple's rate-distortion weight (smaller files after Zstd at some quality); 0 = off.
    pub compression_lambda: f32,

    pub mode: DemoMode,
    /// Option examples: triangle fraction every configuration reduces to.
    pub preset_ratio: f32,
    /// Cluster DAG: [`ClusterOptions::max_triangles`](farthing::cluster::ClusterOptions::max_triangles).
    pub cluster_max_triangles: u32,
    /// Cluster DAG: [`ClusterOptions::max_vertices`](farthing::cluster::ClusterOptions::max_vertices).
    pub cluster_max_vertices: u32,
    /// Cluster DAG: [`ClusterOptions::fast`](farthing::cluster::ClusterOptions::fast).
    pub cluster_fast: bool,
    /// HLOD: copies of the scene along x and z (1 = the scene itself).
    pub hlod_copies: u32,
    /// HLOD: leaf cells across one copy (1 or 2).
    pub hlod_split: u32,
    /// HLOD: triangle budget of a leaf proxy.
    pub hlod_triangles: usize,
    /// HLOD: unwrap and bake every proxy (256 x 256 atlas) instead of geometry only.
    pub hlod_bake: bool,
    /// Splats: surfels in the full cloud.
    pub splat_count: usize,
    /// Splats: shape each Gaussian from its surface patch (footprints) instead of a round disc.
    pub splat_footprints: bool,
    /// HLOD: render visibility (triangle IDs) on the GPU when WebGL2 is available.
    pub gpu_visibility: bool,
    /// Crowd: bones the middle level keeps.
    pub crowd_bones: usize,
    /// Crowd: fraction of the triangles the middle level keeps (the far level a quarter of it).
    pub crowd_ratio: f32,
}

impl Default for LodOptions {
    fn default() -> Self {
        LodOptions {
            levels: 4,
            max_error: 0.0,
            step: 0.5,
            spacing: LevelSpacing::Triangles,
            error_ratio: 2.0,
            fast: false,
            strict_seams: false,
            symmetry: false,
            lock_border: false,
            bake_from: 4,
            atlas: 512,
            supersample: 2,
            planar_uv: false,
            textures: TextureMode::Recut,
            shared_size: 2048,
            view_distance: 0.0,
            view_fov: 60.0,
            view_height: 1080,
            shared_keep_layout: true,
            lod0_error: 0.0,
            texture_shrink: TextureShrink::Keep,
            texture_filter: "box".into(),
            texture_sharpen: 0,
            texture_compression: TextureCompression::Off,
            compression_effort: mipple::format::DEFAULT_EFFORT,
            compression_lambda: 0.0,
            mode: DemoMode::Chain,
            preset_ratio: 0.25,
            cluster_max_triangles: 128,
            cluster_max_vertices: 255,
            cluster_fast: false,
            hlod_copies: 3,
            hlod_split: 1,
            hlod_triangles: 2000,
            hlod_bake: true,
            splat_count: 20_000,
            splat_footprints: true,
            gpu_visibility: true,
            crowd_bones: 12,
            crowd_ratio: 0.4,
        }
    }
}

impl LodOptions {
    /// Whether level `level`, measuring `error` (world units), ends the chain: it is past
    /// [`max_error`](Self::max_error) of `diagonal`, and not the first level.
    /// Error spacing sets every level's deviation instead, so nothing is past it.
    pub(crate) fn past_max_error(&self, level: usize, error: f32, diagonal: f32) -> bool {
        self.spacing == LevelSpacing::Triangles
            && level > 1
            && self.max_error > 0.0
            && error > self.max_error * diagonal
    }

    /// [`LevelSpacing::Error`]: the deviations (world units, ascending) of the reduced levels
    /// after one that deviates `floor`: up to `n` of them, the last at
    /// [`max_error`](Self::max_error) of `diagonal` (2% when that is 0), each one before
    /// [`error_ratio`](Self::error_ratio) times less. Fewer when the floor leaves no room: the
    /// first must deviate at least the ratio's square root times the floor, so it is a level
    /// apart. `None` with triangle spacing.
    pub(crate) fn error_targets(&self, n: usize, floor: f32, diagonal: f32) -> Option<Vec<f32>> {
        if self.spacing != LevelSpacing::Error {
            return None;
        }
        let ratio = self.error_ratio.clamp(1.25, 8.0);
        let max = if self.max_error > 0.0 {
            self.max_error
        } else {
            0.02
        } * diagonal;
        let mut targets: Vec<f32> = (0..n as i32).rev().map(|k| max / ratio.powi(k)).collect();
        targets.retain(|&t| t >= floor * ratio.sqrt());
        Some(targets)
    }

    /// Pixels per radian of the view: `view_height` pixels over `view_fov` degrees (the
    /// defaults give [`reference_ppr`]).
    pub(crate) fn ppr(&self) -> f32 {
        let half = self.view_fov.clamp(5.0, 150.0).to_radians() / 2.0;
        self.view_height.clamp(120, 8640) as f32 / (2.0 * half.tan())
    }

    /// With a viewing budget: the deviation (world units) that spans one pixel at the closest
    /// distance. Its inverse is the most texels per unit that view can show.
    pub(crate) fn visible_error(&self) -> Option<f32> {
        (self.view_distance > 0.0 && self.view_distance.is_finite())
            .then(|| self.view_distance / self.ppr())
    }

    /// The deviation (world units) the first level may have, and its inverse caps the atlas
    /// density: the viewing budget's [`visible_error`](Self::visible_error), or for a `recut`
    /// [`lod0_error`](Self::lod0_error) of `diagonal`, whichever is larger.
    pub(crate) fn allowed_error(&self, recut: bool, diagonal: f32) -> Option<f32> {
        let own = (recut && self.lod0_error > 0.0 && self.lod0_error.is_finite())
            .then(|| self.lod0_error.min(0.5) * diagonal);
        match (self.visible_error(), own) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }

    /// How textures are downsampled (for color and data; normal maps set `normal_map`).
    pub(crate) fn downsample_options(&self) -> DownsampleOptions {
        let f = self.texture_filter.as_str();
        let filter = match f {
            "nearest" => Downsample::Nearest,
            "triangle" => Downsample::Triangle,
            "kaiser" => Downsample::Kaiser,
            "lanczos" => Downsample::Lanczos3,
            _ if f.starts_with("blur") => Downsample::Blur(f[4..].parse().unwrap_or(3)),
            _ => Downsample::Box,
        };
        DownsampleOptions {
            filter,
            sharpen: self.texture_sharpen.min(10),
            ..Default::default()
        }
    }

    /// How many times level `level`'s textures are halved from the first atlas level's
    /// (`first`), for `error` (the level's measured deviation, world units) and `density`
    /// (texels per world unit the first atlas level has). Never fewer than `prev`, and never
    /// below 4 texels on the shorter side `side`.
    pub(crate) fn halvings(
        &self,
        level: usize,
        first: usize,
        error: f32,
        density: f32,
        prev: u32,
        side: u32,
    ) -> u32 {
        let k = match self.texture_shrink {
            TextureShrink::Keep => 0,
            TextureShrink::Halve => level.saturating_sub(first) as u32,
            TextureShrink::Fit => {
                // A texel per pixel at the switch distance is a texel per `error`.
                let at = error.max(self.visible_error().unwrap_or(0.0));
                if level <= first || at <= 0.0 || density <= 0.0 || !(density * at).is_finite() {
                    0
                } else {
                    (density * at).log2().floor().max(0.0) as u32
                }
            }
        };
        let most = (side.max(4) / 4).ilog2();
        k.max(prev).min(most)
    }

    pub(crate) fn cook_options(&self) -> CookOptions {
        CookOptions {
            target: self.texture_compression,
            effort: self.compression_effort,
            lambda: self.compression_lambda,
        }
    }

    fn simplify_options(&self) -> SimplifyOptions {
        SimplifyOptions {
            seams: if self.strict_seams {
                SeamMode::Strict
            } else {
                SeamMode::Permissive
            },
            symmetry: if self.symmetry {
                Symmetry::Auto
            } else {
                Symmetry::Off
            },
            quality: if self.fast {
                Quality::Fast
            } else {
                Quality::Best
            },
            lock_border: self.lock_border,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SceneInfo {
    pub name: String,
    pub triangles: usize,
    pub vertices: usize,
    pub meshes: usize,
    pub instances: usize,
    pub materials: Vec<String>,
    pub textured_materials: usize,
    pub diagonal: f32,
    pub skinned: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelKind {
    Original,
    Reduced,
    /// Reduced from the re-unwrapped scene (shared or recut); textures in its atlas.
    Shared,
    Baked,
    Splats,
}

#[derive(Clone, Debug, Serialize)]
pub struct LevelInfo {
    pub level: usize,
    pub kind: LevelKind,
    pub triangles: usize,
    pub vertices: usize,
    /// Measured deviation from LOD0: two-way Hausdorff distance over area-weighted samples and
    /// every vertex of both, world units.
    pub error: f32,
    /// `error` relative to the scene's bounding-box diagonal.
    pub relative_error: f32,
    /// The simplifier's own estimate, summed over the chain: an upper bound, often loose.
    pub error_bound: f32,
    /// Camera distance from the bounding-box center beyond which `error` projects to at most
    /// one pixel in the view `ppr` describes; see [`switch_distance`].
    pub switch_distance_1px: f32,
    /// Pixels per radian of the view `switch_distance_1px` is given for.
    pub ppr: f32,
    /// Measured levels: mean distance to LOD0 over the area-weighted samples, both ways.
    pub mean_error: Option<f32>,
    /// Measured levels: root mean square of the same distances.
    pub rms_error: Option<f32>,
    /// Measured levels: mean angle in degrees between the level's shading normals and LOD0's
    /// at matching points. `None` for baked levels, whose normal maps make up for it.
    pub normal_error: Option<f32>,
    /// Time to build the level (measuring it not included).
    pub ms: f64,
    /// Baked levels: fraction of texels that found the source surface.
    pub coverage: Option<f32>,
    pub charts: Option<usize>,
    pub utilization: Option<f32>,
    /// Option examples: the configuration's name and what it changes.
    pub label: Option<String>,
    pub note: Option<String>,
    /// Size of the level's download (GLB, or PLY for splats).
    pub bytes: Option<usize>,
    /// GPU memory of the textures the level is drawn with: RGBA8 with full mip chains, each
    /// texture counted once.
    pub texture_bytes: Option<usize>,
    /// Shared and baked levels: atlas size.
    pub atlas_width: Option<u32>,
    pub atlas_height: Option<u32>,
    /// Shared levels: the fraction of the artist's texel density the atlas keeps.
    pub density_scale: Option<f32>,
    /// Shared levels: the level whose GLB carries the atlas (the others reuse its material).
    pub shared_textures: Option<usize>,
    /// Recut LOD0: how far its base color is from the original materials at matching points of
    /// the same triangles, in 8-bit sRGB levels (largest channel): mean and 99th percentile.
    pub texture_error: Option<f32>,
    pub texture_error_p99: Option<f32>,
    /// With a viewing budget: the most texels per world unit the closest view can show (the
    /// shared atlas is capped there), and the fraction of the artist's texels under that cap.
    /// Recut levels with a LOD0 error: a texel per that deviation instead, when it is larger.
    pub view_texel_density: Option<f32>,
    pub view_texels: Option<f32>,
    /// Atlas and proxy levels whose textures were shrunk: how many times each side was halved
    /// from the first atlas level's (or from the bake), with which filter.
    pub texture_halvings: Option<u32>,
    pub texture_filter: Option<String>,
    /// With texture compression: the cooked textures the level draws with (a shared level's
    /// are its carrier's) and what the level ships as.
    pub cooked: Option<CookedInfo>,
}

impl LevelInfo {
    pub(crate) fn new(level: usize, kind: LevelKind) -> LevelInfo {
        LevelInfo {
            level,
            kind,
            triangles: 0,
            vertices: 0,
            error: 0.0,
            relative_error: 0.0,
            error_bound: 0.0,
            switch_distance_1px: 0.0,
            ppr: reference_ppr(),
            mean_error: None,
            rms_error: None,
            normal_error: None,
            ms: 0.0,
            coverage: None,
            charts: None,
            utilization: None,
            label: None,
            note: None,
            bytes: None,
            texture_bytes: None,
            atlas_width: None,
            atlas_height: None,
            density_scale: None,
            shared_textures: None,
            texture_error: None,
            texture_error_p99: None,
            view_texel_density: None,
            view_texels: None,
            texture_halvings: None,
            texture_filter: None,
            cooked: None,
        }
    }

    /// Measures the level `scene` against LOD0 (`source`, flattened): its error, mean and RMS
    /// distance, normal deviation and switch distance. `diagonal` is LOD0's bounding-box
    /// diagonal.
    pub(crate) fn measure(&mut self, source: &Mesh, scene: &Scene, diagonal: f32) {
        let d = surface_distance(source, &scene.flatten().0, MEASURE_SAMPLES);
        self.set_distance(&d, diagonal);
    }

    /// Records `d`, the level measured against LOD0, with the switch distance for `ppr`.
    pub(crate) fn set_distance(&mut self, d: &SurfaceDistance, diagonal: f32) {
        self.error = d.hausdorff;
        self.relative_error = d.hausdorff / diagonal;
        self.switch_distance_1px = switch_distance(d.hausdorff, diagonal, self.ppr);
        self.mean_error = Some(d.mean);
        self.rms_error = Some(d.rms);
        self.normal_error =
            (!matches!(self.kind, LevelKind::Baked)).then_some(d.normal_mean_degrees);
    }
}

/// A typed array for the page.
#[derive(Clone, Debug)]
pub enum Buffer {
    F32(Vec<f32>),
    U32(Vec<u32>),
    U8(Vec<u8>),
}

pub enum Event<'a> {
    /// What the pipeline is doing now, for a status line.
    Stage(&'a str),
    /// A level is done, with what the viewer draws (see [`scene_payload`]): `None` for level 0
    /// (the page already has the original). Reduced levels carry no textures; their materials
    /// are the original's, by slot.
    Level {
        info: &'a LevelInfo,
        scene: Option<(&'a Scene, &'a ExportTextures)>,
    },
    /// A baked texture (sRGB base color, tangent-space normal as 0..1, or glTF ORM data),
    /// RGBA8 rows.
    Texture {
        level: usize,
        channel: &'a str,
        width: u32,
        height: u32,
        rgba: &'a [u8],
    },
    /// Mode-specific results (see [`crate::modes`]): a JSON description and typed arrays.
    Data {
        kind: &'a str,
        json: String,
        buffers: Vec<Buffer>,
    },
}

pub struct Demo {
    pub name: String,
    pub scene: Scene,
    /// Decoded textures are shared 8-bit [`Texture`]s: every level that uses the original
    /// materials holds references, not copies.
    pub materials: Vec<GltfMaterial>,
    /// Results of the last run, by level (0 = original), with the textures to export.
    pub(crate) levels: Vec<(Scene, ExportTextures)>,
    /// With texture compression: each level's cooked textures, by level (shared levels hold
    /// their carrier's).
    pub(crate) cooked: BTreeMap<usize, Arc<Cooked>>,
    /// What the last cluster, HLOD or splat run keeps for selection and downloads.
    pub(crate) extra: Extra,
}

impl Demo {
    /// Loads the first model among `files` (shallowest path, then by format in the order of
    /// [`MODEL_EXTENSIONS`]: GLB, glTF, FBX, ...); `.zip` files are expanded first. Other files
    /// are what the model refers to: glTF buffers and images, MTL libraries, textures.
    pub fn load(files: Vec<(String, Vec<u8>)>) -> Result<Demo, String> {
        let files = expand_zips(files)?;
        let lower = |s: &str| s.to_ascii_lowercase();
        let format = |name: &str| {
            let ext = lower(name.rsplit_once('.').map_or("", |(_, e)| e));
            MODEL_EXTENSIONS.iter().position(|&e| e == ext)
        };
        let mut mains: Vec<(usize, usize)> = (0..files.len())
            .filter_map(|i| Some((i, format(&files[i].0)?)))
            .collect();
        mains.sort_by_key(|&(i, rank)| (files[i].0.matches('/').count(), rank, lower(&files[i].0)));
        let Some(&(main, rank)) = mains.first() else {
            let names: Vec<&str> = files.iter().map(|f| f.0.as_str()).take(12).collect();
            return Err(format!(
                "no model found: reads .{} (got: {})",
                MODEL_EXTENSIONS.join(", ."),
                if names.is_empty() {
                    "nothing".into()
                } else {
                    names.join(", ")
                }
            ));
        };
        let main_name = files[main].0.clone();
        // Other formats go through polysmelt, which finds the files they refer to itself.
        if MODEL_EXTENSIONS[rank] != "glb" && MODEL_EXTENSIONS[rank] != "gltf" {
            let (scene, materials) =
                load_model(&main_name, files).map_err(|e| format!("{main_name}: {e}"))?;
            return Demo::from_scene(&main_name, scene, materials);
        }
        let dir = main_name
            .rfind('/')
            .map_or("", |i| &main_name[..=i])
            .to_owned();
        let by_path: HashMap<String, usize> = files
            .iter()
            .enumerate()
            .map(|(i, f)| (lower(&normalize(&f.0)), i))
            .collect();
        let resolve = |uri: &str| -> Option<Vec<u8>> {
            let key = lower(&normalize(&format!("{dir}{uri}")));
            if let Some(&i) = by_path.get(&key) {
                return Some(files[i].1.clone());
            }
            // Fall back to a unique file with the same base name (flattened uploads).
            let base = lower(key.rsplit('/').next().unwrap_or(&key));
            let mut hits = files
                .iter()
                .filter(|f| lower(f.0.rsplit('/').next().unwrap_or(&f.0)) == base);
            match (hits.next(), hits.next()) {
                (Some(f), None) => Some(f.1.clone()),
                _ => None,
            }
        };
        let (scene, materials) =
            load_gltf_with(&files[main].1, resolve).map_err(|e| format!("{main_name}: {e}"))?;
        Demo::from_scene(&main_name, scene, materials)
    }

    /// The demo of a model loaded from `path`.
    fn from_scene(path: &str, scene: Scene, materials: Vec<GltfMaterial>) -> Result<Demo, String> {
        if scene.triangle_count() == 0 {
            return Err(format!("{path} has no triangles"));
        }
        let name = path.rsplit('/').next().unwrap_or(path).to_owned();
        Ok(Demo {
            name,
            levels: vec![(scene.clone(), original_textures(&materials))],
            scene,
            materials,
            cooked: BTreeMap::new(),
            extra: Extra::None,
        })
    }

    pub fn info(&self) -> SceneInfo {
        let s = &self.scene;
        SceneInfo {
            name: self.name.clone(),
            triangles: s.triangle_count(),
            vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
            meshes: s.meshes.len(),
            instances: s.instances.len(),
            materials: s.materials.iter().map(|m| m.name.clone()).collect(),
            textured_materials: self
                .materials
                .iter()
                .filter(|m| m.base_color_texture.is_some())
                .count(),
            diagonal: s.bounds().diagonal(),
            skinned: !s.skeletons.is_empty(),
        }
    }

    /// The original as GLB with base color, metallic-roughness and normal textures and factors
    /// (what farthing reads of the materials), so every level is shown with the same material
    /// model.
    pub fn original_glb(&self) -> Result<Vec<u8>, String> {
        self.level_glb(0, true)
    }

    /// A level of the last run (0 = the original) as the viewer takes it: see
    /// [`scene_payload`].
    pub fn level_scene(&self, level: usize) -> Result<(serde_json::Value, Vec<Buffer>), String> {
        let (s, t) = self
            .levels
            .get(level)
            .ok_or_else(|| format!("no level {level}; run first"))?;
        Ok(scene_payload(s, t))
    }

    /// A level of the last run as a self-contained GLB: compact (quantized and
    /// meshopt-compressed) for the viewer, or plain glTF, which every tool opens, for download.
    pub fn level_glb(&self, level: usize, compact: bool) -> Result<Vec<u8>, String> {
        let (s, t) = self
            .levels
            .get(level)
            .ok_or_else(|| format!("no level {level}; run first"))?;
        let opts = if compact {
            GlbOptions::compact()
        } else {
            GlbOptions::default()
        };
        glb_bytes_with(s, t, &opts).map_err(|e| e.to_string())
    }

    /// A level of the last run as it would ship with texture compression: a zip of its compact
    /// GLB without textures (quantized, `EXT_meshopt_compression`), the KTX2 files mipple made
    /// (`textures/*.ktx2`, Zstd-supercompressed) and a `manifest.json` that binds them.
    pub fn level_cooked(&self, level: usize) -> Result<Vec<u8>, String> {
        let (s, t) = self
            .levels
            .get(level)
            .ok_or_else(|| format!("no level {level}; run first"))?;
        let c = self
            .cooked
            .get(&level)
            .ok_or_else(|| format!("level {level} has no cooked textures"))?;
        let glb = compact_glb(s, &bare(t)).map_err(|e| e.to_string())?;
        let manifest = serde_json::json!({
            "model": "model.glb",
            "geometry": "glTF 2.0 with KHR_mesh_quantization and EXT_meshopt_compression; material 0 \
                         keeps its factors, and its textures are the KTX2 files below",
            "textures": c.info.textures.iter().zip(&c.files).map(|(t, (file, _))| serde_json::json!({
                "file": file,
                "slot": match t.channel {
                    "base_color" => "pbrMetallicRoughness.baseColorTexture",
                    "normal" => "normalTexture",
                    _ => "pbrMetallicRoughness.metallicRoughnessTexture (G roughness, B metalness)",
                },
                "format": t.format,
                "width": t.width,
                "height": t.height,
                "mips": t.mips,
                "gpu_bytes": t.gpu_bytes,
            })).collect::<Vec<_>>(),
            "texture_target": c.info.target,
            "supercompression": format!("Zstd level {}", c.info.zstd_level),
            "cooked_by": "mipple, https://github.com/Azazel-Labs/mipple",
            "reduced_by": "farthing, https://github.com/Azazel-Labs/farthing",
        });
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        // Every entry is compressed already (meshopt, Zstd), so they are stored as they are.
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let manifest = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
        let entries = [("model.glb", &glb), ("manifest.json", &manifest)]
            .into_iter()
            .chain(c.files.iter().map(|(n, b)| (n.as_str(), b)));
        for (name, bytes) in entries {
            zip.start_file(name, stored).map_err(|e| e.to_string())?;
            zip.write_all(bytes).map_err(|e| e.to_string())?;
        }
        let out = zip.finish().map_err(|e| e.to_string())?;
        Ok(out.into_inner())
    }

    /// Runs `opts.mode`, reporting each result as it finishes. `now` returns milliseconds.
    /// Starting a run drops the previous run's results, whatever its mode.
    pub fn run(
        &mut self,
        opts: &LodOptions,
        now: &dyn Fn() -> f64,
        emit: &mut dyn FnMut(Event),
    ) -> Result<(), String> {
        self.levels.truncate(1);
        self.cooked.clear();
        self.extra = Extra::None;
        match opts.mode {
            DemoMode::Chain => self.run_chain(opts, now, emit),
            DemoMode::Presets => crate::modes::run_presets(self, opts, now, emit),
            DemoMode::Clusters => crate::modes::run_clusters(self, opts, now, emit),
            DemoMode::Hlod => crate::modes::run_hlod(self, opts, now, emit),
            DemoMode::Splats => crate::modes::run_splats(self, opts, now, emit),
            DemoMode::Crowd => crate::modes::run_crowd(self, opts, now, emit),
        }
    }

    /// The classic LOD chain.
    fn run_chain(
        &mut self,
        opts: &LodOptions,
        now: &dyn Fn() -> f64,
        emit: &mut dyn FnMut(Event),
    ) -> Result<(), String> {
        let err = |e: Error| e.to_string();
        let levels = opts.levels.clamp(1, 8);
        let step = opts.step.clamp(0.05, 0.95);
        let bake_from = match opts.bake_from {
            0 => levels + 1,
            b => b.clamp(1, levels),
        };
        let base = opts.simplify_options();
        let diag = self.scene.bounds().diagonal().max(f32::MIN_POSITIVE);
        let input_tris = self.scene.triangle_count();
        let ppr = opts.ppr();
        let visible = opts.visible_error();
        let source = self.scene.flatten().0;
        let measure = |s: &Scene| surface_distance(&source, &s.flatten().0, MEASURE_SAMPLES);
        let level_info = |level, kind, s: &Scene, d: Option<&SurfaceDistance>, bound: f32, ms| {
            let mut info = LevelInfo {
                triangles: s.triangle_count(),
                vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
                error_bound: bound,
                ppr,
                ms,
                ..LevelInfo::new(level, kind)
            };
            if let Some(d) = d {
                info.set_distance(d, diag);
            }
            info.note = visible.map(|v| view_note(level, info.error, v, opts.view_distance));
            info
        };

        let original_bytes = texture_bytes(&original_textures(&self.materials));
        let mut info = level_info(0, LevelKind::Original, &self.scene, None, 0.0, 0.0);
        info.texture_bytes = Some(original_bytes);
        // A recut's levels are named from LOD0, the recut itself.
        if opts.textures == TextureMode::Recut {
            info.label = Some("Original".into());
        }
        if let Some(v) = visible {
            let dens = self.artist_density(&source);
            info.view_texel_density = Some(1.0 / v);
            info.view_texels = texels_under(&source, &dens, 1.0 / v);
        }
        emit(Event::Level {
            info: &info,
            scene: None,
        });
        if opts.textures == TextureMode::Recut {
            self.shared_levels(opts, true, levels + 2, now, emit)?;
            emit(Event::Stage("done"));
            return Ok(());
        }

        // Near levels: each mesh simplified from its previous level (as `lod_chain` does),
        // keeping UVs and material slots.
        let mut prev: Vec<(Mesh, f32)> =
            self.scene.meshes.iter().map(|m| (m.clone(), 0.0)).collect();
        let mesh_scale = crate::modes::mesh_scales(&self.scene);
        // Triangle counts the steps compound from: the original's, or with a viewing budget,
        // the first level's (which may already be well past one step).
        let mut from: Vec<usize> = self.scene.meshes.iter().map(Mesh::triangle_count).collect();
        let mut from_level = 0;
        let shared = opts.textures == TextureMode::Shared;
        if shared && bake_from > 1 && self.shared_levels(opts, false, bake_from, now, emit)? {
            emit(Event::Stage("done"));
            return Ok(());
        }
        // Error spacing: every reduced level's deviation, from the original's.
        let targets = opts.error_targets(levels, 0.0, diag);
        let goal = |level: usize| targets.as_ref().map(|t| t.get(level - 1).copied());
        let mut prev_error = 0.0;
        let first_reduced = if shared { bake_from } else { 1 };
        for level in first_reduced..bake_from {
            let goal = match goal(level) {
                Some(None) => break, // no room for another level apart
                Some(g) => g,
                None => None,
            };
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let t0 = now();
            let ratio = step.powi((level - from_level) as i32);
            let limit = goal.or(visible.filter(|_| level == 1));
            let ((next, s, d), _) = fit_visible(limit, goal.is_some(), |k| {
                let mut next = Vec::with_capacity(prev.len());
                let mut extended = false;
                for (mi, (mesh, error)) in prev.iter().enumerate() {
                    let target = match goal {
                        Some(_) => mesh.triangle_count(),
                        None => ((from[mi] as f32 * ratio) as usize).max(1),
                    };
                    // In mesh units: the largest instance scale turns them into world units.
                    let limit = limit
                        .filter(|_| mesh_scale[mi] > 0.0)
                        .map(|v| remaining(k * v, prev_error) / mesh_scale[mi]);
                    let (m, e, x) = reduce_to(mesh, &base, target, limit).map_err(err)?;
                    extended |= x;
                    next.push((m, error + e));
                }
                let mut s = self.scene.clone();
                for (mi, (m, _)) in next.iter().enumerate() {
                    s.meshes[mi] = m.clone();
                }
                let d = measure(&s);
                Ok(((next, s, d), d.hausdorff, extended))
            })?;
            if opts.past_max_error(level, d.hausdorff, diag) {
                emit(Event::Stage("done"));
                return Ok(());
            }
            prev = next;
            prev_error = d.hausdorff;
            if limit.is_some() {
                from = prev.iter().map(|(m, _)| m.triangle_count()).collect();
                from_level = 1;
            }
            // Mesh errors are in mesh units; the bound is the largest in world units.
            let bound = prev
                .iter()
                .enumerate()
                .map(|(mi, p)| p.1 * mesh_scale[mi])
                .fold(0.0, f32::max);
            let none = ExportTextures::default();
            let glb = compact_glb(&s, &none).map_err(err)?;
            let ms = now() - t0;
            let mut info = level_info(level, LevelKind::Reduced, &s, Some(&d), bound, ms);
            info.bytes = Some(glb.len());
            info.texture_bytes = Some(original_bytes);
            emit(Event::Level {
                info: &info,
                scene: Some((&s, &none)),
            });
            self.levels.push((s, original_textures(&self.materials)));
        }
        if bake_from > levels {
            emit(Event::Stage("done"));
            return Ok(());
        }
        // The far levels compound from the first level when the budget made it.
        let far_from = match self.levels.get(1) {
            Some((s, _)) if visible.is_some() => (s.triangle_count(), 1),
            _ => (input_tris, 0),
        };

        // Far levels: one world-space mesh without seams, re-parameterized and baked.
        emit(Event::Stage("Proxy: merging the scene"));
        let mut geo = {
            let mut m = repair::geometry_only(&self.scene.flatten().0);
            m.skin = None;
            m.morph_targets.clear();
            m.material_ids.iter_mut().for_each(|s| *s = 0);
            m
        };
        let mut geo_error = 0.0;
        let mut host = MaterialHost::new(&self.materials, false);
        let far_base = SimplifyOptions {
            seams: SeamMode::Permissive,
            ..base.clone()
        };
        // How far `geo` deviates: it starts from the original, not the near levels.
        let mut geo_measured = 0.0;
        for level in bake_from..=levels {
            let goal = match goal(level) {
                Some(None) => break, // no room for another level apart
                Some(g) => g,
                None => None,
            };
            let t0 = now();
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let ratio = step.powi((level - far_from.1) as i32);
            let target = match goal {
                Some(_) => geo.triangle_count(),
                None => ((far_from.0 as f32 * ratio) as usize).max(4),
            };
            let limit = goal.or(visible.filter(|_| level == 1));
            // Measured before the unwrap (which only splits vertices), so a budget can size the
            // atlas for the distance the level appears at.
            let ((m, e, mesh, d), error) = fit_visible(limit, goal.is_some(), |k| {
                let l = limit.map(|v| remaining(k * v, geo_measured));
                let (m, e, x) = reduce_to(&geo, &far_base, target, l).map_err(err)?;
                let mut mesh = m.clone();
                mesh.compute_normals(60.0);
                let d = measure(&Scene::from_mesh(mesh.clone()));
                Ok(((m, e, mesh, d), d.hausdorff, x))
            })?;
            // Checked before the unwrap and bake: a level left out costs only its reduction.
            if opts.past_max_error(level, error, diag) {
                break;
            }
            geo = m;
            geo_error += e;
            geo_measured = error;

            emit(Event::Stage(&format!("LOD{level}: unwrapping UVs")));
            // With a budget: a texel per pixel where the level first shows, at its switch
            // distance or the closest view, whichever is farther (but at least about 64 × 64).
            // Fitting textures does the same without a budget: a texel per pixel at the switch
            // distance.
            let floor = 64.0 * (0.5 / mesh.surface_area().max(f32::MIN_POSITIVE)).sqrt();
            let fit = opts.texture_shrink == TextureShrink::Fit && error > 0.0;
            let density = match visible {
                Some(v) => Some((1.0 / error.max(v)).max(floor)),
                None if fit => Some((1.0 / error).max(floor)),
                None => None,
            };
            // Halving bakes each later proxy at half the side of the one before: its charts are
            // padded for that size, where shrinking a finished bake would run them together.
            let halved = match opts.texture_shrink {
                TextureShrink::Halve => (level - bake_from) as u32,
                _ => 0,
            };
            let popts = ParamOptions {
                atlas_size: (opts.atlas >> halved).max(64),
                padding: 2,
                texel_density: density.map(|d| vec![d; mesh.triangle_count()]),
                ..if opts.planar_uv {
                    ParamOptions::planar()
                } else {
                    ParamOptions::default()
                }
            };
            let p = parameterize(&mesh, &popts).map_err(err)?;
            let (w, h) = (p.width, p.height);
            let mut mesh = p.mesh;
            compute_tangents(&mut mesh, 0).map_err(err)?;

            emit(Event::Stage(&format!("LOD{level}: baking {w}×{h}")));
            let (tex, coverage) = bake_textures(
                &mut host,
                &mesh,
                &self.scene,
                &BakeOptions {
                    width: w,
                    height: h,
                    supersample: opts.supersample.clamp(1, 4),
                    gutter: 4,
                    channels: vec![
                        channels::BASE_COLOR.into(),
                        channels::NORMAL.into(),
                        channels::ORM.into(),
                    ],
                    ..Default::default()
                },
                level,
                emit,
            )
            .map_err(err)?;
            let mut s = Scene::from_mesh(mesh);
            s.materials[0].name = "baked".into();
            let cooked = cook_level(level, &s, &tex, opts, now, emit)?;
            let shown = cooked.as_ref().map_or(&tex, |c| &c.1);
            let glb = compact_glb(&s, shown).map_err(err)?;
            let ms = now() - t0;
            let mut info = level_info(level, LevelKind::Baked, &s, Some(&d), geo_error, ms);
            info.coverage = Some(coverage);
            info.charts = Some(p.charts);
            info.utilization = Some(p.utilization);
            info.bytes = Some(glb.len());
            info.texture_bytes = Some(texture_bytes(&tex));
            info.atlas_width = Some(w);
            info.atlas_height = Some(h);
            info.texture_halvings = (halved > 0).then_some(halved);
            if let Some((c, _)) = &cooked {
                info.cooked = Some(c.info.clone());
                self.cooked.insert(level, c.clone());
            }
            emit(Event::Level {
                info: &info,
                scene: Some((&s, shown)),
            });
            self.levels.push((s, tex));
        }
        emit(Event::Stage("done"));
        Ok(())
    }

    /// Shared-atlas levels (1 until `end`): the whole scene as one mesh, its UV islands
    /// repacked (or re-unwrapped) at the artist's texel density (up to `opts.shared_size` per
    /// side) and baked by exact transfer, then reduced level by level. Level 1's GLB carries the
    /// atlas. A `recut` always unwraps afresh, and its level 1 is LOD0: every triangle (unless a
    /// viewing budget drops some), with the texture error of the resampling. Returns whether a
    /// level went past [`LodOptions::max_error`], which ends the chain there.
    fn shared_levels(
        &mut self,
        opts: &LodOptions,
        recut: bool,
        end: usize,
        now: &dyn Fn() -> f64,
        emit: &mut dyn FnMut(Event),
    ) -> Result<bool, String> {
        let err = |e: Error| e.to_string();
        let mut t0 = now();
        let what = if recut { "Recut" } else { "Shared atlas" };
        let name = |level: usize| format!("LOD{}", level - usize::from(recut));
        emit(Event::Stage(&format!("{what}: merging the scene")));
        let (flat, refs) = self.scene.flatten();
        let mut dens = self.artist_density(&flat);
        let diag = self.scene.bounds().diagonal().max(f32::MIN_POSITIVE);
        // The deviation level 1 may have: a viewing budget's pixel at the closest view, or a
        // recut's LOD0 error. Texels finer than it are wasted, so it caps the density too.
        let allowed = opts.allowed_error(recut, diag);
        let cap = allowed.map(|v| 1.0 / v);
        let view_texels = cap.and_then(|c| texels_under(&flat, &dens, c));
        if let Some(c) = cap {
            dens.iter_mut().for_each(|d| *d = d.min(c));
        }
        let textured = dens.iter().any(|&d| d > 0.0);
        // Materials that read textures through two UV sets: their islands may overlap in one
        // set and not in the other.
        let one_set = self.materials.iter().all(|m| {
            let mut sets = [
                m.base_color_texture.as_ref().map(|t| t.1),
                m.metallic_roughness_texture.as_ref().map(|t| t.1),
                m.normal_texture.as_ref().map(|t| t.1),
            ]
            .into_iter()
            .flatten();
            let first = sets.next();
            sets.all(|s| Some(s) == first)
        });
        // Kept islands hold the artist's texels one to one, so a cap that binds unwraps afresh.
        let capped = view_texels.is_some_and(|t| t < 0.999);
        let keep = (!recut && opts.shared_keep_layout && textured && !capped)
            .then(|| source_texels(&flat, &self.largest_textures(), one_set));
        let source = flat.clone();
        let mut mesh = flat;
        mesh.uvs.clear();
        mesh.tangents = None;
        mesh.colors = None;
        mesh.skin = None;
        mesh.morph_targets.clear();
        mesh.material_ids.iter_mut().for_each(|s| *s = 0);
        if mesh.normals.is_none() {
            mesh.compute_normals(60.0);
        }
        // Copies of a vertex on the original's UV seams become one (triangles keep their order).
        repair::weld(&mut mesh, 0.0);

        emit(Event::Stage(&format!("{what}: unwrapping UVs")));
        let size = opts.shared_size.clamp(64, MAX_SHARED_SIZE);
        let size = if textured { size } else { size.min(512) };
        // Charts sit in their own block × block cells, `block` texels apart, so mips stay inside
        // a chart down to the level where a block is one texel: mip 3 of a 2048 atlas, mip 2 of
        // a 1024 one. Scenes with many charts get smaller blocks when those do not fit.
        // With a cap the atlas can come out far smaller than `size`; blocks sized for `size`
        // would then be most of it, so they follow the side the capped texels need instead.
        let need = cap.map(|_| {
            let texels = texels_needed(&mesh, &dens).max(1.0);
            ((texels / 0.5).sqrt().ceil() as u32)
                .next_power_of_two()
                .min(size)
        });
        let mut block = (need.unwrap_or(size) / 256).clamp(2, 8);
        let p = loop {
            let r = parameterize(
                &mesh,
                &ParamOptions {
                    atlas_size: size,
                    padding: block,
                    block,
                    texel_density: textured.then(|| dens.clone()),
                    source_texels: keep.clone(),
                    ..if opts.planar_uv {
                        ParamOptions::planar()
                    } else {
                        ParamOptions::default()
                    }
                },
            );
            match r {
                Ok(p) => break p,
                Err(_) if block > 1 => block /= 2,
                Err(e) => return Err(err(e)),
            }
        };
        let (w, h) = (p.width, p.height);
        emit(Event::Stage(&format!("{what}: baking {w}×{h}")));
        let mut mesh = p.mesh;
        compute_tangents(&mut mesh, 0).map_err(err)?;
        let (mut tex, coverage) = bake_textures(
            &mut MaterialHost::new(&self.materials, false),
            &mesh,
            &self.scene,
            &BakeOptions {
                width: w,
                height: h,
                supersample: 1,
                gutter: block,
                channels: vec![
                    channels::BASE_COLOR.into(),
                    channels::NORMAL.into(),
                    channels::ORM.into(),
                ],
                source_triangles: Some(refs),
                ..Default::default()
            },
            1,
            emit,
        )
        .map_err(err)?;
        let cutoff = self
            .materials
            .iter()
            .filter_map(|m| match m.info.class {
                MaterialClass::Masked { cutoff } => Some(cutoff),
                _ => None,
            })
            .reduce(f32::min);
        if let Some(c) = cutoff {
            tex.alpha_cutoff.insert(0, c);
        }
        let tex_bytes = texture_bytes(&tex);
        let texture_error = recut
            .then(|| self.texture_error(&source, &mesh, tex.base_color.get(&0)?))
            .flatten();

        let input_tris = self.scene.triangle_count();
        let base = opts.simplify_options();
        let step = opts.step.clamp(0.05, 0.95);
        let ppr = opts.ppr();
        let visible = opts.visible_error();
        let note = |level: usize, error: f32| match (visible, allowed) {
            (_, Some(a)) if level == 1 && Some(a) != visible => Some(format!(
                "LOD0 error: reduced until it deviates about {:.3}% of the diagonal; measured {:.3}%",
                100.0 * a / diag,
                100.0 * error / diag
            )),
            (Some(v), _) => Some(view_note(level, error, v, opts.view_distance)),
            _ => None,
        };
        let mut error_bound = 0.0;
        // Steps compound from the original (a recut's LOD0 is a step of none), or with a
        // budget from the first level.
        let mut from = (input_tris, usize::from(recut));
        // Error spacing: a shared chain's levels from the original (as `run_chain`'s far levels
        // count them), a recut's after LOD0, once LOD0's own deviation is known.
        let mut targets = if recut {
            None
        } else {
            opts.error_targets(opts.levels.clamp(1, 8), 0.0, diag)
        };
        let mut prev_error = 0.0;
        // Later levels' textures: how many times halved, the level whose GLB carries them, and
        // the textures. The halvings only grow, so equal ones share the carrier's.
        let density = atlas_density(&mesh, w, h);
        let down = opts.downsample_options();
        let mut shrunk = (0u32, 1usize, tex.clone());
        // The carrier's cooked textures, with the decoded ones the viewer draws.
        let mut cooked: Option<(Arc<Cooked>, ExportTextures)> = None;
        for level in 1..end {
            emit(Event::Stage(&format!("{}: reducing", name(level))));
            let goal = match &targets {
                // A recut's LOD0 keeps its own limit.
                Some(t) if !(recut && level == 1) => match t.get(level - 1 - usize::from(recut)) {
                    Some(&g) => Some(g),
                    None => return Ok(true), // no room for another level apart
                },
                _ => None,
            };
            let ratio = step.powi((level - from.1) as i32);
            let target = match goal {
                Some(_) => mesh.triangle_count(),
                None => ((from.0 as f32 * ratio) as usize).max(4),
            };
            let limit = goal.or(allowed.filter(|_| level == 1));
            let ((m, e, s, d), error) = fit_visible(limit, goal.is_some(), |k| {
                let (m, e, x) = if limit.is_none() && target >= mesh.triangle_count() {
                    (mesh.clone(), 0.0, false)
                } else {
                    let l = limit.map(|v| remaining(k * v, prev_error));
                    reduce_to(&mesh, &base, target, l).map_err(err)?
                };
                let mut t = m.clone();
                compute_tangents(&mut t, 0).map_err(err)?;
                let mut s = Scene::from_mesh(t);
                s.materials[0].name = "shared".into();
                let d = surface_distance(&source, &s.flatten().0, MEASURE_SAMPLES);
                Ok(((m, e, s, d), d.hausdorff, x))
            })?;
            if opts.past_max_error(level, error, diag) {
                return Ok(true);
            }
            mesh = m;
            error_bound += e;
            prev_error = error;
            if limit.is_some() {
                from = (mesh.triangle_count(), 1);
            }
            if recut && level == 1 {
                targets = opts.error_targets(end.saturating_sub(2), error, diag);
            }
            let k = opts.halvings(level, 1, error, density, shrunk.0, w.min(h));
            if k != shrunk.0 {
                emit(Event::Stage(&format!(
                    "{}: downsampling textures",
                    name(level)
                )));
                let t = shrink_textures(&tex, k, &down);
                // Cooking shows the decoded textures instead.
                if opts.texture_compression == TextureCompression::Off {
                    emit_textures(level, &t, emit);
                }
                shrunk = (k, level, t);
            }
            let (k, carrier, ltex) = (shrunk.0, shrunk.1, &shrunk.2);
            if level == carrier {
                cooked = cook_level(level, &s, ltex, opts, now, emit)?;
            }
            let none = ExportTextures::default();
            let shown = if level == carrier {
                cooked.as_ref().map_or(ltex, |c| &c.1)
            } else {
                &none
            };
            let glb = compact_glb(&s, shown).map_err(err)?;
            let ms = now() - t0;
            let mut info = LevelInfo {
                triangles: s.triangle_count(),
                vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
                error_bound,
                ppr,
                ms,
                note: note(level, error),
                view_texel_density: cap,
                view_texels,
                coverage: Some(coverage),
                charts: Some(p.charts),
                utilization: Some(p.utilization),
                bytes: Some(glb.len()),
                texture_bytes: Some(if k == 0 {
                    tex_bytes
                } else {
                    texture_bytes(ltex)
                }),
                atlas_width: Some((w >> k).max(1)),
                atlas_height: Some((h >> k).max(1)),
                density_scale: Some(p.density_scale),
                shared_textures: Some(carrier),
                texture_halvings: (k > 0).then_some(k),
                texture_filter: (k > 0).then(|| opts.texture_filter.clone()),
                label: recut.then(|| name(level)),
                ..LevelInfo::new(level, LevelKind::Shared)
            };
            if level == 1 {
                (info.texture_error, info.texture_error_p99) = texture_error.unzip();
            }
            if let Some((c, _)) = &cooked {
                let mut ci = c.info.clone();
                if level != carrier {
                    // This level ships its own geometry and the carrier's textures.
                    ci.ship_bytes = glb.len() + ci.ktx2_bytes;
                }
                info.cooked = Some(ci);
                self.cooked.insert(level, c.clone());
            }
            info.set_distance(&d, diag);
            emit(Event::Level {
                info: &info,
                scene: Some((&s, shown)),
            });
            self.levels.push((s, ltex.clone()));
            t0 = now();
        }
        Ok(false)
    }

    /// Base color deviation of `mesh` drawn with `atlas` from `source` (the flattened scene)
    /// drawn with the original materials, at matching points of the same triangles: the mean
    /// and 99th percentile of the largest channel's difference, in 8-bit sRGB levels. Points
    /// are spread by area.
    fn texture_error(&self, source: &Mesh, mesh: &Mesh, atlas: &Texture) -> Option<(f32, f32)> {
        let n = source.triangle_count().min(mesh.triangle_count());
        let area = |t: usize| {
            let [a, b, c] = source.triangle(t).map(|v| source.positions[v as usize]);
            (b - a).cross(c - a).length()
        };
        let total: f32 = (0..n).map(area).sum();
        if total <= 0.0 || mesh.uvs.is_empty() {
            return None;
        }
        let to_srgb = |c: Vec4| {
            glam::Vec3::new(
                linear_to_srgb(c.x.clamp(0.0, 1.0)),
                linear_to_srgb(c.y.clamp(0.0, 1.0)),
                linear_to_srgb(c.z.clamp(0.0, 1.0)),
            )
        };
        let mut diffs = Vec::with_capacity(MEASURE_SAMPLES + n);
        let (mut acc, mut i) = (0.0f64, 0usize);
        for t in 0..n {
            acc += f64::from(area(t) / total) * MEASURE_SAMPLES as f64;
            let m = self
                .materials
                .get(self.scene.material_id(source.material_ids[t]).0 as usize);
            let (sv, mv) = (source.triangle(t), mesh.triangle(t));
            while (i as f64) < acc {
                // R2 points folded into the triangle.
                let (mut u, mut w) = (
                    (0.5 + i as f32 * 0.754_877_7).fract(),
                    (0.5 + i as f32 * 0.569_840_3).fract(),
                );
                if u + w > 1.0 {
                    (u, w) = (1.0 - u, 1.0 - w);
                }
                i += 1;
                let k = [1.0 - u - w, u, w];
                let uv_of = |m: &Mesh, v: [u32; 3], set: usize| {
                    m.uvs.get(set).map_or(Vec2::ZERO, |uv| {
                        (0..3).map(|j| uv[v[j] as usize] * k[j]).sum()
                    })
                };
                let mut want = source.colors.as_ref().map_or(Vec4::ONE, |c| {
                    (0..3).map(|j| c[sv[j] as usize] * k[j]).sum()
                });
                if let Some(m) = m {
                    want *= m.base_color_factor;
                    if let Some((tex, set)) = &m.base_color_texture {
                        want *= tex.sample_bilinear(uv_of(source, sv, *set as usize), true);
                    }
                }
                let got = atlas.sample_bilinear(uv_of(mesh, mv, 0), false);
                diffs.push((to_srgb(want) - to_srgb(got)).abs().max_element() * 255.0);
            }
        }
        if diffs.is_empty() {
            return None;
        }
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        diffs.sort_by(f32::total_cmp);
        Some((mean, diffs[(diffs.len() - 1) * 99 / 100]))
    }

    /// The artist's texel density per triangle of the flattened scene, from the largest
    /// texture of its material (texels per world unit; 0 where untextured).
    fn artist_density(&self, flat: &Mesh) -> Vec<f32> {
        let largest = self.largest_textures();
        let sizes = |set: u32| -> Vec<(u32, u32)> {
            largest
                .iter()
                .map(|x| match x {
                    Some((w, h, s)) if *s == set => (*w, *h),
                    _ => (0, 0),
                })
                .collect()
        };
        texel_density(flat, 0, &sizes(0))
            .into_iter()
            .zip(texel_density(flat, 1, &sizes(1)))
            .map(|(a, b)| a.max(b))
            .collect()
    }

    /// Per material slot of the scene, its largest texture: width, height and UV set.
    fn largest_textures(&self) -> Vec<Option<(u32, u32, u32)>> {
        (0..self.scene.materials.len())
            .map(|slot| {
                let m = self
                    .materials
                    .get(self.scene.material_id(slot as u32).0 as usize)?;
                [
                    m.base_color_texture.as_ref().map(|(t, s)| (t, *s)),
                    m.metallic_roughness_texture.as_ref().map(|(t, s)| (t, *s)),
                    m.normal_texture.as_ref().map(|(t, s, _)| (t, *s)),
                ]
                .into_iter()
                .flatten()
                .max_by_key(|(t, _)| u64::from(t.width) * u64::from(t.height))
                .map(|(t, s)| (t.width, t.height, s))
            })
            .collect()
    }
}

/// The deviation a further reduction may add to reach `goal`, from a level that already deviates
/// `already` (the measured distances add at most). At least a tenth of the goal, so a level that
/// measured near it still reduces a little.
fn remaining(goal: f32, already: f32) -> f32 {
    (goal - already).max(0.1 * goal)
}

/// Reduces `mesh` to `target` triangles, then with `limit` (an estimated deviation, mesh
/// units) on until the simplifier's estimate reaches it: the coarser of the two. Returns the
/// mesh, its estimated deviation, and whether the limit took it past the target.
fn reduce_to(
    mesh: &Mesh,
    opts: &SimplifyOptions,
    target: usize,
    limit: Option<f32>,
) -> Result<(Mesh, f32, bool), Error> {
    // A step of none (a recut's LOD0) with a limit: only the limit reduces, so skip the pass
    // that would build the whole reducer to remove nothing.
    if let Some(l) = limit
        && target >= mesh.triangle_count()
    {
        let r = simplify(
            mesh,
            &SimplifyOptions {
                target_ratio: None,
                target_triangles: None,
                max_error: Some(l),
                ..opts.clone()
            },
        )?;
        return Ok((r.mesh, r.error, true));
    }
    let r = simplify(
        mesh,
        &SimplifyOptions {
            target_triangles: Some(target),
            ..opts.clone()
        },
    )?;
    match limit {
        Some(l) if r.error < l => {
            let more = simplify(
                &r.mesh,
                &SimplifyOptions {
                    target_ratio: None,
                    target_triangles: None,
                    max_error: Some(l - r.error),
                    ..opts.clone()
                },
            )?;
            Ok((more.mesh, r.error + more.error, true))
        }
        _ => Ok((r.mesh, r.error, false)),
    }
}

/// Builds a level that may go to a visible limit of `limit` world units: `build(k)` reduces
/// with the limit scaled by `k` and returns the level, its measured deviation (world units),
/// and whether the limit took it past its step. The simplifier's estimate can run well below
/// the measured Hausdorff distance (thin parts, small pieces), so when the measured deviation
/// is over a pixel and a quarter, the scale is bisected (a few builds at most) for the coarsest
/// level that measures within it. With `reach` (levels spaced by error, where the limit is the
/// level's goal), the estimate can also run well above the measurement, so a level that
/// measures under 70% of its goal is searched upward the same way.
fn fit_visible<T>(
    limit: Option<f32>,
    reach: bool,
    mut build: impl FnMut(f32) -> Result<(T, f32, bool), String>,
) -> Result<(T, f32), String> {
    let (level, error, extended) = build(1.0)?;
    let v = match limit {
        Some(v) if extended && error > 1.25 * v => 1.25 * v,
        Some(v) if reach && extended && error < 0.7 * v => {
            return reach_up(v, (level, error), build);
        }
        _ => return Ok((level, error)),
    };
    // The finest scale that measured too coarse, and the coarsest that fit.
    let (mut bad, mut finest) = (1.0f32, (level, error));
    let mut good: Option<(f32, (T, f32))> = None;
    let mut k = (v / finest.1).clamp(0.2, 0.7);
    for _ in 0..4 {
        let (level, error, extended) = build(k)?;
        if extended && error > v {
            bad = k;
            finest = (level, error);
        } else {
            good = Some((k, (level, error)));
            if !extended {
                break; // the step alone: a smaller scale changes nothing
            }
        }
        k = match &good {
            Some((g, _)) if bad / g < 1.15 => break,
            Some((g, _)) => (g * bad).sqrt(),
            None => k * (v / finest.1).clamp(0.2, 0.7),
        };
    }
    Ok(good.map_or(finest, |g| g.1))
}

/// [`fit_visible`]'s upward search: from a level at scale 1 that measured `fit.1`, well under
/// the goal `v`, the coarsest level (a few builds at most) that still measures within it.
fn reach_up<T>(
    v: f32,
    fit: (T, f32),
    mut build: impl FnMut(f32) -> Result<(T, f32, bool), String>,
) -> Result<(T, f32), String> {
    let (mut good, mut bad) = ((1.0f32, fit), None::<f32>);
    let mut k = (v / good.1.1.max(1e-9)).clamp(1.3, 4.0);
    for _ in 0..3 {
        let (level, error, _) = build(k)?;
        if error <= v {
            good = (k, (level, error));
            if error >= 0.7 * v {
                break; // close enough to the goal
            }
        } else {
            bad = Some(k);
        }
        k = match bad {
            Some(b) if b / good.0 < 1.15 => break,
            Some(b) => (good.0 * b).sqrt(),
            None => (good.0 * (v / good.1.1.max(1e-9)).clamp(1.3, 4.0)).min(64.0),
        };
    }
    Ok(good.1)
}

/// The fraction of the artist's texels (area × density², over textured triangles) that a cap
/// of `cap` texels per unit keeps; `None` when nothing is textured.
fn texels_under(flat: &Mesh, dens: &[f32], cap: f32) -> Option<f32> {
    let (mut all, mut kept) = (0.0f64, 0.0f64);
    for (t, &d) in dens.iter().enumerate() {
        if d > 0.0 {
            let [a, b, c] = flat.triangle_positions(t);
            let area = f64::from(farthing::math::triangle_area(a, b, c));
            all += area * f64::from(d) * f64::from(d);
            kept += area * f64::from(d.min(cap)).powi(2);
        }
    }
    (all > 0.0).then(|| (kept / all) as f32)
}

/// Texels `dens` (per triangle, texels per unit) asks for over `mesh`: area × density².
fn texels_needed(mesh: &Mesh, dens: &[f32]) -> f32 {
    dens.iter()
        .enumerate()
        .map(|(t, &d)| {
            let [a, b, c] = mesh.triangle_positions(t);
            f64::from(farthing::math::triangle_area(a, b, c)) * f64::from(d) * f64::from(d)
        })
        .sum::<f64>() as f32
}

/// How a level looks from the closest view of a budget: `visible` is one pixel there.
fn view_note(level: usize, error: f32, visible: f32, distance: f32) -> String {
    let px = error / visible;
    match level {
        0 => format!("Finer than the closest view ({distance} away) can show"),
        1 => format!(
            "Visible limit: reduced until it deviates about a pixel at {distance}; measured {px:.1} px"
        ),
        _ if px <= 1.0 => format!("Within a pixel of LOD0 even at {distance}"),
        _ => format!("{px:.1} px from LOD0 at {distance}"),
    }
}

/// Where each triangle of the flattened scene reads its material's largest texture (`largest`
/// per slot: width, height, UV set), for keeping the artist's UV layout in the shared atlas.
/// Islands that cover the same texels share them if `share` and no vertex colors tint the
/// copies differently.
fn source_texels(flat: &Mesh, largest: &[Option<(u32, u32, u32)>], share: bool) -> SourceTexels {
    let mut uvs = Vec::with_capacity(flat.indices.len());
    let texture = (0..flat.triangle_count())
        .map(|t| {
            let slot = flat.material_ids[t] as usize;
            let set = largest.get(slot).copied().flatten();
            match set.and_then(|(_, _, s)| flat.uvs.get(s as usize)) {
                Some(uv) => {
                    uvs.extend(flat.triangle(t).map(|v| uv[v as usize]));
                    Some(slot as u32)
                }
                None => {
                    uvs.extend([Vec2::ZERO; 3]);
                    None
                }
            }
        })
        .collect();
    let tinted = flat
        .colors
        .as_ref()
        .is_some_and(|c| c.iter().any(|&c| c != Vec4::ONE));
    SourceTexels {
        uvs,
        texture,
        sizes: largest
            .iter()
            .map(|x| x.map_or((0, 0), |(w, h, _)| (w, h)))
            .collect(),
        share: share && !tinted,
    }
}

/// Bakes `mesh` (in world space) from `scene` through `host`, emits the textures for `level`
/// and returns them for export (slot 0), with the bake's coverage.
///
/// Rows are quantized to 8 bits as the bake finishes them, and the correspondence is not
/// kept, so the bake never holds a float image or a correspondence entry for the whole atlas.
fn bake_textures(
    host: &mut dyn RenderHost,
    mesh: &Mesh,
    scene: &Scene,
    opts: &BakeOptions,
    level: usize,
    emit: &mut dyn FnMut(Event),
) -> Result<(ExportTextures, f32), Error> {
    let (w, h) = (opts.width, opts.height);
    let mut texels: BTreeMap<String, Vec<[u8; 4]>> = BTreeMap::new();
    let opts = BakeOptions {
        keep_correspondence: false,
        ..opts.clone()
    };
    let bake = bake_rows_with_host(host, mesh, Mat4::IDENTITY, scene, &opts, &mut |r| {
        let encode = encoder(r.channel, r.encoding);
        texels
            .entry(r.channel.to_owned())
            .or_insert_with(|| Vec::with_capacity(w as usize * h as usize))
            .extend(r.pixels.iter().map(|&p| encode(p)));
    })?;
    let mut tex = ExportTextures::default();
    for (name, texels) in texels {
        // The page shows the texels opaque.
        let opaque: Vec<[u8; 4]> = texels.iter().map(|&[r, g, b, _]| [r, g, b, 255]).collect();
        emit(Event::Texture {
            level,
            channel: &name,
            width: w,
            height: h,
            rgba: opaque.as_flattened(),
        });
        drop(opaque);
        let texture = |srgb| Texture::from_rgba8(w, h, texels.as_flattened(), srgb);
        match name.as_str() {
            channels::BASE_COLOR => {
                tex.base_color.insert(0, texture(true)?);
            }
            channels::NORMAL => {
                tex.normal.insert(0, texture(false)?);
            }
            channels::ORM => {
                tex.metallic_roughness.insert(0, texture(false)?);
            }
            _ => {}
        }
    }
    Ok((tex, bake.coverage))
}

/// `tex` with every texture's sides halved `times` times, normal maps filtered as normals.
fn shrink_textures(tex: &ExportTextures, times: u32, opts: &DownsampleOptions) -> ExportTextures {
    let normal = DownsampleOptions {
        normal_map: true,
        ..*opts
    };
    let mut out = tex.clone();
    out.base_color
        .values_mut()
        .for_each(|t| *t = t.halve(times, opts));
    out.metallic_roughness
        .values_mut()
        .for_each(|t| *t = t.halve(times, opts));
    out.normal
        .values_mut()
        .for_each(|t| *t = t.halve(times, &normal));
    out
}

/// Cooks `t`, the textures level `level` (`s`) carries, when the options ask for compression:
/// shows the decoded textures in the panel and returns the cooked files and stats with the
/// decoded textures for the viewer's GLB.
fn cook_level(
    level: usize,
    s: &Scene,
    t: &ExportTextures,
    opts: &LodOptions,
    now: &dyn Fn() -> f64,
    emit: &mut dyn FnMut(Event),
) -> Result<Option<(Arc<Cooked>, ExportTextures)>, String> {
    let Some((mut c, decoded)) =
        cook(t, &opts.cook_options(), now, &mut |m| emit(Event::Stage(m)))?
    else {
        return Ok(None);
    };
    c.info.level = level;
    let geometry = compact_glb(s, &bare(t)).map_err(|e| e.to_string())?;
    c.info.ship_bytes = geometry.len() + c.info.ktx2_bytes;
    emit_textures(level, &decoded, emit);
    Ok(Some((Arc::new(c), decoded)))
}

/// Shows `tex`'s slot-0 textures in the page's texture panel for `level`, opaque.
fn emit_textures(level: usize, tex: &ExportTextures, emit: &mut dyn FnMut(Event)) {
    let channels = [
        (channels::BASE_COLOR, tex.base_color.get(&0)),
        (channels::NORMAL, tex.normal.get(&0)),
        (channels::ORM, tex.metallic_roughness.get(&0)),
    ];
    for (channel, t) in channels {
        let Some(t) = t else { continue };
        let opaque: Vec<u8> = t
            .rgba8()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[r, g, b, _]| [r, g, b, 255])
            .collect();
        emit(Event::Texture {
            level,
            channel,
            width: t.width,
            height: t.height,
            rgba: &opaque,
        });
    }
}

/// Texels per world unit `mesh`'s UV set 0 gives in a `width × height` atlas, averaged by area.
fn atlas_density(mesh: &Mesh, width: u32, height: u32) -> f32 {
    let Some(uv) = mesh.uvs.first() else {
        return 0.0;
    };
    let (mut texels, mut area) = (0.0f64, 0.0f64);
    for t in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle(t).map(|v| v as usize);
        let [pa, pb, pc] = [a, b, c].map(|v| mesh.positions[v]);
        let [ua, ub, uc] = [a, b, c].map(|v| uv[v] * Vec2::new(width as f32, height as f32));
        area += f64::from((pb - pa).cross(pc - pa).length());
        texels += f64::from((ub - ua).perp_dot(uc - ua).abs());
    }
    if area > 0.0 {
        (texels / area).sqrt() as f32
    } else {
        0.0
    }
}

/// How a baked channel is stored: sRGB base color, signed normals, linear data.
fn encoder(channel: &str, encoding: ChannelEncoding) -> fn(Vec4) -> [u8; 4] {
    match (channel, encoding) {
        (channels::BASE_COLOR, _) | (_, ChannelEncoding::Color) => |p| Texture::encode(p, true),
        (channels::NORMAL, _)
        | (_, ChannelEncoding::NormalTangent | ChannelEncoding::NormalWorld) => {
            Texture::encode_normal
        }
        _ => |p| Texture::encode(p, false),
    }
}

/// Every GLB the demo produces: quantized, meshopt-compressed and reordered for the GPU's
/// caches. The viewer draws from [`scene_payload`]; the GLB's size is what a level ships.
pub(crate) fn compact_glb(s: &Scene, t: &ExportTextures) -> farthing::Result<Vec<u8>> {
    glb_bytes_with(s, t, &GlbOptions::compact())
}

/// A scene as the viewer takes it: JSON naming typed arrays by their index in the buffers,
/// so the page needs no glTF reader.
///
/// * `meshes`: `{positions, normals, uvs, colors, indices, parts}`. Positions and normals are
///   xyz per vertex, uvs (set 0) uv per vertex and colors linear RGB per vertex, or -1 when the
///   mesh has none; normals are computed when it has none. `indices` hold the triangles sorted
///   by material slot, and `parts` are `[first index, index count, slot]`.
/// * `instances`: `{mesh, matrix}`, the world transform column-major.
/// * `materials`, one per slot: `{name, base_color, metallic, roughness, cutoff,
///   base_color_texture, metallic_roughness_texture, normal_texture}` with the export's
///   defaults, textures by index or -1. Alpha-tested slots are double-sided, as exported.
/// * `textures`: `{width, height, srgb, rgba}`, RGBA8 rows top to bottom, each one once.
pub fn scene_payload(s: &Scene, t: &ExportTextures) -> (serde_json::Value, Vec<Buffer>) {
    use serde_json::json;
    let mut buffers = Vec::new();
    let mut push = |b: Buffer| {
        buffers.push(b);
        buffers.len() as i64 - 1
    };
    let meshes: Vec<_> = s
        .meshes
        .iter()
        .map(|m| {
            let smoothed;
            let m = if m.normals.is_none() {
                let mut c = m.clone();
                c.compute_normals(60.0);
                smoothed = c;
                &smoothed
            } else {
                m
            };
            let xyz = |v: &[glam::Vec3]| v.iter().flat_map(|p| p.to_array()).collect::<Vec<f32>>();
            let positions = push(Buffer::F32(xyz(&m.positions)));
            let normals = push(Buffer::F32(xyz(m.normals.as_deref().unwrap_or_default())));
            let uvs = m.uvs.first().map_or(-1, |u| {
                push(Buffer::F32(u.iter().flat_map(|p| p.to_array()).collect()))
            });
            let colors = m.colors.as_ref().map_or(-1, |c| {
                push(Buffer::F32(
                    c.iter().flat_map(|c| c.truncate().to_array()).collect(),
                ))
            });
            let slot = |t: u32| m.material_ids.get(t as usize).copied().unwrap_or(0);
            let mut order: Vec<u32> = (0..(m.indices.len() / 3) as u32).collect();
            order.sort_by_key(|&t| slot(t));
            let mut indices = Vec::with_capacity(m.indices.len());
            let mut parts = Vec::new();
            for run in order.chunk_by(|&a, &b| slot(a) == slot(b)) {
                parts.push([indices.len() as u32, run.len() as u32 * 3, slot(run[0])]);
                for &t in run {
                    indices.extend_from_slice(&m.indices[t as usize * 3..t as usize * 3 + 3]);
                }
            }
            let indices = push(Buffer::U32(indices));
            json!({
                "positions": positions,
                "normals": normals,
                "uvs": uvs,
                "colors": colors,
                "indices": indices,
                "parts": parts,
            })
        })
        .collect();
    let instances: Vec<_> = s
        .instances
        .iter()
        .map(|i| json!({ "mesh": i.mesh, "matrix": i.transform.to_cols_array() }))
        .collect();
    let mut textures: Vec<(Texture, bool)> = Vec::new();
    let mut texture_json = Vec::new();
    let mut texture = |tex: Option<&Texture>, srgb: bool| -> i64 {
        let Some(tex) = tex else { return -1 };
        if let Some(i) = textures
            .iter()
            .position(|(t, s)| *s == srgb && t.same_texels(tex))
        {
            return i as i64;
        }
        textures.push((tex.clone(), srgb));
        let rgba = push(Buffer::U8(tex.rgba8_as(srgb).into_owned()));
        texture_json.push(json!({
            "width": tex.width,
            "height": tex.height,
            "srgb": srgb,
            "rgba": rgba,
        }));
        textures.len() as i64 - 1
    };
    let materials: Vec<_> = s
        .materials
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let i = i as u32;
            let mr_texture = t.metallic_roughness.get(&i);
            let mr =
                t.metallic_roughness_factor
                    .get(&i)
                    .copied()
                    .unwrap_or(if mr_texture.is_some() {
                        Vec2::ONE
                    } else {
                        Vec2::new(0.0, 0.8)
                    });
            json!({
                "name": slot.name,
                "base_color": t.base_color_factor.get(&i).copied().unwrap_or(Vec4::ONE).to_array(),
                "metallic": mr.x,
                "roughness": mr.y,
                "cutoff": t.alpha_cutoff.get(&i),
                "base_color_texture": texture(t.base_color.get(&i), true),
                "metallic_roughness_texture": texture(mr_texture, false),
                "normal_texture": texture(t.normal.get(&i), false),
            })
        })
        .collect();
    let json = json!({
        "meshes": meshes,
        "instances": instances,
        "materials": materials,
        "textures": texture_json,
    });
    (json, buffers)
}

/// GPU memory of export textures: RGBA8 with a full mip chain (4/3), each texture counted
/// once however many slots share it.
pub(crate) fn texture_bytes(t: &ExportTextures) -> usize {
    let mut seen: Vec<&Texture> = Vec::new();
    for tex in t
        .base_color
        .values()
        .chain(t.normal.values())
        .chain(t.metallic_roughness.values())
    {
        if !seen.iter().any(|s| s.same_texels(tex)) {
            seen.push(tex);
        }
    }
    seen.iter()
        .map(|t| t.width as usize * t.height as usize * 16 / 3)
        .sum()
}

fn expand_zips(files: Vec<(String, Vec<u8>)>) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    for (name, data) in files {
        let name = name.replace('\\', "/");
        if !name.to_ascii_lowercase().ends_with(".zip") {
            out.push((name, data));
            continue;
        }
        let mut zip =
            zip::ZipArchive::new(Cursor::new(data)).map_err(|e| format!("{name}: {e}"))?;
        for i in 0..zip.len() {
            let mut f = zip.by_index(i).map_err(|e| format!("{name}: {e}"))?;
            let n = f
                .name()
                .map_err(|e| format!("{name}: entry {i}: {e}"))?
                .replace('\\', "/");
            if f.is_dir() || n.starts_with("__MACOSX/") {
                continue;
            }
            let mut buf = Vec::with_capacity(f.size() as usize);
            f.read_to_end(&mut buf)
                .map_err(|e| format!("{name}/{n}: {e}"))?;
            out.push((n, buf));
        }
    }
    Ok(out)
}

/// `a/./b/../c` → `a/c`.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for p in path.split('/') {
        match p {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

/// The materials' textures and factors for export (the textures are shared, not copied).
pub(crate) fn original_textures(materials: &[GltfMaterial]) -> ExportTextures {
    let mut t = ExportTextures::default();
    for (i, m) in materials.iter().enumerate() {
        let i = i as u32;
        t.base_color_factor.insert(i, m.base_color_factor);
        t.metallic_roughness_factor
            .insert(i, glam::Vec2::new(m.metallic_factor, m.roughness_factor));
        if let MaterialClass::Masked { cutoff } = m.info.class {
            t.alpha_cutoff.insert(i, cutoff);
        }
        if let Some((tex, _)) = &m.base_color_texture {
            t.base_color.insert(i, tex.clone());
        }
        if let Some((tex, _)) = &m.metallic_roughness_texture {
            t.metallic_roughness.insert(i, tex.clone());
        }
        if let Some((tex, _, _)) = &m.normal_texture {
            t.normal.insert(i, tex.clone());
        }
    }
    t
}

/// What a browser demo can know about glTF materials: base color texture × factor × vertex
/// color, metallic-roughness texture × factors, and the source's normal with its normal map
/// applied (when the source has tangents).
///
/// Materials farthing creates ([`RenderHost::create_material`], HLOD proxies) get ids after
/// the glTF's, and their baked textures are kept (as 8-bit textures) so a parent proxy can be
/// baked from its children's materials.
pub(crate) struct MaterialHost<'a> {
    materials: &'a [GltfMaterial],
    baked: Vec<BakedMaterial>,
    /// Try triangle-ID renders on the GPU (WebGL2) before the CPU ray caster.
    gpu: bool,
    /// What the visibility renders cost so far.
    pub(crate) visibility: VisibilityStats,
}

/// Triangle-ID renders (HLOD visibility) a [`MaterialHost`] served, and where.
#[derive(Default)]
pub(crate) struct VisibilityStats {
    /// Renders done on the GPU, and on the CPU.
    pub(crate) gpu_calls: u32,
    pub(crate) cpu_calls: u32,
    /// Views rendered, and the wall time spent rendering them.
    pub(crate) views: u32,
    pub(crate) ms: f64,
}

/// A material made from a bake: base color (sRGB), tangent-space normal and glTF ORM, all in
/// UV set 0.
struct BakedMaterial {
    base_color: Option<Texture>,
    normal: Option<Texture>,
    orm: Option<Texture>,
}

/// What [`MaterialHost`] evaluates at a surface point.
struct Shaded {
    color: Vec4,
    normal: glam::Vec3,
    orm: Vec4,
}

impl<'a> MaterialHost<'a> {
    /// `gpu`: render visibility on the GPU when one is available.
    pub(crate) fn new(materials: &'a [GltfMaterial], gpu: bool) -> MaterialHost<'a> {
        MaterialHost {
            materials,
            baked: Vec::new(),
            gpu,
            visibility: VisibilityStats::default(),
        }
    }

    /// World-space shading normal: the interpolated normal, perturbed by a tangent-space normal
    /// map when the source mesh has tangents.
    fn shading_normal(
        s: &farthing::host::SurfaceSample,
        map: Option<(&Texture, u32, f32)>,
    ) -> glam::Vec3 {
        let n = s.normal.normalize_or_zero();
        let t = s.tangent.truncate();
        let Some((tex, set, scale)) = map else {
            return n;
        };
        if t.length_squared() < 1e-12 || n == glam::Vec3::ZERO {
            return n;
        }
        let t = (t - n * n.dot(t)).normalize_or_zero();
        let b = n.cross(t) * if s.tangent.w < 0.0 { -1.0 } else { 1.0 };
        let m = tex.sample_bilinear(s.uv[(set as usize).min(1)], true);
        let (x, y, z) = (
            (m.x * 2.0 - 1.0) * scale,
            (m.y * 2.0 - 1.0) * scale,
            m.z * 2.0 - 1.0,
        );
        (t * x + b * y + n * z).try_normalize().unwrap_or(n)
    }

    fn shade(&self, s: &farthing::host::SurfaceSample) -> Shaded {
        let id = s.material.0 as usize;
        if let Some(m) = self.materials.get(id) {
            let uv = |set: u32| s.uv[(set as usize).min(1)];
            let mut color = s.color * m.base_color_factor;
            if let Some((tex, set)) = &m.base_color_texture {
                color *= tex.sample_bilinear(uv(*set), true);
            }
            if m.info.class == MaterialClass::Opaque {
                color.w = 1.0;
            }
            // glTF layout: R occlusion (unused here), G roughness, B metalness.
            let mut orm = Vec4::new(1.0, m.roughness_factor, m.metallic_factor, 1.0);
            if let Some((tex, set)) = &m.metallic_roughness_texture {
                let t = tex.sample_bilinear(uv(*set), true);
                orm.y *= t.y;
                orm.z *= t.z;
            }
            let map = m.normal_texture.as_ref().map(|(t, set, k)| (t, *set, *k));
            return Shaded {
                color,
                normal: Self::shading_normal(s, map),
                orm,
            };
        }
        if let Some(b) = id
            .checked_sub(self.materials.len())
            .and_then(|i| self.baked.get(i))
        {
            let sample = |t: &Option<Texture>, fallback: Vec4| {
                t.as_ref()
                    .map_or(fallback, |t| t.sample_bilinear(s.uv[0], false))
            };
            let mut color = sample(&b.base_color, Vec4::ONE);
            color.w = 1.0;
            return Shaded {
                color,
                normal: Self::shading_normal(s, b.normal.as_ref().map(|t| (t, 0, 1.0))),
                orm: sample(&b.orm, Vec4::new(1.0, 0.8, 0.0, 1.0)),
            };
        }
        // Unbaked child proxies (`hlod::PROXY_MATERIAL`) and unknown slots: plain gray.
        Shaded {
            color: Vec4::new(0.7, 0.7, 0.7, 1.0),
            normal: s.normal.normalize_or_zero(),
            orm: Vec4::new(1.0, 0.8, 0.0, 1.0),
        }
    }
}

impl RenderHost for MaterialHost<'_> {
    fn material_info(&self, material: MaterialId) -> MaterialInfo {
        self.materials
            .get(material.0 as usize)
            .map_or_else(MaterialInfo::default, |m| m.info)
    }

    fn bake_surface(&mut self, req: &SurfaceBakeRequest) -> farthing::Result<ChannelImages> {
        let n = (req.width * req.height) as usize;
        let mut color = vec![Vec4::ZERO; n];
        let mut normal = vec![Vec4::ZERO; n];
        let mut orm = vec![Vec4::ZERO; n];
        for (i, s) in req.samples.iter().enumerate() {
            let Some(s) = s else { continue };
            let sh = self.shade(s);
            color[i] = sh.color;
            normal[i] = sh.normal.extend(1.0);
            orm[i] = sh.orm;
        }
        let image = |pixels: &mut Vec<Vec4>| Image {
            width: req.width,
            height: req.height,
            pixels: std::mem::take(pixels),
        };
        let mut out = ChannelImages::new();
        for ch in &req.channels {
            let (img, encoding) = match ch.as_str() {
                channels::BASE_COLOR => (image(&mut color), ChannelEncoding::Color),
                channels::NORMAL => (image(&mut normal), ChannelEncoding::NormalWorld),
                channels::ORM => (image(&mut orm), ChannelEncoding::Data),
                _ => {
                    return Err(Error::Unsupported(
                        "channel (demo host: base_color, normal, orm)",
                    ));
                }
            };
            out.insert(
                ch.clone(),
                ChannelImage {
                    image: img,
                    encoding,
                },
            );
        }
        Ok(out)
    }

    fn render_views(&mut self, _req: &ViewRenderRequest) -> farthing::Result<Vec<ViewImages>> {
        Err(Error::Unsupported("render_views"))
    }

    fn render_ids(&mut self, req: &IdRenderRequest) -> farthing::Result<Vec<IdImage>> {
        let t0 = crate::gpu::now_ms();
        let infos: Vec<MaterialInfo> = req
            .scene
            .materials
            .iter()
            .map(|m| self.material_info(m.id))
            .collect();
        let gpu = self
            .gpu
            .then(|| crate::gpu::render_ids(req, &infos))
            .flatten();
        let v = &mut self.visibility;
        let images = match gpu {
            Some(images) => {
                v.gpu_calls += 1;
                images
            }
            None => {
                v.cpu_calls += 1;
                farthing::raster::cpu_render_ids(req.scene, &req.views, &infos)
            }
        };
        v.views += req.views.len() as u32;
        v.ms += crate::gpu::now_ms() - t0;
        Ok(images)
    }

    fn create_material(&mut self, desc: &BakedMaterialDesc) -> farthing::Result<MaterialId> {
        let get = |name: &str| desc.textures.get(name);
        self.baked.push(BakedMaterial {
            base_color: get(channels::BASE_COLOR).map(|c| Texture::from_image(&c.image, true)),
            normal: get(channels::NORMAL).map(|c| Texture::from_signed_normals(&c.image)),
            orm: get(channels::ORM).map(|c| Texture::from_image(&c.image, false)),
        });
        Ok(MaterialId(
            (self.materials.len() + self.baked.len() - 1) as u32,
        ))
    }
}
