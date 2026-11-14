//! The demo pipeline, independent of the browser so it can be tested natively.
//!
//! * [`Demo::load`]: glTF / GLB from dropped files or a zip (external buffers and textures
//!   resolved by relative path).
//! * [`Demo::run`]: a LOD chain. Near levels are reduced per mesh and keep the original
//!   materials, or ([`TextureMode::Shared`]) are reduced from one re-unwrapped copy of the
//!   whole scene whose atlas holds the original materials at the artist's resolution; far levels (from `bake_from`) are geometry-only proxies of the whole scene with
//!   fresh UVs and base color, normal and roughness/metalness maps baked from the original. Every finished level and
//!   baked texture is reported through an [`Event`] as soon as it exists.
//! * The other example modes ([`DemoMode`]: option examples, cluster DAG, HLOD tree, splats)
//!   live in [`crate::modes`]; [`Demo::run`] dispatches on [`LodOptions::mode`].

use crate::modes::Extra;
use farthing::bake::{bake_with_host, compute_tangents, BakeOptions};
use farthing::host::{
    channels, BakedMaterialDesc, ChannelEncoding, ChannelImage, ChannelImages, IdImage,
    IdRenderRequest, MaterialClass, MaterialInfo, RenderHost, SurfaceBakeRequest, ViewImages,
    ViewRenderRequest,
};
use farthing::image::{linear_to_srgb, Image, Texture};
use farthing::io::{glb_bytes, load_gltf_with, ExportTextures, GltfMaterial};
use farthing::metrics::surface_distance;
use farthing::reduce::{repair, simplify, Quality, SeamMode, SimplifyOptions, Symmetry};
use farthing::uv::{parameterize, texel_density, ParamOptions};
use farthing::{Error, MaterialId, Mesh, Scene};
use glam::{Mat4, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// Surface samples per mesh when measuring a level against LOD0.
pub(crate) const MEASURE_SAMPLES: usize = 20_000;

/// Pixels per radian of the reference view the switch distances are given for: 1080 pixels
/// high, 60 degree vertical field of view.
pub(crate) fn reference_ppr() -> f32 {
    1080.0 / (2.0 * 30f32.to_radians().tan())
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
}

/// Largest shared atlas side: a 2048² bake peaks around half a gigabyte, and 4096² would need
/// four times that, more than a browser gives WebAssembly.
pub const MAX_SHARED_SIZE: u32 = 2048;

/// Which textures the chain's near levels use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureMode {
    /// Each mesh reduced on its own, keeping the original UVs and materials.
    #[default]
    Original,
    /// The whole scene re-unwrapped once (charts cut where farthing wants them, mip-safe
    /// 8 × 8 blocks), the original materials baked into that one atlas at the artist's
    /// texel density, and every near level reduced from that mesh.
    Shared,
}

/// Options the page exposes. Unknown or missing fields take the defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct LodOptions {
    /// Levels after LOD0 (1–8).
    pub levels: usize,
    /// Triangle ratio between consecutive levels.
    pub step: f32,
    pub fast: bool,
    pub strict_seams: bool,
    pub symmetry: bool,
    pub lock_border: bool,
    /// First level (1-based) that becomes a baked proxy; 0 = none.
    pub bake_from: usize,
    /// Baked texture size (square).
    pub atlas: u32,
    /// Bake samples per texel along each axis (1–4).
    pub supersample: u32,
    /// Planar charts instead of LSCM for baked levels and the shared atlas.
    pub planar_uv: bool,
    pub textures: TextureMode,
    /// Shared atlas: largest side (at most [`MAX_SHARED_SIZE`]). The atlas keeps the artist's
    /// texel density when that fits, and shrinks every chart by the same factor when it does not.
    pub shared_size: u32,
    /// Closest distance the scene is ever seen from, in its own units (glTF: metres); 0 = no
    /// viewing budget. With one, nothing is kept that would not show there: the first level is
    /// reduced until it deviates about a pixel at that distance, the shared atlas is capped at
    /// a texel per pixel, and each baked atlas is sized for the distance its level appears at.
    pub view_distance: f32,
    /// Vertical field of view of that view, degrees. Switch distances are given for it.
    pub view_fov: f32,
    /// Screen height of that view, pixels.
    pub view_height: u32,

    pub mode: DemoMode,
    /// Option examples: triangle fraction every configuration reduces to.
    pub preset_ratio: f32,
    /// Cluster DAG: `"nanite"` (128 triangles, 255 vertices) or `"meshlet"` (124, 64).
    pub cluster_layout: String,
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
}

impl Default for LodOptions {
    fn default() -> Self {
        LodOptions {
            levels: 4,
            step: 0.5,
            fast: false,
            strict_seams: false,
            symmetry: false,
            lock_border: false,
            bake_from: 4,
            atlas: 512,
            supersample: 2,
            planar_uv: false,
            textures: TextureMode::Original,
            shared_size: 2048,
            view_distance: 0.0,
            view_fov: 60.0,
            view_height: 1080,
            mode: DemoMode::Chain,
            preset_ratio: 0.25,
            cluster_layout: "nanite".into(),
            cluster_fast: false,
            hlod_copies: 3,
            hlod_split: 1,
            hlod_triangles: 2000,
            hlod_bake: true,
            splat_count: 20_000,
            splat_footprints: true,
            gpu_visibility: true,
        }
    }
}

impl LodOptions {
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
    /// Reduced from the re-unwrapped scene; textures in the shared atlas.
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
    /// Measured deviation from LOD0: sampled two-way Hausdorff distance, world units.
    pub error: f32,
    /// `error` relative to the scene's bounding-box diagonal.
    pub relative_error: f32,
    /// The simplifier's own estimate, summed over the chain: an upper bound, often loose.
    pub error_bound: f32,
    /// Camera distance at which `error` projects to one pixel, for `ppr`.
    pub switch_distance_1px: f32,
    /// Pixels per radian of the view `switch_distance_1px` is given for.
    pub ppr: f32,
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
    /// With a viewing budget: the most texels per world unit the closest view can show (the
    /// shared atlas is capped there), and the fraction of the artist's texels under that cap.
    pub view_texel_density: Option<f32>,
    pub view_texels: Option<f32>,
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
            view_texel_density: None,
            view_texels: None,
        }
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
    /// A level is done. `glb` is empty for level 0 (the page already has the original).
    /// Reduced levels carry no textures; their materials are the original's, by index.
    Level { info: &'a LevelInfo, glb: &'a [u8] },
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
    /// What the last cluster, HLOD or splat run keeps for selection and downloads.
    pub(crate) extra: Extra,
}

impl Demo {
    /// Loads the first `.glb` / `.gltf` (shallowest path, GLB first) among `files`; `.zip`
    /// files are expanded first. Other files resolve the glTF's external URIs.
    pub fn load(files: Vec<(String, Vec<u8>)>) -> Result<Demo, String> {
        let files = expand_zips(files)?;
        let lower = |s: &str| s.to_ascii_lowercase();
        let mut mains: Vec<usize> = (0..files.len())
            .filter(|&i| {
                let n = lower(&files[i].0);
                n.ends_with(".glb") || n.ends_with(".gltf")
            })
            .collect();
        mains.sort_by_key(|&i| {
            let n = lower(&files[i].0);
            (n.matches('/').count(), !n.ends_with(".glb"), n)
        });
        let Some(&main) = mains.first() else {
            let names: Vec<&str> = files.iter().map(|f| f.0.as_str()).take(12).collect();
            return Err(format!(
                "no .glb or .gltf found (got: {})",
                if names.is_empty() {
                    "nothing".into()
                } else {
                    names.join(", ")
                }
            ));
        };
        let main_name = files[main].0.clone();
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
        if scene.triangle_count() == 0 {
            return Err(format!("{main_name} has no triangles"));
        }
        let name = main_name
            .rsplit('/')
            .next()
            .unwrap_or(&main_name)
            .to_owned();
        Ok(Demo {
            name,
            levels: vec![(scene.clone(), original_textures(&materials))],
            scene,
            materials,
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
        self.level_glb(0)
    }

    /// A level of the last run as a self-contained GLB (for download).
    pub fn level_glb(&self, level: usize) -> Result<Vec<u8>, String> {
        let (s, t) = self
            .levels
            .get(level)
            .ok_or_else(|| format!("no level {level}; run first"))?;
        glb_bytes(s, t).map_err(|e| e.to_string())
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
        self.extra = Extra::None;
        match opts.mode {
            DemoMode::Chain => self.run_chain(opts, now, emit),
            DemoMode::Presets => crate::modes::run_presets(self, opts, now, emit),
            DemoMode::Clusters => crate::modes::run_clusters(self, opts, now, emit),
            DemoMode::Hlod => crate::modes::run_hlod(self, opts, now, emit),
            DemoMode::Splats => crate::modes::run_splats(self, opts, now, emit),
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
        let measure =
            |s: &Scene| surface_distance(&source, &s.flatten().0, MEASURE_SAMPLES).hausdorff;
        let level_info = |level, kind, s: &Scene, error: f32, bound: f32, ms| LevelInfo {
            triangles: s.triangle_count(),
            vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
            error,
            relative_error: error / diag,
            error_bound: bound,
            switch_distance_1px: error * ppr,
            ppr,
            ms,
            note: visible.map(|v| view_note(level, error, v, opts.view_distance)),
            ..LevelInfo::new(level, kind)
        };

        let original_bytes = texture_bytes(&original_textures(&self.materials));
        let mut info = level_info(0, LevelKind::Original, &self.scene, 0.0, 0.0, 0.0);
        info.texture_bytes = Some(original_bytes);
        if let Some(v) = visible {
            let dens = self.artist_density(&source);
            info.view_texel_density = Some(1.0 / v);
            info.view_texels = texels_under(&source, &dens, 1.0 / v);
        }
        emit(Event::Level {
            info: &info,
            glb: &[],
        });

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
        if shared && bake_from > 1 {
            self.shared_levels(opts, bake_from, now, emit)?;
        }
        let first_reduced = if shared { bake_from } else { 1 };
        for level in first_reduced..bake_from {
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let t0 = now();
            let ratio = step.powi((level - from_level) as i32);
            let limit = visible.filter(|_| level == 1);
            let ((next, s), error) = fit_visible(limit, |k| {
                let mut next = Vec::with_capacity(prev.len());
                let mut extended = false;
                for (mi, (mesh, error)) in prev.iter().enumerate() {
                    let target = ((from[mi] as f32 * ratio) as usize).max(1);
                    // In mesh units: the largest instance scale turns them into world units.
                    let limit = limit
                        .filter(|_| mesh_scale[mi] > 0.0)
                        .map(|v| k * v / mesh_scale[mi]);
                    let (m, e, x) = reduce_to(mesh, &base, target, limit).map_err(err)?;
                    extended |= x;
                    next.push((m, error + e));
                }
                let mut s = self.scene.clone();
                for (mi, (m, _)) in next.iter().enumerate() {
                    s.meshes[mi] = m.clone();
                }
                let error = measure(&s);
                Ok(((next, s), error, extended))
            })?;
            prev = next;
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
            let glb = glb_bytes(&s, &ExportTextures::default()).map_err(err)?;
            let ms = now() - t0;
            let mut info = level_info(level, LevelKind::Reduced, &s, error, bound, ms);
            info.bytes = Some(glb.len());
            info.texture_bytes = Some(original_bytes);
            emit(Event::Level {
                info: &info,
                glb: &glb,
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
        for level in bake_from..=levels {
            let t0 = now();
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let ratio = step.powi((level - far_from.1) as i32);
            let target = ((far_from.0 as f32 * ratio) as usize).max(4);
            let limit = visible.filter(|_| level == 1);
            // Measured before the unwrap (which only splits vertices), so a budget can size the
            // atlas for the distance the level appears at.
            let ((m, e, mesh), error) = fit_visible(limit, |k| {
                let (m, e, x) =
                    reduce_to(&geo, &far_base, target, limit.map(|v| k * v)).map_err(err)?;
                let mut mesh = m.clone();
                mesh.compute_normals(60.0);
                let error = measure(&Scene::from_mesh(mesh.clone()));
                Ok(((m, e, mesh), error, x))
            })?;
            geo = m;
            geo_error += e;

            emit(Event::Stage(&format!("LOD{level}: unwrapping UVs")));
            // With a budget: a texel per pixel where the level first shows, at its switch
            // distance or the closest view, whichever is farther (but at least about 64 × 64).
            let floor = 64.0 * (0.5 / mesh.surface_area().max(f32::MIN_POSITIVE)).sqrt();
            let density = visible.map(|v| (1.0 / error.max(v)).max(floor));
            let popts = ParamOptions {
                atlas_size: opts.atlas,
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
            let bake = bake_with_host(
                &mut host,
                &mesh,
                Mat4::IDENTITY,
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
            )
            .map_err(err)?;
            let tex = bake_textures(&bake.textures, level, emit);
            let mut s = Scene::from_mesh(mesh);
            s.materials[0].name = "baked".into();
            let glb = glb_bytes(&s, &tex).map_err(err)?;
            let ms = now() - t0;
            let mut info = level_info(level, LevelKind::Baked, &s, error, geo_error, ms);
            info.coverage = Some(bake.coverage);
            info.charts = Some(p.charts);
            info.utilization = Some(p.utilization);
            info.bytes = Some(glb.len());
            info.texture_bytes = Some(texture_bytes(&tex));
            info.atlas_width = Some(w);
            info.atlas_height = Some(h);
            emit(Event::Level {
                info: &info,
                glb: &glb,
            });
            self.levels.push((s, tex));
        }
        emit(Event::Stage("done"));
        Ok(())
    }

    /// Shared-atlas near levels (1 until `bake_from`): the whole scene as one mesh, re-unwrapped
    /// at the artist's texel density (up to `opts.shared_size` per side) and baked by exact
    /// transfer, then reduced level by level. Level 1's GLB carries the atlas.
    fn shared_levels(
        &mut self,
        opts: &LodOptions,
        bake_from: usize,
        now: &dyn Fn() -> f64,
        emit: &mut dyn FnMut(Event),
    ) -> Result<(), String> {
        let err = |e: Error| e.to_string();
        let mut t0 = now();
        emit(Event::Stage("Shared atlas: merging the scene"));
        let (flat, refs) = self.scene.flatten();
        let mut dens = self.artist_density(&flat);
        // A viewing budget caps it at a texel per pixel of the closest view.
        let cap = opts.visible_error().map(|v| 1.0 / v);
        let view_texels = cap.and_then(|c| texels_under(&flat, &dens, c));
        if let Some(c) = cap {
            dens.iter_mut().for_each(|d| *d = d.min(c));
        }
        let textured = dens.iter().any(|&d| d > 0.0);
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

        emit(Event::Stage("Shared atlas: unwrapping UVs"));
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
        emit(Event::Stage(&format!("Shared atlas: baking {w}×{h}")));
        let mut mesh = p.mesh;
        compute_tangents(&mut mesh, 0).map_err(err)?;
        // The correspondence (about 45 bytes a texel) goes as soon as the bake returns.
        let (textures, coverage) = bake_with_host(
            &mut MaterialHost::new(&self.materials, false),
            &mesh,
            Mat4::IDENTITY,
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
        )
        .map(|b| (b.textures, b.coverage))
        .map_err(err)?;
        let mut tex = bake_textures(&textures, 1, emit);
        drop(textures);
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

        let diag = self.scene.bounds().diagonal().max(f32::MIN_POSITIVE);
        let input_tris = self.scene.triangle_count();
        let base = opts.simplify_options();
        let step = opts.step.clamp(0.05, 0.95);
        let ppr = opts.ppr();
        let visible = opts.visible_error();
        let mut error_bound = 0.0;
        // Steps compound from the original, or with a budget from the first level.
        let mut from = (input_tris, 0);
        for level in 1..bake_from {
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let ratio = step.powi((level - from.1) as i32);
            let target = ((from.0 as f32 * ratio) as usize).max(4);
            let limit = visible.filter(|_| level == 1);
            let ((m, e, s), error) = fit_visible(limit, |k| {
                let (m, e, x) =
                    reduce_to(&mesh, &base, target, limit.map(|v| k * v)).map_err(err)?;
                let mut t = m.clone();
                compute_tangents(&mut t, 0).map_err(err)?;
                let mut s = Scene::from_mesh(t);
                s.materials[0].name = "shared".into();
                let error = surface_distance(&source, &s.flatten().0, MEASURE_SAMPLES).hausdorff;
                Ok(((m, e, s), error, x))
            })?;
            mesh = m;
            error_bound += e;
            if limit.is_some() {
                from = (mesh.triangle_count(), 1);
            }
            let glb = if level == 1 {
                glb_bytes(&s, &tex)
            } else {
                glb_bytes(&s, &ExportTextures::default())
            }
            .map_err(err)?;
            let ms = now() - t0;
            let info = LevelInfo {
                triangles: s.triangle_count(),
                vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
                error,
                relative_error: error / diag,
                error_bound,
                switch_distance_1px: error * ppr,
                ppr,
                ms,
                note: visible.map(|v| view_note(level, error, v, opts.view_distance)),
                view_texel_density: cap,
                view_texels,
                coverage: Some(coverage),
                charts: Some(p.charts),
                utilization: Some(p.utilization),
                bytes: Some(glb.len()),
                texture_bytes: Some(tex_bytes),
                atlas_width: Some(w),
                atlas_height: Some(h),
                density_scale: Some(p.density_scale),
                shared_textures: Some(1),
                ..LevelInfo::new(level, LevelKind::Shared)
            };
            emit(Event::Level {
                info: &info,
                glb: &glb,
            });
            self.levels.push((s, tex.clone()));
            t0 = now();
        }
        Ok(())
    }

    /// The artist's texel density per triangle of the flattened scene, from the largest
    /// texture of its material (texels per world unit; 0 where untextured).
    fn artist_density(&self, flat: &Mesh) -> Vec<f32> {
        let largest: Vec<Option<(u32, u32, u32)>> = (0..self.scene.materials.len())
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
            .collect();
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
/// level that measures within it.
fn fit_visible<T>(
    limit: Option<f32>,
    mut build: impl FnMut(f32) -> Result<(T, f32, bool), String>,
) -> Result<(T, f32), String> {
    let (level, error, extended) = build(1.0)?;
    let v = match limit {
        Some(v) if extended && error > 1.25 * v => 1.25 * v,
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

/// Emits a bake's textures for `level` and returns them for export (slot 0).
fn bake_textures(
    textures: &ChannelImages,
    level: usize,
    emit: &mut dyn FnMut(Event),
) -> ExportTextures {
    let mut tex = ExportTextures::default();
    for (name, ch) in textures {
        let rgba = to_rgba8(&ch.image, ch.encoding);
        emit(Event::Texture {
            level,
            channel: name,
            width: ch.image.width,
            height: ch.image.height,
            rgba: &rgba,
        });
        match name.as_str() {
            channels::BASE_COLOR => {
                tex.base_color
                    .insert(0, Texture::from_image(&ch.image, true));
            }
            channels::NORMAL => {
                tex.normal
                    .insert(0, Texture::from_signed_normals(&ch.image));
            }
            channels::ORM => {
                tex.metallic_roughness
                    .insert(0, Texture::from_image(&ch.image, false));
            }
            _ => {}
        }
    }
    tex
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

pub(crate) fn to_rgba8(img: &Image, encoding: ChannelEncoding) -> Vec<u8> {
    let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut out = Vec::with_capacity(img.pixels.len() * 4);
    for p in &img.pixels {
        let rgb = match encoding {
            ChannelEncoding::Color => [
                linear_to_srgb(p.x),
                linear_to_srgb(p.y),
                linear_to_srgb(p.z),
            ],
            ChannelEncoding::NormalTangent | ChannelEncoding::NormalWorld => {
                [p.x * 0.5 + 0.5, p.y * 0.5 + 0.5, p.z * 0.5 + 0.5]
            }
            ChannelEncoding::Data => [p.x, p.y, p.z],
        };
        out.extend_from_slice(&[q(rgb[0]), q(rgb[1]), q(rgb[2]), 255]);
    }
    out
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
        // Unbaked child proxies (`hlod::PROXY_MATERIAL`) and unknown slots: plain grey.
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
                    ))
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
