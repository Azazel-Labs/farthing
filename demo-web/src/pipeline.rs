//! The demo pipeline, independent of the browser so it can be tested natively.
//!
//! * [`Demo::load`]: glTF / GLB from dropped files or a zip (external buffers and textures
//!   resolved by relative path).
//! * [`Demo::run`]: a LOD chain. Near levels are reduced per mesh and keep the original
//!   materials; far levels (from `bake_from`) are geometry-only proxies of the whole scene with
//!   fresh UVs and base color + normal maps baked from the original. Every finished level and
//!   baked texture is reported through an [`Event`] as soon as it exists.

use farthing::bake::{bake_with_host, compute_tangents, BakeOptions};
use farthing::host::{
    channels, BakedMaterialDesc, ChannelEncoding, ChannelImage, ChannelImages, MaterialClass,
    MaterialInfo, RenderHost, SurfaceBakeRequest, ViewImages, ViewRenderRequest,
};
use farthing::image::{linear_to_srgb, Image};
use farthing::metrics::surface_distance;
use farthing::{Error, MaterialId, Mesh, Scene};
use farthing::io::{glb_bytes, load_gltf_with, ExportTextures, GltfMaterial};
use farthing::reduce::{repair, simplify, Quality, SeamMode, SimplifyOptions, Symmetry};
use farthing::uv::{parameterize, ParamOptions};
use glam::{Mat4, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// Surface samples per mesh when measuring a level against LOD0.
const MEASURE_SAMPLES: usize = 20_000;

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
    /// Planar charts instead of LSCM for baked levels.
    pub planar_uv: bool,
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
        }
    }
}

impl LodOptions {
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
    Baked,
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
    /// Camera distance at which `error` projects to one pixel (1080p, 60° vertical FOV).
    pub switch_distance_1px: f32,
    /// Time to build the level (measuring it not included).
    pub ms: f64,
    /// Baked levels: fraction of texels that found the source surface.
    pub coverage: Option<f32>,
    pub charts: Option<usize>,
    pub utilization: Option<f32>,
}

pub enum Event<'a> {
    /// What the pipeline is doing now, for a status line.
    Stage(&'a str),
    /// A level is done. `glb` is empty for level 0 (the page already has the original).
    /// Reduced levels carry no textures; their materials are the original's, by index.
    Level { info: &'a LevelInfo, glb: &'a [u8] },
    /// A baked texture (sRGB base color, or tangent-space normal as 0..1), RGBA8 rows.
    Texture {
        level: usize,
        channel: &'a str,
        width: u32,
        height: u32,
        rgba: &'a [u8],
    },
}

pub struct Demo {
    pub name: String,
    pub scene: Scene,
    pub materials: Vec<GltfMaterial>,
    textures: ExportTextures,
    /// Results of the last run, by level (0 = original).
    levels: Vec<(Scene, ExportTextures)>,
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
                if names.is_empty() { "nothing".into() } else { names.join(", ") }
            ));
        };
        let main_name = files[main].0.clone();
        let dir = main_name.rfind('/').map_or("", |i| &main_name[..=i]).to_owned();
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
        let textures = original_textures(&materials);
        let name = main_name.rsplit('/').next().unwrap_or(&main_name).to_owned();
        Ok(Demo {
            name,
            levels: vec![(scene.clone(), textures.clone())],
            scene,
            materials,
            textures,
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

    /// The original as GLB with base color textures and factors (what farthing reads of the
    /// materials), so every level is shown with the same material model.
    pub fn original_glb(&self) -> Result<Vec<u8>, String> {
        glb_bytes(&self.scene, &self.textures).map_err(|e| e.to_string())
    }

    /// A level of the last run as a self-contained GLB (for download).
    pub fn level_glb(&self, level: usize) -> Result<Vec<u8>, String> {
        let (s, t) = self
            .levels
            .get(level)
            .ok_or_else(|| format!("no level {level}; run first"))?;
        glb_bytes(s, t).map_err(|e| e.to_string())
    }

    /// Builds the chain, reporting each level as it finishes. `now` returns milliseconds.
    pub fn run(
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
        let switch = |e: f32| e * 1080.0 / (2.0 * 30f32.to_radians().tan());
        let source = self.scene.flatten().0;
        let level_info = |level, kind, s: &Scene, bound: f32, ms| {
            let error = if level == 0 {
                0.0
            } else {
                surface_distance(&source, &s.flatten().0, MEASURE_SAMPLES).hausdorff
            };
            LevelInfo {
                level,
                kind,
                triangles: s.triangle_count(),
                vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
                error,
                relative_error: error / diag,
                error_bound: bound,
                switch_distance_1px: switch(error),
                ms,
                coverage: None,
                charts: None,
                utilization: None,
            }
        };

        self.levels.truncate(1);
        let info = level_info(0, LevelKind::Original, &self.scene, 0.0, 0.0);
        emit(Event::Level { info: &info, glb: &[] });

        // Near levels: each mesh simplified from its previous level (as `lod_chain` does),
        // keeping UVs and material slots.
        let mut prev: Vec<(Mesh, f32)> =
            self.scene.meshes.iter().map(|m| (m.clone(), 0.0)).collect();
        let mut mesh_scale = vec![0.0f32; self.scene.meshes.len()];
        for inst in &self.scene.instances {
            let m = glam::Mat3::from_mat4(inst.transform);
            let s = m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length());
            mesh_scale[inst.mesh] = mesh_scale[inst.mesh].max(s);
        }
        for level in 1..bake_from {
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let t0 = now();
            let ratio = step.powi(level as i32);
            let mut next = Vec::with_capacity(prev.len());
            for (mi, (mesh, error)) in prev.iter().enumerate() {
                let target = ((self.scene.meshes[mi].triangle_count() as f32 * ratio) as usize).max(1);
                let r = simplify(
                    mesh,
                    &SimplifyOptions {
                        target_triangles: Some(target),
                        ..base.clone()
                    },
                )
                .map_err(err)?;
                next.push((r.mesh, error + r.error));
            }
            prev = next;
            let mut s = self.scene.clone();
            for (mi, (m, _)) in prev.iter().enumerate() {
                s.meshes[mi] = m.clone();
            }
            // Mesh errors are in mesh units; the bound is the largest in world units.
            let bound = prev
                .iter()
                .enumerate()
                .map(|(mi, p)| p.1 * mesh_scale[mi])
                .fold(0.0, f32::max);
            let glb = glb_bytes(&s, &ExportTextures::default()).map_err(err)?;
            let info = level_info(level, LevelKind::Reduced, &s, bound, now() - t0);
            emit(Event::Level { info: &info, glb: &glb });
            self.levels.push((s, self.textures.clone()));
        }
        if bake_from > levels {
            emit(Event::Stage("done"));
            return Ok(());
        }

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
        let mut host = MaterialHost {
            materials: &self.materials,
        };
        for level in bake_from..=levels {
            let t0 = now();
            emit(Event::Stage(&format!("LOD{level}: reducing")));
            let target = ((input_tris as f32 * step.powi(level as i32)) as usize).max(4);
            let r = simplify(
                &geo,
                &SimplifyOptions {
                    target_triangles: Some(target),
                    seams: SeamMode::Permissive,
                    ..base.clone()
                },
            )
            .map_err(err)?;
            geo = r.mesh;
            geo_error += r.error;
            let mut mesh = geo.clone();
            mesh.compute_normals(60.0);

            emit(Event::Stage(&format!("LOD{level}: unwrapping UVs")));
            let popts = ParamOptions {
                atlas_size: opts.atlas,
                padding: 2,
                ..if opts.planar_uv {
                    ParamOptions::planar()
                } else {
                    ParamOptions::default()
                }
            };
            let p = parameterize(&mesh, &popts).map_err(err)?;
            let mut mesh = p.mesh;
            compute_tangents(&mut mesh, 0).map_err(err)?;

            emit(Event::Stage(&format!("LOD{level}: baking {0}×{0}", opts.atlas)));
            let bake = bake_with_host(
                &mut host,
                &mesh,
                Mat4::IDENTITY,
                &self.scene,
                &BakeOptions {
                    width: opts.atlas,
                    height: opts.atlas,
                    supersample: opts.supersample.clamp(1, 4),
                    gutter: 4,
                    channels: vec![channels::BASE_COLOR.into(), channels::NORMAL.into()],
                    ..Default::default()
                },
            )
            .map_err(err)?;
            let mut tex = ExportTextures::default();
            for (name, ch) in &bake.textures {
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
                        tex.base_color.insert(0, ch.image.clone());
                    }
                    channels::NORMAL => {
                        tex.normal.insert(0, ch.image.clone());
                    }
                    _ => {}
                }
            }
            let mut s = Scene::from_mesh(mesh);
            s.materials[0].name = "baked".into();
            let glb = glb_bytes(&s, &tex).map_err(err)?;
            let mut info = level_info(level, LevelKind::Baked, &s, geo_error, now() - t0);
            info.coverage = Some(bake.coverage);
            info.charts = Some(p.charts);
            info.utilization = Some(p.utilization);
            emit(Event::Level { info: &info, glb: &glb });
            self.levels.push((s, tex));
        }
        emit(Event::Stage("done"));
        Ok(())
    }
}

fn expand_zips(files: Vec<(String, Vec<u8>)>) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    for (name, data) in files {
        let name = name.replace('\\', "/");
        if !name.to_ascii_lowercase().ends_with(".zip") {
            out.push((name, data));
            continue;
        }
        let mut zip = zip::ZipArchive::new(Cursor::new(data)).map_err(|e| format!("{name}: {e}"))?;
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
            f.read_to_end(&mut buf).map_err(|e| format!("{name}/{n}: {e}"))?;
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

fn original_textures(materials: &[GltfMaterial]) -> ExportTextures {
    let mut t = ExportTextures::default();
    for (i, m) in materials.iter().enumerate() {
        let i = i as u32;
        t.base_color_factor.insert(i, m.base_color_factor);
        if let Some((img, _)) = &m.base_color_texture {
            t.base_color.insert(i, img.clone());
        }
        if let MaterialClass::Masked { cutoff } = m.info.class {
            t.alpha_cutoff.insert(i, cutoff);
        }
    }
    t
}

fn to_rgba8(img: &Image, encoding: ChannelEncoding) -> Vec<u8> {
    let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut out = Vec::with_capacity(img.pixels.len() * 4);
    for p in &img.pixels {
        let rgb = match encoding {
            ChannelEncoding::Color => [linear_to_srgb(p.x), linear_to_srgb(p.y), linear_to_srgb(p.z)],
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
/// color, and the source's interpolated normal (source normal maps are not applied).
struct MaterialHost<'a> {
    materials: &'a [GltfMaterial],
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
        for (i, s) in req.samples.iter().enumerate() {
            let Some(s) = s else { continue };
            let mut c = s.color;
            if let Some(m) = self.materials.get(s.material.0 as usize) {
                c *= m.base_color_factor;
                if let Some((tex, set)) = &m.base_color_texture {
                    c *= tex.sample_bilinear(s.uv[(*set as usize).min(1)], true);
                }
                if m.info.class == MaterialClass::Opaque {
                    c.w = 1.0;
                }
            }
            color[i] = c;
            normal[i] = s.normal.normalize_or_zero().extend(1.0);
        }
        let image = |pixels| Image {
            width: req.width,
            height: req.height,
            pixels,
        };
        let mut out = ChannelImages::new();
        for ch in &req.channels {
            let (img, encoding) = match ch.as_str() {
                channels::BASE_COLOR => (image(color.clone()), ChannelEncoding::Color),
                channels::NORMAL => (image(normal.clone()), ChannelEncoding::NormalWorld),
                _ => return Err(Error::Unsupported("channel (demo host: base_color, normal)")),
            };
            out.insert(ch.clone(), ChannelImage { image: img, encoding });
        }
        Ok(out)
    }

    fn render_views(&mut self, _req: &ViewRenderRequest) -> farthing::Result<Vec<ViewImages>> {
        Err(Error::Unsupported("render_views"))
    }

    fn create_material(&mut self, _desc: &BakedMaterialDesc) -> farthing::Result<MaterialId> {
        Err(Error::Unsupported("create_material"))
    }
}
