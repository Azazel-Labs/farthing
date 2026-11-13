use crate::bake::correspondence::{build_correspondence, Correspondence};
use crate::bake::tangents::compute_tangents;
use crate::host::{
    channels, ChannelEncoding, ChannelImage, ChannelImages, RenderHost, SurfaceBakeRequest,
    SurfaceSample,
};
use crate::image::Image;
use crate::{Error, Mesh, Result, Scene};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Channel produced by farthing itself: signed source distance along the destination normal.
pub const DISPLACEMENT: &str = "displacement";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakeOptions {
    pub width: u32,
    pub height: u32,
    /// Samples per texel along each axis (1–4).
    pub supersample: u32,
    /// Ray search distance; None = 3% of the source bounding-box diagonal.
    pub max_distance: Option<f32>,
    /// Destination UV set to bake into.
    pub uv_set: usize,
    /// Gutter dilation in pixels.
    pub gutter: u32,
    pub channels: Vec<String>,
}

impl Default for BakeOptions {
    fn default() -> Self {
        BakeOptions {
            width: 1024,
            height: 1024,
            supersample: 2,
            max_distance: None,
            uv_set: 0,
            gutter: 8,
            channels: vec![channels::BASE_COLOR.into(), channels::NORMAL.into()],
        }
    }
}

#[derive(Clone, Debug)]
pub struct BakeOutput {
    /// Final-resolution textures, dilated. Normals are tangent-space (`NormalTangent`).
    pub textures: ChannelImages,
    /// Fraction of texels with a source point.
    pub coverage: f32,
    /// The correspondence used (keep it to re-bake channels without re-tracing).
    pub correspondence: Correspondence,
}

/// Host-flattened textures: per channel, per (instance, material slot) an image in the source
/// mesh's UV set 0. Use instance `u32::MAX` for a texture shared by every instance of a slot.
#[derive(Clone, Debug, Default)]
pub struct FlattenedTextures {
    pub channels: BTreeMap<String, (ChannelEncoding, HashMap<(u32, u32), Image>)>,
}

impl FlattenedTextures {
    pub fn insert(
        &mut self,
        channel: &str,
        encoding: ChannelEncoding,
        instance: Option<u32>,
        slot: u32,
        image: Image,
    ) {
        self.channels
            .entry(channel.to_owned())
            .or_insert_with(|| (encoding, HashMap::new()))
            .1
            .insert((instance.unwrap_or(u32::MAX), slot), image);
    }
    pub fn get(&self, channel: &str, instance: u32, slot: u32) -> Option<&Image> {
        let (_, m) = self.channels.get(channel)?;
        m.get(&(instance, slot))
            .or_else(|| m.get(&(u32::MAX, slot)))
    }
}

struct Prepared {
    dest: Mesh,
    corr: Correspondence,
}

fn prepare(
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
) -> Result<Prepared> {
    let mut dest = dest.clone();
    if dest.uvs.len() <= opts.uv_set {
        return Err(Error::InvalidMesh(
            "destination mesh needs UVs; run crate::uv::parameterize first".into(),
        ));
    }
    if dest.normals.is_none() {
        dest.compute_normals(180.0);
    }
    compute_tangents(&mut dest, opts.uv_set)?;
    let maxd = opts
        .max_distance
        .unwrap_or_else(|| source.bounds().diagonal() * 0.03);
    let corr = build_correspondence(
        &dest,
        dest_to_world,
        source,
        opts.width,
        opts.height,
        opts.supersample,
        opts.uv_set,
        maxd,
    )?;
    Ok(Prepared { dest, corr })
}

/// Destination tangent frame (world space) at a correspondence sample.
fn dest_frame(dest: &Mesh, m: Mat4, tri: u32, b: Vec3) -> (Vec3, Vec3, Vec3) {
    let [i0, i1, i2] = dest.triangle(tri as usize).map(|v| v as usize);
    let nmat = Mat3::from_mat4(m).inverse().transpose();
    let m3 = Mat3::from_mat4(m);
    let ns = dest.normals.as_ref().unwrap();
    let ts = dest.tangents.as_ref().unwrap();
    let n = (nmat * (ns[i0] * b.x + ns[i1] * b.y + ns[i2] * b.z)).normalize_or_zero();
    let t4 = ts[i0] * b.x + ts[i1] * b.y + ts[i2] * b.z;
    let t = (m3 * t4.truncate()).normalize_or_zero();
    let t = (t - n * n.dot(t)).normalize_or_zero(); // Gram-Schmidt
    let sign = if ts[i0].w < 0.0 { -1.0 } else { 1.0 };
    let bt = n.cross(t) * sign;
    (t, bt, n)
}

/// Turns per-sample channel images into final textures: normal conversion, supersample
/// resolve, dilation.
fn finish(
    mut raw: ChannelImages,
    prep: &Prepared,
    dest_to_world: Mat4,
    opts: &BakeOptions,
) -> ChannelImages {
    let corr = &prep.corr;
    for img in raw.values_mut() {
        // Mark coverage in alpha from the correspondence (hosts may not).
        for (i, e) in corr.entries.iter().enumerate() {
            if e.is_none() {
                img.image.pixels[i] = Vec4::ZERO;
            } else if img.image.pixels[i].w <= 0.0 {
                img.image.pixels[i].w = 1.0;
            }
        }
        if img.encoding == ChannelEncoding::NormalWorld {
            for (i, e) in corr.entries.iter().enumerate() {
                if let Some(e) = e {
                    let nw = img.image.pixels[i].truncate().normalize_or_zero();
                    let (t, b, n) = dest_frame(
                        &prep.dest,
                        dest_to_world,
                        e.dest_triangle,
                        e.dest_barycentrics,
                    );
                    let ts = Vec3::new(nw.dot(t), nw.dot(b), nw.dot(n)).normalize_or_zero();
                    img.image.pixels[i] = ts.extend(1.0);
                }
            }
            img.encoding = ChannelEncoding::NormalTangent;
        }
    }
    raw.into_iter()
        .map(|(name, mut ci)| {
            let mut img = ci.image.downsample_coverage(corr.supersample);
            if ci.encoding == ChannelEncoding::NormalTangent {
                for p in &mut img.pixels {
                    if p.w > 0.0 {
                        *p = p.truncate().normalize_or_zero().extend(p.w);
                    }
                }
            }
            img.dilate(opts.gutter);
            // Covered texels get full alpha; alpha is coverage, not opacity.
            for p in &mut img.pixels {
                if p.w > 0.0 {
                    p.w = 1.0;
                }
            }
            ci.image = img;
            (name, ci)
        })
        .collect()
}

fn displacement_channel(corr: &Correspondence) -> ChannelImage {
    let mut img = Image::new(corr.width, corr.height, Vec4::ZERO);
    for (i, e) in corr.entries.iter().enumerate() {
        if let Some(e) = e {
            img.pixels[i] = Vec4::new(e.distance, e.distance, e.distance, 1.0);
        }
    }
    ChannelImage {
        image: img,
        encoding: ChannelEncoding::Data,
    }
}

fn interpolate_sample(source: &Scene, e: &crate::bake::CorrespondenceEntry) -> SurfaceSample {
    let inst = &source.instances[e.source.instance as usize];
    let mesh = &source.meshes[inst.mesh];
    let [i0, i1, i2] = mesh
        .triangle(e.source.triangle as usize)
        .map(|v| v as usize);
    let b = e.barycentrics;
    let lerp3 = |a: &[Vec3]| a[i0] * b.x + a[i1] * b.y + a[i2] * b.z;
    let m = inst.transform;
    let nmat = Mat3::from_mat4(m).inverse().transpose();
    let position = m.transform_point3(lerp3(&mesh.positions));
    let normal = match &mesh.normals {
        Some(n) => (nmat * lerp3(n)).normalize_or_zero(),
        None => {
            let [a, bb, c] = mesh.triangle_positions(e.source.triangle as usize);
            (nmat * (bb - a).cross(c - a)).normalize_or_zero()
        }
    };
    let tangent = mesh
        .tangents
        .as_ref()
        .map_or(Vec4::new(1.0, 0.0, 0.0, 1.0), |t| {
            let v = t[i0] * b.x + t[i1] * b.y + t[i2] * b.z;
            (Mat3::from_mat4(m) * v.truncate())
                .normalize_or_zero()
                .extend(if t[i0].w < 0.0 { -1.0 } else { 1.0 })
        });
    let uv = |set: usize| {
        mesh.uvs
            .get(set)
            .map_or(Vec2::ZERO, |u| u[i0] * b.x + u[i1] * b.y + u[i2] * b.z)
    };
    let color = mesh
        .colors
        .as_ref()
        .map_or(Vec4::ONE, |c| c[i0] * b.x + c[i1] * b.y + c[i2] * b.z);
    let slot = mesh.material_ids[e.source.triangle as usize];
    SurfaceSample {
        instance: e.source.instance,
        material: source.material_id(slot),
        position,
        normal,
        tangent,
        uv: [uv(0), uv(1)],
        color,
    }
}

/// A bake split into two phases, for engines that shade asynchronously or across an FFI
/// boundary: [`prepare_bake`] → shade [`PreparedBake::request`] → [`finish_bake`].
pub struct PreparedBake {
    prep: Prepared,
    dest_to_world: Mat4,
    opts: BakeOptions,
    /// What the engine must shade (supersampled grid, one optional sample per texel).
    pub request: SurfaceBakeRequest,
}

impl PreparedBake {
    /// Correspondence entries, parallel to `request.samples`.
    pub fn correspondence_entries(&self) -> &[Option<crate::bake::CorrespondenceEntry>] {
        &self.prep.corr.entries
    }
    /// Declares which channels will be submitted (for callers that learn this late, e.g. FFI)
    /// and the gutter size.
    pub fn set_channels(&mut self, channels: Vec<String>, gutter: u32) {
        self.request.channels = channels
            .iter()
            .filter(|c| c.as_str() != DISPLACEMENT)
            .cloned()
            .collect();
        self.opts.channels = channels;
        self.opts.gutter = gutter;
    }
}

/// Phase 1: UV rasterization, correspondence and the surface-sample request.
pub fn prepare_bake(
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
) -> Result<PreparedBake> {
    let prep = prepare(dest, dest_to_world, source, opts)?;
    let request = SurfaceBakeRequest {
        width: prep.corr.width,
        height: prep.corr.height,
        samples: prep
            .corr
            .entries
            .iter()
            .map(|e| e.as_ref().map(|e| interpolate_sample(source, e)))
            .collect(),
        channels: opts
            .channels
            .iter()
            .filter(|c| c.as_str() != DISPLACEMENT)
            .cloned()
            .collect(),
    };
    Ok(PreparedBake {
        prep,
        dest_to_world,
        opts: opts.clone(),
        request,
    })
}

/// Phase 2: takes the shaded channels (request-sized) and produces final textures.
pub fn finish_bake(prepared: PreparedBake, mut shaded: ChannelImages) -> Result<BakeOutput> {
    let req = &prepared.request;
    for c in &req.channels {
        let img = shaded
            .get(c)
            .ok_or_else(|| Error::Host(format!("missing channel `{c}`")))?;
        if (img.image.width, img.image.height) != (req.width, req.height) {
            return Err(Error::Host(format!("channel `{c}` has the wrong size")));
        }
    }
    if prepared.opts.channels.iter().any(|c| c == DISPLACEMENT) {
        shaded.insert(
            DISPLACEMENT.into(),
            displacement_channel(&prepared.prep.corr),
        );
    }
    let coverage = prepared.prep.corr.coverage();
    let textures = finish(
        shaded,
        &prepared.prep,
        prepared.dest_to_world,
        &prepared.opts,
    );
    Ok(BakeOutput {
        textures,
        coverage,
        correspondence: prepared.prep.corr,
    })
}

/// Point-evaluation bake: the host evaluates its real materials at the source points.
pub fn bake_with_host(
    host: &mut dyn RenderHost,
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
) -> Result<BakeOutput> {
    let prepared = prepare_bake(dest, dest_to_world, source, opts)?;
    let shaded = if prepared.request.channels.is_empty() {
        ChannelImages::new()
    } else {
        host.bake_surface(&prepared.request)?
    };
    finish_bake(prepared, shaded)
}

/// Flattened-texture bake: moves texels from host-flattened textures (one per source
/// mesh/material, in source UV set 0) without knowing what the channels mean.
pub fn bake_flattened(
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    flattened: &FlattenedTextures,
    opts: &BakeOptions,
) -> Result<BakeOutput> {
    let prep = prepare(dest, dest_to_world, source, opts)?;
    let mut raw = ChannelImages::new();
    for name in &opts.channels {
        if name == DISPLACEMENT {
            raw.insert(DISPLACEMENT.into(), displacement_channel(&prep.corr));
            continue;
        }
        let Some((encoding, _)) = flattened.channels.get(name) else {
            return Err(Error::InvalidInput(format!(
                "no flattened textures for channel `{name}`"
            )));
        };
        let mut img = Image::new(prep.corr.width, prep.corr.height, Vec4::ZERO);
        for (i, e) in prep.corr.entries.iter().enumerate() {
            let Some(e) = e else { continue };
            let inst = &source.instances[e.source.instance as usize];
            let mesh = &source.meshes[inst.mesh];
            let slot = mesh.material_ids[e.source.triangle as usize];
            if let Some(tex) = flattened.get(name, e.source.instance, slot) {
                let [i0, i1, i2] = mesh
                    .triangle(e.source.triangle as usize)
                    .map(|v| v as usize);
                let b = e.barycentrics;
                let uv = mesh
                    .uvs
                    .first()
                    .map_or(Vec2::ZERO, |u| u[i0] * b.x + u[i1] * b.y + u[i2] * b.z);
                let mut v = tex.sample_bilinear(uv, true);
                v.w = 1.0;
                img.pixels[i] = v;
            }
        }
        // Flattened normals are tangent-space in the *source* frame; re-express them via world space.
        let encoding = if *encoding == ChannelEncoding::NormalTangent {
            for (i, e) in prep.corr.entries.iter().enumerate() {
                let Some(e) = e else { continue };
                let s = interpolate_sample(source, e);
                let n = s.normal;
                let t =
                    (s.tangent.truncate() - n * n.dot(s.tangent.truncate())).normalize_or_zero();
                let b = n.cross(t) * s.tangent.w;
                let ts = img.pixels[i].truncate();
                img.pixels[i] = (t * ts.x + b * ts.y + n * ts.z)
                    .normalize_or_zero()
                    .extend(1.0);
            }
            ChannelEncoding::NormalWorld
        } else {
            *encoding
        };
        raw.insert(
            name.clone(),
            ChannelImage {
                image: img,
                encoding,
            },
        );
    }
    let coverage = prep.corr.coverage();
    let textures = finish(raw, &prep, dest_to_world, opts);
    Ok(BakeOutput {
        textures,
        coverage,
        correspondence: prep.corr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{BakedMaterialDesc, MaterialInfo, ViewImages, ViewRenderRequest};
    use crate::MaterialId;

    /// Test host: a "procedural material" that colors by world position and returns the
    /// geometric normal — farthing never sees this function.
    struct ProceduralHost;
    impl RenderHost for ProceduralHost {
        fn material_info(&self, _m: MaterialId) -> MaterialInfo {
            MaterialInfo::default()
        }
        fn bake_surface(&mut self, req: &SurfaceBakeRequest) -> Result<ChannelImages> {
            let mut out = ChannelImages::new();
            for c in &req.channels {
                let mut img = Image::new(req.width, req.height, Vec4::ZERO);
                for (i, s) in req.samples.iter().enumerate() {
                    if let Some(s) = s {
                        img.pixels[i] = if c == channels::NORMAL {
                            s.normal.extend(1.0)
                        } else {
                            (s.position * 0.5 + 0.5)
                                .clamp(Vec3::ZERO, Vec3::ONE)
                                .extend(1.0)
                        };
                    }
                }
                let encoding = if c == channels::NORMAL {
                    ChannelEncoding::NormalWorld
                } else {
                    ChannelEncoding::Color
                };
                out.insert(
                    c.clone(),
                    ChannelImage {
                        image: img,
                        encoding,
                    },
                );
            }
            Ok(out)
        }
        fn render_views(&mut self, _r: &ViewRenderRequest) -> Result<Vec<ViewImages>> {
            Err(Error::Unsupported("render_views"))
        }
        fn create_material(&mut self, _d: &BakedMaterialDesc) -> Result<MaterialId> {
            Ok(MaterialId(99))
        }
    }

    fn quad(z: f32) -> Mesh {
        let mut m = Mesh::new(
            vec![
                Vec3::new(-1.0, -1.0, z),
                Vec3::new(1.0, -1.0, z),
                Vec3::new(1.0, 1.0, z),
                Vec3::new(-1.0, 1.0, z),
            ],
            vec![0, 1, 2, 0, 2, 3],
        );
        m.normals = Some(vec![Vec3::Z; 4]);
        m.uvs = vec![vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ]];
        m
    }

    #[test]
    fn point_eval_bake_matches_procedural_color() {
        let source = Scene::from_mesh(quad(0.0));
        let dest = quad(0.01);
        let opts = BakeOptions {
            width: 32,
            height: 32,
            supersample: 1,
            max_distance: Some(0.1),
            gutter: 2,
            channels: vec!["base_color".into(), "normal".into(), DISPLACEMENT.into()],
            ..Default::default()
        };
        let out =
            bake_with_host(&mut ProceduralHost, &dest, Mat4::IDENTITY, &source, &opts).unwrap();
        assert!(out.coverage > 0.95, "coverage {}", out.coverage);
        let bc = &out.textures["base_color"].image;
        // Texel (x, y) ↔ uv ((x+.5)/32, (y+.5)/32) ↔ world x = 2u-1, y = 1-2v.
        let p = bc.get(24, 8);
        let wx = 2.0 * (24.5 / 32.0) - 1.0;
        let wy = 1.0 - 2.0 * (8.5 / 32.0);
        assert!(
            (p.x - (wx * 0.5 + 0.5)).abs() < 0.03 && (p.y - (wy * 0.5 + 0.5)).abs() < 0.03,
            "{p:?}"
        );
        // Flat source under flat dest → tangent-space normal is +Z.
        let n = out.textures["normal"].image.get(16, 16);
        assert!((n.truncate() - Vec3::Z).length() < 1e-3, "{n:?}");
        assert_eq!(
            out.textures["normal"].encoding,
            ChannelEncoding::NormalTangent
        );
        // Source is 0.01 below the dest.
        let d = out.textures[DISPLACEMENT].image.get(16, 16).x;
        assert!((d + 0.01).abs() < 1e-3, "{d}");
    }

    #[test]
    fn flattened_bake_moves_texels() {
        let source = Scene::from_mesh(quad(0.0));
        let mut tex = Image::new(4, 4, Vec4::ZERO);
        for y in 0..4 {
            for x in 0..4 {
                tex.set(x, y, Vec4::new(x as f32 / 3.0, y as f32 / 3.0, 0.0, 1.0));
            }
        }
        let mut flat = FlattenedTextures::default();
        flat.insert("base_color", ChannelEncoding::Color, None, 0, tex);
        let mut dest = quad(0.0);
        // Destination UVs rotated 180°: texel content must follow the surface, not the UVs.
        dest.uvs[0] = dest.uvs[0].iter().map(|u| Vec2::ONE - *u).collect();
        let opts = BakeOptions {
            width: 4,
            height: 4,
            supersample: 1,
            max_distance: Some(0.1),
            gutter: 0,
            channels: vec!["base_color".into()],
            ..Default::default()
        };
        let out = bake_flattened(&dest, Mat4::IDENTITY, &source, &flat, &opts).unwrap();
        let img = &out.textures["base_color"].image;
        assert!(
            (img.get(0, 0).x - 1.0).abs() < 0.05 && (img.get(3, 3).x).abs() < 0.05,
            "{:?} {:?}",
            img.get(0, 0),
            img.get(3, 3)
        );
    }
}
