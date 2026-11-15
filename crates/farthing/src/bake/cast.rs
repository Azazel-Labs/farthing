use crate::bake::correspondence::{Correspondence, Tracer};
use crate::bake::tangents::compute_tangents;
use crate::host::{
    ChannelEncoding, ChannelImage, ChannelImages, RenderHost, SurfaceBakeRequest, SurfaceSample,
    channels,
};
use crate::image::{Image, downsample_coverage_rows};
use crate::{Error, Mesh, Result, Scene};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Channel produced by farthing itself: signed source distance along the destination normal.
pub const DISPLACEMENT: &str = "displacement";

/// [`bake_with_host`] traces and shades at most this many samples at a time: the request
/// (about 100 bytes a sample), the correspondence entries (about 45) and the host's output
/// for them stay a few megabytes however large the atlas.
const BAND_SAMPLES: usize = 1 << 17;

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
    /// The destination is a copy of the source with new UVs (say [`Scene::flatten`]'s mesh,
    /// re-unwrapped with [`crate::uv::parameterize`]): per destination triangle, the source
    /// triangle it copies, as `flatten` returns them. Texels then read the source at the same
    /// surface point instead of casting rays: an exact transfer, with `max_distance` unused.
    #[serde(skip)]
    pub source_triangles: Option<Vec<crate::scene::TriRef>>,
    /// Return the correspondence's entries in [`BakeOutput::correspondence`] (about 45 bytes
    /// a sample, held for the whole atlas), to re-bake channels later without tracing again.
    /// Without it the bake only ever holds a band of them, and the output's correspondence
    /// has its size but no entries.
    #[serde(default = "keep_correspondence")]
    pub keep_correspondence: bool,
}

fn keep_correspondence() -> bool {
    true
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
            source_triangles: None,
            keep_correspondence: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BakeOutput {
    /// Final-resolution textures, dilated. Normals are tangent-space (`NormalTangent`).
    pub textures: ChannelImages,
    /// Fraction of texels with a source point.
    pub coverage: f32,
    /// The correspondence used (keep it to re-bake channels without re-tracing). Its entries
    /// are empty unless [`BakeOptions::keep_correspondence`].
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

/// The destination as the bake needs it: UV set checked, normals and tangents present.
fn prepare_dest(dest: &Mesh, opts: &BakeOptions) -> Result<Mesh> {
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
    Ok(dest)
}

/// Source lookups for `opts`' sample grid, `band` rows at a time: exact copies with
/// `source_triangles`, rays otherwise.
fn tracer<'a>(
    dest: &'a Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &'a BakeOptions,
    band: u32,
) -> Result<Tracer<'a>> {
    let ss = opts.supersample.max(1);
    let (w, h) = (opts.width * ss, opts.height * ss);
    match &opts.source_triangles {
        Some(refs) => Tracer::copy(dest, source, refs, w, h, opts.uv_set, band),
        None => {
            let maxd = opts
                .max_distance
                .unwrap_or_else(|| source.bounds().diagonal() * 0.03);
            Tracer::rays(dest, dest_to_world, source, w, h, opts.uv_set, maxd, band)
        }
    }
}

fn prepare(
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
) -> Result<Prepared> {
    let dest = prepare_dest(dest, opts)?;
    let ss = opts.supersample.max(1);
    let corr = tracer(&dest, dest_to_world, source, opts, opts.height * ss)?.whole(ss);
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

/// Finished rows of one channel, owned.
struct Strip<'a> {
    channel: &'a str,
    encoding: ChannelEncoding,
    y: u32,
    pixels: Vec<Vec4>,
}

/// Turns per-sample channel images into final textures: normal conversion, supersample
/// resolve, dilation. Bands of sample rows arrive top to bottom, and finished rows leave as
/// soon as no later band can change them (dilation reaches `gutter` rows).
struct Finisher<'a> {
    dest: &'a Mesh,
    dest_to_world: Mat4,
    supersample: u32,
    /// Output size.
    width: u32,
    height: u32,
    gutter: u32,
    channels: BTreeMap<String, Pending>,
    /// Output rows resolved, and handed out.
    resolved: u32,
    emitted: u32,
}

/// A channel's resolved rows that are not finished yet.
struct Pending {
    /// The encoding of the channel's first band, and what it is finished as.
    input: ChannelEncoding,
    output: ChannelEncoding,
    /// Undilated rows from `lo` up to the finisher's `resolved`.
    lo: u32,
    rows: Vec<Vec4>,
}

impl<'a> Finisher<'a> {
    fn new(
        dest: &'a Mesh,
        dest_to_world: Mat4,
        width: u32,
        height: u32,
        supersample: u32,
        gutter: u32,
    ) -> Finisher<'a> {
        Finisher {
            dest,
            dest_to_world,
            supersample: supersample.max(1),
            width,
            height,
            gutter,
            channels: BTreeMap::new(),
            resolved: 0,
            emitted: 0,
        }
    }

    /// The next `rows` sample rows: `entries` and every channel's pixels, row-major.
    fn band(
        &mut self,
        rows: u32,
        entries: &[Option<crate::bake::CorrespondenceEntry>],
        raw: ChannelImages,
        out: &mut dyn FnMut(Strip<'_>),
    ) {
        let ss = self.supersample;
        let sample_width = self.width * ss;
        let out_rows = rows / ss;
        for (name, ci) in raw {
            let p = self.channels.entry(name).or_insert_with(|| Pending {
                input: ci.encoding,
                output: match ci.encoding {
                    ChannelEncoding::NormalWorld => ChannelEncoding::NormalTangent,
                    e => e,
                },
                lo: 0,
                rows: Vec::new(),
            });
            let mut px = ci.image.pixels;
            // Mark coverage in alpha from the correspondence (hosts may not).
            for (p, e) in px.iter_mut().zip(entries) {
                if e.is_none() {
                    *p = Vec4::ZERO;
                } else if p.w <= 0.0 {
                    p.w = 1.0;
                }
            }
            if p.input == ChannelEncoding::NormalWorld {
                for (p, e) in px.iter_mut().zip(entries) {
                    if let Some(e) = e {
                        let nw = p.truncate().normalize_or_zero();
                        let (t, b, n) = dest_frame(
                            self.dest,
                            self.dest_to_world,
                            e.dest_triangle,
                            e.dest_barycentrics,
                        );
                        let ts = Vec3::new(nw.dot(t), nw.dot(b), nw.dot(n)).normalize_or_zero();
                        *p = ts.extend(1.0);
                    }
                }
            }
            let start = p.rows.len();
            if ss > 1 {
                p.rows
                    .resize(start + (out_rows * self.width) as usize, Vec4::ZERO);
                downsample_coverage_rows(&px, sample_width, ss, &mut p.rows[start..]);
            } else if p.rows.is_empty() {
                p.rows = px;
            } else {
                p.rows.extend_from_slice(&px);
            }
            if p.output == ChannelEncoding::NormalTangent {
                for p in &mut p.rows[start..] {
                    if p.w > 0.0 {
                        *p = p.truncate().normalize_or_zero().extend(p.w);
                    }
                }
            }
        }
        self.resolved += out_rows;
        self.flush(out);
    }

    /// Dilates and hands out the rows no later band can change, in strips of at least eight
    /// gutters (each strip redoes the dilation of a gutter's worth of rows on either side).
    fn flush(&mut self, out: &mut dyn FnMut(Strip<'_>)) {
        let last = self.resolved >= self.height;
        let end = if last {
            self.height
        } else {
            self.resolved.saturating_sub(self.gutter)
        };
        if !last && end < self.emitted + (8 * self.gutter).max(1) {
            return;
        }
        let (w, g, from) = (self.width as usize, self.gutter, self.emitted);
        for (name, p) in &mut self.channels {
            let mut img = Image {
                width: self.width,
                height: self.resolved - p.lo,
                pixels: if last {
                    std::mem::take(&mut p.rows)
                } else {
                    p.rows.clone()
                },
            };
            img.dilate(g);
            let mut px = img.pixels;
            px.truncate((end - p.lo) as usize * w);
            px.drain(..(from - p.lo) as usize * w);
            // Covered texels get full alpha; alpha is coverage, not opacity.
            for p in &mut px {
                if p.w > 0.0 {
                    p.w = 1.0;
                }
            }
            if !last {
                // Keep a gutter of undilated rows above the next strip.
                let lo = end.saturating_sub(g);
                p.rows.drain(..(lo - p.lo) as usize * w);
                p.lo = lo;
            }
            out(Strip {
                channel: name,
                encoding: p.output,
                y: from,
                pixels: px,
            });
        }
        self.emitted = end;
    }
}

/// Collects strips into whole images (`width × height`).
fn collect(textures: &mut ChannelImages, width: u32, height: u32, s: Strip<'_>) {
    let ci = textures
        .entry(s.channel.to_owned())
        .or_insert_with(|| ChannelImage {
            image: Image {
                width,
                height,
                pixels: Vec::new(),
            },
            encoding: s.encoding,
        });
    let px = &mut ci.image.pixels;
    let n = width as usize * height as usize;
    if px.is_empty() && s.pixels.len() == n {
        *px = s.pixels;
    } else {
        px.reserve_exact(n - px.len());
        px.extend_from_slice(&s.pixels);
    }
}

/// [`Finisher`] over whole sample images in one band.
fn finish(
    raw: ChannelImages,
    prep: &Prepared,
    dest_to_world: Mat4,
    opts: &BakeOptions,
) -> ChannelImages {
    let corr = &prep.corr;
    let ss = corr.supersample.max(1);
    let (w, h) = (corr.width / ss, corr.height / ss);
    let mut fin = Finisher::new(&prep.dest, dest_to_world, w, h, ss, opts.gutter);
    let mut textures = ChannelImages::new();
    fin.band(corr.height, &corr.entries, raw, &mut |s| {
        collect(&mut textures, w, h, s)
    });
    textures
}

fn displacement_channel(
    entries: &[Option<crate::bake::CorrespondenceEntry>],
    width: u32,
    height: u32,
) -> ChannelImage {
    let mut img = Image::new(width, height, Vec4::ZERO);
    for (p, e) in img.pixels.iter_mut().zip(entries) {
        if let Some(e) = e {
            *p = Vec4::new(e.distance, e.distance, e.distance, 1.0);
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

/// The channels the host shades (all but [`DISPLACEMENT`]).
fn host_channels(opts: &BakeOptions) -> Vec<String> {
    opts.channels
        .iter()
        .filter(|c| c.as_str() != DISPLACEMENT)
        .cloned()
        .collect()
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
        channels: host_channels(opts),
    };
    Ok(PreparedBake {
        prep,
        dest_to_world,
        opts: opts.clone(),
        request,
    })
}

/// Phase 2: takes the shaded channels (request-sized) and produces final textures.
pub fn finish_bake(prepared: PreparedBake, shaded: ChannelImages) -> Result<BakeOutput> {
    let req = &prepared.request;
    check_shaded(&shaded, &req.channels, req.width, req.height)?;
    Ok(finish_output(
        prepared.prep,
        prepared.dest_to_world,
        &prepared.opts,
        shaded,
    ))
}

/// Every requested channel came back, at the request's size.
fn check_shaded(
    shaded: &ChannelImages,
    channels: &[String],
    width: u32,
    height: u32,
) -> Result<()> {
    for c in channels {
        let img = shaded
            .get(c)
            .ok_or_else(|| Error::Host(format!("missing channel `{c}`")))?;
        let n = width as usize * height as usize;
        if (img.image.width, img.image.height) != (width, height) || img.image.pixels.len() != n {
            return Err(Error::Host(format!("channel `{c}` has the wrong size")));
        }
    }
    Ok(())
}

fn finish_output(
    prep: Prepared,
    dest_to_world: Mat4,
    opts: &BakeOptions,
    mut shaded: ChannelImages,
) -> BakeOutput {
    if opts.channels.iter().any(|c| c == DISPLACEMENT) {
        let c = &prep.corr;
        shaded.insert(
            DISPLACEMENT.into(),
            displacement_channel(&c.entries, c.width, c.height),
        );
    }
    let coverage = prep.corr.coverage();
    let textures = finish(shaded, &prep, dest_to_world, opts);
    output(prep.corr, textures, coverage, opts)
}

/// The output, keeping the correspondence's entries only when `opts` asks.
fn output(
    mut correspondence: Correspondence,
    textures: ChannelImages,
    coverage: f32,
    opts: &BakeOptions,
) -> BakeOutput {
    if !opts.keep_correspondence {
        correspondence.entries = Vec::new();
    }
    BakeOutput {
        textures,
        coverage,
        correspondence,
    }
}

/// Point-evaluation bake: the host evaluates its real materials at the source points.
///
/// The bake runs a band of rows at a time (about 128 thousand samples): the host is called
/// once per band, and neither the request, the correspondence (unless
/// [`BakeOptions::keep_correspondence`]) nor the supersampled channel images ever exist for
/// the whole atlas. The output textures are whole float images; to store them in another
/// format without ever holding those, use [`bake_rows_with_host`].
pub fn bake_with_host(
    host: &mut dyn RenderHost,
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
) -> Result<BakeOutput> {
    let mut textures = ChannelImages::new();
    let (w, h) = (opts.width, opts.height);
    let (coverage, correspondence) =
        bake_bands(host, dest, dest_to_world, source, opts, &mut |s| {
            collect(&mut textures, w, h, s)
        })?;
    Ok(BakeOutput {
        textures,
        coverage,
        correspondence,
    })
}

/// Finished rows of one channel from [`bake_rows_with_host`].
#[derive(Clone, Copy, Debug)]
pub struct BakedRows<'a> {
    /// The channel, as named in [`BakeOptions::channels`].
    pub channel: &'a str,
    /// How to read the texels; normals the host returns world-space arrive tangent-space,
    /// as in [`BakeOutput::textures`].
    pub encoding: ChannelEncoding,
    /// The texture row `pixels` starts at. Each channel's rows arrive top to bottom, each
    /// row once.
    pub y: u32,
    /// Whole rows of [`BakeOptions::width`] texels, exactly the texels
    /// [`BakeOutput::textures`] would hold.
    pub pixels: &'a [Vec4],
}

/// [`bake_with_host`] for callers that keep the textures in their own form (say 8-bit):
/// finished rows go to `rows` as soon as no later band can change them, so the bake never
/// holds a whole float image. Same texels, coverage and correspondence as [`bake_with_host`];
/// the returned [`BakeOutput::textures`] is empty.
pub fn bake_rows_with_host(
    host: &mut dyn RenderHost,
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
    rows: &mut dyn FnMut(BakedRows<'_>),
) -> Result<BakeOutput> {
    let (coverage, correspondence) =
        bake_bands(host, dest, dest_to_world, source, opts, &mut |s| {
            rows(BakedRows {
                channel: s.channel,
                encoding: s.encoding,
                y: s.y,
                pixels: &s.pixels,
            })
        })?;
    Ok(BakeOutput {
        textures: ChannelImages::new(),
        coverage,
        correspondence,
    })
}

/// The banded point-evaluation bake behind [`bake_with_host`] and [`bake_rows_with_host`]:
/// per band of sample rows, trace, shade and hand the rows to a [`Finisher`].
fn bake_bands(
    host: &mut dyn RenderHost,
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    opts: &BakeOptions,
    out: &mut dyn FnMut(Strip<'_>),
) -> Result<(f32, Correspondence)> {
    let dest = prepare_dest(dest, opts)?;
    let ss = opts.supersample.max(1);
    let (w, h) = (opts.width * ss, opts.height * ss);
    // Whole output rows per band.
    let band = ((BAND_SAMPLES / w.max(1) as usize) as u32 / ss).max(1) * ss;
    let mut tracer = Some(tracer(&dest, dest_to_world, source, opts, band)?);
    let bands = tracer.as_ref().map_or(0, Tracer::bands);
    let channels = host_channels(opts);
    let displacement = opts.channels.iter().any(|c| c == DISPLACEMENT);
    let mut fin = Finisher::new(
        &dest,
        dest_to_world,
        opts.width,
        opts.height,
        ss,
        opts.gutter,
    );
    let mut kept = Vec::new();
    if opts.keep_correspondence {
        kept.reserve_exact(w as usize * h as usize);
    }
    let mut covered = 0;
    for i in 0..bands {
        let rows = band.min(h - i as u32 * band);
        let entries = tracer.as_ref().map_or_else(Vec::new, |t| t.rows(i));
        if i + 1 == bands {
            // The source lookup (a BVH when casting rays) goes before the last band is shaded.
            tracer = None;
        }
        covered += entries.iter().filter(|e| e.is_some()).count();
        let mut raw = ChannelImages::new();
        if !channels.is_empty() {
            let req = SurfaceBakeRequest {
                width: w,
                height: rows,
                samples: entries
                    .iter()
                    .map(|e| e.as_ref().map(|e| interpolate_sample(source, e)))
                    .collect(),
                channels: channels.clone(),
            };
            let mut part = host.bake_surface(&req)?;
            drop(req);
            check_shaded(&part, &channels, w, rows)?;
            for c in &channels {
                if let Some(ci) = part.remove(c) {
                    raw.insert(c.clone(), ci);
                }
            }
        }
        if displacement {
            raw.insert(DISPLACEMENT.into(), displacement_channel(&entries, w, rows));
        }
        fin.band(rows, &entries, raw, out);
        if opts.keep_correspondence {
            kept.extend(entries);
        }
    }
    let coverage = covered as f32 / (w as usize * h as usize).max(1) as f32;
    let correspondence = Correspondence {
        width: w,
        height: h,
        supersample: ss,
        entries: kept,
    };
    Ok((coverage, correspondence))
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
            let c = &prep.corr;
            raw.insert(
                DISPLACEMENT.into(),
                displacement_channel(&c.entries, c.width, c.height),
            );
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
    Ok(output(prep.corr, textures, coverage, opts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MaterialId;
    use crate::host::{BakedMaterialDesc, MaterialInfo, ViewImages, ViewRenderRequest};

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

    #[test]
    fn source_points_match_the_destination_on_mirrored_instances() {
        // A lopsided triangle, so swapped barycentrics land somewhere else, placed as is and
        // mirrored. The destination is the flattened scene with new UVs.
        let mut tri = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
            vec![0, 1, 2],
        );
        tri.normals = Some(vec![Vec3::Z; 3]);
        let mut source = Scene::from_mesh(tri);
        let mut mirror = source.instances[0].clone();
        mirror.transform = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0))
            * Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        source.instances.push(mirror);
        let (flat, refs) = source.flatten();
        let dest = crate::uv::parameterize(
            &flat,
            &crate::uv::ParamOptions {
                atlas_size: 64,
                ..crate::uv::ParamOptions::planar()
            },
        )
        .unwrap()
        .mesh;
        let opts = BakeOptions {
            width: 64,
            height: 64,
            supersample: 1,
            max_distance: Some(0.1),
            ..Default::default()
        };
        for copy in [false, true] {
            let opts = BakeOptions {
                source_triangles: copy.then(|| refs.clone()),
                ..opts.clone()
            };
            let prep = prepare(&dest, Mat4::IDENTITY, &source, &opts).unwrap();
            let mut seen = [0; 2];
            for e in prep.corr.entries.iter().flatten() {
                let [i0, i1, i2] = dest.triangle(e.dest_triangle as usize).map(|v| v as usize);
                let b = e.dest_barycentrics;
                let p =
                    dest.positions[i0] * b.x + dest.positions[i1] * b.y + dest.positions[i2] * b.z;
                let q = interpolate_sample(&source, e).position;
                assert!((p - q).length() < 1e-3, "copy {copy}: {p} vs {q}");
                if copy {
                    assert_eq!(e.source, refs[e.dest_triangle as usize]);
                }
                seen[e.source.instance as usize] += 1;
            }
            assert!(seen[0] > 20 && seen[1] > 20, "{seen:?}");
        }
    }

    #[test]
    fn banded_bake_matches_one_request() {
        // Bands of rows, each finished as soon as dilation allows, against one request.
        struct Counting(u32);
        impl RenderHost for Counting {
            fn material_info(&self, m: MaterialId) -> MaterialInfo {
                ProceduralHost.material_info(m)
            }
            fn bake_surface(&mut self, req: &SurfaceBakeRequest) -> Result<ChannelImages> {
                self.0 += 1;
                ProceduralHost.bake_surface(req)
            }
            fn render_views(&mut self, r: &ViewRenderRequest) -> Result<Vec<ViewImages>> {
                ProceduralHost.render_views(r)
            }
            fn create_material(&mut self, d: &BakedMaterialDesc) -> Result<MaterialId> {
                ProceduralHost.create_material(d)
            }
        }
        let mut quad = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2, 0, 2, 3],
        );
        quad.normals = Some(vec![Vec3::Z; 4]);
        // A diamond in UV space: half the texels are gutter, so dilation crosses bands.
        quad.uvs = vec![vec![
            Vec2::new(0.5, 0.0),
            Vec2::new(1.0, 0.5),
            Vec2::new(0.5, 1.0),
            Vec2::new(0.0, 0.5),
        ]];
        let source = Scene::from_mesh(quad.clone());
        for (supersample, gutter) in [(1, 2), (1, 12), (2, 12)] {
            let opts = BakeOptions {
                width: 1024,
                height: 400,
                supersample,
                gutter,
                channels: vec![
                    channels::BASE_COLOR.into(),
                    channels::NORMAL.into(),
                    DISPLACEMENT.into(),
                ],
                source_triangles: Some(source.flatten().1),
                ..Default::default()
            };
            let mut host = Counting(0);
            let banded = bake_with_host(&mut host, &quad, Mat4::IDENTITY, &source, &opts).unwrap();
            let band = BAND_SAMPLES as u32 / (1024 * supersample * supersample) * supersample;
            assert_eq!(host.0, (400 * supersample).div_ceil(band));
            assert!(host.0 > 3, "{} bands", host.0);
            let prepared = prepare_bake(&quad, Mat4::IDENTITY, &source, &opts).unwrap();
            let shaded = ProceduralHost.bake_surface(&prepared.request).unwrap();
            let whole = finish_bake(prepared, shaded).unwrap();
            assert_eq!(banded.textures, whole.textures);
            assert_eq!(banded.coverage, whole.coverage);
            assert!((banded.coverage - 0.5).abs() < 0.01, "{}", banded.coverage);
            assert_eq!(banded.correspondence, whole.correspondence);

            // Rows as they finish, without the correspondence: the same texels.
            let mut rows = ChannelImages::new();
            let opts = BakeOptions {
                keep_correspondence: false,
                ..opts
            };
            let out = bake_rows_with_host(
                &mut ProceduralHost,
                &quad,
                Mat4::IDENTITY,
                &source,
                &opts,
                &mut |r| {
                    let ci = rows
                        .entry(r.channel.into())
                        .or_insert_with(|| ChannelImage {
                            image: Image::new(1024, 0, Vec4::ZERO),
                            encoding: r.encoding,
                        });
                    assert_eq!(ci.image.height, r.y);
                    assert_eq!(r.pixels.len() % 1024, 0);
                    ci.image.height += r.pixels.len() as u32 / 1024;
                    ci.image.pixels.extend_from_slice(r.pixels);
                },
            )
            .unwrap();
            assert_eq!(rows, whole.textures);
            assert!(out.textures.is_empty() && out.correspondence.entries.is_empty());
            assert_eq!(out.coverage, whole.coverage);
        }
    }
}
