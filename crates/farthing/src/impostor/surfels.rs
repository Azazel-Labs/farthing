//! Surfel proxies: a surface covered by small oriented discs instead of triangles.
//!
//! Each surfel is a point on the source surface with a normal and a radius, plus the source
//! triangle and barycentric coordinates it came from, so the host can evaluate its own material
//! there (albedo, roughness, normal-map detail). farthing places and sizes the discs; it never
//! decides what they look like. Because every surfel carries a normal, the host lights it like
//! any other G-buffer content, so the proxy responds to dynamic lighting.
//!
//! Placement is blue noise over the surface area: area-weighted candidates thinned by weighted
//! sample elimination (Yuksel, *Sample Elimination for Generating Poisson Disk Sample Sets*,
//! Eurographics 2015), which spreads discs evenly with no holes or clumps for a given count.
//!
//! For runtime use, [`progressive_surfel_cloud`] orders the surfels so that every prefix is
//! itself an even covering (continuing the elimination in stages, as in Yuksel's progressive
//! sample sets): draw the first `k` at [`SurfelCloud::radius_for`]`(k)` and fade the next one
//! in for continuous LOD with no popping. [`SurfelCloud::encode`] writes a compact stream in
//! that order (quantized positions and normals, optional host attributes), so any prefix of the
//! bytes decodes to a coarser cloud: stream from the start and refine as data arrives.

use glam::{Vec2, Vec3};

/// Reference WGSL for drawing a [`SurfelCloud`] as oriented splats with the dithered fade of
/// [`SurfelCloud::lod`], writing a G-buffer (albedo, normal). Engines adapt it.
pub const SURFEL_SHADER: &str = include_str!("../../shaders/surfel_splat.wgsl");
use crate::{Error, Mesh, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Surfel {
    pub position: Vec3,
    /// Unit normal (interpolated vertex normal when the mesh has normals).
    pub normal: Vec3,
    pub radius: f32,
    /// Source triangle and barycentric coordinates, for the host's material evaluation.
    pub triangle: u32,
    pub barycentrics: Vec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfelOptions {
    /// Number of surfels.
    pub count: usize,
    /// Disc radius relative to the radius that would tile the area exactly once (above 1,
    /// neighbouring discs overlap so no gaps show).
    pub coverage: f32,
    /// Candidates generated per final surfel before elimination.
    pub candidates_per_surfel: usize,
    pub seed: u64,
    /// Optional per-triangle weight in [0, 1] (e.g. the fraction of views that see the
    /// triangle, from `farthing::visibility`): surfels are spread over area × weight, so hidden
    /// interior surface gets none and the discs are sized for the visible surface only.
    #[serde(skip)]
    pub triangle_weights: Option<Vec<f32>>,
    /// With `triangle_weights`: also make the *density* follow the weights (variable-radius
    /// elimination) and size each surfel for its surface (radius ÷ √weight). Off by default:
    /// weights then pick where candidates come from, elimination evens out the spacing, and
    /// every surfel has the same radius. That renders better for visibility weights (hidden
    /// leaves would otherwise get large discs that show through gaps: −0.3 to −0.9 dB on the
    /// bush); turn it on for importance maps where low weight means "fewer, larger discs".
    #[serde(default)]
    pub adaptive_density: bool,
}

impl Default for SurfelOptions {
    fn default() -> Self {
        SurfelOptions {
            count: 2000,
            coverage: 1.7,
            candidates_per_surfel: 5,
            seed: 1,
            triangle_weights: None,
            adaptive_density: false,
        }
    }
}

fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Places `opts.count` surfels on `mesh` (deterministic for a given seed). With
/// [`SurfelOptions::triangle_weights`] candidates are drawn over area × weight; with
/// [`SurfelOptions::adaptive_density`] too, the final density follows the weights and each
/// surfel is sized for the surface it stands for: radius ÷ √weight (weights below
/// [`MIN_SIZING_WEIGHT`] count as that), so low-weight regions get fewer, larger discs.
pub fn surfel_cloud(mesh: &Mesh, opts: &SurfelOptions) -> Result<Vec<Surfel>> {
    Ok(cloud_and_area(mesh, opts)?.0)
}

/// Weight floor for sizing surfels on low-weight triangles (see [`surfel_cloud`]).
pub const MIN_SIZING_WEIGHT: f32 = 0.05;

/// Spacing and radius factor for surfels on triangle `t`: 1 ⁄ √weight (weight floored).
fn sizing_scale(opts: &SurfelOptions, t: usize) -> f32 {
    match &opts.triangle_weights {
        Some(w) if opts.adaptive_density => {
            1.0 / w.get(t).copied().unwrap_or(0.0).clamp(MIN_SIZING_WEIGHT, 1.0).sqrt()
        }
        _ => 1.0,
    }
}

/// The surfels and the (weighted) area they cover.
fn cloud_and_area(mesh: &Mesh, opts: &SurfelOptions) -> Result<(Vec<Surfel>, f64)> {
    mesh.validate()?;
    if opts.count == 0 {
        return Err(Error::InvalidInput(
            "surfel count must be at least 1".into(),
        ));
    }
    let nt = mesh.triangle_count();
    let areas: Vec<f64> = (0..nt)
        .map(|t| {
            let [a, b, c] = mesh.triangle_positions(t);
            let w = opts
                .triangle_weights
                .as_ref()
                .map_or(1.0, |w| w.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0))
                as f64;
            0.5 * (b - a).cross(c - a).length() as f64 * w
        })
        .collect();
    let total: f64 = areas.iter().sum();
    if total <= 0.0 {
        return Err(Error::InvalidInput(
            "mesh has no (weighted) surface area".into(),
        ));
    }
    // Area-weighted candidates: inverse CDF over triangles, uniform within each.
    let mut cdf = Vec::with_capacity(nt);
    let mut acc = 0.0;
    for a in &areas {
        acc += a;
        cdf.push(acc);
    }
    let m = opts.count * opts.candidates_per_surfel.max(1);
    let mut rng = opts.seed ^ 0xD1B5_4A32_D192_ED03;
    let mut cand: Vec<(Vec3, u32, Vec3)> = Vec::with_capacity(m);
    for _ in 0..m {
        let r = splitmix(&mut rng) * total;
        let t = cdf.partition_point(|&c| c < r).min(nt - 1);
        let (mut u, mut v) = (splitmix(&mut rng) as f32, splitmix(&mut rng) as f32);
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let bary = Vec3::new(1.0 - u - v, u, v);
        let [a, b, c] = mesh.triangle_positions(t);
        cand.push((a * bary.x + b * bary.y + c * bary.z, t as u32, bary));
    }
    let scales: Option<Vec<f32>> = (opts.adaptive_density && opts.triangle_weights.is_some())
        .then(|| cand.iter().map(|c| sizing_scale(opts, c.1 as usize)).collect());
    let keep = eliminate(
        &cand.iter().map(|c| c.0).collect::<Vec<_>>(),
        scales.as_deref(),
        opts.count,
        total,
    );
    // Radius that tiles the area once, times coverage.
    let radius =
        ((total / (opts.count as f64 * std::f64::consts::PI)).sqrt() as f32) * opts.coverage;
    let size = |t: usize| radius * sizing_scale(opts, t);
    let surfels = keep
        .into_iter()
        .map(|i| {
            let (p, t, bary) = cand[i];
            let tri = mesh.triangle(t as usize);
            let face = {
                let [a, b, c] = mesh.triangle_positions(t as usize);
                (b - a).cross(c - a).normalize_or_zero()
            };
            let normal = match &mesh.normals {
                Some(n) => {
                    let s = n[tri[0] as usize] * bary.x
                        + n[tri[1] as usize] * bary.y
                        + n[tri[2] as usize] * bary.z;
                    let s = s.normalize_or_zero();
                    if s == Vec3::ZERO {
                        face
                    } else {
                        s
                    }
                }
                None => face,
            };
            Surfel {
                position: p,
                normal,
                radius: size(t as usize),
                triangle: t,
                barycentrics: bary,
            }
        })
        .collect();
    Ok((surfels, total))
}

/// Weighted sample elimination: removes the most crowded candidate until `n` remain. Returns
/// the kept indices in ascending order.
fn eliminate(pts: &[Vec3], scale: Option<&[f32]>, n: usize, area: f64) -> Vec<usize> {
    eliminate_ordered(pts, scale, n, area).0
}

/// [`eliminate`], also returning the indices in the order they were removed.
/// `scale` (≥ 1 per point, optional) widens each point's neighbourhood: spacing grows with it,
/// so a point at scale s ends up with about 1/s² the density (adaptive elimination).
fn eliminate_ordered(
    pts: &[Vec3],
    scale: Option<&[f32]>,
    n: usize,
    area: f64,
) -> (Vec<usize>, Vec<usize>) {
    let m = pts.len();
    if n >= m {
        return ((0..m).collect(), Vec::new());
    }
    let sc = |i: usize| scale.map_or(1.0, |s| s[i]);
    let smax = scale.map_or(1.0, |s| s.iter().copied().fold(1.0f32, f32::max));
    // Neighbourhood radius from the 2D Poisson disk packing density over the surface area.
    let rmax = (2.0 * (area / (2.0 * 3f64.sqrt() * n as f64)).sqrt()) as f32;
    let cell = (rmax * smax).max(1e-12);
    let key = |p: Vec3| -> (i32, i32, i32) {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    };
    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<u32>> = Default::default();
    for (i, &p) in pts.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i as u32);
    }
    let neighbours = |i: usize, out: &mut Vec<(u32, f32)>| {
        out.clear();
        let p = pts[i];
        let (cx, cy, cz) = key(p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in list {
                            if j as usize != i {
                                // Distance in units of the pair's neighbourhood radius.
                                let r = rmax * 0.5 * (sc(i) + sc(j as usize));
                                let d = (pts[j as usize] - p).length() / r;
                                if d < 1.0 {
                                    out.push((j, d));
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    let wfn = |d: f32| (1.0 - d).powi(8);
    let mut weight = vec![0.0f32; m];
    let mut nb = Vec::new();
    for i in 0..m {
        neighbours(i, &mut nb);
        weight[i] = nb.iter().map(|&(_, d)| wfn(d)).sum();
    }
    // Max-heap by weight with lazy invalidation (ties broken by index for determinism).
    #[derive(PartialEq)]
    struct E(f32, u32);
    impl Eq for E {}
    impl PartialOrd for E {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for E {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
        }
    }
    let mut heap: std::collections::BinaryHeap<E> =
        (0..m).map(|i| E(weight[i], i as u32)).collect();
    let mut alive = vec![true; m];
    let mut left = m;
    let mut removed = Vec::with_capacity(m - n);
    while left > n {
        let Some(E(w, i)) = heap.pop() else { break };
        let i = i as usize;
        if !alive[i] || w != weight[i] {
            continue;
        }
        alive[i] = false;
        removed.push(i);
        left -= 1;
        neighbours(i, &mut nb);
        for &(j, d) in &nb {
            let j = j as usize;
            if alive[j] {
                weight[j] -= wfn(d);
                heap.push(E(weight[j], j as u32));
            }
        }
    }
    ((0..m).filter(|&i| alive[i]).collect(), removed)
}

/// Surfels in progressive order, with what is needed to size any prefix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfelCloud {
    /// Every prefix is an even covering of the surface.
    pub surfels: Vec<Surfel>,
    /// (Weighted) surface area the surfels cover.
    pub area: f32,
    /// [`SurfelOptions::coverage`].
    pub coverage: f32,
}

impl SurfelCloud {
    /// Disc radius when the first `k` surfels are drawn: the radius that tiles the area with
    /// `k` discs, times the coverage factor. `k` may be fractional (fading).
    pub fn radius_for(&self, k: f32) -> f32 {
        (self.area / (k.max(1.0) * std::f32::consts::PI)).sqrt() * self.coverage
    }

    /// Surfel count for an object whose bounding sphere spans `screen_diameter_px` pixels, with
    /// each disc covering about `px_per_surfel` pixels across: (visible area in pixels) ÷ (disc
    /// area). Clamped to the cloud's size; fractional for fading.
    pub fn count_for_screen(&self, screen_diameter_px: f32, radius: f32, px_per_surfel: f32) -> f32 {
        // Roughly half the surface faces the camera; pixels per world unit at that size.
        let ppu = screen_diameter_px / (2.0 * radius.max(1e-12));
        let visible_px = 0.5 * self.area * ppu * ppu;
        let disc_px = std::f32::consts::FRAC_PI_4 * px_per_surfel * px_per_surfel;
        (visible_px / disc_px.max(1e-6)).clamp(1.0, self.surfels.len() as f32)
    }

    /// Draw list for a fractional count `k`: the first `floor(k)` surfels at full size and the
    /// next one fading in. Returns (surfels to draw, radius scale, fade of the last one): draw
    /// surfel `i` with radius `surfels[i].radius × scale` (its radius in the full cloud grown by
    /// √(count ⁄ k), so a prefix covers what the whole cloud covers), and the last one at
    /// `fade` (0..1: shrink it by √fade, or dither its coverage, as the reference shader does).
    pub fn lod(&self, k: f32) -> (&[Surfel], f32, f32) {
        let n_all = self.surfels.len().max(1) as f32;
        let k = k.clamp(1.0, n_all);
        let whole = k.floor() as usize;
        let frac = k - whole as f32;
        let n = if frac > 0.0 { (whole + 1).min(self.surfels.len()) } else { whole };
        (&self.surfels[..n], (n_all / k).sqrt(), if frac > 0.0 { frac } else { 1.0 })
    }
}

/// A per-surfel attribute the host evaluated at each surfel's source point (albedo,
/// roughness, …), stored as `components` unorm8 values per surfel in cloud order. farthing
/// carries the bytes; what they mean is the host's business.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfelChannel {
    pub name: String,
    pub components: u8,
    pub data: Vec<u8>,
}

/// A decoded (possibly partial) surfel stream.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedSurfels {
    /// The surfels received so far (a prefix of the encoded cloud, so an even covering).
    pub cloud: SurfelCloud,
    pub channels: Vec<SurfelChannel>,
    /// Surfel count of the whole stream.
    pub total: usize,
}

const MAGIC: &[u8; 4] = b"LFSF";
const VERSION: u16 = 1;
/// Flag: records carry source triangle and barycentrics.
const FLAG_SOURCE: u16 = 1;
/// Flag: records carry a per-surfel radius scale (u8, log2 steps of 1/16 relative to the
/// cloud's base radius, covering ×1/16 to ×16).
const FLAG_RADII: u16 = 2;

fn encode_scale(s: f32) -> u8 {
    ((s.max(1e-6).log2() + 4.0) * 16.0).round().clamp(0.0, 255.0) as u8
}

fn decode_scale(q: u8) -> f32 {
    (q as f32 / 16.0 - 4.0).exp2()
}

impl SurfelCloud {
    /// Bytes per surfel record in [`encode`](Self::encode)'s format (plus one when the radii
    /// vary, see [`has_varying_radii`](Self::has_varying_radii)).
    pub fn record_size(channels: &[SurfelChannel], with_source: bool) -> usize {
        10 + if with_source { 8 } else { 0 }
            + channels.iter().map(|c| c.components as usize).sum::<usize>()
    }

    /// Whether surfels differ in radius (weighted clouds), so the stream stores a scale each.
    pub fn has_varying_radii(&self) -> bool {
        let base = self.radius_for(self.surfels.len() as f32);
        self.surfels.iter().any(|s| (s.radius / base - 1.0).abs() > 1e-4)
    }

    /// Compact stream, little-endian: a header (magic `LFSF`, version, flags, count, area,
    /// coverage, bounds, channel names and sizes), then one fixed-size record per surfel in
    /// progressive order: position as 3 × u16 within the bounds, normal as 2 × u16 octahedral,
    /// optionally the source triangle (u32) and barycentrics (2 × u16), a radius scale (u8,
    /// only for clouds whose radii vary), then each channel's bytes. The base radius comes from
    /// the area and count ([`radius_for`](Self::radius_for)). 10 bytes per surfel without
    /// source references, radii or channels.
    pub fn encode(&self, channels: &[SurfelChannel], with_source: bool) -> Result<Vec<u8>> {
        let n = self.surfels.len();
        for c in channels {
            if c.data.len() != n * c.components as usize || c.name.len() > 255 {
                return Err(Error::InvalidInput(format!(
                    "surfel channel `{}` must have {} × {n} bytes and a name under 256 bytes",
                    c.name, c.components
                )));
            }
        }
        let (lo, hi) = self.surfels.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), s| (lo.min(s.position), hi.max(s.position)),
        );
        let (lo, hi) = if n == 0 { (Vec3::ZERO, Vec3::ZERO) } else { (lo, hi) };
        let radii = self.has_varying_radii();
        let base = self.radius_for(n as f32);
        let mut out = Vec::with_capacity(64 + n * (Self::record_size(channels, with_source) + 1));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let flags = if with_source { FLAG_SOURCE } else { 0 } | if radii { FLAG_RADII } else { 0 };
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for f in [self.area, self.coverage, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z] {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.push(channels.len() as u8);
        for c in channels {
            out.push(c.name.len() as u8);
            out.extend_from_slice(c.name.as_bytes());
            out.push(c.components);
        }
        let ext = (hi - lo).max(Vec3::splat(1e-30));
        let q16 = |x: f32| (x.clamp(0.0, 1.0) * 65535.0).round() as u16;
        for (i, s) in self.surfels.iter().enumerate() {
            let p = (s.position - lo) / ext;
            for v in [p.x, p.y, p.z] {
                out.extend_from_slice(&q16(v).to_le_bytes());
            }
            let o = crate::impostor::oct_encode(s.normal);
            out.extend_from_slice(&q16(o.x).to_le_bytes());
            out.extend_from_slice(&q16(o.y).to_le_bytes());
            if with_source {
                out.extend_from_slice(&s.triangle.to_le_bytes());
                out.extend_from_slice(&q16(s.barycentrics.y).to_le_bytes());
                out.extend_from_slice(&q16(s.barycentrics.z).to_le_bytes());
            }
            if radii {
                out.push(encode_scale(s.radius / base));
            }
            for c in channels {
                let k = c.components as usize;
                out.extend_from_slice(&c.data[i * k..(i + 1) * k]);
            }
        }
        Ok(out)
    }

    /// Decodes a stream from [`encode`](Self::encode), or as much of one as `bytes` holds: the
    /// complete records present are returned (a coarser but even covering), with the stream's
    /// full count in [`DecodedSurfels::total`]. Fails only if the header itself is incomplete
    /// or not a surfel stream.
    pub fn decode(bytes: &[u8]) -> Result<DecodedSurfels> {
        let bad = |m: &str| Error::InvalidInput(format!("surfel stream: {m}"));
        let mut at = 0usize;
        let mut take = |k: usize| -> Result<&[u8]> {
            let s = bytes.get(at..at + k).ok_or_else(|| bad("truncated header"))?;
            at += k;
            Ok(s)
        };
        if take(4)? != MAGIC {
            return Err(bad("not an LFSF stream"));
        }
        let u16le = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
        let version = u16le(take(2)?);
        if version != VERSION {
            return Err(bad(&format!("unsupported version {version}")));
        }
        let flags = u16le(take(2)?);
        let with_source = flags & FLAG_SOURCE != 0;
        let radii = flags & FLAG_RADII != 0;
        let total = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut f = [0f32; 8];
        for v in &mut f {
            *v = f32::from_le_bytes(take(4)?.try_into().unwrap());
        }
        let (area, coverage) = (f[0], f[1]);
        let (lo, hi) = (Vec3::new(f[2], f[3], f[4]), Vec3::new(f[5], f[6], f[7]));
        let nch = take(1)?[0] as usize;
        let mut channels = Vec::with_capacity(nch);
        for _ in 0..nch {
            let len = take(1)?[0] as usize;
            let name = String::from_utf8(take(len)?.to_vec()).map_err(|_| bad("channel name"))?;
            let components = take(1)?[0];
            channels.push(SurfelChannel {
                name,
                components,
                data: Vec::new(),
            });
        }
        let rec = Self::record_size(&channels, with_source) + radii as usize;
        let n = ((bytes.len() - at) / rec).min(total);
        let ext = hi - lo;
        let d16 = |b: &[u8]| u16le(b) as f32 / 65535.0;
        let mut surfels = Vec::with_capacity(n);
        let radius = (area / (total.max(1) as f32 * std::f32::consts::PI)).sqrt() * coverage;
        for i in 0..n {
            let r = &bytes[at + i * rec..at + (i + 1) * rec];
            let position = lo + Vec3::new(d16(&r[0..2]), d16(&r[2..4]), d16(&r[4..6])) * ext;
            let normal = crate::impostor::oct_decode(Vec2::new(d16(&r[6..8]), d16(&r[8..10])));
            let mut o = 10;
            let (triangle, barycentrics) = if with_source {
                let t = u32::from_le_bytes(r[10..14].try_into().unwrap());
                let (u, v) = (d16(&r[14..16]), d16(&r[16..18]));
                o = 18;
                (t, Vec3::new((1.0 - u - v).max(0.0), u, v))
            } else {
                (0, Vec3::new(1.0, 0.0, 0.0))
            };
            let radius = if radii {
                o += 1;
                radius * decode_scale(r[o - 1])
            } else {
                radius
            };
            for c in &mut channels {
                let k = c.components as usize;
                c.data.extend_from_slice(&r[o..o + k]);
                o += k;
            }
            surfels.push(Surfel {
                position,
                normal,
                radius,
                triangle,
                barycentrics,
            });
        }
        Ok(DecodedSurfels {
            cloud: SurfelCloud {
                surfels,
                area,
                coverage,
            },
            channels,
            total,
        })
    }
}

/// Like [`surfel_cloud`], ordered so that every prefix is an even covering (see
/// [`SurfelCloud`]). The set of surfels is the same as `surfel_cloud`'s.
pub fn progressive_surfel_cloud(mesh: &Mesh, opts: &SurfelOptions) -> Result<SurfelCloud> {
    let (surfels, area) = cloud_and_area(mesh, opts)?;
    let area = area as f32;
    let pts: Vec<Vec3> = surfels.iter().map(|s| s.position).collect();
    let scales: Option<Vec<f32>> = (opts.adaptive_density && opts.triangle_weights.is_some())
        .then(|| surfels.iter().map(|s| sizing_scale(opts, s.triangle as usize)).collect());
    let order = progressive_order(&pts, scales.as_deref(), area as f64);
    Ok(SurfelCloud {
        surfels: order.into_iter().map(|i| surfels[i]).collect(),
        area,
        coverage: opts.coverage,
    })
}

/// Ranks points so every prefix is well spread: eliminate in stages (each halving the count,
/// with the neighbourhood radius for that stage's size) down to one point, then reverse.
fn progressive_order(pts: &[Vec3], scale: Option<&[f32]>, area: f64) -> Vec<usize> {
    let n = pts.len();
    let mut alive: Vec<usize> = (0..n).collect();
    let mut eliminated: Vec<usize> = Vec::with_capacity(n);
    while alive.len() > 1 {
        let target = alive.len() / 2;
        let sub: Vec<Vec3> = alive.iter().map(|&i| pts[i]).collect();
        let sub_scale: Option<Vec<f32>> = scale.map(|s| alive.iter().map(|&i| s[i]).collect());
        let keep = eliminate_ordered(&sub, sub_scale.as_deref(), target, area);
        let mut kept = vec![false; sub.len()];
        for &k in &keep.0 {
            kept[k] = true;
        }
        eliminated.extend(keep.1.iter().map(|&k| alive[k]));
        alive = (0..sub.len()).filter(|&k| kept[k]).map(|k| alive[k]).collect();
    }
    eliminated.extend(alive);
    eliminated.reverse();
    eliminated
}

/// Disc geometry for hosts that draw surfels as small meshes: one `sides`-gon fan per surfel,
/// all vertices carrying the surfel normal, the source's attributes (UVs, colors) at the
/// surfel's barycentrics, and the source triangle's material slot.
pub fn surfels_to_mesh(source: &Mesh, surfels: &[Surfel], sides: u32) -> Mesh {
    let sides = sides.max(3);
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uvs: Vec<Vec<Vec2>> = vec![Vec::new(); source.uvs.len()];
    let mut idx = Vec::new();
    let mut mats = Vec::new();
    for s in surfels {
        let n = s.normal;
        let t = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let u = n.cross(t).normalize();
        let v = n.cross(u);
        let tri = source.triangle(s.triangle as usize);
        let base = pos.len() as u32;
        // Centre plus rim; the polygon's area matches the disc's.
        let rim = s.radius
            * (std::f32::consts::TAU
                / (sides as f32 * (std::f32::consts::TAU / sides as f32).sin()))
            .sqrt();
        pos.push(s.position);
        for k in 0..sides {
            let a = k as f32 / sides as f32 * std::f32::consts::TAU;
            pos.push(s.position + (u * a.cos() + v * a.sin()) * rim);
        }
        for _ in 0..=sides {
            nrm.push(n);
        }
        for (set, out) in source.uvs.iter().zip(&mut uvs) {
            let c = set[tri[0] as usize] * s.barycentrics.x
                + set[tri[1] as usize] * s.barycentrics.y
                + set[tri[2] as usize] * s.barycentrics.z;
            out.extend(std::iter::repeat_n(c, sides as usize + 1));
        }
        for k in 0..sides {
            idx.extend_from_slice(&[base, base + 1 + k, base + 1 + (k + 1) % sides]);
            mats.push(source.material_ids[s.triangle as usize]);
        }
    }
    let mut m = Mesh::new(pos, idx);
    m.normals = Some(nrm);
    m.uvs = uvs;
    m.material_ids = mats;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both reference shaders parse and validate (naga: wgpu's shader compiler).
    #[test]
    fn reference_shaders_validate() {
        for (name, src) in [("surfel_splat", SURFEL_SHADER), ("octahedral", crate::impostor::REFERENCE_SHADER)] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    fn plane(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                pos.push(Vec3::new(i as f32 / n as f32, 0.0, j as f32 / n as f32));
            }
        }
        let mut idx = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        Mesh::new(pos, idx)
    }

    #[test]
    fn surfels_cover_the_surface_evenly() {
        let m = plane(8);
        let s = surfel_cloud(
            &m,
            &SurfelOptions {
                count: 200,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s.len(), 200);
        assert_eq!(
            s,
            surfel_cloud(
                &m,
                &SurfelOptions {
                    count: 200,
                    ..Default::default()
                }
            )
            .unwrap()
        );
        // Blue noise: the closest pair is not much closer than the ideal spacing.
        let ideal = (1.0 / 200.0f32).sqrt();
        let mut min_d = f32::INFINITY;
        for i in 0..s.len() {
            for j in i + 1..s.len() {
                min_d = min_d.min((s[i].position - s[j].position).length());
            }
            assert!(s[i].normal.y.abs() > 0.99);
        }
        assert!(min_d > 0.4 * ideal, "min spacing {min_d} vs ideal {ideal}");
        // Every interior point of the plane is within a disc (blue noise thins out at an open
        // border, where the coverage factor alone has to close the gap).
        for k in 0..400 {
            let p = Vec3::new(
                0.05 + 0.9 * (k % 20) as f32 / 19.0,
                0.0,
                0.05 + 0.9 * (k / 20) as f32 / 19.0,
            );
            assert!(
                s.iter().any(|x| (x.position - p).length() <= x.radius),
                "hole at {p}"
            );
        }
        let disc = surfels_to_mesh(&m, &s, 6);
        assert_eq!(disc.triangle_count(), 200 * 6);
        disc.validate().unwrap();
    }

    /// Fraction of interior probe points of the unit plane not inside any of the discs.
    fn holes(s: &[Surfel], r: f32) -> f32 {
        let mut miss = 0;
        for k in 0..900 {
            let p = Vec3::new(
                0.08 + 0.84 * (k % 30) as f32 / 29.0,
                0.0,
                0.08 + 0.84 * (k / 30) as f32 / 29.0,
            );
            if !s.iter().any(|x| (x.position - p).length() <= r) {
                miss += 1;
            }
        }
        miss as f32 / 900.0
    }

    fn min_spacing(s: &[Surfel]) -> f32 {
        let mut d = f32::INFINITY;
        for i in 0..s.len() {
            for j in i + 1..s.len() {
                d = d.min((s[i].position - s[j].position).length());
            }
        }
        d
    }

    #[test]
    fn every_progressive_prefix_covers_the_surface() {
        let m = plane(8);
        let opts = SurfelOptions {
            count: 800,
            ..Default::default()
        };
        let cloud = progressive_surfel_cloud(&m, &opts).unwrap();
        let plain = surfel_cloud(&m, &opts).unwrap();
        // Same surfels, reordered.
        let mut a: Vec<_> = cloud.surfels.iter().map(|s| s.position.to_array().map(f32::to_bits)).collect();
        let mut b: Vec<_> = plain.iter().map(|s| s.position.to_array().map(f32::to_bits)).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert!((cloud.radius_for(800.0) - plain[0].radius).abs() < 1e-6);
        for k in [25usize, 50, 100, 200, 400, 800] {
            let r = cloud.radius_for(k as f32);
            let prefix = &cloud.surfels[..k];
            // About as hole-free and as evenly spread as a cloud generated for k surfels
            // directly (blue noise thins out at the plane's open border, so up to 1% of probes
            // near it can miss; a fresh cloud misses up to 0.1%).
            let fresh = surfel_cloud(&m, &SurfelOptions { count: k, ..opts.clone() }).unwrap();
            let (h, hf) = (holes(prefix, r), holes(&fresh, fresh[0].radius));
            assert!(h < 0.01, "prefix {k}: holes {h} vs fresh {hf}");
            assert!(min_spacing(prefix) >= 0.9 * min_spacing(&fresh), "prefix {k} clumps");
            // A prefix of the unordered cloud leaves holes at the same radius.
            if k < 400 {
                let hp = holes(&plain[..k], r);
                assert!(hp > 0.01 && hp > 3.0 * h, "plain prefix {k}: {hp} vs {h}");
            }
        }
        // Fractional LOD: the next surfel fades in.
        let (draw, scale, last) = cloud.lod(100.25);
        assert_eq!(draw.len(), 101);
        assert!((last - 0.25).abs() < 1e-6);
        assert!((draw[0].radius * scale - cloud.radius_for(100.25)).abs() < 1e-5);
        assert_eq!(cloud.lod(100.0).0.len(), 100);
    }

    #[test]
    fn stream_round_trips_and_decodes_prefixes() {
        let m = plane(8);
        let cloud = progressive_surfel_cloud(
            &m,
            &SurfelOptions {
                count: 300,
                ..Default::default()
            },
        )
        .unwrap();
        let albedo = SurfelChannel {
            name: "albedo".into(),
            components: 3,
            data: (0..300 * 3).map(|i| (i * 7 % 256) as u8).collect(),
        };
        let bytes = cloud.encode(std::slice::from_ref(&albedo), true).unwrap();
        let rec = SurfelCloud::record_size(std::slice::from_ref(&albedo), true);
        assert_eq!(rec, 21);
        let header = bytes.len() - 300 * rec;
        let d = SurfelCloud::decode(&bytes).unwrap();
        assert_eq!((d.total, d.cloud.surfels.len()), (300, 300));
        assert_eq!(d.channels, vec![albedo.clone()]);
        for (a, b) in cloud.surfels.iter().zip(&d.cloud.surfels) {
            assert!((a.position - b.position).length() < 2e-5, "{a:?} {b:?}");
            assert!(a.normal.dot(b.normal) > 0.9999);
            assert_eq!(a.triangle, b.triangle);
            assert!((a.barycentrics - b.barycentrics).abs().max_element() < 1e-4);
            assert!((a.radius - b.radius).abs() < 1e-6);
        }
        // Without source references or channels: 10 bytes per surfel.
        let small = cloud.encode(&[], false).unwrap();
        assert_eq!(small.len(), 45 + 300 * 10);
        // A truncated stream decodes its complete records: a coarser, still even covering.
        let part = SurfelCloud::decode(&bytes[..header + 37 * rec + 5]).unwrap();
        assert_eq!((part.total, part.cloud.surfels.len()), (300, 37));
        assert_eq!(part.channels[0].data, albedo.data[..37 * 3].to_vec());
        assert!(SurfelCloud::decode(&bytes[..10]).is_err());
        assert!(SurfelCloud::decode(b"nope, not surfels").is_err());
    }

    #[test]
    fn weighted_surfels_are_sized_for_their_surface() {
        let m = plane(8);
        // The z < 0.5 half has weight 0.25: a quarter of the density, so twice the radius.
        let w: Vec<f32> = (0..m.triangle_count())
            .map(|t| {
                let [a, b, c] = m.triangle_positions(t);
                if (a.z + b.z + c.z) / 3.0 < 0.5 {
                    0.25
                } else {
                    1.0
                }
            })
            .collect();
        let opts = SurfelOptions {
            count: 500,
            triangle_weights: Some(w),
            adaptive_density: true,
            ..Default::default()
        };
        let cloud = progressive_surfel_cloud(&m, &opts).unwrap();
        let (lo, hi): (Vec<&Surfel>, Vec<&Surfel>) =
            cloud.surfels.iter().partition(|s| s.position.z < 0.5);
        assert!((lo[0].radius / hi[0].radius - 2.0).abs() < 1e-4);
        assert!(lo.len() * 3 < hi.len(), "{} vs {}", lo.len(), hi.len());
        // No holes on either half (interior probes), even where surfels are sparse.
        for k in 0..900 {
            let p = Vec3::new(0.08 + 0.84 * (k % 30) as f32 / 29.0, 0.0, 0.08 + 0.84 * (k / 30) as f32 / 29.0);
            assert!(cloud.surfels.iter().any(|x| (x.position - p).length() <= x.radius), "hole at {p}");
        }
        // The stream keeps the radii (one byte each, within 4.4%).
        assert!(cloud.has_varying_radii());
        let bytes = cloud.encode(&[], false).unwrap();
        assert_eq!(bytes.len(), 45 + 500 * 11);
        let d = SurfelCloud::decode(&bytes).unwrap();
        for (a, b) in cloud.surfels.iter().zip(&d.cloud.surfels) {
            assert!((b.radius / a.radius - 1.0).abs() < 0.023, "{} vs {}", a.radius, b.radius);
        }
    }

    #[test]
    fn zero_weight_triangles_get_no_surfels() {
        let m = plane(8);
        // Hide the half with z < 0.5.
        let w: Vec<f32> = (0..m.triangle_count())
            .map(|t| {
                let [a, b, c] = m.triangle_positions(t);
                if (a.z + b.z + c.z) / 3.0 < 0.5 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let s = surfel_cloud(
            &m,
            &SurfelOptions {
                count: 100,
                triangle_weights: Some(w),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.iter().all(|x| x.position.z >= 0.5 - 1e-6));
    }
}
