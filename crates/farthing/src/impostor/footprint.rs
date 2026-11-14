//! Surfel footprints: the patch of surface each surfel stands for, measured from the mesh.
//!
//! A surfel is placed at one point, but it is drawn to cover a whole patch of surface: its
//! cell in the Voronoi diagram of the surfels over the surface. [`surfel_footprints`] measures
//! that patch, so a disc or a Gaussian splat can take its shape from the geometry instead of
//! a fixed round size (no training, everything comes from the mesh):
//!
//! * **Shape.** Dense surface samples (area-weighted, stratified per triangle) go to their
//!   nearest surfel. The 2D second moment of a surfel's samples in its tangent plane gives two
//!   tangent sigmas and the direction of the major axis, so a leaf strip gives a stretched
//!   footprint along the strip. The aspect ratio is clamped
//!   ([`FootprintOptions::max_aspect`]).
//! * **Coverage.** The assigned area projected onto the surfel plane, divided by the area of
//!   the uniformly filled ellipse with those second moments (4π σ₁σ₂), clamped to 1. A solid
//!   patch scores about 1 (0.99 for a hexagonal cell, 0.95 for a square). A coarse level of
//!   sparse foliage, where one surfel's cell spans several leaves and the gaps between them,
//!   scores the fraction actually covered, and the splat becomes that transparent instead of a
//!   solid blob.
//! * **Averaged attributes.** The normal is the area-weighted mean over the patch (sample
//!   normals flipped into the surfel normal's hemisphere first, so double-sided leaves made of
//!   two opposite sheets don't cancel out). Its spread, `1 − |mean normal|`, is 0 for a flat
//!   patch and grows with curvature or clutter: use it as normal variance (roughness) in the
//!   G-buffer and as thickness for a splat. Host channels (albedo, …) can be averaged over the
//!   patch with [`average_surfel_channel`]: farthing hands the host batches of sample points
//!   (triangle and barycentrics) to evaluate its own materials at, and reduces the results per
//!   surfel with the area weights. farthing never decides what the values mean.
//!
//! Any set of surfels works, in particular a prefix of a progressive
//! [`SurfelCloud`](super::surfels::SurfelCloud): fewer surfels have larger cells, so a
//! coarser prefix gets larger footprints and its own coverage. Pass the prefix's radius scale
//! (from [`SurfelCloud::lod`](super::surfels::SurfelCloud::lod)) or use
//! [`SurfelCloud::footprints`](super::surfels::SurfelCloud::footprints).
//!
//! Memory stays O(surfels): samples are generated on the fly, triangle by triangle, assigned
//! and accumulated immediately. Host evaluation goes in batches of at most [`SAMPLE_BATCH`].
//! Results are deterministic for a given seed.

use super::surfels::{Surfel, SurfelChannel};
use crate::{Error, Mesh, Result};
use glam::{DVec3, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How surface samples are drawn and footprints are fitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootprintOptions {
    /// Surface samples per surfel, on average (every triangle with area gets at least one).
    pub samples_per_surfel: u32,
    /// Largest ratio between the two tangent sigmas; thinner patches get a wider minor axis.
    pub max_aspect: f32,
    /// A sample is assigned to its nearest surfel only within this many drawn radii
    /// (`surfel.radius × radius_scale`); farther surface is left out.
    pub max_distance: f32,
    pub seed: u64,
    /// Optional per-triangle weight in [0, 1], as in
    /// [`SurfelOptions::triangle_weights`](super::surfels::SurfelOptions::triangle_weights):
    /// samples count area × weight, so hidden surface adds no coverage.
    #[serde(skip)]
    pub triangle_weights: Option<Vec<f32>>,
}

impl Default for FootprintOptions {
    fn default() -> Self {
        Self {
            samples_per_surfel: 64,
            max_aspect: 8.0,
            max_distance: 2.0,
            seed: 1,
            triangle_weights: None,
        }
    }
}

/// The measured patch of one surfel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfelFootprint {
    /// Area-weighted centroid of the patch (the splat centre).
    pub center: Vec3,
    /// Unit area-weighted mean normal (the surfel's own normal if the patch is empty or its
    /// normals cancel).
    pub normal: Vec3,
    /// Unit major axis in the tangent plane (perpendicular to `normal`).
    pub tangent: Vec3,
    /// Standard deviations of the patch along `tangent` and the bitangent (x ≥ y), after the
    /// aspect clamp. A uniformly filled ellipse with these moments has semi-axes 2σ.
    pub sigma: Vec2,
    /// Standard deviation across the plane: the larger of the measured spread of the samples
    /// along the normal and `normal_spread × √(σx σy)`, at most σx.
    pub thickness: f32,
    /// `1 − |mean normal|` in [0, 1]: 0 for a flat patch. Normal variance, for roughness.
    pub normal_spread: f32,
    /// Fraction of the footprint ellipse covered by surface, in [0, 1] (the splat's opacity).
    pub coverage: f32,
    /// Surface area assigned to the surfel (× triangle weight). Reconstructed from the other
    /// fields after decoding a stream, so approximate there.
    pub area: f32,
}

impl SurfelFootprint {
    /// `normal × tangent`: the minor axis, so (tangent, bitangent, normal) is right-handed.
    pub fn bitangent(&self) -> Vec3 {
        self.normal.cross(self.tangent)
    }

    /// Semi-axes of the uniformly filled ellipse with the patch's second moments (2σ). Grow
    /// them by the cloud's coverage factor to draw overlapping discs, as for the radius.
    pub fn semi_axes(&self) -> Vec2 {
        2.0 * self.sigma
    }

    /// σx ⁄ σy (≥ 1).
    pub fn aspect(&self) -> f32 {
        self.sigma.x / self.sigma.y.max(f32::MIN_POSITIVE)
    }
}

/// Orthonormal tangent basis (t, b) around unit `n` with t × b = n (Duff et al., *Building an
/// Orthonormal Basis, Revisited*, JCGT 2017). The reference surfel shader builds the same
/// basis, and the stream stores footprint orientation as an angle from `t`.
pub fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// A point on the source surface handed to the host for material evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    /// Index of the surfel (in the slice passed in) the sample belongs to.
    pub surfel: u32,
    pub triangle: u32,
    pub barycentrics: Vec3,
    pub position: Vec3,
}

/// Largest batch [`average_surfel_channel`] hands to the host's evaluator at once.
pub const SAMPLE_BATCH: usize = 4096;

/// A channel to average over each surfel's patch.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSpec {
    pub name: String,
    /// Values per sample (1..=16).
    pub components: u8,
    /// The last component is coverage alpha (alpha-tested leaves): the other components are
    /// averaged weighted by it, so transparent texels don't bleed their colour, and the alpha
    /// itself is the plain area-weighted mean.
    pub alpha_weighted: bool,
}

/// Weighted area of triangle `t`.
fn weighted_area(mesh: &Mesh, weights: Option<&[f32]>, t: usize) -> f64 {
    let [a, b, c] = mesh.triangle_positions(t);
    let w = weights.map_or(1.0, |w| w.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0)) as f64;
    0.5 * (b - a).cross(c - a).length() as f64 * w
}

fn hash01(mut z: u64) -> f64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

fn cell_key(p: Vec3, cell: f32) -> (i32, i32, i32) {
    (
        (p.x / cell).floor() as i32,
        (p.y / cell).floor() as i32,
        (p.z / cell).floor() as i32,
    )
}

/// Nearest-surfel lookup on a uniform grid, searching outward ring by ring. The hash map is
/// only probed, never iterated, so its order cannot reach the results.
struct Nearest<'a> {
    surfels: &'a [Surfel],
    cell: f32,
    map: HashMap<(i32, i32, i32), Vec<u32>>,
    cutoff2: Vec<f32>,
    max_ring: i32,
}

impl<'a> Nearest<'a> {
    fn new(surfels: &'a [Surfel], radius_scale: f32, max_distance: f32, area: f64) -> Self {
        let cutoff: Vec<f32> = surfels
            .iter()
            .map(|s| (s.radius * radius_scale * max_distance).max(0.0))
            .collect();
        let max_cut = cutoff.iter().copied().fold(0.0f32, f32::max);
        // Cell size: about one surfel per cell on a flat surface, and at most 17 rings out to
        // the largest cutoff.
        let spacing = (area / surfels.len().max(1) as f64).sqrt() as f32;
        let cell = spacing.max(max_cut / 16.0).max(1e-12);
        let mut map: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, s) in surfels.iter().enumerate() {
            map.entry(cell_key(s.position, cell))
                .or_default()
                .push(i as u32);
        }
        Nearest {
            surfels,
            cell,
            map,
            cutoff2: cutoff.iter().map(|c| c * c).collect(),
            max_ring: ((max_cut / cell).ceil() as i32 + 1).clamp(1, 17),
        }
    }

    /// Nearest surfel within its cutoff (ties to the lower index).
    fn find(&self, p: Vec3) -> Option<u32> {
        let (cx, cy, cz) = cell_key(p, self.cell);
        let mut best = (f32::INFINITY, u32::MAX);
        for ring in 0..=self.max_ring {
            for dz in -ring..=ring {
                for dy in -ring..=ring {
                    let edge = dz.abs() == ring || dy.abs() == ring;
                    let step = if edge { 1 } else { 2 * ring.max(1) };
                    let mut dx = -ring;
                    while dx <= ring {
                        if let Some(list) = self.map.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in list {
                                let d2 = (self.surfels[j as usize].position - p).length_squared();
                                if d2 < best.0 || (d2 == best.0 && j < best.1) {
                                    best = (d2, j);
                                }
                            }
                        }
                        dx += step;
                    }
                }
            }
            // Every cell of the next ring is at least `ring` cells away.
            let reach = ring as f32 * self.cell;
            if best.0 <= reach * reach {
                break;
            }
        }
        (best.1 != u32::MAX && best.0 <= self.cutoff2[best.1 as usize]).then_some(best.1)
    }
}

/// Streams area-weighted surface samples, each assigned to its nearest surfel:
/// `f(sample, unit normal, area it stands for)`. Returns the total weighted area.
fn for_each_assigned_sample(
    mesh: &Mesh,
    surfels: &[Surfel],
    radius_scale: f32,
    opts: &FootprintOptions,
    mut f: impl FnMut(&SurfaceSample, Vec3, f64),
) -> Result<f64> {
    mesh.validate()?;
    let weights = opts.triangle_weights.as_deref();
    let nt = mesh.triangle_count();
    let total: f64 = (0..nt).map(|t| weighted_area(mesh, weights, t)).sum();
    if total <= 0.0 {
        return Err(Error::InvalidInput(
            "mesh has no (weighted) surface area".into(),
        ));
    }
    if !(radius_scale > 0.0 && radius_scale.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "footprint radius scale must be positive, got {radius_scale}"
        )));
    }
    if surfels.is_empty() {
        return Ok(total);
    }
    let grid = Nearest::new(surfels, radius_scale, opts.max_distance, total);
    let density = opts.samples_per_surfel.max(1) as f64 * surfels.len() as f64 / total;
    // R2 low-discrepancy sequence (Roberts 2018), rotated per triangle.
    const A1: f64 = 0.754_877_666_246_692_7;
    const A2: f64 = 0.569_840_290_998_053_3;
    let seed = opts.seed ^ 0x5851_F42D_4C95_7F2D;
    for t in 0..nt {
        let a = weighted_area(mesh, weights, t);
        if a <= 0.0 {
            continue;
        }
        let count = ((a * density).round() as usize).clamp(1, 1 << 24);
        let w = a / count as f64;
        let h = seed ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (ox, oy) = (hash01(h), hash01(h ^ 0xA076_1D64_78BD_642F));
        let [p0, p1, p2] = mesh.triangle_positions(t);
        let face = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let tri = mesh.triangle(t);
        for i in 0..count {
            let (mut u, mut v) = (
                (ox + i as f64 * A1).fract() as f32,
                (oy + i as f64 * A2).fract() as f32,
            );
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let bary = Vec3::new(1.0 - u - v, u, v);
            let p = p0 * bary.x + p1 * bary.y + p2 * bary.z;
            let Some(s) = grid.find(p) else { continue };
            let n = match &mesh.normals {
                Some(nn) => (nn[tri[0] as usize] * bary.x
                    + nn[tri[1] as usize] * bary.y
                    + nn[tri[2] as usize] * bary.z)
                    .try_normalize()
                    .unwrap_or(face),
                None => face,
            };
            f(
                &SurfaceSample {
                    surfel: s,
                    triangle: t as u32,
                    barycentrics: bary,
                    position: p,
                },
                n,
                w,
            );
        }
    }
    Ok(total)
}

#[derive(Clone, Copy, Default)]
struct Moments {
    w: f64,
    /// Σ w·d, d = sample − surfel position.
    d: DVec3,
    /// Σ w·d dᵀ: xx, xy, xz, yy, yz, zz.
    dd: [f64; 6],
    /// Σ w·n, n flipped into the surfel normal's hemisphere.
    n: DVec3,
}

/// Footprints of `surfels` over `mesh` (one per surfel, in order). `radius_scale` is the scale
/// the surfels are drawn at (1 for a whole cloud; [`SurfelCloud::lod`]'s scale for a prefix);
/// it only bounds how far a sample may be from its surfel ([`FootprintOptions::max_distance`]).
/// A surfel that receives no surface keeps its point and normal with coverage 0.
///
/// [`SurfelCloud::lod`]: super::surfels::SurfelCloud::lod
pub fn surfel_footprints(
    mesh: &Mesh,
    surfels: &[Surfel],
    radius_scale: f32,
    opts: &FootprintOptions,
) -> Result<Vec<SurfelFootprint>> {
    let mut acc = vec![Moments::default(); surfels.len()];
    for_each_assigned_sample(mesh, surfels, radius_scale, opts, |smp, n, w| {
        let s = &surfels[smp.surfel as usize];
        let m = &mut acc[smp.surfel as usize];
        let d = (smp.position - s.position).as_dvec3();
        let n = if n.dot(s.normal) < 0.0 { -n } else { n };
        m.w += w;
        m.d += d * w;
        m.dd[0] += w * d.x * d.x;
        m.dd[1] += w * d.x * d.y;
        m.dd[2] += w * d.x * d.z;
        m.dd[3] += w * d.y * d.y;
        m.dd[4] += w * d.y * d.z;
        m.dd[5] += w * d.z * d.z;
        m.n += n.as_dvec3() * w;
    })?;
    Ok(surfels
        .iter()
        .zip(&acc)
        .map(|(s, m)| fit(s, m, radius_scale, opts.max_aspect))
        .collect())
}

fn fit(s: &Surfel, m: &Moments, radius_scale: f32, max_aspect: f32) -> SurfelFootprint {
    let r = (s.radius * radius_scale).max(1e-12);
    let own_normal = s.normal.try_normalize().unwrap_or(Vec3::Z);
    if m.w <= 0.0 {
        let (t, _) = tangent_basis(own_normal);
        let sg = 0.25 * r;
        return SurfelFootprint {
            center: s.position,
            normal: own_normal,
            tangent: t,
            sigma: Vec2::splat(sg),
            thickness: 0.0,
            normal_spread: 0.0,
            coverage: 0.0,
            area: 0.0,
        };
    }
    let mu = m.d / m.w;
    let nm = m.n / m.w;
    let len = nm.length().min(1.0);
    let normal = if len > 1e-6 {
        (nm / len).as_vec3().normalize()
    } else {
        own_normal
    };
    // Covariance about the centroid.
    let c = [
        m.dd[0] / m.w - mu.x * mu.x,
        m.dd[1] / m.w - mu.x * mu.y,
        m.dd[2] / m.w - mu.x * mu.z,
        m.dd[3] / m.w - mu.y * mu.y,
        m.dd[4] / m.w - mu.y * mu.z,
        m.dd[5] / m.w - mu.z * mu.z,
    ];
    let quad = |a: DVec3, b: DVec3| {
        a.x * (c[0] * b.x + c[1] * b.y + c[2] * b.z)
            + a.y * (c[1] * b.x + c[3] * b.y + c[4] * b.z)
            + a.z * (c[2] * b.x + c[4] * b.y + c[5] * b.z)
    };
    let (t0, b0) = tangent_basis(normal);
    let (t0, b0, nd) = (t0.as_dvec3(), b0.as_dvec3(), normal.as_dvec3());
    let (cxx, cxy, cyy) = (quad(t0, t0), quad(t0, b0), quad(b0, b0));
    let half = 0.5 * (cxx + cyy);
    let disc = (0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy).sqrt();
    let (l1, l2) = ((half + disc).max(0.0), (half - disc).max(0.0));
    let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let tangent = (t0 * angle.cos() + b0 * angle.sin()).as_vec3();
    let tangent = (tangent - normal * tangent.dot(normal))
        .try_normalize()
        .unwrap_or(t0.as_vec3());
    let sx = (l1.sqrt() as f32).max(r * 1e-3);
    let sy = (l2.sqrt() as f32).max(sx / max_aspect.max(1.0));
    let spread = (1.0 - len) as f32;
    let measured = quad(nd, nd).max(0.0).sqrt() as f32;
    let thickness = measured.max(spread * (sx * sy).sqrt()).min(sx);
    // Samples were flipped into one hemisphere, so |Σ w n| is their area projected onto the
    // plane (exact unless some face away from the mean by more than 90°).
    let projected = (m.w * len) as f32;
    let ellipse = 4.0 * std::f32::consts::PI * sx * sy;
    SurfelFootprint {
        center: s.position + mu.as_vec3(),
        normal,
        tangent,
        sigma: Vec2::new(sx, sy),
        thickness,
        normal_spread: spread.clamp(0.0, 1.0),
        coverage: (projected / ellipse).clamp(0.0, 1.0),
        area: m.w as f32,
    }
}

/// Averages a host-evaluated channel over each surfel's patch (the same samples and
/// assignment as [`surfel_footprints`] with the same options). `eval(samples, out)` fills
/// `out` with `spec.components` values per sample (linear 0..1, clamped when quantized), for
/// batches of at most [`SAMPLE_BATCH`] samples. A surfel that receives no samples is
/// evaluated at its own source point, as the single-point path does. Returns unorm8 values in
/// surfel order, ready for [`SurfelCloud::encode`](super::surfels::SurfelCloud::encode) or
/// [`surfels_to_splats`](super::splats::surfels_to_splats).
pub fn average_surfel_channel<F>(
    mesh: &Mesh,
    surfels: &[Surfel],
    radius_scale: f32,
    opts: &FootprintOptions,
    spec: &ChannelSpec,
    mut eval: F,
) -> Result<SurfelChannel>
where
    F: FnMut(&[SurfaceSample], &mut [f32]),
{
    let k = spec.components as usize;
    if !(1..=16).contains(&k) || spec.name.len() > 255 {
        return Err(Error::InvalidInput(format!(
            "channel `{}` needs 1 to 16 components and a name under 256 bytes",
            spec.name
        )));
    }
    if let Some(s) = surfels
        .iter()
        .find(|s| s.triangle as usize >= mesh.triangle_count())
    {
        return Err(Error::InvalidInput(format!(
            "surfel source triangle {} is out of range",
            s.triangle
        )));
    }
    let n = surfels.len();
    let mut red = Reducer {
        k,
        alpha: spec.alpha_weighted && k >= 2,
        wsum: vec![0.0; n],
        wa: vec![0.0; n],
        sums: vec![0.0; n * k],
        cap: SAMPLE_BATCH.min(n.saturating_mul(64)).max(1),
        batch: Vec::new(),
        bw: Vec::new(),
        vals: Vec::new(),
    };
    red.batch.reserve_exact(red.cap);
    red.bw.reserve_exact(red.cap);
    for_each_assigned_sample(mesh, surfels, radius_scale, opts, |smp, _, w| {
        red.batch.push(*smp);
        red.bw.push(w);
        if red.batch.len() >= red.cap {
            red.flush(&mut eval);
        }
    })?;
    red.flush(&mut eval);
    // Single-point fallback for surfels without samples (and, for alpha-weighted channels,
    // the colour of surfels whose patch is fully transparent).
    let lonely: Vec<usize> = (0..n)
        .filter(|&i| red.wsum[i] <= 0.0 || (red.alpha && red.wa[i] <= 0.0))
        .collect();
    let mut point = vec![0.0f32; lonely.len() * k];
    for (c, chunk) in lonely.chunks(SAMPLE_BATCH).enumerate() {
        let samples: Vec<SurfaceSample> = chunk
            .iter()
            .map(|&i| {
                let s = &surfels[i];
                SurfaceSample {
                    surfel: i as u32,
                    triangle: s.triangle,
                    barycentrics: s.barycentrics,
                    position: s.position,
                }
            })
            .collect();
        let at = c * SAMPLE_BATCH * k;
        eval(&samples, &mut point[at..at + chunk.len() * k]);
    }
    let q = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut data = Vec::with_capacity(n * k);
    let mut next_lonely = 0;
    for i in 0..n {
        let pv = if lonely.get(next_lonely) == Some(&i) {
            next_lonely += 1;
            Some(&point[(next_lonely - 1) * k..next_lonely * k])
        } else {
            None
        };
        let s = &red.sums[i * k..(i + 1) * k];
        match pv {
            Some(v) if red.wsum[i] <= 0.0 => data.extend(v.iter().map(|&x| q(x as f64))),
            _ if red.alpha => {
                for c in 0..k - 1 {
                    data.push(q(match pv {
                        Some(v) => v[c] as f64,
                        None => s[c] / red.wa[i],
                    }));
                }
                data.push(q(red.wa[i] / red.wsum[i]));
            }
            _ => data.extend(s.iter().map(|&x| q(x / red.wsum[i]))),
        }
    }
    Ok(SurfelChannel {
        name: spec.name.clone(),
        components: spec.components,
        data,
    })
}

/// Per-surfel weighted sums of host values, fed in batches.
struct Reducer {
    k: usize,
    alpha: bool,
    /// Σ w per surfel.
    wsum: Vec<f64>,
    /// Σ w·α per surfel (alpha-weighted channels).
    wa: Vec<f64>,
    /// Σ w·value (× α for the weighted components) per surfel and component.
    sums: Vec<f64>,
    cap: usize,
    batch: Vec<SurfaceSample>,
    bw: Vec<f64>,
    vals: Vec<f32>,
}

impl Reducer {
    fn flush<F: FnMut(&[SurfaceSample], &mut [f32])>(&mut self, eval: &mut F) {
        if self.batch.is_empty() {
            return;
        }
        let k = self.k;
        self.vals.clear();
        self.vals.resize(self.batch.len() * k, 0.0);
        eval(&self.batch, &mut self.vals);
        for (j, (smp, &w)) in self.batch.iter().zip(&self.bw).enumerate() {
            let s = smp.surfel as usize;
            let v = &self.vals[j * k..(j + 1) * k];
            let acc = &mut self.sums[s * k..(s + 1) * k];
            self.wsum[s] += w;
            if self.alpha {
                let a = w * v[k - 1].clamp(0.0, 1.0) as f64;
                self.wa[s] += a;
                for c in 0..k - 1 {
                    acc[c] += a * v[c] as f64;
                }
            } else {
                for c in 0..k {
                    acc[c] += w * v[c] as f64;
                }
            }
        }
        self.batch.clear();
        self.bw.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impostor::splats::{surfels_to_splats, SplatOptions};
    use crate::impostor::surfels::{progressive_surfel_cloud, surfel_cloud, SurfelOptions};
    use glam::Vec4;

    /// Grid of quads over [0, w] × [0, h] in the XZ plane (normal +Y).
    fn sheet(w: f32, h: f32, nx: u32, nz: u32) -> Mesh {
        let mut pos = Vec::new();
        for j in 0..=nz {
            for i in 0..=nx {
                pos.push(Vec3::new(
                    w * i as f32 / nx as f32,
                    0.0,
                    h * j as f32 / nz as f32,
                ));
            }
        }
        let mut idx = Vec::new();
        for j in 0..nz {
            for i in 0..nx {
                let a = j * (nx + 1) + i;
                idx.extend_from_slice(&[a, a + nx + 1, a + 1, a + 1, a + nx + 1, a + nx + 2]);
            }
        }
        Mesh::new(pos, idx)
    }

    fn surfel_at(p: Vec3, r: f32) -> Surfel {
        Surfel {
            position: p,
            normal: Vec3::Y,
            radius: r,
            triangle: 0,
            barycentrics: Vec3::new(1.0, 0.0, 0.0),
        }
    }

    #[test]
    fn tangent_basis_is_orthonormal_and_right_handed() {
        for n in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(0.3, -0.5, 0.1).normalize(),
            Vec3::new(-0.2, 0.1, -0.9).normalize(),
        ] {
            let (t, b) = tangent_basis(n);
            assert!((t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5);
            assert!(t.dot(n).abs() < 1e-5 && b.dot(n).abs() < 1e-5 && t.dot(b).abs() < 1e-5);
            assert!(t.cross(b).dot(n) > 0.9999);
        }
    }

    /// A long thin strip with surfels spaced along it: each patch is a 0.125 × 0.02 rectangle,
    /// so its footprint (and splat) is stretched along the strip, at any orientation.
    #[test]
    fn strip_gives_anisotropic_footprints_along_it() {
        for deg in [0.0f32, 30.0, 90.0] {
            let rot = glam::Quat::from_rotation_y(deg.to_radians());
            let mut m = sheet(1.0, 0.02, 40, 2);
            m.positions.iter_mut().for_each(|p| *p = rot * *p);
            let along = rot * Vec3::X;
            let s: Vec<Surfel> = (0..8)
                .map(|i| surfel_at(rot * Vec3::new(0.0625 + 0.125 * i as f32, 0.0, 0.01), 0.15))
                .collect();
            // The mesh has only ~3 default samples per (tiny) triangle; use more so the
            // second moments of each rectangle are measured closely.
            let opts = FootprintOptions {
                samples_per_surfel: 512,
                ..Default::default()
            };
            let f = surfel_footprints(&m, &s, 1.0, &opts).unwrap();
            for fp in &f {
                assert!(fp.tangent.dot(along).abs() > 0.99, "{deg}°: {fp:?}");
                let aspect = fp.aspect();
                assert!((aspect - 6.25).abs() < 0.6, "{deg}°: aspect {aspect}");
                assert!((fp.sigma.x - 0.125 / 12f32.sqrt()).abs() < 0.003);
                // A filled rectangle covers 3 ⁄ π of its moment ellipse.
                assert!((fp.coverage - 0.955).abs() < 0.03, "{}", fp.coverage);
                assert!((fp.area - 0.0025).abs() < 2e-4, "{}", fp.area);
                assert!(fp.normal.dot(Vec3::Y) > 0.9999 && fp.normal_spread < 1e-4);
            }
            let sp = surfels_to_splats(&s, Some(&f), None, &SplatOptions::default()).unwrap();
            for (x, fp) in sp.iter().zip(&f) {
                assert!((x.rotation * Vec3::X).dot(along).abs() > 0.99);
                assert!((x.rotation * Vec3::Z).dot(Vec3::Y).abs() > 0.9999);
                assert!(x.sigma.x > 5.0 * x.sigma.y && x.sigma.z < x.sigma.y);
                assert!((x.opacity - 0.98 * fp.coverage).abs() < 1e-5);
                assert!((x.position - fp.center).length() < 1e-6);
            }
        }
        // Aspect clamp: a 16:1 patch is widened to 8:1.
        let m = sheet(1.0, 0.02, 40, 2);
        let s = [0.25f32, 0.75].map(|x| surfel_at(Vec3::new(x, 0.0, 0.01), 0.5));
        let f = surfel_footprints(&m, &s, 1.0, &FootprintOptions::default()).unwrap();
        for fp in &f {
            assert!((fp.aspect() - 8.0).abs() < 1e-3);
        }
    }

    /// Strips half as wide as their period: a coarse surfel's patch spans several strips and
    /// the gaps between them, so it covers about half its footprint.
    #[test]
    fn half_covered_footprint_is_half_opaque() {
        let mut m = Mesh::default();
        for k in 0..25 {
            let mut strip = sheet(1.0, 0.02, 10, 1);
            strip
                .positions
                .iter_mut()
                .for_each(|p| p.z += 0.04 * k as f32);
            m.append(&strip, 0);
        }
        let solid = sheet(1.0, 1.0, 8, 8);
        let opts = SurfelOptions {
            count: 40,
            ..Default::default()
        };
        let mean_interior = |mesh: &Mesh| {
            let s = surfel_cloud(mesh, &opts).unwrap();
            let f = surfel_footprints(mesh, &s, 1.0, &FootprintOptions::default()).unwrap();
            let inner: Vec<f32> = s
                .iter()
                .zip(&f)
                .filter(|(s, _)| s.position.x > 0.2 && s.position.x < 0.8)
                .map(|(_, f)| f.coverage)
                .collect();
            inner.iter().sum::<f32>() / inner.len() as f32
        };
        let half = mean_interior(&m);
        let full = mean_interior(&solid);
        assert!((half - 0.5).abs() < 0.07, "striped coverage {half}");
        assert!(full > 0.93, "solid coverage {full}");
    }

    /// A corrugated sheet (facets at ±30°) has spread normals: higher normal spread
    /// (roughness) and thicker footprints than a flat sheet.
    #[test]
    fn spread_normals_raise_roughness() {
        let n = 40u32;
        let h = (30f32.to_radians()).tan() / n as f32;
        let mut flat = sheet(1.0, 1.0, n, 4);
        let mut wavy = flat.clone();
        for (i, p) in wavy.positions.iter_mut().enumerate() {
            p.y = if (i as u32 % (n + 1)) % 2 == 1 {
                h
            } else {
                0.0
            };
        }
        // Smooth normals on the flat sheet must give zero spread too.
        flat.normals = Some(vec![Vec3::Y; flat.positions.len()]);
        let opts = SurfelOptions {
            count: 30,
            ..Default::default()
        };
        let stats = |m: &Mesh| {
            let s = surfel_cloud(m, &opts).unwrap();
            let f = surfel_footprints(m, &s, 1.0, &FootprintOptions::default()).unwrap();
            let k = f.len() as f32;
            (
                f.iter().map(|f| f.normal_spread).sum::<f32>() / k,
                f.iter().map(|f| f.thickness / f.sigma.y).sum::<f32>() / k,
                f.iter().map(|f| f.normal.y).fold(1.0f32, f32::min),
            )
        };
        let (sf, tf, _) = stats(&flat);
        let (sw, tw, ny) = stats(&wavy);
        assert!(sf < 1e-4, "flat spread {sf}");
        // 1 − cos 30° = 0.134.
        assert!((sw - 0.134).abs() < 0.03, "corrugated spread {sw}");
        assert!(tw > 0.1 && tw > 10.0 * tf, "thickness {tw} vs {tf}");
        assert!(ny > 0.97, "mean normals stay up: {ny}");
    }

    #[test]
    fn double_sided_sheets_do_not_cancel() {
        let mut m = sheet(1.0, 1.0, 4, 4);
        let back = {
            let mut b = m.clone();
            for t in b.indices.chunks_exact_mut(3) {
                t.swap(1, 2);
            }
            b
        };
        m.append(&back, 0);
        let s = surfel_cloud(
            &m,
            &SurfelOptions {
                count: 20,
                ..Default::default()
            },
        )
        .unwrap();
        let f = surfel_footprints(&m, &s, 1.0, &FootprintOptions::default()).unwrap();
        for fp in &f {
            assert!(fp.normal_spread < 1e-4 && fp.normal.y.abs() > 0.9999);
        }
    }

    #[test]
    fn coarser_prefixes_get_larger_footprints() {
        let m = sheet(1.0, 1.0, 8, 8);
        let cloud = progressive_surfel_cloud(
            &m,
            &SurfelOptions {
                count: 400,
                ..Default::default()
            },
        )
        .unwrap();
        let opts = FootprintOptions::default();
        let mean = |f: &[SurfelFootprint]| {
            let k = f.len() as f32;
            (
                f.iter()
                    .map(|f| (f.sigma.x * f.sigma.y).sqrt())
                    .sum::<f32>()
                    / k,
                f.iter().map(|f| f.coverage).sum::<f32>() / k,
                f.iter().map(|f| f.area).sum::<f32>(),
            )
        };
        let (s_all, c_all, a_all) = mean(&cloud.footprints(&m, 400, &opts).unwrap());
        let (s_q, c_q, a_q) = mean(&cloud.footprints(&m, 100, &opts).unwrap());
        assert!((s_q / s_all - 2.0).abs() < 0.2, "{s_q} vs {s_all}");
        assert!(c_all > 0.9 && c_q > 0.9, "{c_all} {c_q}");
        // Every bit of the plane is assigned at both levels.
        assert!((a_all - 1.0).abs() < 1e-3 && (a_q - 1.0).abs() < 1e-3);
    }

    #[test]
    fn footprints_and_channels_are_deterministic() {
        let mut m = sheet(1.0, 1.0, 8, 8);
        m.colors = Some(
            m.positions
                .iter()
                .map(|p| Vec4::new(p.x, p.z, 0.5, 1.0))
                .collect(),
        );
        let s = surfel_cloud(
            &m,
            &SurfelOptions {
                count: 100,
                ..Default::default()
            },
        )
        .unwrap();
        let opts = FootprintOptions::default();
        let a = surfel_footprints(&m, &s, 1.0, &opts).unwrap();
        let b = surfel_footprints(&m, &s, 1.0, &opts).unwrap();
        assert_eq!(a, b);
        let spec = ChannelSpec {
            name: "color".into(),
            components: 4,
            alpha_weighted: true,
        };
        let colors = m.colors.clone().unwrap();
        let eval = |smp: &[SurfaceSample], out: &mut [f32]| {
            for (x, o) in smp.iter().zip(out.chunks_exact_mut(4)) {
                let t = m.triangle(x.triangle as usize);
                let c = colors[t[0] as usize] * x.barycentrics.x
                    + colors[t[1] as usize] * x.barycentrics.y
                    + colors[t[2] as usize] * x.barycentrics.z;
                o.copy_from_slice(&c.to_array());
            }
        };
        let c1 = average_surfel_channel(&m, &s, 1.0, &opts, &spec, eval).unwrap();
        let c2 = average_surfel_channel(&m, &s, 1.0, &opts, &spec, eval).unwrap();
        assert_eq!(c1, c2);
        // The patch average of a linear gradient is the gradient at the patch centroid.
        for (i, fp) in a.iter().enumerate() {
            let px = &c1.data[i * 4..i * 4 + 4];
            assert!(
                (px[0] as f32 / 255.0 - fp.center.x).abs() < 0.01,
                "{px:?} {fp:?}"
            );
            assert!((px[1] as f32 / 255.0 - fp.center.z).abs() < 0.01);
            assert_eq!(px[3], 255);
        }
    }

    /// Alpha-tested texels: the colour average ignores transparent texels, the alpha is the
    /// covered fraction.
    #[test]
    fn alpha_weighted_channels_skip_transparent_texels() {
        let m = sheet(1.0, 1.0, 8, 8);
        let s = [surfel_at(Vec3::new(0.5, 0.0, 0.5), 2.0)];
        let spec = ChannelSpec {
            name: "albedo".into(),
            components: 4,
            alpha_weighted: true,
        };
        let mut calls = 0;
        let c = average_surfel_channel(
            &m,
            &s,
            1.0,
            &FootprintOptions::default(),
            &spec,
            |smp, out| {
                calls += 1;
                assert!(smp.len() <= SAMPLE_BATCH);
                for (x, o) in smp.iter().zip(out.chunks_exact_mut(4)) {
                    let v = if x.position.x < 0.25 {
                        [1.0, 0.0, 0.0, 1.0]
                    } else {
                        [0.0, 1.0, 0.0, 0.0]
                    };
                    o.copy_from_slice(&v);
                }
            },
        )
        .unwrap();
        assert!(calls >= 1);
        assert_eq!(&c.data[..3], &[255, 0, 0]);
        assert!(
            (c.data[3] as f32 / 255.0 - 0.25).abs() < 0.02,
            "{}",
            c.data[3]
        );
    }
}
