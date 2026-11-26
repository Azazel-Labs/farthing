//! Quality metrics: surface distance between meshes and image difference.

use crate::bvh::Bvh;
use crate::image::Image;
use crate::math::triangle_area;
use crate::mesh::Mesh;
use crate::rng::Pcg32;
use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Area-weighted random points on the surface (deterministic for a given seed).
pub fn sample_surface(mesh: &Mesh, count: usize, seed: u64) -> Vec<Vec3> {
    sample_points(mesh, count, seed)
        .into_iter()
        .map(|s| s.position)
        .collect()
}

/// A point on a mesh: where it is, its triangle, and its barycentric weights there.
struct SurfacePoint {
    position: Vec3,
    triangle: u32,
    barycentrics: Vec3,
}

/// [`sample_surface`], keeping where each point lies.
fn sample_points(mesh: &Mesh, count: usize, seed: u64) -> Vec<SurfacePoint> {
    let tri_count = mesh.triangle_count();
    if tri_count == 0 || count == 0 {
        return Vec::new();
    }
    let mut cdf = Vec::with_capacity(tri_count);
    let mut total = 0.0f64;
    for t in 0..tri_count {
        let [a, b, c] = mesh.triangle_positions(t);
        total += triangle_area(a, b, c) as f64;
        cdf.push(total);
    }
    let mut rng = Pcg32::new(seed);
    (0..count)
        .map(|_| {
            let x = rng.f32() as f64 * total;
            let t = cdf.partition_point(|&c| c < x).min(tri_count - 1);
            let [a, b, c] = mesh.triangle_positions(t);
            let (mut u, mut v) = (rng.f32(), rng.f32());
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            SurfacePoint {
                position: a + (b - a) * u + (c - a) * v,
                triangle: t as u32,
                barycentrics: Vec3::new(1.0 - u - v, u, v),
            }
        })
        .collect()
}

/// Every distinct position a triangle corner of `mesh` sits at (unused vertices left out), in
/// a fixed order.
fn corner_points(mesh: &Mesh) -> Vec<Vec3> {
    let mut keys: Vec<[u32; 3]> = mesh
        .indices
        .iter()
        .map(|&i| mesh.positions[i as usize].to_array().map(f32::to_bits))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .map(|k| Vec3::from_array(k.map(f32::from_bits)))
        .collect()
}

/// Shading normal of `mesh` at a point of triangle `t`: the interpolated vertex normal, or the
/// face normal when the mesh has none. Zero when neither is defined.
fn shading_normal(mesh: &Mesh, t: u32, barycentrics: Vec3) -> Vec3 {
    let [a, b, c] = mesh.triangle(t as usize).map(|i| i as usize);
    if let Some(n) = &mesh.normals {
        let n = n[a] * barycentrics.x + n[b] * barycentrics.y + n[c] * barycentrics.z;
        if let Some(n) = n.try_normalize() {
            return n;
        }
    }
    let [pa, pb, pc] = mesh.triangle_positions(t as usize);
    (pb - pa).cross(pc - pa).normalize_or_zero()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceDistance {
    /// max(forward_max, backward_max) — symmetric Hausdorff distance (sampled).
    pub hausdorff: f32,
    /// Mean of both directions, over the area-weighted samples.
    pub mean: f32,
    /// Root mean square of both directions, over the area-weighted samples.
    pub rms: f32,
    /// Max distance from points on A to B: the samples and every vertex of A.
    pub forward_max: f32,
    /// Max distance from points on B to A: the samples and every vertex of B.
    pub backward_max: f32,
    /// Bounding-box diagonal of A, for relative numbers.
    pub diagonal: f32,
    /// hausdorff / diagonal.
    pub relative_hausdorff: f32,
    pub relative_mean: f32,
    /// Mean angle in degrees, both directions, between the shading normal at each
    /// area-weighted sample and the one at its closest point on the other mesh (interpolated
    /// vertex normals, or face normals; normal maps not applied).
    #[serde(default)]
    pub normal_mean_degrees: f32,
}

/// One direction: the largest distance over samples and corners, and per sample its distance
/// and normal angle (degrees; `None` where a normal is undefined).
struct OneWay {
    max: f32,
    distances: Vec<f32>,
    angles: Vec<Option<f32>>,
}

fn one_way(
    from: &Mesh,
    samples: &[SurfacePoint],
    corners: &[Vec3],
    to: &Mesh,
    target: &Bvh,
) -> OneWay {
    let (distances, angles): (Vec<f32>, Vec<Option<f32>>) = samples
        .par_iter()
        .map(|s| match target.closest_point(s.position, f32::INFINITY) {
            Some(c) => {
                let n0 = shading_normal(from, s.triangle, s.barycentrics);
                let n1 = shading_normal(to, c.triangle, c.barycentrics);
                let angle = (n0 != Vec3::ZERO && n1 != Vec3::ZERO)
                    .then(|| n0.dot(n1).clamp(-1.0, 1.0).acos().to_degrees());
                (c.distance, angle)
            }
            None => (f32::INFINITY, None),
        })
        .unzip();
    let corner_max = corners
        .par_iter()
        .map(|&p| {
            target
                .closest_point(p, f32::INFINITY)
                .map_or(f32::INFINITY, |c| c.distance)
        })
        .reduce(|| 0.0, f32::max);
    OneWay {
        max: distances.iter().copied().fold(corner_max, f32::max),
        distances,
        angles,
    }
}

fn summarize(fwd: OneWay, bwd: OneWay, diagonal: f32) -> SurfaceDistance {
    let all: Vec<f32> = fwd
        .distances
        .iter()
        .chain(&bwd.distances)
        .copied()
        .collect();
    let n = all.len().max(1) as f32;
    let mean = all.iter().sum::<f32>() / n;
    let rms = (all.iter().map(|d| d * d).sum::<f32>() / n).sqrt();
    let angles: Vec<f32> = fwd
        .angles
        .iter()
        .chain(&bwd.angles)
        .flatten()
        .copied()
        .collect();
    let normal_mean_degrees = angles.iter().sum::<f32>() / angles.len().max(1) as f32;
    let hausdorff = fwd.max.max(bwd.max);
    SurfaceDistance {
        hausdorff,
        mean,
        rms,
        forward_max: fwd.max,
        backward_max: bwd.max,
        diagonal,
        relative_hausdorff: hausdorff / diagonal,
        relative_mean: mean / diagonal,
        normal_mean_degrees,
    }
}

fn triangle_bvh(m: &Mesh) -> Bvh {
    Bvh::build(
        (0..m.triangle_count())
            .map(|t| m.triangle_positions(t))
            .collect(),
    )
}

/// Symmetric sampled surface distance between two meshes (same coordinate space).
///
/// `samples` area-weighted random points per mesh (fixed seeds, so the result is
/// deterministic) give the mean, RMS and normal deviation. The maxima also measure every
/// distinct vertex position of both meshes: random points alone rarely land on the tip of a
/// small feature that went missing, and can report half the true Hausdorff distance. That
/// costs one closest-point query per distinct vertex position, on top of `2 × samples`.
pub fn surface_distance(a: &Mesh, b: &Mesh, samples: usize) -> SurfaceDistance {
    let (bvh_a, bvh_b) = (triangle_bvh(a), triangle_bvh(b));
    let fwd = one_way(
        a,
        &sample_points(a, samples, 11),
        &corner_points(a),
        b,
        &bvh_b,
    );
    let bwd = one_way(
        b,
        &sample_points(b, samples, 13),
        &corner_points(b),
        a,
        &bvh_a,
    );
    summarize(fwd, bwd, a.bounds().diagonal().max(1e-12))
}

/// How a surface point reflects: what a specular highlight on it looks like.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gloss {
    /// Perceptual roughness in 0..=1, as glTF stores it (the lobe's width is its square).
    pub roughness: f32,
    /// Reflectance at normal incidence in 0..=1: about 0.04 for non-metals, the base color
    /// for metals.
    pub reflectance: f32,
}

impl Gloss {
    /// A matte surface: no highlight to misplace.
    pub const MATTE: Gloss = Gloss {
        roughness: 1.0,
        reflectance: 0.0,
    };

    /// From glTF's metallic-roughness model; `base_luminance` is the base color's luminance.
    pub fn from_metallic_roughness(roughness: f32, metallic: f32, base_luminance: f32) -> Gloss {
        let m = metallic.clamp(0.0, 1.0);
        Gloss {
            roughness: roughness.clamp(0.0, 1.0),
            reflectance: (0.04 * (1.0 - m) + base_luminance.clamp(0.0, 1.0) * m).clamp(0.0, 1.0),
        }
    }

    /// How much a highlight here draws the eye, in 0..=1: √reflectance × (1 − roughness).
    /// The root keeps polished non-metals in play (their highlights are dim in energy but
    /// plain to see); a wide lobe has little contrast whatever its strength.
    pub fn weight(&self) -> f32 {
        self.reflectance.max(0.0).sqrt() * (1.0 - self.roughness).clamp(0.0, 1.0)
    }

    /// Share of the highlight that lands in the wrong place when the shading normal is off by
    /// `angle` radians, in 0..=1: one minus the overlap of two lobes of this roughness that far
    /// apart, 1 − exp(−angle² / 4α²) with α = roughness² (the normal distribution's width).
    /// α is held above `MIN_LOBE`: below it, normal maps and pixel size blur the lobe anyway.
    pub fn misplaced(&self, angle: f32) -> f32 {
        let a = (self.roughness * self.roughness).max(MIN_LOBE);
        1.0 - (-(angle * angle) / (4.0 * a * a)).exp()
    }
}

/// Narrowest lobe [`Gloss::misplaced`] credits, in radians (about 1.7°).
const MIN_LOBE: f32 = 0.03;

/// Specular fidelity of a reduced mesh (see [`specular_error`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecularError {
    /// Surface area of the original that was sampled (for combining several meshes: weigh
    /// each by `area × weight`).
    pub area: f32,
    /// Mean highlight weight over the original's area in 0..=1: how specular the model is.
    pub weight: f32,
    /// Share of the weighted highlight that the reduced mesh puts in the wrong place, in
    /// 0..=1, at the reduced mesh's own scale.
    pub misplaced: f32,
    /// `weight × misplaced`: the error as a share of the whole surface, so a small glossy
    /// trim on a matte model scores low and a polished body scores high.
    pub impact: f32,
    /// Highlight-weighted mean angle, in degrees, between the original's mean shading normal
    /// over each reduced triangle and the reduced mesh's there.
    pub normal_degrees: f32,
    /// Highlight-weighted mean spread, in degrees, of the original's normals within a
    /// reduced triangle: detail below the reduced mesh's scale. It widens the highlight at
    /// that scale (a renderer should add it to roughness), so it is not counted as error.
    pub detail_degrees: f32,
}

/// How far the reduced mesh's highlights are from the original's.
///
/// A still under fixed light hides shading-normal error on glossy surfaces; a moving light or
/// camera shows it as highlights that swim or sit in the wrong place. Area-weighted points on
/// `original` (`samples`, or 16 per reduced triangle if that is more) are each matched to
/// the closest point of `reduced`, and `gloss` (material id and first UV set at the point)
/// says how wide and how strong the highlight is there.
///
/// The comparison is made per reduced triangle, the smallest patch the reduced mesh can
/// shade on its own: the original's mean normal over the points landing on it against the
/// reduced mesh's mean normal over the same points. A polished surface's highlight is
/// misplaced entirely by a few degrees, a rough one's hardly at all. Finer detail of the
/// original inside the triangle is gone whatever the reducer does; it spreads the original's
/// highlight at this scale, so it widens the lobe compared against (as in Toksvig's
/// roughness from normal variance) and is reported apart as `detail_degrees`. Normal maps
/// are not applied. Deterministic.
pub fn specular_error(
    original: &Mesh,
    reduced: &Mesh,
    gloss: impl Fn(u32, glam::Vec2) -> Gloss + Sync,
    samples: usize,
) -> SpecularError {
    let area: f64 = (0..original.triangle_count())
        .map(|t| {
            let [a, b, c] = original.triangle_positions(t);
            triangle_area(a, b, c) as f64
        })
        .sum();
    let points = sample_points(original, samples.max(16 * reduced.triangle_count()), 17);
    if points.is_empty() || reduced.triangle_count() == 0 {
        return SpecularError {
            area: area as f32,
            ..Default::default()
        };
    }
    let bvh = triangle_bvh(reduced);
    let uvs = original.uvs.first();
    // Per point: reduced triangle, highlight weight, lobe width, and the two normals.
    let each: Vec<(u32, f32, f32, Vec3, Vec3)> = points
        .par_iter()
        .map(|s| {
            let tri = original.triangle(s.triangle as usize).map(|i| i as usize);
            let b = s.barycentrics;
            let uv = uvs.map_or(glam::Vec2::ZERO, |u| {
                u[tri[0]] * b.x + u[tri[1]] * b.y + u[tri[2]] * b.z
            });
            let material = original
                .material_ids
                .get(s.triangle as usize)
                .copied()
                .unwrap_or(0);
            let g = gloss(material, uv);
            let n0 = shading_normal(original, s.triangle, b);
            match bvh.closest_point(s.position, f32::INFINITY) {
                Some(c) => (
                    c.triangle,
                    g.weight(),
                    g.roughness * g.roughness,
                    n0,
                    shading_normal(reduced, c.triangle, c.barycentrics),
                ),
                None => (u32::MAX, g.weight(), g.roughness * g.roughness, n0, -n0),
            }
        })
        .collect();
    // Summed in sample order per reduced triangle, so the result doesn't depend on threads.
    #[derive(Clone, Copy, Default)]
    struct Patch {
        w: f64,
        lobe: f64,
        n0: glam::DVec3,
        n1: glam::DVec3,
    }
    let mut patches = vec![Patch::default(); reduced.triangle_count() + 1];
    let mut total = 0f64;
    for &(t, w, lobe, n0, n1) in &each {
        let p = &mut patches[(t as usize).min(reduced.triangle_count())];
        let w = w as f64;
        p.w += w;
        p.lobe += w * lobe as f64;
        p.n0 += n0.as_dvec3() * w;
        p.n1 += n1.as_dvec3() * w;
        total += w;
    }
    let (mut misplaced, mut angle_sum, mut detail_sum) = (0f64, 0f64, 0f64);
    for p in &patches {
        if p.w <= 0.0 {
            continue;
        }
        let angle = match (p.n0.try_normalize(), p.n1.try_normalize()) {
            (Some(a), Some(b)) => a.dot(b).clamp(-1.0, 1.0).acos(),
            _ => std::f64::consts::PI,
        };
        // Angular variance of the original's normals in the patch: 2 (1 − |mean|).
        let variance = (2.0 * (1.0 - p.n0.length() / p.w)).max(0.0);
        let lobe = (p.lobe / p.w).max(MIN_LOBE as f64);
        let wide2 = lobe * lobe + variance;
        misplaced += p.w * (1.0 - (-(angle * angle) / (4.0 * wide2)).exp());
        angle_sum += p.w * angle;
        detail_sum += p.w * variance.sqrt();
    }
    let weight = total / each.len() as f64;
    let share = |x: f64| if total > 0.0 { x / total } else { 0.0 };
    SpecularError {
        area: area as f32,
        weight: weight as f32,
        misplaced: share(misplaced) as f32,
        impact: (weight * share(misplaced)) as f32,
        normal_degrees: share(angle_sum).to_degrees() as f32,
        detail_degrees: share(detail_sum).to_degrees() as f32,
    }
}

/// Like [`surface_distance`], counting only surface that can be seen from outside its own
/// mesh: a sample (or vertex) stays if a ray from it escapes in at least one of 32 fixed
/// directions. An enclosed part (a watch movement in its case) adds nothing, so deleting it
/// isn't penalized, while a visible part that goes missing still is.
pub fn visible_surface_distance(a: &Mesh, b: &Mesh, samples: usize) -> SurfaceDistance {
    let (bvh_a, bvh_b) = (triangle_bvh(a), triangle_bvh(b));
    let diagonal = a.bounds().diagonal().max(1e-12);
    let dirs: Vec<Vec3> = (0..32)
        .map(|i| {
            let z = 1.0 - (i as f32 + 0.5) / 16.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let t = i as f32 * 2.399_963;
            Vec3::new(r * t.cos() + 1e-3, r * t.sin() + 2e-3, z + 3e-3).normalize()
        })
        .collect();
    let eps = diagonal * 1e-5;
    let seen = |p: Vec3, own: &Bvh| {
        dirs.iter()
            .any(|&d| !own.occluded(p, d, eps, diagonal * 4.0))
    };
    let visible = |pts: Vec<SurfacePoint>, own: &Bvh| -> Vec<SurfacePoint> {
        pts.into_par_iter()
            .filter(|s| seen(s.position, own))
            .collect()
    };
    let visible_corners = |pts: Vec<Vec3>, own: &Bvh| -> Vec<Vec3> {
        pts.into_par_iter().filter(|&p| seen(p, own)).collect()
    };
    let fwd = one_way(
        a,
        &visible(sample_points(a, samples, 11), &bvh_a),
        &visible_corners(corner_points(a), &bvh_a),
        b,
        &bvh_b,
    );
    let bwd = one_way(
        b,
        &visible(sample_points(b, samples, 13), &bvh_b),
        &visible_corners(corner_points(b), &bvh_b),
        a,
        &bvh_a,
    );
    summarize(fwd, bwd, diagonal)
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageDiff {
    /// PSNR over RGB of pixels covered in either image (dB; higher is better).
    pub psnr: f32,
    /// Mean absolute RGB error over the same pixels.
    pub mean_abs_error: f32,
    /// Fraction of pixels covered in exactly one image (silhouette mismatch).
    pub coverage_mismatch: f32,
}

/// Compares two renders of the same view. Coverage comes from alpha (> 0.5).
pub fn image_difference(a: &Image, b: &Image) -> ImageDiff {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "image sizes differ"
    );
    let (mut se, mut ae, mut n, mut mismatch) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (p, q) in a.pixels.iter().zip(&b.pixels) {
        let (ca, cb) = (p.w > 0.5, q.w > 0.5);
        if ca != cb {
            mismatch += 1;
        }
        if ca || cb {
            let pa = if ca { p.truncate() } else { Vec3::ZERO };
            let pb = if cb { q.truncate() } else { Vec3::ZERO };
            let d = (pa - pb).abs();
            se += d.length_squared() as f64 / 3.0;
            ae += (d.x + d.y + d.z) as f64 / 3.0;
            n += 1;
        }
    }
    let n = n.max(1) as f64;
    let mse = se / n;
    ImageDiff {
        psnr: if mse <= 1e-12 {
            99.0
        } else {
            (10.0 * (1.0 / mse).log10()) as f32
        },
        mean_abs_error: (ae / n) as f32,
        coverage_mismatch: mismatch as f32 / a.pixels.len().max(1) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube;

    #[test]
    fn specular_error_follows_gloss_and_scale() {
        // A latitude-longitude sphere with smooth normals.
        let uv_sphere = |seg: u32, rings: u32| {
            let mut m = Mesh::default();
            for r in 0..=rings {
                let phi = std::f32::consts::PI * r as f32 / rings as f32;
                for s in 0..seg {
                    let th = std::f32::consts::TAU * s as f32 / seg as f32;
                    m.positions.push(Vec3::new(
                        phi.sin() * th.cos(),
                        phi.cos(),
                        phi.sin() * th.sin(),
                    ));
                }
            }
            for r in 0..rings {
                for s in 0..seg {
                    let (a, b) = (r * seg + s, r * seg + (s + 1) % seg);
                    let (c, d) = (a + seg, b + seg);
                    for t in [[a, b, c], [b, d, c]] {
                        let [x, y, z] = t.map(|i| m.positions[i as usize]);
                        if (y - x).cross(z - x).length() > 1e-9 {
                            m.indices.extend(t);
                            m.material_ids.push(0);
                        }
                    }
                }
            }
            m
        };
        let fine = {
            let mut m = uv_sphere(96, 48);
            m.compute_normals(180.0);
            m
        };
        let polished = |_: u32, _: glam::Vec2| Gloss::from_metallic_roughness(0.1, 1.0, 1.0);
        // The same mesh: nothing misplaced, however glossy.
        let same = specular_error(&fine, &fine, polished, 4000);
        assert!(same.weight > 0.8 && same.misplaced < 1e-3, "{same:?}");
        // A coarse sphere with smooth normals follows the fine one's highlight closely; with
        // faceted normals each face's highlight is off by up to half its angle.
        let mut coarse = uv_sphere(16, 8);
        coarse.compute_normals(180.0);
        let smooth = specular_error(&fine, &coarse, polished, 4000);
        let mut faceted = coarse.clone();
        faceted.normals = None;
        let flat = specular_error(&fine, &faceted, polished, 4000);
        assert!(
            flat.misplaced > 2.0 * smooth.misplaced.max(0.01),
            "{smooth:?} {flat:?}"
        );
        // The same normals matter less on a rough surface and not at all on a matte one.
        let rough = |_: u32, _: glam::Vec2| Gloss::from_metallic_roughness(0.9, 1.0, 1.0);
        let r = specular_error(&fine, &faceted, rough, 4000);
        assert!(r.misplaced < 0.25 * flat.misplaced, "{r:?} {flat:?}");
        let matte = specular_error(&fine, &faceted, |_, _| Gloss::MATTE, 4000);
        assert_eq!((matte.weight, matte.impact), (0.0, 0.0));
    }

    #[test]
    fn identical_meshes_have_zero_distance() {
        let c = cube();
        let d = surface_distance(&c, &c, 2000);
        assert!(d.hausdorff < 1e-5);
    }

    #[test]
    fn scaled_cube_distance() {
        let a = cube();
        let mut b = cube();
        for p in &mut b.positions {
            *p *= 1.2;
        }
        let d = surface_distance(&a, &b, 4000);
        assert!((d.forward_max - 0.1).abs() < 0.02, "{d:?}");
    }

    #[test]
    fn lost_spike_is_caught_by_its_tip() {
        // A needle 1.0 tall on top of the cube, with too little area for 100 samples to find.
        let plain = cube();
        let mut spiked = cube();
        let b = spiked.positions.len() as u32;
        spiked.positions.extend([
            Vec3::new(-1e-3, 0.5, 0.0),
            Vec3::new(1e-3, 0.5, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
        ]);
        spiked.indices.extend([b, b + 1, b + 2]);
        spiked.material_ids.push(0);
        let d = surface_distance(&spiked, &plain, 100);
        assert!((d.forward_max - 1.0).abs() < 1e-4, "{d:?}");
        assert!((d.hausdorff - 1.0).abs() < 1e-4, "{d:?}");
        assert!(d.backward_max < 1e-5, "{d:?}");
        assert_eq!(d, surface_distance(&spiked, &plain, 100));
    }

    #[test]
    fn normal_deviation() {
        let c = cube();
        assert!(surface_distance(&c, &c, 1000).normal_mean_degrees < 1e-3);
        let mut flipped = cube();
        for t in flipped.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        let d = surface_distance(&c, &flipped, 1000);
        assert!(d.hausdorff < 1e-5, "{d:?}");
        assert!(d.normal_mean_degrees > 179.0, "{d:?}");
    }

    #[test]
    fn samples_lie_on_surface() {
        let c = cube();
        for p in sample_surface(&c, 500, 1) {
            assert!((p.abs().max_element() - 0.5).abs() < 1e-5);
        }
    }
}
