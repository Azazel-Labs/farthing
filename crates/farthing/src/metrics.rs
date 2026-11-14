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
