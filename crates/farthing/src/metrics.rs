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
            a + (b - a) * u + (c - a) * v
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceDistance {
    /// max(forward_max, backward_max) — symmetric Hausdorff distance (sampled).
    pub hausdorff: f32,
    /// Mean of both directions.
    pub mean: f32,
    pub rms: f32,
    /// Max distance from samples on A to B.
    pub forward_max: f32,
    /// Max distance from samples on B to A.
    pub backward_max: f32,
    /// Bounding-box diagonal of A, for relative numbers.
    pub diagonal: f32,
    /// hausdorff / diagonal.
    pub relative_hausdorff: f32,
    pub relative_mean: f32,
}

fn one_way(points: &[Vec3], target: &Bvh) -> Vec<f32> {
    points
        .par_iter()
        .map(|p| {
            target
                .closest_point(*p, f32::INFINITY)
                .map_or(f32::INFINITY, |c| c.distance)
        })
        .collect()
}

/// Symmetric sampled surface distance between two meshes (same coordinate space).
pub fn surface_distance(a: &Mesh, b: &Mesh, samples: usize) -> SurfaceDistance {
    let bvh_a = Bvh::build(
        (0..a.triangle_count())
            .map(|t| a.triangle_positions(t))
            .collect(),
    );
    let bvh_b = Bvh::build(
        (0..b.triangle_count())
            .map(|t| b.triangle_positions(t))
            .collect(),
    );
    let fwd = one_way(&sample_surface(a, samples, 11), &bvh_b);
    let bwd = one_way(&sample_surface(b, samples, 13), &bvh_a);
    let maxf = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
    let all: Vec<f32> = fwd.iter().chain(bwd.iter()).copied().collect();
    let n = all.len().max(1) as f32;
    let mean = all.iter().sum::<f32>() / n;
    let rms = (all.iter().map(|d| d * d).sum::<f32>() / n).sqrt();
    let diagonal = a.bounds().diagonal().max(1e-12);
    let hausdorff = maxf(&fwd).max(maxf(&bwd));
    SurfaceDistance {
        hausdorff,
        mean,
        rms,
        forward_max: maxf(&fwd),
        backward_max: maxf(&bwd),
        diagonal,
        relative_hausdorff: hausdorff / diagonal,
        relative_mean: mean / diagonal,
    }
}

/// Like [`surface_distance`], counting only surface that can be seen from outside its own
/// mesh: a sample stays if a ray from it escapes in at least one of 32 fixed directions. An
/// enclosed part (a watch movement in its case) adds nothing, so deleting it isn't penalized,
/// while a visible part that goes missing still is.
pub fn visible_surface_distance(a: &Mesh, b: &Mesh, samples: usize) -> SurfaceDistance {
    let build = |m: &Mesh| {
        Bvh::build(
            (0..m.triangle_count())
                .map(|t| m.triangle_positions(t))
                .collect(),
        )
    };
    let (bvh_a, bvh_b) = (build(a), build(b));
    let diagonal = a.bounds().diagonal().max(1e-12);
    let dirs: Vec<Vec3> = (0..32)
        .map(|i| {
            let z = 1.0 - (i as f32 + 0.5) / 16.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let t = i as f32 * 2.399_963;
            Vec3::new(r * t.cos() + 1e-3, r * t.sin() + 2e-3, z + 3e-3).normalize()
        })
        .collect();
    let visible = |pts: Vec<Vec3>, own: &Bvh| -> Vec<Vec3> {
        let eps = diagonal * 1e-5;
        pts.into_par_iter()
            .filter(|p| dirs.iter().any(|&d| !own.occluded(*p, d, eps, diagonal * 4.0)))
            .collect()
    };
    let fwd = one_way(&visible(sample_surface(a, samples, 11), &bvh_a), &bvh_b);
    let bwd = one_way(&visible(sample_surface(b, samples, 13), &bvh_b), &bvh_a);
    let maxf = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
    let all: Vec<f32> = fwd.iter().chain(bwd.iter()).copied().collect();
    let n = all.len().max(1) as f32;
    let mean = all.iter().sum::<f32>() / n;
    let rms = (all.iter().map(|d| d * d).sum::<f32>() / n).sqrt();
    let hausdorff = maxf(&fwd).max(maxf(&bwd));
    SurfaceDistance {
        hausdorff,
        mean,
        rms,
        forward_max: maxf(&fwd),
        backward_max: maxf(&bwd),
        diagonal,
        relative_hausdorff: hausdorff / diagonal,
        relative_mean: mean / diagonal,
    }
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
    fn samples_lie_on_surface() {
        let c = cube();
        for p in sample_surface(&c, 500, 1) {
            assert!((p.abs().max_element() - 0.5).abs() < 1e-5);
        }
    }
}
