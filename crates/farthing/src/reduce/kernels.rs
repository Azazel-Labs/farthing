//! Fixed-shape numeric kernels for the reducer's hot loop.
//!
//! Quadrics are stored as 12 padded coefficients so evaluation is a 12-wide dot product against
//! a point's precomputed monomials, written as three 4-lane multiply-adds and a fixed pairwise
//! horizontal sum. The association order is spelled out in the source, so the compiler can
//! vectorize (SSE2, AVX2, NEON) without changing a single bit of the result: output stays
//! identical across CPUs and builds.

use glam::DVec3;

/// Monomials of q matching the padded quadric layout
/// `[a00 a01 a02 a03 a11 a12 a13 a22 a23 a33 0 0]`.
#[inline(always)]
pub(crate) fn monomials(q: DVec3) -> [f64; 12] {
    let (x, y, z) = (q.x, q.y, q.z);
    [
        x * x,
        2.0 * x * y,
        2.0 * x * z,
        2.0 * x,
        y * y,
        2.0 * y * z,
        2.0 * y,
        z * z,
        2.0 * z,
        1.0,
        0.0,
        0.0,
    ]
}

/// Homogeneous point for gradient dot products.
#[inline(always)]
pub(crate) fn homogeneous(q: DVec3) -> [f64; 4] {
    [q.x, q.y, q.z, 1.0]
}

/// Σ c[k]·m[k] over 12 lanes: lane-wise (c0m0 + c4m4) + c8m8, then (l0 + l1) + (l2 + l3).
#[inline(always)]
pub(crate) fn dot12(c: &[f64], m: &[f64; 12]) -> f64 {
    let c: &[f64; 12] = c[..12].try_into().unwrap();
    let mut acc = [0.0f64; 4];
    for l in 0..4 {
        acc[l] = c[l] * m[l] + c[4 + l] * m[4 + l] + c[8 + l] * m[8 + l];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// g·h over 4 lanes: (g0h0 + g1h1) + (g2h2 + g3h3).
#[inline(always)]
pub(crate) fn dot4(g: &[f64], h: &[f64; 4]) -> f64 {
    let g: &[f64; 4] = g[..4].try_into().unwrap();
    (g[0] * h[0] + g[1] * h[1]) + (g[2] * h[2] + g[3] * h[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot12_matches_quadric_form() {
        let c = [
            1.5, -0.25, 0.5, 2.0, 3.0, 0.75, -1.0, 0.5, 0.125, 4.0, 0.0, 0.0,
        ];
        let q = DVec3::new(0.3, -1.2, 2.5);
        let (x, y, z) = (q.x, q.y, q.z);
        let direct = c[0] * x * x
            + 2.0 * c[1] * x * y
            + 2.0 * c[2] * x * z
            + 2.0 * c[3] * x
            + c[4] * y * y
            + 2.0 * c[5] * y * z
            + 2.0 * c[6] * y
            + c[7] * z * z
            + 2.0 * c[8] * z
            + c[9];
        assert!((dot12(&c, &monomials(q)) - direct).abs() < 1e-12);
    }
}
