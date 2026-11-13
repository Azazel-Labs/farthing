//! Symmetric 4×4 error quadrics (Garland & Heckbert 1997), stored as 10 coefficients padded
//! to 12 so evaluation is one fixed-shape dot product (see `kernels`).

use glam::DVec3;

#[derive(Clone, Copy, Debug, Default)]
pub struct Quadric {
    // a00 a01 a02 a03 a11 a12 a13 a22 a23 a33 0 0
    a: [f64; 12],
    /// Accumulated weight (area) — used to normalize the error into a mean squared distance.
    pub w: f64,
}

impl Quadric {
    /// Quadric of the plane n·x + d = 0 (n unit length), scaled by `weight`.
    pub fn plane(n: DVec3, d: f64, weight: f64) -> Quadric {
        let (a, b, c) = (n.x, n.y, n.z);
        Quadric {
            a: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
                0.0,
                0.0,
            ]
            .map(|v| v * weight),
            w: weight,
        }
    }

    /// A constraint plane (border/seam guard): contributes error but no normalization weight,
    /// so constraints never dilute the mean surface error they protect.
    pub fn constraint(n: DVec3, d: f64, weight: f64) -> Quadric {
        Quadric {
            w: 0.0,
            ..Quadric::plane(n, d, weight)
        }
    }

    pub fn add(&mut self, o: &Quadric) {
        for i in 0..12 {
            self.a[i] += o.a[i];
        }
        self.w += o.w;
    }

    /// Weighted sum of squared distances from `p` to the accumulated planes.
    pub fn eval(&self, p: DVec3) -> f64 {
        let a = &self.a;
        let (x, y, z) = (p.x, p.y, p.z);
        let v = a[0] * x * x
            + 2.0 * a[1] * x * y
            + 2.0 * a[2] * x * z
            + 2.0 * a[3] * x
            + a[4] * y * y
            + 2.0 * a[5] * y * z
            + 2.0 * a[6] * y
            + a[7] * z * z
            + 2.0 * a[8] * z
            + a[9];
        v.max(0.0)
    }

    /// Raw coefficients (a00 a01 a02 a03 a11 a12 a13 a22 a23 a33).
    pub fn coeffs(&self) -> [f64; 12] {
        self.a
    }

    /// Weighted squared distance at a point given by its precomputed monomials (clamped ≥ 0).
    #[inline(always)]
    pub(crate) fn eval_m(&self, m: &[f64; 12]) -> f64 {
        crate::reduce::kernels::dot12(&self.a, m).max(0.0)
    }

    /// Mean squared distance (eval / weight).
    pub fn mean_error(&self, p: DVec3) -> f64 {
        if self.w <= 0.0 {
            0.0
        } else {
            self.eval(p) / self.w
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_distance() {
        let q = Quadric::plane(DVec3::Y, -1.0, 2.0); // plane y = 1
        assert!((q.eval(DVec3::new(5.0, 3.0, 0.0)) - 8.0).abs() < 1e-9);
        assert!((q.mean_error(DVec3::new(0.0, 3.0, 0.0)) - 4.0).abs() < 1e-9);
    }
}
