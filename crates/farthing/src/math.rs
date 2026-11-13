//! Small geometric helpers.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Axis-aligned bounding box. An empty box has `min > max`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a Vec3>) -> Self {
        let mut b = Self::EMPTY;
        for p in points {
            b.extend(*p);
        }
        b
    }
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }
    pub fn extend(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    pub fn union(&self, o: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn extent(&self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            self.max - self.min
        }
    }
    pub fn diagonal(&self) -> f32 {
        self.extent().length()
    }
    /// Radius of the bounding sphere around the box center.
    pub fn radius(&self) -> f32 {
        self.diagonal() * 0.5
    }
    pub fn surface_area(&self) -> f32 {
        let e = self.extent();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }
    pub fn largest_axis(&self) -> usize {
        let e = self.extent();
        if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        }
    }
}

/// Area of triangle (a, b, c).
#[inline]
pub fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b - a).cross(c - a).length() * 0.5
}

/// Unnormalized face normal (length = 2 × area).
#[inline]
pub fn face_normal_raw(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a)
}

/// Signed area of a 2D triangle (positive when counter-clockwise).
#[inline]
pub fn signed_area_2d(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
}

/// Barycentric coordinates of `p` in 2D triangle (a, b, c). Returns None for degenerate triangles.
#[inline]
pub fn barycentric_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<Vec3> {
    let area = signed_area_2d(a, b, c);
    if area.abs() < 1e-20 {
        return None;
    }
    let w0 = signed_area_2d(p, b, c) / area;
    let w1 = signed_area_2d(a, p, c) / area;
    Some(Vec3::new(w0, w1, 1.0 - w0 - w1))
}

/// An orthonormal basis (t, b) perpendicular to `n` (Frisvad / Duff et al. 2017).
pub fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = 1.0f32.copysign(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bt = Vec3::new(b, sign + n.y * n.y * a, -n.y);
    (t, bt)
}

/// Converts a pixel error at `distance` into a world-space deviation for a perspective
/// camera with vertical field of view `fov_y` (radians) and `viewport_height` pixels.
pub fn pixel_error_to_world(
    pixel_error: f32,
    distance: f32,
    fov_y: f32,
    viewport_height: f32,
) -> f32 {
    pixel_error * 2.0 * distance * (fov_y * 0.5).tan() / viewport_height
}

/// Inverse of [`pixel_error_to_world`]: the on-screen size in pixels of a world deviation.
pub fn world_error_to_pixels(
    world_error: f32,
    distance: f32,
    fov_y: f32,
    viewport_height: f32,
) -> f32 {
    world_error * viewport_height / (2.0 * distance * (fov_y * 0.5).tan())
}

/// Evenly distributed directions on the unit sphere (Fibonacci lattice).
/// With `hemisphere`, only directions with y >= 0 are produced.
pub fn fibonacci_directions(n: usize, hemisphere: bool) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..n)
        .map(|i| {
            let fi = i as f32 + 0.5;
            let y = if hemisphere {
                1.0 - fi / n as f32
            } else {
                1.0 - 2.0 * fi / n as f32
            };
            let r = (1.0 - y * y).max(0.0).sqrt();
            let phi = golden * i as f32;
            Vec3::new(r * phi.cos(), y, r * phi.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_world_roundtrip() {
        let w = pixel_error_to_world(2.0, 50.0, 1.0, 1080.0);
        let p = world_error_to_pixels(w, 50.0, 1.0, 1080.0);
        assert!((p - 2.0).abs() < 1e-4);
    }

    #[test]
    fn basis_is_orthonormal() {
        for n in fibonacci_directions(50, false) {
            let (t, b) = orthonormal_basis(n);
            assert!(t.dot(n).abs() < 1e-5 && b.dot(n).abs() < 1e-5 && t.dot(b).abs() < 1e-5);
            assert!((t.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn hemisphere_directions_point_up() {
        assert!(fibonacci_directions(64, true).iter().all(|d| d.y >= 0.0));
    }
}
