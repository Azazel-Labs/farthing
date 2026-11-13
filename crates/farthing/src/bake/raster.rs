//! UV-space rasterization: which destination triangle (and where in it) covers each texel.

use crate::math::barycentric_2d;
use crate::Mesh;
use glam::{Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexelSample {
    pub triangle: u32,
    pub barycentrics: Vec3,
}

/// Rasterizes `mesh` in UV set `uv_set` at `width × height`, sampling texel centers.
/// Overlapping charts: the later triangle wins.
pub fn rasterize_uv(
    mesh: &Mesh,
    uv_set: usize,
    width: u32,
    height: u32,
) -> Vec<Option<TexelSample>> {
    let mut out = vec![None; (width * height) as usize];
    let Some(uvs) = mesh.uvs.get(uv_set) else {
        return out;
    };
    let size = Vec2::new(width as f32, height as f32);
    for t in 0..mesh.triangle_count() {
        let tri = mesh.triangle(t);
        let p = tri.map(|v| uvs[v as usize] * size);
        let min = p[0].min(p[1]).min(p[2]).floor().max(Vec2::ZERO);
        let max = p[0].max(p[1]).max(p[2]).ceil().min(size);
        for y in min.y as u32..max.y as u32 {
            for x in min.x as u32..max.x as u32 {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                if let Some(b) = barycentric_2d(c, p[0], p[1], p[2]) {
                    if b.min_element() >= -1e-5 {
                        out[(y * width + x) as usize] = Some(TexelSample {
                            triangle: t as u32,
                            barycentrics: b,
                        });
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_quad_covers_all_texels() {
        let mut m = Mesh::new(vec![Vec3::ZERO; 4], vec![0, 1, 2, 0, 2, 3]);
        m.uvs = vec![vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y]];
        let r = rasterize_uv(&m, 0, 16, 16);
        assert!(r.iter().all(|s| s.is_some()));
    }
}
