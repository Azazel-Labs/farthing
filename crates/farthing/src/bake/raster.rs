//! UV-space rasterization: which destination triangle (and where in it) covers each texel.

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
    match mesh.uvs.get(uv_set) {
        Some(uvs) => rasterize_rows(
            mesh,
            uvs,
            width,
            height,
            0,
            height,
            0..mesh.triangle_count(),
        ),
        None => vec![None; (width * height) as usize],
    }
}

/// Triangle `t`'s corners in texel units and the rows its bounding box covers.
fn texel_bounds(mesh: &Mesh, uvs: &[Vec2], t: usize, size: Vec2) -> ([Vec2; 3], Vec2, Vec2) {
    let p = mesh.triangle(t).map(|v| uvs[v as usize] * size);
    let min = p[0].min(p[1]).min(p[2]).floor().max(Vec2::ZERO);
    let max = p[0].max(p[1]).max(p[2]).ceil().min(size);
    (p, min, max)
}

/// Rows `y0..y0 + rows` of [`rasterize_uv`] (UV set `uvs`), from the triangles in
/// `triangles`, in increasing order: every triangle that reaches those rows must be there.
pub(crate) fn rasterize_rows(
    mesh: &Mesh,
    uvs: &[Vec2],
    width: u32,
    height: u32,
    y0: u32,
    rows: u32,
    triangles: impl IntoIterator<Item = usize>,
) -> Vec<Option<TexelSample>> {
    let mut out = vec![None; (width * rows) as usize];
    let size = Vec2::new(width as f32, height as f32);
    let y1 = y0 + rows;
    for t in triangles {
        let (p, min, max) = texel_bounds(mesh, uvs, t, size);
        // Barycentrics from the edges out of corner 0: a few products per texel, and offsets
        // the size of the triangle rather than of the atlas, which keeps their precision.
        let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
        let area2 = e1.perp_dot(e2);
        if (0.5 * area2).abs() < 1e-20 {
            continue;
        }
        let inv = 1.0 / area2;
        for y in (min.y as u32).max(y0)..(max.y as u32).min(y1) {
            let dy = y as f32 + 0.5 - p[0].y;
            for x in min.x as u32..max.x as u32 {
                let dx = x as f32 + 0.5 - p[0].x;
                let u = (dx * e2.y - dy * e2.x) * inv;
                let v = (e1.x * dy - e1.y * dx) * inv;
                let b = Vec3::new(1.0 - u - v, u, v);
                if b.min_element() >= -1e-5 {
                    out[((y - y0) * width + x) as usize] = Some(TexelSample {
                        triangle: t as u32,
                        barycentrics: b,
                    });
                }
            }
        }
    }
    out
}

/// Per band of `band` rows (the last may be shorter), the triangles [`rasterize_rows`] needs
/// for it, in increasing order.
pub(crate) fn band_triangles(
    mesh: &Mesh,
    uvs: &[Vec2],
    width: u32,
    height: u32,
    band: u32,
) -> Vec<Vec<u32>> {
    let band = band.max(1);
    let mut bands = vec![Vec::new(); height.div_ceil(band) as usize];
    let size = Vec2::new(width as f32, height as f32);
    for t in 0..mesh.triangle_count() {
        let (_, min, max) = texel_bounds(mesh, uvs, t, size);
        let (lo, hi) = (min.y as u32, (max.y as u32).min(height));
        if lo < hi {
            for b in &mut bands[(lo / band) as usize..=((hi - 1) / band) as usize] {
                b.push(t as u32);
            }
        }
    }
    bands
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
