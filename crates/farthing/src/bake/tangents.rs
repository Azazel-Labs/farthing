//! MikkTSpace tangents (via the pure-Rust `mikktspace` crate), so baked normal maps decode
//! identically in engines that use MikkTSpace.

use crate::{Error, Mesh, Result};
use glam::{Vec3, Vec4};

struct Geo<'a> {
    mesh: &'a Mesh,
    uv_set: usize,
    normals: &'a [Vec3],
    /// Per corner.
    out: Vec<Option<Vec4>>,
}

impl mikktspace::Geometry for Geo<'_> {
    fn num_faces(&self) -> usize {
        self.mesh.triangle_count()
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.mesh.positions[self.mesh.indices[face * 3 + vert] as usize].to_array()
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.mesh.indices[face * 3 + vert] as usize].to_array()
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.mesh.uvs[self.uv_set][self.mesh.indices[face * 3 + vert] as usize].to_array()
    }
    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.out[face * 3 + vert] = Some(Vec4::from_array(tangent));
    }
}

/// Computes per-vertex MikkTSpace tangents from `uv_set` and stores them in `mesh.tangents`.
/// Computes smooth normals first if the mesh has none. A vertex whose corners need opposite
/// handedness (a vertex on the seam of mirrored UVs that share it) is split in two, so each
/// side gets its own bitangent sign; triangles keep their order.
pub fn compute_tangents(mesh: &mut Mesh, uv_set: usize) -> Result<()> {
    if mesh.uvs.len() <= uv_set {
        return Err(Error::InvalidMesh(format!("mesh has no UV set {uv_set}")));
    }
    if mesh.normals.is_none() {
        mesh.compute_normals(180.0);
    }
    let normals = mesh.normals.clone().unwrap();
    let mut g = Geo {
        mesh,
        uv_set,
        normals: &normals,
        out: vec![None; mesh.indices.len()],
    };
    mikktspace::generate_tangents(&mut g);
    let corners = g.out;
    let n = mesh.vertex_count();
    let mut out = vec![Vec4::new(1.0, 0.0, 0.0, 1.0); n];
    // Handedness seen at each vertex (0: none yet), and its copy for the other handedness.
    let mut sign = vec![0i8; n];
    let mut twin = vec![u32::MAX; n];
    let mut order: Vec<u32> = (0..n as u32).collect();
    for (i, t) in corners.into_iter().enumerate() {
        let Some(t) = t else { continue };
        let v = mesh.indices[i] as usize;
        let s = if t.w < 0.0 { -1 } else { 1 };
        if sign[v] == 0 || sign[v] == s {
            sign[v] = s;
            out[v] = t;
            continue;
        }
        if twin[v] == u32::MAX {
            twin[v] = order.len() as u32;
            order.push(v as u32);
            out.push(t);
        } else {
            out[twin[v] as usize] = t;
        }
        mesh.indices[i] = twin[v];
    }
    if order.len() > n {
        let mut split = mesh.select_vertices(&order);
        split.indices = std::mem::take(&mut mesh.indices);
        split.material_ids = std::mem::take(&mut mesh.material_ids);
        *mesh = split;
    }
    mesh.tangents = Some(out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn flat_quad_tangent_follows_u() {
        let mut m = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, -1.0),
            ],
            vec![0, 1, 2, 0, 2, 3],
        );
        m.normals = Some(vec![Vec3::Y; 4]);
        m.uvs = vec![vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ]];
        compute_tangents(&mut m, 0).unwrap();
        let t = m.tangents.unwrap()[0];
        assert!((t.truncate() - Vec3::X).length() < 1e-4, "{t:?}");
    }

    #[test]
    fn mirrored_uvs_split_the_shared_vertices() {
        // Two quads side by side sharing the middle edge; the right one mirrors the left's UVs
        // (as a symmetric model sharing one texture half does).
        let mut m = Mesh::new(
            vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::ZERO,
                Vec3::new(0.0, 0.0, -1.0),
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::X,
                Vec3::new(1.0, 0.0, -1.0),
            ],
            vec![0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2],
        );
        m.normals = Some(vec![Vec3::Y; 6]);
        m.uvs = vec![vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.0, 0.0),
        ]];
        compute_tangents(&mut m, 0).unwrap();
        // The two middle vertices are split: 6 + 2.
        assert_eq!(m.vertex_count(), 8);
        assert_eq!(m.triangle_count(), 4);
        let ts = m.tangents.as_ref().unwrap();
        for t in 0..4 {
            // u runs along +X on the left and -X on the right; v runs along +Z on both.
            let want = if t < 2 { Vec3::X } else { Vec3::NEG_X };
            for v in m.triangle(t) {
                let tan = ts[v as usize];
                assert!((tan.truncate() - want).length() < 1e-4, "tri {t}: {tan:?}");
                let bitangent = Vec3::Y.cross(tan.truncate()) * tan.w;
                assert!((bitangent - Vec3::Z).length() < 1e-4, "tri {t}: {tan:?}");
            }
        }
        // The copies carry the vertex's position and UV.
        assert_eq!(m.positions[m.triangle(2)[0] as usize], Vec3::ZERO);
        assert_eq!(m.uvs[0][m.triangle(2)[0] as usize], Vec2::new(1.0, 1.0));
    }
}
