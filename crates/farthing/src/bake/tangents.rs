//! MikkTSpace tangents (via the pure-Rust `mikktspace` crate), so baked normal maps decode
//! identically in engines that use MikkTSpace.

use glam::{Vec3, Vec4};
use crate::{Error, Mesh, Result};

struct Geo<'a> {
    mesh: &'a Mesh,
    uv_set: usize,
    normals: &'a [Vec3],
    out: Vec<Vec4>,
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
        let v = self.mesh.indices[face * 3 + vert] as usize;
        self.out[v] = Vec4::from_array(tangent);
    }
}

/// Computes per-vertex MikkTSpace tangents from `uv_set` and stores them in `mesh.tangents`.
/// Computes smooth normals first if the mesh has none.
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
        out: vec![Vec4::new(1.0, 0.0, 0.0, 1.0); mesh.vertex_count()],
    };
    mikktspace::generate_tangents(&mut g);
    let out = g.out;
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
}
