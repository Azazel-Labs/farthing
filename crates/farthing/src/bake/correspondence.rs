//! The correspondence buffer: for every (supersampled) destination texel, which source
//! surface point it maps to.

use crate::bake::raster::rasterize_uv;
use crate::bvh::Bvh;
use crate::scene::TriRef;
use crate::{Error, Mesh, Result, Scene};
use glam::{Mat4, Vec3};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrespondenceEntry {
    /// Source instance and triangle.
    pub source: TriRef,
    /// Barycentrics on the source triangle, in the source mesh's corner order (also for
    /// mirrored instances, whose world triangles swap corners 1 and 2).
    pub barycentrics: Vec3,
    /// Signed distance from the destination surface to the source along the destination
    /// normal (positive = source is outside).
    pub distance: f32,
    /// Destination triangle and barycentrics (for tangent frames).
    pub dest_triangle: u32,
    pub dest_barycentrics: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Correspondence {
    /// Sample grid size (output size × supersample).
    pub width: u32,
    pub height: u32,
    pub supersample: u32,
    pub entries: Vec<Option<CorrespondenceEntry>>,
}

impl Correspondence {
    pub fn coverage(&self) -> f32 {
        self.entries.iter().filter(|e| e.is_some()).count() as f32
            / self.entries.len().max(1) as f32
    }
}

/// Builds the correspondence from `dest` (placed by `dest_to_world`, UV set `uv_set`) to the
/// source scene. Rays start `max_distance` outside the destination surface and travel inward
/// (an implicit cage); if they miss, the closest source point within `2 × max_distance` is used.
pub fn build_correspondence(
    dest: &Mesh,
    dest_to_world: Mat4,
    source: &Scene,
    width: u32,
    height: u32,
    supersample: u32,
    uv_set: usize,
    max_distance: f32,
) -> Result<Correspondence> {
    if dest.uvs.len() <= uv_set {
        return Err(Error::InvalidMesh(format!(
            "destination has no UV set {uv_set}"
        )));
    }
    let ss = supersample.max(1);
    let (w, h) = (width * ss, height * ss);
    let samples = rasterize_uv(dest, uv_set, w, h);
    let (tris, refs) = source.world_triangles();
    let flip = mirrored(source);
    let bvh = Bvh::build(tris);
    let nmat = glam::Mat3::from_mat4(dest_to_world).inverse().transpose();
    let face_n: Vec<Vec3> = (0..dest.triangle_count())
        .map(|t| {
            let [a, b, c] = dest.triangle_positions(t);
            (nmat * (b - a).cross(c - a)).normalize_or_zero()
        })
        .collect();
    let entries = samples
        .par_iter()
        .map(|s| {
            let s = (*s)?;
            let tri = dest.triangle(s.triangle as usize);
            let b = s.barycentrics;
            let p_local = dest.positions[tri[0] as usize] * b.x
                + dest.positions[tri[1] as usize] * b.y
                + dest.positions[tri[2] as usize] * b.z;
            let p = dest_to_world.transform_point3(p_local);
            let n = match &dest.normals {
                Some(ns) => (nmat
                    * (ns[tri[0] as usize] * b.x
                        + ns[tri[1] as usize] * b.y
                        + ns[tri[2] as usize] * b.z))
                    .normalize_or_zero(),
                None => face_n[s.triangle as usize],
            };
            let n = if n == Vec3::ZERO {
                face_n[s.triangle as usize]
            } else {
                n
            };
            let origin = p + n * max_distance;
            let hit = bvh.intersect(origin, -n, 0.0, 2.0 * max_distance);
            let (src_tri, bary, dist) = match hit {
                Some(hh) => (hh.triangle, hh.barycentrics(), max_distance - hh.t),
                None => {
                    let c = bvh.closest_point(p, 2.0 * max_distance)?;
                    (c.triangle, c.barycentrics, (c.point - p).dot(n))
                }
            };
            let source = refs[src_tri as usize];
            Some(CorrespondenceEntry {
                source,
                barycentrics: mesh_order(bary, flip[source.instance as usize]),
                distance: dist,
                dest_triangle: s.triangle,
                dest_barycentrics: b,
            })
        })
        .collect();
    Ok(Correspondence {
        width: w,
        height: h,
        supersample: ss,
        entries,
    })
}

/// Per instance: does its transform mirror (and so swap corners 1 and 2 in world space)?
fn mirrored(source: &Scene) -> Vec<bool> {
    source
        .instances
        .iter()
        .map(|i| i.transform.determinant() < 0.0)
        .collect()
}

/// World-order barycentrics to the source mesh's corner order.
fn mesh_order(b: Vec3, flip: bool) -> Vec3 {
    if flip {
        Vec3::new(b.x, b.z, b.y)
    } else {
        b
    }
}

/// The correspondence for a destination that is a copy of the source (say
/// [`Scene::flatten`]'s mesh with new UVs): `source_triangles[t]` is the source triangle that
/// destination triangle `t` copies, corners in world order as `flatten` lays them out. Each
/// sample reads the source at its own barycentrics, so the transfer is exact and needs no rays.
pub fn copy_correspondence(
    dest: &Mesh,
    source: &Scene,
    source_triangles: &[TriRef],
    width: u32,
    height: u32,
    supersample: u32,
    uv_set: usize,
) -> Result<Correspondence> {
    if dest.uvs.len() <= uv_set {
        return Err(Error::InvalidMesh(format!(
            "destination has no UV set {uv_set}"
        )));
    }
    if source_triangles.len() != dest.triangle_count() {
        return Err(Error::InvalidInput(format!(
            "{} source triangles for {} destination triangles",
            source_triangles.len(),
            dest.triangle_count()
        )));
    }
    for r in source_triangles {
        let ok = source
            .instances
            .get(r.instance as usize)
            .is_some_and(|i| (r.triangle as usize) < source.meshes[i.mesh].triangle_count());
        if !ok {
            return Err(Error::InvalidInput(format!(
                "source triangle {r:?} does not exist"
            )));
        }
    }
    let ss = supersample.max(1);
    let (w, h) = (width * ss, height * ss);
    let flip = mirrored(source);
    let entries = rasterize_uv(dest, uv_set, w, h)
        .par_iter()
        .map(|s| {
            let s = (*s)?;
            let source = source_triangles[s.triangle as usize];
            Some(CorrespondenceEntry {
                source,
                barycentrics: mesh_order(s.barycentrics, flip[source.instance as usize]),
                distance: 0.0,
                dest_triangle: s.triangle,
                dest_barycentrics: s.barycentrics,
            })
        })
        .collect();
    Ok(Correspondence {
        width: w,
        height: h,
        supersample: ss,
        entries,
    })
}
