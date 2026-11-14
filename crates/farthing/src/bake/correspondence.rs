//! The correspondence buffer: for every (supersampled) destination texel, which source
//! surface point it maps to.

use crate::bake::raster::{band_triangles, rasterize_rows, TexelSample};
use crate::bvh::Bvh;
use crate::scene::TriRef;
use crate::{Error, Mesh, Result, Scene};
use glam::{Mat3, Mat4, Vec2, Vec3};
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
    let ss = supersample.max(1);
    let h = height * ss;
    Tracer::rays(
        dest,
        dest_to_world,
        source,
        width * ss,
        h,
        uv_set,
        max_distance,
        h,
    )
    .map(|t| t.whole(ss))
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
    let ss = supersample.max(1);
    let h = height * ss;
    Tracer::copy(dest, source, source_triangles, width * ss, h, uv_set, h).map(|t| t.whole(ss))
}

/// Builds correspondence entries a band of rows at a time, so a bake never needs them all at
/// once ([`build_correspondence`] and [`copy_correspondence`] take a single band).
pub(crate) struct Tracer<'a> {
    dest: &'a Mesh,
    uvs: &'a [Vec2],
    /// Sample grid size.
    width: u32,
    height: u32,
    /// Rows per band, and per band the destination triangles that reach it.
    band: u32,
    bands: Vec<Vec<u32>>,
    flip: Vec<bool>,
    lookup: Lookup<'a>,
}

enum Lookup<'a> {
    Copy(&'a [TriRef]),
    Rays(Box<Rays>),
}

struct Rays {
    bvh: Bvh,
    refs: Vec<TriRef>,
    dest_to_world: Mat4,
    nmat: Mat3,
    face_n: Vec<Vec3>,
    max_distance: f32,
}

impl<'a> Tracer<'a> {
    fn new(
        dest: &'a Mesh,
        source: &Scene,
        width: u32,
        height: u32,
        uv_set: usize,
        band: u32,
        lookup: Lookup<'a>,
    ) -> Tracer<'a> {
        let uvs = &dest.uvs[uv_set];
        let band = band.max(1);
        Tracer {
            dest,
            uvs,
            width,
            height,
            band,
            bands: band_triangles(dest, uvs, width, height, band),
            flip: mirrored(source),
            lookup,
        }
    }

    /// Ray-cast lookups (see [`build_correspondence`]) on a `width × height` sample grid,
    /// `band` rows at a time.
    pub(crate) fn rays(
        dest: &'a Mesh,
        dest_to_world: Mat4,
        source: &Scene,
        width: u32,
        height: u32,
        uv_set: usize,
        max_distance: f32,
        band: u32,
    ) -> Result<Tracer<'a>> {
        check_uv_set(dest, uv_set)?;
        let (tris, refs) = source.world_triangles();
        let bvh = Bvh::build(tris);
        let nmat = Mat3::from_mat4(dest_to_world).inverse().transpose();
        let face_n: Vec<Vec3> = (0..dest.triangle_count())
            .map(|t| {
                let [a, b, c] = dest.triangle_positions(t);
                (nmat * (b - a).cross(c - a)).normalize_or_zero()
            })
            .collect();
        let rays = Rays {
            bvh,
            refs,
            dest_to_world,
            nmat,
            face_n,
            max_distance,
        };
        Ok(Self::new(
            dest,
            source,
            width,
            height,
            uv_set,
            band,
            Lookup::Rays(Box::new(rays)),
        ))
    }

    /// Exact-copy lookups (see [`copy_correspondence`]) on a `width × height` sample grid,
    /// `band` rows at a time.
    pub(crate) fn copy(
        dest: &'a Mesh,
        source: &Scene,
        source_triangles: &'a [TriRef],
        width: u32,
        height: u32,
        uv_set: usize,
        band: u32,
    ) -> Result<Tracer<'a>> {
        check_uv_set(dest, uv_set)?;
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
        Ok(Self::new(
            dest,
            source,
            width,
            height,
            uv_set,
            band,
            Lookup::Copy(source_triangles),
        ))
    }

    /// Bands of rows, top to bottom.
    pub(crate) fn bands(&self) -> usize {
        self.bands.len()
    }

    /// The entries of band `index` (rows from `index × band`), row-major.
    pub(crate) fn rows(&self, index: usize) -> Vec<Option<CorrespondenceEntry>> {
        let y0 = index as u32 * self.band;
        let rows = self.band.min(self.height - y0);
        let samples = rasterize_rows(
            self.dest,
            self.uvs,
            self.width,
            self.height,
            y0,
            rows,
            self.bands[index].iter().map(|&t| t as usize),
        );
        samples.par_iter().map(|s| self.entry(s)).collect()
    }

    /// The whole grid in one band.
    pub(crate) fn whole(self, supersample: u32) -> Correspondence {
        let entries = if self.bands.is_empty() {
            Vec::new()
        } else {
            self.rows(0)
        };
        Correspondence {
            width: self.width,
            height: self.height,
            supersample,
            entries,
        }
    }

    fn entry(&self, s: &Option<TexelSample>) -> Option<CorrespondenceEntry> {
        let s = (*s)?;
        let (source, bary, distance) = match &self.lookup {
            Lookup::Copy(refs) => (refs[s.triangle as usize], s.barycentrics, 0.0),
            Lookup::Rays(r) => {
                let dest = self.dest;
                let tri = dest.triangle(s.triangle as usize);
                let b = s.barycentrics;
                let p_local = dest.positions[tri[0] as usize] * b.x
                    + dest.positions[tri[1] as usize] * b.y
                    + dest.positions[tri[2] as usize] * b.z;
                let p = r.dest_to_world.transform_point3(p_local);
                let n = match &dest.normals {
                    Some(ns) => (r.nmat
                        * (ns[tri[0] as usize] * b.x
                            + ns[tri[1] as usize] * b.y
                            + ns[tri[2] as usize] * b.z))
                        .normalize_or_zero(),
                    None => r.face_n[s.triangle as usize],
                };
                let n = if n == Vec3::ZERO {
                    r.face_n[s.triangle as usize]
                } else {
                    n
                };
                let origin = p + n * r.max_distance;
                let hit = r.bvh.intersect(origin, -n, 0.0, 2.0 * r.max_distance);
                let (src_tri, bary, dist) = match hit {
                    Some(hh) => (hh.triangle, hh.barycentrics(), r.max_distance - hh.t),
                    None => {
                        let c = r.bvh.closest_point(p, 2.0 * r.max_distance)?;
                        (c.triangle, c.barycentrics, (c.point - p).dot(n))
                    }
                };
                (r.refs[src_tri as usize], bary, dist)
            }
        };
        Some(CorrespondenceEntry {
            source,
            barycentrics: mesh_order(bary, self.flip[source.instance as usize]),
            distance,
            dest_triangle: s.triangle,
            dest_barycentrics: s.barycentrics,
        })
    }
}

fn check_uv_set(dest: &Mesh, uv_set: usize) -> Result<()> {
    if dest.uvs.len() <= uv_set {
        return Err(Error::InvalidMesh(format!(
            "destination has no UV set {uv_set}"
        )));
    }
    Ok(())
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
