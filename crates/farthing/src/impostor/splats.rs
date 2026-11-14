//! Gaussian splats from surfels: one flat 3D Gaussian per surfel, written in the 3D Gaussian
//! Splatting `.ply` layout (Kerbl et al. 2023, INRIA reference code), so a foliage proxy built
//! with [`progressive_surfel_cloud`](super::surfels::progressive_surfel_cloud) can be drawn by
//! any splat renderer (SuperSplat, gsplat, three.js / Babylon / Unreal / Unity splat plug-ins).
//!
//! Two ways to shape them, neither trained (everything comes from the mesh):
//!
//! * **Footprint splats** (with [`SurfelFootprint`]s from
//!   [`surfel_footprints`](super::footprint::surfel_footprints)): each Gaussian takes the
//!   shape of the patch of surface its surfel stands for. It is centred on the patch
//!   centroid, its tangent sigmas and rotation come from the patch's second moments (a leaf
//!   strip gives a stretched splat along the strip), its thickness from the patch's normal
//!   spread, and its opacity from the fraction of the footprint actually covered by surface,
//!   so sparse foliage at a coarse level is see-through instead of a solid blob.
//! * **Round splats** (no footprints): a disc of tangent sigma proportional to the surfel
//!   radius, at fixed opacity, rotated so its two wide axes lie in the surfel's tangent plane
//!   and its thin axis along the normal (a 2D-Gaussian-Splatting-style disc).
//!
//! Colour is spherical-harmonics degree 0 from a host-evaluated [`SurfelChannel`] (albedo, or
//! lit radiance if the host baked lighting in): evaluated at each surfel's point, or averaged
//! over its patch with [`average_surfel_channel`](super::footprint::average_surfel_channel).
//! farthing places and shapes the splats; the colours are the host's.
//!
//! The cloud's progressive order is kept, so the first `k` splats are a coarser even covering,
//! as for the discs: with sigmas grown by
//! [`SurfelCloud::lod`](super::surfels::SurfelCloud::lod)'s radius scale, or (better) with
//! footprints measured for that prefix by
//! [`SurfelCloud::footprints`](super::surfels::SurfelCloud::footprints).

use super::footprint::SurfelFootprint;
use super::surfels::{Surfel, SurfelChannel};
use crate::{Error, Result};
use glam::{Mat3, Quat, Vec3};

/// SH degree-0 basis constant: colour = 0.5 + C0 · f_dc.
const SH_C0: f32 = 0.282_094_8;

/// How surfels become Gaussians.
#[derive(Clone, Debug, PartialEq)]
pub struct SplatOptions {
    /// Round splats: tangent-plane sigma as a fraction of the surfel radius. 0.5 puts the disc
    /// edge at 2σ, where neighbouring splats overlap enough to leave no holes on an even
    /// covering.
    pub sigma_scale: f32,
    /// Footprint splats: tangent sigmas as a multiple of the patch's. A patch's own sigmas
    /// tile the surface edge to edge; the default 1.7 (the default surfel coverage factor)
    /// gives the same overlap as round splats, so a fully covered surface shows no holes.
    pub footprint_sigma_scale: f32,
    /// Smallest normal-axis sigma as a fraction of the smaller tangent sigma (flat but not
    /// degenerate, which some renderers reject). Footprint splats are thicker where the
    /// patch's normals spread.
    pub thickness: f32,
    /// Peak opacity, multiplied by the footprint coverage (footprint splats) and by the colour
    /// channel's alpha when it has four components (alpha-tested leaves).
    pub opacity: f32,
    /// Extra size multiplier, e.g. [`SurfelCloud::lod`](super::surfels::SurfelCloud::lod)'s
    /// scale when exporting a prefix of a cloud measured as a whole. Leave it at 1 for
    /// footprints measured for the prefix itself.
    pub radius_scale: f32,
}

impl Default for SplatOptions {
    fn default() -> Self {
        Self {
            sigma_scale: 0.5,
            footprint_sigma_scale: 1.7,
            thickness: 0.05,
            opacity: 0.98,
            radius_scale: 1.0,
        }
    }
}

/// One Gaussian in renderer-ready (activated) form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Splat {
    pub position: Vec3,
    /// Standard deviations along the rotated x, y (tangent) and z (normal) axes.
    pub sigma: Vec3,
    pub rotation: Quat,
    /// Linear 0..1 colour.
    pub color: Vec3,
    pub opacity: f32,
}

/// Gaussians for `surfels`: shaped by `footprints` when given (one per surfel, see the module
/// docs), round otherwise. Coloured from `color` (unorm8 RGB or RGBA per surfel, in cloud
/// order; mid grey without one).
pub fn surfels_to_splats(
    surfels: &[Surfel],
    footprints: Option<&[SurfelFootprint]>,
    color: Option<&SurfelChannel>,
    opts: &SplatOptions,
) -> Result<Vec<Splat>> {
    if let Some(f) = footprints {
        if f.len() != surfels.len() {
            return Err(Error::InvalidInput(format!(
                "{} footprints for {} surfels",
                f.len(),
                surfels.len()
            )));
        }
    }
    if let Some(c) = color {
        if !(3..=4).contains(&c.components) {
            return Err(Error::InvalidInput(format!(
                "splat colour channel `{}` has {} components (need 3 or 4)",
                c.name, c.components
            )));
        }
        if c.data.len() < surfels.len() * c.components as usize {
            return Err(Error::InvalidInput(format!(
                "splat colour channel `{}` covers {} surfels, the cloud has {}",
                c.name,
                c.data.len() / c.components as usize,
                surfels.len()
            )));
        }
    }
    Ok(surfels
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (rgb, a) = match color {
                Some(c) => {
                    let k = c.components as usize;
                    let px = &c.data[i * k..i * k + k];
                    let f = |b: u8| b as f32 / 255.0;
                    (
                        Vec3::new(f(px[0]), f(px[1]), f(px[2])),
                        if k == 4 { f(px[3]) } else { 1.0 },
                    )
                }
                None => (Vec3::splat(0.5), 1.0),
            };
            match footprints.map(|f| &f[i]) {
                Some(f) => footprint_splat(f, rgb, a, opts),
                None => round_splat(s, rgb, a, opts),
            }
        })
        .collect())
}

/// Any unit vector perpendicular to unit `n`.
fn any_tangent(n: Vec3) -> Vec3 {
    let t = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    n.cross(t).normalize()
}

fn round_splat(s: &Surfel, rgb: Vec3, a: f32, opts: &SplatOptions) -> Splat {
    let n = s.normal.try_normalize().unwrap_or(Vec3::Z);
    let u = any_tangent(n);
    let v = n.cross(u); // u × v = n: a proper rotation
    let st = s.radius * opts.radius_scale * opts.sigma_scale;
    Splat {
        position: s.position,
        sigma: Vec3::new(st, st, (st * opts.thickness).max(1e-7)),
        rotation: Quat::from_mat3(&Mat3::from_cols(u, v, n)).normalize(),
        color: rgb,
        opacity: (opts.opacity * a).clamp(0.0, 1.0),
    }
}

fn footprint_splat(f: &SurfelFootprint, rgb: Vec3, a: f32, opts: &SplatOptions) -> Splat {
    let n = f.normal.try_normalize().unwrap_or(Vec3::Z);
    let u = (f.tangent - n * f.tangent.dot(n))
        .try_normalize()
        .unwrap_or_else(|| any_tangent(n));
    let v = n.cross(u);
    let k = opts.footprint_sigma_scale * opts.radius_scale;
    let (sx, sy) = (f.sigma.x * k, f.sigma.y * k);
    let sz = (f.thickness * opts.radius_scale)
        .max(sy * opts.thickness)
        .max(1e-7);
    Splat {
        position: f.center,
        sigma: Vec3::new(sx, sy, sz),
        rotation: Quat::from_mat3(&Mat3::from_cols(u, v, n)).normalize(),
        color: rgb,
        opacity: (opts.opacity * f.coverage * a).clamp(0.0, 1.0),
    }
}

/// The standard 3DGS binary `.ply` (little endian; x y z, nx ny nz, f_dc_0..2, opacity,
/// scale_0..2, rot_0..3 with rot_0 = w). Scales are stored as ln σ, opacity as its logit and
/// colour as SH degree-0 coefficients, as the reference renderer expects. No higher SH bands
/// (view-independent colour), which every common viewer accepts.
pub fn write_splat_ply(splats: &[Splat]) -> Vec<u8> {
    let mut out = format!(
        "ply\nformat binary_little_endian 1.0\ncomment farthing surfel splats\nelement vertex {}\n",
        splats.len()
    )
    .into_bytes();
    for p in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        out.extend_from_slice(format!("property float {p}\n").as_bytes());
    }
    out.extend_from_slice(b"end_header\n");
    out.reserve(splats.len() * 17 * 4);
    for s in splats {
        let n = s.rotation * Vec3::Z;
        let o = s.opacity.clamp(1e-6, 1.0 - 1e-6);
        let q = s.rotation;
        let vals = [
            s.position.x,
            s.position.y,
            s.position.z,
            n.x,
            n.y,
            n.z,
            (s.color.x - 0.5) / SH_C0,
            (s.color.y - 0.5) / SH_C0,
            (s.color.z - 0.5) / SH_C0,
            (o / (1.0 - o)).ln(),
            s.sigma.x.max(1e-12).ln(),
            s.sigma.y.max(1e-12).ln(),
            s.sigma.z.max(1e-12).ln(),
            q.w,
            q.x,
            q.y,
            q.z,
        ];
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surfel(p: Vec3, n: Vec3, r: f32) -> Surfel {
        Surfel {
            position: p,
            normal: n,
            radius: r,
            triangle: 0,
            barycentrics: Vec3::splat(1.0 / 3.0),
        }
    }

    #[test]
    fn splats_lie_flat_on_their_surfels() {
        let ns = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(1.0, 2.0, -3.0).normalize(),
        ];
        let s: Vec<Surfel> = ns.iter().map(|&n| surfel(Vec3::ONE, n, 0.2)).collect();
        let splats = surfels_to_splats(&s, None, None, &SplatOptions::default()).unwrap();
        for (sp, n) in splats.iter().zip(ns) {
            assert!(
                (sp.rotation * Vec3::Z).dot(n) > 0.9999,
                "thin axis along the normal"
            );
            assert!((sp.sigma.x - 0.1).abs() < 1e-6 && sp.sigma.z < 0.01);
        }
    }

    #[test]
    fn ply_has_the_3dgs_layout() {
        let s = [surfel(Vec3::new(1.0, 2.0, 3.0), Vec3::Y, 0.4)];
        let c = SurfelChannel {
            name: "albedo".into(),
            components: 4,
            data: vec![255, 0, 128, 255],
        };
        let sp = surfels_to_splats(&s, None, Some(&c), &SplatOptions::default()).unwrap();
        let ply = write_splat_ply(&sp);
        let h = ply.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let header = std::str::from_utf8(&ply[..h]).unwrap();
        assert!(header.contains("element vertex 1\n"));
        assert!(
            header.contains("property float f_dc_0\n") && header.contains("property float rot_3\n")
        );
        let f: Vec<f32> = ply[h..]
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(f.len(), 17);
        assert_eq!(&f[..3], &[1.0, 2.0, 3.0]);
        assert!((0.5 + SH_C0 * f[6] - 1.0).abs() < 1e-5, "red decodes to 1");
        assert!((0.5 + SH_C0 * f[7]).abs() < 1e-5, "green decodes to 0");
        let opacity = 1.0 / (1.0 + (-f[9]).exp());
        assert!((opacity - 0.98).abs() < 1e-4);
        assert!((f[10].exp() - 0.2).abs() < 1e-5, "sigma = radius × 0.5");
        let q = Quat::from_xyzw(f[14], f[15], f[16], f[13]);
        assert!((q * Vec3::Z).dot(Vec3::Y) > 0.9999);
    }

    #[test]
    fn short_colour_channels_are_rejected() {
        let s = [surfel(Vec3::ZERO, Vec3::Z, 1.0); 2];
        let c = SurfelChannel {
            name: "albedo".into(),
            components: 3,
            data: vec![0; 3],
        };
        assert!(surfels_to_splats(&s, None, Some(&c), &SplatOptions::default()).is_err());
    }
}
