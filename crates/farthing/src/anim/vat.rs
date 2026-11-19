//! Vertex animation textures: a mesh's animated vertex positions (and normals) baked per
//! frame (the FVAT format), for the farthest levels of detail of a crowd.
//!
//! Far away, a unit is a few hundred vertices and skinning it costs more than reading its
//! vertices back: a VAT stores each frame's final positions, so the vertex shader reads one
//! texel per vertex (two, blending frames) and does no skinning. It costs memory per vertex
//! and frame, so it suits only small meshes. [`bake_vat`] samples the clips at the same times
//! as [`super::bake::bake`] with the same options, so the clip records match the palette's
//! and a unit's clip state drives either.
//!
//! # Quantization
//!
//! Positions are unorm16 per axis over the box of every frame's positions:
//! `p = bounds_min + q / 65535 × bounds_extent`; error per axis ≤ `bounds_extent / 131070`.
//! Normals (when the mesh has them) are octahedral, two snorm8 values: error below 1°.
//!
//! # File layout
//!
//! Little-endian, `u32`/`f32` unless stated; sections start on 16-byte boundaries. Clip
//! records and names are as in FANM ([`super::bake`]), and bytes 16–63 of the header read
//! as a uniform the same way.
//!
//! **Header** (80 bytes):
//!
//! | Offset | Field |
//! |---|---|
//! | 0 | magic `b"FVAT"` (4 bytes), version (= [`VERSION`]), header size (80), flags (bit 0: normals) |
//! | 16 | vertex count, clip count, frame count, fps |
//! | 32 | bounds min.xyz, clip array offset |
//! | 48 | bounds extent.xyz, texel array offset |
//! | 64 | name section offset, name section size, 2 × reserved |
//!
//! **Texels**: frame-major, `vertex count` texels a frame, 8 bytes each (one row of an
//! `Rgba16Uint` texture per frame, or a `vec2<u32>` storage array): x, y, z as unorm16, then
//! the normal (snorm8 octahedral x in the low byte, y in the high byte; 0 without normals).

use super::Animation;
use super::bake::{
    BakeError, BakeOptions, PaletteClip, align16, measure_with, read_clips, read_names, skin_point,
    write_clips, write_names,
};
use crate::scene::Skeleton;
use crate::{Error, Mesh, Result};
use glam::{Mat3, Vec3};

/// File magic.
pub const MAGIC: [u8; 4] = *b"FVAT";
/// Format version written by [`VertexAnimation::to_bytes`].
pub const VERSION: u32 = 1;
/// File header size in bytes.
pub const HEADER_SIZE: usize = 80;
/// Header flag: texels carry normals.
pub const FLAG_NORMALS: u32 = 1;

/// A baked vertex animation: see the [module docs](self) for the format.
#[derive(Clone, Debug, PartialEq)]
pub struct VertexAnimation {
    pub vertex_count: u32,
    pub fps: f32,
    pub normals: bool,
    pub bounds_min: Vec3,
    pub bounds_extent: Vec3,
    pub clips: Vec<PaletteClip>,
    /// Frame-major: `vertex count` texels per frame.
    pub texels: Vec<[u16; 4]>,
}

/// Bakes a skinned mesh's vertices over every clip of its skeleton (or the rest pose, as
/// one single-frame clip named "rest").
pub fn bake_vat(skeleton: &Skeleton, mesh: &Mesh, opts: &BakeOptions) -> Result<VertexAnimation> {
    let skin = mesh
        .skin
        .as_ref()
        .ok_or_else(|| Error::InvalidInput("mesh is not skinned".into()))?;
    // The palette's clip records, so the two agree on frames and loops.
    let palette = super::bake::bake(skeleton, opts);
    let rest_clip = Animation::default();
    let rest = skeleton.rest_pose();
    let n = mesh.positions.len();
    let mut positions = Vec::with_capacity(n * palette.frame_count());
    let mut normals = Vec::new();
    for (ci, clip) in palette.clips.iter().enumerate() {
        let anim = skeleton.animations.get(ci).unwrap_or(&rest_clip);
        let k = clip.frame_count;
        let dt = if k > 1 {
            clip.duration / (k - 1) as f32
        } else {
            0.0
        };
        for i in 0..k {
            let mut pose = rest.clone();
            anim.sample_into(dt * i as f32, &mut pose);
            let mats = skeleton.skinning_matrices(&skeleton.world_matrices(&pose));
            for v in 0..n {
                positions.push(skin_point(
                    &mats,
                    skin.joints[v],
                    skin.weights[v],
                    mesh.positions[v],
                ));
            }
            if let Some(ns) = &mesh.normals {
                for v in 0..n {
                    let mut out = Vec3::ZERO;
                    for (j, w) in skin.joints[v].iter().zip(skin.weights[v]) {
                        if let Some(m) = mats.get(*j as usize).filter(|_| w > 0.0) {
                            out += Mat3::from_mat4(*m) * ns[v] * w;
                        }
                    }
                    normals.push(out.normalize_or_zero());
                }
            }
        }
    }
    let (lo, hi) = positions.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let (lo, ext) = if positions.is_empty() {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (lo, hi - lo)
    };
    let q = |v: f32, lo: f32, ext: f32| {
        if ext > 0.0 {
            (((v - lo) / ext).clamp(0.0, 1.0) * 65535.0).round() as u16
        } else {
            0
        }
    };
    let texels = positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            [
                q(p.x, lo.x, ext.x),
                q(p.y, lo.y, ext.y),
                q(p.z, lo.z, ext.z),
                normals.get(i).map_or(0, |n| oct8_encode(*n)),
            ]
        })
        .collect();
    Ok(VertexAnimation {
        vertex_count: n as u32,
        fps: opts.fps,
        normals: mesh.normals.is_some(),
        bounds_min: lo,
        bounds_extent: ext,
        clips: palette.clips,
        texels,
    })
}

/// Octahedral encoding of a unit vector as two snorm8 values (x low, y high).
fn oct8_encode(n: Vec3) -> u16 {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    if l1 <= 0.0 || !l1.is_finite() {
        return 0;
    }
    let n = n / l1;
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let (x, y) = if n.z >= 0.0 {
        (n.x, n.y)
    } else {
        ((1.0 - n.y.abs()) * sign(n.x), (1.0 - n.x.abs()) * sign(n.y))
    };
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8 as u16;
    q(x) | (q(y) << 8)
}

fn oct8_decode(w: u16) -> Vec3 {
    let s = |v: u8| (v as i8 as f32 / 127.0).max(-1.0);
    let (x, y) = (s(w as u8), s((w >> 8) as u8));
    let z = 1.0 - x.abs() - y.abs();
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let (x, y) = if z < 0.0 {
        ((1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y))
    } else {
        (x, y)
    };
    Vec3::new(x, y, z).normalize_or_zero()
}

impl VertexAnimation {
    /// Total frames over all clips.
    pub fn frame_count(&self) -> usize {
        self.texels.len() / (self.vertex_count.max(1) as usize)
    }

    /// Vertex `vertex` of palette-wide frame `frame`: position, and normal (zero without).
    pub fn vertex(&self, frame: u32, vertex: u32) -> (Vec3, Vec3) {
        let t = self.texels[(frame * self.vertex_count + vertex) as usize];
        let un = |v: u16| v as f32 / 65535.0;
        let p = self.bounds_min + Vec3::new(un(t[0]), un(t[1]), un(t[2])) * self.bounds_extent;
        let n = if self.normals {
            oct8_decode(t[3])
        } else {
            Vec3::ZERO
        };
        (p, n)
    }

    /// Every vertex's position and normal `time` seconds into clip `clip`, blending the two
    /// frames around it as a shader would.
    pub fn sample(&self, clip: usize, time: f32) -> (Vec<Vec3>, Vec<Vec3>) {
        let (f0, f1, t) = self.clips[clip].frames_at(time);
        (0..self.vertex_count)
            .map(|v| {
                let (p0, n0) = self.vertex(f0, v);
                let (p1, n1) = self.vertex(f1, v);
                (p0.lerp(p1, t), n0.lerp(n1, t).normalize_or_zero())
            })
            .unzip()
    }

    /// Writes the FVAT file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let names = write_names(&self.clips);
        let clips_offset = HEADER_SIZE;
        let names_offset = align16(clips_offset + 16 * self.clips.len());
        let texels_offset = align16(names_offset + names.len());
        let mut out = Vec::with_capacity(texels_offset + 8 * self.texels.len());
        out.extend_from_slice(&MAGIC);
        let (lo, ext) = (self.bounds_min, self.bounds_extent);
        for v in [
            VERSION,
            HEADER_SIZE as u32,
            if self.normals { FLAG_NORMALS } else { 0 },
            self.vertex_count,
            self.clips.len() as u32,
            self.frame_count() as u32,
            self.fps.to_bits(),
            lo.x.to_bits(),
            lo.y.to_bits(),
            lo.z.to_bits(),
            clips_offset as u32,
            ext.x.to_bits(),
            ext.y.to_bits(),
            ext.z.to_bits(),
            texels_offset as u32,
            names_offset as u32,
            names.len() as u32,
            0,
            0,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        write_clips(&mut out, &self.clips);
        out.resize(names_offset, 0);
        out.extend_from_slice(&names);
        out.resize(texels_offset, 0);
        for t in &self.texels {
            for c in t {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.resize(align16(out.len()), 0);
        out
    }

    /// Reads an FVAT file, validating its layout.
    pub fn from_bytes(bytes: &[u8]) -> Result<VertexAnimation> {
        let bad = |m: &str| Error::InvalidInput(format!("FVAT: {m}"));
        if bytes.len() < HEADER_SIZE || bytes[..4] != MAGIC {
            return Err(bad("not a vertex animation"));
        }
        let u = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let f = |i: usize| f32::from_bits(u(i));
        if u(4) != VERSION {
            return Err(bad(&format!("unsupported version {}", u(4))));
        }
        if u(8) as usize != HEADER_SIZE {
            return Err(bad("unexpected header size"));
        }
        let (verts, clip_count, frames) = (u(16), u(20) as usize, u(24) as usize);
        let (clips_offset, texels_offset) = (u(44) as usize, u(60) as usize);
        let (names_offset, names_len) = (u(64) as usize, u(68) as usize);
        let count = frames
            .checked_mul(verts as usize)
            .ok_or_else(|| bad("too many frames"))?;
        let in_range =
            |off: usize, len: usize| off.checked_add(len).is_some_and(|e| e <= bytes.len());
        if verts == 0
            || !in_range(clips_offset, clip_count.saturating_mul(16))
            || !in_range(names_offset, names_len)
            || !in_range(texels_offset, count.saturating_mul(8))
        {
            return Err(bad("truncated or inconsistent sections"));
        }
        let names = read_names(&bytes[names_offset..names_offset + names_len], clip_count)
            .ok_or_else(|| bad("bad clip names"))?;
        let clips = read_clips(&bytes[clips_offset..], names, frames)
            .ok_or_else(|| bad("bad clip records"))?;
        let h = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let texels = (0..count)
            .map(|t| {
                let o = texels_offset + 8 * t;
                [h(o), h(o + 2), h(o + 4), h(o + 6)]
            })
            .collect();
        Ok(VertexAnimation {
            vertex_count: verts,
            fps: f(28),
            normals: u(12) & FLAG_NORMALS != 0,
            bounds_min: Vec3::new(f(32), f(36), f(40)),
            bounds_extent: Vec3::new(f(48), f(52), f(56)),
            clips,
            texels,
        })
    }

    /// Bytes per frame.
    pub fn frame_bytes(&self) -> usize {
        8 * self.vertex_count as usize
    }
}

/// Measures a VAT against sampling its skeleton's clips exactly, like
/// [`super::bake::measure`]. `mesh` is the mesh it was baked from.
pub fn measure_vat(vat: &VertexAnimation, skeleton: &Skeleton, mesh: &Mesh) -> Result<BakeError> {
    if vat.vertex_count as usize != mesh.positions.len() {
        return Err(Error::InvalidInput(
            "vertex animation and mesh have different vertex counts".into(),
        ));
    }
    measure_with(skeleton, mesh, &vat.clips, |c, t, frame| {
        let positions = match frame {
            Some(f) => (0..vat.vertex_count).map(|v| vat.vertex(f, v).0).collect(),
            None => vat.sample(c, t).0,
        };
        Box::new(move |v| positions[v.index])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::bake::{bake, tests::arm};

    #[test]
    fn vat_matches_palette_clips_and_skinning() {
        let (s, mut m) = arm();
        m.normals = Some(vec![Vec3::X; m.positions.len()]);
        let opts = BakeOptions::default();
        let vat = bake_vat(&s, &m, &opts).unwrap();
        assert_eq!(vat.clips, bake(&s, &opts).clips);
        assert_eq!(vat.frame_count(), 92);
        let e = measure_vat(&vat, &s, &m).unwrap();
        for c in &e.clips {
            assert!(c.quantization < 1e-4, "{c:?}");
            assert!(c.interpolation < 5e-3, "{c:?}");
        }
        // The tip vertex at the wave's peak: bent a quarter turn about the tip joint.
        let (p, n) = vat.sample(0, 0.5);
        assert!(
            p[8].abs_diff_eq(Vec3::new(-1.0, 1.1, 0.0), 1e-3),
            "{}",
            p[8]
        );
        assert!(n[8].abs_diff_eq(Vec3::Y, 0.02), "{}", n[8]);
    }

    #[test]
    fn vat_bytes_round_trip() {
        let (s, m) = arm();
        let vat = bake_vat(&s, &m, &BakeOptions::default()).unwrap();
        let bytes = vat.to_bytes();
        assert_eq!(bytes.len() % 16, 0);
        assert_eq!(VertexAnimation::from_bytes(&bytes).unwrap(), vat);
        assert!(VertexAnimation::from_bytes(&bytes[..90]).is_err());
    }

    #[test]
    fn oct8_round_trips() {
        for n in [Vec3::X, -Vec3::Z, Vec3::new(0.3, -0.5, 0.8).normalize()] {
            assert!(oct8_decode(oct8_encode(n)).angle_between(n) < 1f32.to_radians());
        }
    }
}
