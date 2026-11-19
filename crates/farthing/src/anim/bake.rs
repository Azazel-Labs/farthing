//! A GPU pose palette: a skeleton's clips baked into one flat buffer of quantized skinning
//! transforms (the FANM format) that a vertex shader samples by clip and time.
//!
//! The palette is built for crowds: every unit of a kind shares one palette, and a unit's
//! state is only the clip it plays and when it started, so the CPU writes to the GPU when a
//! unit changes clips, not every frame. [`bake`] builds it from a [`Skeleton`]'s
//! [`animations`](Skeleton::animations), [`PosePalette::to_bytes`] writes the file and
//! [`PosePalette::from_bytes`] reads (and validates) it back. [`PosePalette::sample`] is the
//! CPU reference of the shader's lookup ([`REFERENCE_SHADER`]), and [`measure`] reports what
//! the bake costs in vertex positions, against sampling the clips exactly.
//!
//! # Sampling
//!
//! Each clip is sampled at evenly spaced times from 0 to its duration, both ends included:
//! `n = ceil(duration × fps) + 1` frames (one frame for a clip without keys), so frame `i` is
//! at `duration × i / (n − 1)`. Each frame stores, per joint, the joint's *skinning*
//! transform (its posed matrix times its inverse bind matrix), so the shader skins a vertex
//! with the transforms as they are, without walking the hierarchy.
//!
//! A shader finds the frames around a time `τ` (seconds since the clip started) as
//! `x = τ / duration × (n − 1)`, wrapped into `[0, n − 1)` for a looping clip and clamped to
//! `[0, n − 1]` otherwise, then blends frames `floor(x)` and `floor(x) + 1` (the last frame
//! at the end): the rotations with a normalized lerp (negating the second when the two
//! disagree in sign), translations and scales linearly. A clip loops when its last pose
//! matches its first ([`BakeOptions::looping`]); its first and last frames are then the same
//! pose and the wrap is seamless.
//!
//! # Bone LOD
//!
//! Every level of detail of a mesh can share one palette. [`crate::bones::apply_bone_reduction`]
//! re-links a level's vertices to kept joints without renumbering them, so a reduced level
//! simply reads fewer of the palette's joints (and a level with fewer influences per vertex,
//! after [`crate::bones::limit_influences`], fewer still). Don't compact the skeleton of a
//! level that shares a palette. [`measure`] takes any level.
//!
//! # Quantization
//!
//! A joint's transform in a frame is 16 bytes:
//!
//! * **Rotation**: a unit quaternion, four snorm16 values. Consecutive frames of a clip keep
//!   the same hemisphere (the dot product of a joint's rotation in two neighboring frames is
//!   never negative), so blending neighbors never needs the sign fix. Error below 0.004°.
//! * **Translation**: unorm16 per axis over the box of every translation in the palette:
//!   `t = translation_min + q / 65535 × translation_extent`. Error per axis ≤
//!   `translation_extent / 131070`.
//! * **Scale**: one uniform scale, a 16-bit float. Non-uniform scale and mirroring can't be
//!   stored; [`measure`] shows the error when a skeleton has them. Scale 1 is exact.
//!
//! # File layout
//!
//! All values are little-endian; `u32`/`f32` unless stated. Sections start on 16-byte
//! boundaries. Bytes 16–63 of the header read as a uniform buffer (std140): joint count,
//! clip count, frame count, fps, `vec3` translation min, clip offset, `vec3` translation
//! extent, frame offset.
//!
//! **Header** (80 bytes, offset 0):
//!
//! | Offset | Field |
//! |---|---|
//! | 0 | magic `b"FANM"` (4 bytes), version (= [`VERSION`]), header size (80), flags (0) |
//! | 16 | joint count, clip count, frame count (all clips), fps (nominal) |
//! | 32 | translation min.xyz, clip array offset |
//! | 48 | translation extent.xyz, frame array offset |
//! | 64 | name section offset, name section size in bytes, 2 × reserved |
//!
//! **Clip** (16 bytes each; [`PaletteClip`]): first frame, frame count, duration (seconds),
//! flags (bit 0: loops).
//!
//! **Names**: per clip, its name's byte length (`u32`) and its UTF-8 bytes, padded to 4 bytes.
//!
//! **Frames**: frame-major, `joint count` transforms per frame, 16 bytes each, so joint `j` of
//! frame `f` is `uvec4` number `f × joint count + j`:
//!
//! | Word | Content |
//! |---|---|
//! | 0 | rotation x (low half), y (high half), snorm16: `unpack2x16snorm` |
//! | 1 | rotation z (low), w (high), snorm16 |
//! | 2 | translation x (low), y (high), unorm16: `unpack2x16unorm` |
//! | 3 | translation z (low, unorm16), scale (high, `f16`: `unpack2x16float(w).y`) |

use super::Animation;
use crate::scene::Skeleton;
use crate::{Error, Mesh, Result};
use glam::{Mat4, Quat, Vec3, Vec4};

/// File magic.
pub const MAGIC: [u8; 4] = *b"FANM";
/// Format version written by [`PosePalette::to_bytes`].
pub const VERSION: u32 = 1;
/// File header size in bytes.
pub const HEADER_SIZE: usize = 80;
/// Clip flag: the clip loops.
pub const CLIP_LOOPS: u32 = 1;

/// Reference WGSL for sampling a palette: the frame lookup, decoding and blending of
/// [`PosePalette::sample`], and skinning a vertex with up to four joints.
pub const REFERENCE_SHADER: &str = include_str!("../../shaders/pose_palette.wgsl");

/// Options for [`bake`].
#[derive(Clone, Debug)]
pub struct BakeOptions {
    /// Samples per second. 30 by default.
    pub fps: f32,
    /// Whether clips loop: `None` (the default) decides per clip, looping when the last
    /// pose matches the first (every joint within 0.5° and 0.1% of the palette's size).
    pub looping: Option<bool>,
}

impl Default for BakeOptions {
    fn default() -> Self {
        BakeOptions {
            fps: 30.0,
            looping: None,
        }
    }
}

/// One clip of a [`PosePalette`].
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteClip {
    pub name: String,
    /// Its first frame, counted over the whole palette.
    pub first_frame: u32,
    /// Frames in the clip, at least one.
    pub frame_count: u32,
    /// Seconds.
    pub duration: f32,
    pub looping: bool,
}

impl PaletteClip {
    /// The two frames (palette-wide indices) around `time` seconds into the clip, and how far
    /// between them it is (0–1), as the shader finds them.
    pub fn frames_at(&self, time: f32) -> (u32, u32, f32) {
        let n = self.frame_count.max(1);
        if n == 1 || self.duration <= 0.0 || !time.is_finite() {
            return (self.first_frame, self.first_frame, 0.0);
        }
        let last = (n - 1) as f32;
        let mut x = time / self.duration * last;
        x = if self.looping {
            x.rem_euclid(last)
        } else {
            x.clamp(0.0, last)
        };
        let f0 = (x.floor() as u32).min(n - 1);
        let f1 = (f0 + 1).min(n - 1);
        (self.first_frame + f0, self.first_frame + f1, x - f0 as f32)
    }
}

/// A joint's decoded skinning transform: scale (uniform), then rotation, then translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BonePose {
    pub rotation: Quat,
    pub translation: Vec3,
    pub scale: f32,
}

impl BonePose {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.translation,
        )
    }

    /// Blends two poses as the shader does: normalized lerp of the rotations (with the sign
    /// fix), linear translation and scale.
    pub fn blend(a: BonePose, b: BonePose, t: f32) -> BonePose {
        let (qa, mut qb) = (Vec4::from(a.rotation), Vec4::from(b.rotation));
        if qa.dot(qb) < 0.0 {
            qb = -qb;
        }
        BonePose {
            rotation: Quat::from_vec4(qa.lerp(qb, t).normalize()),
            translation: a.translation.lerp(b.translation, t),
            scale: a.scale + (b.scale - a.scale) * t,
        }
    }
}

/// A baked pose palette: see the [module docs](self) for the format.
#[derive(Clone, Debug, PartialEq)]
pub struct PosePalette {
    pub joint_count: u32,
    /// The rate the clips were sampled at.
    pub fps: f32,
    pub translation_min: Vec3,
    pub translation_extent: Vec3,
    pub clips: Vec<PaletteClip>,
    /// Packed transforms, frame-major: `joint count` per frame, four words each.
    pub bones: Vec<[u32; 4]>,
}

/// Bakes the skeleton's clips (or, without clips, its rest pose as one single-frame clip
/// named "rest") into a palette.
pub fn bake(skeleton: &Skeleton, opts: &BakeOptions) -> PosePalette {
    let joints = skeleton.joints.len();
    let rest_clip;
    let clips: &[Animation] = if skeleton.animations.is_empty() {
        rest_clip = [Animation {
            name: "rest".into(),
            tracks: Vec::new(),
        }];
        &rest_clip
    } else {
        &skeleton.animations
    };
    let rest = skeleton.rest_pose();

    // Exact skinning transforms of every frame.
    let mut poses: Vec<BonePose> = Vec::new();
    let mut records = Vec::new();
    for clip in clips {
        let first = poses.len() / joints.max(1);
        let mut prev: Option<usize> = None;
        let mut pose = rest.clone();
        for t in clip.frame_times(opts.fps) {
            pose.copy_from_slice(&rest);
            clip.sample_into(t, &mut pose);
            let worlds = skeleton.world_matrices(&pose);
            let start = poses.len();
            for m in skeleton.skinning_matrices(&worlds) {
                poses.push(decompose(m));
            }
            // Same hemisphere as the previous frame.
            if let Some(p) = prev {
                for j in 0..joints {
                    if poses[p + j].rotation.dot(poses[start + j].rotation) < 0.0 {
                        poses[start + j].rotation = -poses[start + j].rotation;
                    }
                }
            }
            prev = Some(start);
        }
        records.push(PaletteClip {
            name: clip.name.clone(),
            first_frame: first as u32,
            frame_count: (poses.len() / joints.max(1) - first) as u32,
            duration: clip.duration(),
            looping: false,
        });
    }

    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for p in &poses {
        lo = lo.min(p.translation);
        hi = hi.max(p.translation);
    }
    if poses.is_empty() {
        (lo, hi) = (Vec3::ZERO, Vec3::ZERO);
    }
    let extent = hi - lo;
    let size = extent.max_element();
    for r in &mut records {
        r.looping = match opts.looping {
            Some(l) => l && r.frame_count > 1,
            None => {
                let (a, b) = (
                    r.first_frame as usize,
                    (r.first_frame + r.frame_count - 1) as usize,
                );
                r.frame_count > 2
                    && (0..joints).all(|j| {
                        let (p, q) = (poses[a * joints + j], poses[b * joints + j]);
                        p.rotation.dot(q.rotation).abs() >= (0.25f32.to_radians()).cos()
                            && p.translation.distance(q.translation) <= 1e-3 * size.max(1e-6)
                            && (p.scale - q.scale).abs() <= 1e-3 * p.scale.abs().max(1e-6)
                    })
            }
        };
    }

    let unorm = |v: f32, lo: f32, ext: f32| {
        if ext > 0.0 {
            (((v - lo) / ext).clamp(0.0, 1.0) * 65535.0).round() as u32
        } else {
            0
        }
    };
    let bones = poses
        .iter()
        .map(|p| {
            let q = p.rotation;
            let t = p.translation;
            [
                snorm16(q.x) | (snorm16(q.y) << 16),
                snorm16(q.z) | (snorm16(q.w) << 16),
                unorm(t.x, lo.x, extent.x) | (unorm(t.y, lo.y, extent.y) << 16),
                unorm(t.z, lo.z, extent.z) | ((f32_to_f16(p.scale) as u32) << 16),
            ]
        })
        .collect();
    PosePalette {
        joint_count: joints as u32,
        fps: opts.fps,
        translation_min: lo,
        translation_extent: extent,
        clips: records,
        bones,
    }
}

/// A skinning matrix as rotation, translation and one uniform scale (the mean of the three).
fn decompose(m: Mat4) -> BonePose {
    let (s, r, t) = m.to_scale_rotation_translation();
    BonePose {
        rotation: r.normalize(),
        translation: t,
        scale: (s.x + s.y + s.z) / 3.0,
    }
}

fn snorm16(v: f32) -> u32 {
    ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16) as u16 as u32
}

fn unsnorm16(v: u32) -> f32 {
    (v as u16 as i16 as f32 / 32767.0).max(-1.0)
}

/// `f32` to IEEE half, rounding to nearest even.
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let round = |r: u32, rem: u32, half: u32| {
        if rem > half || (rem == half && r & 1 == 1) {
            r + 1
        } else {
            r
        }
    };
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        return sign | round(m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1)) as u16;
    }
    sign | round(((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000) as u16
}

/// IEEE half to `f32`.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    match exp {
        0 => {
            let v = man as f32 * (-24f32).exp2();
            if sign != 0 { -v } else { v }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

impl PosePalette {
    /// Total frames over all clips.
    pub fn frame_count(&self) -> usize {
        self.bones.len() / (self.joint_count.max(1) as usize)
    }

    /// The clip with this name.
    pub fn clip(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|c| c.name == name)
    }

    /// Decodes joint `joint` of palette-wide frame `frame`.
    pub fn bone(&self, frame: u32, joint: u32) -> BonePose {
        let w = self.bones[(frame * self.joint_count + joint) as usize];
        let un = |v: u32| (v & 0xffff) as f32 / 65535.0;
        let (lo, ext) = (self.translation_min, self.translation_extent);
        BonePose {
            rotation: Quat::from_xyzw(
                unsnorm16(w[0]),
                unsnorm16(w[0] >> 16),
                unsnorm16(w[1]),
                unsnorm16(w[1] >> 16),
            )
            .normalize(),
            translation: lo + Vec3::new(un(w[2]), un(w[2] >> 16), un(w[3])) * ext,
            scale: f16_to_f32((w[3] >> 16) as u16),
        }
    }

    /// Every joint's skinning transform `time` seconds into clip `clip`, as the reference
    /// shader computes it.
    pub fn sample(&self, clip: usize, time: f32) -> Vec<BonePose> {
        let (f0, f1, t) = self.clips[clip].frames_at(time);
        (0..self.joint_count)
            .map(|j| BonePose::blend(self.bone(f0, j), self.bone(f1, j), t))
            .collect()
    }

    /// [`PosePalette::sample`] as matrices.
    pub fn skinning_matrices(&self, clip: usize, time: f32) -> Vec<Mat4> {
        self.sample(clip, time)
            .iter()
            .map(BonePose::matrix)
            .collect()
    }

    /// Writes the FANM file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let names = write_names(&self.clips);
        let clips_offset = HEADER_SIZE;
        let names_offset = align16(clips_offset + 16 * self.clips.len());
        let frames_offset = align16(names_offset + names.len());
        let mut out = Vec::with_capacity(frames_offset + 16 * self.bones.len());
        out.extend_from_slice(&MAGIC);
        let words = |out: &mut Vec<u8>, w: &[u32]| {
            for v in w {
                out.extend_from_slice(&v.to_le_bytes());
            }
        };
        let (lo, ext) = (self.translation_min, self.translation_extent);
        words(
            &mut out,
            &[
                VERSION,
                HEADER_SIZE as u32,
                0,
                self.joint_count,
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
                frames_offset as u32,
                names_offset as u32,
                names.len() as u32,
                0,
                0,
            ],
        );
        write_clips(&mut out, &self.clips);
        out.resize(names_offset, 0);
        out.extend_from_slice(&names);
        out.resize(frames_offset, 0);
        for b in &self.bones {
            words(&mut out, b);
        }
        out
    }

    /// Reads a FANM file, validating its layout.
    pub fn from_bytes(bytes: &[u8]) -> Result<PosePalette> {
        let bad = |m: &str| Error::InvalidInput(format!("FANM: {m}"));
        if bytes.len() < HEADER_SIZE || bytes[..4] != MAGIC {
            return Err(bad("not a pose palette"));
        }
        let u = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let f = |i: usize| f32::from_bits(u(i));
        if u(4) != VERSION {
            return Err(bad(&format!("unsupported version {}", u(4))));
        }
        if u(8) as usize != HEADER_SIZE {
            return Err(bad("unexpected header size"));
        }
        let (joints, clip_count, frames) = (u(16), u(20) as usize, u(24) as usize);
        let clips_offset = u(44) as usize;
        let frames_offset = u(60) as usize;
        let (names_offset, names_len) = (u(64) as usize, u(68) as usize);
        let bone_count = frames
            .checked_mul(joints as usize)
            .ok_or_else(|| bad("too many frames"))?;
        let in_range =
            |off: usize, len: usize| off.checked_add(len).is_some_and(|e| e <= bytes.len());
        if joints == 0
            || !in_range(clips_offset, clip_count.saturating_mul(16))
            || !in_range(names_offset, names_len)
            || !in_range(frames_offset, bone_count.saturating_mul(16))
        {
            return Err(bad("truncated or inconsistent sections"));
        }
        let names = read_names(&bytes[names_offset..names_offset + names_len], clip_count)
            .ok_or_else(|| bad("bad clip names"))?;
        let clips = read_clips(&bytes[clips_offset..], names, frames)
            .ok_or_else(|| bad("bad clip records"))?;
        let bones = (0..bone_count)
            .map(|b| {
                let o = frames_offset + 16 * b;
                [u(o), u(o + 4), u(o + 8), u(o + 12)]
            })
            .collect();
        Ok(PosePalette {
            joint_count: joints,
            fps: f(28),
            translation_min: Vec3::new(f(32), f(36), f(40)),
            translation_extent: Vec3::new(f(48), f(52), f(56)),
            clips,
            bones,
        })
    }
}

#[inline]
pub(crate) fn align16(n: usize) -> usize {
    n.div_ceil(16) * 16
}

pub(crate) fn write_clips(out: &mut Vec<u8>, clips: &[PaletteClip]) {
    for c in clips {
        for v in [
            c.first_frame,
            c.frame_count,
            c.duration.to_bits(),
            if c.looping { CLIP_LOOPS } else { 0 },
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Reads clip records, checking their frames lie within `frames`.
pub(crate) fn read_clips(
    bytes: &[u8],
    names: Vec<String>,
    frames: usize,
) -> Option<Vec<PaletteClip>> {
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let w = |k: usize| {
                u32::from_le_bytes(
                    bytes[16 * i + 4 * k..16 * i + 4 * k + 4]
                        .try_into()
                        .unwrap(),
                )
            };
            let c = PaletteClip {
                name,
                first_frame: w(0),
                frame_count: w(1),
                duration: f32::from_bits(w(2)),
                looping: w(3) & CLIP_LOOPS != 0,
            };
            let end = c.first_frame as usize + c.frame_count as usize;
            (c.frame_count > 0 && end <= frames && c.duration.is_finite() && c.duration >= 0.0)
                .then_some(c)
        })
        .collect()
}

pub(crate) fn write_names(clips: &[PaletteClip]) -> Vec<u8> {
    let mut out = Vec::new();
    for c in clips {
        out.extend_from_slice(&(c.name.len() as u32).to_le_bytes());
        out.extend_from_slice(c.name.as_bytes());
        out.resize(out.len().div_ceil(4) * 4, 0);
    }
    out
}

pub(crate) fn read_names(bytes: &[u8], count: usize) -> Option<Vec<String>> {
    let mut at = 0usize;
    let mut names = Vec::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        let len = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize;
        let s = bytes.get(at + 4..at.checked_add(4 + len)?)?;
        names.push(String::from_utf8(s.to_vec()).ok()?);
        at = (at + 4 + len).div_ceil(4) * 4;
    }
    Some(names)
}

/// How far a bake moves a mesh's vertices from where sampling its clips exactly puts them.
#[derive(Clone, Debug, PartialEq)]
pub struct BakeError {
    /// The mesh's bounding-box diagonal in the bind pose, to put the errors in proportion.
    pub size: f32,
    pub clips: Vec<ClipError>,
}

/// [`BakeError`] of one clip, in mesh units.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipError {
    pub name: String,
    /// The largest vertex error at the baked frames' own times: what quantization costs.
    pub quantization: f32,
    /// The largest vertex error halfway between frames: quantization plus what blending
    /// frames instead of sampling the curves costs. Raise [`BakeOptions::fps`] to lower it.
    pub interpolation: f32,
}

impl BakeError {
    /// The largest error of any clip.
    pub fn max(&self) -> f32 {
        self.clips
            .iter()
            .map(|c| c.quantization.max(c.interpolation))
            .fold(0.0, f32::max)
    }
}

/// Skins a point with up to four joints.
pub(crate) fn skin_point(mats: &[Mat4], joints: [u16; 4], weights: [f32; 4], p: Vec3) -> Vec3 {
    let mut out = Vec3::ZERO;
    let mut total = 0.0;
    for k in 0..4 {
        let w = weights[k];
        if w > 0.0
            && let Some(m) = mats.get(joints[k] as usize)
        {
            out += m.transform_point3(p) * w;
            total += w;
        }
    }
    if total > 0.0 { out / total } else { p }
}

/// Measures every clip of a palette against its skeleton on a skinned mesh (any bone LOD of
/// it): the largest vertex error at the frames and between them. At most 4096 of the mesh's
/// vertices are tested, evenly spread. The palette must have been baked from this skeleton.
pub fn measure(palette: &PosePalette, skeleton: &Skeleton, mesh: &Mesh) -> Result<BakeError> {
    if palette.joint_count as usize != skeleton.joints.len() {
        return Err(Error::InvalidInput(
            "pose palette and skeleton have different joint counts".into(),
        ));
    }
    measure_with(skeleton, mesh, &palette.clips, |c, t, _| {
        let m = palette.skinning_matrices(c, t);
        Box::new(move |v: &SkinnedVertex| skin_point(&m, v.joints, v.weights, v.position))
    })
}

/// A vertex under test: its index, bind position and skin.
pub(crate) struct SkinnedVertex {
    pub index: usize,
    pub position: Vec3,
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

type Decoder<'a> = Box<dyn Fn(&SkinnedVertex) -> Vec3 + 'a>;

/// Shared by the palette and VAT measures: `decode(clip, time, frame)` returns a function from
/// a test vertex to its decoded position; `frame` is the palette-wide frame at the frame times,
/// `None` between frames.
pub(crate) fn measure_with<'a>(
    skeleton: &Skeleton,
    mesh: &Mesh,
    clips: &[PaletteClip],
    decode: impl Fn(usize, f32, Option<u32>) -> Decoder<'a>,
) -> Result<BakeError> {
    let skin = mesh
        .skin
        .as_ref()
        .ok_or_else(|| Error::InvalidInput("mesh is not skinned".into()))?;
    let n = mesh.positions.len();
    let step = n.div_ceil(4096).max(1);
    let verts: Vec<SkinnedVertex> = (0..n)
        .step_by(step)
        .map(|i| SkinnedVertex {
            index: i,
            position: mesh.positions[i],
            joints: skin.joints[i],
            weights: skin.weights[i],
        })
        .collect();
    let (lo, hi) = mesh.positions.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let size = if n > 0 { (hi - lo).length() } else { 0.0 };
    let rest = skeleton.rest_pose();
    let rest_clip = Animation::default();
    let mut out = Vec::new();
    for (ci, clip) in clips.iter().enumerate() {
        let anim = skeleton.animations.get(ci).unwrap_or(&rest_clip);
        let exact = |t: f32| {
            let mut pose = rest.clone();
            anim.sample_into(t, &mut pose);
            skeleton.skinning_matrices(&skeleton.world_matrices(&pose))
        };
        let err = |t: f32, frame: Option<u32>| {
            let m = exact(t);
            let d = decode(ci, t, frame);
            verts
                .iter()
                .map(|v| skin_point(&m, v.joints, v.weights, v.position).distance(d(v)))
                .fold(0.0, f32::max)
        };
        let k = clip.frame_count;
        let dt = if k > 1 {
            clip.duration / (k - 1) as f32
        } else {
            0.0
        };
        let quantization = (0..k)
            .map(|i| err(dt * i as f32, Some(clip.first_frame + i)))
            .fold(0.0, f32::max);
        let interpolation = (0..k.saturating_sub(1))
            .map(|i| err(dt * (i as f32 + 0.5), None))
            .fold(0.0, f32::max);
        out.push(ClipError {
            name: clip.name.clone(),
            quantization,
            interpolation,
        });
    }
    Ok(BakeError { size, clips: out })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::anim::{Curve, Track};
    use crate::{Joint, Skin, Transform};

    /// A two-joint arm, its tip bending a quarter turn back and forth over a second, and a
    /// strip of vertices along it.
    pub(crate) fn arm() -> (Skeleton, Mesh) {
        let mut s = Skeleton {
            name: "arm".into(),
            joints: vec![
                Joint {
                    name: "root".into(),
                    parent: None,
                    inverse_bind: Mat4::IDENTITY,
                    rest: Transform::IDENTITY,
                },
                Joint {
                    name: "tip".into(),
                    parent: Some(0),
                    inverse_bind: Mat4::from_translation(-Vec3::Y),
                    rest: Transform::IDENTITY,
                },
            ],
            animations: Vec::new(),
        };
        s.rest_from_bind();
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        s.animations.push(Animation {
            name: "wave".into(),
            tracks: vec![Track {
                rotation: Some(Curve::linear(
                    vec![0.0, 0.5, 1.0],
                    vec![Quat::IDENTITY, q, Quat::IDENTITY],
                )),
                ..Track::new(1)
            }],
        });
        s.animations.push(Animation {
            name: "reach".into(),
            tracks: vec![Track {
                translation: Some(Curve::linear(
                    vec![0.0, 2.0],
                    vec![Vec3::ZERO, Vec3::X * 3.0],
                )),
                ..Track::new(0)
            }],
        });
        let positions: Vec<Vec3> = (0..9)
            .map(|i| Vec3::new(0.1, i as f32 * 0.25, 0.0))
            .collect();
        let mut m = Mesh::new(positions, Vec::new());
        m.skin = Some(Skin {
            joints: (0..9)
                .map(|i| if i < 4 { [0, 1, 0, 0] } else { [1, 0, 0, 0] })
                .collect(),
            weights: (0..9)
                .map(|i| {
                    if i < 4 {
                        [0.75, 0.25, 0.0, 0.0]
                    } else {
                        [1.0, 0.0, 0.0, 0.0]
                    }
                })
                .collect(),
        });
        (s, m)
    }

    #[test]
    fn half_floats_round_trip() {
        for v in [0.0f32, 1.0, -2.5, 0.333, 65504.0, 1e-5, 6.1e-5, -1e-7] {
            let h = f16_to_f32(f32_to_f16(v));
            assert!((h - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {h}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
    }

    #[test]
    fn bakes_frames_and_detects_loops() {
        let (s, _) = arm();
        let p = bake(&s, &BakeOptions::default());
        assert_eq!(p.joint_count, 2);
        assert_eq!(p.clips.len(), 2);
        assert_eq!(p.clips[0].frame_count, 31);
        assert_eq!(p.clips[1].first_frame, 31);
        assert_eq!(p.clips[1].frame_count, 61);
        assert!(p.clips[0].looping, "wave ends where it starts");
        assert!(!p.clips[1].looping, "reach doesn't");
        assert_eq!(p.frame_count(), 92);
        assert_eq!(p.bones.len(), 92 * 2);
    }

    #[test]
    fn frame_lookup_wraps_or_clamps() {
        let c = PaletteClip {
            name: String::new(),
            first_frame: 10,
            frame_count: 5,
            duration: 2.0,
            looping: true,
        };
        assert_eq!(c.frames_at(0.0), (10, 11, 0.0));
        assert_eq!(c.frames_at(1.25), (12, 13, 0.5));
        let (a, b, t) = c.frames_at(2.25);
        assert_eq!((a, b), (10, 11));
        assert!((t - 0.5).abs() < 1e-5);
        let once = PaletteClip {
            looping: false,
            ..c
        };
        assert_eq!(once.frames_at(9.0), (14, 14, 0.0));
        assert_eq!(once.frames_at(-1.0), (10, 11, 0.0));
    }

    #[test]
    fn decoded_palette_is_close_to_exact_skinning() {
        let (s, m) = arm();
        let p = bake(&s, &BakeOptions::default());
        let e = measure(&p, &s, &m).unwrap();
        assert_eq!(e.clips.len(), 2);
        for c in &e.clips {
            assert!(c.quantization < 2e-4, "{c:?}");
            assert!(c.interpolation < 5e-3, "{c:?}");
        }
        // Fewer frames: blending errs more between them.
        let coarse = bake(
            &s,
            &BakeOptions {
                fps: 4.0,
                ..Default::default()
            },
        );
        let ec = measure(&coarse, &s, &m).unwrap();
        assert!(ec.clips[0].interpolation > e.clips[0].interpolation);
    }

    #[test]
    fn bytes_round_trip_and_validate() {
        let (s, _) = arm();
        let p = bake(&s, &BakeOptions::default());
        let bytes = p.to_bytes();
        assert_eq!(bytes.len() % 16, 0);
        assert_eq!(PosePalette::from_bytes(&bytes).unwrap(), p);
        assert!(PosePalette::from_bytes(&bytes[..100]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(PosePalette::from_bytes(&bad).is_err());
    }

    #[test]
    fn rest_only_skeleton_bakes_one_frame() {
        let (mut s, m) = arm();
        s.animations.clear();
        let p = bake(&s, &BakeOptions::default());
        assert_eq!(p.clips.len(), 1);
        assert_eq!(p.clips[0].name, "rest");
        assert_eq!(p.clips[0].frame_count, 1);
        assert!(measure(&p, &s, &m).unwrap().max() < 1e-4);
    }

    #[test]
    fn reference_shader_validates() {
        let module = naga::front::wgsl::parse_str(REFERENCE_SHADER).expect("parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validate");
    }
}
