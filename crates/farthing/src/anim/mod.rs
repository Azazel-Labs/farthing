//! Skeletal animation.
//!
//! An [`Animation`] is a clip: [`Track`]s of keyframed translation, rotation and scale curves,
//! one track per animated joint, that replace parts of the joint's [`Joint::rest`] transform
//! over time. Clips live on their [`Skeleton`] ([`Skeleton::animations`]); glTF import reads
//! them and export writes them back.
//!
//! [`bake`] turns a skeleton's clips into a compact GPU pose palette (the FANM format) that a
//! vertex shader samples by clip and time, so a crowd of units animates with no per-frame
//! CPU work; [`vat`] bakes a mesh's animated vertex positions into a texture for the
//! farthest LODs. [`sample_poses`] feeds real poses to the bone planner
//! ([`crate::bones::plan_bone_reduction`]).
//!
//! [`Joint::rest`]: crate::Joint::rest

pub mod bake;
pub mod vat;

use crate::scene::{Skeleton, Transform};
use glam::{Mat4, Quat, Vec3, Vec4};

/// How a [`Curve`] fills the time between keys, as glTF's sampler interpolation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// The previous key's value holds until the next key.
    Step,
    /// Straight lines between keys; rotations are spherically interpolated.
    #[default]
    Linear,
    /// Cubic Hermite splines with per-key in and out tangents. Rotations are normalized
    /// afterwards.
    CubicSpline,
}

/// Keyframes of one value over time.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve<T> {
    pub interpolation: Interpolation,
    /// Key times in seconds, increasing.
    pub times: Vec<f32>,
    /// One value per key, or for [`Interpolation::CubicSpline`] three per key, laid out as
    /// glTF has them: in tangent, value, out tangent.
    pub values: Vec<T>,
}

/// A value a [`Curve`] can interpolate.
pub trait Animatable: Copy {
    /// Linear (or for rotations, spherical) interpolation.
    fn interpolate(a: Self, b: Self, t: f32) -> Self;
    /// `h00·p0 + h10·m0 + h01·p1 + h11·m1`, tangents already scaled by the key interval.
    fn hermite(p0: Self, m0: Self, p1: Self, m1: Self, h: [f32; 4]) -> Self;
}

impl Animatable for Vec3 {
    fn interpolate(a: Self, b: Self, t: f32) -> Self {
        a.lerp(b, t)
    }
    fn hermite(p0: Self, m0: Self, p1: Self, m1: Self, h: [f32; 4]) -> Self {
        p0 * h[0] + m0 * h[1] + p1 * h[2] + m1 * h[3]
    }
}

impl Animatable for Quat {
    fn interpolate(a: Self, b: Self, t: f32) -> Self {
        a.slerp(b, t)
    }
    fn hermite(p0: Self, m0: Self, p1: Self, m1: Self, h: [f32; 4]) -> Self {
        let v = Vec4::from(p0) * h[0]
            + Vec4::from(m0) * h[1]
            + Vec4::from(p1) * h[2]
            + Vec4::from(m1) * h[3];
        Quat::from_vec4(v).normalize()
    }
}

impl<T: Animatable> Curve<T> {
    /// A linear curve.
    pub fn linear(times: Vec<f32>, values: Vec<T>) -> Curve<T> {
        Curve {
            interpolation: Interpolation::Linear,
            times,
            values,
        }
    }

    /// Time of the last key, 0 for an empty curve.
    pub fn end(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// The value of key `k`.
    pub fn key(&self, k: usize) -> T {
        match self.interpolation {
            Interpolation::CubicSpline => self.values[3 * k + 1],
            _ => self.values[k],
        }
    }

    /// Whether there are as many values as the interpolation needs for the keys.
    pub fn is_valid(&self) -> bool {
        let per = match self.interpolation {
            Interpolation::CubicSpline => 3,
            _ => 1,
        };
        !self.times.is_empty() && self.values.len() == per * self.times.len()
    }

    /// The value at `time`, holding the first and last keys outside their range. `None` for
    /// an empty or malformed curve.
    pub fn sample(&self, time: f32) -> Option<T> {
        if !self.is_valid() {
            return None;
        }
        let n = self.times.len();
        if n == 1 || time <= self.times[0] || time.is_nan() {
            return Some(self.key(0));
        }
        if time >= self.times[n - 1] {
            return Some(self.key(n - 1));
        }
        // The last key at or before `time`.
        let k = self.times.partition_point(|&t| t <= time) - 1;
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let dt = t1 - t0;
        let u = if dt > 0.0 { (time - t0) / dt } else { 0.0 };
        Some(match self.interpolation {
            Interpolation::Step => self.values[k],
            Interpolation::Linear => T::interpolate(self.values[k], self.values[k + 1], u),
            Interpolation::CubicSpline => {
                let (u2, u3) = (u * u, u * u * u);
                let h = [
                    2.0 * u3 - 3.0 * u2 + 1.0,
                    (u3 - 2.0 * u2 + u) * dt,
                    -2.0 * u3 + 3.0 * u2,
                    (u3 - u2) * dt,
                ];
                let p0 = self.values[3 * k + 1];
                let m0 = self.values[3 * k + 2];
                let p1 = self.values[3 * (k + 1) + 1];
                let m1 = self.values[3 * (k + 1)];
                T::hermite(p0, m0, p1, m1, h)
            }
        })
    }

    /// Applies `f` to every value, and `tangent` to the tangents of a cubic curve.
    fn map(&mut self, f: impl Fn(T) -> T, tangent: impl Fn(T) -> T) {
        let cubic = self.interpolation == Interpolation::CubicSpline;
        for (i, v) in self.values.iter_mut().enumerate() {
            *v = if cubic && i % 3 != 1 {
                tangent(*v)
            } else {
                f(*v)
            };
        }
    }
}

/// The animated parts of one joint. A missing curve leaves that part at the joint's rest.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    /// The joint's index in its [`Skeleton`].
    pub joint: usize,
    pub translation: Option<Curve<Vec3>>,
    pub rotation: Option<Curve<Quat>>,
    pub scale: Option<Curve<Vec3>>,
}

impl Track {
    pub fn new(joint: usize) -> Track {
        Track {
            joint,
            translation: None,
            rotation: None,
            scale: None,
        }
    }

    /// Time of the last key of any curve.
    pub fn end(&self) -> f32 {
        let t = self.translation.as_ref().map_or(0.0, Curve::end);
        let r = self.rotation.as_ref().map_or(0.0, Curve::end);
        let s = self.scale.as_ref().map_or(0.0, Curve::end);
        t.max(r).max(s)
    }

    /// Writes the track's value at `time` over the parts of `pose` it animates.
    pub fn apply(&self, time: f32, pose: &mut Transform) {
        if let Some(v) = self.translation.as_ref().and_then(|c| c.sample(time)) {
            pose.translation = v;
        }
        if let Some(v) = self.rotation.as_ref().and_then(|c| c.sample(time)) {
            pose.rotation = v.normalize();
        }
        if let Some(v) = self.scale.as_ref().and_then(|c| c.sample(time)) {
            pose.scale = v;
        }
    }

    /// Re-expresses the curves under a parent transform `offset`: the joint's transform
    /// becomes `offset · transform`. Exact for an offset with uniform scale (which is how
    /// glTF import folds non-joint nodes between two joints into the child joint).
    pub fn premultiply(&mut self, offset: Transform) {
        let (t_o, r_o, s_o) = (offset.translation, offset.rotation, offset.scale);
        if let Some(c) = &mut self.translation {
            c.map(|t| t_o + r_o * (s_o * t), |m| r_o * (s_o * m));
        }
        if let Some(c) = &mut self.rotation {
            c.map(|r| (r_o * r).normalize(), |m| r_o * m);
        }
        if let Some(c) = &mut self.scale {
            c.map(|s| s_o * s, |m| s_o * m);
        }
    }
}

/// A clip: tracks that animate joints of a [`Skeleton`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Animation {
    pub name: String,
    pub tracks: Vec<Track>,
}

impl Animation {
    /// Length in seconds: from 0 to the last key of any track.
    pub fn duration(&self) -> f32 {
        self.tracks.iter().map(Track::end).fold(0.0, f32::max)
    }

    /// The joints' local transforms at `time`: the skeleton's rest pose with the tracks
    /// applied.
    pub fn sample(&self, skeleton: &Skeleton, time: f32) -> Vec<Transform> {
        let mut pose = skeleton.rest_pose();
        self.sample_into(time, &mut pose);
        pose
    }

    /// Writes the tracks' values at `time` over `pose` (one transform per joint, usually
    /// starting from [`Skeleton::rest_pose`]). Tracks for joints out of range are skipped.
    pub fn sample_into(&self, time: f32, pose: &mut [Transform]) {
        for track in &self.tracks {
            if let Some(p) = pose.get_mut(track.joint) {
                track.apply(time, p);
            }
        }
    }

    /// The clip for a skeleton whose joints were renumbered: joint `j` becomes
    /// `new_index[j]`, and tracks of joints mapped to `usize::MAX` (removed) are dropped.
    pub fn remap_joints(&self, new_index: &[usize]) -> Animation {
        Animation {
            name: self.name.clone(),
            tracks: self
                .tracks
                .iter()
                .filter_map(|t| {
                    let j = *new_index.get(t.joint)?;
                    (j != usize::MAX).then(|| Track {
                        joint: j,
                        ..t.clone()
                    })
                })
                .collect(),
        }
    }

    /// The number of evenly spaced samples [`Animation::frame_times`] takes at `fps`.
    pub fn frame_count(&self, fps: f32) -> usize {
        let d = self.duration();
        if d <= 0.0 || fps <= 0.0 || !d.is_finite() {
            1
        } else {
            (d * fps).ceil() as usize + 1
        }
    }

    /// Evenly spaced sample times from 0 to [`Animation::duration`], both included, at about
    /// `fps` samples a second ([`Animation::frame_count`] of them).
    pub fn frame_times(&self, fps: f32) -> impl Iterator<Item = f32> {
        let n = self.frame_count(fps);
        let d = self.duration();
        (0..n).map(move |i| {
            if n == 1 {
                0.0
            } else {
                d * i as f32 / (n - 1) as f32
            }
        })
    }
}

/// Poses for the bone planner ([`crate::bones::plan_bone_reduction`]): the rest pose, then
/// every clip of the skeleton sampled at `fps`, each as joint matrices in the skeleton's
/// space ([`Skeleton::world_matrices`]).
pub fn sample_poses(skeleton: &Skeleton, fps: f32) -> Vec<Vec<Mat4>> {
    let rest = skeleton.rest_pose();
    let mut poses = vec![skeleton.world_matrices(&rest)];
    for clip in &skeleton.animations {
        for t in clip.frame_times(fps) {
            let mut pose = rest.clone();
            clip.sample_into(t, &mut pose);
            poses.push(skeleton.world_matrices(&pose));
        }
    }
    poses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;

    fn arm() -> Skeleton {
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
        s
    }

    fn bend() -> Animation {
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        Animation {
            name: "bend".into(),
            tracks: vec![Track {
                rotation: Some(Curve::linear(vec![0.0, 1.0], vec![Quat::IDENTITY, q])),
                ..Track::new(0)
            }],
        }
    }

    #[test]
    fn rest_from_bind_round_trips() {
        let s = arm();
        assert!(s.joints[1].rest.translation.abs_diff_eq(Vec3::Y, 1e-6));
        let worlds = s.world_matrices(&s.rest_pose());
        for (j, w) in worlds.iter().enumerate() {
            assert!(w.abs_diff_eq(s.bind_world(j), 1e-5));
        }
        for m in s.skinning_matrices(&worlds) {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        }
    }

    #[test]
    fn linear_and_step_sampling() {
        let c = Curve::linear(vec![1.0, 3.0], vec![Vec3::ZERO, Vec3::X * 4.0]);
        assert_eq!(c.sample(0.0), Some(Vec3::ZERO));
        assert_eq!(c.sample(2.0), Some(Vec3::X * 2.0));
        assert_eq!(c.sample(9.0), Some(Vec3::X * 4.0));
        let s = Curve {
            interpolation: Interpolation::Step,
            ..c.clone()
        };
        assert_eq!(s.sample(2.9), Some(Vec3::ZERO));
        let empty = Curve::<Vec3>::linear(Vec::new(), Vec::new());
        assert_eq!(empty.sample(1.0), None);
    }

    #[test]
    fn cubic_spline_hits_keys_and_follows_tangents() {
        // p(t) = t² on [0, 2]: tangents are the derivative 2t.
        let c = Curve {
            interpolation: Interpolation::CubicSpline,
            times: vec![0.0, 2.0],
            values: vec![
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::X * 4.0,
                Vec3::X * 4.0,
                Vec3::X * 4.0,
            ],
        };
        assert!(c.sample(1.0).unwrap().abs_diff_eq(Vec3::X, 1e-6));
        assert!(c.sample(0.5).unwrap().abs_diff_eq(Vec3::X * 0.25, 1e-6));
        assert_eq!(c.sample(2.0), Some(Vec3::X * 4.0));
    }

    #[test]
    fn clip_poses_the_skeleton() {
        let mut s = arm();
        s.animations.push(bend());
        let clip = &s.animations[0];
        assert_eq!(clip.duration(), 1.0);
        let worlds = s.world_matrices(&clip.sample(&s, 1.0));
        // The tip, one unit up, swings to -x.
        let tip = worlds[1].transform_point3(Vec3::ZERO);
        assert!(tip.abs_diff_eq(-Vec3::X, 1e-5), "{tip}");
        let poses = sample_poses(&s, 10.0);
        assert_eq!(poses.len(), 1 + 11);
        assert!(poses[0][1].abs_diff_eq(s.bind_world(1), 1e-5));
    }

    #[test]
    fn remap_drops_removed_joints() {
        let mut clip = bend();
        clip.tracks.push(Track::new(1));
        let r = clip.remap_joints(&[usize::MAX, 0]);
        assert_eq!(r.tracks.len(), 1);
        assert_eq!(r.tracks[0].joint, 0);
    }

    #[test]
    fn premultiply_matches_matrix_product() {
        let offset = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(0.7),
            scale: Vec3::splat(2.0),
        };
        let mut track = Track {
            translation: Some(Curve::linear(vec![0.0], vec![Vec3::new(0.5, -1.0, 2.0)])),
            rotation: Some(Curve::linear(vec![0.0], vec![Quat::from_rotation_x(0.3)])),
            scale: Some(Curve::linear(vec![0.0], vec![Vec3::splat(1.5)])),
            ..Track::new(0)
        };
        let mut before = Transform::IDENTITY;
        track.apply(0.0, &mut before);
        track.premultiply(offset);
        let mut after = Transform::IDENTITY;
        track.apply(0.0, &mut after);
        assert!(
            after
                .matrix()
                .abs_diff_eq(offset.matrix() * before.matrix(), 1e-5)
        );
    }
}
