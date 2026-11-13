//! Bone reduction: influence limiting and skinned-error bone removal.
//!
//! * [`limit_influences`] — keep the top-k weights per vertex and renormalize.
//! * [`plan_bone_reduction`] — score every bone by the skinned error of re-linking its
//!   vertices to its nearest kept ancestor (over sampled animation poses when given), then
//!   remove bones leaf-first, cheapest first, until a bone budget or error limit is reached.
//!   A locked bone protects itself and all its ancestors.
//! * [`apply_bone_reduction`] — re-link mesh weights through the old→kept remap. The engine
//!   keeps driving the reduced skeleton with the original animations.
//! * [`compact_skeleton`] — optionally drop removed joints and renumber.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

use glam::{Mat4, Vec3};
use crate::{Error, Mesh, Result, Skeleton};
use serde::{Deserialize, Serialize};

/// Keeps the `max` largest weights per vertex and renormalizes. Returns vertices changed.
pub fn limit_influences(mesh: &mut Mesh, max: usize) -> usize {
    let Some(skin) = &mut mesh.skin else { return 0 };
    let max = max.clamp(1, 4);
    let mut changed = 0;
    for (j, w) in skin.joints.iter_mut().zip(skin.weights.iter_mut()) {
        let mut pairs: Vec<(u16, f32)> = j.iter().copied().zip(w.iter().copied()).collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let nonzero = pairs.iter().filter(|p| p.1 > 0.0).count();
        if nonzero <= max {
            continue;
        }
        changed += 1;
        let sum: f32 = pairs[..max].iter().map(|p| p.1).sum::<f32>().max(1e-12);
        for i in 0..4 {
            if i < max {
                j[i] = pairs[i].0;
                w[i] = pairs[i].1 / sum;
            } else {
                j[i] = 0;
                w[i] = 0.0;
            }
        }
    }
    changed
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoneReductionOptions {
    /// Keep at most this many bones.
    pub max_bones: Option<usize>,
    /// Keep this fraction of bones.
    pub ratio: Option<f32>,
    /// Never remove a bone whose estimated error exceeds this (world units).
    pub max_error: Option<f32>,
    /// Bones that must stay (with their ancestors).
    pub locked: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneReduction {
    /// For every original joint, the kept joint it maps to (itself if kept).
    pub remap: Vec<usize>,
    pub removed: Vec<usize>,
    /// Estimated error of each removal, in removal order.
    pub errors: Vec<f32>,
}

impl BoneReduction {
    pub fn kept_count(&self) -> usize {
        self.remap.len() - self.removed.len()
    }
}

/// Plans which bones to remove.
///
/// `poses`: optional animation samples, each a list of joint *world* matrices (one per joint).
/// With poses, the error of re-linking vertex v from bone b to ancestor a is the distance
/// between `M_b·IB_b·v` and `M_a·IB_a·v`, weighted and maximized over frames. Without poses,
/// a bind-pose proxy is used: weight × distance from the vertex to a's joint, scaled by the
/// length of the b→a chain (how far the pivot moves).
pub fn plan_bone_reduction(
    skeleton: &Skeleton,
    meshes: &[&Mesh],
    poses: Option<&[Vec<Mat4>]>,
    opts: &BoneReductionOptions,
) -> Result<BoneReduction> {
    let n = skeleton.joints.len();
    if n == 0 {
        return Err(Error::InvalidInput("skeleton has no joints".into()));
    }
    if let Some(p) = poses {
        if p.iter().any(|f| f.len() != n) {
            return Err(Error::InvalidInput(
                "every pose must have one matrix per joint".into(),
            ));
        }
    }
    // Per joint: list of (bind-space position, weight).
    let mut infl: Vec<Vec<(Vec3, f32)>> = vec![Vec::new(); n];
    for m in meshes {
        let Some(skin) = &m.skin else { continue };
        for (v, (j, w)) in skin.joints.iter().zip(&skin.weights).enumerate() {
            for k in 0..4 {
                if w[k] > 0.0 && (j[k] as usize) < n {
                    infl[j[k] as usize].push((m.positions[v], w[k]));
                }
            }
        }
    }
    let mut locked = vec![false; n];
    for &l in &opts.locked {
        let mut cur = Some(l);
        while let Some(c) = cur {
            if c >= n || locked[c] {
                break;
            }
            locked[c] = true;
            cur = skeleton.joints[c].parent;
        }
    }
    for (j, joint) in skeleton.joints.iter().enumerate() {
        if joint.parent.is_none() {
            locked[j] = true; // roots stay
        }
    }
    let bind_pos: Vec<Vec3> = (0..n)
        .map(|j| skeleton.bind_world(j).w_axis.truncate())
        .collect();
    let mut remap: Vec<usize> = (0..n).collect();
    let mut alive = vec![true; n];
    let target = match (opts.max_bones, opts.ratio) {
        (Some(m), Some(r)) => m.min((n as f32 * r).ceil() as usize),
        (Some(m), None) => m,
        (None, Some(r)) => (n as f32 * r).ceil() as usize,
        (None, None) => n / 2,
    }
    .max(1);
    let kept_ancestor = |j: usize, alive: &[bool]| {
        let mut p = skeleton.joints[j].parent;
        while let Some(pp) = p {
            if alive[pp] {
                return Some(pp);
            }
            p = skeleton.joints[pp].parent;
        }
        None
    };
    let error_of = |j: usize, a: usize, infl: &[(Vec3, f32)]| -> f32 {
        match poses {
            Some(frames) => {
                let ib_j = skeleton.joints[j].inverse_bind;
                let ib_a = skeleton.joints[a].inverse_bind;
                let mut worst = 0.0f32;
                for f in frames {
                    let (mj, ma) = (f[j] * ib_j, f[a] * ib_a);
                    for &(p, w) in infl {
                        worst = worst
                            .max(w * (mj.transform_point3(p) - ma.transform_point3(p)).length());
                    }
                }
                worst
            }
            None => {
                let chain = (bind_pos[j] - bind_pos[a]).length();
                let reach = infl
                    .iter()
                    .map(|&(p, w)| w * (p - bind_pos[a]).length())
                    .fold(0.0f32, f32::max);
                // Rotating the pivot by ~0.25 rad moves points by ~0.25 × distance.
                0.25 * (chain + reach)
                    * infl
                        .iter()
                        .map(|x| x.1)
                        .sum::<f32>()
                        .min(1.0)
                        .max(if infl.is_empty() { 0.0 } else { 0.1 })
            }
        }
    };
    let mut removed = Vec::new();
    let mut errors = Vec::new();
    let mut count = n;
    while count > target {
        // Removable: alive, unlocked, every child already removed.
        let mut best: Option<(usize, usize, f32)> = None;
        for j in 0..n {
            if !alive[j] || locked[j] || skeleton.children(j).any(|c| alive[c]) {
                continue;
            }
            let Some(a) = kept_ancestor(j, &alive) else {
                continue;
            };
            let e = error_of(j, a, &infl[j]);
            if best.is_none_or(|b| e < b.2) {
                best = Some((j, a, e));
            }
        }
        let Some((j, a, e)) = best else { break };
        if opts.max_error.is_some_and(|m| e > m) {
            break;
        }
        alive[j] = false;
        count -= 1;
        removed.push(j);
        errors.push(e);
        // j's influences (and those already folded into it) now belong to a.
        let moved = std::mem::take(&mut infl[j]);
        infl[a].extend(moved);
        for r in remap.iter_mut() {
            if *r == j {
                *r = a;
            }
        }
    }
    Ok(BoneReduction {
        remap,
        removed,
        errors,
    })
}

/// Re-links weights through `red.remap`, merging duplicate joints per vertex (top 4 kept,
/// renormalized). Joint indices still refer to the original skeleton.
pub fn apply_bone_reduction(mesh: &mut Mesh, red: &BoneReduction) {
    let Some(skin) = &mut mesh.skin else { return };
    for (j, w) in skin.joints.iter_mut().zip(skin.weights.iter_mut()) {
        let mut acc: Vec<(u16, f32)> = Vec::with_capacity(4);
        for k in 0..4 {
            if w[k] <= 0.0 {
                continue;
            }
            let nj = red
                .remap
                .get(j[k] as usize)
                .copied()
                .unwrap_or(j[k] as usize) as u16;
            match acc.iter_mut().find(|p| p.0 == nj) {
                Some(p) => p.1 += w[k],
                None => acc.push((nj, w[k])),
            }
        }
        acc.sort_by(|a, b| b.1.total_cmp(&a.1));
        let sum: f32 = acc.iter().map(|p| p.1).sum::<f32>().max(1e-12);
        for k in 0..4 {
            match acc.get(k) {
                Some(&(jj, ww)) => {
                    j[k] = jj;
                    w[k] = ww / sum;
                }
                None => {
                    j[k] = 0;
                    w[k] = 0.0;
                }
            }
        }
    }
}

/// Drops removed joints. Returns the compact skeleton and the old→new index map
/// (`usize::MAX` for removed joints). Apply [`remap_joint_indices`] to meshes afterwards.
pub fn compact_skeleton(skeleton: &Skeleton, red: &BoneReduction) -> (Skeleton, Vec<usize>) {
    let n = skeleton.joints.len();
    let removed: std::collections::HashSet<usize> = red.removed.iter().copied().collect();
    let mut new_index = vec![usize::MAX; n];
    let mut joints = Vec::new();
    for j in 0..n {
        if !removed.contains(&j) {
            new_index[j] = joints.len();
            joints.push(skeleton.joints[j].clone());
        }
    }
    for joint in &mut joints {
        let mut p = joint.parent;
        while let Some(pp) = p {
            if !removed.contains(&pp) {
                break;
            }
            p = skeleton.joints[pp].parent;
        }
        joint.parent = p.map(|pp| new_index[pp]);
    }
    (
        Skeleton {
            name: skeleton.name.clone(),
            joints,
        },
        new_index,
    )
}

/// Rewrites joint indices with a map from [`compact_skeleton`] (after [`apply_bone_reduction`]).
pub fn remap_joint_indices(mesh: &mut Mesh, new_index: &[usize]) {
    if let Some(skin) = &mut mesh.skin {
        for (j, w) in skin.joints.iter_mut().zip(&skin.weights) {
            for k in 0..4 {
                if w[k] > 0.0 {
                    j[k] = new_index[j[k] as usize] as u16;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Joint, Skin};

    /// A 5-joint chain along +Y with a strip of vertices, each weighted to its nearest joint.
    fn chain() -> (Skeleton, Mesh) {
        let joints = (0..5)
            .map(|i| Joint {
                name: format!("j{i}"),
                parent: if i == 0 { None } else { Some(i - 1) },
                inverse_bind: Mat4::from_translation(Vec3::new(0.0, -(i as f32), 0.0)),
            })
            .collect();
        let mut pos = Vec::new();
        let mut j = Vec::new();
        let mut w = Vec::new();
        for i in 0..20 {
            let y = i as f32 * 0.25;
            pos.push(Vec3::new(0.1, y, 0.0));
            let a = (y.floor() as u16).min(4);
            let b = (a + 1).min(4);
            let t = y - y.floor();
            j.push([a, b, 0, 0]);
            w.push(if a == b {
                [1.0, 0.0, 0.0, 0.0]
            } else {
                [1.0 - t, t, 0.0, 0.0]
            });
        }
        let mut m = Mesh::new(pos, Vec::new());
        m.skin = Some(Skin {
            joints: j,
            weights: w,
        });
        (
            Skeleton {
                name: "chain".into(),
                joints,
            },
            m,
        )
    }

    #[test]
    fn leaf_first_removal_and_relink() {
        let (sk, mut m) = chain();
        let red = plan_bone_reduction(
            &sk,
            &[&m],
            None,
            &BoneReductionOptions {
                max_bones: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(red.kept_count(), 3);
        assert_eq!(red.removed, vec![4, 3], "leaf-first along a chain");
        apply_bone_reduction(&mut m, &red);
        let s = m.skin.as_ref().unwrap();
        for (j, w) in s.joints.iter().zip(&s.weights) {
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            for k in 0..4 {
                if w[k] > 0.0 {
                    assert!(j[k] <= 2);
                }
            }
        }
        let (compact, map) = compact_skeleton(&sk, &red);
        assert_eq!(compact.joints.len(), 3);
        remap_joint_indices(&mut m, &map);
        assert!(m.skin.unwrap().joints.iter().flatten().all(|&j| j < 3));
    }

    #[test]
    fn locked_bone_protects_chain() {
        let (sk, m) = chain();
        let red = plan_bone_reduction(
            &sk,
            &[&m],
            None,
            &BoneReductionOptions {
                max_bones: Some(1),
                locked: vec![4],
                ..Default::default()
            },
        )
        .unwrap();
        assert!(red.removed.is_empty());
    }

    #[test]
    fn pose_error_ranks_static_bones_first() {
        let (sk, m) = chain();
        // Animation: only joint 2 rotates; joints 3 and 4 follow rigidly (world matrices).
        let rot = Mat4::from_rotation_z(0.8);
        let frames: Vec<Vec<Mat4>> = vec![(0..5)
            .map(|i| {
                let bind = sk.bind_world(i);
                if i >= 2 {
                    Mat4::from_translation(Vec3::Y * 2.0)
                        * rot
                        * Mat4::from_translation(-Vec3::Y * 2.0)
                        * bind
                } else {
                    bind
                }
            })
            .collect()];
        let red = plan_bone_reduction(
            &sk,
            &[&m],
            Some(&frames),
            &BoneReductionOptions {
                max_bones: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        // Removing 4 and 3 (rigid with 2) costs nothing; removing 2 would cost a lot.
        assert_eq!(red.removed, vec![4, 3]);
        assert!(red.errors.iter().all(|&e| e < 1e-4), "{:?}", red.errors);
    }

    #[test]
    fn influences_are_limited() {
        let mut m = Mesh::new(vec![Vec3::ZERO], Vec::new());
        m.skin = Some(Skin {
            joints: vec![[0, 1, 2, 3]],
            weights: vec![[0.4, 0.3, 0.2, 0.1]],
        });
        assert_eq!(limit_influences(&mut m, 2), 1);
        let s = m.skin.unwrap();
        assert_eq!(s.joints[0][..2], [0, 1]);
        assert!((s.weights[0][0] - 0.4 / 0.7).abs() < 1e-5 && s.weights[0][2] == 0.0);
    }
}
