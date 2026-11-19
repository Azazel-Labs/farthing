//! The Khronos Fox (CC BY 4.0; see web/samples/index.json for its credits): a skinned,
//! animated sample. Its clips import, bake to a pose palette and a vertex animation texture
//! within small errors, survive a GLB round trip, and drive bone reduction with real poses.
//! Skipped (with a note) until `python fetch_samples.py` has downloaded it.

use farthing::anim::bake::{BakeOptions, PosePalette, bake, measure};
use farthing::anim::sample_poses;
use farthing::anim::vat::{VertexAnimation, bake_vat, measure_vat};
use farthing::bones::{BoneReductionOptions, apply_bone_reduction, plan_bone_reduction};
use farthing::io::{ExportTextures, glb_bytes, load_gltf_slice};
use farthing::{Mesh, Scene};

fn fox() -> Option<Scene> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web/samples/Fox.glb");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("no web/samples/Fox.glb; run fetch_samples.py to test the Fox");
        return None;
    };
    Some(load_gltf_slice(&bytes).unwrap().0)
}

fn skinned(scene: &Scene) -> &Mesh {
    scene
        .meshes
        .iter()
        .find(|m| m.skin.is_some())
        .expect("a skinned mesh")
}

#[test]
fn fox_clips_import_and_bake() {
    let Some(scene) = fox() else { return };
    assert_eq!(scene.skeletons.len(), 1);
    let sk = &scene.skeletons[0];
    assert_eq!(sk.joints.len(), 24);
    let names: Vec<&str> = sk.animations.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["Survey", "Walk", "Run"]);
    for a in &sk.animations {
        assert!(a.duration() > 0.5, "{}: {}", a.name, a.duration());
        assert!(!a.tracks.is_empty());
    }
    // The rest pose reproduces the bind pose the inverse bind matrices were made from.
    let rest = sk.skinning_matrices(&sk.world_matrices(&sk.rest_pose()));
    for m in &rest {
        assert!(m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-3), "{m}");
    }

    let mesh = skinned(&scene);
    let palette = bake(sk, &BakeOptions::default());
    assert_eq!(palette.joint_count, 24);
    assert_eq!(palette.clips.len(), 3);
    // Walk and Run cycle; Survey doesn't end where it starts.
    let looping: Vec<bool> = palette.clips.iter().map(|c| c.looping).collect();
    assert_eq!(&looping[1..], [true, true], "{looping:?}");

    let err = measure(&palette, sk, mesh).unwrap();
    eprintln!(
        "Fox palette: {} frames, {} bytes, {err:?}",
        palette.frame_count(),
        palette.to_bytes().len()
    );
    // Interpolation is the worst vertex halfway between two frames: at 30 fps a paw mid-stride
    // in Walk strays about 1.3% of the Fox's size (blending skinning transforms cuts the
    // corner of its arc).
    for c in &err.clips {
        assert!(
            c.quantization < 0.002 * err.size,
            "{}: {c:?} of {}",
            c.name,
            err.size
        );
        assert!(
            c.interpolation < 0.02 * err.size,
            "{}: {c:?} of {}",
            c.name,
            err.size
        );
    }
    let back = PosePalette::from_bytes(&palette.to_bytes()).unwrap();
    assert_eq!(back, palette);

    let vat = bake_vat(sk, mesh, &BakeOptions::default()).unwrap();
    assert_eq!(vat.vertex_count as usize, mesh.positions.len());
    assert_eq!(vat.frame_count(), palette.frame_count());
    let verr = measure_vat(&vat, sk, mesh).unwrap();
    eprintln!("Fox VAT: {} bytes, {verr:?}", vat.to_bytes().len());
    for c in &verr.clips {
        assert!(c.quantization < 0.002 * verr.size, "{}: {c:?}", c.name);
        assert!(c.interpolation < 0.02 * verr.size, "{}: {c:?}", c.name);
    }
    assert_eq!(VertexAnimation::from_bytes(&vat.to_bytes()).unwrap(), vat);
}

#[test]
fn fox_clips_survive_a_glb_round_trip() {
    let Some(scene) = fox() else { return };
    let glb = glb_bytes(&scene, &ExportTextures::default()).unwrap();
    let back = load_gltf_slice(&glb).unwrap().0;
    let (a, b) = (&scene.skeletons[0], &back.skeletons[0]);
    assert_eq!(a.joints.len(), b.joints.len());
    assert_eq!(a.animations.len(), b.animations.len());
    for (ca, cb) in a.animations.iter().zip(&b.animations) {
        assert_eq!(ca.name, cb.name);
        assert!((ca.duration() - cb.duration()).abs() < 1e-5);
        for k in 0..=10 {
            let t = ca.duration() * k as f32 / 10.0;
            let wa = a.world_matrices(&ca.sample(a, t));
            let wb = b.world_matrices(&cb.sample(b, t));
            for (ma, mb) in wa.iter().zip(&wb) {
                assert!(ma.abs_diff_eq(*mb, 1e-3), "{} at {t}", ca.name);
            }
        }
    }
}

#[test]
fn fox_bone_reduction_uses_its_clips() {
    let Some(scene) = fox() else { return };
    let sk = &scene.skeletons[0];
    let mesh = skinned(&scene);
    let poses = sample_poses(sk, 15.0);
    assert!(poses.len() > 30);
    let opts = BoneReductionOptions {
        max_bones: Some(12),
        ..Default::default()
    };
    let posed = plan_bone_reduction(sk, &[mesh], Some(&poses), &opts).unwrap();
    let proxy = plan_bone_reduction(sk, &[mesh], None, &opts).unwrap();
    assert_eq!(posed.kept_count(), 12);
    assert_eq!(proxy.kept_count(), 12);

    // A reduced LOD keeps the original joint indices, so the one palette drives it; its
    // error against the full skeleton's clips is what the planner traded away.
    let palette = bake(sk, &BakeOptions::default());
    let full = measure(&palette, sk, mesh).unwrap();
    let worst = |red| {
        let mut lod = mesh.clone();
        apply_bone_reduction(&mut lod, red);
        let lod_err = measure(&palette, sk, &lod).unwrap();
        // Compare the LOD's skinned positions with the full mesh's: measure() compares each
        // mesh with its own exact skinning, so the reduction's own cost shows in neither.
        // Instead skin both with the exact poses and take the largest gap.
        let mut gap = 0.0f32;
        for frame in poses.iter().step_by(3) {
            let mats = sk.skinning_matrices(frame);
            let (a, b) = (mesh.skin.as_ref().unwrap(), lod.skin.as_ref().unwrap());
            for v in 0..mesh.positions.len() {
                let p = mesh.positions[v];
                let pa = skin(&mats, a.joints[v], a.weights[v], p);
                let pb = skin(&mats, b.joints[v], b.weights[v], p);
                gap = gap.max(pa.distance(pb));
            }
        }
        // The palette's own quantization costs the LOD no more than the full mesh. (Its
        // interpolation error is larger: a vertex hanging from a farther bone swings a wider arc.)
        assert!(
            lod_err
                .clips
                .iter()
                .all(|c| c.quantization < 0.002 * full.size),
            "{lod_err:?}"
        );
        gap
    };
    let (g_posed, g_proxy) = (worst(&posed), worst(&proxy));
    eprintln!(
        "Fox 24 → 12 bones: posed plan moves vertices {g_posed:.3}, bind-pose proxy {g_proxy:.3} (size {:.1})",
        full.size
    );
    assert!(g_posed <= g_proxy * 1.0001, "{g_posed} vs {g_proxy}");
}

fn skin(mats: &[glam::Mat4], j: [u16; 4], w: [f32; 4], p: glam::Vec3) -> glam::Vec3 {
    let mut out = glam::Vec3::ZERO;
    for k in 0..4 {
        if w[k] > 0.0 {
            out += mats[j[k] as usize].transform_point3(p) * w[k];
        }
    }
    out
}
