//! The GPU crowd renderer against farthing's CPU palette decode: a unit it draws, skinned or
//! from a vertex animation texture, mid-clip or mid-fade, matches the same mesh skinned on the
//! CPU and drawn as an ordinary mesh. Thousands of units cull and pick levels of detail on the
//! GPU, and only what changed is written.

use farthing::anim::bake::{BakeOptions, BonePose, PosePalette, bake};
use farthing::anim::vat::bake_vat;
use farthing::anim::{Animation, Curve, Track};
use farthing::bones::{BoneReductionOptions, apply_bone_reduction, plan_bone_reduction};
use farthing::glam::camera::rh::view::look_at_mat4;
use farthing::glam::{Mat3, Mat4, Quat, Vec3};
use farthing::{Joint, Mesh, Skeleton, Skin, Transform};
use farthing_poc_engine::viewer::{Camera, Draw, Material, View, Viewer};
use farthing_poc_engine::{CrowdLod, CrowdLodKind, CrowdUnit};

const SIZE: u32 = 256;

fn viewer() -> Option<Viewer> {
    match Viewer::headless() {
        Ok(v) if v.clusters_supported() => Some(v),
        Ok(_) => {
            eprintln!("skipped: the adapter can't run crowds");
            None
        }
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

/// A three-joint column two units tall (joints at heights 0, 0.7 and 1.4), with a clip that
/// sways it and one that bends its top back and forth.
fn column() -> (Skeleton, Mesh) {
    let joint = |name: &str, parent, y: f32| Joint {
        name: name.into(),
        parent,
        inverse_bind: Mat4::from_translation(Vec3::new(0.0, -y, 0.0)),
        rest: Transform::IDENTITY,
    };
    let mut s = Skeleton {
        name: "column".into(),
        joints: vec![
            joint("base", None, 0.0),
            joint("mid", Some(0), 0.7),
            joint("top", Some(1), 1.4),
        ],
        animations: Vec::new(),
    };
    s.rest_from_bind();
    let turn = |a: f32| Quat::from_rotation_z(a);
    s.animations.push(Animation {
        name: "sway".into(),
        tracks: vec![Track {
            rotation: Some(Curve::linear(
                vec![0.0, 0.5, 1.0],
                vec![turn(-0.3), turn(0.3), turn(-0.3)],
            )),
            ..Track::new(0)
        }],
    });
    s.animations.push(Animation {
        name: "bend".into(),
        tracks: vec![
            Track {
                rotation: Some(Curve::linear(
                    vec![0.0, 0.6, 1.2],
                    vec![turn(0.0), turn(0.9), turn(0.0)],
                )),
                ..Track::new(1)
            },
            Track {
                rotation: Some(Curve::linear(
                    vec![0.0, 0.6, 1.2],
                    vec![turn(0.0), turn(0.7), turn(0.0)],
                )),
                ..Track::new(2)
            },
        ],
    });

    // Rings of 12 around y, 0.25 apart, weighted between the two joints nearest each.
    let (rings, segs) = (9, 12);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut joints = Vec::new();
    let mut weights = Vec::new();
    for r in 0..rings {
        let y = r as f32 * 0.25;
        let x = (y / 0.7).min(1.999);
        let (j, t) = (x.floor() as u16, x.fract());
        for k in 0..segs {
            let a = k as f32 / segs as f32 * std::f32::consts::TAU;
            let n = Vec3::new(a.cos(), 0.0, a.sin());
            positions.push(n * 0.2 + Vec3::Y * y);
            normals.push(n);
            joints.push([j, j + 1, 0, 0]);
            weights.push([1.0 - t, t, 0.0, 0.0]);
        }
    }
    let mut indices = Vec::new();
    for r in 0..rings - 1 {
        for k in 0..segs {
            let (a, b) = (r * segs + k, r * segs + (k + 1) % segs);
            let (c, d) = (a + segs, b + segs);
            indices.extend([a, c, b, b, c, d]);
        }
    }
    let mut m = Mesh::new(positions, indices);
    m.normals = Some(normals);
    m.skin = Some(Skin { joints, weights });
    (s, m)
}

fn flat3(v: &[Vec3]) -> Vec<f32> {
    v.iter().flat_map(|p| p.to_array()).collect()
}

/// A skinned level of `mesh`, owning its arrays.
struct Arrays {
    positions: Vec<f32>,
    normals: Vec<f32>,
    joints: Vec<u16>,
    weights: Vec<f32>,
}

fn arrays(mesh: &Mesh) -> Arrays {
    let skin = mesh.skin.as_ref().unwrap();
    Arrays {
        positions: flat3(&mesh.positions),
        normals: flat3(mesh.normals.as_ref().unwrap()),
        joints: skin.joints.iter().flatten().copied().collect(),
        weights: skin.weights.iter().flatten().copied().collect(),
    }
}

fn skinned<'a>(a: &'a Arrays, indices: &'a [u32], min_pixels: f32) -> CrowdLod<'a> {
    CrowdLod {
        kind: CrowdLodKind::Skinned {
            positions: &a.positions,
            normals: &a.normals,
            joints: &a.joints,
            weights: &a.weights,
        },
        uvs: &[],
        indices,
        min_pixels,
    }
}

fn camera(eye: Vec3, target: Vec3) -> Camera {
    Camera {
        view: look_at_mat4(eye, target, Vec3::Y),
        fov_y: 0.9,
        near: 0.05,
        far: 400.0,
    }
}

/// `unit`'s pose at `time` on the CPU: the palette decode (the fade blended per joint, as the
/// shader does), applied to `mesh` and placed.
fn cpu_pose(
    palette: &PosePalette,
    mesh: &Mesh,
    unit: &CrowdUnit,
    time: f32,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let a = palette.sample(unit.clip as usize, (time - unit.start) * unit.speed);
    let b = palette.sample(
        unit.previous_clip as usize,
        (time - unit.previous_start) * unit.speed,
    );
    let fade = if unit.fade > 0.0 {
        ((time - unit.fade_start) / unit.fade).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let bones: Vec<Mat4> = a
        .iter()
        .zip(&b)
        .map(|(&a, &b)| {
            if fade < 1.0 {
                BonePose::blend(b, a, fade)
            } else {
                a
            }
            .matrix()
        })
        .collect();
    let skin = mesh.skin.as_ref().unwrap();
    let turn = Mat3::from_rotation_y(unit.yaw);
    let place = |p: Vec3| Vec3::from_array(unit.position) + turn * (p * unit.scale);
    let mut ps = Vec::new();
    let mut ns = Vec::new();
    for (v, &p) in mesh.positions.iter().enumerate() {
        let n = mesh.normals.as_ref().unwrap()[v];
        let (mut sp, mut sn) = (Vec3::ZERO, Vec3::ZERO);
        for k in 0..4 {
            let w = skin.weights[v][k];
            if w > 0.0 {
                let m = bones[skin.joints[v][k] as usize];
                sp += w * m.transform_point3(p);
                sn += w * m.transform_vector3(n);
            }
        }
        ps.push(place(sp));
        ns.push((turn * sn).normalize());
    }
    (ps, ns)
}

/// Pixels whose color differs by more than `tolerance` in any channel, and pixels drawn in either.
fn compare(a: &[u8], b: &[u8], tolerance: u8) -> (usize, usize) {
    let (a, b) = (a.as_chunks::<4>().0, b.as_chunks::<4>().0);
    let drawn = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x[3] > 0 || y[3] > 0)
        .count();
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .any(|(p, q)| p.abs_diff(*q) > tolerance)
        })
        .count();
    (differ, drawn)
}

fn draw(mesh: u32) -> [Draw; 1] {
    [Draw {
        mesh,
        model: Mat4::IDENTITY,
        overlays: false,
    }]
}

/// Renders `unit` from the crowd at `time`, then the CPU's pose of it as a plain mesh; how many
/// pixels differ, of how many drawn.
fn against_cpu(vat: bool, unit: CrowdUnit, time: f32) -> Option<(usize, usize)> {
    let mut viewer = viewer()?;
    let (sk, mesh) = column();
    let palette = bake(&sk, &BakeOptions::default());
    let material = viewer.add_material(&Material::default());
    let a = arrays(&mesh);
    let anim = bake_vat(&sk, &mesh, &BakeOptions::default()).unwrap();
    let lod = if vat {
        CrowdLod {
            kind: CrowdLodKind::Vat(&anim),
            uvs: &[],
            indices: &mesh.indices,
            min_pixels: 0.0,
        }
    } else {
        skinned(&a, &mesh.indices, 0.0)
    };
    let crowd = viewer.add_crowd(&palette, &[lod], material).unwrap();
    viewer.set_crowd_units(crowd, 0, &[unit]);
    viewer.set_crowd_time(time);
    let cam = camera(Vec3::new(0.5, 1.2, 4.0), Vec3::new(0.0, 1.0, 0.0));
    fn view<'a>(camera: Camera, draws: &'a [Draw]) -> [View<'a>; 1] {
        [View {
            rect: [0, 0, SIZE, SIZE],
            camera,
            draws,
        }]
    }
    let gpu = viewer.render_image(SIZE, SIZE, &view(cam, &draw(crowd)));

    let (ps, ns) = if vat {
        // A vertex animation level fades by blending positions, not joints.
        let fade = if unit.fade > 0.0 {
            ((time - unit.fade_start) / unit.fade).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let only = |clip, start| CrowdUnit {
            clip,
            start,
            fade: 0.0,
            ..unit
        };
        let (pa, na) = cpu_pose(&palette, &mesh, &only(unit.clip, unit.start), time);
        let (pb, nb) = cpu_pose(
            &palette,
            &mesh,
            &only(unit.previous_clip, unit.previous_start),
            time,
        );
        (
            pb.iter().zip(&pa).map(|(b, a)| b.lerp(*a, fade)).collect(),
            nb.iter()
                .zip(&na)
                .map(|(b, a)| b.lerp(*a, fade).normalize())
                .collect(),
        )
    } else {
        cpu_pose(&palette, &mesh, &unit, time)
    };
    let geometry = viewer.add_geometry(&flat3(&ps), &flat3(&ns), &[], &[]);
    let plain = viewer.add_mesh(geometry, &mesh.indices, material).unwrap();
    let cpu = viewer.render_image(SIZE, SIZE, &view(cam, &draw(plain)));
    let (differ, drawn) = compare(&gpu, &cpu, 12);
    eprintln!("vat {vat}: {differ} of {drawn} pixels differ");
    assert!(drawn > 2000, "the unit is in view: {drawn} pixels");
    Some((differ, drawn))
}

fn unit() -> CrowdUnit {
    CrowdUnit {
        position: [0.1, 0.0, -0.2],
        yaw: 0.6,
        clip: 1,
        previous_clip: 0,
        fade: 0.0,
        scale: 1.1,
        start: 0.1,
        previous_start: 0.0,
        fade_start: 0.0,
        speed: 1.0,
    }
}

#[test]
fn a_skinned_unit_matches_cpu_skinning() {
    // Mid-clip, between baked frames.
    let Some((differ, drawn)) = against_cpu(false, unit(), 0.47) else {
        return;
    };
    assert!(differ * 100 < drawn, "{differ} of {drawn}");
}

#[test]
fn a_fading_unit_matches_cpu_skinning() {
    // Halfway through a fade from sway into bend, played at 1.5×.
    let u = CrowdUnit {
        fade: 0.4,
        fade_start: 0.3,
        speed: 1.5,
        ..unit()
    };
    let Some((differ, drawn)) = against_cpu(false, u, 0.5) else {
        return;
    };
    assert!(differ * 100 < drawn, "{differ} of {drawn}");
}

#[test]
fn a_vertex_animation_unit_matches_cpu_skinning() {
    // The texture holds positions skinned with exact poses rather than the palette's, so
    // allow an edge pixel or two to move.
    let Some((differ, drawn)) = against_cpu(true, unit(), 0.47) else {
        return;
    };
    assert!(differ * 100 < drawn, "{differ} of {drawn}");
    let u = CrowdUnit {
        fade: 0.4,
        fade_start: 0.3,
        ..unit()
    };
    let Some((differ, drawn)) = against_cpu(true, u, 0.5) else {
        return;
    };
    assert!(differ * 100 < drawn, "{differ} of {drawn}");
}

#[test]
fn thousands_of_units_cull_and_pick_levels() {
    let Some(mut viewer) = viewer() else { return };
    let (sk, mesh) = column();
    let palette = bake(&sk, &BakeOptions::default());
    let material = viewer.add_material(&Material::default());

    // Three levels: the full column, a one-bone LOD keeping the palette's joint indices, and a
    // vertex animation texture for the farthest.
    let full = arrays(&mesh);
    let opts = BoneReductionOptions {
        max_bones: Some(1),
        ..Default::default()
    };
    let red = plan_bone_reduction(&sk, &[&mesh], None, &opts).unwrap();
    let mut one_bone = mesh.clone();
    apply_bone_reduction(&mut one_bone, &red);
    let reduced = arrays(&one_bone);
    let anim = bake_vat(&sk, &mesh, &BakeOptions::default()).unwrap();
    let lods = [
        skinned(&full, &mesh.indices, 40.0),
        skinned(&reduced, &mesh.indices, 12.0),
        CrowdLod {
            kind: CrowdLodKind::Vat(&anim),
            uvs: &[],
            indices: &mesh.indices,
            min_pixels: 0.0,
        },
    ];
    let crowd = viewer.add_crowd(&palette, &lods, material).unwrap();

    // 100 × 100 units, 1.5 apart, each its own clip, phase and turn.
    let side = 100;
    let units: Vec<CrowdUnit> = (0..side * side)
        .map(|i| {
            let (x, z) = ((i % side) as f32, (i / side) as f32);
            CrowdUnit {
                position: [x * 1.5, 0.0, -z * 1.5],
                yaw: i as f32 * 0.37,
                clip: (i % 2) as u32,
                previous_clip: 0,
                scale: 1.0,
                start: -(i as f32 * 0.013),
                speed: 1.0,
                ..Default::default()
            }
        })
        .collect();
    viewer.set_crowd_units(crowd, 0, &units);
    assert_eq!(viewer.crowd_unit_count(crowd), side * side);

    // From one corner, looking across the field: the near units get the full column, the far
    // ones the texture, and those behind the camera nothing.
    let cam = camera(Vec3::new(30.0, 6.0, 8.0), Vec3::new(75.0, 0.0, -75.0));
    let draws = draw(crowd);
    let views = [View {
        rect: [0, 0, SIZE, SIZE],
        camera: cam,
        draws: &draws,
    }];
    let frame = |viewer: &mut Viewer, t: f32| {
        viewer.set_crowd_time(t);
        let image = viewer.render_image(SIZE, SIZE, &views);
        viewer.settle();
        image
    };
    let image = frame(&mut viewer, 0.25);
    let covered = image.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
    let culled = viewer.crowd_stats(crowd).unwrap();
    eprintln!("culled: {culled:?}, {covered} pixels");
    let per_lod = &culled.per_lod[0];
    let shown: u32 = per_lod.iter().sum();
    assert!(
        per_lod.iter().all(|&n| n > 0),
        "every level draws some: {per_lod:?}"
    );
    assert!(shown < (side * side) as u32 && shown > 1000, "{shown}");
    assert!(
        per_lod[0] < per_lod[2],
        "far units outnumber near ones: {per_lod:?}"
    );
    assert!(culled.triangles > 0 && covered > 1000);

    // Without frustum culling every unit is drawn, at the same levels the visible ones had.
    viewer.set_crowd_culling(false);
    frame(&mut viewer, 0.25);
    let all = viewer.crowd_stats(crowd).unwrap();
    assert_eq!(
        all.per_lod[0].iter().sum::<u32>(),
        (side * side) as u32,
        "{all:?}"
    );
    viewer.set_crowd_culling(true);

    // A unit changing clips is one record written; the counts stay put as time moves on (the
    // shader animates, the CPU doesn't).
    viewer.set_crowd_units(
        crowd,
        5,
        &[CrowdUnit {
            clip: 1,
            previous_clip: units[5].clip,
            fade: 0.3,
            fade_start: 0.5,
            start: 0.5,
            ..units[5]
        }],
    );
    frame(&mut viewer, 0.6);
    let later = viewer.crowd_stats(crowd).unwrap();
    assert_eq!(later.per_lod[0].iter().sum::<u32>(), shown, "{later:?}");

    // Two views each cull for themselves.
    let behind = camera(Vec3::new(75.0, 6.0, -170.0), Vec3::new(75.0, 0.0, -75.0));
    let two = [
        views[0],
        View {
            rect: [0, 0, SIZE / 2, SIZE / 2],
            camera: behind,
            draws: &draws,
        },
    ];
    viewer.render_image(SIZE, SIZE, &two);
    viewer.settle();
    let both = viewer.crowd_stats(crowd).unwrap();
    assert_eq!(both.per_lod.len(), 2, "{both:?}");
    assert_eq!(
        both.per_lod[0], later.per_lod[0],
        "the first view's counts don't change"
    );
    assert!(both.per_lod[1].iter().sum::<u32>() > 0, "{both:?}");

    // Fewer units: the rest stop drawing.
    viewer.set_crowd_unit_count(crowd, 10);
    viewer.set_crowd_culling(false);
    frame(&mut viewer, 0.7);
    assert_eq!(
        viewer.crowd_stats(crowd).unwrap().per_lod[0]
            .iter()
            .sum::<u32>(),
        10
    );
}

#[test]
fn levels_must_match_the_palette() {
    let Some(mut viewer) = viewer() else { return };
    let (sk, mesh) = column();
    let palette = bake(&sk, &BakeOptions::default());
    let material = viewer.add_material(&Material::default());
    let mut a = arrays(&mesh);
    a.joints[0] = 7;
    let e = viewer.add_crowd(&palette, &[skinned(&a, &mesh.indices, 0.0)], material);
    assert!(e.is_err(), "a joint the palette doesn't have");
    let anim = bake_vat(
        &sk,
        &mesh,
        &BakeOptions {
            fps: 15.0,
            looping: None,
        },
    )
    .unwrap();
    let vat = CrowdLod {
        kind: CrowdLodKind::Vat(&anim),
        uvs: &[],
        indices: &mesh.indices,
        min_pixels: 0.0,
    };
    assert!(
        viewer.add_crowd(&palette, &[vat], material).is_err(),
        "baked at another rate"
    );
}
