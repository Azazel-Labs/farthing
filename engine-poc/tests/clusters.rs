//! The GPU cluster renderer against farthing's CPU cut: with what it needs resident it draws
//! exactly `GpuClusterDag::select`, for every view at once; with a small page pool it draws
//! the cut the resident pages allow, without holes.

use farthing::cluster::{
    ClusterOptions, GpuClusterDag, GpuOptions, LodView, build_cluster_dag,
};
use farthing::glam::camera::rh::view::look_at_mat4;
use farthing::glam::{Mat4, Vec3};
use farthing_poc_engine::scene_gen::bumpy_sphere;
use farthing_poc_engine::viewer::{Camera, Draw, View, Viewer};
use std::sync::OnceLock;

const NONE: u32 = u32::MAX;
const SIZE: u32 = 384;
const FOV: f32 = 1.0;

fn dag() -> &'static GpuClusterDag {
    static DAG: OnceLock<GpuClusterDag> = OnceLock::new();
    DAG.get_or_init(|| {
        let mesh = bumpy_sphere(384, 192, 1.0);
        let dag = build_cluster_dag(&mesh, &ClusterOptions::default()).expect("cluster DAG");
        let opts = GpuOptions {
            page_size: 32 * 1024,
            ..GpuOptions::default()
        };
        let gpu = GpuClusterDag::from_dag(&dag, &mesh, &opts).expect("packed DAG");
        // Through the file format, as the demo loads it.
        GpuClusterDag::from_bytes(&gpu.to_bytes()).expect("decoded DAG")
    })
}

fn viewer() -> Option<Viewer> {
    match Viewer::headless() {
        Ok(v) if v.clusters_supported() => Some(v),
        Ok(_) => {
            eprintln!("skipped: the adapter can't run the cluster renderer");
            None
        }
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

fn camera(eye: Vec3) -> Camera {
    Camera {
        view: look_at_mat4(eye, Vec3::ZERO, Vec3::Y),
        fov_y: FOV,
        near: 0.01,
        far: 100.0,
    }
}

fn lod(eye: Vec3, height: u32) -> LodView {
    LodView {
        eye,
        pixels_per_radian: height as f32 / 2.0 / (FOV / 2.0).tan(),
        threshold: 1.0,
    }
}

fn draws(mesh: u32) -> [Draw; 1] {
    [Draw {
        mesh,
        model: Mat4::IDENTITY,
        overlays: false,
    }]
}

/// Renders until nothing more is requested (or `frames` pass); the last frame's coverage.
fn settle(viewer: &mut Viewer, views: &[View], frames: usize) -> usize {
    let mut quiet = 0;
    let mut last = 0;
    for _ in 0..frames {
        let image = viewer.render_image(SIZE, SIZE, views);
        viewer.settle();
        last = image.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
        let s = viewer.cluster_stats().expect("stats");
        quiet = if s.requested_pages == 0 { quiet + 1 } else { 0 };
        if quiet >= 3 {
            break;
        }
    }
    last
}

/// The clusters instance 0 drew in each view.
fn drawn(viewer: &Viewer) -> Vec<Vec<u32>> {
    viewer
        .cluster_visible()
        .into_iter()
        .map(|v| v.into_iter().filter(|&(i, _)| i == 0).map(|(_, c)| c).collect())
        .collect()
}

fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}

/// The cut over whatever is resident, as the renderer defines it (see `clusters.rs`).
fn resident_cut(dag: &GpuClusterDag, resident: &[bool], view: &LodView) -> Vec<u32> {
    let groups = dag.groups.len();
    let mut child_pages = vec![Vec::new(); groups];
    let mut consumers = vec![Vec::new(); groups];
    for c in &dag.clusters {
        if c.parent_group != NONE {
            child_pages[c.parent_group as usize].push(c.page);
            if c.group != NONE {
                consumers[c.group as usize].push(c.parent_group);
            }
        }
    }
    let mut order: Vec<usize> = (0..groups).collect();
    order.sort_by_key(|&g| std::cmp::Reverse(dag.groups[g].level));
    let mut ready = vec![false; groups];
    for g in order {
        ready[g] = child_pages[g].iter().all(|&p| resident[p as usize])
            && resident[dag.groups[g].page as usize]
            && consumers[g].iter().all(|&q| ready[q as usize]);
    }
    let mut out = Vec::new();
    for (i, c) in dag.clusters.iter().enumerate() {
        let drawable = if c.parent_group == NONE {
            resident[c.page as usize]
        } else {
            ready[c.parent_group as usize]
        };
        if !drawable || view.projected(c.parent_error, &c.parent_bounds) <= view.threshold {
            continue;
        }
        let refine = view.projected(c.lod_error, &c.lod_bounds) > view.threshold && c.group != NONE;
        if refine && ready[c.group as usize] {
            continue;
        }
        out.push(i as u32);
    }
    out
}

#[test]
fn streams_to_the_cpu_cut() {
    let Some(mut viewer) = viewer() else {
        return;
    };
    let dag = dag();
    let mesh = viewer.add_cluster_mesh(dag.clone()).expect("cluster mesh");
    viewer.set_cluster_culling(false, false);
    viewer.set_cluster_streaming(1 << 30, 16);
    let eye = Vec3::new(0.3, 0.4, 1.9);
    let d = draws(mesh);
    let views = [View {
        rect: [0, 0, SIZE, SIZE],
        camera: camera(eye),
        draws: &d,
    }];
    settle(&mut viewer, &views, 400);
    viewer.render_image(SIZE, SIZE, &views);
    let want = sorted(dag.select(&lod(eye, SIZE)));
    assert_eq!(drawn(&viewer)[0], want);
    let resident = viewer.cluster_resident(mesh);
    let count = resident.iter().filter(|&&r| r).count();
    eprintln!("{count} of {} pages resident, {} clusters drawn", resident.len(), want.len());
}

#[test]
fn small_pool_draws_the_resident_cut() {
    let Some(mut viewer) = viewer() else {
        return;
    };
    let dag = dag();
    let mesh = viewer.add_cluster_mesh(dag.clone()).expect("cluster mesh");
    viewer.set_cluster_culling(false, false);
    // Close to opposite sides, so moving between them needs other fine pages.
    let eyes = [Vec3::new(0.2, -0.3, 1.6), Vec3::new(-0.1, 0.3, -1.6)];
    let d = draws(mesh);
    let views = eyes.map(|eye| {
        [View {
            rect: [0, 0, SIZE, SIZE],
            camera: camera(eye),
            draws: &d,
        }]
    });
    viewer.set_cluster_streaming(1 << 30, 16);
    let full = views.each_ref().map(|v| settle(&mut viewer, v, 400));

    // The smallest pool: the pinned pages and 16 more. Every 30 frames the camera moves to
    // the other side, and the pages it needs push out the ones it no longer does.
    viewer.set_cluster_streaming(0, 4);
    for frame in 0..120 {
        let at = frame / 30 % 2;
        let image = viewer.render_image(SIZE, SIZE, &views[at]);
        let resident = viewer.cluster_resident(mesh);
        assert_eq!(
            drawn(&viewer)[0],
            resident_cut(dag, &resident, &lod(eyes[at], SIZE)),
            "frame {frame}"
        );
        viewer.settle();
        let covered = image.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
        let gap = covered.abs_diff(full[at]) as f32 / full[at] as f32;
        assert!(gap < 0.05, "frame {frame}: {covered} pixels covered, {} fully resident", full[at]);
    }
    let s = viewer.cluster_stats().expect("stats");
    assert!(s.resident_pages <= s.pool_pages);
    assert!(s.resident_pages < s.total_pages, "the pool should be too small for everything");
    assert!(s.evicted > 0, "moving the camera should evict pages");
    eprintln!(
        "{} of {} pages in a pool of {}; {} loaded, {} evicted",
        s.resident_pages, s.total_pages, s.pool_pages, s.loaded, s.evicted
    );
}

#[test]
fn split_screen_streams_for_both_views() {
    let Some(mut viewer) = viewer() else {
        return;
    };
    let dag = dag();
    let mesh = viewer.add_cluster_mesh(dag.clone()).expect("cluster mesh");
    viewer.set_cluster_culling(false, false);
    viewer.set_cluster_streaming(1 << 30, 16);
    // Close to opposite sides: each needs fine pages the other doesn't.
    let eyes = [Vec3::new(0.0, 0.2, 1.5), Vec3::new(0.1, -0.2, -1.5)];
    let d = draws(mesh);
    let half = SIZE / 2;
    let views = [
        View {
            rect: [0, 0, half, SIZE],
            camera: camera(eyes[0]),
            draws: &d,
        },
        View {
            rect: [half, 0, half, SIZE],
            camera: camera(eyes[1]),
            draws: &d,
        },
    ];
    settle(&mut viewer, &views, 600);
    viewer.render_image(SIZE, SIZE, &views);
    let got = drawn(&viewer);
    for (v, eye) in eyes.iter().enumerate() {
        assert_eq!(got[v], sorted(dag.select(&lod(*eye, SIZE))), "view {v}");
    }
    // Neither view alone needs everything.
    let mut alone = Viewer::headless().expect("viewer");
    let m = alone.add_cluster_mesh(dag.clone()).expect("cluster mesh");
    alone.set_cluster_culling(false, false);
    alone.set_cluster_streaming(1 << 30, 16);
    let d1 = draws(m);
    let one = [View {
        rect: [0, 0, half, SIZE],
        camera: camera(eyes[0]),
        draws: &d1,
    }];
    settle(&mut alone, &one, 600);
    let both = viewer.cluster_resident(mesh).iter().filter(|&&r| r).count();
    let single = alone.cluster_resident(m).iter().filter(|&&r| r).count();
    eprintln!("split screen: {both} pages resident; one view: {single}");
    assert!(both > single);
}
