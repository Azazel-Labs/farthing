//! Nested HLOD timing on a synthetic city:
//! `bench_hlod_tree [--blocks N] [--cell S] [--from-source] [--bake] [--parallel K]
//! [--save scene.glb]`.
//!
//! N × N blocks, 20 units apart, each with a building (a subdivided box), three props (UV
//! spheres) and a ground tile; one large plaza slab every 8 × 8 blocks goes to a coarser level.
//! Proxies are hashed and dropped as soon as the builder hands them over, as an exporter would.
//! Prints the build time, nodes and triangles per level, and an output hash, so runs at
//! different thread counts can be checked for identical output.
use farthing::bake::BakeOptions;
use farthing::hlod::{build_hlod_tree_with, HlodOptions, HlodTreeOptions, ParentSource};
use farthing::host::{
    BakedMaterialDesc, ChannelEncoding, ChannelImage, ChannelImages, MaterialInfo,
    SurfaceBakeRequest, ViewImages, ViewRenderRequest,
};
use farthing::{Instance, MaterialId, MaterialSlot, Mesh, Result, Scene};
use glam::{Mat4, Quat, Vec3, Vec4};
use std::time::Instant;

fn cube(n: usize) -> Mesh {
    let mut pos = Vec::new();
    let mut idx = Vec::new();
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    for (nrm, u, v) in faces {
        let base = pos.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let s = i as f32 / n as f32 * 2.0 - 1.0;
                let t = j as f32 / n as f32 * 2.0 - 1.0;
                pos.push(nrm + u * s + v * t);
            }
        }
        let w = n as u32 + 1;
        for j in 0..n as u32 {
            for i in 0..n as u32 {
                let a = base + j * w + i;
                idx.extend_from_slice(&[a, a + 1, a + w + 1, a, a + w + 1, a + w]);
            }
        }
    }
    Mesh::new(pos, idx)
}

fn sphere(rings: u32, segments: u32) -> Mesh {
    let mut pos = Vec::new();
    for r in 0..=rings {
        let th = r as f32 / rings as f32 * std::f32::consts::PI;
        for s in 0..=segments {
            let ph = s as f32 / segments as f32 * std::f32::consts::TAU;
            pos.push(Vec3::new(
                th.sin() * ph.cos(),
                th.cos(),
                th.sin() * ph.sin(),
            ));
        }
    }
    let w = segments + 1;
    let mut idx = Vec::new();
    for r in 0..rings {
        for s in 0..segments {
            let a = r * w + s;
            idx.extend_from_slice(&[a, a + 1, a + w + 1, a, a + w + 1, a + w]);
        }
    }
    let mut m = Mesh::new(pos, idx);
    farthing::reduce::repair::remove_degenerate_triangles(&mut m, 0.0);
    m.compact();
    m
}

fn city(blocks: usize) -> Scene {
    let mut s = Scene {
        meshes: vec![cube(8), sphere(16, 32), cube(1)],
        materials: vec![MaterialSlot {
            name: "m".into(),
            id: MaterialId(0),
        }],
        ..Default::default()
    };
    let mut rng = farthing::rng::Pcg32::new(7);
    let add = |s: &mut Scene, name: String, mesh: usize, t: Mat4| {
        s.instances.push(Instance {
            name,
            mesh,
            transform: t,
            skeleton: None,
        })
    };
    for j in 0..blocks {
        for i in 0..blocks {
            let c = Vec3::new(i as f32 * 20.0 + 10.0, 0.0, j as f32 * 20.0 + 10.0);
            let h = 3.0 + rng.f32() * 12.0;
            let rot = Quat::from_rotation_y(rng.f32() * 0.5);
            add(
                &mut s,
                format!("building_{i}_{j}"),
                0,
                Mat4::from_scale_rotation_translation(
                    Vec3::new(4.0 + rng.f32() * 2.0, h, 4.0 + rng.f32() * 2.0),
                    rot,
                    c + Vec3::Y * h,
                ),
            );
            for k in 0..3 {
                let r = 0.4 + rng.f32() * 0.6;
                let off = Vec3::new(rng.f32() * 16.0 - 8.0, r, 8.0 - k as f32 * 0.5);
                add(
                    &mut s,
                    format!("prop_{i}_{j}_{k}"),
                    1,
                    Mat4::from_scale_rotation_translation(Vec3::splat(r), Quat::IDENTITY, c + off),
                );
            }
            add(
                &mut s,
                format!("tile_{i}_{j}"),
                2,
                Mat4::from_scale_rotation_translation(
                    Vec3::new(9.5, 0.05, 9.5),
                    Quat::IDENTITY,
                    c - Vec3::Y * 0.05,
                ),
            );
            if i % 8 == 4 && j % 8 == 4 {
                add(
                    &mut s,
                    format!("plaza_{i}_{j}"),
                    2,
                    Mat4::from_scale_rotation_translation(
                        Vec3::new(60.0, 0.3, 60.0),
                        Quat::IDENTITY,
                        c - Vec3::Y * 0.4,
                    ),
                );
            }
        }
    }
    s
}

/// Colours by world position; registers materials by counting.
struct Host {
    materials: u32,
}

impl farthing::host::RenderHost for Host {
    fn material_info(&self, _m: MaterialId) -> MaterialInfo {
        MaterialInfo::default()
    }
    fn bake_surface(&mut self, req: &SurfaceBakeRequest) -> Result<ChannelImages> {
        let mut out = ChannelImages::new();
        for c in &req.channels {
            let mut img = farthing::image::Image::new(req.width, req.height, Vec4::ZERO);
            for (i, s) in req.samples.iter().enumerate() {
                if let Some(s) = s {
                    img.pixels[i] = (s.position * 0.01).fract().extend(1.0);
                }
            }
            out.insert(
                c.clone(),
                ChannelImage {
                    image: img,
                    encoding: ChannelEncoding::Color,
                },
            );
        }
        Ok(out)
    }
    fn render_views(&mut self, _r: &ViewRenderRequest) -> Result<Vec<ViewImages>> {
        Err(farthing::Error::Unsupported("render_views"))
    }
    fn create_material(&mut self, _d: &BakedMaterialDesc) -> Result<MaterialId> {
        self.materials += 1;
        Ok(MaterialId(1000 + self.materials))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let blocks: usize = arg("--blocks").map_or(32, |v| v.parse().unwrap());
    let cell: f32 = arg("--cell").map_or(80.0, |v| v.parse().unwrap());
    let from_source = args.iter().any(|a| a == "--from-source");
    let bake = args.iter().any(|a| a == "--bake");
    let parallel: usize = arg("--parallel").map_or(0, |v| v.parse().unwrap());
    let t = Instant::now();
    let scene = city(blocks);
    println!(
        "scene: {} instances, {} triangles ({:.2}s)",
        scene.instances.len(),
        scene.triangle_count(),
        t.elapsed().as_secs_f32()
    );
    if let Some(p) = arg("--save") {
        farthing::io::save(&scene, p).expect("save");
        return;
    }
    let d = HlodOptions::default();
    let opts = HlodTreeOptions {
        cell_size: cell,
        parents: if from_source {
            ParentSource::Source
        } else {
            ParentSource::Children
        },
        proxy: HlodOptions {
            bake: bake.then(|| BakeOptions {
                width: 512,
                height: 512,
                supersample: 1,
                channels: vec![farthing::host::channels::BASE_COLOR.into()],
                ..d.bake.clone().unwrap()
            }),
            ..d
        },
        parallel_cells: parallel,
        ..Default::default()
    };
    let mut host = Host { materials: 0 };
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut handed = 0;
    let t = Instant::now();
    let tree = build_hlod_tree_with(&mut host, &scene, &opts, |i, node| {
        let p = node.proxy.take().unwrap();
        handed += 1;
        for x in p.mesh.positions.iter().flat_map(|p| p.to_array()) {
            hash = (hash ^ x.to_bits() as u64 ^ (i as u64) << 32).wrapping_mul(0x0100_0000_01b3);
        }
        Ok(())
    })
    .expect("hlod tree");
    let secs = t.elapsed().as_secs_f32();
    let per_level: Vec<usize> = (0..tree.levels).map(|l| tree.level(l).count()).collect();
    let max_err: Vec<f32> = (0..tree.levels)
        .map(|l| {
            tree.level(l)
                .map(|n| tree.nodes[n as usize].error)
                .fold(0.0, f32::max)
        })
        .collect();
    println!(
        "{} nodes ({handed} handed over), {} levels, threads {}, {:.2}s{}{}",
        tree.nodes.len(),
        tree.levels,
        rayon::current_num_threads(),
        secs,
        if from_source { ", from source" } else { "" },
        if bake { ", baked 512²" } else { "" },
    );
    println!("nodes per level:     {per_level:?}");
    println!("triangles per level: {:?}", tree.level_triangles());
    println!("max error per level: {max_err:.3?}");
    println!("hash {hash:016x}");
}
