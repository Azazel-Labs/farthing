//! The example modes besides the classic chain ([`DemoMode`](crate::pipeline::DemoMode)):
//!
//! * **Option examples** ([`run_presets`]): one reduction target under several named
//!   [`SimplifyOptions`] configurations. farthing has no engine presets; these are the option
//!   differences that matter when targeting one engine or another, given names.
//! * **Cluster DAG** ([`run_clusters`]): a Nanite-style cluster DAG per mesh and its packed GPU
//!   form; the page asks [`Demo::select`] for the view-dependent cut as the camera moves.
//! * **HLOD tree** ([`run_hlod`]): the scene (optionally tiled into a grid of copies) in a
//!   cell tree with a merged proxy per cell; [`Demo::select`] picks the cut.
//! * **Splats** ([`run_splats`]): a progressive surfel cloud turned into Gaussian splats at
//!   three densities.
//!
//! Each run reports through [`Event::Data`] (a JSON description and typed arrays) and keeps
//! only what selection and downloads need in [`Extra`].

use crate::pipeline::{
    original_textures, reference_ppr, Buffer, Demo, Event, LevelInfo, LevelKind, LodOptions,
    MaterialHost, MEASURE_SAMPLES,
};
use farthing::bake::BakeOptions;
use farthing::cluster::{
    build_cluster_dag, ClusterDag, ClusterOptions, GpuClusterDag, GpuOptions, LodView,
};
use farthing::hlod::{build_hlod_tree_with, CellGrid, HlodOptions, HlodTree, HlodTreeOptions};
use farthing::host::{channels, MaterialClass};
use farthing::image::Texture;
use farthing::impostor::footprint::{average_surfel_channel, ChannelSpec, FootprintOptions};
use farthing::impostor::splats::{surfels_to_splats, write_splat_ply, Splat, SplatOptions};
use farthing::impostor::surfels::{progressive_surfel_cloud, SurfelOptions};
use farthing::io::{glb_bytes, ExportTextures, GltfMaterial};
use farthing::metrics::surface_distance;
use farthing::reduce::{
    repair, screen_size_to_max_error, simplify, Quality, SeamMode, SimplifyOptions, Symmetry,
};
use farthing::uv::ParamOptions;
use farthing::{Error, Instance, Mesh, Scene};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use serde::Deserialize;
use serde_json::json;

/// What the last cluster, HLOD or splat run keeps for [`Demo::select`] and downloads.
pub(crate) enum Extra {
    None,
    Clusters(ClusterState),
    Hlod(HlodState),
    /// Splats per splat level (level 1 = the whole cloud).
    Splats(Vec<Vec<Splat>>),
}

pub(crate) struct ClusterState {
    dags: Vec<ClusterDag>,
    gpus: Vec<GpuClusterDag>,
    /// Per instance: its mesh and the inverse of its transform (eye into mesh space).
    instances: Vec<(usize, Mat4)>,
}

pub(crate) struct HlodState {
    tree: HlodTree,
    /// Source triangles of each (tiled) instance.
    instance_triangles: Vec<usize>,
}

/// A camera for [`Demo::select`].
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct ViewRequest {
    pub eye: [f32; 3],
    /// Viewport height in pixels / (2 tan(vertical fov / 2)).
    pub ppr: f32,
    /// Largest acceptable error on screen, in pixels.
    pub threshold: f32,
}

impl ViewRequest {
    fn lod_view(&self, eye: Vec3) -> LodView {
        LodView {
            eye,
            pixels_per_radian: self.ppr.max(1.0),
            threshold: self.threshold.max(0.01),
        }
    }
}

impl Demo {
    /// The view-dependent cut of the last cluster or HLOD run: a JSON summary and the drawn
    /// pieces as `u32` arrays (clusters: per instance `[count, cluster ids…]`; HLOD: proxy
    /// nodes, then source instances).
    pub fn select(&self, view: &ViewRequest) -> Result<(String, Vec<Buffer>), String> {
        let eye = Vec3::from(view.eye);
        match &self.extra {
            Extra::Clusters(st) => {
                let mut out = Vec::new();
                let mut triangles = 0usize;
                let mut clusters = 0usize;
                let mut full = 0usize;
                let mut per_level: Vec<usize> = Vec::new();
                let mut gpu_agrees = true;
                let mut max_error = 0.0f32;
                for &(m, inv) in &st.instances {
                    let dag = &st.dags[m];
                    let lv = view.lod_view(inv.transform_point3(eye));
                    let sel = dag.select(&lv);
                    let mut gpu: Vec<u32> = st.gpus[m]
                        .select(&lv)
                        .iter()
                        .map(|&i| st.gpus[m].clusters[i as usize].source_cluster)
                        .collect();
                    gpu.sort_unstable();
                    let mut cpu = sel.clone();
                    cpu.sort_unstable();
                    gpu_agrees &= gpu == cpu;
                    let scale = inv.inverse().x_axis.truncate().length();
                    out.push(sel.len() as u32);
                    for &c in &sel {
                        let cl = &dag.clusters[c as usize];
                        triangles += cl.triangle_count();
                        let l = cl.level as usize;
                        if per_level.len() <= l {
                            per_level.resize(l + 1, 0);
                        }
                        per_level[l] += 1;
                        max_error = max_error.max(cl.lod_error * scale);
                    }
                    clusters += sel.len();
                    full += dag.level_triangles().first().copied().unwrap_or(0);
                    out.extend_from_slice(&sel);
                }
                let json = json!({
                    "triangles": triangles,
                    "clusters": clusters,
                    "full_triangles": full,
                    "per_level": per_level,
                    "gpu_agrees": gpu_agrees,
                    "max_error": max_error,
                })
                .to_string();
                Ok((json, vec![Buffer::U32(out)]))
            }
            Extra::Hlod(st) => {
                let cut = st.tree.select(&view.lod_view(eye));
                let proxy: usize = cut
                    .nodes
                    .iter()
                    .map(|&n| st.tree.nodes[n as usize].triangles)
                    .sum();
                let source: usize = cut
                    .instances
                    .iter()
                    .map(|&i| st.instance_triangles[i as usize])
                    .sum();
                let max_error = cut
                    .nodes
                    .iter()
                    .map(|&n| st.tree.nodes[n as usize].error)
                    .fold(0.0, f32::max);
                let json = json!({
                    "triangles": proxy + source,
                    "proxy_triangles": proxy,
                    "source_triangles": source,
                    "proxies": cut.nodes.len(),
                    "instances": cut.instances.len(),
                    "full_triangles": st.instance_triangles.iter().sum::<usize>(),
                    "max_error": max_error,
                })
                .to_string();
                Ok((
                    json,
                    vec![Buffer::U32(cut.nodes), Buffer::U32(cut.instances)],
                ))
            }
            _ => Err("no cluster or HLOD run to select from".into()),
        }
    }

    /// A splat level of the last splat run as a 3D Gaussian splatting PLY.
    pub fn level_ply(&self, level: usize) -> Result<Vec<u8>, String> {
        match &self.extra {
            Extra::Splats(levels) => level
                .checked_sub(1)
                .and_then(|i| levels.get(i))
                .map(|s| write_splat_ply(s))
                .ok_or_else(|| format!("no splat level {level}")),
            _ => Err("no splat run".into()),
        }
    }
}

fn err(e: Error) -> String {
    e.to_string()
}

/// Largest axis scale of the instances using each mesh (mesh units to world units).
pub(crate) fn mesh_scales(scene: &Scene) -> Vec<f32> {
    let mut scale = vec![0.0f32; scene.meshes.len()];
    for inst in &scene.instances {
        let m = Mat3::from_mat4(inst.transform);
        let s = m
            .x_axis
            .length()
            .max(m.y_axis.length())
            .max(m.z_axis.length());
        scale[inst.mesh] = scale[inst.mesh].max(s);
    }
    scale
}

/// Reports the original as level 0 (every mode shows it).
fn emit_original(demo: &Demo, emit: &mut dyn FnMut(Event)) {
    let s = &demo.scene;
    let info = LevelInfo {
        triangles: s.triangle_count(),
        vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
        ..LevelInfo::new(0, LevelKind::Original)
    };
    emit(Event::Level {
        info: &info,
        glb: &[],
    });
}

// ---------------------------------------------------------------------------------------------
// Option examples

struct Preset {
    label: &'static str,
    note: &'static str,
    opts: SimplifyOptions,
    /// Drop UVs and normals first (repair::geometry_only) and recompute normals.
    geometry_only: bool,
    /// No triangle target: reduce until the error bound for a 256-pixel object at 1 pixel.
    pixel_bound: bool,
}

/// Named example configurations: each differs from the library defaults in one choice an
/// engine integration typically makes. Not engine presets (farthing has none).
fn presets() -> Vec<Preset> {
    let d = SimplifyOptions::default;
    let p = |label, note, opts| Preset {
        label,
        note,
        opts,
        geometry_only: false,
        pixel_bound: false,
    };
    vec![
        p(
            "Library defaults",
            "SimplifyOptions::default(): permissive seams, quality Best",
            d(),
        ),
        p(
            "Strict seams",
            "seams: Strict (UV/normal seams never open; engines with cached tangents)",
            SimplifyOptions {
                seams: SeamMode::Strict,
                ..d()
            },
        ),
        p(
            "Fast quality",
            "quality: Fast (build-time budget, e.g. runtime or editor previews)",
            SimplifyOptions {
                quality: Quality::Fast,
                ..d()
            },
        ),
        p(
            "Preserve symmetry",
            "symmetry: Auto (mirror-plane collapses; characters, vehicles)",
            SimplifyOptions {
                symmetry: Symmetry::Auto,
                ..d()
            },
        ),
        p(
            "Lock borders",
            "lock_border: true (open edges fixed; tiled terrain, modular kits)",
            SimplifyOptions {
                lock_border: true,
                ..d()
            },
        ),
        p(
            "Classic QEM ranking",
            "error_normalization: 1 (plain quadric order instead of normalized)",
            SimplifyOptions {
                error_normalization: 1.0,
                ..d()
            },
        ),
        p(
            "No silhouette weight",
            "silhouette_weight: 0 (outline not protected)",
            SimplifyOptions {
                silhouette_weight: 0.0,
                ..d()
            },
        ),
        p(
            "Keep all parts",
            "remove_parts: false (small disconnected parts are never dropped)",
            SimplifyOptions {
                remove_parts: false,
                ..d()
            },
        ),
        Preset {
            label: "Geometry only",
            note: "repair::geometry_only, then normals at 60° (UVs dropped; shadow/collision LODs)",
            opts: d(),
            geometry_only: true,
            pixel_bound: false,
        },
        Preset {
            label: "Pixel-error bound",
            note: "no triangle target; max_error = screen_size_to_max_error(diag, 256 px, 1 px)",
            opts: d(),
            geometry_only: false,
            pixel_bound: true,
        },
    ]
}

/// One reduction under each [`presets`] configuration, measured against the original.
pub(crate) fn run_presets(
    demo: &mut Demo,
    opts: &LodOptions,
    now: &dyn Fn() -> f64,
    emit: &mut dyn FnMut(Event),
) -> Result<(), String> {
    emit_original(demo, emit);
    let ratio = opts.preset_ratio.clamp(0.01, 0.95);
    let diag = demo.scene.bounds().diagonal().max(f32::MIN_POSITIVE);
    let scales = mesh_scales(&demo.scene);
    let source = demo.scene.flatten().0;
    for (i, p) in presets().into_iter().enumerate() {
        let level = i + 1;
        emit(Event::Stage(&format!("{}: reducing", p.label)));
        let t0 = now();
        let mut meshes = Vec::with_capacity(demo.scene.meshes.len());
        let mut bound = 0.0f32;
        for (mi, mesh) in demo.scene.meshes.iter().enumerate() {
            let geo;
            let input = if p.geometry_only {
                let mut g = repair::geometry_only(mesh);
                g.compute_normals(60.0);
                geo = g;
                &geo
            } else {
                mesh
            };
            let scale = scales[mi].max(f32::MIN_POSITIVE);
            let so = if p.pixel_bound {
                SimplifyOptions {
                    max_error: Some(screen_size_to_max_error(diag, 256.0, 1.0) / scale),
                    ..p.opts.clone()
                }
            } else {
                SimplifyOptions {
                    target_triangles: Some(
                        ((mesh.triangle_count() as f32 * ratio) as usize).max(1),
                    ),
                    ..p.opts.clone()
                }
            };
            let mut r = simplify(input, &so).map_err(err)?;
            if p.geometry_only {
                r.mesh.compute_normals(60.0);
            }
            bound = bound.max(r.error * scales[mi]);
            meshes.push(r.mesh);
        }
        let s = Scene {
            meshes,
            materials: demo.scene.materials.clone(),
            instances: demo.scene.instances.clone(),
            skeletons: demo.scene.skeletons.clone(),
        };
        // Like the chain: the page gets geometry (it reuses the original's materials), the
        // download gets the original's textures.
        let glb = glb_bytes(&s, &ExportTextures::default()).map_err(err)?;
        let ms = now() - t0;
        let error = surface_distance(&source, &s.flatten().0, MEASURE_SAMPLES).hausdorff;
        let info = LevelInfo {
            triangles: s.triangle_count(),
            vertices: s.meshes.iter().map(Mesh::vertex_count).sum(),
            error,
            relative_error: error / diag,
            error_bound: bound,
            switch_distance_1px: error * reference_ppr(),
            ms,
            label: Some(p.label.into()),
            note: Some(p.note.into()),
            bytes: Some(glb.len()),
            ..LevelInfo::new(level, LevelKind::Reduced)
        };
        emit(Event::Level {
            info: &info,
            glb: &glb,
        });
        demo.levels.push((s, original_textures(&demo.materials)));
    }
    emit(Event::Stage("done"));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Cluster DAG

fn flat_vec3(v: &[Vec3]) -> Vec<f32> {
    v.iter().flat_map(|p| p.to_array()).collect()
}

fn flat_vec2(v: &[Vec2]) -> Vec<f32> {
    v.iter().flat_map(|p| p.to_array()).collect()
}

/// A cluster DAG and its GPU form per mesh. Sends every cluster's triangles once (all levels)
/// so the page can draw any cut without asking for geometry again.
pub(crate) fn run_clusters(
    demo: &mut Demo,
    opts: &LodOptions,
    now: &dyn Fn() -> f64,
    emit: &mut dyn FnMut(Event),
) -> Result<(), String> {
    emit_original(demo, emit);
    let mut copts = if opts.cluster_fast {
        ClusterOptions::fast()
    } else {
        ClusterOptions::default()
    };
    if opts.cluster_layout == "meshlet" {
        // Mesh-shader friendly: 64 vertices, 124 triangles.
        copts.max_triangles = 124;
        copts.max_vertices = 64;
    }
    let t0 = now();
    let scene = &demo.scene;
    let mut dags = Vec::with_capacity(scene.meshes.len());
    let mut gpus = Vec::with_capacity(scene.meshes.len());
    let mut meshes_json = Vec::new();
    let mut buffers = Vec::new();
    buffers.push(Buffer::F32(
        scene
            .instances
            .iter()
            .flat_map(|i| i.transform.to_cols_array())
            .collect(),
    ));
    buffers.push(Buffer::U32(
        scene.instances.iter().map(|i| i.mesh as u32).collect(),
    ));
    for (mi, mesh) in scene.meshes.iter().enumerate() {
        emit(Event::Stage(&format!(
            "Cluster DAG: mesh {}/{}",
            mi + 1,
            scene.meshes.len()
        )));
        let tm = now();
        let dag = build_cluster_dag(mesh, &copts).map_err(err)?;
        let dag_ms = now() - tm;
        let gpu = GpuClusterDag::from_dag(&dag, mesh, &GpuOptions::default()).map_err(err)?;
        let levels = dag.levels() as usize;
        let mut level_clusters = vec![0usize; levels];
        let mut indices = Vec::new();
        let mut tri_start = Vec::with_capacity(dag.clusters.len() + 1);
        let mut materials = Vec::new();
        let mut cluster_level = Vec::with_capacity(dag.clusters.len());
        tri_start.push(0u32);
        for c in &dag.clusters {
            level_clusters[c.level as usize] += 1;
            indices.extend_from_slice(&c.indices);
            materials.extend_from_slice(&c.material_ids);
            tri_start.push((indices.len() / 3) as u32);
            cluster_level.push(c.level);
        }
        meshes_json.push(json!({
            "triangles": mesh.triangle_count(),
            "clusters": dag.clusters.len(),
            "groups": dag.groups.len(),
            "levels": levels,
            "level_triangles": dag.level_triangles(),
            "level_clusters": level_clusters,
            "gpu_bytes": gpu.file_size(),
            "pages": gpu.pages.len(),
            "step": gpu.step,
            "ms": dag_ms,
        }));
        buffers.push(Buffer::F32(flat_vec3(&mesh.positions)));
        buffers.push(Buffer::F32(
            mesh.normals.as_deref().map(flat_vec3).unwrap_or_default(),
        ));
        buffers.push(Buffer::F32(
            mesh.uvs.first().map(|u| flat_vec2(u)).unwrap_or_default(),
        ));
        buffers.push(Buffer::U32(indices));
        buffers.push(Buffer::U32(tri_start));
        buffers.push(Buffer::U32(materials));
        buffers.push(Buffer::U32(cluster_level));
        dags.push(dag);
        gpus.push(gpu);
    }
    let all: usize = meshes_json
        .iter()
        .map(|m| {
            m["level_triangles"]
                .as_array()
                .map_or(0, |a| a.iter().filter_map(|v| v.as_u64()).sum::<u64>())
                as usize
        })
        .sum();
    let json = json!({
        "meshes": meshes_json,
        "clusters": dags.iter().map(|d| d.clusters.len()).sum::<usize>(),
        "levels": dags.iter().map(ClusterDag::levels).max().unwrap_or(0),
        "all_level_triangles": all,
        "gpu_bytes": gpus.iter().map(GpuClusterDag::file_size).sum::<usize>(),
        "max_triangles": copts.max_triangles,
        "max_vertices": copts.max_vertices,
        "ms": now() - t0,
        "reference_ppr": reference_ppr(),
    })
    .to_string();
    emit(Event::Data {
        kind: "clusters",
        json,
        buffers,
    });
    demo.extra = Extra::Clusters(ClusterState {
        dags,
        gpus,
        instances: scene
            .instances
            .iter()
            .map(|i| (i.mesh, i.transform.inverse()))
            .collect(),
    });
    emit(Event::Stage("done"));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// HLOD tree

/// Proxy textures for export: the bake's channels as material 0.
fn proxy_textures(bake: &farthing::bake::BakeOutput) -> ExportTextures {
    let mut tex = ExportTextures::default();
    for (name, ch) in &bake.textures {
        match name.as_str() {
            channels::BASE_COLOR => {
                tex.base_color
                    .insert(0, Texture::from_image(&ch.image, true));
            }
            channels::NORMAL => {
                tex.normal
                    .insert(0, Texture::from_signed_normals(&ch.image));
            }
            channels::ORM => {
                tex.metallic_roughness
                    .insert(0, Texture::from_image(&ch.image, false));
            }
            _ => {}
        }
    }
    tex
}

/// The scene tiled `n × n` on the xz plane (copy offsets centred on the original), plus the
/// offsets.
fn tiled_scene(scene: &Scene, n: u32) -> (Scene, Vec<Vec3>, f32) {
    let b = scene.bounds();
    let e = b.extent();
    let spacing = (e.x.max(e.z) * 1.3).max(b.diagonal() * 0.25).max(1e-6);
    let half = (n as f32 - 1.0) * 0.5;
    let mut offsets = Vec::new();
    for iz in 0..n {
        for ix in 0..n {
            offsets.push(Vec3::new(
                (ix as f32 - half) * spacing,
                0.0,
                (iz as f32 - half) * spacing,
            ));
        }
    }
    let mut instances = Vec::with_capacity(offsets.len() * scene.instances.len());
    for o in &offsets {
        for inst in &scene.instances {
            instances.push(Instance {
                name: inst.name.clone(),
                mesh: inst.mesh,
                transform: Mat4::from_translation(*o) * inst.transform,
                skeleton: None,
            });
        }
    }
    let s = Scene {
        meshes: scene.meshes.clone(),
        materials: scene.materials.clone(),
        instances,
        skeletons: Vec::new(),
    };
    (s, offsets, spacing)
}

/// An HLOD tree over the (tiled) scene. Each proxy is sent as a GLB as soon as the builder is
/// done with it and then freed; the tree itself (bounds, errors, hierarchy) is kept.
pub(crate) fn run_hlod(
    demo: &mut Demo,
    opts: &LodOptions,
    now: &dyn Fn() -> f64,
    emit: &mut dyn FnMut(Event),
) -> Result<(), String> {
    emit_original(demo, emit);
    let t0 = now();
    let n = opts.hlod_copies.clamp(1, 7);
    let (scene, offsets, spacing) = tiled_scene(&demo.scene, n);
    let atlas = 256;
    let proxy = HlodOptions {
        target_triangles: Some(opts.hlod_triangles.clamp(50, 50_000)),
        uv: opts.hlod_bake.then(|| ParamOptions {
            atlas_size: atlas,
            padding: 2,
            ..Default::default()
        }),
        bake: opts.hlod_bake.then(|| BakeOptions {
            width: atlas,
            height: atlas,
            supersample: 1,
            gutter: 4,
            channels: vec![
                channels::BASE_COLOR.into(),
                channels::NORMAL.into(),
                channels::ORM.into(),
            ],
            ..Default::default()
        }),
        ..Default::default()
    };
    let topts = HlodTreeOptions {
        cell_size: spacing / opts.hlod_split.clamp(1, 4) as f32,
        grid: CellGrid::Quadtree,
        proxy,
        parallel_cells: 1,
        ..Default::default()
    };
    let mut host = MaterialHost::new(&demo.materials, opts.gpu_visibility);
    let mut built = 0usize;
    emit(Event::Stage(&format!(
        "HLOD: building {} instances",
        scene.instances.len()
    )));
    let tree = build_hlod_tree_with(&mut host, &scene, &topts, |idx, node| {
        let Some(p) = node.proxy.take() else {
            return Ok(());
        };
        let tex = p.bake.as_ref().map(proxy_textures).unwrap_or_default();
        let baked = p.bake.is_some();
        let coverage = p.bake.as_ref().map(|b| b.coverage);
        drop(p.bake);
        let mut s = Scene::from_mesh(p.mesh);
        s.materials[0].name = format!("hlod_{idx}");
        let glb = glb_bytes(&s, &tex)?;
        built += 1;
        emit(Event::Stage(&format!("HLOD: {built} proxies")));
        emit(Event::Data {
            kind: "hlod_node",
            json: json!({
                "node": idx,
                "level": node.level,
                "triangles": node.triangles,
                "input_triangles": p.input_triangles,
                "hidden_triangles": p.hidden_triangles,
                "error": node.error,
                "baked": baked,
                "coverage": coverage,
                "bytes": glb.len(),
            })
            .to_string(),
            buffers: vec![Buffer::U8(glb)],
        });
        Ok(())
    })
    .map_err(err)?;
    let ppr = reference_ppr();
    let nodes: Vec<_> = tree
        .nodes
        .iter()
        .map(|n| {
            json!({
                "level": n.level,
                "parent": n.parent,
                "children": n.children,
                "own_instances": n.own_instances.len(),
                "instances": n.instances.len(),
                "triangles": n.triangles,
                "source_triangles": n.source_triangles,
                "error": n.error,
                "geometric_error": n.geometric_error,
                "texel_size": n.texel_size,
                "sphere": [n.sphere.center.x, n.sphere.center.y, n.sphere.center.z, n.sphere.radius],
                "switch": n.switch_distance(ppr, 1.0),
            })
        })
        .collect();
    let src = demo.scene.instances.len();
    let instance_triangles: Vec<usize> = scene
        .instances
        .iter()
        .map(|i| scene.meshes[i.mesh].triangle_count())
        .collect();
    let b = scene.bounds();
    let json = json!({
        "nodes": nodes,
        "roots": tree.roots,
        "levels": tree.levels,
        "level_triangles": tree.level_triangles(),
        "source_instances": src,
        "copies": n * n,
        "offsets": offsets.iter().map(|o| o.to_array()).collect::<Vec<_>>(),
        "instance_triangles": instance_triangles.iter().sum::<usize>(),
        "cell_size": topts.cell_size,
        "focus": { "center": b.center().to_array(), "radius": b.diagonal() * 0.5 },
        "joint_nodes": demo.scene.skeletons.iter().map(|s| s.joints.len()).sum::<usize>(),
        "baked": opts.hlod_bake,
        "ms": now() - t0,
        "visibility": {
            "gpu": crate::gpu::gpu_name().filter(|_| host.visibility.gpu_calls > 0),
            "gpu_renders": host.visibility.gpu_calls,
            "cpu_renders": host.visibility.cpu_calls,
            "views": host.visibility.views,
            "ms": host.visibility.ms,
        },
        "reference_ppr": ppr,
    })
    .to_string();
    emit(Event::Data {
        kind: "hlod_tree",
        json,
        buffers: Vec::new(),
    });
    demo.extra = Extra::Hlod(HlodState {
        tree,
        instance_triangles,
    });
    emit(Event::Stage("done"));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Splats

/// Linear base color (× factor × vertex color) of `mesh` triangle `t` at `bary`, as the glTF
/// material describes it. `mesh` is the flattened scene (material ids are scene slots).
fn base_color_at(
    materials: &[GltfMaterial],
    scene: &Scene,
    mesh: &Mesh,
    t: usize,
    bary: Vec3,
) -> Vec4 {
    let [a, b, c] = [0, 1, 2].map(|k| mesh.indices[t * 3 + k] as usize);
    let lerp4 = |v: &[Vec4]| v[a] * bary.x + v[b] * bary.y + v[c] * bary.z;
    let mut color = mesh.colors.as_deref().map_or(Vec4::ONE, lerp4);
    let slot = mesh.material_ids.get(t).copied().unwrap_or(0) as usize;
    let Some(m) = scene
        .materials
        .get(slot)
        .and_then(|s| materials.get(s.id.0 as usize))
    else {
        return Vec4::new(0.7, 0.7, 0.7, 1.0);
    };
    color *= m.base_color_factor;
    if let Some((tex, set)) = &m.base_color_texture {
        if let Some(uv) = mesh.uvs.get(*set as usize).or(mesh.uvs.first()) {
            let p = uv[a] * bary.x + uv[b] * bary.y + uv[c] * bary.z;
            color *= tex.sample_bilinear(p, true);
        }
    }
    if m.info.class == MaterialClass::Opaque {
        color.w = 1.0;
    }
    color
}

/// Floats per splat sent to the page: position, covariance (xx xy xz yy yz zz), linear RGBA,
/// normal.
pub const SPLAT_FLOATS: usize = 16;

fn splat_payload(splats: &[Splat]) -> Vec<f32> {
    let mut out = Vec::with_capacity(splats.len() * SPLAT_FLOATS);
    for s in splats {
        let m = Mat3::from_quat(s.rotation) * Mat3::from_diagonal(s.sigma);
        let c = m * m.transpose();
        let n = s.rotation * Vec3::Z;
        out.extend_from_slice(&s.position.to_array());
        out.extend_from_slice(&[
            c.x_axis.x, c.y_axis.x, c.z_axis.x, c.y_axis.y, c.z_axis.y, c.z_axis.z,
        ]);
        out.extend_from_slice(&s.color.to_array());
        out.push(s.opacity);
        out.extend_from_slice(&n.to_array());
    }
    out
}

/// A progressive surfel cloud of the flattened scene, as Gaussian splats at the whole cloud,
/// a quarter and a sixteenth of it (each prefix covers the surface on its own).
pub(crate) fn run_splats(
    demo: &mut Demo,
    opts: &LodOptions,
    now: &dyn Fn() -> f64,
    emit: &mut dyn FnMut(Event),
) -> Result<(), String> {
    emit_original(demo, emit);
    let t0 = now();
    let diag = demo.scene.bounds().diagonal().max(f32::MIN_POSITIVE);
    let flat = demo.scene.flatten().0;
    let n = opts.splat_count.clamp(1_000, 200_000);
    emit(Event::Stage(&format!("Splats: placing {n} surfels")));
    let cloud = progressive_surfel_cloud(
        &flat,
        &SurfelOptions {
            count: n,
            ..Default::default()
        },
    )
    .map_err(err)?;
    let cloud_ms = now() - t0;
    let n = cloud.surfels.len();
    let fopts = FootprintOptions::default();
    let mut kept = Vec::new();
    for (i, k) in [n, n / 4, n / 16].into_iter().enumerate() {
        let k = k.max(1);
        let level = i + 1;
        let t1 = now();
        emit(Event::Stage(&format!("Splats: {k} splats")));
        let (surfels, scale, _) = cloud.lod(k as f32);
        let footprints = if opts.splat_footprints {
            Some(cloud.footprints(&flat, k, &fopts).map_err(err)?)
        } else {
            None
        };
        let color = average_surfel_channel(
            &flat,
            surfels,
            scale,
            &fopts,
            &ChannelSpec {
                name: channels::BASE_COLOR.into(),
                components: 4,
                alpha_weighted: true,
            },
            |samples, out| {
                for (s, o) in samples.iter().zip(out.as_chunks_mut::<4>().0) {
                    let c = base_color_at(
                        &demo.materials,
                        &demo.scene,
                        &flat,
                        s.triangle as usize,
                        s.barycentrics,
                    );
                    o.copy_from_slice(&c.to_array());
                }
            },
        )
        .map_err(err)?;
        let splats = surfels_to_splats(
            surfels,
            footprints.as_deref(),
            Some(&color),
            &SplatOptions {
                // Round discs of a prefix grow to cover what the whole cloud covers;
                // footprints are measured for the prefix itself.
                radius_scale: if footprints.is_some() { 1.0 } else { scale },
                ..Default::default()
            },
        )
        .map_err(err)?;
        let radius = cloud.radius_for(k as f32);
        // Distance at which `k` splats put about two pixels across each (reference view).
        let ppu =
            (k as f32 * std::f32::consts::FRAC_PI_4 * 4.0 / (0.5 * cloud.area.max(1e-12))).sqrt();
        let ply_bytes = write_splat_ply(&splats).len();
        let info = LevelInfo {
            vertices: k,
            error: radius,
            relative_error: radius / diag,
            switch_distance_1px: reference_ppr() / ppu.max(1e-12),
            ms: now() - t1 + if level == 1 { cloud_ms } else { 0.0 },
            bytes: Some(ply_bytes),
            label: Some(format!("{k} splats")),
            note: Some(if footprints.is_some() {
                "footprint Gaussians (shaped by their surface patch)".into()
            } else {
                "round Gaussians (disc radius)".into()
            }),
            ..LevelInfo::new(level, LevelKind::Splats)
        };
        emit(Event::Data {
            kind: "splats",
            json: serde_json::to_string(&info).unwrap_or_default(),
            buffers: vec![Buffer::F32(splat_payload(&splats))],
        });
        kept.push(splats);
    }
    demo.extra = Extra::Splats(kept);
    emit(Event::Stage("done"));
    Ok(())
}
