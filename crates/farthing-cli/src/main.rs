//! `farthing` — command-line front end. Geometry-only commands run without an engine; commands
//! that need real materials (baking, impostor captures) need a RenderHost and live in the PoC
//! engine (`engine-poc`) or your own integration.

#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use farthing::host::{GeometryOnlyHost, MaterialInfo};
use farthing::metrics::surface_distance;
use farthing::reduce::{lod_chain, screen_size_to_max_error, simplify, SimplifyOptions};
use farthing::{Mesh, Scene};
use glam::Vec4;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "farthing",
    version,
    about = "Clean-room asset optimization toolkit"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum ImpostorType {
    Octahedral,
    Hemi,
    Flipbook,
    Single,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print scene statistics.
    Info { input: PathBuf },
    /// Reduce every mesh.
    Reduce {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Fraction of triangles to keep.
        #[arg(long)]
        ratio: Option<f32>,
        #[arg(long)]
        triangles: Option<usize>,
        /// Max deviation in world units.
        #[arg(long)]
        max_error: Option<f32>,
        /// Target on-screen size (px of the bounding-box diagonal) with 1 px tolerance.
        #[arg(long)]
        screen_size: Option<f32>,
        #[arg(long)]
        lock_border: bool,
        /// Stop only when all targets are met (default: any).
        #[arg(long)]
        all: bool,
        /// Weld vertices within this distance first.
        #[arg(long)]
        weld: Option<f32>,
        /// Drop UVs/normals and ignore seams (for far LODs that will be re-baked).
        #[arg(long)]
        ignore_seams: bool,
        /// Never collapse across UV seams (keeps UV island borders exact; may stop short of the
        /// target). Default: seams are crossed when that is the cheapest collapse.
        #[arg(long)]
        strict_seams: bool,
        /// Detect a mirror plane and keep the LOD symmetric.
        #[arg(long)]
        symmetry: bool,
        /// Fast mode: a lean single-threaded loop, several times faster, slightly higher error.
        #[arg(long)]
        fast: bool,
    },
    /// Build a LOD chain (writes lod0..N and lods.json into a directory).
    Lods {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4)]
        levels: usize,
        /// Triangle ratio between consecutive levels.
        #[arg(long, default_value_t = 0.5)]
        step: f32,
        /// Screen height used to compute switch distances.
        #[arg(long, default_value_t = 1080.0)]
        screen_height: f32,
        /// Vertical field of view (degrees) for switch distances.
        #[arg(long, default_value_t = 60.0)]
        fov: f32,
        /// From this level on, ignore UV seams (geometry-only; re-bake those levels).
        #[arg(long)]
        far_from: Option<usize>,
    },
    /// Watertight proxy of the whole scene.
    Remesh {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 96)]
        resolution: usize,
        #[arg(long, default_value_t = 2000)]
        triangles: usize,
        /// Seal gaps up to this many voxels.
        #[arg(long, default_value_t = 1)]
        close: usize,
        /// Remove everything below this height.
        #[arg(long)]
        clip_below: Option<f32>,
    },
    /// Conservative occluder mesh (and optional box occluders as JSON).
    Occluder {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 64)]
        resolution: usize,
        #[arg(long, default_value_t = 300)]
        triangles: usize,
        /// Also write N inscribed boxes to <output>.boxes.json.
        #[arg(long)]
        boxes: Option<usize>,
    },
    /// Camera-sampled visibility: report, and optionally cull never-seen triangles.
    Visibility {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        cameras: usize,
        #[arg(long)]
        hemisphere: bool,
        #[arg(long, default_value_t = 128)]
        resolution: u32,
        /// Rings of neighbours kept around visible triangles when culling.
        #[arg(long, default_value_t = 1)]
        rings: usize,
    },
    /// Nanite-style cluster DAG (meshlets + LOD groups) for run-time per-cluster LOD.
    Clusters {
        input: PathBuf,
        /// Write the DAG as JSON.
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        max_triangles: usize,
        #[arg(long, default_value_t = 255)]
        max_vertices: usize,
        /// Fast group reductions (about 2x faster build, attributes kept less faithfully).
        #[arg(long)]
        fast: bool,
    },
    /// Distant-LOD proxy of a whole scene: hidden-surface removal, merge, reduce, unwrap.
    Hlod {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2000)]
        triangles: usize,
        /// Max deviation in world units (with --triangles, whichever comes first).
        #[arg(long)]
        max_error: Option<f32>,
        /// Voxel-remesh at this resolution first (fuses interpenetrating objects).
        #[arg(long)]
        remesh: Option<usize>,
        /// Keep hidden surfaces (skip the visibility pass).
        #[arg(long)]
        no_cull: bool,
        /// Visibility cameras on the upper hemisphere only (things that stand on the ground).
        #[arg(long)]
        hemisphere: bool,
        #[arg(long, default_value_t = 64)]
        cameras: usize,
        #[arg(long, default_value_t = 256)]
        resolution: u32,
        /// UV atlas size for the proxy (0: no UVs).
        #[arg(long, default_value_t = 1024)]
        atlas: u32,
    },
    /// New UV atlas for every mesh.
    Uv {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1024)]
        atlas: u32,
        /// Max chart normal-cone angle (default 85 for LSCM, 60 for planar).
        #[arg(long)]
        angle: Option<f32>,
        /// Planar charts instead of LSCM.
        #[arg(long)]
        planar: bool,
    },
    /// Merge all instances into one mesh with one atlas layout (+ layout JSON). Texel copy
    /// needs host-flattened textures (see engine-poc).
    Aggregate {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        atlas: u32,
    },
    /// Bone reduction and influence limiting for skinned meshes.
    Bones {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        max_bones: Option<usize>,
        #[arg(long)]
        ratio: Option<f32>,
        #[arg(long)]
        influences: Option<usize>,
        /// Drop removed joints from the skeleton (default keeps indices, writes a remap).
        #[arg(long)]
        compact: bool,
    },
    /// Plan impostor captures (directions, cameras, atlas layout) as JSON.
    ImpostorPlan {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = ImpostorType::Octahedral)]
        kind: ImpostorType,
        #[arg(long, default_value_t = 9)]
        frames: u32,
        #[arg(long, default_value_t = 128)]
        frame_size: u32,
    },
    /// Foliage / distant proxy as Gaussian splats: blue-noise surfels written as a 3DGS .ply
    /// (progressive order: any prefix is a coarser covering). Each splat takes the shape,
    /// normal and coverage (opacity) of the patch of surface it stands for, coloured with the
    /// vertex colours averaged over that patch.
    Splats {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 20000)]
        count: usize,
        /// Write only the first N splats of the cloud (a coarser level), shaped for that level.
        #[arg(long)]
        prefix: Option<usize>,
        /// Old fixed-size mode: round splats at half the surfel radius, fixed opacity, colour
        /// from one point per surfel.
        #[arg(long)]
        round: bool,
        /// Largest aspect ratio of a footprint splat.
        #[arg(long, default_value_t = 8.0)]
        max_aspect: f32,
        /// Surface samples per surfel used to measure footprints.
        #[arg(long, default_value_t = 64)]
        samples_per_surfel: u32,
    },
    /// Fit billboard-cloud planes (geometry only) and write card quads.
    Billboard {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 24)]
        planes: usize,
        #[arg(long, default_value_t = 0.03)]
        epsilon: f32,
    },
    /// Quad-preserving reduction by edge-ring removal (quad OBJ in/out, or glTF paired into quads).
    QuadReduce {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        ratio: Option<f32>,
        #[arg(long)]
        quads: Option<usize>,
        #[arg(long)]
        max_error: Option<f32>,
    },
    /// Vertex-cache and vertex-fetch optimization.
    Optimize {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Surface distance between two assets (JSON).
    Compare {
        a: PathBuf,
        b: PathBuf,
        #[arg(long, default_value_t = 20000)]
        samples: usize,
    },
    /// Print the reference octahedral impostor WGSL shader.
    Shader,
}

fn load(p: &Path) -> Result<Scene> {
    farthing::io::load(p).with_context(|| format!("loading {}", p.display()))
}

fn save(s: &Scene, p: &Path) -> Result<()> {
    if let Some(dir) = p.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    farthing::io::save(s, p).with_context(|| format!("saving {}", p.display()))
}

fn print(v: serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
}

fn single_mesh(s: &Scene) -> Mesh {
    s.flatten().0
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let t0 = Instant::now();
    match cli.cmd {
        Cmd::Info { input } => {
            let s = load(&input)?;
            let b = s.bounds();
            print(json!({
                "meshes": s.meshes.len(),
                "instances": s.instances.len(),
                "materials": s.materials.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
                "skeletons": s.skeletons.iter().map(|k| json!({"name": k.name, "joints": k.joints.len()})).collect::<Vec<_>>(),
                "triangles": s.triangle_count(),
                "vertices": s.meshes.iter().map(|m| m.vertex_count()).sum::<usize>(),
                "bounds": {"min": b.min.to_array(), "max": b.max.to_array(), "diagonal": b.diagonal()},
                "acmr_16": s.meshes.iter().map(|m| farthing::reduce::optimize::acmr(&m.indices, 16)).collect::<Vec<_>>(),
            }));
        }
        Cmd::Reduce {
            input,
            output,
            ratio,
            triangles,
            max_error,
            screen_size,
            lock_border,
            all,
            weld,
            ignore_seams,
            strict_seams,
            symmetry,
            fast,
        } => {
            let mut s = load(&input)?;
            let diag = s.bounds().diagonal();
            let max_error =
                max_error.or(screen_size.map(|px| screen_size_to_max_error(diag, px, 1.0)));
            let before = s.triangle_count();
            let mut reports = Vec::new();
            for m in &mut s.meshes {
                if ignore_seams {
                    *m = farthing::reduce::repair::geometry_only(m);
                }
                if let Some(eps) = weld {
                    farthing::reduce::repair::weld(m, eps);
                }
                let opts = SimplifyOptions {
                    target_ratio: ratio,
                    target_triangles: triangles,
                    max_error,
                    lock_border,
                    stop: if all {
                        farthing::reduce::StopWhen::All
                    } else {
                        farthing::reduce::StopWhen::Any
                    },
                    seams: if strict_seams {
                        farthing::reduce::SeamMode::Strict
                    } else {
                        farthing::reduce::SeamMode::Permissive
                    },
                    symmetry: if symmetry {
                        farthing::reduce::Symmetry::Auto
                    } else {
                        farthing::reduce::Symmetry::Off
                    },
                    quality: if fast {
                        farthing::reduce::Quality::Fast
                    } else {
                        farthing::reduce::Quality::Best
                    },
                    ..Default::default()
                };
                let r = simplify(m, &opts)?;
                reports.push(json!({"triangles": r.mesh.triangle_count(), "error": r.error, "relative_error": r.relative_error, "symmetry_plane": r.symmetry_plane.map(|p| p.to_array())}));
                *m = r.mesh;
                if ignore_seams {
                    m.compute_normals(60.0);
                }
            }
            save(&s, &output)?;
            print(
                json!({"input_triangles": before, "output_triangles": s.triangle_count(), "meshes": reports, "seconds": t0.elapsed().as_secs_f32()}),
            );
        }
        Cmd::Lods {
            input,
            output,
            levels,
            step,
            screen_height,
            fov,
            far_from,
        } => {
            let s = load(&input)?;
            std::fs::create_dir_all(&output)?;
            let ratios: Vec<f32> = (1..=levels).map(|i| step.powi(i as i32)).collect();
            let near = far_from.map_or(levels, |f| f.saturating_sub(1).min(levels));
            let chains: Vec<Vec<farthing::reduce::LodLevel>> = s
                .meshes
                .iter()
                .map(|m| -> farthing::Result<Vec<farthing::reduce::LodLevel>> {
                    let mut chain = lod_chain(m, &ratios[..near], &SimplifyOptions::default())?;
                    if near < levels {
                        // Far levels: geometry-only from the last near level.
                        let base = chain.last().unwrap();
                        let geo = farthing::reduce::repair::geometry_only(&base.mesh);
                        let rel: Vec<f32> = ratios[near..]
                            .iter()
                            .map(|r| {
                                r * m.triangle_count() as f32 / geo.triangle_count().max(1) as f32
                            })
                            .collect();
                        let far = lod_chain(&geo, &rel, &SimplifyOptions::default())?;
                        let err0 = base.error;
                        for mut l in far.into_iter().skip(1) {
                            l.error += err0;
                            l.mesh.compute_normals(60.0);
                            chain.push(l);
                        }
                    }
                    Ok(chain)
                })
                .collect::<farthing::Result<_>>()?;
            let fov = fov.to_radians();
            let mut info = Vec::new();
            for lvl in 0..=levels {
                let mut ls = s.clone();
                let mut err: f32 = 0.0;
                for (mi, chain) in chains.iter().enumerate() {
                    ls.meshes[mi] = chain[lvl].mesh.clone();
                    err = err.max(chain[lvl].error);
                }
                let path = output.join(format!("lod{lvl}.glb"));
                save(&ls, &path)?;
                // Switch to this level once its error projects to under 1 pixel.
                let switch = if err > 0.0 {
                    err * screen_height / (2.0 * (fov * 0.5).tan())
                } else {
                    0.0
                };
                info.push(json!({"level": lvl, "file": path.file_name().unwrap().to_string_lossy(), "triangles": ls.triangle_count(), "error": err, "switch_distance_1px": switch}));
            }
            std::fs::write(
                output.join("lods.json"),
                serde_json::to_string_pretty(&info)?,
            )?;
            print(json!({"levels": info, "seconds": t0.elapsed().as_secs_f32()}));
        }
        Cmd::Remesh {
            input,
            output,
            resolution,
            triangles,
            close,
            clip_below,
        } => {
            let s = load(&input)?;
            let opts = farthing::voxel::RemeshOptions {
                resolution,
                close_gaps: close,
                target_triangles: Some(triangles),
                clip_planes: clip_below
                    .map(|y| vec![Vec4::new(0.0, -1.0, 0.0, y)])
                    .unwrap_or_default(),
                ..Default::default()
            };
            let r = farthing::voxel::remesh(&s, &opts)?;
            let d = surface_distance(&single_mesh(&s), &r.mesh, 10000);
            save(&Scene::from_mesh(r.mesh.clone()), &output)?;
            print(
                json!({"input_triangles": s.triangle_count(), "output_triangles": r.mesh.triangle_count(), "voxels": r.voxels, "cell_size": r.cell_size, "extracted_triangles": r.extracted_triangles, "relative_mean_distance": d.relative_mean, "seconds": t0.elapsed().as_secs_f32()}),
            );
        }
        Cmd::Occluder {
            input,
            output,
            resolution,
            triangles,
            boxes,
        } => {
            let s = load(&input)?;
            let r = farthing::voxel::occluder(
                &s,
                &farthing::voxel::OccluderOptions {
                    resolution,
                    target_triangles: triangles,
                    ..Default::default()
                },
            )?;
            save(&Scene::from_mesh(r.mesh.clone()), &output)?;
            let mut out = json!({"triangles": r.mesh.triangle_count(), "outside_fraction": r.outside_fraction, "erode_used": r.erode_used});
            if let Some(n) = boxes {
                let b = farthing::voxel::box_occluders(&s, resolution, n)?;
                let path = output.with_extension("boxes.json");
                std::fs::write(&path, serde_json::to_string_pretty(&b)?)?;
                out["boxes"] = json!(b.len());
            }
            out["seconds"] = json!(t0.elapsed().as_secs_f32());
            print(out);
        }
        Cmd::Visibility {
            input,
            output,
            cameras,
            hemisphere,
            resolution,
            rings,
        } => {
            let s = load(&input)?;
            use farthing::visibility::{
                compute_visibility, cull_invisible, CameraSet, VisibilityOptions,
            };
            let set = if hemisphere {
                CameraSet::Hemisphere {
                    count: cameras,
                    distance_factor: 2.5,
                }
            } else {
                CameraSet::Sphere {
                    count: cameras,
                    distance_factor: 2.5,
                }
            };
            let vis = compute_visibility(
                &mut GeometryOnlyHost,
                &s,
                &VisibilityOptions {
                    cameras: set,
                    resolution,
                },
            )?;
            let mut out =
                json!({"views": vis.view_count, "visible_fraction": vis.visible_fraction()});
            if let Some(o) = output {
                let (culled, removed) = cull_invisible(&s, &vis, rings);
                save(&culled, &o)?;
                out["removed_triangles"] = json!(removed);
                out["output_triangles"] = json!(culled.triangle_count());
            }
            out["seconds"] = json!(t0.elapsed().as_secs_f32());
            print(out);
        }
        Cmd::Clusters {
            input,
            output,
            max_triangles,
            max_vertices,
            fast,
        } => {
            let mesh = single_mesh(&load(&input)?);
            let opts = farthing::cluster::ClusterOptions {
                max_triangles,
                max_vertices,
                ..if fast {
                    farthing::cluster::ClusterOptions::fast()
                } else {
                    Default::default()
                }
            };
            let t1 = Instant::now();
            let dag = farthing::cluster::build_cluster_dag(&mesh, &opts)?;
            let build = t1.elapsed().as_secs_f32();
            if let Some(o) = &output {
                std::fs::write(o, serde_json::to_string(&dag)?)?;
            }
            print(
                json!({"input_triangles": mesh.triangle_count(), "clusters": dag.clusters.len(), "groups": dag.groups.len(), "levels": dag.levels(), "level_triangles": dag.level_triangles(), "roots": dag.roots().count(), "build_seconds": build, "seconds": t0.elapsed().as_secs_f32()}),
            );
        }
        Cmd::Hlod {
            input,
            output,
            triangles,
            max_error,
            remesh,
            no_cull,
            hemisphere,
            cameras,
            resolution,
            atlas,
        } => {
            use farthing::hlod::{build_hlod, HlodGeometry, HlodOptions};
            use farthing::visibility::{CameraSet, VisibilityOptions};
            let s = load(&input)?;
            let d = HlodOptions::default();
            let opts = HlodOptions {
                visibility: (!no_cull).then_some(VisibilityOptions {
                    cameras: if hemisphere {
                        CameraSet::Hemisphere {
                            count: cameras,
                            distance_factor: 2.5,
                        }
                    } else {
                        CameraSet::Sphere {
                            count: cameras,
                            distance_factor: 2.5,
                        }
                    },
                    resolution,
                }),
                geometry: match remesh {
                    Some(r) => HlodGeometry::Remesh(farthing::voxel::RemeshOptions {
                        resolution: r,
                        ..Default::default()
                    }),
                    None => HlodGeometry::Reduce,
                },
                target_triangles: Some(triangles),
                max_error,
                uv: (atlas > 0).then(|| farthing::uv::ParamOptions {
                    atlas_size: atlas,
                    ..d.uv.clone().unwrap()
                }),
                // Materials live in the engine: bake there (or through a RenderHost).
                bake: None,
                ..d
            };
            let r = build_hlod(&mut GeometryOnlyHost, &s, &opts)?;
            let dist = surface_distance(&single_mesh(&s), &r.mesh, 20000);
            save(&Scene::from_mesh(r.mesh.clone()), &output)?;
            print(
                json!({"input_triangles": r.input_triangles, "hidden_triangles": r.hidden_triangles, "output_triangles": r.mesh.triangle_count(), "error": r.error, "relative_mean_distance": dist.relative_mean, "seconds": t0.elapsed().as_secs_f32()}),
            );
        }
        Cmd::Uv {
            input,
            output,
            atlas,
            angle,
            planar,
        } => {
            let mut s = load(&input)?;
            let mut rep = Vec::new();
            for m in &mut s.meshes {
                let base = if planar {
                    farthing::uv::ParamOptions::planar()
                } else {
                    farthing::uv::ParamOptions::default()
                };
                let opts = farthing::uv::ParamOptions {
                    atlas_size: atlas,
                    max_chart_angle_deg: angle.unwrap_or(base.max_chart_angle_deg),
                    ..base
                };
                let r = farthing::uv::parameterize(m, &opts)?;
                rep.push(json!({"charts": r.charts, "lscm_charts": r.lscm_charts, "utilization": r.utilization, "mean_stretch": r.mean_stretch, "texels_per_unit": r.texels_per_unit}));
                *m = r.mesh;
            }
            save(&s, &output)?;
            print(json!({"meshes": rep, "seconds": t0.elapsed().as_secs_f32()}));
        }
        Cmd::Aggregate {
            input,
            output,
            atlas,
        } => {
            let s = load(&input)?;
            let infos = vec![MaterialInfo::default(); s.materials.len()];
            let r = farthing::aggregate::aggregate(
                &s,
                &infos,
                &farthing::aggregate::AggregateOptions {
                    atlas_size: atlas,
                    ..Default::default()
                },
            )?;
            save(&r.scene, &output)?;
            let layout = output.with_extension("layout.json");
            std::fs::write(&layout, serde_json::to_string_pretty(&r.placements)?)?;
            print(
                json!({"draw_calls_before": r.draw_calls_before, "draw_calls_after": r.scene.materials.len(), "islands": r.placements.len(), "texels_per_unit": r.texels_per_unit, "layout": layout, "seconds": t0.elapsed().as_secs_f32()}),
            );
        }
        Cmd::Bones {
            input,
            output,
            max_bones,
            ratio,
            influences,
            compact,
        } => {
            let mut s = load(&input)?;
            if s.skeletons.is_empty() {
                bail!("{} has no skeleton", input.display());
            }
            let mut reports = Vec::new();
            for si in 0..s.skeletons.len() {
                let mesh_ids: Vec<usize> = s
                    .instances
                    .iter()
                    .filter(|i| i.skeleton == Some(si))
                    .map(|i| i.mesh)
                    .collect();
                let meshes: Vec<&Mesh> = mesh_ids.iter().map(|&m| &s.meshes[m]).collect();
                let red = farthing::bones::plan_bone_reduction(
                    &s.skeletons[si],
                    &meshes,
                    None,
                    &farthing::bones::BoneReductionOptions {
                        max_bones,
                        ratio,
                        ..Default::default()
                    },
                )?;
                for &m in &mesh_ids {
                    farthing::bones::apply_bone_reduction(&mut s.meshes[m], &red);
                    if let Some(k) = influences {
                        farthing::bones::limit_influences(&mut s.meshes[m], k);
                    }
                }
                if compact {
                    let (sk, map) = farthing::bones::compact_skeleton(&s.skeletons[si], &red);
                    for &m in &mesh_ids {
                        farthing::bones::remap_joint_indices(&mut s.meshes[m], &map);
                    }
                    s.skeletons[si] = sk;
                }
                reports.push(json!({"skeleton": s.skeletons[si].name, "kept": red.kept_count(), "removed": red.removed, "remap": red.remap}));
            }
            save(&s, &output)?;
            print(json!({"skeletons": reports}));
        }
        Cmd::ImpostorPlan {
            input,
            output,
            kind,
            frames,
            frame_size,
        } => {
            let s = load(&input)?;
            use farthing::impostor::ImpostorKind;
            let k = match kind {
                ImpostorType::Octahedral => ImpostorKind::Octahedral {
                    frames_per_side: frames,
                    hemisphere: false,
                },
                ImpostorType::Hemi => ImpostorKind::Octahedral {
                    frames_per_side: frames,
                    hemisphere: true,
                },
                ImpostorType::Flipbook => ImpostorKind::Flipbook {
                    frames,
                    elevation_deg: 10.0,
                },
                ImpostorType::Single => ImpostorKind::SingleView {
                    direction: glam::Vec3::Z,
                },
            };
            let p = farthing::impostor::plan(s.bounds(), &k, frame_size)?;
            let views: Vec<_> = p
                .views()
                .iter()
                .map(|v| json!({"view": v.view.to_cols_array(), "proj": v.proj.to_cols_array()}))
                .collect();
            std::fs::write(
                &output,
                serde_json::to_string_pretty(&json!({"plan": p, "views": views}))?,
            )?;
            print(json!({"frames": p.directions.len(), "atlas": p.atlas_size(), "output": output}));
        }
        Cmd::Splats {
            input,
            output,
            count,
            prefix,
            round,
            max_aspect,
            samples_per_surfel,
        } => {
            use farthing::impostor::footprint::{
                average_surfel_channel, surfel_footprints, ChannelSpec, FootprintOptions,
            };
            use farthing::impostor::splats::{surfels_to_splats, write_splat_ply, SplatOptions};
            use farthing::impostor::surfels::{
                progressive_surfel_cloud, SurfelChannel, SurfelOptions,
            };
            let mesh = single_mesh(&load(&input)?);
            let cloud = progressive_surfel_cloud(
                &mesh,
                &SurfelOptions {
                    count,
                    ..Default::default()
                },
            )?;
            let k = prefix.unwrap_or(count).clamp(1, cloud.surfels.len());
            let surfels = &cloud.surfels[..k];
            // Radius scale of the prefix: a coarser level covers more surface per splat.
            let scale = (cloud.surfels.len() as f32 / k as f32).sqrt();
            let lerp = |c: &[Vec4], tri: usize, b: glam::Vec3| {
                let t = mesh.triangle(tri);
                c[t[0] as usize] * b.x + c[t[1] as usize] * b.y + c[t[2] as usize] * b.z
            };
            let fopts = FootprintOptions {
                samples_per_surfel,
                max_aspect,
                ..Default::default()
            };
            let (footprints, color) = if round {
                let color = mesh.colors.as_ref().map(|c| SurfelChannel {
                    name: "color".into(),
                    components: 4,
                    data: surfels
                        .iter()
                        .flat_map(|s| {
                            lerp(c, s.triangle as usize, s.barycentrics)
                                .to_array()
                                .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                        })
                        .collect(),
                });
                (None, color)
            } else {
                let fp = surfel_footprints(&mesh, surfels, scale, &fopts)?;
                let color = match &mesh.colors {
                    Some(c) => Some(average_surfel_channel(
                        &mesh,
                        surfels,
                        scale,
                        &fopts,
                        &ChannelSpec {
                            name: "color".into(),
                            components: 4,
                            alpha_weighted: true,
                        },
                        |samples, out| {
                            for (s, o) in samples.iter().zip(out.chunks_exact_mut(4)) {
                                o.copy_from_slice(
                                    &lerp(c, s.triangle as usize, s.barycentrics).to_array(),
                                );
                            }
                        },
                    )?),
                    None => None,
                };
                (Some(fp), color)
            };
            let opts = SplatOptions {
                // Footprints are measured for the prefix; round splats grow by its scale.
                radius_scale: if round { scale } else { 1.0 },
                ..Default::default()
            };
            let splats = surfels_to_splats(surfels, footprints.as_deref(), color.as_ref(), &opts)?;
            std::fs::write(&output, write_splat_ply(&splats))?;
            let mean = |f: &dyn Fn(&farthing::impostor::splats::Splat) -> f32| {
                splats.iter().map(f).sum::<f32>() / splats.len().max(1) as f32
            };
            let mut stats = json!({
                "input_triangles": mesh.triangle_count(),
                "splats": splats.len(),
                "mode": if round { "round" } else { "footprint" },
                "colored": color.is_some(),
                "mean_opacity": mean(&|s| s.opacity),
                "mean_aspect": mean(&|s| s.sigma.x / s.sigma.y.max(1e-30)),
            });
            if let Some(fp) = &footprints {
                let n = fp.len().max(1) as f32;
                stats["mean_coverage"] = json!(fp.iter().map(|f| f.coverage).sum::<f32>() / n);
                stats["mean_normal_spread"] =
                    json!(fp.iter().map(|f| f.normal_spread).sum::<f32>() / n);
            }
            stats["seconds"] = json!(t0.elapsed().as_secs_f32());
            print(stats);
        }
        Cmd::Billboard {
            input,
            output,
            planes,
            epsilon,
        } => {
            let s = load(&input)?;
            let flat = single_mesh(&s);
            let c = farthing::impostor::billboard_cloud(
                &flat,
                &farthing::impostor::BillboardOptions {
                    max_planes: planes,
                    epsilon,
                    ..Default::default()
                },
            )?;
            let mut cards = Mesh::default();
            for p in &c.planes {
                let q = Mesh::new(p.corners().to_vec(), vec![0, 1, 2, 0, 2, 3]);
                cards.append(&q, 0);
            }
            save(&Scene::from_mesh(cards), &output)?;
            print(
                json!({"planes": c.planes.len(), "covered_area_fraction": c.covered_area_fraction, "input_triangles": flat.triangle_count()}),
            );
        }
        Cmd::QuadReduce {
            input,
            output,
            ratio,
            quads,
            max_error,
        } => {
            let qm = if input
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("obj"))
            {
                match farthing::io::load_obj_quads(&input) {
                    Ok(q) => q,
                    Err(_) => {
                        farthing::quad::QuadMesh::from_triangles(
                            &single_mesh(&load(&input)?),
                            0.05,
                        )?
                        .0
                    }
                }
            } else {
                farthing::quad::QuadMesh::from_triangles(&single_mesh(&load(&input)?), 0.05)?.0
            };
            let before = qm.quad_count();
            let r = farthing::reduce::reduce_quads(
                &qm,
                &farthing::reduce::QuadReduceOptions {
                    target_ratio: ratio,
                    target_quads: quads,
                    max_error,
                },
            )?;
            if output
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("obj"))
            {
                farthing::io::save_obj_quads(&r.mesh, &output)?;
            } else {
                save(&Scene::from_mesh(r.mesh.to_mesh()), &output)?;
            }
            print(
                json!({"input_quads": before, "output_quads": r.mesh.quad_count(), "rings_removed": r.rings_removed, "error": r.error, "seconds": t0.elapsed().as_secs_f32()}),
            );
        }
        Cmd::Optimize { input, output } => {
            let mut s = load(&input)?;
            let before: Vec<f32> = s
                .meshes
                .iter()
                .map(|m| farthing::reduce::optimize::acmr(&m.indices, 16))
                .collect();
            for m in &mut s.meshes {
                farthing::reduce::optimize::optimize_mesh(m);
            }
            let after: Vec<f32> = s
                .meshes
                .iter()
                .map(|m| farthing::reduce::optimize::acmr(&m.indices, 16))
                .collect();
            save(&s, &output)?;
            print(json!({"acmr_before": before, "acmr_after": after}));
        }
        Cmd::Compare { a, b, samples } => {
            let (sa, sb) = (load(&a)?, load(&b)?);
            let d = surface_distance(&single_mesh(&sa), &single_mesh(&sb), samples);
            print(
                json!({"a_triangles": sa.triangle_count(), "b_triangles": sb.triangle_count(), "distance": d}),
            );
        }
        Cmd::Shader => print!("{}", farthing::impostor::REFERENCE_SHADER),
    }
    Ok(())
}
