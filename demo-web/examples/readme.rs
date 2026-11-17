//! Renders the panels of the README's recut, cluster and splat figures: the demo's own pipeline
//! on a model, drawn offscreen by the engine viewer the page draws with, its editor overlays
//! included. The sheets in `readme/` lay the panels out (see `readme/build.sh`).
//!
//! ```text
//! cargo run --release --features readme --example readme -- web/samples/FlightHelmet.zip readme/out
//! ```
//!
//! Writes transparent PNGs (straight alpha, sRGB) and `readme.json`, the numbers the sheets
//! quote.

use farthing::math::Aabb;
use farthing_demo_web::modes::ViewRequest;
use farthing_demo_web::pipeline::{Buffer, Demo, Event, LodOptions};
use farthing_poc_engine::editor_visualizations::Overlay;
use farthing_poc_engine::orbit::OrbitCamera;
use farthing_poc_engine::viewer::{AlphaMode, Camera, Draw, LineStyle, Material, View, Viewer};
use glam::{Mat4, Vec3, Vec4};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Device pixels per CSS pixel in the sheets: line widths scale with it, as on the page.
const DPR: f32 = 2.0;
/// The view the cuts are chosen for: 1080 pixels high, 60° vertical field of view.
const SCREEN_HEIGHT: f32 = 1080.0;
const SCREEN_FOV: f32 = 60.0;
/// How much of a panel the model fills, edge to edge.
const FIT: f32 = 0.94;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, out] = args.as_slice() else {
        return Err("usage: readme <model or zip> <out-dir>".into());
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    let name = Path::new(model)
        .file_name()
        .map_or_else(|| model.clone(), |n| n.to_string_lossy().into_owned());
    let mut demo = Demo::load(vec![(name, std::fs::read(model)?)])?;
    let mut viewer = Viewer::headless()?;
    let bounds = demo.scene.bounds();
    let original = demo.info();
    let mut stats = json!({
        "model": demo.name,
        "triangles": original.triangles,
        "materials": original.materials.len(),
        "textured_materials": original.textured_materials,
    });
    stats["recut"] = recut(&mut demo, &mut viewer, bounds, &out)?;
    stats["clusters"] = clusters(&mut demo, &mut viewer, bounds, &out)?;
    stats["splats"] = splats(&mut demo, &mut viewer, bounds, &out)?;
    std::fs::write(
        out.join("readme.json"),
        serde_json::to_string_pretty(&stats)?,
    )?;
    println!("wrote {}", out.display());
    Ok(())
}

fn now() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1000.0
}

fn options(v: Value) -> Result<LodOptions> {
    Ok(serde_json::from_value(v)?)
}

/// A data event: its kind, its JSON and its buffers.
type Data = (String, Value, Vec<Buffer>);

/// Runs the demo, printing its stages; returns the level infos and data events.
fn run(demo: &mut Demo, opts: &LodOptions) -> Result<(Vec<Value>, Vec<Data>)> {
    let mut levels = Vec::new();
    let mut data = Vec::new();
    demo.run(opts, &now, &mut |e| match e {
        Event::Stage(s) => eprintln!("  {s}"),
        Event::Level { info, .. } => levels.push(serde_json::to_value(info).unwrap_or_default()),
        Event::Data {
            kind,
            json,
            buffers,
        } => data.push((
            kind.to_string(),
            serde_json::from_str(&json).unwrap_or_default(),
            buffers,
        )),
        Event::Texture { .. } => {}
    })?;
    Ok((levels, data))
}

/// The page's home direction, at the distance where the scene's bounding box just fills a `w`
/// by `h` panel (with a small margin), divided by `zoom`.
fn camera(bounds: Aabb, w: u32, h: u32, fov_y: f32, zoom: f32) -> Camera {
    let focus = (bounds.center(), bounds.diagonal() / 2.0);
    let mut orbit = OrbitCamera::default();
    orbit.focus = focus;
    orbit.fov_y = fov_y;
    orbit.home();
    let tan_y = (fov_y.to_radians() / 2.0).tan();
    let tan_x = tan_y * w as f32 / h as f32;
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 {
                    bounds.min.x
                } else {
                    bounds.max.x
                },
                if i & 2 == 0 {
                    bounds.min.y
                } else {
                    bounds.max.y
                },
                if i & 4 == 0 {
                    bounds.min.z
                } else {
                    bounds.max.z
                },
            )
        })
        .collect();
    // How much of the frame the box fills at distance `d` (1 = touching an edge).
    let fill = |orbit: &mut OrbitCamera, d: f32| {
        orbit.set_distance(d);
        let view = orbit.view_matrix();
        corners
            .iter()
            .map(|&c| {
                let p = view.transform_point3(c);
                let depth = (-p.z).max(1e-6);
                (p.x.abs() / depth / tan_x).max(p.y.abs() / depth / tan_y)
            })
            .fold(0.0, f32::max)
    };
    let (mut lo, mut hi) = (focus.1 * 0.5, orbit.home_distance(focus.1) * 4.0);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if fill(&mut orbit, mid) > FIT {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    orbit.set_distance(hi / zoom);
    orbit.camera()
}

fn view<'a>(w: u32, h: u32, camera: Camera, draws: &'a [Draw]) -> [View<'a>; 1] {
    [View {
        rect: [0, 0, w, h],
        camera,
        draws,
    }]
}

fn save(rgba: &[u8], w: u32, h: u32, path: &Path) -> Result<()> {
    farthing::image::Image::from_rgba8(w, h, rgba, true).save_png(path, true)?;
    Ok(())
}

fn buffer_f32<'a>(b: &'a [Buffer], i: &Value) -> &'a [f32] {
    match i.as_i64().and_then(|i| b.get(i as usize)) {
        Some(Buffer::F32(v)) => v,
        _ => &[],
    }
}

fn buffer_u32<'a>(b: &'a [Buffer], i: &Value) -> &'a [u32] {
    match i.as_i64().and_then(|i| b.get(i as usize)) {
        Some(Buffer::U32(v)) => v,
        _ => &[],
    }
}

fn f(b: &Buffer) -> &[f32] {
    match b {
        Buffer::F32(v) => v,
        _ => &[],
    }
}

fn u(b: &Buffer) -> &[u32] {
    match b {
        Buffer::U32(v) => v,
        _ => &[],
    }
}

fn mat4(v: &Value) -> Mat4 {
    let a: Vec<f32> = v
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect()
        })
        .unwrap_or_default();
    a.get(..16).map_or(Mat4::IDENTITY, Mat4::from_cols_slice)
}

/// A level payload ([`farthing_demo_web::pipeline::scene_payload`]) in the viewer, as the page's
/// `buildModel` puts it there; returns what to draw.
fn load_scene(viewer: &mut Viewer, scene: &Value, b: &[Buffer]) -> Vec<Draw> {
    let textures: Vec<u32> = scene["textures"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            let rgba = match t["rgba"].as_i64().and_then(|i| b.get(i as usize)) {
                Some(Buffer::U8(v)) => v.as_slice(),
                _ => &[],
            };
            let dim = |k: &str| t[k].as_u64().unwrap_or(1) as u32;
            viewer.add_texture(
                dim("width"),
                dim("height"),
                rgba,
                t["srgb"].as_bool() == Some(true),
            )
        })
        .collect();
    let tex = |v: &Value| v.as_i64().and_then(|i| textures.get(i as usize).copied());
    let materials: Vec<u32> = scene["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|m| {
            let c: Vec<f32> = m["base_color"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect();
            viewer.add_material(&Material {
                base_color: Vec4::from_slice(&c),
                metallic: m["metallic"].as_f64().unwrap_or(0.0) as f32,
                roughness: m["roughness"].as_f64().unwrap_or(0.8) as f32,
                alpha: m["cutoff"]
                    .as_f64()
                    .map_or(AlphaMode::Opaque, |c| AlphaMode::Mask(c as f32)),
                // As on the page: payloads don't say which materials are double-sided.
                double_sided: true,
                base_color_texture: tex(&m["base_color_texture"]),
                metallic_roughness_texture: tex(&m["metallic_roughness_texture"]),
                normal_texture: tex(&m["normal_texture"]),
                ..Default::default()
            })
        })
        .collect();
    let fallback = viewer.add_material(&Material {
        base_color: Vec4::new(0.8, 0.8, 0.8, 1.0),
        roughness: 0.8,
        double_sided: true,
        ..Default::default()
    });
    let meshes: Vec<Vec<u32>> = scene["meshes"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|m| {
            let geometry = viewer.add_geometry(
                buffer_f32(b, &m["positions"]),
                buffer_f32(b, &m["normals"]),
                buffer_f32(b, &m["uvs"]),
                buffer_f32(b, &m["colors"]),
            );
            let indices = buffer_u32(b, &m["indices"]);
            m["parts"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|p| {
                    let p: Vec<usize> = p
                        .as_array()?
                        .iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as usize)
                        .collect();
                    let (start, count, slot) = (p[0], p[1], p[2]);
                    let material = materials.get(slot).copied().unwrap_or(fallback);
                    viewer.add_mesh(geometry, &indices[start..start + count], material)
                })
                .collect()
        })
        .collect();
    let mut draws = Vec::new();
    for i in scene["instances"].as_array().into_iter().flatten() {
        let model = mat4(&i["matrix"]);
        let mesh = i["mesh"].as_u64().unwrap_or(0) as usize;
        for &id in meshes.get(mesh).into_iter().flatten() {
            draws.push(Draw {
                mesh: id,
                model,
                overlays: true,
            });
        }
    }
    draws
}

/// The page's overlay colors (`lineStyles` in app.js): sRGB hex to linear.
fn line_styles(viewer: &mut Viewer, dark: bool) {
    let lin = |hex: u32, a: f32| {
        let c = |s: u32| {
            let c = ((hex >> s) & 255) as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        Vec4::new(c(16), c(8), c(0), a)
    };
    let (wire, seam) = if dark {
        (lin(0x8fbcff, 0.75), lin(0xffa94d, 1.0))
    } else {
        (lin(0x1453c4, 0.7), lin(0xe8590c, 1.0))
    };
    viewer.set_line_style(
        Overlay::Wireframe,
        LineStyle {
            color: wire,
            width: 1.25 * DPR,
        },
    );
    viewer.set_line_style(
        Overlay::Seams,
        LineStyle {
            color: seam,
            width: 2.5 * DPR,
        },
    );
}

// ---------------------------------------------------------------------------------------------
// Recut: the original next to LOD0–LOD3.

fn recut(demo: &mut Demo, viewer: &mut Viewer, bounds: Aabb, out: &Path) -> Result<Value> {
    eprintln!("recut");
    let opts = options(json!({ "mode": "chain", "textures": "recut", "levels": 3, "step": 0.35 }))?;
    let (mut levels, _) = run(demo, &opts)?;
    let (w, h) = (440, 550);
    let cam = camera(bounds, w, h, 30.0, 1.0);
    let mut atlas = Value::Null;
    for (l, info) in levels.iter_mut().enumerate() {
        let (scene, buffers) = demo.level_scene(l)?;
        let count = |k: &str| scene[k].as_array().map_or(0, Vec::len);
        info["meshes"] = json!(count("meshes"));
        info["materials"] = json!(count("materials"));
        info["textures"] = json!(count("textures"));
        if l == 1 {
            // The recut atlas's base color, for the sheet.
            if let Some(t) = scene["textures"]
                .as_array()
                .and_then(|t| t.iter().find(|t| t["srgb"].as_bool() == Some(true)))
                && let Some(Buffer::U8(rgba)) =
                    t["rgba"].as_i64().and_then(|i| buffers.get(i as usize))
            {
                let (tw, th) = (
                    t["width"].as_u64().unwrap_or(1),
                    t["height"].as_u64().unwrap_or(1),
                );
                save(rgba, tw as u32, th as u32, &out.join("recut_atlas.png"))?;
                atlas = json!({ "width": tw, "height": th });
            }
        }
        let draws = load_scene(viewer, &scene, &buffers);
        viewer.set_overlays(false, false);
        let img = viewer.render_image(w, h, &view(w, h, cam, &draws));
        save(&img, w, h, &out.join(format!("recut_{l}.png")))?;
        for (theme, dark) in [("light", false), ("dark", true)] {
            line_styles(viewer, dark);
            for (overlay, wire, seams) in [("seams", false, true), ("wire", true, false)] {
                viewer.set_overlays(wire, seams);
                let img = viewer.render_image(w, h, &view(w, h, cam, &draws));
                save(
                    &img,
                    w,
                    h,
                    &out.join(format!("recut_{l}_{overlay}_{theme}.png")),
                )?;
            }
        }
        for d in draws {
            viewer.remove_mesh(d.mesh);
        }
    }
    Ok(json!({ "levels": levels, "atlas": atlas, "panel": [w, h] }))
}

// ---------------------------------------------------------------------------------------------
// Clusters: the DAG's cut for the model at three sizes on screen.

/// The page's `hsl`, taken as linear.
fn hsl(h: f32, s: f32, l: f32) -> [f32; 3] {
    let a = s * l.min(1.0 - l);
    let f = |n: f32| {
        let k = (n + h * 12.0) % 12.0;
        l - a * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0)
    };
    [f(0.0), f(8.0), f(4.0)]
}

/// The page's `clusterColor`: a stable random color per cluster.
fn cluster_color(id: u32) -> [f32; 3] {
    let mut h = (id ^ 0x9e37_79b9).wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    hsl(
        h as f32 / 4_294_967_296.0,
        0.5 + ((h >> 8) & 63) as f32 / 250.0,
        0.45 + ((h >> 16) & 63) as f32 / 400.0,
    )
}

/// Area-weighted vertex normals from the full-detail (DAG level 0) clusters, as the page's
/// `smoothNormals`.
fn smooth_normals(pos: &[f32], idx: &[u32], tri_start: &[u32], level: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; pos.len()];
    let p = |i: u32| Vec3::from_slice(&pos[3 * i as usize..]);
    for (c, _) in level.iter().enumerate().filter(|&(_, &l)| l == 0) {
        for t in tri_start[c] as usize..tri_start[c + 1] as usize {
            let v = &idx[3 * t..3 * t + 3];
            let n = (p(v[1]) - p(v[0])).cross(p(v[2]) - p(v[0]));
            for &o in v {
                for k in 0..3 {
                    out[3 * o as usize + k] += n[k];
                }
            }
        }
    }
    for n in out.chunks_mut(3) {
        let v = Vec3::from_slice(n).normalize_or(Vec3::Y);
        n.copy_from_slice(&v.to_array());
    }
    out
}

fn clusters(demo: &mut Demo, viewer: &mut Viewer, bounds: Aabb, out: &Path) -> Result<Value> {
    eprintln!("clusters");
    let opts = options(json!({ "mode": "clusters" }))?;
    let (_, data) = run(demo, &opts)?;
    let (_, info, b) = data
        .into_iter()
        .find(|d| d.0 == "clusters")
        .ok_or("no cluster data")?;
    let [Buffer::F32(xf), Buffer::U32(inst_mesh), rest @ ..] = b.as_slice() else {
        return Err("unexpected cluster buffers".into());
    };
    struct M<'a> {
        pos: &'a [f32],
        nrm: Vec<f32>,
        idx: &'a [u32],
        tri_start: &'a [u32],
    }
    let meshes: Vec<M> = rest
        .chunks(7)
        .map(|c| {
            let (pos, idx, tri_start) = (f(&c[0]), u(&c[3]), u(&c[4]));
            let nrm = if f(&c[1]).is_empty() {
                smooth_normals(pos, idx, tri_start, u(&c[6]))
            } else {
                f(&c[1]).to_vec()
            };
            M {
                pos,
                nrm,
                idx,
                tri_start,
            }
        })
        .collect();
    let color_mat = viewer.add_material(&Material {
        base_color: Vec4::ONE,
        roughness: 0.7,
        double_sided: true,
        ..Default::default()
    });
    let ppr = SCREEN_HEIGHT / (2.0 * (SCREEN_FOV.to_radians() / 2.0).tan());
    let (w, h) = (740, 740);
    let mut cuts = Vec::new();
    let cam = camera(bounds, w, h, 30.0, 1.0);
    // Where the model is about 94% of a 1080p screen's height (FIT): the cuts are for it
    // `px` high, proportionally farther.
    let near = camera(bounds, 1080, 1080, SCREEN_FOV, 1.0).eye();
    for (k, px) in [800.0f32, 200.0, 50.0].into_iter().enumerate() {
        let eye = bounds.center() + (near - bounds.center()) * (FIT * SCREEN_HEIGHT / px);
        let distance = eye.distance(bounds.center());
        let (sel, ids) = demo.select(&ViewRequest {
            eye: eye.to_array(),
            ppr,
            threshold: 1.0,
        })?;
        let Some(Buffer::U32(ids)) = ids.first() else {
            return Err("no cut".into());
        };
        let mut draws = Vec::new();
        let mut at = 0;
        for (i, &mi) in inst_mesh.iter().enumerate() {
            let n = ids[at] as usize;
            let picked = &ids[at + 1..at + 1 + n];
            at += 1 + n;
            let m = &meshes[mi as usize];
            let (mut pos, mut nrm, mut col) = (Vec::new(), Vec::new(), Vec::new());
            for &c in picked {
                let rgb = cluster_color(c);
                for t in m.tri_start[c as usize]..m.tri_start[c as usize + 1] {
                    for j in 0..3 {
                        let v = 3 * m.idx[3 * t as usize + j] as usize;
                        pos.extend_from_slice(&m.pos[v..v + 3]);
                        nrm.extend_from_slice(&m.nrm[v..v + 3]);
                        col.extend_from_slice(&rgb);
                    }
                }
            }
            if pos.is_empty() {
                continue;
            }
            let geometry = viewer.add_geometry(&pos, &nrm, &[], &col);
            let indices: Vec<u32> = (0..(pos.len() / 3) as u32).collect();
            if let Some(mesh) = viewer.add_mesh(geometry, &indices, color_mat) {
                draws.push(Draw {
                    mesh,
                    model: Mat4::from_cols_slice(&xf[16 * i..16 * i + 16]),
                    overlays: false,
                });
            }
            viewer.remove_geometry(geometry);
        }
        viewer.set_overlays(false, false);
        let img = viewer.render_image(w, h, &view(w, h, cam, &draws));
        save(&img, w, h, &out.join(format!("clusters_{k}.png")))?;
        // The same cut at its true size on that screen, for the inset.
        let s = (px / FIT).round() as u32;
        let small = camera(bounds, s, s, SCREEN_FOV, 1.0);
        let img = viewer.render_image(s, s, &view(s, s, small, &draws));
        save(&img, s, s, &out.join(format!("clusters_{k}_actual.png")))?;
        for d in draws {
            viewer.remove_mesh(d.mesh);
        }
        let mut sel: Value = serde_json::from_str(&sel)?;
        sel["screen_px"] = json!(px);
        sel["distance"] = json!(distance);
        cuts.push(sel);
    }
    Ok(json!({ "info": info, "cuts": cuts, "panel": [w, h] }))
}

// ---------------------------------------------------------------------------------------------
// Splats: the mesh, its surfels and its Gaussians.

fn splats(demo: &mut Demo, viewer: &mut Viewer, bounds: Aabb, out: &Path) -> Result<Value> {
    eprintln!("splats");
    let opts = options(json!({ "mode": "splats" }))?;
    let (_, data) = run(demo, &opts)?;
    // Three times the panel, so the sheet can show a close-up from the same render.
    let (w, h) = (2220, 2220);
    let cam = camera(bounds, w, h, 30.0, 1.0);
    // The mesh.
    let (scene, buffers) = demo.level_scene(0)?;
    let draws = load_scene(viewer, &scene, &buffers);
    viewer.set_overlays(false, false);
    let img = viewer.render_image(w, h, &view(w, h, cam, &draws));
    save(&img, w, h, &out.join("splats_mesh.png"))?;
    for d in draws {
        viewer.remove_mesh(d.mesh);
    }
    let mut levels = Vec::new();
    for (k, (_, info, b)) in data.iter().filter(|d| d.0 == "splats").enumerate() {
        let Some(Buffer::F32(payload)) = b.first() else {
            continue;
        };
        // Blended Gaussians go back to front: far (most negative view z) first.
        let e = cam.view;
        let z = |s: &[f32]| (e * Vec4::new(s[0], s[1], s[2], 1.0)).z;
        let mut order: Vec<&[f32]> = payload.chunks(16).collect();
        order.sort_by(|a, b| z(a).total_cmp(&z(b)));
        let sorted: Vec<f32> = order.concat();
        let id = viewer.add_splats(&sorted);
        let draws = [Draw {
            mesh: id,
            model: Mat4::IDENTITY,
            overlays: false,
        }];
        for (shape, gaussians) in [("discs", false), ("gaussians", true)] {
            viewer.set_splat_shape(gaussians);
            let img = viewer.render_image(w, h, &view(w, h, cam, &draws));
            save(&img, w, h, &out.join(format!("splats_{k}_{shape}.png")))?;
        }
        viewer.remove_mesh(id);
        levels.push(info.clone());
    }
    Ok(json!({ "levels": levels, "panel": [w, h] }))
}
