//! farthing proof-of-concept: runs every farthing processor against a small wgpu engine whose
//! materials farthing never sees, renders the results next to the originals and scores them.
//!
//! Usage: `farthing-poc-engine [scene.glb] [--out DIR]`
//! Without a file, a procedural demo scene is used (its materials depend on world position and
//! procedural noise, so they cannot be expressed as glTF PBR).

mod engine;
mod scene_gen;

use anyhow::Result;
use engine::{Engine, MaterialDef};
use farthing::bake::{bake_with_host, BakeOptions, FlattenedTextures};
use farthing::glam::{Vec3, Vec4};
use farthing::host::{channels, BakedMaterialDesc, BakedMaterialKind, ChannelEncoding, MaterialClass, MaterialInfo, RenderHost, View};
use farthing::image::Image;
use farthing::metrics::{image_difference, ImageDiff};
use farthing::{Instance, MaterialId, MaterialSlot, Mesh, Scene};
use farthing::io::ExportTextures;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const VIEW_SIZE: u32 = 512;

struct Ctx<'a> {
    engine: &'a mut Engine,
    out: PathBuf,
    views: Vec<View>,
    reference: Vec<Image>,
    report: BTreeMap<String, Value>,
    sheet_rows: Vec<Vec<Image>>,
}

fn views_for(scene: &Scene) -> Vec<View> {
    let b = scene.bounds();
    let (c, r) = (b.center(), b.radius());
    [Vec3::new(0.35, 0.45, 1.0), Vec3::new(-1.0, 0.25, 0.4), Vec3::new(0.6, 1.0, -0.5)]
        .iter()
        .map(|d| View::perspective(c + d.normalize() * r * 2.6, c, 40f32.to_radians(), VIEW_SIZE, VIEW_SIZE, r * 0.05, r * 6.0))
        .collect()
}

fn half(img: &Image) -> Image {
    let mut d = img.downsample_coverage(2);
    // Composite on a neutral background for the contact sheet.
    for p in &mut d.pixels {
        let bg = Vec3::splat(0.18);
        *p = (bg.lerp(p.truncate(), p.w)).extend(1.0);
    }
    d
}

impl Ctx<'_> {
    fn score(&mut self, name: &str, renders: &[Image]) -> Vec<ImageDiff> {
        let diffs: Vec<ImageDiff> = renders.iter().zip(&self.reference).map(|(a, b)| image_difference(b, a)).collect();
        for (i, r) in renders.iter().enumerate() {
            let _ = r.save_png(self.out.join(format!("{name}_view{i}.png")), true);
        }
        self.sheet_rows.push(renders.iter().map(half).collect());
        diffs
    }

    fn render_scene(&mut self, scene: &Scene) -> Vec<Image> {
        self.views.clone().iter().map(|v| self.engine.render(scene, v, true).0).collect()
    }

    fn record(&mut self, key: &str, v: Value) {
        println!("  {key}: {}", serde_json::to_string(&v).unwrap());
        self.report.insert(key.into(), v);
    }
}

fn mean_psnr(d: &[ImageDiff]) -> f32 {
    d.iter().map(|x| x.psnr).sum::<f32>() / d.len().max(1) as f32
}
fn mean_mismatch(d: &[ImageDiff]) -> f32 {
    d.iter().map(|x| x.coverage_mismatch).sum::<f32>() / d.len().max(1) as f32
}

/// Single-instance scene, used as a bake source.
fn only_instance(scene: &Scene, i: usize) -> Scene {
    let inst = &scene.instances[i];
    Scene {
        meshes: vec![scene.meshes[inst.mesh].clone()],
        materials: scene.materials.clone(),
        instances: vec![Instance { mesh: 0, ..inst.clone() }],
        skeletons: Vec::new(),
    }
}

/// Bakes the source scene onto `dest` (with new UVs) through the engine and registers the result.
fn bake_onto(engine: &mut Engine, dest: &Mesh, transform: farthing::glam::Mat4, source: &Scene, size: u32, name: &str) -> Result<(MaterialId, farthing::host::ChannelImages, f32)> {
    let opts = BakeOptions { width: size, height: size, supersample: 2, ..Default::default() };
    let out = bake_with_host(engine, dest, transform, source, &opts)?;
    let id = engine.create_material(&BakedMaterialDesc {
        name: name.into(),
        kind: BakedMaterialKind::Surface,
        class: MaterialClass::Opaque,
        textures: out.textures.clone(),
        parameters: BTreeMap::new(),
    })?;
    Ok((id, out.textures, out.coverage))
}

fn save_textured(scene: &Scene, slot_textures: &[(u32, &farthing::host::ChannelImages)], path: &Path) -> Result<()> {
    let mut tex = ExportTextures::default();
    for (slot, t) in slot_textures {
        if let Some(c) = t.get(channels::BASE_COLOR) {
            tex.base_color.insert(*slot, c.image.clone());
        }
        if let Some(n) = t.get(channels::NORMAL).filter(|n| n.encoding == ChannelEncoding::NormalTangent) {
            tex.normal.insert(*slot, n.image.clone());
        }
    }
    farthing::io::save_glb(scene, path, &tex)?;
    Ok(())
}

fn load_gltf_into_engine(engine: &mut Engine, path: &str) -> Result<Scene> {
    let (mut scene, mats) = farthing::io::load_gltf(path)?;
    for (slot, m) in mats.iter().enumerate() {
        let base = m.base_color_texture.as_ref().map(|t| t.0.clone()).unwrap_or_else(|| Image::new(1, 1, Vec4::ONE));
        let cutoff = match m.info.class {
            MaterialClass::Masked { cutoff } => cutoff,
            _ => 0.0,
        };
        let id = engine.add_material(MaterialDef::Textured { base, normal: None, factor: m.base_color_factor, cutoff }, m.info);
        scene.materials[slot].id = id;
    }
    Ok(scene)
}

/// `impostor_shot <asset.glb> <out-dir> [diameter-px...]`: for each on-screen diameter, builds an
/// octahedral impostor sized for it ([`farthing::impostor::frame_size_for_screen`]) and renders
/// the mesh and the impostor at exactly that size from views between capture directions. Writes
/// `d{px}_{mesh|impostor}_view{i}.png` and `impostor_shot.json` (sizes, bytes, PSNR, silhouette).
fn impostor_shot_mode(args: &[String]) -> Result<()> {
    let path = args.first().ok_or_else(|| anyhow::anyhow!("usage: impostor_shot <asset.glb> <out-dir> [diameter-px...]"))?;
    let out = PathBuf::from(args.get(1).map_or("impostor-shot", String::as_str));
    let diameters: Vec<u32> = if args.len() > 2 { args[2..].iter().filter_map(|a| a.parse().ok()).collect() } else { vec![160, 96, 48] };
    std::fs::create_dir_all(&out)?;
    let mut engine = Engine::new()?;
    // Like `load_gltf_into_engine`, plus the asset's tangent-space normal map.
    let (mut scene, mats) = farthing::io::load_gltf(path)?;
    for (slot, m) in mats.iter().enumerate() {
        let base = m.base_color_texture.as_ref().map(|t| t.0.clone()).unwrap_or_else(|| Image::new(1, 1, Vec4::ONE));
        let normal = m.normal_texture.as_ref().map(|t| t.0.clone());
        let cutoff = match m.info.class {
            MaterialClass::Masked { cutoff } => cutoff,
            _ => 0.0,
        };
        let id = engine.add_material(MaterialDef::Textured { base, normal, factor: m.base_color_factor, cutoff }, m.info);
        scene.materials[slot].id = id;
    }
    let b = scene.bounds();
    let (c, r) = (b.center(), b.radius());
    // Directions chosen to fall between the 12x12 capture directions, not on them.
    let dirs = [Vec3::new(0.55, 0.22, 1.0), Vec3::new(-0.9, 0.35, 0.5), Vec3::new(0.15, 0.6, -1.0)];
    let fov = 20f32.to_radians();
    let mut rows = Vec::new();
    for &dpx in &diameters {
        let frame = farthing::impostor::frame_size_for_screen(dpx as f32);
        let kind = farthing::impostor::ImpostorKind::Octahedral { frames_per_side: 12, hemisphere: false };
        let t = Instant::now();
        let imp = farthing::impostor::build_impostor(&mut engine, &scene, &kind, frame, true)?;
        let secs = t.elapsed().as_secs_f32();
        let mat = imp.material.unwrap();
        // Image a little larger than the sphere; distance puts the sphere at `dpx` pixels across.
        let size = (dpx as f32 * 1.15).ceil() as u32;
        let dist = size as f32 * r / (dpx as f32 * (fov * 0.5).tan());
        let mut diffs = Vec::new();
        for (i, d) in dirs.iter().enumerate() {
            let v = View::perspective(c + d.normalize() * dist, c, fov, size, size, (dist - r * 1.5).max(r * 0.01), dist + r * 1.5);
            // Reference rendered at 4x and box-filtered, like an anti-aliased engine would show it
            // (the impostor's frames are supersampled too).
            let v4 = View::perspective(c + d.normalize() * dist, c, fov, size * 4, size * 4, (dist - r * 1.5).max(r * 0.01), dist + r * 1.5);
            let mesh = engine.render(&scene, &v4, true).0.downsample_coverage(4);
            let got = engine.render_impostor(mat, imp.params.center, imp.params.radius, &v)?;
            mesh.save_png(out.join(format!("d{dpx}_mesh_view{i}.png")), true)?;
            got.save_png(out.join(format!("d{dpx}_impostor_view{i}.png")), true)?;
            diffs.push(image_difference(&mesh, &got));
        }
        let atlas = &imp.atlas[channels::BASE_COLOR].image;
        atlas.save_png(out.join(format!("d{dpx}_atlas.png")), true)?;
        // Colour + normal atlases, RGBA8 each, uncompressed.
        let bytes = atlas.width as usize * atlas.height as usize * 4 * 2;
        let row = json!({"diameter_px": dpx, "frame_px": frame, "atlas_px": atlas.width, "atlas_bytes_rgba8_color_normal": bytes,
            "triangles": scene.triangle_count(), "psnr_db": diffs.iter().map(|d| d.psnr).collect::<Vec<_>>(),
            "silhouette_mismatch": diffs.iter().map(|d| d.coverage_mismatch).collect::<Vec<_>>(), "seconds": secs});
        println!("{row}");
        rows.push(row);
    }
    std::fs::write(out.join("impostor_shot.json"), serde_json::to_string_pretty(&json!({"asset": path, "frames_per_side": 12, "fov_deg": 20, "rows": rows}))?)?;
    Ok(())
}

/// `score <original.glb> <lod.glb>...`: renders each LOD with the original's materials (matched
/// by slot index) from the standard views and prints PSNR / silhouette mismatch per file.
/// Progressive surfels as continuous LOD on the bush: each prefix of one progressive cloud
/// against a cloud generated for that count (relit PSNR against the mesh), and the image change
/// between neighbouring LOD steps (popping) for both.
fn surfel_lod_mode() -> Result<()> {
    use farthing::impostor::surfels::{progressive_surfel_cloud, surfel_cloud, surfels_to_mesh, Surfel, SurfelCloud, SurfelOptions};
    let mut engine = Engine::new()?;
    let scene = scene_gen::bush(&mut engine, 3000);
    let mesh = scene.meshes[0].clone();
    let b = scene.bounds();
    let (c, r) = (b.center(), b.radius());
    let views: Vec<View> = [Vec3::new(0.35, 0.45, 1.0), Vec3::new(-1.0, 0.25, 0.4), Vec3::new(0.6, 1.0, -0.5)]
        .iter()
        .map(|d| View::perspective(c + d.normalize() * r * 2.6, c, 40f32.to_radians(), 256, 256, r * 0.05, r * 6.0))
        .collect();
    let light = Vec3::new(0.4, 0.8, 0.3);
    let shade = |engine: &mut Engine, s: &Scene| -> Vec<Image> {
        views.iter().map(|v| { let (a, n, _) = engine.render(s, v, false); shade_gbuffer(&a, &n, light) }).collect()
    };
    let reference = shade(&mut engine, &scene);
    let vis = farthing::visibility::compute_visibility(&mut engine, &scene, &farthing::visibility::VisibilityOptions {
        cameras: farthing::visibility::CameraSet::Sphere { count: 48, distance_factor: 2.5 },
        resolution: 256,
    })?;
    let seen: Vec<f32> = vis.views_seen[0].iter().map(|&n| n as f32 / vis.view_count as f32).collect();
    let n_max = 6000;
    let opts = SurfelOptions { count: n_max, triangle_weights: Some(seen.clone()), ..Default::default() };
    let t = Instant::now();
    let cloud = progressive_surfel_cloud(&mesh, &opts)?;
    let build_s = t.elapsed().as_secs_f32();
    // `scale` multiplies each surfel's own radius; the last one shrinks by √fade.
    let discs = |list: &[Surfel], scale: f32, fade: f32| -> Scene {
        let mut v: Vec<Surfel> = list.iter().map(|s| Surfel { radius: s.radius * scale, ..*s }).collect();
        if let Some(l) = v.last_mut() {
            l.radius *= fade.sqrt();
        }
        Scene { meshes: vec![surfels_to_mesh(&mesh, &v, 6)], ..scene.clone() }
    };
    let mean = |d: &[ImageDiff]| d.iter().map(|x| x.psnr).sum::<f32>() / d.len() as f32;
    let change = |a: &[Image], b: &[Image]| -> f32 {
        let mut s = 0.0;
        let mut n = 0.0;
        for (x, y) in a.iter().zip(b) {
            for (p, q) in x.pixels.iter().zip(&y.pixels) {
                s += (p.truncate() - q.truncate()).abs().element_sum() / 3.0;
                n += 1.0;
            }
        }
        s / n
    };
    for k in [750usize, 1500, 3000, 6000] {
        let (list, rad, last) = cloud.lod(k as f32);
        let prog = shade(&mut engine, &discs(list, rad, last));
        let step = (k as f32 * 1.05).min(n_max as f32);
        let (l2, r2, s2) = cloud.lod(step);
        let prog_next = shade(&mut engine, &discs(l2, r2, s2));
        let fresh_c = surfel_cloud(&mesh, &SurfelOptions { count: k, ..opts.clone() })?;
        let fresh = shade(&mut engine, &discs(&fresh_c, 1.0, 1.0));
        let fresh_n = surfel_cloud(&mesh, &SurfelOptions { count: step as usize, ..opts.clone() })?;
        let fresh_next = shade(&mut engine, &discs(&fresh_n, 1.0, 1.0));
        let dp: Vec<ImageDiff> = reference.iter().zip(&prog).map(|(a, b)| image_difference(a, b)).collect();
        let df: Vec<ImageDiff> = reference.iter().zip(&fresh).map(|(a, b)| image_difference(a, b)).collect();
        println!("{}", json!({"surfels": k, "progressive_prefix_psnr_db": mean(&dp), "fresh_cloud_psnr_db": mean(&df),
            "step_change_progressive": change(&prog, &prog_next), "step_change_fresh_clouds": change(&fresh, &fresh_next)}));
    }
    // The reference splat shader against the disc meshes of the same surfels: coverage and
    // normals should agree (albedo is the host's material, not compared here).
    for k in [1500.0f32, 1500.5] {
        let (list, scale, fade) = cloud.lod(k);
        let inst: Vec<_> = list.iter().map(|s| (s.position.to_array(), s.normal.to_array(), s.radius, [255u8, 255, 255, 255])).collect();
        let disc_scene = discs(list, scale, fade);
        let mut cov = 0.0;
        let mut nerr = 0.0;
        for v in &views {
            let (sa, sn) = engine.render_splats(v, &inst, [scale, (list.len() - 1) as f32, fade, 0.0]);
            let (_, dn, _) = engine.render(&disc_scene, v, false);
            let d = image_difference(&dn, &sn);
            cov += d.coverage_mismatch / 3.0;
            let (mut e, mut n) = (0.0f32, 0.0f32);
            for (a, b) in sn.pixels.iter().zip(&dn.pixels) {
                if a.w > 0.5 && b.w > 0.5 {
                    e += a.truncate().normalize_or_zero().dot(b.truncate().normalize_or_zero()).clamp(-1.0, 1.0).acos().to_degrees();
                    n += 1.0;
                }
            }
            nerr += e / n.max(1.0) / 3.0;
            let _ = sa;
        }
        println!("{}", json!({"splat_shader_vs_disc_mesh": k, "coverage_mismatch": cov, "mean_normal_deg": nerr}));
    }
    let albedo = farthing::impostor::surfels::SurfelChannel { name: "albedo".into(), components: 3, data: vec![128; n_max * 3] };
    let bytes = cloud.encode(std::slice::from_ref(&albedo), false)?;
    let with_src = cloud.encode(&[], true)?;
    let t = Instant::now();
    let back = SurfelCloud::decode(&bytes)?;
    println!("{}", json!({"surfels": n_max, "build_seconds": build_s, "stream_bytes_geometry_plus_rgb": bytes.len(), "bytes_per_surfel": bytes.len() as f32 / n_max as f32,
        "stream_bytes_with_source_refs": with_src.len(), "decode_ms": t.elapsed().as_secs_f64() * 1e3, "decoded": back.cloud.surfels.len()}));
    Ok(())
}

/// One- vs two-layer octahedral impostors on concave props: mean PSNR and silhouette mismatch
/// against the mesh from three close and three distant views. The one-layer impostor is also
/// run at √2 the frame size, the same atlas memory as two layers.
fn layered_mode() -> Result<()> {
    let mut engine = Engine::new()?;
    let demo = scene_gen::demo_scene(&mut engine);
    let props = vec![
        ("vase", scene_gen::vase(&mut engine)),
        ("knot", only_instance(&demo, 0)),
        ("bush", scene_gen::bush(&mut engine, 600)),
    ];
    let kind = farthing::impostor::ImpostorKind::Octahedral { frames_per_side: 12, hemisphere: false };
    for (name, scene) in &props {
        let b = scene.bounds();
        let close = views_for(scene);
        let far: Vec<View> = [Vec3::new(0.35, 0.45, 1.0), Vec3::new(-1.0, 0.25, 0.4), Vec3::new(0.6, 1.0, -0.5)]
            .iter()
            .map(|dir| View::perspective(b.center() + dir.normalize() * b.radius() * 8.0, b.center(), 13f32.to_radians(), VIEW_SIZE, VIEW_SIZE, b.radius(), b.radius() * 12.0))
            .collect();
        let refs: Vec<Image> = close.iter().chain(&far).map(|v| engine.render(scene, v, true).0).collect();
        let variants: [(&str, u32, u8); 6] = [
            ("1 layer", 160, 0),
            ("1 layer, 2x memory", 226, 0),
            ("2 layers", 160, 1),
            ("1 layer, 1.25x memory", 179, 0),
            ("2 layers, half-res back", 160, 2),
            ("auto", 160, 3),
        ];
        for (label, frame, mode) in variants {
            let t = Instant::now();
            let imp = match mode {
                0 => farthing::impostor::build_impostor(&mut engine, scene, &kind, frame, true)?,
                1 => farthing::impostor::build_layered_impostor(&mut engine, scene, &kind, frame, true)?,
                2 => farthing::impostor::build_layered_impostor_with(&mut engine, scene, &kind, frame, true, &farthing::impostor::LayerOptions { auto: false, ..Default::default() })?,
                _ => farthing::impostor::build_layered_impostor_with(&mut engine, scene, &kind, frame, true, &Default::default())?,
            };
            let secs = t.elapsed().as_secs_f32();
            if mode == 1 {
                let h0 = &imp.atlas[farthing::impostor::HEIGHT].image;
                let h1 = &imp.atlas[&farthing::impostor::layer_channel(farthing::impostor::HEIGHT, 1)].image;
                let c0 = &imp.atlas[channels::BASE_COLOR].image;
                let c1 = &imp.atlas[&farthing::impostor::layer_channel(channels::BASE_COLOR, 1)].image;
                let (mut n0, mut n1, mut gap) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..c0.pixels.len() {
                    if c0.pixels[i].w >= 0.5 {
                        n0 += 1.0;
                        if c1.pixels[i].w >= 0.5 {
                            n1 += 1.0;
                            gap += (h0.pixels[i].x - h1.pixels[i].x) as f64;
                        }
                    }
                }
                println!("STATS {name}: layer2/layer1 coverage {:.3}, mean gap {:.3} r", n1 / n0, gap / n1.max(1.0));
            }
            let mat = imp.material.unwrap();
            let mut d = Vec::new();
            for (i, v) in close.iter().chain(&far).enumerate() {
                let got = engine.render_impostor(mat, imp.params.center, imp.params.radius, v)?;
                if i == 0 {
                    let _ = got.save_png(format!("poc-out/layered_{name}_{}.png", label.replace([' ', ','], "_")), true);
                }
                d.push(image_difference(&refs[i], &got));
            }
            let texels: u64 = imp.atlas.values().filter(|c| c.encoding == ChannelEncoding::Color).map(|c| c.image.width as u64 * c.image.height as u64).sum();
            println!(
                "{}",
                json!({"prop": name, "impostor": label, "frame_px": frame, "color_texels": texels, "layers": imp.params.layers,
                    "close_psnr_db": mean_psnr(&d[..3]), "close_silhouette": mean_mismatch(&d[..3]),
                    "far_psnr_db": mean_psnr(&d[3..]), "far_silhouette": mean_mismatch(&d[3..]), "seconds": secs})
            );
        }
    }
    Ok(())
}

fn score_mode(args: &[String]) -> Result<()> {
    let mut engine = Engine::new()?;
    let original = load_gltf_into_engine(&mut engine, &args[0])?;
    let views = views_for(&original);
    let render = |engine: &mut Engine, scene: &Scene| -> Vec<Image> { views.iter().map(|v| engine.render(scene, v, true).0).collect() };
    let reference = render(&mut engine, &original);
    let mut rows = Vec::new();
    for path in &args[1..] {
        let (mut lod, _) = farthing::io::load_gltf(path)?;
        // Same slot layout as the original (both written from the same scene structure).
        for (slot, m) in lod.materials.iter_mut().enumerate() {
            if let Some(src) = original.materials.get(slot) {
                m.id = src.id;
            }
        }
        let r = render(&mut engine, &lod);
        let d: Vec<ImageDiff> = r.iter().zip(&reference).map(|(a, b)| image_difference(b, a)).collect();
        let row = json!({"file": path, "triangles": lod.triangle_count(), "psnr_db": mean_psnr(&d), "silhouette_mismatch": mean_mismatch(&d)});
        println!("{}", serde_json::to_string(&row).unwrap());
        rows.push(row);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "relight") {
        return relight_mode(&args[1..]);
    }
    if args.first().is_some_and(|a| a == "score") {
        return score_mode(&args[1..]);
    }
    if args.first().is_some_and(|a| a == "layered") {
        return layered_mode();
    }
    if args.first().is_some_and(|a| a == "surfel_lod") {
        return surfel_lod_mode();
    }
    if args.first().is_some_and(|a| a == "impostor_shot") {
        return impostor_shot_mode(&args[1..]);
    }
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).map_or_else(|| PathBuf::from("poc-out"), PathBuf::from);
    let input = args.first().filter(|a| !a.starts_with("--")).cloned();
    std::fs::create_dir_all(&out)?;
    let t_all = Instant::now();
    let mut engine = Engine::new()?;
    println!("farthing PoC engine on adapter: {}", engine.adapter_name);
    let scene = match &input {
        Some(p) => load_gltf_into_engine(&mut engine, p)?,
        None => scene_gen::demo_scene(&mut engine),
    };
    let views = views_for(&scene);
    let mut ctx = Ctx { engine: &mut engine, out: out.clone(), views, reference: Vec::new(), report: BTreeMap::new(), sheet_rows: Vec::new() };
    ctx.record("source", json!({"triangles": scene.triangle_count(), "instances": scene.instances.len(), "materials": scene.materials.iter().map(|m| m.name.clone()).collect::<Vec<_>>()}));

    // 0. Reference renders.
    let t = Instant::now();
    ctx.reference = ctx.render_scene(&scene);
    let refs = ctx.reference.clone();
    ctx.score("original", &refs);
    ctx.record("original", json!({"seconds": t.elapsed().as_secs_f32()}));

    // 1. Near LOD: seam-preserving reduction keeps the original materials and UVs.
    let t = Instant::now();
    let mut near = scene.clone();
    for m in &mut near.meshes {
        *m = farthing::reduce::simplify(m, &farthing::reduce::SimplifyOptions::ratio(0.25))?.mesh;
    }
    let r = ctx.render_scene(&near);
    let d = ctx.score("near_lod", &r);
    ctx.record("near_lod_25pct", json!({"triangles": near.triangle_count(), "psnr_db": mean_psnr(&d), "silhouette_mismatch": mean_mismatch(&d), "seconds": t.elapsed().as_secs_f32()}));
    farthing::io::save_glb(&near, out.join("near_lod.glb"), &ExportTextures::default())?;

    // 2. Far LOD: geometry-only reduction + new UVs + bake through the engine (point evaluation).
    let t = Instant::now();
    let mut far = scene.clone();
    let mut far_textures = Vec::new();
    let mut baked_any = Vec::new();
    for (ii, inst) in scene.instances.iter().enumerate() {
        let src = &scene.meshes[inst.mesh];
        if baked_any.contains(&inst.mesh) {
            continue;
        }
        baked_any.push(inst.mesh);
        let geo = farthing::reduce::repair::geometry_only(src);
        let target = (src.triangle_count() / 20).max(12);
        let mut low = farthing::reduce::simplify(&geo, &farthing::reduce::SimplifyOptions { target_triangles: Some(target), ..Default::default() })?.mesh;
        low.compute_normals(60.0);
        let low = farthing::uv::parameterize(&low, &farthing::uv::ParamOptions { atlas_size: 512, ..Default::default() })?.mesh;
        let (id, tex, _cov) = bake_onto(ctx.engine, &low, inst.transform, &only_instance(&scene, ii), 512, &format!("far_{}", inst.name))?;
        far.materials.push(MaterialSlot { name: format!("far_{}", inst.name), id });
        let slot = far.materials.len() as u32 - 1;
        let mut low = low;
        low.material_ids.iter_mut().for_each(|m| *m = slot);
        far.meshes[inst.mesh] = low;
        far_textures.push((slot, tex));
    }
    let r = ctx.render_scene(&far);
    let d = ctx.score("far_lod", &r);
    if let Some((_, t0)) = far_textures.first() {
        let _ = t0[channels::BASE_COLOR].image.save_png(out.join("far_lod_basecolor.png"), true);
        let n = &t0[channels::NORMAL].image;
        let enc = Image { width: n.width, height: n.height, pixels: n.pixels.iter().map(|p| (p.truncate() * 0.5 + 0.5).extend(1.0)).collect() };
        let _ = enc.save_png(out.join("far_lod_normal.png"), false);
    }
    let refs: Vec<(u32, &farthing::host::ChannelImages)> = far_textures.iter().map(|(s, t)| (*s, t)).collect();
    save_textured(&far, &refs, &out.join("far_lod.glb"))?;
    ctx.record("far_lod_5pct_baked", json!({"triangles": far.triangle_count(), "psnr_db": mean_psnr(&d), "silhouette_mismatch": mean_mismatch(&d), "seconds": t.elapsed().as_secs_f32()}));

    // 3. Remeshed proxy of the whole scene, baked through the engine.
    let t = Instant::now();
    let proxy = farthing::voxel::remesh(&scene, &farthing::voxel::RemeshOptions { resolution: 128, target_triangles: Some(3000), ..Default::default() })?;
    let pm = farthing::uv::parameterize(&proxy.mesh, &farthing::uv::ParamOptions { atlas_size: 1024, ..Default::default() })?.mesh;
    let (pid, ptex, pcov) = bake_onto(ctx.engine, &pm, farthing::glam::Mat4::IDENTITY, &scene, 1024, "proxy")?;
    let mut proxy_scene = Scene::from_mesh(pm);
    proxy_scene.materials = vec![MaterialSlot { name: "proxy".into(), id: pid }];
    let r = ctx.render_scene(&proxy_scene);
    let d = ctx.score("proxy", &r);
    save_textured(&proxy_scene, &[(0, &ptex)], &out.join("proxy.glb"))?;
    ctx.record("remesh_proxy", json!({"triangles": proxy_scene.triangle_count(), "voxels": proxy.voxels, "bake_coverage": pcov, "psnr_db": mean_psnr(&d), "silhouette_mismatch": mean_mismatch(&d), "seconds": t.elapsed().as_secs_f32()}));

    // 4. Octahedral impostor of one object (how impostors are used): farthing plans, the engine
    //    captures, farthing assembles; the engine renders it with farthing's reference shader.
    let t = Instant::now();
    let knot = only_instance(&scene, 0);
    let kind = farthing::impostor::ImpostorKind::Octahedral { frames_per_side: 12, hemisphere: false };
    let imp = farthing::impostor::build_impostor(ctx.engine, &knot, &kind, 160, true)?;
    let _ = imp.atlas[channels::BASE_COLOR].image.save_png(out.join("impostor_atlas.png"), true);
    std::fs::write(out.join("impostor_params.json"), serde_json::to_string_pretty(&imp.params)?)?;
    let mat = imp.material.unwrap();
    let score_views = |ctx: &mut Ctx, views: &[View], tag: &str| -> Result<(Vec<ImageDiff>, Vec<Image>)> {
        let mut diffs = Vec::new();
        let mut imgs = Vec::new();
        for (i, v) in views.iter().enumerate() {
            let reference = ctx.engine.render(&knot, v, true).0;
            let got = ctx.engine.render_impostor(mat, imp.params.center, imp.params.radius, v)?;
            let _ = got.save_png(ctx.out.join(format!("impostor_{tag}_view{i}.png")), true);
            let _ = reference.save_png(ctx.out.join(format!("impostor_{tag}_reference{i}.png")), true);
            diffs.push(image_difference(&reference, &got));
            imgs.push(got);
        }
        Ok((diffs, imgs))
    };
    let close_views = ctx.views.clone();
    let (d, imgs) = score_views(&mut ctx, &close_views, "close")?;
    ctx.sheet_rows.push(imgs.iter().map(half).collect());
    // Impostors are for distance: also score from 8 radii with a long lens (same pixel size).
    let b = knot.bounds();
    let far_views: Vec<View> = [Vec3::new(0.35, 0.45, 1.0), Vec3::new(-1.0, 0.25, 0.4), Vec3::new(0.6, 1.0, -0.5)]
        .iter()
        .map(|dir| View::perspective(b.center() + dir.normalize() * b.radius() * 8.0, b.center(), 13f32.to_radians(), VIEW_SIZE, VIEW_SIZE, b.radius(), b.radius() * 12.0))
        .collect();
    let (fd, _) = score_views(&mut ctx, &far_views, "far")?;
    ctx.record("octahedral_impostor_knot", json!({
        "frames": 144,
        "atlas_px": imp.atlas[channels::BASE_COLOR].image.width,
        "close_psnr_db": mean_psnr(&d), "close_silhouette_mismatch": mean_mismatch(&d),
        "far_psnr_db": mean_psnr(&fd), "far_silhouette_mismatch": mean_mismatch(&fd),
        "seconds": t.elapsed().as_secs_f32()
    }));

    // 5. Aggregation: engine flattens each instance's material into its own UVs (the flatten
    //    workflow), farthing repacks the islands into one atlas and copies texels.
    let t = Instant::now();
    let mut flat = FlattenedTextures::default();
    for (ii, inst) in scene.instances.iter().enumerate() {
        let src = &scene.meshes[inst.mesh];
        if src.uvs.is_empty() {
            continue;
        }
        let opts = BakeOptions { width: 512, height: 512, supersample: 2, ..Default::default() };
        let out_tex = bake_with_host(ctx.engine, src, inst.transform, &only_instance(&scene, ii), &opts)?;
        let img = out_tex.textures[channels::BASE_COLOR].image.clone();
        let nrm = out_tex.textures[channels::NORMAL].image.clone();
        let mut slots: Vec<u32> = src.material_ids.clone();
        slots.sort();
        slots.dedup();
        for s in slots {
            flat.insert(channels::BASE_COLOR, ChannelEncoding::Color, Some(ii as u32), s, img.clone());
            flat.insert(channels::NORMAL, ChannelEncoding::NormalTangent, Some(ii as u32), s, nrm.clone());
        }
    }
    let infos: Vec<MaterialInfo> = scene.materials.iter().map(|m| ctx.engine.material_info(m.id)).collect();
    let agg = farthing::aggregate::aggregate(&scene, &infos, &farthing::aggregate::AggregateOptions { atlas_size: 1024, ..Default::default() })?;
    let atlas = farthing::aggregate::copy_texels(&agg, &flat, channels::BASE_COLOR)?;
    let _ = atlas.save_png(out.join("aggregate_atlas.png"), true);
    let mut agg_scene = agg.scene.clone();
    let mut textures = farthing::host::ChannelImages::new();
    textures.insert(channels::BASE_COLOR.into(), farthing::host::ChannelImage { image: atlas, encoding: ChannelEncoding::Color });
    let natlas = farthing::aggregate::copy_tangent_normals(&agg, &flat, channels::NORMAL)?;
    textures.insert(channels::NORMAL.into(), farthing::host::ChannelImage { image: natlas, encoding: ChannelEncoding::NormalTangent });
    let aid = ctx.engine.create_material(&BakedMaterialDesc { name: "aggregate".into(), kind: BakedMaterialKind::Surface, class: MaterialClass::Opaque, textures: textures.clone(), parameters: BTreeMap::new() })?;
    for m in &mut agg_scene.materials {
        m.id = aid;
    }
    let r = ctx.render_scene(&agg_scene);
    let d = ctx.score("aggregate", &r);
    save_textured(&agg_scene, &[(0, &textures)], &out.join("aggregate.glb"))?;
    ctx.record("aggregate", json!({"draw_calls_before": agg.draw_calls_before, "draw_calls_after": agg_scene.materials.len(), "islands": agg.placements.len(), "psnr_db": mean_psnr(&d), "seconds": t.elapsed().as_secs_f32()}));

    // 6. Visibility from the upper hemisphere: hidden and underside triangles.
    let t = Instant::now();
    let vis = farthing::visibility::compute_visibility(ctx.engine, &scene, &farthing::visibility::VisibilityOptions {
        cameras: farthing::visibility::CameraSet::Hemisphere { count: 48, distance_factor: 2.5 },
        resolution: 128,
    })?;
    let (culled, removed) = farthing::visibility::cull_invisible(&scene, &vis, 1);
    let hidden_seen = scene.instances.iter().position(|i| i.name == "hidden_cube").map(|i| vis.views_seen[i].iter().filter(|&&v| v > 0).count());
    farthing::io::save_glb(&culled, out.join("visibility_culled.glb"), &ExportTextures::default())?;
    ctx.record("visibility_hemisphere", json!({"views": vis.view_count, "visible_fraction": vis.visible_fraction(), "removed_triangles": removed, "hidden_cube_triangles_seen": hidden_seen, "seconds": t.elapsed().as_secs_f32()}));

    // 7. Occluder.
    let t = Instant::now();
    let occ = farthing::voxel::occluder(&scene, &Default::default())?;
    farthing::io::save_glb(&Scene::from_mesh(occ.mesh.clone()), out.join("occluder.glb"), &ExportTextures::default())?;
    ctx.record("occluder", json!({"triangles": occ.mesh.triangle_count(), "outside_fraction": occ.outside_fraction, "seconds": t.elapsed().as_secs_f32()}));

    // Contact sheet: rows = original, near LOD, far LOD, proxy, impostor, aggregate; cols = views.
    let cell = VIEW_SIZE / 2;
    let rows = ctx.sheet_rows.len() as u32;
    let mut sheet = Image::new(cell * 3, cell * rows, Vec4::ONE);
    for (r, row) in ctx.sheet_rows.iter().enumerate() {
        for (c, img) in row.iter().enumerate() {
            sheet.blit(img, c as u32 * cell, r as u32 * cell);
        }
    }
    sheet.save_png(out.join("contact_sheet.png"), true)?;
    ctx.record("total_seconds", json!(t_all.elapsed().as_secs_f32()));
    let report = Value::Object(ctx.report.clone().into_iter().collect());
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&report)?)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// Deferred shading on the CPU from an unlit render (albedo, coverage in alpha) and its normal
/// buffer: every representation is lit by exactly the same code for any light direction.
fn shade_gbuffer(albedo: &Image, normal: &Image, light: Vec3) -> Image {
    let l = light.normalize();
    Image {
        width: albedo.width,
        height: albedo.height,
        pixels: albedo
            .pixels
            .iter()
            .zip(&normal.pixels)
            .map(|(a, n)| {
                if a.w <= 0.0 {
                    return Vec4::ZERO;
                }
                let k = 0.2 + 0.8 * n.truncate().normalize_or_zero().dot(l).max(0.0);
                (a.truncate() * k).extend(a.w)
            })
            .collect(),
    }
}

/// `relight [leaves] [budgets...]`: a foliage-like bush against three proxies at equal primitive
/// budgets (triangle LOD, voxel proxy, surfel cloud), each relit from its G-buffer under
/// several light directions and compared with the original under the same light.
fn relight_mode(args: &[String]) -> Result<()> {
    let leaves: usize = args.first().and_then(|a| a.parse().ok()).unwrap_or(3000);
    let budgets: Vec<usize> = if args.len() > 1 { args[1..].iter().filter_map(|a| a.parse().ok()).collect() } else { vec![600, 1500, 3000] };
    let mut engine = Engine::new()?;
    let scene = scene_gen::bush(&mut engine, leaves);
    let mesh = scene.meshes[0].clone();
    // Image size per view: 512 (close) by default; RELIGHT_PX=128 for a distant LOD's size.
    let px: u32 = std::env::var("RELIGHT_PX").ok().and_then(|v| v.parse().ok()).unwrap_or(VIEW_SIZE);
    let views: Vec<View> = {
        let b = scene.bounds();
        let (c, r) = (b.center(), b.radius());
        [Vec3::new(0.35, 0.45, 1.0), Vec3::new(-1.0, 0.25, 0.4), Vec3::new(0.6, 1.0, -0.5)]
            .iter()
            .map(|d| View::perspective(c + d.normalize() * r * 2.6, c, 40f32.to_radians(), px, px, r * 0.05, r * 6.0))
            .collect()
    };
    let lights = [Vec3::new(0.4, 0.8, 0.3), Vec3::new(-0.8, 0.5, 0.2), Vec3::new(0.1, 0.3, -0.9), Vec3::new(0.0, 1.0, 0.0)];
    let gbuf = |engine: &mut Engine, s: &Scene| -> Vec<(Image, Image)> {
        views.iter().map(|v| { let (c, n, _) = engine.render(s, v, false); (c, n) }).collect()
    };
    let reference = gbuf(&mut engine, &scene);
    let out = PathBuf::from(format!("poc-out/relight{}", if px == VIEW_SIZE { String::new() } else { format!("_{px}") }));
    std::fs::create_dir_all(&out)?;
    let score = |name: &str, g: &[(Image, Image)]| -> (f32, f32) {
        let mut psnr = 0.0;
        let mut sil = 0.0;
        let mut n = 0.0;
        for (li, l) in lights.iter().enumerate() {
            for (vi, ((ra, rn), (ca, cn))) in reference.iter().zip(g).enumerate() {
                let r = shade_gbuffer(ra, rn, *l);
                let c = shade_gbuffer(ca, cn, *l);
                let d = image_difference(&r, &c);
                psnr += d.psnr;
                sil += d.coverage_mismatch;
                n += 1.0;
                if vi == 0 {
                    let _ = c.save_png(out.join(format!("{name}_light{li}.png")), true);
                    if name.starts_with("lod") {
                        let _ = r.save_png(out.join(format!("original_light{li}.png")), true);
                    }
                }
            }
        }
        (psnr / n, sil / n)
    };
    let with_mesh = |m: Mesh| Scene { meshes: vec![m], ..scene.clone() };
    // Impostors at fixed frame sizes, for the memory an impostor needs to compete.
    for frame in [16u32, 32, 64] {
        let kind = farthing::impostor::ImpostorKind::Octahedral { frames_per_side: 12, hemisphere: false };
        let imp = farthing::impostor::build_impostor(&mut engine, &scene, &kind, frame, true)?;
        let mat = imp.material.unwrap();
        let g: Vec<(Image, Image)> = views
            .iter()
            .map(|v| -> Result<(Image, Image)> {
                Ok((
                    engine.render_impostor_pass(mat, imp.params.center, imp.params.radius, v, 1)?,
                    engine.render_impostor_pass(mat, imp.params.center, imp.params.radius, v, 2)?,
                ))
            })
            .collect::<Result<_>>()?;
        let (p, sil) = score(&format!("impostor_frame{frame}"), &g);
        let bytes = (frame * 12) * (frame * 12) * 8;
        println!("{}", json!({"budget": 0, "method": "impostor_fixed", "primitives": 0, "frame_px": frame, "atlas_px": frame * 12, "bytes": bytes, "psnr_db": p, "silhouette_mismatch": sil, "seconds": 0.0, "proxy_resolution": 0}));
    }
    println!("bush: {} leaves, {} triangles, {} parts", leaves, mesh.triangle_count(), leaves + 1);
    // Fraction of views around the bush that see each triangle: hidden interior leaves get no
    // surfels.
    let vis = farthing::visibility::compute_visibility(&mut engine, &scene, &farthing::visibility::VisibilityOptions {
        cameras: farthing::visibility::CameraSet::Sphere { count: 48, distance_factor: 2.5 },
        resolution: 256,
    })?;
    let seen: Vec<f32> = vis.views_seen[0].iter().map(|&n| n as f32 / vis.view_count as f32).collect();
    println!("visible triangles: {:.1}%", 100.0 * seen.iter().filter(|&&s| s > 0.0).count() as f32 / seen.len() as f32);
    for &budget in &budgets {
        let t = Instant::now();
        let lod = farthing::reduce::simplify(&mesh, &farthing::reduce::SimplifyOptions { target_triangles: Some(budget), ..Default::default() })?.mesh;
        let lod_s = t.elapsed().as_secs_f32();
        let t = Instant::now();
        // A voxel proxy of foliage is a sponge whose handles collapses cannot remove: lower the
        // resolution until the budget is met (or the floor of 16 cells is reached).
        let mut res = 128;
        let mut proxy = farthing::voxel::remesh(&scene, &farthing::voxel::RemeshOptions { resolution: res, target_triangles: Some(budget), ..Default::default() })?.mesh;
        while proxy.triangle_count() > budget * 11 / 10 && res > 16 {
            res = res * 3 / 4;
            proxy = farthing::voxel::remesh(&scene, &farthing::voxel::RemeshOptions { resolution: res, target_triangles: Some(budget), ..Default::default() })?.mesh;
        }
        let proxy_s = t.elapsed().as_secs_f32();
        let t = Instant::now();
        let surfels = farthing::impostor::surfels::surfel_cloud(&mesh, &farthing::impostor::surfels::SurfelOptions { count: budget, ..Default::default() })?;
        let discs = farthing::impostor::surfels::surfels_to_mesh(&mesh, &surfels, 6);
        let surfel_s = t.elapsed().as_secs_f32();
        let t = Instant::now();
        let vsurfels = farthing::impostor::surfels::surfel_cloud(&mesh, &farthing::impostor::surfels::SurfelOptions { count: budget, triangle_weights: Some(seen.clone()), ..Default::default() })?;
        let vdiscs = farthing::impostor::surfels::surfels_to_mesh(&mesh, &vsurfels, 6);
        let vsurfel_s = t.elapsed().as_secs_f32();
        // The voxel proxy has one material slot of its own: give it the leaf material.
        proxy.material_ids.iter_mut().for_each(|m| *m = 0);
        // Octahedral impostor with about the budget's memory: a surfel is ~24 bytes and an atlas
        // texel 8 (albedo + normal, 4 bytes each), so 3 texels per primitive; 12×12 frames.
        let frame = (((budget * 3) as f32 / 144.0).sqrt().floor() as u32).max(4);
        let t = Instant::now();
        let kind = farthing::impostor::ImpostorKind::Octahedral { frames_per_side: 12, hemisphere: false };
        let imp = farthing::impostor::build_impostor(&mut engine, &scene, &kind, frame, true)?;
        let imp_s = t.elapsed().as_secs_f32();
        let mat = imp.material.unwrap();
        let g: Vec<(Image, Image)> = views
            .iter()
            .map(|v| -> Result<(Image, Image)> {
                Ok((
                    engine.render_impostor_pass(mat, imp.params.center, imp.params.radius, v, 1)?,
                    engine.render_impostor_pass(mat, imp.params.center, imp.params.radius, v, 2)?,
                ))
            })
            .collect::<Result<_>>()?;
        let (p, sil) = score(&format!("impostor_{budget}"), &g);
        println!("{}", json!({"budget": budget, "method": "impostor", "primitives": 0, "frame_px": frame, "atlas_px": frame * 12, "psnr_db": p, "silhouette_mismatch": sil, "seconds": imp_s, "proxy_resolution": 0}));
        for (name, m, secs) in [("lod", lod, lod_s), ("proxy", proxy, proxy_s), ("surfels", discs, surfel_s), ("surfels_visible", vdiscs, vsurfel_s)] {
            let prims = if name.starts_with("surfels") { budget } else { m.triangle_count() };
            let g = gbuf(&mut engine, &with_mesh(m));
            let (p, s) = score(&format!("{name}_{budget}"), &g);
            println!("{}", json!({"budget": budget, "method": name, "primitives": prims, "psnr_db": p, "silhouette_mismatch": s, "seconds": secs, "proxy_resolution": if name == "proxy" { res } else { 0 }}));
        }
    }
    Ok(())
}
