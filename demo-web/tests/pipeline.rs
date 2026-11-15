//! The browser pipeline, run natively: a zipped `.gltf` + `.bin` with a textured sphere,
//! through reduced and baked levels.

use farthing::image::Image;
use farthing::io::{glb_bytes, load_gltf_slice, ExportTextures};
use farthing::{Mesh, Scene};
use farthing_demo_web::modes::ViewRequest;
use farthing_demo_web::pipeline::{
    Buffer, Demo, DemoMode, Event, LevelInfo, LevelKind, LodOptions, TextureMode,
};
use glam::{Vec2, Vec3, Vec4};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Counts live and peak heap bytes, so the sample test can hold the pipeline to the wasm32
/// address space: natively a 64-bit runner with 16 GB happily ran a 7 GB peak that trapped
/// ("unreachable") in the browser.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

/// Peak heap a sample may reach. wasm32 linear memory stops at 4 GiB; the samples peak near
/// 430 MB (FlightHelmet), so 1 GiB catches a regression long before a browser would trap.
const WASM_HEAP_BUDGET: usize = 1 << 30;

fn textured_sphere() -> Vec<u8> {
    let (seg, rings) = (48u32, 24u32);
    let (mut pos, mut nrm, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let th = v * std::f32::consts::PI;
        for s in 0..=seg {
            let u = s as f32 / seg as f32;
            let ph = u * std::f32::consts::TAU;
            let p = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
            pos.push(p);
            nrm.push(p);
            uv.push(Vec2::new(u, v));
        }
    }
    for r in 0..rings {
        for s in 0..seg {
            let a = r * (seg + 1) + s;
            let b = a + seg + 1;
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    let mut m = Mesh::new(pos, idx);
    m.normals = Some(nrm);
    m.uvs = vec![uv];
    // Checker texture: red / white, so the bake has something to carry.
    let mut tex = Image::new(64, 64, Vec4::ONE);
    for y in 0..64 {
        for x in 0..64 {
            if (x / 8 + y / 8) % 2 == 0 {
                tex.pixels[y * 64 + x] = Vec4::new(1.0, 0.0, 0.0, 1.0);
            }
        }
    }
    let mut t = ExportTextures::default();
    t.base_color
        .insert(0, farthing::image::Texture::from_image(&tex, true));
    glb_bytes(&Scene::from_mesh(m), &t).unwrap()
}

/// GLB → (`.gltf` JSON referencing `bin_uri`, `.bin` bytes).
fn split_glb(glb: &[u8], bin_uri: &str) -> (Vec<u8>, Vec<u8>) {
    let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let mut doc: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
    doc["buffers"][0]["uri"] = bin_uri.into();
    (
        serde_json::to_vec(&doc).unwrap(),
        glb[20 + json_len + 8..].to_vec(),
    )
}

fn zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, data) in files {
        w.start_file(*name, opts).unwrap();
        w.write_all(data).unwrap();
    }
    w.finish().unwrap().into_inner()
}

#[test]
fn zipped_gltf_reduces_and_bakes() {
    let (gltf, bin) = split_glb(&textured_sphere(), "buffers/sphere%20data.bin");
    let archive = zip(&[
        ("model/scene.gltf", &gltf),
        ("model/buffers/sphere data.bin", &bin),
        ("__MACOSX/model/._scene.gltf", b"junk"),
    ]);
    let mut demo = Demo::load(vec![("upload.zip".into(), archive)]).unwrap();
    let info = demo.info();
    assert_eq!(info.name, "scene.gltf");
    assert_eq!(info.triangles, 48 * 24 * 2);
    assert_eq!(info.textured_materials, 1);
    load_gltf_slice(&demo.original_glb().unwrap()).unwrap();

    let opts = LodOptions {
        levels: 3,
        bake_from: 3,
        atlas: 64,
        textures: TextureMode::Original,
        ..Default::default()
    };
    let (mut levels, mut textures, mut stages) = (vec![], vec![], 0);
    let t0 = std::time::Instant::now();
    demo.run(
        &opts,
        &|| t0.elapsed().as_secs_f64() * 1e3,
        &mut |e| match e {
            Event::Stage(_) => stages += 1,
            Event::Level { info, glb } => levels.push((info.clone(), glb.to_vec())),
            Event::Texture {
                channel,
                width,
                height,
                rgba,
                ..
            } => {
                assert_eq!(rgba.len(), (width * height * 4) as usize);
                textures.push((channel.to_owned(), rgba.to_vec()));
            }
            Event::Data { .. } => panic!("the chain reports levels, not data"),
        },
    )
    .unwrap();

    assert!(stages > 3);
    assert_eq!(levels.len(), 4, "LOD0..LOD3");
    assert!(levels[0].1.is_empty());
    for w in levels.windows(2) {
        assert!(w[1].0.triangles < w[0].0.triangles, "{:?}", w[1].0);
    }
    for (info, glb) in &levels[1..] {
        let (s, _) = load_gltf_slice(glb).unwrap();
        assert_eq!(s.triangle_count(), info.triangles);
        assert!(info.error > 0.0 && info.relative_error < 0.2, "{info:?}");
    }
    check_measured(&levels, demo.info().diagonal);
    let baked = &levels[3].0;
    assert!(baked.coverage.unwrap() > 0.2, "{baked:?}");
    let names: Vec<&str> = textures.iter().map(|t| t.0.as_str()).collect();
    assert_eq!(names, ["base_color", "normal", "orm"]);
    // The checker survives: the baked base color has both red and white texels.
    let px = textures[0].1.as_chunks::<4>().0.iter();
    let red = px.clone().filter(|p| p[0] > 200 && p[1] < 60).count();
    let white = px.filter(|p| p[0] > 200 && p[1] > 200).count();
    assert!(red > 100 && white > 100, "red {red}, white {white}");

    // The downloadable baked level embeds its textures.
    let full = demo.level_glb(3, false).unwrap();
    assert!(full.windows(9).any(|w| w == b"image/png"));
    assert!(full.windows(24).any(|w| w == b"metallicRoughnessTexture"));
}

/// Measured levels (all but LOD0) report a mean and RMS distance between zero and the error,
/// a normal deviation unless baked, and switch where the error is a pixel even on the side of
/// the model nearest the camera: the error at the reference scale plus half the diagonal.
fn check_measured(levels: &[(LevelInfo, Vec<u8>)], diagonal: f32) {
    let ppr = 1080.0 / (2.0 * 30f32.to_radians().tan());
    let original = &levels[0].0;
    assert_eq!(original.switch_distance_1px, 0.0);
    assert!(original.mean_error.is_none() && original.normal_error.is_none());
    for (info, _) in &levels[1..] {
        let switch = info.error * ppr + 0.5 * diagonal;
        assert!(
            (info.switch_distance_1px - switch).abs() <= 1e-4 * switch,
            "{info:?}"
        );
        let (mean, rms) = (info.mean_error.unwrap(), info.rms_error.unwrap());
        assert!(0.0 < mean && mean <= rms && rms <= info.error, "{info:?}");
        let baked = matches!(info.kind, LevelKind::Baked);
        assert_eq!(info.normal_error.is_none(), baked, "{info:?}");
        if let Some(n) = info.normal_error {
            assert!((0.0..30.0).contains(&n), "{info:?}");
        }
    }
}

#[test]
fn shared_atlas_carries_the_original_materials() {
    // Keeping the artist's UV layout (the default) and unwrapping afresh.
    let mut charts = vec![];
    for keep in [true, false] {
        let mut demo = Demo::load(vec![("sphere.glb".into(), textured_sphere())]).unwrap();
        let opts = LodOptions {
            levels: 3,
            bake_from: 3,
            atlas: 64,
            textures: TextureMode::Shared,
            shared_size: 512,
            shared_keep_layout: keep,
            ..Default::default()
        };
        let (mut levels, mut textures) = (vec![], vec![]);
        let t0 = std::time::Instant::now();
        demo.run(
            &opts,
            &|| t0.elapsed().as_secs_f64() * 1e3,
            &mut |e| match e {
                Event::Level { info, glb } => levels.push((info.clone(), glb.to_vec())),
                Event::Texture {
                    level,
                    channel,
                    width,
                    height,
                    rgba,
                } => textures.push((level, channel.to_owned(), width, height, rgba.to_vec())),
                _ => {}
            },
        )
        .unwrap();

        let kinds: Vec<String> = levels.iter().map(|l| format!("{:?}", l.0.kind)).collect();
        assert_eq!(kinds, ["Original", "Shared", "Shared", "Baked"]);
        let (one, two) = (&levels[1].0, &levels[2].0);
        assert!(two.triangles < one.triangles);
        for info in [one, two] {
            assert_eq!(info.shared_textures, Some(1));
            assert!(info.relative_error < 0.05, "{info:?}");
            let d = info.density_scale.unwrap();
            assert!(d > 0.0 && d <= 1.0, "{info:?}");
            assert!(info.coverage.unwrap() > 0.2, "{info:?}");
        }
        check_measured(&levels, demo.info().diagonal);
        let (w, h) = (one.atlas_width.unwrap(), one.atlas_height.unwrap());
        assert!(w <= 512 && h <= 512 && w.is_power_of_two(), "{w}×{h}");
        // Three textures (base color, normal, ORM) with mips.
        assert_eq!(one.texture_bytes, Some(3 * ((w * h) as usize * 16 / 3)));

        // Level 1's GLB carries the atlas; level 2 reuses it, but downloads embed it.
        assert!(levels[1].1.windows(9).any(|w| w == b"image/png"));
        assert!(!levels[2].1.windows(9).any(|w| w == b"image/png"));
        assert!(demo
            .level_glb(2, false)
            .unwrap()
            .windows(9)
            .any(|w| w == b"image/png"));

        // The atlas comes from level 1 and keeps the checker.
        let shared: Vec<_> = textures.iter().filter(|t| t.0 == 1).collect();
        let names: Vec<&str> = shared.iter().map(|t| t.1.as_str()).collect();
        assert_eq!(names, ["base_color", "normal", "orm"]);
        assert_eq!((shared[0].2, shared[0].3), (w, h));
        let px = shared[0].4.as_chunks::<4>().0.iter();
        let red = px.clone().filter(|p| p[0] > 200 && p[1] < 60).count();
        let white = px.filter(|p| p[0] > 200 && p[1] > 200).count();
        assert!(red > 1000 && white > 1000, "red {red}, white {white}");
        assert!(textures.iter().all(|t| t.0 == 1 || t.0 == 3));
        charts.push(one.charts.unwrap());
    }
    // The sphere's UV islands are kept whole; unwrapped afresh it is cut into more charts.
    assert!(charts[0] < charts[1], "{charts:?}");
}

#[test]
fn recut_keeps_every_triangle_then_reduces() {
    let mut demo = Demo::load(vec![("sphere.glb".into(), textured_sphere())]).unwrap();
    let opts = LodOptions {
        levels: 3,
        shared_size: 512,
        ..Default::default()
    };
    assert_eq!(opts.textures, TextureMode::Recut);
    let (mut levels, mut textures) = (vec![], vec![]);
    demo.run(&opts, &|| 0.0, &mut |e| match e {
        Event::Level { info, glb } => levels.push((info.clone(), glb.to_vec())),
        Event::Texture { level, .. } => textures.push(level),
        _ => {}
    })
    .unwrap();

    // Original, LOD0 and three reductions, all drawn with LOD0's atlas: nothing else baked.
    let names: Vec<String> = levels[1..]
        .iter()
        .map(|l| l.0.label.clone().unwrap())
        .collect();
    assert_eq!(names, ["LOD0", "LOD1", "LOD2", "LOD3"]);
    assert!(levels[1..]
        .iter()
        .all(|l| matches!(l.0.kind, LevelKind::Shared) && l.0.shared_textures == Some(1)));
    assert!(textures.iter().all(|&l| l == 1), "{textures:?}");
    let lod0 = &levels[1].0;
    assert_eq!(lod0.triangles, demo.info().triangles);
    assert!(lod0.relative_error < 1e-5, "{lod0:?}");
    for w in levels[1..].windows(2) {
        let ratio = w[1].0.triangles as f32 / w[0].0.triangles as f32;
        assert!(ratio > 0.4 && ratio < 0.6, "{:?}", w[1].0);
    }
    let reduced: Vec<_> = levels[..1].iter().chain(&levels[2..]).cloned().collect();
    check_measured(&reduced, demo.info().diagonal);

    // The checker survives resampling at the same density, but its hard edges blur: about
    // 8 levels in 255 on average, and up to 100 on the edges themselves.
    let (mean, p99) = (lod0.texture_error.unwrap(), lod0.texture_error_p99.unwrap());
    assert!(mean > 0.0 && mean < 12.0 && p99 < 160.0, "{lod0:?}");
    assert!(levels[2..].iter().all(|l| l.0.texture_error.is_none()));
    assert!(levels[1].1.windows(9).any(|w| w == b"image/png"));
    assert!(!levels[2].1.windows(9).any(|w| w == b"image/png"));

    // A LOD0 error of 2% of the diagonal: LOD0 drops triangles until it deviates about that,
    // and texels finer than it (the checker's 18 a unit, against 14 at 2%) are dropped too.
    let diag = demo.info().diagonal;
    let (coarse, _) = chain_levels(
        &mut demo,
        &LodOptions {
            lod0_error: 0.02,
            ..opts.clone()
        },
    );
    let (full, lod0) = (&levels[1].0, &coarse[1]);
    assert!(lod0.triangles < full.triangles, "{lod0:?}");
    assert!(
        lod0.relative_error > 0.002 && lod0.relative_error < 0.025,
        "{lod0:?}"
    );
    assert!(
        lod0.note.as_deref().unwrap().starts_with("LOD0 error"),
        "{lod0:?}"
    );
    assert!((lod0.view_texel_density.unwrap() * 0.02 * diag - 1.0).abs() < 1e-4);
    let area = |l: &LevelInfo| l.atlas_width.unwrap() * l.atlas_height.unwrap();
    assert!(area(lod0) < area(full), "{lod0:?} vs {full:?}");
    let ratio = coarse[2].triangles as f32 / lod0.triangles as f32;
    assert!(ratio > 0.4 && ratio < 0.6, "{:?}", coarse[2]);
}

/// Runs the chain and returns each level's info and the size of each texture it emitted.
fn chain_levels(demo: &mut Demo, opts: &LodOptions) -> (Vec<LevelInfo>, Vec<(usize, u32, u32)>) {
    let (mut levels, mut textures) = (vec![], vec![]);
    demo.run(opts, &|| 0.0, &mut |e| match e {
        Event::Level { info, .. } => levels.push(info.clone()),
        Event::Texture {
            level,
            width,
            height,
            ..
        } => textures.push((level, width, height)),
        _ => {}
    })
    .unwrap();
    (levels, textures)
}

#[test]
fn viewing_budget_drops_what_the_view_cannot_show() {
    let mut demo = Demo::load(vec![("sphere.glb".into(), textured_sphere())]).unwrap();
    let full = LodOptions {
        levels: 3,
        bake_from: 3,
        atlas: 256,
        textures: TextureMode::Original,
        ..Default::default()
    };
    // The unit sphere seen from 200 units at 1080p, 60°: a pixel is 0.21 units, and the
    // 64 × 64 checker (about 18 texels a unit) is far denser than the view's 4.7.
    let far = LodOptions {
        view_distance: 200.0,
        ..full.clone()
    };
    let (a, a_tex) = chain_levels(&mut demo, &full);
    let (b, b_tex) = chain_levels(&mut demo, &far);
    let pixel = 200.0 / b[0].ppr;
    assert!((b[0].ppr - a[0].ppr).abs() < 1e-3);

    // The first level goes past one step, to about a pixel at the closest view.
    assert!(b[1].triangles < a[1].triangles, "{:?} vs {:?}", b[1], a[1]);
    assert!(b[1].error / pixel < 4.0, "{:?}", b[1]);
    assert!(b[1].note.as_deref().unwrap().starts_with("Visible limit"));
    assert!(a
        .iter()
        .all(|l| l.note.is_none() && l.view_texels.is_none()));
    for w in b.windows(2) {
        assert!(w[1].triangles < w[0].triangles, "{:?}", w[1]);
    }
    // The original shows a small part of its texels from there.
    let kept = b[0].view_texels.unwrap();
    assert!(kept > 0.0 && kept < 0.2, "{kept}");

    // The baked level's atlas is sized for where it appears, not the 256 asked for.
    let baked = |tex: &[(usize, u32, u32)]| *tex.iter().find(|t| t.0 == 3).unwrap();
    assert_eq!(baked(&a_tex), (3, 256, 256));
    let (_, w, h) = baked(&b_tex);
    assert!(w < 256 && h < 256, "{w}×{h}");
    assert_eq!((b[3].atlas_width, b[3].atlas_height), (Some(w), Some(h)));
    assert!(b[3].coverage.unwrap() > 0.2, "{:?}", b[3]);

    // The shared atlas is capped at the view's density.
    let shared = |o: &LodOptions| LodOptions {
        textures: TextureMode::Shared,
        shared_size: 512,
        ..o.clone()
    };
    let (sa, _) = chain_levels(&mut demo, &shared(&full));
    let (sb, _) = chain_levels(&mut demo, &shared(&far));
    let area = |l: &LevelInfo| l.atlas_width.unwrap() * l.atlas_height.unwrap();
    assert!(area(&sb[1]) < area(&sa[1]), "{:?} vs {:?}", sb[1], sa[1]);
    assert!((sb[1].view_texel_density.unwrap() - 1.0 / pixel).abs() < 1e-3);
    assert!(sb[1].view_texels.unwrap() < 0.2);
    assert!(sb[1].triangles < sa[1].triangles);

    // Switch distances are for the view's screen: twice the pixels, twice the distance (past
    // the half diagonal they all add, for the model's near side).
    let (c, _) = chain_levels(
        &mut demo,
        &LodOptions {
            view_height: 2160,
            ..full.clone()
        },
    );
    assert!((c[1].ppr / a[1].ppr - 2.0).abs() < 1e-4);
    let radius = c[1].switch_distance_1px - c[1].error * c[1].ppr;
    assert!(radius > 1.0 && radius < 2.0, "{radius}");
    for l in a[1..].iter().chain(&c[1..]) {
        let switch = l.error * l.ppr + radius;
        assert!(
            (l.switch_distance_1px - switch).abs() <= 1e-4 * switch,
            "{l:?}"
        );
    }
}

#[test]
fn reports_missing_files() {
    let (gltf, _) = split_glb(&textured_sphere(), "missing.bin");
    let e = Demo::load(vec![("a.gltf".into(), gltf)]).err().unwrap();
    assert!(e.contains("missing.bin"), "{e}");
    let e = Demo::load(vec![("notes.txt".into(), b"hi".to_vec())])
        .err()
        .unwrap();
    assert!(e.contains("no .glb or .gltf"), "{e}");
}

/// The sample models the page offers (`python fetch_samples.py` downloads them into
/// web/samples/): each loads and runs through reduced and baked levels with the page's
/// default options. Skipped when they haven't been downloaded; CI fetches them first.
#[test]
fn samples_reduce_and_bake() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web/samples");
    let Ok(index) = std::fs::read(dir.join("index.json")) else {
        eprintln!("no web/samples/index.json; run fetch_samples.py to test the samples");
        return;
    };
    let index: Vec<serde_json::Value> = serde_json::from_slice(&index).unwrap();
    assert!(!index.is_empty());
    for sample in &index {
        let file = sample["file"].as_str().unwrap();
        let t0 = std::time::Instant::now();
        let bytes = std::fs::read(dir.join(file)).unwrap();
        PEAK.store(LIVE.load(Relaxed), Relaxed);
        let mut demo =
            Demo::load(vec![(file.into(), bytes)]).unwrap_or_else(|e| panic!("{file}: {e}"));
        let info = demo.info();
        assert!(info.triangles > 1000, "{file}: {info:?}");
        let opts = LodOptions {
            bake_from: 3,
            atlas: 256,
            textures: TextureMode::Original,
            ..Default::default()
        };
        let mut levels = vec![];
        demo.run(&opts, &|| 0.0, &mut |e| {
            if let Event::Level { info, .. } = e {
                levels.push(info.clone());
            }
        })
        .unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(levels.len(), opts.levels + 1, "{file}");
        for w in levels.windows(2) {
            assert!(w[1].triangles < w[0].triangles, "{file}: {:?}", w[1]);
        }
        // The bake finds the surface under (nearly) every charted texel. How much of the atlas
        // the charts fill depends on how many there are: FlightHelmet's LOD3 has about 800 in
        // 256 × 256.
        let baked = &levels[opts.bake_from];
        let (c, u) = (baked.coverage.unwrap(), baked.utilization.unwrap());
        assert!(c > 0.9 * u && u > 0.15, "{file}: {baked:?}");
        // What the page calls: the original for the viewer, then each level's download.
        demo.original_glb()
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        for l in 1..=opts.levels {
            demo.level_glb(l, false)
                .unwrap_or_else(|e| panic!("{file}: {e}"));
        }
        let peak = PEAK.load(Relaxed);
        eprintln!(
            "{file}: {} tris, LOD{} {} tris, {:.1}s, peak heap {} MB",
            info.triangles,
            opts.levels,
            levels[opts.levels].triangles,
            t0.elapsed().as_secs_f64(),
            peak >> 20
        );
        assert!(
            peak < WASM_HEAP_BUDGET,
            "{file}: peak heap {} MB would not fit wasm32",
            peak >> 20
        );

        // The shared atlas at its largest size, held to the same budget.
        let t0 = std::time::Instant::now();
        PEAK.store(LIVE.load(Relaxed), Relaxed);
        let shared = LodOptions {
            textures: TextureMode::Shared,
            ..opts.clone()
        };
        let mut levels = vec![];
        demo.run(&shared, &|| 0.0, &mut |e| {
            if let Event::Level { info, .. } = e {
                levels.push(info.clone());
            }
        })
        .unwrap_or_else(|e| panic!("{file}: {e}"));
        let peak = PEAK.load(Relaxed);
        let atlas = &levels[1];
        eprintln!(
            "{file}: shared atlas {}×{} at {:.0}% of the artist's density, {:.1}s, peak heap {} MB",
            atlas.atlas_width.unwrap(),
            atlas.atlas_height.unwrap(),
            atlas.density_scale.unwrap() * 100.0,
            t0.elapsed().as_secs_f64(),
            peak >> 20
        );
        assert!(
            peak < WASM_HEAP_BUDGET,
            "{file}: shared peak heap {} MB would not fit wasm32",
            peak >> 20
        );
        assert_eq!(levels.len(), opts.levels + 1, "{file}");
        for info in &levels[1..opts.bake_from] {
            assert_eq!(format!("{:?}", info.kind), "Shared", "{file}");
            assert!(info.coverage.unwrap() > 0.2, "{file}: {info:?}");
            assert!(info.density_scale.unwrap() > 0.0, "{file}: {info:?}");
        }
        for w in levels.windows(2) {
            assert!(w[1].triangles < w[0].triangles, "{file}: {:?}", w[1]);
        }

        // The recut chain (the page's default), held to the same budget.
        let t0 = std::time::Instant::now();
        PEAK.store(LIVE.load(Relaxed), Relaxed);
        let recut = LodOptions::default();
        let mut levels = vec![];
        demo.run(&recut, &|| 0.0, &mut |e| {
            if let Event::Level { info, .. } = e {
                levels.push(info.clone());
            }
        })
        .unwrap_or_else(|e| panic!("{file}: {e}"));
        let peak = PEAK.load(Relaxed);
        let lod0 = &levels[1];
        let last = levels.last().unwrap();
        eprintln!(
            "{file}: recut {}×{} at {:.0}% of the artist's density, {} charts, texture error mean {:.1} p99 {:.0}, {} {} tris ({:.1}%) at {:.4} relative error, {:.1}s, peak heap {} MB",
            lod0.atlas_width.unwrap(),
            lod0.atlas_height.unwrap(),
            lod0.density_scale.unwrap() * 100.0,
            lod0.charts.unwrap(),
            lod0.texture_error.unwrap_or(0.0),
            lod0.texture_error_p99.unwrap_or(0.0),
            last.label.as_deref().unwrap(),
            last.triangles,
            100.0 * last.triangles as f32 / info.triangles as f32,
            last.relative_error,
            t0.elapsed().as_secs_f64(),
            peak >> 20
        );
        assert!(
            peak < WASM_HEAP_BUDGET,
            "{file}: recut peak heap {} MB would not fit wasm32",
            peak >> 20
        );
        assert_eq!(levels.len(), recut.levels + 2, "{file}");
        assert_eq!(lod0.triangles, info.triangles, "{file}");
        for w in levels[1..].windows(2) {
            assert!(w[1].triangles < w[0].triangles, "{file}: {:?}", w[1]);
        }

        // The other modes with the page's defaults, each held to the same budget.
        for mode in [
            DemoMode::Presets,
            DemoMode::Clusters,
            DemoMode::Hlod,
            DemoMode::Splats,
        ] {
            let t0 = std::time::Instant::now();
            PEAK.store(LIVE.load(Relaxed), Relaxed);
            let run = run_mode(&mut demo, mode, LodOptions::default());
            let peak = PEAK.load(Relaxed);
            eprintln!(
                "{file}: {mode:?} {} results, {:.1}s, peak heap {} MB",
                run.levels.len() + run.data.len(),
                t0.elapsed().as_secs_f64(),
                peak >> 20
            );
            assert!(
                peak < WASM_HEAP_BUDGET,
                "{file}: {mode:?} peak heap {} MB would not fit wasm32",
                peak >> 20
            );
            check_mode(&demo, mode, &run, info.diagonal);
        }
    }
}

/// What a run reported.
#[derive(Default)]
struct Run {
    levels: Vec<(LevelInfo, usize)>,
    data: Vec<(String, serde_json::Value, Vec<Buffer>)>,
}

fn run_mode(demo: &mut Demo, mode: DemoMode, opts: LodOptions) -> Run {
    let mut run = Run::default();
    demo.run(&LodOptions { mode, ..opts }, &|| 0.0, &mut |e| match e {
        Event::Level { info, glb } => run.levels.push((info.clone(), glb.len())),
        Event::Data {
            kind,
            json,
            buffers,
        } => run.data.push((
            kind.to_owned(),
            serde_json::from_str(&json).unwrap(),
            buffers,
        )),
        _ => {}
    })
    .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
    run
}

fn u32s(b: &Buffer) -> &[u32] {
    match b {
        Buffer::U32(v) => v,
        _ => panic!("expected u32s"),
    }
}

/// What every mode must report, whatever the asset.
fn check_mode(demo: &Demo, mode: DemoMode, run: &Run, diagonal: f32) {
    let original = run.levels[0].0.triangles;
    match mode {
        DemoMode::Chain => {}
        DemoMode::Presets => {
            assert_eq!(run.levels.len(), 11, "original + 10 configurations");
            for (info, glb) in &run.levels[1..] {
                assert!(info.label.is_some() && info.note.is_some(), "{info:?}");
                assert!(info.triangles < original, "{info:?}");
                assert_eq!(info.bytes, Some(*glb));
                assert!(info.relative_error < 0.5, "{info:?}");
                demo.level_glb(info.level, false).unwrap();
            }
        }
        DemoMode::Clusters => {
            let (kind, json, buffers) = &run.data[0];
            assert_eq!(kind, "clusters");
            let meshes = json["meshes"].as_array().unwrap();
            assert_eq!(buffers.len(), 2 + 7 * meshes.len());
            assert!(json["gpu_bytes"].as_u64().unwrap() > 0);
            // Close up shows full detail, far away a coarser cut; the packed GPU form picks
            // the same clusters.
            let view = |d: f32, threshold| ViewRequest {
                eye: [0.0, 0.0, d * diagonal],
                ppr: 935.0,
                threshold,
            };
            let cut = |v: &ViewRequest| {
                let (j, b) = demo.select(v).unwrap();
                let j: serde_json::Value = serde_json::from_str(&j).unwrap();
                assert!(j["gpu_agrees"].as_bool().unwrap(), "{j}");
                assert!(!u32s(&b[0]).is_empty());
                j["triangles"].as_u64().unwrap()
            };
            let near = cut(&view(0.01, 0.1));
            let far = cut(&view(50.0, 1.0));
            assert!(far < near, "far {far} near {near}");
        }
        DemoMode::Hlod => {
            let nodes = run.data.iter().filter(|d| d.0 == "hlod_node").count();
            let (kind, json, _) = run.data.last().unwrap();
            assert_eq!(kind, "hlod_tree");
            assert_eq!(nodes, json["nodes"].as_array().unwrap().len(), "{json}");
            let at = |d: f32| {
                demo.select(&ViewRequest {
                    eye: [0.0, 0.0, d * diagonal],
                    ppr: 935.0,
                    threshold: 1.0,
                })
                .unwrap()
            };
            let far = at(1000.0);
            let near = at(0.0);
            assert!(!u32s(&far.1[0]).is_empty(), "far away draws proxies");
            assert!(
                !u32s(&near.1[1]).is_empty(),
                "inside draws source instances"
            );
        }
        DemoMode::Splats => {
            assert_eq!(run.data.len(), 3);
            for (i, (kind, json, buffers)) in run.data.iter().enumerate() {
                assert_eq!(kind, "splats");
                let k = json["vertices"].as_u64().unwrap() as usize;
                let Buffer::F32(v) = &buffers[0] else {
                    panic!("expected f32s")
                };
                assert_eq!(v.len(), k * farthing_demo_web::modes::SPLAT_FLOATS);
                assert!(v.iter().all(|x| x.is_finite()));
                let ply = demo.level_ply(i + 1).unwrap();
                assert_eq!(Some(ply.len() as u64), json["bytes"].as_u64());
            }
        }
    }
}

/// Every mode on the textured sphere, with small settings.
#[test]
fn sphere_runs_every_mode() {
    let mut demo = Demo::load(vec![("sphere.glb".into(), textured_sphere())]).unwrap();
    let diag = demo.info().diagonal;
    let small = LodOptions {
        hlod_copies: 3,
        hlod_triangles: 300,
        splat_count: 4000,
        ..Default::default()
    };
    for mode in [
        DemoMode::Presets,
        DemoMode::Clusters,
        DemoMode::Hlod,
        DemoMode::Splats,
    ] {
        let run = run_mode(&mut demo, mode, small.clone());
        check_mode(&demo, mode, &run, diag);
    }
    // A splat run keeps PLYs but no GLB levels.
    assert!(demo.level_glb(1, false).is_err());
    assert!(demo.level_ply(1).is_ok());
}
