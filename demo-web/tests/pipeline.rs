//! The browser pipeline, run natively: a zipped `.gltf` + `.bin` with a textured sphere,
//! through reduced and baked levels.

use farthing::image::Image;
use farthing::{Mesh, Scene};
use farthing_demo_web::pipeline::{Demo, Event, LodOptions};
use farthing::io::{glb_bytes, load_gltf_slice, ExportTextures};
use glam::{Vec2, Vec3, Vec4};
use std::io::Write;

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
    t.base_color.insert(0, tex);
    glb_bytes(&Scene::from_mesh(m), &t).unwrap()
}

/// GLB → (`.gltf` JSON referencing `bin_uri`, `.bin` bytes).
fn split_glb(glb: &[u8], bin_uri: &str) -> (Vec<u8>, Vec<u8>) {
    let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let mut doc: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
    doc["buffers"][0]["uri"] = bin_uri.into();
    (serde_json::to_vec(&doc).unwrap(), glb[20 + json_len + 8..].to_vec())
}

fn zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
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
        ..Default::default()
    };
    let (mut levels, mut textures, mut stages) = (vec![], vec![], 0);
    let t0 = std::time::Instant::now();
    demo.run(&opts, &|| t0.elapsed().as_secs_f64() * 1e3, &mut |e| match e {
        Event::Stage(_) => stages += 1,
        Event::Level { info, glb } => levels.push((info.clone(), glb.to_vec())),
        Event::Texture { channel, width, height, rgba, .. } => {
            assert_eq!(rgba.len(), (width * height * 4) as usize);
            textures.push((channel.to_owned(), rgba.to_vec()));
        }
    })
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
    let baked = &levels[3].0;
    assert!(baked.coverage.unwrap() > 0.2, "{baked:?}");
    let names: Vec<&str> = textures.iter().map(|t| t.0.as_str()).collect();
    assert_eq!(names, ["base_color", "normal", "orm"]);
    // The checker survives: the baked base color has both red and white texels.
    let px = textures[0].1.chunks_exact(4);
    let red = px.clone().filter(|p| p[0] > 200 && p[1] < 60).count();
    let white = px.filter(|p| p[0] > 200 && p[1] > 200).count();
    assert!(red > 100 && white > 100, "red {red}, white {white}");

    // The downloadable baked level embeds its textures.
    let full = demo.level_glb(3).unwrap();
    assert!(full.windows(9).any(|w| w == b"image/png"));
    assert!(full.windows(24).any(|w| w == b"metallicRoughnessTexture"));
}

#[test]
fn reports_missing_files() {
    let (gltf, _) = split_glb(&textured_sphere(), "missing.bin");
    let e = Demo::load(vec![("a.gltf".into(), gltf)]).err().unwrap();
    assert!(e.contains("missing.bin"), "{e}");
    let e = Demo::load(vec![("notes.txt".into(), b"hi".to_vec())]).err().unwrap();
    assert!(e.contains("no .glb or .gltf"), "{e}");
}
