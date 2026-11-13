//! Procedural demo scene: a torus knot (procedural marble), a bumpy sphere (world-space
//! checker) and a cube hidden inside the sphere (only visibility/remeshing should notice).

use crate::engine::{Engine, MaterialDef};
use farthing::glam::{Mat4, Vec2, Vec3, Vec4};
use farthing::host::MaterialInfo;
use farthing::{Instance, MaterialSlot, Mesh, Scene};

fn grid_mesh(nu: u32, nv: u32, f: impl Fn(u32, u32) -> (Vec3, Vec3)) -> Mesh {
    // (nu + 1) × (nv + 1) vertices; the last row/column copy the first (bit-exact seams).
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    for j in 0..=nv {
        for i in 0..=nu {
            let (p, n) = f(i % nu, j % nv);
            pos.push(p);
            nrm.push(n);
            uv.push(Vec2::new(i as f32 / nu as f32, j as f32 / nv as f32));
        }
    }
    let mut idx = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let a = j * (nu + 1) + i;
            let b = a + nu + 1;
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    let mut m = Mesh::new(pos, idx);
    m.normals = Some(nrm);
    m.uvs = vec![uv];
    m
}

pub fn torus_knot(tubular: u32, radial: u32, p: f32, q: f32, radius: f32, tube: f32) -> Mesh {
    let curve = |u: f32| {
        let qu = q * u;
        let r = radius * (2.0 + qu.cos()) * 0.5;
        Vec3::new(r * (p * u).cos(), r * (p * u).sin(), radius * qu.sin() * 0.5)
    };
    let m = grid_mesh(tubular, radial, |i, j| {
        let u = i as f32 / tubular as f32 * std::f32::consts::TAU;
        let v = j as f32 / radial as f32 * std::f32::consts::TAU;
        let p1 = curve(u);
        let p2 = curve(u + 0.01);
        let t = p2 - p1;
        let n0 = p2 + p1;
        let b = t.cross(n0).normalize();
        let n = b.cross(t).normalize();
        let off = n * (-tube * v.cos()) + b * (tube * v.sin());
        (p1 + off, off.normalize())
    });
    // UVs are unique (not tiled): this engine's materials depend on world position, so a
    // tiled UV could not be flattened (see README, "flattened mode limitations").
    m
}

pub fn bumpy_sphere(seg: u32, rings: u32, radius: f32) -> Mesh {
    let mut m = grid_mesh(seg, rings, |i, j| {
        let th = j as f32 / rings as f32 * std::f32::consts::PI;
        let ph = i as f32 / seg as f32 * std::f32::consts::TAU;
        let bump = 1.0 + 0.06 * (5.0 * th).sin() * (4.0 * ph).sin() + 0.03 * (13.0 * th).cos() * (9.0 * ph).cos();
        let r = radius * bump;
        if j == 0 {
            return (Vec3::new(0.0, radius, 0.0), Vec3::Y);
        }
        let d = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
        (d * r, d)
    });
    // Poles: the last ring maps onto ring 0 via `j % nv`; fix it to the south pole.
    let last = rings * (seg + 1);
    for i in 0..=seg {
        let v = (last + i) as usize;
        m.positions[v] = Vec3::new(0.0, -radius, 0.0);
        m.normals.as_mut().unwrap()[v] = Vec3::NEG_Y;
    }
    // Drop degenerate pole triangles, then smooth normals (bumps).
    let mut idx = Vec::new();
    for t in 0..m.triangle_count() {
        let [a, b, c] = m.triangle_positions(t);
        if (b - a).cross(c - a).length_squared() > 1e-14 {
            idx.extend_from_slice(&m.triangle(t));
        }
    }
    m.material_ids = vec![0; idx.len() / 3];
    m.indices = idx;
    m.compute_normals(180.0);
    m
}

pub fn cube(size: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut uvs = Vec::new();
    for (n, u, v) in [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ] {
        let b = positions.len() as u32;
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            positions.push((n + u * su + v * sv) * 0.5 * size);
            uvs.push(Vec2::new(su * 0.5 + 0.5, sv * 0.5 + 0.5));
        }
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    let mut m = Mesh::new(positions, indices);
    m.uvs = vec![uvs];
    m.compute_normals(30.0);
    m
}

/// Builds the demo scene and registers its two procedural materials with the engine.
pub fn demo_scene(engine: &mut Engine) -> Scene {
    let marble = engine.add_material(
        MaterialDef::Marble { a: Vec4::new(0.92, 0.90, 0.85, 1.0), b: Vec4::new(0.12, 0.22, 0.45, 1.0), scale: 7.0 },
        MaterialInfo::default(),
    );
    let checker = engine.add_material(
        MaterialDef::Checker { a: Vec4::new(0.95, 0.50, 0.12, 1.0), b: Vec4::new(0.12, 0.10, 0.09, 1.0), scale: 5.0 },
        MaterialInfo::default(),
    );
    let knot = torus_knot(384, 40, 2.0, 3.0, 1.0, 0.28);
    let mut sphere = bumpy_sphere(128, 64, 0.85);
    sphere.material_ids.iter_mut().for_each(|m| *m = 1);
    let mut hidden = cube(0.5);
    hidden.material_ids.iter_mut().for_each(|m| *m = 1);
    Scene {
        meshes: vec![knot, sphere, hidden],
        materials: vec![MaterialSlot { name: "marble".into(), id: marble }, MaterialSlot { name: "checker".into(), id: checker }],
        instances: vec![
            Instance { name: "knot".into(), mesh: 0, transform: Mat4::from_translation(Vec3::new(-1.35, 0.0, 0.0)), skeleton: None },
            Instance { name: "sphere".into(), mesh: 1, transform: Mat4::from_translation(Vec3::new(1.35, 0.0, 0.0)), skeleton: None },
            Instance { name: "hidden_cube".into(), mesh: 2, transform: Mat4::from_translation(Vec3::new(1.35, 0.0, 0.0)), skeleton: None },
        ],
        skeletons: Vec::new(),
    }
}

/// A foliage-like test asset: a trunk and a canopy of small two-sided leaf quads, the kind of
/// geometry triangle LODs handle worst (thousands of disconnected parts).
pub fn bush(engine: &mut Engine, leaves: usize) -> Scene {
    let leaf_mat = engine.add_material(
        MaterialDef::Marble { a: Vec4::new(0.35, 0.62, 0.18, 1.0), b: Vec4::new(0.08, 0.25, 0.06, 1.0), scale: 9.0 },
        MaterialInfo::default(),
    );
    let bark = engine.add_material(
        MaterialDef::Checker { a: Vec4::new(0.40, 0.27, 0.15, 1.0), b: Vec4::new(0.25, 0.16, 0.08, 1.0), scale: 12.0 },
        MaterialInfo::default(),
    );
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    let mut idx = Vec::new();
    let mut mats = Vec::new();
    for _ in 0..leaves {
        // Point in an ellipsoidal canopy, leaf facing roughly outward with random tilt.
        let d = loop {
            let p = Vec3::new(rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0);
            if p.length_squared() <= 1.0 {
                break p;
            }
        };
        let c = Vec3::new(d.x * 1.0, 1.3 + d.y * 0.75, d.z * 1.0);
        let out = (c - Vec3::new(0.0, 1.3, 0.0)).normalize_or_zero();
        let jitter = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
        let n = (out + jitter * 1.2 + Vec3::Y * 0.3).normalize();
        let t = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let u = n.cross(t).normalize();
        let v = n.cross(u);
        let (w, h) = (0.06 + rnd() * 0.04, 0.10 + rnd() * 0.06);
        let b = pos.len() as u32;
        for (su, sv) in [(-1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 1.0)] {
            pos.push(c + u * (su * w * 0.5) + v * (sv * h - h * 0.5));
            nrm.push(n);
            uv.push(Vec2::new(su * 0.5 + 0.5, sv));
        }
        idx.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        mats.extend_from_slice(&[0, 0]);
    }
    // Trunk: an open cylinder.
    let seg = 16;
    let b = pos.len() as u32;
    for j in 0..2 {
        for i in 0..=seg {
            let a = i as f32 / seg as f32 * std::f32::consts::TAU;
            let r = if j == 0 { 0.12 } else { 0.07 };
            pos.push(Vec3::new(a.cos() * r, j as f32 * 1.1, a.sin() * r));
            nrm.push(Vec3::new(a.cos(), 0.0, a.sin()));
            uv.push(Vec2::new(i as f32 / seg as f32, j as f32));
        }
    }
    for i in 0..seg {
        let (a, c) = (b + i, b + seg + 1 + i);
        idx.extend_from_slice(&[a, c, a + 1, a + 1, c, c + 1]);
        mats.extend_from_slice(&[1, 1]);
    }
    let mut m = Mesh::new(pos, idx);
    m.normals = Some(nrm);
    m.uvs = vec![uv];
    m.material_ids = mats;
    Scene {
        meshes: vec![m],
        materials: vec![MaterialSlot { name: "leaf".into(), id: leaf_mat }, MaterialSlot { name: "bark".into(), id: bark }],
        instances: vec![Instance { name: "bush".into(), mesh: 0, transform: Mat4::IDENTITY, skeleton: None }],
        skeletons: Vec::new(),
    }
}

/// A thick-walled vase (a closed profile turned about y), open at the top and with a hole in
/// the bottom: a concave prop whose inside one view reveals and the next hides.
pub fn vase(engine: &mut Engine) -> Scene {
    // Closed cross-section in the (radius, y) half-plane, counter-clockwise.
    let profile: Vec<(f32, f32)> = {
        let mut p = Vec::new();
        let n = 24;
        for k in 0..=n {
            let t = k as f32 / n as f32;
            let y = -1.0 + 1.9 * t;
            p.push((0.35 + 0.65 * (std::f32::consts::PI * t * 0.9).sin(), y));
        }
        for k in (0..=n).rev() {
            let t = k as f32 / n as f32;
            let y = -0.9 + 1.8 * t;
            p.push((0.27 + 0.6 * (std::f32::consts::PI * t * 0.9).sin(), y));
        }
        p
    };
    let nv = profile.len() as u32;
    let nu = 96;
    let mut m = grid_mesh(nu, nv, |i, j| {
        let a = i as f32 / nu as f32 * std::f32::consts::TAU;
        let (r, y) = profile[j as usize];
        (Vec3::new(r * a.cos(), y, r * a.sin()), Vec3::Y)
    });
    // Outward winding (positive signed volume).
    let vol: f32 = (0..m.triangle_count())
        .map(|t| {
            let [a, b, c] = m.triangle_positions(t);
            a.dot(b.cross(c))
        })
        .sum();
    if vol < 0.0 {
        for t in m.indices.chunks_mut(3) {
            t.swap(1, 2);
        }
    }
    m.compute_normals(50.0);
    let marble = engine.add_material(
        MaterialDef::Marble { a: Vec4::new(0.85, 0.80, 0.70, 1.0), b: Vec4::new(0.45, 0.12, 0.10, 1.0), scale: 5.0 },
        MaterialInfo::default(),
    );
    Scene {
        meshes: vec![m],
        materials: vec![MaterialSlot { name: "vase".into(), id: marble }],
        instances: vec![Instance { name: "vase".into(), mesh: 0, transform: Mat4::IDENTITY, skeleton: None }],
        skeletons: Vec::new(),
    }
}
