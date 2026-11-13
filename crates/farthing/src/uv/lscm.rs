//! Least Squares Conformal Maps (Lévy, Petitjean, Ray & Maillot 2002).
//!
//! Each triangle contributes the complex residual Σ_j W_j·U_j / √(2A), where W_j are the edge
//! vectors of the triangle in its own 2D frame and U_j = u_j + i·v_j. Two vertices are pinned;
//! the real normal equations are solved with Jacobi-preconditioned conjugate gradients.

use glam::{Vec2, Vec3};

/// A sparse row with up to 6 entries over 2·n real unknowns (u_k at 2k, v_k at 2k+1).
struct Row {
    idx: [u32; 6],
    val: [f64; 6],
}

/// Solves LSCM for one chart. `tris` index into `pos`. Returns UVs per vertex (in world-scale
/// units, area-normalized), or None if the solve fails or produces flipped triangles.
pub fn lscm(pos: &[Vec3], tris: &[[u32; 3]]) -> Option<Vec<Vec2>> {
    let n = pos.len();
    if n < 3 || tris.is_empty() {
        return None;
    }
    // Pins: two far-apart vertices (double sweep).
    let far = |from: usize| {
        (0..n)
            .max_by(|&a, &b| {
                (pos[a] - pos[from])
                    .length_squared()
                    .total_cmp(&(pos[b] - pos[from]).length_squared())
            })
            .unwrap()
    };
    let p0 = far(0);
    let p1 = far(p0);
    if p0 == p1 {
        return None;
    }
    let dist = (pos[p1] - pos[p0]).length() as f64;
    let pinned = |k: usize| k == p0 || k == p1;
    let pin_val = |k: usize, comp: usize| -> f64 {
        if k == p1 && comp == 0 {
            dist
        } else {
            0.0
        }
    };
    // Build rows.
    let mut rows = Vec::with_capacity(tris.len() * 2);
    for t in tris {
        let [a, b, c] = t.map(|i| pos[i as usize].as_dvec3());
        let x = (b - a).normalize_or_zero();
        let nrm = (b - a).cross(c - a);
        let area2 = nrm.length();
        if area2 < 1e-24 {
            continue;
        }
        let y = nrm.normalize().cross(x);
        let q = [
            glam::DVec2::ZERO,
            glam::DVec2::new((b - a).length(), 0.0),
            glam::DVec2::new((c - a).dot(x), (c - a).dot(y)),
        ];
        let w = [q[2] - q[1], q[0] - q[2], q[1] - q[0]];
        let s = 1.0 / area2.sqrt();
        let mut r1 = Row {
            idx: [0; 6],
            val: [0.0; 6],
        };
        let mut r2 = Row {
            idx: [0; 6],
            val: [0.0; 6],
        };
        for j in 0..3 {
            let k = t[j];
            let (ar, ai) = (w[j].x * s, w[j].y * s);
            // Re: a u - b v ; Im: b u + a v
            r1.idx[2 * j] = 2 * k;
            r1.val[2 * j] = ar;
            r1.idx[2 * j + 1] = 2 * k + 1;
            r1.val[2 * j + 1] = -ai;
            r2.idx[2 * j] = 2 * k;
            r2.val[2 * j] = ai;
            r2.idx[2 * j + 1] = 2 * k + 1;
            r2.val[2 * j + 1] = ar;
        }
        rows.push(r1);
        rows.push(r2);
    }
    let dim = 2 * n;
    let is_pinned = |i: usize| pinned(i / 2);
    // b = -Aᵀ (A_p x_p), restricted to free unknowns.
    let mut b = vec![0.0f64; dim];
    let mut diag = vec![0.0f64; dim];
    for r in &rows {
        let mut fixed = 0.0;
        for e in 0..6 {
            let i = r.idx[e] as usize;
            if is_pinned(i) {
                fixed += r.val[e] * pin_val(i / 2, i % 2);
            }
        }
        for e in 0..6 {
            let i = r.idx[e] as usize;
            if !is_pinned(i) {
                b[i] -= r.val[e] * fixed;
                diag[i] += r.val[e] * r.val[e];
            }
        }
    }
    let apply = |x: &[f64], out: &mut [f64]| {
        out.iter_mut().for_each(|v| *v = 0.0);
        for r in &rows {
            let mut s = 0.0;
            for e in 0..6 {
                let i = r.idx[e] as usize;
                if !is_pinned(i) {
                    s += r.val[e] * x[i];
                }
            }
            for e in 0..6 {
                let i = r.idx[e] as usize;
                if !is_pinned(i) {
                    out[i] += r.val[e] * s;
                }
            }
        }
    };
    // Direct path: the normal equations AᵀA x = b over the free unknowns, factored once.
    let mut x = vec![0.0f64; dim];
    let mut red = vec![usize::MAX; dim];
    let mut nfree = 0;
    for (i, r) in red.iter_mut().enumerate() {
        if !is_pinned(i) {
            *r = nfree;
            nfree += 1;
        }
    }
    let direct = {
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(rows.len() * 21);
        for r in &rows {
            for e in 0..6 {
                let (i, vi) = (r.idx[e] as usize, r.val[e]);
                if is_pinned(i) {
                    continue;
                }
                for f in e..6 {
                    let (j, vj) = (r.idx[f] as usize, r.val[f]);
                    if is_pinned(j) {
                        continue;
                    }
                    let (a, c) = (red[i].max(red[j]), red[i].min(red[j]));
                    // Each unordered pair once per row (f ≥ e); duplicates across rows are summed.
                    entries.push((a, c, vi * vj));
                }
            }
        }
        crate::uv::sparse::EnvelopeCholesky::factor(nfree, &entries)
    };
    let solved_direct = if let Some(f) = &direct {
        let mut rb: Vec<f64> = (0..dim).filter(|&i| !is_pinned(i)).map(|i| b[i]).collect();
        f.solve(&mut rb);
        for i in 0..dim {
            if !is_pinned(i) {
                x[i] = rb[red[i]];
            }
        }
        true
    } else {
        false
    };
    // Initial guess: planar projection onto the best-fit plane (helps convergence).
    let centroid = pos.iter().copied().sum::<Vec3>() / n as f32;
    let avg_n = tris
        .iter()
        .map(|t| {
            (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize])
        })
        .sum::<Vec3>()
        .normalize_or_zero();
    let ax = (pos[p1] - pos[p0]).normalize_or_zero();
    let ay = avg_n.cross(ax).normalize_or_zero();
    for k in 0..n {
        if !pinned(k) && !solved_direct {
            let d = pos[k] - pos[p0];
            x[2 * k] = d.dot(ax) as f64;
            x[2 * k + 1] = d.dot(ay) as f64;
        }
    }
    let _ = centroid;
    // Preconditioned CG.
    let mut ax_v = vec![0.0; dim];
    apply(&x, &mut ax_v);
    let mut r: Vec<f64> = (0..dim)
        .map(|i| if is_pinned(i) { 0.0 } else { b[i] - ax_v[i] })
        .collect();
    let inv_d: Vec<f64> = diag
        .iter()
        .map(|&d| if d > 1e-30 { 1.0 / d } else { 0.0 })
        .collect();
    let mut z: Vec<f64> = r.iter().zip(&inv_d).map(|(a, b)| a * b).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let max_iter = if solved_direct {
        0
    } else {
        (dim * 2).clamp(200, 5000)
    };
    let mut ap = vec![0.0; dim];
    for _ in 0..max_iter {
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rz / pap;
        for i in 0..dim {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn / b_norm < 1e-9 {
            break;
        }
        for i in 0..dim {
            z[i] = r[i] * inv_d[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..dim {
            p[i] = z[i] + beta * p[i];
        }
    }
    for k in [p0, p1] {
        x[2 * k] = pin_val(k, 0);
        x[2 * k + 1] = pin_val(k, 1);
    }
    let mut uv: Vec<Vec2> = (0..n)
        .map(|k| Vec2::new(x[2 * k] as f32, x[2 * k + 1] as f32))
        .collect();
    if uv.iter().any(|u| !u.is_finite()) {
        return None;
    }
    // Orientation: all triangles must have the same sign; mirror if mostly negative.
    let signed: Vec<f32> = tris
        .iter()
        .map(|t| {
            crate::math::signed_area_2d(
                uv[t[0] as usize],
                uv[t[1] as usize],
                uv[t[2] as usize],
            )
        })
        .collect();
    let neg = signed.iter().filter(|&&s| s < 0.0).count();
    if neg * 2 > signed.len() {
        for u in &mut uv {
            u.y = -u.y;
        }
    }
    let flips = if neg * 2 > signed.len() {
        signed.len() - neg
    } else {
        neg
    };
    if flips > 0 || signed.iter().any(|s| s.abs() < 1e-14) {
        return None;
    }
    // Area-normalize so the chart's UV area equals its 3D area.
    let a3: f32 = tris
        .iter()
        .map(|t| {
            crate::math::triangle_area(
                pos[t[0] as usize],
                pos[t[1] as usize],
                pos[t[2] as usize],
            )
        })
        .sum();
    let a2: f32 = signed.iter().map(|s| s.abs()).sum();
    if a2 <= 0.0 {
        return None;
    }
    let k = (a3 / a2).sqrt();
    for u in &mut uv {
        *u *= k;
    }
    Some(uv)
}

/// True if a chart layout is not an embedding. Exact and linear-time for disk-like charts:
/// a layout is overlap-free when every triangle keeps the chart's orientation (no folds), no
/// vertex's UV angles sum past 2π (no wrap-around at a vertex) and the boundary does not
/// cross itself (checked on a uniform grid). `_res` is kept for API compatibility.
pub fn layout_overlaps(uv: &[Vec2], tris: &[[u32; 3]], _res: usize) -> bool {
    if tris.is_empty() {
        return false;
    }
    let area = |t: &[u32; 3]| {
        let [a, b, c] = t.map(|i| uv[i as usize].as_dvec2());
        (b - a).perp_dot(c - a)
    };
    let total: f64 = tris.iter().map(area).sum();
    let sign = if total >= 0.0 { 1.0 } else { -1.0 };
    let scale = tris
        .iter()
        .map(|t| area(t).abs())
        .fold(0.0f64, f64::max)
        .max(1e-300);
    // 1. Folds and degenerate triangles.
    if tris.iter().any(|t| area(t) * sign <= scale * 1e-12) {
        return true;
    }
    // 2. Angle sums per vertex.
    let mut angle = vec![0.0f64; uv.len()];
    for t in tris {
        for k in 0..3 {
            let (o, a, b) = (
                uv[t[k] as usize].as_dvec2(),
                uv[t[(k + 1) % 3] as usize].as_dvec2(),
                uv[t[(k + 2) % 3] as usize].as_dvec2(),
            );
            let (u, v) = (a - o, b - o);
            angle[t[k] as usize] += u.perp_dot(v).abs().atan2(u.dot(v));
        }
    }
    if angle.iter().any(|&s| s > std::f64::consts::TAU + 1e-6) {
        return true;
    }
    // 3. Boundary self-intersection.
    let mut edges: Vec<(u32, u32)> = tris
        .iter()
        .flat_map(|t| {
            (0..3).map(move |k| {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
        })
        .collect();
    edges.sort_unstable();
    let mut boundary: Vec<(u32, u32)> = Vec::new();
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        if j - i == 1 {
            boundary.push(edges[i]);
        }
        i = j;
    }
    if boundary.len() < 4 {
        return false;
    }
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    let mut len = 0.0f32;
    for &(a, b) in &boundary {
        let (pa, pb) = (uv[a as usize], uv[b as usize]);
        lo = lo.min(pa.min(pb));
        hi = hi.max(pa.max(pb));
        len += pa.distance(pb);
    }
    let cell = (len / boundary.len() as f32 * 2.0).max(1e-12);
    let dims = ((hi - lo) / cell).ceil().max(Vec2::ONE);
    let (gw, gh) = ((dims.x as usize).min(4096), (dims.y as usize).min(4096));
    let cell_of = |p: Vec2| {
        let c = ((p - lo) / (hi - lo).max(Vec2::splat(1e-12)) * Vec2::new(gw as f32, gh as f32))
            .floor();
        ((c.x as usize).min(gw - 1), (c.y as usize).min(gh - 1))
    };
    // (cell, edge) pairs, sorted: edges sharing a cell are tested against each other.
    let mut bins: Vec<(u32, u32)> = Vec::new();
    for (e, &(a, b)) in boundary.iter().enumerate() {
        let (c0, c1) = (
            cell_of(uv[a as usize].min(uv[b as usize])),
            cell_of(uv[a as usize].max(uv[b as usize])),
        );
        for y in c0.1..=c1.1 {
            for x in c0.0..=c1.0 {
                bins.push(((y * gw + x) as u32, e as u32));
            }
        }
    }
    bins.sort_unstable();
    let cross = |e: (u32, u32), f: (u32, u32)| {
        if e.0 == f.0 || e.0 == f.1 || e.1 == f.0 || e.1 == f.1 {
            return false; // adjacent boundary edges meet at their shared vertex
        }
        let (p, q, r, s) = (
            uv[e.0 as usize].as_dvec2(),
            uv[e.1 as usize].as_dvec2(),
            uv[f.0 as usize].as_dvec2(),
            uv[f.1 as usize].as_dvec2(),
        );
        let d1 = (q - p).perp_dot(r - p);
        let d2 = (q - p).perp_dot(s - p);
        let d3 = (s - r).perp_dot(p - r);
        let d4 = (s - r).perp_dot(q - r);
        d1 * d2 < 0.0 && d3 * d4 < 0.0
    };
    let mut i = 0;
    while i < bins.len() {
        let mut j = i + 1;
        while j < bins.len() && bins[j].0 == bins[i].0 {
            j += 1;
        }
        for x in i..j {
            for y in x + 1..j {
                if cross(boundary[bins[x].1 as usize], boundary[bins[y].1 as usize]) {
                    return true;
                }
            }
        }
        i = j;
    }
    false
}

/// Scales a chart's UVs so its UV area equals its 3D surface area (consistent texel density).
pub fn normalize_area(pos: &[Vec3], tris: &[[u32; 3]], uv: &mut [Vec2]) {
    let (mut a3, mut a2) = (0.0f64, 0.0f64);
    for t in tris {
        let [p0, p1, p2] = t.map(|i| pos[i as usize]);
        let [q0, q1, q2] = t.map(|i| uv[i as usize]);
        a3 += 0.5 * (p1 - p0).cross(p2 - p0).length() as f64;
        a2 += 0.5 * ((q1 - q0).perp_dot(q2 - q0)).abs() as f64;
    }
    if a2 > 1e-30 && a3 > 1e-30 {
        let s = (a3 / a2).sqrt() as f32;
        for q in uv.iter_mut() {
            *q *= s;
        }
    }
}

/// As-rigid-as-possible refinement (Liu et al. 2008, local/global): starting from a valid
/// layout, alternates per-triangle best rotations with a cotangent-Laplacian solve. Pulls a
/// conformal (LSCM) map towards isometry. Returns false (and leaves `uv` untouched) if the
/// result would flip a triangle.
pub fn arap_refine(pos: &[Vec3], tris: &[[u32; 3]], uv: &mut [Vec2], iterations: usize) -> bool {
    let n = uv.len();
    if n < 3 || tris.is_empty() || iterations == 0 {
        return true;
    }
    // Local isometric 2D frames and cotangent weights per triangle (edge k opposite vertex k).
    let mut local = Vec::with_capacity(tris.len());
    let mut cot = Vec::with_capacity(tris.len());
    for t in tris {
        let [a, b, c] = t.map(|i| pos[i as usize].as_dvec3());
        let e1 = b - a;
        let nrm = e1.cross(c - a);
        let l1 = e1.length();
        if l1 < 1e-20 || nrm.length() < 1e-24 {
            local.push([glam::DVec2::ZERO; 3]);
            cot.push([0.0; 3]);
            continue;
        }
        let x = e1 / l1;
        let y = nrm.normalize().cross(x);
        let p = [
            glam::DVec2::ZERO,
            glam::DVec2::new(l1, 0.0),
            glam::DVec2::new((c - a).dot(x), (c - a).dot(y)),
        ];
        let ct = |k: usize| {
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            let (u, v) = (p[i] - p[k], p[j] - p[k]);
            (u.dot(v) / u.perp_dot(v).abs().max(1e-20)).max(1e-4)
        };
        local.push(p);
        cot.push([ct(0), ct(1), ct(2)]);
    }
    // Laplacian (CSR-free: adjacency lists with weights).
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut diag = vec![0.0f64; n];
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let (i, j) = (tri[(k + 1) % 3] as usize, tri[(k + 2) % 3] as usize);
            let w = 0.5 * cot[t][k];
            diag[i] += w;
            diag[j] += w;
            adj[i].push((j as u32, w));
            adj[j].push((i as u32, w));
        }
    }
    let pin = 0usize;
    // The system matrix (cotangent Laplacian, pinned vertex removed) is the same in every
    // iteration: factor it once and reuse it (CG fallback for huge charts).
    let direct = {
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(n * 4);
        let red = |v: usize| if v < pin { v } else { v - 1 };
        for i in 0..n {
            if i == pin {
                continue;
            }
            entries.push((red(i), red(i), diag[i]));
            for &(j, w) in &adj[i] {
                let j = j as usize;
                if j != pin && i < j {
                    entries.push((red(i), red(j), -w));
                }
            }
        }
        crate::uv::sparse::EnvelopeCholesky::factor(n - 1, &entries)
    };
    let orig: Vec<Vec2> = uv.to_vec();
    let mut u: Vec<glam::DVec2> = uv.iter().map(|q| q.as_dvec2()).collect();
    let apply = |x: &[f64], out: &mut [f64]| {
        for i in 0..n {
            if i == pin {
                out[i] = x[i];
                continue;
            }
            let mut s = diag[i] * x[i];
            for &(j, w) in &adj[i] {
                if j as usize != pin {
                    s -= w * x[j as usize];
                }
            }
            out[i] = s;
        }
    };
    let sign0: f64 = tris
        .iter()
        .map(|t| {
            let q = t.map(|i| orig[i as usize].as_dvec2());
            (q[1] - q[0]).perp_dot(q[2] - q[0])
        })
        .sum::<f64>()
        .signum();
    let flip_free = |u: &[glam::DVec2]| {
        tris.iter().all(|t| {
            let q = t.map(|i| u[i as usize]);
            (q[1] - q[0]).perp_dot(q[2] - q[0]) * sign0 > 0.0
        })
    };
    let mut last_good = u.clone();
    let mut any = false;
    for _ in 0..iterations {
        // Local step: closest rotation to each triangle's Jacobian.
        let mut rhs = vec![glam::DVec2::ZERO; n];
        for (t, tri) in tris.iter().enumerate() {
            let p = local[t];
            if cot[t] == [0.0; 3] {
                continue;
            }
            let q = tri.map(|i| u[i as usize]);
            // J = [q1-q0, q2-q0] · [p1-p0, p2-p0]^-1
            let pm = glam::DMat2::from_cols(p[1] - p[0], p[2] - p[0]);
            let qm = glam::DMat2::from_cols(q[1] - q[0], q[2] - q[0]);
            if pm.determinant().abs() < 1e-30 {
                continue;
            }
            let j = qm * pm.inverse();
            let ang = (j.x_axis.y - j.y_axis.x).atan2(j.x_axis.x + j.y_axis.y);
            let r = glam::DMat2::from_angle(ang);
            for k in 0..3 {
                let (i, jj) = ((k + 1) % 3, (k + 2) % 3);
                let w = 0.5 * cot[t][k];
                let d = r * (p[i] - p[jj]) * w;
                rhs[tri[i] as usize] += d;
                rhs[tri[jj] as usize] -= d;
            }
        }
        // Global step: L u = rhs, per coordinate, warm-started CG with Jacobi preconditioner.
        for c in 0..2 {
            let mut b: Vec<f64> = rhs.iter().map(|v| if c == 0 { v.x } else { v.y }).collect();
            b[pin] = if c == 0 { u[pin].x } else { u[pin].y };
            // Move the pinned column to the right-hand side.
            let pv = b[pin];
            for &(j, w) in &adj[pin] {
                if j as usize != pin {
                    b[j as usize] += w * pv;
                }
            }
            if let Some(f) = &direct {
                let mut red: Vec<f64> = (0..n).filter(|&i| i != pin).map(|i| b[i]).collect();
                f.solve(&mut red);
                let mut k = 0;
                for i in 0..n {
                    if i == pin {
                        continue;
                    }
                    if c == 0 {
                        u[i].x = red[k]
                    } else {
                        u[i].y = red[k]
                    }
                    k += 1;
                }
                continue;
            }
            let mut x: Vec<f64> = u.iter().map(|v| if c == 0 { v.x } else { v.y }).collect();
            let mut ax = vec![0.0; n];
            apply(&x, &mut ax);
            let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
            let inv: Vec<f64> = (0..n)
                .map(|i| {
                    if i == pin || diag[i] <= 1e-30 {
                        1.0
                    } else {
                        1.0 / diag[i]
                    }
                })
                .collect();
            let mut z: Vec<f64> = r.iter().zip(&inv).map(|(a, b)| a * b).collect();
            let mut p = z.clone();
            let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
            let mut ap = vec![0.0; n];
            for _ in 0..(4 * n).clamp(50, 2000) {
                apply(&p, &mut ap);
                let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
                if pap.abs() < 1e-300 {
                    break;
                }
                let alpha = rz / pap;
                for i in 0..n {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * ap[i];
                }
                if r.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-7 * bn {
                    break;
                }
                for i in 0..n {
                    z[i] = r[i] * inv[i];
                }
                let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
                let beta = rz2 / rz.max(1e-300);
                rz = rz2;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
            }
            for i in 0..n {
                if c == 0 {
                    u[i].x = x[i]
                } else {
                    u[i].y = x[i]
                }
            }
        }
        // Keep the last flip-free iterate; stop at the first fold.
        if !flip_free(&u) {
            break;
        }
        last_good.copy_from_slice(&u);
        any = true;
    }
    if !any {
        return false;
    }
    for (dst, src) in uv.iter_mut().zip(&last_good) {
        *dst = src.as_vec2();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hemisphere patch: LSCM must flatten it without flips and with low angle distortion.
    #[test]
    fn hemisphere_flattens_conformally() {
        let (seg, rings) = (24u32, 8u32);
        let mut pos = vec![Vec3::Y];
        for r in 1..=rings {
            let th = r as f32 / rings as f32 * std::f32::consts::FRAC_PI_2 * 0.95;
            for s in 0..seg {
                let ph = s as f32 / seg as f32 * std::f32::consts::TAU;
                pos.push(Vec3::new(
                    th.sin() * ph.cos(),
                    th.cos(),
                    th.sin() * ph.sin(),
                ));
            }
        }
        let mut tris = Vec::new();
        for s in 0..seg {
            tris.push([0, 1 + (s + 1) % seg, 1 + s]);
        }
        for r in 0..rings - 1 {
            for s in 0..seg {
                let a = 1 + r * seg + s;
                let b = 1 + r * seg + (s + 1) % seg;
                tris.push([a, b, a + seg]);
                tris.push([b, b + seg, a + seg]);
            }
        }
        let uv = lscm(&pos, &tris).expect("lscm");
        assert!(!layout_overlaps(&uv, &tris, 128));
        // Angle preservation: compare the angle at each corner in 3D vs UV.
        let mut worst = 0.0f32;
        for t in &tris {
            for k in 0..3 {
                let (a, b, c) = (
                    t[k] as usize,
                    t[(k + 1) % 3] as usize,
                    t[(k + 2) % 3] as usize,
                );
                let ang3 = (pos[b] - pos[a]).angle_between(pos[c] - pos[a]);
                let ang2 = (uv[b] - uv[a]).angle_to(uv[c] - uv[a]).abs();
                worst = worst.max((ang3 - ang2).abs());
            }
        }
        assert!(worst < 0.35, "worst angle error {worst} rad");
    }
}
