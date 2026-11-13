// Reference runtime shader for farthing octahedral / hemi-octahedral impostors.
// Engines adapt this; the encode/decode functions MUST match farthing::impostor::oct_encode /
// hemi_oct_encode exactly, or frames snap at cell boundaries.
//
// Atlas layout: frames_per_side × frames_per_side frames; frame (i, j) was captured with an
// orthographic camera looking from direction decode(i / (N - 1), j / (N - 1)) toward the
// impostor center (y-up), covering [-radius, radius]².
//
// Per-frame "virtual plane" reprojection (Brucks 2018): the view ray is intersected with each
// blended frame's own capture plane, then refined with that frame's stored height (normal atlas
// alpha), so the three frames sample the same surface point instead of the same billboard UV.
//
// Layered impostors (frames.w = 2) bind a second pair of atlases: per frame, the nearest
// front-facing surface behind the first (depth peeling). The march takes whichever layer the
// ray reaches first, so surfaces a neighbouring view uncovers are drawn instead of stretched.
// Single-layer impostors bind the first pair twice and set frames.w = 1.

struct Impostor {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    center_radius: vec4<f32>,   // xyz center (world), w radius
    frames: vec4<f32>,          // x frames_per_side, y hemisphere (0/1), z output mode, w layers
};

@group(0) @binding(0) var<uniform> imp: Impostor;
@group(0) @binding(1) var atlas_color: texture_2d<f32>;
@group(0) @binding(2) var atlas_normal: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;
@group(0) @binding(4) var atlas_color2: texture_2d<f32>;
@group(0) @binding(5) var atlas_normal2: texture_2d<f32>;

// Coverage (x) and height in radius units (y) of a layer at atlas uv.
fn layer_at(layer: i32, fuv: vec2<f32>) -> vec2<f32> {
    if (layer == 0) {
        return vec2<f32>(textureSampleLevel(atlas_color, atlas_sampler, fuv, 0.0).a,
                         textureSampleLevel(atlas_normal, atlas_sampler, fuv, 0.0).a * 2.0 - 1.0);
    }
    return vec2<f32>(textureSampleLevel(atlas_color2, atlas_sampler, fuv, 0.0).a,
                     textureSampleLevel(atlas_normal2, atlas_sampler, fuv, 0.0).a * 2.0 - 1.0);
}

fn sign_not_zero(v: vec2<f32>) -> vec2<f32> {
    return select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
}

// Full octahedron, y-up. Direction -> [0,1]^2.
fn oct_encode(d: vec3<f32>) -> vec2<f32> {
    let o = d / (abs(d.x) + abs(d.y) + abs(d.z));
    var p = o.xz;
    if (o.y < 0.0) {
        p = (vec2<f32>(1.0) - abs(o.zx)) * sign_not_zero(o.xz);
    }
    return p * 0.5 + 0.5;
}

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let p = uv * 2.0 - 1.0;
    let y = 1.0 - abs(p.x) - abs(p.y);
    var xz = p;
    if (y < 0.0) {
        xz = (vec2<f32>(1.0) - abs(p.yx)) * sign_not_zero(p);
    }
    return normalize(vec3<f32>(xz.x, y, xz.y));
}

// Upper hemisphere (y >= 0) onto the whole square.
fn hemi_oct_encode(d: vec3<f32>) -> vec2<f32> {
    let o = d / (abs(d.x) + abs(d.y) + abs(d.z));
    return vec2<f32>(o.x + o.z, o.z - o.x) * 0.5 + 0.5;
}

fn hemi_oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let a = uv * 2.0 - 1.0;
    let x = (a.x - a.y) * 0.5;
    let z = (a.x + a.y) * 0.5;
    return normalize(vec3<f32>(x, max(1.0 - abs(x) - abs(z), 0.0), z));
}

// Same up-vector rule as farthing::host::View::orthographic (looking along -d).
fn capture_up(d: vec3<f32>) -> vec3<f32> {
    let c = cross(d, vec3<f32>(0.0, 1.0, 0.0));
    if (dot(c, c) < 1e-8) {
        if (abs(d.y) < 0.9) { return vec3<f32>(0.0, 1.0, 0.0); }
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

// Camera-facing quad from vertices with corners in [-1, 1]^2, enlarged so the perspective
// silhouette of the bounding sphere is never clipped.
@vertex
fn vs_main(@location(0) corner: vec2<f32>) -> VsOut {
    let c = imp.center_radius.xyz;
    let r = imp.center_radius.w * 1.2;
    let to_cam = normalize(imp.camera_pos.xyz - c);
    let up = capture_up(to_cam);
    let right = normalize(cross(up, to_cam));
    let true_up = cross(to_cam, right);
    let world = c + (right * corner.x + true_up * corner.y) * r;
    var out: VsOut;
    out.clip = imp.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    return out;
}

fn frame_dir(cell: vec2<f32>, n: f32) -> vec3<f32> {
    let uv = cell / (n - 1.0);
    if (imp.frames.y > 0.5) { return hemi_oct_decode(uv); }
    return oct_decode(uv);
}

struct FrameSample {
    color: vec4<f32>,
    normal: vec3<f32>,
};

// Intersect the view ray with frame `cell`'s capture plane and sample the frame there.
fn sample_frame(cell: vec2<f32>, ray_o: vec3<f32>, ray_d: vec3<f32>, n: f32) -> FrameSample {
    let c = imp.center_radius.xyz;
    let r = imp.center_radius.w;
    let d = frame_dir(cell, n);
    var s: FrameSample;
    s.color = vec4<f32>(0.0);
    s.normal = vec3<f32>(0.0, 0.0, 1.0);
    let denom = dot(ray_d, d);
    if (abs(denom) < 1e-5) { return s; }
    // Capture camera basis: forward = -d, right = forward × up, up' = right × forward.
    let up = capture_up(d);
    let right = normalize(cross(-d, up));
    let true_up = cross(right, -d);
    // March the ray through the frame's height field (relief mapping): the surface sits h above
    // the capture plane (normal atlas alpha = h / radius * 0.5 + 0.5) where the colour alpha
    // says the texel is covered. Step from where the ray enters the slab |height| <= r to where
    // it leaves, take the first step that passes below the surface, then bisect. Unlike a
    // fixed-point iteration this does not jump across depth discontinuities, which left gaps.
    let t_plane = dot(c - ray_o, d) / denom;
    let ta = t_plane + r / denom;
    let tb = t_plane - r / denom;
    let t0 = min(ta, tb);
    let t1 = max(ta, tb);
    var prev_t = t0;
    var hit = false;
    var lo = t0;
    var hi = t0;
    let steps = 16;
    // A height field is solid all the way down; treat it as a shell instead, so rays passing
    // behind the front surface of a neighbouring frame (near silhouettes) do not hit the
    // extrusion. The shell is 2.5 march steps thick so steps cannot skip it (2–4 steps tested;
    // 2.5 balanced close and distant views best).
    let shell = 5.0 * r / f32(steps);
    let layers = i32(max(imp.frames.w, 1.0));
    var layer = 0;
    for (var i = 0; i <= steps; i = i + 1) {
        let t = mix(t0, t1, f32(i) / f32(steps));
        let q = ray_o + ray_d * t - c;
        let lp = vec2<f32>(dot(q, right), -dot(q, true_up)) / (2.0 * r) + 0.5;
        if (all(lp >= vec2<f32>(0.0)) && all(lp <= vec2<f32>(1.0))) {
            let fuv = (cell + lp) / n;
            let rh = dot(q, d);
            for (var k = 0; k < layers; k = k + 1) {
                let ch = layer_at(k, fuv);
                let hgt = ch.y * r;
                if (ch.x >= 0.5 && rh <= hgt && rh >= hgt - shell) {
                    hit = true;
                    layer = k;
                    break;
                }
            }
            if (hit) {
                lo = prev_t;
                hi = t;
                break;
            }
        }
        prev_t = t;
    }
    if (!hit) { return s; }
    var local = vec2<f32>(0.0);
    for (var i = 0; i < 5; i = i + 1) {
        let t = 0.5 * (lo + hi);
        let q = ray_o + ray_d * t - c;
        let lp = clamp(vec2<f32>(dot(q, right), -dot(q, true_up)) / (2.0 * r) + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
        let fuv = (cell + lp) / n;
        let ch = layer_at(layer, fuv);
        if (ch.x >= 0.5 && dot(q, d) <= ch.y * r) { hi = t; } else { lo = t; }
    }
    let qh = ray_o + ray_d * hi - c;
    local = clamp(vec2<f32>(dot(qh, right), -dot(qh, true_up)) / (2.0 * r) + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv = (cell + local) / n;
    if (layer == 0) {
        s.color = textureSampleLevel(atlas_color, atlas_sampler, uv, 0.0);
        s.normal = textureSampleLevel(atlas_normal, atlas_sampler, uv, 0.0).xyz * 2.0 - 1.0;
    } else {
        s.color = textureSampleLevel(atlas_color2, atlas_sampler, uv, 0.0);
        s.normal = textureSampleLevel(atlas_normal2, atlas_sampler, uv, 0.0).xyz * 2.0 - 1.0;
    }
    return s;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = imp.frames.x;
    let c = imp.center_radius.xyz;
    var dir = normalize(imp.camera_pos.xyz - c);
    var g: vec2<f32>;
    if (imp.frames.y > 0.5) {
        dir.y = max(dir.y, 0.0);
        g = hemi_oct_encode(normalize(dir)) * (n - 1.0);
    } else {
        g = oct_encode(dir) * (n - 1.0);
    }
    // The three nearest frames of the triangulated grid.
    let base = floor(g);
    let f = fract(g);
    var a: vec2<f32>; var b: vec2<f32>; var cc: vec2<f32>; var w: vec3<f32>;
    if (f.x + f.y < 1.0) {
        a = base; b = base + vec2<f32>(1.0, 0.0); cc = base + vec2<f32>(0.0, 1.0);
        w = vec3<f32>(1.0 - f.x - f.y, f.x, f.y);
    } else {
        a = base + vec2<f32>(1.0, 1.0); b = base + vec2<f32>(0.0, 1.0); cc = base + vec2<f32>(1.0, 0.0);
        w = vec3<f32>(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
    }
    let hi = vec2<f32>(n - 1.0);
    a = min(a, hi); b = min(b, hi); cc = min(cc, hi);
    let ray_o = imp.camera_pos.xyz;
    let ray_d = normalize(in.world - ray_o);
    let sa = sample_frame(a, ray_o, ray_d, n);
    let sb = sample_frame(b, ray_o, ray_d, n);
    let sc = sample_frame(cc, ray_o, ray_d, n);
    let color = sa.color * w.x + sb.color * w.y + sc.color * w.z;
    if (color.a < 0.5) { discard; }
    let nrm = normalize(sa.normal * w.x + sb.normal * w.y + sc.normal * w.z);
    let rgb = (sa.color.rgb * sa.color.a * w.x + sb.color.rgb * sb.color.a * w.y + sc.color.rgb * sc.color.a * w.z) / max(color.a, 1e-4);
    // frames.z selects the output: 0 = lit preview, 1 = albedo, 2 = world normal (G-buffer
    // passes for hosts that light impostors with their own deferred lighting).
    if (imp.frames.z > 1.5) { return vec4<f32>(nrm, 1.0); }
    if (imp.frames.z > 0.5) { return vec4<f32>(rgb, 1.0); }
    let light = max(dot(nrm, normalize(vec3<f32>(0.4, 0.8, 0.3))), 0.0) * 0.8 + 0.2;
    return vec4<f32>(rgb * light, 1.0);
}
