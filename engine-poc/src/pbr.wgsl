// Surfaces: glTF metallic-roughness lit only by a procedural room, an approximation of the
// three.js RoomEnvironment the demo used: walls, floor and ceiling as a gradient, and the
// room's light panels as lobes that widen with roughness (split-sum, with Karis's analytic
// environment BRDF). Output is linear and premultiplied; the target is an sRGB view, so the
// GPU encodes it.

struct Material {
    base: vec4f,
    // RGB, then the occlusion strength.
    emissive: vec4f,
    // Metallic, roughness, alpha cutoff (0: none), blend (0 or 1).
    params: vec4f,
    // xy, zero without a normal texture; z is 1 for flat shading.
    normal_scale: vec4f,
}

@group(2) @binding(0) var<uniform> mat: Material;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var base_tex: texture_2d<f32>;
@group(2) @binding(3) var mr_tex: texture_2d<f32>;
@group(2) @binding(4) var normal_tex: texture_2d<f32>;
@group(2) @binding(5) var emissive_tex: texture_2d<f32>;
@group(2) @binding(6) var occlusion_tex: texture_2d<f32>;

const PI: f32 = 3.14159265;

struct Varyings {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec3f,
}

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f, @location(3) color: vec3f) -> Varyings {
    let world = obj.model * vec4f(position, 1.0);
    var out: Varyings;
    out.clip = frame.view_proj * world;
    out.world = world.xyz;
    out.normal = (obj.normal * vec4f(normal, 0.0)).xyz;
    out.uv = uv;
    out.color = color;
    return out;
}

// The room's light panels as seen from the model: direction, and energy (radiance times solid
// angle). The last is the bright middle of the ceiling.
const PANELS = array<vec4f, 7>(
    vec4f(-0.698, 0.622, 0.355, 0.459),
    vec4f(-0.631, 0.706, -0.322, 0.344),
    vec4f(0.770, 0.631, -0.095, 0.975),
    vec4f(-0.027, 0.522, 0.853, 3.044),
    vec4f(0.187, 0.664, -0.724, 0.255),
    vec4f(0.0, 1.0, 0.0, 0.25),
    vec4f(0.0, 1.0, 0.0, 1.0),
);
// Each panel's size as the exponent of a cosine lobe of the same solid angle.
const PANEL_N = array<f32, 7>(684.0, 912.0, 109.0, 88.0, 491.0, 2512.0, 6.5);

// The walls, floor and ceiling (no panels): radiance seen along `d`.
fn room(d: vec3f) -> f32 {
    if d.y > 0.0 {
        return 0.15 + 0.55 * smoothstep(0.0, 0.7, d.y);
    }
    return 0.15 + 0.55 * smoothstep(0.0, 1.0, -d.y);
}

// The walls' irradiance over π for a normal with this y (fit to `room`, integrated).
fn room_diffuse(y: f32) -> f32 {
    return 0.42 + 0.04 * y + 0.16 * y * y;
}

fn diffuse_light(n: vec3f) -> f32 {
    var e = room_diffuse(n.y);
    for (var i = 0; i < 7; i++) {
        e += PANELS[i].w * max(dot(n, PANELS[i].xyz), 0.0) / PI;
    }
    return e;
}

// The room's radiance along `r`, blurred as a surface of this roughness reflects it.
fn specular_light(r: vec3f, rough: f32) -> f32 {
    let a = rough * rough;
    let n_rough = max(2.0 / max(a * a, 1e-4) - 2.0, 1e-3);
    var l = mix(room(r), room_diffuse(r.y), rough);
    for (var i = 0; i < 7; i++) {
        let n = 1.0 / (1.0 / n_rough + 1.0 / PANEL_N[i]);
        l += PANELS[i].w * (n + 1.0) / (2.0 * PI) * pow(max(dot(r, PANELS[i].xyz), 0.0), n);
    }
    return l;
}

// Karis, "Physically Based Shading on Mobile".
fn env_brdf(f0: vec3f, rough: f32, nov: f32) -> vec3f {
    let r = rough * vec4f(-1.0, -0.0275, -0.572, 0.022) + vec4f(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

@fragment
fn fs_main(in: Varyings, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    return shade(in, front, obj.normal[3].w);
}

// The lit surface. `det` is the sign of the model's determinant (its triangles face the other
// way when mirrored). A zero normal shades flat. Call it from uniform control flow: it takes
// derivatives.
fn shade(in: Varyings, front: bool, det: f32) -> vec4f {
    // Everything that needs derivatives first, in uniform control flow.
    let base_s = textureSample(base_tex, samp, in.uv);
    let mr = textureSample(mr_tex, samp, in.uv);
    let nt = textureSample(normal_tex, samp, in.uv).xyz * 2.0 - 1.0;
    let emissive_s = textureSample(emissive_tex, samp, in.uv).rgb;
    let occlusion_s = textureSample(occlusion_tex, samp, in.uv).r;
    let dp1 = dpdx(in.world);
    let dp2 = dpdy(in.world);
    let duv1 = dpdx(in.uv);
    let duv2 = dpdy(in.uv);

    let face = select(-1.0, 1.0, front) * det;
    let flat = mat.normal_scale.z != 0.0 || dot(in.normal, in.normal) == 0.0;
    var n = select(normalize(in.normal), vec3f(0.0, 0.0, 1.0), flat) * face;
    let v = normalize(frame.eye.xyz - in.world);
    if flat {
        // Flat shading: the triangle's own normal, facing the camera.
        let f = normalize(cross(dp1, dp2));
        n = select(-f, f, dot(f, v) >= 0.0);
    } else if any(mat.normal_scale.xy != vec2f(0.0)) {
        // The cotangent frame from derivatives (Schüler), as three.js builds it without tangents.
        let q1perp = cross(dp2, n);
        let q0perp = cross(n, dp1);
        let t = q1perp * duv1.x + q0perp * duv2.x;
        let b = q1perp * duv1.y + q0perp * duv2.y;
        let det = max(dot(t, t), dot(b, b));
        let s = select(inverseSqrt(det), 0.0, det == 0.0) * face;
        let m = vec3f(nt.xy * mat.normal_scale.xy, nt.z);
        n = normalize(t * s * m.x + b * s * m.y + n * m.z);
    }

    var base = mat.base * base_s * vec4f(in.color, 1.0);
    if mat.params.z > 0.0 {
        if base.a < mat.params.z {
            discard;
        }
        base.a = 1.0;
    } else if mat.params.w == 0.0 {
        base.a = 1.0;
    }
    let rough = clamp(mat.params.y * mr.g, 0.04, 1.0);
    let metal = clamp(mat.params.x * mr.b, 0.0, 1.0);
    let nov = clamp(dot(n, v), 1e-4, 1.0);
    let ao = mix(1.0, occlusion_s, mat.emissive.w);
    let diffuse = base.rgb * (1.0 - metal) * diffuse_light(n);
    let specular = env_brdf(mix(vec3f(0.04), base.rgb, metal), rough, nov) * specular_light(reflect(-v, n), rough);
    let color = (diffuse + specular) * ao + mat.emissive.rgb * emissive_s;
    return vec4f(color * base.a, base.a);
}
