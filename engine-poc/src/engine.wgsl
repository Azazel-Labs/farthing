// farthing PoC engine shaders. farthing never sees this file: the material system below is
// the engine's own. Two of the materials depend on world position and procedural noise, so
// they cannot be expressed as glTF PBR — exactly what the RenderHost boundary must handle.

struct Draw {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    camera_pos: vec4<f32>,
    view: mat4x4<f32>,
    lit: vec4<f32>,   // x lit, y peel, z peel bias
};

struct Material {
    kind: u32,        // 0 = world-space checker, 1 = procedural marble + bump, 2 = textured
    flags: u32,       // bit 0: has normal map
    scale: f32,
    cutoff: f32,      // alpha-test threshold (0 = opaque)
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    factor: vec4<f32>,
};

struct BakeInfo {
    material: vec4<f32>, // x = material index + 1
};

@group(0) @binding(0) var<uniform> draw: Draw;
// Depth layer to peel (lit.y = 1): linear depth in r, coverage in a; lit.z = bias.
@group(0) @binding(1) var peel_depth: texture_2d<f32>;

@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var base_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var samp: sampler;

@group(2) @binding(0) var in_pos: texture_2d<f32>;
@group(2) @binding(1) var in_nrm: texture_2d<f32>;
@group(2) @binding(2) var in_uv: texture_2d<f32>;
@group(2) @binding(3) var in_tan: texture_2d<f32>;
@group(2) @binding(4) var<uniform> bake: BakeInfo;

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 0.0)), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
    let b = mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y);
    return mix(a, b, u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var amp = 0.5;
    var sum = 0.0;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        sum = sum + amp * noise3(q);
        q = q * 2.03;
        amp = amp * 0.5;
    }
    return sum;
}

fn marble_height(p: vec3<f32>) -> f32 {
    return sin(p.x * mat.scale + 6.0 * fbm(p * mat.scale * 0.35)) * 0.5 + 0.5;
}

struct MatOut {
    albedo: vec4<f32>,
    normal: vec3<f32>,
};

fn eval_material(p: vec3<f32>, n_in: vec3<f32>, uv: vec2<f32>, t_in: vec4<f32>) -> MatOut {
    var o: MatOut;
    let n = normalize(n_in);
    o.normal = n;
    o.albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (mat.kind == 0u) {
        let c = floor(p * mat.scale);
        let k = (i32(c.x) + i32(c.y) + i32(c.z)) & 1;
        o.albedo = select(mat.color_a, mat.color_b, k == 1);
    } else if (mat.kind == 1u) {
        let h = marble_height(p);
        o.albedo = mix(mat.color_a, mat.color_b, h);
        let e = 0.004;
        let g = vec3<f32>(marble_height(p + vec3<f32>(e, 0.0, 0.0)) - h,
                          marble_height(p + vec3<f32>(0.0, e, 0.0)) - h,
                          marble_height(p + vec3<f32>(0.0, 0.0, e)) - h) / e;
        let gt = g - n * dot(g, n);
        o.normal = normalize(n - gt * 0.015);
    } else {
        o.albedo = textureSampleLevel(base_tex, samp, uv, 0.0) * mat.factor;
        if ((mat.flags & 1u) != 0u) {
            let ts = textureSampleLevel(normal_tex, samp, uv, 0.0).xyz * 2.0 - 1.0;
            let t = normalize(t_in.xyz - n * dot(t_in.xyz, n));
            let b = cross(n, t) * t_in.w;
            o.normal = normalize(t * ts.x + b * ts.y + n * ts.z);
        }
    }
    return o;
}

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tan: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) t: vec4<f32>,
    @location(4) vdepth: f32,
};

@vertex
fn vs_scene(v: VIn) -> VOut {
    var o: VOut;
    let w = draw.model * vec4<f32>(v.pos, 1.0);
    o.clip = draw.view_proj * w;
    o.wpos = w.xyz;
    o.n = (draw.normal_mat * vec4<f32>(v.nrm, 0.0)).xyz;
    o.t = vec4<f32>((draw.model * vec4<f32>(v.tan.xyz, 0.0)).xyz, v.tan.w);
    o.uv = v.uv;
    o.vdepth = -(draw.view * w).z;
    return o;
}

struct FOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) depth: vec4<f32>,
};

fn shade(albedo: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let l = normalize(vec3<f32>(0.4, 0.8, 0.3));
    return albedo * (0.2 + 0.8 * max(dot(n, l), 0.0));
}

@fragment
fn fs_scene(i: VOut, @builtin(front_facing) ff: bool) -> FOut {
    if (draw.lit.y > 0.5) {
        // Second layer: the nearest front face behind the first layer's surface.
        let pd = textureLoad(peel_depth, vec2<i32>(i.clip.xy), 0);
        if (!ff || pd.a <= 0.0 || i.vdepth <= pd.r + draw.lit.z) { discard; }
    }
    var n = normalize(i.n);
    if (!ff) { n = -n; }
    let m = eval_material(i.wpos, n, i.uv, i.t);
    if (m.albedo.a < mat.cutoff) { discard; }
    var c = m.albedo.rgb;
    if (draw.lit.x > 0.5) { c = shade(c, m.normal); }
    var o: FOut;
    o.color = vec4<f32>(c, 1.0);
    o.normal = vec4<f32>(m.normal, 1.0);
    o.depth = vec4<f32>(i.vdepth, 0.0, 0.0, 1.0);
    return o;
}

// ---- Point-evaluation bake: one full-screen pass per material ----

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct BakeOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_bake(@builtin(position) fc: vec4<f32>) -> BakeOut {
    let c = vec2<i32>(fc.xy);
    let p = textureLoad(in_pos, c, 0);
    if (abs(p.w - bake.material.x) > 0.5) { discard; }
    let n = textureLoad(in_nrm, c, 0).xyz;
    let uv = textureLoad(in_uv, c, 0);
    let t = textureLoad(in_tan, c, 0);
    let m = eval_material(p.xyz, n, uv.xy, t);
    var o: BakeOut;
    o.albedo = vec4<f32>(m.albedo.rgb, m.albedo.a);
    o.normal = vec4<f32>(m.normal, 1.0);
    return o;
}
