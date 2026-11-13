// Reference runtime shader for farthing surfel clouds (progressive order, see
// farthing::impostor::surfels::SurfelCloud::lod). Engines adapt this.
//
// Draw `count` instances of a 6-vertex quad (triangle list, no vertex buffer for the corners),
// one instance per surfel, in cloud order: count = ceil(k) for a fractional LOD k. Each
// instance is an oriented disc in its surfel's tangent plane with radius
// `radius × params.x` (params.x = the radius scale `lod(k)` returns). The last instance
// (index params.y) fades in: its pixels survive with probability params.z (screen-space
// dithering, stable per surfel), so stepping k up or down never pops a whole disc.
//
// Outputs a small G-buffer (albedo, world normal) so the host lights the cloud with its own
// lighting. Albedo comes from a per-surfel RGBA8 channel the host evaluated (e.g. stored in
// the LFSF stream as a channel).

struct Splat {
    view_proj: mat4x4<f32>,
    // x: radius scale, y: index of the fading surfel (as f32), z: its fade 0..1, w: unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> splat: Splat;

struct Instance {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) radius: f32,
    @location(3) albedo: vec4<f32>, // Unorm8x4
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) albedo: vec4<f32>,
    @location(3) @interpolate(flat) index: u32,
};

// Orthonormal basis around n (Duff et al. 2017), matching farthing's surfels_to_mesh up to
// rotation about n (irrelevant for a disc).
fn basis(n: vec3<f32>) -> mat2x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let u = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat2x3<f32>(t, u);
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32, inst: Instance) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[v % 6u];
    let n = normalize(inst.normal);
    let tb = basis(n);
    let r = inst.radius * splat.params.x;
    let world = inst.position + (tb[0] * c.x + tb[1] * c.y) * r;
    var out: VsOut;
    out.clip = splat.view_proj * vec4<f32>(world, 1.0);
    out.corner = c;
    out.normal = n;
    out.albedo = inst.albedo;
    out.index = i;
    return out;
}

fn hash(p: vec2<u32>, i: u32) -> f32 {
    var h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u ^ i * 0xcb1ab31fu;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h & 0xffffffu) / 16777216.0;
}

struct GBuffer {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) ff: bool) -> GBuffer {
    if (dot(in.corner, in.corner) > 1.0) { discard; }
    // The fading surfel: dithered coverage, so its contribution grows smoothly with k.
    if (f32(in.index) == splat.params.y && hash(vec2<u32>(in.clip.xy), in.index) >= splat.params.z) {
        discard;
    }
    var o: GBuffer;
    o.albedo = vec4<f32>(in.albedo.rgb, 1.0);
    o.normal = vec4<f32>(select(-in.normal, in.normal, ff), 1.0);
    return o;
}
