// Draws the clusters the cull pass appended to one bin's region of `visible`: one instance per
// cluster, one triangle per three vertices, decoded straight from the resident pages. Shaded by
// `pbr.wgsl`'s `shade`, with the wireframe drawn from barycentrics.

@group(3) @binding(0) var<storage, read> scene: array<u32>;
@group(3) @binding(1) var<storage, read> state: array<u32>;
@group(3) @binding(2) var<storage, read> instances: array<Instance>;
@group(3) @binding(3) var<storage, read> pool: array<u32>;
@group(3) @binding(4) var<storage, read> visible: array<vec2u>;
@group(3) @binding(5) var<uniform> bin: BinU;

struct BinU {
    // Where the bin's region of `visible` starts, and its capacity.
    region: u32,
    cap: u32,
    // The material slot it draws, and whether to skip mixed clusters' other slots' triangles.
    slot: u32,
    check_material: u32,
    // 0: white (the material's color), 1: a color per cluster, 2: a color per DAG level.
    color_mode: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct ClusterVaryings {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec3f,
    @location(4) bary: vec3f,
    @location(5) @interpolate(flat) wire: u32,
}

fn off_screen() -> ClusterVaryings {
    var o: ClusterVaryings;
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
}

fn oct_decode(w: u32) -> vec3f {
    let s = max(vec2f(f32(bitcast<i32>(w << 16u) >> 16u), f32(bitcast<i32>(w) >> 16u)) / 32767.0, vec2f(-1.0));
    let z = 1.0 - abs(s.x) - abs(s.y);
    var p = s;
    if z < 0.0 {
        p = (1.0 - abs(s.yx)) * select(vec2f(-1.0), vec2f(1.0), s >= vec2f(0.0));
    }
    let n = vec3f(p, z);
    let l = length(n);
    return select(vec3f(0.0), n / l, l > 0.0);
}

// The demo's colors (`app.js`'s hsl, levelColor and clusterColor).
fn hsl(h: f32, s: f32, l: f32) -> vec3f {
    let a = s * min(l, 1.0 - l);
    let k = (vec3f(0.0, 8.0, 4.0) + h * 12.0) % 12.0;
    return l - a * max(vec3f(-1.0), min(min(k - 3.0, 9.0 - k), vec3f(1.0)));
}

fn level_color(level: u32) -> vec3f {
    return hsl(fract(0.58 + f32(level) * 0.137), 0.62, 0.55);
}

fn cluster_color(id: u32) -> vec3f {
    var h = (id ^ 0x9e3779b9u) * 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return hsl(f32(h) / 4294967296.0, 0.5 + f32((h >> 8u) & 63u) / 250.0, 0.45 + f32((h >> 16u) & 63u) / 400.0);
}

fn pool_byte(base: u32, offset: u32) -> u32 {
    return (pool[base + (offset >> 2u)] >> ((offset & 3u) * 8u)) & 0xffu;
}

@vertex
fn vs_cluster(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ClusterVaryings {
    if ii >= bin.cap {
        return off_screen();
    }
    let e = visible[bin.region + ii];
    let inst = instances[e.x];
    let c = e.y;
    let t = vi / 3u;
    if t >= (cw(c, 27u) >> 16u) {
        return off_screen();
    }
    let m = mesh_at(inst.mesh);
    let slot = page_slot(m, cw(c, 24u));
    if slot == NONE {
        return off_screen();
    }
    let base = slot * scene[7];
    if bin.check_material != 0u && cw(c, 28u) == NONE {
        let o = cw(c, 29u) + 2u * t;
        let id = (pool[base + (o >> 2u)] >> ((o & 2u) * 8u)) & 0xffffu;
        if select(0u, id, id < m.slot_count) != bin.slot {
            return off_screen();
        }
    }
    // A mirrored instance's triangles wind the other way: swap two corners so every instance
    // of the bin culls and faces alike.
    var k = vi % 3u;
    if inst.normal[3].w < 0.0 && k != 0u {
        k = 3u - k;
    }
    let v = pool[base + (cw(c, 25u) >> 2u) + pool_byte(base, cw(c, 26u) + 3u * t + k)];
    let o = base + 12u + 4u * v;
    let w = vec4u(pool[o], pool[o + 1u], pool[o + 2u], pool[o + 3u]);
    let d = vec3u(w.x & 0x1fffffu, (w.x >> 21u) | ((w.y & 0x3ffu) << 11u), (w.y >> 10u) & 0x1fffffu);
    let grid = vec3u(pool[base + 4u], pool[base + 5u], pool[base + 6u]) + d;
    let lp = m.origin + vec3f(grid) * m.step;
    var normal = vec3f(0.0);
    if (m.flags & 1u) != 0u {
        normal = oct_decode(w.z);
    }
    let world = inst.model * vec4f(lp, 1.0);
    var out: ClusterVaryings;
    out.clip = frame.view_proj * world;
    out.world = world.xyz;
    out.normal = (inst.normal * vec4f(normal, 0.0)).xyz;
    out.uv = m.uv_min + vec2f(f32(w.w & 0xffffu), f32(w.w >> 16u)) * m.uv_step;
    switch bin.color_mode {
        case 1u: {
            out.color = cluster_color(cw(c, 30u));
        }
        case 2u: {
            out.color = level_color(cw(c, 21u));
        }
        default: {
            out.color = vec3f(1.0);
        }
    }
    let corner = vi % 3u;
    out.bary = vec3f(f32(corner == 0u), f32(corner == 1u), f32(corner == 2u));
    out.wire = inst.wire;
    return out;
}

@fragment
fn fs_cluster(in: ClusterVaryings, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let fw = fwidth(in.bary);
    var v: Varyings;
    v.clip = in.clip;
    v.world = in.world;
    v.normal = in.normal;
    v.uv = in.uv;
    v.color = in.color;
    // The corners were swapped for mirrored instances, so every triangle faces as if unmirrored.
    let c = shade(v, front, 1.0);
    let px = in.bary / max(fw, vec3f(1e-6));
    let edge = min(px.x, min(px.y, px.z));
    let hw = frame.widths.x * 0.5;
    let cover = (1.0 - smoothstep(hw - 0.5, hw + 0.5, edge)) * frame.wire.a * f32(in.wire);
    return vec4f(frame.wire.rgb * cover, cover) + c * (1.0 - cover);
}
