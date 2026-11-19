// Draws one level of a crowd: instance i is the i-th unit the cull appended to the level's
// region of `visible`. Skinned levels blend the palette's frames around each unit's clip time
// and skin with up to four joints; a vertex animation level (FVAT) reads its baked positions.
// Shaded by `pbr.wgsl`'s `shade`. Follows `common.wgsl`, `pbr.wgsl` and `crowd.wgsl`.

struct DrawU {
    model: mat4x4f,
    // The inverse transpose of `model`.
    normal: mat4x4f,
    // Where the level's region of `visible` starts.
    region: u32,
    // The clock the units' start times are on, seconds.
    time: f32,
    // A vertex animation level's vertex count and flags (bit 0: normals).
    vat_vertices: u32,
    vat_flags: u32,
    vat_min: vec4f,
    vat_extent: vec4f,
    // The level's color when showing the levels (alpha 0), replacing the material's color;
    // white with alpha 1 otherwise.
    tint: vec4f,
}

@group(3) @binding(0) var<uniform> palette: Palette;
@group(3) @binding(1) var<storage, read> clips: array<Clip>;
@group(3) @binding(2) var<storage, read> bones: array<vec4u>;
@group(3) @binding(3) var<storage, read> units: array<Unit>;
@group(3) @binding(4) var<storage, read> visible: array<u32>;
@group(3) @binding(5) var<uniform> draw: DrawU;
@group(3) @binding(6) var<storage, read> vat: array<vec2u>;

struct Bone {
    rotation: vec4f,
    translation: vec3f,
    scale: f32,
}

fn decode_bone(frame: u32, joint: u32) -> Bone {
    let w = bones[frame * palette.joint_count + joint];
    let q = vec4f(unpack2x16snorm(w.x), unpack2x16snorm(w.y));
    let t = vec3f(unpack2x16unorm(w.z), unpack2x16unorm(w.w).x);
    return Bone(normalize(q), palette.translation_min + t * palette.translation_extent, unpack2x16float(w.w).y);
}

fn blend_bones(a: Bone, b: Bone, t: f32) -> Bone {
    var qb = b.rotation;
    if dot(a.rotation, qb) < 0.0 {
        qb = -qb;
    }
    return Bone(normalize(mix(a.rotation, qb, t)), mix(a.translation, b.translation, t), mix(a.scale, b.scale, t));
}

fn sample_bone(f: Frames, joint: u32) -> Bone {
    return blend_bones(decode_bone(f.f0, joint), decode_bone(f.f1, joint), f.t);
}

fn rotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// A unit's clip state now: the frames of the clip it plays, of the one it fades out of, and
// how far the fade has gone (1: done).
struct UnitPose {
    a: Frames,
    b: Frames,
    fade: f32,
}

fn unit_pose(unit: Unit) -> UnitPose {
    let a = frames_at(clips[unit.clip], (draw.time - unit.start) * unit.speed);
    let b = frames_at(clips[unit.previous_clip], (draw.time - unit.previous_start) * unit.speed);
    var fade = 1.0;
    if unit.fade > 0.0 {
        fade = clamp((draw.time - unit.fade_start) / unit.fade, 0.0, 1.0);
    }
    return UnitPose(a, b, fade);
}

fn turn(yaw: f32) -> mat3x3f {
    let c = cos(yaw);
    let s = sin(yaw);
    return mat3x3f(vec3f(c, 0.0, -s), vec3f(0.0, 1.0, 0.0), vec3f(s, 0.0, c));
}

// Places a unit-space point and normal, and fills the varyings.
fn place(unit: Unit, p: vec3f, n: vec3f, uv: vec2f) -> Varyings {
    let r = turn(unit.yaw);
    let world = draw.model * vec4f(unit.position + r * (p * unit.scale), 1.0);
    var out: Varyings;
    out.clip = frame.view_proj * world;
    out.world = world.xyz;
    out.normal = (draw.normal * vec4f(r * n, 0.0)).xyz;
    out.uv = uv;
    out.color = vec3f(1.0);
    return out;
}

@vertex
fn vs_skinned(
    @builtin(instance_index) ii: u32,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) joints: vec4u,
    @location(4) weights: vec4f,
) -> Varyings {
    let unit = units[visible[draw.region + ii]];
    let pose = unit_pose(unit);
    // Linear blend skinning with up to four joints; the weights add up to 1.
    var p = vec3f(0.0);
    var n = vec3f(0.0);
    for (var k = 0u; k < 4u; k++) {
        let w = weights[k];
        if w > 0.0 {
            var bone = sample_bone(pose.a, joints[k]);
            if pose.fade < 1.0 {
                bone = blend_bones(sample_bone(pose.b, joints[k]), bone, pose.fade);
            }
            p += w * (rotate(bone.rotation, position * bone.scale) + bone.translation);
            n += w * rotate(bone.rotation, normal);
        }
    }
    return place(unit, p, n, uv);
}

fn vat_texel(frame: u32, v: u32) -> vec2u {
    return vat[frame * draw.vat_vertices + v];
}

fn vat_position(w: vec2u) -> vec3f {
    return draw.vat_min.xyz + vec3f(unpack2x16unorm(w.x), unpack2x16unorm(w.y).x) * draw.vat_extent.xyz;
}

fn vat_normal(w: vec2u) -> vec3f {
    if (draw.vat_flags & 1u) == 0u {
        return vec3f(0.0);
    }
    let e = unpack4x8snorm(w.y >> 16u).xy;
    var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
    if n.z < 0.0 {
        n = vec3f((1.0 - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)), n.z);
    }
    return normalize(n);
}

// The vertex at a pair of frames, blended.
fn vat_at(f: Frames, v: u32) -> array<vec3f, 2> {
    let w0 = vat_texel(f.f0, v);
    let w1 = vat_texel(f.f1, v);
    return array<vec3f, 2>(mix(vat_position(w0), vat_position(w1), f.t), mix(vat_normal(w0), vat_normal(w1), f.t));
}

@vertex
fn vs_vat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, @location(2) uv: vec2f) -> Varyings {
    let unit = units[visible[draw.region + ii]];
    let pose = unit_pose(unit);
    var a = vat_at(pose.a, vi);
    if pose.fade < 1.0 {
        let b = vat_at(pose.b, vi);
        a = array<vec3f, 2>(mix(b[0], a[0], pose.fade), mix(b[1], a[1], pose.fade));
    }
    return place(unit, a[0], a[1], uv);
}

@fragment
fn fs_crowd(in: Varyings, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let c = shade(in, front, 1.0);
    if draw.tint.a > 0.5 {
        return c;
    }
    // Showing the levels: the level's color, lit as brightly as the material is, so a level
    // reads the same on any texture.
    let light = dot(c.rgb, vec3f(0.2126, 0.7152, 0.0722));
    return vec4f(draw.tint.rgb * (0.3 + 0.7 * smoothstep(0.0, 0.5, light)), c.a);
}
