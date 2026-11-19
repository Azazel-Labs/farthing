// Reference runtime shader for farthing pose palettes (the FANM format, see
// farthing::anim::bake). Engines adapt this.
//
// Draws a crowd of skinned units with one instanced draw: instance i is units[i]. A unit's
// whole animation state is its record: the clip it plays and when it started, and the clip it
// fades out of. The CPU writes a record when a unit changes clips (or moves), never per frame;
// the shader finds each joint's transform from the clock in `view.time`.
//
// Upload the palette file as it is: bytes 16..64 as the `palette` uniform, the clip array as
// `clips` and the frame array as `bones` (offsets in the header). Keep `view.time` small (rebase
// it and the units' start times now and then): f32 seconds lose a millisecond after ~2 hours.

struct View {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    // Seconds on the clock the units' start times are measured on.
    time: f32,
};

// FANM header bytes 16..64.
struct Palette {
    joint_count: u32,
    clip_count: u32,
    frame_count: u32,
    fps: f32,
    translation_min: vec3<f32>,
    clip_offset: u32,
    translation_extent: vec3<f32>,
    frame_offset: u32,
};

struct Clip {
    first_frame: u32,
    frame_count: u32,
    duration: f32,
    // Bit 0: loops.
    flags: u32,
};

// 48 bytes.
struct Unit {
    position: vec3<f32>,
    // Turn about +y, radians.
    yaw: f32,
    // The clip playing, and the one it fades in over.
    clip: u32,
    previous_clip: u32,
    // Seconds the fade from previous_clip to clip takes (0: cut).
    fade: f32,
    scale: f32,
    // Clock times `clip` and `previous_clip` started, and the fade began.
    start: f32,
    previous_start: f32,
    fade_start: f32,
    // Playback rate (1 = as authored).
    speed: f32,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> palette: Palette;
@group(0) @binding(2) var<storage, read> clips: array<Clip>;
@group(0) @binding(3) var<storage, read> bones: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> units: array<Unit>;

struct Bone {
    rotation: vec4<f32>,
    translation: vec3<f32>,
    scale: f32,
};

fn decode_bone(frame: u32, joint: u32) -> Bone {
    let w = bones[frame * palette.joint_count + joint];
    let q = vec4<f32>(unpack2x16snorm(w.x), unpack2x16snorm(w.y));
    let t = vec3<f32>(unpack2x16unorm(w.z), unpack2x16unorm(w.w).x);
    return Bone(normalize(q), palette.translation_min + t * palette.translation_extent, unpack2x16float(w.w).y);
}

fn blend_bones(a: Bone, b: Bone, t: f32) -> Bone {
    var qb = b.rotation;
    if dot(a.rotation, qb) < 0.0 {
        qb = -qb;
    }
    return Bone(normalize(mix(a.rotation, qb, t)), mix(a.translation, b.translation, t), mix(a.scale, b.scale, t));
}

struct Frames {
    f0: u32,
    f1: u32,
    t: f32,
};

// The two frames around `time` seconds into a clip (farthing's PaletteClip::frames_at).
fn frames_at(clip: Clip, time: f32) -> Frames {
    let n = max(clip.frame_count, 1u);
    if n == 1u || clip.duration <= 0.0 {
        return Frames(clip.first_frame, clip.first_frame, 0.0);
    }
    let last = f32(n - 1u);
    var x = time / clip.duration * last;
    if (clip.flags & 1u) != 0u {
        x = x - floor(x / last) * last;
    } else {
        x = clamp(x, 0.0, last);
    }
    let f0 = min(u32(floor(x)), n - 1u);
    let f1 = min(f0 + 1u, n - 1u);
    return Frames(clip.first_frame + f0, clip.first_frame + f1, x - f32(f0));
}

fn sample_bone(f: Frames, joint: u32) -> Bone {
    return blend_bones(decode_bone(f.f0, joint), decode_bone(f.f1, joint), f.t);
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

struct VsIn {
    @builtin(instance_index) instance: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<u32>, // Uint16x4
    @location(3) weights: vec4<f32>, // Unorm8x4
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let unit = units[v.instance];
    let a = frames_at(clips[unit.clip], (view.time - unit.start) * unit.speed);
    let b = frames_at(clips[unit.previous_clip], (view.time - unit.previous_start) * unit.speed);
    var fade = 1.0;
    if unit.fade > 0.0 {
        fade = clamp((view.time - unit.fade_start) / unit.fade, 0.0, 1.0);
    }
    // Linear blend skinning with up to four joints; weights add up to 1.
    var p = vec3<f32>(0.0);
    var n = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let w = v.weights[k];
        if w > 0.0 {
            var bone = sample_bone(a, v.joints[k]);
            if fade < 1.0 {
                bone = blend_bones(sample_bone(b, v.joints[k]), bone, fade);
            }
            p += w * (rotate(bone.rotation, v.position * bone.scale) + bone.translation);
            n += w * rotate(bone.rotation, v.normal);
        }
    }
    let c = cos(unit.yaw);
    let s = sin(unit.yaw);
    let turn = mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));
    var out: VsOut;
    out.clip = view.view_proj * vec4<f32>(unit.position + turn * (p * unit.scale), 1.0);
    out.normal = turn * n;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let lambert = max(dot(normalize(in.normal), -view.light_dir), 0.0);
    return vec4<f32>(vec3<f32>(0.2 + 0.8 * lambert), 1.0);
}
