// Crowds: units animated from a farthing pose palette (FANM, `farthing::anim::bake`), culled
// and given a level of detail on the GPU, then drawn with one indirect instanced draw per level.
// Adapted from farthing's reference shader (`farthing::anim::bake::REFERENCE_SHADER`): a unit's
// whole animation state is its record, so the CPU writes one only when the unit changes clips
// or moves, and the shader finds every joint's transform from the clock.

const MAX_LODS: u32 = 4u;

// 48 bytes, as `CrowdUnit`.
struct Unit {
    position: vec3f,
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
}

// FANM header bytes 16..64.
struct Palette {
    joint_count: u32,
    clip_count: u32,
    frame_count: u32,
    fps: f32,
    translation_min: vec3f,
    clip_offset: u32,
    translation_extent: vec3f,
    frame_offset: u32,
}

struct Clip {
    first_frame: u32,
    frame_count: u32,
    duration: f32,
    // Bit 0: loops.
    flags: u32,
}

struct Frames {
    f0: u32,
    f1: u32,
    t: f32,
}

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
