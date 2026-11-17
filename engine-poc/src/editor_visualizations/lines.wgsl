// Overlay lines: each segment is an instanced quad of a constant width in pixels, in the
// overlay's flat color (from the frame, so the page can follow light and dark mode).

struct Out {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
}

fn line(p0: vec3f, p1: vec3f, corner: u32, color: vec4f, width: f32) -> Out {
    var out: Out;
    out.color = vec4f(color.rgb * color.a, color.a);
    var a = frame.view_proj * obj.model * vec4f(p0, 1.0);
    var b = frame.view_proj * obj.model * vec4f(p1, 1.0);
    // Clip to the near plane (z = 0 in WebGPU's clip space) so both ends project.
    if a.z < 0.0 && b.z < 0.0 {
        out.clip = vec4f(2.0, 2.0, 2.0, 1.0); // outside the view: dropped
        return out;
    }
    if a.z < 0.0 {
        a = mix(a, b, a.z / (a.z - b.z));
    } else if b.z < 0.0 {
        b = mix(b, a, b.z / (b.z - a.z));
    }
    let half_px = frame.viewport.xy * 0.5;
    let d = b.xy / b.w * half_px - a.xy / a.w * half_px;
    let len = length(d);
    let dir = select(vec2f(1.0, 0.0), d / len, len > 1e-6);
    // Two triangles: corners 1, 2 and 4 are at the far end, 2, 4 and 5 on the left side.
    let far = ((22u >> corner) & 1u) == 1u;
    let side = select(-1.0, 1.0, ((52u >> corner) & 1u) == 1u);
    let p = select(a, b, far);
    // Out to the side, and past the end by half the width so joints close.
    let offset = (vec2f(-dir.y, dir.x) * side + dir * select(-1.0, 1.0, far)) * (width * 0.5) / half_px;
    out.clip = vec4f(p.xy + offset * p.w, p.z, p.w);
    return out;
}

@vertex
fn vs_wire(@builtin(vertex_index) corner: u32, @location(0) p0: vec3f, @location(1) p1: vec3f) -> Out {
    return line(p0, p1, corner, frame.wire, frame.widths.x);
}

@vertex
fn vs_seam(@builtin(vertex_index) corner: u32, @location(0) p0: vec3f, @location(1) p1: vec3f) -> Out {
    return line(p0, p1, corner, frame.seam, frame.widths.y);
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
    return in.color;
}
