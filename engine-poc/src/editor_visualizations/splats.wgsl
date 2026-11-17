// Splats: a surfel cloud drawn as camera-facing quads, one instance per surfel. Each carries
// its center, its 3D covariance (the upper triangle: xx xy xz, yy yz zz), a linear RGBA color and
// a normal. The covariance is projected to a 2D ellipse on screen (EWA, as 3D Gaussian splatting
// does) and the quad spans its axes.
//
// - fs_disc: opaque discs that write depth, so order doesn't matter.
// - fs_gauss: Gaussian falloff, blended; the caller draws them far to near.

struct Splat {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
    // Where in the quad: ±2 at the edges, in standard deviations.
    @location(1) pos: vec2f,
}

const QUAD = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));

@vertex
fn vs_splat(@builtin(vertex_index) corner: u32, @location(0) center: vec3f, @location(1) cov_a: vec3f, @location(2) cov_b: vec3f, @location(3) rgba: vec4f, @location(4) nrm: vec3f) -> Splat {
    var out: Splat;
    let model_view = frame.view * obj.model;
    let cam = model_view * vec4f(center, 1.0);
    let clip = frame.view_proj * obj.model * vec4f(center, 1.0);
    if cam.z > -1e-6 || abs(clip.x) > 1.3 * clip.w || abs(clip.y) > 1.3 * clip.w {
        out.clip = vec4f(0.0, 0.0, 2.0, 1.0); // behind or well off screen: dropped
        return out;
    }
    let v = mat3x3f(cov_a.x, cov_a.y, cov_a.z, cov_a.y, cov_b.x, cov_b.y, cov_a.z, cov_b.y, cov_b.z);
    let w = mat3x3f(model_view[0].xyz, model_view[1].xyz, model_view[2].xyz);
    let f = frame.focal.xy;
    let iz = 1.0 / cam.z;
    // The projection's Jacobian at the center; screen y points up, as in clip space.
    let j = mat3x3f(-f.x * iz, 0.0, 0.0, 0.0, -f.y * iz, 0.0, f.x * cam.x * iz * iz, f.y * cam.y * iz * iz, 0.0);
    let t = j * w;
    let s = t * v * transpose(t);
    // Widened by 0.3 px² so no splat is thinner than a pixel.
    let a = s[0][0] + 0.3;
    let b = s[0][1];
    let c = s[1][1] + 0.3;
    let mid = 0.5 * (a + c);
    let rad = length(vec2f(0.5 * (a - c), b));
    let l1 = mid + rad;
    let l2 = max(mid - rad, 0.1);
    var d1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), a >= c);
    if abs(b) > 1e-9 {
        d1 = normalize(vec2f(b, l1 - a));
    }
    let major = min(sqrt(2.0 * l1), 1024.0) * d1;
    let minor = min(sqrt(2.0 * l2), 1024.0) * vec2f(-d1.y, d1.x); // keeps the quad counter-clockwise
    out.pos = QUAD[corner] * 2.0;
    let off = (out.pos.x * major + out.pos.y * minor) * 2.0 * frame.viewport.zw;
    out.clip = vec4f((clip.xy / clip.w + off) * clip.w, clip.z, clip.w);
    let n = normalize(w * nrm);
    let lit = 0.55 + 0.45 * abs(dot(n, normalize(cam.xyz)));
    out.color = vec4f(rgba.rgb * lit, rgba.a);
    return out;
}

@fragment
fn fs_disc(in: Splat) -> @location(0) vec4f {
    if dot(in.pos, in.pos) > 2.0 {
        discard;
    }
    return vec4f(in.color.rgb, 1.0);
}

@fragment
fn fs_gauss(in: Splat) -> @location(0) vec4f {
    let r2 = dot(in.pos, in.pos);
    let alpha = in.color.a * exp(-r2);
    if r2 > 4.0 || alpha < 1.0 / 255.0 {
        discard;
    }
    return vec4f(in.color.rgb * alpha, alpha);
}
