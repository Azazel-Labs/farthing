// Culls a crowd for one view: one invocation per unit, appending it to the region of
// `visible` of the level of detail its projected size picks. Follows `crowd.wgsl`.

struct CullU {
    view_proj: mat4x4f,
    // The crowd's placement (every unit's position is in it).
    model: mat4x4f,
    // World-space frustum planes (xyz · p + w ≥ 0 inside).
    planes: array<vec4f, 6>,
    // xyz: the eye; w: the focal length in pixels.
    eye: vec4f,
    // Per level: the least projected radius (pixels) it's drawn at; the last level takes the rest.
    lod_pixels: vec4f,
    // Unit count, level count, region size (units), first indirect record.
    counts: vec4u,
    // x: the unit bounding radius (unit space), y: the model's largest axis scale, z: whether
    // to frustum cull.
    params: vec4f,
}

@group(0) @binding(0) var<storage, read> cull_units: array<Unit>;
@group(0) @binding(1) var<storage, read_write> cull_visible: array<u32>;
@group(0) @binding(2) var<storage, read_write> cull_args: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cull: CullU;

@compute @workgroup_size(64)
fn cull_crowd(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= cull.counts.x {
        return;
    }
    let u = cull_units[i];
    let center = (cull.model * vec4f(u.position, 1.0)).xyz;
    let radius = cull.params.x * u.scale * cull.params.y;
    if cull.params.z != 0.0 {
        for (var p = 0u; p < 6u; p++) {
            if dot(cull.planes[p].xyz, center) + cull.planes[p].w < -radius {
                return;
            }
        }
    }
    let distance = max(length(center - cull.eye.xyz), 1e-4);
    let pixels = radius * cull.eye.w / distance;
    let levels = min(cull.counts.y, MAX_LODS);
    var lod = levels - 1u;
    for (var k = 0u; k + 1u < levels; k++) {
        if pixels >= cull.lod_pixels[k] {
            lod = k;
            break;
        }
    }
    let record = cull.counts.w + lod;
    let slot = atomicAdd(&cull_args[record * 5u + 1u], 1u);
    if slot < cull.counts.z {
        cull_visible[record * cull.counts.z + slot] = i;
    }
}
