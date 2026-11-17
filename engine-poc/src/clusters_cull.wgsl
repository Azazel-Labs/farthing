// The cluster LOD cut, on the GPU: one dispatch per view, one thread per (instance, cluster).
// A cluster is drawn when it is the finest resident cluster that is still coarse enough for the
// view (farthing's `GpuClusterDag::select`, restricted to what is resident), is inside the
// frustum and doesn't face away. Drawn clusters are appended to their bins' regions of
// `visible`, counted into the bins' indirect draw arguments, and they record which groups the
// view needs resident, and how badly, in `requests`.

@group(0) @binding(0) var<storage, read> scene: array<u32>;
@group(0) @binding(1) var<storage, read> state: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> visible: array<vec2u>;
// Per view and bin, the indirect draw arguments (vertex count, instance count, 0, 0); then per
// view, STATS words of statistics.
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
// Per group (across all meshes), the most any view needs it: a projected error over the
// threshold, as f32 bits (positive floats order like their bits).
@group(0) @binding(5) var<storage, read_write> requests: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> cv: CullView;

struct CullView {
    // World-space planes, inside where dot(xyz, p) + w >= 0.
    planes: array<vec4f, 6>,
    eye: vec3f,
    // Pixels per radian: viewport height / (2 tan(fov / 2)).
    ppr: f32,
    threshold: f32,
    inst_first: u32,
    inst_count: u32,
    view: u32,
    // 1: frustum culling, 2: cone (back-face) culling, 4: bin by material.
    flags: u32,
    stats_base: u32,
    pad0: u32,
    pad1: u32,
}

fn world_sphere(inst: Instance, s: vec4f) -> vec4f {
    return vec4f((inst.model * vec4f(s.xyz, 1.0)).xyz, s.w * inst.scale);
}

fn request(group: u32, need: f32) {
    atomicMax(&requests[group], bitcast<u32>(min(need, 1.0e30)));
}

fn append(bin: u32, inst: u32, c: u32) {
    let vb = cv.view * scene[3] + bin;
    let region = scene[6] + 2u * vb;
    let i = atomicAdd(&counters[4u * vb + 1u], 1u);
    if i < state[region + 1u] {
        visible[state[region] + i] = vec2u(inst, c);
    }
}

fn visit(ii: u32, inst: Instance, m: Mesh, c: u32) {
    let group = cw(c, 22u);
    let parent_group = cw(c, 23u);
    // Drawable at all: its page and every page its coarser chain needs are resident.
    if parent_group == NONE {
        if page_slot(m, cw(c, 24u)) == NONE {
            return;
        }
    } else if !group_ready(m, parent_group) {
        return;
    }
    // Coarse enough that its parent won't do...
    let pe = projected(cf(c, 20u) * inst.scale, world_sphere(inst, sphere(c, 8u)), cv.eye, cv.ppr);
    if pe <= cv.threshold {
        return;
    }
    // ...and either fine enough, or its refinement isn't resident yet.
    let le = projected(cf(c, 19u) * inst.scale, world_sphere(inst, sphere(c, 4u)), cv.eye, cv.ppr);
    let refine = le > cv.threshold && group != NONE;
    if refine && group_ready(m, group) {
        return;
    }
    let stats = cv.stats_base;
    atomicAdd(&counters[stats], 1u);

    if (cv.flags & 1u) != 0u {
        let b = world_sphere(inst, sphere(c, 0u));
        for (var k = 0u; k < 6u; k++) {
            if dot(cv.planes[k].xyz, b.xyz) + cv.planes[k].w < -b.w {
                return;
            }
        }
    }

    // What this view needs resident: the chain this cluster is drawn from, and its refinement.
    if parent_group != NONE {
        request(m.group_first + parent_group, pe / cv.threshold);
    }
    if refine {
        request(m.group_first + group, le / cv.threshold);
    }

    let slots = cw(c, 31u);
    let cutoff = cf(c, 15u);
    if (cv.flags & 2u) != 0u && (slots & m.double_mask) == 0u && cutoff <= 1.0 {
        // In model space, so a mirrored or non-uniformly scaled instance tests the same way.
        let eye = (inst.inv * vec4f(cv.eye, 1.0)).xyz;
        let d = vec3f(cf(c, 12u), cf(c, 13u), cf(c, 14u)) - eye;
        let dir = select(vec3f(0.0), normalize(d), dot(d, d) > 0.0);
        if dot(dir, vec3f(cf(c, 16u), cf(c, 17u), cf(c, 18u))) >= cutoff {
            return;
        }
    }

    atomicAdd(&counters[stats + 1u], 1u);
    atomicAdd(&counters[stats + 2u], cw(c, 27u) >> 16u);
    atomicAdd(&counters[stats + 4u + min(cw(c, 21u), STAT_LEVELS - 1u)], 1u);
    let material = cw(c, 28u);
    if (cv.flags & 4u) == 0u {
        append(m.slot_first, ii, c);
    } else if material != NONE {
        append(m.slot_first + select(0u, material, material < m.slot_count), ii, c);
    } else {
        for (var s = 0u; s < m.slot_count; s++) {
            if s >= 32u || ((slots >> s) & 1u) != 0u {
                append(m.slot_first + s, ii, c);
            }
        }
    }
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    if gid.y >= cv.inst_count {
        return;
    }
    let ii = cv.inst_first + gid.y;
    let inst = instances[ii];
    let m = mesh_at(inst.mesh);
    for (var k = gid.x; k < m.cluster_count; k += groups.x * 64u) {
        visit(ii, inst, m, m.cluster_first + k);
    }
}
