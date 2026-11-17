// What the cluster cull (compute) and draw (vertex) passes share: the layout of the scene,
// state and instance buffers, and the cluster and page decoding. Each pass declares the
// bindings itself; see `clusters.rs` for how the buffers are built.
//
// `scene` (u32 words, rebuilt when the cluster meshes change):
//   header: mesh count, mesh array offset, cluster array offset, bin count, largest cluster
//   triangle count, ready-flag offset in `state`, region offset in `state`, words per pool slot
//   mesh records (16 words each, see `Mesh`)
//   cluster records (32 words each): the `.fcld` cluster header, except that `page`, `group`
//   and `parent group` stay mesh-local and the reserved word holds the cluster's material slot
//   bits (all set when the mesh has more than 32 slots)
// `state` (u32 words): per page its pool slot (NONE: not resident), per group 1 when it is
//   ready (every page its refinement needs is resident), then per view and bin the region of
//   `visible` it appends to (offset, capacity).

const NONE: u32 = 0xffffffffu;
const CLUSTER_WORDS: u32 = 32u;
const MESH_WORDS: u32 = 16u;
// Per view in `counters`, after the indirect arguments: clusters in the cut, clusters drawn,
// triangles drawn, (unused), then clusters drawn per DAG level (the last level holds the rest).
const STATS: u32 = 20u;
const STAT_LEVELS: u32 = 16u;
// Stands in for an infinite projected error.
const BIG: f32 = 3.0e38;

struct Instance {
    model: mat4x4f,
    // The inverse transpose; normal[3].w is the sign of the determinant.
    normal: mat4x4f,
    // World to model.
    inv: mat4x4f,
    // The mesh record.
    mesh: u32,
    // Draw the wireframe over it.
    wire: u32,
    // The largest axis scale.
    scale: f32,
    pad: u32,
}

struct Mesh {
    origin: vec3f,
    step: f32,
    uv_min: vec2f,
    uv_step: vec2f,
    cluster_first: u32,
    cluster_count: u32,
    page_first: u32,
    group_first: u32,
    flags: u32,
    // Its first bin: slot s draws in bin slot_first + s.
    slot_first: u32,
    slot_count: u32,
    // The slots whose materials are double-sided (never back-face culled).
    double_mask: u32,
}

fn mesh_at(m: u32) -> Mesh {
    let o = scene[1] + m * MESH_WORDS;
    var r: Mesh;
    r.origin = vec3f(bitcast<f32>(scene[o]), bitcast<f32>(scene[o + 1u]), bitcast<f32>(scene[o + 2u]));
    r.step = bitcast<f32>(scene[o + 3u]);
    r.uv_min = vec2f(bitcast<f32>(scene[o + 4u]), bitcast<f32>(scene[o + 5u]));
    r.uv_step = vec2f(bitcast<f32>(scene[o + 6u]), bitcast<f32>(scene[o + 7u]));
    r.cluster_first = scene[o + 8u];
    r.cluster_count = scene[o + 9u];
    r.page_first = scene[o + 10u];
    r.group_first = scene[o + 11u];
    r.flags = scene[o + 12u];
    r.slot_first = scene[o + 13u];
    r.slot_count = scene[o + 14u];
    r.double_mask = scene[o + 15u];
    return r;
}

// Word `k` of cluster `c`'s record.
fn cw(c: u32, k: u32) -> u32 {
    return scene[scene[2] + c * CLUSTER_WORDS + k];
}

fn cf(c: u32, k: u32) -> f32 {
    return bitcast<f32>(cw(c, k));
}

// The sphere at word `k` of cluster `c`'s record: center, radius.
fn sphere(c: u32, k: u32) -> vec4f {
    return vec4f(cf(c, k), cf(c, k + 1u), cf(c, k + 2u), cf(c, k + 3u));
}

// The projected size of `error` at the sphere's nearest point, in pixels (farthing's
// `LodView::projected`, with BIG for infinity).
fn projected(error: f32, s: vec4f, eye: vec3f, ppr: f32) -> f32 {
    if error <= 0.0 {
        return 0.0;
    }
    if (bitcast<u32>(error) & 0x7f800000u) == 0x7f800000u {
        return BIG;
    }
    let d = distance(s.xyz, eye) - s.w;
    if d <= 0.0 {
        return BIG;
    }
    return error * ppr / d;
}

fn page_slot(m: Mesh, page: u32) -> u32 {
    return state[m.page_first + page];
}

fn group_ready(m: Mesh, group: u32) -> bool {
    return state[scene[5] + m.group_first + group] != 0u;
}
