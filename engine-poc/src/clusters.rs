//! GPU-driven cluster rendering: farthing's packed cluster DAGs (`.fcld`, see
//! `farthing::cluster::gpu`) cut, culled and drawn on the GPU, their pages streamed into a
//! fixed pool.
//!
//! Each frame:
//!
//! 1. **Cut** (`clusters_cull.wgsl`, one compute dispatch per view). Every instance's every
//!    cluster runs farthing's LOD test against that view, limited to what is resident: a cluster
//!    is drawn when its coarser chain is resident, its parent is too coarse, and it is fine
//!    enough or its refinement isn't resident yet. Then it's frustum and normal-cone culled and
//!    appended to its bins' regions of the visible list, counting into the bins' indirect draw
//!    arguments. So the cut follows the camera every frame, with no CPU round trip.
//! 2. **Draw** (`clusters_draw.wgsl`): one indirect draw per view and bin (a mesh's material
//!    slot), one instance per visible cluster. The vertex shader decodes the cluster's triangles
//!    straight from its page in the pool.
//! 3. **Streaming**. The cut also records, per group, the most any view and instance needs it
//!    (its projected error over the threshold): the groups its drawn clusters hang from, and the
//!    groups that would refine them. That list is read back a frame or two later. Need flows to
//!    coarser groups, then to pages: a page's priority is the most any group needing it wants,
//!    and a page's dependencies are at least as urgent as it. Split screen or several
//!    instances therefore stream the union of what every view needs, by the largest need.
//!    Missing pages load highest priority first, each after its missing dependencies, a few per
//!    frame. When the pool is full the page loaded instead is the least urgent, least recently
//!    needed page nothing resident depends on, and only if it's clearly less urgent (1.5×).
//!    Pages holding the roots (and what they depend on) stay resident.
//!
//! A group is *ready* when everything drawing its refinement needs is resident: the pages of
//! the clusters it consumed, the page of the clusters it produced, and readiness of every
//! group that consumes those. The cut only refines into ready groups, so a cut over a partly
//! resident DAG never overlaps itself or leaves holes.

use crate::viewer::{DEPTH, GpuMaterial, SAMPLES};
use bytemuck::{Pod, Zeroable};
use farthing::cluster::gpu::{FLAG_NORMALS, FLAG_UVS, GpuClusterDag};
use farthing::glam::{Mat3, Mat4, Vec3, Vec4};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const NONE: u32 = u32::MAX;
const HEADER_WORDS: usize = 8;
const MESH_WORDS: usize = 16;
const CLUSTER_WORDS: usize = 32;
/// Statistics words per view (see `clusters_common.wgsl`).
const STATS: usize = 20;
pub const STAT_LEVELS: usize = 16;
/// Material slots per mesh; materials past it draw with slot 0's.
const MAX_SLOTS: u32 = 256;
const READBACKS: usize = 3;
/// How much more urgent a page must be than the one it replaces.
const HYSTERESIS: f32 = 1.5;

/// One view of the cluster meshes this frame.
pub(crate) struct ClusterView {
    pub view_proj: Mat4,
    pub eye: Vec3,
    /// Viewport height / (2 tan(fov / 2)).
    pub ppr: f32,
    pub draws: Vec<ClusterDraw>,
}

pub(crate) struct ClusterDraw {
    /// The renderer's mesh id.
    pub mesh: u32,
    pub model: Mat4,
    pub wire: bool,
}

/// What one view's cut drew, from the latest readback.
#[derive(Clone, Debug, Default)]
pub struct ClusterViewStats {
    /// Clusters the LOD cut picked, before culling.
    pub cut: u32,
    /// Clusters drawn after frustum and cone culling.
    pub clusters: u32,
    pub triangles: u32,
    /// Clusters drawn per DAG level (the last counts the rest).
    pub per_level: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct ClusterStats {
    pub views: Vec<ClusterViewStats>,
    pub resident_pages: usize,
    pub total_pages: usize,
    pub resident_bytes: u64,
    pub total_bytes: u64,
    /// The pool's size: pages it holds × the largest page.
    pub pool_bytes: u64,
    pub pool_pages: usize,
    /// Pages some view needs that aren't resident.
    pub requested_pages: usize,
    /// Pages loaded and evicted so far.
    pub loaded: u64,
    pub evicted: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceG {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    inv: [[f32; 4]; 4],
    mesh: u32,
    wire: u32,
    scale: f32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullViewU {
    planes: [[f32; 4]; 6],
    eye: [f32; 3],
    ppr: f32,
    threshold: f32,
    inst_first: u32,
    inst_count: u32,
    view: u32,
    flags: u32,
    stats_base: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BinU {
    region: u32,
    cap: u32,
    slot: u32,
    check_material: u32,
    color_mode: u32,
    pad: [u32; 3],
}

/// A mesh and what streaming needs to know about its DAG.
struct ClusterMesh {
    dag: GpuClusterDag,
    /// The viewer material for each material id (missing: the default).
    materials: Vec<u32>,
    slot_count: u32,
    /// Per cluster, the slots its triangles use (all bits set when it uses one past 31).
    slot_masks: Vec<u32>,
    /// Per slot, how many clusters can draw in it.
    slot_clusters: Vec<u32>,
    /// Per group, the pages of the clusters it consumed.
    child_pages: Vec<Vec<u32>>,
    /// Per group, the groups that consume the clusters it produced.
    consumers: Vec<Vec<u32>>,
    /// Groups, finest level first.
    by_level: Vec<u32>,
    /// Pages that stay resident: the roots' and what they depend on.
    pinned: Vec<bool>,
}

impl ClusterMesh {
    fn new(dag: GpuClusterDag) -> ClusterMesh {
        let ids: Vec<Vec<u32>> = (0..dag.clusters.len())
            .map(|c| {
                let mut m = dag.cluster_materials(c);
                m.sort_unstable();
                m.dedup();
                m
            })
            .collect();
        let slot_count = ids
            .iter()
            .flatten()
            .map(|&m| m.saturating_add(1))
            .max()
            .unwrap_or(1)
            .clamp(1, MAX_SLOTS);
        let slot = |m: u32| if m < slot_count { m } else { 0 };
        let mut slot_masks = Vec::with_capacity(ids.len());
        let mut slot_clusters = vec![0u32; slot_count as usize];
        for m in &ids {
            let mut slots: Vec<u32> = m.iter().map(|&m| slot(m)).collect();
            slots.sort_unstable();
            slots.dedup();
            let mixed = dag.clusters[slot_masks.len()].material == NONE;
            if mixed && slot_count > 32 {
                // The cull pass appends it to every slot.
                slot_clusters.iter_mut().for_each(|n| *n += 1);
            } else {
                for &s in &slots {
                    slot_clusters[s as usize] += 1;
                }
            }
            slot_masks.push(if slots.iter().any(|&s| s >= 32) {
                u32::MAX
            } else {
                slots.iter().fold(0, |b, &s| b | 1 << s)
            });
        }
        let groups = dag.groups.len();
        let mut child_pages = vec![Vec::new(); groups];
        let mut consumers = vec![Vec::new(); groups];
        let mut pinned = vec![false; dag.pages.len()];
        for c in &dag.clusters {
            if c.parent_group == NONE {
                pinned[c.page as usize] = true;
            } else {
                child_pages[c.parent_group as usize].push(c.page);
                if c.group != NONE {
                    consumers[c.group as usize].push(c.parent_group);
                }
            }
        }
        for v in child_pages.iter_mut().chain(consumers.iter_mut()) {
            v.sort_unstable();
            v.dedup();
        }
        // Dependencies have lower indices.
        for p in (0..dag.pages.len()).rev() {
            if pinned[p] {
                for &d in dag.page_dependencies(p) {
                    pinned[d as usize] = true;
                }
            }
        }
        let mut by_level: Vec<u32> = (0..groups as u32).collect();
        by_level.sort_by_key(|&g| dag.groups[g as usize].level);
        ClusterMesh {
            dag,
            materials: Vec::new(),
            slot_count,
            slot_masks,
            slot_clusters,
            child_pages,
            consumers,
            by_level,
            pinned,
        }
    }
}

/// The meshes laid out in GPU buffers, and their residency.
struct Built {
    generation: u64,
    /// Mesh ids in record order.
    order: Vec<usize>,
    /// Per mesh id, its record (NONE: none).
    record: Vec<u32>,
    page_first: Vec<u32>,
    group_first: Vec<u32>,
    cluster_first: Vec<u32>,
    /// Per bin: its record and slot.
    bins: Vec<(u32, u32)>,
    pages: usize,
    groups: usize,
    max_clusters: u32,
    max_tris: u32,
    scene: wgpu::Buffer,
    requests: wgpu::Buffer,
    pool: wgpu::Buffer,
    slot_bytes: u64,
    pool_slots: u32,
    /// Per global page.
    page_slot: Vec<u32>,
    owner: Vec<(u32, u32)>,
    prio: Vec<f32>,
    last_used: Vec<u64>,
    dependents: Vec<u32>,
    pinned: Vec<bool>,
    free: Vec<u32>,
    /// Per global group.
    ready: Vec<bool>,
    requested: usize,
}

enum Fetch {
    Idle,
    Copied,
    Mapping,
}

struct Readback {
    buffer: wgpu::Buffer,
    fetch: Fetch,
    done: Arc<AtomicBool>,
    frame: u64,
    generation: u64,
    views: usize,
    groups: usize,
}

/// One indirect draw: a view's bin.
struct BinDraw {
    /// Its record in the bin uniforms and the draw arguments: view × bins + bin.
    record: usize,
    pipe: usize,
    material: u32,
    blend: bool,
}

#[derive(Default)]
struct FrameBuffers {
    state: Option<wgpu::Buffer>,
    instances: Option<wgpu::Buffer>,
    visible: Option<wgpu::Buffer>,
    counters: Option<wgpu::Buffer>,
    cull_u: Option<wgpu::Buffer>,
    bin_u: Option<wgpu::Buffer>,
    binds: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
}

pub(crate) struct ClusterRenderer {
    cull: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    draw_bgl: wgpu::BindGroupLayout,
    /// Indexed by blend × 2 + back-face culling.
    surfaces: Vec<wgpu::RenderPipeline>,
    pub(crate) default_material: u32,
    meshes: Vec<Option<ClusterMesh>>,
    built: Option<Built>,
    scene_dirty: bool,
    threshold: f32,
    frustum: bool,
    cone: bool,
    freeze: bool,
    /// 0: the meshes' materials, 1: a color per cluster, 2: a color per level.
    color_mode: u32,
    color_material: Option<u32>,
    budget: u64,
    pages_per_frame: usize,
    frozen: Vec<CullViewU>,
    frame: u64,
    generation: u64,
    buffers: FrameBuffers,
    readbacks: Vec<Readback>,
    /// Per view, this frame's draws.
    draws: Vec<Vec<BinDraw>>,
    /// Per view and bin, this frame's region of `visible` (offset, capacity).
    regions: Vec<u32>,
    /// Per instance this frame: the first instance of its view, and its mesh record.
    instance_log: Vec<(u32, u32)>,
    cull_stride: u64,
    bin_stride: u64,
    view_stats: Vec<ClusterViewStats>,
    loaded: u64,
    evicted: u64,
}

fn storage(binding: u32, read_only: bool, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dynamic_uniform(binding: u32, visibility: wgpu::ShaderStages, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

/// The largest buffer the renderer binds.
fn buffer_limit(device: &wgpu::Device) -> u64 {
    let l = device.limits();
    l.max_storage_buffer_binding_size.min(l.max_buffer_size)
}

/// Makes `slot` a buffer of at least `size` bytes; true when it made a new one.
fn ensure(
    device: &wgpu::Device,
    slot: &mut Option<wgpu::Buffer>,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> bool {
    let size = size.max(16).next_multiple_of(16);
    if slot.as_ref().is_some_and(|b| b.size() >= size) {
        return false;
    }
    let size = size.next_power_of_two().min(buffer_limit(device)).max(size);
    *slot = Some(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }));
    true
}

/// Records `stride` bytes apart.
fn records<T: Pod>(items: &[T], stride: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; items.len().max(1) * stride as usize];
    for (i, r) in items.iter().enumerate() {
        let b = bytemuck::bytes_of(r);
        bytes[i * stride as usize..][..b.len()].copy_from_slice(b);
    }
    bytes
}

/// The view's frustum planes from its view-projection matrix (WebGPU depth, 0 to 1).
fn frustum_planes(m: Mat4) -> [[f32; 4]; 6] {
    let r = [m.row(0), m.row(1), m.row(2), m.row(3)];
    [r[3] + r[0], r[3] - r[0], r[3] + r[1], r[3] - r[1], r[2], r[3] - r[2]].map(|p: Vec4| {
        let l = p.truncate().length();
        (if l > 0.0 { p / l } else { p }).to_array()
    })
}

/// Whether this device can run the cluster renderer.
pub(crate) fn supported(device: &wgpu::Device) -> bool {
    let l = device.limits();
    l.max_storage_buffers_per_shader_stage >= 6
        && l.max_compute_workgroups_per_dimension >= 1
        && l.max_storage_buffer_binding_size > 0
        && l.max_compute_invocations_per_workgroup >= 64
}

impl ClusterRenderer {
    pub(crate) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        frames: &wgpu::BindGroupLayout,
        models: &wgpu::BindGroupLayout,
        material: &wgpu::BindGroupLayout,
        default_material: u32,
    ) -> ClusterRenderer {
        let compute = wgpu::ShaderStages::COMPUTE;
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster cull"),
            entries: &[
                storage(0, true, compute),
                storage(1, true, compute),
                storage(2, true, compute),
                storage(3, false, compute),
                storage(4, false, compute),
                storage(5, false, compute),
                dynamic_uniform(6, compute, size_of::<CullViewU>() as u64),
            ],
        });
        let vertex = wgpu::ShaderStages::VERTEX;
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster draw"),
            entries: &[
                storage(0, true, vertex),
                storage(1, true, vertex),
                storage(2, true, vertex),
                storage(3, true, vertex),
                storage(4, true, vertex),
                dynamic_uniform(5, vertex, size_of::<BinU>() as u64),
            ],
        });
        let common = include_str!("clusters_common.wgsl");
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster cull"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{common}\n{}",
                include_str!("clusters_cull.wgsl")
            ))),
        });
        let cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster cull"),
            layout: Some(&cull_layout),
            module: &cull_shader,
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster draw"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{}\n{}\n{common}\n{}",
                include_str!("common.wgsl"),
                include_str!("pbr.wgsl"),
                include_str!("clusters_draw.wgsl")
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(frames), Some(models), Some(material), Some(&draw_bgl)],
            immediate_size: 0,
        });
        let mut surfaces = Vec::new();
        for blend in [false, true] {
            for cull in [None, Some(wgpu::Face::Back)] {
                surfaces.push(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("cluster surface"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &draw_shader,
                        entry_point: Some("vs_cluster"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        cull_mode: cull,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH,
                        depth_write_enabled: Some(!blend),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        bias: wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 1.5,
                            clamp: 0.0,
                        },
                    }),
                    multisample: wgpu::MultisampleState {
                        count: SAMPLES,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &draw_shader,
                        entry_point: Some("fs_cluster"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                }));
            }
        }
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        ClusterRenderer {
            cull,
            cull_bgl,
            draw_bgl,
            surfaces,
            default_material,
            meshes: Vec::new(),
            built: None,
            scene_dirty: false,
            threshold: 1.0,
            frustum: true,
            cone: true,
            freeze: false,
            color_mode: 0,
            color_material: None,
            budget: 64 << 20,
            pages_per_frame: 16,
            frozen: Vec::new(),
            frame: 0,
            generation: 0,
            buffers: FrameBuffers::default(),
            readbacks: Vec::new(),
            draws: Vec::new(),
            regions: Vec::new(),
            instance_log: Vec::new(),
            cull_stride: (size_of::<CullViewU>() as u64).next_multiple_of(align),
            bin_stride: (size_of::<BinU>() as u64).next_multiple_of(align),
            view_stats: Vec::new(),
            loaded: 0,
            evicted: 0,
        }
    }

    pub(crate) fn add_mesh(&mut self, dag: GpuClusterDag) -> u32 {
        let m = ClusterMesh::new(dag);
        self.built = None;
        match self.meshes.iter().position(Option::is_none) {
            Some(i) => {
                self.meshes[i] = Some(m);
                i as u32
            }
            None => {
                self.meshes.push(Some(m));
                self.meshes.len() as u32 - 1
            }
        }
    }

    pub(crate) fn remove_mesh(&mut self, id: u32) {
        if let Some(m) = self.meshes.get_mut(id as usize) {
            *m = None;
            self.built = None;
        }
    }

    /// The viewer material for each of the mesh's material ids.
    pub(crate) fn set_materials(&mut self, id: u32, materials: &[u32]) {
        if let Some(Some(m)) = self.meshes.get_mut(id as usize) {
            m.materials = materials.to_vec();
            self.scene_dirty = true;
        }
    }

    pub(crate) fn set_shading(&mut self, mode: u32, color_material: Option<u32>) {
        self.color_mode = mode.min(2);
        self.color_material = color_material;
    }

    pub(crate) fn set_threshold(&mut self, px: f32) {
        self.threshold = if px.is_finite() { px.max(0.01) } else { 1.0 };
    }

    /// The pool's size in bytes and how many pages may load per frame. A new size starts
    /// streaming over.
    pub(crate) fn set_streaming(&mut self, budget: u64, pages_per_frame: usize) {
        if budget != self.budget {
            self.built = None;
        }
        self.budget = budget;
        self.pages_per_frame = pages_per_frame.max(1);
    }

    pub(crate) fn set_culling(&mut self, frustum: bool, cone: bool) {
        self.frustum = frustum;
        self.cone = cone;
    }

    /// Keeps cutting and culling from the cameras as they are now.
    pub(crate) fn set_freeze(&mut self, on: bool) {
        if !on {
            self.frozen.clear();
        }
        self.freeze = on;
    }

    pub(crate) fn stats(&self) -> ClusterStats {
        let mut s = ClusterStats {
            views: self.view_stats.clone(),
            loaded: self.loaded,
            evicted: self.evicted,
            ..Default::default()
        };
        let Some(b) = &self.built else {
            return s;
        };
        s.total_pages = b.pages;
        s.pool_pages = b.pool_slots as usize;
        s.pool_bytes = u64::from(b.pool_slots) * b.slot_bytes;
        s.requested_pages = b.requested;
        for (p, &(r, local)) in b.owner.iter().enumerate() {
            let Some(Some(m)) = self.meshes.get(b.order[r as usize]) else {
                continue;
            };
            let size = u64::from(m.dag.pages[local as usize].size);
            s.total_bytes += size;
            if b.page_slot[p] != NONE {
                s.resident_pages += 1;
                s.resident_bytes += size;
            }
        }
        s
    }

    /// The scene buffer's words: header, mesh records, cluster records.
    fn scene_words(&self, b: &Built, materials: &[Option<GpuMaterial>]) -> Vec<u32> {
        let clusters: usize = b
            .order
            .iter()
            .filter_map(|&id| self.meshes[id].as_ref())
            .map(|m| m.dag.clusters.len())
            .sum();
        let mesh_base = HEADER_WORDS;
        let cluster_base = mesh_base + MESH_WORDS * b.order.len();
        let mut w = Vec::with_capacity(cluster_base + CLUSTER_WORDS * clusters);
        w.extend([
            b.order.len() as u32,
            mesh_base as u32,
            cluster_base as u32,
            b.bins.len() as u32,
            b.max_tris,
            b.pages as u32,
            (b.pages + b.groups) as u32,
            (b.slot_bytes / 4) as u32,
        ]);
        let f = f32::to_bits;
        let mut slot_first = 0u32;
        for (r, &id) in b.order.iter().enumerate() {
            let m = self.meshes[id].as_ref().expect("built mesh");
            let d = &m.dag;
            let double_mask = (0..m.slot_count.min(32)).fold(0u32, |mask, s| {
                let id = m.materials.get(s as usize).copied().unwrap_or(self.default_material);
                let double = materials
                    .get(id as usize)
                    .and_then(Option::as_ref)
                    .is_some_and(|g| g.double_sided);
                mask | (u32::from(double) << s)
            });
            w.extend([
                f(d.origin.x),
                f(d.origin.y),
                f(d.origin.z),
                f(d.step),
                f(d.uv_min.x),
                f(d.uv_min.y),
                f(d.uv_step.x),
                f(d.uv_step.y),
                b.cluster_first[r],
                d.clusters.len() as u32,
                b.page_first[r],
                b.group_first[r],
                d.flags & (FLAG_NORMALS | FLAG_UVS),
                slot_first,
                m.slot_count,
                double_mask,
            ]);
            slot_first += m.slot_count;
        }
        for &id in &b.order {
            let m = self.meshes[id].as_ref().expect("built mesh");
            for (c, cl) in m.dag.clusters.iter().enumerate() {
                let s = |s: &farthing::cluster::Sphere| {
                    [f(s.center.x), f(s.center.y), f(s.center.z), f(s.radius)]
                };
                w.extend(s(&cl.bounds));
                w.extend(s(&cl.lod_bounds));
                w.extend(s(&cl.parent_bounds));
                w.extend([
                    f(cl.cone.apex.x),
                    f(cl.cone.apex.y),
                    f(cl.cone.apex.z),
                    f(cl.cone.cutoff),
                    f(cl.cone.axis.x),
                    f(cl.cone.axis.y),
                    f(cl.cone.axis.z),
                    f(cl.lod_error),
                    f(cl.parent_error),
                    cl.level,
                    cl.group,
                    cl.parent_group,
                    cl.page,
                    cl.vertex_offset,
                    cl.triangle_offset,
                    cl.vertex_count | cl.triangle_count << 16,
                    cl.material,
                    cl.material_offset,
                    cl.source_cluster,
                    m.slot_masks[c],
                ]);
            }
        }
        w
    }

    /// Lays the meshes out and starts streaming over: only the pinned pages resident.
    fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, materials: &[Option<GpuMaterial>]) {
        self.generation += 1;
        self.frozen.clear();
        let order: Vec<usize> = (0..self.meshes.len())
            .filter(|&i| self.meshes[i].is_some())
            .collect();
        let mut record = vec![NONE; self.meshes.len()];
        let (mut page_first, mut group_first, mut cluster_first) = (Vec::new(), Vec::new(), Vec::new());
        let (mut pages, mut groups, mut clusters) = (0u32, 0u32, 0u32);
        let mut bins = Vec::new();
        let mut owner = Vec::new();
        let mut pinned = Vec::new();
        let (mut max_clusters, mut max_tris, mut slot_bytes) = (0u32, 0u32, 16u64);
        for (r, &id) in order.iter().enumerate() {
            let m = self.meshes[id].as_ref().expect("mesh");
            let d = &m.dag;
            record[id] = r as u32;
            page_first.push(pages);
            group_first.push(groups);
            cluster_first.push(clusters);
            bins.extend((0..m.slot_count).map(|s| (r as u32, s)));
            owner.extend((0..d.pages.len() as u32).map(|p| (r as u32, p)));
            pinned.extend_from_slice(&m.pinned);
            pages += d.pages.len() as u32;
            groups += d.groups.len() as u32;
            clusters += d.clusters.len() as u32;
            max_clusters = max_clusters.max(d.clusters.len() as u32);
            max_tris = max_tris.max(d.clusters.iter().map(|c| c.triangle_count).max().unwrap_or(0));
            slot_bytes = slot_bytes.max(d.page_data.iter().map(|p| p.len() as u64).max().unwrap_or(0));
        }
        let slot_bytes = slot_bytes.next_multiple_of(16);
        let pinned_count = pinned.iter().filter(|&&p| p).count() as u64;
        let limit = buffer_limit(device);
        let pool_slots = (self.budget / slot_bytes)
            .max(pinned_count + 16)
            .min(u64::from(pages).max(1))
            .min(limit / slot_bytes)
            .max(1) as u32;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster pages"),
            size: u64::from(pool_slots) * slot_bytes,
            usage,
            mapped_at_creation: false,
        });
        let requests = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster requests"),
            size: (u64::from(groups) * 4).max(16).next_multiple_of(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pages = pages as usize;
        let mut b = Built {
            generation: self.generation,
            order,
            record,
            page_first,
            group_first,
            cluster_first,
            bins,
            pages,
            groups: groups as usize,
            max_clusters,
            max_tris,
            // Replaced below, once the records can be written.
            scene: requests.clone(),
            requests,
            pool,
            slot_bytes,
            pool_slots,
            page_slot: vec![NONE; pages],
            owner,
            prio: vec![0.0; pages],
            last_used: vec![0; pages],
            dependents: vec![0; pages],
            pinned,
            free: (0..pool_slots).rev().collect(),
            ready: vec![false; groups as usize],
            requested: 0,
        };
        let words = self.scene_words(&b, materials);
        b.scene = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster scene"),
            size: (words.len() as u64 * 4).next_multiple_of(16),
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b.scene, 0, bytemuck::cast_slice(&words));
        for p in 0..pages {
            if b.pinned[p] {
                let Some(slot) = b.free.pop() else {
                    break;
                };
                load(&self.meshes, &mut b, queue, p, slot);
            }
        }
        compute_ready(&self.meshes, &mut b);
        self.built = Some(b);
        self.scene_dirty = false;
        self.buffers.binds = None;
        self.view_stats.clear();
    }

    /// Reads back what's arrived; streams by the newest.
    fn collect(&mut self, queue: &wgpu::Queue) {
        let generation = self.built.as_ref().map_or(0, |b| b.generation);
        let mut newest: Option<(u64, Vec<f32>)> = None;
        for rb in &mut self.readbacks {
            if !matches!(rb.fetch, Fetch::Mapping) || !rb.done.load(Ordering::Acquire) {
                continue;
            }
            if rb.generation == generation
                && let Ok(data) = rb.buffer.slice(..).get_mapped_range()
            {
                let words: &[u32] = bytemuck::cast_slice(&data[..(rb.groups + rb.views * STATS) * 4]);
                if newest.as_ref().is_none_or(|(f, _)| rb.frame > *f) {
                    let need = words[..rb.groups].iter().map(|&b| f32::from_bits(b)).collect();
                    newest = Some((rb.frame, need));
                    self.view_stats = words[rb.groups..]
                        .as_chunks::<STATS>()
                        .0
                        .iter()
                        .map(|s| ClusterViewStats {
                            cut: s[0],
                            clusters: s[1],
                            triangles: s[2],
                            per_level: s[4..4 + STAT_LEVELS].to_vec(),
                        })
                        .collect();
                }
            }
            rb.buffer.unmap();
            rb.fetch = Fetch::Idle;
        }
        if let Some((_, need)) = newest {
            self.stream(queue, need);
        }
    }

    /// Updates priorities from the views' needs, then loads (and evicts) pages.
    fn stream(&mut self, queue: &wgpu::Queue, mut need: Vec<f32>) {
        let frame = self.frame;
        let ppf = self.pages_per_frame;
        let meshes = &self.meshes;
        let Some(b) = self.built.as_mut() else {
            return;
        };
        if need.len() != b.groups {
            return;
        }
        b.prio.fill(0.0);
        for (r, &id) in b.order.iter().enumerate() {
            let m = meshes[id].as_ref().expect("built mesh");
            let (gf, pf) = (b.group_first[r] as usize, b.page_first[r] as usize);
            // A group needs its coarser chain: finest first, so need flows all the way up.
            for &g in &m.by_level {
                let n = need[gf + g as usize];
                if n > 0.0 {
                    for &q in &m.consumers[g as usize] {
                        let q = &mut need[gf + q as usize];
                        *q = q.max(n);
                    }
                }
            }
            for (g, group) in m.dag.groups.iter().enumerate() {
                let n = need[gf + g];
                if n > 0.0 {
                    for &p in m.child_pages[g].iter().chain([&group.page]) {
                        let p = &mut b.prio[pf + p as usize];
                        *p = p.max(n);
                    }
                }
            }
            for p in (0..m.dag.pages.len()).rev() {
                let n = b.prio[pf + p];
                if n > 0.0 {
                    for &d in m.dag.page_dependencies(p) {
                        let d = &mut b.prio[pf + d as usize];
                        *d = d.max(n);
                    }
                }
            }
        }
        for p in 0..b.pages {
            if b.prio[p] > 0.0 {
                b.last_used[p] = frame;
            }
        }
        let mut wanted: Vec<usize> = (0..b.pages)
            .filter(|&p| b.page_slot[p] == NONE && b.prio[p] > 0.0)
            .collect();
        wanted.sort_by(|&x, &y| b.prio[y].total_cmp(&b.prio[x]).then(x.cmp(&y)));
        b.requested = wanted.len();
        let mut touched = vec![false; b.pages];
        let mut loads = 0;
        'pages: for &p in &wanted {
            if loads >= ppf {
                break;
            }
            if b.page_slot[p] != NONE {
                continue;
            }
            let chain = missing_chain(meshes, b, p);
            if loads > 0 && loads + chain.len() > ppf {
                break;
            }
            let urgency = b.prio[p];
            for q in chain {
                let slot = match b.free.pop() {
                    Some(s) => s,
                    None => match victim(b, &touched, urgency) {
                        Some(v) => {
                            self.evicted += 1;
                            evict(meshes, b, v)
                        }
                        None => break 'pages,
                    },
                };
                load(meshes, b, queue, q, slot);
                touched[q] = true;
                loads += 1;
                self.loaded += 1;
            }
        }
        if loads > 0 {
            compute_ready(meshes, b);
        }
    }

    /// Cuts and culls `views` into this frame's draws, recording it into `enc`. Call after
    /// the frame's uniforms are written and before the render pass.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        views: &[ClusterView],
        materials: &[Option<GpuMaterial>],
    ) {
        let _ = device.poll(wgpu::PollType::Poll);
        self.frame += 1;
        self.draws.clear();
        if views.iter().all(|v| v.draws.is_empty()) || self.meshes.iter().all(Option::is_none) {
            return;
        }
        match &self.built {
            None => self.build(device, queue, materials),
            Some(b) if self.scene_dirty => {
                let words = self.scene_words(b, materials);
                queue.write_buffer(&b.scene, 0, bytemuck::cast_slice(&words));
                self.scene_dirty = false;
            }
            Some(_) => {}
        }
        self.collect(queue);
        let b = self.built.as_ref().expect("built");
        let (nv, nb, nr) = (views.len(), b.bins.len(), b.order.len());

        let mut instances = Vec::new();
        let mut log = Vec::new();
        let mut spans = Vec::with_capacity(nv);
        let mut counts = vec![0u64; nv * nr];
        let max_instances = device.limits().max_compute_workgroups_per_dimension as usize;
        for (v, view) in views.iter().enumerate() {
            let first = instances.len();
            for d in &view.draws {
                let r = b.record.get(d.mesh as usize).copied().unwrap_or(NONE);
                if r == NONE || instances.len() - first >= max_instances {
                    continue;
                }
                let det = d.model.determinant();
                let mut normal = Mat4::from_mat3(Mat3::from_mat4(d.model).inverse().transpose());
                normal.w_axis.w = if det < 0.0 { -1.0 } else { 1.0 };
                let scale = d.model.x_axis.truncate().length()
                    .max(d.model.y_axis.truncate().length())
                    .max(d.model.z_axis.truncate().length());
                instances.push(InstanceG {
                    model: d.model.to_cols_array_2d(),
                    normal: normal.to_cols_array_2d(),
                    inv: d.model.inverse().to_cols_array_2d(),
                    mesh: r,
                    wire: u32::from(d.wire),
                    scale,
                    pad: 0,
                });
                counts[v * nr + r as usize] += 1;
                log.push((first as u32, r));
            }
            spans.push((first as u32, (instances.len() - first) as u32));
        }

        // Each view's bins get a region of `visible` big enough for every cluster that can
        // draw in them.
        let by_material = self.color_mode == 0;
        let limit = buffer_limit(device) / 8;
        let mut regions = vec![0u32; nv * nb * 2];
        let mut total = 0u64;
        for v in 0..nv {
            for (bi, &(r, s)) in b.bins.iter().enumerate() {
                let m = self.meshes[b.order[r as usize]].as_ref().expect("built mesh");
                let per = if by_material {
                    m.slot_clusters[s as usize]
                } else if s == 0 {
                    m.dag.clusters.len() as u32
                } else {
                    0
                };
                let cap = (counts[v * nr + r as usize] * u64::from(per)).min(limit - total);
                regions[2 * (v * nb + bi)] = total as u32;
                regions[2 * (v * nb + bi) + 1] = cap as u32;
                total += cap;
            }
        }
        let mut state: Vec<u32> = Vec::with_capacity(b.pages + b.groups + regions.len());
        state.extend_from_slice(&b.page_slot);
        state.extend(b.ready.iter().map(|&r| u32::from(r)));
        state.extend_from_slice(&regions);

        let args = nv * nb * 4;
        let mut counters = vec![0u32; args + nv * STATS];
        for a in counters[..args].as_chunks_mut::<4>().0 {
            a[0] = 3 * b.max_tris;
        }

        let mut culls = Vec::with_capacity(nv);
        let flags = u32::from(self.frustum) | u32::from(self.cone) << 1 | u32::from(by_material) << 2;
        for (v, view) in views.iter().enumerate() {
            let live = CullViewU {
                planes: frustum_planes(view.view_proj),
                eye: view.eye.to_array(),
                ppr: view.ppr,
                ..Zeroable::zeroed()
            };
            if !self.freeze || self.frozen.len() <= v {
                self.frozen.truncate(v);
                self.frozen.push(live);
            }
            let mut u = if self.freeze { self.frozen[v] } else { live };
            u.threshold = self.threshold;
            (u.inst_first, u.inst_count) = spans[v];
            u.view = v as u32;
            u.flags = flags;
            u.stats_base = (args + v * STATS) as u32;
            culls.push(u);
        }

        let mut bin_us = Vec::with_capacity(nv * nb);
        let mut draws = Vec::with_capacity(nv);
        let has = |id: u32| materials.get(id as usize).is_some_and(Option::is_some);
        for v in 0..nv {
            let mut list = Vec::new();
            for (bi, &(r, s)) in b.bins.iter().enumerate() {
                let record = v * nb + bi;
                let m = self.meshes[b.order[r as usize]].as_ref().expect("built mesh");
                bin_us.push(BinU {
                    region: regions[2 * record],
                    cap: regions[2 * record + 1],
                    slot: s,
                    check_material: u32::from(by_material && m.slot_count > 1),
                    color_mode: self.color_mode,
                    pad: [0; 3],
                });
                if regions[2 * record + 1] == 0 {
                    continue;
                }
                let wanted = if by_material {
                    m.materials.get(s as usize).copied()
                } else {
                    self.color_material
                };
                let material = wanted.filter(|&id| has(id)).unwrap_or(self.default_material);
                let Some(Some(g)) = materials.get(material as usize) else {
                    continue;
                };
                // In the debug colors, a mesh with any double-sided material stays two-sided.
                let any_double = (0..m.slot_count).any(|s| {
                    let id = m.materials.get(s as usize).copied().unwrap_or(self.default_material);
                    materials.get(id as usize).and_then(Option::as_ref).is_some_and(|g| g.double_sided)
                });
                let double = g.double_sided || (!by_material && any_double);
                list.push(BinDraw {
                    record,
                    pipe: usize::from(g.blend) * 2 + usize::from(!double),
                    material,
                    blend: g.blend,
                });
            }
            draws.push(list);
        }

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let fb = &mut self.buffers;
        let mut fresh = fb.binds.is_none();
        fresh |= ensure(device, &mut fb.state, state.len() as u64 * 4, storage, "cluster state");
        fresh |= ensure(device, &mut fb.instances, (instances.len() * size_of::<InstanceG>()) as u64, storage, "cluster instances");
        fresh |= ensure(device, &mut fb.visible, total * 8, storage | wgpu::BufferUsages::COPY_SRC, "cluster visible");
        fresh |= ensure(
            device,
            &mut fb.counters,
            counters.len() as u64 * 4,
            storage | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            "cluster counters",
        );
        fresh |= ensure(device, &mut fb.cull_u, nv as u64 * self.cull_stride, uniform, "cluster cull views");
        fresh |= ensure(device, &mut fb.bin_u, (nv * nb) as u64 * self.bin_stride, uniform, "cluster bins");
        if fresh {
            let cull = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cluster cull"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: b.scene.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: frame_buffer(&fb.state).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: frame_buffer(&fb.instances).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: frame_buffer(&fb.visible).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: frame_buffer(&fb.counters).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: b.requests.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: fb.cull_u.as_ref().expect("cull uniforms"),
                            offset: 0,
                            size: wgpu::BufferSize::new(size_of::<CullViewU>() as u64),
                        }),
                    },
                ],
            });
            let draw = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cluster draw"),
                layout: &self.draw_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: b.scene.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: frame_buffer(&fb.state).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: frame_buffer(&fb.instances).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: b.pool.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: frame_buffer(&fb.visible).as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: fb.bin_u.as_ref().expect("bin uniforms"),
                            offset: 0,
                            size: wgpu::BufferSize::new(size_of::<BinU>() as u64),
                        }),
                    },
                ],
            });
            fb.binds = Some((cull, draw));
        }
        queue.write_buffer(frame_buffer(&fb.state), 0, bytemuck::cast_slice(&state));
        if !instances.is_empty() {
            queue.write_buffer(frame_buffer(&fb.instances), 0, bytemuck::cast_slice(&instances));
        }
        queue.write_buffer(frame_buffer(&fb.counters), 0, bytemuck::cast_slice(&counters));
        queue.write_buffer(frame_buffer(&fb.cull_u), 0, &records(&culls, self.cull_stride));
        queue.write_buffer(frame_buffer(&fb.bin_u), 0, &records(&bin_us, self.bin_stride));

        enc.clear_buffer(&b.requests, 0, None);
        {
            let (cull_bind, _) = fb.binds.as_ref().expect("binds");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cluster cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull);
            let x = b.max_clusters.div_ceil(64).clamp(1, max_instances as u32);
            for (v, &(_, count)) in spans.iter().enumerate() {
                if count > 0 {
                    pass.set_bind_group(0, cull_bind, &[(v as u64 * self.cull_stride) as u32]);
                    pass.dispatch_workgroups(x, count, 1);
                }
            }
        }

        // Read the requests and statistics back when a buffer is free.
        let bytes = ((b.groups + nv * STATS) * 4) as u64;
        let idle = self.readbacks.iter().position(|r| matches!(r.fetch, Fetch::Idle));
        let idle = idle.or_else(|| {
            (self.readbacks.len() < READBACKS).then(|| {
                self.readbacks.push(Readback {
                    buffer: readback_buffer(device, bytes),
                    fetch: Fetch::Idle,
                    done: Arc::new(AtomicBool::new(false)),
                    frame: 0,
                    generation: 0,
                    views: 0,
                    groups: 0,
                });
                self.readbacks.len() - 1
            })
        });
        if let Some(i) = idle {
            let rb = &mut self.readbacks[i];
            if rb.buffer.size() < bytes {
                rb.buffer = readback_buffer(device, bytes);
            }
            if b.groups > 0 {
                enc.copy_buffer_to_buffer(&b.requests, 0, &rb.buffer, 0, (b.groups * 4) as u64);
            }
            enc.copy_buffer_to_buffer(
                frame_buffer(&fb.counters),
                (args * 4) as u64,
                &rb.buffer,
                (b.groups * 4) as u64,
                (nv * STATS * 4) as u64,
            );
            rb.fetch = Fetch::Copied;
            rb.frame = self.frame;
            rb.generation = b.generation;
            rb.views = nv;
            rb.groups = b.groups;
        }
        self.instance_log = log;
        self.regions = regions;
        self.draws = draws;
    }

    /// Draws view `view`'s opaque (or blended) bins. Groups 0 and 1 must be bound.
    pub(crate) fn draw(
        &self,
        pass: &mut wgpu::RenderPass,
        view: usize,
        blended: bool,
        materials: &[Option<GpuMaterial>],
    ) {
        let (Some(list), Some((_, bind)), Some(counters)) = (
            self.draws.get(view),
            self.buffers.binds.as_ref(),
            self.buffers.counters.as_ref(),
        ) else {
            return;
        };
        for d in list.iter().filter(|d| d.blend == blended) {
            let Some(Some(g)) = materials.get(d.material as usize) else {
                continue;
            };
            pass.set_pipeline(&self.surfaces[d.pipe]);
            pass.set_bind_group(2, &g.bind, &[]);
            pass.set_bind_group(3, bind, &[(d.record as u64 * self.bin_stride) as u32]);
            pass.draw_indirect(counters, d.record as u64 * 16);
        }
    }

    /// Whether `view` draws any clusters this frame.
    pub(crate) fn draws_in(&self, view: usize) -> bool {
        self.draws.get(view).is_some_and(|l| !l.is_empty())
    }

    /// Starts reading back what this frame's cut asked for. Call after submitting.
    pub(crate) fn after_submit(&mut self) {
        for rb in &mut self.readbacks {
            if matches!(rb.fetch, Fetch::Copied) {
                rb.done.store(false, Ordering::Release);
                let done = rb.done.clone();
                rb.buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| {
                    done.store(true, Ordering::Release);
                });
                rb.fetch = Fetch::Mapping;
            }
        }
    }
}

fn frame_buffer(b: &Option<wgpu::Buffer>) -> &wgpu::Buffer {
    b.as_ref().expect("frame buffer")
}

fn readback_buffer(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cluster readback"),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn mesh_of<'a>(meshes: &'a [Option<ClusterMesh>], b: &Built, p: usize) -> (&'a ClusterMesh, usize, usize) {
    let (r, local) = b.owner[p];
    let m = meshes[b.order[r as usize]].as_ref().expect("built mesh");
    (m, b.page_first[r as usize] as usize, local as usize)
}

fn load(meshes: &[Option<ClusterMesh>], b: &mut Built, queue: &wgpu::Queue, p: usize, slot: u32) {
    let (m, pf, local) = mesh_of(meshes, b, p);
    queue.write_buffer(&b.pool, u64::from(slot) * b.slot_bytes, &m.dag.page_data[local]);
    b.page_slot[p] = slot;
    for &d in m.dag.page_dependencies(local) {
        b.dependents[pf + d as usize] += 1;
    }
}

fn evict(meshes: &[Option<ClusterMesh>], b: &mut Built, p: usize) -> u32 {
    let (m, pf, local) = mesh_of(meshes, b, p);
    for &d in m.dag.page_dependencies(local) {
        b.dependents[pf + d as usize] -= 1;
    }
    std::mem::replace(&mut b.page_slot[p], NONE)
}

/// The page to give up for one of `urgency`: resident, not pinned, nothing resident depending
/// on it, not just loaded, and clearly less urgent; the least urgent, then least recently
/// needed.
fn victim(b: &Built, touched: &[bool], urgency: f32) -> Option<usize> {
    (0..b.pages)
        .filter(|&p| {
            b.page_slot[p] != NONE
                && !b.pinned[p]
                && b.dependents[p] == 0
                && !touched[p]
                && b.prio[p] * HYSTERESIS < urgency
        })
        .min_by(|&x, &y| {
            b.prio[x]
                .total_cmp(&b.prio[y])
                .then(b.last_used[x].cmp(&b.last_used[y]))
        })
}

/// Page `p` and its missing dependencies, dependencies first.
fn missing_chain(meshes: &[Option<ClusterMesh>], b: &Built, p: usize) -> Vec<usize> {
    let (m, pf, _) = mesh_of(meshes, b, p);
    let mut out = vec![p];
    let mut stack = vec![p];
    while let Some(q) = stack.pop() {
        for &d in m.dag.page_dependencies(q - pf) {
            let d = pf + d as usize;
            if b.page_slot[d] == NONE && !out.contains(&d) {
                out.push(d);
                stack.push(d);
            }
        }
    }
    out.sort_unstable();
    out
}

/// Which groups' refinements can be drawn: coarsest first, so a group's consumers are known.
fn compute_ready(meshes: &[Option<ClusterMesh>], b: &mut Built) {
    for (r, &id) in b.order.iter().enumerate() {
        let m = meshes[id].as_ref().expect("built mesh");
        let (gf, pf) = (b.group_first[r] as usize, b.page_first[r] as usize);
        for &g in m.by_level.iter().rev() {
            let g = g as usize;
            let resident = |p: u32| b.page_slot[pf + p as usize] != NONE;
            let ready = m.child_pages[g].iter().all(|&p| resident(p))
                && resident(m.dag.groups[g].page)
                && m.consumers[g].iter().all(|&q| b.ready[gf + q as usize]);
            b.ready[gf + g] = ready;
        }
    }
}

#[cfg(feature = "native")]
impl ClusterRenderer {
    /// Blocks until the GPU is idle and every readback has arrived.
    pub(crate) fn settle(&self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// The last frame's visible lists: per view, (instance within the view, mesh-local
    /// cluster), sorted. Blocks.
    pub(crate) fn read_visible(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Vec<(u32, u32)>> {
        let (Some(b), Some(visible), Some(counters)) = (
            self.built.as_ref(),
            self.buffers.visible.as_ref(),
            self.buffers.counters.as_ref(),
        ) else {
            return Vec::new();
        };
        let nb = b.bins.len();
        let nv = self.draws.len();
        let read = |src: &wgpu::Buffer, size: u64| -> Vec<u32> {
            let dst = readback_buffer(device, size);
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(src, 0, &dst, 0, size.next_multiple_of(4));
            queue.submit([enc.finish()]);
            dst.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let data = dst.slice(..).get_mapped_range().expect("mapped");
            bytemuck::cast_slice(&data[..size as usize]).to_vec()
        };
        let args = read(counters, (nv * nb * 16) as u64);
        let vis = read(visible, visible.size());
        let inst = &self.instance_log;
        let mut out = vec![Vec::new(); nv];
        for (v, list) in out.iter_mut().enumerate() {
            for bi in 0..nb {
                let rec = v * nb + bi;
                let (off, cap) = (self.regions[2 * rec] as usize, self.regions[2 * rec + 1]);
                let n = args[4 * rec + 1].min(cap) as usize;
                for &[i, c] in vis[2 * off..2 * (off + n)].as_chunks::<2>().0 {
                    let (first, r) = inst[i as usize];
                    list.push((i - first, c - b.cluster_first[r as usize]));
                }
            }
            list.sort_unstable();
            list.dedup();
        }
        out
    }

    /// Per page of mesh `id`, whether it's resident.
    pub(crate) fn resident(&self, id: u32) -> Vec<bool> {
        let Some(b) = &self.built else {
            return Vec::new();
        };
        let r = b.record.get(id as usize).copied().unwrap_or(NONE);
        if r == NONE {
            return Vec::new();
        }
        let m = self.meshes[id as usize].as_ref().expect("mesh");
        let pf = b.page_first[r as usize] as usize;
        (0..m.dag.pages.len()).map(|p| b.page_slot[pf + p] != NONE).collect()
    }
}
