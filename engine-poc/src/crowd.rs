//! GPU crowds: thousands of skinned units animated from one farthing pose palette (`.fanm`, see
//! `farthing::anim::bake`), culled and drawn without per-unit CPU work.
//!
//! A unit's whole animation state is a 48-byte record ([`CrowdUnit`]): where it stands, the
//! clip it plays and when it started, and the clip it fades out of. The CPU writes a record
//! only when a unit changes clips or moves; the shaders find every joint's transform from the
//! viewer's clock ([`crate::viewer::Viewer::set_crowd_time`]).
//!
//! Each frame, per view a crowd is drawn in:
//!
//! 1. **Cull** (`crowd_cull.wgsl`, one compute dispatch): every unit's bounding sphere is
//!    frustum tested, and its projected radius picks its level of detail. It's appended to that
//!    level's region of the visible list, counting into the level's indirect draw.
//! 2. **Draw** (`crowd_draw.wgsl`): one indexed indirect draw per level, one instance per
//!    visible unit. Skinned levels blend the two palette frames around each joint's clip time
//!    (and the faded-out clip's) and skin with four influences. A vertex animation level
//!    (`.fvat`, `farthing::anim::vat`) reads baked positions instead, for the farthest units.
//!    It can't blend bones, so during a fade it blends the two clips' positions and normals:
//!    close enough at the size those units are on screen.
//!
//! Every level is a bone LOD of the same skeleton that keeps its joint indices
//! (`farthing::bones::apply_bone_reduction`), so they all share the one palette.

use crate::viewer::{DEPTH, GpuMaterial, SAMPLES};
use bytemuck::{Pod, Zeroable};
use farthing::anim::bake::PosePalette;
use farthing::anim::vat::VertexAnimation;
use farthing::glam::{Mat3, Mat4, Vec3};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Levels of detail a crowd can have.
pub const MAX_LODS: usize = 4;
/// Words per indirect record (an indexed draw's arguments).
const ARGS_WORDS: usize = 5;

/// One unit of a crowd (48 bytes, as `crowd.wgsl` reads it).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct CrowdUnit {
    /// In the crowd's space (its draw's model transform).
    pub position: [f32; 3],
    /// Turn about +y, radians.
    pub yaw: f32,
    /// The palette clip playing, and the one it fades in over.
    pub clip: u32,
    pub previous_clip: u32,
    /// Seconds the fade from `previous_clip` to `clip` takes (0: a cut).
    pub fade: f32,
    pub scale: f32,
    /// Clock times `clip` and `previous_clip` started, and the fade began.
    pub start: f32,
    pub previous_start: f32,
    pub fade_start: f32,
    /// Playback rate (1: as authored).
    pub speed: f32,
}

/// The vertices of a level of detail.
pub enum CrowdLodKind<'a> {
    /// Skinned from the palette: per vertex xyz positions and normals (`3n` floats; normals may
    /// be empty to shade flat), four joint indices and four weights (`4n` each).
    Skinned {
        positions: &'a [f32],
        normals: &'a [f32],
        joints: &'a [u16],
        weights: &'a [f32],
    },
    /// Baked positions (and normals) per frame. Baked with the palette's options, so its clip
    /// records match the palette's.
    Vat(&'a VertexAnimation),
}

/// One level of detail of a crowd's unit.
pub struct CrowdLod<'a> {
    pub kind: CrowdLodKind<'a>,
    /// Per vertex uvs (`2n` floats), or empty.
    pub uvs: &'a [f32],
    /// Three per triangle.
    pub indices: &'a [u32],
    /// The least projected bounding radius, in pixels, a unit is drawn with this level at;
    /// smaller units take the next. The last level draws whatever is left.
    pub min_pixels: f32,
}

/// What a crowd drew, from the latest readback.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrowdStats {
    /// Units per level, per view the crowd was drawn in.
    pub per_lod: Vec<Vec<u32>>,
    /// Triangles drawn, all views.
    pub triangles: u64,
}

/// A crowd drawn in a view this frame.
pub(crate) struct CrowdDraw {
    pub crowd: u32,
    pub model: Mat4,
}

/// One view's crowds this frame.
pub(crate) struct CrowdView {
    pub view_proj: Mat4,
    pub eye: Vec3,
    /// Viewport height / (2 tan(fov / 2)).
    pub focal: f32,
    pub draws: Vec<CrowdDraw>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullU {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    planes: [[f32; 4]; 6],
    eye: [f32; 4],
    lod_pixels: [f32; 4],
    counts: [u32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawU {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    region: u32,
    time: f32,
    vat_vertices: u32,
    vat_flags: u32,
    vat_min: [f32; 4],
    vat_extent: [f32; 4],
    tint: [f32; 4],
}

enum LodVertices {
    Skinned {
        positions: wgpu::Buffer,
        normals: wgpu::Buffer,
        joints: wgpu::Buffer,
        weights: wgpu::Buffer,
    },
    Vat {
        texels: wgpu::Buffer,
        vertices: u32,
        normals: bool,
        min: Vec3,
        extent: Vec3,
    },
}

struct GpuLod {
    vertices: LodVertices,
    uvs: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    min_pixels: f32,
}

/// One cull and its draws: a crowd in a view.
struct Record {
    view: usize,
    /// Its cull record's and its first draw record's index in the frame's uniform arrays.
    cull: usize,
    draw: usize,
    /// Its first indirect record in the crowd's args (one per level).
    first: usize,
}

struct Crowd {
    palette: wgpu::Buffer,
    clips: wgpu::Buffer,
    bones: wgpu::Buffer,
    clip_count: u32,
    /// Bounding radius about a unit's origin at scale 1, every frame of every clip.
    radius: f32,
    lods: Vec<GpuLod>,
    material: u32,
    units: Vec<CrowdUnit>,
    /// The units' GPU buffer and how many units it holds.
    gpu_units: Option<(wgpu::Buffer, usize)>,
    /// The range of `units` the GPU doesn't have yet.
    dirty: Option<(usize, usize)>,
    /// Visible lists and indirect records, sized for `capacity.0` units and `capacity.1`
    /// records (a crowd drawn in two views has two).
    visible: Option<wgpu::Buffer>,
    args: Option<wgpu::Buffer>,
    capacity: (usize, usize),
    /// This frame's culls, and their bind groups.
    records: Vec<Record>,
    binds: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
    /// The latest stats, and the readback bringing the next.
    stats: CrowdStats,
    readback: Option<Readback>,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Fetch,
    done: Arc<AtomicBool>,
    /// Per record: its view.
    views: Vec<usize>,
    lods: usize,
    /// Triangles per level.
    triangles: Vec<u64>,
}

enum Fetch {
    Idle,
    Copied,
    Mapping,
}

pub(crate) struct CrowdRenderer {
    cull: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    draw_bgl: wgpu::BindGroupLayout,
    /// Skinned then vertex animation, each culling back faces then not.
    pipelines: [wgpu::RenderPipeline; 4],
    crowds: Vec<Option<Crowd>>,
    cull_u: Uniforms,
    draw_u: Uniforms,
    /// A placeholder for levels without vertex animation.
    no_vat: wgpu::Buffer,
    time: f32,
    frustum: bool,
    /// Tint each level its own color.
    show_lods: bool,
}

/// A growable uniform buffer of records read at dynamic offsets.
struct Uniforms {
    size: u64,
    stride: u64,
    buffer: Option<wgpu::Buffer>,
}

impl Uniforms {
    fn new(device: &wgpu::Device, size: usize) -> Uniforms {
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        Uniforms {
            size: size as u64,
            stride: (size as u64).next_multiple_of(align),
            buffer: None,
        }
    }

    fn write<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, records: &[T]) {
        let stride = self.stride as usize;
        let mut bytes = vec![0u8; records.len().max(1) * stride];
        for (i, r) in records.iter().enumerate() {
            bytes[i * stride..][..self.size as usize].copy_from_slice(bytemuck::bytes_of(r));
        }
        if !self
            .buffer
            .as_ref()
            .is_some_and(|b| b.size() >= bytes.len() as u64)
        {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("crowd uniforms"),
                size: (bytes.len() as u64).next_power_of_two(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(self.buffer.as_ref().expect("uniforms"), 0, &bytes);
    }

    fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer.as_ref().expect("uniforms written"),
            offset: 0,
            size: wgpu::BufferSize::new(self.size),
        })
    }

    fn offset(&self, i: usize) -> u32 {
        (i as u64 * self.stride) as u32
    }
}

fn storage(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
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

fn uniform(
    binding: u32,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
    size: usize,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(size as u64),
        },
        count: None,
    }
}

fn buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("crowd"),
        size: (data.len() as u64).max(16).next_multiple_of(4),
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b, 0, data);
    b
}

/// World-space frustum planes of a view-projection matrix (0..1 depth).
fn frustum_planes(m: Mat4) -> [[f32; 4]; 6] {
    let r = [m.row(0), m.row(1), m.row(2), m.row(3)];
    [
        r[3] + r[0],
        r[3] - r[0],
        r[3] + r[1],
        r[3] - r[1],
        r[2],
        r[3] - r[2],
    ]
    .map(|p| {
        let l = p.truncate().length();
        (if l > 0.0 { p / l } else { p }).to_array()
    })
}

/// The colors levels are drawn in when shown (as the demo's level colors); alpha 0 tells the
/// shader to replace the material's color.
const LOD_TINTS: [[f32; 4]; MAX_LODS] = [
    [0.35, 0.75, 1.0, 0.0],
    [0.45, 0.9, 0.45, 0.0],
    [1.0, 0.8, 0.3, 0.0],
    [1.0, 0.45, 0.45, 0.0],
];

/// Whether this device can draw crowds (compute and vertex-stage storage: WebGPU or native).
pub(crate) fn supported(device: &wgpu::Device) -> bool {
    let l = device.limits();
    l.max_storage_buffers_per_shader_stage >= 5 && l.max_compute_invocations_per_workgroup >= 64
}

impl CrowdRenderer {
    pub(crate) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        frames: &wgpu::BindGroupLayout,
        models: &wgpu::BindGroupLayout,
        material: &wgpu::BindGroupLayout,
    ) -> CrowdRenderer {
        let compute = wgpu::ShaderStages::COMPUTE;
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("crowd cull"),
            entries: &[
                storage(0, true, compute),
                storage(1, false, compute),
                storage(2, false, compute),
                uniform(3, compute, true, size_of::<CullU>()),
            ],
        });
        let vertex = wgpu::ShaderStages::VERTEX;
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("crowd draw"),
            entries: &[
                uniform(0, vertex, false, 48),
                storage(1, true, vertex),
                storage(2, true, vertex),
                storage(3, true, vertex),
                storage(4, true, vertex),
                // The fragment stage reads the level tint.
                uniform(5, vertex | wgpu::ShaderStages::FRAGMENT, true, size_of::<DrawU>()),
                storage(6, true, vertex),
            ],
        });
        let common = include_str!("crowd.wgsl");
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("crowd cull"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{common}\n{}",
                include_str!("crowd_cull.wgsl")
            ))),
        });
        let cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("crowd cull"),
            layout: Some(&cull_layout),
            module: &cull_shader,
            entry_point: Some("cull_crowd"),
            compilation_options: Default::default(),
            cache: None,
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("crowd draw"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{}\n{}\n{common}\n{}",
                include_str!("common.wgsl"),
                include_str!("pbr.wgsl"),
                include_str!("crowd_draw.wgsl")
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(frames), Some(models), Some(material), Some(&draw_bgl)],
            immediate_size: 0,
        });
        let attr = |location, format| wgpu::VertexAttribute {
            format,
            offset: 0,
            shader_location: location,
        };
        let skinned_attrs = [
            [attr(0, wgpu::VertexFormat::Float32x3)],
            [attr(1, wgpu::VertexFormat::Float32x3)],
            [attr(2, wgpu::VertexFormat::Float32x2)],
            [attr(3, wgpu::VertexFormat::Uint16x4)],
            [attr(4, wgpu::VertexFormat::Float32x4)],
        ];
        let strides = [12, 12, 8, 8, 16];
        let skinned: Vec<Option<wgpu::VertexBufferLayout>> = skinned_attrs
            .iter()
            .zip(strides)
            .map(|(a, stride)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: stride,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: a,
                })
            })
            .collect();
        let vat = [skinned[2].clone()];
        let pipeline = |entry: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("crowd"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: SAMPLES,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_shader,
                    entry_point: Some("fs_crowd"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let back = Some(wgpu::Face::Back);
        let pipelines = [
            pipeline("vs_skinned", &skinned, back),
            pipeline("vs_skinned", &skinned, None),
            pipeline("vs_vat", &vat, back),
            pipeline("vs_vat", &vat, None),
        ];
        let no_vat = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crowd no vat"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        CrowdRenderer {
            cull,
            cull_bgl,
            draw_bgl,
            pipelines,
            crowds: Vec::new(),
            cull_u: Uniforms::new(device, size_of::<CullU>()),
            draw_u: Uniforms::new(device, size_of::<DrawU>()),
            no_vat,
            time: 0.0,
            frustum: true,
            show_lods: false,
        }
    }

    /// Adds a crowd with no units. Fails when a level's vertices don't match the palette.
    pub(crate) fn add(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        palette: &PosePalette,
        lods: &[CrowdLod],
        material: u32,
    ) -> Result<u32, String> {
        if lods.is_empty() || lods.len() > MAX_LODS {
            return Err(format!("a crowd needs 1 to {MAX_LODS} levels of detail"));
        }
        let usage = wgpu::BufferUsages::VERTEX;
        let joint_count = palette.joint_count;
        let mut gpu = Vec::new();
        // The largest distance any vertex can be from the unit's origin: |R(s·p) + t| ≤
        // s·|p| + |t| for every bone the palette holds.
        let (mut max_p, mut max_scale, mut vat_radius) = (0.0f32, 0.0f32, 0.0f32);
        for (li, lod) in lods.iter().enumerate() {
            let bad = |m: &str| Err(format!("level {li}: {m}"));
            let (vertices, n) = match &lod.kind {
                CrowdLodKind::Skinned {
                    positions,
                    normals,
                    joints,
                    weights,
                } => {
                    let n = positions.len() / 3;
                    if joints.len() != 4 * n || weights.len() != 4 * n {
                        return bad("needs four joints and weights per vertex");
                    }
                    if joints.iter().any(|&j| u32::from(j) >= joint_count) {
                        return bad("names a joint the palette doesn't have");
                    }
                    for p in positions.as_chunks::<3>().0 {
                        max_p = max_p.max(Vec3::from_array(*p).length());
                    }
                    let normals: Cow<[f32]> = if normals.len() == 3 * n {
                        Cow::Borrowed(normals)
                    } else {
                        Cow::Owned(vec![0.0; 3 * n])
                    };
                    (
                        LodVertices::Skinned {
                            positions: buffer(
                                device,
                                queue,
                                bytemuck::cast_slice(&positions[..3 * n]),
                                usage,
                            ),
                            normals: buffer(device, queue, bytemuck::cast_slice(&normals), usage),
                            joints: buffer(device, queue, bytemuck::cast_slice(joints), usage),
                            weights: buffer(device, queue, bytemuck::cast_slice(weights), usage),
                        },
                        n,
                    )
                }
                CrowdLodKind::Vat(v) => {
                    let same = v.frame_count() == palette.frame_count()
                        && v.clips.len() == palette.clips.len()
                        && v.clips.iter().zip(&palette.clips).all(|(a, b)| {
                            (a.first_frame, a.frame_count, a.looping)
                                == (b.first_frame, b.frame_count, b.looping)
                        });
                    if !same {
                        return bad("its vertex animation wasn't baked like the palette");
                    }
                    for k in 0..8 {
                        let corner = Vec3::new(
                            if k & 1 == 0 { 0.0 } else { 1.0 },
                            if k & 2 == 0 { 0.0 } else { 1.0 },
                            if k & 4 == 0 { 0.0 } else { 1.0 },
                        );
                        vat_radius =
                            vat_radius.max((v.bounds_min + corner * v.bounds_extent).length());
                    }
                    let texels: &[u8] = bytemuck::cast_slice(&v.texels);
                    (
                        LodVertices::Vat {
                            texels: buffer(device, queue, texels, wgpu::BufferUsages::STORAGE),
                            vertices: v.vertex_count,
                            normals: v.normals,
                            min: v.bounds_min,
                            extent: v.bounds_extent,
                        },
                        v.vertex_count as usize,
                    )
                }
            };
            let uvs: Cow<[f32]> = if lod.uvs.len() == 2 * n {
                Cow::Borrowed(lod.uvs)
            } else {
                Cow::Owned(vec![0.0; 2 * n])
            };
            let indices: Vec<u32> = lod
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter(|t| t.iter().all(|&i| (i as usize) < n))
                .flatten()
                .copied()
                .collect();
            gpu.push(GpuLod {
                vertices,
                uvs: buffer(device, queue, bytemuck::cast_slice(&uvs), usage),
                indices: buffer(
                    device,
                    queue,
                    bytemuck::cast_slice(&indices),
                    wgpu::BufferUsages::INDEX,
                ),
                index_count: indices.len() as u32,
                min_pixels: lod.min_pixels,
            });
        }
        let mut max_t = 0.0f32;
        for f in 0..palette.frame_count() as u32 {
            for j in 0..joint_count {
                let b = palette.bone(f, j);
                max_scale = max_scale.max(b.scale.abs());
                max_t = max_t.max(b.translation.length());
            }
        }
        let radius = (max_p * max_scale + max_t).max(vat_radius).max(1e-3);

        let fanm = palette.to_bytes();
        let clip_records: Vec<u32> = palette
            .clips
            .iter()
            .flat_map(|c| {
                [
                    c.first_frame,
                    c.frame_count,
                    c.duration.to_bits(),
                    u32::from(c.looping),
                ]
            })
            .collect();
        let crowd = Crowd {
            palette: buffer(device, queue, &fanm[16..64], wgpu::BufferUsages::UNIFORM),
            clips: buffer(
                device,
                queue,
                bytemuck::cast_slice(&clip_records),
                wgpu::BufferUsages::STORAGE,
            ),
            bones: buffer(
                device,
                queue,
                bytemuck::cast_slice(&palette.bones),
                wgpu::BufferUsages::STORAGE,
            ),
            clip_count: palette.clips.len() as u32,
            radius,
            lods: gpu,
            material,
            units: Vec::new(),
            gpu_units: None,
            dirty: None,
            visible: None,
            args: None,
            capacity: (0, 0),
            records: Vec::new(),
            binds: None,
            stats: CrowdStats::default(),
            readback: None,
        };
        let id = match self.crowds.iter().position(Option::is_none) {
            Some(free) => {
                self.crowds[free] = Some(crowd);
                free
            }
            None => {
                self.crowds.push(Some(crowd));
                self.crowds.len() - 1
            }
        };
        Ok(id as u32)
    }

    pub(crate) fn remove(&mut self, id: u32) {
        if let Some(c) = self.crowds.get_mut(id as usize) {
            *c = None;
        }
    }

    fn crowd(&mut self, id: u32) -> Option<&mut Crowd> {
        self.crowds.get_mut(id as usize).and_then(Option::as_mut)
    }

    /// Writes units `first..first + units.len()`, growing the crowd to hold them. Clip indices
    /// past the palette's play its first clip.
    pub(crate) fn set_units(&mut self, id: u32, first: usize, units: &[CrowdUnit]) {
        let Some(c) = self.crowd(id) else { return };
        let end = first + units.len();
        if c.units.len() < end {
            c.units.resize(end, CrowdUnit::default());
        }
        let clips = c.clip_count;
        for (dst, src) in c.units[first..end].iter_mut().zip(units) {
            *dst = *src;
            if dst.clip >= clips {
                dst.clip = 0;
            }
            if dst.previous_clip >= clips {
                dst.previous_clip = 0;
            }
        }
        c.dirty = Some(match c.dirty {
            Some((a, b)) => (a.min(first), b.max(end)),
            None => (first, end),
        });
    }

    /// Keeps the first `count` units.
    pub(crate) fn set_unit_count(&mut self, id: u32, count: usize) {
        if let Some(c) = self.crowd(id) {
            c.units.truncate(count);
            c.dirty = c.dirty.map(|(a, b)| (a.min(count), b.min(count)));
        }
    }

    pub(crate) fn units(&self, id: u32) -> usize {
        self.crowds
            .get(id as usize)
            .and_then(Option::as_ref)
            .map_or(0, |c| c.units.len())
    }

    pub(crate) fn set_time(&mut self, seconds: f32) {
        self.time = seconds;
    }

    pub(crate) fn set_frustum_culling(&mut self, on: bool) {
        self.frustum = on;
    }

    pub(crate) fn set_show_lods(&mut self, on: bool) {
        self.show_lods = on;
    }

    pub(crate) fn stats(&self, id: u32) -> Option<CrowdStats> {
        self.crowds
            .get(id as usize)?
            .as_ref()
            .map(|c| c.stats.clone())
    }

    /// Uploads changed units, then records every crowd's culls for `views`.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        views: &[CrowdView],
    ) {
        for c in self.crowds.iter_mut().flatten() {
            c.records.clear();
            c.binds = None;
            c.poll_readback();
        }
        let mut culls = Vec::new();
        let mut draws = Vec::new();
        for (vi, v) in views.iter().enumerate() {
            let planes = frustum_planes(v.view_proj);
            for d in &v.draws {
                let Some(Some(c)) = self.crowds.get_mut(d.crowd as usize) else {
                    continue;
                };
                if c.units.is_empty() {
                    continue;
                }
                let first_record = c.records.len() * c.lods.len();
                let mut lod_pixels = [0.0; 4];
                for (k, l) in c.lods.iter().enumerate() {
                    lod_pixels[k] = l.min_pixels;
                }
                let m3 = Mat3::from_mat4(d.model);
                let scale = m3
                    .x_axis
                    .length()
                    .max(m3.y_axis.length())
                    .max(m3.z_axis.length());
                culls.push(CullU {
                    view_proj: v.view_proj.to_cols_array_2d(),
                    model: d.model.to_cols_array_2d(),
                    planes,
                    eye: v.eye.extend(v.focal).to_array(),
                    lod_pixels,
                    counts: [
                        c.units.len() as u32,
                        c.lods.len() as u32,
                        0,
                        first_record as u32,
                    ],
                    params: [c.radius, scale, f32::from(u8::from(self.frustum)), 0.0],
                });
                let normal = Mat4::from_mat3(m3.inverse().transpose());
                let first_draw = draws.len();
                for (k, l) in c.lods.iter().enumerate() {
                    let mut u = DrawU {
                        model: d.model.to_cols_array_2d(),
                        normal: normal.to_cols_array_2d(),
                        region: 0,
                        time: self.time,
                        vat_vertices: 0,
                        vat_flags: 0,
                        vat_min: [0.0; 4],
                        vat_extent: [0.0; 4],
                        tint: if self.show_lods {
                            LOD_TINTS[k]
                        } else {
                            [1.0; 4]
                        },
                    };
                    if let LodVertices::Vat {
                        vertices,
                        normals,
                        min,
                        extent,
                        ..
                    } = &l.vertices
                    {
                        u.vat_vertices = *vertices;
                        u.vat_flags = u32::from(*normals);
                        u.vat_min = min.extend(0.0).to_array();
                        u.vat_extent = extent.extend(0.0).to_array();
                    }
                    draws.push(u);
                }
                c.records.push(Record {
                    view: vi,
                    cull: culls.len() - 1,
                    draw: first_draw,
                    first: first_record,
                });
            }
        }
        // Size each drawn crowd's buffers, and fill in the region sizes.
        for c in self.crowds.iter_mut().flatten() {
            if c.records.is_empty() {
                continue;
            }
            c.upload_units(device, queue);
            let (units, records) = (c.units.len(), c.records.len() * c.lods.len());
            if c.capacity.0 < units || c.capacity.1 < records {
                let cap = (units.next_power_of_two(), records.max(c.capacity.1));
                c.visible = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("crowd visible"),
                    size: (cap.0 * cap.1 * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }));
                c.args = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("crowd args"),
                    size: (cap.1 * ARGS_WORDS * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
                c.capacity = cap;
                c.readback = None;
            }
            let cap = c.capacity.0 as u32;
            let mut args = Vec::with_capacity(records * ARGS_WORDS);
            for r in &c.records {
                culls[r.cull].counts[2] = cap;
                for (k, l) in c.lods.iter().enumerate() {
                    draws[r.draw + k].region = (culls[r.cull].counts[3] + k as u32) * cap;
                    args.extend_from_slice(&[l.index_count, 0, 0, 0, 0]);
                }
            }
            queue.write_buffer(
                c.args.as_ref().expect("args"),
                0,
                bytemuck::cast_slice(&args),
            );
        }
        if culls.is_empty() {
            return;
        }
        self.cull_u.write(device, queue, &culls);
        self.draw_u.write(device, queue, &draws);
        for c in self.crowds.iter_mut().flatten() {
            if c.records.is_empty() {
                continue;
            }
            let units = &c.gpu_units.as_ref().expect("units").0;
            let (visible, args) = (
                c.visible.as_ref().expect("visible"),
                c.args.as_ref().expect("args"),
            );
            let cull = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crowd cull"),
                layout: &self.cull_bgl,
                entries: &[
                    entry(0, units.as_entire_binding()),
                    entry(1, visible.as_entire_binding()),
                    entry(2, args.as_entire_binding()),
                    entry(3, self.cull_u.binding()),
                ],
            });
            // One bind group serves every level: a vertex animation level's texels are bound
            // per level, so only crowds with one get a second group (below).
            let vat = c.lods.iter().find_map(|l| match &l.vertices {
                LodVertices::Vat { texels, .. } => Some(texels),
                LodVertices::Skinned { .. } => None,
            });
            let draw = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crowd draw"),
                layout: &self.draw_bgl,
                entries: &[
                    entry(0, c.palette.as_entire_binding()),
                    entry(1, c.clips.as_entire_binding()),
                    entry(2, c.bones.as_entire_binding()),
                    entry(3, units.as_entire_binding()),
                    entry(4, visible.as_entire_binding()),
                    entry(5, self.draw_u.binding()),
                    entry(6, vat.unwrap_or(&self.no_vat).as_entire_binding()),
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("crowd cull"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cull);
                for r in &c.records {
                    pass.set_bind_group(0, &cull, &[self.cull_u.offset(r.cull)]);
                    pass.dispatch_workgroups((c.units.len() as u32).div_ceil(64), 1, 1);
                }
            }
            c.binds = Some((cull, draw));
            c.start_readback(device, enc);
        }
    }

    /// Draws the crowds `view` culled. Group 0 (the frame) and group 1 (any model) are bound.
    pub(crate) fn draw(
        &self,
        pass: &mut wgpu::RenderPass,
        view: usize,
        materials: &[Option<GpuMaterial>],
    ) {
        for c in self.crowds.iter().flatten() {
            let (Some((_, bind)), Some(args), Some(Some(mat))) =
                (&c.binds, &c.args, materials.get(c.material as usize))
            else {
                continue;
            };
            pass.set_bind_group(2, &mat.bind, &[]);
            let cull_face = usize::from(mat.double_sided);
            for r in c.records.iter().filter(|r| r.view == view) {
                for (k, l) in c.lods.iter().enumerate() {
                    pass.set_bind_group(3, bind, &[self.draw_u.offset(r.draw + k)]);
                    match &l.vertices {
                        LodVertices::Skinned {
                            positions,
                            normals,
                            joints,
                            weights,
                        } => {
                            pass.set_pipeline(&self.pipelines[cull_face]);
                            pass.set_vertex_buffer(0, positions.slice(..));
                            pass.set_vertex_buffer(1, normals.slice(..));
                            pass.set_vertex_buffer(2, l.uvs.slice(..));
                            pass.set_vertex_buffer(3, joints.slice(..));
                            pass.set_vertex_buffer(4, weights.slice(..));
                        }
                        LodVertices::Vat { .. } => {
                            pass.set_pipeline(&self.pipelines[2 + cull_face]);
                            pass.set_vertex_buffer(0, l.uvs.slice(..));
                        }
                    }
                    pass.set_index_buffer(l.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed_indirect(args, ((r.first + k) * ARGS_WORDS * 4) as u64);
                }
            }
        }
    }

    /// Whether `view` draws any crowd this frame.
    pub(crate) fn draws_in(&self, view: usize) -> bool {
        self.crowds
            .iter()
            .flatten()
            .any(|c| c.binds.is_some() && c.records.iter().any(|r| r.view == view))
    }

    /// Starts reading back this frame's counts. Call after submitting.
    pub(crate) fn after_submit(&mut self) {
        for c in self.crowds.iter_mut().flatten() {
            if let Some(rb) = &mut c.readback
                && matches!(rb.state, Fetch::Copied)
            {
                rb.done.store(false, Ordering::Release);
                let done = rb.done.clone();
                rb.buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |_| {
                        done.store(true, Ordering::Release);
                    });
                rb.state = Fetch::Mapping;
            }
        }
    }
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

impl Crowd {
    fn upload_units(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let n = self.units.len();
        if !self.gpu_units.as_ref().is_some_and(|(_, cap)| *cap >= n) {
            let cap = n.next_power_of_two().max(64);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("crowd units"),
                size: (cap * size_of::<CrowdUnit>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.gpu_units = Some((buf, cap));
            self.dirty = Some((0, n));
        }
        if let Some((a, b)) = self.dirty.take()
            && a < b
        {
            let buf = &self.gpu_units.as_ref().expect("units").0;
            let offset = (a * size_of::<CrowdUnit>()) as u64;
            queue.write_buffer(buf, offset, bytemuck::cast_slice(&self.units[a..b]));
        }
    }

    fn start_readback(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder) {
        let records = self.records.len() * self.lods.len();
        let size = (records * ARGS_WORDS * 4) as u64;
        let views: Vec<usize> = self.records.iter().map(|r| r.view).collect();
        let triangles = self
            .lods
            .iter()
            .map(|l| u64::from(l.index_count / 3))
            .collect();
        let rb = self.readback.get_or_insert_with(|| Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("crowd readback"),
                size: (self.capacity.1 * ARGS_WORDS * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Fetch::Idle,
            done: Arc::new(AtomicBool::new(false)),
            views: Vec::new(),
            lods: 0,
            triangles: Vec::new(),
        });
        if !matches!(rb.state, Fetch::Idle) || rb.buffer.size() < size {
            return;
        }
        enc.copy_buffer_to_buffer(self.args.as_ref().expect("args"), 0, &rb.buffer, 0, size);
        rb.views = views;
        rb.lods = self.lods.len();
        rb.triangles = triangles;
        rb.state = Fetch::Copied;
    }

    fn poll_readback(&mut self) {
        let Some(rb) = &mut self.readback else { return };
        if !matches!(rb.state, Fetch::Mapping) || !rb.done.load(Ordering::Acquire) {
            return;
        }
        let size = (rb.views.len() * rb.lods * ARGS_WORDS * 4) as u64;
        {
            let data = rb.buffer.slice(..size).get_mapped_range().expect("mapped");
            let words: &[u32] = bytemuck::cast_slice(&data);
            let nv = rb.views.iter().max().map_or(0, |v| v + 1);
            let mut stats = CrowdStats {
                per_lod: vec![vec![0; rb.lods]; nv],
                triangles: 0,
            };
            for (r, &v) in rb.views.iter().enumerate() {
                for k in 0..rb.lods {
                    let n = words[(r * rb.lods + k) * ARGS_WORDS + 1];
                    stats.per_lod[v][k] += n;
                    stats.triangles += u64::from(n) * rb.triangles[k];
                }
            }
            self.stats = stats;
        }
        rb.buffer.unmap();
        rb.state = Fetch::Idle;
    }
}

#[cfg(feature = "native")]
impl CrowdRenderer {
    /// Blocks until every readback has arrived, then takes them in.
    pub(crate) fn settle(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        for c in self.crowds.iter_mut().flatten() {
            c.poll_readback();
        }
    }
}
