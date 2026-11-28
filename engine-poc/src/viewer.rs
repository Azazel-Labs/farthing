//! The interactive viewer: textures, materials and meshes held on the GPU, drawn from one or
//! more cameras into any texture view.
//!
//! It owns no window or canvas. The browser build hands it a canvas surface's texture each frame
//! (see `web.rs`); a native caller can hand it an offscreen target. Surfaces are glTF
//! metallic-roughness under a procedural room light (`pbr.wgsl`); the overlays drawn over them
//! are in [`crate::editor_visualizations`].

use crate::clusters::{ClusterDraw, ClusterRenderer, ClusterStats, ClusterView};
use crate::crowd::{CrowdDraw, CrowdLod, CrowdRenderer, CrowdStats, CrowdUnit, CrowdView};
use crate::editor_visualizations::{self as viz, Overlay};
use farthing::cluster::GpuClusterDag;
use bytemuck::{Pod, Zeroable};
use farthing::glam::{Mat3, Mat4, Vec3, Vec4};
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Multisampling: WebGPU guarantees 4× for the canvas formats.
pub(crate) const SAMPLES: u32 = 4;

/// Where the scene is seen from.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World to camera (right-handed, looking down -Z).
    pub view: Mat4,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        // Right-handed, depth 0 at `near` to 1 at `far` (WebGPU's clip space).
        let f = 1.0 / (self.fov_y / 2.0).tan();
        let r = self.far / (self.near - self.far);
        let proj = Mat4::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, -1.0),
            Vec4::new(0.0, 0.0, r * self.near, 0.0),
        );
        proj * self.view
    }

    pub fn eye(&self) -> Vec3 {
        self.view.inverse().w_axis.truncate()
    }
}

/// How a material's alpha is used (glTF's `alphaMode`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    /// Cut out below the cutoff.
    Mask(f32),
    /// Blended over what's behind, drawn after the opaque surfaces, far to near.
    Blend,
}

/// A glTF metallic-roughness material. Textures are ids from [`Viewer::add_texture`].
#[derive(Clone, Debug)]
pub struct Material {
    /// Linear RGBA.
    pub base_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    /// Linear RGB, already scaled by any strength.
    pub emissive: Vec3,
    /// The tangent-space normal's xy scale.
    pub normal_scale: [f32; 2],
    pub occlusion_strength: f32,
    pub alpha: AlphaMode,
    pub double_sided: bool,
    pub base_color_texture: Option<u32>,
    /// Roughness in green, metalness in blue.
    pub metallic_roughness_texture: Option<u32>,
    pub normal_texture: Option<u32>,
    pub emissive_texture: Option<u32>,
    /// Occlusion in red.
    pub occlusion_texture: Option<u32>,
    /// Shade each triangle with its own normal rather than the vertex normals.
    pub flat: bool,
}

impl Default for Material {
    fn default() -> Material {
        Material {
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 1.0,
            emissive: Vec3::ZERO,
            normal_scale: [1.0, 1.0],
            occlusion_strength: 1.0,
            alpha: AlphaMode::Opaque,
            double_sided: false,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            emissive_texture: None,
            occlusion_texture: None,
            flat: false,
        }
    }
}

/// An overlay's look: its premultiplied-on-draw linear RGBA and its width in target pixels.
#[derive(Clone, Copy, Debug)]
pub struct LineStyle {
    pub color: Vec4,
    pub width: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameU {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    eye: [f32; 4],
    /// Width, height, and their reciprocals, in target pixels.
    viewport: [f32; 4],
    wire: [f32; 4],
    seam: [f32; 4],
    /// The wireframe's and the seams' widths, in target pixels.
    widths: [f32; 4],
    /// The focal lengths in target pixels (x, y).
    focal: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModelU {
    model: [[f32; 4]; 4],
    /// The inverse transpose; its last element is the sign of the model's determinant (a
    /// mirrored model's triangles face the other way).
    normal: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialU {
    base: [f32; 4],
    /// RGB, then the occlusion strength.
    emissive: [f32; 4],
    /// Metallic, roughness, alpha cutoff (0: none), blend (0 or 1).
    params: [f32; 4],
    /// xy, zero without a normal texture; z is 1 for flat shading.
    normal_scale: [f32; 4],
}

pub(crate) struct GpuMaterial {
    pub(crate) bind: wgpu::BindGroup,
    pub(crate) blend: bool,
    pub(crate) double_sided: bool,
}

/// Vertices that meshes index into. Meshes hold their own handles, so it can be removed while
/// they're still drawn.
#[derive(Clone)]
struct GpuGeometry {
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    uvs: wgpu::Buffer,
    /// None draws white.
    colors: Option<wgpu::Buffer>,
    center: Vec3,
    /// Kept for the overlays, which are built the first time they're shown.
    cpu: Arc<CpuVertices>,
}

struct CpuVertices {
    positions: Box<[f32]>,
    uvs: Box<[f32]>,
}

struct GpuMesh {
    geometry: GpuGeometry,
    indices: wgpu::Buffer,
    count: u32,
    material: u32,
    cpu_indices: Box<[u32]>,
    /// Per overlay: not built yet, or its segments (none when it has no lines).
    lines: [Option<Option<(wgpu::Buffer, u32)>>; 2],
}

/// A splat cloud: [`viz::SPLAT_FLOATS`] floats per splat, drawn in the order given.
struct GpuSplats {
    buffer: wgpu::Buffer,
    count: u32,
    center: Vec3,
}

/// What a [`Draw`] can name.
enum Drawable {
    Mesh(Box<GpuMesh>),
    Splats(GpuSplats),
    /// A cluster mesh: its id in [`Viewer::clusters`].
    Cluster(u32),
    /// A crowd: its id in [`Viewer::crowds`].
    Crowd(u32),
}

/// One mesh or splat cloud to draw this frame, where.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    /// From [`Viewer::add_mesh`], [`Viewer::add_splats`], [`Viewer::add_cluster_mesh`] or
    /// [`Viewer::add_crowd`].
    pub mesh: u32,
    pub model: Mat4,
    /// Whether the overlays that are on are drawn over it.
    pub overlays: bool,
}

/// One camera's view into part of the target.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    /// x, y (from the top left), width and height, in target pixels.
    pub rect: [u32; 4],
    pub camera: Camera,
    pub draws: &'a [Draw],
}

/// A uniform buffer of records `stride` bytes apart, bound at a dynamic offset.
struct Records {
    layout: wgpu::BindGroupLayout,
    size: usize,
    stride: usize,
    buffer: Option<(wgpu::Buffer, wgpu::BindGroup, usize)>,
}

impl Records {
    fn new(device: &wgpu::Device, label: &str, size: usize) -> Records {
        let stride =
            size.next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as usize);
        Records {
            layout: uniform_layout(device, label, true),
            size,
            stride,
            buffer: None,
        }
    }

    /// Writes `records` and returns the bind group.
    fn write<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[T],
    ) -> &wgpu::BindGroup {
        let mut bytes = vec![0u8; records.len().max(1) * self.stride];
        for (i, r) in records.iter().enumerate() {
            bytes[i * self.stride..][..self.size].copy_from_slice(bytemuck::bytes_of(r));
        }
        if !matches!(&self.buffer, Some((_, _, cap)) if *cap >= bytes.len()) {
            let cap = bytes.len().next_power_of_two();
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: cap as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(self.size as u64),
                    }),
                }],
            });
            self.buffer = Some((buf, bind, cap));
        }
        let (buf, bind, _) = self.buffer.as_ref().expect("record buffer");
        queue.write_buffer(buf, 0, &bytes);
        bind
    }

    fn offset(&self, i: usize) -> u32 {
        (i * self.stride) as u32
    }
}

struct Targets {
    size: (u32, u32),
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    /// With a [`Composite`]: the resolved frame, and the bind group it's read through.
    resolved: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

/// For a target that isn't sRGB (a canvas): the frame is drawn and resolved in an sRGB texture,
/// then copied over as premultiplied sRGB-encoded color, which is what browsers composite.
/// Blending through an sRGB view instead encodes premultiplied color after the fact, so partly
/// covered pixels (every antialiased edge, line and Gaussian) end up brighter than their alpha
/// allows and composite as a light fringe.
struct Composite {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

const COMPOSITE_WGSL: &str = "
@group(0) @binding(0) var frame_in: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let c = textureLoad(frame_in, vec2i(pos.xy), 0);
    if c.a <= 0.0 {
        return vec4f(0.0);
    }
    let l = min(c.rgb / c.a, vec3f(1.0));
    let srgb = select(1.055 * pow(l, vec3f(1.0 / 2.4)) - 0.055, l * 12.92, l <= vec3f(0.0031308));
    return vec4f(srgb * c.a, c.a);
}
";

/// The format frames are drawn in when the target isn't sRGB.
const COMPOSITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct Viewer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// What the pipelines draw into: the target's format, or [`COMPOSITE_FORMAT`].
    format: wgpu::TextureFormat,
    composite: Option<Composite>,
    /// Indexed by `blend as usize * 3 + cull`, cull being none, back, front.
    surfaces: Vec<wgpu::RenderPipeline>,
    lines: [wgpu::RenderPipeline; 2],
    /// Discs, then Gaussians.
    splats: [wgpu::RenderPipeline; 2],
    frames: Records,
    models: Records,
    material_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    flat: wgpu::TextureView,
    /// White vertex colors for geometry without its own, and how many vertices it covers.
    no_colors: (wgpu::Buffer, usize),
    targets: Option<Targets>,
    textures: Vec<Option<wgpu::TextureView>>,
    materials: Vec<Option<GpuMaterial>>,
    geometries: Vec<Option<GpuGeometry>>,
    meshes: Vec<Option<Drawable>>,
    styles: [LineStyle; 2],
    overlays: [bool; 2],
    gaussians: bool,
    /// Made with the first cluster mesh.
    clusters: Option<ClusterRenderer>,
    /// Made with the first crowd.
    crowds: Option<CrowdRenderer>,
}

fn uniform_layout(device: &wgpu::Device, label: &str, dynamic: bool) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Puts `item` in the first free slot of `slots`; returns its index.
fn insert<T>(slots: &mut Vec<Option<T>>, item: T) -> u32 {
    match slots.iter().position(Option::is_none) {
        Some(free) => {
            slots[free] = Some(item);
            free as u32
        }
        None => {
            slots.push(Some(item));
            slots.len() as u32 - 1
        }
    }
}

fn free<T>(slots: &mut [Option<T>], id: u32) {
    if let Some(s) = slots.get_mut(id as usize) {
        *s = None;
    }
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> u8 {
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// The mip chain below an RGBA8 image, each level a 2×2 box filter of the last (edge texels
/// repeated for odd sizes). sRGB color is averaged in linear light.
pub(crate) fn mip_chain(
    width: u32,
    height: u32,
    rgba: &[u8],
    srgb: bool,
) -> Vec<(u32, u32, Vec<u8>)> {
    let lut: Vec<f32> = (0..=255u8)
        .map(|c| {
            if srgb {
                srgb_to_linear(c)
            } else {
                f32::from(c) / 255.0
            }
        })
        .collect();
    let encode = |l: f32| {
        if srgb {
            linear_to_srgb(l)
        } else {
            (l * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        }
    };
    let mut out: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    let (mut w, mut h) = (width as usize, height as usize);
    while w > 1 || h > 1 {
        let src = out.last().map_or(rgba, |m| &m.2);
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut dst = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            let ys = [(2 * y).min(h - 1), (2 * y + 1).min(h - 1)];
            for x in 0..nw {
                let xs = [(2 * x).min(w - 1), (2 * x + 1).min(w - 1)];
                for c in 0..4 {
                    let sum: f32 = ys
                        .iter()
                        .flat_map(|&sy| xs.iter().map(move |&sx| (sy * w + sx) * 4 + c))
                        .map(|i| {
                            if c == 3 {
                                f32::from(src[i]) / 255.0
                            } else {
                                lut[src[i] as usize]
                            }
                        })
                        .sum();
                    let avg = sum / 4.0;
                    dst[(y * nw + x) * 4 + c] = if c == 3 {
                        (avg * 255.0 + 0.5) as u8
                    } else {
                        encode(avg)
                    };
                }
            }
        }
        out.push((nw as u32, nh as u32, dst));
        (w, h) = (nw, nh);
    }
    out
}

impl Viewer {
    /// A viewer drawing into targets of `format`. An sRGB format is drawn into directly, with
    /// premultiplied linear color; any other (a canvas's) gets premultiplied sRGB-encoded color.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Viewer {
        let (format, composite) = if target_format.is_srgb() {
            (target_format, None)
        } else {
            (
                COMPOSITE_FORMAT,
                Some(Self::composite(&device, target_format)),
            )
        };
        let frames = Records::new(&device, "frame", size_of::<FrameU>());
        let models = Records::new(&device, "model", size_of::<ModelU>());
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture(2),
                texture(3),
                texture(4),
                texture(5),
                texture(6),
            ],
        });
        let common = include_str!("common.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{common}\n{}",
                include_str!("pbr.wgsl")
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&frames.layout),
                Some(&models.layout),
                Some(&material_bgl),
            ],
            immediate_size: 0,
        });
        let position = wgpu::vertex_attr_array![0 => Float32x3];
        let normal = wgpu::vertex_attr_array![1 => Float32x3];
        let uv = wgpu::vertex_attr_array![2 => Float32x2];
        let color = wgpu::vertex_attr_array![3 => Float32x3];
        let buffer = |attributes, array_stride| wgpu::VertexBufferLayout {
            array_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes,
        };
        let buffers = [
            Some(buffer(&position[..], 12)),
            Some(buffer(&normal[..], 12)),
            Some(buffer(&uv[..], 8)),
            Some(buffer(&color[..], 12)),
        ];
        let multisample = wgpu::MultisampleState {
            count: SAMPLES,
            ..Default::default()
        };
        let mut surfaces = Vec::new();
        for blend in [false, true] {
            for cull in [None, Some(wgpu::Face::Back), Some(wgpu::Face::Front)] {
                surfaces.push(
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("surface"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_main"),
                            compilation_options: Default::default(),
                            buffers: &buffers,
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
                            // Pushed back a little so the overlays' lines, on the same edges, win.
                            bias: wgpu::DepthBiasState {
                                constant: 2,
                                slope_scale: 1.5,
                                clamp: 0.0,
                            },
                        }),
                        multisample,
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs_main"),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    }),
                );
            }
        }
        let lines = viz::line_pipelines(
            &device,
            &[&frames.layout, &models.layout],
            format,
            DEPTH,
            SAMPLES,
        );
        let splats = viz::splat_pipelines(
            &device,
            &[&frames.layout, &models.layout],
            format,
            DEPTH,
            SAMPLES,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 4,
            ..Default::default()
        });
        Viewer {
            white: Self::pixel(&device, &queue, [255; 4]),
            flat: Self::pixel(&device, &queue, [128, 128, 255, 255]),
            no_colors: (Self::white_colors(&device, &queue, 0), 0),
            device,
            queue,
            format,
            composite,
            surfaces,
            lines,
            splats,
            frames,
            models,
            material_bgl,
            sampler,
            targets: None,
            textures: Vec::new(),
            materials: Vec::new(),
            geometries: Vec::new(),
            meshes: Vec::new(),
            styles: [
                LineStyle {
                    color: Vec4::new(0.0, 0.1, 0.55, 0.7),
                    width: 1.25,
                },
                LineStyle {
                    color: Vec4::new(0.8, 0.1, 0.0, 1.0),
                    width: 2.5,
                },
            ],
            overlays: [false; 2],
            gaussians: true,
            clusters: None,
            crowds: None,
        }
    }

    fn composite(device: &wgpu::Device, target: wgpu::TextureFormat) -> Composite {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPOSITE_WGSL)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Composite { pipeline, layout }
    }

    /// A vertex buffer of white RGB for at least `vertices` vertices.
    fn white_colors(device: &wgpu::Device, queue: &wgpu::Queue, vertices: usize) -> wgpu::Buffer {
        let colors = vec![1.0f32; 3 * vertices.max(1)];
        let b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("white"),
            size: (colors.len() * 4) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b, 0, bytemuck::cast_slice(&colors));
        b
    }

    fn buffer(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (data.len() as u64).max(4).next_multiple_of(4),
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&b, 0, data);
        b
    }

    fn pixel(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4]) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            tex.as_image_copy(),
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        tex.create_view(&Default::default())
    }

    /// Uploads an RGBA8 image (rows top to bottom) with a mip chain built from it. `srgb` for
    /// color (base color, emissive), not for data (normals, metal-roughness, occlusion).
    pub fn add_texture(&mut self, width: u32, height: u32, rgba: &[u8], srgb: bool) -> u32 {
        let (width, height) = (width.max(1), height.max(1));
        let rgba: Cow<[u8]> = if rgba.len() == (width * height * 4) as usize {
            Cow::Borrowed(rgba)
        } else {
            Cow::Owned(vec![255; (width * height * 4) as usize])
        };
        let mips = mip_chain(width, height, &rgba, srgb);
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1 + mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, (w, h, data)) in std::iter::once((width, height, &*rgba))
            .chain(mips.iter().map(|(w, h, d)| (*w, *h, &d[..])))
            .enumerate()
        {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        insert(&mut self.textures, tex.create_view(&Default::default()))
    }

    /// Frees a texture; materials already made with it keep it.
    pub fn remove_texture(&mut self, id: u32) {
        free(&mut self.textures, id);
    }

    pub fn add_material(&mut self, m: &Material) -> u32 {
        let (cutoff, blend) = match m.alpha {
            AlphaMode::Opaque => (0.0, false),
            AlphaMode::Mask(c) => (c.max(1e-4), false),
            AlphaMode::Blend => (0.0, true),
        };
        let has_normals = m
            .normal_texture
            .is_some_and(|t| self.textures.get(t as usize).is_some_and(Option::is_some));
        let u = MaterialU {
            base: m.base_color.to_array(),
            emissive: m.emissive.extend(m.occlusion_strength).to_array(),
            params: [m.metallic, m.roughness, cutoff, f32::from(u8::from(blend))],
            normal_scale: [
                if has_normals { m.normal_scale[0] } else { 0.0 },
                if has_normals { m.normal_scale[1] } else { 0.0 },
                f32::from(u8::from(m.flat)),
                0.0,
            ],
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<MaterialU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&u));
        let view = |t: Option<u32>, fallback| {
            t.and_then(|t| self.textures.get(t as usize)?.as_ref())
                .unwrap_or(fallback)
        };
        let views = [
            view(m.base_color_texture, &self.white),
            view(m.metallic_roughness_texture, &self.white),
            view(m.normal_texture, &self.flat),
            view(m.emissive_texture, &self.white),
            view(m.occlusion_texture, &self.white),
        ];
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
        ];
        entries.extend(views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry {
            binding: 2 + i as u32,
            resource: wgpu::BindingResource::TextureView(v),
        }));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.material_bgl,
            entries: &entries,
        });
        insert(
            &mut self.materials,
            GpuMaterial {
                bind,
                blend,
                double_sided: m.double_sided,
            },
        )
    }

    /// Frees a material; meshes using it aren't drawn until given another.
    pub fn remove_material(&mut self, id: u32) {
        free(&mut self.materials, id);
    }

    /// Uploads vertices for meshes to index: xyz positions and normals (`3 * n` floats each),
    /// uvs (`2 * n` floats, or none) and linear RGB colors (`3 * n` floats, or none for white).
    pub fn add_geometry(
        &mut self,
        positions: &[f32],
        normals: &[f32],
        uvs: &[f32],
        colors: &[f32],
    ) -> u32 {
        let n = positions.len() / 3;
        let positions = &positions[..3 * n];
        let normals = if normals.len() == 3 * n {
            Cow::Borrowed(normals)
        } else {
            Cow::Owned(vec![0.0; 3 * n])
        };
        let uvs: Cow<[f32]> = if uvs.len() == 2 * n {
            Cow::Borrowed(uvs)
        } else {
            Cow::Owned(vec![0.0; 2 * n])
        };
        let colors = (colors.len() == 3 * n && n > 0)
            .then(|| self.buffer(bytemuck::cast_slice(colors), wgpu::BufferUsages::VERTEX));
        if colors.is_none() && n > self.no_colors.1 {
            let cover = n.next_power_of_two();
            self.no_colors = (Self::white_colors(&self.device, &self.queue, cover), cover);
        }
        let (lo, hi) = positions.as_chunks::<3>().0.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(Vec3::from_array(*p)), hi.max(Vec3::from_array(*p))),
        );
        let geometry = GpuGeometry {
            positions: self.buffer(bytemuck::cast_slice(positions), wgpu::BufferUsages::VERTEX),
            normals: self.buffer(bytemuck::cast_slice(&normals), wgpu::BufferUsages::VERTEX),
            uvs: self.buffer(bytemuck::cast_slice(&uvs), wgpu::BufferUsages::VERTEX),
            colors,
            center: if n > 0 { (lo + hi) / 2.0 } else { Vec3::ZERO },
            cpu: Arc::new(CpuVertices {
                positions: positions.into(),
                uvs: uvs.into_owned().into(),
            }),
        };
        insert(&mut self.geometries, geometry)
    }

    /// Frees a geometry; meshes already made from it keep its vertices.
    pub fn remove_geometry(&mut self, id: u32) {
        free(&mut self.geometries, id);
    }

    /// Makes a triangle mesh from a geometry's vertices: `indices` (three per triangle) drawn
    /// with `material`. Returns the id a [`Draw`] names, or None for an unknown geometry.
    pub fn add_mesh(&mut self, geometry: u32, indices: &[u32], material: u32) -> Option<u32> {
        let geometry = self.geometries.get(geometry as usize)?.clone()?;
        let n = geometry.cpu.positions.len() / 3;
        let indices: Vec<u32> = indices
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|t| t.iter().all(|&i| (i as usize) < n))
            .flatten()
            .copied()
            .collect();
        let mesh = GpuMesh {
            indices: self.buffer(bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX),
            count: indices.len() as u32,
            geometry,
            material,
            cpu_indices: indices.into(),
            lines: [None, None],
        };
        Some(insert(&mut self.meshes, Drawable::Mesh(Box::new(mesh))))
    }

    /// Uploads a splat cloud, [`viz::SPLAT_FLOATS`] floats per splat, drawn in this order (far
    /// to near, for Gaussians). Returns the id a [`Draw`] names.
    pub fn add_splats(&mut self, splats: &[f32]) -> u32 {
        let count = splats.len() / viz::SPLAT_FLOATS;
        let splats = &splats[..count * viz::SPLAT_FLOATS];
        let sum = splats
            .as_chunks::<{ viz::SPLAT_FLOATS }>()
            .0
            .iter()
            .fold(Vec3::ZERO, |c, s| c + Vec3::from_slice(&s[..3]));
        let buffer = self.buffer(bytemuck::cast_slice(splats), wgpu::BufferUsages::VERTEX);
        insert(
            &mut self.meshes,
            Drawable::Splats(GpuSplats {
                buffer,
                count: count as u32,
                center: sum / count.max(1) as f32,
            }),
        )
    }

    /// Replaces a splat cloud's splats (re-sorted, say).
    pub fn set_splats(&mut self, id: u32, splats: &[f32]) {
        let count = splats.len() / viz::SPLAT_FLOATS;
        let splats = &splats[..count * viz::SPLAT_FLOATS];
        let fits = matches!(self.meshes.get(id as usize), Some(Some(Drawable::Splats(s))) if s.buffer.size() >= (splats.len() * 4) as u64);
        let grown =
            (!fits).then(|| self.buffer(bytemuck::cast_slice(splats), wgpu::BufferUsages::VERTEX));
        let Some(Some(Drawable::Splats(s))) = self.meshes.get_mut(id as usize) else {
            return;
        };
        match grown {
            Some(b) => s.buffer = b,
            None => self
                .queue
                .write_buffer(&s.buffer, 0, bytemuck::cast_slice(splats)),
        }
        s.count = count as u32;
    }

    /// Whether splats are drawn as blended Gaussians (the default) or opaque discs.
    pub fn set_splat_shape(&mut self, gaussians: bool) {
        self.gaussians = gaussians;
    }

    /// Frees a mesh or splat cloud; its id may be reused.
    pub fn remove_mesh(&mut self, id: u32) {
        if let (Some(Some(Drawable::Cluster(c))), Some(r)) =
            (self.meshes.get(id as usize), self.clusters.as_mut())
        {
            r.remove_mesh(*c);
        }
        if let (Some(Some(Drawable::Crowd(c))), Some(r)) = (self.meshes.get(id as usize), self.crowds.as_mut()) {
            r.remove(*c);
        }
        free(&mut self.meshes, id);
    }

    /// Whether this device can draw cluster meshes (WebGPU or native; not WebGL2).
    pub fn clusters_supported(&self) -> bool {
        crate::clusters::supported(&self.device)
    }

    /// A packed cluster DAG, cut, culled and drawn on the GPU, its pages streamed in as the
    /// views need them (see `clusters.rs`). Each draw naming it is an instance. Its triangles
    /// draw with the default material until [`Viewer::set_cluster_materials`]. None when
    /// [`Viewer::clusters_supported`] is false.
    pub fn add_cluster_mesh(&mut self, dag: GpuClusterDag) -> Option<u32> {
        if !self.clusters_supported() {
            return None;
        }
        if self.clusters.is_none() {
            let default = self.add_material(&Material::default());
            self.clusters = Some(ClusterRenderer::new(
                &self.device,
                self.format,
                &self.frames.layout,
                &self.models.layout,
                &self.material_bgl,
                default,
            ));
        }
        let c = self.clusters.as_mut()?.add_mesh(dag);
        Some(insert(&mut self.meshes, Drawable::Cluster(c)))
    }

    /// A crowd of skinned units animated from `palette` (see `crowd.rs`), with up to
    /// [`crate::CROWD_MAX_LODS`] levels of detail that are bone LODs of the palette's
    /// skeleton, drawn with `material`. Each draw naming it places the whole crowd. It has no
    /// units until [`Viewer::set_crowd_units`]. Fails when [`Viewer::clusters_supported`] is
    /// false (crowds need the same compute and storage support) or a level doesn't match the
    /// palette.
    pub fn add_crowd(
        &mut self,
        palette: &farthing::anim::bake::PosePalette,
        lods: &[CrowdLod],
        material: u32,
    ) -> Result<u32, String> {
        if !crate::crowd::supported(&self.device) {
            return Err("crowds need WebGPU".into());
        }
        let r = self.crowds.get_or_insert_with(|| {
            CrowdRenderer::new(&self.device, self.format, &self.frames.layout, &self.models.layout, &self.material_bgl)
        });
        let c = r.add(&self.device, &self.queue, palette, lods, material)?;
        Ok(insert(&mut self.meshes, Drawable::Crowd(c)))
    }

    fn crowd_id(&self, id: u32) -> Option<u32> {
        match self.meshes.get(id as usize) {
            Some(Some(Drawable::Crowd(c))) => Some(*c),
            _ => None,
        }
    }

    /// Writes a crowd's units `first..first + units.len()`, growing it to hold them. Only the
    /// changed range is uploaded, at the next render.
    pub fn set_crowd_units(&mut self, id: u32, first: usize, units: &[CrowdUnit]) {
        if let (Some(c), Some(r)) = (self.crowd_id(id), self.crowds.as_mut()) {
            r.set_units(c, first, units);
        }
    }

    /// Keeps a crowd's first `count` units.
    pub fn set_crowd_unit_count(&mut self, id: u32, count: usize) {
        if let (Some(c), Some(r)) = (self.crowd_id(id), self.crowds.as_mut()) {
            r.set_unit_count(c, count);
        }
    }

    /// How many units a crowd has.
    pub fn crowd_unit_count(&self, id: u32) -> usize {
        match (self.crowd_id(id), &self.crowds) {
            (Some(c), Some(r)) => r.units(c),
            _ => 0,
        }
    }

    /// The clock crowd units' start times are on, seconds.
    pub fn set_crowd_time(&mut self, seconds: f32) {
        if let Some(r) = self.crowds.as_mut() {
            r.set_time(seconds);
        }
    }

    /// Whether crowds skip units outside the view (on by default).
    pub fn set_crowd_culling(&mut self, on: bool) {
        if let Some(r) = self.crowds.as_mut() {
            r.set_frustum_culling(on);
        }
    }

    /// Whether crowd units are tinted by their level of detail.
    pub fn set_crowd_show_lods(&mut self, on: bool) {
        if let Some(r) = self.crowds.as_mut() {
            r.set_show_lods(on);
        }
    }

    /// What a crowd drew, a frame or two behind (read back without stalling).
    pub fn crowd_stats(&self, id: u32) -> Option<CrowdStats> {
        self.crowds.as_ref()?.stats(self.crowd_id(id)?)
    }

    fn cluster_id(&self, id: u32) -> Option<u32> {
        match self.meshes.get(id as usize) {
            Some(Some(Drawable::Cluster(c))) => Some(*c),
            _ => None,
        }
    }

    /// The material each of a cluster mesh's material ids draws with.
    pub fn set_cluster_materials(&mut self, id: u32, materials: &[u32]) {
        if let (Some(c), Some(r)) = (self.cluster_id(id), self.clusters.as_mut()) {
            r.set_materials(c, materials);
        }
    }

    /// How cluster meshes are colored: 0 their materials, 1 a color per cluster, 2 a color
    /// per DAG level. The colors multiply `material`'s base color (None: the default material).
    pub fn set_cluster_shading(&mut self, mode: u32, material: Option<u32>) {
        if let Some(r) = self.clusters.as_mut() {
            r.set_shading(mode, material);
        }
    }

    /// The largest error the cluster cut leaves on screen, in pixels.
    pub fn set_cluster_threshold(&mut self, pixels: f32) {
        if let Some(r) = self.clusters.as_mut() {
            r.set_threshold(pixels);
        }
    }

    /// The page pool's size and how many pages may load per frame. A new size starts
    /// streaming over.
    pub fn set_cluster_streaming(&mut self, budget_bytes: u64, pages_per_frame: usize) {
        if let Some(r) = self.clusters.as_mut() {
            r.set_streaming(budget_bytes, pages_per_frame);
        }
    }

    /// Frustum and normal-cone culling of clusters.
    pub fn set_cluster_culling(&mut self, frustum: bool, cone: bool) {
        if let Some(r) = self.clusters.as_mut() {
            r.set_culling(frustum, cone);
        }
    }

    /// Keeps the cluster cut and culling where the cameras are now, to look at it from
    /// elsewhere.
    pub fn set_cluster_freeze(&mut self, on: bool) {
        if let Some(r) = self.clusters.as_mut() {
            r.set_freeze(on);
        }
    }

    /// What the cluster cut drew per view and what's resident, a frame or two behind.
    pub fn cluster_stats(&self) -> Option<ClusterStats> {
        self.clusters.as_ref().map(ClusterRenderer::stats)
    }

    pub fn set_line_style(&mut self, overlay: Overlay, style: LineStyle) {
        self.styles[overlay as usize] = style;
    }

    /// Which overlays are drawn over the draws that take them.
    pub fn set_overlays(&mut self, wireframe: bool, seams: bool) {
        self.overlays = [wireframe, seams];
    }

    fn targets(&mut self, width: u32, height: u32) {
        if self
            .targets
            .as_ref()
            .is_some_and(|t| t.size == (width, height))
        {
            return;
        }
        let make = |format, label, sample_count, usage| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let resolved = self.composite.as_ref().map(|c| {
            let view = make(
                self.format,
                "resolved",
                1,
                attachment | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite"),
                layout: &c.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
            (view, bind)
        });
        self.targets = Some(Targets {
            size: (width, height),
            color: make(self.format, "color", SAMPLES, attachment),
            depth: make(DEPTH, "depth", SAMPLES, attachment),
            resolved,
        });
    }

    /// Builds the overlays that are on for the meshes `views` draw them over, if not yet built.
    fn build_lines(&mut self, views: &[View]) {
        for (o, _) in self.overlays.iter().enumerate().filter(|(_, on)| **on) {
            for d in views.iter().flat_map(|v| v.draws).filter(|d| d.overlays) {
                let Some(Some(Drawable::Mesh(m))) = self.meshes.get_mut(d.mesh as usize) else {
                    continue;
                };
                if m.lines[o].is_some() {
                    continue;
                }
                let (p, uv) = (&m.geometry.cpu.positions, &m.geometry.cpu.uvs);
                let segments = if o == Overlay::Wireframe as usize {
                    viz::wire_segments(p, &m.cpu_indices)
                } else {
                    viz::seam_segments(p, uv, &m.cpu_indices)
                };
                m.lines[o] = Some((!segments.is_empty()).then(|| {
                    let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("lines"),
                        size: (segments.len() * 4) as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue
                        .write_buffer(&b, 0, bytemuck::cast_slice(&segments));
                    (b, (segments.len() / 6) as u32)
                }));
            }
        }
    }

    /// Draws `views` into `target` (`width` by `height`), cleared to transparent first. Draws
    /// naming a removed mesh or material are skipped.
    pub fn render(&mut self, target: &wgpu::TextureView, width: u32, height: u32, views: &[View]) {
        let (width, height) = (width.max(1), height.max(1));
        self.targets(width, height);
        self.build_lines(views);
        let style = |s: LineStyle| s.color.to_array();
        let mut frames = Vec::new();
        let mut models = Vec::new();
        let mut passes = Vec::new();
        let mut cluster_views = Vec::new();
        let mut crowd_views = Vec::new();
        let wire = self.overlays[0];
        for v in views {
            let [x, y, w, h] = v.rect;
            let (x, y) = (x.min(width - 1), y.min(height - 1));
            let (w, h) = (w.clamp(1, width - x), h.clamp(1, height - y));
            let focal = h as f32 / 2.0 / (v.camera.fov_y / 2.0).tan();
            frames.push(FrameU {
                view_proj: v.camera.view_proj(w as f32 / h as f32).to_cols_array_2d(),
                view: v.camera.view.to_cols_array_2d(),
                eye: v.camera.eye().extend(1.0).to_array(),
                viewport: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
                wire: style(self.styles[0]),
                seam: style(self.styles[1]),
                widths: [self.styles[0].width, self.styles[1].width, 0.0, 0.0],
                focal: [focal, focal, 0.0, 0.0],
            });
            let mut items = Vec::new();
            let mut cluster_draws = Vec::new();
            let mut crowd_draws = Vec::new();
            for d in v.draws {
                let det = d.model.determinant();
                let (pipe, center) = match self.meshes.get(d.mesh as usize) {
                    Some(Some(Drawable::Cluster(c))) => {
                        cluster_draws.push(ClusterDraw {
                            mesh: *c,
                            model: d.model,
                            wire: wire && d.overlays,
                        });
                        continue;
                    }
                    Some(Some(Drawable::Crowd(c))) => {
                        crowd_draws.push(CrowdDraw {
                            crowd: *c,
                            model: d.model,
                        });
                        continue;
                    }
                    Some(Some(Drawable::Mesh(mesh))) => {
                        let Some(Some(mat)) = self.materials.get(mesh.material as usize) else {
                            continue;
                        };
                        let cull = if mat.double_sided {
                            0
                        } else if det < 0.0 {
                            2
                        } else {
                            1
                        };
                        (
                            Pipe::Surface(usize::from(mat.blend) * 3 + cull),
                            mesh.geometry.center,
                        )
                    }
                    Some(Some(Drawable::Splats(s))) => (Pipe::Splats(self.gaussians), s.center),
                    _ => continue,
                };
                let mut normal = Mat4::from_mat3(Mat3::from_mat4(d.model).inverse().transpose());
                normal.w_axis.w = if det < 0.0 { -1.0 } else { 1.0 };
                let depth = (v.camera.view * d.model).transform_point3(center).z;
                items.push(Item {
                    model: models.len(),
                    id: d.mesh as usize,
                    pipe,
                    depth,
                    overlays: d.overlays,
                });
                models.push(ModelU {
                    model: d.model.to_cols_array_2d(),
                    normal: normal.to_cols_array_2d(),
                });
            }
            // Opaque first, grouped by pipeline; blended after, far (most negative z) to near.
            items.sort_by(|a, b| {
                a.pipe
                    .blended()
                    .cmp(&b.pipe.blended())
                    .then(if a.pipe.blended() {
                        a.depth.total_cmp(&b.depth)
                    } else {
                        a.pipe.cmp(&b.pipe)
                    })
            });
            passes.push(([x, y, w, h], items));
            cluster_views.push(ClusterView {
                view_proj: v.camera.view_proj(w as f32 / h as f32),
                eye: v.camera.eye(),
                ppr: focal,
                draws: cluster_draws,
            });
            crowd_views.push(CrowdView {
                view_proj: v.camera.view_proj(w as f32 / h as f32),
                eye: v.camera.eye(),
                focal,
                draws: crowd_draws,
            });
        }
        let frame_bind = self
            .frames
            .write(&self.device, &self.queue, &frames)
            .clone();
        let model_bind = self
            .models
            .write(&self.device, &self.queue, &models)
            .clone();
        let targets = self.targets.as_ref().expect("render targets");
        let mut enc = self.device.create_command_encoder(&Default::default());
        if let Some(c) = self.clusters.as_mut() {
            c.prepare(&self.device, &self.queue, &mut enc, &cluster_views, &self.materials);
        }
        if let Some(c) = self.crowds.as_mut() {
            c.prepare(&self.device, &self.queue, &mut enc, &crowd_views);
        }
        let clusters = self.clusters.as_ref();
        let crowds = self.crowds.as_ref();
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color,
                    depth_slice: None,
                    resolve_target: Some(targets.resolved.as_ref().map_or(target, |r| &r.0)),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (vi, ([x, y, w, h], items)) in passes.iter().enumerate() {
                pass.set_viewport(*x as f32, *y as f32, *w as f32, *h as f32, 0.0, 1.0);
                pass.set_scissor_rect(*x, *y, *w, *h);
                pass.set_bind_group(0, &frame_bind, &[self.frames.offset(vi)]);
                // Cluster pipelines take a model group they don't read.
                let clusters = clusters.filter(|c| c.draws_in(vi));
                if let Some(c) = clusters {
                    pass.set_bind_group(1, &model_bind, &[0]);
                    c.draw(&mut pass, vi, false, &self.materials);
                }
                // Crowd pipelines too.
                if let Some(c) = crowds.filter(|c| c.draws_in(vi)) {
                    pass.set_bind_group(1, &model_bind, &[0]);
                    c.draw(&mut pass, vi, &self.materials);
                }
                let mut bound = None;
                for item in items {
                    if bound != Some(item.pipe) {
                        pass.set_pipeline(match item.pipe {
                            Pipe::Surface(i) => &self.surfaces[i],
                            Pipe::Splats(gauss) => &self.splats[usize::from(gauss)],
                        });
                        bound = Some(item.pipe);
                    }
                    pass.set_bind_group(1, &model_bind, &[self.models.offset(item.model)]);
                    match &self.meshes[item.id] {
                        Some(Drawable::Mesh(m)) => {
                            let Some(Some(mat)) = self.materials.get(m.material as usize) else {
                                continue;
                            };
                            let g = &m.geometry;
                            pass.set_bind_group(2, &mat.bind, &[]);
                            pass.set_vertex_buffer(0, g.positions.slice(..));
                            pass.set_vertex_buffer(1, g.normals.slice(..));
                            pass.set_vertex_buffer(2, g.uvs.slice(..));
                            pass.set_vertex_buffer(
                                3,
                                g.colors.as_ref().unwrap_or(&self.no_colors.0).slice(..),
                            );
                            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..m.count, 0, 0..1);
                        }
                        Some(Drawable::Splats(s)) if s.count > 0 => {
                            pass.set_vertex_buffer(0, s.buffer.slice(..));
                            pass.draw(0..6, 0..s.count);
                        }
                        _ => {}
                    }
                }
                if let Some(c) = clusters {
                    pass.set_bind_group(1, &model_bind, &[0]);
                    c.draw(&mut pass, vi, true, &self.materials);
                }
                for o in (0..2).filter(|&o| self.overlays[o]) {
                    pass.set_pipeline(&self.lines[o]);
                    for item in items.iter().filter(|i| i.overlays) {
                        let Some(Drawable::Mesh(m)) = &self.meshes[item.id] else {
                            continue;
                        };
                        let Some(Some((buf, n))) = &m.lines[o] else {
                            continue;
                        };
                        pass.set_bind_group(1, &model_bind, &[self.models.offset(item.model)]);
                        pass.set_vertex_buffer(0, buf.slice(..));
                        pass.draw(0..6, 0..*n);
                    }
                }
            }
        }
        if let (Some(c), Some((_, bind))) = (&self.composite, &targets.resolved) {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&c.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        if let Some(c) = self.clusters.as_mut() {
            c.after_submit();
        }
        if let Some(c) = self.crowds.as_mut() {
            c.after_submit();
        }
    }
}

/// Which pipeline draws an item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Pipe {
    /// Indexed like [`Viewer::surfaces`].
    Surface(usize),
    /// Gaussians (true) or discs.
    Splats(bool),
}

impl Pipe {
    /// Drawn after everything opaque, far to near.
    fn blended(self) -> bool {
        matches!(self, Pipe::Surface(3..) | Pipe::Splats(true))
    }
}

/// One draw of a view, ready to sort and record.
struct Item {
    /// Its model record.
    model: usize,
    id: usize,
    pipe: Pipe,
    /// Its center's camera-space z.
    depth: f32,
    overlays: bool,
}

#[cfg(feature = "native")]
impl Viewer {
    /// A viewer on the default adapter, for drawing offscreen with [`Viewer::render_image`].
    pub fn headless() -> anyhow::Result<Viewer> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("viewer"),
                ..Default::default()
            }))?;
        Ok(Viewer::new(device, queue, COMPOSITE_FORMAT))
    }

    /// Draws `views` into a new `width` by `height` image and reads it back: sRGB RGBA8 rows,
    /// straight (not premultiplied) alpha, transparent where nothing was drawn.
    pub fn render_image(&mut self, width: u32, height: u32, views: &[View]) -> Vec<u8> {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.render(&tex.create_view(&Default::default()), width, height, views);
        let bpr = (width * 4).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bpr * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            tex.size(),
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("mapped");
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for row in data.chunks(bpr as usize).take(height as usize) {
            for p in row[..(width * 4) as usize].chunks(4) {
                // Premultiplied linear color stored as sRGB: unpremultiply in linear.
                let a = f32::from(p[3]) / 255.0;
                if a <= 0.0 {
                    out.extend([0, 0, 0, 0]);
                    continue;
                }
                for &c in &p[..3] {
                    out.push(linear_to_srgb(srgb_to_linear(c) / a));
                }
                out.push(p[3]);
            }
        }
        drop(data);
        buf.unmap();
        out
    }

    /// Waits for the GPU, so the next render streams by everything rendered so far.
    pub fn settle(&mut self) {
        if let Some(c) = &self.clusters {
            c.settle(&self.device);
        }
        if let Some(c) = self.crowds.as_mut() {
            c.settle(&self.device);
        }
    }

    /// The clusters the last render drew: per view, (instance, cluster) sorted, where the
    /// instance counts that view's draws of cluster meshes. For tests.
    pub fn cluster_visible(&self) -> Vec<Vec<(u32, u32)>> {
        self.clusters
            .as_ref()
            .map(|c| c.read_visible(&self.device, &self.queue))
            .unwrap_or_default()
    }

    /// Per page of a cluster mesh, whether it's resident. For tests.
    pub fn cluster_resident(&self, id: u32) -> Vec<bool> {
        match (self.cluster_id(id), &self.clusters) {
            (Some(c), Some(r)) => r.resident(c),
            _ => Vec::new(),
        }
    }
}
