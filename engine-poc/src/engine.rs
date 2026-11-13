//! The PoC engine: a headless wgpu renderer with its own material system, implementing
//! farthing's `RenderHost`. It talks to farthing only through the public API.

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use farthing::glam::{Mat4, Vec3, Vec4};
use farthing::host::{
    channels, BakedMaterialDesc, BakedMaterialKind, ChannelEncoding, ChannelImage, ChannelImages, MaterialClass, MaterialInfo, RenderHost,
    SurfaceBakeRequest, View, ViewImages, ViewRenderRequest,
};
use farthing::image::{linear_to_srgb, Image};
use farthing::{MaterialId, Mesh, Scene};
use std::borrow::Cow;

const RT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The engine's own material description. farthing only ever sees a `MaterialId`.
#[derive(Clone, Debug)]
pub enum MaterialDef {
    /// World-space 3D checker (depends on world position).
    Checker { a: Vec4, b: Vec4, scale: f32 },
    /// Procedural marble with bump-mapped normals (noise in world space).
    Marble { a: Vec4, b: Vec4, scale: f32 },
    /// Texture-driven (glTF base color, or a farthing bake). Normal map is tangent space.
    Textured { base: Image, normal: Option<Image>, factor: Vec4, cutoff: f32 },
}

struct GpuMaterial {
    #[allow(dead_code)]
    def: MaterialDef,
    info: MaterialInfo,
    bind: wgpu::BindGroup,
    /// For impostor materials: atlas textures and layout.
    impostor: Option<ImpostorMat>,
}

/// One surfel for [`Engine::render_splats`]: position, normal, radius, RGBA8 albedo.
pub type SplatInstance = ([f32; 3], [f32; 3], f32, [u8; 4]);

struct ImpostorMat {
    color: wgpu::TextureView,
    normal: wgpu::TextureView,
    /// Second depth layer (layered impostors); the first layer's views otherwise.
    color2: Option<wgpu::TextureView>,
    normal2: Option<wgpu::TextureView>,
    frames_per_side: u32,
    hemisphere: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawU {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    view: [[f32; 4]; 4],
    lit: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatU {
    kind: u32,
    flags: u32,
    scale: f32,
    cutoff: f32,
    color_a: [f32; 4],
    color_b: [f32; 4],
    factor: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImpU {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    center_radius: [f32; 4],
    frames: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
    uv: [f32; 2],
    tan: [f32; 4],
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_name: String,
    materials: Vec<GpuMaterial>,
    draw_bgl: wgpu::BindGroupLayout,
    mat_bgl: wgpu::BindGroupLayout,
    bake_bgl: wgpu::BindGroupLayout,
    imp_bgl: wgpu::BindGroupLayout,
    scene_pipeline: wgpu::RenderPipeline,
    bake_pipeline: wgpu::RenderPipeline,
    imp_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    no_peel: wgpu::TextureView,
}

fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

impl Engine {
    pub fn new() -> Result<Engine> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        let adapter_name = adapter.get_info().name;
        // Three RGBA32F render targets need 48 bytes per sample (default limit is 32).
        let mut limits = wgpu::Limits::default();
        limits.max_color_attachment_bytes_per_sample = adapter.limits().max_color_attachment_bytes_per_sample;
        if limits.max_color_attachment_bytes_per_sample < 48 {
            return Err(anyhow!("adapter allows only {} color bytes per sample; need 48", limits.max_color_attachment_bytes_per_sample));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("farthing-poc"), required_limits: limits, ..Default::default() }))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("engine"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("engine.wgsl"))) });
        let imp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("impostor"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(farthing::impostor::REFERENCE_SHADER)),
        });
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        // Binding 1: depth layer to peel (linear depth in r, coverage in a) for layered captures.
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("draw"), entries: &[uniform_entry(0, vf), tex_entry(1, false)] });
        let mat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake-inputs"),
            entries: &[tex_entry(0, false), tex_entry(1, false), tex_entry(2, false), tex_entry(3, false), uniform_entry(4, wgpu::ShaderStages::FRAGMENT)],
        });
        let imp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("impostor"),
            entries: &[
                uniform_entry(0, vf),
                tex_entry(1, true),
                tex_entry(2, true),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                tex_entry(4, true),
                tex_entry(5, true),
            ],
        });
        let target = |f| Some(wgpu::ColorTargetState { format: f, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let depth = Some(wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&draw_bgl), Some(&mat_bgl)], immediate_size: 0 });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_scene"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &attrs })],
            },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: depth.clone(),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_scene"), compilation_options: Default::default(), targets: &[target(RT), target(RT), target(RT)] }),
            multiview_mask: None,
            cache: None,
        });
        let bake_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[None, Some(&mat_bgl), Some(&bake_bgl)], immediate_size: 0 });
        let bake_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bake"),
            layout: Some(&bake_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_bake"), compilation_options: Default::default(), targets: &[target(RT), target(RT)] }),
            multiview_mask: None,
            cache: None,
        });
        let imp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&imp_bgl)], immediate_size: 0 });
        let imp_attrs = wgpu::vertex_attr_array![0 => Float32x2];
        let imp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("impostor"),
            layout: Some(&imp_layout),
            vertex: wgpu::VertexState {
                module: &imp_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: 8, step_mode: wgpu::VertexStepMode::Vertex, attributes: &imp_attrs })],
            },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: depth,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &imp_shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[target(RT)] }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let white = upload_rgba8(&device, &queue, &Image::new(1, 1, Vec4::ONE), false);
        let no_peel = upload_f32(&device, &queue, 1, 1, &[[0.0; 4]]);
        Ok(Engine {
            device,
            queue,
            adapter_name,
            materials: Vec::new(),
            draw_bgl,
            mat_bgl,
            bake_bgl,
            imp_bgl,
            scene_pipeline,
            bake_pipeline,
            imp_pipeline,
            sampler,
            white,
            no_peel,
        })
    }

    fn upload_rgba8(&self, img: &Image, srgb: bool) -> wgpu::TextureView {
        upload_rgba8(&self.device, &self.queue, img, srgb)
    }
}

fn upload_f32(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, px: &[[f32; 4]]) -> wgpu::TextureView {
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(px),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 16), rows_per_image: Some(h) },
            size,
        );
        tex.create_view(&Default::default())
    }

fn upload_rgba8(device: &wgpu::Device, queue: &wgpu::Queue, img: &Image, srgb: bool) -> wgpu::TextureView {
        let mut data = Vec::with_capacity(img.pixels.len() * 4);
        for p in &img.pixels {
            let rgb = if srgb { [linear_to_srgb(p.x), linear_to_srgb(p.y), linear_to_srgb(p.z)] } else { [p.x, p.y, p.z] };
            for c in rgb {
                data.push((c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            data.push((p.w.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        let size = wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(img.width * 4), rows_per_image: Some(img.height) },
            size,
        );
        tex.create_view(&Default::default())
}

impl Engine {

    fn upload_f32(&self, w: u32, h: u32, px: &[[f32; 4]]) -> wgpu::TextureView {
        upload_f32(&self.device, &self.queue, w, h, px)
    }

    fn uniform<T: Pod>(&self, v: &T) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(v));
        buf
    }

    fn target(&self, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn read_rgba32f(&self, tex: &wgpu::Texture, w: u32, h: u32) -> Image {
        let bpr = (w * 16).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("mapped");
        let mut img = Image::new(w, h, Vec4::ZERO);
        for y in 0..h {
            let row: &[f32] = bytemuck::cast_slice(&data[(y * bpr) as usize..(y * bpr + w * 16) as usize]);
            for x in 0..w {
                let i = (x * 4) as usize;
                img.set(x, y, Vec4::new(row[i], row[i + 1], row[i + 2], row[i + 3]));
            }
        }
        drop(data);
        buf.unmap();
        img
    }

    /// Registers an engine material; returns the handle farthing will carry around.
    pub fn add_material(&mut self, def: MaterialDef, info: MaterialInfo) -> MaterialId {
        let (u, base, normal) = match &def {
            MaterialDef::Checker { a, b, scale } => (MatU { kind: 0, flags: 0, scale: *scale, cutoff: 0.0, color_a: a.to_array(), color_b: b.to_array(), factor: [1.0; 4] }, None, None),
            MaterialDef::Marble { a, b, scale } => (MatU { kind: 1, flags: 0, scale: *scale, cutoff: 0.0, color_a: a.to_array(), color_b: b.to_array(), factor: [1.0; 4] }, None, None),
            MaterialDef::Textured { base, normal, factor, cutoff } => (
                MatU { kind: 2, flags: normal.is_some() as u32, scale: 1.0, cutoff: *cutoff, color_a: [0.0; 4], color_b: [0.0; 4], factor: factor.to_array() },
                Some(self.upload_rgba8(base, true)),
                normal.as_ref().map(|n| {
                    let enc = Image { width: n.width, height: n.height, pixels: n.pixels.iter().map(|p| (p.truncate() * 0.5 + 0.5).extend(1.0)).collect() };
                    self.upload_rgba8(&enc, false)
                }),
            ),
        };
        let ubuf = self.uniform(&u);
        let base = base.as_ref().unwrap_or(&self.white);
        let normal = normal.as_ref().unwrap_or(&self.white);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mat_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.materials.push(GpuMaterial { def, info, bind, impostor: None });
        MaterialId(self.materials.len() as u32 - 1)
    }

    fn vertex_buffer(&self, mesh: &Mesh) -> (wgpu::Buffer, wgpu::Buffer, Vec<(u32, u32, u32)>) {
        let mut m = mesh.clone();
        if m.normals.is_none() {
            m.compute_normals(60.0);
        }
        if m.tangents.is_none() && !m.uvs.is_empty() {
            let _ = farthing::bake::compute_tangents(&mut m, 0);
        }
        let normals = m.normals.as_ref().unwrap();
        let verts: Vec<Vertex> = (0..m.vertex_count())
            .map(|v| Vertex {
                pos: m.positions[v].to_array(),
                nrm: normals[v].to_array(),
                uv: m.uvs.first().map_or([0.0; 2], |u| u[v].to_array()),
                tan: m.tangents.as_ref().map_or([1.0, 0.0, 0.0, 1.0], |t| t[v].to_array()),
            })
            .collect();
        // Group triangles by material slot: (slot, first index, count).
        let mut order: Vec<usize> = (0..m.triangle_count()).collect();
        order.sort_by_key(|&t| m.material_ids[t]);
        let mut idx = Vec::with_capacity(m.indices.len());
        let mut groups = Vec::new();
        for &t in &order {
            let slot = m.material_ids[t];
            match groups.last_mut() {
                Some((s, _, n)) if *s == slot => *n += 3,
                _ => groups.push((slot, idx.len() as u32, 3)),
            }
            idx.extend_from_slice(&m.triangle(t));
        }
        let vb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (verts.len().max(1) * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&vb, 0, bytemuck::cast_slice(&verts));
        let ib = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (idx.len().max(1) * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ib, 0, bytemuck::cast_slice(&idx));
        (vb, ib, groups)
    }

    fn draw_uniform(&self, v: &View, model: Mat4, lit: bool, peel: Option<(&wgpu::TextureView, f32)>) -> wgpu::BindGroup {
        let u = DrawU {
            view_proj: v.view_proj().to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            normal_mat: Mat4::from_mat3(farthing::glam::Mat3::from_mat4(model).inverse().transpose()).to_cols_array_2d(),
            camera_pos: v.eye().extend(1.0).to_array(),
            view: v.view.to_cols_array_2d(),
            lit: [lit as u32 as f32, peel.is_some() as u32 as f32, peel.map_or(0.0, |p| p.1), 0.0],
        };
        let buf = self.uniform(&u);
        let peel_view = peel.map_or(&self.no_peel, |p| p.0);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.draw_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(peel_view) },
            ],
        })
    }

    /// Renders a scene from one view. Returns (color, normal, depth) images.
    pub fn render(&self, scene: &Scene, v: &View, lit: bool) -> (Image, Image, Image) {
        self.render_peeled(scene, v, lit, None)
    }

    /// [`Engine::render`] keeping only front faces behind `peel`'s depth by more than the bias
    /// (depth peeling for layered impostors).
    pub fn render_peeled(&self, scene: &Scene, v: &View, lit: bool, peel: Option<(&Image, f32)>) -> (Image, Image, Image) {
        let (w, h) = (v.width, v.height);
        let peel_tex = peel.map(|(img, _)| {
            let px: Vec<[f32; 4]> = img.pixels.iter().map(|p| p.to_array()).collect();
            self.upload_f32(img.width, img.height, &px)
        });
        let peel = peel_tex.as_ref().zip(peel.map(|p| p.1));
        let color = self.target(w, h, RT);
        let normal = self.target(w, h, RT);
        let depth_rt = self.target(w, h, RT);
        let depth = self.target(w, h, DEPTH);
        let buffers: Vec<_> = scene.meshes.iter().map(|m| self.vertex_buffer(m)).collect();
        let draws: Vec<wgpu::BindGroup> = scene.instances.iter().map(|i| self.draw_uniform(v, i.transform, lit, peel)).collect();
        let cv = color.create_view(&Default::default());
        let nv = normal.create_view(&Default::default());
        let dv = depth_rt.create_view(&Default::default());
        let zv = depth.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let att = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[att(&cv), att(&nv), att(&dv)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &zv, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.scene_pipeline);
            for (ii, inst) in scene.instances.iter().enumerate() {
                let (vb, ib, groups) = &buffers[inst.mesh];
                pass.set_bind_group(0, &draws[ii], &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                for &(slot, first, count) in groups {
                    let id = scene.material_id(slot).0 as usize;
                    let Some(mat) = self.materials.get(id) else { continue };
                    pass.set_bind_group(1, &mat.bind, &[]);
                    pass.draw_indexed(first..first + count, 0, 0..1);
                }
            }
        }
        self.queue.submit([enc.finish()]);
        (self.read_rgba32f(&color, w, h), self.read_rgba32f(&normal, w, h), self.read_rgba32f(&depth_rt, w, h))
    }

    /// Renders surfels with farthing's reference splat shader (`surfels::SURFEL_SHADER`):
    /// one instance per surfel (position, normal, radius, RGBA8 albedo), radius scale, and the
    /// fading surfel's index and fade. Returns (albedo, normal) G-buffer images.
    pub fn render_splats(&self, v: &View, inst: &[SplatInstance], params: [f32; 4]) -> (Image, Image) {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Inst {
            pos: [f32; 3],
            nrm: [f32; 3],
            radius: f32,
            albedo: [u8; 4],
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct U {
            view_proj: [[f32; 4]; 4],
            params: [f32; 4],
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(farthing::impostor::surfels::SURFEL_SHADER)),
        });
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT)] });
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32, 3 => Unorm8x4];
        let target = Some(wgpu::ColorTargetState { format: RT, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Inst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &attrs })],
            },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[target.clone(), target] }),
            multiview_mask: None,
            cache: None,
        });
        let data: Vec<Inst> = inst.iter().map(|&(pos, nrm, radius, albedo)| Inst { pos, nrm, radius, albedo }).collect();
        let vb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (data.len().max(1) * std::mem::size_of::<Inst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        if !data.is_empty() {
            self.queue.write_buffer(&vb, 0, bytemuck::cast_slice(&data));
        }
        let ubuf = self.uniform(&U { view_proj: v.view_proj().to_cols_array_2d(), params });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }] });
        let (w, h) = (v.width, v.height);
        let color = self.target(w, h, RT);
        let normal = self.target(w, h, RT);
        let depth = self.target(w, h, DEPTH);
        let (cv, nv, zv) = (color.create_view(&Default::default()), normal.create_view(&Default::default()), depth.create_view(&Default::default()));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let att = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splats"),
                color_attachments: &[att(&cv), att(&nv)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &zv, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..6, 0..data.len() as u32);
        }
        self.queue.submit([enc.finish()]);
        (self.read_rgba32f(&color, w, h), self.read_rgba32f(&normal, w, h))
    }

    /// Renders an octahedral impostor (created via `create_material`) with farthing's
    /// reference shader — proving the encode functions match between bake and runtime.
    pub fn render_impostor(&self, material: MaterialId, center: Vec3, radius: f32, v: &View) -> Result<Image> {
        self.render_impostor_pass(material, center, radius, v, 0)
    }

    /// `mode`: 0 = lit preview, 1 = albedo, 2 = world normal.
    pub fn render_impostor_pass(&self, material: MaterialId, center: Vec3, radius: f32, v: &View, mode: u32) -> Result<Image> {
        let mat = self.materials.get(material.0 as usize).ok_or_else(|| anyhow!("unknown material"))?;
        let imp = mat.impostor.as_ref().ok_or_else(|| anyhow!("material is not an impostor"))?;
        let u = ImpU {
            view_proj: v.view_proj().to_cols_array_2d(),
            camera_pos: v.eye().extend(1.0).to_array(),
            center_radius: center.extend(radius).to_array(),
            frames: [imp.frames_per_side as f32, imp.hemisphere as u32 as f32, mode as f32, if imp.color2.is_some() { 2.0 } else { 1.0 }],
        };
        let ubuf = self.uniform(&u);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.imp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&imp.color) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&imp.normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(imp.color2.as_ref().unwrap_or(&imp.color)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(imp.normal2.as_ref().unwrap_or(&imp.normal)) },
            ],
        });
        let corners: [[f32; 2]; 6] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let vb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 48, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&vb, 0, bytemuck::cast_slice(&corners));
        let color = self.target(v.width, v.height, RT);
        let depth = self.target(v.width, v.height, DEPTH);
        let cv = color.create_view(&Default::default());
        let zv = depth.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("impostor"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &cv, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &zv, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.imp_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([enc.finish()]);
        Ok(self.read_rgba32f(&color, v.width, v.height))
    }
}

impl RenderHost for Engine {
    fn material_info(&self, material: MaterialId) -> MaterialInfo {
        self.materials.get(material.0 as usize).map_or(MaterialInfo::default(), |m| m.info)
    }

    fn bake_surface(&mut self, req: &SurfaceBakeRequest) -> farthing::Result<ChannelImages> {
        let (w, h) = (req.width, req.height);
        let n = (w * h) as usize;
        let mut pos = vec![[0.0f32; 4]; n];
        let mut nrm = vec![[0.0f32; 4]; n];
        let mut uv = vec![[0.0f32; 4]; n];
        let mut tan = vec![[0.0f32; 4]; n];
        let mut used: Vec<u32> = Vec::new();
        for (i, s) in req.samples.iter().enumerate() {
            if let Some(s) = s {
                pos[i] = s.position.extend(s.material.0 as f32 + 1.0).to_array();
                nrm[i] = s.normal.extend(0.0).to_array();
                uv[i] = [s.uv[0].x, s.uv[0].y, s.uv[1].x, s.uv[1].y];
                tan[i] = s.tangent.to_array();
                if !used.contains(&s.material.0) {
                    used.push(s.material.0);
                }
            }
        }
        let views = [self.upload_f32(w, h, &pos), self.upload_f32(w, h, &nrm), self.upload_f32(w, h, &uv), self.upload_f32(w, h, &tan)];
        let albedo = self.target(w, h, RT);
        let normal = self.target(w, h, RT);
        let av = albedo.create_view(&Default::default());
        let nv = normal.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (k, &m) in used.iter().enumerate() {
            let Some(mat) = self.materials.get(m as usize) else { return Err(farthing::Error::Host(format!("unknown material {m}"))) };
            let info = self.uniform(&[m as f32 + 1.0, 0.0f32, 0.0, 0.0]);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bake_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&views[0]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&views[1]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&views[2]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&views[3]) },
                    wgpu::BindGroupEntry { binding: 4, resource: info.as_entire_binding() },
                ],
            });
            let load = if k == 0 { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load };
            let att = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store } });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bake"),
                color_attachments: &[att(&av), att(&nv)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.bake_pipeline);
            pass.set_bind_group(1, &mat.bind, &[]);
            pass.set_bind_group(2, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        let mut out = ChannelImages::new();
        for c in &req.channels {
            match c.as_str() {
                channels::BASE_COLOR => {
                    out.insert(c.clone(), ChannelImage { image: self.read_rgba32f(&albedo, w, h), encoding: ChannelEncoding::Color });
                }
                channels::NORMAL => {
                    out.insert(c.clone(), ChannelImage { image: self.read_rgba32f(&normal, w, h), encoding: ChannelEncoding::NormalWorld });
                }
                other => return Err(farthing::Error::Host(format!("PoC engine has no `{other}` channel"))),
            }
        }
        Ok(out)
    }

    fn render_views(&mut self, req: &ViewRenderRequest) -> farthing::Result<Vec<ViewImages>> {
        Ok(req
            .views
            .iter()
            .map(|v| {
                let (color, normal, depth) = self.render(req.scene, v, req.lit);
                let mut channels = ChannelImages::new();
                for c in &req.channels {
                    match c.as_str() {
                        channels::NORMAL => channels.insert(c.clone(), ChannelImage { image: normal.clone(), encoding: ChannelEncoding::NormalWorld }),
                        _ => channels.insert(c.clone(), ChannelImage { image: color.clone(), encoding: ChannelEncoding::Color }),
                    };
                }
                ViewImages { channels, depth }
            })
            .collect())
    }

    fn render_views_behind(&mut self, req: &ViewRenderRequest, behind: &[Image], bias: f32) -> farthing::Result<Vec<ViewImages>> {
        Ok(req
            .views
            .iter()
            .zip(behind)
            .map(|(v, b)| {
                let (color, normal, depth) = self.render_peeled(req.scene, v, req.lit, Some((b, bias)));
                let mut channels = ChannelImages::new();
                for c in &req.channels {
                    match c.as_str() {
                        channels::NORMAL => channels.insert(c.clone(), ChannelImage { image: normal.clone(), encoding: ChannelEncoding::NormalWorld }),
                        _ => channels.insert(c.clone(), ChannelImage { image: color.clone(), encoding: ChannelEncoding::Color }),
                    };
                }
                ViewImages { channels, depth }
            })
            .collect())
    }

    fn create_material(&mut self, desc: &BakedMaterialDesc) -> farthing::Result<MaterialId> {
        let base = desc.textures.get(channels::BASE_COLOR).map(|c| c.image.clone()).ok_or_else(|| farthing::Error::Host("baked material needs base_color".into()))?;
        let cutoff = match desc.class {
            MaterialClass::Masked { cutoff } => cutoff,
            _ => 0.0,
        };
        let normal = desc.textures.get(channels::NORMAL).filter(|c| c.encoding == ChannelEncoding::NormalTangent).map(|c| c.image.clone());
        let info = MaterialInfo { class: desc.class, two_sided: cutoff > 0.0 };
        let id = self.add_material(MaterialDef::Textured { base: base.clone(), normal, factor: Vec4::ONE, cutoff }, info);
        if let BakedMaterialKind::OctahedralImpostor { frames_per_side, hemisphere } = desc.kind {
            // Pack height (in radius units) into the normal atlas alpha for parallax.
            let packed = |layer: u32| -> Option<Image> {
                let nimg = desc.textures.get(&farthing::impostor::layer_channel(channels::NORMAL, layer))?.image.clone();
                let himg = desc.textures.get(&farthing::impostor::layer_channel(farthing::impostor::HEIGHT, layer)).map(|c| c.image.clone());
                Some(Image {
                    width: nimg.width,
                    height: nimg.height,
                    pixels: nimg
                        .pixels
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            let h = himg.as_ref().map_or(0.0, |hi| hi.pixels[i].x);
                            (p.truncate() * 0.5 + 0.5).extend(h * 0.5 + 0.5)
                        })
                        .collect(),
                })
            };
            let enc = packed(0).unwrap_or_else(|| Image::new(1, 1, Vec4::new(0.5, 0.5, 1.0, 0.5)));
            let layer2 = desc.textures.get(&farthing::impostor::layer_channel(channels::BASE_COLOR, 1)).zip(packed(1));
            let (color2, normal2) = match layer2 {
                Some((c, n)) => (Some(self.upload_rgba8(&c.image, true)), Some(self.upload_rgba8(&n, false))),
                None => (None, None),
            };
            let imp = ImpostorMat { color: self.upload_rgba8(&base, true), normal: self.upload_rgba8(&enc, false), color2, normal2, frames_per_side, hemisphere };
            self.materials[id.0 as usize].impostor = Some(imp);
        }
        Ok(id)
    }
}
