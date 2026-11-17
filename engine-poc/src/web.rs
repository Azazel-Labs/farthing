//! The browser side: a [`Viewer`] drawing into a canvas, for the web demo.
//!
//! The page creates one with `createView(canvas)`, uploads textures, materials, geometries,
//! meshes, splat clouds and cluster meshes as typed arrays, and each animation frame calls `beginFrame`, `view` once per camera (compare
//! mode has two), then `endFrame`. An `Orbit` turns the camera with the page's pointer and wheel
//! events.

use crate::editor_visualizations::Overlay;
use crate::orbit::OrbitCamera;
use crate::viewer::{AlphaMode, Camera, Draw, LineStyle, Material, Viewer};
use farthing::glam::{Mat4, Vec2, Vec3, Vec4};
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// A view queued by `view` for `endFrame`.
struct QueuedView {
    rect: [u32; 4],
    camera: Camera,
    draws: Vec<Draw>,
}

#[wasm_bindgen]
pub struct EngineView {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    viewer: Viewer,
    backend: String,
    views: Vec<QueuedView>,
}

/// Opens the GPU and binds it to `canvas`. Fails when the browser has no WebGPU (or, in a
/// `webgl` build, no WebGL2 either).
#[wasm_bindgen(js_name = createView)]
pub async fn create_view(canvas: web_sys::HtmlCanvasElement) -> Result<EngineView, JsValue> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL;
    let instance = wgpu::util::new_instance_with_webgpu_detection(desc).await;
    let (width, height) = (canvas.width().max(1), canvas.height().max(1));
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(js_err)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(js_err)?;
    let backend = format!("{:?}", adapter.get_info().backend);
    // WebGPU gets its full default limits (storage buffers and compute, for the cluster
    // renderer) and storage bindings as large as the adapter allows, up to 1 GiB.
    let available = adapter.limits();
    let required_limits = if adapter.get_info().backend == wgpu::Backend::BrowserWebGpu {
        let mut l = wgpu::Limits::default().using_resolution(available.clone());
        l.max_storage_buffer_binding_size = available.max_storage_buffer_binding_size.min(1 << 30);
        l.max_buffer_size = available.max_buffer_size.min(1 << 30);
        l
    } else {
        wgpu::Limits::downlevel_webgl2_defaults().using_resolution(available)
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("farthing-viewer"),
            required_limits,
            ..Default::default()
        })
        .await
        .map_err(js_err)?;
    let mut config = surface
        .get_default_config(&adapter, width, height)
        .ok_or("the canvas can't be drawn to by this adapter")?;
    // Transparent where nothing's drawn, so the page shows through. Every WebGPU canvas takes
    // premultiplied alpha, though wgpu's capabilities list only opaque there.
    let caps = surface.get_capabilities(&adapter);
    if adapter.get_info().backend == wgpu::Backend::BrowserWebGpu
        || caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
    {
        config.alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
    }
    // Canvases are 8-bit unorm, holding sRGB-encoded premultiplied color; the viewer composites
    // into that from its own sRGB target.
    surface.configure(&device, &config);
    let viewer = Viewer::new(device.clone(), queue.clone(), config.format);
    Ok(EngineView {
        surface,
        device,
        queue,
        config,
        viewer,
        backend,
        views: Vec::new(),
    })
}

fn texture(id: i32) -> Option<u32> {
    u32::try_from(id).ok()
}

#[wasm_bindgen]
impl EngineView {
    /// The graphics API in use: "BrowserWebGpu" or "Gl".
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// Uploads an RGBA8 image (rows top to bottom); the engine builds its mips. `srgb` for color
    /// (base color, emissive), not data. Returns its id.
    #[wasm_bindgen(js_name = addTexture)]
    pub fn add_texture(&mut self, width: u32, height: u32, rgba: &[u8], srgb: bool) -> u32 {
        self.viewer.add_texture(width, height, rgba, srgb)
    }

    #[wasm_bindgen(js_name = removeTexture)]
    pub fn remove_texture(&mut self, id: u32) {
        self.viewer.remove_texture(id);
    }

    /// Makes a glTF metallic-roughness material. `factors` (15 or 16 floats): base color RGBA
    /// (linear), metallic, roughness, normal scale xy, occlusion strength, emissive RGB
    /// (linear, scaled), alpha mode (0 opaque, 1 mask, 2 blend), alpha cutoff, double-sided
    /// (0 or 1), and optionally flat shading (0 or 1: face normals). `textures` (5 ids, -1 for none): base color, metal-roughness, normal, emissive,
    /// occlusion. Returns its id.
    #[wasm_bindgen(js_name = addMaterial)]
    pub fn add_material(&mut self, factors: &[f32], textures: &[i32]) -> Result<u32, JsValue> {
        let (Ok(f), Ok(t)) = (
            <&[f32; 15]>::try_from(factors.get(..15).unwrap_or(&[])),
            <&[i32; 5]>::try_from(textures),
        ) else {
            return Err("a material needs 15 or 16 factors and 5 texture ids".into());
        };
        if factors.len() > 16 {
            return Err("a material needs 15 or 16 factors and 5 texture ids".into());
        }
        Ok(self.viewer.add_material(&Material {
            base_color: Vec4::from_slice(&f[0..4]),
            metallic: f[4],
            roughness: f[5],
            normal_scale: [f[6], f[7]],
            occlusion_strength: f[8],
            emissive: Vec3::from_slice(&f[9..12]),
            alpha: match f[12] as u32 {
                1 => AlphaMode::Mask(f[13]),
                2 => AlphaMode::Blend,
                _ => AlphaMode::Opaque,
            },
            double_sided: f[14] != 0.0,
            flat: factors.get(15).is_some_and(|&x| x != 0.0),
            base_color_texture: texture(t[0]),
            metallic_roughness_texture: texture(t[1]),
            normal_texture: texture(t[2]),
            emissive_texture: texture(t[3]),
            occlusion_texture: texture(t[4]),
        }))
    }

    #[wasm_bindgen(js_name = removeMaterial)]
    pub fn remove_material(&mut self, id: u32) {
        self.viewer.remove_material(id);
    }

    /// Uploads vertices for meshes to index: positions and normals (xyz per vertex), uvs (uv per
    /// vertex, or empty) and linear RGB colors (per vertex, or empty for white). Returns its id.
    #[wasm_bindgen(js_name = addGeometry)]
    pub fn add_geometry(
        &mut self,
        positions: &[f32],
        normals: &[f32],
        uvs: &[f32],
        colors: &[f32],
    ) -> u32 {
        self.viewer.add_geometry(positions, normals, uvs, colors)
    }

    /// Frees a geometry; meshes made from it keep drawing.
    #[wasm_bindgen(js_name = removeGeometry)]
    pub fn remove_geometry(&mut self, id: u32) {
        self.viewer.remove_geometry(id);
    }

    /// Makes a mesh of a geometry's triangles: `indices` (three per triangle) drawn with
    /// `material`. Returns its id.
    #[wasm_bindgen(js_name = addMesh)]
    pub fn add_mesh(
        &mut self,
        geometry: u32,
        indices: &[u32],
        material: u32,
    ) -> Result<u32, JsValue> {
        self.viewer
            .add_mesh(geometry, indices, material)
            .ok_or_else(|| "no such geometry".into())
    }

    /// Uploads a splat cloud, 16 floats per splat: center xyz, covariance xx xy xz yy yz zz,
    /// linear RGBA, normal xyz. Gaussians draw in the order given, so send them far to near.
    /// Returns an id a view draws like a mesh's.
    #[wasm_bindgen(js_name = addSplats)]
    pub fn add_splats(&mut self, splats: &[f32]) -> u32 {
        self.viewer.add_splats(splats)
    }

    /// Replaces a splat cloud's splats, re-sorted for a new camera, say.
    #[wasm_bindgen(js_name = setSplats)]
    pub fn set_splats(&mut self, id: u32, splats: &[f32]) {
        self.viewer.set_splats(id, splats);
    }

    /// Whether splats draw as blended Gaussians (the default) or opaque discs.
    #[wasm_bindgen(js_name = setSplatShape)]
    pub fn set_splat_shape(&mut self, gaussians: bool) {
        self.viewer.set_splat_shape(gaussians);
    }

    /// Whether cluster meshes draw on the GPU (WebGPU; not WebGL2).
    #[wasm_bindgen(getter, js_name = gpuClusters)]
    pub fn gpu_clusters(&self) -> bool {
        self.viewer.clusters_supported()
    }

    /// A packed cluster DAG (a `.fcld` file), cut, culled and drawn on the GPU with its pages
    /// streamed in as the views need them. Draws naming the returned id are its instances.
    #[wasm_bindgen(js_name = addClusterMesh)]
    pub fn add_cluster_mesh(&mut self, fcld: &[u8]) -> Result<u32, JsValue> {
        let dag = farthing::cluster::GpuClusterDag::from_bytes(fcld).map_err(js_err)?;
        self.viewer
            .add_cluster_mesh(dag)
            .ok_or_else(|| "cluster meshes need WebGPU".into())
    }

    /// The material (from `addMaterial`) each of a cluster mesh's material ids draws with.
    #[wasm_bindgen(js_name = setClusterMaterials)]
    pub fn set_cluster_materials(&mut self, id: u32, materials: &[u32]) {
        self.viewer.set_cluster_materials(id, materials);
    }

    /// 0: the cluster meshes' materials, 1: a color per cluster, 2: a color per DAG level, on
    /// material `material` (-1: the default).
    #[wasm_bindgen(js_name = setClusterShading)]
    pub fn set_cluster_shading(&mut self, mode: u32, material: i32) {
        self.viewer.set_cluster_shading(mode, u32::try_from(material).ok());
    }

    /// The largest error the cluster cut leaves on screen, in device pixels.
    #[wasm_bindgen(js_name = setClusterThreshold)]
    pub fn set_cluster_threshold(&mut self, pixels: f32) {
        self.viewer.set_cluster_threshold(pixels);
    }

    /// The page pool's size in bytes and how many pages may upload per frame. A new size
    /// starts streaming over.
    #[wasm_bindgen(js_name = setClusterStreaming)]
    pub fn set_cluster_streaming(&mut self, budget: f64, pages_per_frame: u32) {
        self.viewer
            .set_cluster_streaming(budget.max(0.0) as u64, pages_per_frame as usize);
    }

    #[wasm_bindgen(js_name = setClusterCulling)]
    pub fn set_cluster_culling(&mut self, frustum: bool, cone: bool) {
        self.viewer.set_cluster_culling(frustum, cone);
    }

    /// Keeps the cluster cut where the cameras are now.
    #[wasm_bindgen(js_name = setClusterFreeze)]
    pub fn set_cluster_freeze(&mut self, on: bool) {
        self.viewer.set_cluster_freeze(on);
    }

    /// What the cluster cut drew and what's resident, as JSON: `views` (per view `cut`,
    /// `clusters`, `triangles`, `perLevel`), `residentPages`, `totalPages`, `residentBytes`,
    /// `totalBytes`, `poolBytes`, `poolPages`, `requestedPages`, `loaded`, `evicted`. Empty
    /// without cluster meshes.
    #[wasm_bindgen(js_name = clusterStats)]
    pub fn cluster_stats(&self) -> String {
        let Some(s) = self.viewer.cluster_stats() else {
            return String::new();
        };
        let list = |v: &[u32]| {
            v.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        };
        let views: Vec<String> = s
            .views
            .iter()
            .map(|v| {
                format!(
                    r#"{{"cut":{},"clusters":{},"triangles":{},"perLevel":[{}]}}"#,
                    v.cut,
                    v.clusters,
                    v.triangles,
                    list(&v.per_level)
                )
            })
            .collect();
        format!(
            r#"{{"views":[{}],"residentPages":{},"totalPages":{},"residentBytes":{},"totalBytes":{},"poolBytes":{},"poolPages":{},"requestedPages":{},"loaded":{},"evicted":{}}}"#,
            views.join(","),
            s.resident_pages,
            s.total_pages,
            s.resident_bytes,
            s.total_bytes,
            s.pool_bytes,
            s.pool_pages,
            s.requested_pages,
            s.loaded,
            s.evicted
        )
    }

    /// Frees a mesh or splat cloud.
    #[wasm_bindgen(js_name = removeMesh)]
    pub fn remove_mesh(&mut self, id: u32) {
        self.viewer.remove_mesh(id);
    }

    /// An overlay's look: `kind` 0 is the wireframe and 1 the UV seams; `color` is linear RGBA
    /// and `width` in device pixels.
    #[wasm_bindgen(js_name = setLineStyle)]
    pub fn set_line_style(&mut self, kind: u32, color: &[f32], width: f32) -> Result<(), JsValue> {
        let overlay = if kind == 0 {
            Overlay::Wireframe
        } else {
            Overlay::Seams
        };
        let color = <[f32; 4]>::try_from(color).map_err(|_| "color needs 4 floats")?;
        self.viewer.set_line_style(
            overlay,
            LineStyle {
                color: Vec4::from(color),
                width,
            },
        );
        Ok(())
    }

    /// Which overlays are drawn, over the draws that take them.
    #[wasm_bindgen(js_name = setOverlays)]
    pub fn set_overlays(&mut self, wireframe: bool, seams: bool) {
        self.viewer.set_overlays(wireframe, seams);
    }

    /// Starts a frame of `width` by `height` device pixels, resizing the canvas's drawing
    /// buffer to it.
    #[wasm_bindgen(js_name = beginFrame)]
    pub fn begin_frame(&mut self, width: u32, height: u32) {
        self.views.clear();
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) != (self.config.width, self.config.height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Queues a view of the frame: a rect in device pixels from the top left, mesh `meshes[i]`
    /// with model matrix `models[16 * i..]` for each i, overlays over those with `overlays[i]`
    /// set, and the camera. Matrices are column-major; `view` is world to camera and `fov_y` is
    /// in degrees.
    #[allow(clippy::too_many_arguments)]
    pub fn view(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        meshes: &[u32],
        models: &[f32],
        overlays: &[u8],
        view: &[f32],
        fov_y: f32,
        near: f32,
        far: f32,
    ) -> Result<(), JsValue> {
        if view.len() != 16 || models.len() != meshes.len() * 16 || overlays.len() != meshes.len() {
            return Err("view needs 16 floats, models 16 per mesh and overlays 1 per mesh".into());
        }
        let draws = meshes
            .iter()
            .zip(models.as_chunks::<16>().0)
            .zip(overlays)
            .map(|((&mesh, m), &o)| Draw {
                mesh,
                model: Mat4::from_cols_array(m),
                overlays: o != 0,
            })
            .collect();
        let camera = Camera {
            view: Mat4::from_cols_slice(view),
            fov_y: fov_y.to_radians(),
            near,
            far,
        };
        self.views.push(QueuedView {
            rect: [x, y, width, height],
            camera,
            draws,
        });
        Ok(())
    }

    /// Draws the queued views and presents the frame.
    #[wasm_bindgen(js_name = endFrame)]
    pub fn end_frame(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let target = frame.texture.create_view(&Default::default());
        let views: Vec<_> = self
            .views
            .iter()
            .map(|v| crate::viewer::View {
                rect: v.rect,
                camera: v.camera,
                draws: &v.draws,
            })
            .collect();
        self.viewer
            .render(&target, self.config.width, self.config.height, &views);
        self.queue.present(frame);
    }
}

/// The camera: the page forwards its canvas's pointer and wheel events, calls `update` each
/// frame and passes `viewMatrix`, `near` and `far` to `EngineView::view`. Lengths are in scene
/// units, positions in CSS pixels.
#[wasm_bindgen]
#[derive(Default)]
pub struct Orbit(OrbitCamera);

#[wasm_bindgen]
impl Orbit {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Orbit {
        Orbit::default()
    }

    /// The model's bounding sphere: the target stays near it, and near and far fit it.
    #[wasm_bindgen(js_name = setFocus)]
    pub fn set_focus(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        self.0.focus = (Vec3::new(x, y, z), radius.max(1e-6));
    }

    /// Looks at the focus from the default angle.
    pub fn home(&mut self) {
        self.0.home();
    }

    /// The distance at which a sphere of `radius` fills the view.
    #[wasm_bindgen(js_name = homeDistance)]
    pub fn home_distance(&self, radius: f32) -> f32 {
        self.0.home_distance(radius)
    }

    /// Moves the target to the focus center, the eye with it.
    pub fn recenter(&mut self) {
        self.0.recenter();
    }

    /// Puts the eye `d` from the target.
    #[wasm_bindgen(js_name = setDistance)]
    pub fn set_distance(&mut self, d: f32) {
        self.0.set_distance(d);
    }

    /// The eye's distance from the focus center.
    #[wasm_bindgen(getter)]
    pub fn distance(&self) -> f32 {
        self.0.focus_distance()
    }

    #[wasm_bindgen(getter, js_name = maxDistance)]
    pub fn max_distance(&self) -> f32 {
        self.0.max_distance
    }

    #[wasm_bindgen(setter, js_name = maxDistance)]
    pub fn set_max_distance(&mut self, d: f32) {
        self.0.max_distance = d;
    }

    #[wasm_bindgen(setter, js_name = panEnabled)]
    pub fn set_pan_enabled(&mut self, on: bool) {
        self.0.pan_enabled = on;
    }

    /// Vertical field of view in degrees.
    #[wasm_bindgen(getter)]
    pub fn fov(&self) -> f32 {
        self.0.fov_y
    }

    /// `button` as `PointerEvent.button`; `modifier` when shift, ctrl or meta is down (left
    /// drag pans then).
    #[wasm_bindgen(js_name = pointerDown)]
    pub fn pointer_down(&mut self, id: i32, x: f32, y: f32, button: u32, modifier: bool) {
        self.0.pointer_down(id, Vec2::new(x, y), button, modifier);
    }

    #[wasm_bindgen(js_name = pointerMove)]
    pub fn pointer_move(&mut self, id: i32, x: f32, y: f32) {
        self.0.pointer_move(id, Vec2::new(x, y));
    }

    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self, id: i32) {
        self.0.pointer_up(id);
    }

    /// `WheelEvent.deltaY` and `deltaMode`; `pinch` when ctrl is down.
    pub fn wheel(&mut self, delta_y: f32, mode: u32, pinch: bool) {
        self.0.wheel(delta_y, mode, pinch);
    }

    /// Eases the camera on; call once per frame with the view's height in CSS pixels.
    pub fn update(&mut self, height: f32) {
        self.0.update(height);
    }

    #[wasm_bindgen(getter)]
    pub fn eye(&self) -> Vec<f32> {
        self.0.eye.to_array().to_vec()
    }

    /// World to camera, column-major.
    #[wasm_bindgen(getter, js_name = viewMatrix)]
    pub fn view_matrix(&self) -> Vec<f32> {
        self.0.view_matrix().to_cols_array().to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn near(&self) -> f32 {
        self.0.planes().0
    }

    #[wasm_bindgen(getter)]
    pub fn far(&self) -> f32 {
        self.0.planes().1
    }
}
