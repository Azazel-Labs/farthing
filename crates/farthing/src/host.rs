//! The engine boundary.
//!
//! farthing does geometry, UVs, correspondence and packing. It never decides what anything
//! looks like. Every step that needs real materials calls into a [`RenderHost`] the engine
//! implements:
//!
//! | Need | Call |
//! |---|---|
//! | Is a material opaque, alpha-tested or blended? | [`RenderHost::material_info`] |
//! | What does the source surface look like at these points? | [`RenderHost::bake_surface`] |
//! | Render these views (impostors, image metrics) | [`RenderHost::render_views`] |
//! | Which triangle covers each pixel (visibility) | [`RenderHost::render_ids`] |
//! | Make an engine material from baked textures | [`RenderHost::create_material`] |

use crate::image::Image;
use crate::scene::{MaterialId, Scene, TriRef};
use crate::Result;
use glam::camera::rh::proj::directx as proj;
use glam::camera::rh::view::look_at_mat4;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::BTreeMap;

/// Well-known channel names. Hosts may define more; farthing treats all channels as opaque
/// data except for the encoding the host declares.
pub mod channels {
    pub const BASE_COLOR: &str = "base_color";
    pub const NORMAL: &str = "normal";
    pub const ORM: &str = "orm";
    pub const EMISSIVE: &str = "emissive";
    pub const OPACITY: &str = "opacity";
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MaterialClass {
    #[default]
    Opaque,
    /// Alpha-tested: fragments with opacity below `cutoff` are discarded.
    Masked { cutoff: f32 },
    /// Alpha-blended: never occludes.
    Blended,
}

/// The few facts geometry code needs about a material.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MaterialInfo {
    pub class: MaterialClass,
    pub two_sided: bool,
}

/// How farthing should treat a channel image the host returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelEncoding {
    /// Linear-light color; written to disk as sRGB.
    Color,
    /// Linear data (roughness, metalness, masks); written as-is.
    Data,
    /// World-space unit normals in xyz. farthing converts these to the destination's
    /// tangent space when baking onto a mesh.
    NormalWorld,
    /// Tangent-space normals in xyz (already converted).
    NormalTangent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChannelImage {
    pub image: Image,
    pub encoding: ChannelEncoding,
}

pub type ChannelImages = BTreeMap<String, ChannelImage>;

/// One point on a source surface, with every input a material graph may read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    pub instance: u32,
    pub material: MaterialId,
    /// World-space position.
    pub position: Vec3,
    /// World-space interpolated geometric normal of the source.
    pub normal: Vec3,
    /// World-space source tangent (w = handedness), if the source has tangents.
    pub tangent: Vec4,
    /// Source UV sets 0 and 1.
    pub uv: [Vec2; 2],
    pub color: Vec4,
}

/// "Evaluate your real materials at these points": one optional sample per output texel.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceBakeRequest {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width * height` entries. `None` = no surface (leave transparent).
    pub samples: Vec<Option<SurfaceSample>>,
    /// Channels wanted, e.g. [`channels::BASE_COLOR`], [`channels::NORMAL`].
    pub channels: Vec<String>,
}

/// A camera. Projection matrices follow the wgpu / D3D convention (depth in 0..1, right-handed view).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub view: Mat4,
    pub proj: Mat4,
    pub width: u32,
    pub height: u32,
}

impl View {
    pub fn perspective(
        eye: Vec3,
        target: Vec3,
        fov_y: f32,
        width: u32,
        height: u32,
        near: f32,
        far: f32,
    ) -> View {
        let up = if (target - eye)
            .normalize_or_zero()
            .cross(Vec3::Y)
            .length_squared()
            < 1e-6
        {
            Vec3::Z
        } else {
            Vec3::Y
        };
        View {
            view: look_at_mat4(eye, target, up),
            proj: proj::perspective(fov_y, width as f32 / height as f32, near, far),
            width,
            height,
        }
    }

    /// Orthographic camera looking along `dir` at `center`, covering `half_width × half_height`.
    pub fn orthographic(
        center: Vec3,
        dir: Vec3,
        up: Vec3,
        half_width: f32,
        half_height: f32,
        depth: f32,
        width: u32,
        height: u32,
    ) -> View {
        let dir = dir.normalize();
        let up = if dir.cross(up).length_squared() < 1e-8 {
            if dir.y.abs() < 0.9 {
                Vec3::Y
            } else {
                Vec3::Z
            }
        } else {
            up
        };
        let eye = center - dir * depth;
        View {
            view: look_at_mat4(eye, center, up),
            proj: proj::orthographic(
                -half_width,
                half_width,
                -half_height,
                half_height,
                0.0,
                2.0 * depth,
            ),
            width,
            height,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.view.inverse().w_axis.truncate()
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj * self.view
    }

    /// World-space ray through the center of pixel (x, y); y = 0 is the top row.
    pub fn ray(&self, x: u32, y: u32) -> (Vec3, Vec3) {
        let inv = self.view_proj().inverse();
        let nx = (x as f32 + 0.5) / self.width as f32 * 2.0 - 1.0;
        let ny = 1.0 - (y as f32 + 0.5) / self.height as f32 * 2.0;
        let near = inv.project_point3(Vec3::new(nx, ny, 0.0));
        let far = inv.project_point3(Vec3::new(nx, ny, 1.0));
        (near, (far - near).normalize_or_zero())
    }
}

/// "Render these views of this scene with your real materials."
#[derive(Clone, Debug)]
pub struct ViewRenderRequest<'a> {
    pub scene: &'a Scene,
    pub views: Vec<View>,
    pub channels: Vec<String>,
    /// Unlit channels (impostor captures) vs. lit beauty renders (image comparisons).
    pub lit: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ViewImages {
    pub channels: ChannelImages,
    /// Linear view-space depth in `r`, coverage in `a` (0 = background).
    pub depth: Image,
}

#[derive(Clone, Debug)]
pub struct IdRenderRequest<'a> {
    pub scene: &'a Scene,
    pub views: Vec<View>,
}

/// Which triangle covers each pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct IdImage {
    pub width: u32,
    pub height: u32,
    pub ids: Vec<Option<TriRef>>,
}

/// What a baked output material should be.
#[derive(Clone, Debug, PartialEq)]
pub enum BakedMaterialKind {
    /// Ordinary surface material sampling UV set 0.
    Surface,
    /// Octahedral / hemi-octahedral impostor atlas.
    OctahedralImpostor {
        frames_per_side: u32,
        hemisphere: bool,
    },
    /// Flipbook impostor around the up axis.
    Flipbook { frames: u32 },
    /// Alpha-tested cards (billboard cloud, single view).
    Cards,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BakedMaterialDesc {
    pub name: String,
    pub kind: BakedMaterialKind,
    pub class: MaterialClass,
    pub textures: ChannelImages,
    pub parameters: BTreeMap<String, f32>,
}

/// The engine side of the boundary. See the module docs.
pub trait RenderHost {
    /// Facts geometry code needs about a material.
    fn material_info(&self, material: MaterialId) -> MaterialInfo;

    /// Evaluate real materials at source surface points; one image per requested channel,
    /// `req.width × req.height`.
    fn bake_surface(&mut self, req: &SurfaceBakeRequest) -> Result<ChannelImages>;

    /// Render views with real materials.
    fn render_views(&mut self, req: &ViewRenderRequest) -> Result<Vec<ViewImages>>;

    /// Depth peeling, for layered impostors: like [`render_views`](Self::render_views), but
    /// per pixel keep only front-facing surfaces whose linear depth exceeds `behind[i]`'s
    /// (`r`, coverage in `a`, as returned in [`ViewImages::depth`] at the same resolution) by
    /// more than `bias`; pixels `behind[i]` doesn't cover stay empty. The default reports it
    /// unsupported, and layered impostors then fail with [`crate::Error::Unsupported`].
    fn render_views_behind(
        &mut self,
        req: &ViewRenderRequest,
        behind: &[Image],
        bias: f32,
    ) -> Result<Vec<ViewImages>> {
        let _ = (req, behind, bias);
        Err(crate::Error::Unsupported("render_views_behind"))
    }

    /// Triangle-ID renders. The default is farthing's CPU ray caster, which honours
    /// [`MaterialClass::Blended`] (never occludes) but treats masked materials as opaque.
    /// Hosts that can evaluate alpha should override this.
    fn render_ids(&mut self, req: &IdRenderRequest) -> Result<Vec<IdImage>> {
        let infos: Vec<MaterialInfo> = req
            .scene
            .materials
            .iter()
            .map(|m| self.material_info(m.id))
            .collect();
        Ok(crate::raster::cpu_render_ids(req.scene, &req.views, &infos))
    }

    /// Build the engine material for baked, merged or impostor outputs.
    fn create_material(&mut self, desc: &BakedMaterialDesc) -> Result<MaterialId>;
}

/// A host with no materials: every material is opaque, rendering is unsupported.
/// Enough for reduction, remeshing, occluders, bone reduction and CPU visibility.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeometryOnlyHost;

impl RenderHost for GeometryOnlyHost {
    fn material_info(&self, _material: MaterialId) -> MaterialInfo {
        MaterialInfo::default()
    }
    fn bake_surface(&mut self, _req: &SurfaceBakeRequest) -> Result<ChannelImages> {
        Err(crate::Error::Unsupported("bake_surface"))
    }
    fn render_views(&mut self, _req: &ViewRenderRequest) -> Result<Vec<ViewImages>> {
        Err(crate::Error::Unsupported("render_views"))
    }
    fn create_material(&mut self, _desc: &BakedMaterialDesc) -> Result<MaterialId> {
        Err(crate::Error::Unsupported("create_material"))
    }
}
