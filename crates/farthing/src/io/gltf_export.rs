use crate::anim::{Curve, Interpolation};
use crate::image::Texture;
use crate::reduce::optimize::optimize_mesh;
use crate::{Error, Mesh, Result, Scene};
use glam::{Mat4, Vec3};
use polysmelt::meshopt::{
    ExpMode, encode_filter_color, encode_filter_exp, encode_filter_quat, encode_index_buffer,
    encode_oct8, encode_vertex_buffer_level,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;

/// Optional textures to embed per material slot, so outputs open in any glTF viewer.
#[derive(Clone, Debug, Default)]
pub struct ExportTextures {
    /// slot index → base color, sampled with UV set 0. Written as sRGB (converted if the
    /// texture holds linear bytes); build one from a bake with [`Texture::from_image`].
    pub base_color: BTreeMap<u32, Texture>,
    /// slot index → tangent-space normal map stored as glTF does (xyz · 0.5 + 0.5; see
    /// [`Texture::from_signed_normals`]).
    pub normal: BTreeMap<u32, Texture>,
    /// slot index → alpha cutoff, for alpha-tested outputs (impostors, cards).
    pub alpha_cutoff: BTreeMap<u32, f32>,
    /// slot index → base color factor (linear RGBA; multiplies the base color texture).
    /// Default white.
    pub base_color_factor: BTreeMap<u32, glam::Vec4>,
    /// slot index → glTF metallic-roughness texture (linear data: G = roughness,
    /// B = metalness), sampled with UV set 0.
    pub metallic_roughness: BTreeMap<u32, Texture>,
    /// slot index → (metallic, roughness) factors. Default (0, 0.8) without a texture,
    /// (1, 1) with one.
    pub metallic_roughness_factor: BTreeMap<u32, glam::Vec2>,
}

/// Which meshoptimizer compression extension [`GlbOptions::compress`] writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshoptCodec {
    /// `EXT_meshopt_compression`, attribute codec version 0: what most viewers read today.
    #[default]
    Ext,
    /// `KHR_meshopt_compression`, attribute codec version 1: smaller vertex data, and the
    /// `COLOR` filter for quantized vertex colors. `level` is how hard the encoder searches,
    /// as meshoptimizer's `meshopt_encodeVertexBufferLevel`: 0 is fastest, 2 is its default
    /// and 3 also tries the 32-bit XOR deltas that suit floats (higher values encode as 3).
    Khr {
        /// Encoder effort, 0 to 3.
        level: u8,
    },
}

impl MeshoptCodec {
    /// `KHR_meshopt_compression` at level 3, the smallest output.
    pub const KHR: MeshoptCodec = MeshoptCodec::Khr { level: 3 };

    /// The glTF extension name.
    pub fn extension(self) -> &'static str {
        match self {
            MeshoptCodec::Ext => "EXT_meshopt_compression",
            MeshoptCodec::Khr { .. } => "KHR_meshopt_compression",
        }
    }

    fn encode_vertices(self, data: &[u8], stride: usize) -> Vec<u8> {
        match self {
            MeshoptCodec::Ext => encode_vertex_buffer_level(data, stride, 2, 0),
            MeshoptCodec::Khr { level } => {
                encode_vertex_buffer_level(data, stride, level.min(9), 1)
            }
        }
    }
}

/// How [`glb_bytes_with`] encodes geometry. The default writes plain float attributes that
/// any glTF reader opens; [`GlbOptions::compact`] turns everything on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlbOptions {
    /// `KHR_mesh_quantization`: positions as 16-bit integers (the dequantizing scale and offset
    /// folded into each instance's node matrix), normals and tangents as 8-bit, and texture
    /// coordinates and colors within [0, 1] as 16-bit normalized. Skinned and morphed meshes
    /// keep float positions, since their node matrices can't carry the dequantization.
    pub quantize: bool,
    /// meshoptimizer compression (the extension [`GlbOptions::codec`] picks): vertex, index,
    /// animation and inverse bind matrix data through meshoptimizer's codecs. With `quantize`
    /// it also applies its lossy filters: octahedral for normals and tangents, quaternion
    /// (12 bits) for rotation keys, exponential (16 bits) for translation and scale keys, and
    /// with `KHR_meshopt_compression` the color filter (12 bits) for vertex colors. The file
    /// then requires the extension (three.js: `GLTFLoader.setMeshoptDecoder`).
    pub compress: bool,
    /// The compression extension `compress` writes.
    pub codec: MeshoptCodec,
    /// Reorders each mesh's triangles for the vertex cache and for overdraw, and its vertices
    /// by first use ([`optimize_mesh`]); compression works best on that order.
    pub optimize: bool,
}

impl GlbOptions {
    /// Quantized, compressed and optimized: the smallest files, for viewers that support
    /// both extensions (three.js, Babylon.js and others that read gltfpack's output).
    pub fn compact() -> Self {
        GlbOptions {
            quantize: true,
            compress: true,
            codec: MeshoptCodec::Ext,
            optimize: true,
        }
    }

    /// [`GlbOptions::compact`] with `KHR_meshopt_compression` ([`MeshoptCodec::KHR`]):
    /// smaller still, for viewers that read the newer extension.
    pub fn compact_khr() -> Self {
        GlbOptions {
            codec: MeshoptCodec::KHR,
            ..Self::compact()
        }
    }
}

const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const I8: u32 = 5120;
const I16: u32 = 5122;
const U16: u32 = 5123;
const U32: u32 = 5125;
const F32: u32 = 5126;

struct Bin {
    data: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    /// Encode vertex, index and animation views with this meshopt extension.
    compress: Option<MeshoptCodec>,
    /// Apply meshopt's lossy filters (quantization was asked for).
    filters: bool,
    /// Size of the fallback buffer the compressed views decode into.
    fallback: usize,
    /// An accessor uses a type only `KHR_mesh_quantization` allows.
    quantized: bool,
}

fn align4(n: usize) -> usize {
    n.div_ceil(4) * 4
}

impl Bin {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        self.data.resize(align4(self.data.len()), 0);
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        let mut v = json!({ "buffer": 0, "byteOffset": offset, "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }
    /// A view whose `raw_len` bytes decode from `encoded` (in the binary chunk) into the
    /// fallback buffer; `ext` holds the codec's mode and layout.
    fn compressed_view(
        &mut self,
        codec: MeshoptCodec,
        raw_len: usize,
        encoded: &[u8],
        mut ext: Value,
        target: Option<u32>,
    ) -> usize {
        self.data.resize(align4(self.data.len()), 0);
        ext["buffer"] = json!(0);
        ext["byteOffset"] = json!(self.data.len());
        ext["byteLength"] = json!(encoded.len());
        self.data.extend_from_slice(encoded);
        self.fallback = align4(self.fallback);
        let mut v = json!({
            "buffer": 1, "byteOffset": self.fallback, "byteLength": raw_len,
            "extensions": { codec.extension(): ext }
        });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.fallback += raw_len;
        self.views.len() - 1
    }
    /// A view of `stride`-byte elements, through the attribute codec when compressing (with
    /// `filter` applied on decode); `target` is set for vertex attributes, which also get a
    /// byte stride.
    fn elements_view(
        &mut self,
        bytes: &[u8],
        stride: usize,
        filter: Option<&str>,
        target: Option<u32>,
    ) -> usize {
        let view = if let Some(codec) = self.compress {
            let mut ext = json!({ "byteStride": stride, "count": bytes.len() / stride, "mode": "ATTRIBUTES" });
            if let Some(f) = filter {
                ext["filter"] = json!(f);
            }
            let encoded = codec.encode_vertices(bytes, stride);
            self.compressed_view(codec, bytes.len(), &encoded, ext, target)
        } else {
            self.view(bytes, target)
        };
        if target.is_some() {
            self.views[view]["byteStride"] = json!(stride);
        }
        view
    }
    /// A vertex attribute view of `stride`-byte elements.
    fn attribute_view(&mut self, bytes: &[u8], stride: usize, filter: Option<&str>) -> usize {
        self.elements_view(bytes, stride, filter, Some(ARRAY_BUFFER))
    }
    /// Floats in elements of `comps`, outside vertex data (animation keys, matrices).
    fn floats_view(&mut self, data: &[f32], comps: usize) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.elements_view(&bytes, 4 * comps, None, None)
    }
    fn accessor(
        &mut self,
        view: usize,
        component: u32,
        normalized: bool,
        count: usize,
        kind: &str,
    ) -> usize {
        let mut a =
            json!({ "bufferView": view, "componentType": component, "count": count, "type": kind });
        if normalized {
            a["normalized"] = json!(true);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }
    fn accessor_f32(&mut self, data: &[f32], comps: usize, kind: &str, minmax: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = self.attribute_view(&bytes, 4 * comps, None);
        let count = data.len() / comps;
        let a = self.accessor(view, F32, false, count, kind);
        if minmax && count > 0 {
            let mut min = vec![f32::INFINITY; comps];
            let mut max = vec![f32::NEG_INFINITY; comps];
            for c in data.chunks_exact(comps) {
                for i in 0..comps {
                    min[i] = min[i].min(c[i]);
                    max[i] = max[i].max(c[i]);
                }
            }
            self.accessors[a]["min"] = json!(min);
            self.accessors[a]["max"] = json!(max);
        }
        a
    }
    /// 16-bit components, `comps` used of each element's `padded` (padding is zero).
    fn accessor_u16(
        &mut self,
        data: &[u16],
        comps: usize,
        padded: usize,
        kind: &str,
        normalized: bool,
    ) -> usize {
        let bytes: Vec<u8> = data
            .chunks_exact(comps)
            .flat_map(|c| (0..padded).flat_map(|i| c.get(i).copied().unwrap_or(0).to_le_bytes()))
            .collect();
        let view = self.attribute_view(&bytes, 2 * padded, None);
        self.accessor(view, U16, normalized, data.len() / comps, kind)
    }
    /// Unit vectors (with a fourth component for tangents) as normalized bytes, through the
    /// octahedral filter when compressing.
    fn accessor_unit8(&mut self, v: &[[f32; 4]], kind: &str) -> usize {
        let filter = self.compress.is_some().then_some("OCTAHEDRAL");
        let bytes: Vec<u8> = v
            .iter()
            .flat_map(|&[x, y, z, w]| {
                if filter.is_some() {
                    encode_oct8([x, y, z], w)
                } else {
                    let n = Vec3::new(x, y, z).normalize_or_zero();
                    [n.x, n.y, n.z, w].map(|c| (c.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8)
                }
            })
            .collect();
        let view = self.attribute_view(&bytes, 4, filter);
        self.quantized = true;
        self.accessor(view, I8, true, v.len(), kind)
    }
    /// RGBA colors in [0, 1] as normalized 16-bit, through the `COLOR` filter (12 bits) when
    /// compressing with `KHR_meshopt_compression`, else plain.
    fn accessor_color16(&mut self, colors: &[f32], plain: &[u16]) -> usize {
        if !matches!(self.compress, Some(MeshoptCodec::Khr { .. })) {
            return self.accessor_u16(plain, 4, 4, "VEC4", true);
        }
        let rgba: Vec<[f32; 4]> = colors
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let view = self.attribute_view(&encode_filter_color(&rgba, 8, 12), 8, Some("COLOR"));
        self.accessor(view, U16, true, rgba.len(), "VEC4")
    }
    /// Triangle indices, 16-bit when every vertex fits.
    fn accessor_indices(&mut self, data: &[u32], vertex_count: usize) -> usize {
        // WebGL 2 always treats 65535 as a primitive restart in 16-bit indices.
        let (component, stride) = if vertex_count <= 0xffff {
            (U16, 2)
        } else {
            (U32, 4)
        };
        let bytes: Vec<u8> = if stride == 2 {
            data.iter()
                .flat_map(|&i| (i as u16).to_le_bytes())
                .collect()
        } else {
            data.iter().flat_map(|i| i.to_le_bytes()).collect()
        };
        let view = if let Some(codec) = self.compress {
            let ext = json!({ "byteStride": stride, "count": data.len(), "mode": "TRIANGLES" });
            self.compressed_view(
                codec,
                bytes.len(),
                &encode_index_buffer(data),
                ext,
                Some(ELEMENT_ARRAY_BUFFER),
            )
        } else {
            self.view(&bytes, Some(ELEMENT_ARRAY_BUFFER))
        };
        self.accessor(view, component, false, data.len(), "SCALAR")
    }
    /// An animation sampler: its key times (one accessor per distinct run of times, shared)
    /// and its values, flattened to `comps` floats each. With compression and filters,
    /// rotations (`comps` 4) go through the quaternion filter and the rest through the
    /// exponential one; cubic spline tangents stay exact.
    fn sampler<T: Copy>(
        &mut self,
        times_cache: &mut HashMap<Vec<u32>, usize>,
        curve: &Curve<T>,
        comps: usize,
        flatten: impl Fn(T) -> Vec<f32>,
    ) -> Value {
        let key: Vec<u32> = curve.times.iter().map(|t| t.to_bits()).collect();
        let input = match times_cache.get(&key) {
            Some(&a) => a,
            None => {
                let view = self.floats_view(&curve.times, 1);
                let a = self.accessor(view, F32, false, curve.times.len(), "SCALAR");
                let (lo, hi) = (curve.times[0], curve.times[curve.times.len() - 1]);
                self.accessors[a]["min"] = json!([lo]);
                self.accessors[a]["max"] = json!([hi]);
                times_cache.insert(key, a);
                a
            }
        };
        let values: Vec<f32> = curve.values.iter().flat_map(|&v| flatten(v)).collect();
        let kind = if comps == 4 { "VEC4" } else { "VEC3" };
        let filtered = self.compress.is_some()
            && self.filters
            && !matches!(curve.interpolation, Interpolation::CubicSpline);
        let output = if filtered && comps == 4 {
            let q: Vec<[f32; 4]> = values
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect();
            let view = self.elements_view(&encode_filter_quat(&q, 12), 8, Some("QUATERNION"), None);
            // Rotation outputs may be normalized shorts in core glTF.
            self.accessor(view, I16, true, q.len(), kind)
        } else if filtered {
            let bytes = encode_filter_exp(&values, 4 * comps, 16, ExpMode::SharedComponent);
            let view = self.elements_view(&bytes, 4 * comps, Some("EXPONENTIAL"), None);
            self.accessor(view, F32, false, curve.values.len(), kind)
        } else {
            let view = self.floats_view(&values, comps);
            self.accessor(view, F32, false, curve.values.len(), kind)
        };
        let interpolation = match curve.interpolation {
            Interpolation::Step => "STEP",
            Interpolation::Linear => "LINEAR",
            Interpolation::CubicSpline => "CUBICSPLINE",
        };
        json!({ "input": input, "output": output, "interpolation": interpolation })
    }
    fn accessor_mat4(&mut self, mats: &[Mat4]) -> usize {
        let data: Vec<f32> = mats.iter().flat_map(|m| m.to_cols_array()).collect();
        let view = self.floats_view(&data, 16);
        self.accessor(view, F32, false, mats.len(), "MAT4")
    }
}

/// Positions as 16-bit integers in a cube over the mesh's bounds, and the matrix that maps
/// them back.
fn quantize_positions(p: &[Vec3]) -> (Vec<u16>, Mat4) {
    let (min, max) = p
        .iter()
        .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let extent = (max - min).max_element();
    let scale = if extent > 0.0 { extent / 65535.0 } else { 1.0 };
    let q = p
        .iter()
        .flat_map(|&v| {
            ((v - min) / scale)
                .round()
                .clamp(Vec3::ZERO, Vec3::splat(65535.0))
                .to_array()
        })
        .map(|c| c as u16)
        .collect();
    (
        q,
        Mat4::from_translation(min) * Mat4::from_scale(Vec3::splat(scale)),
    )
}

/// Components within [0, 1] as 16-bit normalized integers.
fn unorm16(v: impl IntoIterator<Item = f32>) -> Option<Vec<u16>> {
    v.into_iter()
        .map(|c| {
            (0.0..=1.0)
                .contains(&c)
                .then(|| (c * 65535.0).round() as u16)
        })
        .collect()
}

fn png_bytes(tex: &Texture, srgb: bool) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(
            &tex.rgba8_as(srgb),
            tex.width,
            tex.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| Error::Image(e.to_string()))?;
    Ok(out)
}

/// Writes a binary glTF. Each mesh becomes one glTF mesh with one primitive per material
/// slot used (vertex accessors shared). Instances become nodes; skeletons become joint nodes
/// (with their rest transforms), and their clips animations.
/// Attributes are floats, and indices 16-bit when a mesh has at most 65,535 vertices.
pub fn save_glb(scene: &Scene, path: impl AsRef<Path>, textures: &ExportTextures) -> Result<()> {
    save_glb_with(scene, path, textures, &GlbOptions::default())
}

/// [`save_glb`] with quantization, compression or reordering ([`GlbOptions`]).
pub fn save_glb_with(
    scene: &Scene,
    path: impl AsRef<Path>,
    textures: &ExportTextures,
    options: &GlbOptions,
) -> Result<()> {
    std::fs::write(path.as_ref(), glb_bytes_with(scene, textures, options)?)?;
    Ok(())
}

/// [`save_glb`] into memory: the bytes of a binary glTF.
pub fn glb_bytes(scene: &Scene, textures: &ExportTextures) -> Result<Vec<u8>> {
    glb_bytes_with(scene, textures, &GlbOptions::default())
}

/// [`save_glb_with`] into memory.
pub fn glb_bytes_with(
    scene: &Scene,
    textures: &ExportTextures,
    options: &GlbOptions,
) -> Result<Vec<u8>> {
    let mut bin = Bin {
        data: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
        compress: options.compress.then_some(options.codec),
        filters: options.quantize,
        fallback: 0,
        quantized: false,
    };
    let mut images = Vec::new();
    let mut gl_textures = Vec::new();

    let materials: Vec<Value> = scene
        .materials
        .iter()
        .enumerate()
        .map(|(slot, m)| -> Result<Value> {
            let factor = textures
                .base_color_factor
                .get(&(slot as u32))
                .copied()
                .unwrap_or(glam::Vec4::ONE);
            let mr_tex = textures.metallic_roughness.get(&(slot as u32));
            let mr = textures
                .metallic_roughness_factor
                .get(&(slot as u32))
                .copied()
                .unwrap_or(if mr_tex.is_some() { glam::Vec2::ONE } else { glam::Vec2::new(0.0, 0.8) });
            let mut mat = json!({
                "name": m.name,
                "pbrMetallicRoughness": { "baseColorFactor": factor.to_array(), "metallicFactor": mr.x, "roughnessFactor": mr.y }
            });
            if let Some(img) = mr_tex {
                let view = bin.view(&png_bytes(img, false)?, None);
                images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
                gl_textures.push(json!({ "source": images.len() - 1 }));
                mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = json!({ "index": gl_textures.len() - 1 });
            }
            if let Some(img) = textures.base_color.get(&(slot as u32)) {
                let view = bin.view(&png_bytes(img, true)?, None);
                images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
                gl_textures.push(json!({ "source": images.len() - 1 }));
                mat["pbrMetallicRoughness"]["baseColorTexture"] = json!({ "index": gl_textures.len() - 1 });
            }
            if let Some(img) = textures.normal.get(&(slot as u32)) {
                let view = bin.view(&png_bytes(img, false)?, None);
                images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
                gl_textures.push(json!({ "source": images.len() - 1 }));
                mat["normalTexture"] = json!({ "index": gl_textures.len() - 1 });
            }
            if let Some(c) = textures.alpha_cutoff.get(&(slot as u32)) {
                mat["alphaMode"] = json!("MASK");
                mat["alphaCutoff"] = json!(c);
                mat["doubleSided"] = json!(true);
            }
            Ok(mat)
        })
        .collect::<Result<_>>()?;

    let optimized: Vec<Mesh>;
    let source = if options.optimize {
        optimized = scene
            .meshes
            .iter()
            .map(|m| {
                let mut m = m.clone();
                if m.triangle_count() > 0 {
                    optimize_mesh(&mut m);
                }
                m
            })
            .collect();
        &optimized
    } else {
        &scene.meshes
    };
    let mut meshes = Vec::new();
    // Per mesh, the matrix that maps its quantized positions back.
    let mut dequant: Vec<Option<Mat4>> = Vec::new();
    for (mi, mesh) in source.iter().enumerate() {
        mesh.validate()?;
        let flat3 = |v: &[glam::Vec3]| v.iter().flat_map(|p| p.to_array()).collect::<Vec<f32>>();
        let q = options.quantize;
        // Skinning ignores the node matrix the dequantization rides on, and morph deltas
        // would need it too.
        let skinned = mesh.skin.is_some()
            || scene
                .instances
                .iter()
                .any(|i| i.mesh == mi && i.skeleton.is_some());
        let integer_positions =
            q && !skinned && mesh.morph_targets.is_empty() && mesh.vertex_count() > 0;
        let mut attrs = serde_json::Map::new();
        let position = if integer_positions {
            let (qp, m) = quantize_positions(&mesh.positions);
            dequant.push(Some(m));
            let a = bin.accessor_u16(&qp, 3, 4, "VEC3", false);
            let (mut min, mut max) = ([u16::MAX; 3], [0u16; 3]);
            for c in qp.chunks_exact(3) {
                for i in 0..3 {
                    min[i] = min[i].min(c[i]);
                    max[i] = max[i].max(c[i]);
                }
            }
            bin.accessors[a]["min"] = json!(min);
            bin.accessors[a]["max"] = json!(max);
            bin.quantized = true;
            a
        } else {
            dequant.push(None);
            bin.accessor_f32(&flat3(&mesh.positions), 3, "VEC3", true)
        };
        attrs.insert("POSITION".into(), json!(position));
        if let Some(n) = &mesh.normals {
            let a = if q {
                let v: Vec<[f32; 4]> = n.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
                bin.accessor_unit8(&v, "VEC3")
            } else {
                bin.accessor_f32(&flat3(n), 3, "VEC3", false)
            };
            attrs.insert("NORMAL".into(), json!(a));
        }
        if let Some(t) = &mesh.tangents {
            let a = if q {
                let v: Vec<[f32; 4]> = t.iter().map(|v| v.to_array()).collect();
                bin.accessor_unit8(&v, "VEC4")
            } else {
                let d: Vec<f32> = t.iter().flat_map(|v| v.to_array()).collect();
                bin.accessor_f32(&d, 4, "VEC4", false)
            };
            attrs.insert("TANGENT".into(), json!(a));
        }
        for (i, uv) in mesh.uvs.iter().enumerate() {
            let d: Vec<f32> = uv.iter().flat_map(|v| v.to_array()).collect();
            let a = match unorm16(d.iter().copied()).filter(|_| q) {
                Some(u) => bin.accessor_u16(&u, 2, 2, "VEC2", true),
                None => bin.accessor_f32(&d, 2, "VEC2", false),
            };
            attrs.insert(format!("TEXCOORD_{i}"), json!(a));
        }
        if let Some(c) = &mesh.colors {
            let d: Vec<f32> = c.iter().flat_map(|v| v.to_array()).collect();
            let a = match unorm16(d.iter().copied()).filter(|_| q) {
                Some(u) => bin.accessor_color16(&d, &u),
                None => bin.accessor_f32(&d, 4, "VEC4", false),
            };
            attrs.insert("COLOR_0".into(), json!(a));
        }
        if let Some(s) = &mesh.skin {
            let j: Vec<u16> = s.joints.iter().flatten().copied().collect();
            attrs.insert(
                "JOINTS_0".into(),
                json!(bin.accessor_u16(&j, 4, 4, "VEC4", false)),
            );
            let d: Vec<f32> = s.weights.iter().flatten().copied().collect();
            attrs.insert(
                "WEIGHTS_0".into(),
                json!(bin.accessor_f32(&d, 4, "VEC4", false)),
            );
        }
        let targets: Vec<Value> = mesh
            .morph_targets
            .iter()
            .map(|t| json!({ "POSITION": bin.accessor_f32(&flat3(&t.position_deltas), 3, "VEC3", true) }))
            .collect();
        // One primitive per material slot.
        let mut by_slot: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for t in 0..mesh.triangle_count() {
            by_slot
                .entry(mesh.material_ids[t])
                .or_default()
                .extend_from_slice(&mesh.triangle(t));
        }
        let prims: Vec<Value> = by_slot
            .into_iter()
            .map(|(slot, idx)| {
                let indices = bin.accessor_indices(&idx, mesh.vertex_count());
                let mut p = json!({ "attributes": attrs.clone(), "indices": indices, "mode": 4 });
                if (slot as usize) < materials.len() {
                    p["material"] = json!(slot);
                }
                if !targets.is_empty() {
                    p["targets"] = json!(targets);
                }
                p
            })
            .collect();
        let mut m = json!({ "primitives": prims });
        if !mesh.morph_targets.is_empty() {
            m["extras"] = json!({ "targetNames": mesh.morph_targets.iter().map(|t| t.name.clone()).collect::<Vec<_>>() });
        }
        meshes.push(m);
    }

    // Nodes: skeleton joints first, then instances.
    let mut nodes: Vec<Value> = Vec::new();
    let mut scene_roots: Vec<usize> = Vec::new();
    let mut skins = Vec::new();
    let mut animations: Vec<Value> = Vec::new();
    let mut times_cache = HashMap::new();
    for sk in &scene.skeletons {
        let base = nodes.len();
        for joint in &sk.joints {
            let r = joint.rest;
            let mut node = json!({ "name": joint.name });
            if r.translation != Vec3::ZERO {
                node["translation"] = json!(r.translation.to_array());
            }
            if r.rotation != glam::Quat::IDENTITY {
                node["rotation"] = json!(r.rotation.to_array());
            }
            if r.scale != Vec3::ONE {
                node["scale"] = json!(r.scale.to_array());
            }
            nodes.push(node);
        }
        for clip in &sk.animations {
            let mut samplers = Vec::new();
            let mut channels = Vec::new();
            for track in clip.tracks.iter().filter(|t| t.joint < sk.joints.len()) {
                let node = base + track.joint;
                let mut add = |sampler: Value, path: &str| {
                    channels.push(json!({ "sampler": samplers.len(), "target": { "node": node, "path": path } }));
                    samplers.push(sampler);
                };
                if let Some(c) = track.translation.as_ref().filter(|c| c.is_valid()) {
                    add(
                        bin.sampler(&mut times_cache, c, 3, |v| v.to_array().to_vec()),
                        "translation",
                    );
                }
                if let Some(c) = track.rotation.as_ref().filter(|c| c.is_valid()) {
                    add(
                        bin.sampler(&mut times_cache, c, 4, |v| v.to_array().to_vec()),
                        "rotation",
                    );
                }
                if let Some(c) = track.scale.as_ref().filter(|c| c.is_valid()) {
                    add(
                        bin.sampler(&mut times_cache, c, 3, |v| v.to_array().to_vec()),
                        "scale",
                    );
                }
            }
            if !channels.is_empty() {
                animations
                    .push(json!({ "name": clip.name, "samplers": samplers, "channels": channels }));
            }
        }
        for (j, joint) in sk.joints.iter().enumerate() {
            match joint.parent {
                Some(p) => {
                    let children = nodes[base + p]
                        .get("children")
                        .cloned()
                        .unwrap_or(json!([]));
                    let mut c: Vec<usize> = serde_json::from_value(children).unwrap_or_default();
                    c.push(base + j);
                    nodes[base + p]["children"] = json!(c);
                }
                None => scene_roots.push(base + j),
            }
        }
        let ibm = bin.accessor_mat4(&sk.joints.iter().map(|j| j.inverse_bind).collect::<Vec<_>>());
        skins.push(json!({ "name": sk.name, "joints": (base..base + sk.joints.len()).collect::<Vec<_>>(), "inverseBindMatrices": ibm }));
    }
    for inst in &scene.instances {
        let mut n = json!({ "name": inst.name, "mesh": inst.mesh });
        let transform = match dequant.get(inst.mesh).copied().flatten() {
            Some(d) => inst.transform * d,
            None => inst.transform,
        };
        if transform != Mat4::IDENTITY {
            n["matrix"] = json!(transform.to_cols_array());
        }
        if let Some(s) = inst.skeleton {
            n["skin"] = json!(s);
        }
        nodes.push(n);
        scene_roots.push(nodes.len() - 1);
    }

    bin.data.resize(align4(bin.data.len()), 0);
    let mut buffers = vec![json!({ "byteLength": bin.data.len() })];
    let mut extensions = Vec::new();
    if bin.quantized {
        extensions.push("KHR_mesh_quantization");
    }
    if let Some(codec) = bin.compress.filter(|_| bin.fallback > 0) {
        extensions.push(codec.extension());
        buffers.push(json!({
            "byteLength": align4(bin.fallback),
            "extensions": { codec.extension(): { "fallback": true } }
        }));
    }
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": "farthing" },
        "scene": 0,
        "scenes": [{ "nodes": scene_roots }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": bin.accessors,
        "bufferViews": bin.views,
        "buffers": buffers,
    });
    if !extensions.is_empty() {
        doc["extensionsUsed"] = json!(extensions);
        doc["extensionsRequired"] = json!(extensions);
    }
    if !skins.is_empty() {
        doc["skins"] = json!(skins);
    }
    if !animations.is_empty() {
        doc["animations"] = json!(animations);
    }
    if !images.is_empty() {
        doc["images"] = json!(images);
        doc["textures"] = json!(gl_textures);
        doc["samplers"] = json!([{ "magFilter": 9729, "minFilter": 9987 }]);
        for t in doc["textures"].as_array_mut().unwrap() {
            t["sampler"] = json!(0);
        }
    }

    let mut json_bytes =
        serde_json::to_vec(&doc).map_err(|e| Error::InvalidInput(e.to_string()))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.data.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin.data.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    out.extend_from_slice(&bin.data);
    Ok(out)
}
