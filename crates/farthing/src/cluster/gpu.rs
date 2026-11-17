//! A packed, streamable form of a [`ClusterDag`]: flat little-endian buffers a renderer uploads
//! as they are, with the vertex data quantized and cut into fixed-size streaming pages.
//!
//! [`GpuClusterDag::from_dag`] builds it from a DAG and its source mesh, [`GpuClusterDag::to_bytes`]
//! writes the file and [`GpuClusterDag::from_bytes`] reads (and validates) it back.
//! [`GpuClusterDag::select`] is a CPU reference of the per-cluster LOD test a GPU runs; it picks
//! the same clusters as [`ClusterDag::select`] (their errors and spheres are stored as exact
//! `f32`s).
//!
//! # Pages
//!
//! Clusters are packed into pages of at most [`GpuOptions::page_size`] bytes (128 KiB by
//! default) by [`plan_pages`], with this format's sizes ([`FcldLayout`]).
//! The unit of packing is a group's output: the clusters one [`ClusterGroup`](super::ClusterGroup)
//! produced share their LOD error and bounds, so they become visible together, and they
//! always land in one page. Source-level clusters (no producing group) are packed by the group
//! that consumes them. Units go in coarse-to-fine order (level descending), so the root
//! clusters are in the first pages and a renderer that streams pages in file order always has
//! a cluster's coarser replacement resident before the cluster itself.
//!
//! Each page lists its dependencies: the pages holding the clusters its clusters' parent groups
//! produced (the coarser clusters that are drawn instead of them). A dependency always has a
//! lower page index; a page may be made resident once its dependencies are.
//!
//! For a format of your own, plan with [`plan_pages`] and your own
//! [`PageLayout`]; see the [`pages`](super::pages) module.
//!
//! # Quantization
//!
//! * **Positions** snap to one grid over the whole mesh: `origin` is the mesh's bounding-box
//!   minimum and `step = largest extent / (2^position_bits − 1)` ([`GpuOptions::position_bits`],
//!   16 by default, at most 21). Each page stores its minimum grid corner `base`, and each
//!   vertex its offset from `base` in 21 bits per axis. A vertex shared by two pages decodes to
//!   the same bits in both (the grid is global), so cuts that mix pages stay crack-free.
//!   Decoded error per axis ≤ `step / 2` (plus `f32` rounding). Decode as
//!   `origin + vec3(base + delta) * step`, adding the integers *before* converting to float.
//! * **Normals**: octahedral encoding, two snorm16 values. Angular error below 0.01°.
//! * **UVs** (set 0): unorm16 over the mesh's UV bounds: `uv = uv_min + q * uv_step` with
//!   `uv_step = (uv_max − uv_min) / 65535`; error per axis ≤ `uv_step / 2` (1/131070 of the UV
//!   range: 0.008 texels of a 1024-texel texture over a 0–1 range). The bounds are global for
//!   the same reason as the position grid.
//!
//! These are always stored when the mesh has them. The rest are opt-in ([`GpuOptions`]), each
//! stored only when asked for and present in the mesh, and each one `u32` word per vertex
//! unless stated:
//!
//! * **Tangents** ([`FLAG_TANGENTS`]): the direction like a normal (oct snorm16 x2), with the
//!   lowest bit of the y half replaced by the handedness (set: `w = −1`). Decode the direction
//!   as is; the stolen bit moves it by less than 0.002°.
//! * **Colors** ([`FLAG_COLORS`], vertex color set 0): RGBA8 unorm, red in the low byte, values
//!   clamped to 0–1 (as stored in the mesh, which is linear for glTF).
//! * **UVs, set 1** ([`FLAG_UV1`], needs set 0): unorm16 like set 0, over its own bounds
//!   (`uv1_min`, `uv1_step`).
//! * **Skin** ([`FLAG_SKIN`], three words): joints as four `u16` (two words, joint 0 in the low
//!   half of the first), then weights as RGBA8 unorm, normalized so the four bytes add up to
//!   exactly 255.
//!
//! Further UV and color sets and morph targets are not stored.
//!
//! # File layout
//!
//! All values are little-endian; `u32`/`f32` unless stated. Every section and every page
//! starts on a 16-byte boundary, and every record except the vertex is a whole number of
//! `vec4`/`uvec4`s, so each array reads as a std430 struct array. Vertices are a whole number
//! of `u32` words ([`vertex_words`]): 16 bytes with no optional attributes.
//!
//! **File header** (112 bytes, offset 0):
//!
//! | Offset | Field |
//! |---|---|
//! | 0 | magic `b"FCLD"` (4 bytes), version (= [`VERSION`]), flags, header size (112) |
//! | 16 | cluster count, group count, page count, dependency count |
//! | 32 | page size, position bits, vertex stride (bytes), cluster stride (128) |
//! | 48 | origin.xyz, step |
//! | 64 | uv_min.xy, uv_step.xy |
//! | 80 | cluster array offset, group array offset, page table offset, dependency array offset |
//! | 96 | uv1_min.xy, uv1_step.xy (zero without [`FLAG_UV1`]) |
//!
//! Flags: bit 0 normals, bit 1 UVs, bit 2 tangents, bit 3 colors, bit 4 UV set 1, bit 5 skin.
//! [`GpuClusterDag::from_bytes`] also reads version 1 files, which have the same layout with a
//! 96-byte header and only bits 0 and 1.
//!
//! **Cluster header** (128 bytes each, in page order; [`GpuCluster`]):
//!
//! | Offset | Field |
//! |---|---|
//! | 0 | bounds: center.xyz, radius (culling sphere of the cluster's triangles) |
//! | 16 | lod bounds: center.xyz, radius |
//! | 32 | parent bounds: center.xyz, radius |
//! | 48 | cone apex.xyz, cone cutoff (> 1: never back-face culled) |
//! | 64 | cone axis.xyz, lod error |
//! | 80 | parent error (`+inf` for roots), level, group (`0xFFFFFFFF` at level 0), parent group (`0xFFFFFFFF` for roots) |
//! | 96 | page, vertex-index offset, triangle offset, counts (vertex count in bits 0–15, triangle count in bits 16–31) |
//! | 112 | material (the cluster's single material, or `0xFFFFFFFF` when mixed), material offset (0 when not mixed), source cluster (index in the [`ClusterDag`]), reserved |
//!
//! Offsets in a cluster header are byte offsets from the start of its page.
//!
//! **Group** (48 bytes each, in [`ClusterDag::groups`] order; [`GpuGroup`]): bounds
//! center.xyz, radius | error, level, page, first cluster | cluster count, 3 × reserved.
//!
//! **Page table entry** (32 bytes each; [`GpuPage`]): file offset low, file offset high, byte
//! size, first cluster, cluster count, first dependency, dependency count, level (the highest
//! cluster level in the page).
//!
//! **Dependencies**: `u32` page indices, each page's run sorted ascending, padded to 16 bytes.
//!
//! **Page** (at most `page size` bytes, self-contained, so it can sit in any slot of a page pool):
//!
//! | Offset | Content |
//! |---|---|
//! | 0 | page header: first cluster, cluster count, vertex count, vertex offset (48) |
//! | 16 | base.xyz (grid units), vertex-index section offset |
//! | 32 | triangle section offset, material section offset, byte size, page index |
//! | 48 | vertices, `vertex stride` bytes each: word 0 = Δx \| Δy << 21, word 1 = Δy >> 11 \| Δz << 10 (21 bits per axis), word 2 = normal (oct snorm16 x in the low half, y in the high half), word 3 = uv (unorm16 u low, v high), then the optional attributes in flag order: tangent, color, uv1, skin (3 words) |
//! | vertex-index offset | per cluster, `vertex count` page-local vertex indices (`u32`) |
//! | triangle offset | per cluster, three `u8` cluster-local vertex indices per triangle (meshlet micro-indices), each cluster's run padded to 4 bytes |
//! | material offset | per mixed-material cluster, one `u16` material id per triangle, padded to 4 bytes |
//!
//! Sections start on 16-byte boundaries; the byte size is a multiple of 16. A triangle's
//! vertex `k` is `vertices[indices[vertex_offset/4 + tri[k]]]`.

use super::pages::{PageLayout, PageTally, plan_pages};
use super::{Cluster, ClusterDag, Cone, LodView, Sphere};
use crate::{Error, Mesh, Result, Skin};
use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

/// File magic.
pub const MAGIC: [u8; 4] = *b"FCLD";
/// Format version written by [`GpuClusterDag::to_bytes`].
pub const VERSION: u32 = 2;
/// File header size in bytes.
pub const HEADER_SIZE: usize = 112;
/// File header size of version 1 files.
pub const HEADER_SIZE_V1: usize = 96;
/// Cluster header size in bytes.
pub const CLUSTER_STRIDE: usize = 128;
/// Group record size in bytes.
pub const GROUP_STRIDE: usize = 48;
/// Page table entry size in bytes.
pub const PAGE_ENTRY_STRIDE: usize = 32;
/// Page header size in bytes.
pub const PAGE_HEADER_SIZE: usize = 48;
/// Size in bytes of a vertex with no optional attributes (position, normal, uv).
pub const VERTEX_STRIDE: usize = 16;

/// Flag: vertices carry normals.
pub const FLAG_NORMALS: u32 = 1;
/// Flag: vertices carry UVs (set 0).
pub const FLAG_UVS: u32 = 2;
/// Flag: vertices carry tangents (one word).
pub const FLAG_TANGENTS: u32 = 4;
/// Flag: vertices carry colors (one word).
pub const FLAG_COLORS: u32 = 8;
/// Flag: vertices carry UV set 1 (one word).
pub const FLAG_UV1: u32 = 16;
/// Flag: vertices carry joints and weights (three words).
pub const FLAG_SKIN: u32 = 32;
/// Every flag this version knows.
pub const FLAGS_ALL: u32 = 63;

const NONE: u32 = u32::MAX;

/// `u32` words per vertex for `flags`: 4, plus each optional attribute's words.
pub fn vertex_words(flags: u32) -> usize {
    4 + [FLAG_TANGENTS, FLAG_COLORS, FLAG_UV1]
        .iter()
        .filter(|&&f| flags & f != 0)
        .count()
        + if flags & FLAG_SKIN != 0 { 3 } else { 0 }
}

/// The word within a vertex where the optional attribute `flag` starts (`None` when `flags`
/// lacks it). Attributes follow the four base words in flag order.
pub fn attribute_word(flags: u32, flag: u32) -> Option<usize> {
    if flags & flag == 0 || !matches!(flag, FLAG_TANGENTS | FLAG_COLORS | FLAG_UV1 | FLAG_SKIN) {
        return None;
    }
    Some(vertex_words(
        flags & (flag - 1) & !(FLAG_NORMALS | FLAG_UVS),
    ))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuOptions {
    /// Largest page in bytes (a multiple of 16, at least 4 KiB). Every group's output must fit
    /// in one page: eight full 128-triangle clusters take about 17 KiB, so use 32 KiB or more
    /// with the default cluster options.
    pub page_size: usize,
    /// Position grid resolution per axis, 1 to 21 bits over the mesh's largest extent. Every
    /// value costs the same 8 bytes per vertex.
    pub position_bits: u32,
    /// Store tangents when the mesh has them (4 bytes per vertex).
    pub tangents: bool,
    /// Store vertex colors (set 0) when the mesh has them (4 bytes per vertex).
    pub colors: bool,
    /// Store UV set 1 when the mesh has two sets (4 bytes per vertex).
    pub uv1: bool,
    /// Store joints and weights when the mesh is skinned (12 bytes per vertex).
    pub skin: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            page_size: 128 * 1024,
            position_bits: 16,
            tangents: false,
            colors: false,
            uv1: false,
            skin: false,
        }
    }
}

impl GpuOptions {
    /// Default options that store every attribute the format knows.
    pub fn all_attributes() -> Self {
        GpuOptions {
            tangents: true,
            colors: true,
            uv1: true,
            skin: true,
            ..Default::default()
        }
    }
}

/// The page sizes of this format, for [`plan_pages`]: what [`GpuClusterDag::from_dag`] packs
/// with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FcldLayout {
    pub page_size: usize,
    /// Bytes per vertex: `4 * vertex_words(flags)`.
    pub vertex_stride: usize,
}

/// Bytes a run of `n` bytes takes once padded to 4.
#[inline]
fn pad4(n: usize) -> usize {
    n.div_ceil(4) * 4
}

impl PageLayout for FcldLayout {
    /// Vertex indices, triangles, and per-triangle materials when mixed.
    fn cluster_bytes(&self, cluster: &Cluster, vertices: usize) -> [usize; 4] {
        let t = cluster.triangle_count();
        let m0 = cluster.material_ids.first();
        let mixed = cluster.material_ids.iter().any(|m| Some(m) != m0);
        [
            4 * vertices,
            pad4(3 * t),
            if mixed { pad4(2 * t) } else { 0 },
            0,
        ]
    }

    fn page_bytes(&self, t: &PageTally) -> usize {
        align16(PAGE_HEADER_SIZE + t.vertices * self.vertex_stride)
            + align16(t.bytes[0])
            + align16(t.bytes[1])
            + align16(t.bytes[2])
    }

    fn max_page_bytes(&self) -> usize {
        self.page_size
    }
}

/// A cluster header (see the module docs for its 128-byte layout).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuCluster {
    pub bounds: Sphere,
    pub lod_bounds: Sphere,
    pub parent_bounds: Sphere,
    pub cone: Cone,
    pub lod_error: f32,
    pub parent_error: f32,
    pub level: u32,
    /// Producing group (`u32::MAX` at level 0) and consuming group (`u32::MAX` for roots), as
    /// indices into [`GpuClusterDag::groups`].
    pub group: u32,
    pub parent_group: u32,
    pub page: u32,
    /// Byte offset (from the page start) of the cluster's `u32` page-local vertex indices.
    pub vertex_offset: u32,
    /// Byte offset (from the page start) of the cluster's `u8` triangle micro-indices.
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
    /// The material of every triangle, or `u32::MAX` when they differ (then
    /// `material_offset` points at one `u16` per triangle).
    pub material: u32,
    pub material_offset: u32,
    /// Index of this cluster in the source [`ClusterDag`].
    pub source_cluster: u32,
}

/// A group record (48 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuGroup {
    pub bounds: Sphere,
    pub error: f32,
    /// Level of the clusters it consumed.
    pub level: u32,
    /// Page holding the clusters it produced.
    pub page: u32,
    /// The clusters it produced: `cluster_first..cluster_first + cluster_count`.
    pub cluster_first: u32,
    pub cluster_count: u32,
}

/// A page table entry plus the page header fields.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuPage {
    /// Byte offset of the page in the file.
    pub offset: u64,
    pub size: u32,
    pub cluster_first: u32,
    pub cluster_count: u32,
    pub dep_first: u32,
    pub dep_count: u32,
    /// Highest cluster level in the page.
    pub level: u32,
    pub vertex_count: u32,
    /// Minimum grid corner of the page's vertices.
    pub base: [u32; 3],
}

/// A decoded vertex. Attributes the file lacks are zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    /// Direction and handedness (`w` = ±1).
    pub tangent: Vec4,
    pub color: Vec4,
    pub uv1: Vec2,
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

/// The packed DAG: the metadata decoded, the pages as raw bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuClusterDag {
    /// The `FLAG_*` attributes the vertices carry.
    pub flags: u32,
    pub page_size: u32,
    pub position_bits: u32,
    pub origin: Vec3,
    pub step: f32,
    pub uv_min: Vec2,
    pub uv_step: Vec2,
    pub uv1_min: Vec2,
    pub uv1_step: Vec2,
    pub clusters: Vec<GpuCluster>,
    pub groups: Vec<GpuGroup>,
    pub pages: Vec<GpuPage>,
    pub dependencies: Vec<u32>,
    /// Each page's bytes, exactly as in the file (header included).
    pub page_data: Vec<Vec<u8>>,
}

/// Builds the packed form of `dag` (over `mesh`'s vertices) and writes it out.
pub fn encode_gpu(dag: &ClusterDag, mesh: &Mesh, opts: &GpuOptions) -> Result<Vec<u8>> {
    Ok(GpuClusterDag::from_dag(dag, mesh, opts)?.to_bytes())
}

/// Reads a file written by [`encode_gpu`].
pub fn decode_gpu(bytes: &[u8]) -> Result<GpuClusterDag> {
    GpuClusterDag::from_bytes(bytes)
}

fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::InvalidInput(msg.into()))
}

#[inline]
fn align16(n: usize) -> usize {
    n.div_ceil(16) * 16
}

/// Octahedral encoding of a unit vector as two snorm16 values (x low, y high).
fn oct_encode(n: Vec3) -> u32 {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    if l1 <= 0.0 || !l1.is_finite() {
        return 0;
    }
    let n = n / l1;
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let (x, y) = if n.z >= 0.0 {
        (n.x, n.y)
    } else {
        ((1.0 - n.y.abs()) * sign(n.x), (1.0 - n.x.abs()) * sign(n.y))
    };
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16) as u16 as u32;
    q(x) | (q(y) << 16)
}

fn oct_decode(w: u32) -> Vec3 {
    let s = |v: u16| (v as i16 as f32 / 32767.0).max(-1.0);
    let (x, y) = (s(w as u16), s((w >> 16) as u16));
    let z = 1.0 - x.abs() - y.abs();
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let (x, y) = if z < 0.0 {
        ((1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y))
    } else {
        (x, y)
    };
    Vec3::new(x, y, z).normalize_or_zero()
}

/// Little-endian writer.
struct W(Vec<u8>);

impl W {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn vec3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn sphere(&mut self, s: &Sphere) {
        self.vec3(s.center);
        self.f32(s.radius);
    }
    fn pad16(&mut self) {
        self.0.resize(align16(self.0.len()), 0);
    }
}

/// Bounds-checked little-endian reads.
struct R<'a>(&'a [u8]);

impl R<'_> {
    fn u32(&self, at: usize) -> Result<u32> {
        match self.0.get(at..at.wrapping_add(4)) {
            Some(b) if at.checked_add(4).is_some() => {
                Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            _ => invalid("cluster file truncated"),
        }
    }
    fn f32(&self, at: usize) -> Result<f32> {
        self.u32(at).map(f32::from_bits)
    }
    fn vec3(&self, at: usize) -> Result<Vec3> {
        Ok(Vec3::new(
            self.f32(at)?,
            self.f32(at + 4)?,
            self.f32(at + 8)?,
        ))
    }
    fn sphere(&self, at: usize) -> Result<Sphere> {
        Ok(Sphere {
            center: self.vec3(at)?,
            radius: self.f32(at + 12)?,
        })
    }
}

/// The bounds frame of a unorm16 UV set: `(min, step)`.
fn uv_frame(u: &[Vec2]) -> (Vec2, Vec2) {
    if u.is_empty() {
        return (Vec2::ZERO, Vec2::ZERO);
    }
    let lo = u
        .iter()
        .copied()
        .fold(Vec2::splat(f32::INFINITY), Vec2::min);
    let hi = u
        .iter()
        .copied()
        .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max);
    (lo, (hi - lo) / 65535.0)
}

/// A UV as unorm16 x2 in its frame.
fn uv_encode(t: Vec2, lo: Vec2, step: Vec2) -> u32 {
    let q = |x: f32, lo: f32, s: f32| {
        if s > 0.0 {
            ((x - lo) / s).round().clamp(0.0, 65535.0) as u32
        } else {
            0
        }
    };
    q(t.x, lo.x, step.x) | (q(t.y, lo.y, step.y) << 16)
}

fn uv_decode(w: u32, lo: Vec2, step: Vec2) -> Vec2 {
    lo + Vec2::new((w & 0xFFFF) as f32, (w >> 16) as f32) * step
}

fn unorm8x4(v: [f32; 4]) -> u32 {
    v.iter().enumerate().fold(0, |w, (k, &x)| {
        w | (((x.clamp(0.0, 1.0) * 255.0).round() as u32) << (8 * k))
    })
}

fn unorm8x4_decode(w: u32) -> [f32; 4] {
    [0, 1, 2, 3].map(|k| ((w >> (8 * k)) & 0xFF) as f32 / 255.0)
}

/// Weights as unorm8 x4 adding up to exactly 255 (all zero when they add up to nothing).
fn weights_encode(w: [f32; 4]) -> u32 {
    let w = w.map(|x| if x.is_finite() { x.max(0.0) } else { 0.0 });
    let sum: f32 = w.iter().sum();
    if sum <= 0.0 {
        return 0;
    }
    let mut q = w.map(|x| (x / sum * 255.0).round() as i32);
    // Rounding is off by at most 2; the largest weight absorbs it.
    let big = (0..4).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap_or(0);
    q[big] += 255 - q.iter().sum::<i32>();
    q.iter().enumerate().fold(0, |acc, (k, &x)| {
        acc | ((x.clamp(0, 255) as u32) << (8 * k))
    })
}

/// A tangent as oct snorm16 x2, the handedness in bit 16 (set: negative).
fn tangent_encode(t: Vec4) -> u32 {
    (oct_encode(t.truncate()) & !0x1_0000) | (u32::from(t.w < 0.0) << 16)
}

fn tangent_decode(w: u32) -> Vec4 {
    let s = if w & 0x1_0000 != 0 { -1.0 } else { 1.0 };
    oct_decode(w).extend(s)
}

impl GpuClusterDag {
    /// Packs `dag`, whose clusters index `mesh`'s vertices.
    pub fn from_dag(dag: &ClusterDag, mesh: &Mesh, opts: &GpuOptions) -> Result<GpuClusterDag> {
        if !(1..=21).contains(&opts.position_bits) {
            return invalid("position_bits must be within 1..=21");
        }
        if opts.page_size < 4096
            || !opts.page_size.is_multiple_of(16)
            || opts.page_size > u32::MAX as usize
        {
            return invalid("page_size must be a multiple of 16, at least 4096");
        }
        let nv = mesh.vertex_count();
        for c in &dag.clusters {
            if c.indices.iter().any(|&v| v as usize >= nv) {
                return invalid("cluster index out of range of the mesh");
            }
            if c.material_ids.len() * 3 != c.indices.len() {
                return invalid("cluster material count does not match its triangles");
            }
        }

        // Meshlet layout per cluster.
        struct Local {
            verts: Vec<u32>,
            tris: Vec<u8>,
            material: u32,
        }
        let mut local: Vec<Local> = Vec::with_capacity(dag.clusters.len());
        for c in &dag.clusters {
            let (verts, tris) = c.meshlet();
            if verts.len() > 256 {
                return invalid("a cluster has more than 256 vertices (u8 micro-indices)");
            }
            let m0 = c.material_ids.first().copied().unwrap_or(0);
            let mixed = c.material_ids.iter().any(|&m| m != m0);
            if mixed && c.material_ids.iter().any(|&m| m > u16::MAX as u32) {
                return invalid("material ids above 65535 in a mixed-material cluster");
            }
            local.push(Local {
                verts,
                tris,
                material: if mixed { NONE } else { m0 },
            });
        }

        // Quantization frames and the attributes stored.
        let b = mesh.bounds();
        let origin = b.min;
        let extent = (b.max - b.min).max_element();
        let qmax = ((1u64 << opts.position_bits) - 1) as f32;
        let step = if extent > 0.0 && extent.is_finite() {
            extent / qmax
        } else {
            1.0
        };
        let normals = mesh.normals.as_ref().filter(|n| n.len() == nv);
        let uvs = mesh.uvs.first().filter(|u| u.len() == nv);
        let tangents = mesh
            .tangents
            .as_ref()
            .filter(|t| opts.tangents && t.len() == nv);
        let colors = mesh
            .colors
            .as_ref()
            .filter(|c| opts.colors && c.len() == nv);
        let uv1 = mesh
            .uvs
            .get(1)
            .filter(|u| opts.uv1 && uvs.is_some() && u.len() == nv);
        let skin = mesh
            .skin
            .as_ref()
            .filter(|s| opts.skin && s.joints.len() == nv && s.weights.len() == nv);
        let (uv_min, uv_step) = uvs.map_or((Vec2::ZERO, Vec2::ZERO), |u| uv_frame(u));
        let (uv1_min, uv1_step) = uv1.map_or((Vec2::ZERO, Vec2::ZERO), |u| uv_frame(u));
        let flag = |on: bool, f: u32| if on { f } else { 0 };
        let flags = flag(normals.is_some(), FLAG_NORMALS)
            | flag(uvs.is_some(), FLAG_UVS)
            | flag(tangents.is_some(), FLAG_TANGENTS)
            | flag(colors.is_some(), FLAG_COLORS)
            | flag(uv1.is_some(), FLAG_UV1)
            | flag(skin.is_some(), FLAG_SKIN);
        let stride = 4 * vertex_words(flags);
        let grid = |v: u32| -> [u32; 3] {
            let p = (mesh.positions[v as usize] - origin) / step;
            [p.x, p.y, p.z].map(|c| c.round().clamp(0.0, qmax) as u32)
        };
        // Words 2.. of a vertex.
        let vertex_attrs = |v: u32, w: &mut W| {
            let v = v as usize;
            w.u32(normals.map_or(0, |n| oct_encode(n[v])));
            w.u32(uvs.map_or(0, |u| uv_encode(u[v], uv_min, uv_step)));
            if let Some(t) = tangents {
                w.u32(tangent_encode(t[v]));
            }
            if let Some(c) = colors {
                w.u32(unorm8x4(c[v].to_array()));
            }
            if let Some(u) = uv1 {
                w.u32(uv_encode(u[v], uv1_min, uv1_step));
            }
            if let Some(s) = skin {
                let j = s.joints[v].map(u32::from);
                w.u32(j[0] | (j[1] << 16));
                w.u32(j[2] | (j[3] << 16));
                w.u32(weights_encode(s.weights[v]));
            }
        };

        let layout = FcldLayout {
            page_size: opts.page_size,
            vertex_stride: stride,
        };
        let plan = plan_pages(dag, &layout)?;
        let mut gpu_of = vec![NONE; dag.clusters.len()];
        for (i, c) in plan.cluster_order().enumerate() {
            gpu_of[c as usize] = i as u32;
        }
        let gpage = &plan.group_page;

        let mut clusters: Vec<GpuCluster> = Vec::with_capacity(dag.clusters.len());
        let mut pages: Vec<GpuPage> = Vec::with_capacity(plan.pages.len());
        let mut page_data: Vec<Vec<u8>> = Vec::with_capacity(plan.pages.len());
        let mut dependencies: Vec<u32> = Vec::new();
        // Page-local index of each source vertex, valid for the page being written.
        let mut vmap = vec![0u32; nv];
        let mut first = 0usize;
        for (p, planned) in plan.pages.iter().enumerate() {
            let members = &planned.clusters;
            let verts = &planned.vertices;
            for (i, &v) in verts.iter().enumerate() {
                vmap[v as usize] = i as u32;
            }
            let q: Vec<[u32; 3]> = verts.iter().map(|&v| grid(v)).collect();
            let base = q.iter().fold([u32::MAX; 3], |a, g| {
                [a[0].min(g[0]), a[1].min(g[1]), a[2].min(g[2])]
            });
            let base = if q.is_empty() { [0; 3] } else { base };
            let t = &planned.tally;
            let vertex_offset = PAGE_HEADER_SIZE;
            let index_offset = align16(vertex_offset + verts.len() * stride);
            let triangle_offset = index_offset + align16(t.bytes[0]);
            let material_offset = triangle_offset + align16(t.bytes[1]);
            let size = material_offset + align16(t.bytes[2]);
            debug_assert_eq!(size, layout.page_bytes(t));

            let mut w = W(Vec::with_capacity(size));
            for v in [
                first as u32,
                members.len() as u32,
                verts.len() as u32,
                vertex_offset as u32,
            ] {
                w.u32(v);
            }
            for v in [base[0], base[1], base[2], index_offset as u32] {
                w.u32(v);
            }
            for v in [
                triangle_offset as u32,
                material_offset as u32,
                size as u32,
                p as u32,
            ] {
                w.u32(v);
            }
            for (&v, g) in verts.iter().zip(&q) {
                let d = [g[0] - base[0], g[1] - base[1], g[2] - base[2]];
                w.u32(d[0] | (d[1] << 21));
                w.u32((d[1] >> 11) | (d[2] << 10));
                vertex_attrs(v, &mut w);
            }
            w.pad16();
            let (mut io, mut to, mut mo) = (index_offset, triangle_offset, material_offset);
            for &c in members {
                for &v in &local[c as usize].verts {
                    w.u32(vmap[v as usize]);
                }
            }
            w.pad16();
            debug_assert_eq!(w.0.len(), triangle_offset);
            for &c in members {
                w.0.extend_from_slice(&local[c as usize].tris);
                w.0.resize(pad4(w.0.len()), 0);
            }
            w.pad16();
            for &c in members {
                if local[c as usize].material == NONE {
                    for &m in &dag.clusters[c as usize].material_ids {
                        w.0.extend_from_slice(&(m as u16).to_le_bytes());
                    }
                    w.0.resize(pad4(w.0.len()), 0);
                }
            }
            w.pad16();
            debug_assert_eq!(w.0.len(), size);

            // Cluster headers.
            for &c in members {
                let s = &dag.clusters[c as usize];
                let l = &local[c as usize];
                let mixed = l.material == NONE;
                clusters.push(GpuCluster {
                    bounds: s.bounds,
                    lod_bounds: s.lod_bounds,
                    parent_bounds: s.parent_bounds,
                    cone: s.cone,
                    lod_error: s.lod_error,
                    parent_error: s.parent_error,
                    level: s.level,
                    group: s.group,
                    parent_group: s.parent_group,
                    page: p as u32,
                    vertex_offset: io as u32,
                    triangle_offset: to as u32,
                    vertex_count: l.verts.len() as u32,
                    triangle_count: (l.tris.len() / 3) as u32,
                    material: l.material,
                    material_offset: if mixed { mo as u32 } else { 0 },
                    source_cluster: c,
                });
                io += 4 * l.verts.len();
                to += pad4(l.tris.len());
                if mixed {
                    mo += pad4(2 * l.tris.len() / 3);
                }
            }
            pages.push(GpuPage {
                offset: 0,
                size: size as u32,
                cluster_first: first as u32,
                cluster_count: members.len() as u32,
                dep_first: dependencies.len() as u32,
                dep_count: planned.dependencies.len() as u32,
                level: planned.level,
                vertex_count: verts.len() as u32,
                base,
            });
            dependencies.extend_from_slice(&planned.dependencies);
            page_data.push(w.0);
            first += members.len();
        }
        let groups = dag
            .groups
            .iter()
            .enumerate()
            .map(|(g, grp)| GpuGroup {
                bounds: grp.bounds,
                error: grp.error,
                level: grp.level,
                page: if gpage[g] == NONE { 0 } else { gpage[g] },
                cluster_first: grp.clusters.first().map_or(0, |&c| gpu_of[c as usize]),
                cluster_count: grp.clusters.len() as u32,
            })
            .collect();
        let mut out = GpuClusterDag {
            flags,
            page_size: opts.page_size as u32,
            position_bits: opts.position_bits,
            origin,
            step,
            uv_min,
            uv_step,
            uv1_min,
            uv1_step,
            clusters,
            groups,
            pages,
            dependencies,
            page_data,
        };
        out.layout();
        Ok(out)
    }

    /// Bytes per vertex.
    pub fn vertex_stride(&self) -> usize {
        4 * vertex_words(self.flags)
    }

    /// Offsets of (clusters, groups, page table, dependencies, first page).
    fn sections(&self) -> [usize; 5] {
        let c = HEADER_SIZE;
        let g = c + self.clusters.len() * CLUSTER_STRIDE;
        let p = g + self.groups.len() * GROUP_STRIDE;
        let d = p + self.pages.len() * PAGE_ENTRY_STRIDE;
        let first = align16(d + 4 * self.dependencies.len());
        [c, g, p, d, first]
    }

    /// Sets each page's file offset.
    fn layout(&mut self) {
        let mut at = self.sections()[4] as u64;
        for (p, d) in self.pages.iter_mut().zip(&self.page_data) {
            p.offset = at;
            at += align16(d.len()) as u64;
        }
    }

    /// The file bytes (see the module docs for the layout).
    pub fn to_bytes(&self) -> Vec<u8> {
        let [c_off, g_off, p_off, d_off, first] = self.sections();
        let total = first
            + self
                .page_data
                .iter()
                .map(|d| align16(d.len()))
                .sum::<usize>();
        let mut w = W(Vec::with_capacity(total));
        w.0.extend_from_slice(&MAGIC);
        for v in [VERSION, self.flags, HEADER_SIZE as u32] {
            w.u32(v);
        }
        for v in [
            self.clusters.len(),
            self.groups.len(),
            self.pages.len(),
            self.dependencies.len(),
        ] {
            w.u32(v as u32);
        }
        for v in [
            self.page_size,
            self.position_bits,
            self.vertex_stride() as u32,
            CLUSTER_STRIDE as u32,
        ] {
            w.u32(v);
        }
        w.vec3(self.origin);
        w.f32(self.step);
        for v in [self.uv_min.x, self.uv_min.y, self.uv_step.x, self.uv_step.y] {
            w.f32(v);
        }
        for v in [c_off, g_off, p_off, d_off] {
            w.u32(v as u32);
        }
        for v in [
            self.uv1_min.x,
            self.uv1_min.y,
            self.uv1_step.x,
            self.uv1_step.y,
        ] {
            w.f32(v);
        }
        debug_assert_eq!(w.0.len(), HEADER_SIZE);
        for c in &self.clusters {
            w.sphere(&c.bounds);
            w.sphere(&c.lod_bounds);
            w.sphere(&c.parent_bounds);
            w.vec3(c.cone.apex);
            w.f32(c.cone.cutoff);
            w.vec3(c.cone.axis);
            w.f32(c.lod_error);
            w.f32(c.parent_error);
            for v in [
                c.level,
                c.group,
                c.parent_group,
                c.page,
                c.vertex_offset,
                c.triangle_offset,
            ] {
                w.u32(v);
            }
            w.u32(c.vertex_count | (c.triangle_count << 16));
            for v in [c.material, c.material_offset, c.source_cluster, 0] {
                w.u32(v);
            }
        }
        for g in &self.groups {
            w.sphere(&g.bounds);
            w.f32(g.error);
            for v in [g.level, g.page, g.cluster_first, g.cluster_count, 0, 0, 0] {
                w.u32(v);
            }
        }
        // Page offsets are recomputed so a hand-edited struct still writes a consistent file.
        let mut at = first as u64;
        for (p, d) in self.pages.iter().zip(&self.page_data) {
            w.u32(at as u32);
            w.u32((at >> 32) as u32);
            for v in [
                d.len() as u32,
                p.cluster_first,
                p.cluster_count,
                p.dep_first,
                p.dep_count,
                p.level,
            ] {
                w.u32(v);
            }
            at += align16(d.len()) as u64;
        }
        for &d in &self.dependencies {
            w.u32(d);
        }
        w.pad16();
        for d in &self.page_data {
            w.0.extend_from_slice(d);
            w.pad16();
        }
        w.0
    }

    /// Reads and validates a file written by [`GpuClusterDag::to_bytes`] (or a version 1 file).
    /// Every offset and index is checked, so the accessors never index out of bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<GpuClusterDag> {
        let r = R(bytes);
        if bytes.len() < HEADER_SIZE_V1 || bytes[..4] != MAGIC {
            return invalid("not a farthing cluster file (bad magic)");
        }
        let version = r.u32(4)?;
        let (header_size, known) = match version {
            1 => (HEADER_SIZE_V1, FLAG_NORMALS | FLAG_UVS),
            VERSION => (HEADER_SIZE, FLAGS_ALL),
            _ => {
                return invalid(format!(
                    "unsupported cluster file version {version} (expected 1 or {VERSION})"
                ));
            }
        };
        let flags = r.u32(8)?;
        if flags & !known != 0 || (flags & FLAG_UV1 != 0 && flags & FLAG_UVS == 0) {
            return invalid("unknown or inconsistent vertex flags");
        }
        if bytes.len() < header_size
            || r.u32(12)? as usize != header_size
            || r.u32(40)? as usize != 4 * vertex_words(flags)
            || r.u32(44)? as usize != CLUSTER_STRIDE
        {
            return invalid("unexpected header, vertex or cluster size");
        }
        let stride = 4 * vertex_words(flags);
        let (uv1_min, uv1_step) = if version == 1 {
            (Vec2::ZERO, Vec2::ZERO)
        } else {
            (
                Vec2::new(r.f32(96)?, r.f32(100)?),
                Vec2::new(r.f32(104)?, r.f32(108)?),
            )
        };
        let [nc, ng, np, nd] = [16, 20, 24, 28].map(|o| r.u32(o).map(|v| v as usize));
        let (nc, ng, np, nd) = (nc?, ng?, np?, nd?);
        let page_size = r.u32(32)?;
        let position_bits = r.u32(36)?;
        if !(1..=21).contains(&position_bits) {
            return invalid("position_bits out of range");
        }
        let origin = r.vec3(48)?;
        let step = r.f32(60)?;
        let uv_min = Vec2::new(r.f32(64)?, r.f32(68)?);
        let uv_step = Vec2::new(r.f32(72)?, r.f32(76)?);
        let [c_off, g_off, p_off, d_off] = [80, 84, 88, 92].map(|o| r.u32(o).map(|v| v as usize));
        let (c_off, g_off, p_off, d_off) = (c_off?, g_off?, p_off?, d_off?);
        let fits = |off: usize, n: usize, stride: usize| {
            n.checked_mul(stride)
                .and_then(|s| off.checked_add(s))
                .is_some_and(|end| end <= bytes.len())
                && off.is_multiple_of(4)
        };
        if !fits(c_off, nc, CLUSTER_STRIDE)
            || !fits(g_off, ng, GROUP_STRIDE)
            || !fits(p_off, np, PAGE_ENTRY_STRIDE)
            || !fits(d_off, nd, 4)
        {
            return invalid("cluster file section out of bounds");
        }
        let opt = |v: u32, n: usize| v == NONE || (v as usize) < n;
        let mut clusters = Vec::with_capacity(nc);
        for i in 0..nc {
            let o = c_off + i * CLUSTER_STRIDE;
            let counts = r.u32(o + 108)?;
            let c = GpuCluster {
                bounds: r.sphere(o)?,
                lod_bounds: r.sphere(o + 16)?,
                parent_bounds: r.sphere(o + 32)?,
                cone: Cone {
                    apex: r.vec3(o + 48)?,
                    cutoff: r.f32(o + 60)?,
                    axis: r.vec3(o + 64)?,
                },
                lod_error: r.f32(o + 76)?,
                parent_error: r.f32(o + 80)?,
                level: r.u32(o + 84)?,
                group: r.u32(o + 88)?,
                parent_group: r.u32(o + 92)?,
                page: r.u32(o + 96)?,
                vertex_offset: r.u32(o + 100)?,
                triangle_offset: r.u32(o + 104)?,
                vertex_count: counts & 0xFFFF,
                triangle_count: counts >> 16,
                material: r.u32(o + 112)?,
                material_offset: r.u32(o + 116)?,
                source_cluster: r.u32(o + 120)?,
            };
            if !opt(c.group, ng) || !opt(c.parent_group, ng) || c.page as usize >= np {
                return invalid("cluster group or page out of range");
            }
            clusters.push(c);
        }
        let mut groups = Vec::with_capacity(ng);
        for i in 0..ng {
            let o = g_off + i * GROUP_STRIDE;
            let g = GpuGroup {
                bounds: r.sphere(o)?,
                error: r.f32(o + 16)?,
                level: r.u32(o + 20)?,
                page: r.u32(o + 24)?,
                cluster_first: r.u32(o + 28)?,
                cluster_count: r.u32(o + 32)?,
            };
            if g.page as usize >= np
                || (g.cluster_first as usize).saturating_add(g.cluster_count as usize) > nc
            {
                return invalid("group page or clusters out of range");
            }
            groups.push(g);
        }
        let dependencies: Vec<u32> = (0..nd)
            .map(|i| r.u32(d_off + 4 * i))
            .collect::<Result<_>>()?;
        if dependencies.iter().any(|&d| d as usize >= np) {
            return invalid("page dependency out of range");
        }
        let mut pages = Vec::with_capacity(np);
        let mut page_data = Vec::with_capacity(np);
        for i in 0..np {
            let o = p_off + i * PAGE_ENTRY_STRIDE;
            let offset = r.u32(o)? as u64 | ((r.u32(o + 4)? as u64) << 32);
            let size = r.u32(o + 8)?;
            let start = usize::try_from(offset).ok();
            let Some(data) = start
                .and_then(|s| Some(s..s.checked_add(size as usize)?))
                .and_then(|range| bytes.get(range))
            else {
                return invalid("page out of bounds");
            };
            if (size as usize) < PAGE_HEADER_SIZE || size > page_size {
                return invalid("page size out of range");
            }
            let pr = R(data);
            let p = GpuPage {
                offset,
                size,
                cluster_first: r.u32(o + 12)?,
                cluster_count: r.u32(o + 16)?,
                dep_first: r.u32(o + 20)?,
                dep_count: r.u32(o + 24)?,
                level: r.u32(o + 28)?,
                vertex_count: pr.u32(8)?,
                base: [pr.u32(16)?, pr.u32(20)?, pr.u32(24)?],
            };
            if (p.cluster_first as usize).saturating_add(p.cluster_count as usize) > nc
                || (p.dep_first as usize).saturating_add(p.dep_count as usize) > nd
                || pr.u32(0)? != p.cluster_first
                || pr.u32(4)? != p.cluster_count
                || pr.u32(40)? != size
            {
                return invalid("page header does not match the page table");
            }
            let vend = (pr.u32(12)? as usize)
                .saturating_add((p.vertex_count as usize).saturating_mul(stride));
            if pr.u32(12)? as usize != PAGE_HEADER_SIZE || vend > size as usize {
                return invalid("page vertices out of bounds");
            }
            pages.push(p);
            page_data.push(data.to_vec());
        }
        let out = GpuClusterDag {
            flags,
            page_size,
            position_bits,
            origin,
            step,
            uv_min,
            uv_step,
            uv1_min,
            uv1_step,
            clusters,
            groups,
            pages,
            dependencies,
            page_data,
        };
        // Per-cluster ranges inside their pages, and every index they hold.
        for c in &out.clusters {
            let page = &out.pages[c.page as usize];
            let data = &out.page_data[c.page as usize];
            let (vo, to) = (c.vertex_offset as usize, c.triangle_offset as usize);
            let (nv, nt) = (c.vertex_count as usize, c.triangle_count as usize);
            let in_page =
                |o: usize, len: usize| o.checked_add(len).is_some_and(|e| e <= data.len());
            if vo % 4 != 0 || !in_page(vo, 4 * nv) || !in_page(to, 3 * nt) {
                return invalid("cluster data out of its page");
            }
            if c.material == NONE
                && (c.material_offset % 2 != 0 || !in_page(c.material_offset as usize, 2 * nt))
            {
                return invalid("cluster materials out of its page");
            }
            let pr = R(data);
            for k in 0..nv {
                if pr.u32(vo + 4 * k)? >= page.vertex_count {
                    return invalid("vertex index out of its page");
                }
            }
            if data[to..to + 3 * nt].iter().any(|&t| t as usize >= nv) {
                return invalid("triangle index out of its cluster");
            }
        }
        Ok(out)
    }

    /// The bytes of the whole file this decodes to, i.e. [`GpuClusterDag::to_bytes`]'s length.
    pub fn file_size(&self) -> usize {
        self.sections()[4]
            + self
                .page_data
                .iter()
                .map(|d| align16(d.len()))
                .sum::<usize>()
    }

    /// Clusters to draw (indices into [`GpuClusterDag::clusters`]): the same test as
    /// [`ClusterDag::select`], so the same clusters (see [`GpuCluster::source_cluster`]).
    pub fn select(&self, view: &LodView) -> Vec<u32> {
        (0..self.clusters.len() as u32)
            .filter(|&i| {
                let c = &self.clusters[i as usize];
                view.projected(c.lod_error, &c.lod_bounds) <= view.threshold
                    && view.projected(c.parent_error, &c.parent_bounds) > view.threshold
            })
            .collect()
    }

    /// The pages page `p` depends on.
    pub fn page_dependencies(&self, p: usize) -> &[u32] {
        let pg = &self.pages[p];
        &self.dependencies[pg.dep_first as usize..(pg.dep_first + pg.dep_count) as usize]
    }

    /// Decodes vertex `v` of page `p`.
    pub fn vertex(&self, p: usize, v: u32) -> GpuVertex {
        let page = &self.pages[p];
        let r = R(&self.page_data[p]);
        let o = PAGE_HEADER_SIZE + v as usize * self.vertex_stride();
        let word = |k: usize| r.u32(o + 4 * k).unwrap_or(0);
        let w = [0, 1, 2, 3].map(word);
        let d = [
            w[0] & 0x1F_FFFF,
            (w[0] >> 21) | ((w[1] & 0x3FF) << 11),
            (w[1] >> 10) & 0x1F_FFFF,
        ];
        let g = [0, 1, 2].map(|k| page.base[k].wrapping_add(d[k]) as f32);
        let at = |f: u32| attribute_word(self.flags, f);
        let mut out = GpuVertex {
            position: self.origin + Vec3::from_array(g) * self.step,
            normal: if self.flags & FLAG_NORMALS != 0 {
                oct_decode(w[2])
            } else {
                Vec3::ZERO
            },
            uv: uv_decode(w[3], self.uv_min, self.uv_step),
            ..Default::default()
        };
        if let Some(k) = at(FLAG_TANGENTS) {
            out.tangent = tangent_decode(word(k));
        }
        if let Some(k) = at(FLAG_COLORS) {
            out.color = Vec4::from_array(unorm8x4_decode(word(k)));
        }
        if let Some(k) = at(FLAG_UV1) {
            out.uv1 = uv_decode(word(k), self.uv1_min, self.uv1_step);
        }
        if let Some(k) = at(FLAG_SKIN) {
            let (a, b) = (word(k), word(k + 1));
            out.joints = [a as u16, (a >> 16) as u16, b as u16, (b >> 16) as u16];
            out.weights = unorm8x4_decode(word(k + 2));
        }
        out
    }

    /// Cluster `c`'s page-local vertex indices.
    pub fn cluster_vertices(&self, c: usize) -> Vec<u32> {
        let cl = &self.clusters[c];
        let r = R(&self.page_data[cl.page as usize]);
        (0..cl.vertex_count as usize)
            .map(|k| r.u32(cl.vertex_offset as usize + 4 * k).unwrap_or(0))
            .collect()
    }

    /// Cluster `c`'s triangles as cluster-local (micro) indices.
    pub fn cluster_triangles(&self, c: usize) -> Vec<[u8; 3]> {
        let cl = &self.clusters[c];
        let d = &self.page_data[cl.page as usize];
        let o = cl.triangle_offset as usize;
        d[o..o + 3 * cl.triangle_count as usize]
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect()
    }

    /// Cluster `c`'s material id per triangle.
    pub fn cluster_materials(&self, c: usize) -> Vec<u32> {
        let cl = &self.clusters[c];
        let n = cl.triangle_count as usize;
        if cl.material != NONE {
            return vec![cl.material; n];
        }
        let d = &self.page_data[cl.page as usize];
        let o = cl.material_offset as usize;
        d[o..o + 2 * n]
            .chunks_exact(2)
            .map(|m| u16::from_le_bytes([m[0], m[1]]) as u32)
            .collect()
    }

    /// The selected clusters as one mesh of decoded vertices (one vertex per page vertex used;
    /// vertices shared across pages are equal but not merged).
    pub fn to_mesh(&self, selection: &[u32]) -> Mesh {
        let mut map: std::collections::HashMap<(u32, u32), u32> = Default::default();
        let mut verts: Vec<GpuVertex> = Vec::new();
        let mut indices = Vec::new();
        let mut materials = Vec::new();
        for &c in selection {
            let cl = &self.clusters[c as usize];
            let vs = self.cluster_vertices(c as usize);
            for t in self.cluster_triangles(c as usize) {
                for k in t {
                    let pv = vs[k as usize];
                    let id = *map.entry((cl.page, pv)).or_insert_with(|| {
                        verts.push(self.vertex(cl.page as usize, pv));
                        verts.len() as u32 - 1
                    });
                    indices.push(id);
                }
            }
            materials.extend(self.cluster_materials(c as usize));
        }
        let mut m = Mesh::new(verts.iter().map(|v| v.position).collect(), indices);
        m.material_ids = materials;
        if self.flags & FLAG_NORMALS != 0 {
            m.normals = Some(verts.iter().map(|v| v.normal).collect());
        }
        if self.flags & FLAG_UVS != 0 {
            m.uvs = vec![verts.iter().map(|v| v.uv).collect()];
            if self.flags & FLAG_UV1 != 0 {
                m.uvs.push(verts.iter().map(|v| v.uv1).collect());
            }
        }
        if self.flags & FLAG_TANGENTS != 0 {
            m.tangents = Some(verts.iter().map(|v| v.tangent).collect());
        }
        if self.flags & FLAG_COLORS != 0 {
            m.colors = Some(verts.iter().map(|v| v.color).collect());
        }
        if self.flags & FLAG_SKIN != 0 {
            m.skin = Some(Skin {
                joints: verts.iter().map(|v| v.joints).collect(),
                weights: verts.iter().map(|v| v.weights).collect(),
            });
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{blob, edge_counts};
    use super::super::{ClusterOptions, build_cluster_dag};
    use super::*;
    use std::collections::HashMap;

    fn mesh_with_uvs(n: i32) -> Mesh {
        let mut m = blob(n);
        let uv = m
            .positions
            .iter()
            .map(|p| Vec2::new(p.x * 0.5 + 0.5, p.y * 0.25 + 0.75))
            .collect();
        m.uvs = vec![uv];
        // Two materials, so some clusters are mixed.
        m.material_ids = (0..m.triangle_count())
            .map(|t| (m.triangle_positions(t)[0].x > 0.3) as u32 * 3)
            .collect();
        m
    }

    fn views() -> Vec<LodView> {
        let mut v = Vec::new();
        for d in [1.5f32, 3.0, 8.0, 30.0, 200.0] {
            for t in [0.5f32, 1.0, 4.0] {
                v.push(LodView {
                    eye: Vec3::new(d, d * 0.3, -d * 0.2),
                    pixels_per_radian: 1000.0,
                    threshold: t,
                });
            }
        }
        v
    }

    #[test]
    fn round_trip_is_exact() {
        let m = mesh_with_uvs(40);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let gpu = GpuClusterDag::from_dag(&dag, &m, &GpuOptions::default()).unwrap();
        let bytes = gpu.to_bytes();
        assert_eq!(bytes.len(), gpu.file_size());
        assert_eq!(bytes.len() % 16, 0);
        assert_eq!(&bytes[..4], b"FCLD");
        let back = decode_gpu(&bytes).unwrap();
        assert_eq!(back, gpu);
        assert_eq!(back.to_bytes(), bytes);
        assert_eq!(encode_gpu(&dag, &m, &GpuOptions::default()).unwrap(), bytes);
        assert_eq!(gpu.flags, FLAG_NORMALS | FLAG_UVS);

        // Every DAG cluster appears once, with its fields, triangles and materials intact.
        assert_eq!(gpu.clusters.len(), dag.clusters.len());
        let mut seen = vec![false; dag.clusters.len()];
        for (i, g) in gpu.clusters.iter().enumerate() {
            let s = &dag.clusters[g.source_cluster as usize];
            assert!(!std::mem::replace(
                &mut seen[g.source_cluster as usize],
                true
            ));
            assert_eq!(
                (
                    g.lod_error,
                    g.parent_error,
                    g.lod_bounds,
                    g.parent_bounds,
                    g.level
                ),
                (
                    s.lod_error,
                    s.parent_error,
                    s.lod_bounds,
                    s.parent_bounds,
                    s.level
                )
            );
            assert_eq!((g.bounds, g.cone), (s.bounds, s.cone));
            let (verts, tris) = s.meshlet();
            let pv = gpu.cluster_vertices(i);
            assert_eq!(pv.len(), verts.len());
            assert_eq!(gpu.cluster_triangles(i).concat(), tris);
            assert_eq!(gpu.cluster_materials(i), s.material_ids);
            // Page-local vertices decode to the source vertex they stand for.
            for (k, &v) in verts.iter().enumerate() {
                let d = gpu.vertex(g.page as usize, pv[k]);
                assert!(d.position.distance(m.positions[v as usize]) <= gpu.step);
            }
        }
        assert!(gpu.clusters.iter().any(|c| c.material == u32::MAX));
        for (gi, g) in gpu.groups.iter().enumerate() {
            let src = &dag.groups[gi];
            assert_eq!(
                (g.error, g.bounds, g.level),
                (src.error, src.bounds, src.level)
            );
            let produced: Vec<u32> = (g.cluster_first..g.cluster_first + g.cluster_count)
                .map(|c| gpu.clusters[c as usize].source_cluster)
                .collect();
            assert_eq!(produced, src.clusters);
        }
    }

    #[test]
    fn rejects_bad_input() {
        let m = mesh_with_uvs(12);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let bytes = encode_gpu(&dag, &m, &GpuOptions::default()).unwrap();
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(decode_gpu(&bad).is_err());
        let mut bad = bytes.clone();
        bad[4] = 99;
        assert!(decode_gpu(&bad).is_err());
        for cut in [0, 50, HEADER_SIZE, bytes.len() / 2, bytes.len() - 1] {
            assert!(decode_gpu(&bytes[..cut]).is_err(), "truncated at {cut}");
        }
        // Corrupting any byte must never panic, in decoding or in the accessors.
        for i in 0..bytes.len() {
            let mut b = bytes.clone();
            b[i] ^= 0xA5;
            if let Ok(g) = decode_gpu(&b) {
                let all: Vec<u32> = (0..g.clusters.len() as u32).collect();
                let _ = g.to_mesh(&all);
                for p in 0..g.pages.len() {
                    let _ = g.page_dependencies(p);
                }
            }
        }
        let opts = GpuOptions {
            position_bits: 22,
            ..Default::default()
        };
        assert!(GpuClusterDag::from_dag(&dag, &m, &opts).is_err());
        let opts = GpuOptions {
            page_size: 4096,
            ..Default::default()
        };
        // A group's eight 128-triangle clusters need more than 4 KiB.
        assert!(GpuClusterDag::from_dag(&dag, &m, &opts).is_err());
    }

    #[test]
    fn selection_matches_the_dag() {
        let m = mesh_with_uvs(40);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let bytes = encode_gpu(&dag, &m, &GpuOptions::default()).unwrap();
        let gpu = decode_gpu(&bytes).unwrap();
        let mut sizes = Vec::new();
        for v in views() {
            let mut want = dag.select(&v);
            let mut got: Vec<u32> = gpu
                .select(&v)
                .iter()
                .map(|&c| gpu.clusters[c as usize].source_cluster)
                .collect();
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, want);
            sizes.push(got.len());
        }
        // The views span several cuts.
        sizes.sort_unstable();
        sizes.dedup();
        assert!(sizes.len() > 3, "{sizes:?}");
    }

    #[test]
    fn pages_respect_budget_and_order() {
        let m = mesh_with_uvs(80); // 38,400 triangles
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        for page_size in [32 * 1024, 64 * 1024, 128 * 1024] {
            let opts = GpuOptions {
                page_size,
                ..Default::default()
            };
            let gpu = GpuClusterDag::from_dag(&dag, &m, &opts).unwrap();
            let bytes = gpu.to_bytes();
            assert!(gpu.pages.len() > 1);
            let mut next_cluster = 0;
            let mut prev_level = u32::MAX;
            for (p, page) in gpu.pages.iter().enumerate() {
                assert!(page.size as usize <= page_size);
                assert_eq!(page.offset % 16, 0);
                assert_eq!(page.size % 16, 0);
                assert_eq!(page.cluster_first, next_cluster);
                next_cluster += page.cluster_count;
                // Coarse to fine.
                let levels = (page.cluster_first..page.cluster_first + page.cluster_count)
                    .map(|c| gpu.clusters[c as usize].level);
                assert!(levels.clone().max().unwrap() <= prev_level);
                prev_level = levels.min().unwrap();
                for &d in gpu.page_dependencies(p) {
                    assert!((d as usize) < p, "page {p} depends on later page {d}");
                }
                let o = page.offset as usize;
                assert_eq!(&bytes[o..o + page.size as usize], &gpu.page_data[p][..]);
            }
            assert_eq!(next_cluster as usize, gpu.clusters.len());
            // A cluster's parent group is in its own page or a dependency, and a group's
            // output never spans pages.
            for c in &gpu.clusters {
                if c.parent_group != u32::MAX {
                    let pp = gpu.groups[c.parent_group as usize].page;
                    assert!(pp == c.page || gpu.page_dependencies(c.page as usize).contains(&pp));
                }
                if c.group != u32::MAX {
                    assert_eq!(gpu.groups[c.group as usize].page, c.page);
                }
            }
            // The roots come first.
            let roots = gpu.clusters.iter().filter(|c| c.parent_group == u32::MAX);
            assert!(
                roots.clone().all(|c| c.page < 2),
                "{:?}",
                roots.map(|c| c.page).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn quantization_error_is_bounded() {
        let m = mesh_with_uvs(40);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let b = m.bounds();
        let extent = (b.max - b.min).max_element();
        for bits in [10, 16, 21] {
            let opts = GpuOptions {
                position_bits: bits,
                ..Default::default()
            };
            let gpu = decode_gpu(&encode_gpu(&dag, &m, &opts).unwrap()).unwrap();
            assert_eq!(gpu.step, extent / ((1u32 << bits) - 1) as f32);
            let (mut pe, mut ne, mut ue) = (0.0f32, 0.0f32, Vec2::ZERO);
            for (i, c) in gpu.clusters.iter().enumerate() {
                let (verts, _) = dag.clusters[c.source_cluster as usize].meshlet();
                for (k, &pv) in gpu.cluster_vertices(i).iter().enumerate() {
                    let d = gpu.vertex(c.page as usize, pv);
                    let v = verts[k] as usize;
                    pe = pe.max((d.position - m.positions[v]).abs().max_element());
                    let n = m.normals.as_ref().unwrap()[v].normalize();
                    ne = ne.max(d.normal.dot(n).clamp(-1.0, 1.0).acos());
                    ue = ue.max((d.uv - m.uvs[0][v]).abs());
                }
            }
            // Half a step, plus f32 rounding of origin + q * step.
            let slack = 4.0 * f32::EPSILON * (extent + b.min.abs().max_element());
            assert!(
                pe <= gpu.step * 0.5 + slack,
                "bits {bits}: {pe} > {}",
                gpu.step * 0.5
            );
            assert!(ne < 1e-3, "normal error {ne} rad");
            assert!(ue.x <= gpu.uv_step.x * 0.5 + 1e-6, "{ue}");
            assert!(ue.y <= gpu.uv_step.y * 0.5 + 1e-6, "{ue}");
        }
    }

    #[test]
    fn decoded_cuts_are_crack_free() {
        let m = mesh_with_uvs(40);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let opts = GpuOptions {
            page_size: 32 * 1024,
            ..Default::default()
        };
        let gpu = decode_gpu(&encode_gpu(&dag, &m, &opts).unwrap()).unwrap();
        assert!(gpu.pages.len() > 4);
        for v in views() {
            let mut cut = gpu.to_mesh(&gpu.select(&v));
            // Weld by exact decoded position: pages share one grid, so a vertex split across
            // pages decodes to identical bits.
            let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
            let remap: Vec<u32> = cut
                .positions
                .iter()
                .map(|p| {
                    let n = ids.len() as u32;
                    *ids.entry(p.to_array().map(f32::to_bits)).or_insert(n)
                })
                .collect();
            for i in &mut cut.indices {
                *i = remap[*i as usize];
            }
            for (e, n) in edge_counts(&cut) {
                assert_eq!(n, 2, "edge {e:?} used {n} times");
            }
        }
    }

    /// `mesh_with_uvs` plus tangents, colors, a second UV set and a skin.
    fn mesh_with_everything(n: i32) -> Mesh {
        let mut m = mesh_with_uvs(n);
        let normals = m.normals.clone().unwrap();
        m.tangents = Some(
            normals
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let t = n.any_orthonormal_vector();
                    t.extend(if i % 3 == 0 { -1.0 } else { 1.0 })
                })
                .collect(),
        );
        m.colors = Some(
            m.positions
                .iter()
                .map(|p| (p.abs().min(Vec3::ONE)).extend(0.5))
                .collect(),
        );
        let uv1 = m
            .positions
            .iter()
            .map(|p| Vec2::new(p.z * 3.0 - 1.0, p.x * 7.0))
            .collect();
        m.uvs.push(uv1);
        let nv = m.vertex_count();
        m.skin = Some(Skin {
            joints: (0..nv)
                .map(|i| [i as u16, (i * 7) as u16, 300, u16::MAX])
                .collect(),
            weights: (0..nv)
                .map(|i| {
                    let a = (i % 10) as f32;
                    [a, 10.0 - a, 1.0, 0.0]
                })
                .collect(),
        });
        m
    }

    #[test]
    fn optional_attributes_round_trip() {
        let m = mesh_with_everything(30);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let plain = GpuClusterDag::from_dag(&dag, &m, &GpuOptions::default()).unwrap();
        assert_eq!(plain.flags, FLAG_NORMALS | FLAG_UVS);
        assert_eq!(plain.vertex_stride(), VERTEX_STRIDE);
        let gpu = GpuClusterDag::from_dag(&dag, &m, &GpuOptions::all_attributes()).unwrap();
        assert_eq!(gpu.flags, FLAGS_ALL);
        assert_eq!(gpu.vertex_stride(), 40);
        let bytes = gpu.to_bytes();
        let back = decode_gpu(&bytes).unwrap();
        assert_eq!(back, gpu);
        // Same clusters, more pages: the vertices are bigger.
        assert_eq!(gpu.clusters.len(), plain.clusters.len());
        assert!(gpu.file_size() > plain.file_size());

        let tans = m.tangents.as_ref().unwrap();
        let cols = m.colors.as_ref().unwrap();
        let skin = m.skin.as_ref().unwrap();
        for (i, c) in gpu.clusters.iter().enumerate() {
            let (verts, _) = dag.clusters[c.source_cluster as usize].meshlet();
            for (k, &pv) in gpu.cluster_vertices(i).iter().enumerate() {
                let d = gpu.vertex(c.page as usize, pv);
                let v = verts[k] as usize;
                // The base attributes decode as in the plain file.
                let p = plain
                    .clusters
                    .iter()
                    .position(|x| x.source_cluster == c.source_cluster);
                let pc = &plain.clusters[p.unwrap()];
                let pd = plain.vertex(pc.page as usize, plain.cluster_vertices(p.unwrap())[k]);
                assert_eq!(
                    (d.position, d.normal, d.uv),
                    (pd.position, pd.normal, pd.uv)
                );
                assert_eq!(pd.tangent, Vec4::ZERO);

                assert_eq!(d.tangent.w, tans[v].w);
                assert!(d.tangent.truncate().dot(tans[v].truncate()) > 1e-3f32.cos());
                assert!((d.color - cols[v]).abs().max_element() <= 0.5 / 255.0 + 1e-6);
                let e = (d.uv1 - m.uvs[1][v]).abs();
                assert!(e.x <= gpu.uv1_step.x * 0.5 + 1e-6 && e.y <= gpu.uv1_step.y * 0.5 + 1e-6);
                assert_eq!(d.joints, skin.joints[v]);
                let sum: f32 = skin.weights[v].iter().sum();
                let qsum: u32 = d.weights.iter().map(|w| (w * 255.0).round() as u32).sum();
                assert_eq!(qsum, 255);
                for (a, b) in d.weights.iter().zip(skin.weights[v]) {
                    assert!(
                        (a - b / sum).abs() <= 2.0 / 255.0 + 1e-6,
                        "{:?} {:?}",
                        d.weights,
                        skin.weights[v]
                    );
                }
            }
        }
        let all: Vec<u32> = (0..gpu.clusters.len() as u32).collect();
        let mesh = gpu.to_mesh(&all);
        assert!(mesh.tangents.is_some() && mesh.colors.is_some() && mesh.skin.is_some());
        assert_eq!(mesh.uvs.len(), 2);

        // Only what is asked for and present.
        let opts = GpuOptions {
            colors: true,
            skin: true,
            ..Default::default()
        };
        let mut no_skin = m.clone();
        no_skin.skin = None;
        let g = GpuClusterDag::from_dag(&dag, &no_skin, &opts).unwrap();
        assert_eq!(g.flags, FLAG_NORMALS | FLAG_UVS | FLAG_COLORS);
        assert_eq!(attribute_word(g.flags, FLAG_COLORS), Some(4));
        assert_eq!(attribute_word(gpu.flags, FLAG_SKIN), Some(7));
        assert_eq!(attribute_word(g.flags, FLAG_SKIN), None);
    }

    #[test]
    fn layout_matches_written_pages() {
        let m = mesh_with_everything(40);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        for opts in [GpuOptions::default(), GpuOptions::all_attributes()] {
            let opts = GpuOptions {
                page_size: 32 * 1024,
                ..opts
            };
            let gpu = GpuClusterDag::from_dag(&dag, &m, &opts).unwrap();
            let layout = FcldLayout {
                page_size: opts.page_size,
                vertex_stride: gpu.vertex_stride(),
            };
            let plan = plan_pages(&dag, &layout).unwrap();
            assert_eq!(plan.pages.len(), gpu.pages.len());
            for (pp, gp) in plan.pages.iter().zip(&gpu.pages) {
                assert_eq!(layout.page_bytes(&pp.tally), gp.size as usize);
                assert_eq!(pp.vertices.len(), gp.vertex_count as usize);
            }
            let order: Vec<u32> = plan.cluster_order().collect();
            let src: Vec<u32> = gpu.clusters.iter().map(|c| c.source_cluster).collect();
            assert_eq!(order, src);
        }
    }

    /// Rewrites a version 2 file without optional attributes as version 1.
    fn as_v1(v2: &[u8]) -> Vec<u8> {
        let u = |at: usize| u32::from_le_bytes(v2[at..at + 4].try_into().unwrap());
        let shift = (HEADER_SIZE - HEADER_SIZE_V1) as u32;
        let mut out = v2[..HEADER_SIZE_V1].to_vec();
        out[4..8].copy_from_slice(&1u32.to_le_bytes());
        out[12..16].copy_from_slice(&(HEADER_SIZE_V1 as u32).to_le_bytes());
        for at in [80, 84, 88, 92] {
            out[at..at + 4].copy_from_slice(&(u(at) - shift).to_le_bytes());
        }
        out.extend_from_slice(&v2[HEADER_SIZE..]);
        // Page table file offsets.
        let (p_off, np) = ((u(88) - shift) as usize, u(24) as usize);
        for i in 0..np {
            let at = p_off + i * PAGE_ENTRY_STRIDE;
            let o = u32::from_le_bytes(out[at..at + 4].try_into().unwrap());
            out[at..at + 4].copy_from_slice(&(o - shift).to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_version_1() {
        let m = mesh_with_uvs(20);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        let gpu = GpuClusterDag::from_dag(&dag, &m, &GpuOptions::default()).unwrap();
        let v1 = as_v1(&gpu.to_bytes());
        let back = decode_gpu(&v1).unwrap();
        assert_eq!(back.clusters, gpu.clusters);
        assert_eq!(back.page_data, gpu.page_data);
        assert_eq!(back.to_bytes(), gpu.to_bytes());
        // Version 1 knows no optional attributes.
        let mut bad = v1.clone();
        bad[8] |= FLAG_COLORS as u8;
        assert!(decode_gpu(&bad).is_err());
    }

    #[test]
    fn oct_encoding_round_trips() {
        for i in 0..2000 {
            let t = i as f32 * 0.37;
            let n =
                Vec3::new(t.sin() * (t * 1.3).cos(), (t * 0.7).cos(), (t * 2.1).sin()).normalize();
            let d = oct_decode(oct_encode(n));
            assert!(d.dot(n) > 1e-3f32.cos(), "{n} -> {d}");
        }
        for n in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            assert!(oct_decode(oct_encode(n)).dot(n) > 0.99999);
        }
    }
}
