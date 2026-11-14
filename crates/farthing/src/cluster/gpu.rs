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
//! default). The unit of packing is a group's output: the clusters one [`ClusterGroup`]
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
//! Tangents, colors, further UV sets, skins and morph targets are not stored.
//!
//! # File layout
//!
//! All values are little-endian; `u32`/`f32` unless stated. Every section and every page
//! starts on a 16-byte boundary, and every record is a whole number of `vec4`/`uvec4`s, so
//! each array reads as a std430 struct array.
//!
//! **File header** (96 bytes, offset 0):
//!
//! | Offset | Field |
//! |---|---|
//! | 0 | magic `b"FCLD"` (4 bytes), version (= [`VERSION`]), flags (bit 0: normals, bit 1: UVs), header size (96) |
//! | 16 | cluster count, group count, page count, dependency count |
//! | 32 | page size, position bits, vertex stride (16), cluster stride (128) |
//! | 48 | origin.xyz, step |
//! | 64 | uv_min.xy, uv_step.xy |
//! | 80 | cluster array offset, group array offset, page table offset, dependency array offset |
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
//! | 48 | vertices, 16 bytes each (`uvec4`): `x` = Δx \| Δy << 21, `y` = Δy >> 11 \| Δz << 10 (21 bits per axis), `z` = normal (oct snorm16 x in the low half, y in the high half), `w` = uv (unorm16 u low, v high) |
//! | vertex-index offset | per cluster, `vertex count` page-local vertex indices (`u32`) |
//! | triangle offset | per cluster, three `u8` cluster-local vertex indices per triangle (meshlet micro-indices), each cluster's run padded to 4 bytes |
//! | material offset | per mixed-material cluster, one `u16` material id per triangle, padded to 4 bytes |
//!
//! Sections start on 16-byte boundaries; the byte size is a multiple of 16. A triangle's
//! vertex `k` is `vertices[indices[vertex_offset/4 + tri[k]]]`.

use super::{ClusterDag, Cone, LodView, Sphere};
use crate::{Error, Mesh, Result};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// File magic.
pub const MAGIC: [u8; 4] = *b"FCLD";
/// Format version written by [`GpuClusterDag::to_bytes`].
pub const VERSION: u32 = 1;
/// File header size in bytes.
pub const HEADER_SIZE: usize = 96;
/// Cluster header size in bytes.
pub const CLUSTER_STRIDE: usize = 128;
/// Group record size in bytes.
pub const GROUP_STRIDE: usize = 48;
/// Page table entry size in bytes.
pub const PAGE_ENTRY_STRIDE: usize = 32;
/// Page header size in bytes.
pub const PAGE_HEADER_SIZE: usize = 48;
/// Vertex record size in bytes.
pub const VERTEX_STRIDE: usize = 16;

/// Flag: vertices carry normals.
pub const FLAG_NORMALS: u32 = 1;
/// Flag: vertices carry UVs.
pub const FLAG_UVS: u32 = 2;

const NONE: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuOptions {
    /// Largest page in bytes (a multiple of 16, at least 4 KiB). Every group's output must fit
    /// in one page: eight full 128-triangle clusters take about 17 KiB, so use 32 KiB or more
    /// with the default cluster options.
    pub page_size: usize,
    /// Position grid resolution per axis, 1 to 21 bits over the mesh's largest extent. Every
    /// value costs the same 8 bytes per vertex.
    pub position_bits: u32,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            page_size: 128 * 1024,
            position_bits: 16,
        }
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

/// A decoded vertex.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

/// The packed DAG: the metadata decoded, the pages as raw bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuClusterDag {
    pub flags: u32,
    pub page_size: u32,
    pub position_bits: u32,
    pub origin: Vec3,
    pub step: f32,
    pub uv_min: Vec2,
    pub uv_step: Vec2,
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

/// One packing unit: clusters that always share a page.
struct Unit {
    level: u32,
    clusters: Vec<u32>,
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
        let ng = dag.groups.len();
        if dag.clusters.iter().any(|c| {
            (c.group != NONE && c.group as usize >= ng)
                || (c.parent_group != NONE && c.parent_group as usize >= ng)
        }) {
            return invalid("cluster group out of range");
        }

        // Quantization frames.
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
        let (uv_min, uv_step) = match uvs {
            Some(u) if !u.is_empty() => {
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
            _ => (Vec2::ZERO, Vec2::ZERO),
        };
        let flags = if normals.is_some() { FLAG_NORMALS } else { 0 }
            | if uvs.is_some() { FLAG_UVS } else { 0 };
        let grid = |v: u32| -> [u32; 3] {
            let p = (mesh.positions[v as usize] - origin) / step;
            [p.x, p.y, p.z].map(|c| c.round().clamp(0.0, qmax) as u32)
        };
        let vertex_attrs = |v: u32| -> (u32, u32) {
            let n = normals.map_or(0, |n| oct_encode(n[v as usize]));
            let uv = uvs.map_or(0, |u| {
                let q = |x: f32, lo: f32, s: f32| {
                    if s > 0.0 {
                        ((x - lo) / s).round().clamp(0.0, 65535.0) as u32
                    } else {
                        0
                    }
                };
                let t = u[v as usize];
                q(t.x, uv_min.x, uv_step.x) | (q(t.y, uv_min.y, uv_step.y) << 16)
            });
            (n, uv)
        };

        // Units: each group's output, then source-level clusters by consuming group; coarse
        // to fine.
        let mut units: Vec<Unit> = dag
            .groups
            .iter()
            .map(|g| Unit {
                level: g.level + 1,
                clusters: g.clusters.clone(),
            })
            .collect();
        let mut by_parent: Vec<Vec<u32>> = vec![Vec::new(); ng + 1];
        for (i, c) in dag.clusters.iter().enumerate() {
            if c.group == NONE {
                let k = if c.parent_group == NONE {
                    ng
                } else {
                    c.parent_group as usize
                };
                by_parent[k].push(i as u32);
            }
        }
        for list in by_parent {
            if !list.is_empty() {
                units.push(Unit {
                    level: dag.clusters[list[0] as usize].level,
                    clusters: list,
                });
            }
        }
        // Stable: within a level, groups keep their (spatially coherent) build order.
        units.sort_by_key(|u| std::cmp::Reverse(u.level));

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

        // Greedy page packing, sizes counted exactly as written. `in_page[v]` is the page
        // (+1) that holds source vertex `v`, `seen[v]` the unit (+1) that last counted it.
        let mut in_page = vec![0u32; nv];
        let mut seen = vec![0u32; nv];
        let mut page_units: Vec<Vec<usize>> = Vec::new();
        let mut open: Vec<usize> = Vec::new();
        // Vertices, index bytes, triangle bytes and material bytes of the open page.
        let mut sums = [0usize; 4];
        let bytes = |s: [usize; 4]| {
            align16(PAGE_HEADER_SIZE + s[0] * VERTEX_STRIDE)
                + align16(s[1])
                + align16(s[2])
                + align16(s[3])
        };
        let too_big = |n: usize| {
            invalid(format!(
                "a group's clusters need {n} bytes, more than page_size {}",
                opts.page_size
            ))
        };
        for u in 0..units.len() {
            let page = page_units.len() as u32 + 1;
            let tag = u as u32 + 1;
            let mut grown = sums;
            for &c in &units[u].clusters {
                let l = &local[c as usize];
                for &v in &l.verts {
                    let v = v as usize;
                    if in_page[v] != page && seen[v] != tag {
                        seen[v] = tag;
                        grown[0] += 1;
                    }
                }
                grown[1] += 4 * l.verts.len();
                grown[2] += l.tris.len().div_ceil(4) * 4;
                if l.material == NONE {
                    grown[3] += (2 * l.tris.len() / 3).div_ceil(4) * 4;
                }
            }
            if bytes(grown) > opts.page_size {
                if open.is_empty() {
                    return too_big(bytes(grown));
                }
                page_units.push(std::mem::take(&mut open));
                sums = [0; 4];
            }
            let page = page_units.len() as u32 + 1;
            for &c in &units[u].clusters {
                let l = &local[c as usize];
                for &v in &l.verts {
                    let v = v as usize;
                    if in_page[v] != page {
                        in_page[v] = page;
                        sums[0] += 1;
                    }
                }
                sums[1] += 4 * l.verts.len();
                sums[2] += l.tris.len().div_ceil(4) * 4;
                if l.material == NONE {
                    sums[3] += (2 * l.tris.len() / 3).div_ceil(4) * 4;
                }
            }
            if bytes(sums) > opts.page_size {
                return too_big(bytes(sums));
            }
            open.push(u);
        }
        if !open.is_empty() {
            page_units.push(open);
        }

        // Cluster order and pages.
        let mut gpu_of = vec![NONE; dag.clusters.len()];
        let mut order: Vec<u32> = Vec::with_capacity(dag.clusters.len());
        let mut page_of_cluster: Vec<u32> = Vec::with_capacity(dag.clusters.len());
        for (p, us) in page_units.iter().enumerate() {
            for &u in us {
                for &c in &units[u].clusters {
                    gpu_of[c as usize] = order.len() as u32;
                    order.push(c);
                    page_of_cluster.push(p as u32);
                }
            }
        }
        if gpu_of.contains(&NONE) {
            return invalid("a cluster belongs to no group and is consumed by none");
        }
        // A group's page: where its produced clusters are (`u32::MAX` if it produced none).
        let gpage: Vec<u32> = dag
            .groups
            .iter()
            .map(|g| {
                g.clusters
                    .first()
                    .map_or(NONE, |&c| page_of_cluster[gpu_of[c as usize] as usize])
            })
            .collect();

        let mut clusters: Vec<GpuCluster> = Vec::with_capacity(order.len());
        let mut pages: Vec<GpuPage> = Vec::with_capacity(page_units.len());
        let mut page_data: Vec<Vec<u8>> = Vec::with_capacity(page_units.len());
        let mut dependencies: Vec<u32> = Vec::new();
        let mut first = 0usize;
        for (p, us) in page_units.iter().enumerate() {
            let members: Vec<u32> = us
                .iter()
                .flat_map(|&u| units[u].clusters.iter().copied())
                .collect();
            // Page-local vertices in first-use order.
            let mut vmap: std::collections::HashMap<u32, u32> = Default::default();
            let mut verts: Vec<u32> = Vec::new();
            for &c in &members {
                for &v in &local[c as usize].verts {
                    vmap.entry(v).or_insert_with(|| {
                        verts.push(v);
                        verts.len() as u32 - 1
                    });
                }
            }
            let q: Vec<[u32; 3]> = verts.iter().map(|&v| grid(v)).collect();
            let base = q.iter().fold([u32::MAX; 3], |a, g| {
                [a[0].min(g[0]), a[1].min(g[1]), a[2].min(g[2])]
            });
            let base = if q.is_empty() { [0; 3] } else { base };
            let vertex_offset = PAGE_HEADER_SIZE;
            let index_offset = align16(vertex_offset + verts.len() * VERTEX_STRIDE);
            let index_bytes: usize = members
                .iter()
                .map(|&c| 4 * local[c as usize].verts.len())
                .sum();
            let triangle_offset = index_offset + align16(index_bytes);
            let tri_bytes: usize = members
                .iter()
                .map(|&c| local[c as usize].tris.len().div_ceil(4) * 4)
                .sum();
            let material_offset = triangle_offset + align16(tri_bytes);
            let mat_bytes: usize = members
                .iter()
                .filter(|&&c| local[c as usize].material == NONE)
                .map(|&c| (2 * local[c as usize].tris.len() / 3).div_ceil(4) * 4)
                .sum();
            let size = material_offset + align16(mat_bytes);
            debug_assert!(size <= opts.page_size);

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
                let (n, uv) = vertex_attrs(v);
                w.u32(d[0] | (d[1] << 21));
                w.u32((d[1] >> 11) | (d[2] << 10));
                w.u32(n);
                w.u32(uv);
            }
            w.pad16();
            let (mut io, mut to, mut mo) = (index_offset, triangle_offset, material_offset);
            for &c in &members {
                let l = &local[c as usize];
                for &v in &l.verts {
                    w.u32(vmap[&v]);
                }
            }
            w.pad16();
            debug_assert_eq!(w.0.len(), triangle_offset);
            for &c in &members {
                w.0.extend_from_slice(&local[c as usize].tris);
                w.0.resize(w.0.len().div_ceil(4) * 4, 0);
            }
            w.pad16();
            for &c in &members {
                if local[c as usize].material == NONE {
                    for &m in &dag.clusters[c as usize].material_ids {
                        w.0.extend_from_slice(&(m as u16).to_le_bytes());
                    }
                    w.0.resize(w.0.len().div_ceil(4) * 4, 0);
                }
            }
            w.pad16();
            debug_assert_eq!(w.0.len(), size);

            // Cluster headers.
            for &c in &members {
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
                to += l.tris.len().div_ceil(4) * 4;
                if mixed {
                    mo += (2 * l.tris.len() / 3).div_ceil(4) * 4;
                }
            }
            // Dependencies: pages of the parent groups' clusters.
            let mut deps: Vec<u32> = members
                .iter()
                .map(|&c| dag.clusters[c as usize].parent_group)
                .filter(|&g| g != NONE)
                .map(|g| gpage[g as usize])
                .filter(|&dp| dp != NONE && dp != p as u32)
                .collect();
            deps.sort_unstable();
            deps.dedup();
            pages.push(GpuPage {
                offset: 0,
                size: size as u32,
                cluster_first: first as u32,
                cluster_count: members.len() as u32,
                dep_first: dependencies.len() as u32,
                dep_count: deps.len() as u32,
                level: members
                    .iter()
                    .map(|&c| dag.clusters[c as usize].level)
                    .max()
                    .unwrap_or(0),
                vertex_count: verts.len() as u32,
                base,
            });
            dependencies.extend(deps);
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
            clusters,
            groups,
            pages,
            dependencies,
            page_data,
        };
        out.layout();
        Ok(out)
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
            VERTEX_STRIDE as u32,
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

    /// Reads and validates a file written by [`GpuClusterDag::to_bytes`]. Every offset and index
    /// is checked, so the accessors never index out of bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<GpuClusterDag> {
        let r = R(bytes);
        if bytes.len() < HEADER_SIZE || bytes[..4] != MAGIC {
            return invalid("not a farthing cluster file (bad magic)");
        }
        let version = r.u32(4)?;
        if version != VERSION {
            return invalid(format!(
                "unsupported cluster file version {version} (expected {VERSION})"
            ));
        }
        let flags = r.u32(8)?;
        if r.u32(12)? as usize != HEADER_SIZE
            || r.u32(40)? as usize != VERTEX_STRIDE
            || r.u32(44)? as usize != CLUSTER_STRIDE
        {
            return invalid("unexpected header, vertex or cluster size");
        }
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
            let vend =
                (pr.u32(12)? as usize).saturating_add(p.vertex_count as usize * VERTEX_STRIDE);
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
        let o = PAGE_HEADER_SIZE + v as usize * VERTEX_STRIDE;
        let w = [0, 4, 8, 12].map(|k| r.u32(o + k).unwrap_or(0));
        let d = [
            w[0] & 0x1F_FFFF,
            (w[0] >> 21) | ((w[1] & 0x3FF) << 11),
            (w[1] >> 10) & 0x1F_FFFF,
        ];
        let g = [0, 1, 2].map(|k| page.base[k].wrapping_add(d[k]) as f32);
        GpuVertex {
            position: self.origin + Vec3::from_array(g) * self.step,
            normal: if self.flags & FLAG_NORMALS != 0 {
                oct_decode(w[2])
            } else {
                Vec3::ZERO
            },
            uv: self.uv_min + Vec2::new((w[3] & 0xFFFF) as f32, (w[3] >> 16) as f32) * self.uv_step,
        }
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
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{blob, edge_counts};
    use super::super::{build_cluster_dag, ClusterOptions};
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
