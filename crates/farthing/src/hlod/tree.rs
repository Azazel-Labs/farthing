//! Nested HLOD cells: a level partitioned into a tree of proxies.
//!
//! [`build_hlod_tree`] assigns every instance to one cell of a loose octree (or quadtree) by
//! the centre of its bounds, builds a proxy per leaf cell from its instances with the
//! [`build_hlod`](super::build_hlod) pipeline, then a proxy per parent cell from its
//! children's proxies (or from the source, see [`ParentSource`]). Instances too large for a
//! cell go up to the first level whose cells are big enough. [`HlodTree::select`] picks, for a
//! camera, which proxies to draw and which instances stay as originals.
//!
//! # Budgets and errors
//!
//! Cells double in size per level. The error budget doubles with them
//! (`max_error × 2^level`), and the triangle budget is `target_triangles ×
//! triangle_scale^level`, capped for a parent at `parent_reduction` × its input. With the
//! default `triangle_scale` of 1, every node's proxy covers about the same number of pixels
//! at its switch distance, so every node gets about the same number of triangles.
//!
//! A node's [`error`](HlodNode::error) is a world-space deviation, in the same convention as
//! [`crate::cluster`]: the proxy may replace the instances once
//! [`LodView::projected`] of that error at the node's bounding sphere is at most the pixel
//! threshold, that is beyond [`HlodNode::switch_distance`] from the sphere's centre. Built from
//! children, a parent's error is its own reduction error plus the largest child error. With
//! [`HlodTreeOptions::texel_error`], one atlas texel (in world units) counts as error too, so
//! a proxy is not drawn while its texels are larger than the pixel threshold. A node's error
//! is never below its children's.
//!
//! # Host, memory and threads
//!
//! Visibility and baking go through the [`RenderHost`] exactly as
//! [`build_hlod`](super::build_hlod) does, one node at a time; when baking, every node's textures are also registered with
//! [`RenderHost::create_material`] (its [`HlodNode::material`]), because a parent built from
//! its children is baked from their materials. Merging, reduction and unwrapping run in
//! parallel for a batch of cells. Hidden-surface removal for a node considers everything that
//! node replaces at once: its own instances and all its children's proxies (or, from source,
//! every instance below it).
//!
//! Only one batch of culled cell sources is alive at a time
//! ([`HlodTreeOptions::parallel_cells`]), merged full-resolution meshes
//! live only inside each cell's reduction, and bake correspondences are dropped
//! ([`HlodTreeOptions::keep_correspondence`]). [`build_hlod_tree_with`] hands every finished
//! node to a callback as soon as the builder no longer needs its proxy (after its parent is
//! built, or at once when parents are built from source), so a caller can write proxies out
//! and drop them instead of holding the whole tree's meshes and textures.

use super::{bake_stage, cull_stage, shape_stage, Culled, HlodOptions, HlodResult};
use crate::cluster::{LodView, Sphere};
use crate::host::{BakedMaterialDesc, BakedMaterialKind, MaterialClass, RenderHost};
use crate::{Aabb, Error, Instance, MaterialId, MaterialSlot, Mesh, Result, Scene};
use glam::{Mat4, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Material slot id of unbaked child proxies inside a parent's input. Hosts are asked
/// [`RenderHost::material_info`] for it during visibility and should report it opaque (the
/// default [`crate::host::MaterialInfo`]).
pub const PROXY_MATERIAL: MaterialId = MaterialId(u32::MAX);

/// How space is divided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellGrid {
    /// Cubic cells, 8 children per parent.
    Octree,
    /// Columns over the XZ plane (Y up, the glTF convention), 4 children per parent, as a
    /// world-partition grid. Heights never split cells or push instances up a level.
    Quadtree,
}

/// What a parent cell's proxy is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParentSource {
    /// Its children's proxies (and its own large instances): cheap, errors add up per level.
    /// How world-partition HLOD layers stack.
    Children,
    /// Every source instance below it: higher quality, the cost of a source build per level.
    Source,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HlodTreeOptions {
    /// Edge length of a leaf cell in world units. Level `l` cells are `cell_size × 2^l`.
    pub cell_size: f32,
    pub grid: CellGrid,
    /// An instance goes to the first level whose cells are at least its largest extent
    /// divided by this (1: the loose-octree rule, an instance fits a cell when it fits within
    /// the cell grown by half a cell on every side).
    pub loose_factor: f32,
    /// Most levels, leaves included (None: up to a single root). With a cap the top level
    /// can have several roots.
    pub max_levels: Option<u32>,
    pub parents: ParentSource,
    /// Proxy settings for leaf cells. Parents use the same with scaled budgets.
    pub proxy: HlodOptions,
    /// Triangle budget multiplier per level (`target_triangles × triangle_scale^level`).
    pub triangle_scale: f32,
    /// A parent's triangle target is at most this fraction of its input (children's proxies
    /// plus its own instances), so every level coarsens. Like any triangle target it yields
    /// to `max_error` (whichever is reached first).
    pub parent_reduction: f32,
    /// Count one atlas texel (world units) as error, so a proxy is not drawn while its
    /// texels are larger than the pixel threshold.
    pub texel_error: bool,
    /// Keep each bake's texel correspondence (large: one entry per supersampled texel).
    pub keep_correspondence: bool,
    /// Cells whose culled sources and reductions are in flight at once (0: one per rayon
    /// thread). Peak memory grows with it; the output does not depend on it.
    pub parallel_cells: usize,
}

impl Default for HlodTreeOptions {
    fn default() -> Self {
        HlodTreeOptions {
            cell_size: 64.0,
            grid: CellGrid::Octree,
            loose_factor: 1.0,
            max_levels: None,
            parents: ParentSource::Children,
            proxy: HlodOptions::default(),
            triangle_scale: 1.0,
            parent_reduction: 0.5,
            texel_error: true,
            keep_correspondence: false,
            parallel_cells: 0,
        }
    }
}

/// One cell of the tree.
#[derive(Clone, Debug)]
pub struct HlodNode {
    /// 0 for the finest cells.
    pub level: u32,
    /// Integer cell coordinates on this level's grid (`origin + cell × cell_size × 2^level`
    /// is the cell's minimum corner; y is 0 with [`CellGrid::Quadtree`]).
    pub cell: [u32; 3],
    /// Bounds of everything the node replaces, and of its proxy.
    pub bounds: Aabb,
    /// The sphere around `bounds` used for selection.
    pub sphere: Sphere,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
    /// Instances placed at this node itself (too large for its children's cells).
    pub own_instances: Vec<u32>,
    /// Every instance the proxy replaces (its own and all its descendants'), ascending.
    pub instances: Vec<u32>,
    /// The proxy (None once taken by the [`build_hlod_tree_with`] callback).
    pub proxy: Option<HlodResult>,
    /// The engine material made from the proxy's bake.
    pub material: Option<MaterialId>,
    /// Proxy triangles (kept when the proxy is taken).
    pub triangles: usize,
    /// Source triangles of all replaced instances.
    pub source_triangles: usize,
    /// Accumulated geometric deviation from the source, world units.
    pub geometric_error: f32,
    /// World size of one atlas texel (0 without UVs).
    pub texel_size: f32,
    /// The error selection uses: geometric error, texel size (when counted) and at least
    /// every child's error.
    pub error: f32,
}

impl HlodNode {
    /// Distance from `sphere.center` beyond which the proxy's error projects to at most
    /// `threshold` pixels ([`LodView::projected`]); 0 when it is always acceptable.
    pub fn switch_distance(&self, pixels_per_radian: f32, threshold: f32) -> f32 {
        if self.error <= 0.0 {
            0.0
        } else {
            self.error * pixels_per_radian / threshold.max(f32::MIN_POSITIVE) + self.sphere.radius
        }
    }
}

/// The result of [`build_hlod_tree`].
#[derive(Clone, Debug)]
pub struct HlodTree {
    /// Children always come before their parents.
    pub nodes: Vec<HlodNode>,
    pub roots: Vec<u32>,
    pub levels: u32,
    pub cell_size: f32,
    pub grid: CellGrid,
    /// Minimum corner of cell (0, 0, 0) on every level.
    pub origin: Vec3,
    /// Per source instance, the node it was placed at (None for instances without triangles,
    /// which no node replaces and selection never lists).
    pub instance_node: Vec<Option<u32>>,
}

/// What to draw for one view: proxies and the instances that stay as originals. Every
/// instance with triangles is covered exactly once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HlodCut {
    /// Nodes whose proxies are drawn, ascending.
    pub nodes: Vec<u32>,
    /// Source instances drawn as themselves, ascending.
    pub instances: Vec<u32>,
}

impl HlodTree {
    /// The coarsest acceptable proxies for `view`: from each root down, a node is drawn when
    /// its error projects to at most `view.threshold` pixels; otherwise its own instances
    /// are drawn as originals and its children are tested in turn.
    pub fn select(&self, view: &LodView) -> HlodCut {
        let mut cut = HlodCut::default();
        let mut stack: Vec<u32> = self.roots.iter().rev().copied().collect();
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            if view.projected(node.error, &node.sphere) <= view.threshold {
                cut.nodes.push(n);
            } else {
                cut.instances.extend_from_slice(&node.own_instances);
                stack.extend(node.children.iter().rev());
            }
        }
        cut.nodes.sort_unstable();
        cut.instances.sort_unstable();
        cut
    }

    /// Node indices of one level.
    pub fn level(&self, level: u32) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len() as u32).filter(move |&n| self.nodes[n as usize].level == level)
    }

    /// Proxy triangles per level.
    pub fn level_triangles(&self) -> Vec<usize> {
        let mut t = vec![0; self.levels as usize];
        for n in &self.nodes {
            t[n.level as usize] += n.triangles;
        }
        t
    }
}

/// Builds the HLOD tree of `scene`. See the module docs.
pub fn build_hlod_tree(
    host: &mut dyn RenderHost,
    scene: &Scene,
    opts: &HlodTreeOptions,
) -> Result<HlodTree> {
    build_hlod_tree_with(host, scene, opts, |_, _| Ok(()))
}

/// [`build_hlod_tree`], handing each node to `on_node` (with its index) once the builder no
/// longer needs its proxy, in a deterministic order. The callback may take the proxy
/// (`node.proxy.take()`) to write it out and free it; everything else stays in the tree.
pub fn build_hlod_tree_with(
    host: &mut dyn RenderHost,
    scene: &Scene,
    opts: &HlodTreeOptions,
    mut on_node: impl FnMut(u32, &mut HlodNode) -> Result<()>,
) -> Result<HlodTree> {
    if !(opts.cell_size.is_finite() && opts.cell_size > 0.0) {
        return Err(Error::InvalidInput("cell_size must be positive".into()));
    }
    if !(opts.loose_factor > 0.0 && opts.triangle_scale > 0.0 && opts.parent_reduction > 0.0) {
        return Err(Error::InvalidInput(
            "loose_factor, triangle_scale and parent_reduction must be positive".into(),
        ));
    }
    if opts.proxy.bake.is_some() && opts.proxy.uv.is_none() {
        return Err(Error::InvalidInput(
            "an HLOD bake needs `uv` options".into(),
        ));
    }
    let quad = opts.grid == CellGrid::Quadtree;
    let flat = |v: Vec3| if quad { Vec3::new(v.x, 0.0, v.z) } else { v };

    // Instance bounds (mesh boxes transformed, corner by corner).
    let mesh_bounds: Vec<Aabb> = scene.meshes.iter().map(Mesh::bounds).collect();
    let mesh_tris: Vec<usize> = scene.meshes.iter().map(Mesh::triangle_count).collect();
    let placed: Vec<u32> = (0..scene.instances.len() as u32)
        .filter(|&i| mesh_tris[scene.instances[i as usize].mesh] > 0)
        .collect();
    if placed.is_empty() {
        return Err(Error::InvalidInput("HLOD of an empty scene".into()));
    }
    let inst_bounds: Vec<Aabb> = scene
        .instances
        .iter()
        .map(|inst| transform_box(&mesh_bounds[inst.mesh], inst.transform))
        .collect();
    let mut origin = Vec3::splat(f32::INFINITY);
    for &i in &placed {
        origin = origin.min(inst_bounds[i as usize].center());
    }
    let origin = flat(origin);

    // Leaf cell of each instance's centre, and the level whose cells fit it.
    let mut coord0 = vec![[0u32; 3]; scene.instances.len()];
    let mut level = vec![0u32; scene.instances.len()];
    for &i in &placed {
        let b = &inst_bounds[i as usize];
        let rel = (flat(b.center()) - origin) / opts.cell_size;
        if rel.max_element() >= (1u64 << 30) as f32 {
            return Err(Error::InvalidInput(
                "scene spans more than 2^30 cells; use a larger cell_size".into(),
            ));
        }
        coord0[i as usize] = rel.floor().max(Vec3::ZERO).as_uvec3().to_array();
        let extent = flat(b.extent()).max_element() / opts.loose_factor;
        let mut l = 0;
        while opts.cell_size * ((1u64 << l) as f32) < extent && l < 31 {
            l += 1;
        }
        level[i as usize] = l;
    }
    // Top level: everything in one cell (and every instance fits), unless capped.
    let mut top = placed.iter().map(|&i| level[i as usize]).max().unwrap_or(0);
    let max_c = placed.iter().fold([0u32; 3], |m, &i| {
        let c = coord0[i as usize];
        [m[0].max(c[0]), m[1].max(c[1]), m[2].max(c[2])]
    });
    while max_c.iter().any(|&c| c >> top != 0) {
        top += 1;
    }
    if let Some(cap) = opts.max_levels {
        top = top.min(cap.max(1) - 1);
    }
    let mut instance_node = vec![None; scene.instances.len()];

    let mut nodes: Vec<HlodNode> = Vec::new();
    // Nodes of the previous level, by their cell.
    let mut below: BTreeMap<[u32; 3], u32> = BTreeMap::new();
    let batch = match opts.parallel_cells {
        0 => rayon::current_num_threads().max(1),
        n => n,
    };
    for l in 0..=top {
        // Cells of this level: own instances and children.
        let mut cells: BTreeMap<[u32; 3], (Vec<u32>, Vec<u32>)> = BTreeMap::new();
        for &i in &placed {
            if level[i as usize].min(top) == l {
                let c = coord0[i as usize].map(|c| c >> l);
                cells.entry(c).or_default().0.push(i);
            }
        }
        for (&c, &n) in &below {
            cells.entry(c.map(|c| c >> 1)).or_default().1.push(n);
        }
        let first = nodes.len() as u32;
        for (cell, (own, children)) in cells {
            let n = nodes.len() as u32;
            let mut instances = own.clone();
            let mut bounds = Aabb::EMPTY;
            let mut source_triangles = 0;
            for &i in &own {
                instance_node[i as usize] = Some(n);
                bounds = bounds.union(&inst_bounds[i as usize]);
                source_triangles += mesh_tris[scene.instances[i as usize].mesh];
            }
            for &ch in &children {
                let child = &mut nodes[ch as usize];
                child.parent = Some(n);
                instances.extend_from_slice(&child.instances);
                bounds = bounds.union(&child.bounds);
                source_triangles += child.source_triangles;
            }
            instances.sort_unstable();
            nodes.push(HlodNode {
                level: l,
                cell,
                bounds,
                sphere: Sphere::default(),
                parent: None,
                children,
                own_instances: own,
                instances,
                proxy: None,
                material: None,
                triangles: 0,
                source_triangles,
                geometric_error: 0.0,
                texel_size: 0.0,
                error: 0.0,
            });
        }
        let level_nodes: Vec<u32> = (first..nodes.len() as u32).collect();
        for chunk in level_nodes.chunks(batch) {
            build_batch(host, scene, opts, l, chunk, &mut nodes)?;
        }
        // Children are no longer needed once their parents exist (from children), and
        // nothing is needed again when building from source.
        let done: Vec<u32> = match opts.parents {
            ParentSource::Children => level_nodes
                .iter()
                .flat_map(|&n| nodes[n as usize].children.clone())
                .collect(),
            ParentSource::Source => level_nodes.clone(),
        };
        for n in done {
            on_node(n, &mut nodes[n as usize])?;
        }
        below = level_nodes
            .iter()
            .map(|&n| (nodes[n as usize].cell, n))
            .collect();
    }
    let roots: Vec<u32> = (0..nodes.len() as u32)
        .filter(|&n| nodes[n as usize].parent.is_none())
        .collect();
    if opts.parents == ParentSource::Children {
        for &r in &roots {
            on_node(r, &mut nodes[r as usize])?;
        }
    }
    Ok(HlodTree {
        nodes,
        roots,
        levels: top + 1,
        cell_size: opts.cell_size,
        grid: opts.grid,
        origin,
        instance_node,
    })
}

/// The 8 corners of `b` through `m`.
fn transform_box(b: &Aabb, m: Mat4) -> Aabb {
    let mut out = Aabb::EMPTY;
    if b.is_empty() {
        return out;
    }
    for k in 0..8 {
        let p = Vec3::new(
            if k & 1 == 0 { b.min.x } else { b.max.x },
            if k & 2 == 0 { b.min.y } else { b.max.y },
            if k & 4 == 0 { b.min.z } else { b.max.z },
        );
        out.extend(m.transform_point3(p));
    }
    out
}

/// Proxy options for a node of `level`. `parent_input` is, for a parent, its children's proxy
/// triangles plus its own instances' triangles.
fn level_options(opts: &HlodTreeOptions, level: u32, parent_input: Option<usize>) -> HlodOptions {
    let scale = (1u64 << level) as f32;
    let mut target = opts.proxy.target_triangles.map(|t| {
        ((t as f64 * (opts.triangle_scale as f64).powi(level as i32)).round() as usize).max(1)
    });
    if let Some(input) = parent_input {
        let cap = ((input as f64 * opts.parent_reduction as f64).ceil() as usize).max(1);
        target = Some(target.map_or(cap, |t| t.min(cap)));
    }
    HlodOptions {
        target_triangles: target,
        max_error: opts.proxy.max_error.map(|e| e * scale),
        ..opts.proxy.clone()
    }
}

/// The input of one node: its own source instances plus its children's proxies (from
/// children) or every source instance below it (from source).
fn node_input(scene: &Scene, opts: &HlodTreeOptions, nodes: &[HlodNode], n: u32) -> Scene {
    let node = &nodes[n as usize];
    let from_source = opts.parents == ParentSource::Source || node.children.is_empty();
    let sources: &[u32] = if from_source {
        &node.instances
    } else {
        &node.own_instances
    };
    let mut input = Scene {
        materials: scene.materials.clone(),
        ..Default::default()
    };
    let mut mesh_map: HashMap<usize, usize> = HashMap::new();
    for &i in sources {
        let inst = &scene.instances[i as usize];
        let m = *mesh_map.entry(inst.mesh).or_insert_with(|| {
            input.meshes.push(scene.meshes[inst.mesh].clone());
            input.meshes.len() - 1
        });
        input.instances.push(Instance {
            name: inst.name.clone(),
            mesh: m,
            transform: inst.transform,
            skeleton: None,
        });
    }
    if !from_source {
        for &c in &node.children {
            let child = &nodes[c as usize];
            let Some(proxy) = &child.proxy else { continue };
            if proxy.mesh.triangle_count() == 0 {
                continue;
            }
            let slot = input.materials.len() as u32;
            input.materials.push(MaterialSlot {
                name: format!("hlod_{c}"),
                id: child.material.unwrap_or(PROXY_MATERIAL),
            });
            let mut mesh = proxy.mesh.clone();
            mesh.material_ids.iter_mut().for_each(|m| *m = slot);
            input.meshes.push(mesh);
            input.instances.push(Instance {
                name: format!("hlod_{c}"),
                mesh: input.meshes.len() - 1,
                transform: Mat4::IDENTITY,
                skeleton: None,
            });
        }
    }
    input
}

/// Builds the proxies of `batch` (nodes of one level): visibility through the host one node
/// at a time, geometry in parallel, bakes through the host one at a time.
fn build_batch(
    host: &mut dyn RenderHost,
    scene: &Scene,
    opts: &HlodTreeOptions,
    level: u32,
    batch: &[u32],
    nodes: &mut [HlodNode],
) -> Result<()> {
    // Hidden-surface removal over each node's whole input.
    let mut culled: Vec<(Culled, HlodOptions)> = Vec::with_capacity(batch.len());
    for &n in batch {
        let input = node_input(scene, opts, nodes, n);
        let node = &nodes[n as usize];
        let parent_input = (!node.children.is_empty()).then(|| {
            let own: usize = node
                .own_instances
                .iter()
                .map(|&i| scene.meshes[scene.instances[i as usize].mesh].triangle_count())
                .sum();
            own + node
                .children
                .iter()
                .map(|&c| nodes[c as usize].triangles)
                .sum::<usize>()
        });
        let lo = level_options(opts, level, parent_input);
        let c = cull_stage(host, &input, &lo)?;
        drop(input);
        culled.push((c, lo));
    }
    // Merge, reduce, unwrap.
    let shaped: Vec<Result<Option<super::Shaped>>> = culled
        .par_iter()
        .map(|(c, lo)| {
            if c.visible.triangle_count() == 0 {
                Ok(None)
            } else {
                shape_stage(&c.visible, lo).map(Some)
            }
        })
        .collect();
    for ((&n, (c, lo)), shaped) in batch.iter().zip(culled).zip(shaped) {
        let shaped = shaped?;
        let (mesh, step_error, texels_per_unit) = match shaped {
            Some(s) => (s.mesh, s.error, s.texels_per_unit),
            None => (Mesh::default(), 0.0, None),
        };
        let mut bake = if mesh.triangle_count() > 0 {
            bake_stage(host, &mesh, &c.visible, &lo)?
        } else {
            None
        };
        drop(c.visible);
        let node = &nodes[n as usize];
        let mut material = None;
        if let Some(b) = &mut bake {
            material = Some(host.create_material(&BakedMaterialDesc {
                name: format!(
                    "hlod_l{}_{}_{}_{}",
                    node.level, node.cell[0], node.cell[1], node.cell[2]
                ),
                kind: BakedMaterialKind::Surface,
                class: MaterialClass::Opaque,
                textures: b.textures.clone(),
                parameters: BTreeMap::new(),
            })?);
            if !opts.keep_correspondence {
                b.correspondence.entries = Vec::new();
            }
        }
        let child_geo = node
            .children
            .iter()
            .map(|&ch| nodes[ch as usize].geometric_error)
            .fold(0.0f32, f32::max);
        let child_err = node
            .children
            .iter()
            .map(|&ch| nodes[ch as usize].error)
            .fold(0.0f32, f32::max);
        let geometric_error = match opts.parents {
            ParentSource::Children => step_error + child_geo,
            ParentSource::Source => step_error.max(child_geo),
        };
        let texel_size = texels_per_unit.map_or(0.0, |t| if t > 0.0 { 1.0 / t } else { 0.0 });
        let mut error = geometric_error.max(child_err);
        if opts.texel_error {
            error = error.max(texel_size);
        }
        let node = &mut nodes[n as usize];
        if mesh.triangle_count() > 0 {
            node.bounds = node.bounds.union(&mesh.bounds());
        }
        node.sphere = Sphere {
            center: node.bounds.center(),
            radius: node.bounds.radius(),
        };
        node.triangles = mesh.triangle_count();
        node.geometric_error = geometric_error;
        node.texel_size = texel_size;
        node.error = error;
        node.material = material;
        node.proxy = Some(HlodResult {
            mesh,
            bake,
            input_triangles: c.input_triangles,
            hidden_triangles: c.hidden_triangles,
            error: step_error,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::GeometryOnlyHost;
    use crate::visibility::{CameraSet, VisibilityOptions};

    /// An `n × n` grid of subdivided boxes on the XZ plane, `spacing` apart, plus an
    /// optional ground slab spanning them.
    fn grid(n: usize, spacing: f32, ground: bool) -> Scene {
        let mut s = Scene {
            meshes: vec![super::super::tests::cube(4)],
            materials: vec![MaterialSlot {
                name: "m".into(),
                id: MaterialId(0),
            }],
            ..Default::default()
        };
        for j in 0..n {
            for i in 0..n {
                s.instances.push(Instance {
                    name: format!("box_{i}_{j}"),
                    mesh: 0,
                    transform: Mat4::from_translation(Vec3::new(
                        i as f32 * spacing + 0.5 * spacing,
                        1.0,
                        j as f32 * spacing + 0.5 * spacing,
                    )),
                    skeleton: None,
                });
            }
        }
        if ground {
            let half = n as f32 * spacing * 0.5;
            s.instances.push(Instance {
                name: "ground".into(),
                mesh: 0,
                transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(half, 0.1, half),
                    Default::default(),
                    Vec3::new(half, -0.1, half),
                ),
                skeleton: None,
            });
        }
        s
    }

    fn opts(cell: f32) -> HlodTreeOptions {
        HlodTreeOptions {
            cell_size: cell,
            grid: CellGrid::Quadtree,
            proxy: HlodOptions {
                visibility: Some(VisibilityOptions {
                    cameras: CameraSet::Hemisphere {
                        count: 16,
                        distance_factor: 2.5,
                    },
                    resolution: 64,
                }),
                target_triangles: Some(150),
                uv: None,
                bake: None,
                ..HlodOptions::default()
            },
            ..HlodTreeOptions::default()
        }
    }

    /// Every instance with triangles covered exactly once.
    fn assert_valid_cut(tree: &HlodTree, cut: &HlodCut, instances: usize) {
        let mut seen = vec![0u32; instances];
        for &n in &cut.nodes {
            for &i in &tree.nodes[n as usize].instances {
                seen[i as usize] += 1;
            }
        }
        for &i in &cut.instances {
            seen[i as usize] += 1;
        }
        for (i, &c) in seen.iter().enumerate() {
            let expected = tree.instance_node[i].is_some() as u32;
            assert_eq!(c, expected, "instance {i} covered {c} times");
        }
    }

    #[test]
    fn grid_of_boxes_makes_the_expected_cells() {
        // 8 × 8 boxes, 4 apart, in 8-unit cells: 2 × 2 boxes per leaf.
        let s = grid(8, 4.0, false);
        let tree = build_hlod_tree(&mut GeometryOnlyHost, &s, &opts(8.0)).unwrap();
        let per_level: Vec<usize> = (0..tree.levels).map(|l| tree.level(l).count()).collect();
        assert_eq!(per_level, vec![16, 4, 1], "{per_level:?}");
        assert_eq!(tree.roots.len(), 1);
        for n in tree.level(0) {
            assert_eq!(tree.nodes[n as usize].instances.len(), 4);
        }
        let root = &tree.nodes[tree.roots[0] as usize];
        assert_eq!(root.instances, (0..64).collect::<Vec<u32>>());
        // Children come before parents, and every parent replaces exactly its children.
        for (i, n) in tree.nodes.iter().enumerate() {
            let mut from_children: Vec<u32> = n.own_instances.clone();
            for &c in &n.children {
                assert!(c < i as u32);
                assert_eq!(tree.nodes[c as usize].parent, Some(i as u32));
                from_children.extend(&tree.nodes[c as usize].instances);
            }
            from_children.sort_unstable();
            assert_eq!(from_children, n.instances);
        }
    }

    #[test]
    fn parents_are_coarser_than_their_children() {
        let s = grid(8, 4.0, false);
        let tree = build_hlod_tree(&mut GeometryOnlyHost, &s, &opts(8.0)).unwrap();
        for n in &tree.nodes {
            if n.children.is_empty() {
                continue;
            }
            let sum: usize = n
                .children
                .iter()
                .map(|&c| tree.nodes[c as usize].triangles)
                .sum();
            assert!(n.triangles < sum, "{} vs children {sum}", n.triangles);
            for &c in &n.children {
                assert!(n.error >= tree.nodes[c as usize].error);
            }
            assert!(n.error > 0.0);
        }
        // Budgets hold per level (150 × 1^level, at most half the input for parents).
        for n in &tree.nodes {
            assert!(n.triangles <= 150, "{}", n.triangles);
        }
    }

    #[test]
    fn large_instances_go_up_and_selection_is_a_valid_cut() {
        let s = grid(8, 4.0, true);
        let ground = s.instances.len() as u32 - 1;
        let o = opts(8.0);
        let tree = build_hlod_tree(&mut GeometryOnlyHost, &s, &o).unwrap();
        // The 32-unit slab fits only the 32-unit root cell.
        let at = tree.instance_node[ground as usize].unwrap();
        assert_eq!(tree.nodes[at as usize].level, 2);
        assert_eq!(tree.roots, vec![at]);
        let mut kinds = std::collections::BTreeSet::new();
        for d in [0.5f32, 5.0, 20.0, 60.0, 200.0, 1e4] {
            for eye in [
                Vec3::new(16.0, 2.0, 16.0),
                Vec3::new(-d, d * 0.3, 16.0),
                Vec3::new(2.0, 1.0, 2.0 - d),
            ] {
                let view = LodView {
                    eye,
                    pixels_per_radian: 1000.0,
                    threshold: 1.0,
                };
                let cut = tree.select(&view);
                assert_valid_cut(&tree, &cut, s.instances.len());
                // Selection agrees with the switch distances.
                for &n in &cut.nodes {
                    let node = &tree.nodes[n as usize];
                    assert!(node.sphere.center.distance(eye) >= node.switch_distance(1000.0, 1.0));
                }
                kinds.insert((cut.nodes.len(), cut.instances.len()));
            }
        }
        // Far away only the root is drawn; up close the cells around the eye are refined
        // (the flat-faced leaf proxies are lossless, so the rest stay proxies).
        assert!(kinds.iter().any(|&(n, i)| n > 1 && i > 0), "{kinds:?}");
        assert!(kinds.contains(&(1, 0)));
        assert!(kinds.len() > 2, "{kinds:?}");
    }

    #[test]
    fn deterministic() {
        let s = grid(6, 3.0, true);
        for parents in [ParentSource::Children, ParentSource::Source] {
            let o = HlodTreeOptions {
                parents,
                grid: CellGrid::Octree,
                ..opts(4.0)
            };
            let a = build_hlod_tree(&mut GeometryOnlyHost, &s, &o).unwrap();
            let b = rayon::ThreadPoolBuilder::new()
                .num_threads(3)
                .build()
                .unwrap()
                .install(|| {
                    let o = HlodTreeOptions {
                        parallel_cells: 2,
                        ..o.clone()
                    };
                    build_hlod_tree(&mut GeometryOnlyHost, &s, &o).unwrap()
                });
            assert_eq!(a.nodes.len(), b.nodes.len());
            for (x, y) in a.nodes.iter().zip(&b.nodes) {
                assert_eq!(x.instances, y.instances);
                assert_eq!(x.error.to_bits(), y.error.to_bits());
                let (px, py) = (x.proxy.as_ref().unwrap(), y.proxy.as_ref().unwrap());
                assert_eq!(px.mesh.positions, py.mesh.positions);
                assert_eq!(px.mesh.indices, py.mesh.indices);
            }
        }
    }

    #[test]
    fn callback_gets_every_node_once_after_its_parent() {
        let s = grid(8, 4.0, false);
        let mut order = Vec::new();
        let mut parents_built = std::collections::BTreeSet::new();
        let tree = build_hlod_tree_with(&mut GeometryOnlyHost, &s, &opts(8.0), |n, node| {
            if let Some(p) = node.parent {
                parents_built.insert(p);
            }
            assert!(node.proxy.take().is_some());
            order.push(n);
            Ok(())
        })
        .unwrap();
        order.sort_unstable();
        assert_eq!(order, (0..tree.nodes.len() as u32).collect::<Vec<_>>());
        assert!(tree
            .nodes
            .iter()
            .all(|n| n.proxy.is_none() && n.triangles > 0));
    }

    #[test]
    fn hidden_surfaces_go_across_the_whole_cell() {
        // Two boxes pressed face to face in one cell: the touching faces vanish only when
        // both are culled together.
        let mut s = grid(1, 4.0, false);
        let mut second = s.instances[0].clone();
        second.transform = Mat4::from_translation(Vec3::new(4.0, 1.0, 2.0));
        s.instances.push(second);
        let mut o = opts(16.0);
        o.proxy.target_triangles = Some(10_000);
        o.proxy.grow_rings = 0;
        let tree = build_hlod_tree(&mut GeometryOnlyHost, &s, &o).unwrap();
        assert_eq!(tree.nodes.len(), 1);
        let p = tree.nodes[0].proxy.as_ref().unwrap();
        // The touching faces (16 triangles each) and the undersides would not be seen.
        assert!(p.hidden_triangles >= 32, "{}", p.hidden_triangles);
    }

    #[test]
    fn bakes_every_node_and_parents_from_their_children_materials() {
        let s = grid(4, 4.0, false);
        let mut o = opts(8.0);
        o.proxy.uv = Some(crate::uv::ParamOptions {
            atlas_size: 64,
            padding: 2,
            ..Default::default()
        });
        o.proxy.bake = Some(crate::bake::BakeOptions {
            width: 64,
            height: 64,
            supersample: 1,
            gutter: 2,
            channels: vec![crate::host::channels::BASE_COLOR.into()],
            ..Default::default()
        });
        let tree = build_hlod_tree(&mut super::super::tests::PositionHost, &s, &o).unwrap();
        assert_eq!(tree.levels, 2);
        for n in &tree.nodes {
            let p = n.proxy.as_ref().unwrap();
            let b = p.bake.as_ref().expect("baked");
            assert!(b.coverage > 0.0 && b.correspondence.entries.is_empty());
            assert_eq!(n.material, Some(MaterialId(1)));
            assert!(n.texel_size > 0.0 && n.error >= n.texel_size);
        }
    }
}
