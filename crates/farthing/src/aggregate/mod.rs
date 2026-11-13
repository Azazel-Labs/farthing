//! Scene aggregation: merge instances into one mesh with a shared, repacked atlas.
//!
//! Merges every instance of a scene into one mesh with one shared UV atlas, so a cluster of
//! props becomes one draw call per material class (opaque / masked / blended).
//!
//! Chart repack mode: each (instance, material slot) UV island keeps its shape, is sized by
//! world-space texel density, and is packed into the atlas. Texels are then *copied* from the
//! host's flattened textures through each chart's 2D transform — a resample, not a ray cast.
//! Tiling UVs work: the repeated texture is baked into a chart sized by world area.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

use glam::{Vec2, Vec4};
use crate::bake::FlattenedTextures;
use crate::host::{MaterialClass, MaterialInfo};
use crate::image::Image;
use crate::math::triangle_area;
use crate::{Error, MaterialId, MaterialSlot, Mesh, Result, Scene};
use crate::uv::pack_rects;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggregateOptions {
    pub atlas_size: u32,
    pub padding: u32,
    /// Texels per world unit; None = as large as fits.
    pub texels_per_unit: Option<f32>,
    /// Optional per-triangle weights in [`Scene::flatten`] order (e.g. visibility); chart
    /// texel area scales with the chart's mean weight.
    #[serde(skip)]
    pub triangle_weights: Option<Vec<f32>>,
}

impl Default for AggregateOptions {
    fn default() -> Self {
        AggregateOptions {
            atlas_size: 2048,
            padding: 4,
            texels_per_unit: None,
            triangle_weights: None,
        }
    }
}

/// Where one source UV island landed in the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChartPlacement {
    pub instance: u32,
    pub slot: u32,
    /// Source UV rectangle of the island.
    pub src_min: Vec2,
    pub src_max: Vec2,
    /// Destination pixel rectangle.
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub rotated: bool,
}

#[derive(Clone, Debug)]
pub struct AggregateResult {
    /// Merged scene: one mesh, one instance at identity, one material slot per class used.
    pub scene: Scene,
    /// Material class of each output slot.
    pub slot_classes: Vec<MaterialClass>,
    pub placements: Vec<ChartPlacement>,
    pub atlas_size: u32,
    pub texels_per_unit: f32,
    pub draw_calls_before: usize,
}

fn class_key(c: MaterialClass) -> u8 {
    match c {
        MaterialClass::Opaque => 0,
        MaterialClass::Masked { .. } => 1,
        MaterialClass::Blended => 2,
    }
}

/// Builds the merged mesh and atlas layout. `infos[slot]` gives each source slot's class.
pub fn aggregate(
    scene: &Scene,
    infos: &[MaterialInfo],
    opts: &AggregateOptions,
) -> Result<AggregateResult> {
    let mut merged = Mesh::default();
    struct Island {
        instance: u32,
        slot: u32,
        verts: Vec<u32>, // indices into merged
        uv_min: Vec2,
        uv_max: Vec2,
        world_area: f32,
        weight: f32,
        tris: u32,
    }
    let mut islands: Vec<Island> = Vec::new();
    let mut draw_calls_before = 0;
    let mut flat_tri = 0usize;
    let mut out_mats: Vec<u32> = Vec::new();
    let mut class_slots: Vec<MaterialClass> = Vec::new();
    for (ii, inst) in scene.instances.iter().enumerate() {
        let src = &scene.meshes[inst.mesh];
        let mut m = src.clone();
        m.transform(inst.transform);
        if m.uvs.is_empty() {
            return Err(Error::InvalidMesh(format!(
                "instance `{}` has no UVs; parameterize it first",
                inst.name
            )));
        }
        let mut slots: Vec<u32> = m.material_ids.clone();
        slots.sort();
        slots.dedup();
        draw_calls_before += slots.len();
        // Duplicate vertices per slot so each island owns its vertices.
        let mut vkey: HashMap<(u32, u32), u32> = HashMap::new();
        let mut order: Vec<u32> = Vec::new();
        let mut idx = Vec::with_capacity(m.indices.len());
        for t in 0..m.triangle_count() {
            let slot = m.material_ids[t];
            for v in m.triangle(t) {
                let id = *vkey.entry((v, slot)).or_insert_with(|| {
                    order.push(v);
                    order.len() as u32 - 1
                });
                idx.push(id);
            }
        }
        let mut part = m.select_vertices(&order);
        part.indices = idx;
        part.material_ids = m.material_ids.clone();
        // Islands: union-find over triangles sharing a vertex.
        let nv = part.vertex_count();
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for t in 0..part.triangle_count() {
            let [a, b, c] = part.triangle(t);
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            parent[rb as usize] = ra;
            let rc = find(&mut parent, c);
            parent[rc as usize] = ra;
        }
        let base = merged.vertex_count() as u32;
        let mut root_island: HashMap<u32, usize> = HashMap::new();
        for t in 0..part.triangle_count() {
            let tri = part.triangle(t);
            let root = find(&mut parent, tri[0]);
            let slot = part.material_ids[t];
            let isl = *root_island.entry(root).or_insert_with(|| {
                islands.push(Island {
                    instance: ii as u32,
                    slot,
                    verts: Vec::new(),
                    uv_min: Vec2::INFINITY,
                    uv_max: Vec2::NEG_INFINITY,
                    world_area: 0.0,
                    weight: 0.0,
                    tris: 0,
                });
                islands.len() - 1
            });
            let [pa, pb, pc] = part.triangle_positions(t);
            islands[isl].world_area += triangle_area(pa, pb, pc);
            islands[isl].tris += 1;
            islands[isl].weight += opts
                .triangle_weights
                .as_ref()
                .and_then(|w| w.get(flat_tri + t).copied())
                .unwrap_or(1.0);
            for v in tri {
                let uv = part.uvs[0][v as usize];
                islands[isl].uv_min = islands[isl].uv_min.min(uv);
                islands[isl].uv_max = islands[isl].uv_max.max(uv);
            }
        }
        for v in 0..nv as u32 {
            let root = find(&mut parent, v);
            if let Some(&isl) = root_island.get(&root) {
                islands[isl].verts.push(base + v);
            }
        }
        // Output slot per material class.
        for t in 0..part.triangle_count() {
            let class = infos
                .get(part.material_ids[t] as usize)
                .map_or(MaterialClass::Opaque, |i| i.class);
            let pos = class_slots
                .iter()
                .position(|c| class_key(*c) == class_key(class))
                .unwrap_or_else(|| {
                    class_slots.push(class);
                    class_slots.len() - 1
                });
            out_mats.push(pos as u32);
        }
        flat_tri += part.triangle_count();
        merged.append(&part, 0);
    }
    merged.material_ids = out_mats;
    merged.uvs.truncate(1);

    // Size islands by world texel density, fit the atlas.
    let atlas = opts.atlas_size;
    let world_sizes: Vec<Vec2> = islands
        .iter()
        .map(|i| {
            let uv_ext = (i.uv_max - i.uv_min).max(Vec2::splat(1e-6));
            let mean_w = (i.weight / i.tris.max(1) as f32).clamp(0.05, 20.0);
            // World length per UV unit, assuming the island fills ~half its UV rect.
            let k = (i.world_area / (uv_ext.x * uv_ext.y * 0.5)).sqrt() * mean_w.sqrt();
            uv_ext * k
        })
        .collect();
    let total: f32 = world_sizes.iter().map(|s| s.x * s.y).sum();
    let mut density = opts
        .texels_per_unit
        .unwrap_or_else(|| (0.75 * (atlas * atlas) as f32 / total.max(1e-12)).sqrt());
    let mut packed = None;
    for _ in 0..60 {
        let sizes: Vec<(u32, u32)> = world_sizes
            .iter()
            .map(|s| {
                (
                    ((s.x * density).ceil() as u32).clamp(1, atlas),
                    ((s.y * density).ceil() as u32).clamp(1, atlas),
                )
            })
            .collect();
        if let Some(p) = pack_rects(&sizes, atlas, atlas, opts.padding, true) {
            packed = Some(p);
            break;
        }
        density *= 0.92;
    }
    let packed =
        packed.ok_or_else(|| Error::InvalidInput("islands do not fit the atlas".into()))?;
    let mut placements = Vec::with_capacity(islands.len());
    let uvs = &mut merged.uvs[0];
    for (isl, pl) in islands.iter().zip(&packed) {
        let ext = (isl.uv_max - isl.uv_min).max(Vec2::splat(1e-6));
        for &v in &isl.verts {
            let local = (uvs[v as usize] - isl.uv_min) / ext; // 0..1 within the island
            let l = if pl.rotated {
                Vec2::new(local.y, local.x)
            } else {
                local
            };
            uvs[v as usize] = (Vec2::new(pl.x as f32, pl.y as f32)
                + l * Vec2::new(pl.w as f32, pl.h as f32))
                / atlas as f32;
        }
        placements.push(ChartPlacement {
            instance: isl.instance,
            slot: isl.slot,
            src_min: isl.uv_min,
            src_max: isl.uv_max,
            x: pl.x,
            y: pl.y,
            w: pl.w,
            h: pl.h,
            rotated: pl.rotated,
        });
    }
    merged.tangents = None;
    let materials = class_slots
        .iter()
        .enumerate()
        .map(|(i, c)| MaterialSlot {
            name: format!(
                "aggregate_{}",
                ["opaque", "masked", "blended"][class_key(*c) as usize]
            ),
            id: MaterialId(i as u32),
        })
        .collect();
    Ok(AggregateResult {
        scene: Scene {
            meshes: vec![merged],
            materials,
            instances: vec![crate::Instance {
                name: "aggregate".into(),
                mesh: 0,
                transform: glam::Mat4::IDENTITY,
                skeleton: None,
            }],
            skeletons: Vec::new(),
        },
        slot_classes: class_slots,
        placements,
        atlas_size: atlas,
        texels_per_unit: density,
        draw_calls_before,
    })
}

/// Copies one channel of host-flattened textures into the atlas, then dilates by `padding`.
pub fn copy_texels(
    result: &AggregateResult,
    flattened: &FlattenedTextures,
    channel: &str,
) -> Result<Image> {
    let mut atlas = Image::new(result.atlas_size, result.atlas_size, Vec4::ZERO);
    for p in &result.placements {
        let Some(tex) = flattened.get(channel, p.instance, p.slot) else {
            return Err(Error::InvalidInput(format!(
                "no flattened `{channel}` texture for instance {} slot {}",
                p.instance, p.slot
            )));
        };
        for y in 0..p.h {
            for x in 0..p.w {
                let mut l = Vec2::new((x as f32 + 0.5) / p.w as f32, (y as f32 + 0.5) / p.h as f32);
                if p.rotated {
                    l = Vec2::new(l.y, l.x);
                }
                let uv = p.src_min + (p.src_max - p.src_min) * l;
                let mut v = tex.sample_bilinear(uv, true);
                v.w = 1.0;
                atlas.set(p.x + x, p.y + y, v);
            }
        }
    }
    atlas.dilate(4);
    Ok(atlas)
}

/// Like [`copy_texels`] for tangent-space normal maps: charts the packer transposed get their
/// X and Y swapped so the normals stay correct under the new UV axes (the engine recomputes
/// MikkTSpace tangents from the new UVs).
pub fn copy_tangent_normals(
    result: &AggregateResult,
    flattened: &FlattenedTextures,
    channel: &str,
) -> Result<Image> {
    let mut atlas = copy_texels(result, flattened, channel)?;
    for p in &result.placements {
        if !p.rotated {
            continue;
        }
        for y in p.y..p.y + p.h {
            for x in p.x..p.x + p.w {
                let v = atlas.get(x, y);
                atlas.set(x, y, Vec4::new(v.y, v.x, v.z, v.w));
            }
        }
    }
    for px in &mut atlas.pixels {
        if px.w > 0.0 {
            *px = px.truncate().normalize_or_zero().extend(px.w);
        }
    }
    Ok(atlas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};
    use crate::host::ChannelEncoding;
    use crate::Instance;

    fn quad() -> Mesh {
        let mut m = Mesh::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
            vec![0, 1, 2, 0, 2, 3],
        );
        m.uvs = vec![vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ]];
        m
    }

    #[test]
    fn two_instances_two_materials_become_one_drawcall() {
        let mut a = quad();
        a.material_ids = vec![0, 1];
        let mut s = Scene::from_mesh(a);
        s.instances.push(Instance {
            name: "b".into(),
            mesh: 0,
            transform: Mat4::from_translation(Vec3::X * 3.0),
            skeleton: None,
        });
        let r = aggregate(
            &s,
            &[MaterialInfo::default(); 2],
            &AggregateOptions {
                atlas_size: 256,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.draw_calls_before, 4);
        assert_eq!(r.scene.materials.len(), 1);
        assert_eq!(r.placements.len(), 4);
        let m = &r.scene.meshes[0];
        m.validate().unwrap();
        assert!(m.uvs[0]
            .iter()
            .all(|u| (0.0..=1.0).contains(&u.x) && (0.0..=1.0).contains(&u.y)));
        // Copy texels: slot 0 red, slot 1 green.
        let mut flat = FlattenedTextures::default();
        flat.insert(
            "base_color",
            ChannelEncoding::Color,
            None,
            0,
            Image::new(8, 8, Vec4::new(1.0, 0.0, 0.0, 1.0)),
        );
        flat.insert(
            "base_color",
            ChannelEncoding::Color,
            None,
            1,
            Image::new(8, 8, Vec4::new(0.0, 1.0, 0.0, 1.0)),
        );
        let atlas = copy_texels(&r, &flat, "base_color").unwrap();
        let p = r.placements.iter().find(|p| p.slot == 1).unwrap();
        assert_eq!(
            atlas.get(p.x + p.w / 2, p.y + p.h / 2),
            Vec4::new(0.0, 1.0, 0.0, 1.0)
        );
    }

    #[test]
    fn masked_material_gets_its_own_slot() {
        let mut a = quad();
        a.material_ids = vec![0, 1];
        let s = Scene::from_mesh(a);
        let infos = [
            MaterialInfo::default(),
            MaterialInfo {
                class: MaterialClass::Masked { cutoff: 0.5 },
                two_sided: true,
            },
        ];
        let r = aggregate(
            &s,
            &infos,
            &AggregateOptions {
                atlas_size: 128,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.scene.materials.len(), 2);
    }
}
