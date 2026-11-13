//! CPU reference renderer for triangle-ID buffers (the default [`crate::host::RenderHost::render_ids`]).

use crate::bvh::Bvh;
use crate::host::{IdImage, MaterialClass, MaterialInfo, View};
use crate::scene::{Scene, TriRef};
use rayon::prelude::*;

/// Casts one ray per pixel. Blended materials are skipped (they never occlude).
pub fn cpu_render_ids(scene: &Scene, views: &[View], infos: &[MaterialInfo]) -> Vec<IdImage> {
    let (mut tris, mut refs) = scene.world_triangles();
    // Drop triangles whose material never occludes.
    let keep: Vec<bool> = refs
        .iter()
        .map(|r| {
            let inst = &scene.instances[r.instance as usize];
            let slot = scene.meshes[inst.mesh].material_ids[r.triangle as usize] as usize;
            !matches!(
                infos.get(slot).map(|i| i.class),
                Some(MaterialClass::Blended)
            )
        })
        .collect();
    if keep.iter().any(|k| !k) {
        let mut i = 0;
        tris.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let mut j = 0;
        refs.retain(|_| {
            j += 1;
            keep[j - 1]
        });
    }
    let bvh = Bvh::build(tris);
    views
        .iter()
        .map(|v| {
            let ids: Vec<Option<TriRef>> = (0..v.width * v.height)
                .into_par_iter()
                .map(|i| {
                    let (o, d) = v.ray(i % v.width, i / v.width);
                    bvh.intersect(o, d, 0.0, f32::INFINITY)
                        .map(|h| refs[h.triangle as usize])
                })
                .collect();
            IdImage {
                width: v.width,
                height: v.height,
                ids,
            }
        })
        .collect()
}
