//! Triangle-ID renders on the GPU for farthing's visibility pass (HLOD hidden-surface
//! removal). In the browser they go to WebGL2 in the worker (`web/gpu.js`); natively, or when
//! WebGL2 is missing or fails, [`render_ids`] returns `None` and the host falls back to
//! farthing's CPU ray caster. The GPU result can differ from the CPU one by the odd pixel
//! along triangle edges (rasterization rules vs ray hits), so HLOD output then depends on the
//! GPU; the library's own output stays deterministic.

use farthing::host::{IdImage, IdRenderRequest, MaterialClass, MaterialInfo};

/// Milliseconds since some fixed point, for timing.
pub(crate) fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1e3
    }
}

/// The GPU's name when GPU renders are available.
pub(crate) fn gpu_name() -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        imp::gpu_name()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

/// `req` rendered on the GPU, or `None` to use the CPU. Triangles of blended materials never
/// occlude and are left out, as in farthing's CPU caster.
pub(crate) fn render_ids(req: &IdRenderRequest, infos: &[MaterialInfo]) -> Option<Vec<IdImage>> {
    let first = req.views.first()?;
    let (w, h) = (first.width, first.height);
    if req.views.iter().any(|v| v.width != w || v.height != h) {
        return None;
    }
    let (tris, refs) = req.scene.world_triangles();
    let mut flat = Vec::with_capacity(tris.len() * 9);
    let mut kept = Vec::with_capacity(refs.len());
    for (t, r) in tris.iter().zip(&refs) {
        let inst = &req.scene.instances[r.instance as usize];
        let slot = req.scene.meshes[inst.mesh].material_ids[r.triangle as usize] as usize;
        if matches!(
            infos.get(slot).map(|i| i.class),
            Some(MaterialClass::Blended)
        ) {
            continue;
        }
        flat.extend(t.iter().flat_map(|p| p.to_array()));
        kept.push(*r);
    }
    drop((tris, refs));
    let view_proj: Vec<f32> = req
        .views
        .iter()
        .flat_map(|v| v.view_proj().to_cols_array())
        .collect();
    let n = (w * h) as usize;
    let images = gpu_ids(&flat, &view_proj, w, h, req.views.len(), |ids| {
        ids.iter()
            .map(|&k| {
                (k as usize)
                    .checked_sub(1)
                    .and_then(|k| kept.get(k).copied())
            })
            .collect()
    })?;
    Some(
        images
            .into_iter()
            .map(|ids: Vec<_>| {
                debug_assert_eq!(ids.len(), n);
                IdImage {
                    width: w,
                    height: h,
                    ids,
                }
            })
            .collect(),
    )
}

/// Renders the views and maps each view's raw ids (1-based triangle index, 0 = none) with
/// `map`, one view at a time so the raw readback is never copied whole into wasm memory.
#[cfg(target_arch = "wasm32")]
fn gpu_ids<T>(
    tris: &[f32],
    view_proj: &[f32],
    w: u32,
    h: u32,
    views: usize,
    mut map: impl FnMut(&[u32]) -> T,
) -> Option<Vec<T>> {
    let all = imp::gpu_render_ids(tris, view_proj, w, h);
    if all.is_undefined() || all.is_null() {
        return None;
    }
    let all = js_sys::Uint32Array::from(all);
    let n = w * h;
    if all.length() != n * views as u32 {
        return None;
    }
    let mut buf = vec![0u32; n as usize];
    Some(
        (0..views as u32)
            .map(|i| {
                all.subarray(i * n, (i + 1) * n).copy_to(&mut buf);
                map(&buf)
            })
            .collect(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn gpu_ids<T>(
    _tris: &[f32],
    _view_proj: &[f32],
    _w: u32,
    _h: u32,
    _views: usize,
    _map: impl FnMut(&[u32]) -> T,
) -> Option<Vec<T>> {
    None
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(module = "/web/gpu.js")]
    extern "C" {
        #[wasm_bindgen(js_name = gpuRenderIds)]
        pub fn gpu_render_ids(tris: &[f32], view_proj: &[f32], width: u32, height: u32) -> JsValue;
        #[wasm_bindgen(js_name = gpuName)]
        pub fn gpu_name() -> Option<String>;
    }
}
