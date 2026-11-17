//! farthing's proof-of-concept engine.
//!
//! - [`engine`] (feature `native`): a headless wgpu renderer with its own material system that
//!   implements farthing's `RenderHost`, so farthing can bake, build impostors and score against
//!   it without ever seeing its materials.
//! - [`viewer`]: the interactive renderer the web demo draws with. It owns no window: it draws
//!   into whatever texture view it's given, a canvas surface in the browser (feature `web`) or an
//!   offscreen target natively.
//! - [`editor_visualizations`]: the overlays the viewer draws over a model to show what farthing
//!   did (wireframe, UV seams).
//! - [`orbit`]: the camera the web demo turns, pans and dollies with the pointer.
//!
//! It uses only farthing's public API.

pub mod editor_visualizations;
pub mod orbit;
pub mod viewer;

#[cfg(feature = "native")]
pub mod engine;
#[cfg(feature = "native")]
pub mod scene_gen;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web;
