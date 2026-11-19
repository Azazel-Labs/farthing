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
//! - `clusters`: the viewer's GPU-driven cluster renderer: a compute LOD cut with frustum and
//!   cone culling, then indirect draws decoding vertices straight from the streamed pages.
//! - `crowd`: the viewer's GPU crowds: thousands of skinned units animated from one farthing
//!   pose palette, culled and given a level of detail by a compute pass, then drawn with one
//!   indirect instanced draw per level. The CPU writes a unit only when its state changes.
//! - [`orbit`]: the camera the web demo turns, pans and dollies with the pointer.
//!
//! It uses only farthing's public API.

mod clusters;
mod crowd;
pub mod editor_visualizations;
pub mod orbit;
pub mod viewer;

pub use clusters::{ClusterStats, ClusterViewStats, STAT_LEVELS};
pub use crowd::{CrowdLod, CrowdLodKind, CrowdStats, CrowdUnit, MAX_LODS as CROWD_MAX_LODS};

#[cfg(feature = "native")]
pub mod engine;
#[cfg(feature = "native")]
pub mod scene_gen;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web;
