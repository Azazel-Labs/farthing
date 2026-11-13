//! Material baking: transfer a source surface's appearance onto a proxy's UVs.
//!
//! Material casting without evaluating materials:
//!
//! 1. Rasterize the destination mesh in UV space ([`raster`]).
//! 2. Ray-cast each sample to the source surface ([`correspondence`]) — the reusable,
//!    serializable "where did this texel come from" buffer.
//! 3. Either ask the host to evaluate its real materials at those points
//!    ([`bake_with_host`], point-evaluation mode) or move texels from host-flattened textures
//!    ([`bake_flattened`], for hosts that pre-flatten materials to textures).
//! 4. Convert world-space normals to the destination's MikkTSpace frame, resolve
//!    supersampling and dilate into gutters.
//!
//! Displacement (signed distance along the normal) is pure geometry and is produced here.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

mod cast;
pub mod correspondence;
pub mod raster;
pub mod tangents;

pub use cast::{
    bake_flattened, bake_with_host, finish_bake, prepare_bake, BakeOptions, BakeOutput,
    FlattenedTextures, PreparedBake, DISPLACEMENT,
};
pub use correspondence::{build_correspondence, Correspondence, CorrespondenceEntry};
pub use tangents::compute_tangents;
