//! Mesh reduction: triangle and quad simplification, LOD chains, repair and vertex-cache ordering.
//!
//! * [`simplify`] — half-edge collapse driven by quadric error metrics
//!   (Garland & Heckbert 1997) plus attribute penalties for normals, UVs, colors, skin weights
//!   and morph deltas. Seams are preserved through wedge mapping; borders through constraint
//!   planes and border-only collapses.
//! * [`lod_chain`] — successive reductions with per-level error, for LOD switch distances.
//! * [`repair`] — welding, degenerate/duplicate triangle removal.
//! * [`optimize`] — vertex-cache (Forsyth) and vertex-fetch ordering, ACMR analysis.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

mod fast;
mod kernels;
pub mod optimize;
mod parts;
pub(crate) use parts::parts_of;
pub mod quad;
mod quadric;
pub mod repair;
mod simplify;
pub mod symmetry;

pub use quad::{reduce_quads, QuadReduceOptions, QuadReduceResult};
pub use simplify::{
    lod_chain, screen_size_to_max_error, simplify, simplify_scene, CollapseOrder, LodLevel,
    Quality, SeamMode, SimplifyOptions, SimplifyResult, StopWhen,
};
pub use symmetry::{detect_symmetry, Symmetry};
