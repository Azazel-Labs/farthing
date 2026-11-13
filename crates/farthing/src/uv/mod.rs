//! UV parameterization (LSCM, ARAP, planar) and atlas packing.
//!
//! * [`pack`] — the atlas packer (the `snugly` crate, re-exported): rectangles and
//!   irregular chart outlines, shared by parameterization, aggregation and impostor atlases.
//! * [`parameterize`] — new non-overlapping UVs: charts by normal-cone region growing,
//!   PCA-aligned planar projection, uniform texel density, packed into one atlas.
//!
//! Charts are flattened with least squares conformal maps (Lévy et al. 2002, [`lscm`]) and
//! fall back to planar projection when a chart would fold or overlap.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

pub mod lscm;
/// The atlas packer, now its own crate ([`snugly`]); re-exported here.
pub use snugly as pack;
mod param;
pub mod sparse;

pub use pack::{pack_rects, Placement};
pub use param::{parameterize, ParamMethod, ParamOptions, ParamResult};
