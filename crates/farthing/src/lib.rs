//! # farthing
//!
//! Mesh LOD for real-time 3D, engine-agnostic: farthing works on geometry and opaque material
//! slots, and reaches real materials only through the [`host::RenderHost`] boundary.
//!
//! The data model lives at the crate root:
//!
//! * [`Mesh`] — an indexed triangle mesh whose vertices are *wedges*: one vertex per
//!   unique attribute set, so a UV seam is two vertices at one position (the GPU layout).
//! * [`Scene`] — meshes, instances with world transforms, skeletons and opaque material
//!   slots. farthing never knows what a material looks like.
//! * [`host::RenderHost`] — the boundary to the engine. Anything that needs real materials
//!   (baking, impostor captures, alpha-aware visibility, image metrics) goes through it.
//! * [`bvh::Bvh`] — binned-SAH bounding volume hierarchy for ray casts and closest-point queries.
//! * [`image::Image`], [`metrics`] — float images and quality metrics.
//!
//! The processors are modules:
//!
//! * [`reduce`] — attribute-aware quadric edge-collapse reduction, repair, vertex-cache optimization.
//! * [`uv`] — chart-based UV parameterization and atlas packing.
//! * [`bake`] — material casting from a source to a target surface through the host.
//! * [`aggregate`] — many meshes and materials into one mesh with one atlas.
//! * [`cluster`] — Nanite-style cluster DAGs for per-cluster LOD selection at run time.
//! * [`hlod`] — many objects into one distant proxy: hidden-surface removal, merge, reduce, bake.
//! * [`voxel`] — voxel remeshing (watertight proxies) and conservative occluders.
//! * [`visibility`] — camera-sampled visibility culling and texel budgets.
//! * [`impostor`] — octahedral, flipbook, single-view and billboard-cloud impostors.
//! * [`bones`] — skinning LOD: influence limiting and bone removal.
//! * `io` — glTF 2.0 and OBJ import/export (feature `io`, on by default).

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

// Data model.
pub mod bvh;
pub mod error;
pub mod host;
pub mod image;
pub mod math;
pub mod mesh;
pub mod metrics;
pub mod quad;
pub mod raster;
pub mod rng;
pub mod scene;

// Processors.
pub mod aggregate;
pub mod bake;
pub mod bones;
pub mod cluster;
pub mod hlod;
pub mod impostor;
#[cfg(feature = "io")]
pub mod io;
pub mod reduce;
pub mod uv;
pub mod visibility;
pub mod voxel;

pub use error::{Error, Result};
pub use glam;
pub use math::Aabb;
pub use mesh::{Mesh, MorphTarget, Skin};
pub use scene::{Instance, Joint, MaterialId, MaterialSlot, Scene, Skeleton};
