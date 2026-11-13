# Provenance

farthing is a clean-room implementation. Its sources are published papers, permissively
licensed open source used as dependencies, and Simplygon's *public* documentation, read only to
scope features (what a capability does, never how it is implemented). No Simplygon SDK
binaries, headers, samples or plugins were used, inspected or traced.

## Algorithms by module

| Module | Technique | Source |
|---|---|---|
| `farthing::reduce::simplify` | Quadric error metrics, edge collapse | Garland & Heckbert, *Surface Simplification Using Quadric Error Metrics*, SIGGRAPH 1997 |
| | Border constraint planes | Garland & Heckbert 1997, §6 |
| | Attribute quadrics per wedge | Hoppe, *New Quadric Metric for Simplifying Meshes with Appearance Attributes*, IEEE Vis 1999; Garland & Heckbert, *Simplifying Surfaces with Color and Texture using Quadric Error Metrics*, 1998 |
| | Symmetry-aware collapses | Mirror-plane detection by reflected-point matching (standard); cf. Mitra, Guibas & Pauly, *Partial and Approximate Symmetry Detection*, 2006 |
| `farthing::reduce::quad` | Quad strip / edge-ring removal | Daniels, Silva, Shepherd & Cohen, *Quadrilateral Mesh Simplification*, 2008; Tarini et al., *Practical Quad Mesh Simplification*, 2010 |
| | Half-edge collapse, geomorph targets | Hoppe, *Progressive Meshes*, SIGGRAPH 1996 |
| | Link condition | Dey, Edelsbrunner, Guha, Nekhayev, *Topology Preserving Edge Contraction*, 1999 |
| `farthing::reduce::simplify` (queue) | Indexed binary heap | Standard (Cormen et al., *Introduction to Algorithms*) |
| | Cluster-parallel reduction with locked partition borders | Partition-and-lock parallel decimation, cf. Franc & Skala, *Parallel Triangular Mesh Decimation Without Sorting*, SCCG 2001; Morton-order partitioning (Morton, 1966) |
| | Fixed-association SIMD dot products, runtime ISA dispatch | Standard practice (deterministic reductions; `#[target_feature]` multiversioning) |
| | Batched rounds of non-conflicting collapses (independent-set decimation) | cf. DeCoro & Tatarchuk, *Real-time Mesh Simplification Using the GPU*, I3D 2007; Papageorgiou & Platis, *Triangular Mesh Simplification on the GPU*, The Visual Computer 2015. The freshness rule for skipping re-evaluation is farthing's own |
| | Lazy re-ranking with monotone lower bounds | Standard lazy-greedy evaluation (cf. Minoux, *Accelerated Greedy Algorithms*, 1978) |
| | Area-weighted (unnormalized) error ranking, area-scaled normal term, constraint planes without normalization weight | farthing's own choices, selected by the benchmark in `bench-compare/` (see docs/BENCHMARKS.md) |
| `farthing::reduce::optimize` | Linear-speed vertex cache optimisation | Forsyth, *Linear-Speed Vertex Cache Optimisation*, 2006 |
| `farthing::bvh` | Binned SAH BVH | Wald, *On fast Construction of SAH-based Bounding Volume Hierarchies*, 2007 |
| | Ray/triangle | Möller & Trumbore, *Fast, Minimum Storage Ray/Triangle Intersection*, 1997 |
| | Closest point on triangle | Ericson, *Real-Time Collision Detection*, 2005, §5.1.5 |
| `farthing::math` | Orthonormal basis | Duff et al., *Building an Orthonormal Basis, Revisited*, JCGT 2017 |
| `farthing::uv` | Chart growing + planar projection | Standard practice; cf. Sander et al., *Texture Mapping Progressive Meshes*, 2001 |
| | Least squares conformal maps | Lévy, Petitjean, Ray & Maillot, *Least Squares Conformal Maps for Automatic Texture Atlas Generation*, SIGGRAPH 2002 |
| | Preconditioned conjugate gradients | Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, 1994 |
| | Skyline packing | Jylänki, *A Thousand Ways to Pack the Bin*, 2010 |
| | As-rigid-as-possible refinement (local/global) | Liu, Zhang, Xu, Gotsman & Gortler, *A Local/Global Approach to Mesh Parameterization*, SGP 2008 |
| | Envelope (profile) Cholesky with reverse Cuthill–McKee ordering | George & Liu, *Computer Solution of Large Sparse Positive Definite Systems*, 1981; Cuthill & McKee, 1969 |
| | Embedding test (orientation + vertex angle sums + simple boundary) | Standard result for disk-topology maps; cf. Floater & Hormann, *Surface Parameterization: a Tutorial and Survey*, 2005 |
| | Minimum-area bounding rectangle (rotating calipers), convex hull | Freeman & Shapira, 1975; Toussaint, *Solving Geometric Problems with the Rotating Calipers*, 1983; Andrew, 1979 |
| | Chart merging under distortion limits | Standard practice in atlas generation; cf. Sander et al. 2001, Lévy et al. 2002 |
| | L2 stretch metric | Sander, Snyder, Gortler & Hoppe, *Texture Mapping Progressive Meshes*, SIGGRAPH 2001 |
| | Shape-aware packing (rasterized chart outlines on a per-column skyline, 4 rotations) | Built on the skyline idea above; conservative triangle/box overlap as in Akenine-Möller 2001 |
| `farthing::bake` | Ray-cast transfer with cage-like offset | Standard normal/texture baking practice |
| | Tangent space | Mikkelsen, *Simulation of Wrinkled Surfaces Revisited*, 2008 (via the `mikktspace` crate) |
| `farthing::reduce` | Whole-part removal during reduction (orientation-aware removal error, lazily refined lower bounds) | Own design; the stall analysis (closed parts stuck at minimal triangulations) prompted it |
| | Counting-sort CSR for the ordering graph | Standard sparse-matrix practice |
| | Multiply-rotate small-key hasher | After rustc's FxHash (public domain description) |
| `farthing::reduce` | Expected-silhouette weight on face quadrics (edge silhouette probability = dihedral angle / π for uniformly random views) | Derived in the v8 perception research (docs/research/perception.md), marked there as the report's own derivation |
| `farthing::impostor` | Surfel proxies: blue-noise placement by weighted sample elimination | Yuksel, *Sample Elimination for Generating Poisson Disk Sample Sets*, Eurographics 2015; surfels after Pfister et al., *Surfels: Surface Elements as Rendering Primitives*, SIGGRAPH 2000 |
| `farthing::voxel` | Per-pocket gap sealing (keep only regions of the closing deeper than the closing radius) | Own; simplified from the persistence-based pocket ranking suggested in docs/research |
| `farthing::impostor` | Relief-marched impostor parallax (linear march plus bisection through each frame's height field), height field treated as a shell | Linear/binary search after Policarpo, Oliveira and Comba, *Real-Time Relief Mapping on Arbitrary Polygonal Surfaces*, I3D 2005; the shell rule is our own |
| | Supersampled impostor captures (box filter with coverage in alpha) | Standard |
| `farthing::voxel` | Triangle/box overlap (SAT) | Akenine-Möller, *Fast 3D Triangle-Box Overlap Testing*, JGT 2001 |
| | Surface nets | Gibson, *Constrained Elastic Surface Nets*, 1998 |
| | Morphological closing for gap sealing | Standard image morphology; applied only when the unsealed flood loses >10% of the solid (own heuristic) |
| | Manifold solids before extraction (filling 2×2 face and 2×2×2 corner critical configurations) | Making the voxel set *well-composed*: Latecki, *3D Well-Composed Pictures*, Graphical Models and Image Processing, 1997 |
| | Fold-safe projection onto the source (revert collapsed/flipped faces, back coincident snaps off by 2% of a cell) | Own; closest points from the BVH |
| `farthing::visibility` | Fibonacci sphere sampling | González, *Measurement of Areas on a Sphere Using Fibonacci and Latitude–Longitude Lattices*, 2010 |
| `farthing::impostor` | Octahedral impostors, virtual-plane reprojection, depth parallax | Brucks, *Octahedral Impostors*, 2018 (shaderbits.com); parallax refinement cf. Kaneko et al., *Detailed Shape Representation with Parallax Mapping*, 2001 |
| | Octahedral mapping | Cigolle et al., *A Survey of Efficient Representations for Independent Unit Vectors*, JCGT 2014 |
| | Billboard clouds | Décoret, Durand, Sillion & Dorsey, *Billboard Clouds for Extreme Model Simplification*, SIGGRAPH 2003 |
| `farthing` | Parallel BVH build with a fixed layout (same tree for any thread count) | Own; SAH binning as in Wald, *On fast Construction of SAH-based Bounding Volume Hierarchies*, 2007 |
| `farthing::reduce` | Quad edge rings as union-find classes, walked per class | Union-find: Tarjan, *Efficiency of a Good But Not Linear Set Union Algorithm*, JACM 1975 |
| | Deterministic parallel quadric setup (gather in triangle order) and speculative candidate evaluation | Own |
| `snugly` | Hole-filling packer: bottom-left placement over per-column free intervals, branch and bound on a column-subsampled lower bound | Bottom-left placement: Baker, Coffman & Rivest, *Orthogonal Packings in Two Dimensions*, SIAM J. Comput. 1980; Chazelle, *The Bottom-Left Bin-Packing Heuristic*, IEEE TC 1983 |
| `farthing::voxel` | Sub-voxel signed boundary: centre sign by angle-weighted pseudo-normals, confirmed by ray winding counts; thin-sheet and lone-sheet guards | Bærentzen & Aanæs, *Signed Distance Computation Using the Angle Weighted Pseudonormal*, IEEE TVCG 2005; ray crossing parity (classic point-in-polyhedron test); guards are own |
| `farthing::reduce` | Parallel commits by deterministic reservations | Blelloch, Fineman, Gibbons & Shun, *Internally Deterministic Parallel Algorithms Can Be Fast*, PPoPP 2012 |
| `farthing::reduce::fast` | Fast mode: f32 quadrics, sorted passes over all edges, part deletion on stall | Quadrics: Garland & Heckbert 1997; pass-based scheduling (sort candidate collapses by cost, apply non-conflicting ones in order) is standard independent-set decimation (see the batched-rounds row); attribute penalty Σ area·|Δattribute|² is own |
| `farthing::reduce` | Visible-sample part removal error (rays from each sample must escape the other parts), transparent-material pass-through, error capped at gap + part size, protection on stall | Own; the visibility test is ambient-occlusion-style ray casting (cf. Zhukov, Iones & Kronin, *An Ambient Light Illumination Model*, 1998) |
| | Sliding along non-manifold junctions; twin faces counted once when classifying edges | Own extension of border sliding; collapse validity still by the link condition (Dey et al. 1999) |
| `snugly` | MaxRects (maximal free rectangles, split and prune) with best-short-side, best-long-side, best-area, bottom-left and contact-point rules | Jylänki, *A Thousand Ways to Pack the Bin*, 2010; running several rules and orders and keeping the best is standard practice |
| | Interior holes as free intervals (nesting in holes) | Own extension of the hole-filling packer |
| `farthing::impostor` | Layered impostors: second layer by depth peeling (front faces only, behind the first layer by a bias), shader marches both layers | Depth peeling: Everitt, *Interactive Order-Independent Transparency*, NVIDIA 2001; layered depth images: Shade, Gortler, He & Szeliski, *Layered Depth Images*, SIGGRAPH 1998. Front-face rule and two-layer march are own |
| | Progressive surfel order (staged elimination, reversed) | Yuksel 2015, §progressive sample sets |
| | LFSF surfel stream (quantized positions, octahedral normals, progressive records) | Own format; octahedral normals: Cigolle et al. 2014 |
| `farthing::uv` | Splitting sprawling charts (two-means on UV face centroids, connected components, kept when the realigned rectangles shrink) | Own; k-means after Lloyd, *Least Squares Quantization in PCM*, 1982 |
| `snugly` | Bounding-rectangle candidate searched only below the outline result, pruned by an area bound | Own |
| `farthing::impostor` | Half-resolution back layer, `layer_benefit` (coverage × depth gap) for automatic layer choice | Own |
| | Variable-radius (adaptive) sample elimination for weighted surfels | Yuksel 2015 (adaptive sampling with a per-sample radius) |
| | Surfel splats with dithered fade (screen-space hash) | Splatting after Pfister et al. 2000 and Zwicker et al., *Surface Splatting*, SIGGRAPH 2001; dithered (stochastic) transparency as in Enderton et al., *Stochastic Transparency*, I3D 2010 |
| `farthing::reduce` | Junction endpoints and branch points collapsing along junctions | Own extension of border sliding; link condition (Dey et al. 1999) |
| `farthing::metrics` | Visible-surface distance (samples kept if a ray escapes their own mesh) | Own benchmark metric |
| `farthing::bones` | Skinned-error bone removal | Linear blend skinning error; cf. Landreneau & Schaefer, *Simplification of Articulated Meshes*, 2009 |

## Benchmark harness (not part of the library)

`bench-compare/` links meshoptimizer (`meshopt` crate, MIT), xatlas (`xatlas-rs-v2` crate, MIT)
and baby_shark (MIT) **only to measure against them**. No code from them is used in farthing;
the harness is a separate, excluded workspace and needs a C++ toolchain and libclang.

## Patents

Clean-room work protects against copyright claims, not patent claims. Get a freedom-to-operate
review before any commercial release.
