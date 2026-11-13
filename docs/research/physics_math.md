# farthing: ideas from computational physics and applied mathematics

Scope: level sets, eikonal solvers, interface reconstruction (VOF/MOF), N-body/Barnes–Hut, multigrid,
discrete exterior calculus / heat methods, threshold dynamics (mean-curvature flow), computational
topology (persistence, discrete Morse theory, digital topology, link conditions), and packing.
Focus: W3 (sub-voxel signed fields robust to open/thin input), W7 (controlled topology change), plus
W2 and W4 where a real transfer exists.

## What I looked at in the code first (so the ideas fit what's there)

- `crates/farthing/src/voxel/mod.rs::classify`: voxels that overlap a triangle (`tri_box_overlap`)
  are marked blocked. They are dilated by `close_gaps` cells, the outside is flood-filled from the
  grid border, solid = not outside, and then the solid is eroded. So every input sheet becomes a
  solid layer at least one voxel thick. Any crevice narrower than about 1 to 2 cells is filled by
  construction. Projection back to the surface (search radius 1.5 cells) cannot reopen it.
- `remesh`: `adaptive_gaps` picks between a sealed and an unsealed classification with a single
  global test: did the solid shrink by more than 10%? Gap closing is therefore all-or-nothing and
  global. A single leak forces the whole proxy to be closed, which fills every real crevice too.
- `grid.rs::manifoldize` patches the classic 2x2 face-diagonal and corner-diagonal ambiguities by
  filling voxels.
- `crates/farthing/src/reduce/simplify.rs` (around lines 567-596): both endpoints of any edge shared
  by more than 2 triangles are locked, and so are border vertices of degree >2. The link check
  (around lines 1055-1084) is the manifold version: it compares common neighbours with shared
  triangles.

These points come up again in the ideas below.

---

## Ranked ideas (expected payoff / effort)

| # | Idea | Addresses | Effort | Already standard in graphics? |
|---|------|-----------|--------|------------------------------|
| 1 | Persistence-gated, *local* leak sealing and part merging on the exterior distance field (merge tree / 0-dimensional persistent homology) | W7, W3 | Low–Med | Not as such (alpha wrapping / morphology use a single global scale) |
| 2 | Generalized winding number sign via Barnes–Hut (fast winding numbers) + exact unsigned narrow band + ε-thickening of sheets | W3 | Med | **Yes**, standard (libigl, Houdini, fTetWild); farthing just lacks it |
| 3 | Signed Heat Method on the voxel grid, solved with geometric multigrid | W3, W7 | Med | Recent (SIGGRAPH 2024); not yet common in remeshers |
| 4 | Sub-voxel crevice detection via Hamilton–Jacobi skeleton flux / MOF-style "disconnected pieces per cell", then local refinement | W3 | Med–High | Pieces exist (adaptive DC, octrees); this detector-driven refinement is not standard |
| 5 | Full link condition for non-manifold 2-complexes instead of locking | W2 | Low–Med | Manifold link condition is standard; the general-complex version is not used in game LOD tools as far as I found |
| 6 | Digital-topology-controlled edits (simple points) + saddle-pair persistence for handle removal | W7 | Med–High | Partly (Wood et al. 2002–04 isosurface topology simplification); not in asset tools |
| 7 | Threshold dynamics (MBO / volume-preserving MBO) for scale-controlled proxy simplification | W7 (W5) | Low | Rare in mesh processing; morphology (Nooruddin–Turk 2003) is the graphics analogue |
| 8 | FFT/NTT cross-correlation placement for UV charts (spectral packing in 2D) | W4 | Med | Spectral packing is 3D/printing (2023). I found no published UV packer using it |
| 9 | (Minor) Re-distancing with sub-cell fix + fast sweeping after any field edit | W3 support | Low | Standard in level-set physics codes and OpenVDB |

---

## 1. Persistence-gated local leak sealing and part merging (W7, W3)

**Source.** 0-dimensional persistent homology / merge trees of a scalar field (Edelsbrunner–
Letscher–Zomorodian 2002 and later). Fast, localized, parallel versions come from scientific
visualization of physics data: Lukasczyk, Garth, Maciejewski, Tierny, *Localized Topological
Simplification of Scalar Data*, IEEE VIS 2020 ([arXiv:2009.00083](https://arxiv.org/abs/2009.00083)).
Given a set of extrema to keep, it produces a nearby function with only those extrema, by locally
flattening the sub/superlevel set components of the unwanted ones. The authors report up to 36x
speedups and about 70% parallel efficiency. TTK ships it
([doc](https://topology-tool-kit.github.io/doc/html/classttk_1_1LocalizedTopologicalSimplification.html)).

**Mechanism.** Take the unsigned distance d(x) to the input triangles on the voxel grid, restricted to
air. Sweep d from large to small (a superlevel-set filtration) with union-find:
- Each local maximum of d starts an "air pocket" (an outside region or an internal cavity).
- Two pockets merge at a saddle of d. The saddle value s is half the width of the narrowest channel
  joining them.
- The persistence pair (birth = pocket's max distance, death = s) tells you two things: how big the
  cavity is, and how narrow its leak is.

Symmetrically, the sublevel filtration of d on the input (or of the SDF) gives the offset at which
separate parts merge. Pair (part i, part j) merges at offset s_ij. That is exactly the "merge nearby
parts" decision of W7, one pair at a time.

**Transfer to farthing.**
- Replace the global `close_gaps` plus 10% heuristic in `remesh` with this:
  1. Compute d (exact in the narrow band via the existing `Bvh::closest_point`, propagated outward
     by fast sweeping or a chamfer pass).
  2. Run union-find over voxels sorted by d (descending), seeded with the border/outside component.
  3. For every interior cavity, record the saddle voxel and the saddle value s at which it joins
     the outside.
  4. Seal only the leaks whose s is below a gap threshold, *and* whose cavity has persistence
     (max d − s) above a significance threshold (a real enclosure rather than noise). Sealing
     writes a plug: a small ball of radius about s around the saddle voxel is marked blocked.
     Then flood once.
- Real crevices elsewhere stay open, because nothing is dilated globally.
- The same machinery with the sublevel filtration on the solid gives a ranked list of "part i merges
  with part j at gap s_ij". A proxy can merge parts with gap below τ, locally, by filling the
  saddle region (a convex-hull plug or a capsule between the two closest points).

**Why it may beat the state of the art.**
- Alpha wrapping (Portaneri et al., *Alpha Wrapping with an Offset*, SIGGRAPH 2022;
  [CGAL](https://doc.cgal.org/latest/Alpha_wrap_3/)) and volumetric closing (Nooruddin & Turk,
  [TVCG 2003](https://faculty.cc.gatech.edu/~turk/my_papers/volumetric_simp.pdf)) both use a single
  global scale. The same alpha that seals a 3 cm leak also fills every 3 cm crevice.
- Persistence separates the two cases: a leak is a narrow channel with a big cavity behind it; a
  crevice is a narrow channel with nothing behind it. This directly improves Hausdorff error (W3)
  on assets that need sealing, which today lose all their crevices when one leak triggers the
  sealed path.

**Cost/risk (pure Rust).**
- Low to medium: a sort (or bucket sort on quantized d), union-find, and a plug rasterizer, roughly
  300-500 lines.
- Deterministic as long as ties in d are broken by voxel index.
- Risk: a leak through a long thin slot has a saddle *line*, not a point. Plug the whole set of air
  voxels with d ≤ s within a radius of the saddle (iterate until the cavity separates). Check
  separation by re-running the flood.

**Absorbed by graphics?** Merge trees and persistence are common in scientific visualization (TTK)
and appear in topology-aware ML losses. I did not find a published mesh-repair or proxy pipeline that
uses saddle values of the exterior distance field to plug leaks selectively. **Uncertain**: something
similar may exist in medical-imaging cavity filling. Treat this as a novel combination, not a
published method.

---

## 2. Robust sign: fast generalized winding numbers + exact unsigned band + sheet thickening (W3)

**Source.** Generalized winding number (Jacobson, Kavan, Sorkine-Hornung 2013), made fast by a
Barnes–Hut / multipole-style tree borrowed from N-body physics: Barill, Dickson, Schmidt, Levin,
Jacobson, *Fast Winding Numbers for Soups and Clouds*, SIGGRAPH 2018
([project](https://dgp.toronto.edu/projects/fast-winding-numbers),
[paper](https://www.dgp.toronto.edu/projects/fast-winding-numbers/fast-winding-numbers-for-soups-and-clouds-siggraph-2018-barill-et-al.pdf)).
SideFX published an implementation ([github.com/sideeffects/WindingNumber](https://github.com/sideeffects/WindingNumber)),
and libigl has `igl::fast_winding_number`.

**Mechanism.**
- The winding number w(x) is the sum of the signed solid angles of all triangles, divided by 4π.
- Far clusters are approximated by a dipole Taylor expansion (low order), exactly like a Barnes–Hut
  gravity tree.
- w is about 1 inside, about 0 outside, and degrades gracefully with holes and self-intersections.

**Transfer.**
- Replace "flood from the border through non-blocked voxels" as the *sign* source:
  - sign(x) = inside if w(x) > 0.5.
  - Compute w only at voxels in or near the surface band. Deep cells inherit the sign by flood fill,
    because w is harmonic and cannot change sign without crossing the surface.
- Compute exact unsigned distance at corners in the band (`Bvh::closest_point` is already there).
  The SDF is ±d. Contour it with surface nets / dual contouring, placing each vertex at the QEF
  minimizer of the closest points. Surfaces then sit at sub-voxel accuracy instead of being
  inflated by one voxel layer.
- **Thin and open sheets (important caveat).** For a single open sheet (a leaf quad, a cloth panel),
  w is about +0.5 just on one side and −0.5 on the other. Thresholding at 0.5 gives *no* volume. So
  use solid = {w > 0.5} ∪ {d < ε_sheet}, i.e. thicken only the parts that the winding number cannot
  claim. ε_sheet can be well below a voxel if the contouring uses the exact d.
  - This is what the current pipeline does implicitly with a one-voxel thickness, but here it
    applies only to sheets, not to closed solids.

**Why better than the current approach.** The current sign comes from blocked-voxel flood fill. Its
sign error is up to about one voxel everywhere, and a single hole flips entire regions. Winding
numbers remove the inflation on closed parts and tolerate holes smaller than the part.

**Cost/risk.**
- Medium: a BVH with per-node dipole moments (area-weighted normal sum, centroid) plus a
  second-order term. farthing already has a BVH. About 400-600 lines.
- Deterministic if each query traverses in fixed order (per-query parallelism only).
- Barnes–Hut accuracy parameter: β ≈ 2 in the paper (from memory; check against the paper).
- Contouring w directly is fragile; Feng notes contouring winding numbers fails on about 10% of a
  benchmark ([SHM page](https://nzfeng.github.io/research/SignedHeatMethod)). Use w only for the
  sign and the exact d for geometry.

**Absorbed?** **Yes, it is mainstream graphics** (libigl, Houdini, fTetWild-style robust
inside/outside). Included because it is the cheapest correct baseline that farthing is missing, and
ideas 1, 3, 4 and 6 build on it.

---

## 3. Signed Heat Method on the voxel grid with multigrid (W3, W7)

**Source.** Nicole Feng and Keenan Crane, *A Heat Method for Generalized Signed Distance*,
ACM TOG / SIGGRAPH 2024 ([project](https://nzfeng.github.io/research/SignedHeatMethod),
[PDF](https://nzfeng.github.io/research/SignedHeatMethod/SignedDistance.pdf),
[geometry-central docs](https://geometry-central.net/surface/algorithms/signed_heat_method/)).
Its lineage is Varadhan's formula / the heat method (Crane, Weischedel, Wardetzky 2013), which comes
from PDE theory.

**Mechanism.** Three steps:
1. Diffuse the source normals into the domain with vector heat diffusion (one screened-Poisson
   solve).
2. Normalize the result to get a unit field approximating ∇SDF.
3. Solve a Poisson equation for the scalar whose gradient best matches that field.

For small t, the diffused normals are parallel to the SDF gradient. The authors claim robustness to
holes, noise, self-intersections and non-manifold input, and an inside/outside classification often
more robust than winding numbers. It works on regular grids and in any dimension. Their stated
limitations: it needs global solves, it is not output-sensitive, and longer diffusion gives more
harmonic hole-fills but a worse distance approximation.

**Transfer.**
- Two linear solves on farthing's grid: a 7-point Laplacian on a regular box. This is the textbook
  case for **geometric multigrid** (V-cycles with red-black Gauss–Seidel; O(N), no matrix assembly).
- The source term is the area-weighted triangle normals splatted into the voxels they overlap
  (already enumerated by `voxelize`).
- Outputs:
  - (a) a generalized SDF whose zero set is sub-voxel accurate near the input and *implicitly
    completes holes*. That gives a sign for open meshes with large missing regions, where winding
    numbers become ambiguous (w ≈ 0.5 over large areas).
  - (b) a physically meaningful W7 knob: diffusion time t ≈ (hole size)² controls how large a hole is
    bridged smoothly.
- Combine with idea 1: run SHM only when idea 2 reports many voxels with w in [0.3, 0.7] (ambiguous
  sign).

**Why it may beat the state of the art.** Winding numbers give labels, not distances, and are
ambiguous for big holes. SHM gives a distance-like field with consistent completion in one pipeline,
and on a regular grid it costs two multigrid solves: about 2 × (5-10 V-cycles) × O(N). At 128³ that
is tens of milliseconds per solve on one core (my estimate, not measured).

**Cost/risk.**
- Medium: a multigrid Poisson solver on a box with Neumann/Dirichlet boundaries is about 400 lines
  of Rust. Splatting and normalization are trivial.
- Determinism: use fixed V-cycle counts and red-black ordering. Parallelize by slabs with no
  floating-point reductions, or reduce in fixed order.
- Risk: the grid must cover the domain with padding. The mesh is open, so the outer boundary
  condition matters. The paper's open questions (pseudonormal-like completions) mean results on
  very sparse inputs may be implausible.

**Absorbed?** It is in geometry-central and has a standalone library
([signed-heat-docs](https://nzfeng.github.io/signed-heat-docs)). I found no evidence it is used in
game-asset proxy generators. The multigrid part is standard (screened Poisson reconstruction uses
it).

---

## 4. Detecting sub-voxel crevices and refining locally (W3)

This is the honest core of W3. Any field sampled on a uniform grid of spacing h, whether signed
distance, winding number or heat, *cannot* represent a slot narrower than about h. The Hausdorff
error of a filled slot is its depth, not its width. The fix is local refinement, and physics codes
have two good detectors for *where* to refine.

**4a. Hamilton–Jacobi skeleton flux (shock detection).**
- Source: Siddiqi, Bouix, Tannenbaum, Zucker, *Hamilton–Jacobi Skeletons*, IJCV 2002
  ([PDF copy](https://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Hamilton-Jacobi-Skeleton-Siddiqi-Bouix.pdf)).
- Mechanism: the medial axis is where the eikonal equation forms shocks, i.e. where ∇d is
  discontinuous. The *average outward flux* of ∇d through a small sphere is strongly negative there
  and about 0 elsewhere.
- Transfer: compute the exact unsigned distance d *and the closest-point direction* at voxel
  corners (BVH query).
  - A cell whose corners' closest-point vectors point at *opposite walls* (strongly negative flux,
    i.e. the exterior medial sheet passes through the cell) and whose d is below h marks a slot
    narrower than the cell.
  - These cells, and only these, get subdivided (2x or 4x per axis, as an octree/sparse block) before
    classification and contouring.
- Cost scales with crevice area, not volume.

**4b. "Disconnected pieces in a cell" (Moment-of-Fluid filament capturing).**
- Source: Jemison, Sussman, Shashkov, *Filament capturing with the multimaterial moment-of-fluid
  method*, J. Comput. Phys. 285 (2015)
  ([OSTI](https://www.osti.gov/pages/biblio/1247141-filament-capturing-multimaterial-moment-fluid-method)).
- Mechanism: under-resolved filaments are detected as *multiple disconnected regions of one material
  inside a cell*. Each piece is then reconstructed as its own "material" with volume conserved.
  Adaptive mesh refinement is applied only near filament tips. The authors report accuracy gains
  comparable to one extra refinement level at lower cost.
- Transfer: clip the input triangles to each surface voxel. Exact clipping in the spirit of
  Powell & Abel, *An exact general remeshing scheme applied to physically conservative
  voxelization*, J. Comput. Phys. 2015 ([arXiv:1412.4941](https://arxiv.org/abs/1412.4941); r3d
  library) is robust to degeneracies with combinatorial decisions. Then count connected air
  components within the cell (a flood on the clipped arrangement, or on a 4³ sub-sample).
  - A cell with two or more air pieces needs refinement, or multiple dual vertices per cell, in the
    style of manifold dual contouring.
- 4b is more exact than 4a but more expensive.

**Why it may beat the state of the art.** Adaptive dual contouring and OpenVDB-style sparse grids
exist, but their refinement criteria are usually curvature/QEF-error based. A slot can have zero
curvature (two flat parallel walls) and zero QEF error per wall, so it is missed. Shock/flux and
piece-count detectors target exactly the topology-carrying sub-voxel features.

**Cost/risk.** Medium-high: needs a sparse 2-level grid and surface nets on mixed resolution, or
simply run the uniform pipeline on refined bricks and stitch them. 4a is cheap to prototype as a
diagnostic first: count flagged cells per asset and correlate with the measured Hausdorff error
before building refinement. Deterministic.

**Absorbed?** The HJ skeleton is known in vision and graphics for skeletonization, not as a
refinement trigger for remeshing. MOF and filament capturing are confined to multiphase-flow codes
(I found no graphics use). **Uncertain** whether 4a's flux signal is clean enough on triangle soups;
prototype before committing.

---

## 5. Full link condition for non-manifold complexes instead of locks (W2)

**Source.** Dey, Edelsbrunner, Guha, Nekhayev, *Topology preserving edge contraction*, Publ. Inst.
Math. (Beograd) 66(80), 1999 ([EuDML](https://eudml.org/doc/256270),
[summary page](https://www.cs.purdue.edu/homes/tamaldey/decim.htm)). It extends the guarantees to
2-dimensional simplicial complexes and 3-manifolds, and underlies Geomagic's decimator. The later
summary by Dey, Hirani, Krishnamoorthy, Smith, *Edge Contractions and Simplicial Homology*, 2013
([arXiv:1304.0664](https://arxiv.org/abs/1304.0664)), states that link-condition contractions
"preserve homeomorphism in low dimensional complexes, and homotopy in general". It also gives
cheaper per-dimension "p-link conditions" that preserve specific homology groups.

**Mechanism.** Contract edge ab only if Lk(a) ∩ Lk(b) = Lk(ab), checked at every level:
- vertices: common neighbours of a and b are exactly the apexes of triangles on ab;
- edges: no edge xy with both triangles axy and bxy present;
- boundary: handled with a dummy vertex joined to the boundary.

This is defined for *any* simplicial complex, including edges with 3+ triangles.

**Transfer.**
- Today, `simplify.rs` locks both endpoints of every non-manifold edge, plus border vertices of
  degree >2. At 1% targets those locks are a likely cause of the 20-60% overshoot on 2 of 8 assets.
  (I have not measured which share is due to locks and which to flip guards. The `Validity` tally
  in the code should tell you; check it first.)
- Instead, treat the mesh as a 2-complex:
  - drop the blanket lock;
  - extend the existing marker-array check to the edge level (the "axy & bxy" test) and to
    non-manifold edge stars (Lk(ab) = all apexes of the 3+ triangles on ab);
  - add the dummy-vertex rule for boundaries.
- Collapses along a non-manifold edge chain (e.g. a T-junction seam where a fin meets a body) then
  become legal whenever homotopy type is preserved.
- Use the p-link variants if you deliberately want to allow *some* topology change, e.g. preserve
  H₀ (components) but allow H₁ (handles) to die, as a W7 mode. That is exactly what Dey et al. 2013
  characterise.

**Why better than the state of the art.** Mainstream QEM implementations either restrict themselves
to manifolds (and lock non-manifold structure, as farthing does) or allow arbitrary vertex pairs
(Garland–Heckbert virtual pairs, Popović–Hoppe 1997 progressive simplicial complexes), which give no
topological control. The general link condition is a principled middle ground.

**Cost/risk.**
- Low to medium. The vertex-level check already exists; the edge-level check is another marker pass
  over the neighbours' fans.
- The real risk is the attribute/wedge mapping (`map_wedges`) around non-manifold fans, and the
  flip test at a non-manifold edge, where "the" normal is ambiguous. Run the flip test per incident
  fan.
- Deterministic.

**Absorbed?** The manifold link condition is standard in graphics. Applying the general-complex
version to game-asset LOD is uncommon as far as I could find, but the math is 25 years old.

---

## 6. Topology-controlled voxel edits: simple points + saddle persistence (W7)

**Sources.**
- Digital topology: Bertrand & Malandain, *A new characterization of three-dimensional simple
  points*, Pattern Recognition Letters 1994. A voxel is *simple* if flipping it changes no topology,
  decidable from its 26-neighbourhood.
- Topology-preserving level sets: Han, Xu, Prince, *A topology preserving level set method for
  geometric deformable models*, IEEE TPAMI 2003.
- Ségonne, *Active contours under topology control — genus preserving level sets*, IJCV 2008
  ([PDF](https://surfer.nmr.mgh.harvard.edu/ftp/articles/2007/2007_-_Segonne_-_International_Journal_of_Computer_Vision.pdf)):
  this one allows *selected* topology changes (merges/splits) while forbidding genus changes.
- Handle detection and removal in 3D scalar fields:
  - Guillou, Vidal, Tierny, *Discrete Morse Sandwich: fast computation of persistence diagrams for
    scalar data*, IEEE TVCG 2023 ([arXiv:2206.13932](https://arxiv.org/abs/2206.13932));
  - Kissi, Pont, Levine, Tierny, *A Practical Solver for Scalar Data Topological Simplification*,
    IEEE VIS 2024 ([arXiv:2407.12399](https://arxiv.org/abs/2407.12399)). Unlike extremum-only
    methods, this also cancels saddle pairs in 3D, so isosurfaces lose handles. "Genus repair" is
    one of its applications.
- The graphics precedent: Wood, Hoppe, Desbrun, Schröder, *Isosurface Topology Simplification*
  (MSR-TR-2002-28, later TOG 2004) ([PDF](https://hhoppe.com/isotoposimp_tr.pdf)). It finds handles
  by Reeb-graph sweep, measures each handle by its shortest Reeb/cross loop, and removes handles
  below τ by writing a spanning "wall" into the volume.

**Mechanism and transfer.**
- (a) Make every voxel flip in farthing topology-aware. `manifoldize`, closing/erosion and the plugs
  of idea 1 flip voxels freely today. Instead, test the flip with a simple-point check: an
  Euler-characteristic octant LUT plus a 26/6-connectivity check, as used in skeletonization codes
  such as Lee–Kashyap–Chu 1994.
- (b) A flip that is *not* simple is a topology change: classify it as merge, split, hole-fill or
  handle creation/removal from the change in component counts and Euler number. Allow only the
  classes the user asked for (Ségonne's idea).
- (c) For handles, compute the saddle-saddle persistence pairs of the SDF (Discrete Morse Sandwich
  is the fast route). Pair (s1, s2) with small persistence is a handle or tunnel whose thickness is
  set by the pair's values. Kill it either with the Kissi et al. optimisation, or (simpler, Wood
  et al. style) by filling or cutting along the shortest loop.

The result is a W7 control surface:
- "close holes < x"
- "merge parts closer than y" (from idea 1)
- "remove handles thinner than z"
- "never create new handles"

**Why better.** Current morphology changes topology uncontrollably: it closes some tunnels and opens
others, and `manifoldize` can create handles. Simple-point gating makes every change deliberate and
logged, which matters for determinism audits too.

**Cost/risk.**
- (a)+(b): medium; the LUT approach is well documented.
- (c): high. Discrete Morse/persistence in 3D is substantial code, roughly 2-4k lines for a careful
  implementation. Wood et al.'s sweep is simpler but still significant. Prototype (a)+(b) first.

**Absorbed?** Wood et al. is graphics, but I see no evidence that asset-optimization tools ship it.
Simple points and topology-preserving level sets are standard in medical imaging, not in LOD tools.

---

## 7. Threshold dynamics (MBO) for scale-controlled proxy simplification (W7, some W5)

**Sources.**
- Merriman–Bence–Osher 1992 (MBO scheme).
- Esedoğlu & Otto, *Threshold Dynamics for Networks with Arbitrary Surface Tensions*, Comm. Pure
  Appl. Math. 2015 ([MPI preprint](https://www.mis.mpg.de/publications/preprints/2013/prepr2013-2.html)).
- Ruuth & Wetton, *A simple scheme for volume-preserving motion by mean curvature*, J. Sci. Comput.
  2003 ([record](https://portal.mardi4nfdi.de/wiki/Publication:1418893)).
- Higher-order variants: Esedoğlu, Ruuth, Tsai, *Threshold dynamics for high order geometric
  motions* ([PDF](https://dept.math.lsa.umich.edu/~esedoglu/Papers_Preprints/esedoglu_ruuth_tsai.pdf)).

**Mechanism.** Mean-curvature flow on a voxel occupancy is approximated by repeating two steps:
1. convolve the indicator with a Gaussian of variance 2t;
2. threshold at 1/2.

Topology changes (pinch-off of thin necks, merging of close parts, closing of small holes) happen
automatically, at a spatial scale of about √t. The volume-preserving variant chooses the threshold
so that volume is conserved: in practice, threshold at the quantile that keeps the voxel count.

**Transfer.** A one-knob proxy pre-filter after classification. A few MBO steps at scale √t equal to
the proxy's on-screen pixel footprint:
- remove features that cannot be seen;
- round voxel staircases;
- make the topology change at that scale, coherently.

Use the volume-preserving threshold so proxies don't shrink. If you have fractional occupancy (idea
4b clipping, or a smooth SDF), MBO on the fractional field is smoother. Protect silhouettes or
user-locked regions with a mask.

**Why it may beat the current approach.** Dilate/erode closing is anisotropic (box/cross structuring
element) and has no notion of curvature. MBO is isotropic (Gaussian), unconditionally stable, and
provably converges to mean-curvature flow. Separable Gaussians make it O(N) per step and
SIMD-friendly. As a scale-space filter it lines up with "how it looks at screen size" (W5) better
than a cell-count closing radius.

**Cost/risk.**
- Low: a separable 1D Gaussian (or a few box-blur passes) plus a threshold or quantile select.
  About 150 lines.
- Deterministic with a fixed kernel and fixed-order summation. To be bit-exact across CPUs, use
  integer box blurs.
- Risk: mean-curvature flow *removes* thin structures (poles, railings) entirely, which can be
  worse for silhouettes. Gate it with idea 6's topology classes, or add a thin-feature protection
  mask (thin = large |∇d| discontinuity / skeleton flux from idea 4a).

**Absorbed?** In graphics, volumetric morphology (Nooruddin & Turk 2003) is the standard analogue.
MBO itself is used in image segmentation and multiphase physics. I found no mesh-LOD use.

---

## 8. FFT/NTT cross-correlation placement for UV charts (W4)

**Source.** Cui, Rong, Chen, Matusik, *Dense, Interlocking-Free and Scalable Spectral Packing of
Generic 3D Objects*, ACM TOG 42(4), SIGGRAPH 2023
([PDF](https://cdfg.mit.edu/pdf/journal/2023-cui-dense-interlocking-free-and-scalable-spectral-packing-of.pdf)).
- A collision metric = correlation of the new object's voxels with the occupancy of the already
  placed objects, computed by FFT; zero entries are free offsets.
- A proximity metric = correlation with an unsigned distance field of the placed set.
- Placement = the minimum of proximity plus height penalty over *all* offsets at once.
- They report 3 ms for collision detection on a 24 M-voxel grid, versus 9.4 s brute force on a
  smaller grid. The paper is 3D (printing); it does not address UV atlases.

The related applied-math school of phi-functions (Chernov, Stoyan, Romanova, *Mathematical model and
efficient algorithms for object packing problem*, Annals of OR 2010) gives exact continuous
non-overlap constraints for local compaction. It is higher cost and noted only as a possible
refinement stage.

**Transfer.** For each chart, in decreasing size order:
1. Rasterize at atlas texel resolution (farthing already rasterizes charts).
2. Correlate the chart mask with the current atlas occupancy, for each of 2-8 rotations/flips, via
   2D FFT.
3. Among zero-collision offsets, pick the one minimizing ⟨chart, distance-to-occupied⟩. This is the
   "nestles into concavities" score, plus a small bottom-left / bounding-box growth penalty.

This is xatlas's brute-force raster placement, but evaluated for every offset in O(N log N) per chart
with a proximity objective, rather than first-fit. Use it for the few large irregular charts, which
are exactly the W4 failure case, and keep the skyline packer for the many small ones.

**Determinism caveat (important for farthing).** Floating-point FFT libraries choose SIMD paths at
runtime, so results may differ in the last bits across CPUs. I believe this applies to `rustfft`
(it auto-selects AVX/SSE/NEON), but I have **not verified** it. Two ways out:
- the collision counts are integers, so round to nearest and the result is exact as long as the
  error is < 0.5, which holds comfortably at atlas sizes; quantize the proximity cost the same way;
- or use a number-theoretic transform (NTT) over a prime field for bit-exact integer correlation.

**Why it may beat the state of the art.** xatlas wins on tightness through exhaustive raster testing,
but it is slow when brute-forcing large atlases. FFT gives the exhaustive search fast, and the
distance-field proximity term explicitly prefers concave nesting.

**Cost/risk.**
- Medium: a pure-Rust FFT exists, or an NTT is about 200 lines; plus padding, rotations and incremental
  occupancy updates.
- Per-chart cost is O(A log A) on the atlas area A: fine for the "few large charts" regime, too slow
  for thousands of tiny charts (hence the hybrid).
- Uncertain payoff: whether it closes the full 4-9% gap needs a benchmark.

**Absorbed?** I found no published UV packer using FFT correlation (searched; UVPackmaster's internals
are not public). Spectral packing is new even in 3D (2023).

---

## 9. (Support) Re-distancing with sub-cell fix and fast sweeping (W3)

Ideas 1, 3, 6 and 7 all edit a scalar field. Before surface nets and projection, the edited field
must be turned back into a distance field *without moving its zero crossing*. Level-set physics codes
do this as follows:
- Russo & Smereka, *A remark on computing distance functions*, J. Comput. Phys. 2000
  ([record](https://portal.mardi4nfdi.de/wiki/Publication:1587862)): the "subcell fix" uses the
  interface location implied by the neighbours to keep the zero set fixed during re-initialization.
- Zhao, *A fast sweeping method for Eikonal equations*, Math. Comp. 2005
  ([doi](https://www.doi.org/10.1090/S0025-5718-04-01678-3)): O(N) Gauss–Seidel sweeps in 2^d
  orderings. It is deterministic and branch-light, unlike heap-based fast marching.

In farthing, use exact BVH distances where the field still equals the input surface. Use the
sub-cell fix only in cells where the field was edited (plugs, MBO, heat completion), then sweep
outward. This is cheap (about 200 lines) and **standard in physics codes and OpenVDB**; it is listed
so the pipeline stays sub-voxel accurate after topology edits.

---

## Considered and not recommended

- **Quasicrystals / low-discrepancy sets for impostor view sampling (W6):** spherical Fibonacci and
  related lattices are already standard in graphics; no new transfer found.
- **Homogenization theory for LOD of fine geometry:** appearance-level homogenization (e.g. SGGX,
  aggregate BRDFs) is graphics and outside "no materials in the library". No geometric transfer
  found.
- **Phase-field / Cahn–Hilliard inpainting** (Bertozzi, Esedoğlu, Gillette 2007) for hole filling: it
  works, but it is a stiff 4th-order PDE. SHM (idea 3) plus MBO (idea 7) cover the same need more
  cheaply.
- **MD / N-body space-filling-curve particle reordering for cache locality (W1):** real gains in MD
  codes, but Morton/Hilbert vertex reordering is already standard in mesh processing (meshoptimizer).
  Worth checking that farthing sorts positions spatially before building `adj`/quadric arrays; not
  an exotic idea.
- **Fracture / elasticity simulation for topology change:** too indirect; ideas 1 and 6 give the same
  control with far less machinery.

## Suggested order of work

1. **Idea 2 + idea 9**: the sub-voxel SDF baseline. Expect most of the W3 gap on closed and mostly
   closed inputs.
2. **Idea 1**: replaces the global 10% sealing heuristic, which is likely a big Hausdorff win on
   leaky assets.
3. **Idea 4a** as a diagnostic, then refinement if flagged cells correlate with the worst Hausdorff
   cases.
4. **Idea 5** for W2. Check the `Validity` tally first to confirm that locks are the cause.
5. Ideas 3, 7, 6(a,b), then 8, then 6(c).
