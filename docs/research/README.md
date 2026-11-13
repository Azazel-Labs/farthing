# Ideas from outside graphics

Five research passes (November 2026) looked for recent or under-used discoveries in other
sciences that could help farthing's weak spots. The shared brief, with the weak spots W1–W7, is
in [BRIEF.md](BRIEF.md). Each report gives sources, the mechanism, the transfer to farthing,
cost, and whether graphics has already absorbed it.

| Report | Fields |
|---|---|
| [physics_math.md](physics_math.md) | Level sets, fast winding numbers, signed heat method, multigrid, threshold dynamics |
| [biology_materials.md](biology_materials.md) | Protein docking, jamming, colloids, cut-or-fill topology repair |
| [perception.md](perception.md) | Contrast sensitivity, silhouettes, LOD popping studies |
| [optimization_topology.md](optimization_topology.md) | Persistent homology, non-manifold link conditions, nesting, parallel greedy |
| [hpc_architecture.md](hpc_architecture.md) | Profiled hotspots, deterministic reservations, branch-free kernels |

## Where several reports agree

- **Seal leaks per pocket, not globally (W3, W7).** Two reports independently propose ranking
  air pockets of the voxel distance field by neck width with union-find (0-dimensional
  persistence) and plugging only real leaks. This replaces v7's all-or-nothing
  `adaptive_gaps`. The physics report adds a robust sign from fast winding numbers and the
  signed heat method for large holes.
- **Collapse along non-manifold junctions instead of locking them (W2).** The topology report
  proposes the link condition for non-manifold complexes (Dey et al. 1999).
- **Deterministic parallel commits (W1).** Two reports point at deterministic reservations
  (Blelloch et al., PPoPP 2012) to run each batched round's collapses in parallel with today's
  exact order.
- **Silhouette-aware, screen-size-aware error (W5).** The perception report's cheapest win:
  scale each face's plane quadric by its expected silhouette length (dihedral angle / π), and an
  optional screen-size view for a contrast-sensitivity weighted cost.

## Legal note

FFT correlation placement of UV charts against an occupancy map, proposed in two reports, is
claimed by US 12,112,510 B2 (Tencent America, priority 2022, granted 2024). It is not to be
implemented. Contact-scored placement over free-space pockets, also proposed, is the
alternative to explore, and would get its own check first. All patents farthing avoids are in
the README's patent shame list.

## Status

| Idea | Status |
|---|---|
| Packing outlines from edges, not pixel masks (HPC #1) | Done in v8: Avocado UV 89 → 19 ms |
| Linear-time sparse structure, small-key hashing (HPC #2) | Done in v8, bit-identical |
| Whole-part removal during reduction (W2, W7; our own design prompted by the stall analysis) | Done in v8 |
| Per-pocket leak sealing (simplified: keep only deep regions of what closing adds) | Done: DamagedHelmet proxy mean error −7%, dragon Hausdorff −5% |
| Non-manifold junction collapses | Next |
| Deterministic reservations | Planned |
| Expected-silhouette quadric weight | Done (weight 0.5): Hausdorff −3%, UV −3%, image space +0.10 dB |
| Contact-scored pocket placement (W4) | Planned, after the patent-safe design is checked |
| Surfel proxies with visibility weighting (user idea, perception + relighting research) | Prototype in v9: beats the triangle LOD by 1.7–2.9 dB on a bush under moving light |
