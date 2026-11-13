# Ideas from optimization, topology and data mathematics for farthing

Assigned areas: optimal transport, topological data analysis (TDA) and persistent homology, compressed
sensing, the statistical physics of optimization, submodular optimization, randomized numerical linear
algebra (RandNLA) and coresets, matroid and independent-set scheduling, and cartographic generalization.
Focus: W2, W7, W1, W4.

Before proposing changes I read the relevant farthing code, so each transfer points at real code:
- `farthing/src/reduce/simplify.rs`: `State::new` locking, `valid()`, `run_batched`, the half-edge
  collapse policy
- `farthing/src/reduce/parts.rs`: part removal
- `farthing/src/voxel/mod.rs`: `classify`, `close_gaps`, `adaptive_gaps`
- `farthing/src/uv/pack.rs`: the skyline packer
- `docs/BENCHMARKS.md`

I also read meshoptimizer's `simplifier.cpp` (current master) so I could say honestly what graphics
practice already does.

Note: the reducer source changed while I was reading it (a `diagnose()` routine and `parts.rs`
appeared), so line references may drift.

## Ranked summary

| # | Idea | W | Payoff | Effort | Already in graphics? |
|---|---|---|---|---|---|
| 1 | Seal each pocket using persistent homology of the air distance field (0-dim persistence, union-find), plus closing of tunnels under topology control | W7, W3 | high | low–medium | The parts exist (TDA, digital topology, Ju group 2020–22). I found no proxy/LOD tool doing per-pocket sealing |
| 2 | Generalized-boundary link condition: junction curves collapse like borders instead of locking | W2 | high (hits the Lantern failure directly) | medium | The theory dates from 1999. Production tools either lock (farthing) or allow it with no guarantee (meshoptimizer) |
| 3 | Exact integer (NTT/FFT) correlation placement for the K largest charts, from "spectral packing" | W4 | medium–high | medium | Published for 3D printing (TOG 2023). I found no UV-packing use |
| 4 | (1+ε)-threshold rounds instead of the 1/8 quantile. Order commits by locality inside a round. Optional deterministic-reservation commit | W1 | medium (uncertain) | medium | Partly: meshoptimizer's 12-bit radix passes and PaMO's local-minimum rounds are close relatives |
| 5 | Overlap-elimination guided local search (sparrow / jagua-rs, Rust) as a time-boxed post-pass for atlases with few charts | W4 | medium | medium–high | Comes from operations research (OR), 2025. Not in UV tools as far as I found |
| 6 | Kernel-constrained placement: a tiny convex QP / LP over the no-flip polytope, used only to escape stalls | W2 | medium | low–medium | 2D-LP precedent (Cohen–Manocha–Olano). CGAL and meshoptimizer reject the collapse instead |
| 7 | Cartographic generalization: cell-decomposition aggregation (Kada) for architectural proxies, and Töpfer's radical law for budgets of retained parts | W7 (W2) | low–medium | high / tiny | Known in GIS and in 3D city models (CityGML). Rare in game LOD |
| – | Considered and rejected: OT-driven simplification, compressed sensing, coresets, statistical-physics annealing, RandNLA solvers | – | – | – | see end |

---

## 1. Seal each pocket using persistent homology of the distance field (W7; also W3)

**Sources**
- Edelsbrunner, Letscher and Zomorodian (2002) introduced persistence. Boutry et al. later proved that
  morphological *dynamics* (watershed significance, Grimaud 1992) is equivalent to 0-dimensional
  persistent homology in n-D ([LRDE DGMM 2021](https://www.lrde.epita.fr/wiki/Publications/boutry.21.dgmm.1),
  [JMIV 2023 version](https://www.lre.epita.fr/dload/papers/boutry.23.jmiv.pdf)). Union-find flooding
  therefore computes exact 0-dim persistence.
- Graphics state of the art for topological simplification of voxel shapes: Zeng, Chambers, Letscher
  and Ju, "To Cut or to Fill", ACM TOG 39(6) 2020. It finds a globally optimal set of cuts and fills,
  framed as graph labelling and reduced to the node-weighted Steiner tree problem
  ([WUSTL record](https://profiles.wustl.edu/en/publications/to-cut-or-to-fill-a-global-optimization-approach-to-topological-s/)).
  A nested-shape follow-up appeared in CGF 41(5) 2022
  ([EG diglib](https://diglib.eg.org/items/ea08fbe1-dfb0-4afd-a217-2f78cc77f400/full)). Chambers, Ju et
  al. (CCCG 2018) propose a persistence-based candidate heuristic for the NP-hard homological
  simplification problem ([NSF PAR](https://par.nsf.gov//servlets/purl/10095830)).
- Engineering meshing: "Tusqh" (Shawcroft, Shepherd, Mitchell, arXiv 2502.10609, 2025)
  ([arXiv](https://arxiv.org/abs/2502.10609)). It controls output topology with persistent homology
  over a volume-fraction filtration on a background grid. Volume fractions come from generalized
  winding numbers. It also removes pinches with "topological anti-aliasing". The threshold is picked
  by an analyst, globally. The authors call the prototype slow (a Grasshopper plugin with PHAT).
- Topology-preserving growth in medical imaging: simple-point tests (Bertrand and Malandain 1994;
  cited from memory, not re-verified), Han–Xu–Prince TGDM (2003), and Ségonne's genus-preserving level
  sets, IJCV 2007 ([PDF](https://surfer.nmr.mgh.harvard.edu/ftp/articles/2007/2007_-_Segonne_-_International_Journal_of_Computer_Vision.pdf)).

**Mechanism**
- Compute the exact Euclidean distance d from the surface voxels.
- Visit air voxels in decreasing d and union them with already-visited neighbours, using the elder
  rule.
- Each air pocket is born at its inscribed radius b and dies at the neck radius s where it joins the
  outside-seeded component. The voxel where the join happens is the neck's location.
- This yields a per-feature list (b, s, saddle voxel, volume) in O(n α(n)) after a radix sort of the
  quantized d.

**Transfer (in `farthing::voxel::classify`)**

Today `classify` does a global dilate by `close_gaps`, then a flood, then an erode. `adaptive_gaps`
makes one all-or-nothing choice: it seals everywhere or nowhere, depending on whether the unsealed
solid is less than 90% of the sealed one. The proposed replacement:

1. Voxelize with `close_gaps = 0`. Compute an exact EDT (separable, linear time, e.g.
   Felzenszwalb–Huttenlocher) from the surface voxels.
2. Run the 0-dim persistence pass on air, with the outside seeds (grid border, clip cells) as the
   oldest component.
3. Mark a pocket as a *leak* when s ≤ τ_leak (default about 1–1.5 cells, i.e. what `close_gaps=1`
   does today) and b/s ≥ κ (a big interior behind a small opening). τ_leak can also come from the
   proxy's screen size, which makes this a W7 dial.
4. Plug each leak locally: block the air voxels within s+1 cells of its saddle voxel, then re-flood
   from outside. Everything else keeps its unclosed shape. No global erosion is needed.
5. Optional W7 extension for tunnels (handles through the solid: windows, gaps between chair slats,
   mug handles):
   - Grow the solid in increasing d with a simple-point test.
   - A non-simple voxel whose addition lowers β1 (from the local Euler-characteristic change plus
     6/26-connectivity counts) closes a tunnel at radius d. Accept it only if d ≤ τ_tunnel(LOD).
   - Never accept additions that *create* handles or pinches. This plays the role of Tusqh's
     topological anti-aliasing.

**Why it might beat the current approach**
- Global closing fills every crevice of about 2 cells. That is part of the W3 worst-case error
  budget, and the current adaptive switch can only turn closing off entirely.
- Per-pocket plugs change only the voxels at real leaks. Outside a leak's plug, the solid equals the
  open classification.
- It also gives the user a meaningful and monotone control: "seal interiors reachable only through
  openings ≤ X". Examples are a building's doors and windows at far proxies, or a hollow prop with a
  pinhole.
- Compared with Zeng 2020: their global NWST optimization is far heavier and is aimed at
  medical/biological noise. Proxies mostly need *fills* (fill-only is the easy half), and they need a
  time budget in milliseconds.

**Cost / risk**
- The core (EDT, radix sort, union-find, plug, re-flood) is roughly 300 lines of Rust. It runs in
  tens of ms at 128³. It is deterministic if sort keys break ties by voxel index.
- Risk: conservative voxelization creates many tiny pockets. The b/s filter handles this, but the
  threshold κ needs tuning on the kitbash and remesh suites.
- The tunnel extension needs a correct 3D simple-point test (26/6 adjacency). That is well documented
  but fiddly: medium effort.
- Full H1 persistence *pairing* (Cubical Ripser, [Kaji et al. 2020](https://arxiv.org/abs/2005.12692);
  Robins–Wood–Sheppard discrete Morse, TPAMI 2011) is not needed for fill-only decisions. I'd avoid it.

**Already absorbed?**
- Persistence and dynamics are standard in TDA and image analysis.
- Topological simplification of voxel shapes is published graphics research (Zeng/Ju 2020–22).
- I found no public game-proxy or LOD remesher that seals pockets one by one using persistence.
  Simplygon's remesher is closed, so this is uncertain.
- Kissi, Pont, Levine and Tierny ([arXiv 2407.12399](https://arxiv.org/pdf/2407.12399), a TTK
  module) can edit a scalar field so that only chosen persistence pairs survive. It handles 3D saddle
  pairs, so it could act on an SDF before contouring and pair with an SDF-based W3 fix. Their reported
  times are minutes at 256³, too slow for a default path.

## 2. Generalized-boundary link condition: collapse along non-manifold junction curves (W2)

**Sources**
- Dey, Edelsbrunner, Guha and Nekhayev, "Topology preserving edge contraction", Publ. Inst. Math.
  (Beograd) 66(80), 1999
  ([abstract](https://emis.univie.ac.at/journals/PIMB/080/3.html)). It gives local conditions, based
  on a *generalized notion of boundary*, under which contractions in simplicial complexes preserve
  topological type.
- Per the survey in Dey, Hirani, Krishnamoorthy and Smith ([arXiv 1304.0664](https://ar5iv.labs.arxiv.org/html/1304.0664)):
  - the link condition preserves homeomorphism for 2-complexes and 3-manifolds;
  - Attali, Lieutier and Salinas extended it to homotopy preservation for all finite simplicial
    complexes.

**Mechanism**
- Lk(a) ∩ Lk(b) = Lk(ab), evaluated on the full complex (vertices *and* edges), together with a
  hierarchy of generalized boundaries.
- In a 2-complex, edges with ≥3 incident faces form "junction curves". These play the same role as
  border curves: a vertex in a curve's interior may contract only along that curve.

**Transfer**

Today, `State::new` locks both endpoints of every edge with face count > 2 (lines ~586–589), and also
any border vertex of border-degree > 2. Under aggressive targets, every T-junction seam (glass in a
frame, intersecting kitbash sheets welded by position) becomes a frozen polyline. That is the
documented Lantern failure. The change:

1. Classify edges by face count: 1 = border, 2 = manifold, ≥3 = junction (keep the multiplicity).
2. A vertex with exactly two incident junction edges of the *same multiplicity* becomes a
   "junction-interior" vertex. Generalize `border`/`bnbr` to a "feature curve" class and neighbour
   list, so `valid()` allows p→q only along a junction edge.
3. Endpoints and branch points (junction degree 1 or ≥3, or mixed multiplicity) stay locked, like
   today's border-degree > 2 rule.
4. Extend the link check (the `common > shared` count in `valid()`) to the complex version:
   - common neighbours of p and q must equal the opposite vertices of all triangles on pq (k of them
     for a junction edge of multiplicity k);
   - common *edges* in the two links must be empty, the 1-dimensional part of Lk(ab).
5. Add a junction-curve quadric: perpendicular planes per incident sheet, the same construction as
   border quadrics, so the junction polyline keeps its shape.

**Why it might beat the current approach**
- It removes the specific lock that stalls 1% targets, and it keeps a topology guarantee.
- meshoptimizer takes the permissive route instead. Its `Kind_Complex` vertices "can move as long as
  all wedges move to the target vertex" (from the `simplifier.cpp` comments). That reaches targets
  but gives no topological guarantee.

**Cost / risk**
- About 200–300 lines in `State::new`, `valid()` and the quadric setup, plus tests (a T-junction of
  three sheets, a book-binding of k pages).
- Main risk: `map_wedges` and corner fans assume at most two sheets per edge. They need per-sheet fan
  enumeration around junction vertices.
- `parts.rs` part removal still backstops whatever remains.
- Measure first: `diagnose()` already tallies `Locked`, `Link` and `Flip`. Count how many surviving
  positions on Lantern are locked junction-interior vertices.

**Already absorbed?**
- The theory has existed since 1999. Popović and Hoppe's Progressive Simplicial Complexes (1997)
  simplify arbitrary complexes, but deliberately change topology.
- The production simplifiers I checked either lock (farthing) or collapse without a guarantee
  (meshoptimizer).
- I could not check Simplygon (internals out of scope).

## 3. Exact correlation ("spectral") placement for the largest charts (W4)

**Source**

Cui, Rong, Chen and Matusik, "Dense, Interlocking-Free and Scalable Spectral Packing of Generic 3D
Objects", ACM TOG 42(4) 2023
([PDF](https://cdfg.mit.edu/pdf/journal/2023-cui-dense-interlocking-free-and-scalable-spectral-packing-of.pdf)):
- Collision for *all translations at once* is a correlation of voxel grids, computed by FFT in
  O(n log n). Zero entries mark collision-free offsets.
- Tightness comes from correlating the object with the *distance field* of what is already packed,
  plus a height penalty.
- Orientations are sampled. Objects are placed largest first.
- They report 9.4 s brute force vs 0.003 s FFT for one collision query on a 240×123×100 grid.

**Mechanism**

Placing chart A on an occupancy bitmap O:
- overlap(t) = (O ⋆ A)(t);
- tightness(t) = (D_O ⋆ A)(t), where D_O is a capped distance field of O.

Both are convolutions, so one transform pair scores every position.

**Transfer**

This applies to `farthing::uv::pack_charts` for the "few large irregular charts" regime where xatlas
wins (WaterBottle, Fox). The documented failed attempt was an exact bitmap packer that took
2.4–17.9 s because it tested positions one by one. Correlation evaluates all positions in a few
transforms.

1. Use the correlation placer for the K largest charts (e.g. every chart when there are ≤ 32 charts,
   or until 80% of the area is placed). The skyline packer handles the small remainder.
2. Run hierarchically to stay within the time budget:
   - Max-pool masks 4× and correlate at 256² (dilate the chart by one coarse cell so sub-cell
     offsets stay conservative). Zero entries at coarse level are then guaranteed free at full
     resolution.
   - Refine the best few candidates with the existing exact mask test at full resolution.
3. Score = tightness + λ·(distance from the atlas origin, or skyline height). This directly rewards
   nesting a chart inside another chart's concavity, which is the documented gap.
4. Use a number-theoretic transform (NTT, e.g. modulo 998244353) instead of a float FFT. All inputs
   are small integers (mask bits, capped integer distances), so results are *exact*. That keeps the
   output bit-identical across CPUs and SIMD paths, which a float FFT with runtime ISA dispatch would
   not guarantee.

**Why it might beat the current approach**
- Skyline cannot place a chart under an overhang or inside a concavity; correlation can, by
  construction.
- The cost does not depend on the number of candidate positions, which is what killed the exact
  bitmap packer.
- xatlas uses brute-force bitmap placement with random or limited candidate positions (from memory
  of its source; not re-verified here).

**Cost / risk**
- An NTT (radix-2, 256² or 512²) plus the hierarchy is about 400 lines of Rust with no dependencies.
  `rustfft` (pure Rust) is a fallback if a float FFT is acceptable for prototyping.
- Rough budget: about 3 transforms per chart and rotation at 256² is about 1 ms each, so 4 rotations ×
  20 charts ≈ 0.1 s. That is comparable to the current 94–137 ms for WaterBottle. This is an
  estimate, not measured.
- Risk: greedy largest-first order may still leave a gap. Cui et al. found beam search over the
  order added a little density at about 4× the cost.

**Already absorbed?**
- Published in graphics for 3D printing bins.
- The UV-packing literature I found uses no-fit polygons, learning, or chart cutting instead:
  - Yang et al., "Learning based 2D irregular shape packing", TOG 42(6) 2023
    ([arXiv](https://arxiv.org/pdf/2309.10329)), reports 5–10% over xatlas but is about 20× slower;
  - Box Cutter (Limper et al. 2018);
  - Liu et al. 2019.
- I found no FFT/NTT UV packer, but cannot rule out commercial ones (e.g. UVPackmaster).

## 4. Parallel greedy with a quality guarantee: (1+ε) thresholds, locality-ordered commits, deterministic reservations (W1)

**Sources**
- Blelloch, Peng and Tangwongsan, "Linear-work greedy parallel approximate set cover and variants",
  SPAA 2011 ([PDF](https://ldhulipala.github.io/readings/manis.pdf)). Items are bucketed by (1+ε)
  utility classes, and each bucket is processed with a maximal nearly-independent set (MaNIS). This
  recovers the sequential greedy bound up to (1+ε) in polylog depth.
- Badanidiyuru and Vondrák, "Fast algorithms for maximizing submodular functions", SODA 2014
  ([dblp](https://dblp.dagstuhl.de/rec/conf/soda/BadanidiyuruV14.html)): decreasing-threshold greedy.
- Blelloch, Fineman, Gibbons and Shun, "Internally deterministic parallel algorithms can be fast",
  PPoPP 2012 ([PDF](https://jshun.csail.mit.edu/determ.pdf)). Deterministic reservations: each round
  takes a prefix in priority order, iterates priority-reserve their data in parallel, and winners
  commit. Same output on any schedule.
- Blelloch, Fineman and Shun 2012 ([arXiv 1202.3205](https://arxiv.org/pdf/1202.3205)): with
  *random* priorities, greedy maximal independent set (MIS) has polylogarithmic dependence length.

**Mechanism**

Within a cost class [τ, (1+ε)τ) the order is irrelevant up to (1+ε). The scheduler is therefore free
to choose any conflict-free subset and any execution order inside it.

**Transfer (in `run_batched`)**

Today the round limit is the m-th cheapest cost, with m = min(needed, 1/8 of candidates). The 1/8 was
tuned empirically, and the round's collapses run sequentially in cost order. The proposal:

1. Set the round limit to (1+ε)·(cheapest live candidate), with a floor on the round size so
   near-zero-cost flat regions don't degenerate. Every executed collapse then costs at most (1+ε)
   times the cheapest available at round start. This guarantee is independent of the cost
   distribution; the 1/8 quantile carries no such guarantee.
2. Within a round, sort the winners by position index (or Morton code) instead of by cost. Today's
   `list.sort_unstable_by(cmp)` walks memory in cost order, i.e. randomly. If vertices are
   spatially coherent (after a vertex-fetch-style reorder), index order turns the collapse loop's
   random accesses into near-sequential ones. This targets the W1 cache-miss bottleneck. The
   approximation guarantee is what licenses the reordering.
3. Optional, for multi-core: select winners with deterministic reservations. Each candidate
   priority-writes the key (cost bucket, hash(p)) into its 2-ring. Candidates that hold all their
   reservations have disjoint 2-rings, so they can collapse in parallel. Hash priorities inside a
   bucket give polylog round counts, per Blelloch et al. 2012.

**Why it might beat the current approach**
- (1) replaces a tuned constant with a guarantee.
- (2) is a direct lever on the stated W1 bottleneck: cache misses in the collapse loop.
- (3) makes the commit phase parallel. On the 2-core benchmark box that is at most about 2× on that
  phase, so (2) matters more there.

**Cost / risk**
- (1) and (2) are about 50 lines. (3) is about 300 lines plus careful handling of `touched` and
  `checked_at`.
- The quality effect is unknown. The 1/8 rule already matched exact greedy on the benchmark, so the
  win is speed and robustness, not quality.
- Prototype (2) first and measure.

**Already absorbed?** Largely, in spirit:
- meshoptimizer's simplifier works in passes. It radix-sorts collapses on a 12-bit key from the float
  cost, executes them in order while skipping any collapse whose endpoints were already used in the
  pass (`collapse_locked`), and stops a pass at 1.5× the error of the goal-th collapse
  (`simplifier.cpp`, `performEdgeCollapses`). That is an ε≈0.5 bucketed MIS round.
- PaMO (Pacific Graphics 2025, [arXiv 2509.05595](https://arxiv.org/pdf/2509.05595)) selects local
  minima with 64-bit (cost, id) atomicMin keys on the GPU, and states no quality bound.
- What is *not* absorbed: the explicit (1+ε)-of-minimum guarantee, and using that freedom to reorder
  for cache locality.

## 5. Overlap-elimination local search from nesting research, as a time-boxed post-pass (W4)

**Sources** (KU Leuven)
- Gardeyn, Vanden Berghe and Wauters, "An open-source heuristic to reboot 2D nesting research",
  EJOR, [arXiv 2509.13329](https://arxiv.org/html/2509.13329v3). The solver is sparrow.
- "Decoupling geometry from optimization in 2D irregular cutting and packing problems: an
  open-source collision detection engine", [arXiv 2508.08341](https://arxiv.org/html/2508.08341v4).
  The engine is jagua-rs.
- Both are **Rust**:
  - sparrow: MIT license (checked its LICENSE and Cargo.toml);
  - jagua-rs: MPL-2.0 (checked LICENSE).

**Mechanism**
- Strip packing becomes a sequence of fixed-size feasibility problems.
- Guided local search uses dynamic item-pair weights. Colliding items are re-sampled (globally and
  near their current spot) and refined by coordinate descent.
- Overlap is measured with pole-of-inaccessibility circles.
- Collision checks use a quadtree of "hazards" plus fail-fast surrogates (inner circles, and "piers"
  for thin parts).
- Reported expected density gains over the previous state of the art are about +0.3 to +2.4 points
  on standard instances. Those runs used **20-minute** budgets on 3 threads.

**Transfer**

Applies only to atlases with few charts (the WaterBottle and Fox regime, at most a few dozen charts):
1. Take the skyline (or idea #3) layout at scale s.
2. Raise s by 1–2%, i.e. shrink the container in chart units, which creates overlaps.
3. Eliminate the overlaps with sparrow-style guided local search under a budget of 50–200 ms and
   4 rotations.
4. Repeat while it succeeds.

Seed the random number generator from the asset hash and run single-threaded for determinism.

**Why it might beat the current approach**
- This is the current OR state of the art for exactly the irregular-nesting problem that skyline
  approximates.
- Starting from a good layout with a small shrink step is the cheap regime of such searches.

**Cost / risk**
- The time budget is the risk: published numbers are minutes, not milliseconds. Gains under 100 ms
  are unmeasured.
- Using jagua-rs directly brings MPL-2.0 (file-level copyleft; fine as a dependency of an MIT crate,
  but a policy decision). A clean-room re-implementation of the pole surrogate plus edge tests is
  roughly 1–2k lines.
- Do idea #3 first. Use this only if a gap remains.

**Already absorbed?** Not in UV tools that I found. Nesting metaheuristics are standard in OR.

## 6. Kernel-constrained placement as a stall escape (W2)

**Sources**
- Convex QP / LP in 3 variables (Goldfarb–Idnani dual active set; Seidel's randomized LP).
- Graphics precedent: Cohen, Manocha and Olano, "Successive Mappings" (IJCGA; Vis'97 origin)
  ([PDF](https://userpages.cs.umbc.edu/olano/papers/SuccessiveMapping.IJCGA.pdf)). They use 2D linear
  programming to place the new vertex in the kernel of the projected star, which guarantees a
  bijective mapping.
- Current practice rejects instead:
  - CGAL's `Bounded_normal_change_filter` "rejects the placement if any triangle in the profile
    changes the normal by more than 90 degree"
    ([CGAL 6.0 doc](https://doc.cgal.org/6.0/Surface_mesh_simplification/classCGAL_1_1Surface__mesh__simplification_1_1Bounded__normal__change__filter.html));
  - meshoptimizer skips on `hasTriangleFlips`.

**Mechanism**
- For a triangle (x, b, c) with old normal n0, the no-flip condition n0·((b−x)×(c−x)) ≥ δ expands to
  x·((b−c)×n0) + n0·(b×c) ≥ δ. That is **linear in x**.
- So the flip-free placements form a convex polytope (the kernel of the merged star).
- farthing's `min_face_dot` cone bound can be inner-approximated by a few extra half-spaces.
- Minimizing the (PSD) quadric over this polytope is a 3-variable QP with about 10–20 constraints:
  microseconds per call.

**Transfer**

farthing uses **half-edge** collapses (q keeps its input position, and `vertex_remap` promises exact
copies). So this cannot be the default without breaking that invariant. Proposal:
- In the stall-escalation branch of `run_batched` (stall ≥ 1), and only for candidates `diagnose()`
  classifies as `Flip`, allow a *full* collapse to the QP-optimal kernel point.
- Attributes come from the attribute-quadric optimum, or from barycentric interpolation at the
  closest source point.
- If the LP finds the kernel empty, no placement can fix that collapse. That tells you an edge flip
  (Hoppe-style) or a part removal is needed instead.

**Why it might help**
- It turns some `Flip` rejections at 1% into valid collapses at slightly higher error, exactly where
  progress stalls.
- The LP emptiness test is also a cheap diagnostic separating "flip guard too strict" from "geometry
  truly folded".

**Cost / risk**
- About 200 lines (tiny active-set QP, fixed iteration order, so deterministic).
- Risk: it breaks the exact-copy guarantee for the affected vertices, which affects geomorph remap.
  Gate it behind an option.
- Payoff is lower than #2: on Lantern the non-manifold locks probably dominate. Check with
  `diagnose()`.

**Already absorbed?** Partially (the 1990s LP-in-kernel idea). It is not in QEM production
simplifiers that I checked.

## 7. Cartographic generalization (W7, minor W2)

**Sources**
- Kada, "Aggregation of 3D buildings with a hybrid data model", ISPRS 2010
  ([PDF](https://www.isprs.org/proceedings/XXXVIII/part4/files/Kada.pdf)). Planes fitted to façades
  and roofs split a block into cells. Cells are kept by raster overlap with the footprint, and
  morphological closing first bridges gaps between buildings. In their example, overlap of critical
  cells rose from 78% to 98%. One tolerance controls detail.
- Dyken, Dæhlen and Sevaldrud, "Simultaneous curve simplification", J. Geogr. Syst. 11(3) 2009
  ([RePEc](https://econpapers.repec.org/RePEc:kap:jgeosy:v:11:y:2009:i:3:p:273-289)). It simplifies
  networks of curves without changing their topological relations. CGAL's
  `Polyline_simplification_2` cites it.
- Töpfer and Pillewizer's radical law, "The principles of selection", Cartographic J. 1966 (from
  memory, not re-verified): retained count n_f = n_a·sqrt(M_a/M_f).

**Transfers**
- (a) W7, hard-surface and architectural proxies: plane-arrangement cell decomposition, with cells
  classified by the voxel solid from idea #1 (which already gives closing and sealing). This yields
  crisp, low-poly, watertight building proxies whose topology simplifies with the tolerance, as a
  third remesh mode next to surface nets. Effort is high (robust plane arrangement). Geometry
  processing has related methods (PolyFit 2017, Kinetic Shape Reconstruction 2020), so this is
  partly absorbed.
- (b) W2 and aggregation: use the radical law as a *default budget* for how many separate small parts
  survive when the screen size shrinks by a factor k (keep about 1/sqrt(k) of them). It would rank
  input to `parts.rs::plan_removal` rather than only reacting to triangle stalls. Effort is tiny.
  The payoff is empirical; it is a cartographic rule of thumb, not a theorem.
- (c) W4 support: topology-consistent curve-network simplification (Dyken) gives conservative
  simplified chart outlines, for faster collision in ideas #3 and #5. jagua-rs already includes
  polygon simplification.

## Considered and rejected (honest negatives)

- **Optimal transport for simplification and topology change.** Digne, Cohen-Steiner, Alliez, de Goes
  and Desbrun (JMIV 2014; [PDF](https://perso.liris.cnrs.fr/jdigne/articles/jmiv.pdf)) drive
  half-edge collapses by transport cost and naturally produce non-manifold or topology-changing
  complexes. But they report about 10 h (2 h on 8 cores) for 30k points and use local LPs (Clp). The
  attractive part, a mass-weighted cost that tolerates merges, is already approximated by QEM on
  virtual edges plus the part-removal error. Sliced or unbalanced OT did not map to any W.
- **Compressed sensing / sparse approximation.** Sparse-proxy ideas in graphics (VSA, L0 smoothing)
  are absorbed and do not touch W1, W2, W4 or W7.
- **Coresets / RandNLA.** QEM quadrics are 4×4 to (3+k)², so there is nothing to sketch.
  `parts.rs` already subsamples 256 vertices for removal error, which is effectively a coreset. For UV
  solves, approximate-elimination Laplacian solvers (Kyng–Sachdeva 2016; Gao–Kyng–Spielman 2023)
  would only matter for charts too large for the existing envelope Cholesky. I did not dig further.
- **Statistical physics (annealing, message passing).** Stalls at 1% look like "jamming", but I found
  no published evidence that Metropolis-style acceptance of temporarily invalid collapses beats
  structural fixes (#2, #6, part removal, edge flips). Chart segmentation by message passing or graph
  cuts is already standard in graphics.
- **Submodular / matroid.** The useful content is the threshold-greedy pattern, folded into #4. Bucket
  selection is a matching / independent-set problem, not a matroid intersection with extra structure.
  Part removal minimizes the *maximum* error, so the existing sort-by-error greedy is already optimal
  for that bottleneck objective. Submodularity adds nothing there.

## Uncertain claims to verify before acting

- The xatlas packer internals (#3) are from memory.
- The Bertrand–Malandain and Töpfer citations are from memory.
- For Dey et al. 1999, I verified only the abstract. The exact generalized-boundary rule for
  junction-interior vertices in #2 is my practical reading of it, not quoted from the paper.
- The time estimates in #3 and #5 are not measured.
- "Not found in UV packing / proxy tools" means not found by my searches. Closed tools (Simplygon,
  UVPackmaster) could not be checked.
