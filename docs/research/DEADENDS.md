# Dead ends

Things that were tried or examined and did not pay off, so nobody spends the time twice. Each
entry says what was tried, what happened, and what would have to change for it to be worth
another look. Add to it whenever an experiment is dropped. Experiments that worked are in
[../BENCHMARKS.md](../BENCHMARKS.md); ideas not yet tried are in [README.md](README.md).

## Realtime reduction

| Tried | Result | Revisit if |
|---|---|---|
| Collapse tail worked out ahead in parallel (the next 64 heap entries per round, results kept while their neighborhoods are clean) | Slower than the serial tail. Each round wakes the pool, which costs about 70 µs on Windows, and a round's work is smaller than that | Workers that spin through the tail instead of sleeping, or rounds of thousands of independent collapses |
| Tail in rounds: pop the cheapest K entries, work out their turns (cost, checks, wedge map) in parallel, then commit in cost order those whose neighborhoods no earlier commit of the round touched. K one eighth of the collapses left, at most 2,048; results identical for any thread count (commits fd9b9b1 and 92f707e, reverted) | Slower on a 4-thread runner: tail 3.1 → 4.8 ms at a 3,000-triangle target and 15.1 → 19.1 ms at 20,000. With K fixed at 1 or 8 (serial) the restructuring alone costs 26%; at K = 64 (parallel) the tail is 50% slower than serial, since handing 64 turns to the pool costs more than working them out. Turns worked out early are also dirtied by the round's own commits and done again. Quality was unchanged (geometric means within 0.1%) | Many more cores than 4, workers that spin through the tail, or targets in the hundreds of thousands |
| Tail: first heap keys as geometry-only lower bounds, exact costs only when a cluster reaches the top (at a 20,000-triangle target only a third of the clusters costed up front ever collapse) | Slower on a 4-thread runner: tail 12.0 → 14.6 ms at 20,000 triangles, 3.1 → 3.4 ms at 3,000. The up-front exact pass runs in parallel; costing on demand moves that work into the serial loop | A single-threaded build, where it should win |
| Tail: after a collapse p → q, keeping the cached best of q's other neighbors unless it was q (their triangles and other costs are unchanged, and the cost onto q only rises) | Output bit-identical, but no faster at 3,000 triangles and 3–13% slower at 20,000 on all three assets. Not understood | Someone profiles why |
| Vertex importance from material gloss (1 + k x highlight weight, k = 4, 8, 16) to cut specular error | Specular error within 3% of unchanged in realtime and the default reducer: importance moves triangles between glossy and matte areas, and most assets are glossy all over | Assets with mixed materials |
| Gloss-scaled normal weight in realtime's tail, and the normal-spread split weight x4 and x16 | No change in specular error (1.21% to 1.26% misplaced). Realtime's normals come from the clustering and the per-wedge mean, which is already the low-pass the metric asks for | — |
| Wedge joins without the atomic swap: a chunk joins vertices of its own cells with plain stores, and the joins that reach into another chunk wait for a serial pass | Output identical, but the pass got slower on the runner (watch 0.83 → 0.94 ms, car 1.49 → 1.98 ms). The swap was not the cost; the extra branch, the deferred list and the serial pass were | — |
| Tail: a candidate's cost as Qp(q) + Qq(q), with q's own value, position and trace in a 32-byte record, in place of copying and summing q's quadric | Same output on the three large assets, but slower: with it the other tail changes' gain at 20,000 triangles fell from 6–12% to 2–6%, and to nothing at 3,000. Keeping the record current on every collapse and the extra array cost more than the 80-byte copy | — |
| Tail in passes (cost every edge, sort, apply with endpoint locks), as the fast reducer and meshoptimizer do | Not built: the lazy heap does about 6,400 exact evaluations for 1,900 collapses at a 3,000-triangle target; a pass scheme that only commits clean neighborhoods needs 5 to 6 passes of nearly everything, about three times as many, and with stale costs it is fast mode, which scores lower at these ratios | — |
| Multiple-choice collapse (best of 8 random candidates, no queue) for the tail | Not built: 8 evaluations per collapse against about 1.4 in the loop now, and it gives up the global order the tail's quality comes from | — |
| Tail: collapses costing under 5% (or 20%) of the dearest collapse's estimated cost go ahead as soon as their exact cost is known, without going back in line (about half of the collapses at a 3,000-triangle target are under 5%) | Quality unchanged (geometric means within 0.3%), and no faster: tail within 2% at 3,000 and 20,000 triangles for both settings. The push, pop and second gather it saves are cheap; the exact evaluations stay | — |
| Entering the tail at a smaller overshoot where it is large (the watch and the car enter at 2.0 to 2.2 times the target's triangles, the dragon scene at 1.2, because their cells hold several sheets): clusters per target triangle 0.55 → 0.42 → 0.32 | The assets with the large overshoot need it: watch 19.1 → 18.1 → 16.9 dB, car 23.7 → 23.2 → 22.5 dB, dragon scene 26.5 → 24.7 → 23.0 dB at 10%; mean over 11 assets 23.87 → 22.91 → 21.83 dB | — |
| One scatter on the top 10 bits of the Morton code, then per-bucket sorts, in place of three 10-bit scatter passes | Same order, no measurable change in time. Kept because it is simpler | — |
| Keeping freed memory mapped between calls (allocator purge off) to avoid page faults on 15–22 MB per call | No change in time: the faults are not the cost | The per-call allocation grows several times |
| Evaluating a cluster's representative by quadric error alone | Flat regions tie, and the tie picked vertices at cell corners, folding a flat grid's outline. Fixed with a small pull to the cluster's centroid (`CENTROID_PULL`) | — |
| More or fewer clusters per target triangle than 0.55 before the tail | Fewer leaves the tail no room and quality drops; more makes the tail the whole cost | The tail gets several times faster |
| Bounding the result's Hausdorff distance by the quadric error | The quadric error is a mean squared distance to planes, not a distance to the surface; `max_error` holds only loosely (within about 0.06 of the diagonal in the test) | A cheap one-sided distance check per cluster |
| Solving each surviving cluster's position freely against its quadric after the tail | Rendered PSNR up on untextured-looking assets (Duck +1.3 dB) but down on detailed textures (BoomBox −0.7 dB, Corset −0.5 dB): a sideways move slides the texture under attributes copied from a member vertex. Moving along the normal only keeps the gains without the losses and shipped | UVs re-fit together with the position |
| Representing a cluster that touches an open border by one of its border vertices (open fan with no vertex at the same position, so not a UV seam) | Mean rendered PSNR down 0.24 dB at both 10% and 3%: Lantern gains 1.2 dB, but WaterBottle loses 1.9 dB and Avocado 0.8 dB. The border planes already in the quadric do most of the job; forcing the choice gives up too much error elsewhere in the cluster | The cloth-edge retreat on the dragon scene is fixed some other way, such as a larger border-plane weight |
| Border planes along seams at a reduced weight (3% or 10% of the border weight) instead of none | Mean rendered PSNR 0.12 and 0.24 dB below no seam planes at 10%. The helmet prefers some weight (+0.1 dB), most assets none | Per-asset tuning |
| Telling a seam from a border by whether any vertex shares the position | Wrong on unwelded meshes, where border vertices have same-facing duplicates, and on double-sided sheets: the dragon scene's cloth edge collapsed. The test has to be per end edge (does a co-located fan end at the same place) | — |
| Unweighted mean of member normals per wedge | 0.15 dB worse than the area-weighted mean at 10% | — |
| Compute shaders for a 100k–200k triangle call | Not built. Dispatch costs tens of microseconds each and the readback waits on the browser's scheduler, which is the order of the whole 10 ms budget; published GPU decimation results are for meshes several times larger | Inputs of millions of triangles |
| Collapsing every edge that is the cheapest in its neighborhood, in parallel rounds | Not built: the pattern (propagate a candidate's cost to its neighbors, confirm, collapse) is claimed by US 12,579,751 B2, and published versions are not deterministic | A reservation scheme on vertices in a fixed prefix of the greedy order, which is a different mechanism |
| Cell sums as differences of running sums along the Morton order | Not built: claimed by US 10,417,821 B2, and differencing quadrics cancels badly. Cell sums stay hierarchical | The patent lapses (about 2036) |
| Octree clustering on the GPU | Not built: claimed by US 8,760,450 B2 | The patent lapses (about 2031) |
| Splitting the tail into spatial blocks with frozen borders | Not built: at 6,000 triangles in 8 blocks the frozen borders are a large share of all edges | — |

## Triangle reduction (default and fast)

| Tried | Result | Revisit if |
|---|---|---|
| Attribute terms evaluated in f32 with f64 sums | See "Reducer work per collapse" in the benchmarks: not faster once measured over a whole run | — |
| Scaling a part's error by its average visibility before deleting hidden parts | Worse than the per-part rule that shipped (see "Aggressive targets") | — |
| Sweeps of the UV and normal weights, `error_normalization` and the greedy order to win the last three image-space rows against meshoptimizer's permissive mode | Each sweep that won a row cost more elsewhere | A cost with texture content in it |

| Specular error compared point by point (original normal against the reduced mesh's at the closest point) | Saturates on detail no reduced mesh keeps: 63% to 72% "misplaced" on the dragon at 7° for every reducer, and a 1,000x normal weight did not move it. Replaced by comparing mean normals per reduced triangle, with lost detail widening the lobe | — |
| A global normal weight 10x to 1,000x in the default reducer, for specular error | Uneven: BoomBox 5.4% to 3.7% misplaced, the watch 10.6% to 14.3%; mean slightly worse and image score down 0.1 to 1.2 dB | The watch's response is understood |
| Gutter guard off in the default reducer | Best geometry on seam-dense meshes (dragon to 1,000: 4.38% Hausdorff against 11.2%; 8-asset means 2.11 / 0.149), but 1.4% of UV area ends up over half a texel into the gutter, worst 10.3% | A re-bake per level makes gutters moot |
| A blanket spill cap growing with the dearest collapse made (`gutter_tolerance` 0.01, 0.05; shipped for a day, removed) | Rendered PSNR 24.43 → 24.97 → 25.24 dB at 10%, but 0.51% of UV area over half a texel into the gutter at 0.05, and on the watch nearly all the spill landed inside another island. Replaced by `gutter_use`, which probes for room | — |
| Gutter room measured ahead per boundary edge (distance to the nearest other boundary, then only to boundary in front of the edge) | Too pessimistic on seam-dense assets: median clearance 0.39 texel on the dragon, 0.42 on the watch, 0 on the car, because charts are packed end to end and the nearest obstacle is beside an edge, not where a spill would go. No gain (dragon to 1,000: 6.20 / 0.640 at share 0.25). Probing the spill itself replaced it | — |
| Spill obstacles limited to islands of the same material | Flattened watch to 3,000: 7.86 / 0.806 against 9.11 / 0.928. Not adopted: two materials can share an atlas and the mesh does not say | The caller can say which materials share a texture |
| Spill price 4x (at share 0.25) | 8-asset means 2.373 / 0.1629 against 2.288 / 0.1574 at 1x: most of the gain gone. At 0x the same error with a third more spill | — |
| Lower tear price alone (`TEAR_UV` 0.1, 0.02) on the dragon | 11.16 / 0.651 (no change) and 8.65 / 0.625: no help while the gutter guard is hard | With the guard relaxed: guard off, no normal growth and `TEAR_UV` 0.05 reached fast mode's level (2.41 / 0.291) |
| Normal term not growing with region size | Dragon 4.76 / 0.513 but normals 27.7°; 8-asset normals 8.12° against 6.93° | — |
| Clusters of the cluster-parallel path stopping at 2x or 4x the final density | Dragon scene: 0.473 / 0.0158 and 0.318 / 0.0188 against 0.527 / 0.0141. Mixed | — |
| Line quadrics in fast mode (Liu, Rahimzadeh, Zordan 2025): per face corner, weight x area x squared distance to the line through the corner along the face normal; weights 0.001, 0.003, 0.01, 0.1 | 8-asset means at 0.01: Hausdorff 2.257 → 2.375, mean 0.1413 → 0.1496, UV p90 0.0093 → 0.0076, normals 8.18 → 8.24. Rendered PSNR over 11 assets 24.02 → 23.62 at 10% (23.80 at 0.003) and 18.94 → 18.84 at 3% (19.01 at 0.003). With endpoint placement the term only reorders collapses, and the order it prefers costs geometry | A reducer that places vertices; untested in the default reducer |
| Probabilistic quadrics (Trettner and Kobbelt 2020) | Not built: their gain is a well-conditioned solve for an optimal position, which half-edge collapse never does, and a later comparison has them tied with plain quadrics on Hausdorff distance | The reducer starts placing vertices |
| Intrinsic error metrics, spectral coarsening | Not built: they preserve a mesh's operators for solvers, not its appearance | — |
| Differentiable-rendering simplification inside the library | Not built: fixed topology, hours on a GPU per asset, and patented (US 11,450,077, US 11,615,602) | — |
| Neural simplification and mesh generators | Not built: no seam or attribute handling, not deterministic | — |
| Curvature or saliency weights on the quadrics | Not built: a commercial tool deprecated its own, and scaling a quadric does nothing on flat regions, where the ties are | — |
| Texture-content weighting of UV error | Not built: no source reports image-space gains, and it conflicts with visual masking (busy texture hides error) | A color-at-closest-point metric shows headroom |
| Optimal vertex placement during collapse | Not built: an older study found no significant gain in rendered images over choosing an endpoint, and the trial refit in the thin-features work was dropped | — |

## Voxel proxies

| Tried | Result | Revisit if |
|---|---|---|
| A second pass opening boundary cells whose center is outside the source (v13 slot pass) | Reverted; see "Voxel proxies" in the benchmarks | — |
| A guarded carving pass over solid cells outside the source | Dropped; see the same section | — |

## UV atlases

| Tried | Result | Revisit if |
|---|---|---|
| FFT correlation placement of charts against an occupancy map | Not implemented: claimed by US 12,112,510 B2 | The patent lapses |

## Tooling

| Tried | Result | Revisit if |
|---|---|---|
| Timing on the development machine while other work runs | Phase times move by 2× between runs; only minimums over many runs while the load is under about 35% mean anything | — |
| Remote agent isolation for timing | Fails to start on this checkout (the worktree path differs from git's by drive-letter case). The `Bench` workflow on a hosted runner is used instead | — |
