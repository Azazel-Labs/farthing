# Benchmarks: farthing vs open-source alternatives

All numbers come from `bench-compare/`, a separate workspace that links the competitors only to
measure against them. Raw data: [`benchmarks/results-reduce.json`](benchmarks/results-reduce.json),
[`benchmarks/results-uv.json`](benchmarks/results-uv.json),
[`benchmarks/results-remesh.json`](benchmarks/results-remesh.json),
[`benchmarks/image-space-vs-meshopt.jsonl`](benchmarks/image-space-vs-meshopt.jsonl). Machine: 2 CPU cores (cloud VM). Timings
are noisy at the ±20% level; quality numbers are deterministic.

```sh
cd bench-compare
cargo run --release -- reduce <assets...>   # reduction table + summary
python3 pairwise.py results-reduce.json      # matched-triangle-count comparison
cargo run --release --features xatlas -- uv <assets...>   # UV atlas table vs xatlas (needs libclang)
cargo run --release -- remesh <assets...>    # voxel proxies vs baby_shark
RD_RATIOS=0.01,0.003,0.001 cargo run --release -- reduce <assets...>   # aggressive targets
LF_OPTS='{"error_normalization":1.0}' cargo run --release -- reduce ... # option overrides
cargo run --release -- pack                  # rectangle packers
cargo run --release -- clusters <assets...>  # cluster LOD DAGs vs a clusterlod.h-style build
(cd ../engine-poc && cargo run --release -- layered)      # one- vs two-layer impostors
(cd ../engine-poc && cargo run --release -- surfel_lod)   # progressive surfels
cargo run --release -- export <asset> <ratio> <dir>   # LODs for image-space scoring, then:
(cd ../engine-poc && cargo run --release -- score <asset> <dir>/*.glb)
```

Assets: Khronos glTF samples (DamagedHelmet, Fox, Duck, Lantern, WaterBottle, Avocado, BoomBox)
and meshoptimizer's `pirate.obj`. Not bundled.

## Triangle reduction

Competitors: **meshoptimizer** 1.3 (via the `meshopt` 0.7 crate), in three modes (positions only;
attribute-aware with gltfpack-style weights 0.5 for normals, 1.0 for UVs; the same plus
`Permissive`), and **baby_shark** 0.3.12 (geometry only).

Metrics, all relative to the bounding-box diagonal where they are distances:

- **Hausdorff / mean**: symmetric sampled surface distance (40k samples each way).
- **UV p90**: 90th percentile of |UV(original sample) − UV(closest point on the LOD)|.
- **Normal**: mean angle between the original's normal and the LOD's interpolated normal at the
  closest point.

Each asset is reduced to 50%, 20%, 5% and 1%. A strict reducer can stall above the target, and
the stalled result then looks better because it has more triangles. So the fair comparison only
uses the cases where both reducers land within 3% of the same triangle count.

### Matched triangle counts (farthing ÷ other, geometric mean; below 1 means farthing is better)

| vs | cases | Hausdorff | mean | UV p90 | normal | time |
|---|---|---|---|---|---|---|
| meshoptimizer, attribute-aware | 20 | **0.69×** (16/20 wins) | **0.41×** (20/20) | **0.27×** (20/20) | **0.89×** (12/20) | 6.2× slower |
| meshoptimizer, attribute-aware + permissive | 29 | **0.80×** (20/29) | **0.57×** (28/29) | **0.33×** (27/29) | **0.99×** (13/29) | 5.2× slower |
| meshoptimizer, positions only | 20 | 1.17× (5/20) | **0.62×** (16/20) | **0.29×** (18/20) | **0.40×** (18/20) | 9.4× slower |

(v8 numbers. Timings are from a busier run of the same 2-core VM than earlier versions'; part
removal, below, costs 10–20% of reduction time on multi-part assets and nothing on single-part
ones.)

Positions-only meshoptimizer keeps a lower worst-case distance, but it ignores normals and UVs
(it is 2.5–3.5× worse on both). baby_shark overshoots most targets (the Fox collapses to 2
triangles at a 50% target). In the overall summary its Hausdorff is 3.7× farthing's, even though
it usually keeps more triangles.

### Reaching the requested count

| | reached target (≤ 5% over) |
|---|---|
| farthing (default, permissive seams) | **100%** |
| meshoptimizer attribute-aware (strict seams) | 62% |
| meshoptimizer attribute-aware + permissive | 100% |
| baby_shark | 88% |

**Part removal (v8).** Edge collapses never change topology, so an asset made of many parts
spends budget on small parts it cannot simplify further (a ring stops at about 14 triangles),
and at 1% the Lantern and DamagedHelmet used to stall at 148 and 195 triangles (targets 53 and
154). The batched reducer now deletes whole connected parts: a part goes once removing it
moves the surface less than a quarter of the deviation the collapses already accepted, and when
collapses stall above the target the cheapest part goes. The removal error is the largest
distance from the part's surface samples to the rest of the mesh *facing the same way* (so the
outer of two back-to-back shells is never free), measured against the parts still kept (so two
coincident copies cannot both go), with lazily refined lower bounds. Parts with a locked vertex,
over `max_error`, or holding half the remaining triangles stay; removed vertices geomorph to the
nearest remaining vertex.

| 8 assets × 4 ratios, geometric mean | Hausdorff | mean | UV p90 | normal | targets reached |
|---|---|---|---|---|---|
| v7 (no part removal) | 2.49% | 0.220% | 0.0068 | 7.11° | 97% |
| v8 | 2.26% | 0.190% | 0.0066 | **6.81°** | **100%** |
| v9 (adds silhouette weight 0.5, the default) | **2.19%** | **0.188%** | **0.0064** | 6.81° | **100%** |

At 1% the Lantern went from 148 triangles at 76.8% Hausdorff and 19.2% mean error (its main body
crushed while small rings held their minimum) to 52 triangles at 9.6% / 0.56% (meshoptimizer
permissive at 52: 15.9% / 1.26%). DamagedHelmet at 1%: 195 → 137 triangles, 26.5% → 10.4%
Hausdorff. CarConcept at 5%: mean 0.092% → 0.065%, UV p90 0.33 → 0.15, normal 25.3° → 19.1°
(Hausdorff 1.63% → 2.03%, back to 1.63% with v9's silhouette weight).

**Silhouette weight (v9).** From the perception research: an edge is on the silhouette for a
fraction of random view directions equal to its dihedral angle over π (every view, for an open
border), and silhouette shifts are what the eye catches. Each face's plane quadric is scaled by
1 + `silhouette_weight` × the mean of that fraction over its edges.

| weight | Hausdorff | mean | UV p90 | normal | image space vs meshoptimizer |
|---|---|---|---|---|---|
| 0 (v8) | 2.26% | 0.190% | 0.0066 | 6.81° | +1.41 dB, 21/24 |
| **0.5 (default)** | 2.19% | 0.188% | 0.0064 | 6.81° | **+1.51 dB**, 21/24 |
| 1 | 2.13% | 0.185% | 0.0064 | 6.87° | +1.43 dB, 21/24 (Lantern 10% −1.15 dB) |
| 2 | 2.10% | 0.180% | 0.0065 | 6.97° | +1.37 dB, 20/24 |

Geometry keeps improving up to weight 2–8, but rendered images peak at 0.5, so that is the
default. On the PoC scene (procedural materials) weight 0.5 costs 0.06 dB on the 25% LOD and
0.15 dB on the baked 5% LOD. Large meshes: DragonAttenuation 25% 0.111% → 0.108% Hausdorff,
ABeautifulGame 25% 0.045% → 0.065% (worse), geometric mean over the six cases 0.238% vs 0.222%,
mean error 7% lower. Occluders keep weight 0 (silhouette emphasis pushed more of the occluder
outside the source).

### Image space (engine PoC, real textures)

25% LOD rendered with the asset's own textures from three views and compared with the original.
The "v2-like" column emulates the previous defaults: strict seams, mean-normalized error, no
normal term, old skin weight.

| asset | v2-like PSNR | v3 PSNR |
|---|---|---|
| DamagedHelmet | 20.2 dB | **23.2 dB** |
| WaterBottle | 34.3 dB | **37.2 dB** |
| Avocado | 31.6 dB | **33.9 dB** |
| BoomBox | 26.3 dB | **27.9 dB** |
| Duck | 26.1 dB | **27.4 dB** |
| Lantern | 26.3 dB | **27.2 dB** |

**Head-to-head with meshoptimizer (v7).** `bench-compare export` reduces each mesh with
meshoptimizer (attribute-aware, then also `Permissive`) and then reduces it with farthing to exactly the
triangle count meshoptimizer reached. The engine PoC (`score`) renders each LOD with the
asset's own textures from three views and compares it with the original. Six assets × two
ratios (25%, 10%) × two meshoptimizer modes:

| | farthing better | mean PSNR difference | silhouette mismatch (lf / mo, mean) |
|---|---|---|---|
| vs meshopt attr | **12/12** | **+1.98 dB** | **0.0052** / 0.0096 |
| vs meshopt attr + Permissive | **9/12** | **+1.04 dB** | **0.0061** / 0.0075 |

Overall 21/24 at **+1.51 dB** (v9). The three losses are all at 10% against Permissive and
small: Lantern −0.28 dB, DamagedHelmet −0.21 dB, BoomBox −0.13 dB. (v7: +1.37 dB; v8: +1.41 dB.) Every row is in
[`benchmarks/image-space-vs-meshopt.txt`](benchmarks/image-space-vs-meshopt.txt). Sweeps of
the UV and normal weights, of `error_normalization`, and of the greedy order did not remove
these three losses without costing more elsewhere, so the defaults stay as they are.

### What made the difference

1. **Constraint planes no longer dilute the error.** Open-border planes used to add their weight
   to the normalization, so border collapses looked about 10× cheaper than they were.
2. **Area-weighted ranking.** Dividing the quadric error by the area a vertex has absorbed hides
   real deviations as the region grows ("error dilution"). Ranking by the area-weighted sum
   (`error_normalization = 0`) lowered mean error by 2× and UV error by 2× at once.
3. **Normals scaled by region size.** A normal deviation over area A shades like a bump of height
   about θ·√A. Weighting the normal term by area keeps normals relevant on coarse LODs without
   letting them dominate fine ones. The small default weight (5e-5) was chosen in image space.
4. **Permissive seams by default.** Running strict until it stalls, then crossing seams, does
   far worse than letting seam crossings compete on cost from the start. Strict meshoptimizer's
   stalls (38% of cases) are the same failure.
5. **Skin weight calibrated.** Joint-weight differences are O(1) while geometric errors are
   O(1e-3), so the old weight of 1.0 made skinned meshes nearly irreducible. At 0.003 the Fox's
   skin-weight error is 2× lower than meshoptimizer's, which is not skin-aware at all.

### Throughput

Measured on a 2-core VM, min of 5 runs, interleaved with the previous build to cancel machine
noise (`cargo run --release -p farthing --example bench_assets -- <assets>`):

| | v3 | v4 | v5 |
|---|---|---|---|
| 262k-tri sphere, 25% + 5% | 2.5 s | 1.17–1.27 s | **1.11–1.23 s** |
| Five small assets (4–15k tris), 25% + 5% | 222 ms | 221–225 ms | **200–217 ms** |

**Thread scaling** (DragonAttenuation + CarConcept at 25%, geometric mean):

| loop | 1 thread | 2 threads | speedup |
|---|---|---|---|
| greedy, single pass | 1202 ms | 1101 ms | 1.09× |
| batched, single pass | 1065 ms | 893 ms | 1.19× |
| greedy, clustered (v4 default) | 989 ms | 702 ms | 1.41× |
| **batched, clustered (v5 default)** | 997 ms | **575 ms** | **1.73×** |

By Amdahl's law the v5 default is about 84% parallel. That projects to roughly 3.8× over one
thread on 8 cores (about 260 ms for these two meshes, where meshoptimizer takes about 205 ms
single-threaded). This is a projection; only 2 cores were available to measure.

**Multi-core scaling (v11).** Same machine (2 vCPUs), `RAYON_NUM_THREADS` = 1 and 2. Output
hashes are identical across thread counts.

| workload | 1 thread | 2 threads | speedup | parallel share (Amdahl) | projected 8 / 16 cores |
|---|---|---|---|---|---|
| reducer, `bench_assets` (sphere 512, Helmet, Lantern, CarConcept; 25% and 5%), v10 | 5124 ms | 3016 ms | 1.70× | 82% | 3.5× / 4.3× |
| reducer, same, **v11** | 5605 ms | 3127 ms | **1.79×** | **88%** | **4.4× / 5.8×** |
| quad reduction, 60k quads, v10 | 1469 ms | 1333 ms | 1.10× | 18% | 1.2× / 1.2× |
| quad reduction, 60k quads, **v11** | **545 ms** | **438 ms** | 1.24× | 39% | 1.5× / 1.6× |
| quad reduction, 240k quads, **v11** | **2.45 s** | **1.98 s** | 1.24× | 39% | 1.5× / 1.6× |
| UV, eight assets, **v11** | 1863 ms | 1357 ms | 1.37× | 54% | 1.9× / 2.0× |

(1-thread times on this VM vary by about ±5% between runs.) What became parallel in the
reducer:

- **Quadric setup.** Face quadrics are computed in parallel a chunk at a time. Every position
  and every wedge then sums its own faces in triangle order, also in parallel. That gives the
  same sums as the serial scatter, bit for bit.
- **Silhouette weights** come from the position-to-triangle lists, one face per task, with no
  global edge sort.
- **Candidate lists**, the orphan remap after part removal, and the per-round list of live
  candidates.
- **BVH build.** Subtrees above 8192 triangles are built with `rayon::join` and laid out root,
  children, left descendants, right descendants. The tree is identical for any thread count, a
  test checks it, and queries are unchanged.
- **Speculative evaluation.** Each round's stale candidates are evaluated in parallel against
  the state at the start of the round. The serial commit loop uses a precomputed result only
  while neither endpoint has been touched this round, the same rule as the existing "fresh"
  shortcut, so results are identical.

The serial remainder is the commit loop itself. The quad and UV rows are bound by memory
traffic and by serial passes (ring union-find, the take-in-order loop, the density search's
sequence of packings). On this box the two vCPUs share caches, so memory-bound phases barely
scale. Machines with more cores and memory channels should do better than these projections
for them.

**Parallel commits (v12).** The serial remainder in the table above is the loop that applies
each round's collapses. v12 can run it by deterministic reservations (Blelloch, Fineman,
Gibbons & Shun, *Internally Deterministic Parallel Algorithms Can Be Fast*, PPoPP 2012), all
in safe Rust:

1. Take the next 256 live candidates of the round.
2. Each claims its edge's two one-rings with `AtomicU32::fetch_min` of its position in the
   round's list. A candidate that holds every claim cannot interact with any earlier pending
   one, because a collapse only reads and writes inside those rings.
3. Winners are planned in parallel from a shared borrow. The plan is the triangles that die and
   move, the renamed wedges, and the new quadric and attribute rows, summed in the same order
   as the in-place code.
4. The plans are written in list order. Losers return to the front with their claim sets
   cached, until a commit marks their edge.

Rounds whose collapses could together reach the triangle target, and the tail of each round
(fewer than 32 winners out of 256), run serially so the stop point stays exact.

The output is bit-identical to the serial loop. A test runs a 30k-triangle multi-part mesh at
1 and 8 threads and checks that the reservation path ran. All benchmark hashes (26 runs over 11
meshes) match at 1, 2 and 8 threads.

| | 1 thread | 2 threads | parallel share | projected 8 / 16 cores |
|---|---|---|---|---|
| serial commits | 5.84 s | 3.30 s | 87% | 1.40 / 1.08 s |
| reservations | 6.64 s | 3.60 s | 91% | 1.35 / 0.97 s |

(A/B of the same build pair, back to back, three runs each.) Reservations cost about 13% more
total work: half from writing a plan instead of editing in place, half from claims and
retries. Only about a third of each prefix wins, because a round's cheapest candidates sit
close together. Reservations are therefore used only from 8 threads.

**Single-thread profile (v13).** Callgrind, one thread, DamagedHelmet + Duck + BoomBox at 25%
and 5%, counting only the reduction (1.55 G instructions). Inclusive shares:

- candidate ranking: 41%, of which attribute-quadric cost 20%, wedge mapping 14%, validity 11%;
- `State::new` setup: 17%;
- part analysis: 12%;
- incremental updates: 10%;
- evaluation in the commit loop: 5%;
- the collapses themselves: 4%.

What each quality feature costs in wall time (one thread, Helmet + Duck + BoomBox + Lantern):

| options | time |
|---|---|
| defaults | 271 ms |
| `remove_parts: false` | 225 ms |
| `silhouette_weight: 0` | 270 ms |
| `uv_weight: 0` | 240 ms |
| positions only (all attribute weights 0) | 196 ms |
| positions only, no parts, no silhouettes | 167 ms |

Even with every quality feature off, the reducer is only 1.6× faster. The remaining cost is
structural (f64 quadrics, link and flip checks, batched rounds), which is why it trails
meshoptimizer on speed while beating it on every error metric. A single-wedge fast path in
wedge mapping (no seam through the vertex: map it to q's wedge in the first shared triangle,
exactly what the general rule does) cut 2.2% of instructions with identical output.

**`unsafe` removed (v13).** The two AVX2 builds (reducer cost kernel, packer resting height)
were the plain code compiled for wider registers. A/B on one thread: reducer 5.39 vs 5.38 s,
UV 2.51 vs 2.44 s (AVX2 vs not), so both were deleted. The reducer's per-thread scratch used a
reference with an extended lifetime; it now uses `thread_local::ThreadLocal` inside the state.
All output hashes are unchanged, and every farthing crate has `#![forbid(unsafe_code)]`.

**Measuring scaling on other machines.** `scripts/scaling.sh [ASSET_DIR]` runs the reducer
(with and without parallel commits), quad reduction, UV atlases and voxel proxies at 1, 2,
4, … threads up to the core count. It checks every output hash against the 1-thread run, and
prints speedups, the parallel share and the thread count from which parallel commits pay off.
Keep `MAX_THREADS` at or below the physical core count: oversubscribed runs understate the
parallel share.

Small assets in the matched-count table average about 11 ms, against 1.7–2.6 ms for
meshoptimizer.

What the speed work changed, and what it found:

- **Batched collapse order (v5 default).** Instead of one collapse at a time from a priority
  queue, each round ranks candidates in parallel and takes the cheapest eighth. It executes
  them in cost order, each position at most once per round. A candidate is re-evaluated only
  if an earlier collapse in the round touched its neighbourhood; otherwise its cached cost,
  wedge map and validity are provably still exact. Changed positions are re-ranked in parallel
  between rounds, and a stalled round escalates (all candidates, then a full re-rank) before
  stopping. There is no heap. On the benchmark it matches or beats the exact greedy order on
  every metric (8 assets × 3 ratios: Hausdorff −3%, mean −2%, normals −2%, UV equal),
  presumably because each round spreads collapses across the mesh. `collapse_order: Greedy`
  keeps the classic order; symmetric runs use it automatically.
- **Data layout (v4).** Per-position triangle lists live in one arena. Triangles carry their
  position ids. Dead triangles leave the lists eagerly.
- **SIMD kernels (v4).** Quadrics are padded to 12 coefficients and evaluated as fixed-shape
  dot products against precomputed monomials, with an AVX2 build of the cost function picked
  at runtime. The association order is fixed in the source, so the AVX2 and baseline builds
  give identical results (tested).
- **Where time goes.** Callgrind cache and branch simulation shows the loop is bound by branch
  mispredicts on short neighbour loops and by attribute-quadric cache misses, not by
  arithmetic. SIMD cut instructions by about 8% but wall time by only 3–4%. Compiling the whole
  loop for AVX2 with forced inlining was slower (instruction-cache pressure).
- **Cluster-parallel reduction** (meshes ≥ 120k triangles): 32k-triangle Morton clusters are
  reduced concurrently with locked borders (interiors to the final density), then one pass
  handles the border bands. Clusters share the whole mesh's normalization frame. Output is
  identical on any machine and thread count.

### Large meshes

Clustered by default; `cluster_threshold = 0` gives the single-pass result. Times are single
runs. Raw output:
[`benchmarks/large-meshes-clustered.txt`](benchmarks/large-meshes-clustered.txt),
[`benchmarks/large-meshes-exact.txt`](benchmarks/large-meshes-exact.txt).

| asset (tris) | target | farthing Hausdorff / mean | meshoptimizer (attr) Hausdorff / mean | farthing / meshopt time |
|---|---|---|---|---|
| DragonAttenuation (135k) | 25% | **0.11% / 0.005%** | 0.59% / 0.062% | 549 / 192 ms |
| CarConcept (213k) | 25% | **0.19% / 0.003%** | 0.28% / 0.012% | 725 / 221 ms |
| ABeautifulGame (1.5M) | 25% | 0.045% / **0.0005%** | **0.038%** / 0.0008% | 3.9 / 1.9 s |
| DragonAttenuation | 5% | **0.53% / 0.040%** | stalls at 24% | — |
| CarConcept | 5% | **2.03% / 0.065%** | stalls at 10% | — |
| ABeautifulGame | 5% | **0.16% / 0.003%** | stalls at 5.9% (4.9% / 0.088%) | — |
| ChronographWatch (100k, single pass) | 25% | **0.75% / 0.015%** | stalls at 33% (3.3% / 0.18%) | 554 / 104 ms |

(CarConcept rows are v8, with part removal; the others are unchanged.) Over the six large-mesh
cases, v5's clustered default matches v4's exact single-pass quality
(Hausdorff 0.222% vs 0.222%, mean 0.0070% vs 0.0070%) in 1.13 s vs 2.59 s.

### Fast mode (v14)

`Quality::Fast` (`farthing reduce --fast`) is a separate collapse loop for when speed matters
more than the last bit of fidelity. It uses f32 quadrics in the unit-diagonal frame with an
attribute penalty of Σ area·|Δattribute|² instead of attribute quadrics. Each pass rebuilds a
position-to-triangle table, costs every edge in its cheaper allowed direction, sorts by cost
and collapses in order. Only the two endpoints are locked per collapse, and the link condition
and flip test are checked. Border vertices slide along border edges, and when a pass stalls the
smallest part is deleted.

Same 8 assets × 4 ratios, geometric means, default thread count (2 cores):

| method | Hausdorff % | mean % | UV p90 | normals ° | ms | targets reached |
|---|---|---|---|---|---|---|
| farthing (default) | **2.19** | 0.187 | **0.0055** | **6.80** | 18.3 | 100% |
| farthing fast | 2.47 | **0.178** | 0.0101 | 8.42 | 4.5 | 100% |
| meshoptimizer attr + permissive | 2.56 | 0.304 | 0.0151 | 6.86 | **2.5** | 100% |

Fast mode is 4× faster than the default and still has lower Hausdorff, mean and UV error than
meshoptimizer's attribute-aware mode; its normals are 1.6° worse. It doesn't do junction
sliding or visibility-aware part removal (v14's aggressive-target work below), or symmetry.

### Aggressive targets (v14)

11 assets (the 8 above plus ABeautifulGame 1.5M tris, CarConcept 213k, ChronographWatch 100k)
at 1%, 0.3% and 0.1%. *Visible Hausdorff* only counts surface that can be seen from outside its
own mesh (a sample is kept if a ray escapes in one of 32 directions), so deleting a hidden part
isn't penalized. Geometric means:

| | Hausdorff % | mean % | visible Hausdorff % | UV p90 | normals ° | targets reached |
|---|---|---|---|---|---|---|
| farthing v13 | 17.73 | 2.98 | — | 0.516 | 52.3 | 82% |
| **farthing v14** | **12.88** | **1.96** | **12.59** | **0.457** | **47.5** | 73% |
| meshoptimizer attr + permissive | 13.04 | 2.14 | 12.81 | 0.568 | 53.2 | 88% |

| asset at ratio | v13 | v14 | meshoptimizer |
|---|---|---|---|
| ABeautifulGame 0.1% | 28.3% (overshot: 792 of 1,499 tris) | **2.98%** (1,498) | 1.77% (1,486) |
| ABeautifulGame 0.3% | 1.23% | **0.98%** | 1.15% |
| ChronographWatch 1% | 58.7% (3,114 of 1,000) | **16.0%** (1,937) | 4.65% (979) |
| CarConcept 1% | 8.81% (overshot: 1,738 of 2,133) | 9.62% (2,133; visible 5.26%) | 6.22% (visible 4.86%) |
| DamagedHelmet 1% | 9.68% | 9.68% | 7.59% |

What changed, in the order the problems were found:

1. **Part removal respected nothing but its threshold.** A part went once its removal error fell
   below a quarter of the deviation collapses had accepted, however many triangles that took
   away. ABeautifulGame at 0.1% fell to 792 triangles (target 1,499) and lost visible pieces.
   Removal now never takes the count below the target.
2. **Visibility.** A part's removal error is now taken over its *visible* samples only (a ray
   from the sample escapes the other kept parts in some outward direction), so parts enclosed by
   others cost nothing. `transparent_materials` names material ids that let rays through (glass).
   Scaling the error by the part's average visibility was tried first and was worse: a part that
   is 80% hidden still matters fully where it shows.
3. **Small parts were over-charged.** The facing rule (distance to kept surface facing within
   60°) gives a small knob on a big surface a large error from its sides. The error is now at
   most the distance to any kept surface plus the part's own diagonal; back-to-back shells
   (near, but large) keep the facing rule.
4. **Visible-part protection.** When collapses stall, a part is deleted only if its error is at
   most the larger of `part_protection` (5% of the diagonal) and the usual threshold. Without
   it the watch deleted parts with 29% and 55% error. Capping the regular threshold too was
   tried and hurt the helmet (26% instead of 9.7%): blocking a needed deletion forces collapses
   into worse damage.
5. **Locked junctions.** Every endpoint of an edge shared by more than two triangles was locked.
   On the watch, 765 of the last 871 positions were locked. Junction vertices with exactly two
   junction edges (and not on a border) now slide along the junction like border vertices.
6. **Duplicated faces.** The watch's junction degrees went up to 14 with edge multiplicity 4:
   double-sided geometry made of reversed twin triangles, where every edge looks non-manifold.
   Faces over the same three positions now count once when classifying edges, so twins reduce
   like one sheet, and their open edges are borders (before, those edges were "interior" and
   free to erode).
7. **Wedge limit.** Vertices with more than 8 attribute wedges were never collapse sources (for
   speed). Once collapses stall, they now are.

Still behind: the watch (about 350 junction branch points stay locked; at extreme ratios the
attribute term outweighs geometry: geometry-only gives 1.7% vs 3.5% at 5%) and the pirate at 1%,
where a boot is deleted because its removal error is measured against the *unsimplified* other
parts. The default ratios are unchanged (2.19% / 0.187%). Raw rows:
[`benchmarks/v14-reduce-aggressive.txt`](benchmarks/v14-reduce-aggressive.txt).

### v15: junction branch points, fast-mode parity

Junction endpoints and branch points (one or 3+ non-manifold edges, not on a border) may now
collapse onto a junction neighbour, which shortens or merges junction curves. The link
condition still guards topology. The watch at 1% reaches 1,175 triangles (v14: 1,937). Fast
mode now has the same handling as the default: twin faces count once, junction vertices
slide, and stalled parts are removed by visible error with the target floor and protection.
Near aggressive targets (at most a tenth of the input) it also deletes parts below a quarter
of the accepted deviation. Defaults, 8 assets × 4 ratios:

```
method                  hausd%     mean%   vis hd%    uv p90     n deg       ms  reached
farthing                2.1781   0.18730    2.1475   0.00547     6.783    17.73     100%
farthing fast           2.3351   0.16337    2.2474   0.00990     8.314     5.62      97%
meshopt attr perm       2.5640   0.30379    2.5355   0.01507     6.863     2.52     100%
```

Aggressive targets (11 assets × 1%, 0.3%, 0.1%): default 12.85% Hausdorff / 1.98% mean (v14:
12.88 / 1.96), fast 16.15% / 2.47% (v14 fast, measured before the change: 17.20% / 2.77%),
meshoptimizer 13.04% / 2.14%. Raw: [`benchmarks/v15-reduce-aggressive.txt`](benchmarks/v15-reduce-aggressive.txt).

Tried and reverted, each measured on the aggressive set:

| change | geomean Hausdorff | notes |
|---|---|---|
| current rule (quarter of the accepted quadric deviation) | **12.85%** | |
| part error measured against the simplified surface | 13.00% | pirate 1%: 19.6% → 14.7%; helmet 1%: 9.7% → 19.3% |
| threshold = measured collapse distance × 1 / × 2 / × 4 | 13.19% / 16.63% / 18.72% | reaches more targets (76–85%), worse quality |
| × 2 plus simplified-surface errors | 13.34% | |
| attribute weights × 0.2 at aggressive ratios (v14 test) | 14.48% | only the watch improves |

The watch's remaining error (16% at 1%) is on the strap lugs. It persists with part removal
off (11.6%), with `error_normalization` 1.0 (17.3%) and with attributes off (16.3%), so it's a
collapse-ordering problem, not part removal.

## UV atlases

### Best against best (v16)

`bench-compare uv`, both tools tuned for their best result in one 1024² atlas with 2 px
padding. Both DamagedHelmet atlases were checked for overlap by rasterizing each chart at
2048²: 0 overlapping pixels. Raw rows:
[`benchmarks/v16-results-uv.json`](benchmarks/v16-results-uv.json). Its `ms` values are noisy
because other work was running; the timing table below comes from quiet runs.

- **farthing:** default options. Hole filling now runs on every atlas (`HoleFill::Always`).
  Before, it ran only when there were 32 charts or fewer.
- **xatlas** (`xatlas-rs-v2`): `max_iterations` 4 for chart growing. The texel density is
  searched for the largest value that still fits one 1024² atlas: a verified lower bound,
  then grow until it fails, bisect to 0.3%, then try 25 steps above (packing isn't monotonic
  in density). The final density is asserted to fit. Brute-force placement was worse on
  DamagedHelmet (0.414 against 0.446) and takes about 90 s per asset. Bilinear packing didn't
  help either, so both are off.

| asset | charts (lf / xa) | L2 stretch (lf / xa) | utilization (lf / xa) | seams % (lf / xa) | xa util, 1 px pad |
|---|---|---|---|---|---|
| DamagedHelmet | **277** / 848 | **1.0034** / 1.0064 | **0.697** / 0.457 | **13.6** / 25.7 | 0.527 |
| Duck | **12** / 42 | **1.0037** / 1.0061 | **0.677** / 0.593 | **4.9** / 8.0 | 0.603 |
| pirate | **110** / 194 | **1.0013** / 1.0026 | **0.630** / 0.560 | **20.3** / 23.8 | 0.574 |
| WaterBottle | **23** / 47 | **1.0030** / 1.0034 | 0.659 / **0.690** | **7.8** / 9.2 | 0.676 |
| Avocado | **5** / 9 | 1.0019 / **1.0016** | **0.624** / 0.586 | **9.1** / 10.4 | 0.584 |
| Fox | **29** / 42 | **1.0017** / 1.0081 | **0.634** / 0.591 | **19.8** / 21.7 | 0.596 |
| BoomBox | **62** / 66 | 1.0063 / **1.0031** | **0.669** / 0.593 | **5.6** / 9.4 | 0.613 |
| Lantern | **163** / 286 | 1.0013 / **1.0009** | **0.792** / 0.639 | **11.3** / 22.0 | 0.672 |

farthing has fewer charts on 8/8, shorter seams on 8/8, higher utilization on 7/8 (mean 0.673
against 0.589) and lower stretch on 5/8. xatlas packs the water bottle 3 points tighter.
Where it has lower stretch, the margin is at most 0.003. Every result is within 1% of
isometric.

**Padding is compared by measured spacing, not by the setting.** The two tools apply padding
differently. The closest gap between any two charts was measured at 4096²:

- farthing at 2 px: 3.75 texels (4.25 on the Duck, 4.75 on the Avocado).
- xatlas at 2 px: 4.25–5.25 texels, except DamagedHelmet at 2.75.
- xatlas at 1 px: 1.75–2.75 texels, tighter than farthing. Even then, farthing has the higher
  utilization on every asset except the water bottle (last column).

**Time.** xatlas's 3–9 s per asset is mostly the density search packing many times over. One
pack at a fixed density takes about 90–350 ms. Always running the hole filler roughly doubles
farthing's UV time on atlases with hundreds of charts:

| asset | hole filling ≤ 32 charts (v15) | hole filling always (v16) |
|---|---|---|
| DamagedHelmet | 0.610, ~230 ms | 0.697, ~540 ms |
| pirate | 0.587, ~150 ms | 0.630, ~500 ms |
| Lantern | 0.744, ~110 ms | 0.792, ~400 ms |
| BoomBox | 0.671, ~104 ms | 0.669, ~210 ms |

The eight assets take about 2.0 s, up from 1.1 s. Assets with 32 charts or fewer are
unchanged. `ParamOptions::fill_holes = false` keeps only the skyline packer.

**A correction to the older rows.** With default options, xatlas picks its own density. On
DamagedHelmet that made a 1319² atlas, which the v11–v15 rows below rescale to 1024². Tuned
xatlas with a density that fits 1024² scores lower there (0.457 against 0.470), so the old
row slightly flattered xatlas.

**Holes inside charts.** On DamagedHelmet, one 137-triangle chart has a one-triangle hole.
The missing triangle is its own chart. Merging it back would raise that chart's L2 stretch to
1.0088, past the 1.005 merge limit. Allowing merges of fully enclosed charts up to 1.01 or 1.02
stretch closes such holes. It removes 3–8 charts on the helmet and 6 on the Lantern, and cuts
seams by up to 0.2 points. But the reshaped charts pack worse on the Lantern (0.792 →
0.772), so the limit stays at 1.005 for every merge.

### Default options (v6–v15)

Competitor: **xatlas** (via `xatlas-rs-v2`), default chart and pack options, 1024² atlas,
2 px padding.

- **L2 stretch**: normalized Sander L2 over the whole atlas; 1.0 means isometric at one texel
  density.
- **Utilization**: UV area ÷ (atlas side)².
- **Seams**: 3D length of UV-seam edges ÷ total edge length.

| asset | charts (lf / xa) | L2 stretch (lf / xa) | utilization (lf / xa) | seams % (lf / xa) | ms (lf / xa) |
|---|---|---|---|---|---|
| DamagedHelmet | **245** / 847 | **1.0034** / 1.0058 | **0.554** / 0.470 | **13.3** / 25.8 | 463 / 371 |
| Duck | **12** / 30 | **1.0037** / 1.0085 | **0.677** / 0.575 | **4.9** / 7.6 | **280** / 314 |
| pirate | **107** / 194 | **1.0013** / 1.0021 | **0.577** / 0.543 | **20.3** / 24.4 | **217** / 243 |
| WaterBottle | **22** / 48 | **1.0030** / 1.0040 | 0.639 / **0.640** | **7.8** / 9.7 | 132 / 102 |
| Avocado | **5** / 9 | **1.0019** / 1.0023 | **0.624** / 0.515 | **9.1** / 10.2 | 58 / 15 |
| Fox | **23** / 42 | **1.0017** / 1.0070 | 0.526 / **0.547** | **18.2** / 22.1 | 137 / 122 |
| BoomBox | **57** / 95 | **1.0063** / 1.0236 | **0.545** / 0.168 | **5.5** / 12.0 | 139 / 130 |
| Lantern | **157** / 284 | **1.0013** / 1.0023 | **0.714** / 0.626 | **11.3** / 21.9 | 125 / 116 |

v11 numbers (2 threads). farthing has lower stretch on 8/8, shorter seams on 8/8, fewer charts
on 8/8 and better utilization on 6/8 (22 of 24 quality comparisons). xatlas still packs the
water bottle 0.2% tighter and the Fox 4% tighter, down from 4% and 9% in v8. All eight take
1.55 s against xatlas's 1.41 s. In v8 farthing was faster (1.14 s vs 1.28 s). The extra time
is the hole-filling packer on the few-chart assets (Avocado, Duck, Fox, WaterBottle).

**What changed in v11: a hole-filling packer.** The skyline keeps one top per atlas column, so
the space under a chart's overhang is lost for good. With few, large, irregular charts, that
lost space is a large share of the atlas. The new packer keeps a list of free intervals per
column. A position's lowest fit is the skyline's resting height when no column in the window
has a hole. Otherwise it is the first y in the intersection of each holed column's fitting
ranges. Each chart takes the position, over four rotations, that raises the atlas top least,
then the lowest, then the leftmost. x is ranked at a stride with every fourth column (a lower
bound), and refined exactly around the best three. When there are at most 32 charts it runs
alongside the skyline at every density the search tries, and the lower atlas wins. Utilization
went from 0.650 to 0.677 on the Duck, 0.617 to 0.639 on the water bottle and 0.500 to 0.526 on
the Fox; the other assets are unchanged. A test packs an H-shaped chart with two blocks: the
skyline can fill only the upper notch, the new packer fills both. This is classic bottom-left
placement on intervals, not FFT or correlation placement (see the patent list).

**v15: 8 of 8.** Same setup, 2 threads:

| asset | charts (lf / xa) | L2 stretch (lf / xa) | utilization (lf / xa) | seams % (lf / xa) | ms (lf / xa) |
|---|---|---|---|---|---|
| DamagedHelmet | **277** / 847 | **1.0034** / 1.0058 | **0.610** / 0.470 | **13.6** / 25.8 | 392 / 351 |
| Duck | **12** / 30 | **1.0037** / 1.0085 | **0.677** / 0.575 | **4.9** / 7.6 | 271 / 245 |
| pirate | **110** / 194 | **1.0013** / 1.0021 | **0.587** / 0.543 | **20.3** / 24.4 | 261 / 241 |
| WaterBottle | **23** / 48 | **1.0030** / 1.0040 | **0.659** / 0.640 | **7.8** / 9.7 | 131 / 91 |
| Avocado | **5** / 9 | **1.0019** / 1.0023 | **0.624** / 0.515 | **9.1** / 10.2 | 70 / 14 |
| Fox | **29** / 42 | **1.0017** / 1.0070 | **0.634** / 0.547 | **19.8** / 22.1 | 127 / 105 |
| BoomBox | **62** / 95 | **1.0063** / 1.0236 | **0.671** / 0.168 | **5.6** / 12.0 | 160 / 111 |
| Lantern | **163** / 284 | **1.0013** / 1.0023 | **0.744** / 0.626 | **11.3** / 21.9 | 204 / 131 |

farthing is ahead on every quality column for every asset. All eight take 1.62 s against
xatlas's 1.29 s.

- **Sprawling charts are split.** The Fox's two largest charts held 47% of its UV area but
  filled only 32% and 34% of their rectangles. A chart filling under 45% is split in two:
  two-means on face centroids in UV, seeded at the ends of the rectangle's long side, then
  connected components. The split is kept if the pieces' realigned rectangles total at least
  15% less, and pieces split again while that holds (depth ≤ 3). A subset of a valid
  flattening is still valid, so stretch is unchanged; seams grow a little (Fox 18.2 → 19.8%,
  BoomBox 5.5 → 5.6%). Fox 0.526 → 0.634, BoomBox 0.545 → 0.671, helmet 0.554 → 0.610.
- **Bounding rectangles through MaxRects.** `pack()` also packs the shapes' bounding rectangles
  with the rectangle packers and keeps the lowest atlas. Only heights below the outline result
  are searched, and the search is skipped when the rectangles' area rules it out (the
  low-fill charts). Lantern 0.714 → 0.744, water bottle up slightly; others unchanged.

**v13: packer speed and its own crate.** The packer is now `snugly`, with no mesh
dependencies. The hole-filling packer ranks each refinement window by its cheap lower bound
(every fourth column), then checks positions exactly in that order, stopping once the bound
exceeds the best exact score over all rotations so far. The placements are the same as
checking every position exactly (identical hashes). One-thread UV times: Fox 147 → 112 ms,
Duck 325 → 282 ms, water bottle 186 → 167 ms, eight assets 2.00 → 1.86 s. Against xatlas at
default threads: 1.37 s vs 1.27 s for the eight (v12: 1.55 vs 1.41).

What changed in v8 (same atlases except DamagedHelmet, whose packing order moved slightly: utilization 0.562 → 0.554):

- **Outlines, not pixel masks.** The packer only reads each chart's per-column and per-row
  extent, but v6 rasterized, dilated and scanned a full pixel mask for every chart at every
  density the search tried (63% of the Avocado's time). The extents now come straight from the
  triangles: each triangle's y-range inside each column slab, then the square dilation applied
  to the extents (the dilated mask's column spans the union of the columns within r, grown by
  r). A test checks the outlines equal the old raster's on random charts.
- **Linear-time sparse structure.** The envelope Cholesky's adjacency was built by sorting all
  2E index pairs, which cost more than the factorization on the helmet. It is now a counting
  sort by row with each short row sorted in place: the same structure, bit-identical output.
- **Small-key hashing.** The parameterizer's maps (vertex, face and corner ids) use a
  multiply-rotate hasher instead of SipHash. Output never depended on map order, so results
  are bit-identical.

What changed in v6 (v5 → v6 on the same eight assets: 4.2 s → 1.5 s):

- **Exact overlap test.** The old check rasterized every chart into a 256² hash grid; it was
  53% of the run and could miss thin overlaps. It is replaced by an exact linear-time test: no
  folded triangle, no vertex whose UV angles sum past 2π, and a boundary that does not cross
  itself (uniform grid). Together these certify an embedding for disk-like charts.
- **Packer kernels.** The placement test became branch-free with identical results. Where a
  chart comes to rest is an `i32` max over the skyline window, with an AVX2 build picked at
  runtime. The wasted area is a prefix-sum window. The best positions come from partial
  selection, not a full sort.
- **Factor once.** ARAP's cotangent Laplacian and LSCM's normal equations are factored with an
  envelope Cholesky (reverse Cuthill–McKee order, contiguous rows, fixed-order dot products).
  This replaces thousands of conjugate-gradient iterations per chart, with a CG fallback for
  very large charts.
- **Chart merging.** Neighbouring charts merge, longest shared seam first, while the union
  still flattens without overlap. A merge also has to keep per-chart L2 stretch ≤ 1.005 and
  fill its bounding rectangle at least as well as its worse part (or 70%). Each pass evaluates
  a matching of pairs in parallel, so the result is identical on any thread count.
- **Minimum-area alignment.** Charts are rotated to their minimum-area bounding rectangle
  (rotating calipers on the convex hull) instead of their principal axis.

Tried and reverted: an exact bitmap packer that tests chart masks against an occupancy
bitmap, used in place of the skyline packer. It closed part of the utilization gap on the
water bottle and the Fox, but took 2.4–17.9 s for all eight assets instead of 1.5 s. The
skyline packer stays.

## Rectangle packing (v14)

`bench-compare pack`. Every packer gets the same rectangles, sorted longest side first, and
a fixed atlas width. The fixed-size packers are binary-searched on height; farthing's strip
packer reports its own height. Utilization = rectangle area ÷ atlas area, no padding. farthing
runs on 4 rayon threads; crunch, rectangle-pack, guillotiere and etagere are single-threaded.
Each farthing row is a different entry point: `pack` on rectangle shapes, `pack`
without rotations, and `pack_rects` into a fixed bin (binary-searched like the others).

| dataset | packer | atlas | utilization | ms |
|---|---|---|---|---|
| uniform 500 (8–64 px) | snugly pack (rot) | 1024×592 | 0.981 | 185.9 |
| uniform 500 (8–64 px) | snugly pack | 1024×592 | 0.981 | 79.1 |
| uniform 500 (8–64 px) | snugly pack_rects fixed-size (rot) | 1024×592 | 0.981 | 641.0 |
| uniform 500 (8–64 px) | crunch (rot) | 1024×604 | 0.962 | 166.9 |
| uniform 500 (8–64 px) | crunch | 1024×607 | 0.957 | 118.7 |
| uniform 500 (8–64 px) | rectangle-pack | 1024×626 | 0.928 | 25.8 |
| uniform 500 (8–64 px) | etagere | 1024×760 | 0.764 | 0.3 |
| uniform 500 (8–64 px) | etagere bucketed | 1024×741 | 0.784 | 0.3 |
| uniform 500 (8–64 px) | guillotiere | 1024×616 | 0.943 | 1.6 |
| sprites 300 (16–256 px, log) | snugly pack (rot) | 2048×1144 | 0.980 | 186.9 |
| sprites 300 (16–256 px, log) | snugly pack | 2048×1145 | 0.979 | 87.2 |
| sprites 300 (16–256 px, log) | snugly pack_rects fixed-size (rot) | 2048×1144 | 0.980 | 690.8 |
| sprites 300 (16–256 px, log) | crunch (rot) | 2048×1160 | 0.966 | 65.6 |
| sprites 300 (16–256 px, log) | crunch | 2048×1168 | 0.960 | 40.5 |
| sprites 300 (16–256 px, log) | rectangle-pack | 2048×1285 | 0.872 | 12.8 |
| sprites 300 (16–256 px, log) | etagere | 2048×1440 | 0.778 | 0.2 |
| sprites 300 (16–256 px, log) | etagere bucketed | 2048×1546 | 0.725 | 0.1 |
| sprites 300 (16–256 px, log) | guillotiere | 2048×1207 | 0.929 | 0.8 |
| bars 200 (1:3–1:25) | snugly pack (rot) | 1024×445 | 0.965 | 85.1 |
| bars 200 (1:3–1:25) | snugly pack | 1024×456 | 0.942 | 42.6 |
| bars 200 (1:3–1:25) | snugly pack_rects fixed-size (rot) | 1024×448 | 0.959 | 280.5 |
| bars 200 (1:3–1:25) | crunch (rot) | 1024×478 | 0.899 | 24.8 |
| bars 200 (1:3–1:25) | crunch | 1024×496 | 0.866 | 15.5 |
| bars 200 (1:3–1:25) | rectangle-pack | 1024×703 | 0.611 | 7.3 |
| bars 200 (1:3–1:25) | etagere | 1024×656 | 0.655 | 0.1 |
| bars 200 (1:3–1:25) | etagere bucketed | 1024×625 | 0.687 | 0.1 |
| bars 200 (1:3–1:25) | guillotiere | 1024×507 | 0.847 | 0.5 |
| few large 24 (48–512 px) | snugly pack (rot) | 1024×699 | 0.921 | 2.7 |
| few large 24 (48–512 px) | snugly pack | 1024×720 | 0.895 | 1.4 |
| few large 24 (48–512 px) | snugly pack_rects fixed-size (rot) | 1024×699 | 0.921 | 7.5 |
| few large 24 (48–512 px) | crunch (rot) | 1024×744 | 0.866 | 0.4 |
| few large 24 (48–512 px) | crunch | 1024×738 | 0.873 | 0.4 |
| few large 24 (48–512 px) | rectangle-pack | 1024×822 | 0.784 | 0.5 |
| few large 24 (48–512 px) | etagere | 1024×1136 | 0.567 | 0.0 |
| few large 24 (48–512 px) | etagere bucketed | 1024×1115 | 0.578 | 0.0 |
| few large 24 (48–512 px) | guillotiere | 1024×950 | 0.678 | 0.0 |
| impostor frames + props 64 | snugly pack (rot) | 2048×926 | 0.978 | 3.5 |
| impostor frames + props 64 | snugly pack | 2048×949 | 0.954 | 1.0 |
| impostor frames + props 64 | snugly pack_rects fixed-size (rot) | 2048×926 | 0.978 | 17.4 |
| impostor frames + props 64 | crunch (rot) | 2048×930 | 0.974 | 1.9 |
| impostor frames + props 64 | crunch | 2048×949 | 0.954 | 1.3 |
| impostor frames + props 64 | rectangle-pack | 2048×949 | 0.954 | 1.3 |
| impostor frames + props 64 | etagere | 2048×1168 | 0.775 | 0.0 |
| impostor frames + props 64 | etagere bucketed | 2048×1157 | 0.783 | 0.0 |
| impostor frames + props 64 | guillotiere | 2048×949 | 0.954 | 0.1 |

farthing has the highest utilization on all five sets: 0.981 vs crunch's 0.962 (uniform), 0.980
vs 0.966 (sprites), 0.965 vs 0.899 (bars), 0.921 vs 0.873 (few large) and 0.978 vs 0.974
(impostor frames). In v13 the same sets gave 0.926, 0.920, 0.901, 0.885 and 0.955: the skyline
alone, behind crunch on three of the five.

**What changed:**

- **MaxRects with many variants.** snugly now keeps the free space as maximal empty
  rectangles (Jylänki 2010). It tries 5 placement rules × 5 input orders, plus each pair
  without rotation when rotations are allowed. All variants run in parallel next to the
  skyline, and the lowest packing wins. Ties go to the earliest variant in a fixed order, so
  results are identical on any thread count (tested at 1, 4 and 7 threads).
- **Strip search.** `pack_rects_strip` binary-searches the height between the area bound and
  the skyline's height. `pack` takes this path when every shape is a full rectangle.
- **Contact rule cap.** The contact rule is quadratic, so it is skipped above 400 rectangles.

Trade-off: `pack_rects` into a fixed atlas now costs up to 25–50 MaxRects runs instead of
one skyline pass. Callers that search atlas sizes (aggregation, billboard impostors) pay that
per size tried; it is well under a second for their rectangle counts. UV atlases are unchanged
(identical hashes): charts are rarely full rectangles, so they keep the outline packers.

**Interior holes.** Shapes now record uncovered intervals inside their outlines. With
`PackOptions::nest_in_holes`, the hole-filling packer returns them as free space after
placing the shape, so a block can sit inside a ring (tested). It is off by default: on UV
charts the holes are slivers that fragment the free space. Water bottle utilization went
0.626 → 0.601 with it on (bench_uv numbers), so farthing::uv leaves it off.

## Voxel proxies (remesh)

Competitor: **baby_shark** 0.3.12 `VoxelRemesher`, both its dual-contouring
(`FeaturePreserving`) and marching-cubes (`Manifold`) modes, followed by its edge decimator.
farthing runs `farthing::voxel::remesh`. Both use the same voxel size (1/128 of the longest side)
and a 3000-triangle budget. baby_shark's decimator stops at about 2606 triangles, so farthing
is also run at that exact count (the "matched" rows). Open % is the share of edges, welded by
position, that are not shared by exactly two triangles (0 means closed and manifold).

| asset | farthing matched: Hausdorff % / mean % / open % / ms | baby_shark DC | baby_shark MC |
|---|---|---|---|
| Duck | 0.942 / **0.0190** / **0** / **432** | 0.536 / 0.0247 / 0.59 / 1667 | **0.497** / 0.0262 / 0 / 1533 |
| Avocado | 7.379 / **0.1276** / 0 / **235** | 7.392 / 0.1287 / 0 / 1068 | **7.361** / 0.1308 / 0 / 1100 |
| Fox | **0.760** / **0.0156** / 0 / **110** | decimator panicked | 1.118 / 0.0278 / 0 / 719 |
| BoomBox | 2.314 / 0.0284 / **0** / **638** | **2.195** / **0.0270** / 1.45 / 1950 | 2.235 / 0.0282 / 0 / 1981 |
| DamagedHelmet | 3.509 / 0.1759 / 0 / 851 (3000 tris) | decimator panicked | decimator panicked |
| Lantern | **3.479** / 0.0915 / **0** / **73** | 3.493 / **0.0889** / 0.64 / 704 | 3.542 / 0.0935 / 0 / 691 |
| DragonAttenuation | 1.480 / 0.0436 / **0** / **588** | 1.159 / **0.0365** / 10.35 / 1530 | **0.869** / 0.0391 / 6.29 / 1429 |

farthing is 3–10× faster, and its proxies are closed and manifold on every asset. baby_shark's
DC output has open or non-manifold edges on four of the five assets it finished, and its MC
output on the dragon. farthing's mean error is lower on Duck, Avocado and Fox, about equal on
BoomBox, between DC and MC on the Lantern, and higher on the dragon. Its Hausdorff error is
higher on Duck, BoomBox and the dragon. The cause is in the voxel stage, not the
reduction: on the same farthing proxy, farthing's reducer beats baby_shark's decimator on the
dragon (mean 0.066% vs 0.072%, measured before the clash fix below). baby_shark contours a sub-voxel signed distance field. farthing
extracts a binary, conservatively voxelized solid, which fills crevices narrower than about a
cell.

**At equal time.** farthing's voxel stage is fast enough to run at a finer grid in baby_shark's
time. At resolution 192 (1/192 of the longest side), still 3000 triangles, v9:

| asset | farthing res 192: Hausdorff % / mean % / ms | best baby_shark at res 128 |
|---|---|---|
| Duck | 0.562 / **0.0116** / **1210** | MC **0.497** / 0.0262 / 1533 |
| BoomBox | 2.245 / **0.0173** / **1657** | DC **2.195** / 0.0270 / 1950 (1.45% open edges) |
| Lantern | 3.508 / **0.0832** / **204** | DC **3.493** / 0.0889 / 704 (0.64% open) |
| DragonAttenuation | 1.399 / **0.0345** / **1200** | MC **0.869** / 0.0391 / 1429 (6.29% open) |

Mean error is 12–56% lower and every farthing proxy is closed; worst-case distance is within
3% on BoomBox and Lantern, 13% behind on the Duck, and still well behind on the dragon. (At 256
the Duck reaches 0.457% Hausdorff, but the dragon's worst case jumps to 2.79%: a thin part lost
in reduction.)

**v12: sub-voxel signed boundary.** Same harness, same voxel size (1/128 of the longest side),
3000-triangle budget, and the matched rows at baby_shark's triangle count. ⚠ marks open or
non-manifold output.

| asset | farthing v12 matched: Hausdorff % / mean % / open % / ms | baby_shark DC | baby_shark MC |
|---|---|---|---|
| Duck | **0.447** / **0.0208** / 0 / **638** | 0.656 / 0.0239 / 0.59 ⚠ / 1825 | 0.497 / 0.0262 / 0 / 1783 |
| Avocado | 7.384 / **0.1283** / 0 / **367** | 7.392 / 0.1286 / 0 / 1368 | **7.361** / 0.1308 / 0 / 1131 |
| Fox | **0.451** / **0.0152** / 0 / **145** | decimator panicked | 1.118 / 0.0278 / 0 / 684 |
| BoomBox | 2.297 / **0.0239** / 0 / **999** | 2.236 / 0.0275 / 1.45 ⚠ / 2028 | **2.235** / 0.0282 / 0 / 2158 |
| DamagedHelmet | **3.162 / 0.1706** / 0 / 1303 (3000 tris) | decimator panicked | decimator panicked |
| Lantern | **3.479** / 0.0895 / 0 / **172** | 3.493 / **0.0892** / 0.64 ⚠ / 828 | 3.542 / 0.0935 / 0 / 780 |
| DragonAttenuation | 1.391 / 0.0430 / 0 / **1033** | 1.159 / **0.0363** / 10.24 ⚠ / 1743 | **0.869** / 0.0391 / 6.29 ⚠ / 1602 |

v11 matched rows for comparison: Duck 0.942 / 0.0190, Fox 0.760 / 0.0156, BoomBox 2.314 /
0.0284, Lantern 3.479 / 0.0915, dragon 1.480 / 0.0436, Avocado 7.379 / 0.1276; DamagedHelmet
3.509 / 0.1759 at 3000. Worst-case distance fell on every asset. Mean error fell everywhere
except the Duck (+9%) and Avocado (+0.5%). At resolution 192 (3000 triangles), the Duck reaches
0.398 / 0.0133, BoomBox 2.265 / 0.0149, Lantern 3.470 / 0.0773 and the dragon 1.399 / 0.0348.

How it works (`crates/farthing/src/voxel/sdf.rs`). The flood still decides inside and outside
for cells the source does not touch. A touched cell is now solid only if its centre is inside
the source:

1. **Sign.** The closest point's angle-weighted pseudo-normal (face, edge or vertex: Bærentzen
   & Aanæs 2005) gives the sign, exact for closed, consistently wound meshes. A centre found
   outside is carved only if none of three winding rays finds it enclosed. Each ray counts +1
   where it leaves a face and −1 where it enters. Inside a body, the closest face can belong
   to a nested or overlapping piece that faces outwards; the rays still see the body. Rays
   that leave through a hole of an open mesh count 0 or −1, never "enclosed". The dragon is
   open: 1.2% of its faces read as inverted and 4% as ambiguous.
2. **No lost surface.** Surface samples at half-cell spacing each need a solid cell centre
   behind them, within a cell, that lies inside a volume: an interior cell, or a boundary cell
   that all three rays find enclosed. Otherwise the cell stays solid. Sheets thinner than a cell
   have no such centre and survive, where a signed-distance contourer loses them. On the dragon,
   baby_shark's MC leaves 11,402 source samples more than half a cell outside its unreduced
   proxy; farthing leaves 1,541. The other side is where the dragon still trails: 55,990
   samples lie more than half a cell inside farthing's unreduced proxy (v11: 58,765;
   baby_shark: 2,275). They sit in slots between pieces, where no cell centre falls in the gap.
3. **Lone sheets keep their slab.** A sample with no volume on either side keeps every boundary
   cell around it. Without this, the dragon's backdrop (an open sheet) thinned to a staircase
   that fell apart into islands, which the reducer then deleted (Hausdorff 1.87%).

Steps taken on the way, measured on the dragon (3000 triangles, Hausdorff / mean %):

| variant | dragon | notes |
|---|---|---|
| v11 | 1.550 / 0.0394 | |
| sign by closest face only | 1.747 / 0.0486 | dents where nested pieces face outwards |
| + winding rays (any ray "inside" keeps the cell) | 1.747 / 0.0486 | same dents (open mesh: rays through the hole) |
| + support only from volumes all rays agree on | 1.874 / 0.0406 | backdrop sheet thins and breaks up |
| + lone sheets keep their slab (v12) | **1.402 / 0.0390** | Duck 0.471, Fox 0.451, Helmet 3.162 at 3000 |

**v13: slot pass tried and reverted.** A second pass opened boundary cells whose centre is
inside but which touch air, hold facing walls, have a point of their 2×2×2 sub-grid outside the
source, and keep volume behind every surface sample. It repeated so slots open from the mouth.
Dragon unreduced forward error went 1.302 → 1.232%, but reduced it stayed at 1.402%. The Fox
went 0.451 → 0.660% and DamagedHelmet 3.162 → 3.299%, so it was reverted. The dragon's slots
are about 0.15 cells wide; resolution 192 doesn't see them either (1.399%). They need about 8×
local refinement.

**Where the dragon's worst case comes from (v11).** On the unreduced proxy (128k triangles,
1.30% forward Hausdorff), 540k area-weighted source samples were classified by which side of
the proxy they fall. 58,765 lie more than half a cell inside the proxy, and 499 lie more than
half a cell outside it. The error is buried surface: slots and crevices narrower than about a
cell, which conservative voxelization fills to their mouths. The worst sample is 2.8 cells
deep in a slot 0.15 cells wide. No closing is involved: with `close_gaps = 0` the counts are
the same. A flood-unreachable pocket test (ray parity on enclosed air regions) found none.

A guarded carving pass was tried. It carves solid cells whose centre lies outside the source:
the closest point is inside a face, the centre is on the face's outer side, and there are facing
walls within a cell. A cell is carved only if every surface sample in it keeps a kept solid
cell behind it among the 26 neighbours. Results:

| variant | unreduced: worst / samples > ½ cell inside / outside | 3000 tris: Hausdorff / mean | ms |
|---|---|---|---|
| v10 | 2.81 cells / 58,765 / 499 | **1.55% / 0.0394%** | **600** |
| carve, no facing-wall test | 2.43 / 32,215 / 9,620 | 6.89% / 0.267% (stalls at 5676 tris) | 1015 |
| carve, facing walls | 2.61 / 44,074 / 2,980 | **1.45%** / 0.0400% | 1113 |

Even the guarded version leaves 395 islands of 1–3 cells that the reducer must spend triangles
on, raises mean error and doubles the time, so it was reverted. The fix that would match a
signed-distance contourer here is a sub-voxel signed field in the boundary cells.

What changed in v7:

- **Gap sealing only where the input leaks.** Morphological closing (`close_gaps`) also fills
  real crevices. With the new default `adaptive_gaps`, the solid is classified with and without
  closing, and the unclosed one is used unless it is more than 10% smaller (meaning the flood
  leaked inside through a hole). Of eleven test assets, only DamagedHelmet leaks; it keeps 18%
  of its volume without closing. *v8 refines this to per-pocket sealing:* of the cells closing
  adds, each connected region is kept only if some cell lies more than `close_gaps + 1` steps
  inside it (a cavity behind a leak); thin crevice fills revert to air. DamagedHelmet's proxy
  went from 3.51% / 0.176% (global sealing) to 3.44% / 0.163% Hausdorff / mean, the dragon from
  1.55% to 1.48% Hausdorff; the others are unchanged.
- **Manifold solids.** Before extraction, ambiguous 2×2 face and 2×2×2 corner configurations
  are filled, so surface nets cannot produce non-manifold edges or vertices.
- **Surface nets at the grid edge.** Quads were skipped for grid edges that leave the first
  sample layer. Without closing, geometry lying exactly on the bounds reaches that layer, and
  the skipped quads left holes.
- **Fold-safe projection.** Vertices are snapped to the source surface. Faces that the snap would collapse or flip
  are reverted rather than deleted, so no holes open. Several vertices landing on the same source point (a corner,
  or both sides of a thin sheet) are backed off it by 2% of a cell instead of being welded. On
  the dragon, 16% of the vertices had been reverted for this reason; that alone cut its mean
  error from 0.066% to 0.039%.
- **Smoothing after projection.** Laplacian smoothing now moves only the vertices the
  projection had to leave in place, instead of rounding every vertex before projection.

Duck/Fox/BoomBox/Lantern/dragon mean error went from 0.0172/0.0213/0.0393/0.1065/0.0891
(v6 defaults, 3000 tris) to 0.0153/0.0151/0.0249/0.0915/0.0394.

### v14: the dragon's error is in the reduction

v12 left the dragon at 1.39% Hausdorff against baby_shark's 0.87%, and adaptive (octree)
contouring was the planned fix. Measured first: at resolution 512 the *unreduced* proxy is
within 0.31% of the source, but the reduction to 3,000 triangles takes it to 1.38%. The worst
points are the backdrop slab's corners, which the reducer cuts. Finer contouring can't fix error
the reduction adds, so it wasn't built. `RemeshOptions.simplify` now passes options to that final
reduction: `error_normalization: 0.5` gives 1.24% on the dragon, but costs the other assets, so it
isn't the default.

## Surfel proxies (prototype, v9)

A relightable stand-in for distant foliage-like assets: the surface covered by blue-noise
oriented discs (`farthing::impostor::surfels`), each carrying a normal and the source triangle
and barycentrics so the host evaluates its own material there. The host lights them through its
G-buffer like any other geometry, so they respond to moving lights, which baked impostors and
photo-captured splats do not.

Test asset: a procedural bush in the PoC engine, 3000 two-sided leaf quads and a trunk (6032
triangles, 3001 parts). Each representation renders an unlit G-buffer (albedo and normal) from
three views; every image is then lit on the CPU by the same code under four light directions
and compared with the original under the same light (12 comparisons per row). The voxel proxy's
resolution is lowered until it meets the budget (foliage voxelizes into a sponge whose handles
collapses cannot remove; at resolution 128 it stalls near 18k triangles). "Surfels" spread over
all leaf area; "surfels, visible" weight each triangle by the fraction of 48 views that see it
(`farthing::visibility`), so interior leaves get none. Raw rows:
[`benchmarks/relight-bush.jsonl`](benchmarks/relight-bush.jsonl); images:
[`poc/relight_bush.png`](poc/relight_bush.png).

| budget | triangle LOD | voxel proxy | octahedral impostor, same memory | surfels | surfels, visible |
|---|---|---|---|---|---|
| 600 | 18.1 dB (560 tris), silhouette 0.124 | **21.8 dB**, **0.018** (res 54) | 20.8 dB, 0.047 (48² atlas) | 19.5 dB, 0.072 | 20.9 dB, 0.036 |
| 1500 | 19.8 dB (1293 tris), 0.063 | **21.8 dB**, **0.016** (res 54) | 20.8 dB, 0.042 (60²) | 20.3 dB, 0.051 | 21.7 dB, 0.025 |
| 3000 | 19.8 dB (1293 tris), 0.063 | 21.5 dB, **0.019** (res 72) | 20.9 dB, 0.044 (84²) | 21.2 dB, 0.036 | **22.2 dB**, 0.021 |

(v9 reducer with the silhouette weight; the LOD stops at 1293 triangles at the 3000 budget
because part removal deletes leaves that cost less than the accepted error.)

The octahedral impostor is the usual answer for distant foliage. It relights here too, from its
albedo and normal atlases (the reference shader gained G-buffer output modes for this). At the
same memory as the surfels (a surfel ≈ 24 bytes, an atlas texel 8: albedo plus normal) its 12×12
frames are only 4–7 pixels and it is a blur. With more memory it does well:

| impostor atlas | memory | PSNR | silhouette |
|---|---|---|---|
| 192² (16 px frames) | 288 KB | 22.2 dB | 0.028 |
| 384² (32 px frames) | 1.2 MB | 22.8 dB | 0.023 |
| 768² (64 px frames) | 4.7 MB | **23.5 dB** | 0.021 |

(Impostors with v10's anti-aliased captures and relief-marched parallax, below.) So 3000
visibility-weighted surfels (72 KB) match a 288 KB impostor, and 1500 (36 KB) come within 0.5 dB
of it: about 4× less memory for the same quality, with the better silhouette, and they are real
geometry (correct parallax and depth, no frame blending). Impostors still win on
absolute quality when memory is no object. The same ranking holds when the bush is drawn at
128 × 128 pixels instead of 512 ([`benchmarks/relight-bush-128px.jsonl`](benchmarks/relight-bush-128px.jsonl)).

Visibility-weighted surfels beat the triangle LOD at every budget (+1.9 to +2.8 dB, half the
silhouette error or better), tie the voxel proxy at 1500 and beat it at 3000. They also keep
the leafy look the proxy smooths into a blob: PSNR rewards that smoothing, so the table flatters
the proxy. The proxy keeps the best silhouette and wins at 600, and in this test it carries a
single material (its trunk is leaf-coloured; a real pipeline would bake it). Surfels are fast to
make (0.01–0.1 s, against about 1 s for the LOD and 6 s for the fitted proxy). The PoC draws
each disc as a 6-triangle fan through the ordinary mesh pipeline, opaque, so no blending is
involved; a production renderer would draw one quad per disc.

Not done yet: a per-surfel normal-map term (the procedural bump already varies per pixel here)
and a perceptual metric instead of PSNR.

### Progressive order and streaming (v14)

`progressive_surfel_cloud` orders the same surfels so every prefix is an even covering. It
continues the weighted sample elimination in halving stages, each with the neighbourhood radius
for its target size, and reverses the elimination order. The runtime draws the first `k` at
`radius_for(k)` and fades the next one in (`lod(k)` takes fractional `k`). Bush (3,000 leaves,
visibility-weighted, 256 px views, relit, against the mesh):

| surfels | prefix of one 6,000 cloud | cloud generated for that count | mean pixel change per 5% LOD step: progressive / separate clouds |
|---|---|---|---|
| 750 | 19.92 dB | 20.24 dB | **0.0014** / 0.0094 |
| 1,500 | 20.37 dB | 20.48 dB | **0.0016** / 0.0111 |
| 3,000 | 20.99 dB | 21.12 dB | **0.0016** / 0.0103 |
| 6,000 | 21.61 dB | 21.61 dB | — |

One cloud serves every level at 0.1–0.3 dB below dedicated clouds, and a LOD step changes the
image about 7× less than switching between separately generated clouds. The LFSF stream stores
fixed-size records in progressive order: position 3 × u16 in the bounds, octahedral normal
2 × u16, optional source triangle and barycentrics, then host channels. That's 10 bytes per
surfel, or 13 with RGB (6,000 surfels: 78 KB; 108 KB with source references), and decoding
6,000 takes 0.2 ms. A truncated stream decodes its complete records, which are a coarser cloud.
Raw: [`benchmarks/v14-surfel-lod.jsonl`](benchmarks/v14-surfel-lod.jsonl).

### v15: splat shader, per-surfel radii

`SURFEL_SHADER` draws a cloud as instanced oriented discs (radius × the `lod(k)` scale). The
last surfel fades in by dithered coverage, and the output is a G-buffer. naga validates it in a
unit test. In the PoC it matches the disc meshes of the same 1,500 surfels to 0.25% coverage
(circles vs hexagons) and 5° mean normal difference (overlap order), for whole and fractional
k. Per-surfel radii ride `lod()` (a scale now, not one radius) and the stream (+1 byte per
surfel when radii vary, within 2.3%). Adaptive density, where low weight gives fewer, larger
discs through variable-radius elimination, works as designed in tests (a quarter weight gives
a quarter of the density at twice the radius, no holes). But with visibility weights on the
bush it rendered 0.3–0.9 dB worse, since hidden leaves' large discs show through gaps, so it's
opt-in (`adaptive_density`), and the default reproduces v14 exactly.

## Octahedral impostors (v10)

PoC scene: the knot, 12×12 frames of 160 px (1920² atlas), compared with the mesh from the three
standard views (close) and from three distant views with a long lens (far).

| | close PSNR | close silhouette | far PSNR | far silhouette |
|---|---|---|---|---|
| v9 | 18.32 dB | 0.0146 | 21.18 dB | 0.0099 |
| + anti-aliased captures | 18.62 dB | 0.0137 | 21.60 dB | 0.0088 |
| + relief-marched parallax (v10) | **21.83 dB** | **0.0071** | **23.24 dB** | **0.0082** |

- **Anti-aliased captures.** Frames are rendered at twice the frame size and box-filtered down
  (`build_impostor`; `build_impostor_supersampled` picks the factor). The covered fraction is
  kept in alpha, and the shader's 0.5 alpha test puts the silhouette where half a texel is
  covered.
- **Relief-marched parallax.** v8 refined each frame's sample point with three fixed-point
  iterations on the stored height, which jumped across depth discontinuities and left the dark
  gaps noted in the limitations. The shader now marches the view ray through each frame's
  height field (16 steps between the slab planes, then 5 bisection steps) and treats the height
  field as a shell 2.5 steps thick rather than solid, so rays passing behind a neighbouring
  frame's front surface near the silhouette no longer hit its extrusion (with a solid height
  field the far views got worse, 19.96 dB, and silhouettes grew). Shell thickness was swept from
  1.5 to 4 steps.
- **G-buffer outputs.** The reference shader's `frames.z` selects lit preview, albedo or world
  normal, for hosts that light impostors with their own deferred lighting.

### Two layers (v14)

`build_layered_impostor` captures a second layer per frame through
`RenderHost::render_views_behind`: per pixel, the nearest *front-facing* surface more than 1% of
the radius behind the first layer. Front faces only, because a closed shell's next surface is
the back of its own front wall, which no view sees. The reference shader marches both height
fields and takes the layer the ray reaches first. 12×12 frames, three close and three distant
views:

| prop | 1 layer, 160 px | 1 layer, 226 px (same memory) | 2 layers, 160 px |
|---|---|---|---|
| vase, close: PSNR / silhouette | 26.12 dB / 0.0047 | 26.44 / 0.0050 | **26.52 / 0.0034** |
| vase, far | 26.41 / 0.0034 | 26.55 / 0.0036 | **27.41 / 0.0025** |
| knot, close / far PSNR | 22.04 / 23.24 | 22.22 / **23.86** | **22.29** / 23.73 |
| bush, close / far PSNR | 19.66 / 20.60 | **19.78 / 20.80** | 19.69 / 20.63 |

On the concave vase (a thick-walled vessel open at the top), the second layer beats spending
the same memory on larger frames: +1.0 dB far and 30% less silhouette error. It doesn't help the
knot or the bush, so it's an option for concave props. Raw:
[`benchmarks/v14-layered-impostors.jsonl`](benchmarks/v14-layered-impostors.jsonl).

### v15: half-resolution back layer, automatic choice

| prop | variant | color texels | close PSNR / silhouette | far PSNR / silhouette |
|---|---|---|---|---|
| vase | 1 layer, 160 px | 3.69 M | 26.12 / 0.0047 | 26.41 / 0.0034 |
| vase | 1 layer, 179 px (1.25× memory) | 4.61 M | 26.20 / 0.0048 | 26.52 / 0.0034 |
| vase | 2 layers, full-res back | 7.37 M | 26.52 / 0.0034 | 27.41 / 0.0025 |
| vase | **2 layers, half-res back** | **4.61 M** | **26.51 / 0.0035** | **27.42 / 0.0025** |
| knot | 1 layer, 1.25× memory / 2 layers half-res | 4.61 M | 22.03 / 22.27 | 23.38 / 23.63 |
| bush | 1 layer, 1.25× memory / 2 layers half-res | 4.61 M | 19.71 / 19.69 | 20.67 / 20.63 |

The half-resolution back layer keeps the whole gain on the vase for a quarter of the extra
memory. `layer_benefit` = (share of front-layer texels with a back-layer surface) × (mean
depth gap in radii): vase 0.51 (0.66 × 0.77), knot 0.14 (0.28 × 0.50), bush 0.03
(0.22 × 0.14). Auto mode keeps the layer at ≥ 0.25: it kept the vase's and dropped the knot's
and the bush's. Raw: [`benchmarks/v15-layered-impostors.jsonl`](benchmarks/v15-layered-impostors.jsonl).

## Quad reduction (v10, v11)

Bumpy quad spheres (a subdivided cube projected to a sphere with bumps), reduced to 10% of the
quads by edge-ring removal (`examples/bench_quads.rs`). v9 found every ring, costed it and
rebuilt the mesh after each single ring: quadratic. v10 removes many rings per pass: all rings
are costed once, then taken in cost order up to a quarter of them and never dearer than the
quarter's cost, provided their strips plus every quad touching them are vertex-disjoint (so
their collapses cannot interact and each is validated locally), and the mesh is rebuilt once.

| quads in | v9 time | v9 Hausdorff / mean | v10 time | v10 Hausdorff / mean | v11 time, 1 / 2 threads |
|---|---|---|---|---|---|
| 600 | 9 ms | **3.90% / 0.69%** | **6 ms** | 4.65% / 0.80% | — |
| 2,400 | 57 ms | 1.47% / 0.29% | 34 ms | **1.15% / 0.28%** | **16** / 19 ms |
| 9,600 | 399 ms | 0.64% / 0.117% | 105 ms | **0.51% / 0.112%** | **62** / 66 ms |
| 21,600 | 1,899 ms | 0.31% / 0.053% | 396 ms | **0.26% / 0.051%** | 147 / **140** ms |
| 60,000 | (not run; ~15 s extrapolated) | — | 1.4 s | 0.11% / 0.021% | 0.55 / **0.44** s |
| 240,000 | (minutes) | — | 8.8 s | 0.032% / 0.006% | 2.45 / **1.98** s |

v11 output is identical to v10's (same quads, same error). The time went to finding rings, not
removing them. Each pass walked every ring with a `contains()` check per step (quadratic in ring
length) and hashed every edge. v11 finds them in linear time:

- edges are numbered by listing each vertex's sides to higher-numbered vertices, sorted
  locally, in parallel vertex blocks (the same order as a global sort);
- rings are union-find classes of the edges linked across quads, and each class is walked from
  its smallest edge, all classes in parallel. Each side stores its edge and the side across
  it, so a step reads one cache line;
- ring costs are computed in parallel. Validity checks use per-thread arrays stamped per call
  instead of hash maps, and run in parallel for blocks of the next unblocked rings in cost
  order. Rings are still taken one by one in cost order, so the result does not depend on the
  block size or thread count.

Two ring rules are now explicit: edges shared by more than two quads are never rungs, and a ring
that crosses itself (both side pairs of one quad) is dropped whole. Neither occurs in the
benchmark meshes. Scaling on 2 threads is weak (1.24× at 240k) because the walk and the side
table are memory-bound and these two vCPUs share caches. The union-find, the side-table
numbering and the take-in-order loop are still serial.

Without the per-pass cost cap, rings blocked by a neighbour let dearer ones through, and error
at 21,600 quads rose 55% (mean 0.082%); with it, quality matches or beats the one-at-a-time
order except on the smallest mesh.

## Cluster LOD DAGs (v16)

`clusters` builds a Nanite-style DAG with `farthing::cluster::build_cluster_dag` and compares
it with the same scheme assembled from meshoptimizer 0.7's own primitives, as its
`demo/clusterlod.h` does: `build_meshlets(255, 128)`, then `partition_clusters_with_positions`
(groups of about 8), then each group simplified to half with its border locked
(`Sparse | ErrorAbsolute`, plus `Permissive` and normal/UV attributes for "meshopt attr", as in
clusterlod.h). A group that keeps more than 85% of its triangles passes through to the next level,
for every method. Both use 128-triangle, 255-vertex clusters.

**How quality is scored.** The camera sits 4 bounding radii out (1080p, 60° vertical FOV). The
screen-error threshold is swept from 0.02 to 200 px in steps of ×1.5. Each cut is
`dag.select` → `to_mesh`, and its mean surface distance to the source is *measured* (40k
samples, in pixels at the bounds' nearest point). Each cell is the fewest triangles any cut needs
to stay under that measured error. A method that under-reports its own error gains nothing.
Raw data, including every cut's (triangles, mean px, max px) curve:
[`benchmarks/results-clusters.jsonl`](benchmarks/results-clusters.jsonl). Machine: 22 threads,
Windows 11, mimalloc. Lower is better.

| Asset (triangles) | Method | Build | Levels / roots | ≤ 0.25 px | ≤ 0.5 px | ≤ 1 px | ≤ 2 px |
|---|---|---|---|---|---|---|---|
| DamagedHelmet (15k) | farthing | 67 ms | 8 / 1 | **5,927** | 5,927 | 3,888 | 1,943 |
| | farthing fast | 31 ms | 8 / 1 | 7,270 | **4,400** | **1,943** | 1,943 |
| | meshopt | 36 ms | 4 / 26 | 7,720 | 7,049 | 5,416 | 5,416 |
| | meshopt attr | 57 ms | 8 / 1 | 7,721 | 7,721 | 3,858 | **1,926** |
| pirate (5k) | farthing | 26 ms | 7 / 1 | 2,785 | 1,630 | 1,630 | 621 |
| | farthing fast | 12 ms | 7 / 1 | 3,272 | **1,123** | **625** | **311** |
| | meshopt | 11 ms | 6 / 2 | **2,503** | 1,250 | 1,250 | 625 |
| | meshopt attr | 13 ms | 7 / 1 | 2,504 | 2,504 | 1,251 | 625 |
| Lantern (5k) | farthing | 27 ms | 7 / 1 | 674 | 674 | 336 | 168 |
| | farthing fast | 18 ms | 7 / 1 | 674 | 674 | 336 | 336 |
| | meshopt | 10 ms | 4 / 6 | 1,347 | 1,347 | 1,347 | 672 |
| | meshopt attr | 20 ms | 7 / 1 | **661** | **330** | **330** | **82** |
| DragonAttenuation (135k) | farthing | 208 ms | 12 / 1 | 16,685 | 8,741 | 7,781 | 3,323 |
| | farthing fast | 112 ms | 12 / 1 | **12,531** | **8,187** | **2,791** | **799** |
| | meshopt | 397 ms | 7 / 363 | 47,457 | 38,607 | 38,607 | 38,607 |
| | meshopt attr | 508 ms | 12 / 1 | 24,592 | 24,592 | 13,196 | 13,196 |
| Avocado (682) | all four | 1–5 ms | 4 / 1 | 682 | 341–682 | 341–682 | 169–170 |

Build time on ABeautifulGame (1.5M triangles, 22 threads): farthing 1.85 s, fast 1.04 s,
meshopt 3.3 s (973 roots), meshopt attr 3.9 s (160 roots).

What the numbers say:

- **Reaching one root.** farthing's DAG always closes to a single root (two on the chess set,
  whose pieces are disconnected). Geometry-only meshopt stalls when groups can't reach 85%
  under locked borders: 363 roots on the dragon, 973 on the chess set. So its coarsest cut is
  still 38,607 triangles, however far away the dragon is. Permissive attributes help (one
  root on the dragon) but still leave 160 on the chess set. Both methods pass a stalled group on
  to be regrouped. farthing's reducer keeps finding collapses under locked borders where
  meshoptimizer's runs out.
- **Triangles at matched error.** On the dragon, farthing needs 1.7–4× fewer triangles than
  meshopt attr at every error level, and fast mode needs up to 16× fewer at 2 px. On the helmet
  and the pirate it is level or better at most targets. **meshoptimizer wins on the Lantern**
  (330 vs 674 triangles at 0.5 px). Its permissive attribute path keeps the lantern's thin frame
  at half the triangles.
- **Fast mode** (`ClusterOptions::fast()`, `Quality::Fast` group reductions) builds about 2×
  faster, with geometric error as low or lower. It keeps normals and UVs less faithfully, which
  this table, measuring positions only, doesn't show.
- **Speed.** The meshopt baseline here is serial. clusterlod.h itself runs groups in parallel,
  and zeux reports about 10.6M triangles/s on 16 threads, which is faster than farthing's
  roughly 0.8M (default) and 1.4M (fast) triangles/s here. Per core (dragon, one thread):
  farthing 1.2 s, fast 0.56 s, meshopt 0.39 s. farthing's levels are parallel: groups are reduced
  concurrently, and clustering uses a parallel spatial bisection and parallel adjacency. With
  the Windows system allocator, parallel scaling stalls; the bench links mimalloc, and engines should too.

### HLOD (v16)

`farthing::hlod::build_hlod` (CLI: `farthing hlod`) makes one distant proxy out of a whole
scene. It runs camera-sampled visibility, then `cull_invisible`, then merges and reduces (or
voxel-remeshes), unwraps, and bakes from the culled source. The Lantern on its own, with 64
cameras, reduced to 500 triangles: 122 of its 5,394 triangles are never seen, and the result has a
mean distance of 0.026% of the bounds. In the test scene (a crate with a box hidden inside it,
standing on a ground slab, with pebbles), every triangle of the hidden box goes, the faces pressed
against the ground go, and the 200-triangle budget holds.

Small parts are kept or dropped whole before culling (`small_part_pixels`, 8 px of the visibility
render by default). Without that, partly seen pebbles leave shards the reducer can't collapse,
and a hard triangle budget then removes the large parts instead. Visibility is camera ID renders
only. Unlike EA's US 9,978,176, nothing is decided by casting rays along normals.

### Gaussian splats from surfels (v16)

`farthing::impostor::splats` (CLI: `farthing splats`) writes a progressive surfel cloud as a
standard 3D Gaussian Splatting `.ply`: one flat Gaussian per surfel, with tangent σ at half the
disc radius and normal σ at 5% of that, SH degree 0 colour from a host channel, and logit
opacity. The progressive order carries over: the first k splats, grown by `SurfelCloud::lod`'s
radius scale, are a coarser even covering. These are placed, not optimized. There is no
photometric training, so the colour is whatever the host evaluates at each surfel (albedo, or
baked lighting), and the result is view-independent.

