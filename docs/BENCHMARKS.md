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
cargo run --release -- realtime <assets...> # frame-budget reduction to RT_TARGET (3000)
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
  if an earlier collapse in the round touched its neighborhood; otherwise its cached cost and
  wedge map are provably still exact. Validity is checked at commit; see
  [Reducer work per collapse](#reducer-work-per-collapse). Changed positions are re-ranked in parallel
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
  mispredicts on short neighbor loops and by attribute-quadric cache misses, not by
  arithmetic. SIMD cut instructions by about 8% but wall time by only 3–4%. Compiling the whole
  loop for AVX2 with forced inlining was slower (instruction-cache pressure).
- **Cluster-parallel reduction** (meshes ≥ 120k triangles): 32k-triangle Morton clusters are
  reduced concurrently with locked borders (interiors to the final density), then one pass
  handles the border bands. Clusters share the whole mesh's normalization frame. Output is
  identical on any machine and thread count.

### Reducer work per collapse

Why farthing is slower than meshoptimizer: the quality options account for about 1.6×. The
rest is work per candidate. farthing checks validity eagerly for every ranked candidate,
builds wedge maps in f64 for each one, and costs every neighbor in full. meshoptimizer works
in f32, checks flips only when it commits a collapse (about 1.03–1.3 checks per collapse),
and finishes in 4–6 passes. Two changes cut farthing's share of that work:

- **Bounded ranking.** A position's candidates are first ranked by a geometric lower bound on
  their cost. Attribute and seam terms only add to the cost, so the attribute quadrics and
  wedge maps are evaluated cheapest-bound first, and the search stops once the next bound
  can't beat the best full cost found so far. Ties keep neighbor order. Within one ranking,
  the wedge map built for costing is reused by the validity and UV-drift checks, and by the
  collapse itself. Output is bit-identical (all bench-compare hashes match).
- **Lazy validity.** The batched loop ranks positions by cost alone and checks validity only
  when a candidate comes up for commit, the way meshoptimizer does. If a candidate's
  neighborhood hasn't changed since it was ranked, the commit re-runs only the validity check,
  not the cost. When a candidate fails at commit, its position goes back to eager ranking, so
  the same invalid collapse can't come back every round. A round that only switches positions
  to eager ranking doesn't count as a stall.

Work counts over bench-compare's 36 reductions (9 assets × 50/20/5/1%), against the eager
build. Counts, not times, because this machine is too noisy to time:

| per collapse | before | after | change |
|---|---|---|---|
| wedge maps | 35.9 | 23.9 | −33% |
| full cost evaluations | 27.0 | 22.0 | −18% |
| attribute-quadric evaluations | 198 | 155 | −21% |
| validity checks (link + flip) | 8.30 | 4.10 | −50% |
| UV-drift checks | 7.70 | 4.41 | −43% |
| rounds (total) | 1803 | 1884 | +4.5% |

Lazy validity changes which collapses get taken, so quality moves by about as much as
changing the batch fraction from 0.125 to 0.12 or 0.13. Over the run, Hausdorff is +1.4%,
image-space Hausdorff +2.5%, mean error and UV p90 are equal or better, normals +2%, and more
cases reach their target (92% vs 89%). Hausdorff by target (geometric mean, after ÷ before,
only rows with equal triangle counts): 0.98 at 50%, 0.95 at 20%, 1.06 at 5%, 1.22 at 1%.
The 1% rows keep only 50–200 triangles, so single collapses decide them; batch-fraction noise
alone moves them by 11–14%.

We also tried evaluating the attribute terms in f32, keeping the sums in f64. Over the run it
gave +7% Hausdorff, +12% mean error and +19% UV p90. That is at the top of the noise band
(the batch-fraction runs gave +4–6%, +9% and +13–18%). Mean error was +12% both with and
without lazy validity. Its speed gain can't be measured here, so it was left out.

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

### Realtime mode (v21)

`Quality::Realtime` (`farthing reduce --realtime`) is for reductions that must fit in a frame,
such as streaming a heavy asset to a viewer. Edge collapse spends nearly all of its time on the
first 90% of collapses, which barely affect the result; the visible error comes from the last
ones. So stage one gets close to the target in linear time and only the tail is collapsed edge
by edge:

1. Vertices are radix-sorted by Morton code, so every octree cell is a contiguous range. Each
   vertex carries a moment-matched quadric of its fan, plus border planes found without an edge
   table (the next and previous corner-id sums, and their squares, cancel around a closed fan
   and give an open fan's two border neighbors).
2. Cells are split worst-first by quadric error plus a normal-cone term. Flat regions stay
   coarse and curved ones refine, which a uniform grid can't do. Splitting stops at about 0.55
   clusters per target triangle.
3. Each cluster keeps the summed quadric of the original surface and is represented by its
   member vertex of least error, so no vertex is invented and attributes stay exact. Members
   joined by an edge inside the cluster share a wedge, so UV seams and hard edges survive as
   separate wedges at one position.
4. A lazy greedy collapse tail takes the roughly 2× overshoot down to the target. It uses the
   inherited quadrics, a wedge-distance attribute cost, border sliding, and link and flip checks.

Steps 1 to 3 run over a fixed number of chunks of the Morton order. Wedges, seams and clusters
all lie inside one cell, so each chunk finishes on its own, and a shared union-find that keeps
each set's smallest rank as its root makes the result identical for any thread count. A
competitor can't get the same speedup by adding workers to edge collapse: the parallelism comes
from the clusters being local by construction.

Three large assets reduced to 3,000 triangles, every mesh as one (`RD_METHODS=... cargo run
--release -- realtime <assets...>`). Machine: Core Ultra 7 155H pinned to 4 P-cores (8
threads). Time is the best of 5 runs; peak MB counts heap allocated during the call.

| asset | method | Hausdorff % | mean % | UV p90 | normals ° | ms | peak MB |
|---|---|---|---|---|---|---|---|
| ChronographWatch 100k | farthing realtime | 2.150 | 0.188 | **0.765** | **43.5** | 14.4 | 10.3 |
| | meshopt sloppy | 2.670 | 0.370 | 2.241 | 64.3 | **6.7** | 9.9 |
| | meshopt attr + permissive | **2.087** | 0.206 | 0.775 | 44.2 | 184 | 9.9 |
| | farthing fast | 3.105 | 0.285 | 0.794 | 51.4 | 400 | 51.4 |
| CarConcept 213k | farthing realtime | 3.725 | **0.212** | **17.70** | **45.6** | 20.0 | 17.4 |
| | meshopt sloppy | **3.649** | 0.322 | 129.2 | 73.8 | **14.5** | 17.1 |
| | meshopt attr + permissive | 6.103 | 0.323 | 29.38 | 55.9 | 302 | 17.1 |
| | farthing fast | 8.043 | 0.369 | 80.70 | 64.4 | 1592 | 104.2 |
| DragonAttenuation 135k | farthing realtime | 1.985 | 0.172 | 0.0123 | 13.0 | 11.6 | 13.5 |
| | meshopt sloppy | 1.905 | 0.098 | 0.3417 | 13.0 | **11.1** | 9.9 |
| | meshopt attr + permissive | **0.751** | 0.082 | 0.0045 | **6.2** | 449 | 9.9 |
| | farthing fast | 0.754 | **0.037** | **0.0016** | 6.8 | 500 | 63.4 |

Geometric means:

| method | Hausdorff % | mean % | UV p90 | normals ° | ms |
|---|---|---|---|---|---|
| farthing realtime | 2.51 | 0.190 | 0.550 | 29.5 | 15.0 |
| meshopt sloppy | 2.65 | 0.227 | 4.62 | 39.5 | **10.3** |
| meshopt attr + permissive | **2.12** | 0.176 | **0.467** | **24.8** | 292 |
| farthing fast | 2.66 | **0.157** | 0.471 | 28.2 | 683 |

Realtime is about 20× faster than meshoptimizer's attribute-aware simplifier at a small cost in
error, and better than its sloppy simplifier (the only method this fast) on every geometric
mean, with 8× lower UV error. That's because sloppy ignores attributes and seams. The default
reducer isn't listed: at these ratios it deletes hidden parts, and plain Hausdorff counts that
against it.

### Realtime mode, v22: finish step and seam-aware border planes

Three changes, each judged by rendered images (`bench-compare rtexport`, then the engine PoC's
`score`: every mesh of an asset reduced by the same ratio, rendered with its own textures from
three views, PSNR against the original; 11 assets):

1. Each output normal is the area-weighted mean over every input vertex merged into its wedge
   (it was one member's).
2. After the tail, each free cluster moves along its surface normal to the minimum of its
   summed quadric, within a quarter of its shortest edge. Moving it freely scored worse on
   detailed textures, because the UVs stay those of a member vertex.
3. Border planes go only on true border edges. An open fan's end edge where another fan at
   the same position (facing the same way) also ends is a seam or an unwelded edge; planes
   there pinned most vertices of an unwelded mesh. A position hash in the sort key puts such
   vertices next to each other, so the test reads only rank neighbors.

Mean PSNR over the 11 assets (dB, higher is better):

| | 10% of input | 3% of input |
|---|---|---|
| realtime, v21 | 21.98 | 18.64 |
| + wedge normals | 22.55 | 19.12 |
| + position solve along the normal | 22.74 | 19.29 |
| + seam-aware border planes (v22) | **23.13** | **19.50** |
| farthing fast | 24.02 | 18.94 |
| farthing (default) | 23.81 | 18.76 |
| meshopt attr + permissive | 24.01 | 19.47 |
| meshopt sloppy | 18.94 | 17.06 |

Realtime now leads every method at 3% and is 0.9 dB behind fast mode at 10%. The same three
large assets at 3,000 triangles as above:

| asset | Hausdorff % | mean % | UV p90 | normals ° |
|---|---|---|---|---|
| ChronographWatch | 2.491 (v21: 2.150) | 0.154 (0.188) | 0.742 (0.765) | 38.0 (43.5) |
| CarConcept | 3.690 (3.725) | 0.175 (0.212) | 15.31 (17.70) | 41.2 (45.6) |
| DragonAttenuation | 1.493 (1.985) | 0.113 (0.172) | 0.0515 (0.0123) | 10.6 (13.0) |
| geometric mean | 2.39 (2.51) | 0.145 (0.190) | 0.836 (0.550) | 25.5 (29.5) |

The mean error is now below fast mode's (0.157) and meshoptimizer's attribute-aware mode
(0.176) at this count. The dragon scene's UV error rose (its cloth has no seam planes holding
its unwelded vertices any more) while its rendered score went up 0.2 dB.

Time, on a GitHub-hosted Linux runner (4 logical cores; `Bench` workflow, minimum over 60
calls, v21 and v22 taking turns): ChronographWatch 9.4 → 9.9 ms, CarConcept 12.5 → 13.4 ms,
DragonAttenuation 9.0 → 9.6 ms across the three changes. Local timings on the development
machine are not comparable with the v21 table (it was busy), so that table's times stand for
v21.

### Realtime mode: small parts and folds (v23)

- **Small targets get more clusters.** The cluster budget assumes a closed surface's two
  triangles per cluster. A cluster on an open border has one fewer, and at a small target
  most clusters are on a border: the watch's crystal, a 62-triangle disc with a target of 6,
  came out as 2 triangles (a square) where fast mode and meshoptimizer kept a hexagon. The
  budget is now scaled by 1 + 24 / (target + 8), which fades out by a few hundred triangles,
  and a mesh of at most 20,000 triangles that still lands a quarter under its target is
  reduced again with more.
- **Folds go first.** Clusters stand at member vertices chosen one by one, so three of them
  can form a triangle facing the other way from the one it stands for: 0.6% to 6% of the
  triangles entering the tail (the watch 4.4%, DamagedHelmet 2.9%). The tail now takes
  collapses that remove such a triangle before any other, and refuses a collapse that turns
  a triangle against the original surface (a triangle of no area had no "before" to compare
  with). Folded triangles left at 10%: DamagedHelmet 41 to 1, the watch 395 to 82, the dragon
  scene 285 to 23, the car 503 to 216.

Mean PSNR over 11 assets: 23.87 to 24.02 dB at 10%, 19.88 to 20.10 dB at 3%. On the runner a
single large model takes 2 to 5% longer (ChronographWatch 13.9 to 14.6 ms, most of it the
fold test's reads when triangles are built) and a scene reduced mesh by mesh 5 to 17%
(CarConcept's 97 meshes to 10%: 39.8 to 44.7 ms).

A larger budget everywhere scores higher still: 0.715 clusters per target triangle in place
of 0.55 gives 24.45 dB at 10% and 20.28 dB at 3%, because the tail then decides more of the
result. It is not the default: the tail does about 55% more work (single models 11 to 18%
slower, scenes 25 to 30%), and the car as one mesh at 3,000 triangles gets worse (Hausdorff
3.7% to 4.1%).

### Realtime mode and the allocator (v22)

A realtime call allocates 15 to 22 MB in a few dozen large blocks and frees them when it
returns. What that costs depends on the program's allocator. Same code, same runner, the
`Bench` workflow with `bench-compare`'s `system-alloc` feature against its default (mimalloc),
3,000-triangle target, 4 logical cores:

| asset | Windows, mimalloc | Windows, system allocator | Linux, mimalloc | Linux, system allocator |
|---|---|---|---|---|
| ChronographWatch | 15.6 ms | 19.8 ms | 17.2 ms | 17.9 ms |
| CarConcept | 20.6 ms | 26.7 ms | 24.2 ms | 24.6 ms |
| DragonAttenuation | 16.2 ms | 21.1 ms | 18.0 ms | 18.8 ms |

(The two machines differ, so only compare within one system.) On Windows the system heap hands
each large block back to the system at once, and the next call takes page faults on fresh
zeroed pages again: 27 to 30% slower, spread over every phase that fills a new array (the face
pass goes from 2.7 to 4.1 ms on the watch). On Linux it is 2 to 4%. A program that calls the
realtime reducer repeatedly on Windows should set a caching global allocator such as mimalloc;
every time in this document is measured with it.

### Specular error (v22)

A still under fixed light hides shading-normal error on glossy surfaces; a moving light or
camera shows it as highlights in the wrong place. `metrics::specular_error` measures that
without a renderer. Points on the original are matched to the closest point of the reduced
mesh; each material's roughness, metalness and base color (textures included) give the
highlight's width and strength there; and per reduced triangle the original's mean shading
normal is compared with the reduced mesh's. Detail of the original inside a reduced triangle
is lost whatever the reducer does, so it widens the lobe compared against (roughness from
normal variance) instead of counting as error. The score is the share of the model's
highlight, weighted by area and gloss, that lands in the wrong place. A first version that
compared point by point saturated on fine detail (63% "misplaced" on the dragon at 7° of
error for every reducer) and no reducer setting moved it.

Mean over 11 assets (`bench-compare rtexport` prints it per asset), with the image score
beside it:

| method | misplaced highlight, 10% | PSNR, 10% | misplaced highlight, 3% | PSNR, 3% |
|---|---|---|---|---|
| farthing (default) | 1.22% | 23.82 | 2.41% | 18.76 |
| farthing realtime | **1.21%** | 23.29 | **1.51%** | **19.53** |
| meshopt attr + permissive | 1.46% | 24.01 | 2.69% | 19.47 |
| farthing fast | 2.05% | **24.02** | 3.50% | 18.94 |
| farthing fast with `material_gloss` | 1.73% | 23.88 | 3.01% | 18.82 |
| meshopt sloppy | 4.78% | 18.94 | 4.53% | 17.06 |

Fast mode is the weak one: 24% of the watch's highlight misplaced at 10%, against 9% for
realtime. `SimplifyOptions::material_gloss` (a weight per material, from
`GltfMaterial::gloss_weight`) makes a deviating normal cost up to 30 times more on glossy
materials in fast mode: 15% less specular error for 0.13 dB. Realtime did not respond to the
same weighting in its tail or to importance in its clustering, and is already the lowest;
the default reducer responded unevenly (BoomBox better, the watch worse), so neither uses the
option yet.

Realtime's normal-spread weight in cell splitting was retuned on these runs (0.5 to 0.125):
+0.17 dB at 10%, +0.03 dB at 3%, specular error unchanged or lower.

Realtime's wedges that meet at one position with the same UVs and colors (split by normals
only: hard edges, unwelded faces) now take their attributes from one member, the nearest to
the cluster's position among them all, and keep their own mean normal. Before, a wedge with
no member at that position copied the UV of its own nearest member and shifted the texture
by that distance; the dragon scene's checkered cloth showed it over its whole surface. Mean
PSNR 23.29 to 23.87 dB at 10% and 19.53 to 19.88 dB at 3% (dragon scene 20.4 to 26.5 dB at
10%), specular error unchanged. It costs 0.75 to 1.0 ms per call on the runner (about 7%:
ChronographWatch 13.0 to 14.0 ms in that run), most of it reading each duplicate's UVs.

### Default reducer on seam-dense meshes (v22)

The dragon scene's cloth has 62,570 vertices on 22,199 positions, split by normals only.
The default reducer treated a wedge with no partner across the collapsed edge as a UV tear
unless it and its target wedge had the same UVs, which two wedges at different positions
never do. Every such merge was priced as a full tear, so the wedge kept its own attributes
at the new position instead, planting a real UV break inside one chart; the gutter guard
then refused merges around it and the reduction locked up. A merge is now tear-free when the
wedge has the UVs of a partnered wedge whose target has the target wedge's UVs.

| case (Hausdorff % / mean % / UV p90 / normals °) | before | after | fast |
|---|---|---|---|
| cloth to 1,000 triangles | 3.051 / 0.1269 / 0.0924 / 5.15 | 0.506 / 0.0316 / 0.0009 / 1.56 | 0.431 / 0.0214 / 0.0009 / 3.16 |
| cloth, welded first | 30.07 / 4.299 / 0.492 / 31.8 (stalls at 1,223) | 0.506 / 0.0315 / 0.0008 / 1.54 | 0.431 / 0.0215 / 0.0009 / 3.16 |
| dragon scene as one mesh, 13,500 (cluster-parallel) | 1.586 / 0.0513 / 0.0034 / 2.59 | 0.527 / 0.0141 / 0.0004 / 1.90 | 0.280 / 0.0100 / 0.0004 / 2.71 |
| 8 assets x 4 ratios, geometric means | 2.499 / 0.1895 / 0.0099 / 7.38, 94% reached | 2.396 / 0.1723 / 0.0084 / 6.93, 97% reached | 2.257 / 0.1413 / 0.0093 / 8.18 |

Meshes with real UV seams everywhere (the dragon itself, the watch, the car) are a second
problem, not fixed by default: removing a seam vertex along a seam that bends in UV always
spills past the island's edge, the gutter guard (`DRIFT_UV`, 1/2048) forbids it, and the
vertex can then only slide, at a price that drags the accepted geometric error up. On the
dragon to 1,000 triangles, half of the UV validity checks fail on spill.

(Since v24 the default prices spill instead of forbidding it, see "The default reducer
against the fast one" below; what follows describes `Gutter::Guarded`.)

The opt-in `gutter_use` lets a seam vertex go where the atlas has room for it. A spill of
depth d is allowed if no other island's source triangle, and no texture edge, lies within
d x (1 - share) / share of it (with a small margin), so a spill takes at most its share of
the gutter's width at that place and two islands spilling toward each other cannot meet.
The room is probed where the spill lands, against a grid of the source UV triangles. Each
spilling triangle also pays `uv_weight` x its area x d squared, through the same cost as any
other UV error, so a collapse that does not spill wins when it is about as good. The depth is
capped at share / 32 of the texture, half a texel is always allowed as before, and
`texel_size`, when given, is added to the free distance required. A material whose islands
cross a texture repeat keeps the old rule. At 0 (the default) nothing changes.

| case (Hausdorff % / mean % / UV p90 / normals deg) | 0 | 0.1 | 0.25 | 0.4 |
|---|---|---|---|---|
| dragon, 1,000 | 11.16 / 0.651 / 0.824 / 32.3 | 5.59 / 0.585 / 0.826 / 28.9 | 4.90 / 0.517 / 0.774 / 25.8 | 8.95 / 0.451 / 0.763 / 24.3 |
| dragon, 10,000 | 1.064 / 0.0622 / 0.550 / 7.03 | 0.838 / 0.0580 / 0.471 / 6.60 | 0.751 / 0.0548 / 0.420 / 6.31 | 0.850 / 0.0548 / 0.385 / 6.12 |
| watch, 3,000 | 9.11 / 0.963 / 2.02 / 72.0 | 15.79 / 1.542 / 2.09 / 76.8 | 8.27 / 0.889 / 2.16 / 68.3 | 12.43 / 1.019 / 2.07 / 71.7 |
| car, 10,000 | 6.02 / 0.188 / 22.2 / 39.2 | 4.96 / 0.200 / 22.7 / 39.5 | 4.96 / 0.197 / 22.5 / 38.8 | 4.96 / 0.197 / 22.7 / 38.9 |
| 8 assets, means | 2.396 / 0.1723 / 0.0084 / 6.93, 97% | 2.290 / 0.1627 / 0.0075 / 6.79, 97% | 2.288 / 0.1574 / 0.0070 / 6.66, 100% | 2.151 / 0.1559 / 0.0067 / 6.58, 100% |
| UV area over half a texel off its island (mean, worst) | 0.008%, 0.26% | 0.016%, 0.10% | 0.058%, 0.50% | 0.111%, 0.92% |
| UV area inside another island | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| rendered PSNR, 11 assets, 10% of input | 24.43 dB | 24.53 | 24.52 | 24.64 |
| rendered PSNR, 3% of input | 19.38 dB | 19.60 | 20.03 | 20.18 |
| misplaced highlight, 10% / 3% | 1.37% / 2.36% | 1.31% / 2.32% | 1.28% / 2.26% | 1.24% / 2.11% |

The dragon is where it pays (mean error down 12 to 21% at 0.25). The watch and the car, each
flattened to one mesh, get almost nothing: their meshes stack in one UV space and many of
their materials tile, so nearly every probe is blocked, and the watch at 3,000 triangles
swings between 8.3% and 15.8% with no trend. A cap of the same depth with no room test (the
first version of this option, `gutter_tolerance`, removed) scored higher in renders (24.97
and 25.24 dB at 10% for 0.01 and 0.05) but put 0.51% of UV area over half a texel into the
gutter at 0.05 and, on the watch, nearly all of its spill inside another island.

It is off by default because of its cost and because
`uv_islands_never_spill_into_the_gutter` asserts the old rule. The closest-point UV metric
overstates slides (a seam moved by one edge length reads as a jump between islands), so the
UV columns here are harsher on the default reducer than a render is.

Time with it on, on a GitHub-hosted Linux runner (`Bench` workflow's `run` job, each asset as
one mesh to 10,000 triangles, best of 3): dragon scene 575 → 1,386 ms, car 1,521 → 2,659 ms,
watch 630 → 1,249 ms, about twice as long. Probing every spill against the source triangles
is the cost; it needs to get cheaper before the option can be on by default.

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
collapse onto a junction neighbor, which shortens or merges junction curves. The link
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
- **Chart merging.** Neighboring charts merge, longest shared seam first, while the union
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
for cells the source does not touch. A touched cell is now solid only if its center is inside
the source:

1. **Sign.** The closest point's angle-weighted pseudo-normal (face, edge or vertex: Bærentzen
   & Aanæs 2005) gives the sign, exact for closed, consistently wound meshes. A center found
   outside is carved only if none of three winding rays finds it enclosed. Each ray counts +1
   where it leaves a face and −1 where it enters. Inside a body, the closest face can belong
   to a nested or overlapping piece that faces outwards; the rays still see the body. Rays
   that leave through a hole of an open mesh count 0 or −1, never "enclosed". The dragon is
   open: 1.2% of its faces read as inverted and 4% as ambiguous.
2. **No lost surface.** Surface samples at half-cell spacing each need a solid cell center
   behind them, within a cell, that lies inside a volume: an interior cell, or a boundary cell
   that all three rays find enclosed. Otherwise the cell stays solid. Sheets thinner than a cell
   have no such center and survive, where a signed-distance contourer loses them. On the dragon,
   baby_shark's MC leaves 11,402 source samples more than half a cell outside its unreduced
   proxy; farthing leaves 1,541. The other side is where the dragon still trails: 55,990
   samples lie more than half a cell inside farthing's unreduced proxy (v11: 58,765;
   baby_shark: 2,275). They sit in slots between pieces, where no cell center falls in the gap.
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

**v13: slot pass tried and reverted.** A second pass opened boundary cells whose center is
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

A guarded carving pass was tried. It carves solid cells whose center lies outside the source:
the closest point is inside a face, the center is on the face's outer side, and there are facing
walls within a cell. A cell is carved only if every surface sample in it keeps a kept solid
cell behind it among the 26 neighbors. Results:

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

### Realtime: borders, seam tips and hard edges (v24)

Four changes from looking at why realtime trails the default on the watch and the helmet:

- A vertex where a UV seam ends has an open fan whose two ends are copies of one position; it
  was given two border planes (122 such vertices on the Duck, which has no border). No more.
- A cluster on both a border and a junction slides along its border instead of being locked
  (1,880 of 37,500 clusters at 10%).
- Border clusters take part in the position solve along the normal (since withdrawn for
  clusters on an open border, see below).
- In the tail, a wedge with no partner across the collapsed edge is kept when its normal
  differs by more than about 29 degrees, instead of merging into the nearest wedge.

Mean rendered PSNR over the 11 assets: 24.02 to 24.46 dB at 10%, 20.10 to 20.37 at 3%. Largest
gains at 10%: WaterBottle +1.55, Lantern +1.32, Avocado +1.11, Fox +0.56. Largest loss: the
watch, -0.16 at 10% and -0.10 at 3%. Specular impact unchanged. Same-run A/B on a 4-thread
Linux runner: 14.34 to 14.64 ms and 18.21 to 17.97 ms per call at a 3,000-triangle target,
29.98 to 30.99 and 31.60 to 32.27 ms at 20,000; the tail alone is 7 to 11% slower (the position
solve runs on half as many clusters again).

The watch and helmet themselves did not move. The helmet loses to seams leaving their line in
stage one (in 294 of its 695 seam clusters the representative is not a seam vertex) and to
struts thinner than a cell being pressed flat; the watch to plain geometric error in three
large meshes (mean distance about twice the default's). Both want smaller cells, that is, a
larger budget. Not measured: cracks between separately reduced meshes that share a border,
whose border vertices can now move up to a quarter of their shortest edge along the normal
(`lock_border` still pins them). Measured next.

Cracks between meshes reduced one at a time (`bench-compare rtcrack`): source border edges
are marked shared where both ends coincide with another part's border. Shared border is
67.8% of the watch's border length, 7.5% of the car's, none on the dragon scene. Each reduced
border vertex is measured against the neighboring part's reduced border; "sealed" is the
share that lands on it. Gap in % of the scene diagonal.

| shared borders (sealed / mean gap / 95th pct) | before | position solve on borders | open borders kept |
|---|---|---|---|
| watch, 10% | 22.6% / 0.162 / 0.81 | 0.5% / 0.176 / 0.87 | 30.3% / 0.145 / 0.83 |
| watch, 3% | 7.4% / 1.31 / 8.3 | 0.7% / 1.03 / 4.2 | 11.6% / 1.03 / 4.8 |
| car, 10% | 31.7% / 0.279 / 1.25 | 1.9% / 0.292 / 1.15 | 32.2% / 0.271 / 1.25 |
| car, 3% | 17.3% / 0.587 / 1.99 | 0.0% / 0.661 / 2.55 | 17.3% / 0.587 / 1.99 |

Moving border clusters along the normal unsealed nearly every vertex that two neighbors had
picked in common. A cluster holding an open-border input vertex now keeps that position
(junction-only clusters still move). Mean rendered PSNR: 24.46 to 24.19 dB at 10%, 20.37 to
20.23 at 3%, still above the 24.02 / 20.10 before these changes. The loss is mostly
WaterBottle (30.93 to 29.38 at 10%) and Avocado (29.16 to 28.04), single meshes with nothing
to crack against, but one call cannot tell. Sealing at most a third of shared vertices still
leaves gaps of 0.15 to 1% of the diagonal on average: meshes reduced apart diverge whatever
the border rule, and `lock_border` or reducing a scene as one mesh are what close them.

### Realtime: border planes per position (v24)

Realtime finds open borders without an edge table, from sums of each vertex's next and
previous corner ids. Those sums were kept per vertex, so on an unwelded mesh each copy of a
vertex held only a piece of its fan, every piece read as an open border, and the border planes
went in at full weight. The dragon scene's cloth (62,570 vertices on 22,199 positions) was
nailed flat by them. Corners now count by position, so the copies at one position facing one
way add up to the whole fan, and a third sum (a non-linear mix of the corner ids, kept in the
sums' padding) rejects fans of two arcs that the two-moment recovery read as borders. On the
cloth, mean distance falls from 5.65e-4 to 2.19e-4 and normal error from 4.0° to 2.1°.

| rendered PSNR, dB | 10%: before | now | 3%: before | now |
|---|---|---|---|---|
| ChronographWatch | 19.78 | 20.02 | 15.33 | 15.32 |
| DragonAttenuation | 26.85 | 28.80 | 21.48 | 24.66 |
| other nine assets | within ±0.06 | | within ±0.06 | |
| mean of 11 | 24.19 | 24.38 | 20.23 | 20.52 |

Shared borders sealed between separately reduced meshes (`bench-compare rtcrack`): watch
30.3% to 40.0% at 10% and 11.6% to 14.7% at 3%; car 32.2% to 34.3% and 17.3% to 17.4%. Time on
a hosted runner (4 cores, minimum of 12 rounds, same run): no measurable cost (watch 30.90 to
30.38 ms, car 57.49 to 57.15, dragon 28.83 to 27.62 at 10%). `bench-compare rtmesh` gives
per-mesh distance and normal blame against the default reducer, and `rtsolo` writes one file
per mesh with only that mesh reduced, for rendered blame.

### Realtime: nine smaller passes (v24)

Output is byte-identical (hashes of mesh, remap, source vertices, error and collapse counts over
the whole corpus at 300, 3,000 and 20,000 triangles, an error bound, locked borders, and each
mesh at 10% and 3%). Changes: runs of one position joined without pairwise compares,
fine-level counts in plain loops over blocks, face sums filled in parallel, stage-two triangle
data gathered in parallel, fold checks skipped in stars without folds, Morton bits spread from
a table, fine cells keeping their end codes for the octree, and the stage-two triangle set in
one 64-bit word per slot (8 bytes in place of 20). A hosted runner (4 threads, minimum over 16
rounds, same run against the previous main):

| ms | ChronographWatch | CarConcept | DragonAttenuation |
|---|---|---|---|
| 3,000 triangles | 14.36 → 13.70 | 18.80 → 17.98 | 13.86 → 12.77 |
| 20,000 triangles | 30.49 → 28.52 | 31.69 → 29.92 | 26.05 → 24.06 |

The serial collapse tail is now the largest single phase at a 3,000 target (about 2.5 to 3.5
ms): on the watch, 4,672 heap pops for 1,884 collapses at roughly 0.75 to 1 µs each, spread
over cost evaluation, link checks, wedge mapping and the collapse itself with no single hot
spot.

### Realtime: thin cells cut into sectors (v24)

A tube or strut narrower than an octree cell fell to one cluster per cell and its triangles
vanished (the BoomBox antenna, straps and loops). A cell is a candidate when its normals cancel
by at least 30%, its RMS distance to the centroid's planes is under half the cell size, and its
normal spread is that of a tube rather than a plate. It counts as three clusters while it is a
leaf, so the budget stays the same (1.2 × the clusters with no cut gives +0.45 / +0.10 dB on its
own, so every comparison here is at equal budget). When the budget is reached each candidate is
checked over its vertices: an axis from the normal moment, three 120° sectors around it, and
the error those sectors remove against the heap's next split. Kept cuts become three cells;
rejected ones hand their clusters back and refinement resumes. Two sectors leave flat ribbons
facing apart that draw as dark slivers; four gain nothing over three. No cuts on cells holding
an open-border vertex, or on meshes with open borders under 300 clusters, which keeps shared
borders sealed exactly as before (watch 40.0% / 14.7%, car 34.3% / 17.4% at 10% / 3%).

| rendered PSNR, dB | 10%: before | now | 3%: before | now |
|---|---|---|---|---|
| BoomBox | 22.01 | 23.47 | 19.44 | 19.44 |
| CarConcept | 23.94 | 23.82 | 20.26 | 20.24 |
| ChronographWatch | 20.02 | 19.95 | 15.32 | 15.33 |
| DragonAttenuation | 28.80 | 28.89 | 24.66 | 24.88 |
| Duck | 21.84 | 22.03 | 17.11 | 17.20 |
| Fox | 18.16 | 18.23 | 15.47 | 16.10 |
| other five | unchanged | | unchanged | |
| mean of 11 | 24.38 | 24.53 | 20.52 | 20.60 |

The helmet's hoses are not caught: a threshold loose enough for them costs 0.9 dB elsewhere.
Time on a hosted runner (same run, 3,000-triangle target on the merged scene): watch 14.13 to
14.43 ms, car 18.61 to 18.70, dragon 13.73 to 14.11, all in the octree phase.

### Fast: wedges carried across UV seams (v24)

When p moved onto q, a wedge of p that no triangle joins to q was mapped to the q wedge with
the nearest attributes, on whatever UV island that was; its triangles then spanned two
islands and showed the atlas between them, and later collapses stretched them (the Lantern's
post: 22.3% of its surface with UVs outside the source islands, 484 of 2,443 collapses at 10%
tore a wedge). Such a wedge is now carried: it keeps its own attributes and moves to q's
position, at a price of its area x 16 x the largest squared UV move of the joined wedges.
Off when UVs are not weighted or `locked_vertices` is given (the cluster builder rebuilds
triangles from `source_vertex` and needs exact copies).

Mean rendered PSNR over the 11 assets: 24.02 to 25.03 dB at 10%, 18.94 to 20.05 at 3%.
Lantern +5.79 / +5.39, DamagedHelmet +1.43 / +0.92, watch +1.37 / +0.19, Fox +1.07 / +1.83,
Avocado +0.79 / +2.59. One case worse: Corset at 3%, 23.70 to 22.86, a case that moves 0.8 dB
under 10% weight changes. Specular impact 0.0205 to 0.0160 at 10%, 0.0350 to 0.0268 at 3%.
Output vertices rise 21% (to the default reducer's level). Work counts flat (attribute-cost
evaluations down 7%); on cloud runners the dragon, watch and car flattened to 10,000
triangles took 437 ms before and 405 ms after, on different machines.

Fast now renders closer than the default by more than 0.1 dB on WaterBottle and Corset at
10% and on the watch, dragon and Fox at 3%. Still open in fast: the car's dented panels
(3.5 dB behind the default) and the Lantern's skewed post top.

### Fast: attribute cost along the surface gradient (v24)

Fast mode priced a collapse by how far the removed wedge's UVs and normals sat from the target
wedge's, as they are. On a smooth panel or inside a UV island every collapse paid for the
distance it moved, an edge-length term that spread the budget evenly and broke shading on
curved panels (the car's hood alone: 18.74 dB, against 21.72 for the default reducer). Each
wedge now carries its attributes' area-weighted surface gradient J and pays for
attr_q − attr_p − J_p·(x_q − x_p), near zero wherever attributes change linearly. Sharp
creases (over 30°) beside a face narrow across the edge (area under 0.3 × length²) also get a
constraint plane, weight 2 × length² × (1 − cos)/2, so a thin panel's outline no longer sinks
inward (the windshield). The default reducer's two v24 caps do not carry over: fast mode's
attribute term grows with area directly, and crease planes on every edge cost it.

Means over five neutral seeds:

| rendered PSNR, dB | 10%: before | now | default | 3%: before | now | default |
|---|---|---|---|---|---|---|
| Avocado | 28.45 | 28.89 | 28.74 | 22.81 | 22.50 | 23.19 |
| BoomBox | 23.22 | 23.88 | 24.08 | 19.41 | 19.83 | 20.28 |
| CarConcept | 23.99 | 27.48 | 27.55 | 18.65 | 20.20 | 21.98 |
| ChronographWatch | 21.31 | 21.65 | 21.44 | 15.87 | 16.38 | 16.16 |
| Corset | 30.29 | 30.71 | 30.10 | 23.02 | 23.99 | 25.91 |
| DamagedHelmet | 19.42 | 19.76 | 20.17 | 14.86 | 15.56 | 15.24 |
| DragonAttenuation | 29.95 | 29.50 | 29.76 | 25.32 | 24.85 | 24.97 |
| Duck | 21.46 | 21.32 | 22.07 | 16.88 | 16.80 | 16.56 |
| Fox | 18.22 | 18.82 | 18.72 | 15.67 | 15.04 | 15.07 |
| Lantern | 26.28 | 28.71 | 29.49 | 24.00 | 24.71 | 24.98 |
| WaterBottle | 32.34 | 31.86 | 32.25 | 23.99 | 23.55 | 24.13 |
| mean | 25.00 | 25.69 | 25.85 | 20.04 | 20.31 | 20.77 |

The dragon loses 0.45 dB: its UVs drive only an unrendered thickness texture, and the old
edge-length pull helped it by accident. The car at 3% stays 1.8 dB behind, mostly the wipers
(thin non-manifold rods), and so does the Corset at 3% (its base disk becomes an octagon).

Time on a hosted runner (`fast=10000` in the Bench workflow, dragon, watch and car flattened,
12 rounds, same run): 1.061 × the previous fast mode in all (watch 0.997, car 1.045, dragon
1.196). Wedge data now sits in one block per wedge, and a pass near the target that makes
under a quarter of its intended collapses lifts the cost limit for the next one (the watch
used to spend five or six passes collapsing 0 to 27 edges each). The dragon's extra time is
passes: cheap candidates now cluster in flat regions and block each other, 8 passes against 6.

### The default reducer against the fast one (v24)

The fast reducer rendered closer to the source than the default on 9 of 22 cases (11 assets
at 10% and 3% of their triangles), by 4.8 dB on the watch. Causes, all in the default:

- The gutter guard forbade nearly every seam and border move on a part whose UVs run outside
  0..1 (the watch strap), and a seam tear was priced as a UV jump of 1.0, as dear as moving
  the surface 70% of the bounding-box diagonal. A collapse a fifth of the model wide was
  cheaper than a tear and went first: 23.1% Hausdorff on that part, against 1.46% in fast.
- Positions with more than 8 wedges were never removed until a stall (the watch dial).
- The geometric cost counted only the planes of the vertex removed, so error piled up on a
  few targets.

Changed: `Gutter::Priced` is the default (a spill costs like a UV error of its depth, a UV
fold like a tear, neither more than a tear; `Gutter::Guarded` keeps the hard rule and
`gutter_use`), the tear level is 0.05, the geometric cost includes the error the target
already carries, and the wedge limit is 64.

| rendered PSNR, dB | 10%: before | fast | now | 3%: before | fast | now |
|---|---|---|---|---|---|---|
| Avocado | 28.27 | 27.69 | 28.61 | 22.92 | 20.85 | 23.33 |
| BoomBox | 23.70 | 23.05 | 24.06 | 20.31 | 19.52 | 20.69 |
| CarConcept | 26.42 | 23.71 | 27.33 | 20.49 | 18.37 | 22.19 |
| ChronographWatch | 15.15 | 19.97 | 21.36 | 13.43 | 15.57 | 15.26 |
| Corset | 28.78 | 30.36 | 30.10 | 21.61 | 23.70 | 25.44 |
| DamagedHelmet | 18.44 | 18.08 | 20.12 | 13.97 | 13.98 | 15.21 |
| DragonAttenuation | 28.82 | 29.45 | 29.63 | 23.02 | 24.38 | 24.56 |
| Duck | 21.99 | 21.80 | 22.11 | 15.29 | 16.15 | 16.82 |
| Fox | 17.99 | 17.15 | 18.71 | 15.35 | 13.88 | 14.78 |
| Lantern | 28.73 | 20.47 | 28.60 | 24.12 | 18.49 | 24.83 |
| WaterBottle | 30.40 | 32.49 | 31.18 | 22.65 | 23.41 | 24.02 |
| mean | 24.43 | 24.02 | 25.62 | 19.38 | 18.94 | 20.65 |

Fast still leads on three cases: WaterBottle at 10% (1.3 dB, consistent across variants; the
default's cap comes out lopsided at 451 triangles), Corset at 10% (0.26) and the watch at 3%
(0.31), the last two inside the 0.5 to 1 dB a case moves under changes that should not
matter. Specular impact (weighted share of misplaced highlights) is 0.0089 against fast's
0.0205 at 10%, and 0.0169 against 0.0350 at 3%.

Eight assets by four ratios, geometric means (Hausdorff % / mean % / UV p90 / normals deg):
2.396 / 0.1723 / 0.0084 / 6.93 at 97% reached before, 2.101 / 0.1466 / 0.0062 / 6.36 at 100%
now, fast 2.257 / 0.1413 / 0.0093 / 8.18. With both attribute weights at zero the default's
mean error matches fast's, so the rest of that gap is what the normal term buys.

What pricing costs: UV area more than half a texel off its island is 0.21% on average (1.09%
worst), against 0.008% (0.26%) guarded; 0.009% lands inside another island (0.11% worst),
against none. The Lantern, the asset the guard was written for, renders the same (28.73 to
28.60 dB at 10%, 24.12 to 24.83 at 3%).

### Gentle creases and normal growth capped (v24)

The WaterBottle did not lose at the cap's lip. Its body and cap sides are tessellated as tall
thin strips with creases of about 9°, and a crease plane weighed edge length²: on a strip
that outweighed the faces' own planes about five times and pinned every strip, so the cap's
outline and dome ran short of vertices. The Lantern post and the watch dial lost the same way.
A crease under 30° now weighs at most 12 times its face's area; sharp creases keep the full
weight, because there the face across may be a thin part's rim. Capping every crease cost the
car 0.9 to 1.4 dB at 3% (sliver faces along sharp rims need it). Separately, the normal term's
growth with region size stops at 32 average input triangles: past that, dense smooth meshes
(the watch, the dragon) kept the shading of a few wide regions at the cost of their outline.

Single runs move a case 0.5 to 1 dB, so these are means over five neutral seeds (the normal
weight, which both reducers use, scaled by 0.94 to 1.06; jittering the target ratio does not
decorrelate runs):

| rendered PSNR, dB | 10%: before | now | fast | 3%: before | now | fast |
|---|---|---|---|---|---|---|
| Avocado | 28.51 | 28.74 | 28.45 | 23.61 | 23.19 | 22.81 |
| BoomBox | 24.04 | 24.08 | 23.22 | 20.39 | 20.28 | 19.41 |
| CarConcept | 27.41 | 27.55 | 23.99 | 22.16 | 21.98 | 18.65 |
| ChronographWatch | 21.41 | 21.44 | 21.31 | 15.49 | 16.16 | 15.87 |
| Corset | 30.09 | 30.10 | 30.29 | 25.57 | 25.91 | 23.02 |
| DamagedHelmet | 20.15 | 20.17 | 19.42 | 15.29 | 15.24 | 14.86 |
| DragonAttenuation | 29.65 | 29.76 | 29.95 | 24.57 | 24.97 | 25.32 |
| Duck | 21.95 | 22.07 | 21.46 | 16.91 | 16.56 | 16.88 |
| Fox | 18.72 | 18.72 | 18.22 | 14.96 | 15.07 | 15.67 |
| Lantern | 28.61 | 29.49 | 26.28 | 24.86 | 24.98 | 24.00 |
| WaterBottle | 31.01 | 32.25 | 32.34 | 23.62 | 24.13 | 23.99 |
| mean | 25.60 | 25.85 | 24.99 | 20.67 | 20.77 | 20.04 |

With default options: 25.87 / 20.75 dB, against 25.62 / 20.65 before and fast's 25.03 /
20.05. Cases behind fast: 7 (4.02 dB in all) before, 6 (1.75 dB) now, the largest 0.60 dB (Fox
at 3%, 16 triangles). The dragon's remaining gap is its UVs, which drive only a thickness
texture the scorer does not render: reduced without UVs the default scores 29.97 / 24.89
against 29.63 / 24.56 with them. `bench-compare rtexport` takes `MIX_FAST=<mesh indices>`
(those meshes reduced by fast mode) and `UV0_MESH=<mesh indices>` (reduced without UVs, UVs
copied back) to find which mesh of a scene a gap comes from.

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
single material (its trunk is leaf-colored; a real pipeline would bake it). Surfels are fast to
make (0.01–0.1 s, against about 1 s for the LOD and 6 s for the fitted proxy). The PoC draws
each disc as a 6-triangle fan through the ordinary mesh pipeline, opaque, so no blending is
involved; a production renderer would draw one quad per disc.

Not done yet: a per-surfel normal-map term (the procedural bump already varies per pixel here)
and a perceptual metric instead of PSNR.

### Progressive order and streaming (v14)

`progressive_surfel_cloud` orders the same surfels so every prefix is an even covering. It
continues the weighted sample elimination in halving stages, each with the neighborhood radius
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
  field as a shell 2.5 steps thick rather than solid, so rays passing behind a neighboring
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

Without the per-pass cost cap, rings blocked by a neighbor let dearer ones through, and error
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

"meshopt par" and "meshopt attr par" are the same builds with each level's groups simplified,
re-split and bounded concurrently on rayon's pool, the way clusterlod.h lets a caller run them.
Results are committed in group order, so their DAGs are identical to the serial ones (checked
cut by cut on five assets). Only build time and peak heap differ. The tables below predate
them and compare farthing against the serial baseline.

**How quality is scored.** The camera sits 4 bounding radii out (1080p, 60° vertical FOV). The
screen-error threshold is swept from 0.02 to 200 px in steps of ×1.5. Each cut is
`dag.select` → `to_mesh`, and its mean surface distance to the source is *measured* (40k
samples, in pixels at the bounds' nearest point). Each cell is the fewest triangles any cut needs
to stay under that measured error. A method that under-reports its own error gains nothing.
Raw data, including every cut's (triangles, mean px, max px) curve:
[`benchmarks/results-clusters.jsonl`](benchmarks/results-clusters.jsonl). Machine: 22 threads,
Windows 11, mimalloc. Lower is better.

| Asset (triangles) | Method | Build | Peak heap | Levels / roots | ≤ 0.25 px | ≤ 0.5 px | ≤ 1 px | ≤ 2 px |
|---|---|---|---|---|---|---|---|---|
| DamagedHelmet (15k) | farthing | 70 ms | 16.1 MB | 8 / 1 | **5,927** | 5,927 | 3,888 | 1,943 |
| | farthing fast | 34 ms | 5.2 MB | 8 / 1 | 7,270 | **4,400** | **1,943** | 1,943 |
| | meshopt | 41 ms | 1.5 MB | 4 / 26 | 7,720 | 7,049 | 5,416 | 5,416 |
| | meshopt attr | 66 ms | 1.5 MB | 8 / 1 | 7,721 | 7,721 | 3,858 | **1,926** |
| pirate (5k) | farthing | 30 ms | 7.9 MB | 7 / 1 | 2,785 | 1,630 | 1,630 | 621 |
| | farthing fast | 15 ms | 1.7 MB | 7 / 1 | 3,272 | **1,123** | **625** | **311** |
| | meshopt | 11 ms | 0.5 MB | 6 / 2 | **2,503** | 1,250 | 1,250 | 625 |
| | meshopt attr | 14 ms | 0.5 MB | 7 / 1 | 2,504 | 2,504 | 1,251 | 625 |
| Lantern (5k) | farthing | 32 ms | 8.2 MB | 7 / 1 | 674 | 674 | 336 | 168 |
| | farthing fast | 21 ms | 1.9 MB | 7 / 1 | 674 | 674 | 336 | 336 |
| | meshopt | 14 ms | 0.5 MB | 4 / 6 | 1,347 | 1,347 | 1,347 | 672 |
| | meshopt attr | 27 ms | 0.5 MB | 7 / 1 | **661** | **330** | **330** | **82** |
| DragonAttenuation (135k) | farthing | 287 ms | 29.1 MB | 12 / 1 | 16,685 | 8,741 | 7,781 | 3,323 |
| | farthing fast | 143 ms | 12.4 MB | 12 / 1 | **12,531** | **8,187** | **2,791** | **799** |
| | meshopt | 510 ms | 14.3 MB | 7 / 363 | 47,457 | 38,607 | 38,607 | 38,607 |
| | meshopt attr | 553 ms | 14.4 MB | 12 / 1 | 24,592 | 24,592 | 13,196 | 13,196 |
| Avocado (682) | all four | 1–9 ms | 0.1–1.1 MB | 4 / 1 | 682 | 341–682 | 341–682 | 169–170 |

Build is the best of three on all 22 threads. Peak heap is the most heap live during the build,
above what was live before it (the bench wraps mimalloc in a counting allocator). Most of
farthing's peak comes from parallelism: each group being reduced at the same time holds its own
reducer state. On one thread the dragon peaks at 9.7 MB (default) and 8.6 MB (fast), below
meshopt's 14.3 MB. The machine was busier for this run than for the v16 one (meshopt's dragon
build took 510 ms here, 397 ms then), so compare times within this table only.

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
- **Speed.** The meshopt rows here are serial, so farthing's multi-thread lead over them is not
  like for like; rerun with the `par` rows (`ONLY="meshopt attr par"`) for that. clusterlod.h
  itself runs groups in parallel, and zeux reports about 10.6M triangles/s on 16 threads. That is faster than farthing's
  roughly 0.5M (default) and 0.9M (fast) triangles/s in this run. farthing's levels are
  parallel: groups are reduced at the same time, and clustering uses a parallel spatial
  bisection and parallel adjacency. With the Windows system allocator, parallel scaling stalls;
  the bench links mimalloc, and engines should too.
- **Per core (v17).** Four changes came out of profiling one-thread builds. None of them changes
  the output: the clusters and errors are bit-identical, so the quality columns can't move.
  1. Part analysis is skipped when every part has a locked vertex. Group reductions lock their
     whole border, so that is nearly always the case, and the analysis took about a fifth of
     each group's reduction.
  2. The reducer state no longer makes an allocation per face, and its position weld sorts
     precomputed keys.
  3. Cluster growth numbers a block's local vertices with one sort instead of binary searches.
     It also skips candidates that already lose on their count of new vertices alone.
  4. When cluster growth falls back to the nearest loose triangle, it walks outwards along the
     sweep order with a distance bound, instead of scanning the whole block.

  Before and after are each the best of three runs, alternating the two binaries
  (`RAYON_NUM_THREADS=1` for one thread, `SPEED_ONLY=1`):

  | Asset | Method | 1 thread, before | 1 thread, after | 22 threads, before | 22 threads, after |
  |---|---|---|---|---|---|
  | DragonAttenuation | farthing | 1,507 ms | **1,235 ms** | 251 ms | **236 ms** |
  | | farthing fast | 690 ms | **653 ms** | 130 ms | **120 ms** |
  | | meshopt | 527 ms | 500 ms | 400 ms | 417 ms |
  | | meshopt attr | 541 ms | 546 ms | 500 ms | 517 ms |
  | DamagedHelmet | farthing | 204 ms | **164 ms** | 79 ms | **72 ms** |
  | | farthing fast | 84 ms | **73 ms** | 36 ms | **32 ms** |
  | | meshopt | 32 ms | 31 ms | 40 ms | 35 ms |
  | | meshopt attr | 54 ms | 56 ms | 61 ms | 53 ms |

  meshopt's code did not change, so its rows show the run-to-run noise (5–10%). Per core,
  farthing is still slower than meshoptimizer. On the dragon, default mode takes 2.3× as long as
  meshopt attr and fast mode 1.2×; on the helmet, 2.9× and 1.3×. Most of what remains is the
  reducer's own work (collapse costs with attribute quadrics, link and flip checks), plus
  cluster growth. Peak heap on 22 threads fell too, from 36.3 MB to 30.7 MB on the dragon.

### Cracks from slid wedges (v24)

The DAG builder rebuilds each simplified group's triangles from `source_vertex`, indexing the
source mesh's vertices. The default reducer slides a seam wedge with no partner onto the
vertex it collapses onto, so the rebuilt triangle put that corner back at its input position
while its neighbors stood at the target: about half the group reductions opened cracks (1,683
extra open edges over the helmet's 63 groups, 316 over the Corset's 78). Groups are now
reduced with `exact_vertices`, which turns slides (and fast mode's carry) off: no extra open
edges. The measured-error table needs more triangles where the cracked DAG had looked
cheaper, because the sampled distance does not see cracks:

| triangles for mean error at most 0.25 / 0.5 / 1 / 2 px | before (cracked) | now |
|---|---|---|
| DamagedHelmet | 5231 / 3885 / 2962 / 1940 | 7273 / 4393 / 2967 / 2967 |
| Corset | 3305 / 1650 / 1140 / 634 | 4333 / 1651 / 1140 / 629 |

### Packed GPU format (v17)

`farthing::cluster::gpu` (CLI: `farthing clusters --gpu out.fcld`) packs a DAG into flat
little-endian buffers a renderer can upload as they are. Every record is a whole number of
16-byte vectors, so each array reads as a std430 struct array. The file holds:
- 128-byte cluster headers and 48-byte group records;
- a page table that lists each page's dependencies;
- fixed-size streaming pages of quantized vertices and indices.

Each vertex takes 16 bytes. Positions sit on one grid over the whole mesh (16 bits per axis by
default), so cuts that mix pages stay crack-free. Normals are octahedral and UVs unorm16.
(Format version 2 later added opt-in tangents, colors, a second UV set and skin weights,
which grow the vertex past 16 bytes, and made the page planner public as
`farthing::cluster::plan_pages` for custom formats.)

The clusters one group produced always land in one page. Pages run coarse to fine, so a page's
dependencies always come earlier in the file. `GpuClusterDag::select` is a CPU reference of the
GPU's LOD test, and it picks the same clusters as `ClusterDag::select`. With the default 128 KiB
pages and 16-bit positions:

| Asset (triangles) | Pages | File size |
|---|---|---|
| DragonAttenuation (135k) | 55 | 7.0 MB (all 12 levels) |
| DamagedHelmet (15k) | 6 | 0.7 MB |

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
disc radius and normal σ at 5% of that, SH degree 0 color from a host channel, and logit
opacity. The progressive order carries over: the first k splats, grown by `SurfelCloud::lod`'s
radius scale, are a coarser even covering. These are placed, not optimized. There is no
photometric training, so the color is whatever the host evaluates at each surfel (albedo, or
baked lighting), and the result is view-independent.

