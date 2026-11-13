# farthing research: computer architecture, HPC and algorithm engineering (W1 focus)

Scope: ideas from database engines, deterministic parallel algorithms, bucket queues, sparse direct
solvers, bioinformatics SIMD, particle-physics reconstruction and image morphology. The goal is a
much faster reducer (batched rounds) and UV pipeline on 2-8 cores, with output that stays
deterministic.

## 0. Grounding: where the time goes today (measured here, not guessed)

Before searching I read the hot code and profiled the current release binary with callgrind
(instruction counts; `--branch-sim=yes` for the reducer run). `perf` does not work on this kernel.

Caveats:
- These are instruction counts, not wall time. Cache misses are not shown.
- The CLI also decodes textures (PNG/JPEG, `powf`). Below I subtract that.
- `target/release/farthing` is newer than the UV sources. It is *older* than the `simplify.rs`
  edit at 16:21, so the reducer numbers describe the version just before that edit.

| run | function (self instructions) | share |
|---|---|---|
| `uv Avocado.glb` (farthing 89 ms vs xatlas 14 ms in BENCHMARKS.md) | `ChartMask::from_triangles` 0.92 G | about 63% of UV-internal work (27% of the whole process) |
| | `pack_charts` + `resting_height` 0.31 G | about 21% |
| | `EnvelopeCholesky::factor` 0.02 G | about 1% |
| `uv DamagedHelmet.glb` | `ipnsort/quicksort<(u32,u32)>` 1.5 G, of which 1.51 G is called from `EnvelopeCholesky::factor` | about 23% of UV-internal |
| | `EnvelopeCholesky::factor` (numeric) 0.82 G | about 12% |
| | `from_triangles` 0.72 G | about 11% |
| | SipHash (`DefaultHasher::write` + `hash_one`) 0.40 G | about 6% |
| | `factor` called **2,585 times** for 245 final charts (the merge trials) | — |
| `reduce DamagedHelmet --ratio 0.1` (reducer about 0.3 G instructions) | `cost_avx2` 88 M, `map_from_corners` 79 M, `valid` 46 M, `State::new` 37 M, `best_candidate` 26 M, `collapse` 14 M | — |
| | mispredicts: `map_from_corners` 1.33 M, `valid` 0.92 M, `best_candidate` 0.57 M | roughly 40% / 28% / 17% of reducer mispredicts |

BENCHMARKS.md also gives the batched single-pass loop 1.19x on 2 threads. By Amdahl that is
only about 32% parallel. The serial part is the in-round commit loop of `run_batched`, which
evaluates and collapses candidates one at a time, plus the per-round list build.

The ideas below are ranked by expected payoff divided by effort, using this profile.

---

## 1. UV: never build the chart bitmask; compute the four packing envelopes from edges (late materialization + van Herk/Gil-Werman)

- **Source.**
  - Database "late materialization / aggregate push-down": compute only the aggregate the
    consumer reads, never the full intermediate.
  - Linear-time running min/max filters: M. van Herk, *Pattern Recognition Letters* 1992
    ([dblp](https://dblp.org/rec/journals/prl/Herk92)); Gil & Werman, IEEE PAMI 1993
    ([record](https://cris.huji.ac.il/en/publications/computing-2-d-min-median-and-max-filters/));
    streaming variant by Lemire 2006 ([arXiv cs/0610046](https://arxiv.org/pdf/cs/0610046v3)).
- **Mechanism.** A running min/max over a window costs O(1) per sample whatever the window
  width. A square dilation is separable.
- **Transfer (W1, UV).**
  - `pack_charts` reads only `rotated_profiles()`: per column (lowest, highest) and per row
    (leftmost, rightmost). It never uses the bitmask beyond that.
  - Yet every `try_pack` call does three area-sized passes per chart: it rasterizes a full
    `Vec<bool>` with a per-pixel triangle/box SAT test, dilates it in 2D, and scans it. The
    density search calls `try_pack` up to 10 times.
  - Replacement: the lowest covered row of a column strip [x, x+1) is set by the minimum over
    triangle edges clipped to that strip. Walk the edges, about O(perimeter in pixels), and
    update column min/max and row min/max directly.
  - Padding: a square dilation of radius r turns each column envelope into a running min/max
    over ±r columns, shifted by r. That is exactly a van Herk/Gil-Werman filter, O(width).
  - The `covered` sort key can be the chart's UV area times density² (or the envelope area).
- **Why it should win.** Per-chart work drops from O(area · 10 iterations) to
  O(perimeter + width). On Avocado this is the 63% hotspot, and Avocado is the worst asset
  (6.4x slower than xatlas).
- **Determinism.** Integer envelopes. You can reproduce the current conservative SAT semantics
  exactly (pixel touched by the closed triangle), so output stays bit-identical. Test this
  against the current `from_triangles` on all assets.
- **Cost / risk.** Low, about 150 lines. The risk is in the boundary conventions only.
- **Already in graphics?** Conservative rasterization is standard, and envelope-from-edges is
  elementary geometry. The finding is that the bitmask is never needed. Not novel research,
  but probably the largest single UV speed win available.

## 2. UV: linear-time sparse-structure build and compressed (2×2 block) ordering for LSCM/ARAP (radix partitioning, supervariables)

- **Source.**
  - Database radix partitioning, and sort-vs-hash in main memory: Balkesen et al., "Multi-core,
    main-memory joins: sort vs. hash revisited", PVLDB 2013
    ([pdf](https://cs.uwaterloo.ca/~tozsu/publications/other/p168-balkesen.pdf)).
  - Sparse direct solvers: C. Ashcraft, "Compressed graphs and the minimum degree algorithm",
    SIAM J. Sci. Comput. 1995 ([doi](https://epubs.siam.org/doi/10.1137/0916081)).
- **Mechanism.**
  - When keys are small dense integers, a histogram + prefix-sum scatter (counting sort) builds
    a CSR structure in O(nnz). It beats comparison sorting, and it beats hashing outright.
  - Variables that share an adjacency pattern (the u and v of one vertex) are ordered as one
    supervariable.
- **Transfer (W1, UV).**
  - `EnvelopeCholesky::factor` builds the reverse Cuthill-McKee (RCM) adjacency by sorting every
    (i, j) entry pair. For LSCM that is about 21 pairs per A-row, with duplicates. This sort costs
    *more instructions than the factorization itself* (1.51 G vs 0.82 G on DamagedHelmet).
  - Replacement: build the vertex graph once from the chart triangles (each triangle adds 3
    edges). Run RCM on the n/2 vertex nodes. Expand the order to interleaved (u, v) 2×2 blocks.
  - Build CSR by counting sort on the row index. Sort each row's few neighbours with an insertion
    sort, or keep them in triangle order, which is deterministic.
  - Accumulate numeric entries straight into the envelope rows. The `entries` vector then
    disappears.
  - Also replace `std::HashMap` (SipHash) in `param.rs` with flat arrays indexed by face × 3
    (`Chart::uv: HashMap<(usize,usize),Vec2>` → `Vec<Vec2>`), or with sort-based grouping.
    This is the 6% hashing cost, and it also speeds up `fill()` and the final remap.
- **Why it should win.** It removes about 23% + 6% of UV instructions on DamagedHelmet. Ordering
  on the compressed graph halves the ordering work. 2×2 blocks also make the envelope rows
  friendlier to SIMD.
- **Determinism.** Same ordering rule, so identical results are achievable. Note that RCM on
  the compressed graph can give a *different* (equally valid) ordering. Solutions agree only to
  rounding, and stay deterministic.
- **Cost / risk.** Low. **Already in graphics?** Counting-sort CSR is everyday HPC practice.
  libigl/Eigen users mostly use generic triplet assembly, so it is common but not universal.

## 3. Reducer: deterministic reservations to parallelize the in-round commit loop (Blelloch, Fineman, Gibbons, Shun)

- **Source.**
  - Blelloch, Fineman, Gibbons, Shun, "Internally deterministic parallel algorithms can be fast",
    PPoPP 2012 ([pdf](https://jshun.csail.mit.edu/determ.pdf)).
  - Blelloch, Fineman, Shun, "Greedy sequential maximal independent set and matching are
    parallel on average", SPAA 2012 ([arXiv 1202.3205](https://arxiv.org/pdf/1202.3205)).
- **Mechanism.**
  - A loop over priority-ordered iterates runs in rounds over a prefix.
  - *Reserve*: each iterate priority-writes its rank into every shared cell it may touch.
  - *Commit*: iterates that still own all their cells execute in parallel; the rest retry next
    round.
  - With a deterministically chosen prefix size the run is internally deterministic. The basic
    MIS, spanning-forest and minimum-spanning-forest versions reproduce the sequential greedy
    result.
  - Reported self-speedups were 14-21x on 32 hyper-threaded cores, with 1.4-2x single-thread
    overhead (Table 2 of the paper).
- **Transfer (W1).**
  - Today `run_batched` executes the round's m candidates serially in rank order, re-evaluating
    any candidate whose ring was touched earlier in the round.
  - Replace that with `speculative_for` over the rank-sorted list:
    1. **Reserve (parallel).** Candidate p→q writes `rank` with `fetch_min` into `res[v]` for
       every v in N[p] ∪ N[q] (closed one-rings, about 10 vertices). Per-vertex `AtomicU32`.
    2. **Commit (parallel).** If the candidate owns all its cells, evaluate and collapse.
       Evaluation is still needed when it is not "fresh".
    3. Losers stay in rank order for the next sub-round.
  - I checked the read/write sets in the code:
    - `validity`, `map_wedges` and `cost` read only N[p] ∪ N[q] and those triangles.
    - `collapse` writes the adjacency of p, q and the third vertices of the dying triangles
      (all in N(p)), `tris/tpos` of triangles around p, `quadric[q]`, the wedges of p and q,
      and `bnbr` of border neighbours of p.
    - The `touched`/`stamp` marks lie inside the set.
  - So winners are independent, and every candidate is decided only after all
    higher-ranked overlapping candidates. That reproduces the current serial in-round order
    except at the triangle-target cutoff.
  - At the cutoff, either truncate committed winners by rank and run the last sub-rounds with a
    small prefix, or accept a different but still deterministic stop.
- **Engineering prerequisites.**
  - `Adjacency::push` relocates a list to the end of the shared arena when it overflows. Make
    that parallel-safe: reserve capacity len(p) + len(q) for q in the reserve phase, or do the
    relocations in a serial prefix-sum pass after commit.
  - Alternatively, adopt meshoptimizer's design of rebuilding a flat adjacency every pass. Its
    `simplifier.cpp` calls `updateEdgeAdjacency` per pass and skips per-collapse list
    maintenance.
  - `alive_tris` and `worst_geom2` become integer and max reductions, which are exact in any
    order.
- **Why it might beat graphics practice.**
  - GPU simplifiers already use the *reserve* half. PaMO (Pacific Graphics 2025,
    [arXiv 2509.05595](https://arxiv.org/pdf/2509.05595), following Gautron & Kubisch 2023 as
    cited there) collapses only edges whose cost is the minimum over their one-ring, using a
    64-bit (cost, id) `atomicMin`.
  - Franc & Skala (ALGORITMY 2000,
    [pdf](https://emis.univie.ac.at/journals/AMUC/_contributed/algo2000/francskala.pdf)) used
    greedy "super independent" sets. They reported 1.83x on 8 CPUs.
  - What these lack is the retry-in-rank-order discipline. Local-minimum rules discard non-minima
    each round (more rounds, different result). Deterministic reservations keep the exact
    sequential-greedy semantics of the current batched loop, so the quality the benchmarks
    validated is preserved.
  - The SPAA 2012 result says the dependence chains of random-order greedy MIS are short
    (polylog). That suggests few sub-rounds per round. This is an *assumption* for cost-ordered
    collapses and must be measured.
- **Expected payoff (speculative).** It attacks the roughly 68% serial share of single-pass
  batched runs, which covers all meshes under 120k triangles. On 2 cores perhaps 1.19x → 1.5x.
  More on 4-8 cores.
- **Cost / risk.** Medium, about 300 lines plus the arena change. On tiny meshes, use a serial
  fallback below a size threshold; the result is identical either way.

## 4. Reducer: replace the neighbour × corner nested loop with one partitioning pass and branch-free compaction (Ross; Polychroniou-Raghavan-Ross; Inoue)

- **Source.**
  - K. A. Ross, "Conjunctive selection conditions in main memory", PODS 2002
    ([pdf](https://courses.cs.washington.edu/courses/cse591d/02sp/109-Ross.pdf)): no-branch plans
    win at unpredictable selectivities.
  - Polychroniou, Raghavan, Ross, "Rethinking SIMD vectorization for in-memory databases",
    SIGMOD 2015 ([pdf](https://15721.courses.cs.cmu.edu/spring2016/papers/p1493-polychroniou.pdf)):
    selective stores and branch-free compaction.
  - Inoue, Ohara, Taura, PVLDB 2014: set intersection with fewer mispredicts. About 2x from
    branch reduction alone and about 2x more with SIMD
    ([dblp](https://dblp.org/rec/journals/pvldb/InoueOT14),
    [abstract](https://repository.dl.itc.u-tokyo.ac.jp/record/48871/files/A32992_abstract.pdf)).
  - Han, Zou, Yu, SIGMOD 2018 ([code](https://github.com/pkumod/GraphSetIntersection)):
    SIMD set intersection for graph algorithms.
- **Mechanism.** Turn data-dependent branches into masks and conditional moves. Write every
  element unconditionally and advance the output index by the predicate. Do small-set membership
  as an all-pairs SIMD compare instead of an early-exit scan.
- **Transfer (W1).**
  - The main mispredict source today: `best_candidate` calls `map_from_corners` once per
    neighbour q. Each call scans *all* corners of p with `if c.pa == q ... else if c.pb == q ...
    else continue`, which is taken about 2 times in 6, so it mispredicts. It then runs
    `out.iter().find` over a tiny variable-length list.
  - That is a nested-loop join (neighbours × corners). Replace it with one pass over p's corners
    that scatters each corner to its two neighbours (pa, pb). Neighbour slots come from a small
    position→slot table built in the same `neighbors_into` pass. You get all neighbours' wedge
    maps in O(corners) with no data-dependent branches.
  - In `validity`, the link-condition count uses epoch marker writes and reads into an array of
    size np. Replace it with a SIMD all-pairs compare of N(p) (≤ 8 lanes) against N(q), which
    needs no marker array. Fall back to the scalar path above valence 8.
  - Make the flip loop branch-free: compute all n0·n1 tests, OR the masks, and exit once. Ross's
    rule says this pays exactly when the outcome is unpredictable. Flips are rare, so first
    measure that path's mispredicts separately.
- **Why it should win.** Callgrind attributes about 85% of reducer mispredicts to these three
  functions. BENCHMARKS.md already identifies mispredicts on short neighbour loops as the
  bottleneck.
- **Determinism.** Keep corner order and first-match semantics, and the outputs are
  bit-identical. Verify with the existing `simd_and_baseline_builds_agree` and determinism tests.
- **Cost / risk.** Low to medium. Safe Rust with `std::arch` or a portable-SIMD crate.
  **Already in graphics?** Branch-free predicates are folklore, but I found no simplifier that
  restructures wedge mapping as a partitioned single pass.

## 5. Reducer: inter-candidate SIMD ("Matriplex" / SWIPE layout) plus SELL-C-σ valence grouping

- **Source.**
  - CMS mkFit: Lantz et al., "Speeding up particle track reconstruction using a parallel Kalman
    filter algorithm", JINST 15 P09030 (2020) ([arXiv 2006.00071](https://arxiv.org/abs/2006.00071)).
    Its *Matriplex* stores n small N×N matrices "matrix-major", so one vector instruction
    processes the same element of n matrices. Track building was 2.7-2.9x faster at width 16.
  - Genomics: T. Rognes, "Faster Smith-Waterman database searches with inter-sequence SIMD
    parallelisation", BMC Bioinformatics 2011 ([doi](https://doi.org/10.1186%2F1471-2105-12-221)).
    It processes 16 database sequences per vector instead of vectorizing within one alignment,
    and is about 2.5x faster than Farrar's intra-sequence striped method on one thread.
  - Sparse HPC: Kreutzer, Hager, Wellein, Fehske, Bishop, SELL-C-σ (SIAM J. Sci. Comput. 2014,
    [arXiv 1307.6209](https://arxiv.org/pdf/1307.6209)). It sorts rows by length inside windows
    of σ so that chunks of C rows have equal trip counts.
- **Transfer (W1).**
  - Today's SIMD is *intra*-candidate: a 12-wide dot product per quadric. It cut instructions
    about 8% but wall time only 3-4%.
  - Switch to *inter*-candidate lanes. In `best_candidate(p)`, every neighbour q is evaluated
    against the same `Q_p`. Broadcast Q_p's 10 coefficients, put 4 (AVX2 f64) or 8 (AVX-512)
    q-positions' monomials in SoA lanes, and get 4-8 costs from about 10 FMAs, with no
    horizontal sums.
  - Attribute terms: lanes are (wedge, q) pairs taken from the map that idea 4 builds.
  - In the re-rank phase, order `full_list` inside fixed windows by valence (SELL-C-σ). Lanes in
    a chunk then share trip counts and the loop exit becomes predictable. Windows must be fixed
    (not thread-dependent) to stay deterministic.
- **Determinism.** Lane arithmetic matches scalar arithmetic if each lane runs the same
  operation sequence. With no horizontal reduction, the AVX2 and baseline builds stay
  bit-identical. That is actually easier than the current fixed-association horizontal sum.
- **Payoff / cost.** Moderate: up to about 1.3x on the cost/attribute portion, my estimate.
  Medium effort. Do it after idea 4, because mispredicts dominate.
- **Already in graphics?** GPU simplifiers get this implicitly through SIMT. I found no CPU
  QEM simplifier using an explicit inter-candidate SoA layout (meshoptimizer is scalar per
  candidate).

## 6. Reducer: group prefetching / AMAC for attribute-quadric misses, with spatial renumbering

- **Source.**
  - Chen, Ailamaki, Gibbons, Mowry, "Improving hash join performance through prefetching", ICDE
    2004 ([record](https://infoscience.epfl.ch/record/133207/)): group and software-pipelined
    prefetching.
  - Kocberber, Falsafi, Grot, "Asynchronous memory access chaining", PVLDB 9(4) 2015
    ([pdf](https://core.ac.uk/download/43718028.pdf)). It interleaves about 8-10 independent
    lookups through a state ring. Hash join was 4.3x faster, and gains held under skew where
    group prefetching lost most of its advantage.
  - Ainsworth & Jones, "Software prefetching for indirect memory accesses", CGO 2017
    ([pdf](https://www.cl.cam.ac.uk/~sa614/papers/Software-Prefetching-CGO2017.pdf)). For
    A[B[i]] patterns the geomean was 1.3x on Haswell and 2.1-2.7x on in-order cores.
- **Transfer (W1).**
  - The per-wedge attribute block `aq` is 28 + 4·dim f64 values. That is about 250 B, or 4 cache
    lines, for normals+UV, reached via adj → tris → wedge. This is the "attribute-quadric cache
    miss" named in BENCHMARKS.md.
  - In the parallel re-rank (`full_list`, `inc`) and in the round's evaluation phase, process
    candidates in groups of G = 8 with three stages:
    1. prefetch `adj.start/len` for G positions;
    2. prefetch their triangle and `tpos` rows;
    3. prefetch the `aq` lines of the wedges involved, then compute.
  - Use `core::arch::x86_64::_mm_prefetch`; prefetches are hints, so this is deterministic.
  - Complement: renumber positions, triangles and wedges in Morton or Hilbert order once in
    `State::new`, so ring neighbours share lines. Weigh this against Balaji & Lucia (IISWC 2018,
    [pdf](https://bvignesh.github.io/papers/Reordering-IISWC18.pdf)): reordering pays only when
    the reuse it creates outweighs its cost, and it can slow some inputs down.
  - Optionally store `aq` as f32 to halve the lines (meshoptimizer works in f32). This is a
    numerical risk; test it against the quality benchmarks.
- **Caveats.** Small meshes (a few thousand positions) fit mostly in L2, so expect little there.
  Gains appear on the 50k-1.5M triangle meshes. Renumbering changes tie-break order, so output
  changes but stays deterministic.
- **Already in graphics?** Cache-oblivious mesh layouts are graphics (Yoon, Lindstrom, Pascucci,
  Manocha, SIGGRAPH 2005, [record](https://history.siggraph.org/?p=113923)). Software
  prefetching in simplifiers is not common, as far as I found.
- **Cost.** Low.

## 7. UV: reuse child factorizations in chart-merge trials (domain decomposition / Schur complement)

- **Source.** Classic HPC substructuring and Schwarz domain decomposition (Smith, Bjørstad &
  Gropp, *Domain Decomposition*, 1996). Also mixed-precision refinement: Carson & Higham, SIAM J.
  Sci. Comput. 2018 ([pdf](https://eprints.maths.manchester.ac.uk/2629/1/cahi18.pdf)).
- **Mechanism.** A system on the union of two subdomains is solved by reusing each subdomain's
  factor, either as a preconditioner (additive Schwarz) or exactly through the interface Schur
  complement. Only the interface work is new.
- **Transfer (W1, and lets W4 merging go further).**
  - `merge_charts` re-runs a fresh `lscm_chart` on every candidate union: assemble, order,
    factor, ARAP ×10 iterations, overlap test. That is 2,585 factorizations for 245 final charts
    on DamagedHelmet.
  - For the *trial*, use PCG on the merged LSCM system. Warm-start from the two children's UVs,
    rigidly aligned across the seam by 2D Procrustes. Precondition with the children's cached
    envelope factors (block Jacobi / additive Schwarz) plus a diagonal on seam and former-pin
    unknowns.
  - Accept or reject on the stretch bound. Factor exactly only for accepted merges.
- **Determinism.** Fixed iteration count or fixed tolerance, with sequential dot products inside
  each chart. The merge pass already uses a deterministic matching, so it stays deterministic.
  Accept/reject decisions may differ marginally from the exact solve.
- **Risk.** Medium-high. LSCM pins differ between children and union. ARAP is part of the
  current acceptance test. Payoff is unmeasured: profile first how much of the `factor` count
  is trials that fail.
- **Already in graphics?** Hierarchical/multigrid parameterization exists. I found no use of
  child-factor reuse for merge *testing*.
- **Mixed precision.** Factor in f32 and refine in f64 (Carson-Higham) only if `factor` stays
  significant after idea 2.

## 8. Packer: exact best position by threshold-style bounds instead of coarse stride

- **Source.** Fagin, Lotem, Naor, "Optimal aggregation algorithms for middleware" (the
  threshold algorithm for top-k aggregation), PODS 2001 / JCSS 2003. Also Moerkotte, "Small
  materialized aggregates" (zone maps), VLDB 1998
  ([pdf](https://pi3.informatik.uni-mannheim.de/~moer/Publications/MA-98-01.pdf)). Both prune
  with cheap bounds and stop once no remaining item can beat the best.
- **Transfer (W1 + W4).**
  - The resting height y0(x) = max_c(sky[x+c] − lo[c]) is a max. Evaluating it on *any subset*
    of columns (for example 8 representative columns, or block maxima of `sky`) gives a valid
    lower bound.
  - The score (waste + raise·width) increases with y0, so a bound on y0 bounds the score.
  - Compute the bounds for all x cheaply. Visit x in ascending bound (tie-break by x). Evaluate
    exactly and stop when bound ≥ best.
  - This finds the *exact* optimum of the current objective, where the coarse stride + 4-refine
    heuristic can miss it. That may recover part of the 4-9% gap on few-large-chart assets (W4),
    possibly at equal or lower cost.
- **Determinism.** Exact pruning with a fixed tie-break, so deterministic.
- **Risk.** Low to medium. Speed depends on how tight the bound is, which is unmeasured.
- **Already in graphics?** Not in skyline packers I know of. xatlas brute-forces a bitmap
  (my recollection; not verified here).

## 9. Bucketed monotone priority for rounds (Julienne). Graphics has partly absorbed this.

- **Source.** Dhulipala, Blelloch, Shun, "Julienne: a framework for parallel graph algorithms
  using work-efficient bucketing", SPAA 2017 ([pdf](https://jshun.csail.mit.edu/bucketing.pdf)).
  Only a window of buckets (default 128) is materialized; further entries go to an overflow
  bucket. Parallel updates use per-block (2048) histograms + scan, which gives a stable,
  thread-count-independent placement.
- **Transfer (W1).**
  - Each round now scans all np candidates, then runs `select_nth_unstable` and a sort.
  - Instead, key candidates by the top bits of an order-preserving f64→u64 map in a bucket
    window. A round takes the lowest buckets until it has m. Exact (cost, id) ordering is needed
    only inside the boundary bucket. Updates from re-ranking move entries between buckets in
    O(1).
- **Honest status.** meshoptimizer already sorts collapses by a 12-bit float key with a counting
  sort (`sortEdgeCollapses`, which keeps 8 exponent bits + 4 mantissa bits). In my profile, list
  build + select + sort are about 3% of reducer instructions on a small mesh. Low payoff on small
  meshes, somewhat more on big ones. It pairs naturally with idea 3, where the prefix becomes the
  lowest bucket. Low effort.

---

## Considered and rejected (or deprioritized)

- **vqsort / Highway-style SIMD sorting** (Blacher, Giesen, Sanders, Wassenberg, 2022,
  [arXiv 2205.05982](https://arxiv.org/pdf/2205.05982)). Sorting is not a reducer hotspot. The
  large UV sort cost goes away by *not sorting* (idea 2), not by sorting faster. Rust's std
  unstable sort is already ipnsort (it appears in the profile).
- **Approximate Cholesky for Laplacians** (Kyng & Sachdeva 2016; Gao, Kyng, Spielman 2023,
  [arXiv 2303.00709](https://arxiv.org/abs/2303.00709)). The guarantee covers SDDM matrices.
  LSCM normal equations are not SDDM, and cotangent Laplacians stop being SDDM with obtuse
  triangles. Charts here are small, so the envelope factor is cheap once idea 2 is done.
  Relevant only for very large single charts, as a preconditioner built from a clamped
  Laplacian.
- **Supernodal / nested-dissection Cholesky (faer).** Pure-Rust faer has supernodal and
  simplicial sparse Cholesky with AMD/COLAMD
  ([source](https://docs.rs/crate/faer/0.24.0/source/src/sparse/linalg/mod.rs)). On a 2D grid,
  the envelope method needs about O(n^1.5) storage and O(n²) flops, while nested dissection
  needs O(n log n) fill and O(n^1.5) flops (standard results). The crossover is only at charts
  of several thousand vertices. Keep the envelope for small charts, and switch to faer above a
  measured size. Low priority for the current assets.
- **Succinct / compressed structures.** rank/select add instructions. The meshes are small
  enough that memory footprint is not the limit.
- **Reproducible summation** (Demmel & Nguyen, ReproBLAS,
  [tech report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-229.html)). Not
  needed now: the code avoids parallel float reductions. Keep that rule as a guard when adding
  parallelism (idea 3): use only integer/max reductions, or fixed-shape trees. I believe rayon's
  `sum()` split tree depends on scheduling, so do not use it for floats in output-affecting
  paths.

## Suggested order of work

1. Ideas 1 + 2 (UV, low effort, measured hotspots). Expect most of the gap to xatlas to close,
   probably all of it on Avocado.
2. Idea 4 (reducer mispredicts, bit-identical).
3. Idea 3 (parallel commit). This is the main multicore lever for meshes under 120k triangles.
4. Ideas 5 and 6, then 8, 7, 9, guided by re-profiling.
