# Dead ends

Things that were tried or examined and did not pay off, so nobody spends the time twice. Each
entry says what was tried, what happened, and what would have to change for it to be worth
another look. Add to it whenever an experiment is dropped. Experiments that worked are in
[../BENCHMARKS.md](../BENCHMARKS.md); ideas not yet tried are in [README.md](README.md).

## Realtime reduction

| Tried | Result | Revisit if |
|---|---|---|
| Collapse tail worked out ahead in parallel (the next 64 heap entries per round, results kept while their neighborhoods are clean) | Slower than the serial tail. Each round wakes the pool, which costs about 70 µs on Windows, and a round's work is smaller than that | Workers that spin through the tail instead of sleeping, or rounds of thousands of independent collapses |
| One scatter on the top 10 bits of the Morton code, then per-bucket sorts, in place of three 10-bit scatter passes | Same order, no measurable change in time. Kept because it is simpler | — |
| Keeping freed memory mapped between calls (allocator purge off) to avoid page faults on 15–22 MB per call | No change in time: the faults are not the cost | The per-call allocation grows several times |
| Evaluating a cluster's representative by quadric error alone | Flat regions tie, and the tie picked vertices at cell corners, folding a flat grid's outline. Fixed with a small pull to the cluster's centroid (`CENTROID_PULL`) | — |
| More or fewer clusters per target triangle than 0.55 before the tail | Fewer leaves the tail no room and quality drops; more makes the tail the whole cost | The tail gets several times faster |
| Bounding the result's Hausdorff distance by the quadric error | The quadric error is a mean squared distance to planes, not a distance to the surface; `max_error` holds only loosely (within about 0.06 of the diagonal in the test) | A cheap one-sided distance check per cluster |
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
