# farthing research brief (shared by all research agents)

farthing is a clean-room, pure-Rust 3D asset optimization library (a Simplygon alternative): LOD
generation by edge-collapse mesh simplification (QEM with attribute quadrics, batched/parallel
collapse order, SIMD cost kernels), UV atlas generation (chart segmentation, LSCM + ARAP
flattening, chart merging, skyline packing), voxel remeshing for proxies (conservative
voxelization, flood fill, surface nets, projection back to the source surface), occluders,
impostors, material baking via a RenderHost boundary, aggregation, visibility culling, bone
reduction. Constraints: pure Rust, only publicly published knowledge (no Simplygon internals),
no materials inside the library, deterministic output across threads/CPUs.

Current weaknesses we want ideas for (cite which one an idea addresses):
W1 Speed: reducer ~5x slower than meshoptimizer on small meshes (2-core box), bound by branch
   mispredicts and cache misses in the collapse loop; UV atlas ~1.3x slower than xatlas.
W2 Aggressive targets: at 1% triangle targets, 2 of 8 assets stop 20-60% above target due to
   non-manifold locks and flip guards.
W3 Remesh worst-case (Hausdorff) error: binary conservative voxelization fills crevices narrower
   than ~1 voxel; competitor (baby_shark) contours a signed distance field with sub-voxel
   accuracy. Thin sheets / open meshes make inside-outside signs unreliable.
W4 UV packing of a few large irregular charts is 4-9% less tight than xatlas (skyline packer
   cannot nest charts into each other's concavities); chart segmentation/seam placement.
W5 Perceptual error: we optimize geometric/attribute distances; what actually matters is how the
   LOD looks at its screen size (silhouette, shading, texture).
W6 Quad-dominant reduction scales poorly (~50k quads); impostor parallax artifacts.
W7 Topology: proxies and aggressive LODs should be allowed to simplify topology (close holes,
   merge nearby parts) in a controlled way.

Your job: look OUTSIDE mainstream computer graphics for exotic or recent discoveries (roughly
2015-2026, older if underexploited) in the sciences assigned to you that could transfer to these
problems. For each idea give: the source discovery (paper/author/year/link), the mechanism in a
sentence or two, the concrete transfer to farthing (which W#, what algorithm change), why it might
beat the current graphics state of the art, implementation cost/risk in pure Rust, and whether
graphics has already absorbed it (search to check - say so honestly if it is already standard).
Prefer depth over breadth: 6-10 well-checked ideas, ranked by expected payoff/effort. No
speculation presented as fact; mark uncertain claims. Use web search to verify citations.

Write your report as markdown to the file path given in your task, then reply with a short summary
(top 3 ideas, one line each).
