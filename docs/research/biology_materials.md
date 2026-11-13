# farthing: ideas from biology, biophysics and materials science

Scope: protein docking and packing, colloids, jamming and glasses, tissue mechanics, thin-sheet
elasticity and fracture, topology optimization, biological image analysis, transport networks.
Main focus: W4 (packing irregular charts, segmentation and seams), W7 (topology), W2.

How I checked: every citation below was looked up on the web this session unless marked
*(not re-verified)*. Before writing, I read farthing's current code so the transfers fit what
exists:
- `crates/farthing/src/uv/pack.rs`: chart masks drop onto a per-column skyline by their lower
  outline, so a chart can never go under an overhang or into a pocket that opens sideways or
  upward.
- `crates/farthing/src/uv/param.rs`: segmentation grows regions by normal cone, then flattens
  with LSCM+ARAP, splitting on failure. `merge_charts` tests every candidate merge with a full
  trial flattening.
- `crates/farthing/src/reduce/simplify.rs`: Link, Flip and Degenerate guards; non-manifold
  vertices (border degree > 2) are locked; stall escalation re-ranks and then gives up.
- `crates/farthing/src/reduce/parts.rs`: deletes whole parts when closed parts reach their
  minimal triangulation.

**Patent warning for W4 (found while checking idea 1).** US 12,112,510 B2, "Convolutional
approach to fast and compact packing of 3D mesh into 2D maps", is assigned to Tencent America.
Its priority date is 2022-03-25, it was granted 2024-10-08, and Google Patents estimates
expiry in 2043. Claim 1 covers placing a mesh chart by computing a correlation matrix between
the atlas occupancy map and the chart's occupancy matrix, then picking a zero-valued entry.
Dependent claims add the FFT.
([Google Patents](https://patents.google.com/patent/US12112510),
[USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12112510))
This is the obvious FFT-docking transfer, so **I do not recommend FFT or whole-grid
correlation placement in farthing without legal review**. Idea 1 is written to avoid it.
This is not legal advice.

---

## Ranking (expected payoff divided by effort)

| # | Idea | W# | Payoff | Effort | Already in graphics? |
|---|------|----|--------|--------|----------------------|
| 1 | Lock-and-key / docking contact score + free-space placement that can nest charts | W4 | High | Low to medium | Partly. xatlas places freely on bitmaps but scores by position, not contact. Touching-perimeter scoring exists for rectangles in operations research. |
| 2 | Jamming-protocol compaction: allow overlap, relax it, shrink the container | W4 | High | Medium (low if sparrow is adopted) | No for UV atlases. Standard in 2D nesting research. |
| 3 | Cut-or-fill global topology simplification (developed for biomedical and plant-root imaging) | W7 (and W2) | High | Medium to high | Published in graphics (SIGGRAPH Asia 2020, SGP 2022). Not in LOD tools as far as I could find. |
| 4 | Curvature-potential (thin-sheet stress) predictor to prune chart merges and guide seams | W4 (and W1) | Medium | Low | Yes in theory (Sharp & Crane 2018, Ben-Chen et al. 2008). The use as a cheap merge filter is new to farthing. |
| 5 | Unjamming the decimator: swap moves (T1) and soft flip guards with local relaxation | W2 | Medium | Medium | Edge swaps are old (Hoppe 1993). Their use as a stall-breaking schedule in QEM LOD is uncommon. |
| 6 | Topology-optimization length-scale control (robust eroded/dilated projection) as cut/fill candidates and an error bound | W7, W3 | Medium | Low | Mostly yes: it is morphology with better bookkeeping. |
| 7 | Physarum / adaptive-network Steiner trees for seam graphs | W4 | Low | Medium | No, but the payoff is doubtful. |

---

## 1. Lock-and-key contact scoring + concavity-capable placement (W4)

**Source discoveries**
- **Protein docking.** Katchalski-Katzir et al., "Molecular surface recognition: determination
  of geometric fit between proteins and their ligands by correlation techniques", PNAS
  89(6):2195-2199, 1992 (doi:10.1073/pnas.89.6.2195,
  [Wikipedia summary](https://en.wikipedia.org/wiki/Katchalski-Katzir_algorithm)).
  - Each grid cell is labelled outside, surface or interior.
  - The score rewards surface-on-surface contact and heavily penalizes interior overlap.
  - The usual parameterization has a surface layer of +1 and a large negative interior
    weight; the exact values are *from memory, not re-verified*.
- **Lock-and-key colloids.** Sacanna, Irvine, Chaikin & Pine, "Lock and key colloids", Nature
  464:575-578, 2010 ([PDF](https://irvinelab.uchicago.edu/papers/nature08906.pdf)).
  - Depletion attraction scales with the overlap of the particles' excluded-volume shells.
  - The paper's words: "surfaces that fit together maximize the overlap of excluded volume
    and their binding is thus favoured."
  - So the physics selects shape-complementary (concave-into-convex) binding because it
    maximizes shell overlap.

**Mechanism.** Score a candidate placement by how much of the chart's one-pixel outer shell
(its "excluded-volume shell") touches occupied pixels or the atlas border. Interior overlap is
forbidden. Greedy placement that maximizes this score pushes charts into each other's pockets,
which is exactly lock-and-key binding.

**Transfer to farthing** (replaces or extends `pack_charts`)
1. **Free-space search that is not a skyline.** Keep an occupancy bitboard: one `u64` per 64
   columns per row. The chart masks already exist (`ChartMask`).
   - Generate candidate offsets from three sources:
     - (a) the current skyline positions, which keep today's behaviour as a baseline;
     - (b) concave pockets: local maxima of an exact Euclidean distance transform of free
       space;
     - (c) a seeded random sample, as xatlas does.
   - Prune with a "pole" test, the same trick sparrow and jagua-rs use (see idea 2). Precompute
     each chart's pole of inaccessibility (centre and inscribed radius r). An offset survives
     only if the free-space distance transform at the pole's landing pixel is at least r. That
     is one lookup.
   - Then run an exact overlap test row by row: `AND` plus early exit. This per-candidate test
     is the classic bitmap approach xatlas has used since before 2022. Avoid computing a full
     overlap or correlation matrix over all offsets (see the patent warning above).
2. **Score = contact count**, computed as `popcount(shell_row & occupied_row)` summed over
   rows, plus contact with the atlas border.
   - Break ties by bottom-left.
   - Normalize by shell length so large and small charts compete fairly.
   - This is the docking "surface complementarity" term. It is cheap because each candidate
     needs only the shell rows, not the whole interior.
3. Try the 4 rotations already supported, plus mirror if the pipeline allows flipped charts.
   Docking samples rotations in the same way.
4. **Determinism.** Everything is integer: bitboards, popcounts and a seeded RNG for candidate
   sampling. Charts are processed sequentially, or candidates are scored in parallel and then
   reduced by (score, y, x, rot), so the result does not depend on the thread count.

**Why it might beat the current state of the art**
- The 4-9% gap to xatlas is mostly structural. The skyline cannot place under overhangs,
  whereas xatlas tests arbitrary bitmap positions.
- xatlas then picks positions by a simple criterion (to my knowledge, extent of the bounding
  region, *not re-verified in its source*).
- A contact-maximizing score actively prefers nesting. This is the docking and colloid
  insight, and it should matter most in exactly the W4 case of a few large irregular charts.
- Prior art I found:
  - The 2026 GPU packer TABI (Gu, Vining & Sheffer, [arXiv 2602.07782](https://arxiv.org/abs/2602.07782v1))
    explicitly does **not** pack into interior voids and lists this as future work.
  - Its benchmark treats xatlas (random and brute-force modes) as the offline quality
    reference.
  - Blender 3.6 added an "Exact Shape (Concave)" packing method
    ([release notes](https://developer.blender.org/docs/release_notes/3.6/modeling/)), but
    its scoring is not documented there.

**Cost and risk**
- About 300-500 lines of Rust, with no dependencies.
- Risk 1: candidate explosion on large atlases. Mitigate with a coarse-to-fine occupancy
  pyramid (test conservatively dilated masks at 1/4 resolution first) and with pole pruning.
- Risk 2: greedy contact scoring can leave long thin voids. Idea 2 is the fix.

**Absorbed by graphics?**
- Partly. "Touching perimeter" is a known placement rule for rectangles in operations research
  (Lodi, Martello & Vigo 1999, *not re-verified*).
- FFT docking was absorbed for **3D** packing by Cui, Rong, Chen & Matusik, "Dense,
  Interlocking-Free and Scalable Spectral Packing of Generic 3D Objects", SIGGRAPH 2023
  ([PDF](https://cdfg.mit.edu/pdf/journal/2023-cui-dense-interlocking-free-and-scalable-spectral-packing-of.pdf)).
  - That paper cites Katchalski-Katzir as its inspiration.
  - It uses a proximity term based on a distance field.
  - It reports a collision correlation in 0.003 s versus 9.4 s brute force on a
    240x123x100 grid.
  - It addresses no 2D or UV case.
- For 2D UV atlases the correlation route is covered by the Tencent patent. The
  contact-scoring part is not in that patent's dependent claims: it selects by smallest index,
  lp-norm or a preferred location.

---

## 2. Jamming-protocol compaction: soft overlaps, relax, compress (W4)

**Source discoveries**
- **Soft-sphere jamming protocol.** O'Hern, Silbert, Liu & Nagel, "Jamming at zero
  temperature and zero applied stress: the epitome of disorder", PRE 68:011306, 2003
  ([arXiv cond-mat/0304421](https://arxiv.org/abs/cond-mat/0304421)).
  - Particles are allowed to overlap with a soft penalty.
  - Energy is minimized at fixed density, and density is raised step by step until overlaps
    can no longer be relaxed away.
- **Adaptive Shrinking Cell.** Torquato & Jiao, "Dense packings of the Platonic and
  Archimedean solids", Nature 460:876, 2009 ([arXiv 0908.4107](https://arxiv.org/abs/0908.4107v1)).
  - Monte Carlo particle moves are combined with deformation and shrinking of the container,
    treated as an optimization variable.
  - It found the densest known packings of hard polyhedra at the time.
- **The operations-research embodiment, in Rust.** Gardeyn, Vanden Berghe & Wauters, "An
  open-source heuristic to reboot 2D nesting research" (sparrow),
  [arXiv 2509.13329](https://arxiv.org/abs/2509.13329v1), 2025.
  - It decomposes strip packing into a sequence of feasibility problems at fixed strip
    length, then resolves collisions that are temporarily allowed.
  - Overlap is approximated by penetration depth between precomputed pole-of-inaccessibility
    circles, with guided-local-search pair weights.
  - It claims to "consistently outperform the state of the art."
  - Its collision engine is jagua-rs ([arXiv 2508.08341](https://arxiv.org/html/2508.08341v4)).
  - I checked the repos: sparrow is Rust, MIT license, `edition = "2024"`, and depends on
    `jagua-rs = "=2.0.0"`. jagua-rs is **MPL-2.0**.

**Mechanism.** Constructive greedy packing (skyline, xatlas, idea 1) freezes each decision.
Jamming-style compaction instead starts from a feasible packing, shrinks the container (here,
raises texel scale) until overlaps appear, then relaxes the overlaps by moving and rotating
pieces under a penalty that grows on persistent overlaps. It accepts the shrink when
overlap-free and backs off otherwise. The physics shows that such protocols reach much denser
disordered states than sequential deposition, the "random sequential adsorption" analogue of
greedy packing.

**Transfer to farthing.** Add an optional post-pass after `pack_charts`:
1. Fix the atlas size and binary-search the uniform chart scale s. Each probe is a feasibility
   problem: can the charts at scale s be made overlap-free? sparrow's decomposition maps
   one-to-one if "strip length" is replaced by "scale".
2. Overlap proxy:
   - Either pixel-exact: popcount of the AND of the two masks, integer and deterministic.
   - Or sparrow's pole-circle penetration, which is continuous and gives gradients.
3. Moves:
   - Translate a colliding chart to the best of N sampled positions (random plus local), using
     the idea-1 search.
   - Allow the 4 discrete rotations.
   - Increase the pair weights of persistently colliding pairs (guided local search).
4. Stop at a time or iteration budget. Few large charts (the W4 case) means a tiny instance
   (tens of items), which is where these methods excel.

**Why it might beat the current state of the art.** xatlas is constructive and greedy. In 2D
nesting benchmarks, overlap-minimization methods are well established as better than
constructive ones; sparrow is the latest of them. TABI's comparison table shows nobody in UV
packing currently does this.

**Cost and risk**
- Two options:
  - (a) Depend on sparrow/jagua-rs: fast to try, but needs polygon outlines of the charts
    (farthing has hulls and triangle sets), and MPL-2.0 brings file-level copyleft for
    jagua-rs modifications.
  - (b) Clean-room roughly 600-1000 lines on the existing pixel masks: deterministic, slower
    per move.
- Determinism of sparrow itself is **unverified**: it uses `rand` and `rayon`. Seeding and
  ordered reduction would need checking.
- Runtime is the main risk. Expose it as a quality level, the way xatlas offers brute force.

**Absorbed by graphics?** Not for UV atlases, as far as I found. This is standard in
operations-research nesting, with physics-style compaction as the underlying idea.

---

## 3. Cut-or-fill global topology simplification (W7, also helps W2)

**Source discoveries**
- Zeng, Chambers, Letscher & Ju, "To cut or to fill: a global optimization approach to
  topological simplification", ACM TOG (SIGGRAPH Asia) 2020
  ([project page](https://danzeng8.github.io/topo-simplifier),
  [paper](https://par.nsf.gov/servlets/purl/10195909)).
- Follow-up: "Topological Simplification of Nested Shapes", CGF (SGP) 2022
  ([EG diglib](https://diglib.eg.org/handle/10.1111/cgf14611)).
- The work comes from biological imaging. Its test data are brain vessels (angiography), a
  heart (CT), a cryo-EM spike protein, and X-ray CT of corn roots and sorghum panicles.

**Mechanism**
- **Candidates.** Each candidate removes one topological feature (component, handle, cavity)
  either by cutting solid or by filling void.
  - Fill voxels come from deflating an outer seed; cut voxels come from inflating an inner
    seed, both ordered by distance to the boundary.
  - Alternatively: cuts = shape minus its morphological opening, fills = closing minus shape.
- **Energy.** E = 2λ(β0+β2) − λχ + Σ geometric cost. Here β0 is components, β2 is cavities
  and χ is the Euler characteristic, so the β1 (handle) term enters through χ, and λ exceeds
  the total cost of all candidates.
- **Solver.** The problem is reduced to a node-weighted Steiner tree, solved by branch and
  bound with an 8 s cap.
- **Results.** About 3-45 s per model, for example a filigree going from 48 handles to 0.
  Greedy labelling either leaves handles behind or makes large fills.

**Transfer to farthing** (in the `farthing::voxel` proxy path, after flood fill and before
surface nets)
1. **Candidates** come from the existing binary voxel solid plus an exact distance transform:
   - fills = closing(r_fill) minus solid;
   - cuts = solid minus opening(r_cut).
   - Set r from the LOD's screen-space size, so "features under k pixels may change topology".
   - Group candidate voxels into connected clusters; these are the graph nodes.
2. **Topology bookkeeping without homology software.** β0 comes from union-find on solid
   voxels. β2+1 equals β0 of the complement, by union-find on empty voxels. χ is a local sum
   over voxel cells (vertices − edges + faces − cubes), which can be updated incrementally per
   cluster. So the full energy can be evaluated with union-find plus local counts, and β1 never
   has to be computed directly. This is the paper's own formulation.
3. **Solver, in stages.**
   - (a) Greedy, deterministic: process clusters in order of (cost, index); accept when the
     energy drops.
   - (b) If greedy is visibly worse on handles, which the paper shows can happen, add an exact
     or heuristic node-weighted Steiner tree solver. The graphs are small: 5-1,076 edges in the
     paper's examples.
4. **Policy for games.** Do not force genus 0. Give each feature a cost limit tied to the screen
   error budget, so tunnels narrower than the budget get filled, thin handles get cut, and
   cavities always get filled because they are invisible.
   - Nearby parts merge via fills that reduce β0. This is the "merge nearby parts" requirement
     of W7.
   - It also addresses one W2 cause noted in `parts.rs`: closed parts stall at their minimal
     triangulation. After a fill merges them they can be decimated further, instead of being
     deleted outright.

**Why it might beat the current state of the art.** Common practice, such as Nooruddin & Turk
2003 *(not re-verified)*, is morphological closing or opening alone. Those either only fill
(blobbing) or only cut (shredding). The cut-or-fill choice per feature is exactly the
controlled topology change W7 asks for, and the paper reports better topology and less
geometric distortion than greedy or single-sided methods.

**Cost and risk.** Medium: about 800 lines for greedy plus bookkeeping; the Steiner-tree solver
is an extra. Risks:
- Voxel resolution limits which features can be seen. This shares W3's sub-voxel problem.
- The paper notes that the quality of the candidates bounds the result.
- Determinism is easy (integer voxels, ordered greedy).

**Absorbed by graphics?** Published in graphics venues. I found no LOD or proxy product that
does it, but absence from public documentation is not proof.

---

## 4. Curvature-potential predictor for chart merging and seam placement (W4, W1)

**Source discoveries**
- Mitchell, Koning, Bowick & Vitelli, "Fracture in sheets draped on curved surfaces", Nature
  Materials 2017 ([arXiv 1512.04061](https://arxiv.org/pdf/1512.04061)).
  - A flat sheet forced onto a surface of Gaussian curvature G develops an isotropic stress
    σ_kk = EΦ, where E is Young's modulus.
  - Φ is a geometric potential satisfying the Poisson equation ∇²Φ = −G, like electrostatics
    with G as the charge.
  - Positive curvature promotes crack initiation, negative curvature suppresses it, and
    gradients of Φ steer and arrest cracks.
- Related: Cohen, Pandey, Zhang, Maor & Moshe, "Curvature potential formulation for thin
  elastic sheets", [arXiv 2512.03270](https://arxiv.org/html/2512.03270v1), Dec 2025. It
  extends this to large slopes, but only from a flat reference state.

**Mechanism.** The strain needed to flatten a patch can be predicted, without flattening it,
from one scalar Poisson solve with curvature as the source.

**Transfer to farthing**
- In `merge_charts`, every candidate pair currently runs a full LSCM+ARAP trial. Instead,
  first solve the discrete Poisson equation L Φ = −K on the union. Use cotan L, angle-defect
  K, and Dirichlet Φ = 0 on the union's boundary.
- Reject the pair outright if max|Φ| scaled by chart size exceeds a calibrated threshold.
  Calibrate it against farthing's own L2-stretch outcomes on the test assets.
- This is one sparse scalar SPD solve, against LSCM's 2n-unknown system plus ARAP iterations,
  and it can warm-start from the two parts' potentials.
- For splits, cut where Φ indicates high stress. Start cuts from boundary points facing the
  Φ maximum, following the "crack is attracted to high-Φ" rule, instead of re-growing with a
  tighter normal cone.

**Why it might help.** It is mainly a speed win for W1 (the UV atlas runs 1.3x slower than
xatlas), and it lets more merges be tried within the same time, which helps W4. It also gives
splits physically motivated seam paths rather than normal-cone boundaries.

**Cost and risk.** Low: farthing already has a sparse solver (`farthing/src/uv/sparse.rs`).
Risk: the predictor is linear (small-strain) and ignores the free-boundary conditions of real
fracture mechanics. Use it only as a conservative pre-filter, keeping the exact trial for
borderline pairs.

**Absorbed by graphics? Yes, mathematically.**
- This is the linearized conformal-factor equation (Yamabe/Poisson) used by Ben-Chen, Gotsman
  & Bunin 2008 for cone placement
  ([EG diglib](https://diglib.eg.org/handle/10.2312/CGF.v27i2pp449-458)).
- Sharp & Crane's "Variational Surface Cutting" (SIGGRAPH 2018,
  [project](https://www.cs.cmu.edu/~kmcrane/Projects/VariationalCuts/index.html)) evaluates
  cut cost from the Yamabe equation "without explicitly parameteriz[ing] the surface".
- The physics adds intuition and the crack-steering rule. Its use as a cheap merge pre-filter
  in a region-growing pipeline like farthing's is the practical new part.

---

## 5. Unjamming the decimator: swap moves and soft constraints (W2)

**Source discoveries**
- **Swap Monte Carlo.** Ninarello, Berthier & Coslovich, "Models and algorithms for the next
  generation of glass transition studies", PRX 7:021039, 2017
  ([APS](https://link.aps.org/doi/10.1103/PhysRevX.7.021039)).
  - Adding non-physical moves that swap particle identities lets jammed glass-formers relax
    many orders of magnitude faster than local moves alone.
  - The precise speed-up depends on the model; I did not re-check the paper's quoted figure.
- **Tissue rigidity.** Bi, Lopez, Schwarz & Manning, "A density-independent rigidity transition
  in biological tissues", Nature Physics 2015 ([arXiv 1409.0593](https://arxiv.org/pdf/1409.0593)).
  - Whether a confluent tissue is solid or fluid is controlled by the energy barriers to T1
    (neighbour-exchange) transitions, not by density.

**Mechanism.** A system stuck because every local move violates a hard constraint becomes
mobile when a second, connectivity-changing move class is added, or when the hard constraint
is softened into a penalty with relaxation.

**Transfer to farthing**
- When `executed == 0` triggers stall escalation in `simplify.rs`, insert a "swap round" before
  giving up:
  - (a) For each collapse rejected by `Validity::Flip` or `Validity::Degenerate`, try the
    edge flips (T1 moves) in the one-ring of the endpoint that make the collapse valid.
    Accept a flip if it does not raise the local quadric error by more than ε, then
    re-validate the collapse.
  - (b) "Soft guard": accept a blocked collapse if, after re-optimizing the positions of the
    one-ring vertices for a few Gauss-Newton steps (QEM plus a barrier on normal deviation),
    all normals pass `min_face_dot`. If not, roll back.
- Both are deterministic when processed in canonical order.
- This targets the "flip guard" half of W2. The non-manifold-lock half is better served by
  idea 3 or by splitting non-manifold vertices.

**Why it might help.** At around 1% of the original triangles, most blocked collapses are
geometric: needle-like one-rings that a flip would fix. Flips do not reduce the triangle count
but unblock collapses that do.

**Cost and risk.** About 200-300 lines (flip operator plus validity re-check). Risks: flip
cascades (cap them per round) and attribute seams (do not flip seam edges).

**Absorbed by graphics?** Edge swaps are in Hoppe et al. 1993 "Mesh Optimization"
*(not re-verified)* and in isotropic remeshing. The 2024 paper "Mesh Simplification for
Unfolding" ([arXiv 2408.06944](https://arxiv.org/pdf/2408.06944)) reports the same symptom
("every edge in the initial collapse queue is rejected") and uses only collapse-side exit
heuristics, not flips. Honest assessment: this is modest novelty. Physics mainly argues for
the schedule, which is to add a second move class exactly when the first jams.

---

## 6. Topology-optimization length-scale control (W7 candidates, W3 error bound)

**Source.** Wang, Lazarov & Sigmund, "On projection methods, convergence and robust
formulations in topology optimization", SMO 2011
([DTU Orbit](https://orbit.dtu.dk/en/publications/on-projection-methods-convergence-and-robust-formulations-in-topo/)).
Analytical rules linking filter radius and thresholds to minimum solid and void sizes are in
Fernández et al. 2021 ([arXiv 2101.08605](https://arxiv.org/pdf/2101.08605)). Bone remodelling
is the biological analogue (Wolff's law) of the same density-filter idea.

**Mechanism.** Filter a density field with a radius-R kernel, then project it at three
thresholds (eroded, intermediate, dilated). This yields three nested designs and guarantees
minimum feature sizes for solid and for void separately.

**Transfer.**
- Use the eroded and dilated pair as the cut and fill candidate generators for idea 3, with
  separate r_solid and r_void.
- The nesting eroded ⊆ result ⊆ dilated gives a free per-region bound on the Hausdorff change,
  which can be reported in the proxy's error metadata.
- Optionally build the filter from a smoothed occupancy or distance field rather than binary
  voxels, so thresholds can be placed at sub-voxel positions. This helps W3 only marginally;
  a signed-distance approach remains the real W3 fix.

**Cost.** Low (separable blur plus thresholds).

**Absorbed?** Mostly. It is morphology in different clothes. Value is moderate, mainly as
idea 3's front end.

---

## 7. Physarum / adaptive transport networks for seam graphs (W4), low priority

**Source**
- Tero et al., "Rules for biologically inspired adaptive network design", Science 327:439,
  2010 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/20093467/)).
- Physarum-inspired Steiner-tree solvers, for example Y. Sun,
  [arXiv 1903.08926](https://arxiv.org/pdf/1903.08926): optimal on 17 of 18 OR-Library B
  instances in the author's own comparison, but too slow on large graphs without partitioning.

**Transfer.** Treat the seam cut graph, connecting cone points to the boundary, as a network
whose edge conductances adapt to "flux" between terminals. Edge cost would combine visibility,
curvature and texture importance.

**Assessment.** Farthing's cut graphs are small. A shortest-path or MST Steiner heuristic is
nearly as good and far simpler, and the Physarum literature reports evaluation counts rather
than wall time. I would not pursue this unless seam visibility weighting becomes a priority.

---

## Checked and judged absorbed or not transferable

These come from background knowledge and were not individually re-verified this session.

- **Kirigami and auxetics for conforming sheets** (Konaković-Luković et al. SIGGRAPH 2016 and
  2018), and **developability flows** (Stein, Grinspun & Crane 2018): already graphics. They
  give seams at creases, but at the cost of remeshing geometry, which is unsuitable for UV of
  fixed LOD meshes.
- **Defects in curved crystals** (Bausch et al. 2003; Irvine, Vitelli & Chaikin 2010
  "pleats"): the insight that curvature is screened by defects is the same as cone placement
  (Ben-Chen 2008, BFF). Nothing extra for farthing.
- **Leaf venation and space colonization** (Runions et al. 2005): graphics already uses it,
  for growth modelling, not LOD.
- **Phyllotaxis, scutoids, epithelial Voronoi packing, Cellpose flow-field segmentation:** no
  concrete advantage over existing chart growing or watershed segmentation.
- **Lubachevsky-Stillinger growth packing:** a special case of idea 2.

## Summary of uncertainty

- The expected W4 gains (ideas 1 and 2) are my estimates, not measured. Benchmark against
  farthing's xatlas comparison set (`bench-compare/`) before committing.
- US 12,112,510 B2 should be reviewed by counsel before any correlation-based placement ships.
