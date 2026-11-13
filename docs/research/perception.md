# farthing research: vision science, psychophysics, neuroscience → W5 (perceptual error), W6 (impostors)

Scope: turn human-vision findings into **cheap per-vertex / per-face / per-chart terms** that the
edge-collapse reducer (`crates/farthing/src/reduce/simplify.rs`, cost at ~L1000-1030) and the UV
texel-density allocator (`crates/farthing/src/uv/param.rs`, density search at ~L602) can use
**without rendering each candidate**, plus impostor parameter choices (`crates/farthing/src/impostor`).

Read this first: most of the *psychophysics* below is old and solid. The perceptual mesh
community has already applied a lot of it in research code (2001-2016). As far as I can tell
from public material, production simplifiers have not. So the payoff is mostly engineering:
make it cheap, deterministic and calibrated. It is not a new scientific discovery. Each idea
says where it has already been absorbed.

Current farthing state that matters here (I read the code):
- Cost = `sqrt(geom2 + attr2) * importance`. The normal term uses a hand heuristic ("a normal
  deviation over area A shades like a bump of height θ·√A", `en += n * (wi / avg_area)`) with a
  fixed `normal_weight = 5e-5`. **No option knows the screen size the LOD will be viewed at.**
- `vertex_importance` (multiplicative) already exists. `farthing::visibility` already produces
  per-vertex visibility importance.
- The UV allocator uses one uniform texel density for all charts (`lscm.rs:424`, `param.rs:602`).
- Impostors: octahedral (Brucks 2018) with a HEIGHT channel for parallax, flipbook, billboard clouds.

---

## Ranking (expected payoff / effort)

| # | Idea | W | Payoff | Effort | Already in graphics? |
|---|------|---|--------|--------|----------------------|
| 1 | Screen-size-aware JND cost: CSF-weighted shading term + JND stop criterion | W5 | High | Medium | Research yes (2001-2016), production no (as far as I know) |
| 2 | Closed-form expected-silhouette weight (dihedral angle/π, curvature), bake-aware | W5 | High | Low | The probability fact is known in NPR (1997); using it as a simplification weight: not found |
| 3 | Masking map (texture + baked-normal-map detail + roughness) → threshold elevation | W5 | Med-High | Low-Med | Research yes (Ferwerda 97, Qu-Meyer, Lavoué 09); not in QEM reducers |
| 4 | Pop-aware LOD chain spacing (temporal visibility binds before static) | W5 | Med-High | Low-Med | New (2023) finding; not standard |
| 5 | One-time calibration of the cheap model against ColorVideoVDP/FLIP | W5 | High (enables 1-4) | Medium | Metrics standard; using them to *fit* a per-vertex proxy is not |
| 6 | Perceptual texel-density allocation ρ_c ∝ √w_c, capped at display Nyquist | W5 (UV) | Medium | Low | Partly (signal-specialized param. 2002; visibility weighting 2002) |
| 7 | Gloss-scaled normal tolerance | W5 | Medium (glossy assets) | Low | Partly (Williams 2003, Menzel-Guthe 2010) |
| 8 | Low-frequency error tolerance (bas-relief / affine ambiguity, high-pass error) for proxies | W3/W5/W7 | Medium, uncertain | Medium | Compression yes (Sorkine 2003); simplification/remesh no |
| 9 | Impostor parameters from stereo / motion-parallax / ghosting thresholds | W6 | Medium | Low | Heuristic practice; no psychophysical rule in common use |
| 10 | Mesh saliency, foveation, crowding | – | Low (negative result) | – | Mesh saliency is absorbed but has weak evidence |

---

## 1. Screen-size-aware JND cost (CSF + contrast from normal deviation)

**Source.**
- Luebke & Hallen 2001, *Perceptually Driven Simplification for Interactive Rendering*, EGWR
  ([EG DL](https://diglib.eg.org/items/2bee1720-f9cc-4b83-9bb0-40d3a4555305)). Each local
  simplification is scored by the contrast it induces and the spatial frequency of the change,
  using a simple, conservative model. They reported their models could be simplified 2-3×
  further without visible artifacts (as quoted by Williams et al. 2003).
- Williams, Luebke, Cohen, Kelley, Schubert 2003, *Perceptually Guided Simplification of Lit,
  Textured Meshes*, I3D ([PDF](https://scispace.com/pdf/perceptually-guided-simplification-of-lit-textured-meshes-3iz3u4sb7d.pdf)).
  - Frequency comes from screen-space feature size ("a texture deviation of ε can create or
    destroy features no larger than 2ε"), clamped to no lower than the CSF peak.
  - Contrast comes from texel luminance ranges and normal-cone lighting bounds.
  - The Daly CSF is inverted into an "imperceptibility distance".
  - Biggest gains were on vertex-lit models, mainly through better handling of specular
    highlights.
  - About 15 min preprocessing for a 500k-triangle model (2003 hardware).
- Menzel & Guthe 2010, *Towards Perceptual Simplification of Models with Arbitrary Materials*,
  CGF ([EG DL](https://diglib.eg.org/handle/10.2312/CGF.v29i7pp2261-2270)). Moves the
  perceptual error from image space to vertex space (contrast and frequency at a vertex) to
  avoid per-pixel work. Validated by a user study.
- Nader, Wang, Hétroy-Wheeler, Dupont 2016:
  - *Just Noticeable Distortion Profile for Flat-Shaded 3D Mesh Surfaces*, IEEE TVCG
    ([project](https://gnader.github.io/publications/2016-jnd.html)).
  - *Visual Contrast Sensitivity and Discrimination for 3D Meshes and their Applications*, CGF
    35(7) ([EG DL](https://diglib.eg.org/items/1ab83fa2-1831-4b77-a937-07dbedea3147/full),
    [supp.](https://core.ac.uk/download/49316925.pdf)).
  - Their psychophysics on meshes gives a mesh-specific CSF peaking at about 3.5 cpd, plus
    contrast discrimination: the threshold rises roughly linearly once the base contrast is
    visible.
  - Smooth-shaded thresholds are 5-10× larger than flat-shaded ones.
  - Contrast inside a triangle is computed in closed form from the vertex normals. A
    light-independent mode takes the worst case over sampled light directions.
  - They give a per-vertex, direction-dependent displacement JND: displacement along the normal
    is barely tolerable on flat areas, tangential displacement is more tolerable.
  - They already drive edge collapses with it. Edge impact = (v2−v1)·(v_jnd2−v_jnd1)/‖e‖², and
    simplification stops when all remaining edges are above threshold. Distance-dependent
    thresholds give adaptive LODs.
  - **Cost is the problem**: roughly 500 s for 1M vertices with 12 light samples (read off
    their plot).
- Modern CSF: castleCSF, Ashraf, Mantiuk, Chapiro, Wuerger 2024, J. Vision 24(4)
  ([project](https://www.cl.cam.ac.uk/research/rainbow/projects/castleCSF/)). Inputs: spatial
  and temporal frequency, colour direction, luminance, eccentricity, stimulus area.

**Mechanism.** A geometric change is visible if the luminance contrast it produces, divided by
the detection threshold at its retinal spatial frequency (the CSF), exceeds 1 (one JND). The
threshold is raised further by masking (idea 3). Retinal frequency depends on the footprint of
the change **in pixels**, so it depends on the screen size the LOD is viewed at.

**Transfer to farthing (W5).** Add `view: Option<PerceptualView>` to `SimplifyOptions`:
`{ pixels_per_unit_at_switch: f32, pixels_per_degree: f32, adapt_luminance: f32, silhouette_tol_px: f32 }`.
farthing generates LODs offline for known switch distances, so `s = pixels_per_unit` is known
for each LOD. Then:

1. *Footprint frequency.* The quadric's accumulated area `qd.w` is already available in the cost
   function. Use footprint radius `r ≈ sqrt(w/π)` and `f = ppd / (2·r·s)` in cpd. Clamp
   f ≥ f_peak, as Williams did; this is the conservative choice.
2. *Shading contrast without a light.* Worst case over directional light:
   `c ≈ |Δn|` (radians), since `max_l |Δn·l| = |Δn|`. A smooth-shading factor
   `k_smooth ≈ 1/5…1/10` (Nader 2016) applies when vertex normals are interpolated. Δn is
   already estimated by the normal attribute quadric.
3. *Weight.* Replace the `wi/avg_area` heuristic by `W_n(f) = (S(f)·k_smooth / M)²`:
   - S is a CSF lookup table, 64 entries, built once per LOD.
   - M is the masking elevation from idea 3.
   - At small screen sizes S(f) → 0 for small footprints, so the reducer stops paying for
     invisible shading detail and spends the triangle budget on silhouettes.
   - At large screen sizes it tightens. **The normal weight then varies with LOD index
     automatically instead of being one constant.**
   - Implementation: 1 sqrt plus 1 table lookup per evaluation, branch-free.
4. *Stop criterion.* `max_jnd: Option<f32>` stops when the cheapest collapse would exceed V = 1,
   giving a "visually lossless at distance D" LOD. This complements `target_ratio`.
5. *Which CSF.* For a pure-Rust LUT, use a simple analytic CSF:
   - Mannos-Sakrison or Barten form, re-peaked to Nader's mesh-measured ~3.5 cpd, or
   - castleCSF's achromatic, static, foveal slice tabulated from the paper's equations. The
     repo's license is not stated on the project page, so implement clean-room from the paper.
   - The absolute level is set by calibration (idea 5). Only the shape matters here.

**Why it may beat current practice.** Current geometric/attribute QEM uses one weighting for
every LOD. Perceptually, the right weighting of shading vs silhouette vs UV error flips with
screen size. Williams 2003 already showed lower image error than screen-space metrics at equal
triangle counts, especially for lit and specular meshes. Luebke reported 2-3× slack in its own
conservative model.

**Cost/risk.** About 200-400 lines of Rust.

Risks:
- CSF models are threshold models. For aggressive LODs (1%) everything is suprathreshold, so V
  should be used as a *ranking* currency, with Minkowski pooling, not as a hard visibility
  guarantee.
- Williams explicitly calls CSF use for suprathreshold a stopgap.

Deterministic: no rendering, pure functions of mesh and options.

**Absorbed?** In research, yes (2001, 2003, 2010, 2016). In meshoptimizer/xatlas-class tools,
not as far as I can find. I did not verify Simplygon internals and per the brief did not try.

---

## 2. Closed-form expected-silhouette weight (+ bake-aware geometric cost)

**Source (vision).**
- Koenderink 1984, *What does the occluding contour tell us about solid shape?*, Perception 13
  ([DOI](https://journals.sagepub.com/doi/10.1068/p130321)): the sign of contour curvature
  equals the sign of Gaussian curvature.
- Cole et al. 2009, *How well do line drawings depict shape?*, SIGGRAPH
  ([project](https://pixl.cs.princeton.edu/pubs/Cole_2009_HWD/index.php)): people interpret
  certain shapes almost as well from line drawings as from shaded images, and errors trace to
  specific line properties.
- Most directly: Surace, Tursun, Celikcan, Didyk 2023, *Gaze-Contingent Perceptual Level of
  Detail Prediction*, EGSR ([PDF](https://diglib.eg.org/bitstream/handle/10.2312/sr20231130/119-130.pdf)).
  - They model LOD visibility as a separate **silhouette term**, G = (pixels in the XOR of the
    two silhouette masks, weighted by eccentricity) / reference silhouette perimeter.
  - A FovVideoVDP shading term is computed with the silhouette region grayed out.
  - The two are Minkowski-combined, exponent 5.39.
  - The separate silhouette term raised prediction from r = 0.72 (VDP alone) to 0.81
    (cross-validated 0.69 → 0.74).

**Mechanism (my derivation; standard geometry, check before relying on it).**
Silhouette error in the image is approximately (normal-direction displacement d_n) × (length of
contour passing through the region). At the contour the surface normal lies in the image plane,
so **only the normal component of displacement moves the silhouette**, and that is exactly what
the QEM plane distance measures. For a uniformly random view direction:
- *Discrete crease edge* with angle θ between its two face normals: the edge is a contour edge
  iff the two faces face opposite ways, which happens on two lunes of total area 4θ. So
  P = θ/π. Expected contour length = ℓ_e·θ_e/π.
- *Open border edges*: P = 1 (always a potential silhouette).
- *Smooth limit*: expected contour length per unit area = (perimeter of the ellipse with semi-axes
  |κ1|, |κ2|)/(4π) ≈ (|κ1|+|κ2|)/4.
- Flat regions contribute zero. Their distance error is invisible at the silhouette and shows
  only through shading and texture.

Markosian et al. 1997 (*Real-time NPR*,
[PDF](https://www.cs.princeton.edu/courses/archive/fall00/cs597b/papers/rt-npr-s97.pdf)) already
used "probability an edge is a silhouette ∝ dihedral angle" to find silhouettes by random
sampling. The fact is known in NPR. I found no simplifier that uses it as an *expected
silhouette error* weight.

Expected silhouette XOR area for a candidate collapse is then approximately
`s · Σ_faces |d_n,f| · L_f`, normalized by the expected total perimeter. Visibility probability
from `farthing::visibility` can multiply L_f.

**Transfer (W5).**
- Per input face, compute `L_f` once: half of each edge's ℓθ/π, plus border edges, times
  visibility.
- Scale the face's plane quadric by `w_f = ε + (s/τ_sil)² · L_f/A_f`. Since QEM is quadratic,
  this is the L2 surrogate of the linear XOR area.
- Quadrics already sum under collapse, so cost per evaluation is **zero**. Precompute is one
  pass over the edges.
- **Bake-aware mode.** When the pipeline will bake a normal map from the source onto the LOD,
  interior shading error is largely restored by the bake. The geometric term should then be
  mostly the silhouette term (ε small); keep texture-deviation/UV and parallax concerns. When no
  bake follows, keep the shading term from idea 1. This gives a principled split for the "LOD +
  baked normals" workflow that Simplygon-class pipelines target.

**Why it may beat current practice.** Visibility-weighted QEM exists (Zhang & Turk 2002,
*Visibility-Guided Simplification*, IEEE Vis,
[PDF](https://web.engr.oregonstate.edu/~zhange/images/surfviz.pdf)). It cut image error by
2-52% at equal triangle count with <1% simplify-time overhead once precomputed. Visibility
alone, though, does not distinguish "visible but never on the silhouette" (flat front faces,
cheap when baked) from "often on the silhouette" (rims, creases, thin parts). Surace et al. show
the silhouette term carries independent predictive power.

**Cost/risk.** About 100 lines. Low risk.

Caveats:
- The view distribution is assumed uniform. For ground-bound assets it should be the
  game-camera distribution; the θ/π formula then becomes an integral over that distribution,
  which can be Monte Carlo precomputed with a fixed seed.
- The smooth-curvature version needs a curvature estimator. The discrete θ/π version does not.

**Absorbed?** Parts: silhouette-preserving view-dependent LOD (normal cones; Luebke, Hoppe) and
visibility weighting. The closed-form view-averaged silhouette weight and the bake-aware
split: not found in my searches. Mark as **probably novel in combination**; the individual
pieces are not novel.

---

## 3. Masking map: texture, baked normal-map detail, and roughness raise thresholds

**Source.**
- Ferwerda, Pattanaik, Shirley, Greenberg 1997, *A Model of Visual Masking for Computer
  Graphics*, SIGGRAPH ([page](https://en-cg-web.coecis.cornell.edu/pubs/1997/FPSG97.html)):
  psychophysics-based masking model that predicts which textures hide faceting from polygonal
  tessellation as a function of texture contrast, frequency, orientation and tessellation.
- Qu & Meyer, *Perceptually Driven Interactive Geometry Remeshing*
  ([PDF](https://www-users.cse.umn.edu/~gmeyer/papers/qu-meyer-2006.pdf)):
  - They unwrap texture, bump and lighting into a 2D map and compute masking with the Sarnoff
    VDM against a blurred copy.
  - Fewer samples land in textured regions, more around eyes, nose and mouth.
  - They explicitly chose remeshing because per-collapse masking evaluation was too expensive.
- Lavoué 2009, *A local roughness measure for 3D meshes and its application to visual masking*,
  ACM TAP ([HAL](https://hal.archives-ouvertes.fr/hal-01437612)), and the fast roughness metric
  FMPD of Wang, Torkhani, Montanvert 2012, Computers & Graphics
  ([HAL](https://hal-ecp.archives-ouvertes.fr/GIPSA-AGPIG/hal-00734726)). I could not open
  these pages (bot wall); citation only.
- Nehmé, Delanoy, Dupont, Farrugia, Le Callet, Lavoué 2023, *Textured Mesh Quality Assessment:
  Large-Scale Dataset and Deep Learning-based Quality Metric*, ACM TOG
  ([arXiv](https://arxiv.org/pdf/2202.02397)):
  - 343,750 stimuli, 148,929 crowdsourced judgments.
  - Geometric complexity and rich colour texture both mask geometric artifacts. Near-uniform
    textures are less sensitive to UV quantization. Front lighting masks geometric detail.
  - Texture JPEG quality ≥ 50 and sizes > 712² had little effect on their rendered stimuli.

**Mechanism.** Contrast masking: a detection threshold rises with the contrast of co-located
content at similar frequency and orientation. The usual form is
`M = max(1, (C_mask/C_thr)^ε)` with ε ≈ 0.6-0.7 (Legge-Foley/Daly style; textbook value, not
re-verified here).

**Transfer (W5, and UV in idea 6).** Per vertex or face, compute a masker contrast `C_mask`
from:
- (a) albedo-texture local RMS contrast in the band around the LOD's footprint frequency. The
  host supplies it as a per-face scalar through the existing bake/RenderHost boundary; farthing
  never interprets materials.
- (b) **detail that will be baked into the LOD's normal map**: normal variation of the source
  inside the footprint, which the reducer already sees as the residual normal error. This is
  the useful coupling. Detail the bake restores keeps masking the LOD's remaining geometric
  error.
- (c) Lavoué-style roughness: Laplacian-smoothed vs original curvature difference.

Divide the shading term of idea 1 by M. Optionally divide the UV/texture-deviation term by M
too. Precompute only: O(n) over the source mesh, and textures sampled once per face.

**Why better.** It moves triangle budget off busy or rough or textured regions, where
simplification artifacts are invisible, onto smooth, uniformly coloured areas, where faceting is
glaring. Pure QEM does the opposite: rough areas have large quadric error.

**Cost/risk.** About 150 lines plus a host hook for per-face texture statistics.

Risk: masking is orientation- and frequency-specific, and a scalar M is a crude isotropic
approximation. It is fine for ranking, **uncertain** for absolute thresholds.

**Absorbed?** Research yes (1997, 2006, 2009, 2023 dataset evidence). In QEM-collapse reducers:
not found.

---

## 4. Pop-aware LOD chain spacing

**Source.** Surace et al. 2023 (above). Their temporal model is adapted from Tursun & Didyk 2022,
*Perceptual Visibility Model for Temporal Contrast Changes in Periphery*, ACM TOG
([arXiv](https://arxiv.org/pdf/2205.00108)). It predicts the visibility of an instantaneous LOD
switch with r = 0.91, against 0.53 for the original formulation.

**Key empirical finding.** When choosing the coarsest acceptable LOD under both a spatial and a
temporal (popping) constraint, **the temporal constraint decided the choice in 84% of stimuli
with 4 LODs**. With 13 LODs, the spatial model decided in 41%.

**Transfer (W5).** For coarse LOD chains, which are typical (3-5 LODs), *switch-pop visibility*,
not static error vs the original, is the binding constraint. farthing can therefore:
1. Run the reducer once down to the smallest target and record the collapse log (Δd_n, Δn, area
   per collapse). The run is deterministic.
2. Choose LOD cut points and **switch screen sizes** so that the predicted pop between
   consecutive LODs, at their switch distance, equals a target. Pop is the Minkowski-pooled V
   of ideas 1-3 over the collapses between the two cut points.
3. Emit the switch screen sizes as metadata. Today the user typically picks ratios such as
   50/25/12/6% and distances by eye.

**Why better.** It equalizes perceived pops instead of triangle ratios. It also lets farthing
answer the question users actually ask: "how many LODs, and when to switch".

**Cost/risk.** Low-medium. The collapse log is cheap.

Risk: the pooling exponent and threshold need calibration (idea 5). Surace's own calibration
used single objects, simple shading, uniform backgrounds and 5-15 participants.

**Absorbed?** Engines pick switch distances by screen-size heuristics. A psychophysically
calibrated pop criterion for *choosing the chain* is not standard.

---

## 5. Calibrate the cheap model once against ColorVideoVDP / FLIP; never render per candidate

**Source.**
- ColorVideoVDP, Mantiuk et al. 2024, SIGGRAPH ([repo](https://github.com/gfxdisp/ColorVideoVDP/),
  MIT license, PyTorch): display-model-aware (resolution, distance, size, luminance) JOD
  quality and heat maps, built on castleCSF.
- FLIP, NVIDIA (BSD-3, C++/CUDA/Python/PyTorch) ([repo](https://github.com/NVlabs/flip)).
- Lavoué, Larabi, Váša, *On the Efficiency of Image Metrics for Evaluating the Visual Quality of
  3D Models*, IEEE TVCG ~2016 ([PDF](https://perso.liris.cnrs.fr/guillaume.lavoue/revue/TVCG2015.pdf)):
  - Image metrics on 42 rendered views reach about 90-95% correlation on simplification *per
    object*, but only 65-83% pooled across objects.
  - IW-SSIM was the most stable, max-pooling worked best, and non-frontal lighting was better.
- Watson, Friedman, McGaffey 2001, *Measuring and predicting visual fidelity*, SIGGRAPH
  ([PDF](https://arxiv.org/pdf/2507.11857)): automatic measures predict ratings well but naming
  times poorly. QSlim beat vertex clustering, by a large margin at 80% reduction.

**Transfer (W5).** Do not put a VDP in the loop. Instead:
- Build a calibration harness in `bench-compare`:
  - render the source and each LOD from ~42 views, using `core/raster.rs` or the RenderHost,
    with 2 lighting setups;
  - compute ColorVideoVDP, run as an external dev-only Python tool and not shipped, or a
    pure-Rust FLIP port (FLIP is small; about 1-2k lines);
  - split silhouette and interior as Surace does: XOR mask area plus a VDP with the silhouette
    region grayed out.
- **Fit** the 4-6 scalars of ideas 1-4 by least squares: silhouette tolerance, CSF level,
  masking exponent, Minkowski exponent, smooth-shading factor, pop threshold.
- Use the same harness as the W5 regression metric for every reducer change.

**Why better.** It converts the "uncertain constants" in ideas 1-4 into measured ones, at zero
runtime cost.

**Cost/risk.** Medium (harness). Low risk. Determinism is unaffected because the calibration
output is a constant table checked into the repo.

**Absorbed?** The metrics are standard. Using them to fit a per-vertex proxy for an offline
reducer is not a standard published practice as far as I found.

---

## 6. Perceptual texel-density allocation for the UV atlas

**Source.**
- Visibility weighting: Zhang & Turk 2002 (above).
- Signal-adaptive density: Sander, Gortler, Snyder, Hoppe 2002, *Signal-Specialized
  Parametrization* ([PDF](https://hhoppe.com/ssp.pdf)).
- Masking: idea 3.
- Natural-image 1/f amplitude spectra: Field 1987, JOSA A. Classic; not re-fetched.
- Cauchy's surface-area formula: the mean projected area of a convex body is area/4, i.e.
  E[max(0, n·v)] = 1/4 for every surface element under uniform views.

**Mechanism and derivation (mine; check).** Minimise Σ_c w_c·A_c·D(ρ_c) subject to
Σ_c A_c·ρ_c² = B (atlas area), where D is perceptual loss at texel density ρ.
- With 1/f amplitude spectra, the energy lost above the cutoff is ∝ log(ρ_max/ρ) (equal energy
  per octave). This gives **ρ_c ∝ √w_c**.
- If D ∝ ρ^−α (steeper spectra), ρ_c ∝ w_c^{1/(α+2)}.
- Cap ρ_c at the display Nyquist for the nearest distance the LOD is shown at,
  ρ_cap ≈ s_max·(1…1.5) texels per unit. Texels beyond that are invisible.
- By Cauchy, uniform density is already exposure-optimal for convex, unoccluded objects under
  uniform views. Only occlusion, non-uniform camera distributions, texture content and
  semantics should move density.

**Transfer.** `ParamOptions { chart_weight: Option<Vec<f32>>, max_texels_per_unit: Option<f32> }`.
Per-chart weight is w_c = V_c · T_c · U_c:
- V_c: mean visibility, from `farthing::visibility::triangle_weights` or BVH accessibility rays;
- T_c: masking-adjusted, CSF-weighted texture detail energy above the current density, from
  host per-face statistics;
- U_c: user/semantic weight, e.g. faces.

Scale each chart's UVs by √w_c before packing. Keep the existing binary search on the global
density at `param.rs:602`. This also helps W4 slightly: shrinking low-weight charts frees area.

**Why better.** Uniform density spends texels on occluded interiors, flat-colour regions and
detail the eye cannot resolve at the LOD's distance. Nehmé et al. found texture size above
~712² barely mattered on their stimuli, which suggests substantial slack.

**Cost/risk.** Low (about 100 lines plus a host hook). Risk: density discontinuities between
adjacent charts can show as a texture-sharpness seam. Limit the ratio between neighbours
(e.g. ≤ 2×). This is a **heuristic, unverified** rule.

**Absorbed?** Partly. Signal-specialized parametrization and visibility-guided ideas exist. The
combined √w rule with a Nyquist cap tied to LOD distance is not, as far as I found.

---

## 7. Gloss-scaled normal tolerance

**Source.**
- Fleming, Torralba, Adelson 2004, *Specular reflections and the perception of shape*, J.
  Vision 4(9) ([abstract](https://persci.mit.edu/pub_abstracts/Fleming_jov-4-9-10.html)):
  specular image compressions depend on the second derivative of depth (curvature), unlike
  texture, which depends on the first. The resulting orientation fields are diagnostic of 3D
  shape.
- Williams 2003: perceptual gains were largest via specular highlights.
- Lavoué et al. 2018 (below): specular highlights attract fixations.

**Mechanism.** A highlight's intensity changes by roughly |Δn|/σ of its peak, where σ is the
BRDF lobe width. Its *shape* is driven by curvature, so faceting a curved glossy surface breaks
highlights into facet-shaped patches, which observers read as shape change.

**Transfer.** A per-material scalar `gloss_hint` (lobe width σ), passed as data and not as a
material model. Multiply the normal-attribute weight by (σ_ref/max(σ, σ_min))². Optionally add
a curvature-error (second-order) term on glossy materials only.

**Evidence quality: mixed.** Qu & Meyer reported a *shiny* teaspoon needing fewer vertices than
a diffuse one, because environment reflections masked errors, although they excluded
highlights. Treat as **uncertain**; validate with the idea-5 harness.

**Cost.** Trivial. **Absorbed?** Partly (Williams 2003, Menzel-Guthe 2010).

---

## 8. Low-frequency error tolerance: bas-relief / affine ambiguity → high-pass error for proxies

**Source.**
- Belhumeur, Kriegman, Yuille 1999, *The Bas-Relief Ambiguity*, IJCV
  ([page](https://vision.ucsd.edu/node/253)): under Lambertian, orthographic viewing, a surface
  and its generalized-bas-relief transform produce identical images (including shadows) under
  suitably transformed lights.
- Koenderink, van Doorn, Kappers, Todd 2001, *Ambiguity and the 'mental eye' in pictorial
  relief*, Perception 30 ([page](https://research.vu.nl/en/publications/ambiguity-and-the-mental-eye-in-pictorial-relief/)):
  observers' perceived relief differs greatly, but the differences almost vanish once the
  "essential ambiguity" is factored out. My understanding is that this ambiguity is an affine
  depth scaling/shear, but I could not confirm the exact form from the accessible text.
- Graphics analogue: Sorkine, Cohen-Or, (Chen), Toledo 2003/2005, high-pass quantization
  ([PDF](https://www.math.tau.ac.il/~stoledo/Pubs/tog_lowres.pdf)): low-frequency displacements
  are "much less noticeable" than high-frequency ones. Their visual error weights the Laplacian
  term heavily (α < 0.15), which they call ad hoc.

**Transfer (W3, W7, aggressive W2 LODs).** When evaluating or projecting voxel-remeshed proxies,
measure error in differential (Laplacian) coordinates plus a small absolute term. This tolerates
smooth offset or inflation, which is useful for conservative proxies, and penalizes crevice
filling and ridge loss. Projection back to the source can then accept a smooth bias to avoid
high-frequency noise. For QEM, a Laplacian-difference attribute is possible but expensive.

**Why it might help.** It favours proxies that keep relief structure over ones that minimize
Hausdorff. **Uncertain**: the perceptual ambiguity result is about pictorial relief *without*
reference. In side-by-side or popping comparisons, absolute differences become visible again.
So use it for the far proxy LODs only.

**Cost.** Medium (cotangent Laplacian already common). **Absorbed?** In compression yes; in
simplification/remeshing metrics, partially (MSDM2/curvature-based quality metrics).

---

## 9. Impostor parameters from stereo, motion-parallax and ghosting thresholds (W6)

**Source.**
- Stereoacuity is typically ~0.5 arcmin, with ~0.1 arcmin for the best observers
  ([summary](https://en.wikipedia.org/wiki/Stereoscopic_acuity)). Detectable depth difference is
  dz ≈ z²·θ/a, e.g. about 8 cm at 6 m for a = 6.5 cm and θ = 0.5'.
- Motion parallax yields depth with thresholds of similar form to stereo: Rogers & Graham 1982,
  Vision Res. 22 ([DOI](https://doi.org/10.1016/0042-6989(82)90126-2)). I could not access
  the numbers.
- Kellnhofer et al. 2016, *Motion parallax in stereo 3D*, SIGGRAPH Asia
  ([project](https://resources.mpi-inf.mpg.de/StereoParallax)): a joint disparity-parallax
  model.
- Hamill, McDonnell, Dobbyn, O'Sullivan 2005, *Perceptual Evaluation of Impostor
  Representations for Virtual Humans and Buildings*, CGF 24(3)
  ([EG DL](https://diglib.eg.org/handle/10.2312/CGF.v24i3pp623-633)). I could not access its
  findings, so I cite it only as the right prior study.

**Transfer.**
- (a) *Mesh→impostor switch distance in stereo.* For an impostor of depth extent h, flat
  rendering is indistinguishable in stereo when z > √(a·h/θ).
  - For h = 1 m, a = 0.063 m: about 21 m at θ = 30″, about 8.5 m at HMD-pixel-limited
    θ ≈ 3′ (≈20 ppd).
  - The HMD figure is my assumption; sub-pixel disparity from antialiasing may lower it.
  - Use θ from the display's ppd and emit the minimum switch distance per impostor.
- (b) *Monocular motion parallax.* Replace a by the expected camera translation T during a
  glance (a host-supplied parameter): z* = √(T·h/θ_mp). Fast-strafing cameras reveal
  flatness much farther away. This decides **flat billboard vs octahedral with HEIGHT
  parallax**.
- (c) *Octahedral frame count (derivation, uncertain).* Angular frame spacing is roughly
  Δφ ≈ π/N (N = frames_per_side; approximate). Residual misregistration between blended frames
  for content at height h off the capture plane is about h·s·Δφ/2 pixels before height
  reprojection. Requiring this ≤ τ_ghost gives N ≥ π·h·s/(2τ). The HEIGHT reprojection reduces
  the residual to occlusion/disocclusion and height-quantization errors.
  - The rule mainly says *when frames alone cannot suffice* and parallax refinement is
    mandatory.
  - τ_ghost (double-image visibility) is not something I verified. Calibrate with idea 5's
    harness in its 2-frame video mode, as Surace did for pops.
- (d) *Frame texels.* frame_size ≈ projected pixels at switch distance × 1-1.5. Texels beyond
  display Nyquist are wasted. This is standard practice.
- (e) *Mesh↔impostor and frame-transition pops*: reuse idea 4's temporal criterion. Cross-fade
  duration should keep the transient below threshold. Temporal CSF parameters come from
  castleCSF's temporal dimension (not computed here).

**Cost.** Low (formulas plus metadata). **Absorbed?** Engines use heuristics. Psychophysical
switch rules for impostors are not common practice as far as I know.

---

## 10. Low-value or negative results (so nobody spends time on them)

- **Mesh saliency (Lee, Varshney, Jacobs 2005).** It is already absorbed into many simplifiers
  as a weight. Evidence it predicts gaze is weak:
  - Kim et al. 2010, ACM TAP ([PDF](https://www.cs.umd.edu/gvil/papers/kim_TAP10.pdf)): beats
    random and curvature models significantly (18 participants).
  - Lavoué, Cordier, Seo, Larabi 2018, *Visual Attention for Rendered 3D Shapes*, CGF/EG
    ([PDF](https://perso.liris.cnrs.fr/guillaume.lavoue/revue/EG2018.pdf)):
    - a **centre-bias model beat all four mesh-saliency models**;
    - the best methods do not exceed 0.50 linear correlation / 0.76 AUC;
    - attention goes to faces and semantic features, large protrusions, holes, small details on
      smooth areas, and specular highlights;
    - the authors explicitly note observer interest in the silhouette.
  - Recommendation: no geometric saliency term. Keep the user/semantic weight channel (faces),
    the silhouette term (idea 2) and gloss (idea 7).
- **Foveation / crowding** (Rosenholtz's summary-statistic account: Balas, Nakano, Rosenholtz
  2009, J. Vision 9(12):13, [DOI](https://doi.org/10.1167/9.12.13); Tursun et al. 2019
  luminance-contrast-aware foveation, [project](https://www.pdf.inf.usi.ch/projects/AdaptiveFoveation)).
  These are runtime and gaze-dependent, so they do not apply to offline LOD *generation*. The
  one cheap export: a Surace-style lookup table of coarsest-acceptable LOD per
  (distance, eccentricity) as metadata for engines with eye tracking.
- **Absolute mesh-quality metrics** (MSDM2, FMPD, TPDM, Graphics-LPIPS) are good for
  evaluation, but they compare two meshes or two renders. They are not incremental per-collapse
  costs.

---

## Suggested implementation order

1. Idea 2: silhouette face weights. Zero runtime cost, about 100 lines. Test on the 8-asset
   bench with a render-based metric.
2. Idea 5: harness. FLIP port, or ColorVideoVDP as a dev tool, plus the XOR-silhouette split.
3. Idea 1: CSF-weighted normal term replacing the `wi/avg_area` heuristic. `PerceptualView`
   options, `max_jnd` stop.
4. Idea 3: masking map, with the baked-normal coupling first.
5. Idea 4: pop-aware chain plus switch-size metadata.
6. Idea 6: UV √w allocation with Nyquist cap.
7. Ideas 7-9 as validated by the harness.

Uncertain or derived claims are marked inline. The θ/π silhouette probability, the √w texel
rule and the impostor frame-count rule are my derivations from standard geometry and
optimisation, not published results.
