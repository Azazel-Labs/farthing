# Plan: long, thin things survive reduction

Goal: chains, cables, ropes, antennas and railings stay visible as a model gets farther away,
instead of thinning to nothing or breaking into dashes. Far levels draw them as opaque rods
with a few texels of color, in the same material and draw call as the rest of the model.
Nothing becomes alpha-tested that wasn't before.

## Where things stand

- **Error is distance, and thin things are always cheap.** The reducer and the part remover
  measure how far the result strays from the original surface. Squashing a tube or deleting a
  chain link never moves any point farther than the part's own thickness, so a 1 cm cable costs
  half a centimeter of error however long it is. `silhouette_weight` scales the distance near
  likely outlines, but it's still a distance. What's really lost is coverage: the pixels the
  part covered, which grow with length times thickness.
- **Measured (phase 0): thin features lose coverage 5 to 15 times faster than the rest of the
  model.** At the level where the whole model is still within about a pixel, a chain has lost a
  third of its pixels. See [Phase 0 results](#phase-0-results).
- **Collapses do it first, part removal later.** Through the first three levels (half, a
  quarter and an eighth of the triangles) the losses are the same with part removal off: links
  and beads are collapsed toward flat slivers, each staying within its own thickness. Part
  removal joins in from the fourth level, deleting face-on links while edge-on ones stay, which
  gives the dashed chain.
- **Edge collapse can't merge parts.** A link is a closed ring that stops at about 14
  triangles and never joins its neighbors, so a 40-link chain can't get below about 560
  triangles by collapses alone. Getting smaller means deleting or replacing it.
- Crease planes (`CREASE_WEIGHT` in `reduce/simplify.rs`) already keep thin closed panels from
  sliding into slivers. Nothing does the same for a tube's cross-section or a ring's hole.

## Shape of the result

```
crates/farthing/src/
  reduce/parts.rs     coverage cost per part
  reduce/simplify.rs  coverage term in collapse costs, so tubes keep their girth
  thin/mod.rs         find line-like assemblies, fit their center line, build rods
  thin/palette.rs     the rods' color cells: a small reserved block of the atlas
```

- **Coverage is a cost alongside distance.** A part's mean projected area over all view
  directions is what it covers; at a level's switch distance that's a pixel count. A part may go
  only when both its distance error and the coverage it takes away fit the level's budget.
  Cauchy's formula (a quarter of the surface area) is exact only for convex bodies, and a ring
  or a run of overlapping links covers much less than its area suggests, so the estimate is a
  quarter of the convex hull's area (an upper bound), or coverage counted from ID renders over a
  few views where that's affordable.
- **Assemblies, not parts.** Parts nearer each other than their own size (links, beads, posts)
  form one assembly that's kept or replaced whole. No dashed gaps. Removing an assembly whole
  inside today's reducer made far levels worse (see [Phase 1 results](#phase-1-results)), so
  assemblies only matter once there's something to replace them with.
- **Line-like assemblies become rods.** Their center line is simplified within the error bound
  and swept with a triangular prism, 6 triangles per segment and no end caps, with smooth vertex
  normals so it shades like a round rod. A 40-link chain of about 4,000 triangles becomes about
  50.
- **The rod matches coverage, not the outline.** A chain fills maybe 50–70% of its outline. A
  rod as wide as the outline would look heavier and darker, so its width is the outline's width
  times the fraction filled. Its projected area then equals the chain's mean coverage. Its
  distance error is at most the outline's radius, inside the budget like any other change.
- **Shading like the links, not like a pipe.** Matching coverage alone doesn't match appearance:
  hair LOD work finds thick strands look "heavy and dry". The links' normals spread far more than
  a rod's smooth normals, so a rod's highlight would be too sharp. Its roughness is widened by
  that spread (Toksvig's normal-map mipmapping, applied to the links' normals), folded into the
  roughness texel.
- **A few texels of color.** The center line is split into cells, like a very coarse line of
  voxels. Each cell gets the visible links' area-weighted mean base color, roughness and metal,
  in one texel. Every vertex of a cross-section maps to the same texel center, and each spine
  vertex to its cell, so bilinear filtering blends cells along the length and never across it.
  The normal map texel is flat, and the vertex normals shade.
- **Mipmaps stay out of the way.** The rod's UVs barely change across it and change slowly
  along it, so the GPU samples the full-size level and a one-texel-high row doesn't blur into
  its neighbors. A row needs one texel of padding: (N+2)×3 texels per rod. If farthing halves
  the atlas for a level, each cell becomes a 2^h block for h halvings. If the engine streams
  textures and the full-size level isn't resident, the rod samples a smaller one, so cells are
  sized for the smallest level the engine keeps, or the palette block is told to it as
  must-keep.
- **Block compression.** Palette cells and their padding align to 4×4 blocks, so a BC or ASTC
  block never mixes a cell with the atlas around it, and each cell's block is a flat color the
  encoders reproduce exactly.
- **Reserved, not scavenged.** Whether something is line-like is pure geometry, so a recut
  finds its assemblies before it packs the atlas and reserves one small palette block (32×32 or
  less, about 0.02% of a 2048² atlas). That's simpler and safer than packing into leftover
  space later.
- **Opaque, always.** Flat lattices (grilles, wire mesh, chain-link fence) have no opaque
  stand-in: one would be a solid plate you can't see through. They keep their geometry longer
  and go whole, under the coverage cost.

Per level, a thin assembly is:

| On screen | Drawn as |
|---|---|
| link or strand detail above about a pixel | its geometry, reduced with the coverage term |
| detail under a pixel, length still many pixels | a rod, when that loses less coverage than removal |
| length under about a pixel | nothing |

## Phase 0 results

`crates/farthing/examples/thin_features.rs` measures this. It finds thin features in each model:

- **Strands:** parts whose median inward ray length (shape diameter) is small next to the length
  their area implies, and whose box can hold that length, so panels don't count.
- **Assemblies:** four or more similar small or slender parts within a quarter of their size of
  each other.

It then builds a chain halving the triangles each level (`lod_chain`, default options) and
renders 10 orthographic ID views at 768² per level. "Lost" is the original's coverage left with
nothing drawn, counted in pixel blocks as large as a pixel where the level would switch in (its
Hausdorff deviation one pixel wide).

The models:

- **Procedural:** a lamp with a 40-link chain, a sagging cable and a string of beads; a railing;
  a grille.
- **The Lantern** sample.
- **Eight CC0 models from Poly Haven** (1k glTF): `caged_hanging_light`, `pull_chain_light_socket`,
  `modular_electric_cables`, `hanging_industrial_lamp`, `lantern_chandelier_01`,
  `modular_chainlink_fence`, `Chandelier_01` and `old_drill_press`.

Coverage lost where each level switches in, for the thin features and for the whole model:

| Model | Thin features | ½ tris: thin / all | ¼: thin / all | ⅛: thin / all (model width) |
|---|---|---|---|---|
| procedural lamp | chain, cable, beads | 5.7% / 1.0% | 14% / 2.7% | 25% / 5.4% (252 px) |
| pull_chain_light_socket | ball chain | 8.5% / 0.5% | 30% / 2.2% | 57% / 4.5% (111 px) |
| caged_hanging_light | two chains, cable | 0.8% / 0.2% | 13% / 1.7% | 30% / 4.3% (104 px) |
| hanging_industrial_lamp | chain, cable | 5.5% / 0.4% | 12% / 1.3% | 30% / 3.9% |
| Chandelier_01 | arms, pendants, wires | 3.1% / 0.6% | 9.8% / 1.9% | 30% / 6.5% (54 px) |
| Lantern | chain, curved bracket | 0.4% / 0.0% | 1.1% / 0.1% | 3.4% / 0.5% (160 px) |
| old_drill_press | cable, levers | 0.4% / 0.1% | 2.2% / 0.6% | 6.2% / 1.5% (65 px) |
| modular_electric_cables | everything is cable | 0.4% / 0.5% | 2.2% / 2.7% | 6.1% / 7.2% |
| modular_chainlink_fence | posts, wire | 0.0% / 0.0% | 0.2% / 0.2% | 0.7% / 0.4% (420 px) |

Notes:

- **What the strips show.**
  - Ball chain: beads shrink, then flatten into wedges, until only the thin wire between them
    is left.
  - Link chains: face-on rings collapse to slivers, then vanish while edge-on links stay.
  - Some rings close into flat plates, which gain coverage inside the ring instead.
- **Part removal off** (`--keep-parts`): the same losses through ⅛, to within a percent.
  Dashes come from part removal at 1/16 and beyond.
- **The ratio is the finding.** Where a model is all thin features (the cable set), they fare
  like everything else. Where thin features hang off a body, they lose coverage several times
  faster.
- **Caveats.**
  - The Hausdorff deviation counts hidden parts. hanging_industrial_lamp's bulb goes at the
    first level, so that model's switch distances are understated.
  - Each pixel is one ray.
  - "Lost" ignores color, and ignores coverage gained elsewhere.

## Phase 1 results

Measured with `thin_features` on the same models. The example now also reports "all added":
coverage a level draws where the original had none, so a level can't hide loss by getting
fatter. Nothing below shipped. The library is unchanged.

**Why the near levels lose thin features.** It's the erosion the error budget allows: about a
pixel at the switch distance, which is a large share of a part only a few pixels wide.
Half-edge collapse makes it worse. Every vertex stays on the original surface, so a coarse tube
is inscribed in the round one and shrinks all along its length. A cost for the area a collapse
takes away doesn't single out thin parts: for a round part it comes to about twice the
deviation whatever the radius. So the collapse coverage cost wasn't built.

**Refit: tried and dropped.** After each level, every vertex moved along its normal to fit the
original surface (least squares), by at most a quarter of its mean edge length. Coverage lost
at full resolution (thin features), and lost and added where the level switches in (whole
model):

| Model, level | Thin lost | All lost | All added | Deviation (% diag) |
|---|---|---|---|---|
| procedural lamp, ¼ | 13.8% → 2.5% | 2.7% → 0.5% | 0.3% → 1.4% | 0.19 → 0.12 |
| pull_chain_light_socket, ¼ | 31.1% → 13.6% | 2.2% → 1.0% | 0.1% → 0.8% | 0.75 → 0.60 |
| caged_hanging_light, ¼ | 13.1% → 3.7% | 1.7% → 0.5% | 0.1% → 1.3% | 0.65 → 0.45 |
| Chandelier_01, ⅛ | 27.4% → 13.7% | 6.5% → 2.2% | 0.6% → 3.1% | 1.86 → 2.05 |
| old_drill_press, ⅛ | 5.6% → 2.0% | 1.5% → 0.5% | 0.5% → 1.5% | 1.53 → 1.15 |
| modular_electric_cables, 1/16 | 13.2% → 5.0% | 16.0% → 6.3% | 1.0% → 12.6% | 1.01 → 0.72 |
| procedural grille, ½ | 2.9% → 0.9% | 0.3% → 0.1% | 0.0% → 0.3% | 0.47 → 0.98 |

- Thin features keep two to five times more coverage, and the deviation is no worse on the real
  models.
- But lost plus added stays about the same: refit mostly turns shrinking into fattening, and
  overshoots on the cables and Chandelier_01 at 1/16 and beyond.
- Boxy models get worse: vertex normals at sharp corners push the corners out.
- Without the per-vertex limit (only the level's error), the deviation tripled on some levels,
  because a level's error includes the parts it removed.
- Refit can't run inside `simplify`: `source_vertex` promises exact copies of input vertices,
  and cluster mode relies on that for shared borders.

**Assemblies removed whole: tried and reverted.** In `Parts::analyze`, parts up to a fifth of the
model's size, within 3× of each other's size and nearer than the smaller one's size were merged
into one removal unit.

- The grouping chained from part to part and took in most of the Lantern. At 1/32 it lost 47%
  of the whole model's coverage instead of 8%, and 98% at 1/64.
- The procedural lamp's 30-link assembly cost too much to remove, so the chain stalled at 828
  triangles against targets of 576, 288 and 144. Getting unstuck, the reducer deleted the
  lattice: 94% of the model lost at 1/32, against 22%.
- Chandelier_01 got worse even at 1/16: 56% of thin coverage lost instead of 45%.
- Where far levels improved (caged_hanging_light, the cables), it was only because they missed
  their triangle targets.
- The cause: with links grouped, the reducer has fewer small, cheap parts to delete when
  collapses stall, so it deletes a big one. Dashes need links merged into something, not
  removed together.

**Next.** The far levels are where thin features go (50–90% lost at 1/32), and that needs
links merged, not deleted. The candidate is Garland and Heckbert's virtual pairs (1997, cited
by the EA '716 examiner): let the reducer contract nearby vertex pairs that share no edge, so
neighboring links fuse and then reduce like one strand, with no fitted shape or proxy.

## Phases

0. **Measure first (gate). Done**: the dashed chain and the flattening are confirmed, above.
1. **Coverage cost and assemblies. Tried; nothing shipped** (see
   [Phase 1 results](#phase-1-results)). In order of what phase 0 found:
   - Collapses first: penalize a collapse by the projected area its part loses, so a tube
     shrinks from 12 sides to 3 instead of flattening into a ribbon, and a ring keeps its hole
     open until it can go whole. This is the near levels' loss.
   - Parts: the coverage cost (convex-hull area / 4, converted to pixels at the level's switch
     distance), checked against a coverage budget alongside `max_error`.
   - Assemblies: parts grouped by gap under their own size, removed whole; their removal error
     is measured against everything outside the assembly. This is the far levels' dashes.
   - `thin_features` shows thin losses near the whole model's at each level; benchmark results
     don't regress (by work counts, not wall-clock).
2. **Rods.**
   - Line-likeness per assembly: PCA of its samples (length ≫ both widths), with the center
     line from part centroids ordered along the chain, or from voxel thinning for single parts.
   - The center line simplified within the error bound; a triangular prism swept along it; width
     from the coverage match; skin weights per spine vertex blended from the nearest links.
   - The reducer gets rods as replacement candidates inside its one greedy queue: where it would
     remove an assembly it inserts the rod instead when that loses less coverage, and counts the
     rod's triangles.
   - **Gated on a patent opinion** (EA '716 under Risks). Phase 1 doesn't depend on it.
   - Try first: virtual-pair contraction (see Phase 1 results, Next), which may get chains to
     rod-like strands without fitting any shape.
3. **Palette texels.**
   - The recut finds line-like assemblies before packing and reserves the palette block; cells
     baked by the exact transfer the atlas already uses (base color, widened roughness, metal,
     flat normal), aligned to 4×4 blocks.
   - Rod UVs at texel centers, with 2^h blocks when the atlas is halved.
   - Checked in the engine at the switch distances: no bleed from neighbors, color matching the
     links within the texture-error measure the chain already reports.
4. **Demo and README.**
   - The demo's chain levels show rods in the seams and wireframe overlays like any other mesh.
   - A README note under the recut section, and phase 0's assets in BENCHMARKS.

## Prior art

- **Coverage and visibility in the error.**
  - Zhang & Turk, *Visibility-Guided Simplification* (Vis 2002).
  - Luebke's dissertation (section 4.2) on keeping silhouettes.
  - Garland notes that a fixed distance tolerance mishandles long thin parts.
- **Keeping area as detail goes.** Cook et al., *Stochastic Simplification of Aggregate Detail*
  (SIGGRAPH 2007), prunes elements and scales up the survivors, with a contrast correction.
- **Thin parts as lines or prisms.**
  - Weber & Penn (1995) and Deussen et al., *Interactive Visualization of Complex Plant
    Ecosystems* (2002), draw thin parts as lines.
  - Zhang et al., *Tree Branch LOD Models* (CGF 2017), merge branches into a prism or a line,
    faded with transparency where farthing stays opaque.
  - Lange & Rodriguez, *LOD+* (2010), tried cylinder and box proxies.
- **Coverage-matched width.**
  - Huang et al., *Real-time LoD Strand-based Rendering* (CGF 2025), sizes thick strands from
    screen coverage, and notes they look heavy and dry.
  - Ward et al. (2003) on strands, clusters and strips.
- **Normal spread into roughness.** Toksvig's normal-map mipmapping; LEAN mapping.
- **Palette texturing** is ordinary practice, and constant UVs across a triangle sample the top
  mip level.
- **Not used here.** Billboard clouds (Simplygon's thin-object answer) and phone-wire AA (widen
  to a pixel, fade by coverage) both need alpha.

## Risks

- **Shimmer.** A rod narrower than a pixel can shimmer or break up without multisampling. The
  original chain does the same at that distance, so this is no worse. farthing doesn't widen it
  to hide that, which would bring back over-coverage.
- **False lines.** A ladder, a cluster of bolts or a row of buttons could pass as a chain. The
  test is length ≫ width with consecutive parts touching or nearly; phase 0's detector already
  needed a tube test (the length an area implies must fit the part's box) to keep thin panels
  out, and phase 2's assets should include near misses.
- **Animated chains.** Skinned or morphing chains need weights on rod vertices that keep the
  rod inside the links' envelope as they move. A first version can skip rods for those and keep
  only the coverage cost.
- **Patents.** Except for EA '716, these summaries come from search snippets, not full claims,
  and finding nothing is not proof that nothing exists. This is not legal advice.
  - **EA US 12,008,716** (*Systems and methods for generating a simplified polygonal mesh*,
    filed 2021-11-19, granted 2024-06-11, active until about 2042-12). Full claims read.
    - Claim 1 (and its twins, 12 and 20) needs all of these steps:
      1. split the mesh into submeshes;
      2. fit shapes and keep those that approximate a submesh within a threshold;
      3. build proxy LODs from those shapes;
      4. build traditionally simplified LODs for every submesh;
      5. make the result by picking one proxy or one simplified LOD per submesh.
    - Only dependent claims add the rest:
      - the knapsack optimization (10);
      - cost and benefit, with fidelity from comparing renders and importance from silhouette
        size (6–9);
      - the list of shapes, including cylinder, capsule, hull, card and voxel (11);
      - root groups by material or part, with compound LODs (3–5).
    - Competing inside one greedy queue avoids those dependent claims, not claim 1. Claim 1 needs
      no global optimizer, and "a set" can be a single LOD.
    - Phase 2 as written would map onto every step of claim 1: test an assembly for
      line-likeness against a threshold, build a rod, and use it where it beats the simplified
      geometry. The spec even describes the same failure ("neither well-formed nor completely
      removed") and an "empty LOD" for removal.
    - Phase 1 is clear of it. A coverage cost, collapses and removing whole assemblies build no
      proxy shapes.
    - Phase 2 needs a patent attorney's opinion before it ships, since farthing is public. Points
      to put to them, none of them certain:
      - **A reduction, not a choice.** The rod would be one more reduction step on the current
        mesh, contracting the assembly's already-reduced geometry onto its spine. It wouldn't be
        a fitted shape picked over a simplified alternative.
      - **Prior art the examiner didn't cite**, from before 2021-11: Zhang et al. 2017 (merging
        branches into a prism or a line) and Lange & Rodriguez 2010 (cylinder and box proxies).
        The examiner did cite Funkhouser & Séquin 1993 and Maciel & Shirley 1995, so the claims
        were allowed over those.
    - If rods ship, score them analytically (convex hull), not by rendering and comparing
      (claims 8 and 9).
  - **EA US 11,354,861 / 12,293,464** (proxy meshes for groups of sub-meshes via voxel
    volumes). Assemblies here group by gap and build no voxel proxy. farthing's existing
    `voxel/` and `aggregate/` modules should be checked against these on their own.
  - **Pixar US 7,973,789** (stochastic pruning, scaling up the survivors; about 2028-11).
    Rods replace a whole assembly. No "drop some links, thicken the rest" until it expires.
  - Lower risk:
    - Activision US 12,134,038: grid-cell proxies.
    - Google US 8,390,623: textured proxy planes.
    - Nvidia US 12,682,558: crossed strips for ray-tracing curves.
  - No patents were found on palette or reserved-atlas texturing, or on coverage costs from
    surface area.
