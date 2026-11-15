# farthing browser demo

Drop a model, adjust the LOD options and watch the chain build and bake. Everything runs
locally in WebAssembly in a Web Worker. Nothing is uploaded.

- **Input:** a `.glb`, a `.gltf` together with its `.bin` and texture files, a folder, or a
  `.zip` of any of these. External files are resolved by relative path, with a fallback to a
  unique file name.
- **Textures: Recut skin** (the default) treats the whole scene's skin as one surface to cut
  again: farthing merges the scene, unwraps it from scratch into mip-safe charts, and resamples
  every texel of the original materials (base color, normal map, roughness/metalness) into one
  atlas at the artist's texel density. That is **LOD0**: every triangle of the *Original*, in
  one material, with the waste of the artist's layout gone. LOD1 onward are reduced from it
  step by step and drawn with the same atlas, so nothing is baked per level. The normal map is
  tangent-space and made for LOD0, so far levels shade flatter than a bake for them would
  (bevels and rivets fade as the triangles that carried them go). The panel shows how far the
  resampled base color is from the original, in levels of 255: hard edges blur a little,
  since a fresh layout puts texels at new places. A fresh layout also can't share texels
  between mirrored or repeated parts the way the artist's could, and every chart border is a
  seam the reduction keeps, so many charts put a floor under how coarse the levels go.
  *LOD0 error* (a percentage of the bounding-box diagonal; 0 keeps everything) throws out the
  waste: LOD0 is reduced until it measures about that deviation, and the atlas keeps at most a
  texel per that distance, since texels finer than the geometry keeps don't show. LOD0 then has
  a switch distance of its own, and Auto LOD shows the *Original* closer than that.
- **Textures: Original materials** reduces the near levels (LOD1 up to *Bake proxy from*) mesh
  by mesh. Each level is built from the previous one and keeps the original's UVs and
  materials.
- **Textures: Shared atlas** lays the whole scene out in one atlas instead. With *Keep
  the artist's UV layout* (the default), the artist's UV islands are repacked whole, so the
  bake copies the original texels one to one, and islands that reuse the same texels (mirrored
  halves, repeated parts) share them. Unticked, farthing re-unwraps the scene and chooses the
  chart cuts. The original materials are baked into that one atlas by exact transfer, at the
  artist's texel density (the densest texture on each surface), and every near level is
  reduced from that mesh. Charts sit in whole 2ᵏ-texel cells, so mipmaps never
  mix two charts. *Atlas up to* caps the side (for the recut too) (2048 at most: a larger bake doesn't fit
  in a browser's WebAssembly memory). When the artist's density doesn't fit the cap, every
  chart shrinks by the same factor, which the panel shows. Every shared level's download
  carries the atlas, but the page sends it once, with LOD1, and draws the later shared levels
  with it. That one atlas is what the *Textures* column (GPU memory) counts.
- **Far levels** merge the scene into one mesh and reduce it. The mesh is then unwrapped
  (LSCM, or planar charts) and the original's base color, normals (including normal maps) and
  roughness/metalness are baked into an atlas.
  The textures appear in the panel as soon as they are baked.
- **Viewing budget** is for models that are only ever seen from so close: a rock that never
  comes nearer than a metre doesn't need detail finer than a pixel there. Set *Closest view*
  (in the model's units: metres for glTF), the camera's field of view and the screen height,
  and the chain keeps nothing that view can't show. LOD1 is reduced until it deviates about a
  pixel at that distance (or by one step, if that goes further), and the later levels step
  from it. The recut and shared atlases are capped at a texel per pixel there (the shared
  one re-unwrapped, when that cap is below the artist's resolution); a recut's LOD0 is reduced
  to about a pixel there too, and each baked atlas is
  sized for the distance its level first appears at. The panel shows how big the model is,
  what a pixel is at that distance, and how much of the artist's texture detail the view can
  ever see.
- **Error** is the measured deviation from the original (*Original* in a recut chain): a sampled two-way Hausdorff distance,
  relative to the scene's bounding-box diagonal. *Switch at* is the camera distance where that
  error drops below one pixel, for the viewing budget's field of view and screen (1080p,
  60° unless you change them). **Auto LOD** mode switches levels using those
  distances, scaled to your window. Zoom out, or drag the distance slider (its ticks mark the
  switches), to watch it change. Auto mode orbits the model's center, and once the model is
  small a magnifier shows its rendered pixels up close.
- **Reset view** (or F, or double-click) re-frames the model. Panning can't drift away from it.
- **Compare** shows the original next to the selected level. Each level downloads as a
  self-contained `.glb`.
- Changing an option restarts the run. A wasm call can't be interrupted, so the page replaces
  the worker.

## Examples

The *Example* picker runs the other ways farthing builds LODs on the same model. The table
gains a *Size* column (GLB, PLY or packed GPU bytes).

- **Classic LOD chain.** Everything described above.
- **Option examples.** Ten rows, each one the same reduction with one `SimplifyOptions`
  choice changed: library defaults, strict seams, fast quality, preserve symmetry, lock
  borders, classic QEM ranking, no silhouette weight, keep all parts, geometry only
  (`repair::geometry_only` first), and a pixel-error bound (`screen_size_to_max_error`).
  farthing has no per-engine presets. These are the option differences an engine integration
  would choose between. Hover a row's name to see the exact option.
- **Cluster DAG.** `build_cluster_dag` per mesh (Nanite or meshlet cluster limits, optionally
  `ClusterOptions::fast()`), packed with `GpuClusterDag`. The *Cut* row is the view-dependent
  selection at the current camera: every cluster is drawn or replaced by its coarser parent
  depending on its own error on screen, so moving the camera changes it. The worker runs both
  the CPU `LodView` cut and the GPU-format `select` and reports whether they agree. You can
  color the cut by material, by cluster or by DAG level, set the pixel threshold, and freeze
  the cut so you can inspect it from another angle.
- **HLOD cells.** The model is tiled on a grid (1, 3×3 or 5×5 copies) so there are cells to
  merge. `build_hlod_tree_with` gives each cell, and each group of cells above it, a merged
  proxy, which is reduced and can have its textures baked. *Full* draws every copy. *Cut* is
  `HlodTree::select` at the current camera: far cells are drawn as their proxy and near cells
  as the source instances. You can tint proxies by tree level. The bake panel shows the root
  proxy's atlas. The panel also says how far away the nearest leaf proxy takes over. Node
  errors add up the tree (a parent's error includes its children's), so on a dense model with
  thin parts this can be far away. Zoom out further or raise *Max error*.
- **Gaussian splats.** `progressive_surfel_cloud` → `surfels_to_splats`. The full cloud plus
  its first quarter and first sixteenth (a progressive cloud's prefixes are its LODs), each
  shaped from its surface footprint. They are drawn as screen-space Gaussians, blended
  back to front, or as opaque discs. Each level downloads as a 3DGS-layout `.ply`.

## Build and run

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129   # must match the pin in Cargo.toml
./build.sh                                         # → web/pkg/
python fetch_samples.py                            # optional: sample models → web/samples/
python -m http.server -d web 8000                  # then open http://localhost:8000
```

Any static file server works if it serves `.js` as JavaScript. Opening `index.html` from disk
(`file://`) doesn't work, because module workers and wasm need http. If
[binaryen](https://github.com/WebAssembly/binaryen)'s `wasm-opt` is on `PATH`, `build.sh`
also optimizes the module for size.

**Sample models.** `fetch_samples.py` downloads a few CC0 models from the
[Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), pinned to
one commit, into `web/samples/` with an `index.json`, and each model's screenshot from that
repo as its thumbnail. The page shows them as cards in the sidebar. They
aren't committed: CI fetches them when it builds the page, and `cargo test` runs the pipeline
on each one when they're present. The script refuses any model whose metadata lists a license
other than CC0. To change the set, edit `SAMPLES` in the script.

`index.html?model=<url>` loads a model on start, for example `?model=samples/Lantern.glb`.

The pipeline itself (`src/pipeline.rs`) has no browser dependencies. `cargo test` runs it
natively on a zipped `.gltf`.

## Limits

- The bake reads base color (texture × factor × vertex color), normals (mesh normals plus
  source normal maps) and roughness/metalness (texture × factor). Occlusion, emissive and
  other extensions are not baked.
- Base color textures that use a second UV set are exported on the first set.
- Baked proxies, recut and shared-atlas levels are static: skins and morph targets are dropped.
  Reduced levels keep them.
- The wasm build is single-threaded (rayon falls back to the calling thread). A large model
  with a 2048² atlas takes a while.
