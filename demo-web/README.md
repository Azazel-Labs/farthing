# farthing browser demo

Drop a model, adjust the LOD options and watch the chain build and bake. Everything runs
locally in WebAssembly in a Web Worker. Nothing is uploaded.

- **Input:** a `.glb`, a `.gltf` together with its `.bin` and texture files, a folder, or a
  `.zip` of any of these. External files are resolved by relative path, with a fallback to a
  unique file name.
- **Near levels** (LOD1 up to *Bake proxy from*) are reduced mesh by mesh. Each level is built
  from the previous one and keeps the original's UVs and materials.
- **Far levels** merge the scene into one mesh and reduce it. The mesh is then unwrapped
  (LSCM, or planar charts) and the original's base color, normals (including normal maps) and
  roughness/metalness are baked into an atlas.
  The textures appear in the panel as soon as they are baked.
- **Error** is the measured deviation from LOD0: a sampled two-way Hausdorff distance,
  relative to the scene's bounding-box diagonal. *Switch at* is the camera distance where that
  error drops below one pixel (1080p, 60° FOV). **Auto LOD** mode switches levels using those
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
- Baked proxies are static: skins and morph targets are dropped. Reduced levels keep them.
- The wasm build is single-threaded (rayon falls back to the calling thread). A large model
  with a 2048² atlas takes a while.
