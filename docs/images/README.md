# Images

| File | Size | Use |
|---|---|---|
| `hero-light.png`, `hero-dark.png` | 2560 × 960 | README banner (switched by `prefers-color-scheme`) |
| `recut-light.png`, `recut-dark.png` | 2560 wide | README: a recut chain sharing one atlas |
| `clusters-light.png`, `clusters-dark.png` | 2560 wide | README: cluster DAG cuts at three distances |
| `splats-light.png`, `splats-dark.png` | 2560 wide | README: the mesh, its surfels and its Gaussians |
| `social.png` | 1280 × 640 | GitHub social preview (Settings → General → Social preview) and Open Graph |

The models are real farthing output: the Flight Helmet's LOD chain at a step of 0.25, built by
the browser demo's pipeline (`demo-web/src/pipeline.rs`). LOD1 and LOD2 are reduced and keep the
original materials; LOD3 is a merged, unwrapped proxy with baked base color and normals. The
left half of each model is shaded and the right half shows its wireframe.

## Recut, clusters and splats

These three are built by `demo-web/readme/build.sh` (run it from `demo-web/`; it needs Node and
Chrome, or a Chrome path as its argument). It fetches the samples if needed, runs the
`readme` example (`cargo run --release --features readme --example readme`), which drives the
browser demo's pipeline and renders headless with the PoC engine's viewer and its
`editor_visualizations` overlays into `demo-web/readme/out/`, and then has headless Chrome lay
out the sheets in `demo-web/readme/*.html` and capture them here, light and dark. Each sheet is
as tall as its content. The numbers on the sheets come from the same run (`out/readme.json`).

Model: [Flight Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/FlightHelmet)
by Gary Hsu, from the Khronos glTF Sample Assets, CC0 1.0.
