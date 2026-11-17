# Images

| File | Size | Use |
|---|---|---|
| `hero-light.png`, `hero-dark.png` | 2560 × 960 | README banner (switched by `prefers-color-scheme`) |
| `recut-light.png`, `recut-dark.png` | 2560 wide | README: a recut chain sharing one atlas |
| `clusters-light.png`, `clusters-dark.png` | 2560 wide | README: cluster DAG cuts at three distances |
| `splats-light.png`, `splats-dark.png` | 2560 wide | README: the mesh, its surfels and its Gaussians |
| `h2h`, `bench`, `switch`, `atlas`, `impostor` (`-light.png`, `-dark.png`) | 2560 wide | README: the comparisons against meshoptimizer and xatlas, switch distances, impostors |
| `social.png` | 1280 × 640 | GitHub social preview (Settings → General → Social preview) and Open Graph |

The models are real farthing output: the Flight Helmet's LOD chain at a step of 0.25, built by
the browser demo's pipeline (`demo-web/src/pipeline.rs`). LOD1 and LOD2 are reduced and keep the
original materials; LOD3 is a merged, unwrapped proxy with baked base color and normals. The
left half of each model is shaded and the right half shows its wireframe.

The figures below the banner carry no title, lead or footer: the README has that text, so
it can be searched, read by screen readers and translated. Each section there is the figure's
heading (with the italic accent the figures used), its lead paragraph, the image, and the
footer as a `<sub>` caption. When a figure's numbers change, change that text with it.
`h2h`, `bench`, `switch`, `atlas` and `impostor` have no source in the repo; they were cropped
from earlier sheets that had the text drawn in.

## Recut, clusters and splats

These three are built by `demo-web/readme/build.sh` (run it from `demo-web/`; it needs Node and
Chrome, or a Chrome path as its argument). It fetches the samples if needed, runs the
`readme` example (`cargo run --release --features readme --example readme`), which drives the
browser demo's pipeline and renders headless with the PoC engine's viewer and its
`editor_visualizations` overlays into `demo-web/readme/out/`, and then has headless Chrome lay
out the sheets in `demo-web/readme/*.html` and capture them here, light and dark. Each sheet is
as tall as its content. The numbers on the sheets come from the same run (`out/readme.json`);
the README's text around them quotes that run, so update it when they change.

Model: [Flight Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/FlightHelmet)
by Gary Hsu, from the Khronos glTF Sample Assets, CC0 1.0.
