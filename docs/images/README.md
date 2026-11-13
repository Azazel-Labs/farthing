# Images

| File | Size | Use |
|---|---|---|
| `hero-light.png`, `hero-dark.png` | 2560 × 960 | README banner (switched by `prefers-color-scheme`) |
| `social.png` | 1280 × 640 | GitHub social preview (Settings → General → Social preview) and Open Graph |

The models are real farthing output: the Flight Helmet's LOD chain at a step of 0.25, built by
the browser demo's pipeline (`demo-web/src/pipeline.rs`). LOD1 and LOD2 are reduced and keep the
original materials; LOD3 is a merged, unwrapped proxy with baked base color and normals. The
left half of each model is shaded and the right half shows its wireframe.

Model: [Flight Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/FlightHelmet)
by Gary Hsu, from the Khronos glTF Sample Assets, CC0 1.0.
