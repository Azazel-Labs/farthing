# Plan: the PoC engine renders the web demo

Goal: one small wgpu engine draws the browser demo and the README images, so people see
farthing used by a real (if tiny) engine, and the README pictures are exactly what the demo
shows. The demo's three.js viewer goes away once the engine covers what it does.

## Where things stand

- `engine-poc/` is a native, headless wgpu 30 renderer (about 2,000 lines) in its own
  workspace that may only use farthing's public API. It implements `RenderHost` (baking,
  impostors, visibility), renders to `Rgba32Float` targets and reads them back, and has a surfel
  splat pass. It has no canvas output, no glTF PBR material, and nothing for recut or clusters.
- (Before this plan.) The demo builds everything in a worker (farthing compiled to wasm) and draws with three.js:
  PBR with an image-based room light, fat-line wireframes, the UV seams overlay, cluster
  colors, the HLOD cut that follows the camera, splats, `THREE.LOD` switching and the magnifier.

## Shape of the result

```
engine-poc/
  Cargo.toml          features: native (default; headless modes, pollster), web (wasm-bindgen)
  src/lib.rs          the crate's modules
  src/engine.rs       RenderHost and the headless render-and-read-back passes (as today)
  src/viewer.rs       Viewer: textures, materials, geometries, meshes, splat clouds, and views
                      drawn into any target
  src/common.wgsl     the frame and model uniforms every pipeline shares
  src/pbr.wgsl        glTF metallic-roughness with a procedural room light: the demo's look
  src/editor_visualizations/
    mod.rs            which overlays exist, their segments, and their pipelines
    lines.wgsl        wireframe and UV seam edges as screen-space quads of constant pixel width
    splats.wgsl       surfels as opaque discs or sorted Gaussians (the demo's splat shader, moved)
  src/web.rs          the canvas viewer the demo calls (feature web)
  src/main.rs         the native modes, plus `readme`, which writes the README images
```

- **The engine stays tiny.** It draws only what the demo needs and reads nothing itself: the
  worker already holds every level as a farthing `Scene` with decoded textures, so it posts
  typed arrays (positions, normals, tangents, UVs, indices, RGBA8 or block-compressed texels).
  The web engine then needs no glTF, Draco or meshopt code, and only links farthing for
  shared math and types where that adds little.
- **One render core, two outputs.** The same passes draw into the canvas surface or into an
  offscreen texture that is read back (README images, scoring, baking). A canvas gets
  premultiplied sRGB-encoded color, composited from the engine's own sRGB target, because
  blending through an sRGB view of the canvas leaves a light fringe on every antialiased edge.
- **The orbit camera lives in the engine.** It's small, and it shows the whole loop in one place.
  The page forwards pointer and wheel events.
- **WebGPU first.** It's in Chrome, Edge and Safari, and in Firefox on Windows. wgpu's WebGL2
  backend adds about 920 KB gzipped (see phase 0), so it's never in the main module: a second
  build with it is fetched only when the browser has no WebGPU. With neither, the page says so.
- **Block-compressed textures go to the GPU as they are** when the adapter supports BC, ETC2
  or ASTC, which three.js couldn't do from our in-memory cooks. Without that support, the
  decoded texels are uploaded as today.

## Phases

0. **Skeleton and size (gate). Done.**
   - `engine-poc` is a lib + bin. The `native` feature (default) has the headless modes, and
     `web` has the canvas viewer (`src/viewer.rs`, `src/web.rs`).
   - `build-web.sh` writes `demo-web/web/engine/`, and CI builds it with the demo.
   - With `?viewer=engine`, the engine draws into a canvas over three's. What it can't draw yet
     falls back to three.js for that frame, and the camera
     is three's until phase 3.
   - Measured without wasm-opt: **WebGPU only is 354 KB, 81 KB gzipped.** With WebGL2 it's
     3.8 MB, 1.0 MB gzipped, so WebGL2 becomes the separately fetched fallback described above.
1. **Chain parity. Done.**
   - glTF metallic-roughness with all five textures, alpha mask and blend, double-sided, and
     normal maps; the room light approximates three's `RoomEnvironment` procedurally.
   - Wireframe and UV seams overlays, single, compare and auto modes (auto uses three's `LOD`
     pick until the camera moves in), and every texture mode, since the engine draws whatever
     textures the level's material holds. The magnifier crops the engine's canvas.
   - Checked in headless Chrome on BoomBox and Lantern against three.js: the same views, with
     near-identical shading and overlays.
   - WebGPU only, without wasm-opt: **448 KB, 109 KB gzipped**.
   - The panel's textures stay 2D canvases as now.
2. **The other modes. Done.**
   - The viewer splits geometry (vertices, with optional vertex colors) from meshes (an index
     range and a material), so a cluster cut re-sends only its indices. Materials take a flat
     shading flag, and splat clouds draw as discs or as Gaussians the page sorts back to front.
   - Clusters: every color mode, including the cluster and DAG-level colors, which arrive as
     vertex colors, so there is no separate cluster shader. The cut still follows three's
     camera until phase 3.
   - HLOD: proxies, source instances and the tree-level tint. Instanced meshes draw once per
     instance.
   - Splats: discs and sorted Gaussians, in single and compare modes.
   - Presets: the same chain path as phase 1.
   - Checked in headless Chrome on BoomBox against three.js in each mode, and on Lantern again
     for phase 1's views.
   - WebGPU only, without wasm-opt: **467 KB, 115 KB gzipped**.
3. **Switch over. Done.**
   - The engine is the only viewer: three.js and the import map are gone.
   - The orbit camera (`src/orbit.rs`) lives in the engine, damped like three's
     `OrbitControls`. The page forwards pointer and wheel events to it.
   - The page tries the WebGPU build (`web/engine/`), then the WebGL2 one (`web/engine-gl/`,
     `build-web.sh webgl`). With neither, it says so where the drop hint is.
   - The worker posts the farthing `Scene` as typed arrays (positions, normals, UVs, indices,
     material parts, metallic-roughness factors and RGBA8 textures), so the page has no glTF,
     Draco or meshopt code. Downloads still build GLBs on request.
   - CI builds both variants with wasm-opt and fails past 600,000 bytes (WebGPU) or
     4,500,000 bytes (WebGL2).
   - Checked in headless Chrome on BoomBox in every mode, on WebGPU and with WebGPU hidden
     (WebGL2: the same pictures), and on Lantern with Original materials. With neither
     backend, the message shows.
   - Without wasm-opt: **WebGPU 511 KB, 131 KB gzipped; WebGL2 3.9 MB, 1.0 MB gzipped.**
4. **README images. Done.**
   - `demo-web/readme/build.sh` renders, light and dark (a demo-web example rather than a
     `farthing-poc-engine` command, since it drives the demo's pipeline):
     - **recut**: the original next to LOD0–LOD3, with the seams overlay on one half;
     - **clusters**: the DAG cut at three distances in cluster colors;
     - **splats**: the mesh, its surfels and its Gaussians.
   - They're built headless with the same `editor_visualizations` shaders the demo uses, from a
     CC0 sample (the Flight Helmet, as the hero image is), and added to the README next to the
     existing images, with `docs/images/README.md` noting how to rebuild them.

## Risks

- **Size.** wgpu's wasm is the main cost, and WebGL2 adds naga's GLSL output. Phase 0 measures
  it before anything else depends on it.
- **Parity work.** Fat lines, the magnifier and sorted splats each need their own pass. Each is
  small, but there are many of them.
- **Two wasm modules.** The engine and farthing download separately. The engine bar covers
  both, and the viewer can start before the farthing module finishes.
- **The boundary.** The engine keeps using only farthing's public API. If it needs anything
  more, that addition goes into farthing's API on purpose, not through a back door.
