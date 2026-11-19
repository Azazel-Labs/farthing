// farthing LOD demo: page logic. The wasm runs in worker.js; this file handles input, the
// options, the viewer (farthing's PoC engine, engine-poc), the level table and the example
// modes (cluster cuts, HLOD cuts, splats).

const $ = (s) => document.querySelector(s);
const form = $("#opts");
const statusEl = $("#status");
const rowsEl = $("#level-rows");

const STORE_KEY = "farthing-demo-options";
/** Saved options from before version 2 kept Original materials only as the old default: they get the recut. */
const STORE_VERSION = 2;

// ---------------------------------------------------------------- state

let files = null; // [{ name, bytes: ArrayBuffer }], kept so a respawned worker can reload
let assetInfo = null;
let original = null; // { model, center: [x, y, z], radius }
let worker = null;
let busy = false; // the worker is loading or running
let runId = 0;
let runStart = 0;
let runOpts = null; // options of the current run
let selected = 0;
let texPinned = false; // the texture panel stays on the selected level, which the user picked
let texShown = -1; // the level the texture panel was last asked for
let texMaterial = -1; // the original's material slot the panel shows, when it has several
let mode = "single";
let wireOn = false;
let seamOn = false;
/** levels[n] = { info, model (anything with draws() and dispose()), tex: { channel: ImageData }, stale } */
let levels = [];
let pendingDownload = null;
let stoppedAt = null; // the last chain level when the run stopped short at the error limit

// ---------------------------------------------------------------- viewer
//
// farthing's PoC engine draws the view: engine-poc built for the web, on WebGPU, or on WebGL2
// (the engine-gl build) where there is no WebGPU. Its shaders are the ones in its
// editor_visualizations folder, which also draw the README's images. The page hands it the
// worker's scenes as vertices, materials and textures, and forwards the pointer to its orbit
// camera; it has no 3D code of its own.

const view = $("#view");
let canvas = $("#gl");
let gpu = null; // the engine's EngineView
let orbit = null; // its orbit camera
const dpr = () => Math.min(devicePixelRatio, 2);
const EMPTY = new Float32Array(0);

const viewerReady = (async () => {
  const builds = navigator.gpu ? ["engine", "engine-gl"] : ["engine-gl"];
  for (const dir of builds) {
    try {
      const mod = await import(`./${dir}/farthing_poc_engine.js`);
      await mod.default();
      gpu = await mod.createView(canvas);
      orbit = new mod.Orbit();
      listen(canvas);
      console.info(`viewer: farthing's engine on ${gpu.backend}`);
      return;
    } catch (e) {
      console.warn(`viewer: the ${dir} build could not start:`, e);
      // A canvas keeps the first kind of context asked of it: the next build gets a fresh one.
      const fresh = canvas.cloneNode(false);
      canvas.replaceWith(fresh);
      canvas = fresh;
    }
  }
  throw new Error("This browser has neither WebGPU nor WebGL2, so the viewer can't draw.");
})();
viewerReady.catch((e) => {
  const p = $("#no-gpu");
  p.textContent = e.message;
  p.hidden = false;
  setStatus(e.message, "error");
});

/** Forwards the pointer and wheel to the orbit camera. */
function listen(c) {
  c.addEventListener("pointerdown", (e) => {
    c.setPointerCapture(e.pointerId);
    orbit.pointerDown(e.pointerId, e.clientX, e.clientY, e.button, e.shiftKey || e.ctrlKey || e.metaKey);
  });
  c.addEventListener("pointermove", (e) => orbit.pointerMove(e.pointerId, e.clientX, e.clientY));
  const up = (e) => orbit.pointerUp(e.pointerId);
  c.addEventListener("pointerup", up);
  c.addEventListener("pointercancel", up);
  c.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      orbit.wheel(e.deltaY, e.deltaMode, e.ctrlKey);
    },
    { passive: false },
  );
  c.addEventListener("contextmenu", (e) => e.preventDefault());
  c.addEventListener("dblclick", () => home());
}

/** Tells the camera which sphere to keep in reach and fit its planes around. */
function syncFocus() {
  const f = focus();
  if (f && orbit) orbit.setFocus(f.center[0], f.center[1], f.center[2], f.radius);
}

/** Points the camera at the model from the default angle (also: F, double-click, Reset view). */
function home() {
  if (!orbit || !focus()) return;
  syncFocus();
  orbit.home();
  // A focus may ask to start closer than the distance that frames all of it.
  if (focus().distance) orbit.setDistance(focus().distance);
}

// Line overlays: colors are sRGB like CSS, the engine wants linear ones; widths are CSS pixels.
const dark = matchMedia("(prefers-color-scheme: dark)");
const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linear = (hex, a) => new Float32Array([(hex >> 16) & 255, (hex >> 8) & 255, hex & 255].map((v) => toLinear(v / 255)).concat(a));
function lineStyles() {
  const k = dpr();
  gpu.setLineStyle(0, dark.matches ? linear(0x8fbcff, 0.75) : linear(0x1453c4, 0.7), 1.25 * k);
  gpu.setLineStyle(1, dark.matches ? linear(0xffa94d, 1) : linear(0xe8590c, 1), 2.5 * k);
}

/** HSL (all 0-1) to RGB, taken as linear values, as three.js's setHSL gave them. */
function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}
/** A color per DAG or tree level. */
const levelColor = (l) => hsl((0.58 + l * 0.137) % 1, 0.62, 0.55);
/** A stable random color per cluster. */
function clusterColor(id) {
  let h = Math.imul(id ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return hsl((h >>> 0) / 4294967296, 0.5 + ((h >>> 8) & 63) / 250, 0.45 + ((h >>> 16) & 63) / 400);
}

/** A plain material: `rgb` linear, matte unless said otherwise. */
function colorMaterial(rgb, roughness = 0.8) {
  return gpu.addMaterial(new Float32Array([...rgb, 1, 0, roughness, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]), Int32Array.of(-1, -1, -1, -1, -1));
}
let fallbackMaterial = -1;
/** For parts whose material is gone or never came. */
const fallback = () => (fallbackMaterial >= 0 ? fallbackMaterial : (fallbackMaterial = colorMaterial([0.8, 0.8, 0.8])));

/** A payload material (glTF metallic-roughness) in the engine; texture indices into `textures`. */
function payloadMaterial(m, textures) {
  const tex = (i) => (i >= 0 && textures[i] ? textures[i].id : -1);
  const mask = m.cutoff != null;
  // Payloads don't say which materials are double-sided: draw both sides of everything.
  const factors = [...m.base_color, m.metallic, m.roughness, 1, 1, 1, 0, 0, 0, mask ? 1 : 0, m.cutoff ?? 0, 1, 0];
  const ids = [tex(m.base_color_texture), tex(m.metallic_roughness_texture), tex(m.normal_texture), -1, -1];
  return gpu.addMaterial(new Float32Array(factors), Int32Array.from(ids));
}

const IDENTITY = Float32Array.of(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

/**
 * A scene from the worker (the original, a level, an HLOD proxy) in the engine: its textures,
 * materials, vertices and one engine mesh per material part. Scenes that come without their
 * own materials (reduced levels, shared levels after the one carrying the atlas) get theirs
 * from `borrow(slot)`. `draws(tint)` lists what to draw: [model matrix, engine mesh, overlays];
 * a tint material id draws every part with it instead.
 */
function buildModel(scene, borrow = null) {
  const b = scene.buffers;
  const textures = borrow
    ? []
    : scene.textures.map((t) => ({
        id: gpu.addTexture(t.width, t.height, b[t.rgba], t.srgb),
        width: t.width,
        height: t.height,
        rgba: b[t.rgba],
        canvas: null,
      }));
  const materials = borrow ? [] : scene.materials.map((m) => payloadMaterial(m, textures));
  const material = (slot) => (borrow ? borrow(slot) : materials[slot]) ?? fallback();
  const meshes = scene.meshes.map((m) => {
    const positions = b[m.positions];
    const uvs = m.uvs >= 0 ? b[m.uvs] : null;
    const indices = b[m.indices];
    const geometry = gpu.addGeometry(positions, b[m.normals], uvs ?? EMPTY, m.colors >= 0 ? b[m.colors] : EMPTY);
    const parts = m.parts.map(([start, count, slot]) => ({
      start,
      count,
      slot,
      mesh: gpu.addMesh(geometry, indices.subarray(start, start + count), material(slot)),
    }));
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3)
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], positions[i + a]);
        hi[a] = Math.max(hi[a], positions[i + a]);
      }
    return { geometry, parts, uvs, indices, box: [lo, hi], tinted: new Map() };
  });
  const instances = scene.instances.map((i) => ({ mesh: i.mesh, matrix: Float32Array.from(i.matrix) }));
  return {
    meshes,
    instances,
    materials,
    textures,
    json: scene.materials,
    draws(tint = null, out = [], offset = null) {
      for (const inst of instances) {
        const m = meshes[inst.mesh];
        let matrix = inst.matrix;
        if (offset) {
          matrix = matrix.slice();
          matrix[12] += offset[0];
          matrix[13] += offset[1];
          matrix[14] += offset[2];
        }
        if (tint === null) for (const p of m.parts) out.push([matrix, p.mesh, true]);
        else {
          let ids = m.tinted.get(tint);
          if (!ids) m.tinted.set(tint, (ids = m.parts.map((p) => gpu.addMesh(m.geometry, m.indices.subarray(p.start, p.start + p.count), tint))));
          for (const id of ids) out.push([matrix, id, true]);
        }
      }
      return out;
    },
    dispose() {
      for (const m of meshes) {
        for (const p of m.parts) gpu.removeMesh(p.mesh);
        for (const ids of m.tinted.values()) for (const id of ids) gpu.removeMesh(id);
        gpu.removeGeometry(m.geometry);
      }
      for (const id of materials) gpu.removeMaterial(id);
      for (const t of textures) gpu.removeTexture(t.id);
    },
  };
}

/** A model's world bounds: center and bounding-sphere radius. */
function bounds(model) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const inst of model.instances) {
    const [a, b] = model.meshes[inst.mesh].box;
    if (!(a[0] <= b[0])) continue;
    const e = inst.matrix;
    for (let c = 0; c < 8; c++) {
      const p = [c & 1 ? b[0] : a[0], c & 2 ? b[1] : a[1], c & 4 ? b[2] : a[2]];
      for (let k = 0; k < 3; k++) {
        const v = e[k] * p[0] + e[4 + k] * p[1] + e[8 + k] * p[2] + e[12 + k];
        lo[k] = Math.min(lo[k], v);
        hi[k] = Math.max(hi[k], v);
      }
    }
  }
  if (!(lo[0] <= hi[0])) return { center: [0, 0, 0], radius: 1 };
  const center = lo.map((v, k) => (v + hi[k]) / 2);
  return { center, radius: Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 1e-6) };
}

/** A payload texture as a canvas for the texture panel, made the first time it is shown. */
function drawable(t) {
  if (!t.canvas) {
    t.canvas = new OffscreenCanvas(t.width, t.height);
    const rgba = new Uint8ClampedArray(t.rgba.buffer, t.rgba.byteOffset, t.rgba.byteLength);
    t.canvas.getContext("2d").putImageData(new ImageData(rgba, t.width, t.height), 0, 0);
  }
  return t.canvas;
}

function showOriginal(scene) {
  const model = buildModel(scene);
  original = { model, ...bounds(model) };
  orbit.maxDistance = original.radius * 400;
  home();
  levels[0] = { info: null, model, tex: {}, stale: false };
}

function resetViewer() {
  extra?.dispose();
  for (const [i, l] of levels.entries()) if (i > 0) disposeLevel(l);
  dropIncoming();
  extra = null;
  focusOverride = null;
  shownMode = "";
  original?.model.dispose();
  levels = [];
  original = null;
  selected = 0;
  texPinned = false;
  texMaterial = -1;
  syncCutPanel();
}

/**
 * Auto mode's levels: each ready level and the camera distance from the model's center it
 * switches in at. That is the level's 1-pixel switch distance: the bounding radius plus a part
 * for the run's view (info.ppr pixels per radian), rescaled from there to this canvas.
 */
function autoLevels() {
  const ppr = canvas.height / (2 * Math.tan((orbit.fov * Math.PI) / 360));
  const r = diag() / 2;
  let d = 0;
  const out = [];
  for (const l of levels) {
    if (!l?.model || !l.info || l.stale) continue;
    const s = l.info.switch_distance_1px;
    d = Math.max(d, s > 0 ? r + ((s - r) * ppr) / (l.info.ppr || REF_PPR) : 0);
    out.push({ level: l.info.level, distance: d, model: l.model });
  }
  return out;
}

let autoKey = "";
/** Room to zoom out past the last switch, wherever it is, and the slider's ticks. */
function syncAuto(list) {
  const key = list.map((l) => `${l.level}@${l.distance}`).join() + original.radius;
  if (key === autoKey) return;
  autoKey = key;
  orbit.maxDistance = Math.max(original.radius * 400, (list.at(-1)?.distance ?? 0) * 2);
  drawTicks(list);
}

// Auto mode's distance slider: logarithmic, from close up to the zoom limit.
const distEl = $("#dist");
function distRange() {
  return [orbit.homeDistance(original.radius) * 0.5, orbit.maxDistance];
}
function distToSlider(d) {
  const [lo, hi] = distRange();
  return (1000 * Math.log(Math.max(d, lo) / lo)) / Math.log(hi / lo);
}
function drawTicks(list) {
  $("#ticks").innerHTML = list
    .slice(1)
    .map((l) => {
      const t = Math.min(distToSlider(l.distance) / 1000, 1);
      return `<i style="--t:${t}" title="${esc(lodName(l.level))} from ${l.distance.toPrecision(3)}"></i>`;
    })
    .join("");
}
distEl.addEventListener("input", () => {
  if (!original || !orbit) return;
  const [lo, hi] = distRange();
  orbit.setDistance(lo * (hi / lo) ** (+distEl.value / 1000));
});

// Auto mode's magnifier: far away the model is a few pixels, so this crops the frame just drawn
// around it and scales it up without smoothing, to show the rendered pixels themselves.
const magEl = $("#mag");
const magCanvas = magEl.querySelector("canvas");
const magCtx = magCanvas.getContext("2d");
function magnify() {
  const size = Math.round(160 * dpr());
  const [bw, bh] = [canvas.width, canvas.height];
  const v = orbit.viewMatrix;
  const c = original.center;
  const p = [0, 1, 2].map((k) => v[k] * c[0] + v[4 + k] * c[1] + v[8 + k] * c[2] + v[12 + k]);
  const f = 1 / Math.tan((orbit.fov * Math.PI) / 360);
  const depth = -p[2];
  const x = depth > 0 ? (f * bh * p[0]) / (bw * depth) : 9;
  const y = depth > 0 ? (f * p[1]) / depth : 9;
  const rpx = depth > 0 ? ((original.radius * f) / Math.hypot(...p)) * (bh / 2) : 0;
  const zoom = Math.floor(size / Math.max(2 * rpx * 1.15, 4));
  magEl.hidden = zoom < 2 || Math.abs(x) > 1.2 || Math.abs(y) > 1.2;
  if (magEl.hidden) return;
  if (magCanvas.width !== size) magCanvas.width = magCanvas.height = size;
  const src = size / zoom;
  magCtx.imageSmoothingEnabled = false;
  magCtx.clearRect(0, 0, size, size);
  magCtx.drawImage(canvas, ((x + 1) / 2) * bw - src / 2, ((1 - y) / 2) * bh - src / 2, src, src, 0, 0, size, size);
  magEl.querySelector("figcaption").textContent = `${zoom}× · rendered pixels`;
}

function displayed(level) {
  return levels[level]?.model ? level : 0;
}

let hudText = "";
function hud(left, right = "") {
  const t = left + "|" + right;
  if (t === hudText) return;
  hudText = t;
  $("#hud").innerHTML = (left ? `<span>${left}</span>` : "<i></i>") + (right ? `<span>${right}</span>` : "");
}

/** A level's name: its label (a recut's levels count from LOD0, the recut), else LODn. */
const lodName = (level) => levels[level]?.info?.label ?? `LOD${level}`;

function label(level) {
  const l = levels[level];
  if (!l?.info) return `LOD${level}`;
  const i = l.info;
  const stale = l.stale ? " (updating)" : "";
  if (i.kind === "splats") return `LOD${level} · ${fmtInt(i.vertices)} Gaussian splats${stale}`;
  const name = esc(lodName(level));
  const more = i.hud ? ` · ${esc(i.hud)}` : "";
  return `${name} · ${KIND_TEXT[i.kind] ?? i.kind} · ${fmtInt(i.triangles)} tris${more}${stale}`;
}

/** Queues `model`'s draws as a view across device pixels x0 to x1. */
function drawView(model, x0, x1, camera) {
  const draws = model?.draws() ?? [];
  const models = new Float32Array(draws.length * 16);
  draws.forEach(([m], i) => models.set(m, i * 16));
  const ids = Uint32Array.from(draws, (d) => d[1]);
  const overlays = Uint8Array.from(draws, (d) => (d[2] ? 1 : 0));
  gpu.view(x0, 0, x1 - x0, canvas.height, ids, models, overlays, ...camera);
}

let cleared = false;
function render() {
  requestAnimationFrame(render);
  if (!gpu) return;
  const [w, h] = [view.clientWidth, view.clientHeight];
  const k = dpr();
  const [bw, bh] = [Math.max(1, Math.round(w * k)), Math.max(1, Math.round(h * k))];
  if (canvas.width !== bw || canvas.height !== bh) [canvas.width, canvas.height] = [bw, bh];
  // The GPU cluster cut can split the view with a second camera.
  const second = original && mode !== "compare" && mode !== "auto" && displayed(selected) === 1 ? extra?.second?.() : null;
  $("#divider").hidden = !(mode === "compare" || second) || !original;
  if (!original) {
    hud("");
    magEl.hidden = true;
    if (!cleared) {
      gpu.beginFrame(bw, bh);
      gpu.endFrame();
      cleared = true;
    }
    return;
  }
  cleared = false;
  syncFocus();
  orbit.update(h);
  requestCut();
  sortSplats();
  lineStyles();
  gpu.setOverlays(wireOn, seamOn);
  gpu.setSplatShape($("#splat-draw").value === "gauss");
  gpu.beginFrame(bw, bh);
  const camera = [orbit.viewMatrix, orbit.fov, orbit.near, orbit.far];
  if (mode === "compare") {
    const half = Math.round(Math.floor(w / 2) * k);
    drawView(levels[displayed(0)].model, 0, half, camera);
    drawView(levels[displayed(selected)].model, half, bw, camera);
    gpu.endFrame();
    magEl.hidden = true;
    hud(label(0), label(selected));
  } else if (mode === "auto") {
    const list = autoLevels();
    syncAuto(list);
    const dist = orbit.distance;
    const i = Math.max(0, list.findLastIndex((l) => dist >= l.distance));
    drawView(list[i]?.model, 0, bw, camera);
    gpu.endFrame();
    magnify();
    const cur = list[i]?.level ?? 0;
    const next = list[i + 1];
    if (document.activeElement !== distEl) distEl.value = String(distToSlider(dist));
    const ahead = next ? ` · ${lodName(next.level)} at ${next.distance.toPrecision(3)}` : "";
    hud(`Auto: ${label(cur)}`, `distance ${dist.toPrecision(3)}${ahead}`);
  } else {
    const shown = displayed(selected);
    if (second) {
      const half = Math.round(Math.floor(w / 2) * k);
      drawView(levels[shown].model, 0, half, camera);
      drawView(levels[shown].model, half, bw, second);
    } else drawView(levels[shown].model, 0, bw, camera);
    gpu.endFrame();
    magEl.hidden = true;
    hud(label(shown), second ? "Second camera, circling" : (shown === 1 && extra?.hud?.()) || "");
  }
}

const divider = document.createElement("div");
divider.id = "divider";
divider.hidden = true;
divider.style.cssText =
  "position:absolute;top:0;bottom:0;left:50%;width:1px;background:var(--line);pointer-events:none";
view.append(divider);
requestAnimationFrame(render);

// ---------------------------------------------------------------- example modes
//
// Besides the classic chain, the page runs five other examples (the Example picker):
// option configurations (levels, like the chain), a cluster DAG and an HLOD tree (one "Cut"
// level the worker re-selects as the camera moves), Gaussian splats (drawn here) and a GPU
// crowd of a skinned model (animated here, drawn by the engine).

const MODE_KIND = { presets: "reduced", clusters: "clusters", hlod: "hlod", splats: "splats", crowd: "crowd" };
const MODE_LEVELS = { presets: 10, clusters: 1, hlod: 1, splats: 3, crowd: 1 };
const KIND_TEXT = { clusters: "cluster DAG", hlod: "HLOD", crowd: "GPU crowd" };
const MODE_HINTS = {
  chain: "Reduced levels, then baked proxies, each built from the last. Auto LOD switches them by distance.",
  presets:
    "The same reduction under named option configurations. Pick a row to view it, or Compare it with the original.",
  clusters:
    "A Nanite-style cluster DAG per mesh. Each cluster is kept or swapped for a coarser one by its own error on screen: zoom in and out and watch the cut follow.",
  hlod: "The model tiled on a grid and split into cells; each cell, and each group of cells above it, gets a merged proxy. Far cells draw their proxy instead.",
  splats: "A progressive surfel cloud as Gaussian splats: the whole cloud, its first quarter and its first sixteenth.",
  crowd:
    "Thousands of a skinned, animated model (try the Fox) drawn by the GPU: one pose palette, a level per unit by its size on screen (all bones, fewer bones, a baked vertex animation), and the page writes a unit only when it changes clip.",
};
const AUTO_MODES = new Set(["chain", "splats"]);

let shownMode = ""; // the example whose results are on screen ("" before the first run)
/** Cluster or HLOD state: { kind, model, info, cut(sel, buffers), stats(), dispose(), full? } */
let extra = null;
let hlodIncoming = null; // HLOD proxies built while the tree builds
let focusOverride = null; // HLOD: the tiled grid's bounds replace the model's for the camera
let cutSeq = 0;
let cutInflight = false;
let cutKey = "";

function focus() {
  return focusOverride ?? original;
}

function fmtBytes(n) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)} MB` : `${(n / 1e3).toFixed(n < 1e4 ? 1 : 0)} KB`;
}
const fmtPx = (p) => (p < 1 ? p.toFixed(2) : p.toPrecision(2));
const diag = () => assetInfo?.diagonal || 1;
/** Pixels per radian of the reference view (1080 pixels high, 60° vertical FOV). */
const REF_PPR = 1080 / (2 * Math.tan(Math.PI / 6));
/** Pixels per radian of this canvas. */
const canvasPpr = () => canvas.height / (2 * Math.tan((orbit.fov * Math.PI) / 360));

function disposeLevel(l) {
  if (!l?.model) return;
  if (l.model !== original?.model) l.model.dispose();
  l.model = null;
}

function dropIncoming() {
  for (const p of hlodIncoming?.proxies.values() ?? []) p.model.dispose();
  hlodIncoming = null;
}

/** Drops everything a mode put on screen; level 0 is the original again. */
function clearModeState() {
  for (let i = levels.length - 1; i > 0; i--) disposeLevel(levels[i]);
  levels.length = Math.min(levels.length, 1);
  extra?.dispose();
  extra = null;
  dropIncoming();
  if (levels[0] && original) levels[0].model = original.model;
  if (focusOverride) {
    focusOverride = null;
    if (original) {
      orbit.maxDistance = original.radius * 400;
      home();
    }
  }
  cutKey = "";
  autoKey = "";
}

function switchMode(m) {
  clearModeState();
  shownMode = m;
  selected = m === "chain" ? 0 : 1;
  if (!AUTO_MODES.has(m) && mode === "auto") setMode("single");
  syncCutPanel();
  drawTextures(-1);
}

function syncCutPanel() {
  const m = shownMode;
  $("#cut").hidden = !(m === "clusters" || m === "hlod" || m === "splats" || m === "crowd");
  for (const el of document.querySelectorAll("[data-cut]")) el.hidden = !el.dataset.cut.split(" ").includes(m);
  // Streaming and culling controls: only the GPU-driven cluster cut has them.
  for (const el of document.querySelectorAll("[data-gpu]")) el.hidden ||= !extra?.gpu;
  $("#upload-out").value = $("#upload").value;
  $("#cut-title").textContent = { clusters: "Cluster DAG cut", hlod: "HLOD cut", splats: "Splats", crowd: "GPU crowd" }[m] ?? "";
  $("#threshold-out").value = `${fmtPx(2 ** +$("#threshold").value)} px`;
  $("#bake").hidden = !!m && !(m === "chain" || m === "hlod");
  $('[data-mode="auto"]').hidden = !!m && !AUTO_MODES.has(m);
  cutStats();
}

function cutStats() {
  const rows = extra?.stats() ?? (shownMode === "splats" ? splatStats() : []);
  $("#cut-stats").innerHTML = rows
    .map(([k, v, cls]) => `<dt>${esc(k)}</dt><dd class="${cls ?? ""}">${esc(v)}</dd>`)
    .join("");
}

/** Asks the worker for the cut at the current camera, when it changed and none is pending. */
function requestCut() {
  if (extra?.live) return extra.live();
  if (!extra || busy || cutInflight || extra.failed || $("#freeze").checked) return;
  const ppr = canvasPpr();
  const threshold = 2 ** +$("#threshold").value;
  const p = orbit.eye;
  const key = `${p[0].toPrecision(5)},${p[1].toPrecision(5)},${p[2].toPrecision(5)}|${ppr.toFixed(0)}|${threshold}`;
  if (key === cutKey) return;
  cutKey = key;
  cutInflight = true;
  worker.postMessage({ type: "select", seq: ++cutSeq, view: { eye: [...p], ppr, threshold } });
}

function onSelect(m) {
  cutInflight = false;
  if (!extra || m.seq < extra.minSeq) return; // asked for the previous result
  if (m.error) {
    extra.failed = true;
    return setStatus(`Cut: ${m.error}`, "error");
  }
  extra.cut(m.info, m.buffers);
  updateRow(1);
  cutStats();
}

/** Puts a cluster or HLOD result on screen as level 1 (and, for HLOD, level 0). */
function installExtra(x) {
  disposeLevel(levels[1]);
  extra?.dispose();
  extra = x;
  x.minSeq = cutSeq + 1;
  const l1 = entry(1);
  Object.assign(l1, { info: x.info, model: x.model, stale: false, tex: {} });
  if (x.full) {
    const l0 = entry(0);
    l0.model = x.full;
    l0.info = { ...(l0.info ?? { level: 0, kind: "original" }), ...x.fullInfo };
    l0.stale = false;
  }
  cutKey = "";
  syncCutPanel();
  renderRows();
  drawTextures(selected);
}

function onData(m) {
  const { kind, info, buffers } = m;
  const id = m.runId;
  if (kind === "clusters") installExtra(buildClusters(info, buffers));
  else if (kind === "crowd") installExtra(buildCrowd(info, buffers));
  else if (kind === "hlod_node") {
    if (hlodIncoming?.runId !== id) {
      dropIncoming();
      hlodIncoming = { runId: id, proxies: new Map() };
    }
    hlodIncoming.proxies.set(info.node, { model: buildModel({ ...info.scene, buffers }), info });
  } else if (kind === "hlod_tree") {
    const proxies = hlodIncoming?.runId === id ? hlodIncoming.proxies : new Map();
    hlodIncoming = null;
    installExtra(buildHlod(info, proxies));
  } else if (kind === "splats") {
    const l = entry(info.level);
    disposeLevel(l);
    Object.assign(l, { info, model: buildSplats(buffers[0]), stale: false, tex: {} });
    renderRows();
    cutStats();
  }
}

// ---- GPU crowd

/** Floats per unit `setCrowdUnits` takes: x, y, z, yaw, clip, previousClip, fade, scale, start,
 *  previousStart, fadeStart, speed. */
const UNIT = 12;
/** Units in a squad: a 5 x 5 block that changes clip together, so each change is one
 *  contiguous write. */
const SQUAD = 25;
/** Seconds a unit keeps a clip, on average, and the cross-fade into the next. */
const CROWD_HOLD = 8;
const CROWD_FADE = 0.35;
const LEVEL_TINTS = ["blue", "green", "amber"];

/**
 * Thousands of copies of the skinned model on the engine's GPU crowd. Every unit samples the
 * one pose palette on the GPU from its clip and start time; a compute pass culls the units and
 * gives each a level by its size on screen (all bones, fewer bones on a reduced mesh, a baked
 * vertex animation), then one indirect instanced draw per level. The page writes a unit only
 * when it changes clip: a squad at a time, one contiguous upload.
 */
function buildCrowd(info, buffers) {
  const [fanm, ...rest] = buffers;
  const level = (k) => {
    const [positions, normals, uvs, joints, weights, indices] = rest.slice(6 * k, 6 * k + 6);
    return { positions, normals, uvs, joints: Uint16Array.from(joints), weights, indices };
  };
  const [full, mid] = [level(0), level(1)];
  const [fvat, farUvs, farIndices, fewerJoints, fewerWeights] = rest.slice(12);
  const fewer = { ...full, joints: Uint16Array.from(fewerJoints), weights: fewerWeights };
  const cutInfo = {
    level: 1,
    kind: "crowd",
    label: "Crowd",
    triangles: 0,
    vertices: 0,
    full_triangles: 0,
    error: info.bone_gap,
    relative_error: info.bone_gap / (assetInfo?.diagonal || info.size),
    error_title: `the bone reduction moves a vertex up to ${info.bone_gap.toPrecision(3)} (all bones against ${info.bones})`,
    switch_distance_1px: 0,
    ms: info.ms,
    bytes: info.palette_bytes + info.vat_bytes,
  };
  const fail = (why) => ({
    kind: "crowd",
    info: cutInfo,
    model: { draws: () => [], dispose() {} },
    live() {},
    cut() {},
    stats: () => [["Crowd", why, "bad"]],
    dispose() {},
  });
  if (!gpu.gpuClusters) return fail("needs WebGPU: the crowd samples its palette in compute and vertex shaders");

  const material = original?.model.materials[info.material] ?? fallback();
  const make = (levels) => {
    gpu.beginCrowd(fanm);
    for (const l of levels) {
      if (l.vat) gpu.addCrowdVatLod(l.vat, l.uvs, l.indices, l.pixels);
      else gpu.addCrowdSkinnedLod(l.positions, l.normals, l.uvs, l.joints, l.weights, l.indices, l.pixels);
    }
    return gpu.endCrowd(material);
  };
  let ids;
  try {
    ids = {
      crowd: make([
        { ...full, pixels: info.levels[0].pixels },
        { ...mid, pixels: info.levels[1].pixels },
        { vat: fvat, uvs: farUvs, indices: farIndices, pixels: 0 },
      ]),
      before: make([{ ...full, pixels: 0 }]),
      after: make([{ ...fewer, pixels: 0 }]),
    };
  } catch (e) {
    return fail(String(e?.message ?? e));
  }

  // Units stand on the ground of the clips' bounds, spaced by their reach.
  const [lo, hi] = info.bounds;
  const reach = Math.max(hi[0] - lo[0], hi[2] - lo[2], info.size * 0.25);
  const spacing = reach * 1.15;
  const clips = info.clips;
  const t0 = performance.now() / 1000;
  const clock = () => performance.now() / 1000 - t0;
  let units = new Float32Array(0);
  let count = 0;
  let squads = 0;
  let side = 1; // squads across the field
  let due = 0; // squad changes owed
  let lastTick = 0;
  let written = []; // [time, units] of recent writes
  const pick = () => Math.floor(Math.random() * clips.length);

  /** Unit u's clip state, new: `clip` from `now`, fading in from what it played. */
  function setClip(u, clip, now, speed) {
    const o = u * UNIT;
    const d = clips[clip].duration || 1;
    units[o + 5] = units[o + 4];
    units[o + 9] = units[o + 8];
    units[o + 4] = clip;
    units[o + 8] = now - Math.random() * d; // a random phase, so a squad doesn't march in step
    units[o + 10] = now;
    units[o + 6] = CROWD_FADE;
    if (speed) units[o + 11] = speed;
  }

  function layout(n) {
    count = n;
    squads = Math.ceil(n / SQUAD);
    side = Math.ceil(Math.sqrt(squads));
    units = new Float32Array(n * UNIT);
    const now = clock();
    for (let s = 0; s < squads; s++) {
      const clip = pick();
      const [sx, sz] = [s % side, Math.floor(s / side)];
      for (let i = 0; i < SQUAD; i++) {
        const u = s * SQUAD + i;
        if (u >= n) break;
        const o = u * UNIT;
        const jitter = () => (Math.random() - 0.5) * 0.35 * spacing;
        units[o] = ((sx - side / 2) * 5 + (i % 5) + 0.5) * spacing + jitter();
        units[o + 1] = -lo[1];
        units[o + 2] = ((sz - side / 2) * 5 + Math.floor(i / 5) + 0.5) * spacing + jitter();
        units[o + 3] = Math.random() * Math.PI * 2;
        units[o + 7] = 1;
        setClip(u, clip, now, 0.85 + Math.random() * 0.3);
        units[o + 5] = clip;
        units[o + 6] = 0; // no fade at the start
      }
    }
    gpu.setCrowdUnits(ids.crowd, 0, units);
    gpu.setCrowdUnitCount(ids.crowd, n);
    written = [];
    cutInfo.full_triangles = n * info.levels[0].triangles;
    // The camera keeps the whole field in its planes but starts low over the middle, so the
    // near units draw on every bone and the far ones baked; Reset view comes back here.
    const half = (side * 5 * spacing) / 2;
    const prev = focusOverride;
    focusOverride = {
      center: [0, (hi[1] - lo[1]) / 2, 0],
      radius: half * 0.75,
      distance: Math.min(orbit.homeDistance(half * 0.75), orbit.homeDistance(reach) * 1.5),
    };
    orbit.maxDistance = half * 400;
    if (!prev || Math.abs(prev.radius - focusOverride.radius) > 1e-6 * prev.radius) home();
  }

  // The bone comparison: one unit on all bones, one on the reduced ones, side by side in step.
  let pairClip = -1;
  let pairAt = 0;
  const pair = new Float32Array(2 * UNIT);
  function comparePair(now) {
    if (pairClip >= 0 && now < pairAt) return;
    pairClip = (pairClip + 1) % clips.length;
    pairAt = now + Math.max(4, 2 * (clips[pairClip].duration || 1));
    for (let k = 0; k < 2; k++) {
      const o = k * UNIT;
      pair.set([(k ? 0.6 : -0.6) * reach, -lo[1], 0, 0.5, pairClip, pairClip, 0, 1, now, now, now, 1], o);
    }
    gpu.setCrowdUnits(ids.before, 0, pair.subarray(0, UNIT));
    gpu.setCrowdUnits(ids.after, 0, pair.subarray(UNIT));
  }
  let comparing = false;
  function syncCompare() {
    const on = $("#crowd-compare").checked;
    if (on === comparing && (on || count)) return;
    comparing = on;
    if (on) {
      pairClip = -1;
      focusOverride = { center: [0, (hi[1] - lo[1]) / 2, 0], radius: reach * 0.9 };
      orbit.maxDistance = reach * 400;
      home();
    } else layout(+$("#crowd-units").value);
  }

  let latest = null;
  let statsAt = 0;
  return {
    kind: "crowd",
    gpu: true,
    info: cutInfo,
    model: {
      draws: () =>
        comparing
          ? [
              [IDENTITY, ids.before, false],
              [IDENTITY, ids.after, false],
            ]
          : [[IDENTITY, ids.crowd, false]],
      dispose() {
        for (const id of Object.values(ids)) gpu.removeMesh(id);
      },
    },
    /** Each frame: the clock, the panel's settings, the squads whose clip changes, the stats. */
    live() {
      const now = clock();
      gpu.setCrowdTime(now);
      gpu.setCrowdOptions($("#crowd-cull").checked, $("#crowd-lods").checked);
      syncCompare();
      if (comparing) comparePair(now);
      else {
        const want = +$("#crowd-units").value;
        if (want !== count) layout(want);
        // Every unit changes about every CROWD_HOLD seconds; at most a few squads a frame, and
        // consecutive ones, so the frame's write stays one range.
        due += Math.min(0.25, now - lastTick) * (squads / CROWD_HOLD);
        const k = Math.min(Math.floor(due), squads, 8);
        if (k > 0) {
          due -= k;
          const first = Math.floor(Math.random() * (squads - k + 1));
          for (let s = first; s < first + k; s++) {
            const clip = pick();
            for (let i = 0; i < SQUAD && s * SQUAD + i < count; i++) setClip(s * SQUAD + i, clip, now);
          }
          const a = first * SQUAD;
          const b = Math.min(count, (first + k) * SQUAD);
          gpu.setCrowdUnits(ids.crowd, a, units.subarray(a * UNIT, b * UNIT));
          written.push([now, b - a]);
        }
      }
      lastTick = now;
      written = written.filter(([t]) => now - t < 2);
      if (performance.now() - statsAt < 250) return;
      statsAt = performance.now();
      const j = gpu.crowdStats(comparing ? ids.after : ids.crowd);
      latest = j ? JSON.parse(j) : null;
      if (latest) {
        cutInfo.triangles = latest.triangles;
        cutInfo.hud = comparing
          ? "2 units"
          : `${fmtInt((latest.perLod[0] ?? []).reduce((a, n) => a + n, 0))} of ${fmtInt(count)} units drawn`;
      }
      updateRow(1);
      cutStats();
    },
    hud: () => (comparing ? `Left: all ${info.joints} bones · right: ${info.bones} bones` : ""),
    cut() {},
    stats() {
      const rows = [["Renderer", "GPU: compute culling and level pick, one indirect instanced draw per level"]];
      const per = latest?.perLod[0];
      const names = [
        `all ${info.joints} bones, ${fmtInt(info.levels[0].triangles)} tris`,
        `${info.bones} bones, ${fmtInt(info.levels[1].triangles)} tris`,
        `vertex animation, ${fmtInt(info.levels[2].triangles)} tris`,
      ];
      info.levels.forEach((l, k) => {
        const from = l.pixels ? `from ${l.pixels} px radius` : "smaller";
        const n = !comparing && per ? `${fmtInt(per[k] ?? 0)} units · ` : "";
        rows.push([`Level ${k} (${LEVEL_TINTS[k]})`, `${n}${names[k]} · ${from}`]);
      });
      if (!comparing) {
        const drawn = per ? per.reduce((a, n) => a + n, 0) : 0;
        rows.push(["Culled", per ? `${fmtInt(count - drawn)} of ${fmtInt(count)} units` : "waiting for the GPU…"]);
        const perSecond = written.reduce((a, [, n]) => a + n, 0) / 2;
        rows.push([
          "CPU writes",
          `${fmtInt(Math.round(perSecond))} units/s (${fmtBytes(perSecond * UNIT * 4)}/s): only clip changes`,
        ]);
      }
      if (latest) rows.push(["Triangles", fmtInt(latest.triangles)]);
      const clipText = clips.map((c) => `${c.name} ${c.duration.toFixed(2)} s${c.looping ? " ↻" : ""}`).join(", ");
      rows.push(
        ["Clips", clipText],
        ["Pose palette", `${fmtBytes(info.palette_bytes)}: ${fmtInt(info.frames)} frames × ${info.joints} bones at ${info.fps} fps`],
        ["Palette error", `${info.palette_error.toPrecision(3)} (${((100 * info.palette_error) / info.size).toPrecision(2)}% of the size)`],
        ["Vertex animation", `${fmtBytes(info.vat_bytes)}, error ${info.vat_error.toPrecision(3)}`],
        [
          "Bone reduction",
          `${info.joints} → ${info.bones} bones moves a vertex up to ${info.bone_gap.toPrecision(3)} ` +
            `(${((100 * info.bone_gap) / info.size).toPrecision(2)}% of the size)`,
        ],
        ["Built in", `${(info.ms / 1000).toFixed(2)} s`],
      );
      return rows;
    },
    dispose() {}, // level 1 owns the model
  };
}

// ---- cluster DAG

/**
 * Draws any cut of the DAG: every cluster's triangles arrive once, indexing the mesh's vertices.
 * With WebGPU the engine cuts, culls and streams the packed DAG itself (buildClustersGpu).
 */
function buildClusters(info, buffers) {
  const [xf, instMesh, ...rest] = buffers;
  const per = (i) => rest.slice(8 * i, 8 * i + 8);
  if (gpu.gpuClusters) return buildClustersGpu(info, xf, instMesh, per);
  const meshes = info.meshes.map((json, i) => {
    const [pos, nrm, uv, idx, triStart, mats, clusterLevel] = per(i);
    const normal = nrm.length ? nrm : smoothNormals(pos, idx, triStart, clusterLevel);
    const slots = [...new Set(mats)].sort((a, b) => a - b);
    const slotOf = new Map(slots.map((s, k) => [s, k]));
    const tk = Uint32Array.from(mats, (s) => slotOf.get(s));
    // The vertices stay in the engine for the materials view; cuts index them.
    const geometry = gpu.addGeometry(pos, normal, uv, EMPTY);
    return { json, pos, normal, geometry, idx, triStart, tk, clusterLevel, slots, scratch: new Uint32Array(idx.length) };
  });
  const colorMat = CLUSTER_COLOR();
  const instances = Array.from(instMesh, (mi, i) => ({
    mesh: meshes[mi],
    matrix: xf.slice(16 * i, 16 * i + 16),
    ids: [], // engine meshes of this instance's cut
    sel: new Uint32Array(0),
  }));
  const cutInfo = {
    level: 1,
    kind: "clusters",
    label: "Cut",
    triangles: 0,
    vertices: 0,
    full_triangles: 0,
    error: 0,
    relative_error: 0,
    error_title: "largest simplification error among the drawn clusters",
    switch_distance_1px: 0,
    ms: info.ms,
    bytes: info.gpu_bytes,
  };
  let last = null;
  const apply = (ids) => {
    const shading = $("#shading").value;
    let at = 0;
    for (const inst of instances) {
      if (ids) {
        const n = ids[at];
        inst.sel = ids.subarray(at + 1, at + 1 + n);
        at += 1 + n;
      }
      for (const id of inst.ids) gpu.removeMesh(id);
      inst.ids = [];
      const m = inst.mesh;
      if (shading === "materials") {
        for (const [k, start, count] of indexedParts(m, inst.sel)) {
          const material = original?.model.materials[m.slots[k]] ?? fallback();
          inst.ids.push(gpu.addMesh(m.geometry, m.scratch.subarray(start, start + count), material));
        }
      } else {
        const flat = flatGeometry(m, inst.sel, shading);
        if (flat) {
          inst.ids.push(gpu.addMesh(flat.geometry, flat.indices, colorMat));
          gpu.removeGeometry(flat.geometry); // the mesh keeps it
        }
      }
    }
  };
  return {
    kind: "clusters",
    model: {
      draws() {
        const out = [];
        for (const inst of instances) for (const id of inst.ids) out.push([inst.matrix, id, true]);
        return out;
      },
      dispose() {
        for (const inst of instances) for (const id of inst.ids) gpu.removeMesh(id);
        for (const m of meshes) gpu.removeGeometry(m.geometry);
        gpu.removeMaterial(colorMat);
      },
    },
    info: cutInfo,
    cut(sel, [ids]) {
      last = sel;
      apply(ids);
      Object.assign(cutInfo, {
        triangles: sel.triangles,
        full_triangles: sel.full_triangles,
        error: sel.max_error,
        relative_error: sel.max_error / diag(),
        hud: `${fmtInt(sel.clusters)} clusters`,
      });
    },
    reshade: () => apply(null),
    stats() {
      if (!last) return [["Cut", "waiting for the camera…"]];
      const s = last;
      return [
        ["Drawn", `${fmtInt(s.triangles)} of ${fmtInt(s.full_triangles)} tris`],
        ["Clusters", `${fmtInt(s.clusters)} of ${fmtInt(info.clusters)}`],
        ["By DAG level", s.per_level.map((n, l) => `L${l}:${n}`).join(" ")],
        ["Max error", `${s.max_error.toPrecision(3)} (${((100 * s.max_error) / diag()).toPrecision(2)}%)`],
        ["DAG", `${info.levels} levels, ${fmtInt(info.all_level_triangles)} tris in all`],
        ["Cluster size", `≤ ${info.max_triangles} tris, ≤ ${info.max_vertices} verts`],
        ["GPU format", `${fmtBytes(info.gpu_bytes)}, ${info.meshes.reduce((a, m) => a + m.pages, 0)} pages`],
        ["GPU select", s.gpu_agrees ? "same clusters as CPU" : "differs from CPU", s.gpu_agrees ? "ok" : "bad"],
        ["Built in", `${(info.ms / 1000).toFixed(2)} s`],
      ];
    },
    dispose() {}, // level 1 owns the model
  };
}

const CLUSTER_COLOR = () =>
  gpu.addMaterial(new Float32Array([1, 1, 1, 1, 0, 0.7, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]), Int32Array.of(-1, -1, -1, -1, -1));

/**
 * The GPU-driven cut: each frame the engine runs the LOD cut, frustum and back-face (normal
 * cone) culling in a compute pass and draws what's left with indirect draws that decode the
 * vertices straight from the packed pages. Pages stream into a fixed pool by what every view
 * needs most, so the worker is never asked for a cut.
 */
function buildClustersGpu(info, xf, instMesh, per) {
  const meshes = info.meshes.map((json, i) => {
    const [, , , , , mats, , fcld] = per(i);
    let top = 0;
    for (const s of mats) if (s > top) top = s;
    return { json, id: gpu.addClusterMesh(fcld), top };
  });
  const colorMat = CLUSTER_COLOR();
  const instances = Array.from(instMesh, (mi, i) => ({ mesh: meshes[mi], matrix: xf.slice(16 * i, 16 * i + 16) }));
  const fullTris = instances.reduce((a, inst) => a + inst.mesh.json.triangles, 0);
  const cutInfo = {
    level: 1,
    kind: "clusters",
    label: "Cut",
    triangles: 0,
    vertices: 0,
    full_triangles: fullTris,
    error: 0,
    relative_error: 0,
    error_title: "not measured on the GPU: every drawn cluster's error is under the max on screen",
    switch_distance_1px: 0,
    ms: info.ms,
    bytes: info.gpu_bytes,
  };
  const shade = () => {
    for (const m of meshes) {
      const mats = Uint32Array.from({ length: m.top + 1 }, (_, k) => original?.model.materials[k] ?? fallback());
      gpu.setClusterMaterials(m.id, mats);
    }
    const mode = { materials: 0, clusters: 1, levels: 2 }[$("#shading").value] ?? 0;
    gpu.setClusterShading(mode, colorMat);
  };
  shade();
  let latest = null;
  let statsAt = 0;
  return {
    kind: "clusters",
    gpu: true,
    /** Each frame, instead of asking the worker: the panel's settings, and the stats now and then. */
    live() {
      gpu.setClusterThreshold(2 ** +$("#threshold").value);
      gpu.setClusterFreeze($("#freeze").checked);
      gpu.setClusterCulling($("#cull-frustum").checked, $("#cull-cone").checked);
      gpu.setClusterStreaming(+$("#pool").value * 1e6, +$("#upload").value);
      const now = performance.now();
      if (now - statsAt < 250) return;
      statsAt = now;
      const j = gpu.clusterStats();
      latest = j ? JSON.parse(j) : null;
      const v = latest?.views[0];
      if (v) Object.assign(cutInfo, { triangles: v.triangles, hud: `${fmtInt(v.clusters)} clusters` });
      updateRow(1);
      cutStats();
    },
    /** The second camera, when on: circling the model close up. */
    second() {
      if (!$("#second-camera").checked || !focus()) return null;
      const { center, radius } = focus();
      const d = orbit.homeDistance(radius) * 0.55;
      const a = performance.now() / 6000;
      const eye = [center[0] + d * Math.cos(a) * 0.9, center[1] + d * 0.35, center[2] + d * Math.sin(a) * 0.9];
      return [lookAt(eye, center), orbit.fov, orbit.near, orbit.far];
    },
    model: {
      draws: () => instances.map((inst) => [inst.matrix, inst.mesh.id, true]),
      dispose() {
        for (const m of meshes) gpu.removeMesh(m.id);
        gpu.removeMaterial(colorMat);
      },
    },
    info: cutInfo,
    cut() {},
    reshade: shade,
    stats() {
      const rows = [["Renderer", "GPU: compute cut and culling, indirect draws from streamed pages"]];
      const s = latest;
      if (!s?.views.length) return [...rows, ["Cut", "waiting for the GPU…"]];
      s.views.forEach((v, k) => {
        const cam = s.views.length > 1 ? ` (camera ${k + 1})` : "";
        rows.push([`Drawn${cam}`, `${fmtInt(v.triangles)} of ${fmtInt(fullTris)} tris, ${fmtInt(v.clusters)} clusters`]);
        rows.push([`Culled${cam}`, `${fmtInt(v.cut - v.clusters)} of ${fmtInt(v.cut)} clusters in the cut`]);
      });
      const lv = s.views[0].perLevel;
      const used = lv.findLastIndex((n) => n > 0) + 1;
      rows.push(
        ["By DAG level", lv.slice(0, used).map((n, l) => `L${l}:${n}`).join(" ")],
        [
          "Resident",
          `${fmtInt(s.residentPages)} of ${fmtInt(s.totalPages)} pages, ${fmtBytes(s.residentBytes)} of ${fmtBytes(s.totalBytes)}`,
        ],
        ["Page pool", `${fmtBytes(s.poolBytes)}, ${fmtInt(s.poolPages)} pages`],
        s.requestedPages ? ["Streaming", `${fmtInt(s.requestedPages)} pages wanted`] : ["Streaming", "everything needed is resident", "ok"],
        ["Uploaded", `${fmtInt(s.loaded)} pages, ${fmtInt(s.evicted)} evicted`],
        ["DAG", `${info.levels} levels, ${fmtInt(info.all_level_triangles)} tris in all`],
        ["Cluster size", `≤ ${info.max_triangles} tris, ≤ ${info.max_vertices} verts`],
        ["Built in", `${(info.ms / 1000).toFixed(2)} s`],
      );
      return rows;
    },
    dispose() {}, // level 1 owns the model
  };
}

/** A right-handed world-to-camera matrix, column-major, looking from `eye` at `at` with +Y up. */
function lookAt(eye, at) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (a) => {
    const l = Math.hypot(...a) || 1;
    return a.map((x) => x / l);
  };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const f = norm(sub(at, eye));
  const r = norm(cross(f, [0, 1, 0]));
  const u = cross(r, f);
  return new Float32Array([r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, eye), -dot(u, eye), dot(f, eye), 1]);
}

/** Area-weighted vertex normals from the full-detail (DAG level 0) clusters. */
function smoothNormals(pos, idx, triStart, clusterLevel) {
  const out = new Float32Array(pos.length);
  for (let c = 0; c < clusterLevel.length; c++) {
    if (clusterLevel[c] !== 0) continue;
    for (let t = triStart[c]; t < triStart[c + 1]; t++) {
      const [a, b, d] = [3 * idx[3 * t], 3 * idx[3 * t + 1], 3 * idx[3 * t + 2]];
      const u = [pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]];
      const v = [pos[d] - pos[a], pos[d + 1] - pos[a + 1], pos[d + 2] - pos[a + 2]];
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      for (const o of [a, b, d]) for (let k = 0; k < 3; k++) out[o + k] += n[k];
    }
  }
  for (let o = 0; o < out.length; o += 3) {
    const l = Math.hypot(out[o], out[o + 1], out[o + 2]) || 1;
    for (let k = 0; k < 3; k++) out[o + k] /= l;
  }
  return out;
}

/**
 * Writes the triangles of the selected clusters into the mesh's scratch indices, grouped by
 * material; returns the groups as [slot index, start, count].
 */
function indexedParts(m, sel) {
  const { triStart, tk, idx, slots, scratch } = m;
  const start = new Uint32Array(slots.length + 1);
  for (const c of sel) for (let t = triStart[c]; t < triStart[c + 1]; t++) start[tk[t] + 1]++;
  for (let k = 0; k < slots.length; k++) start[k + 1] += start[k];
  const fill = start.slice(0, slots.length);
  for (const c of sel)
    for (let t = triStart[c]; t < triStart[c + 1]; t++) {
      const o = 3 * fill[tk[t]]++;
      scratch[o] = idx[3 * t];
      scratch[o + 1] = idx[3 * t + 1];
      scratch[o + 2] = idx[3 * t + 2];
    }
  const out = [];
  for (let k = 0; k < slots.length; k++)
    if (start[k + 1] > start[k]) out.push([k, 3 * start[k], 3 * (start[k + 1] - start[k])]);
  return out;
}

/** The cut as unshared triangles with a color per cluster or per DAG level, in the engine. */
function flatGeometry(m, sel, shading) {
  let tris = 0;
  for (const c of sel) tris += m.triStart[c + 1] - m.triStart[c];
  if (!tris) return null;
  const pos = new Float32Array(9 * tris);
  const nrm = new Float32Array(9 * tris);
  const col = new Float32Array(9 * tris);
  const P = m.pos;
  const N = m.normal;
  let o = 0;
  for (const c of sel) {
    const [r, g, b] = shading === "levels" ? levelColor(m.clusterLevel[c]) : clusterColor(c);
    for (let t = m.triStart[c]; t < m.triStart[c + 1]; t++)
      for (let j = 0; j < 3; j++) {
        const v = 3 * m.idx[3 * t + j];
        pos[o] = P[v];
        pos[o + 1] = P[v + 1];
        pos[o + 2] = P[v + 2];
        nrm[o] = N[v];
        nrm[o + 1] = N[v + 1];
        nrm[o + 2] = N[v + 2];
        col[o] = r;
        col[o + 1] = g;
        col[o + 2] = b;
        o += 3;
      }
  }
  const indices = Uint32Array.from({ length: 3 * tris }, (_, i) => i);
  return { geometry: gpu.addGeometry(pos, nrm, EMPTY, col), indices };
}

// ---- HLOD

function buildHlod(info, proxies) {
  const src = info.source_instances;
  // The tiled instances, copy-major as the worker numbers them: the original's instances
  // moved by each copy's offset, drawn with its meshes and materials.
  const tiled = info.offsets.map((o) => {
    const draws = original.model.draws(null, [], o);
    const perSource = Array.from({ length: src }, () => []);
    // draws() walks the original's instances in order, its parts within each.
    let at = 0;
    for (const [s, inst] of original.model.instances.entries()) {
      const n = original.model.meshes[inst.mesh].parts.length;
      if (s < src) perSource[s].push(...draws.slice(at, at + n));
      at += n;
    }
    return perSource;
  });
  const cutInst = tiled.flat(); // draws of tiled instance i
  const full = {
    draws: () => cutInst.flat(),
    dispose() {}, // borrows the original's meshes
  };
  let shownProxies = [];
  let shownInst = [];
  const tints = new Map(); // tree level -> material
  const tintFor = (level) => {
    if (!$("#tint").checked) return null;
    let m = tints.get(level);
    if (m === undefined) tints.set(level, (m = colorMaterial(levelColor(level))));
    return m;
  };
  const model = {
    draws() {
      const out = [];
      for (const i of shownInst) for (const d of cutInst[i] ?? []) out.push(d);
      for (const p of shownProxies) p.model.draws(tintFor(p.info.level), out);
      return out;
    },
    dispose() {
      for (const p of proxies.values()) p.model.dispose();
      for (const m of tints.values()) gpu.removeMaterial(m);
    },
  };
  // The bake panel shows the root proxy's textures (the coarsest one).
  let texProxy = null;
  for (const p of proxies.values()) if (p.info.baked && (!texProxy || p.info.level > texProxy.info.level)) texProxy = p;
  const proxyBytes = [...proxies.values()].reduce((a, p) => a + p.info.bytes, 0);
  const total = info.instance_triangles;
  const cutInfo = {
    level: 1,
    kind: "hlod",
    label: "Cut",
    triangles: 0,
    vertices: 0,
    full_triangles: total,
    error: 0,
    relative_error: 0,
    error_title: "largest error among the drawn proxies",
    switch_distance_1px: 0,
    ms: info.ms,
    bytes: proxyBytes,
  };
  const prev = focusOverride;
  focusOverride = { center: info.focus.center, radius: info.focus.radius };
  orbit.maxDistance = focusOverride.radius * 400;
  if (!prev || Math.abs(prev.radius - focusOverride.radius) > 1e-6 * prev.radius) home();
  let last = null;
  return {
    kind: "hlod",
    model,
    full,
    fullInfo: { label: `Full (${info.copies} ${info.copies === 1 ? "copy" : "copies"})`, triangles: total, full_triangles: total },
    info: cutInfo,
    texProxy,
    cut(sel, [nodes, inst]) {
      last = sel;
      shownProxies = [...nodes].map((n) => proxies.get(n)).filter(Boolean);
      shownInst = [...inst];
      Object.assign(cutInfo, {
        triangles: sel.triangles,
        full_triangles: sel.full_triangles,
        error: sel.max_error,
        relative_error: sel.max_error / diag(),
        hud: `${sel.proxies} ${sel.proxies === 1 ? "proxy" : "proxies"}, ${sel.instances} ${sel.instances === 1 ? "instance" : "instances"}`,
      });
    },
    stats() {
      const rows = [];
      if (last) {
        const s = last;
        rows.push(
          ["Drawn", `${fmtInt(s.triangles)} of ${fmtInt(s.full_triangles)} tris`],
          ["Proxies", `${s.proxies} of ${proxies.size} (${fmtInt(s.proxy_triangles)} tris)`],
          ["Source instances", `${s.instances} of ${cutInst.length} (${fmtInt(s.source_triangles)} tris)`],
          ["Max proxy error", `${s.max_error.toPrecision(3)} (${((100 * s.max_error) / diag()).toPrecision(2)}%)`],
        );
      } else rows.push(["Cut", "waiting for the camera…"]);
      // Where the closest leaf cell gives way to its proxy (LodView::projected, solved for
      // distance), so dense models whose proxies only show far away still explain themselves.
      const ppr = canvasPpr();
      const t = 2 ** +$("#threshold").value;
      const eye = orbit.eye;
      let near = null;
      for (const n of info.nodes) {
        if (n.level !== 0 || !(n.error > 0)) continue;
        const [x, y, z, r] = n.sphere;
        const d = Math.hypot(eye[0] - x, eye[1] - y, eye[2] - z);
        if (!near || d < near.d) near = { d, at: (n.error * ppr) / t + r };
      }
      if (near)
        rows.push([
          "Nearest leaf proxy",
          `from ${near.at.toPrecision(3)} away (now ${near.d.toPrecision(3)})`,
          near.d >= near.at ? "ok" : "",
        ]);
      rows.push(
        ["Tree", `${info.levels} levels, ${info.nodes.length} nodes, ${info.roots.length} root${info.roots.length === 1 ? "" : "s"}`],
        ["Proxy tris by level", info.level_triangles.map((n, l) => `L${l}:${fmtInt(n)}`).join(" ")],
        ["Leaf cell", `${info.cell_size.toPrecision(3)} wide, ${info.copies} ${info.copies === 1 ? "copy" : "copies"}`],
        ["Proxies", info.baked ? `baked 256², ${fmtBytes(proxyBytes)} of GLB` : `geometry only, ${fmtBytes(proxyBytes)}`],
        ["Visibility", visibilityText(info.visibility)],
        ["Built in", `${(info.ms / 1000).toFixed(2)} s`],
      );
      return rows;
    },
    dispose() {
      if (levels[0] && original) levels[0].model = original.model;
    },
  };
}

/** Where HLOD's visibility renders ran and what they cost. */
function visibilityText(v) {
  if (!v || !v.views) return "—";
  const where = v.gpu_renders && !v.cpu_renders ? `GPU (${v.gpu})` : v.gpu_renders ? `GPU ${v.gpu_renders}, CPU ${v.cpu_renders}` : "CPU";
  return `${where}: ${fmtInt(v.views)} views in ${(v.ms / 1000).toFixed(2)} s`;
}

/** The bake panel in HLOD mode: the root proxy's baked textures. */
function drawHlodTextures() {
  const bake = $("#bake");
  const p = extra?.kind === "hlod" ? extra.texProxy : null;
  const m = p?.model.json[0];
  const tex = (i) => (i >= 0 ? p.model.textures[i] : null);
  const maps = m ? { base_color: tex(m.base_color_texture), normal: tex(m.normal_texture), orm: tex(m.metallic_roughness_texture) } : {};
  bake.classList.toggle("empty", !maps.base_color);
  $("#bake-title").textContent = p ? `Proxy textures · node ${p.info.node} (tree level ${p.info.level})` : "Proxy textures";
  if (!maps.base_color) {
    setStats(extra?.kind === "hlod" ? "Proxies are geometry only (Bake proxy textures is off)." : "");
    return;
  }
  for (const [ch, t] of Object.entries(maps)) if (t) putTexture(ch, drawable(t));
  setStats(`${fmtInt(p.info.triangles)} tris · ${fmtBytes(p.info.bytes)} · ${fmtPct(p.info.coverage)} covered`);
}

// ---- splats

// The engine draws each splat as a screen-space quad shaped by its projected 3D covariance (EWA
// splatting, as 3D Gaussian splatting viewers do), as blended Gaussians or opaque discs.
// Payload per splat: position, covariance (xx xy xz yy yz zz), linear RGBA, normal.

/** A splat cloud as a level's model; `splat` holds what sorting it needs. */
function buildSplats(payload) {
  const k = payload.length / 16;
  const id = gpu.addSplats(payload);
  return {
    splat: { id, payload, k, key: "", depth: new Float32Array(k), keys: new Uint16Array(k), order: new Uint32Array(k), sorted: new Float32Array(payload.length) },
    draws: () => [[IDENTITY, id, false]],
    dispose: () => gpu.removeMesh(id),
  };
}

let sortCounts = null;
/** Blended Gaussians are drawn back to front: re-sort (counting sort on depth) when the view turns. */
function sortSplats() {
  if ($("#splat-draw").value !== "gauss") return;
  let e = null;
  let key = "";
  for (const l of levels) {
    const s = l?.model?.splat;
    if (!s) continue;
    if (!e) {
      e = orbit.viewMatrix;
      key = `${e[2].toFixed(4)},${e[6].toFixed(4)},${e[10].toFixed(4)},${e[14].toPrecision(5)}`;
    }
    if (s.key === key) continue;
    s.key = key;
    const { payload, k, depth, keys, order, sorted } = s;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < k; i++) {
      const o = 16 * i;
      const z = e[2] * payload[o] + e[6] * payload[o + 1] + e[10] * payload[o + 2] + e[14];
      depth[i] = z;
      if (z < lo) lo = z;
      if (z > hi) hi = z;
    }
    const B = 65536;
    sortCounts ??= new Uint32Array(B + 1);
    const counts = sortCounts.fill(0);
    const scale = (B - 1) / Math.max(hi - lo, 1e-12);
    for (let i = 0; i < k; i++) {
      const q = ((depth[i] - lo) * scale) | 0;
      keys[i] = q;
      counts[q + 1]++;
    }
    for (let b = 0; b < B; b++) counts[b + 1] += counts[b];
    for (let i = 0; i < k; i++) order[counts[keys[i]]++] = i; // farthest (most negative z) first
    for (let j = 0; j < k; j++) sorted.set(payload.subarray(16 * order[j], 16 * order[j] + 16), 16 * j);
    gpu.setSplats(s.id, sorted);
  }
}

function splatStats() {
  const l = levels[selected]?.info?.kind === "splats" ? levels[selected] : levels[1];
  const i = l?.info;
  if (i?.kind !== "splats") return [];
  return [
    ["Level", `LOD${i.level}: ${fmtInt(i.vertices)} splats`],
    ["Shape", i.note ?? ""],
    ["Surfel radius", `${i.error.toPrecision(3)} (${(i.relative_error * 100).toPrecision(2)}%)`],
    ["PLY (3DGS layout)", i.bytes ? fmtBytes(i.bytes) : ""],
    ["Draw", $("#splat-draw").value === "gauss" ? "blended, sorted back to front" : "opaque discs, depth-tested"],
  ];
}

$("#threshold").addEventListener("input", () => {
  $("#threshold-out").value = `${fmtPx(2 ** +$("#threshold").value)} px`;
});
$("#freeze").addEventListener("change", () => (cutKey = ""));
$("#upload").addEventListener("input", () => ($("#upload-out").value = $("#upload").value));
$("#shading").addEventListener("change", () => {
  $("#show-clusters").checked = $("#shading").value === "clusters";
  extra?.reshade?.();
});
// The toolbar's shortcut for the cut panel's Color: per cluster, or back to the materials.
$("#show-clusters").addEventListener("change", (e) => {
  $("#shading").value = e.target.checked ? "clusters" : "materials";
  $("#shading").dispatchEvent(new Event("change"));
});
$("#splat-draw").addEventListener("change", () => {
  for (const l of levels) if (l?.model?.splat) l.model.splat.key = "";
  cutStats();
});

// ---------------------------------------------------------------- worker

// The engine: the wasm module, streamed in once with a progress bar (compiling as it arrives)
// and handed, compiled, to every worker, so respawning one costs neither again.
const engineProgress = { got: 0, total: 0, done: false }; // done: downloaded, maybe compiling
let engineModule = null;
let waitingForEngine = false; // a load is waiting for it, so the overlay shows its progress
const engine = (async () => {
  const pkg = (file) => new URL(`./pkg/${file}`, import.meta.url);
  // A gzipped response's Content-Length is the compressed size, so build.sh writes the real one.
  fetch(pkg("wasm-size.txt"))
    .then((r) => (r.ok ? r.text() : ""))
    .then((t) => {
      if (!engineProgress.total && +t > 0) engineProgress.total = +t;
      showEngine();
    })
    .catch(() => {});
  const r = await fetch(pkg("farthing_demo_web_bg.wasm"));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  if (!r.headers.get("content-encoding")) engineProgress.total = +r.headers.get("content-length") || 0;
  if (!r.body) return WebAssembly.compile(await r.arrayBuffer());
  const [counted, compiled] = r.body.tee();
  const module = WebAssembly.compileStreaming
    ? WebAssembly.compileStreaming(new Response(compiled, { headers: { "content-type": "application/wasm" } }))
    : new Response(compiled).arrayBuffer().then((b) => WebAssembly.compile(b));
  module.catch(() => {}); // reported through `engine` below, not as an unhandled rejection
  const reader = counted.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    engineProgress.got += value.length;
    showEngine();
  }
  engineProgress.done = true;
  showEngine();
  return module;
})();
engine.then(
  (module) => {
    engineModule = module;
    $("#engine").hidden = true;
  },
  (err) => {
    const el = $("#engine");
    el.classList.remove("indeterminate");
    el.querySelector(".nums").textContent = `Could not load the engine: ${err.message ?? err}`;
    setStatus(`Could not load the engine: ${err.message ?? err}`, "error");
  },
);
/** The engine's download, under the empty-state hint and, while a load waits for it, in the overlay. */
function showEngine() {
  if (engineModule) return;
  const { got, done } = engineProgress;
  const total = engineProgress.total >= got ? engineProgress.total : 0;
  const phase = done ? "Preparing the engine…" : "Downloading the engine…";
  const el = $("#engine");
  el.hidden = false;
  el.classList.toggle("indeterminate", done || !total);
  el.querySelector(".meter").style.setProperty("--p", total ? got / total : 0);
  el.querySelector(".nums").textContent = done ? phase : `${phase} ${mb(got)}${total ? ` of ${mb(total)}` : ""}`;
  if (waitingForEngine) {
    if (!done) loadingProgress(phase, got, total);
    else if (loadingEl.querySelector(".phase").textContent !== phase) loadingBusy(phase);
  }
}

function spawn() {
  worker?.terminate();
  cutInflight = false;
  const w = (worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" }));
  engine.then((module) => w.postMessage({ type: "engine", module }), () => {});
  worker.onmessage = ({ data }) => onWorker(data);
  worker.onerror = (e) => {
    busy = false;
    hideLoading();
    setStatus(`Worker failed to start: ${e.message ?? "see the console"}`, "error");
  };
}

// The viewport overlay while a model downloads and while the worker decodes it: big enough
// that clicking a sample visibly does something, with a way out.
const loadingEl = $("#loading");
let loadingT0 = 0;
let loadingTimer = 0;
function showLoading(title, thumb) {
  loadingEl.querySelector(".title").textContent = title;
  const img = loadingEl.querySelector("img");
  img.hidden = !thumb;
  if (thumb) img.src = thumb;
  loadingEl.hidden = false;
  $("#empty").hidden = true;
}
function loadingProgress(phase, got, total) {
  clearInterval(loadingTimer);
  loadingEl.classList.toggle("indeterminate", !total);
  loadingEl.querySelector(".phase").textContent = phase;
  loadingEl.querySelector(".meter").style.setProperty("--p", total ? got / total : 0);
  loadingEl.querySelector(".nums").textContent = total
    ? `${mb(got)} of ${mb(total)} · ${Math.floor((100 * got) / total)}%`
    : got
      ? mb(got)
      : "";
}
/** Indeterminate phase with an elapsed-time counter (decoding a textured model takes seconds). */
function loadingBusy(phase) {
  loadingProgress(phase, 0, 0);
  loadingT0 = performance.now();
  const nums = loadingEl.querySelector(".nums");
  const tick = () => (nums.textContent = `${((performance.now() - loadingT0) / 1000).toFixed(1)} s`);
  tick();
  loadingTimer = setInterval(tick, 100);
}
function hideLoading() {
  clearInterval(loadingTimer);
  loadingEl.hidden = true;
  if (!original) $("#empty").hidden = false;
}
loadingEl.querySelector(".cancel").addEventListener("click", () => {
  fetching?.abort();
  loadToken++; // a load waiting for the engine gives up
  waitingForEngine = false;
  if (busy && !original) {
    // Decoding: wasm can't be interrupted, so drop the worker.
    spawn();
    busy = false;
    files = null;
  }
  for (const b of cards.values()) b.removeAttribute("aria-current");
  hideLoading();
  setStatus("Canceled");
});

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = cls;
}

async function onWorker(m) {
  if (m.runId !== undefined && m.runId !== runId) return; // a canceled run
  switch (m.type) {
    case "loaded":
      if (m.quiet) return;
      busy = false;
      assetInfo = m.info;
      applyViewPreset();
      showInfo(m.info);
      try {
        await viewerReady;
        showOriginal(m.scene);
      } catch (e) {
        hideLoading();
        return setStatus(`Could not display the model: ${e.message ?? e}`, "error");
      }
      hideLoading();
      $("#empty").hidden = true;
      startRun();
      break;
    case "stage":
      if (m.label === "done") break;
      setStatus(`${m.label}…`, "busy");
      break;
    case "texture": {
      const l = entry(m.level);
      l.tex[m.channel] = new ImageData(new Uint8ClampedArray(m.rgba), m.width, m.height);
      if (followTextures(m.level)) drawTextures(m.level);
      break;
    }
    case "level":
      onLevel(m.info, m.scene, m.runId);
      break;
    case "data":
      onData(m);
      break;
    case "select":
      onSelect(m);
      break;
    case "done": {
      busy = false;
      const chain = shownMode === "chain";
      let n = chain ? chainLevels(runOpts) : MODE_LEVELS[shownMode];
      // The chain stops at the first level past the error limit, which is left out.
      const built = levels.findLastIndex((l) => l?.info && !l.stale);
      if (chain && built < n) n = stoppedAt = Math.max(built, 0);
      for (let i = levels.length - 1; i > n; i--) {
        disposeLevel(levels[i]);
        levels.pop();
      }
      if (selected > n) select(n);
      drawTextures(texPinned ? selected : Math.min(texShown, n));
      const what = {
        chain: `${n + 1} levels`,
        presets: `${n} configurations`,
        clusters: "the cluster DAG",
        hlod: "the HLOD tree",
        splats: `${n} splat levels`,
        crowd: "the GPU crowd",
      }[shownMode];
      const live =
        shownMode === "clusters" || shownMode === "hlod"
          ? "; the cut follows the camera"
          : shownMode === "crowd"
            ? "; each unit's level follows the camera"
            : "";
      const limit = `${+((runOpts.max_error || 0.02) * 100).toFixed(2)}%`;
      const byError = runOpts.spacing === "error";
      const stop = byError
        ? `; spaced by error, ${runOpts.error_ratio}× apart up to ${limit}` +
          (stoppedAt != null ? ` (no room for more above LOD0's error)` : "")
        : stoppedAt != null
          ? `; stopped there, as the next level deviates more than ${limit}`
          : "";
      setStatus(`Built ${what} in ${((performance.now() - runStart) / 1000).toFixed(2)} s${stop}${live}`);
      cutKey = "";
      renderRows();
      cutStats();
      break;
    }
    case "download": {
      const type = { ply: "application/octet-stream", zip: "application/zip" }[m.format] ?? "model/gltf-binary";
      const url = URL.createObjectURL(new Blob([m.glb], { type }));
      const stem = `${(assetInfo?.name ?? "model").replace(/\.[^./]+$/, "")}_${lodName(m.level)}`;
      const a = Object.assign(document.createElement("a"), {
        href: url,
        download: m.format === "zip" ? `${stem}_cooked.zip` : `${stem}.${m.format}`,
      });
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
      if (pendingDownload === m.level) {
        pendingDownload = null;
        if (!busy) setStatus("");
      }
      break;
    }
    case "error":
      busy = false;
      hideLoading();
      setStatus(m.message, "error");
      break;
    case "crash": {
      // The worker's wasm is dead after a trap: replace it, and reload the files quietly so
      // the model on screen can be run again with other options.
      busy = false;
      hideLoading();
      console.error("farthing wasm trap:", m.message);
      spawn();
      if (files && original) worker.postMessage({ type: "load", files, quiet: true });
      else files = null;
      setStatus(
        `farthing crashed in the browser (wasm: ${m.message}), most likely out of memory. ` +
          "The worker was restarted; try fewer levels, a smaller atlas or a smaller model.",
        "error",
      );
      break;
    }
  }
}

function entry(level) {
  return (levels[level] ??= { info: null, model: null, tex: {}, stale: true });
}

function onLevel(info, scene, id) {
  if (id !== runId) return;
  const l = entry(info.level);
  if (info.level === 0) {
    l.info = info;
    l.stale = false;
    return renderRows();
  }
  // Reduced levels keep the original's materials; a shared level after the one carrying the
  // atlas draws with that level's. Both come without materials of their own.
  const src = info.kind === "shared" && info.shared_textures !== info.level ? info.shared_textures : null;
  let borrow = null;
  if (info.kind === "reduced") borrow = (slot) => original?.model.materials[slot];
  else if (src !== null) {
    const from = levels[src]?.model?.materials ?? [];
    borrow = (slot) => from[slot] ?? from[0];
  }
  disposeLevel(l);
  if (src !== null) l.tex = levels[src]?.tex ?? {};
  else if (info.kind !== "baked" && info.kind !== "shared") l.tex = {};
  l.model = scene ? buildModel(scene, borrow) : null;
  l.info = info;
  l.stale = false;
  renderRows();
  if (info.kind !== "reduced" ? followTextures(info.level) : selected === info.level) drawTextures(info.level);
}

function startRun() {
  if (!files || !original) return;
  const id = ++runId;
  if (busy) {
    // Cancel: wasm can't be interrupted, so replace the worker and reload the files.
    spawn();
    worker.postMessage({ type: "load", files, quiet: true });
  }
  busy = true;
  runStart = performance.now();
  runOpts = readOptions();
  if (runOpts.mode !== shownMode) switchMode(runOpts.mode);
  for (const l of levels) if (l) l.stale = true;
  stoppedAt = null;
  renderRows();
  setStatus("Starting…", "busy");
  worker.postMessage({ type: "run", runId: id, options: runOpts });
}

let loadToken = 0; // bumped by every load and cancel, so a load waiting for the engine can tell
async function load(list, title) {
  if (!list.length) return;
  if (loadingEl.hidden) showLoading(title ?? (list.length === 1 ? list[0].name : `${list.length} files`));
  const token = ++loadToken;
  if (!engineModule) {
    waitingForEngine = true;
    loadingProgress("Downloading the engine…", 0, 0);
    showEngine();
    try {
      await engine;
    } catch (err) {
      if (token === loadToken) {
        waitingForEngine = false;
        hideLoading();
        setStatus(`Could not load the engine: ${err.message ?? err}`, "error");
      }
      return;
    }
    if (token !== loadToken) return;
    waitingForEngine = false;
  }
  loadingBusy("Decoding geometry and textures…");
  resetViewer();
  rowsEl.replaceChildren();
  drawTextures(-1);
  files = list;
  assetInfo = null;
  runId++;
  spawn();
  busy = true;
  setStatus(`Loading ${list.length === 1 ? list[0].name : `${list.length} files`}…`, "busy");
  worker.postMessage({ type: "load", files });
}

// ---------------------------------------------------------------- UI: info, table, textures

const fmtInt = (n) => n.toLocaleString("en-US");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

function showInfo(i) {
  $("#asset").hidden = false;
  $("#asset-name").textContent = i.name;
  const rows = [
    ["Triangles", fmtInt(i.triangles)],
    ["Vertices", fmtInt(i.vertices)],
    ["Meshes / instances", `${i.meshes} / ${i.instances}`],
    ["Materials (textured)", `${i.materials.length} (${i.textured_materials})`],
    ["Size (diagonal)", i.diagonal.toPrecision(3)],
    ["Load", `${i.load_ms.toFixed(0)} ms`],
  ];
  $("#asset-stats").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  const warn = $("#asset-warn");
  warn.hidden = !i.skinned;
  warn.textContent = i.skinned ? "Skinned: reduced levels keep the skin; baked proxies are static." : "";
  syncViewHint();
}

function renderRows() {
  const m = shownMode || form.elements.mode.value;
  const chain = m === "chain";
  // A recut chain is the original, LOD0 (the recut) and the levels reduced from it.
  const recut = chain && form.elements.textures.value === "recut";
  const n = chain ? (stoppedAt ?? chainLevels(readOptions())) : MODE_LEVELS[m];
  const bakeFrom = recut ? Infinity : +form.elements.bake_from.value || Infinity;
  const near = form.elements.textures.value === "original" ? "reduced" : "shared";
  const total = assetInfo?.triangles ?? 0;
  const out = [];
  for (let i = 0; i <= Math.max(n, levels.length - 1); i++) {
    const l = levels[i];
    const info = l?.info;
    const kind = info?.kind ?? (i === 0 ? "original" : chain ? (i >= bakeFrom ? "baked" : near) : MODE_KIND[m]);
    const pending = !info || l.stale;
    const live = kind === "clusters" || kind === "hlod" || kind === "crowd";
    const per = { hlod: "cell", crowd: "unit" }[kind] ?? "cluster";
    const of = info?.full_triangles || total;
    const errTitle =
      info?.error_title ??
      [
        info?.error_bound != null
          ? `${info.error.toPrecision(3)} measured; simplifier bound ${info.error_bound.toPrecision(3)}`
          : `${info?.error?.toPrecision(3)}`,
        info?.mean_error != null ? `mean ${info.mean_error.toPrecision(3)}, RMS ${info.rms_error.toPrecision(3)}` : "",
        info?.normal_error != null ? `normals off by ${info.normal_error.toFixed(1)}° on average` : "",
        info?.texture_error != null ? texErrorText(info) : "",
      ]
        .filter(Boolean)
        .join("\n");
    const splats = kind === "splats";
    const cells = info
      ? [
          splats ? `<span title="Splats have no triangles">—</span>` : fmtInt(info.triangles),
          splats ? "—" : of ? `${((100 * info.triangles) / of).toPrecision(3)}` : "",
          i === 0 ? "—" : `<span title="${esc(errTitle)}">${(info.relative_error * 100).toPrecision(2)}%</span>`,
          i === 0
            ? "—"
            : live
              ? `<span title="Every ${per} switches at its own distance">per ${per}</span>`
              : info.switch_distance_1px.toPrecision(3),
          i === 0 ? "—" : `${info.ms.toFixed(0)} ms`,
          info.bytes ? fmtBytes(info.bytes) : "—",
          info.texture_bytes != null ? texCell(info) : "—",
        ]
      : ["…", "", "", "", "", "", ""];
    const name =
      splats && info
        ? `${fmtInt(info.vertices)} splats`
        : (info?.label ?? (i === 0 && (!chain || recut) ? "Original" : `LOD${i - (recut ? 1 : 0)}`));
    const kindText = recut && kind === "shared" ? "recut" : kind;
    const ext = kind === "splats" ? "ply" : "glb";
    const cooked = info?.cooked;
    const dl =
      pending || live || (i === 0 && m === "hlod")
        ? ""
        : `<button type="button" data-dl="${i}" data-fmt="${ext}" title="Download ${esc(name)} as .${ext}">.${ext}</button>` +
          (cooked
            ? ` <button type="button" data-dl="${i}" data-fmt="zip" title="${esc(name)} as it ships, ${fmtBytes(cooked.ship_bytes)}: ` +
              `compact GLB plus mipple's KTX2 textures (${esc(cookedFormats(cooked))}, Zstd ${cooked.zstd_level}) and a manifest">.zip</button>`
            : "");
    out.push(
      `<tr data-level="${i}" class="${i === selected ? "selected" : ""} ${pending ? "pending" : ""}">` +
        `<td class="label" title="${esc(info?.note ?? "")}">${esc(name)}</td><td><span class="kind ${kind}">${kindText}</span></td>` +
        cells.map((c) => `<td class="n">${c}</td>`).join("") +
        `<td>${dl}</td></tr>`,
    );
  }
  rowsEl.innerHTML = out.join("");
  syncViewHint();
}

/** A level's texture memory; shared levels after the first draw with the first's atlas. */
function texCell(info) {
  const c = info.cooked;
  const reused = info.kind === "shared" && info.shared_textures !== info.level;
  const carried = reused ? `The atlas ${lodName(info.shared_textures)} carries, not a second copy. ` : "";
  if (!c) {
    const bytes = fmtBytes(info.texture_bytes);
    return reused ? `<span title="${esc(carried)}">${bytes}</span>` : bytes;
  }
  const title =
    `${carried}GPU memory as ${cookedFormats(c)} with mips, cooked by mipple; RGBA8 would be ${fmtBytes(c.rgba_bytes)}. ` +
    `Ships as ${fmtBytes(c.ktx2_bytes)} of KTX2 + Zstd.`;
  return `<span class="cooked" title="${esc(title)}">${fmtBytes(c.gpu_bytes)}</span>`;
}

/** "BC7 + BC5" style names of a level's cooked formats. */
function cookedFormats(c) {
  return [...new Set(c.textures.map((t) => FORMAT_NAMES[t.format] ?? t.format.toUpperCase()))].join(" + ");
}

const FORMAT_NAMES = {
  "etc2-rgb": "ETC2",
  "etc2-rgba": "ETC2 RGBA",
  "eac-rg11": "EAC RG11",
  "astc-4x4": "ASTC 4×4",
  "astc-6x6": "ASTC 6×6",
  "astc-8x8": "ASTC 8×8",
};

const TEX_CAPTIONS = { base_color: "Base color", normal: "Normal", orm: "Rough / metal" };

// The panel shows only the textures and their channel names; everything else about them lives in
// the tooltip, so nothing under the images changes size when another level is picked.
let statsTip = "";
let cookedTip = "";
let formatTips = {};
function setTexTip() {
  for (const ch of Object.keys(TEX_CAPTIONS))
    $(`#tex-${ch}`).title = [formatTips[ch] ?? TEX_CAPTIONS[ch], statsTip, cookedTip].filter(Boolean).join("\n\n");
}

/** Sets the textures' tooltip: a short summary followed by the details. */
function setStats(short, details = []) {
  statsTip = [short, ...details].filter(Boolean).join("\n");
  setTexTip();
}

/** A percentage with two significant digits, never in exponent notation. */
const fmtPct = (x) => {
  const p = x * 100;
  return p >= 10 ? `${Math.round(p)}%` : `${p.toPrecision(2)}%`;
};

/** Adds the mipple cook (formats, sizes, quality) to the textures' tooltips. */
function drawCooked(i) {
  const c = i?.cooked;
  formatTips = {};
  for (const t of c?.textures ?? []) formatTips[t.channel] = `${TEX_CAPTIONS[t.channel]} · ${cookedFormats({ textures: [t] })}`;
  cookedTip = "";
  if (!c) return setTexTip();
  const quality = c.textures
    .map((t) => {
      const q = t.normal_error != null ? `${t.normal_error.toFixed(2)}° off` : t.psnr != null ? `${t.psnr.toFixed(1)} dB` : "lossless";
      return `${CHANNEL_NAMES[t.channel]} ${q}`;
    })
    .join(", ");
  const rdo = c.lambda > 0 ? `, RDO λ ${c.lambda}` : "";
  cookedTip =
    `Cooked by mipple for ${c.target} (effort ${c.effort}${rdo}) in ${(c.ms / 1000).toFixed(1)} s and drawn decoded: ` +
    `what you see is what the GPU samples.\n${quality}\n` +
    `${fmtBytes(c.gpu_bytes)} on the GPU instead of ${fmtBytes(c.rgba_bytes)} as RGBA8 ` +
    `(${(c.rgba_bytes / c.gpu_bytes).toFixed(1)}× less)\n` +
    `${fmtBytes(c.ktx2_bytes)} of KTX2 + Zstd on disk, ${fmtBytes(c.ship_bytes)} with the geometry (the row's .zip)`;
  setTexTip();
}

/** How far a recut's base color is from the original materials. */
const texErrorText = (i) =>
  `Base color ${i.texture_error.toFixed(1)} levels in 255 from the original on average after resampling ` +
  `(99th percentile ${i.texture_error_p99.toFixed(0)})`;

const updateRow = () => renderRows();

/** Rows after the original in a chain: a recut adds LOD0, the recut itself, before `levels` reductions. */
function chainLevels(o) {
  return o.levels + (o.textures === "recut" ? 1 : 0);
}

function select(level) {
  selected = level;
  texPinned = true;
  for (const tr of rowsEl.children) tr.classList.toggle("selected", +tr.dataset.level === level);
  if (hasTextures(level)) drawTextures(level);
  if (shownMode === "splats") cutStats();
}

/** Chain levels drawn with the original's materials: the original, and Original-materials reductions. */
const usesOriginal = (level) =>
  shownMode === "chain" && !!original && (level === 0 || levels[level]?.info?.kind === "reduced");

/** Whether the texture panel has something of this level's own to show. */
const hasTextures = (level) => !!levels[level]?.tex.base_color || usesOriginal(level);

/** Whether textures arriving for `level` should replace what the panel shows. */
const followTextures = (level) => level === selected || !texPinned || !hasTextures(selected);

// Original textures can be large; the panel shows them at most this many texels across.
const PANEL_SIDE = 1024;
// The UV overlay is skipped above this many triangles: it would be a solid fill anyway.
const UV_MAX_TRIANGLES = 300_000;

/** Puts `src` (ImageData or a drawable image; null clears) on the panel canvas of channel `ch`. */
function putTexture(ch, src) {
  const c = $(`#tex-${ch}`);
  if (!c) return null;
  const ctx = c.getContext("2d", { willReadFrequently: ch === "orm" });
  if (src instanceof ImageData) {
    c.width = src.width;
    c.height = src.height;
    ctx.putImageData(src, 0, 0);
    if (ch === "orm") roughMetalOnly(ctx, c);
    return c;
  }
  const w = src?.width ?? 0;
  const h = src?.height ?? 0;
  const s = Math.min(1, PANEL_SIDE / Math.max(w, h, 1));
  c.width = Math.max(1, Math.round(w * s));
  c.height = Math.max(1, Math.round(h * s));
  ctx.clearRect(0, 0, c.width, c.height);
  if (!w || !h) return null;
  try {
    ctx.drawImage(src, 0, 0, c.width, c.height);
  } catch {
    return null; // a compressed texture (KTX2 and the like) has no drawable image
  }
  if (ch === "orm") roughMetalOnly(ctx, c);
  return c;
}

/**
 * Clears red on a metallic-roughness preview. glTF reads only green (roughness) and blue
 * (metalness) there; red is occlusion when the texture doubles as one, which bakes write as 1
 * (none) and many originals leave 0, so the same material would look recolored.
 */
function roughMetalOnly(ctx, c) {
  const img = ctx.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < img.data.length; i += 4) img.data[i] = 0;
  ctx.putImageData(img, 0, 0);
}

/**
 * Draws the UV triangles of `model`'s parts whose material slot passes `keep` over canvas `c`,
 * each triangle moved into the 0-1 tile it starts in. Returns how many triangles there are
 * (each instance's).
 */
function drawUvs(c, model, keep = () => true) {
  const parts = (m) => (m.uvs ? m.parts.filter((p) => keep(p.slot)) : []);
  const used = new Set();
  let tris = 0;
  for (const inst of model?.instances ?? []) {
    const m = model.meshes[inst.mesh];
    const n = parts(m).reduce((a, p) => a + p.count / 3, 0);
    tris += n;
    if (n) used.add(m);
  }
  if (!c || !$("#uv-wire").checked || !tris || tris > UV_MAX_TRIANGLES) return tris;
  const { width: w, height: h } = c;
  const ctx = c.getContext("2d");
  ctx.save();
  // About one screen pixel wide, and inverting what is under it so it shows on any texture.
  ctx.lineWidth = Math.max(1, w / (c.clientWidth || 96));
  ctx.strokeStyle = "#fff";
  ctx.globalCompositeOperation = "difference";
  ctx.beginPath();
  for (const m of used) {
    const { uvs, indices } = m;
    for (const p of parts(m))
      for (let t = p.start; t < p.start + p.count; t += 3) {
        const [a, b, d] = [2 * indices[t], 2 * indices[t + 1], 2 * indices[t + 2]];
        const ou = Math.floor(Math.min(uvs[a], uvs[b], uvs[d]));
        const ov = Math.floor(Math.min(uvs[a + 1], uvs[b + 1], uvs[d + 1]));
        ctx.moveTo((uvs[a] - ou) * w, (uvs[a + 1] - ov) * h);
        ctx.lineTo((uvs[b] - ou) * w, (uvs[b + 1] - ov) * h);
        ctx.lineTo((uvs[d] - ou) * w, (uvs[d + 1] - ov) * h);
        ctx.closePath();
      }
  }
  ctx.stroke();
  ctx.restore();
  return tris;
}

/** The original's materials that have textures, by glTF material index. */
function texturedOriginals() {
  const t = original?.model.textures ?? [];
  const tex = (i) => (i >= 0 ? (t[i] ?? null) : null);
  return (original?.model.json ?? [])
    .map((m, index) => ({
      m,
      index,
      maps: { base_color: tex(m.base_color_texture), normal: tex(m.normal_texture), orm: tex(m.metallic_roughness_texture) },
    }))
    .filter(({ maps }) => maps.base_color || maps.normal || maps.orm);
}

const CHANNEL_NAMES = { base_color: "base color", normal: "normal", orm: "rough / metal" };

/** The panel for the original and Original-materials levels: a source material's textures, this level's UVs on them. */
function drawOriginalTextures(level) {
  const list = texturedOriginals();
  const pick = $("#bake-material");
  if (!list.some((e) => e.index === texMaterial)) texMaterial = list[0]?.index ?? -1;
  const label = (e) => e.m.name || `Material ${e.index}`;
  const key = list.map((e) => e.index).join();
  if (pick.dataset.key !== key) {
    pick.replaceChildren(...list.map((e) => new Option(label(e), String(e.index))));
    pick.dataset.key = key;
  }
  pick.hidden = list.length < 2;
  pick.value = String(texMaterial);
  const e = list.find((x) => x.index === texMaterial);
  const l = levels[level];
  $("#bake").classList.toggle("empty", !e);
  $("#bake-title").textContent = `Original textures · ${lodName(level)}${l?.stale ? " (updating)" : ""}`;
  if (!e) {
    setStats("No textures: plain-color materials.", ["The original has no textures: its materials are plain colors."]);
    return;
  }
  const maps = e.maps;
  const drawn = {};
  for (const [ch, t] of Object.entries(maps)) drawn[ch] = putTexture(ch, t ? drawable(t) : null);
  // Levels draw with the original's materials by slot: keep the slots sharing a texture with this one.
  const texIds = (m) => [m.base_color_texture, m.normal_texture, m.metallic_roughness_texture].filter((i) => i >= 0);
  const mine = new Set(texIds(e.m));
  const json = original.model.json;
  const keep = (slot) => !!json[slot] && texIds(json[slot]).some((i) => mine.has(i));
  const tris = drawUvs(drawn.base_color ?? drawn.normal ?? drawn.orm, l?.model, keep);
  const img = maps.base_color ?? maps.normal ?? maps.orm;
  const short = [];
  const details = [label(e)];
  if (img?.width) {
    short.push(`${img.width} × ${img.height}`);
    const big = img.width > PANEL_SIDE || img.height > PANEL_SIDE;
    details.push(`${img.width} × ${img.height}${big ? ` (shown at most ${PANEL_SIDE} across)` : ""}`);
  }
  const missing = Object.entries(maps)
    .filter(([, t]) => !t)
    .map(([ch]) => CHANNEL_NAMES[ch]);
  if (missing.length) details.push(`No ${missing.join(" or ")} map`);
  if (Object.entries(maps).some(([ch, t]) => t && !drawn[ch])) {
    short.push("compressed");
    details.push("Compressed textures can't be shown");
  }
  short.push(`${fmtInt(tris)} UV tris`);
  details.push(
    tris > UV_MAX_TRIANGLES && $("#uv-wire").checked
      ? `${fmtInt(tris)} triangles, too many to draw their UVs`
      : `UVs of ${fmtInt(tris)} triangles`,
  );
  if (list.length > 1) details.push(`${list.length} textured materials`);
  setStats(short.join(" · "), details);
}

$("#bake-material").addEventListener("change", (e) => {
  texMaterial = +e.target.value;
  drawTextures(texShown);
});
$("#uv-wire").addEventListener("change", () => drawTextures(texShown));

/** Unreal-style names of the downsample filters. */
const FILTER_NAMES = {
  nearest: "unfiltered",
  box: "simple average",
  triangle: "triangle",
  kaiser: "Kaiser",
  lanczos: "Lanczos",
};
const filterName = (f) => FILTER_NAMES[f] ?? (f?.startsWith("blur") ? `blur ${f.slice(4)}` : f);

function drawTextures(level) {
  if (shownMode === "hlod") return drawHlodTextures();
  texShown = level;
  const bake = $("#bake");
  if (usesOriginal(level)) {
    drawCooked(null);
    return drawOriginalTextures(level);
  }
  $("#bake-material").hidden = true;
  // Show this level's bake, else the most recent baked level.
  let l = levels[level];
  if (!l?.tex.base_color) {
    level = levels.findLastIndex((x) => x?.tex.base_color);
    l = levels[level];
  }
  bake.classList.toggle("empty", !l);
  drawCooked(l?.info);
  if (!l) {
    $("#bake-title").textContent = "Baked textures";
    setStats(
      form.elements.textures.value !== "recut" && +form.elements.bake_from.value === 0
        ? "No proxy levels (Bake proxy from: Never)."
        : "",
    );
    return;
  }
  for (const [ch, img] of Object.entries(l.tex)) putTexture(ch, img);
  drawUvs($("#tex-base_color"), l.model);
  const i = l.info;
  const shared = i?.kind === "shared";
  // A recut's levels carry labels; a shared chain's do not.
  const what = shared ? (i.label ? "Recut atlas" : "Shared atlas") : "Baked textures";
  $("#bake-title").textContent = `${what} · ${lodName(level)}${l.stale ? " (updating)" : ""}`;
  const short = [];
  const details = [];
  const k = i?.texture_halvings ?? 0;
  if (shared) {
    const density = i.density_scale / 2 ** k;
    short.push(`${i.atlas_width} × ${i.atlas_height}`, `${fmtPct(density)} density`);
    details.push(
      `${i.atlas_width} × ${i.atlas_height} atlas`,
      `${fmtPct(density)} of the ${i.view_texel_density ? "capped" : "artist's"} texel density`,
    );
  }
  if (k) {
    const sharpen = runOpts?.texture_sharpen ? `, sharpen ${runOpts.texture_sharpen}` : "";
    const from = shared ? `LOD${i.label ? 0 : 1}'s atlas` : "the proxy atlas";
    details.push(`Drawn with ${from} at 1/${2 ** k} the side (${filterName(i.texture_filter ?? runOpts?.texture_filter)}${sharpen})`);
  } else if (shared && i.shared_textures != null && i.shared_textures !== level)
    details.push(`Drawn with ${lodName(i.shared_textures)}'s atlas`);
  if (shared && i.view_texel_density)
    details.push(
      `Capped at ${fmtInt(Math.round(i.view_texel_density))} texels per unit, a texel per the error LOD${i.label ? 0 : 1} may have` +
        (i.view_texels != null ? ` (${fmtPct(i.view_texels)} of the artist's texels)` : ""),
    );
  if (i?.texture_error != null) {
    short.push(`Δ ${i.texture_error.toFixed(1)}/255`);
    details.push(texErrorText(i));
  }
  if (i?.coverage != null) {
    if (!shared) short.push(`${fmtPct(i.coverage)} covered`);
    short.push(`${fmtPct(i.utilization)} packed`);
    details.push(`Coverage ${fmtPct(i.coverage)}, ${i.charts} charts, utilization ${fmtPct(i.utilization)}`);
  }
  setStats(short.join(" · "), details);
}

rowsEl.addEventListener("click", (e) => {
  const dl = e.target.closest("[data-dl]");
  if (dl) {
    const level = +dl.dataset.dl;
    pendingDownload = level;
    if (busy) setStatus(`${lodName(level)} download starts when the run finishes…`, "busy");
    worker.postMessage({ type: "download", level, format: dl.dataset.fmt });
    return;
  }
  const tr = e.target.closest("tr[data-level]");
  if (tr) select(+tr.dataset.level);
});

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea") || !original) return;
  const n = levels.length - 1;
  if (e.key === "ArrowDown" || e.key === "]") select(Math.min(n, selected + 1));
  else if (e.key === "ArrowUp" || e.key === "[") select(Math.max(0, selected - 1));
  else if (e.key === "f" || e.key === "F") home();
  else return;
  e.preventDefault();
});

$("#home").addEventListener("click", () => home());

function setMode(m) {
  mode = m;
  for (const o of document.querySelectorAll("[data-mode]"))
    o.setAttribute("aria-pressed", String(o.dataset.mode === m));
  // Auto mode orbits the model's center: switch distances are measured from there, and zooming
  // out around a panned-away target would leave the model behind.
  const auto = m === "auto";
  if (orbit) {
    orbit.panEnabled = !auto;
    if (auto && original) {
      syncFocus();
      orbit.recenter();
    }
  }
  $("#dist-wrap").hidden = !auto;
}
for (const b of document.querySelectorAll("[data-mode]")) b.addEventListener("click", () => setMode(b.dataset.mode));

$("#wire").addEventListener("change", (e) => {
  wireOn = e.target.checked;
});
$("#seams").addEventListener("change", (e) => {
  seamOn = e.target.checked;
});

// ---------------------------------------------------------------- options

function readOptions() {
  const f = form.elements;
  return {
    levels: +f.levels.value,
    step: +f.step.value,
    spacing: f.spacing.value,
    error_ratio: +f.error_ratio.value,
    bake_from: +f.bake_from.value,
    atlas: +f.atlas.value,
    supersample: +f.supersample.value,
    planar_uv: f.planar_uv.checked,
    textures: f.textures.value,
    lod0_error: Math.max(0, +f.lod0_error.value || 0) / 100,
    max_error: Math.max(0, +f.max_error.value || 0) / 100,
    shared_size: +f.shared_size.value,
    view_distance: Math.max(0, +f.view_distance.value || 0),
    view_preset: f.view_preset.value,
    view_density: Math.max(0, +f.view_density.value || 0),
    view_density_preset: f.view_density_preset.value,
    view_fov: +f.view_fov.value,
    view_height: +f.view_height.value,
    shared_keep_layout: f.shared_keep_layout.checked,
    texture_shrink: f.texture_shrink.value,
    texture_filter: f.texture_filter.value,
    texture_sharpen: +f.texture_sharpen.value,
    texture_compression: f.texture_compression.value,
    compression_effort: +f.compression_effort.value,
    compression_lambda: +f.compression_lambda.value,
    fast: f.fast.checked,
    strict_seams: f.strict_seams.checked,
    symmetry: f.symmetry.checked,
    lock_border: f.lock_border.checked,
    mode: f.mode.value,
    preset_ratio: +f.preset_ratio.value,
    cluster_max_triangles: +f.cluster_max_triangles.value,
    cluster_max_vertices: +f.cluster_max_vertices.value,
    cluster_fast: f.cluster_fast.checked,
    hlod_copies: +f.hlod_copies.value,
    hlod_split: +f.hlod_split.value,
    hlod_triangles: +f.hlod_triangles.value,
    hlod_bake: f.hlod_bake.checked,
    gpu_visibility: f.gpu_visibility.checked,
    splat_count: +f.splat_count.value,
    splat_footprints: f.splat_footprints.checked,
    crowd_bones: +f.crowd_bones.value,
    crowd_ratio: +f.crowd_ratio.value,
  };
}

/** A cluster layout preset fills in the two limits; it names no engine in the options.
 *  Custom keeps the limits as they are and shows them for editing. */
function applyClusterPreset() {
  const f = form.elements;
  const [tris, verts] = f.cluster_preset.value.split("x");
  if (verts === undefined) return;
  f.cluster_max_triangles.value = tris;
  f.cluster_max_vertices.value = verts;
}

/** Shows the option groups of the chosen example. */
function syncModeUi() {
  const m = form.elements.mode.value;
  for (const fs of form.querySelectorAll("fieldset[data-modes]")) fs.hidden = !fs.dataset.modes.split(" ").includes(m);
  const custom = form.elements.cluster_preset.value === "custom";
  for (const el of form.querySelectorAll(".cluster-custom")) el.hidden = !custom;
  const tex = form.elements.textures.value;
  $("#shared-size").hidden = tex === "original";
  $("#shared-layout").hidden = tex !== "shared";
  $("#lod0-error").hidden = $("#lod0-hint").hidden = tex !== "recut";
  // A recut bakes nothing per level.
  for (const el of form.querySelectorAll(".baked-only")) el.hidden = tex === "recut";
  const shrink = form.elements.texture_shrink.value;
  $("#tex-filter").hidden = $("#tex-sharpen").hidden = shrink === "keep";
  const only =
    tex === "original" ? "Only the baked proxies shrink: the reduced levels before them draw with the original materials. " : "";
  $("#shrink-hint").textContent =
    shrink === "keep"
      ? "Every level samples LOD0's atlas at full size."
      : only +
        (shrink === "halve"
          ? "Each level stores the atlas at half the side of the level before: a quarter of the memory, one mip of LOD bias per level."
          : "Each level drops whole mips for as long as a texel still covers at most a pixel where the level switches in.");
  const cook = form.elements.texture_compression.value;
  $("#cook-effort").hidden = $("#cook-rdo").hidden = cook === "off";
  const rdo = +form.elements.compression_lambda.value > 0;
  $("#cook-hint").textContent =
    cook === "off"
      ? "Atlases and proxies stay RGBA8: 32 bits a texel. Pick a target to see them as they would ship."
      : "Every atlas and proxy texture is encoded with its mips, then drawn decoded, so the viewer shows the real artifacts; " +
        "each row's .zip holds the level as it ships (compact GLB, KTX2 with Zstd). " +
        (tex === "original" ? "Reduced levels before the proxies keep the original materials uncompressed. " : "") +
        "The browser encodes on one thread, so large atlases take a while" +
        (rdo ? "; with rate-distortion on, expect tens of seconds per 512² texture." : ".");
  $("#mode-hint").textContent = MODE_HINTS[m] ?? "";
  syncViewHint();
}

/** What the LOD0 error is for this model, in its own units. */
function syncLod0Hint() {
  const e = Math.max(0, +form.elements.lod0_error.value || 0) / 100;
  $("#lod0-hint").textContent = !e
    ? "LOD0 keeps every triangle and the artist's texel density."
    : assetInfo
      ? `LOD0 may deviate ${fmtLen(e * assetInfo.diagonal)}, and keeps at most ${fmtInt(Math.round(1 / (e * assetInfo.diagonal)))} texels per unit.`
      : `LOD0 may deviate ${(e * 100).toPrecision(2)}% of the model's size, with a texel per that distance at most.`;
}

/** What the viewing budget means for this model: its size, and what a pixel is at the closest view. */
function syncViewHint() {
  syncLod0Hint();
  const { view_distance: d, view_fov: fov, view_height: h, view_preset, view_density: ppm } = readOptions();
  const density = view_preset === "density";
  $("#view-density-preset").hidden = $("#view-density").hidden = !density;
  const size = assetInfo ? `The model is ${fmtLen(assetInfo.diagonal)} across (bounding-box diagonal). ` : "";
  if (!d) {
    $("#view-hint").textContent =
      `${size}Pick how much of the screen the model fills when the camera is closest, a texel density target, ` +
      "or type a distance, to drop detail that view cannot show. A prop the player walks up to fills about the " +
      "screen or half of it; scenery that stays in the distance a tenth or less.";
    return;
  }
  const ppu = h / (2 * d * Math.tan(((fov * Math.PI) / 180) / 2));
  const px = assetInfo ? assetInfo.diagonal * ppu : 0;
  if (density) {
    const there = assetInfo
      ? ` There the model spans about ${fmtInt(Math.round(px))} px, ${Math.round((px * 100) / h)}% of the screen height.`
      : "";
    const kept = levels[0]?.stale ? null : levels[0]?.info?.view_texels;
    const texels = kept != null ? ` That keeps ${(kept * 100).toPrecision(2)}% of the artist's texels.` : "";
    $("#view-hint").textContent =
      `${size}${fmtInt(ppm)} px/m (${+(ppm / 100).toPrecision(4)} px/cm, a texel every ${fmtLen(1 / ppm)}) ` +
      `is a texel per pixel from ${fmtLen(d)} on a ${h}-pixel-high screen at ${fov}°.${there}${texels}`;
    return;
  }
  const spans = assetInfo ? ` it spans about ${fmtInt(Math.round(px))} px, ${Math.round((px * 100) / h)}% of the screen height, and` : "";
  const kept = levels[0]?.stale ? null : levels[0]?.info?.view_texels;
  const texels = kept != null ? ` That view shows ${(kept * 100).toPrecision(2)}% of the artist's texels.` : "";
  $("#view-hint").textContent = `${size}From ${fmtLen(d)}${spans} a pixel is ${fmtLen(1 / ppu)} (${fmtInt(Math.round(ppu))} px per unit).${texels}`;
}

/**
 * The closest view's preset: the distance at which the model's bounding-box diagonal fills that
 * fraction of the screen height, rounded to two significant digits.
 */
function applyViewPreset() {
  const f = form.elements;
  if (f.view_preset.value === "custom") return;
  const tan = Math.tan(((+f.view_fov.value * Math.PI) / 180) / 2);
  if (f.view_preset.value === "density") {
    // The distance at which a screen pixel covers one texel of that density.
    const ppm = +f.view_density.value;
    f.view_distance.value = ppm > 0 ? String(+(+f.view_height.value / (2 * ppm * tan)).toPrecision(3)) : "0";
    return;
  }
  const x = +f.view_preset.value;
  if (!x) {
    f.view_distance.value = "0";
    return;
  }
  if (!assetInfo?.diagonal) return;
  const d = assetInfo.diagonal / (2 * x * tan);
  f.view_distance.value = String(+d.toPrecision(2));
}

/** A texel density preset ("512@1080"): its density, and the screen it is meant for. */
function applyDensityPreset() {
  const f = form.elements;
  const [ppm, h] = f.view_density_preset.value.split("@");
  if (!h) return;
  f.view_density.value = ppm;
  f.view_height.value = h;
}

/** A length in scene units, assumed meters (glTF). */
function fmtLen(m) {
  if (m >= 1) return `${m.toPrecision(3)} m`;
  if (m >= 0.01) return `${(m * 100).toPrecision(3)} cm`;
  return `${(m * 1000).toPrecision(3)} mm`;
}

let lastLevels = 4;
function syncOutputs() {
  const f = form.elements;
  const n = +f.levels.value;
  const sel = f.bake_from;
  const prev = sel.options.length ? +sel.value : n;
  const want = prev === 0 ? 0 : prev === lastLevels ? n : Math.min(prev, n);
  sel.innerHTML =
    `<option value="0">Never</option>` +
    Array.from({ length: n }, (_, i) => `<option value="${i + 1}">LOD${i + 1}</option>`).join("");
  sel.value = String(want);
  lastLevels = n;
  f.levels_out.value = String(n);
  f.step_out.value = `${Math.round(+f.step.value * 100)}%`;
  f.error_ratio_out.value = `${+f.error_ratio.value}×`;
  const byError = f.spacing.value === "error";
  for (const el of form.querySelectorAll(".by-error")) el.hidden = !byError;
  for (const el of form.querySelectorAll(".by-triangles")) el.hidden = byError;
  f.supersample_out.value = `${f.supersample.value}×${f.supersample.value}`;
  f.preset_ratio_out.value = `${Math.round(+f.preset_ratio.value * 100)}%`;
  f.texture_sharpen_out.value = f.texture_sharpen.value === "0" ? "off" : f.texture_sharpen.value;
  syncModeUi();
}

function restoreOptions() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(STORE_KEY));
  } catch {}
  if (saved) {
    if (saved.version !== STORE_VERSION) delete saved.textures;
    // Saved without the cluster preset: the one the limits match, else Custom.
    if (saved.cluster_preset === undefined && saved.cluster_max_triangles !== undefined) {
      const key = `${saved.cluster_max_triangles}x${saved.cluster_max_vertices}`;
      const known = [...form.elements.cluster_preset.options].some((o) => o.value === key);
      saved.cluster_preset = known ? key : "custom";
    }
    // Saved before the presets: a distance was typed.
    if (saved.view_preset === undefined) saved.view_preset = saved.view_distance > 0 ? "custom" : "0";
    const f = form.elements;
    for (const [k, v] of Object.entries(saved)) {
      const el = f[k];
      if (!el || k === "bake_from" || k === "version") continue;
      if (el.type === "checkbox") el.checked = !!v;
      else el.value = String(v);
    }
    lastLevels = +f.levels.value;
    applyClusterPreset();
    syncOutputs();
    if (saved.bake_from !== undefined) f.bake_from.value = String(Math.min(saved.bake_from, lastLevels));
  } else syncOutputs();
}

let debounce = 0;
form.addEventListener("input", (e) => {
  // A typed distance is a custom one; a preset or another field of view moves the distance.
  // A density target sets the distance from the density, field of view and screen.
  const f = form.elements;
  const name = e.target.name;
  if (name === "view_distance") f.view_preset.value = +e.target.value ? "custom" : "0";
  else {
    if (name === "view_density_preset") applyDensityPreset();
    else if (name === "view_density") f.view_density_preset.value = "custom";
    else if (name === "view_height" && !f.view_density_preset.value.endsWith(`@${f.view_height.value}`))
      f.view_density_preset.value = "custom";
    if (name.startsWith("view_density")) f.view_preset.value = "density";
    if (name === "cluster_preset") applyClusterPreset();
    if (["view_preset", "view_fov", "view_height"].includes(name) || name.startsWith("view_density")) applyViewPreset();
  }
  syncOutputs();
  try {
    // The preset is page state only; the run takes the two limits.
    const cluster_preset = form.elements.cluster_preset.value;
    localStorage.setItem(STORE_KEY, JSON.stringify({ ...readOptions(), cluster_preset, version: STORE_VERSION }));
  } catch {}
  renderRows();
  clearTimeout(debounce);
  debounce = setTimeout(startRun, 300);
});
form.addEventListener("submit", (e) => e.preventDefault());
restoreOptions();

// ---------------------------------------------------------------- input: drop, picker, samples

async function readEntries(items) {
  // Entries must be taken synchronously, before the first await.
  const entries = [...items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  const out = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ name: prefix + file.name, bytes: await file.arrayBuffer() });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, `${prefix}${entry.name}/`);
      }
    }
  };
  for (const e of entries) await walk(e, "");
  return out;
}

const fromFiles = (list) =>
  Promise.all([...list].map(async (f) => ({ name: f.name, bytes: await f.arrayBuffer() })));

let dragDepth = 0;
addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragDepth++;
  document.body.classList.add("dragging");
});
addEventListener("dragleave", () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    document.body.classList.remove("dragging");
  }
});
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", async (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  const dt = e.dataTransfer;
  const pending = dt.items?.length && dt.items[0].webkitGetAsEntry ? readEntries(dt.items) : fromFiles(dt.files);
  try {
    await load(await pending);
  } catch (err) {
    setStatus(`Could not read the dropped files: ${err.message ?? err}`, "error");
  }
});

const picker = $("#file");
$("#pick").addEventListener("click", () => picker.click());
$("#drop").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target === e.currentTarget) picker.click();
});
picker.addEventListener("change", async () => {
  await load(await fromFiles(picker.files));
  picker.value = "";
});

const mb = (bytes) => `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;

/** Fetches `url`, calling `progress(received, total)` as it streams in. */
async function download(url, signal, progress, expected = 0) {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const total = +r.headers.get("content-length") || expected;
  if (!r.body || !total) return r.arrayBuffer();
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    progress(got, Math.max(total, got));
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out.buffer;
}

// Only one download at a time: picking another model cancels the one in flight.
let fetching = null;
async function loadUrl(url, name, expected, onProgress = () => {}, thumb = null) {
  fetching?.abort();
  const ctl = (fetching = new AbortController());
  setStatus(`Downloading ${name}…`, "busy");
  showLoading(name, thumb);
  loadingProgress("Downloading…", 0, expected);
  try {
    const bytes = await download(url, ctl.signal, (got, total) => {
      setStatus(`Downloading ${name}… ${mb(got)} of ${mb(total)}`, "busy");
      loadingProgress("Downloading…", got, total);
      onProgress(got / total);
    }, expected);
    if (ctl.signal.aborted) return;
    const file = decodeURIComponent(new URL(url, location.href).pathname.split("/").pop());
    await load([{ name: file, bytes }], name);
  } catch (err) {
    if (!ctl.signal.aborted) {
      hideLoading();
      setStatus(`Could not download ${name}: ${err.message ?? err}`, "error");
    }
  } finally {
    if (fetching === ctl) fetching = null;
    onProgress(null);
  }
}

// Sample models: CC0 and CC BY assets that fetch_samples.py downloads into samples/ at build
// time, listed in samples/index.json, each shown as a card in the sidebar; an attributed one
// shows its attribution while it's open. Without the index (a local build that skipped the
// script) the section stays hidden.
const cards = new Map(); // sample file -> its card button
function card(s) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "card";
  b.title = `${s.name} · ${mb(s.bytes)} · ${s.attribution ?? `by ${s.credit}`} · ${s.license}`;
  b.innerHTML =
    (s.thumb ? `<img src="samples/${esc(s.thumb)}" alt="" loading="lazy">` : `<span class="noimg"></span>`) +
    `<span class="name">${esc(s.name)}</span><span class="size">${mb(s.bytes)}</span><span class="bar"></span>`;
  b.addEventListener("click", () => loadSample(s));
  return b;
}

/** The open sample's attribution, with links to its source and license, when it needs one. */
function showAttribution(s) {
  const el = $("#sample-attribution");
  el.hidden = !s?.attribution;
  if (el.hidden) return;
  const link = (href, text) => `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(text)}</a>`;
  el.innerHTML =
    `${link(s.source, s.name)}: ${esc(s.attribution)} ` + link(s.license_url, s.license.replace(/-/g, " "));
}

function loadSample(s) {
  for (const [file, b] of cards) b.setAttribute("aria-current", String(file === s.file));
  showAttribution(s);
  const b = cards.get(s.file);
  loadUrl(
    `samples/${s.file}`,
    s.name,
    s.bytes,
    (p) => {
      b.classList.toggle("loading", p !== null);
      b.style.setProperty("--p", p ?? 0);
    },
    s.thumb ? `samples/${s.thumb}` : null,
  );
}

fetch("samples/index.json")
  .then((r) => (r.ok ? r.json() : []))
  .catch(() => [])
  .then((samples) => {
    if (!samples.length) return;
    for (const s of samples) {
      const b = card(s);
      $("#sample-cards").append(b);
      cards.set(s.file, b);
    }
    $("#samples").hidden = false;
  });

// Choosing or dropping a file of your own cancels a sample download and clears its highlight.
const clearCurrent = () => {
  showAttribution(null);
  fetching?.abort();
  for (const b of cards.values()) b.removeAttribute("aria-current");
};
picker.addEventListener("change", clearCurrent);
addEventListener("drop", clearCurrent);

// ?model=<url> loads a .glb (or .zip) on start, e.g. index.html?model=samples/Lantern.glb
const startUrl = new URLSearchParams(location.search).get("model");
if (startUrl) loadUrl(startUrl, startUrl, 0);

// Under the hint from the start, before the first bytes arrive (here, once `mb` and the overlay exist).
showEngine();
