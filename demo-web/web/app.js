// farthing LOD demo: page logic. The wasm runs in worker.js; this file handles input, the
// options, the three.js viewer, the level table and the example modes (cluster cuts, HLOD
// cuts, splats).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { Wireframe } from "three/addons/lines/Wireframe.js";
import { WireframeGeometry2 } from "three/addons/lines/WireframeGeometry2.js";

const $ = (s) => document.querySelector(s);
const form = $("#opts");
const statusEl = $("#status");
const rowsEl = $("#level-rows");

const STORE_KEY = "farthing-demo-options";

// ---------------------------------------------------------------- state

let files = null; // [{ name, bytes: ArrayBuffer }], kept so a respawned worker can reload
let assetInfo = null;
let original = null; // { object, materials: Material[], cache: Map, center, radius }
let worker = null;
let busy = false; // the worker is loading or running
let runId = 0;
let runStart = 0;
let runOpts = null; // options of the current run
let selected = 0;
let mode = "single";
let wireOn = false;
/** levels[n] = { info, object, tex: { channel: ImageData }, stale } */
let levels = [];
let pendingDownload = null;

// ---------------------------------------------------------------- viewer

const view = $("#view");
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(2, 1.5, 3);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
const root = new THREE.Group(); // one child per level (single and compare modes)
const lod = new THREE.LOD(); // auto mode: clones of the levels
lod.visible = false;
scene.add(root, lod);

// Wireframe: screen-space "fat" lines (plain GL lines are 1 device pixel, too thin to read on a
// high-DPI screen), drawn over surfaces that polygonOffset pushes back so the two don't z-fight.
// Its resolution must match the viewport, so render() sets it for each one.
const dark = matchMedia("(prefers-color-scheme: dark)");
const wireMat = new LineMaterial({ linewidth: 1.25, transparent: true, depthWrite: false });
const setWireColor = () => {
  wireMat.color.set(dark.matches ? 0x8fbcff : 0x1453c4);
  wireMat.opacity = dark.matches ? 0.75 : 0.7;
};
setWireColor();
dark.addEventListener("change", setWireColor);

const loader = new GLTFLoader();

/** The camera distance that fits a sphere of `radius` in view. */
const homeDistance = (radius) => (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.05;

function frame(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-6);
  controls.maxDistance = radius * 400;
  home({ center, radius });
  return { center, radius };
}

/** Points the camera at the model from the default angle (also: F, double-click, Reset view). */
function home(at = focus()) {
  if (!at) return;
  const dir = new THREE.Vector3(0.6, 0.35, 1).normalize();
  camera.position.copy(at.center).addScaledVector(dir, homeDistance(at.radius));
  controls.target.copy(at.center);
  controls.update();
}

/**
 * Per frame: keep the orbit target within reach of the model, so panning can't lose it, and fit
 * near/far tightly around the model's bounding sphere for depth precision at any distance.
 */
function constrainCamera() {
  const { center, radius } = focus();
  const off = controls.target.clone().sub(center);
  if (off.length() > radius * 1.5) {
    const shift = off.clone().setLength(radius * 1.5).sub(off);
    controls.target.add(shift);
    camera.position.add(shift);
  }
  const dist = camera.position.distanceTo(center);
  camera.near = Math.max(radius / 500, (dist - radius) * 0.5);
  camera.far = dist + radius * 2;
}

function setWire(object) {
  object.traverse((o) => {
    if (!o.isMesh || o.name === "__wire" || o.userData.noWire) return;
    let w = o.children.find((c) => c.name === "__wire");
    if (!w && wireOn) {
      w = new Wireframe(wireFor(o.geometry), wireMat);
      w.name = "__wire";
      w.renderOrder = 1;
      o.add(w);
    }
    if (w) w.visible = wireOn;
    for (const m of [o.material].flat()) {
      m.polygonOffset = wireOn;
      m.polygonOffsetFactor = 1;
      m.polygonOffsetUnits = 1;
    }
  });
}

function dispose(object, ownMaterials) {
  object?.traverse((o) => {
    dropWire(o.geometry);
    o.geometry?.dispose();
    if (ownMaterials && o.material && o.material !== wireMat) {
      for (const k of ["map", "normalMap"]) o.material[k]?.dispose();
      o.material.dispose();
    }
  });
  object?.removeFromParent();
}

/** The original's material for glTF material `index`, in the variant `like` needs. */
function originalMaterial(index, like) {
  const key = `${index}:${!!like.vertexColors}:${!!like.flatShading}`;
  let m = original.cache.get(key);
  if (!m) {
    const base = original.materials[index];
    if (!base) return like;
    m = base.clone();
    m.vertexColors = like.vertexColors;
    m.flatShading = like.flatShading;
    original.cache.set(key, m);
  }
  return m;
}

/** Reduced levels are exported without textures: give them the original's materials. */
function adoptMaterials(gltf) {
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const index = gltf.parser.associations.get(o.material)?.materials;
    if (index === undefined) return;
    const mine = o.material;
    o.material = originalMaterial(index, mine);
    if (o.material !== mine) mine.dispose();
  });
}

async function showOriginal(glb) {
  const gltf = await loader.parseAsync(glb, "");
  const materials = await gltf.parser.getDependencies("material");
  const nodes = await gltf.parser.getDependencies("node");
  original = { object: gltf.scene, materials, cache: new Map(), nodes, assoc: gltf.parser.associations };
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const index = gltf.parser.associations.get(o.material)?.materials;
    if (index !== undefined)
      original.cache.set(`${index}:${!!o.material.vertexColors}:${!!o.material.flatShading}`, o.material);
  });
  Object.assign(original, frame(gltf.scene));
  root.add(gltf.scene);
  setWire(gltf.scene);
  levels[0] = { info: null, object: gltf.scene, tex: {}, stale: false };
}

function resetViewer() {
  extra?.dispose();
  for (const [i, l] of levels.entries()) if (i > 0) disposeLevel(l);
  extra = null;
  hlodIncoming = null;
  focusOverride = null;
  shownMode = "";
  if (original) {
    dispose(original.object, true);
    for (const m of original.cache.values()) m.dispose();
  }
  for (const w of wires.values()) w.dispose();
  wires.clear();
  lod.clear();
  levels = [];
  original = null;
  selected = 0;
  syncCutPanel();
}

// Auto mode: THREE.LOD with each level's 1-pixel switch distance, scaled from the 1080p the
// numbers are computed for to this canvas's height.
let lodKey = "";
function syncLod() {
  const h = renderer.getDrawingBufferSize(new THREE.Vector2()).y;
  const ready = levels.filter((l) => l?.object && l.info && !l.stale);
  const key = ready.map((l) => l.object.uuid).join() + h + wireOn + lodVersion;
  if (key === lodKey) return;
  lodKey = key;
  lod.clear();
  lod.levels.length = 0;
  if (!original) return;
  lod.position.copy(original.center);
  let d = 0;
  for (const l of ready) {
    d = Math.max(d, (l.info.switch_distance_1px * h) / 1080);
    const g = new THREE.Group();
    g.position.copy(original.center).negate();
    // The level may be hidden in root (show() hides all but one); the clone copies that flag.
    const copy = l.object.clone();
    copy.visible = true;
    g.add(copy);
    g.userData.level = l.info.level;
    lod.addLevel(g, d);
  }
  // Room to zoom out past the last switch, wherever it is.
  controls.maxDistance = Math.max(original.radius * 400, d * 2);
  drawTicks();
}

// Auto mode's distance slider: logarithmic, from close up to the zoom limit.
const distEl = $("#dist");
function distRange() {
  return [homeDistance(original.radius) * 0.5, controls.maxDistance];
}
function distToSlider(d) {
  const [lo, hi] = distRange();
  return (1000 * Math.log(Math.max(d, lo) / lo)) / Math.log(hi / lo);
}
function drawTicks() {
  $("#ticks").innerHTML = lod.levels
    .slice(1)
    .map((l) => {
      const t = Math.min(distToSlider(l.distance) / 1000, 1);
      return `<i style="--t:${t}" title="LOD${l.object.userData.level} from ${l.distance.toPrecision(3)}"></i>`;
    })
    .join("");
}
distEl.addEventListener("input", () => {
  if (!original) return;
  const [lo, hi] = distRange();
  const d = lo * (hi / lo) ** (+distEl.value / 1000);
  const dir = camera.position.clone().sub(controls.target).normalize();
  camera.position.copy(controls.target).addScaledVector(dir, d);
});

// Auto mode's magnifier: far away the model is a few pixels, so this crops the frame just drawn
// around it and scales it up without smoothing, to show the rendered pixels themselves.
const magEl = $("#mag");
const magCanvas = magEl.querySelector("canvas");
const magCtx = magCanvas.getContext("2d");
function magnify() {
  const size = Math.round(160 * renderer.getPixelRatio());
  const buf = renderer.getDrawingBufferSize(new THREE.Vector2());
  const p = original.center.clone().project(camera);
  const dist = camera.position.distanceTo(original.center);
  const rpx = (original.radius / (dist * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) * (buf.y / 2);
  const zoom = Math.floor(size / Math.max(2 * rpx * 1.15, 4));
  const onScreen = Math.abs(p.x) < 1.2 && Math.abs(p.y) < 1.2 && p.z < 1;
  magEl.hidden = zoom < 2 || !onScreen;
  if (magEl.hidden) return;
  if (magCanvas.width !== size) magCanvas.width = magCanvas.height = size;
  const src = size / zoom;
  const x = ((p.x + 1) / 2) * buf.x - src / 2;
  const y = ((1 - p.y) / 2) * buf.y - src / 2;
  magCtx.imageSmoothingEnabled = false;
  magCtx.clearRect(0, 0, size, size);
  magCtx.drawImage(canvas, x, y, src, src, 0, 0, size, size);
  magEl.querySelector("figcaption").textContent = `${zoom}× · rendered pixels`;
}

function displayed(level) {
  return levels[level]?.object ?? levels[0]?.object;
}

function show(object) {
  for (const c of root.children) c.visible = c === object;
}

function resize() {
  const w = view.clientWidth;
  const h = view.clientHeight;
  const size = renderer.getSize(new THREE.Vector2());
  if (size.x !== w || size.y !== h) renderer.setSize(w, h, false);
  return [w, h];
}

let hudText = "";
function hud(left, right = "") {
  const t = left + "|" + right;
  if (t === hudText) return;
  hudText = t;
  $("#hud").innerHTML = (left ? `<span>${left}</span>` : "<i></i>") + (right ? `<span>${right}</span>` : "");
}

function label(level) {
  const l = levels[level];
  if (!l?.info) return `LOD${level}`;
  const i = l.info;
  const stale = l.stale ? " (updating)" : "";
  if (i.kind === "splats") return `LOD${level} · ${fmtInt(i.vertices)} Gaussian splats${stale}`;
  const name = i.label ? esc(i.label) : `LOD${level}`;
  const more = i.hud ? ` · ${esc(i.hud)}` : "";
  return `${name} · ${KIND_TEXT[i.kind] ?? i.kind} · ${fmtInt(i.triangles)} tris${more}${stale}`;
}

function render() {
  requestAnimationFrame(render);
  controls.update();
  const [w, h] = resize();
  if (!original) {
    hud("");
    magEl.hidden = true;
    renderer.clear();
    return;
  }
  constrainCamera();
  requestCut();
  sortSplats();
  root.visible = mode !== "auto";
  lod.visible = mode === "auto";
  if (mode !== "auto") magEl.hidden = true;
  if (mode === "compare") {
    const half = Math.floor(w / 2);
    camera.aspect = half / h;
    camera.updateProjectionMatrix();
    wireMat.resolution.set(half, h);
    renderer.setScissorTest(true);
    show(displayed(0));
    renderer.setViewport(0, 0, half, h);
    renderer.setScissor(0, 0, half, h);
    renderer.render(scene, camera);
    show(displayed(selected));
    renderer.setViewport(half, 0, w - half, h);
    renderer.setScissor(half, 0, w - half, h);
    renderer.render(scene, camera);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    hud(label(0), label(selected));
  } else {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    wireMat.resolution.set(w, h);
    if (mode === "auto") {
      syncLod();
      renderer.render(scene, camera);
      magnify();
      const i = lod.getCurrentLevel();
      const cur = lod.levels[i]?.object.userData.level ?? 0;
      const next = lod.levels[i + 1];
      const dist = camera.position.distanceTo(original.center);
      if (document.activeElement !== distEl) distEl.value = String(distToSlider(dist));
      const ahead = next ? ` · LOD${next.object.userData.level} at ${next.distance.toPrecision(3)}` : "";
      hud(`Auto: ${label(cur)}`, `distance ${dist.toPrecision(3)}${ahead}`);
    } else {
      show(displayed(selected));
      renderer.render(scene, camera);
      hud(label(levels[selected]?.object ? selected : 0));
    }
  }
  $("#divider").hidden = mode !== "compare";
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
// Besides the classic chain, the page runs four other examples (the Example picker):
// option configurations (GLB levels, like the chain), a cluster DAG and an HLOD tree (one
// "Cut" level the worker re-selects as the camera moves) and Gaussian splats (drawn here).

const MODE_KIND = { presets: "reduced", clusters: "clusters", hlod: "hlod", splats: "splats" };
const MODE_LEVELS = { presets: 10, clusters: 1, hlod: 1, splats: 3 };
const KIND_TEXT = { clusters: "cluster DAG", hlod: "HLOD" };
const MODE_HINTS = {
  chain: "Reduced levels, then baked proxies, each built from the last. Auto LOD switches them by distance.",
  presets:
    "The same reduction under named option configurations. Pick a row to view it, or Compare it with the original.",
  clusters:
    "A Nanite-style cluster DAG per mesh. Each cluster is kept or swapped for a coarser one by its own error on screen: zoom in and out and watch the cut follow.",
  hlod: "The model tiled on a grid and split into cells; each cell, and each group of cells above it, gets a merged proxy. Far cells draw their proxy instead.",
  splats: "A progressive surfel cloud as Gaussian splats: the whole cloud, its first quarter and its first sixteenth.",
};
const AUTO_MODES = new Set(["chain", "splats"]);

let shownMode = ""; // the example whose results are on screen ("" before the first run)
/** Cluster or HLOD state: { kind, group, info, cut(sel, buffers), stats(), dispose(), full? } */
let extra = null;
let hlodIncoming = null; // HLOD proxies parsed while the tree builds
let focusOverride = null; // HLOD: the tiled grid's bounds replace the model's for the camera
let cutSeq = 0;
let cutInflight = false;
let cutKey = "";
let lodVersion = 0;
const wires = new Map(); // geometry -> its wireframe, shared by every mesh drawing it
const plain = { vertexColors: false, flatShading: false };
const tmpV2 = new THREE.Vector2();
const tmpV3 = new THREE.Vector3();
const tmpV4 = new THREE.Vector4();

function focus() {
  return focusOverride ?? original;
}

function fmtBytes(n) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)} MB` : `${(n / 1e3).toFixed(n < 1e4 ? 1 : 0)} KB`;
}
const fmtPx = (p) => (p < 1 ? p.toFixed(2) : p.toPrecision(2));
const diag = () => assetInfo?.diagonal || 1;

function wireFor(geometry) {
  let w = wires.get(geometry);
  if (!w) {
    w = new WireframeGeometry2(geometry);
    wires.set(geometry, w);
  }
  return w;
}
function dropWire(geometry) {
  const w = geometry && wires.get(geometry);
  if (w) {
    w.dispose();
    wires.delete(geometry);
  }
}

const tint = new THREE.Color();
/** A color per DAG or tree level (shared object: read it right away). */
function levelColor(l) {
  return tint.setHSL((0.58 + l * 0.137) % 1, 0.62, 0.55);
}
/** A stable random color per cluster (shared object: read it right away). */
function clusterColor(id) {
  let h = Math.imul(id ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return tint.setHSL((h >>> 0) / 4294967296, 0.5 + ((h >>> 8) & 63) / 250, 0.45 + ((h >>> 16) & 63) / 400);
}

function disposeLevel(l) {
  if (!l?.object) return;
  if (l.object.userData.dispose) {
    l.object.userData.dispose();
    l.object.removeFromParent();
  } else dispose(l.object, l.info?.kind === "baked");
  l.object = null;
}

/** Drops everything a mode put on screen; level 0 is the original again. */
function clearModeState() {
  for (let i = levels.length - 1; i > 0; i--) disposeLevel(levels[i]);
  levels.length = Math.min(levels.length, 1);
  extra?.dispose();
  extra = null;
  hlodIncoming = null;
  if (levels[0] && original) levels[0].object = original.object;
  if (focusOverride) {
    focusOverride = null;
    if (original) {
      controls.maxDistance = original.radius * 400;
      home();
    }
  }
  cutKey = "";
  lodKey = "";
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
  $("#cut").hidden = !(m === "clusters" || m === "hlod" || m === "splats");
  for (const el of document.querySelectorAll("[data-cut]")) el.hidden = !el.dataset.cut.split(" ").includes(m);
  $("#cut-title").textContent = { clusters: "Cluster DAG cut", hlod: "HLOD cut", splats: "Splats" }[m] ?? "";
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
  if (!extra || busy || cutInflight || extra.failed || $("#freeze").checked) return;
  const h = renderer.getDrawingBufferSize(tmpV2).y;
  const ppr = h / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  const threshold = 2 ** +$("#threshold").value;
  const p = camera.position;
  const key = `${p.x.toPrecision(5)},${p.y.toPrecision(5)},${p.z.toPrecision(5)}|${ppr.toFixed(0)}|${threshold}`;
  if (key === cutKey) return;
  cutKey = key;
  cutInflight = true;
  worker.postMessage({ type: "select", seq: ++cutSeq, view: { eye: [p.x, p.y, p.z], ppr, threshold } });
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
  Object.assign(l1, { info: x.info, object: x.group, stale: false, tex: {} });
  root.add(x.group);
  setWire(x.group);
  if (x.full) {
    const l0 = entry(0);
    l0.object = x.full;
    l0.info = { ...(l0.info ?? { level: 0, kind: "original" }), ...x.fullInfo };
    l0.stale = false;
    root.add(x.full);
    setWire(x.full);
  }
  cutKey = "";
  syncCutPanel();
  renderRows();
  drawTextures(selected);
}

async function onData(m) {
  const { kind, info, buffers } = m;
  const id = m.runId;
  if (kind === "clusters") installExtra(buildClusters(info, buffers));
  else if (kind === "hlod_node") {
    if (hlodIncoming?.runId !== id) hlodIncoming = { runId: id, proxies: new Map(), pending: [] };
    const inc = hlodIncoming;
    const u8 = buffers[0];
    inc.pending.push(
      loader.parseAsync(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength), "").then((gltf) => {
        if (id !== runId) return dispose(gltf.scene, true);
        inc.proxies.set(info.node, { object: gltf.scene, info });
      }),
    );
  } else if (kind === "hlod_tree") {
    const inc = hlodIncoming?.runId === id ? hlodIncoming : { proxies: new Map(), pending: [] };
    await Promise.all(inc.pending);
    if (id !== runId) return;
    hlodIncoming = null;
    installExtra(buildHlod(info, inc.proxies));
  } else if (kind === "splats") {
    const l = entry(info.level);
    disposeLevel(l);
    const mesh = buildSplats(buffers[0]);
    Object.assign(l, { info, object: mesh, stale: false, tex: {} });
    root.add(mesh);
    renderRows();
    cutStats();
  }
}

// ---- cluster DAG

/** Draws any cut of the DAG: every cluster's triangles arrive once, indexing the mesh's vertices. */
function buildClusters(info, buffers) {
  const [xf, instMesh, ...rest] = buffers;
  const meshes = info.meshes.map((json, i) => {
    const [pos, nrm, uv, idx, triStart, mats, clusterLevel] = rest.slice(7 * i, 7 * i + 7);
    const position = new THREE.BufferAttribute(pos, 3);
    let normal = nrm.length ? new THREE.BufferAttribute(nrm, 3) : null;
    if (!normal) {
      // Smooth normals from the full-detail clusters.
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", position);
      const full = [];
      for (let c = 0; c < clusterLevel.length; c++)
        if (clusterLevel[c] === 0)
          for (let t = triStart[c]; t < triStart[c + 1]; t++) full.push(idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]);
      g.setIndex(full);
      g.computeVertexNormals();
      normal = g.getAttribute("normal");
    }
    const slots = [...new Set(mats)].sort((a, b) => a - b);
    const slotOf = new Map(slots.map((s, k) => [s, k]));
    const tk = Uint32Array.from(mats, (s) => slotOf.get(s));
    const sphere = new THREE.Box3().setFromBufferAttribute(position).getBoundingSphere(new THREE.Sphere());
    const uvA = uv.length ? new THREE.BufferAttribute(uv, 2) : null;
    return { json, position, normal, uv: uvA, idx, triStart, tk, clusterLevel, slots, sphere };
  });
  const group = new THREE.Group();
  const colorMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7, metalness: 0 });
  const instances = Array.from(instMesh, (mi, i) => {
    const holder = new THREE.Group();
    holder.matrixAutoUpdate = false;
    holder.matrix.fromArray(xf, 16 * i);
    group.add(holder);
    return { mesh: meshes[mi], holder, object: null, indexed: null, flat: null, sel: new Uint32Array(0) };
  });
  group.userData.dispose = () => {
    for (const inst of instances) {
      for (const g of [inst.indexed, inst.flat]) {
        dropWire(g);
        g?.dispose();
      }
    }
    colorMat.dispose();
  };
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
      const m = inst.mesh;
      let g;
      if (shading === "materials") {
        g = inst.indexed;
        if (!g) {
          g = inst.indexed = new THREE.BufferGeometry();
          g.setAttribute("position", m.position);
          g.setAttribute("normal", m.normal);
          if (m.uv) g.setAttribute("uv", m.uv);
          g.setIndex(new THREE.BufferAttribute(new Uint32Array(m.idx.length), 1));
          g.boundingSphere = m.sphere;
        }
        fillIndexed(g, m, inst.sel);
      } else {
        dropWire(inst.flat);
        inst.flat?.dispose();
        g = inst.flat = flatGeometry(m, inst.sel, shading);
      }
      const material = shading === "materials" ? m.slots.map((s) => originalMaterial(s, plain)) : colorMat;
      if (!inst.object) {
        inst.object = new THREE.Mesh(g, material);
        inst.holder.add(inst.object);
      } else {
        inst.object.geometry = g;
        inst.object.material = material;
      }
      // New triangles: the wireframe is rebuilt.
      const w = inst.object.children.find((c) => c.name === "__wire");
      if (w) {
        inst.object.remove(w);
        dropWire(g);
      }
    }
    if (wireOn) setWire(group);
  };
  return {
    kind: "clusters",
    group,
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
    dispose() {}, // level 1 owns the group
  };
}

/** Fills the indexed cut: triangles of the selected clusters, grouped by material. */
function fillIndexed(g, m, sel) {
  const { triStart, tk, idx, slots } = m;
  const start = new Uint32Array(slots.length + 1);
  for (const c of sel) for (let t = triStart[c]; t < triStart[c + 1]; t++) start[tk[t] + 1]++;
  for (let k = 0; k < slots.length; k++) start[k + 1] += start[k];
  const fill = start.slice(0, slots.length);
  const out = g.index.array;
  for (const c of sel)
    for (let t = triStart[c]; t < triStart[c + 1]; t++) {
      const o = 3 * fill[tk[t]]++;
      out[o] = idx[3 * t];
      out[o + 1] = idx[3 * t + 1];
      out[o + 2] = idx[3 * t + 2];
    }
  g.clearGroups();
  for (let k = 0; k < slots.length; k++)
    if (start[k + 1] > start[k]) g.addGroup(3 * start[k], 3 * (start[k + 1] - start[k]), k);
  const n = 3 * start[slots.length];
  g.index.clearUpdateRanges();
  g.index.addUpdateRange(0, n);
  g.index.needsUpdate = true;
  g.setDrawRange(0, n);
}

/** The cut as unshared triangles with a color per cluster or per DAG level. */
function flatGeometry(m, sel, shading) {
  let tris = 0;
  for (const c of sel) tris += m.triStart[c + 1] - m.triStart[c];
  const pos = new Float32Array(9 * tris);
  const nrm = new Float32Array(9 * tris);
  const col = new Float32Array(9 * tris);
  const P = m.position.array;
  const N = m.normal.array;
  let o = 0;
  for (const c of sel) {
    const k = shading === "levels" ? levelColor(m.clusterLevel[c]) : clusterColor(c);
    const [r, g, b] = [k.r, k.g, k.b];
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
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.boundingSphere = m.sphere.clone();
  return geo;
}

// ---- HLOD

/** The meshes glTF node `node` draws itself (not its child nodes). */
function nodeMeshes(node) {
  const out = node.isMesh ? [node] : [];
  for (const c of node.children)
    if (c.isMesh && c.name !== "__wire" && original.assoc.get(c)?.nodes === undefined) out.push(c);
  return out;
}

function buildHlod(info, proxies) {
  original.object.updateMatrixWorld(true);
  const src = info.source_instances;
  const T = new THREE.Matrix4();
  // One group per tiled instance (copy-major, as the worker numbers them), sharing the original's
  // geometry and materials. Skinned meshes are drawn in their bind pose.
  const makeInstances = () => {
    const out = [];
    for (const o of info.offsets) {
      T.makeTranslation(o[0], o[1], o[2]);
      for (let s = 0; s < src; s++) {
        const g = new THREE.Group();
        const node = original.nodes[info.joint_nodes + s];
        for (const mesh of node ? nodeMeshes(node) : []) {
          const c = new THREE.Mesh(mesh.geometry, mesh.material);
          c.matrixAutoUpdate = false;
          c.matrix.multiplyMatrices(T, mesh.matrixWorld);
          g.add(c);
        }
        out.push(g);
      }
    }
    return out;
  };
  const full = new THREE.Group();
  full.add(...makeInstances());
  full.userData.dispose = () => {};
  const group = new THREE.Group();
  const cutInst = makeInstances();
  group.add(...cutInst);
  const tints = new Map();
  for (const p of proxies.values()) {
    p.object.visible = false;
    p.object.traverse((o) => {
      if (o.isMesh) o.userData.base = o.material;
    });
    group.add(p.object);
  }
  group.userData.dispose = () => {
    for (const p of proxies.values()) {
      p.object.traverse((o) => {
        if (o.isMesh) o.material = o.userData.base;
      });
      dispose(p.object, true);
    }
    for (const m of tints.values()) m.dispose();
  };
  const applyTint = () => {
    const on = $("#tint").checked;
    for (const p of proxies.values()) {
      let m = tints.get(p.info.level);
      if (on && !m) {
        m = new THREE.MeshStandardMaterial({ color: levelColor(p.info.level).clone(), roughness: 0.8 });
        tints.set(p.info.level, m);
      }
      p.object.traverse((o) => {
        if (o.isMesh) o.material = on ? m : o.userData.base;
      });
    }
  };
  applyTint();
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
  focusOverride = { center: new THREE.Vector3(...info.focus.center), radius: info.focus.radius };
  controls.maxDistance = focusOverride.radius * 400;
  if (!prev || Math.abs(prev.radius - focusOverride.radius) > 1e-6 * prev.radius) home();
  let last = null;
  return {
    kind: "hlod",
    group,
    full,
    fullInfo: { label: `Full (${info.copies} ${info.copies === 1 ? "copy" : "copies"})`, triangles: total, full_triangles: total },
    info: cutInfo,
    texProxy,
    cut(sel, [nodes, inst]) {
      last = sel;
      const ns = new Set(nodes);
      const is = new Set(inst);
      for (const [n, p] of proxies) p.object.visible = ns.has(n);
      cutInst.forEach((g, i) => (g.visible = is.has(i)));
      Object.assign(cutInfo, {
        triangles: sel.triangles,
        full_triangles: sel.full_triangles,
        error: sel.max_error,
        relative_error: sel.max_error / diag(),
        hud: `${sel.proxies} ${sel.proxies === 1 ? "proxy" : "proxies"}, ${sel.instances} ${sel.instances === 1 ? "instance" : "instances"}`,
      });
    },
    tint: applyTint,
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
      const ppr = renderer.getDrawingBufferSize(tmpV2).y / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
      const t = 2 ** +$("#threshold").value;
      let near = null;
      for (const n of info.nodes) {
        if (n.level !== 0 || !(n.error > 0)) continue;
        const [x, y, z, r] = n.sphere;
        const d = camera.position.distanceTo(tmpV3.set(x, y, z));
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
        ["Built in", `${(info.ms / 1000).toFixed(2)} s`],
      );
      return rows;
    },
    dispose() {
      full.removeFromParent();
      if (levels[0] && original) levels[0].object = original.object;
    },
  };
}

/** The bake panel in HLOD mode: the root proxy's baked textures. */
function drawHlodTextures() {
  const bake = $("#bake");
  const p = extra?.kind === "hlod" ? extra.texProxy : null;
  let mat = null;
  p?.object.traverse((o) => {
    if (o.isMesh && !mat) mat = o.userData.base ?? o.material;
  });
  const maps = { base_color: mat?.map, normal: mat?.normalMap, orm: mat?.roughnessMap };
  bake.classList.toggle("empty", !maps.base_color?.image);
  $("#bake-title").textContent = p ? `Proxy textures · node ${p.info.node} (tree level ${p.info.level})` : "Proxy textures";
  if (!maps.base_color?.image) {
    $("#bake-stats").textContent = extra?.kind === "hlod" ? "Proxies are geometry only (Bake proxy textures is off)." : "";
    return;
  }
  for (const [ch, t] of Object.entries(maps)) {
    const c = $(`#tex-${ch}`);
    const img = t?.image;
    if (!c || !img) continue;
    c.width = img.width;
    c.height = img.height;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0);
  }
  $("#bake-stats").textContent = `coverage ${(p.info.coverage * 100).toFixed(1)}% · ${fmtInt(p.info.triangles)} tris · ${fmtBytes(p.info.bytes)}`;
}

// ---- splats

// Each splat is a screen-space quad around its center, shaped by projecting the 3D covariance
// (EWA splatting, as 3D Gaussian splatting viewers do). Payload per splat: position, covariance
// (xx xy xz yy yz zz), linear RGBA, normal.
const SPLAT_VS = /* glsl */ `
attribute vec3 center;
attribute vec3 covA;
attribute vec3 covB;
attribute vec4 rgba;
attribute vec3 nrm;
uniform vec2 viewport;
uniform vec2 focal;
varying vec4 vColor;
varying vec2 vPos;
void main() {
  vec4 cam = modelViewMatrix * vec4(center, 1.0);
  vec4 clip = projectionMatrix * cam;
  if (cam.z > -1e-6 || abs(clip.x) > 1.3 * clip.w || abs(clip.y) > 1.3 * clip.w) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  mat3 V = mat3(covA.x, covA.y, covA.z, covA.y, covB.x, covB.y, covA.z, covB.y, covB.z);
  mat3 W = mat3(modelViewMatrix);
  float iz = 1.0 / cam.z;
  mat3 J = mat3(-focal.x * iz, 0.0, 0.0,
                0.0, -focal.y * iz, 0.0,
                focal.x * cam.x * iz * iz, focal.y * cam.y * iz * iz, 0.0);
  mat3 T = J * W;
  mat3 S = T * V * transpose(T);
  float a = S[0][0] + 0.3;
  float b = S[0][1];
  float c = S[1][1] + 0.3;
  float mid = 0.5 * (a + c);
  float rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  vec2 d1 = abs(b) > 1e-9 ? normalize(vec2(b, l1 - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * d1;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(-d1.y, d1.x); // (major, minor) keeps the quad counter-clockwise
  vPos = position.xy * 2.0;
  vec2 off = (vPos.x * major + vPos.y * minor) * 2.0 / viewport;
  gl_Position = vec4((clip.xy / clip.w + off) * clip.w, clip.z, clip.w);
  vec3 n = normalize(mat3(modelViewMatrix) * nrm);
  float lit = 0.55 + 0.45 * abs(dot(n, normalize(cam.xyz)));
  vColor = vec4(rgba.rgb * lit, rgba.a);
}`;
const SPLAT_FS = /* glsl */ `
varying vec4 vColor;
varying vec2 vPos;
void main() {
  float r2 = dot(vPos, vPos);
#ifdef DISC
  if (r2 > 2.0) discard;
  gl_FragColor = vec4(vColor.rgb, 1.0);
#else
  if (r2 > 4.0) discard;
  float alpha = vColor.a * exp(-r2);
  if (alpha < 1.0 / 255.0) discard;
  gl_FragColor = vec4(vColor.rgb, alpha);
#endif
  #include <colorspace_fragment>
}`;

const splatMats = {};
function splatMaterial(kind) {
  if (splatMats[kind]) return splatMats[kind];
  const disc = kind === "disc";
  const m = new THREE.ShaderMaterial({
    uniforms: { viewport: { value: new THREE.Vector2(1, 1) }, focal: { value: new THREE.Vector2(1, 1) } },
    vertexShader: SPLAT_VS,
    fragmentShader: SPLAT_FS,
    defines: disc ? { DISC: 1 } : {},
    transparent: !disc,
    depthWrite: disc,
  });
  // Per viewport (compare mode draws two): its size and focal length in pixels.
  m.onBeforeRender = (r, s, cam) => {
    r.getCurrentViewport(tmpV4);
    m.uniforms.viewport.value.set(tmpV4.z, tmpV4.w);
    const e = cam.projectionMatrix.elements;
    m.uniforms.focal.value.set((e[0] * tmpV4.z) / 2, (e[5] * tmpV4.w) / 2);
    m.uniformsNeedUpdate = true;
  };
  return (splatMats[kind] = m);
}

const splatData = new WeakMap(); // geometry -> { payload, buf, k, key, depth, keys, order }
function buildSplats(payload) {
  const k = payload.length / 16;
  const geom = new THREE.InstancedBufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
  geom.setIndex([0, 1, 2, 0, 2, 3]);
  const buf = new THREE.InstancedInterleavedBuffer(new Float32Array(payload), 16, 1);
  buf.setUsage(THREE.DynamicDrawUsage);
  for (const [name, size, offset] of [
    ["center", 3, 0],
    ["covA", 3, 3],
    ["covB", 3, 6],
    ["rgba", 4, 9],
    ["nrm", 3, 13],
  ])
    geom.setAttribute(name, new THREE.InterleavedBufferAttribute(buf, size, offset));
  geom.instanceCount = k;
  splatData.set(geom, { payload, buf, k, key: "", depth: new Float32Array(k), keys: new Uint16Array(k), order: new Uint32Array(k) });
  const mesh = new THREE.Mesh(geom, splatMaterial($("#splat-draw").value));
  mesh.frustumCulled = false;
  mesh.userData.noWire = true;
  mesh.userData.dispose = () => geom.dispose();
  return mesh;
}

let sortCounts = null;
/** Blended Gaussians are drawn back to front: re-sort (counting sort on depth) when the view turns. */
function sortSplats() {
  if ($("#splat-draw").value !== "gauss") return;
  camera.updateMatrixWorld();
  const e = camera.matrixWorldInverse.elements;
  const key = `${e[2].toFixed(4)},${e[6].toFixed(4)},${e[10].toFixed(4)},${e[14].toPrecision(5)}`;
  for (const l of levels) {
    const s = l?.object?.geometry && splatData.get(l.object.geometry);
    if (!s || s.key === key) continue;
    s.key = key;
    const { payload, k, depth, keys, order } = s;
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
    const out = s.buf.array;
    for (let j = 0; j < k; j++) {
      const from = 16 * order[j];
      const to = 16 * j;
      for (let c = 0; c < 16; c++) out[to + c] = payload[from + c];
    }
    s.buf.needsUpdate = true;
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
$("#shading").addEventListener("change", () => extra?.reshade?.());
$("#tint").addEventListener("change", () => extra?.tint?.());
$("#splat-draw").addEventListener("change", () => {
  const m = splatMaterial($("#splat-draw").value);
  for (const l of levels) {
    const s = l?.object?.geometry && splatData.get(l.object.geometry);
    if (!s) continue;
    l.object.material = m;
    s.key = "";
  }
  lodVersion++;
  cutStats();
});

// ---------------------------------------------------------------- worker

function spawn() {
  worker?.terminate();
  cutInflight = false;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
  if (busy && !original) {
    // Decoding: wasm can't be interrupted, so drop the worker.
    spawn();
    busy = false;
    files = null;
  }
  for (const b of cards.values()) b.removeAttribute("aria-current");
  hideLoading();
  setStatus("Cancelled");
});

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = cls;
}

async function onWorker(m) {
  if (m.runId !== undefined && m.runId !== runId) return; // a cancelled run
  switch (m.type) {
    case "loaded":
      if (m.quiet) return;
      busy = false;
      assetInfo = m.info;
      showInfo(m.info);
      try {
        await showOriginal(m.glb);
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
      drawTextures(m.level);
      break;
    }
    case "level":
      await onLevel(m.info, m.glb, m.runId);
      break;
    case "data":
      await onData(m);
      break;
    case "select":
      onSelect(m);
      break;
    case "done": {
      busy = false;
      const chain = shownMode === "chain";
      const n = chain ? runOpts.levels : MODE_LEVELS[shownMode];
      for (let i = levels.length - 1; i > n; i--) {
        disposeLevel(levels[i]);
        levels.pop();
      }
      if (selected > n) select(n);
      drawTextures(selected);
      const what = {
        chain: `${n + 1} levels`,
        presets: `${n} configurations`,
        clusters: "the cluster DAG",
        hlod: "the HLOD tree",
        splats: `${n} splat levels`,
      }[shownMode];
      const live = shownMode === "clusters" || shownMode === "hlod" ? "; the cut follows the camera" : "";
      setStatus(`Built ${what} in ${((performance.now() - runStart) / 1000).toFixed(2)} s${live}`);
      cutKey = "";
      renderRows();
      cutStats();
      break;
    }
    case "download": {
      const ply = m.format === "ply";
      const url = URL.createObjectURL(
        new Blob([m.glb], { type: ply ? "application/octet-stream" : "model/gltf-binary" }),
      );
      const a = Object.assign(document.createElement("a"), {
        href: url,
        download: `${(assetInfo?.name ?? "model").replace(/\.(glb|gltf|zip)$/i, "")}_LOD${m.level}.${ply ? "ply" : "glb"}`,
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
  return (levels[level] ??= { info: null, object: null, tex: {}, stale: true });
}

async function onLevel(info, glb, id) {
  const l = entry(info.level);
  if (info.level === 0) {
    l.info = info;
    l.stale = false;
    return renderRows();
  }
  const gltf = await loader.parseAsync(glb, "");
  if (id !== runId) return dispose(gltf.scene, info.kind === "baked");
  if (info.kind === "reduced") adoptMaterials(gltf);
  disposeLevel(l);
  if (info.kind !== "baked") l.tex = {};
  l.object = gltf.scene;
  l.info = info;
  l.stale = false;
  root.add(gltf.scene);
  setWire(gltf.scene);
  renderRows();
  if (info.kind === "baked" && (selected === info.level || !levels[selected]?.tex.base_color))
    drawTextures(info.level);
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
  renderRows();
  setStatus("Starting…", "busy");
  worker.postMessage({ type: "run", runId: id, options: runOpts });
}

async function load(list, title) {
  if (!list.length) return;
  if (loadingEl.hidden) showLoading(title ?? (list.length === 1 ? list[0].name : `${list.length} files`));
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
}

function renderRows() {
  const m = shownMode || form.elements.mode.value;
  const chain = m === "chain";
  const n = chain ? +form.elements.levels.value : MODE_LEVELS[m];
  const bakeFrom = +form.elements.bake_from.value || Infinity;
  const total = assetInfo?.triangles ?? 0;
  const out = [];
  for (let i = 0; i <= Math.max(n, levels.length - 1); i++) {
    const l = levels[i];
    const info = l?.info;
    const kind = info?.kind ?? (i === 0 ? "original" : chain ? (i >= bakeFrom ? "baked" : "reduced") : MODE_KIND[m]);
    const pending = !info || l.stale;
    const live = kind === "clusters" || kind === "hlod";
    const of = info?.full_triangles || total;
    const errTitle =
      info?.error_title ??
      (info?.error_bound != null
        ? `${info.error.toPrecision(3)} measured; simplifier bound ${info.error_bound.toPrecision(3)}`
        : `${info?.error?.toPrecision(3)}`);
    const splats = kind === "splats";
    const cells = info
      ? [
          splats ? `<span title="Splats have no triangles">—</span>` : fmtInt(info.triangles),
          splats ? "—" : of ? `${((100 * info.triangles) / of).toPrecision(3)}` : "",
          i === 0 ? "—" : `<span title="${esc(errTitle)}">${(info.relative_error * 100).toPrecision(2)}%</span>`,
          i === 0
            ? "—"
            : live
              ? `<span title="Every ${kind === "hlod" ? "cell" : "cluster"} switches at its own distance">per ${kind === "hlod" ? "cell" : "cluster"}</span>`
              : info.switch_distance_1px.toPrecision(3),
          i === 0 ? "—" : `${info.ms.toFixed(0)} ms`,
          info.bytes ? fmtBytes(info.bytes) : "—",
        ]
      : ["…", "", "", "", "", ""];
    const name = splats && info ? `${fmtInt(info.vertices)} splats` : (info?.label ?? (i === 0 && !chain ? "Original" : `LOD${i}`));
    const ext = kind === "splats" ? "ply" : "glb";
    const dl =
      pending || live || (i === 0 && m === "hlod")
        ? ""
        : `<button type="button" data-dl="${i}" data-fmt="${ext}" title="Download ${esc(name)} as .${ext}">.${ext}</button>`;
    out.push(
      `<tr data-level="${i}" class="${i === selected ? "selected" : ""} ${pending ? "pending" : ""}">` +
        `<td class="label" title="${esc(info?.note ?? "")}">${esc(name)}</td><td><span class="kind ${kind}">${kind}</span></td>` +
        cells.map((c) => `<td class="n">${c}</td>`).join("") +
        `<td>${dl}</td></tr>`,
    );
  }
  rowsEl.innerHTML = out.join("");
}

const updateRow = () => renderRows();

function select(level) {
  selected = level;
  for (const tr of rowsEl.children) tr.classList.toggle("selected", +tr.dataset.level === level);
  if (levels[level]?.tex.base_color) drawTextures(level);
  if (shownMode === "splats") cutStats();
}

function drawTextures(level) {
  if (shownMode === "hlod") return drawHlodTextures();
  const bake = $("#bake");
  // Show this level's bake, else the most recent baked level.
  let l = levels[level];
  if (!l?.tex.base_color) {
    level = levels.findLastIndex((x) => x?.tex.base_color);
    l = levels[level];
  }
  bake.classList.toggle("empty", !l);
  if (!l) {
    $("#bake-title").textContent = "Baked textures";
    $("#bake-stats").textContent =
      +form.elements.bake_from.value === 0 ? "No proxy levels (Bake proxy from: Never)." : "";
    return;
  }
  for (const [ch, img] of Object.entries(l.tex)) {
    const c = $(`#tex-${ch}`);
    if (!c) continue;
    c.width = img.width;
    c.height = img.height;
    c.getContext("2d").putImageData(img, 0, 0);
  }
  $("#bake-title").textContent = `Baked textures · LOD${level}${l.stale ? " (updating)" : ""}`;
  const i = l.info;
  $("#bake-stats").textContent =
    i?.coverage != null
      ? `coverage ${(i.coverage * 100).toFixed(1)}% · ${i.charts} charts · utilization ${(i.utilization * 100).toFixed(1)}%`
      : "";
}

rowsEl.addEventListener("click", (e) => {
  const dl = e.target.closest("[data-dl]");
  if (dl) {
    const level = +dl.dataset.dl;
    pendingDownload = level;
    if (busy) setStatus(`LOD${level} download starts when the run finishes…`, "busy");
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
canvas.addEventListener("dblclick", () => home());

function setMode(m) {
  mode = m;
  for (const o of document.querySelectorAll("[data-mode]"))
    o.setAttribute("aria-pressed", String(o.dataset.mode === m));
  // Auto mode orbits the model's center: switch distances are measured from there, and zooming
  // out around a panned-away target would leave the model behind.
  const auto = m === "auto";
  controls.enablePan = !auto;
  if (auto && original) {
    const shift = original.center.clone().sub(controls.target);
    controls.target.add(shift);
    camera.position.add(shift);
  }
  $("#dist-wrap").hidden = !auto;
}
for (const b of document.querySelectorAll("[data-mode]")) b.addEventListener("click", () => setMode(b.dataset.mode));

$("#wire").addEventListener("change", (e) => {
  wireOn = e.target.checked;
  for (const l of levels) if (l?.object) setWire(l.object);
});

// ---------------------------------------------------------------- options

function readOptions() {
  const f = form.elements;
  return {
    levels: +f.levels.value,
    step: +f.step.value,
    bake_from: +f.bake_from.value,
    atlas: +f.atlas.value,
    supersample: +f.supersample.value,
    planar_uv: f.planar_uv.checked,
    fast: f.fast.checked,
    strict_seams: f.strict_seams.checked,
    symmetry: f.symmetry.checked,
    lock_border: f.lock_border.checked,
    mode: f.mode.value,
    preset_ratio: +f.preset_ratio.value,
    cluster_layout: f.cluster_layout.value,
    cluster_fast: f.cluster_fast.checked,
    hlod_copies: +f.hlod_copies.value,
    hlod_split: +f.hlod_split.value,
    hlod_triangles: +f.hlod_triangles.value,
    hlod_bake: f.hlod_bake.checked,
    splat_count: +f.splat_count.value,
    splat_footprints: f.splat_footprints.checked,
  };
}

/** Shows the option groups of the chosen example. */
function syncModeUi() {
  const m = form.elements.mode.value;
  for (const fs of form.querySelectorAll("fieldset[data-modes]")) fs.hidden = !fs.dataset.modes.split(" ").includes(m);
  $("#mode-hint").textContent = MODE_HINTS[m] ?? "";
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
  f.supersample_out.value = `${f.supersample.value}×${f.supersample.value}`;
  f.preset_ratio_out.value = `${Math.round(+f.preset_ratio.value * 100)}%`;
  syncModeUi();
}

function restoreOptions() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(STORE_KEY));
  } catch {}
  if (saved) {
    const f = form.elements;
    for (const [k, v] of Object.entries(saved)) {
      const el = f[k];
      if (!el || k === "bake_from") continue;
      if (el.type === "checkbox") el.checked = !!v;
      else el.value = String(v);
    }
    lastLevels = +f.levels.value;
    syncOutputs();
    if (saved.bake_from !== undefined) f.bake_from.value = String(Math.min(saved.bake_from, lastLevels));
  } else syncOutputs();
}

let debounce = 0;
form.addEventListener("input", () => {
  syncOutputs();
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(readOptions()));
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

// Sample models: CC0 assets that fetch_samples.py downloads into samples/ at build time, listed
// in samples/index.json, each shown as a card in the sidebar. Without the index (a local build
// that skipped the script) the section stays hidden.
const cards = new Map(); // sample file -> its card button
function card(s) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "card";
  b.title = `${s.name} · ${mb(s.bytes)} · by ${s.credit} · ${s.license}`;
  b.innerHTML =
    (s.thumb ? `<img src="samples/${esc(s.thumb)}" alt="" loading="lazy">` : `<span class="noimg"></span>`) +
    `<span class="name">${esc(s.name)}</span><span class="size">${mb(s.bytes)}</span><span class="bar"></span>`;
  b.addEventListener("click", () => loadSample(s));
  return b;
}

function loadSample(s) {
  for (const [file, b] of cards) b.setAttribute("aria-current", String(file === s.file));
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
  fetching?.abort();
  for (const b of cards.values()) b.removeAttribute("aria-current");
};
picker.addEventListener("change", clearCurrent);
addEventListener("drop", clearCurrent);

// ?model=<url> loads a .glb (or .zip) on start, e.g. index.html?model=samples/Lantern.glb
const startUrl = new URLSearchParams(location.search).get("model");
if (startUrl) loadUrl(startUrl, startUrl, 0);
