// farthing LOD demo: page logic. The wasm runs in worker.js; this file handles input, the
// options, the three.js viewer and the level table.
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
function home(at = original) {
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
  const { center, radius } = original;
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
    if (!o.isMesh || o.name === "__wire") return;
    let w = o.children.find((c) => c.name === "__wire");
    if (!w && wireOn) {
      w = new Wireframe(new WireframeGeometry2(o.geometry), wireMat);
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
  original = { object: gltf.scene, materials, cache: new Map() };
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
  for (const [i, l] of levels.entries()) if (i > 0) dispose(l?.object, l?.info?.kind === "baked");
  if (original) {
    dispose(original.object, true);
    for (const m of original.cache.values()) m.dispose();
  }
  lod.clear();
  levels = [];
  original = null;
  selected = 0;
}

// Auto mode: THREE.LOD with each level's 1-pixel switch distance, scaled from the 1080p the
// numbers are computed for to this canvas's height.
let lodKey = "";
function syncLod() {
  const h = renderer.getDrawingBufferSize(new THREE.Vector2()).y;
  const ready = levels.filter((l) => l?.object && l.info && !l.stale);
  const key = ready.map((l) => l.object.uuid).join() + h + wireOn;
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
  const stale = l.stale ? " (updating)" : "";
  return `LOD${level} · ${l.info.kind} · ${fmtInt(l.info.triangles)} tris${stale}`;
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

// ---------------------------------------------------------------- worker

function spawn() {
  worker?.terminate();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => onWorker(data);
  worker.onerror = (e) => {
    busy = false;
    setStatus(`Worker failed to start: ${e.message ?? "see the console"}`, "error");
  };
}

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
        return setStatus(`Could not display the model: ${e.message ?? e}`, "error");
      }
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
    case "done": {
      busy = false;
      const n = runOpts.levels;
      for (let i = levels.length - 1; i > n; i--) {
        dispose(levels[i]?.object, levels[i]?.info?.kind === "baked");
        levels.pop();
      }
      if (selected > n) select(n);
      drawTextures(selected);
      setStatus(`Built ${n + 1} levels in ${((performance.now() - runStart) / 1000).toFixed(2)} s`);
      renderRows();
      break;
    }
    case "download": {
      const url = URL.createObjectURL(new Blob([m.glb], { type: "model/gltf-binary" }));
      const a = Object.assign(document.createElement("a"), {
        href: url,
        download: `${(assetInfo?.name ?? "model").replace(/\.(glb|gltf)$/i, "")}_LOD${m.level}.glb`,
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
      setStatus(m.message, "error");
      break;
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
  dispose(l.object, l.info?.kind === "baked");
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
  for (const l of levels) if (l) l.stale = true;
  renderRows();
  setStatus("Starting…", "busy");
  runOpts = readOptions();
  worker.postMessage({ type: "run", runId: id, options: runOpts });
}

async function load(list) {
  if (!list.length) return;
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
  const n = +form.elements.levels.value;
  const bakeFrom = +form.elements.bake_from.value || Infinity;
  const total = assetInfo?.triangles ?? 0;
  const out = [];
  for (let i = 0; i <= Math.max(n, levels.length - 1); i++) {
    const l = levels[i];
    const info = l?.info;
    const kind = info?.kind ?? (i === 0 ? "original" : i >= bakeFrom ? "baked" : "reduced");
    const pending = !info || l.stale;
    const cells = info
      ? [
          fmtInt(info.triangles),
          total ? `${((100 * info.triangles) / total).toPrecision(3)}` : "",
          i === 0 ? "—" : `<span title="${info.error.toPrecision(3)} measured; simplifier bound ${info.error_bound.toPrecision(3)}">${(info.relative_error * 100).toPrecision(2)}%</span>`,
          i === 0 ? "—" : info.switch_distance_1px.toPrecision(3),
          i === 0 ? "—" : `${info.ms.toFixed(0)} ms`,
        ]
      : ["…", "", "", "", ""];
    const dl = pending ? "" : `<button type="button" data-dl="${i}" title="Download LOD${i} as .glb">.glb</button>`;
    out.push(
      `<tr data-level="${i}" class="${i === selected ? "selected" : ""} ${pending ? "pending" : ""}">` +
        `<td>LOD${i}</td><td><span class="kind ${kind}">${kind}</span></td>` +
        cells.map((c) => `<td class="n">${c}</td>`).join("") +
        `<td>${dl}</td></tr>`,
    );
  }
  rowsEl.innerHTML = out.join("");
}

function select(level) {
  selected = level;
  for (const tr of rowsEl.children) tr.classList.toggle("selected", +tr.dataset.level === level);
  if (levels[level]?.tex.base_color) drawTextures(level);
}

function drawTextures(level) {
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
    worker.postMessage({ type: "download", level });
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
  };
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
async function loadUrl(url, name, expected, onProgress = () => {}) {
  fetching?.abort();
  const ctl = (fetching = new AbortController());
  setStatus(`Downloading ${name}…`, "busy");
  try {
    const bytes = await download(url, ctl.signal, (got, total) => {
      setStatus(`Downloading ${name}… ${mb(got)} of ${mb(total)}`, "busy");
      onProgress(got / total);
    }, expected);
    if (ctl.signal.aborted) return;
    await load([{ name: decodeURIComponent(new URL(url, location.href).pathname.split("/").pop()), bytes }]);
  } catch (err) {
    if (!ctl.signal.aborted) setStatus(`Could not download ${name}: ${err.message ?? err}`, "error");
  } finally {
    if (fetching === ctl) fetching = null;
    onProgress(null);
  }
}

// Sample models: CC0 assets that fetch_samples.py downloads into samples/ at build time, listed
// in samples/index.json. Each one gets a card in the sidebar and in the empty viewport's gallery.
// Without the index (a local build that skipped the script) neither shows.
const cards = new Map(); // sample file -> its card buttons
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
  for (const [file, list] of cards) for (const b of list) b.setAttribute("aria-current", String(file === s.file));
  loadUrl(`samples/${s.file}`, s.name, s.bytes, (p) => {
    for (const b of cards.get(s.file)) {
      b.classList.toggle("loading", p !== null);
      b.style.setProperty("--p", p ?? 0);
    }
  });
}

fetch("samples/index.json")
  .then((r) => (r.ok ? r.json() : []))
  .catch(() => [])
  .then((samples) => {
    if (!samples.length) return;
    for (const s of samples) {
      const pair = [card(s), card(s)];
      $("#sample-cards").append(pair[0]);
      $("#gallery").append(pair[1]);
      cards.set(s.file, pair);
    }
    $("#samples").hidden = false;
  });

// Choosing or dropping a file of your own cancels a sample download and clears its highlight.
const clearCurrent = () => {
  fetching?.abort();
  for (const list of cards.values()) for (const b of list) b.removeAttribute("aria-current");
};
picker.addEventListener("change", clearCurrent);
addEventListener("drop", clearCurrent);

// ?model=<url> loads a .glb (or .zip) on start, e.g. index.html?model=samples/Lantern.glb
const startUrl = new URLSearchParams(location.search).get("model");
if (startUrl) loadUrl(startUrl, startUrl, 0);
