// farthing LOD demo: page logic. The wasm runs in worker.js; this file handles input, the
// options, the three.js viewer and the level table.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

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

const dark = matchMedia("(prefers-color-scheme: dark)");
const wireMat = new THREE.LineBasicMaterial({ transparent: true, depthWrite: false });
const setWireColor = () => {
  wireMat.color.set(dark.matches ? 0xffffff : 0x000000);
  wireMat.opacity = dark.matches ? 0.22 : 0.3;
};
setWireColor();
dark.addEventListener("change", setWireColor);

const loader = new GLTFLoader();

function frame(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-6);
  const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.05;
  camera.near = radius / 200;
  camera.far = radius * 4000;
  camera.position.copy(center).add(new THREE.Vector3(0.6, 0.35, 1).normalize().multiplyScalar(dist));
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.maxDistance = radius * 400;
  controls.update();
  return { center, radius };
}

function setWire(object) {
  object.traverse((o) => {
    if (!o.isMesh) return;
    let w = o.children.find((c) => c.name === "__wire");
    if (!w && wireOn) {
      w = new THREE.LineSegments(new THREE.WireframeGeometry(o.geometry), wireMat);
      w.name = "__wire";
      w.renderOrder = 1;
      o.add(w);
    }
    if (w) w.visible = wireOn;
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
    g.add(l.object.clone());
    g.userData.level = l.info.level;
    lod.addLevel(g, d);
  }
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
    renderer.clear();
    return;
  }
  root.visible = mode !== "auto";
  lod.visible = mode === "auto";
  if (mode === "compare") {
    const half = Math.floor(w / 2);
    camera.aspect = half / h;
    camera.updateProjectionMatrix();
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
    if (mode === "auto") {
      syncLod();
      renderer.render(scene, camera);
      const cur = lod.levels[lod.getCurrentLevel()]?.object.userData.level ?? 0;
      const dist = camera.position.distanceTo(original.center);
      hud(`Auto: ${label(cur)}`, `distance ${dist.toPrecision(3)}`);
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
  else return;
  e.preventDefault();
});

for (const b of document.querySelectorAll("[data-mode]")) {
  b.addEventListener("click", () => {
    mode = b.dataset.mode;
    for (const o of document.querySelectorAll("[data-mode]"))
      o.setAttribute("aria-pressed", String(o === b));
  });
}

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

// Sample models: CC0 assets that fetch_samples.py downloads into samples/ at build time, listed
// in samples/index.json. Without it (a local build that skipped the script) the row stays hidden.
async function loadSample(s) {
  setStatus(`Downloading ${s.name} (${(s.bytes / 1e6).toFixed(0)} MB)…`, "busy");
  try {
    const r = await fetch(`samples/${s.file}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await load([{ name: s.file, bytes: await r.arrayBuffer() }]);
  } catch (err) {
    setStatus(`Could not download ${s.name}: ${err.message ?? err}`, "error");
  }
}

fetch("samples/index.json")
  .then((r) => (r.ok ? r.json() : []))
  .catch(() => [])
  .then((samples) => {
    if (!samples.length) return;
    const row = $("#samples");
    for (const s of samples) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = s.name;
      b.title = `${(s.bytes / 1e6).toFixed(0)} MB · ${s.credit} · ${s.license}`;
      b.addEventListener("click", () => loadSample(s));
      row.append(b);
    }
    row.hidden = false;
    $("#samples-credit").hidden = false;
  });

// ?model=<url> loads a .glb (or .zip) on start, e.g. index.html?model=samples/Lantern.glb
const startUrl = new URLSearchParams(location.search).get("model");
if (startUrl) {
  setStatus(`Downloading ${startUrl}…`, "busy");
  fetch(startUrl)
    .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((bytes) => load([{ name: decodeURIComponent(new URL(startUrl, location.href).pathname.split("/").pop()), bytes }]))
    .catch((err) => setStatus(`Could not download ${startUrl}: ${err.message ?? err}`, "error"));
}
