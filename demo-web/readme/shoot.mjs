// Captures the README sheets: serves this folder, opens each sheet in headless Chrome at
// 1280 CSS pixels wide (as tall as its content) and twice that in device pixels, light and
// dark, and writes docs/images/<sheet>-<theme>.png. Needs the panels in out/ (see build.sh).
//
//   node shoot.mjs [chrome] [sheet...]   (all sheets when none is named)
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const images = join(here, "..", "..", "docs", "images");
const SHEETS = ["recut", "clusters", "splats"];
const args = process.argv.slice(2);
const only = args.filter((a) => SHEETS.includes(a));
const chrome = args.find((a) => !SHEETS.includes(a)) ?? process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";

const types = { ".html": "text/html", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".js": "text/javascript" };
const server = createServer((req, res) => {
  const path = normalize(join(here, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!path.startsWith(here)) return res.writeHead(403).end();
  try {
    const body = readFileSync(path);
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" }).end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const port = 9334;
const proc = spawn(chrome, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "shoot-"))}`,
  "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try {
    target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
  } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const waiting = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && waiting.has(d.id)) (waiting.get(d.id)(d), waiting.delete(d.id));
  if (d.method === "Runtime.consoleAPICalled") console.log(`  [console.${d.params.type}]`, d.params.args.map((a) => a.value ?? a.description).join(" "));
  if (d.method === "Runtime.exceptionThrown") console.log("  [exception]", d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text);
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result?.result?.value;
await send("Runtime.enable");

let failed = false;
for (const sheet of SHEETS.filter((s) => !only.length || only.includes(s))) {
  for (const theme of ["light", "dark"]) {
    await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 800, deviceScaleFactor: 2, mobile: false });
    await send("Page.navigate", { url: `${base}/${sheet}.html?theme=${theme}` });
    const t0 = Date.now();
    let ok = false;
    while (Date.now() - t0 < 60000) {
      await sleep(200);
      if (await evaluate("window.__ready === true")) { ok = true; break; }
    }
    // The sheet is as tall as its content.
    const height = await evaluate("Math.ceil(document.body.getBoundingClientRect().height)");
    await send("Emulation.setDeviceMetricsOverride", { width: 1280, height, deviceScaleFactor: 2, mobile: false });
    await sleep(300);
    const out = join(images, `${sheet}-${theme}.png`);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(out, Buffer.from(shot.result.data, "base64"));
    console.log(`${ok ? "ok" : "TIMEOUT"} ${out} (${2 * 1280}×${2 * height})`);
    failed ||= !ok;
  }
}
ws.close();
proc.kill();
server.close();
process.exit(failed ? 1 : 0);
