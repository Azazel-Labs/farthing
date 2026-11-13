// Runs farthing (wasm) off the main thread. One Session per worker; the page terminates
// and respawns the worker to cancel a run, then re-sends the files.
import init, { Session } from "./pkg/farthing_demo_web.js";

const ready = init();
let session = null;

const post = (msg, transfer = []) => self.postMessage(msg, transfer);

self.onmessage = async ({ data }) => {
  try {
    await ready;
    if (data.type === "load") {
      session?.free();
      session = null;
      const t0 = performance.now();
      session = new Session(data.files.map((f) => [f.name, new Uint8Array(f.bytes)]));
      const info = JSON.parse(session.info());
      info.load_ms = performance.now() - t0;
      if (data.quiet) return post({ type: "loaded", quiet: true, info });
      const glb = session.originalGlb();
      post({ type: "loaded", info, glb: glb.buffer }, [glb.buffer]);
    } else if (data.type === "run") {
      const { runId } = data;
      session.run(JSON.stringify(data.options), (kind, json, bytes) => {
        // Every `bytes` is a fresh JS copy, so its buffer can be transferred.
        if (kind === "stage") post({ type: "stage", runId, label: json });
        else if (kind === "level")
          post({ type: "level", runId, info: JSON.parse(json), glb: bytes.buffer }, [bytes.buffer]);
        else post({ type: "texture", runId, ...JSON.parse(json), rgba: bytes.buffer }, [bytes.buffer]);
      });
      post({ type: "done", runId });
    } else if (data.type === "download") {
      const glb = data.level === 0 ? session.originalGlb() : session.levelGlb(data.level);
      post({ type: "download", level: data.level, glb: glb.buffer }, [glb.buffer]);
    }
  } catch (e) {
    post({ type: "error", runId: data.runId, message: String(e?.message ?? e) });
  }
};
