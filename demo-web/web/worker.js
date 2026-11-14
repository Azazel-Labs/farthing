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
      session.run(JSON.stringify(data.options), (kind, json, payload) => {
        // Every payload is a fresh JS copy, so its buffers can be transferred.
        if (kind === "stage") post({ type: "stage", runId, label: json });
        else if (kind === "level")
          post({ type: "level", runId, info: JSON.parse(json), glb: payload.buffer }, [payload.buffer]);
        else if (kind === "texture")
          post({ type: "texture", runId, ...JSON.parse(json), rgba: payload.buffer }, [payload.buffer]);
        else {
          // Mode data (clusters, HLOD nodes and tree, splats): JSON plus typed arrays.
          const buffers = [...payload];
          post({ type: "data", runId, kind, info: JSON.parse(json), buffers }, buffers.map((b) => b.buffer));
        }
      });
      post({ type: "done", runId });
    } else if (data.type === "select") {
      const [json, ...buffers] = session.select(JSON.stringify(data.view));
      post({ type: "select", seq: data.seq, info: JSON.parse(json), buffers }, buffers.map((b) => b.buffer));
    } else if (data.type === "download") {
      const ply = data.format === "ply";
      const bytes = ply
        ? session.levelPly(data.level)
        : data.level === 0
          ? session.originalGlb()
          : session.levelGlb(data.level);
      post({ type: "download", level: data.level, format: ply ? "ply" : "glb", glb: bytes.buffer }, [bytes.buffer]);
    }
  } catch (e) {
    // A wasm trap (`unreachable`, out of bounds...) poisons the module: nothing in this worker
    // can run again, so the page respawns it. Out-of-memory is the usual cause: allocation
    // failure aborts without reaching the panic hook.
    const trap = e instanceof WebAssembly.RuntimeError;
    if (!trap && data.type === "select") return post({ type: "select", seq: data.seq, error: String(e?.message ?? e) });
    post({ type: trap ? "crash" : "error", runId: data.runId, message: String(e?.message ?? e) });
  }
};
