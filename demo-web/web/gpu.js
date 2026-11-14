// Triangle-ID renders on the GPU (WebGL2 on an OffscreenCanvas), for farthing's visibility
// pass. The wasm imports `gpuRenderIds` and calls it synchronously from the worker; anything
// that goes wrong returns undefined and the wasm falls back to farthing's CPU ray caster.
//
// All views are drawn into tiles of one framebuffer and read back with a single readPixels.
// Each pixel holds the 1-based index of the triangle covering it (gl_VertexID / 3 + 1, packed
// into RGBA8), 0 where nothing is hit. Rows come back top to bottom, as farthing's View::ray
// numbers them.

const VS = `#version 300 es
uniform mat4 uViewProj;
in vec3 aPos;
flat out uint vId;
void main() {
  vec4 p = uViewProj * vec4(aPos, 1.0);
  p.z = 2.0 * p.z - p.w; // farthing's projections map depth to [0, 1]; GL clips to [-1, 1]
  gl_Position = p;
  vId = uint(gl_VertexID / 3) + 1u;
}`;

const FS = `#version 300 es
precision highp float;
precision highp int;
flat in uint vId;
out vec4 outColor;
void main() {
  outColor = vec4(float(vId & 255u), float((vId >> 8) & 255u),
                  float((vId >> 16) & 255u), float(vId >> 24)) / 255.0;
}`;

let state; // undefined: not tried yet; null: unavailable

function init() {
  if (typeof OffscreenCanvas === "undefined") return null;
  const gl = new OffscreenCanvas(1, 1).getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  });
  if (!gl) return null;
  const shader = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const max = Math.min(4096, gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), ...gl.getParameter(gl.MAX_VIEWPORT_DIMS));
  const renderer = (() => {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  })();
  return {
    gl,
    prog,
    uViewProj: gl.getUniformLocation(prog, "uViewProj"),
    vbo: gl.createBuffer(),
    vao: gl.createVertexArray(),
    fbo: gl.createFramebuffer(),
    color: gl.createRenderbuffer(),
    depth: gl.createRenderbuffer(),
    size: [0, 0],
    max,
    renderer,
  };
}

function get() {
  if (state === undefined) {
    try {
      state = init();
    } catch (e) {
      console.warn("GPU visibility unavailable:", e);
      state = null;
    }
  }
  if (state && state.gl.isContextLost()) state = null;
  return state;
}

/** The GPU's name, or undefined when WebGL2 isn't available in this worker. */
export function gpuName() {
  return get()?.renderer;
}

/**
 * `tris`: 9 floats (three world-space corners) per triangle. `viewProj`: 16 floats
 * (column-major) per view, all `width` x `height`. Returns a Uint32Array of
 * views x height x width triangle indices + 1 (0 = background), or undefined.
 */
export function gpuRenderIds(tris, viewProj, width, height) {
  const s = get();
  if (!s || width < 1 || height < 1 || width > s.max || height > s.max) return undefined;
  try {
    return render(s, tris, viewProj, width, height);
  } catch (e) {
    console.warn("GPU visibility failed, using the CPU:", e);
    state = null;
    return undefined;
  }
}

function render(s, tris, viewProj, width, height) {
  const { gl } = s;
  const views = viewProj.length / 16;
  const cols = Math.max(1, Math.floor(s.max / width));
  const rowsMax = Math.max(1, Math.floor(s.max / height));
  const perPass = cols * rowsMax;
  const out = new Uint32Array(views * width * height);

  gl.bindVertexArray(s.vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, s.vbo);
  gl.bufferData(gl.ARRAY_BUFFER, tris, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.useProgram(s.prog);
  gl.disable(gl.BLEND);
  gl.disable(gl.DITHER);
  gl.disable(gl.CULL_FACE); // the CPU caster hits both sides
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LESS);
  gl.depthMask(true);
  gl.colorMask(true, true, true, true);
  gl.enable(gl.SCISSOR_TEST);

  for (let first = 0; first < views; first += perPass) {
    const n = Math.min(perPass, views - first);
    const c = Math.min(cols, n);
    const r = Math.ceil(n / c);
    const W = c * width;
    const H = r * height;
    if (s.size[0] !== W || s.size[1] !== H) {
      gl.bindRenderbuffer(gl.RENDERBUFFER, s.color);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, W, H);
      gl.bindRenderbuffer(gl.RENDERBUFFER, s.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, W, H);
      gl.bindFramebuffer(gl.FRAMEBUFFER, s.fbo);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, s.color);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, s.depth);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error("incomplete framebuffer");
      s.size = [W, H];
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, s.fbo);
    gl.scissor(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    for (let k = 0; k < n; k++) {
      const x = (k % c) * width;
      const y = Math.floor(k / c) * height;
      gl.viewport(x, y, width, height);
      gl.scissor(x, y, width, height);
      gl.uniformMatrix4fv(s.uViewProj, false, viewProj.subarray((first + k) * 16, (first + k + 1) * 16));
      gl.drawArrays(gl.TRIANGLES, 0, tris.length / 3);
    }
    const px = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const ids = new Uint32Array(px.buffer); // little-endian: R is the low byte
    for (let k = 0; k < n; k++) {
      const x = (k % c) * width;
      const y = Math.floor(k / c) * height;
      const base = (first + k) * width * height;
      for (let row = 0; row < height; row++) {
        // GL rows run bottom to top; farthing's run top to bottom.
        const src = (y + height - 1 - row) * W + x;
        out.set(ids.subarray(src, src + width), base + row * width);
      }
    }
  }
  gl.disable(gl.SCISSOR_TEST);
  return out;
}
