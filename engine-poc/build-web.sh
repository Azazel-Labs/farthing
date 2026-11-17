#!/usr/bin/env sh
# Builds the engine's canvas viewer for the web demo, which draws with it. Needs the wasm32
# target and the same wasm-bindgen CLI as the demo (see demo-web/build.sh).
#   ./build-web.sh          WebGPU only, into ../demo-web/web/engine
#   ./build-web.sh webgl    with wgpu's WebGL2 backend as well, into ../demo-web/web/engine-gl,
#                           which the page fetches instead when the browser has no WebGPU
# If binaryen's wasm-opt is on PATH, the output is size-optimized too.
set -eu
cd "$(dirname "$0")"
features=web
out=../demo-web/web/engine
if [ "${1:-}" = webgl ]; then
  features=webgl
  out=../demo-web/web/engine-gl
fi
cargo build --profile web --target wasm32-unknown-unknown --lib --no-default-features --features "$features"
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  target/wasm32-unknown-unknown/web/farthing_poc_engine.wasm
wasm="$out/farthing_poc_engine_bg.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals --enable-reference-types --enable-multivalue "$wasm" -o "$wasm"
elif [ -n "${CI:-}" ]; then
  echo "wasm-opt is not on PATH" >&2
  exit 1
fi
wc -c < "$wasm" | tr -d ' ' > "$out/wasm-size.txt"
echo "$features: $(cat "$out/wasm-size.txt") bytes, $(gzip -9c "$wasm" | wc -c | tr -d ' ') gzipped"
