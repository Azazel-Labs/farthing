#!/usr/bin/env sh
# Builds the wasm module into web/pkg. Needs the wasm32 target and a wasm-bindgen CLI whose
# version matches the `wasm-bindgen = "=…"` pin in Cargo.toml:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.129
# If binaryen's wasm-opt is on PATH, the output is size-optimized too.
set -eu
cd "$(dirname "$0")"
profile=${1:-release}
cargo build --profile "$profile" --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  "target/wasm32-unknown-unknown/$profile/farthing_demo_web.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int \
    web/pkg/farthing_demo_web_bg.wasm -o web/pkg/farthing_demo_web_bg.wasm
fi
ls -l web/pkg
