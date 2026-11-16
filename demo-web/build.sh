#!/usr/bin/env sh
# Builds the wasm module into web/pkg. Needs the wasm32 target and a wasm-bindgen CLI whose
# version matches the `wasm-bindgen = "=…"` pin in Cargo.toml:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.129
# mipple's Zstd (zstd-sys) is C, so it also needs a clang that can target wasm32: `clang` on
# PATH, or CC_wasm32_unknown_unknown (and AR_wasm32_unknown_unknown, an llvm-ar). Ubuntu's clang
# package works; on Windows the LLVM or Swift toolchain's clang does.
# If binaryen's wasm-opt is on PATH, the output is size-optimized too.
set -eu
cd "$(dirname "$0")"
profile=${1:-release}
cargo build --profile "$profile" --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  "target/wasm32-unknown-unknown/$profile/farthing_demo_web.wasm"
# The features enabled are the ones Rust's wasm32-unknown-unknown target emits by default. CI
# installs wasm-opt, so there a missing one is an error rather than a bigger download.
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt --version
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals --enable-reference-types --enable-multivalue \
    web/pkg/farthing_demo_web_bg.wasm -o web/pkg/farthing_demo_web_bg.wasm
elif [ -n "${CI:-}" ]; then
  echo "wasm-opt is not on PATH" >&2
  exit 1
fi
# The module's size, for the page's download bar: a gzipped response's Content-Length is the
# compressed size, not the bytes the page counts.
wc -c < web/pkg/farthing_demo_web_bg.wasm | tr -d ' ' > web/pkg/wasm-size.txt
ls -l web/pkg
