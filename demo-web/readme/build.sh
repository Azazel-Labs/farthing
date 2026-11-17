#!/usr/bin/env bash
# Rebuilds the README's recut, cluster and splat figures (docs/images/{recut,clusters,splats}-
# {light,dark}.png): renders the panels with the demo's pipeline and the engine viewer, then
# lays them out in the sheets here with headless Chrome. Needs a GPU (or a software adapter)
# and Node 22+.
#
#   readme/build.sh [chrome]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f web/samples/FlightHelmet.zip ] || python fetch_samples.py
cargo run --release --features readme --example readme -- web/samples/FlightHelmet.zip readme/out
node readme/shoot.mjs "$@"
