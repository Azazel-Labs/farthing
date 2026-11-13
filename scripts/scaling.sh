#!/usr/bin/env bash
# Thread-scaling benchmark for a many-core machine.
#
#   scripts/scaling.sh [ASSET_DIR]
#
# Runs the reducer (with and without parallel commits), quad reduction, UV atlases and voxel
# proxies at 1, 2, 4, ... threads up to the core count (override with MAX_THREADS=n), checks
# that every run's output hash equals the 1-thread run's, and prints wall times, speedups, the
# parallel share by Amdahl's law, and the thread count from which parallel commits pay off on
# this machine (set it with FARTHING_PAR_COMMIT_THREADS, or change PAR_COMMIT_THREADS).
#
# ASSET_DIR may hold any of the Khronos glTF samples used in docs/BENCHMARKS.md
# (DamagedHelmet.glb, Lantern.glb, Duck.glb, BoomBox.glb, Fox.glb, Avocado.glb,
# WaterBottle.glb, big/CarConcept.glb, big/DragonAttenuation.glb); missing ones are skipped.
# Without it, only the generated meshes run (reducer sphere, quad spheres).
set -euo pipefail
cd "$(dirname "$0")/.."

ASSETS_DIR="${1:-}"
pick() { for f in "$@"; do [ -n "$ASSETS_DIR" ] && [ -f "$ASSETS_DIR/$f" ] && printf '%s ' "$ASSETS_DIR/$f"; done; return 0; }
RED_ASSETS=$(pick DamagedHelmet.glb Lantern.glb big/CarConcept.glb)
UV_ASSETS=$(pick Avocado.glb BoomBox.glb DamagedHelmet.glb Duck.glb Fox.glb Lantern.glb WaterBottle.glb)
VOX_ASSETS=$(pick Duck.glb Fox.glb BoomBox.glb Lantern.glb big/DragonAttenuation.glb)

CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
MAX=${MAX_THREADS:-$CORES}
THREADS="1"
t=2
while [ "$t" -lt "$MAX" ]; do THREADS="$THREADS $t"; t=$((t * 2)); done
[ "$MAX" -gt 1 ] && THREADS="$THREADS $MAX"

echo "Building benchmarks..."
cargo build --release -q --examples -p farthing
BIN=target/release/examples

# Each workload prints result lines and "TOTAL <ms> ms". Fingerprint = result lines without
# timings (hashes, or counts and errors for quads).
run() { # workload threads
  local w=$1 th=$2
  case $w in
    reduce) RAYON_NUM_THREADS=$th FARTHING_PAR_COMMIT_THREADS=100000 "$BIN/bench_assets" -n 3 $RED_ASSETS ;;
    reduce+commits) RAYON_NUM_THREADS=$th FARTHING_PAR_COMMIT_THREADS=1 "$BIN/bench_assets" -n 3 $RED_ASSETS ;;
    quads) RAYON_NUM_THREADS=$th "$BIN/bench_quads" 100 200 |
      awk '{ print; t += $7 } END { printf "TOTAL %.1f ms\n", t }' ;;
    uv) RAYON_NUM_THREADS=$th "$BIN/bench_uv" -n 3 $UV_ASSETS ;;
    remesh) RAYON_NUM_THREADS=$th "$BIN/bench_remesh" -n 2 $VOX_ASSETS ;;
  esac
}
fingerprint() {
  grep -v '^TOTAL' | awk '{
    if ($0 ~ /hash/) { print $1, $NF }
    else { print $2, $4, $6, $10, $12 }   # quads: n, in, out, hausdorff, mean
  }'
}

WORKLOADS="reduce reduce+commits quads"
[ -n "$UV_ASSETS" ] && WORKLOADS="$WORKLOADS uv"
[ -n "$VOX_ASSETS" ] && WORKLOADS="$WORKLOADS remesh"

declare -A MS
STATUS=0
printf '\n%-16s' "workload"; for th in $THREADS; do printf '%14s' "${th}t"; done; printf '%12s\n' "parallel"
for w in $WORKLOADS; do
  printf '%-16s' "$w"
  ref=""
  for th in $THREADS; do
    out=$(run "$w" "$th" 2>/dev/null)
    ms=$(echo "$out" | awk '/^TOTAL/ {print $2}')
    fp=$(echo "$out" | fingerprint)
    MS[$w,$th]=$ms
    mark=""
    if [ "$th" = 1 ]; then ref=$fp; elif [ "$fp" != "$ref" ]; then mark="!"; STATUS=1; fi
    if [ "$th" = 1 ]; then printf '%14s' "${ms}ms"; else
      printf '%14s' "$(awk -v a="${MS[$w,1]}" -v b="$ms" 'BEGIN{printf "%.2fx", a/b}')$mark"; fi
  done
  # Amdahl from 1 thread and the largest count: p = (1 - Tn/T1) / (1 - 1/n).
  awk -v a="${MS[$w,1]}" -v b="${MS[$w,$MAX]}" -v n="$MAX" \
    'BEGIN{ if (n > 1) printf "%11.0f%%\n", 100 * (1 - b/a) / (1 - 1/n); else print "" }'
done

echo
best=""
for th in $THREADS; do
  if awk -v a="${MS[reduce,$th]}" -v b="${MS[reduce+commits,$th]}" 'BEGIN{exit !(b < a)}'; then best=$th; break; fi
done
if [ -n "$best" ]; then
  echo "Parallel commits are faster from $best threads on this machine:"
  echo "  export FARTHING_PAR_COMMIT_THREADS=$best   (built-in default: 8)"
else
  echo "Parallel commits were not faster at any tested thread count; keep the default (8) or raise it."
fi
if [ $STATUS -ne 0 ]; then
  echo "WARNING: some outputs differed from the 1-thread run (marked !). Please report this."
else
  echo "All outputs identical to the 1-thread run."
fi
exit $STATUS
