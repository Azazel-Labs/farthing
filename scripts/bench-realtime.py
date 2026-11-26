#!/usr/bin/env python3
"""A/B timing of the realtime reducer, for a quiet machine (the `bench` workflow runs it).

    scripts/bench-realtime.py ASSET_DIR ROUNDS LABEL=EXE [LABEL=EXE ...]

Each EXE is a bench-compare binary. The builds take turns, ROUNDS times per asset, so drift in
the machine's load hits them alike; every phase's time is its minimum over all rounds. Then
each build runs at 1, 2, 4, ... threads to show which phases scale. Prints Markdown.
"""

import collections
import os
import re
import subprocess
import sys

ASSETS = ["ChronographWatch", "CarConcept", "DragonAttenuation"]
PHASE = re.compile(r"^\s+(\w+)\s+([\d.]+) ms")


def run(exe, asset, threads=None):
    """Per-phase minimums and the best whole-call time of one process (several calls)."""
    env = dict(os.environ, FARTHING_RT_PROFILE="1", RD_METHODS="farthing realtime", RT_RUNS="5")
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    out = subprocess.run([exe, "realtime", asset], env=env, capture_output=True, text=True)
    phases = collections.OrderedDict()
    for line in out.stderr.splitlines():
        m = PHASE.match(line)
        if m:
            phases[m[1]] = min(phases.get(m[1], 1e9), float(m[2]))
    total = None
    for line in out.stdout.splitlines():
        f = line.split()
        if "realtime" in f and not line.startswith("farthing"):
            total = float(f[-2])
    return phases, total


def merge(into, phases, total):
    for k, v in phases.items():
        into[k] = min(into.get(k, 1e9), v)
    if total is not None:
        into["CALL"] = min(into.get("CALL", 1e9), total)


def table(cols, rows):
    names = list(rows)
    print("| phase | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for c in cols:
        print(f"| {c} | " + " | ".join(f"{rows[n].get(c, float('nan')):.2f}" for n in names) + " |")
    print()


def main():
    assets_dir, rounds = sys.argv[1], int(sys.argv[2])
    builds = [a.split("=", 1) for a in sys.argv[3:]]
    cores = os.cpu_count() or 1
    print(f"{cores} logical cores, {rounds} rounds of 5 calls, minimum per phase (ms)\n")
    for name in ASSETS:
        asset = os.path.join(assets_dir, name + ".glb")
        if not os.path.exists(asset):
            continue
        best = {label: collections.OrderedDict() for label, _ in builds}
        for _ in range(rounds):
            for label, exe in builds:
                merge(best[label], *run(exe, asset))
        # Every phase any build has, in the order first seen.
        cols = []
        for b in best.values():
            cols += [c for c in b if c != "CALL" and c not in cols]
        for b in best.values():
            b["sum"] = sum(b.get(c, 0.0) for c in cols)
        print(f"### {name}\n")
        table(cols + ["sum", "CALL"], best)
        label, exe = builds[-1]
        scale = collections.OrderedDict()
        t = 1
        while t <= cores:
            row = collections.OrderedDict()
            for _ in range(max(2, rounds // 4)):
                merge(row, *run(exe, asset, t))
            row["sum"] = sum(row.get(c, 0.0) for c in cols)
            scale[f"{t} thr"] = row
            t *= 2
        print(f"Thread scaling ({label}):\n")
        table(cols + ["sum", "CALL"], scale)


main()
