import json, math, sys
d = json.load(open(sys.argv[1]))
rows = d["rows"]
by = {}
for r in rows: by.setdefault((r["asset"], r["ratio"]), {})[r["method"]] = r
base = "farthing"
metrics = [("hausdorff","hausd"),("mean","mean"),("uv_p90","uv90"),("normal_deg","normal"),("ms","time")]
print(f"{'vs method':<20} {'cases':>5} " + " ".join(f"{n:>14}" for _,n in metrics))
for other in ["meshopt","meshopt attr","meshopt attr perm","baby_shark (geom)"]:
    ratios = {k: [] for k,_ in metrics}; wins = {k: 0 for k,_ in metrics}; n = 0
    for key, ms in by.items():
        if base not in ms or other not in ms: continue
        a, b = ms[base], ms[other]
        if abs(a["tris"] - b["tris"]) > max(3, 0.03 * b["tris"]): continue
        n += 1
        for k,_ in metrics:
            x, y = a[k], b[k]
            if x is None or y is None or (isinstance(y,float) and math.isnan(y)) or (isinstance(x,float) and math.isnan(x)): continue
            x, y = max(x, 1e-7), max(y, 1e-7)
            ratios[k].append(math.log(x / y))
            wins[k] += x <= y
    cells = []
    for k,_ in metrics:
        v = ratios[k]
        cells.append(f"{math.exp(sum(v)/len(v)):5.2f}x w{wins[k]:>2}/{len(v):<2}" if v else f"{'-':>14}")
    print(f"{other:<20} {n:>5} " + " ".join(f"{c:>14}" for c in cells))
print("\n(ratio = farthing / other, geometric mean; <1 means farthing better; w = farthing wins)")
