#!/usr/bin/env python3
"""Downloads the demo's sample models into web/samples/, with an index.json the page reads.

The models are CC0 assets from the Khronos glTF Sample Assets, pinned to one commit. They're
fetched at build time and served with the page, not committed. A model whose metadata lists
any license other than CC0-1.0 is refused. A .glb is copied as is; a multi-file .gltf is packed
into a .zip with its buffers and textures (the demo loads both). Files already downloaded are
kept, so a rerun only fetches what's missing.

    python fetch_samples.py          # then serve web/ as usual
"""

import io
import json
import pathlib
import sys
import urllib.parse
import urllib.request
import zipfile

REPO = "KhronosGroup/glTF-Sample-Assets"
COMMIT = "8f492390ddc7a2ae221a1369ef560589cf8c70fb"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/Models"
BROWSE = f"https://github.com/{REPO}/tree/{COMMIT}/Models"

# (directory in the sample repo, variant). In the order the page lists them. Models whose
# extensionsRequired farthing doesn't support (KHR_texture_transform, say) fail to load, so
# they're left out; tests/pipeline.rs runs every one listed here.
SAMPLES = [
    ("FlightHelmet", "glTF"),
    ("SciFiHelmet", "glTF"),
    ("Corset", "glTF-Binary"),
    ("BoomBox", "glTF-Binary"),
    ("Lantern", "glTF-Binary"),
    ("WaterBottle", "glTF-Binary"),
]

OUT = pathlib.Path(__file__).resolve().parent / "web" / "samples"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "farthing-demo-build"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(name: str, variant: str) -> dict:
    meta = json.loads(get(f"{RAW}/{name}/metadata.json"))
    licenses = {entry.get("license") for entry in meta.get("legal", [])}
    if licenses != {"CC0-1.0"}:
        sys.exit(f"{name}: licensed {sorted(licenses)}, not CC0-1.0 only; refusing to bundle it")

    if variant == "glTF-Binary":
        file = f"{name}.glb"
        dest = OUT / file
        if not dest.exists():
            dest.write_bytes(get(f"{RAW}/{name}/glTF-Binary/{name}.glb"))
    else:
        file = f"{name}.zip"
        dest = OUT / file
        if not dest.exists():
            base = f"{RAW}/{name}/glTF"
            gltf = get(f"{base}/{name}.gltf")
            doc = json.loads(gltf)
            uris = [b["uri"] for b in doc.get("buffers", []) if "uri" in b]
            uris += [i["uri"] for i in doc.get("images", []) if "uri" in i]
            buf = io.BytesIO()
            # Stored, not deflated: PNG/JPEG don't compress and the browser unzips faster.
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                z.writestr(f"{name}/{name}.gltf", gltf)
                for uri in dict.fromkeys(u for u in uris if not u.startswith("data:")):
                    z.writestr(f"{name}/{urllib.parse.unquote(uri)}", get(f"{base}/{uri}"))
            dest.write_bytes(buf.getvalue())

    names = [x for e in meta["legal"] for x in (e.get("artist"), e.get("owner"))]
    credit = ", ".join(dict.fromkeys(x for x in names if x and x != "Public")) or "Khronos"
    size = dest.stat().st_size
    print(f"{name:14} {file:18} {size / 1e6:6.1f} MB  CC0-1.0  {credit}")
    return {
        "name": meta.get("name", name),
        "file": file,
        "bytes": size,
        "credit": credit,
        "license": "CC0-1.0",
        "source": f"{BROWSE}/{name}",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = [fetch(name, variant) for name, variant in SAMPLES]
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"{len(index)} samples, {sum(s['bytes'] for s in index) / 1e6:.1f} MB, in {OUT}")


if __name__ == "__main__":
    main()
