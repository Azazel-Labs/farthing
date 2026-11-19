#!/usr/bin/env python3
"""Downloads the demo's sample models into web/samples/, with an index.json the page reads.

The models are from the Khronos glTF Sample Assets, pinned to one commit. They're fetched at
build time and served with the page, not committed. Each is CC0, except the models listed in
ATTRIBUTED, whose parts may also be CC BY 4.0: the index then carries an attribution naming
every author, their part and its license, which the page shows with the model. A model with any
other license is refused. A .glb is copied as is; a multi-file .gltf is packed
into a .zip with its buffers and textures (the demo loads both). Each model's screenshot is saved
next to it as its thumbnail. Files already downloaded are kept, so a rerun only fetches what's
missing.

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
    ("Fox", "glTF-Binary"),
]

# Models allowed to have CC BY 4.0 parts (credited in the index and on the page). The Fox's
# rigging, animation and glTF conversion are CC BY; it's the demo's animated, skinned model.
ATTRIBUTED = {"Fox"}
LICENSE_NAMES = {"CC0-1.0": "CC0 1.0", "CC-BY-4.0": "CC BY 4.0"}

OUT = pathlib.Path(__file__).resolve().parent / "web" / "samples"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "farthing-demo-build"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(name: str, variant: str) -> dict:
    meta = json.loads(get(f"{RAW}/{name}/metadata.json"))
    licenses = {entry.get("license") for entry in meta.get("legal", [])}
    allowed = {"CC0-1.0", "CC-BY-4.0"} if name in ATTRIBUTED else {"CC0-1.0"}
    if not licenses or not licenses <= allowed:
        sys.exit(f"{name}: licensed {sorted(licenses)}, not {' / '.join(sorted(allowed))}; refusing to bundle it")

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

    # The sample repo's own screenshot (about 130 px square, a few KB) is the page's thumbnail.
    thumb = None
    if shot := meta.get("screenshot"):
        thumb = f"{name}{pathlib.PurePosixPath(shot).suffix}"
        if not (OUT / thumb).exists():
            (OUT / thumb).write_bytes(get(f"{RAW}/{name}/{shot}"))

    names = [x for e in meta["legal"] for x in (e.get("artist"), e.get("owner"))]
    credit = ", ".join(dict.fromkeys(x for x in names if x and x != "Public")) or "Khronos"
    license = "CC-BY-4.0" if "CC-BY-4.0" in licenses else "CC0-1.0"
    size = dest.stat().st_size
    print(f"{name:14} {file:18} {size / 1e6:6.1f} MB  {license:9}  {credit}")
    entry = {
        "name": meta.get("name", name),
        "file": file,
        "bytes": size,
        "thumb": thumb,
        "credit": credit,
        "license": license,
        "source": f"{BROWSE}/{name}",
    }
    if license != "CC0-1.0":
        # "Model by PixelMannen (CC0 1.0); rigging & animation by tomkranis (CC BY 4.0); ..."
        parts = []
        for e in meta["legal"]:
            who = e.get("artist") or e.get("owner") or "unknown"
            what = (e.get("what") or "model").lower().replace("gltf", "glTF")
            parts.append(f"{what} by {who} ({LICENSE_NAMES.get(e.get('license'), e.get('license'))})")
        text = "; ".join(parts)
        entry["attribution"] = text[0].upper() + text[1:] + ". Modified by the demo (reduced, rebaked)."
        entry["license_url"] = "https://creativecommons.org/licenses/by/4.0/"
    return entry


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = [fetch(name, variant) for name, variant in SAMPLES]
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"{len(index)} samples, {sum(s['bytes'] for s in index) / 1e6:.1f} MB, in {OUT}")


if __name__ == "__main__":
    main()
