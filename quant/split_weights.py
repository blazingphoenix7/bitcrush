"""Split web/weights-qwen3/qwen3.bin into <100 MB parts for GitHub Pages hosting
(Pages rejects files over 100 MB; the site fetches qwen3.bin.000... same-origin).
Writes web/weights-remote/{manifest.json, qwen3.bin.NNN}.  Run after export_qwen3i.py."""
import io, json, os, sys

SRC = os.path.join("web", "weights-qwen3")
DST = os.path.join("web", "weights-remote")
PART = 95 * 1024 * 1024

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.makedirs(DST, exist_ok=True)

manifest = json.load(io.open(os.path.join(SRC, "manifest.json"), encoding="utf-8"))
size = os.path.getsize(os.path.join(SRC, "qwen3.bin"))
parts = (size + PART - 1) // PART
manifest["parts"] = parts

with io.open(os.path.join(SRC, "qwen3.bin"), "rb") as f:
    for i in range(parts):
        chunk = f.read(PART)
        with io.open(os.path.join(DST, "qwen3.bin.%03d" % i), "wb") as o:
            o.write(chunk)
        print("part %03d: %d bytes" % (i, len(chunk)))

json.dump(manifest, io.open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8"))
print("wrote %d parts (+manifest with parts=%d) to %s" % (parts, parts, DST))
