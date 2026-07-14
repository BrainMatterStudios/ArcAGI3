#!/usr/bin/env python3
"""Build submission/_duck_shadow/duck-shadow.ipynb — the FLOOR-SAFE shadow-replay submission.

Transforms the pure base-duck notebook (submission/_repro/duck-repro.ipynb) with a SINGLE change:
`await bm.run(...)` -> `await run_with_shadow(bm, ...)`. The duck solver is UNTOUCHED (banks its
~1.26 floor); the concurrent shadow banks each duck level-win efficiently as a fresh play via
max-over-plays (validated offline: tu93 0.03 -> 2.22). Floor-safe: worst case == duck.

Embeds only shadow_run.py + shadow_replay.py (tiny). Run: python submission/_duck_shadow/build_duck_shadow.py
"""
import base64, json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
BASE = REPO / "submission/_repro/duck-repro.ipynb"
SCAF = REPO / "scratchpad/ewm_pathb"
OUT = REPO / "submission/_duck_shadow"

BLOB = {name: base64.b64encode((SCAF / name).read_bytes()).decode()
        for name in ("shadow_run.py", "shadow_replay.py")}

EMBED = (
    "import base64, json, os, sys\n"
    "from pathlib import Path\n"
    "SH = Path('/kaggle/working/shadow'); SH.mkdir(parents=True, exist_ok=True)\n"
    "for rel, b64 in json.loads(%r).items():\n"
    "    (SH / rel).write_bytes(base64.b64decode(b64))\n"
    "if str(SH) not in sys.path: sys.path.insert(0, str(SH))\n"
    "from shadow_run import run_with_shadow   # taaf already importable from the source bundle\n"
    "print('[shadow] run_with_shadow ready')\n"
) % json.dumps(BLOB)

nb = json.loads(BASE.read_text())
new_cells = []
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if c["cell_type"] == "code" and "setup_commands.json" in src:
        new_cells.append(c)  # keep serve-Qwen cell
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": EMBED.splitlines(keepends=True)})
        continue
    if c["cell_type"] == "code" and "await bm.run(" in src:
        # the ONLY functional change: route the run through the concurrent shadow
        src = src.replace("await bm.run(", "await run_with_shadow(bm, ")
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.splitlines(keepends=True)})
        continue
    new_cells.append(c)
nb["cells"] = new_cells

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "duck-shadow.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "ahmedmobasher86/arc-agi-3-duck-shadow",
    "title": "arc-agi-3-duck-shadow",
    "code_file": "duck-shadow.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                        "ahmedmobasher86/taaf-src-hybrid",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [], "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT/'duck-shadow.ipynb'} ({len(nb['cells'])} cells, embedded {len(BLOB)} files)")
