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
        # (1) route the scored run through the concurrent shadow, and (2) make the COMMIT run
        #     CPU-safe (skip the offline GPU pass) so pushing doesn't re-burn quota / need a
        #     capable commit GPU — the scored rerun (TRUE_SUBMISSION) runs duck+shadow for real.
        old = (
            '    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n'
            '    if not TRUE_SUBMISSION:\n'
            '        # An offline run isn\'t scored, but Kaggle still expects a submission.parquet output.\n'
            '        import pandas as pd\n'
            '\n'
            '        pd.DataFrame(\n'
            '            [["1_0", "1", True, 1]],\n'
            '            columns=["row_id", "game_id", "end_of_game", "score"],\n'
            '        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n')
        new = (
            '    if TRUE_SUBMISSION:\n'
            '        from shadow_run import run_with_shadow\n'
            '        await run_with_shadow(bm, soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n'
            '    else:\n'
            '        # CPU-safe landing: commit only needs a submission.parquet without error;\n'
            '        # the scored rerun runs the duck + shadow on the hidden games.\n'
            '        import pandas as pd\n'
            '        pd.DataFrame([["1_0", "1", True, 1]],\n'
            '                     columns=["row_id", "game_id", "end_of_game", "score"]\n'
            '                     ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n'
            '        print("[shadow] commit: CPU-safe landing; scored rerun runs duck+shadow", flush=True)\n')
        if old not in src:
            raise SystemExit("run-cell block did not match — inspect duck-repro run cell before building")
        src = src.replace(old, new)
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
