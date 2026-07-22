#!/usr/bin/env python3
"""Build submission/_duck_l2/duck-l2.ipynb.

Layer-2 bundle: base duck + the two verified defect fixes, NOTHING else.

  1. CPU-safe commit guards (same as _duck_base)
  2. FIX A — death-safe memory: game_over no longer wipes the working world model
     (auto-reset resumes the SAME level; level_transition/run_complete wipes and the
     cross_level_notes carry-over are deliberate design, preserved untouched)
  3. FIX B — image-aware token estimator: images counted at true vision cost
     (~102 tok) instead of len(base64)/3 (~380 tok); recovers up to ~29% of the
     context budget at full 30-turn history

  NO ACTION7 fix. NO animation metadata. NO serving retune. NO adaptive budget.
  Model, sampling, concurrency, budgets, game list: unchanged.

Usage:  .venv/bin/python submission/_duck_l2/build_duck_l2.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_repro/duck-repro.ipynb"
FIXES = REPO / "submission/_duck_fixes/duck_fixes.py"
OUT = Path(__file__).parent / "duck-l2.ipynb"

CPU_SAFE_SETUP_OLD = (
    'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
    '    print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
    '    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n'
    "    # Re-read in case the command persisted new env keys.\n"
    "    env = _command_env()\n"
    "    os.environ.update(env)\n"
)
CPU_SAFE_SETUP_NEW = (
    "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it\n"
    '    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
    '        print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
    "        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n"
    "        env = _command_env()\n"
    "        os.environ.update(env)\n"
    "else:\n"
    '    print("[duck] commit: skipping setup_commands (no GPU serve)", flush=True)\n'
)

CPU_SAFE_RUN_OLD = (
    "    await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
    "    if not TRUE_SUBMISSION:\n"
    "        # An offline run isn't scored, but Kaggle still expects a submission.parquet output.\n"
    "        import pandas as pd\n"
    "\n"
    "        pd.DataFrame(\n"
    '            [["1_0", "1", True, 1]],\n'
    '            columns=["row_id", "game_id", "end_of_game", "score"],\n'
    "        ).to_parquet(WORKING_DIR / \"submission.parquet\", index=False)\n"
)
CPU_SAFE_RUN_NEW = (
    "    if TRUE_SUBMISSION:\n"
    "        await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
    "    else:\n"
    "        import pandas as pd\n"
    '        pd.DataFrame([["1_0", "1", True, 1]],\n'
    '                     columns=["row_id", "game_id", "end_of_game", "score"]\n'
    "                     ).to_parquet(WORKING_DIR / \"submission.parquet\", index=False)\n"
    '        print("[duck] commit: CPU-safe landing; scored rerun runs duck-l2", flush=True)\n'
)

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"


def fix_cell_source() -> str:
    body = FIXES.read_text()
    return (
        "# ============================================================================\n"
        "# Layer-2 fix cell. Inlined from submission/_duck_fixes/duck_fixes.py by\n"
        "# build_duck_l2.py — edit that file and rebuild, never edit this cell.\n"
        "# FIX A: game_over keeps the working world model (auto-reset resumes the SAME\n"
        "#        level; wiping it was a pure re-discovery tax on deep levels).\n"
        "# FIX B: images counted at true vision-token cost, not len(base64)/3 — stops\n"
        "#        premature history eviction (measured ~29% of budget at full history).\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "_fix_results = apply_all()\n"
        "if any('FAIL' in line for line in _fix_results):\n"
        "    print('[duck-fix] WARNING: at least one fix did not apply', flush=True)\n"
    )


def main() -> None:
    nb = json.loads(BASE.read_text())
    seen = {"setup": False, "run": False, "hook": False}
    cells = []

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if cell["cell_type"] == "code" and "setup_commands.json" in src:
            if CPU_SAFE_SETUP_OLD not in src:
                raise SystemExit("setup_commands loop did not match — upstream changed")
            src = src.replace(CPU_SAFE_SETUP_OLD, CPU_SAFE_SETUP_NEW)
            seen["setup"] = True

        elif cell["cell_type"] == "code" and HOOK_MARKER in src:
            src = fix_cell_source()
            seen["hook"] = True

        elif cell["cell_type"] == "code" and "await bm.run(" in src:
            if CPU_SAFE_RUN_OLD not in src:
                raise SystemExit("run-cell block did not match — upstream changed")
            src = src.replace(CPU_SAFE_RUN_OLD, CPU_SAFE_RUN_NEW)
            seen["run"] = True

        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise SystemExit(f"anchors never found: {missing}")

    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    attachments = sum(1 for c in cells if c.get("attachments"))
    print(f"wrote {OUT}  ({len(cells)} cells, {attachments} with attachments, "
          f"anchors: {sorted(k for k, v in seen.items() if v)})")


if __name__ == "__main__":
    main()
