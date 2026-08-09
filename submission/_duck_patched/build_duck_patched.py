#!/usr/bin/env python3
"""Build submission/_duck_patched/duck-patched.ipynb.

Takes the accessible duck bundle (submission/_repro/duck-repro.ipynb) and applies:

  1. The CPU-safe-commit guard, identical to _duck_base: interactive kernels only get a
     P100, which cannot serve Qwen3.6-27B-FP8, so the vLLM setup and the benchmark run
     are both gated on TRUE_SUBMISSION. The commit lands a dummy parquet; the scored
     rerun does the real work on the RTX 6000.

  2. The in-memory harness patches from duck_patches.py, inlined into the notebook's
     existing "Customization hook" cell (cell 11 in the source notebook) — after the
     source bundle is on sys.path and the benchmark is loaded, before bm.run().

Nothing here changes the model, sampling parameters, concurrency, per-game budget, the
game list, or the submission path. The only behavioural deltas are the ACTION7 round
trip and the animation metadata.

Usage:  .venv/bin/python submission/_duck_patched/build_duck_patched.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_repro/duck-repro.ipynb"
PATCHES = Path(__file__).parent / "duck_patches.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-patched.ipynb"

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
    '        print("[duck] commit: CPU-safe landing; scored rerun runs the patched duck", flush=True)\n'
)

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"


# Experiment-arm pins, emitted into the hook cell before apply_all. This build is
# the clean WMR draw: watchdog/HUD-mask/replay ON (their defaults), the un-A/B'd
# patch11/12 pinned OFF, grid-burner already default-off. Change these ONLY when
# the submitted experiment changes, and keep the submission description in sync
# (scripts/submit_gated.py cross-checks it).
# v9 (struct live probe, 2026-08-09): this dict is a LITERAL copy of ARM_ENV from
# submission/_ab_patch_closure/struct/struct-screen.ipynb cell 12 — the arm that
# produced the replicated 2.5x plan adoption and the uncertified +2 levels/wave.
# Copied literally (including the inert GRAPH_GRIND_* keys, dead while TAAF_GRAPH=0)
# so the live probe is a faithful transfer test of the configuration we measured,
# not a hybrid. Two knobs therefore differ from the settled patched arm v7/v8 in
# addition to the four struct flags: TAAF_ANIMATION (default 1 -> 0) and
# TAAF_WATCHDOG_STALL_S (default 600 -> 900).
EXPERIMENT_ENV = {
    "TAAF_WATCHDOG": "1",
    "TAAF_WATCHDOG_STALL_S": "900",
    "TAAF_HUD_MASK": "1",
    "TAAF_WIN_REPLAY": "1",
    "TAAF_ANTIFREEZE": "1",
    "TAAF_ANIMATION": "0",
    "TAAF_GRAPH": "0",
    "TAAF_GRAPH_GRIND_AGE_ACTIONS": "120",
    "TAAF_GRAPH_GRIND_AGE_TURNS": "10",
    "TAAF_GRAPH_GRIND_MAX_PER_LEVEL": "2",
    "TAAF_COMPACT": "0",
    # patch13 archetype playbook: default ON outside pins, but pinned OFF here so
    # the settled v6 arm stays clean until the playbook is A/B'd on its own.
    "TAAF_PLAYBOOK": "0",
    "TAAF_GRID_BURNER": "0",
    "TAAF_DIFF_LINES": "1",
    "TAAF_WIGGLE": "1",
    "TAAF_DISPATCH": "1",
    "TAAF_STRUCT": "1",
}


def patch_cell_source() -> str:
    """The customization-hook cell: duck_patches.py inlined, then applied."""
    body = PATCHES.read_text()
    pins = "".join(
        f'_os.environ["{k}"] = "{v}"\n' for k, v in sorted(EXPERIMENT_ENV.items())
    )
    return (
        "# ============================================================================\n"
        "# In-memory harness patches. Inlined from submission/_duck_patched/duck_patches.py\n"
        "# by build_duck_patched.py — edit that file and rebuild, never edit this cell.\n"
        "#\n"
        "# The Kaggle source dataset is read-only, so these are monkey-patches. They run\n"
        "# after the bundle is on sys.path and the benchmark is loaded, before bm.run().\n"
        "# Model, sampling, concurrency, budgets and the game list are untouched.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "# --- experiment-arm pins (see EXPERIMENT_ENV in build_duck_patched.py) ---\n"
        "import os as _os\n"
        f"{pins}"
        'print(f"[duck-patch] experiment pins: {' + repr(sorted(EXPERIMENT_ENV.items())) + '}", flush=True)\n'
        "\n"
        "_patch_results = apply_all()\n"
        "if any"
        '("FAIL" in line for line in _patch_results):\n'
        "    # Loud but non-fatal: a failed patch means upstream moved, and we want that in\n"
        "    # the log rather than a silently unpatched scored run.\n"
        '    print("[duck-patch] WARNING: at least one patch did not apply", flush=True)\n'
        "\n"
        "# Patch-12 diagnostics land in the kernel log at interpreter exit (bm.run() has\n"
        "# no after-run hook in this notebook; per-event [compact]/[plan-queue] lines\n"
        "# stream during the run regardless).\n"
        "import atexit as _atexit\n"
        "_atexit.register(\n"
        '    lambda: print(f"[duck-patch] compact diagnostics: {COMPACT_DIAGNOSTICS} '
        'antifreeze: {ANTIFREEZE_DIAGNOSTICS} diff-lines: {DIFF_LINES_DIAGNOSTICS} '
        'run-probe: {RUN_PROBE_DIAGNOSTICS} dispatch: {DISPATCH_DIAGNOSTICS} '
        'verify: {VERIFY_DIAGNOSTICS}", flush=True)\n'
        ")\n"
    )


def main() -> None:
    nb = json.loads(BASE.read_text())
    seen = {"setup": False, "run": False, "hook": False}
    cells = []

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if cell["cell_type"] == "code" and "setup_commands.json" in src:
            if CPU_SAFE_SETUP_OLD not in src:
                raise SystemExit("setup_commands loop did not match — upstream notebook changed")
            src = src.replace(CPU_SAFE_SETUP_OLD, CPU_SAFE_SETUP_NEW)
            seen["setup"] = True

        elif cell["cell_type"] == "code" and "DATASET_SOURCES =" in src:
            if "ahmedmobasher86/arcagi3-agent" not in src:
                src = src.replace(
                    'DATASET_SOURCES = [', 
                    'DATASET_SOURCES = ["ahmedmobasher86/arcagi3-agent", '
                )

        elif cell["cell_type"] == "code" and "await bm.run(" in src:
            if CPU_SAFE_RUN_OLD not in src:
                raise SystemExit("run-cell block did not match — upstream notebook changed")
            src = src.replace(CPU_SAFE_RUN_OLD, CPU_SAFE_RUN_NEW)
            seen["run"] = True

        elif cell["cell_type"] == "code" and HOOK_MARKER in src:
            src = patch_cell_source()
            seen["hook"] = True

        # Copy the original cell and override only `source`. Rebuilding a cell from
        # scratch drops keys nbconvert needs — notably `attachments`, which carries the
        # embedded logo in cell 0; without it the commit fails at render time with
        # "InvalidNotebook: missing attachment: tufa_labs.png" even though every code
        # cell ran correctly.
        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    missing = [name for name, found in seen.items() if not found]
    if missing:
        raise SystemExit(f"never found these anchor cells: {missing}")

    # A dropped cell key is invisible until Kaggle renders the notebook and the commit
    # fails. Catch it here instead.
    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    attachments = sum(1 for c in cells if c.get("attachments"))
    print(
        f"wrote {OUT}  ({len(cells)} cells, {attachments} with attachments, "
        f"anchors: {sorted(seen)})"
    )


if __name__ == "__main__":
    main()
