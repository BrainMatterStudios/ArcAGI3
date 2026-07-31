#!/usr/bin/env python3
"""Build submission/_duck_effects/duck-effects.ipynb — base duck + grounded effect memory (B1).

Takes submission/_duck_base/duck-base.ipynb (which already carries the CPU-safe commit
guards) and inlines effect_memory.py into the notebook's "Customization hook" cell —
after the source bundle is on sys.path and the benchmark is loaded, before bm.run().
This is the same mechanism duck-patched uses; the Kaggle source dataset is read-only,
so the pack must be a monkey-patch installed from the notebook itself.

HISTORY — why this file was rewritten (2026-07-31):
    The previous version only did `os.environ["EFFECT_MEMORY"] = "1"` and never imported
    effect_memory.py. Nothing in the shipped bundle reads that variable (its only other
    consumer is scratchpad/rl_gate/run_rollout.py, a local-harness path gated on
    APPLY_EFFECTS_PATCH). The kernel therefore ran as plain base duck while being
    labelled as the effect-memory arm. Setting a toggle is not the same as shipping the
    code the toggle gates — hence the hard verification block below, which fails the
    cell loudly rather than letting a scored run proceed unpatched.

Nothing here changes the model, sampling parameters, concurrency, per-game budget, the
game list, or the submission path. The only behavioural delta is the measured effect
block in the user prompt plus the system line binding CONFIRMED to observed transitions.

Usage:  python3 submission/_duck_effects/build_duck_effects.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
PACK = Path(__file__).parent / "effect_memory.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-effects.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"


def patch_cell_source() -> str:
    """The customization-hook cell: effect_memory.py inlined, enabled, then verified."""
    body = PACK.read_text()
    return (
        "# ============================================================================\n"
        "# B1 grounded effect memory. Inlined from submission/_duck_effects/effect_memory.py\n"
        "# by build_duck_effects.py — edit that file and rebuild, never edit this cell.\n"
        "#\n"
        "# The Kaggle source dataset is read-only, so this is a monkey-patch. It runs after\n"
        "# the bundle is on sys.path and the benchmark is loaded, before bm.run().\n"
        "# Model, sampling, concurrency, budgets and the game list are untouched.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "# The toggle effect_memory.py reads at call time. Set here so the arm identity\n"
        "# lives in one place alongside the code it gates.\n"
        "import os\n"
        'os.environ["EFFECT_MEMORY"] = "1"\n'
        "\n"
        "_effect_results = apply_all()\n"
        "\n"
        "# Hard verification. A scored run that silently serves an unpatched harness is\n"
        "# indistinguishable from base duck and burns a submission slot on a mislabelled\n"
        "# result — so prove the wrappers actually landed and the toggle actually reads on.\n"
        "from inference.agent import tool_agent as _ta\n"
        "\n"
        "_checks = {\n"
        '    "_summarize_step_sequence wrapped": getattr(\n'
        '        _ta.ToolAgent._summarize_step_sequence, "_effect_memory", False),\n'
        '    "_build_user_prompt wrapped": getattr(\n'
        '        _ta.ToolAgent._build_user_prompt, "_effect_memory", False),\n'
        '    "toggle reads enabled": _enabled(),\n'
        "}\n"
        "for _name, _ok in _checks.items():\n"
        '    print(f"[effects] verify {_name}: {\'OK\' if _ok else \'FAIL\'}", flush=True)\n'
        "if not all(_checks.values()):\n"
        '    raise RuntimeError(f"[effects] effect-memory pack did not apply: {_checks}")\n'
        "\n"
        "# End-to-end proof the measured block actually renders, using a throwaway memory:\n"
        "_probe = fresh_memory()\n"
        "record_results(_probe, [\n"
        '    {"executed": True, "level": 0, "executed_actions": ["ACTION1"], "board_changed": False},\n'
        '    {"executed": True, "level": 0, "executed_actions": ["ACTION1"], "board_changed": False},\n'
        '    {"executed": True, "level": 0, "executed_actions": ["MOUSE(row=3, col=4)"],\n'
        '     "board_changed": True},\n'
        "])\n"
        "_block = build_effect_block(_probe)\n"
        'assert "ACTION1 (0/2)" in _block and "(3,4): 1/1" in _block, _block\n'
        'print("[effects] verify block renders: OK", flush=True)\n'
        'print(f"[effects] ARM ACTIVE — EFFECT_MEMORY={os.environ[\'EFFECT_MEMORY\']}", flush=True)\n'
    )


def main() -> None:
    nb = json.loads(BASE.read_text())
    seen = {"hook": False}
    cells = []

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if cell["cell_type"] == "code" and HOOK_MARKER in src:
            src = patch_cell_source()
            seen["hook"] = True

        # Copy the original cell and override only `source`. Rebuilding a cell from
        # scratch drops keys nbconvert needs — notably `attachments`, which carries the
        # embedded logo in cell 0; without it the commit fails at render time.
        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    missing = [name for name, found in seen.items() if not found]
    if missing:
        raise SystemExit(f"never found these anchor cells: {missing}")

    # duck-base must already carry the CPU-safe guards; if it does not, the commit run
    # would try to serve Qwen on a P100 and die.
    joined = "\n".join("".join(c["source"]) for c in cells)
    for guard in ("if TRUE_SUBMISSION:", "skipping setup_commands"):
        if guard not in joined:
            raise SystemExit(f"duck-base lost its CPU-safe guard ({guard!r}) — refusing to build")

    # A dropped cell key is invisible until Kaggle renders the notebook and the commit
    # fails. Catch it here instead.
    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))

    meta = {
        "id": "ahmedmobasher86/arc-agi-3-duck-effects",
        "title": "arc-agi-3-duck-effects",
        "code_file": "duck-effects.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "ahmedmobasher86/taaf-src-hybrid",
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    attachments = sum(1 for c in cells if c.get("attachments"))
    print(
        f"wrote {OUT}  ({len(cells)} cells, {attachments} with attachments, "
        f"anchors: {sorted(seen)})"
    )


if __name__ == "__main__":
    main()
