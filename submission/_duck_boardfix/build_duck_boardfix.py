#!/usr/bin/env python3
"""Build submission/_duck_boardfix/duck-boardfix.ipynb — base duck + corrected board_changed.

Takes submission/_duck_base/duck-base.ipynb (which carries the CPU-safe commit guards)
and inlines board_fix.py into the "Customization hook" cell — after the source
bundle is on sys.path and the benchmark object has been restored, before bm.run(). The
Kaggle source dataset is read-only, so the pack must be a monkey-patch applied from the
notebook. Same mechanism as _duck_effects, and the same hard-verification stance: this
cell raises rather than warning, because a scored run cannot be inspected afterwards and
a warning would be invisible forever.

THE HYPOTHESIS (one variable: the harness tells the model something false)
    `board_changed` is computed as an exact frame comparison (framework/solver.py:704),
    but 18 of 25 games tick a step-budget bar into the frame BEFORE the legality check.
    Measured over 800 random actions per game, the harness asserts "the board changed"
    on 100% of actions in lf52, vc33, s5i5, tu93 and bp35, where the truth is 0.4%,
    0.5%, 4.7%, 35.4% and 63.2%. Corpus-wide, 56.4% of actions carry a wrong value.

    That value is put in front of the model every turn as a fact about the world
    (tool_agent.py:1095-1098). This arm recomputes it with the HUD region excluded and
    changes nothing else. Games with no HUD are untouched -- ar25 and ft09 measure
    exactly 0.000 corrected, which is the control.

    Falsifiable and could lose: the model may be leaning on the miscalibrated signal in
    a way that accidentally helps. An honest "nothing happened" could equally provoke a
    repeat-until-something-moves loop. That is why this is an arm, not a bugfix.

Usage:  python3 submission/_duck_boardfix/build_duck_boardfix.py
"""
import ast
import io
import json
import re
import tokenize
from pathlib import Path


def _strip_prose(src: str) -> str:
    """Return `src` with comments and string literals blanked out.

    The single-variable check below must look at what the cell *does*, not at what
    its documentation says it deliberately leaves alone.
    """
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        # Fall back to the raw source: a tokenizer failure must not silently
        # disable the guard.
        return src
    return " ".join(out)

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
PACK = Path(__file__).parent / "board_fix.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-boardfix.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
ARM_MARKER = "[boardfix] ARM ACTIVE"


def patch_cell_source() -> str:
    body = PACK.read_text()
    return (
        "# ============================================================================\n"
        "# Corrected board_changed. Inlined from submission/_duck_boardfix/board_fix.py\n"
        "# by build_duck_boardfix.py — edit that file and rebuild, never edit this cell.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "_bf = apply_all()\n"
        "\n"
        "# Hard verification: scored-rerun logs are unreadable, so ERROR-vs-score is the\n"
        "# only signal that reaches us, and an ERROR costs no submission slot.\n"
        "from inference.framework import solver as _sv\n"
        "import numpy as _np\n"
        "\n"
        "for _n, _ok in _bf.items():\n"
        "    print(f\"[boardfix] verify {_n}: {'OK' if _ok else 'FAIL'}\", flush=True)\n"
        "if not all(_bf.values()):\n"
        "    raise RuntimeError(f\"[boardfix] pack did not apply: {_bf}\")\n"
        "if not getattr(_sv._HarnessGameSession._execute_action, '_boardfix', False):\n"
        "    raise RuntimeError(\"[boardfix] _execute_action is not wrapped\")\n"
        "\n"
        "# Functional proof on synthetic boards: a lone HUD-row change must read as\n"
        "# 'no board change' on a HUD game and as 'changed' on a game without one.\n"
        "_a = _np.zeros((64, 64), dtype=_np.int8); _b = _a.copy(); _b[63, :7] = 3\n"
        "if boards_equal(_a, _b, 'tu93') is not True:\n"
        "    raise RuntimeError('[boardfix] HUD row not masked on tu93')\n"
        "if boards_equal(_a, _b, 'ar25') is not False:\n"
        "    raise RuntimeError('[boardfix] masked a game that has no HUD (ar25)')\n"
        "_c = _a.copy(); _c[30, 30] = 5\n"
        "if boards_equal(_a, _c, 'tu93') is not False:\n"
        "    raise RuntimeError('[boardfix] real board change was masked away')\n"
        "print('[boardfix] verify masking semantics: OK', flush=True)\n"
        "\n"
        f'print("{ARM_MARKER} — board_changed recomputed with the HUD region excluded", flush=True)\n'
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
        # Copy the original cell and override only `source`; rebuilding from scratch
        # drops keys nbconvert needs, notably `attachments` (the logo in cell 0).
        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    missing = [name for name, found in seen.items() if not found]
    if missing:
        raise SystemExit(f"never found these anchor cells: {missing}")

    joined = "\n".join("".join(c["source"]) for c in cells)
    for guard in ("if TRUE_SUBMISSION:", "skipping setup_commands"):
        if guard not in joined:
            raise SystemExit(f"duck-base lost its CPU-safe guard ({guard!r}) — refusing to build")

    # Nothing in this arm may touch serving, sampling or scheduling. Match actual
    # mutation — assignment, attribute write, env poke — not prose, so the docstrings
    # that *describe* what the arm leaves alone don't trip the check.
    hook_src = next(s for s in ("".join(c["source"]) for c in cells) if ARM_MARKER in s)
    code_only = _strip_prose(hook_src)
    for forbidden in ("MODEL_PATH", "temperature", "concurrency", "max_runtime", "setup_commands"):
        hit = re.search(rf"(?<![\w.]){re.escape(forbidden)}\s*=(?!=)|\.\s*{re.escape(forbidden)}\s*=(?!=)", code_only)
        if hit:
            raise SystemExit(
                f"injected cell assigns {forbidden!r} ({hit.group()!r}) — this arm must change board_changed only"
            )

    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))

    meta = {
        "id": "ahmedmobasher86/arc-agi-3-duck-boardfix",
        "title": "arc-agi-3-duck-boardfix",
        "code_file": "duck-boardfix.ipynb",
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
    print(f"wrote {OUT}  ({len(cells)} cells, {attachments} with attachments, anchors: {sorted(seen)})")


if __name__ == "__main__":
    main()
