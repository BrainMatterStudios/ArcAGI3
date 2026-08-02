#!/usr/bin/env python3
"""Build submission/_duck_depth/duck-depth.ipynb — base duck + depth pack.

One variable: everything in the pack targets LEVELS COMPLETED, the only quantity that
still scores (efficiency cap binds on 8/10 rig-completed and 67% of 66
episode-completed levels; 0.36 levels/game x 3.52% cap = 1.27 = our exact LB score).

D1a death-safe memory (re-ship of duck-l2 FIX A)
D1b mechanical world-model carry into cross_level_notes at level transitions
D1c RESET advertised with cost/semantics guidance (was stripped unconditionally)
D1d ACTION7 explained as UNDO (bare mapping fix measured inert: offered 229, chosen 0)

Verification is fail-not-warn and checks EFFECTS, not installs: the pack's verify()
exercises the wipe/carry logic on a synthetic agent state, confirms RESET survives the
name filter, and confirms the guidance renders exactly once.
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
PACK = Path(__file__).parent / "depth_pack.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-depth.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
ARM_MARKER = "[depth] ARM ACTIVE"


def patch_cell_source() -> str:
    body = PACK.read_text()
    return (
        "# ============================================================================\n"
        "# Depth pack. Inlined from submission/_duck_depth/depth_pack.py by\n"
        "# build_duck_depth.py — edit that file and rebuild, never edit this cell.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "_depth_results = apply_all()\n"
        "for _n, _ok in _depth_results.items():\n"
        "    print(f\"[depth] verify {_n}: {'OK' if _ok else 'FAIL'}\", flush=True)\n"
        "if not all(_depth_results.values()):\n"
        '    raise RuntimeError(f"[depth] pack did not apply: {_depth_results}")\n'
        "# Effect checks, not install checks. Raises on failure; scored-rerun logs are\n"
        "# unreadable so ERROR-vs-score is the only signal that reaches us.\n"
        "verify()\n"
        'print("[depth] ARM ACTIVE — death-safe memory, level carry, RESET+UNDO guidance", flush=True)\n'
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
                f"injected cell assigns {forbidden!r} ({hit.group()!r}) — this arm must be context-only"
            )

    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))

    meta = {
        "id": "ahmedmobasher86/arc-agi-3-duck-depth",
        "title": "arc-agi-3-duck-depth",
        "code_file": "duck-depth.ipynb",
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
