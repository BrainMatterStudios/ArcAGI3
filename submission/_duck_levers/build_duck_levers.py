#!/usr/bin/env python3
"""Build submission/_duck_levers/duck-levers.ipynb — base duck + measured context levers.

Takes submission/_duck_base/duck-base.ipynb (which carries the CPU-safe commit guards)
and inlines harness_levers.py into the "Customization hook" cell — after the source
bundle is on sys.path and the benchmark object has been restored, before bm.run(). The
Kaggle source dataset is read-only, so the pack must be a monkey-patch applied from the
notebook. Same mechanism as _duck_effects, and the same hard-verification stance: this
cell raises rather than warning, because a scored run cannot be inspected afterwards and
a warning would be invisible forever.

THE HYPOTHESIS (one variable: the agent is context-starved)
    Both levers serve a single claim — the duck throws away most of the context it
    has already paid for, for two compounding reasons, one of them an arithmetic bug.

    L1  `_estimate_tokens` is len(json)//3. Measured against real usage.prompt_tokens
        over 3,556 recorded requests, that over-counts by 1.44x (p50 1.39, p90 1.66).
        The trimmer evicts oldest history until the *estimate* fits, and sits pinned
        at exactly the 31,744 ceiling on 11.8% of requests — so real prompts cap near
        22.0k of a 31.7k budget. //4 brings the ratio to ~1.08x, still erring high,
        which is the safe direction.
    L2  The analyzer window is 32,768 while vLLM serves --max-model-len 65536.
        Raised to 49,152, NOT the served maximum: `_reply_reserve_tokens` is
        hardcoded to 512, so a 65,536 window would leave ~5.8k tokens for generation
        against long thinking traces. At 49,152 the real prompt caps near 44.6k and
        generation keeps ~21k.

    Together: real prompt ceiling ~22.0k -> ~44.6k.

    Independent support for the direction: ARC Prize's own controlled experiment
    (technical report 4.3.1) moved one environment 0.0% -> 97.1% on a fixed model by
    changing the harness, and names context management over 64x64 frames as the
    central challenge. Published agent results put naive plan eviction at 34.7
    percentage points of task success.

    Falsifiable and could lose: more retained history means longer prefills against a
    132-min per-game wall clock at concurrency 28. The counterweight is that a larger
    window evicts less often, and front-eviction is what invalidates the vLLM prefix
    cache. Net throughput effect is genuinely unknown — that is what this draw buys.

DELIBERATELY NOT IN THIS BUNDLE — both rejected on measurement today, see
harness_levers.py for the full evidence so neither gets re-derived:
    L3 hard no-op guard   — unsafe. A byte-identical frame does not imply unchanged
                            state (13/25 games); wa30 ends the game on two
                            frame-identical actions. 83.8% of its savings are one game.
    L4 board lattice      — inert. The upscaled games do not render block-constant
                            frames, so the scale is not recoverable from frames, and
                            competition mode exposes no camera object.

Model, sampling, concurrency, per-game budget, game list and submission path: untouched.

Usage:  python3 submission/_duck_levers/build_duck_levers.py
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
PACK = Path(__file__).parent / "harness_levers.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-levers.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
ARM_MARKER = "[levers] ARM ACTIVE"
CONTEXT_WINDOW = 49152


def patch_cell_source() -> str:
    body = PACK.read_text()
    return (
        "# ============================================================================\n"
        "# Measured context levers. Inlined from submission/_duck_levers/harness_levers.py\n"
        "# by build_duck_levers.py — edit that file and rebuild, never edit this cell.\n"
        "#\n"
        "# Runs after the bundle is on sys.path and the benchmark object is restored,\n"
        "# before bm.run(). Model, sampling, concurrency, budgets, game list untouched.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        f"_lever_results = apply_all(context_window={CONTEXT_WINDOW})\n"
        "\n"
        "# Hard verification. A scored run that silently serves an unpatched harness is\n"
        "# indistinguishable from base duck and burns the draw on a mislabelled result.\n"
        "# Scored-rerun logs are unreadable, so this raises instead of warning.\n"
        "import json as _json\n"
        "from inference.agent import tool_agent as _ta\n"
        "\n"
        "for _name, _ok in _lever_results.items():\n"
        "    print(f\"[levers] verify {_name}: {'OK' if _ok else 'FAIL'}\", flush=True)\n"
        "if not all(_lever_results.values()):\n"
        '    raise RuntimeError(f"[levers] pack did not apply: {_lever_results}")\n'
        "\n"
        "# Functional proof, not just a flag: the estimator must now yield ~4 chars per\n"
        "# token instead of ~3 on a realistic payload.\n"
        '_probe = {"role": "user", "content": "x" * 3000, "meta": {"a": [1, 2, 3] * 50}}\n'
        "_chars = len(_json.dumps(_probe, ensure_ascii=True, sort_keys=True, default=str))\n"
        "_ratio = _chars / _ta._estimate_tokens(_probe)\n"
        'print(f"[levers] estimator chars/token: {_ratio:.2f} (was 3.00)", flush=True)\n'
        "if not 3.9 <= _ratio <= 4.1:\n"
        '    raise RuntimeError(f"[levers] L1 did not land: chars/token={_ratio:.2f}")\n'
        "\n"
        f"if _ta._LOCAL_ANALYZER_CONTEXT_WINDOW != {CONTEXT_WINDOW}:\n"
        "    raise RuntimeError(\n"
        f'        f"[levers] L2 did not land: window={{_ta._LOCAL_ANALYZER_CONTEXT_WINDOW}}, expected {CONTEXT_WINDOW}")\n'
        "\n"
        "# The window must stay under what vLLM actually serves, or every request that\n"
        "# reaches the ceiling is a hard API error rather than a longer prompt.\n"
        "_served = 65536\n"
        f"_headroom = _served - int(({CONTEXT_WINDOW} - 1024) / 1.08)\n"
        'print(f"[levers] generation headroom at ceiling: ~{_headroom} tokens", flush=True)\n'
        "if _headroom < 12000:\n"
        '    raise RuntimeError(f"[levers] context window leaves only {_headroom} tokens to generate")\n'
        "\n"
        f'print("{ARM_MARKER} — estimator //4, context window {CONTEXT_WINDOW}", flush=True)\n'
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
        "id": "ahmedmobasher86/arc-agi-3-duck-levers",
        "title": "arc-agi-3-duck-levers",
        "code_file": "duck-levers.ipynb",
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
