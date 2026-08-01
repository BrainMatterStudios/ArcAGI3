#!/usr/bin/env python3
"""Build submission/_duck_variance/duck-variance.ipynb — base duck, wider sampling.

A max-over-draws arm. See variance_pack.py for the argument; in short, private
scores are fixed at run time and we keep the best two submissions ever made, so the
objective is the maximum over draws rather than the mean, and variance is an asset.

Built on PLAIN base duck, deliberately not on the context-levers arm: the n=8 base
distribution (mean 0.9288, sd 0.1947) is the only distribution we have measured, and
this arm is calibrated against it. Stacking two changes would leave us comparing a
draw against a distribution that no longer describes it.

Usage:  python3 submission/_duck_variance/build_duck_variance.py
"""
import io
import json
import re
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
PACK = Path(__file__).parent / "variance_pack.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-variance.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
ARM_MARKER = "[variance] ARM ACTIVE"

WANT_TEMPERATURE = 0.9
WANT_TOP_K = 50
WANT_TOP_P = 0.98


def _strip_prose(src: str) -> str:
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return " ".join(out)


def patch_cell_source() -> str:
    body = PACK.read_text()
    return (
        "# ============================================================================\n"
        "# Deliberate variance widening. Inlined from submission/_duck_variance/\n"
        "# variance_pack.py by build_duck_variance.py — edit that file and rebuild.\n"
        "#\n"
        "# Runs after the bundle is on sys.path and the benchmark object is restored,\n"
        "# before bm.run(). Model, serving, concurrency, budgets, game list untouched;\n"
        "# the ONLY delta is the sampling distribution.\n"
        "# ============================================================================\n"
        "\n"
        f"{body}\n"
        "\n"
        "_variance_result = apply_all()\n"
        "\n"
        "# Hard verification. Scored-rerun logs are unreadable, so a warning would be\n"
        "# invisible forever; ERROR-vs-score is the only signal that reaches us, and an\n"
        "# ERROR costs no submission slot.\n"
        "from inference.agent import tool_agent as _ta\n"
        "\n"
        "print(f\"[variance] before: {_variance_result['before']}\", flush=True)\n"
        "print(f\"[variance] after : {_variance_result['after']}\", flush=True)\n"
        "\n"
        f'_want = {{"temperature": {WANT_TEMPERATURE}, "top_k": {WANT_TOP_K}, "top_p": {WANT_TOP_P}}}\n'
        "_got = {\n"
        '    "temperature": float(_ta._LOCAL_ANALYZER_TEMPERATURE),\n'
        '    "top_k": int(_ta._LOCAL_ANALYZER_TOP_K),\n'
        '    "top_p": float(_ta._LOCAL_ANALYZER_TOP_P),\n'
        "}\n"
        "if _got != _want:\n"
        '    raise RuntimeError(f"[variance] sampling did not land: got {_got}, want {_want}")\n'
        "\n"
        "# The sampler must stay unseeded or every draw is the same draw.\n"
        "if int(_ta._LOCAL_ANALYZER_SEED) != -1:\n"
        '    raise RuntimeError(f"[variance] sampler seed pinned to {_ta._LOCAL_ANALYZER_SEED}")\n'
        "\n"
        f'print("{ARM_MARKER} — temp 0.6->0.9, top_k 20->50, top_p 0.95->0.98", flush=True)\n'
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
        out_cell = dict(cell)
        out_cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out_cell["execution_count"] = None
            out_cell["outputs"] = []
        cells.append(out_cell)

    missing = [n for n, found in seen.items() if not found]
    if missing:
        raise SystemExit(f"never found these anchor cells: {missing}")

    joined = "\n".join("".join(c["source"]) for c in cells)
    for guard in ("if TRUE_SUBMISSION:", "skipping setup_commands"):
        if guard not in joined:
            raise SystemExit(f"duck-base lost its CPU-safe guard ({guard!r}) — refusing to build")

    # This arm changes sampling and NOTHING else. Anything touching serving, the
    # model path, scheduling or budgets would confound it against the n=8 baseline.
    hook_src = next(s for s in ("".join(c["source"]) for c in cells) if ARM_MARKER in s)
    code_only = _strip_prose(hook_src)
    for forbidden in ("MODEL_PATH", "concurrency", "max_runtime", "setup_commands",
                      "CONTEXT_WINDOW", "_estimate_tokens"):
        if re.search(rf"(?<![\w.]){re.escape(forbidden)}\s*=(?!=)", code_only):
            raise SystemExit(f"injected cell assigns {forbidden!r} — this arm must be sampling-only")

    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))

    meta = {
        "id": "ahmedmobasher86/arc-agi-3-duck-variance",
        "title": "arc-agi-3-duck-variance",
        "code_file": "duck-variance.ipynb",
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
    print(f"wrote {OUT}  ({len(cells)} cells, anchors: {sorted(seen)})")


if __name__ == "__main__":
    main()
