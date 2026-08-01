#!/usr/bin/env python3
"""Build the GPU measurement rig — a commit kernel that scores arms without a slot.

WHY. Until now every arm cost a submission slot to evaluate, at one per day, against
a public-LB noise floor that needs ~60 draws to resolve +0.10. Thirty-four scored
submissions produced one usable distribution and zero measured improvements.

`make_benchmark_kaggle_official_110(competition_sim=True)` builds the submission-shaped
110-game benchmark from the 25 public games and starts a local CompetitionArcadeServer:
shared scorecard, hidden baselines, clone IDs, one make() per environment, level-resets
forced. A commit run therefore produces a real, submission-shaped score, and commit
output IS retrievable (`kaggle kernels output`) where scored-rerun output is not.

BUDGET. 110 games at concurrency 28 is 4 waves, so wall time is about 4x the per-game
box plus ~10 min of model load:
    7920s/game (the real box)  ->  8.8h  ->  ~3 sweeps per 30h week
     900s/game                 ->  1.0h  ->  ~25 sweeps
     600s/game                 ->  0.7h  ->  ~35 sweeps
A shortened box changes the regime, so absolute scores are NOT comparable to real
submissions -- but arms are comparable to each other, which is what an A/B needs.

WHAT IS REWRITTEN relative to duck-base:
  cell 8   force the serve. duck-base skips setup_commands unless
           KAGGLE_IS_COMPETITION_RERUN, and the rig needs Qwen actually served.
  cell 12  optional pack injection (the arm under test), same hook cell every arm uses.
  cell 14  replaced wholesale. duck-base's submission branch polls a Kaggle gateway
           that does not exist in a commit run, so the rig substitutes its own run:
           competition_sim benchmark, budget override, per-game score dump.

Usage:
    python3 submission/_rig/build_rig.py                          # baseline, no pack
    RIG_PACK=submission/_duck_boardfix/board_fix.py \
    RIG_LABEL=boardfix RIG_BUDGET=600 python3 submission/_rig/build_rig.py
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
OUT_DIR = Path(__file__).parent

PACK = os.environ.get("RIG_PACK", "")
LABEL = os.environ.get("RIG_LABEL", "base")
BUDGET = float(os.environ.get("RIG_BUDGET", 600))
SLUG = f"arc-agi-3-rig-{LABEL}"
OUT = OUT_DIR / f"rig-{LABEL}.ipynb"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
SERVE_GUARD = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it"
RUN_MARKER = "# Build the live competition game list from the gateway's available environments."


def hook_cell() -> str:
    if not PACK:
        return (
            "# rig: baseline arm, no pack injected.\n"
            f'print("[rig] pack: none (label={LABEL})", flush=True)\n'
        )
    body = Path(PACK).read_text()
    return (
        f"# rig: arm under test, inlined from {PACK}.\n"
        f"{body}\n"
        "\n"
        "_rig_pack = apply_all()\n"
        "for _n, _ok in _rig_pack.items():\n"
        "    print(f\"[rig] pack {_n}: {'OK' if _ok else 'FAIL'}\", flush=True)\n"
        "if not all(_rig_pack.values()):\n"
        '    raise RuntimeError(f"[rig] pack did not apply: {_rig_pack}")\n'
        f'print("[rig] pack: {LABEL} ACTIVE", flush=True)\n'
    )


def run_cell() -> str:
    return f'''# rig: competition-simulated run. Replaces duck-base's submission cell, whose
# gateway poll cannot succeed in a commit run.
import json, time, traceback
import taaf.standard_benchmarks as _sb

print((BUNDLE_DIR / "preamble.txt").read_text())
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

_t0 = time.time()
_rig = _sb.make_benchmark_kaggle_official_110(solver=bm.solver, competition_sim=True)

# Shorten the per-game box so a sweep fits the weekly quota. This changes the regime,
# so scores are comparable BETWEEN ARMS and not to real submissions.
_budget = {BUDGET}
for _attr in ("max_runtime_s_per_game", "max_runtime_s"):
    if hasattr(_rig.solver, _attr):
        setattr(_rig.solver, _attr, _budget)
print(f"[rig] games={{len(_rig.games)}} per_game_budget={{_budget}}s "
      f"concurrency={{getattr(_rig.solver, 'concurrency', '?')}}", flush=True)

_err = None
try:
    await _rig.run(soft_end_time=None, runtime_environment=target, minimal_diagnostics=False)
except Exception:
    _err = traceback.format_exc()
    print("[rig] run raised:\\n" + _err, flush=True)

# Per-game scores are the deliverable. Dump whatever the benchmark exposes rather than
# assuming a shape -- an unreadable measurement must not silently produce a verdict.
_rows = []
for _gr in (getattr(_rig, "game_runs", None) or []):
    try:
        _rows.append({{
            "game_id": getattr(_gr, "game_id", None),
            "score": getattr(_gr, "score", None),
            "levels_completed": getattr(_gr, "levels_completed", None),
            "actions": len(getattr(_gr, "history", []) or []),
        }})
    except Exception as _e:
        _rows.append({{"error": repr(_e)}})

_out = {{
    "label": "{LABEL}", "pack": "{PACK}", "per_game_budget": _budget,
    "elapsed_s": round(time.time() - _t0, 1), "n_games": len(_rows),
    "error": _err, "rows": _rows,
}}
(WORKING_DIR / "rig_result.json").write_text(json.dumps(_out, indent=1, default=str))

_scored = [r.get("score") for r in _rows if isinstance(r.get("score"), (int, float))]
if _scored:
    print(f"[rig] RESULT label={LABEL} n={{len(_scored)}} "
          f"mean={{sum(_scored)/len(_scored):.4f}} max={{max(_scored):.4f}}", flush=True)
else:
    print(f"[rig] RESULT label={LABEL} — NO SCORES PARSED ({{len(_rows)}} rows); "
          "see rig_result.json", flush=True)
print(f"[rig] DONE in {{_out['elapsed_s']}}s", flush=True)
'''


def main() -> None:
    nb = json.loads(BASE.read_text())
    seen = {"serve": False, "hook": False, "run": False}
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            if SERVE_GUARD in src:
                src = src.replace(SERVE_GUARD, "if True:  # rig: force the serve — a commit run must serve Qwen")
                seen["serve"] = True
            elif HOOK_MARKER in src:
                src = hook_cell()
                seen["hook"] = True
            elif RUN_MARKER in src:
                src = run_cell()
                seen["run"] = True
        out = dict(cell)
        out["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out["execution_count"] = None
            out["outputs"] = []
        cells.append(out)

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise SystemExit(f"never found anchors: {missing}")
    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)}")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{SLUG}",
        "title": SLUG,
        "code_file": OUT.name,
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
    }, indent=2))
    print(f"wrote {OUT}  (label={LABEL} pack={PACK or 'none'} budget={BUDGET}s, anchors {sorted(seen)})")


if __name__ == "__main__":
    main()
