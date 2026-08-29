#!/usr/bin/env python3
"""build_duck38_tp1.py — THE PACK-1 FLIGHT ARM: duck38-v12 base + the
throughput graft, in its exact flight configuration.

ARM: the tracked v12 notebook (submission/_duck38_v12/arc3-duck38-v12.ipynb,
code-cell sha256 dc2c36f8…, the bytes that flew 1.55) replicated EXACTLY, plus
ONE inserted cell immediately before the run cell:
  * submission/_throughput_v1/graft_throughput.py embedded inline, written to
    WORKING_DIR/tp_bundle, imported, install() behind a HARD assert
  * every TP_* flag set explicitly (FLIGHT_CONFIG below)
  * stale grafts purged (effort_medium, explorer v7/v8)
  * the 9 h time guard applied to bm.solver.max_runtime_s_per_game on the
    scored path only (true_submission), from the measured setup time

FLIGHT CONFIG (the smoke arm's "tp" phase, byte-for-byte):
  TP_ENABLE=1 TP_TRIM_LOW_WATER=0.5 TP_CONTEXT_WINDOW=24576
  TP_YIELD_SECONDS=900 TP_TOOL_STEPS=8 TP_KEEP_NOTES_ON_GAME_OVER=1
  TP_BATCH_CAP=10

GATE: flies only after kernel arc3-tp-smoke reads PASS (or INCONCLUSIVE with
a positive levels delta) per docs/superpowers/plans/2026-08-29-pack1-throughput.md.

Usage:
  .venv/bin/python submission/_duck38_tp1/build_duck38_tp1.py
  .venv/bin/python submission/_duck38_tp1/validate_duck38_tp1.py
  cd submission/_duck38_tp1 && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
BASE_NB = SUB / "_duck38_v12" / "arc3-duck38-v12.ipynb"
GRAFT_PY = SUB / "_throughput_v1" / "graft_throughput.py"
KERNEL_SLUG = "arc3-duck38-tp1"
BASE_CODE_SHA256_PREFIX = "dc2c36f8"

FLIGHT_CONFIG = {
    "TP_ENABLE": "1",
    "TP_TRIM_LOW_WATER": "0.5",
    "TP_CONTEXT_WINDOW": "24576",
    "TP_YIELD_SECONDS": "900",
    "TP_TOOL_STEPS": "8",
    "TP_KEEP_NOTES_ON_GAME_OVER": "1",
    "TP_BATCH_CAP": "10",
}
PURGE = ("EFFORT_MEDIUM", "EFFORT_DEAD_RETRY", "YIELD_CARRYOVER", "YIELD_SLICE_CAP",
         "EXPLORER", "EXPLORER_V8")

MARK_RUN = "run_context = contextlib.nullcontext()"
MARK_ATTEST = "attest: OK"
MARK_SMOKE = 'SMOKE_GAMES = ["vc33-5430563c", "sb26-7fbdac44", "tn36-ef4dde99"]'

GRAFT_CELL_HEAD = r'''# THE ONE DELTA vs flown v12: the Pack-1 throughput graft.
#
# Every flag is set EXPLICITLY so the scored run's own log states the entire
# configuration. WHY (docs/research-2026-08-29/R4-harness-throughput-audit.md):
#   TRIM_LOW_WATER=0.5   stock drops one history block per turn, so the vLLM
#                        prefix cache is invalidated on nearly every call
#                        (hit 0-20%; 12-15 prefill tokens per generated
#                        token). Cutting to half the budget in one go keeps
#                        the prefix stable across turns.
#   CONTEXT_WINDOW=24576 real working memory is already 4-9 turns; fewer
#                        resident tokens => faster per-stream decode.
#   YIELD_SECONDS=900    the stock 60 s yield is shorter than one call
#                        (118 s median at concurrency 28): 26% of play wall
#                        was spent in slices that ended with no action.
#   TOOL_STEPS=8         bounds a turn's call loop (the yield binds first).
#   KEEP_NOTES=1         stock wipes the carried world model on GAME_OVER.
#   BATCH_CAP=10         blind 20-140-action batches burn efficiency and
#                        trigger GAME_OVERs.
#
# effort_medium and the explorer grafts are DELIBERATELY ABSENT and purged.
'''

GRAFT_CELL_TAIL = r'''
_TP_DIR = WORKING_DIR / "tp_bundle"
_TP_DIR.mkdir(parents=True, exist_ok=True)
(_TP_DIR / "graft_throughput.py").write_text(_TP_SOURCE, encoding="utf-8")
if str(_TP_DIR) not in sys.path:
    sys.path.insert(0, str(_TP_DIR))

import importlib as _importlib

_tpmod = _importlib.import_module("graft_throughput")
_tp_status = _tpmod.install()
print("[throughput]", _tp_status)
# HARD GATE: a scored run that silently plays stock v12 poisons the read; an
# ERROR costs no slot (2026-08-01 audit law).
assert _tp_status == "throughput: OK", "throughput graft must be live, got: " + repr(_tp_status)
_tp_state = _tpmod.status()
assert _tp_state["enabled"] and _tp_state["trim_low_water"] == 0.5 \
    and _tp_state["context_window"] == 24576 and _tp_state["yield_seconds"] == 900.0 \
    and _tp_state["tool_steps"] == 8 and _tp_state["keep_notes_on_game_over"] \
    and _tp_state["batch_cap"] == 10, "flight config not in effect: " + repr(_tp_state)
print("[throughput] status:", _tp_state)

# 9 h time guard (scored path only): 4 x 7,920 s + ~630 s setup left ~90 s of
# slack under the box; shrink the per-game cap only as far as the measured
# setup time requires (never grows it).
if true_submission:
    _stock_cap = float(getattr(bm.solver, "max_runtime_s_per_game", None) or 7920.0)
    _guarded_cap = _tpmod.time_guard_per_game_s(
        _stock_cap,
        setup_elapsed_s=max(0.0, time.time() - NOTEBOOK_START_EPOCH),
        games=110,
        concurrency=int(getattr(bm.solver, "concurrency", None) or 28),
    )
    bm.solver.max_runtime_s_per_game = _guarded_cap
    print(f"[throughput] time guard: per-game cap {_stock_cap:.0f} -> {_guarded_cap:.0f} s")
else:
    print("[throughput] time guard: inert (not a scored rerun)")
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    base_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    base_sha = hashlib.sha256(base_code.encode()).hexdigest()
    assert base_sha.startswith(BASE_CODE_SHA256_PREFIX), (
        f"base v12 notebook drifted from the flown-1.55 bytes ({base_sha[:16]}) — re-attest")

    graft_src = GRAFT_PY.read_text()
    assert "def install() -> str:" in graft_src
    assert "def time_guard_per_game_s" in graft_src

    run_hits = [i for i, c in enumerate(nb["cells"])
                if c["cell_type"] == "code" and MARK_RUN in "".join(c["source"])]
    assert len(run_hits) == 1, run_hits
    run_idx = run_hits[0]

    flag_lines = "".join(f'os.environ["{k}"] = "{v}"\n' for k, v in FLIGHT_CONFIG.items())
    purge = ("for _stale in " + repr(PURGE) + ":\n    os.environ.pop(_stale, None)\n\n")
    graft_cell_text = (GRAFT_CELL_HEAD + flag_lines + purge
                       + "_TP_SOURCE = " + repr(graft_src) + "\n" + GRAFT_CELL_TAIL)
    assert repr(graft_src) in graft_cell_text

    nb["cells"].insert(run_idx, {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": graft_cell_text.splitlines(keepends=True),
    })

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined and MARK_SMOKE in joined
    assert joined.index(MARK_ATTEST) < joined.index(MARK_SMOKE)
    assert joined.index(MARK_SMOKE) < joined.index('os.environ["TP_ENABLE"] = "1"')
    for flag, val in FLIGHT_CONFIG.items():
        line = f'os.environ["{flag}"] = "{val}"'
        assert joined.count(line) == 1, (flag, joined.count(line))
        assert joined.index(line) < joined.index(MARK_RUN), flag
    assert 'EFFORT_MEDIUM"] = "1"' not in joined
    assert "KAGGLE_IS_COMPETITION_RERUN" in joined
    assert "Qwen/Qwen3.8-27B-FP8" in joined
    assert len(nb["cells"]) == 12

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(nb["cells"]), "notebook bytes:", (HERE / f"{KERNEL_SLUG}.ipynb").stat().st_size)


if __name__ == "__main__":
    main()
