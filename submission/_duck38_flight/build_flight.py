#!/usr/bin/env python3
"""build_flight.py — THE PACK FLIGHT ARMS (tp1 / tp2 / tp24): duck38-v12 base + the
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
GRAFT2_PY = SUB / "_throughput_v1" / "graft_control.py"
GRAFT4_PY = SUB / "_throughput_v1" / "graft_explore.py"
EXPLORER_PY = SUB / "_throughput_v1" / "frontier_explorer.py"
GRAFT5_PY = SUB / "_throughput_v1" / "graft_emission.py"
BASE_CODE_SHA256_PREFIX = "dc2c36f8"

# Pack 1 flight config = the tp1b "mech24" smoke phase: hysteresis trim + 24k
# window + notes + time guard, STOCK yield (60 s) and tool steps, batch cap 30.
# (The first tp smoke read FAIL with yield 900 / tool steps 8 / cap 10.)
# tp1b (stock vs hysteresis 24k/50%) also read FAIL: the deep cut starved the
# model of recent context (1-3 turns of history vs stock's 5-8) and it
# re-investigated instead of acting. Pack 1 in the flight arms is therefore
# reduced to its zero-risk parts: notes survive GAME_OVER + the 9 h time guard.
PACK1 = {
    "TP_ENABLE": "1", "TP_TRIM_LOW_WATER": "1.0", "TP_CONTEXT_WINDOW": "0",
    "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1", "TP_KEEP_NOTES_ON_GAME_OVER": "1",
    "TP_BATCH_CAP": "0",
}
PACK2 = {
    "TP2_SUMMARY": "1", "TP2_PROBE": "1", "TP2_PROBE_CLICKS": "3", "TP2_STALL": "1",
    "TP2_STALL_T1": "10", "TP2_STALL_T2": "30", "TP2_STALL_RESETS_PER_LEVEL": "2",
    "TP2_STREAK": "1", "TP2_STREAK_N": "3", "TP2_DIFF": "1",
}
PACK4 = {"TP4_STALL_T3": "30", "TP4_BUDGET": "800", "TP4_RUNS_PER_LEVEL": "1", "TP4_ENDGAME_S": "300"}
PACK5 = {"TP5_WM_FROM_REASONING": "1", "TP5_ACT_FLOOR": "3"}
ARMS = {
    "tp1": {"slug": "arc3-duck38-tp1", "flags": {**PACK1, **PACK2, **PACK4, **PACK5, "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}},
    "tp2": {"slug": "arc3-duck38-tp2", "flags": {**PACK1, **PACK2, **PACK4, **PACK5, "TP2_ENABLE": "1", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}},
    "tp24": {"slug": "arc3-duck38-tp24", "flags": {**PACK1, **PACK2, **PACK4, **PACK5, "TP2_ENABLE": "1", "TP4_ENABLE": "1", "TP5_ENABLE": "0"}},
    # THE SMOKE-PASSED ARM (arc3-tp5-smoke, 08-30: levels 0.88->1.04, zero 9->6,
    # score 3.50->4.40; tn36 0->2, tr87/m0r0/wa30/bp35 0->1): stock harness +
    # emission graft ONLY. Pack 1 stays neutral (notes+guard), Packs 2/4 off.
    "tp5em": {"slug": "arc3-duck38-tp5em", "flags": {**PACK1, **PACK2, **PACK4, **PACK5,
                                                     "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "1"}},
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
# All three graft modules ride every arm; TP2_ENABLE / TP4_ENABLE select the arm.
# effort_medium and the v7/v8 explorer grafts are DELIBERATELY ABSENT and purged.
'''

GRAFT_CELL_TAIL = r'''
_TP_DIR = WORKING_DIR / "tp_bundle"
_TP_DIR.mkdir(parents=True, exist_ok=True)
(_TP_DIR / "graft_throughput.py").write_text(_TP_SOURCE, encoding="utf-8")
(_TP_DIR / "graft_control.py").write_text(_TP2_SOURCE, encoding="utf-8")
(_TP_DIR / "frontier_explorer.py").write_text(_FE_SOURCE, encoding="utf-8")
(_TP_DIR / "graft_explore.py").write_text(_TP4_SOURCE, encoding="utf-8")
(_TP_DIR / "graft_emission.py").write_text(_TP5_SOURCE, encoding="utf-8")
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
assert _tp_state["enabled"] and _tp_state["trim_low_water"] == 1.0 \
    and _tp_state["context_window"] == 0 and _tp_state["yield_seconds"] == -1.0 \
    and _tp_state["tool_steps"] == -1 and _tp_state["keep_notes_on_game_over"] \
    and _tp_state["batch_cap"] == 0, "flight config not in effect: " + repr(_tp_state)
print("[throughput] status:", _tp_state)
_tcmod = _importlib.import_module("graft_control")
_tc_status = _tcmod.install()
print("[control]", _tc_status)
assert _tc_status == "control: OK", "control graft must be live, got: " + repr(_tc_status)
_temod = _importlib.import_module("graft_explore")
_te_status = _temod.install()
print("[explore]", _te_status)
assert _te_status == "explore: OK", "explore graft must be live, got: " + repr(_te_status)
_tmmod = _importlib.import_module("graft_emission")
_tm_status = _tmmod.install()
print("[emission]", _tm_status)
assert _tm_status == "emission: OK", "emission graft must be live, got: " + repr(_tm_status)
assert _tmmod.enabled() == (os.environ.get("TP5_ENABLE") == "1")
print("[emission] status:", _tmmod.status())
print("[control] status:", _tcmod.status())
print("[explore] status:", _temod.status())
assert _tcmod.enabled() == (os.environ.get("TP2_ENABLE") == "1")
assert _temod.enabled() == (os.environ.get("TP4_ENABLE") == "1")

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


def main(arm: str = "tp1") -> None:
    spec = ARMS[arm]
    slug = spec["slug"]
    flight_config = spec["flags"]
    out_dir = HERE / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = json.loads(BASE_NB.read_text())
    base_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    base_sha = hashlib.sha256(base_code.encode()).hexdigest()
    assert base_sha.startswith(BASE_CODE_SHA256_PREFIX), (
        f"base v12 notebook drifted from the flown-1.55 bytes ({base_sha[:16]}) — re-attest")

    graft_src = GRAFT_PY.read_text()
    graft2_src = GRAFT2_PY.read_text()
    graft4_src = GRAFT4_PY.read_text()
    fe_src = EXPLORER_PY.read_text()
    graft5_src = GRAFT5_PY.read_text()
    assert "def install() -> str:" in graft5_src
    assert "def install() -> str:" in graft_src and "def time_guard_per_game_s" in graft_src
    assert "def run_probe" in graft2_src and "def run_explorer" in graft4_src and "class FrontierExplorer" in fe_src

    run_hits = [i for i, c in enumerate(nb["cells"])
                if c["cell_type"] == "code" and MARK_RUN in "".join(c["source"])]
    assert len(run_hits) == 1, run_hits
    run_idx = run_hits[0]

    flag_lines = "".join(f'os.environ["{k}"] = "{v}"\n' for k, v in flight_config.items())
    purge = ("for _stale in " + repr(PURGE) + ":\n    os.environ.pop(_stale, None)\n\n")
    graft_cell_text = (GRAFT_CELL_HEAD + flag_lines + purge
                       + "_TP_SOURCE = " + repr(graft_src) + "\n_TP2_SOURCE = " + repr(graft2_src)
                       + "\n_FE_SOURCE = " + repr(fe_src) + "\n_TP4_SOURCE = " + repr(graft4_src)
                       + "\n_TP5_SOURCE = " + repr(graft5_src) + "\n"
                       + GRAFT_CELL_TAIL)
    for s_ in (graft_src, graft2_src, graft4_src, fe_src, graft5_src):
        assert repr(s_) in graft_cell_text

    nb["cells"].insert(run_idx, {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": graft_cell_text.splitlines(keepends=True),
    })

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined and MARK_SMOKE in joined
    assert joined.index(MARK_ATTEST) < joined.index(MARK_SMOKE)
    assert joined.index(MARK_SMOKE) < joined.index('os.environ["TP_ENABLE"] = "1"')
    for flag, val in flight_config.items():
        line = f'os.environ["{flag}"] = "{val}"'
        assert joined.count(line) == 1, (flag, joined.count(line))
        assert joined.index(line) < joined.index(MARK_RUN), flag
    assert 'EFFORT_MEDIUM"] = "1"' not in joined
    assert "KAGGLE_IS_COMPETITION_RERUN" in joined
    assert "Qwen/Qwen3.8-27B-FP8" in joined
    assert len(nb["cells"]) == 12

    (out_dir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
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
    print("built", slug, "arm", arm, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(nb["cells"]), "notebook bytes:", (out_dir / f"{slug}.ipynb").stat().st_size)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "tp1")
