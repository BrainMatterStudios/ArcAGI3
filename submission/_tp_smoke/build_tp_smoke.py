#!/usr/bin/env python3
"""build_tp_smoke.py — the THROUGHPUT A/B SMOKE (Pack 1, 2026-08-29).

ARM: the duck38-v12 base (Aug-07 anim bundle + Qwen3.8-27B-FP8 repacked,
boot-attested — the bytes that flew 1.55) + ONE graft,
submission/_throughput_v1/graft_throughput.py, run as TWO PHASES inside one
vLLM boot on the scored GPU class:

  phase "stock"  TP_ENABLE=0   every graft seam is a pass-through
  phase "tp"     TP_ENABLE=1   hysteresis trim (low water 0.5), context
                               window 24576, yield 900 s, tool steps 8,
                               notes survive GAME_OVER, batch cap 10

GEOMETRY: all 25 public games per phase, 7,920 s per game, concurrency 28
(the serialized solver's) => eval geometry, ~2.2 h per phase, ~4.6 h kernel.

THE READ (pre-registered, docs/superpowers/plans/2026-08-29-pack1-throughput.md
Task 8): per phase, mean actions/game, mean levels/game, mean score, turns,
generated tokens, prompt tokens, vLLM prefix-cache hit rate, gen tok/s.
  PASS          tp actions/game >= 2.0x stock AND levels/game >= stock - 0.15
                AND prefix hit >= 50%
  INCONCLUSIVE  actions >= 1.5x with levels within noise
  FAIL          levels/game < stock - 0.15 or actions < 1.5x

Usage:
  .venv/bin/python submission/_tp_smoke/build_tp_smoke.py
  .venv/bin/python submission/_tp_smoke/validate_tp_smoke.py
  cd submission/_tp_smoke && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
ROOT = SUB.parent
BASE_NB = SUB / "_duck38_v12" / "arc3-duck38-v12.ipynb"
GRAFT_PY = SUB / "_throughput_v1" / "graft_throughput.py"
GRAFT2_PY = SUB / "_throughput_v1" / "graft_control.py"
ARMS = {
    # arm: (kernel slug, phases as (name, env), read rule)
    "tp": {
        "slug": "arc3-tp-smoke",
        "phases": [("stock", {"TP_ENABLE": "0", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}),
                   ("tp", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0"})],
        "read": "throughput",
    },
    # Pack 1b: the mechanical half only (hysteresis trim + 24k window + notes +
    # time guard), with the STOCK 60 s yield / unlimited tool steps that the
    # tp smoke showed re-ground the model after every investigation call, and
    # a looser batch cap. Stock vs mech24 inside one boot.
    "tp1b": {
        "slug": "arc3-tp1b-smoke",
        "phases": [("stock", {"TP_ENABLE": "0", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}),
                   ("mech24", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0",
                               "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1",
                               "TP_CONTEXT_WINDOW": "24576", "TP_BATCH_CAP": "30"})],
        "read": "throughput",
    },
    "tp2": {
        "slug": "arc3-tp2-smoke",
        "phases": [("tp", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}),
                   ("tp2", {"TP_ENABLE": "1", "TP2_ENABLE": "1", "TP4_ENABLE": "0", "TP5_ENABLE": "0"})],
        "read": "control",
    },
    # Pack 2 on the corrected Pack-1 base (tp1b mech24): async/cheap summaries.
    "tp2b": {
        "slug": "arc3-tp2b-smoke",
        "phases": [("mech24", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0",
                               "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1",
                               "TP_CONTEXT_WINDOW": "24576", "TP_BATCH_CAP": "30"}),
                   ("mech24_ctl", {"TP_ENABLE": "1", "TP2_ENABLE": "1", "TP4_ENABLE": "0",
                                   "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1",
                                   "TP_CONTEXT_WINDOW": "24576", "TP_BATCH_CAP": "30"})],
        "read": "control",
    },
    # Clean Pack 2 read: STOCK Pack 1 (only notes-on-GAME_OVER + time guard kept)
    # vs the same + control (async summaries).
    "tp2c": {
        "slug": "arc3-tp2c-smoke",
        "phases": [("stock", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0",
                              "TP_TRIM_LOW_WATER": "1.0", "TP_CONTEXT_WINDOW": "0",
                              "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1", "TP_BATCH_CAP": "0"}),
                   ("stock_ctl", {"TP_ENABLE": "1", "TP2_ENABLE": "1", "TP4_ENABLE": "0",
                                  "TP_TRIM_LOW_WATER": "1.0", "TP_CONTEXT_WINDOW": "0",
                                  "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1", "TP_BATCH_CAP": "0"})],
        "read": "control",
    },
    # Gentle hysteresis: 43k window, cut 25% at a time (post-cut context >= stock's
    # steady state), everything else stock. tp1b showed 24k/50% starves recency.
    "tp1c": {
        "slug": "arc3-tp1c-smoke",
        "phases": [("stock", {"TP_ENABLE": "0", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}),
                   ("hyst43", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0",
                               "TP_TRIM_LOW_WATER": "0.75", "TP_CONTEXT_WINDOW": "43008",
                               "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1", "TP_BATCH_CAP": "0"})],
        "read": "throughput",
    },
    # Pack 5 (forensics-derived): stock vs stock + emission (wm-from-reasoning
    # + act-floor 3). Single change vs stock; read rule = control (levels/zero).
    "tp5": {
        "slug": "arc3-tp5-smoke",
        "phases": [("stock", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "0",
                              "TP_TRIM_LOW_WATER": "1.0", "TP_CONTEXT_WINDOW": "0",
                              "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1", "TP_BATCH_CAP": "0"}),
                   ("stock_em", {"TP_ENABLE": "1", "TP2_ENABLE": "0", "TP4_ENABLE": "0", "TP5_ENABLE": "1",
                                 "TP_TRIM_LOW_WATER": "1.0", "TP_CONTEXT_WINDOW": "0",
                                 "TP_YIELD_SECONDS": "-1", "TP_TOOL_STEPS": "-1", "TP_BATCH_CAP": "0"})],
        "read": "control",
    },
    "tp4": {
        "slug": "arc3-tp4-smoke",
        "phases": [("tp2", {"TP_ENABLE": "1", "TP2_ENABLE": "1", "TP4_ENABLE": "0", "TP5_ENABLE": "0"}),
                   ("tp24", {"TP_ENABLE": "1", "TP2_ENABLE": "1", "TP4_ENABLE": "1", "TP5_ENABLE": "0"})],
        "read": "control",
    },
}
GRAFT4_PY = SUB / "_throughput_v1" / "graft_explore.py"
EXPLORER_PY = SUB / "_throughput_v1" / "frontier_explorer.py"
GRAFT5_PY = SUB / "_throughput_v1" / "graft_emission.py"

PER_GAME_S = 7920
SOFT_END_S = 19800          # 5.5 h global backstop: boot + 2 x 2.2 h + slack

MARK_IMPORTS = "NOTEBOOK_START_EPOCH = time.time()"
MARK_SMOKE = "# Smoke/eval hook:"
MARK_RUN = "run_context = contextlib.nullcontext()"
MARK_ATTEST = "attest: OK"

RUN_BLOCK_OLD = """    try:
        await bm.run(
            soft_end_time=soft_end,
            runtime_environment=target,
            minimal_diagnostics=run_as_submission,
        )
"""

RUN_BLOCK_NEW = '''    try:
        # tp smoke: TWO phases (stock, then graft) inside ONE vLLM boot. The
        # scored-rerun path takes exactly one pass with the competition games.
        for _phase_name, _phase_games, _phase_cap, _phase_env in SMOKE_PHASES:
            if not run_as_submission:
                TP_ALL_RUNS.extend(bm.game_runs)
                bm.game_runs = []
                bm.games = [GameAPI(env_name=_n, arcade_spec=_spec) for _n in _phase_games]
                bm.n_passes = 1
                bm.game_weights = None
                bm.label = "tp-smoke-" + _phase_name
                bm.solver.max_runtime_s_per_game = float(_phase_cap)
                for _k, _v in _phase_env.items():
                    os.environ[_k] = _v
                print(f"=== PHASE {_phase_name}: {len(_phase_games)} games "
                      f"per_game_cap={_phase_cap}s env={_phase_env} "
                      f"graft_status={_tpmod.status()} ===", flush=True)
            _tp_phase_begin(_phase_name)
            try:
                await bm.run(
                    soft_end_time=soft_end,
                    runtime_environment=target,
                    minimal_diagnostics=run_as_submission,
                )
            except Exception as _phase_exc:  # noqa: BLE001
                import traceback

                TP_PHASE_ERRORS.append(f"{_phase_name}: {type(_phase_exc).__name__}: {_phase_exc}")
                print(f"PHASE {_phase_name} RAISED {type(_phase_exc).__name__}: {_phase_exc}",
                      flush=True)
                traceback.print_exc()
            try:
                _tp_phase_end(_phase_name, list(bm.game_runs))
            except Exception:  # noqa: BLE001
                pass
            if run_as_submission:
                break
'''

GPU_ASSERT_CELL = r'''# Fail-fast GPU assert: metadata machine_shape + --accelerator alone can still
# bind P100; the competition source attachment is the real RTX Pro 6000 gate.
import subprocess as _sp

_gpu = _sp.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("boot gpu:", (_gpu.stdout or "").strip() or (_gpu.stderr or "").strip())
_gpu_name = (_gpu.stdout or "").upper()
assert "RTX" in _gpu_name and "6000" in _gpu_name, (
    f"GPU misbind — expected RTX Pro 6000, got: {_gpu.stdout!r} {_gpu.stderr!r}"
)
'''


def _smoke_cell(games: list[str], phases: list[tuple[str, dict]]) -> str:
    games_repr = json.dumps(games, indent=4)
    phase_lines = "\n".join(f"    ({name!r}, GAMES_25, {PER_GAME_S}, {env!r})," for name, env in phases)
    return f'''# Smoke/eval hook: a NORMAL COMMIT runs the THROUGHPUT A/B SMOKE on the scored
# GPU class. The scored rerun path (KAGGLE_IS_COMPETITION_RERUN) never enters
# this branch.
#
# Two phases, same 25 public games, eval geometry (7,920 s per game,
# concurrency 28 from the serialized solver): "stock" with every graft seam a
# pass-through (TP_ENABLE=0), then "tp" with the Pack-1 flags on.
GAMES_25 = {games_repr}
SMOKE_PHASES = [
{phase_lines}
]
TP_PHASE_ERRORS = []
TP_ALL_RUNS = []

if not run_as_submission:
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    def _resolve_env_dir():
        candidates = [
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
            Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
        ]
        for cand in candidates:
            if cand.is_dir():
                return str(cand)
        for hit in Path("/kaggle/input").rglob("environment_files"):
            if hit.is_dir():
                return str(hit)
        raise RuntimeError("environment_files dir not found in /kaggle/input")

    _env_dir = _resolve_env_dir()
    _spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_env_dir)
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in SMOKE_PHASES[0][1]]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = "tp-smoke-" + SMOKE_PHASES[0][0]
    bm.solver.max_runtime_s_per_game = float(SMOKE_PHASES[0][2])
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds={SOFT_END_S})
    print(f"smoke hook: phases={{[(p[0], len(p[1]), p[2], p[3]) for p in SMOKE_PHASES]}} "
          f"env_dir={{_env_dir}} concurrency={{bm.solver.concurrency}} "
          f"per_game_cap={{bm.solver.max_runtime_s_per_game}}s soft_end={{soft_end}}")
else:
    print("scored rerun: smoke hook inert — full competition games")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''


GRAFT_CELL_HEAD = r'''# Graft install — ONE graft: throughput (Pack 1). Every TP_* flag is set
# EXPLICITLY so the run log states the whole config; the phase loop flips
# TP_ENABLE only. Stale grafts are purged unconditionally.
os.environ["TP_ENABLE"] = "1"
os.environ["TP_TRIM_LOW_WATER"] = "0.5"
os.environ["TP_CONTEXT_WINDOW"] = "24576"
os.environ["TP_YIELD_SECONDS"] = "900"
os.environ["TP_TOOL_STEPS"] = "8"
os.environ["TP_KEEP_NOTES_ON_GAME_OVER"] = "1"
os.environ["TP_BATCH_CAP"] = "10"
os.environ["TP2_ENABLE"] = "1"
os.environ["TP2_SUMMARY"] = "1"
os.environ["TP2_PROBE"] = "1"
os.environ["TP2_PROBE_CLICKS"] = "3"
os.environ["TP2_STALL"] = "1"
os.environ["TP2_STALL_T1"] = "10"
os.environ["TP2_STALL_T2"] = "40"
os.environ["TP2_STALL_RESETS_PER_LEVEL"] = "2"
os.environ["TP2_STREAK"] = "1"
os.environ["TP2_STREAK_N"] = "3"
os.environ["TP2_DIFF"] = "1"
os.environ["TP4_ENABLE"] = "1"
os.environ["TP4_STALL_T3"] = "30"
os.environ["TP4_BUDGET"] = "800"
os.environ["TP4_RUNS_PER_LEVEL"] = "1"
os.environ["TP4_ENDGAME_S"] = "300"
os.environ["TP5_ENABLE"] = "1"
os.environ["TP5_WM_FROM_REASONING"] = "1"
os.environ["TP5_ACT_FLOOR"] = "3"
for _stale in ("EFFORT_MEDIUM", "EFFORT_DEAD_RETRY", "YIELD_CARRYOVER", "YIELD_SLICE_CAP",
               "EXPLORER", "EXPLORER_V8"):
    os.environ.pop(_stale, None)

_TP_SOURCE = '''

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
assert _tp_status == "throughput: OK", "throughput graft must be live, got: " + repr(_tp_status)
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
print("[grafts] flags:", {k: v for k, v in os.environ.items() if k.startswith("TP")})
print("[grafts] status:", _tpmod.status(), _tcmod.status(), _temod.status())
'''

TELEMETRY_CELL = r'''# ---- tp smoke telemetry: THE READ. Per phase: per-game actions / levels /
# score / turns / tokens from the framework mirror + the ToolAgent session
# counters, and vLLM /metrics deltas (prompt + generation tokens, prefix-cache
# queries/hits, request count) — the mechanical quantities Pack 1 targets.
import json as _tel_json
import re as _tel_re
import threading as _tel_threading
import time as _tel_time
import urllib.request as _tel_rq

from inference.framework import solver as _solver_mod

_tel_lock = _tel_threading.Lock()
_TEL_PATH = WORKING_DIR / "tp_smoke_results.json"
TP_SESSIONS = {}      # phase -> {game_id: {...}} filled at session exit
TP_PHASES = []        # ordered phase records
_TP_CURRENT = {"phase": None, "t0": None, "metrics0": None}


def _metrics_url():
    base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/metrics"


_METRIC_KEYS = (
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:request_success_total", "vllm:num_preemptions_total",
)


def _scrape_metrics():
    out = {}
    try:
        with _tel_rq.urlopen(_metrics_url(), timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for key in _METRIC_KEYS:
            if line.startswith(key):
                m = _tel_re.match(r"^\S+(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
                if m:
                    try:
                        out[key] = out.get(key, 0.0) + float(m.group(1))
                    except ValueError:
                        pass
    return out


def _tp_phase_begin(phase):
    with _tel_lock:
        _TP_CURRENT["phase"] = phase
        _TP_CURRENT["t0"] = _tel_time.monotonic()
        _TP_CURRENT["metrics0"] = _scrape_metrics()
        TP_SESSIONS.setdefault(phase, {})
    print(f"[tp-tel] phase {phase} begin metrics0={_TP_CURRENT['metrics0']}", flush=True)


def _tp_phase_end(phase, game_runs):
    m1 = _scrape_metrics()
    m0 = _TP_CURRENT.get("metrics0") or {}
    delta = {k: (m1.get(k, 0.0) - m0.get(k, 0.0)) for k in _METRIC_KEYS if k in m1}
    wall = _tel_time.monotonic() - (_TP_CURRENT.get("t0") or _tel_time.monotonic())
    games = []
    for game_run in game_runs:
        apl = list(game_run.actions_per_level or [])
        actions = sum(apl) if apl else len(game_run.history)
        sess = TP_SESSIONS.get(phase, {}).get(game_run.game_id, {})
        games.append({
            "game_id": game_run.game_id,
            "state": game_run.state,
            "levels_completed": game_run.levels_completed,
            "number_of_levels": game_run.number_of_levels,
            "final_score": game_run.final_score,
            "actions": actions,
            "actions_per_level": apl,
            "base_actions_per_level": list(game_run.base_actions_per_level or []),
            "wallclock_s": game_run.final_wallclock_seconds,
            "solver_note": game_run.solver_note,
            "turns": sess.get("turns"),
            "session_total_tokens": sess.get("total_tokens"),
            "session_generated_tokens": sess.get("generated_tokens"),
            "history_messages_at_exit": sess.get("history_messages"),
            "context_budget_tokens": sess.get("context_budget_tokens"),
            "yield_seconds": sess.get("yield_seconds"),
            "tool_steps": sess.get("tool_steps"),
        })
    n = max(1, len(games))
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    hits = delta.get("vllm:prefix_cache_hits_total", 0.0)
    gen = delta.get("vllm:generation_tokens_total", 0.0)
    prompt = delta.get("vllm:prompt_tokens_total", 0.0)
    rec = {
        "phase": phase,
        "env": {k: v for k, v in os.environ.items() if k.startswith("TP_")},
        "graft_status": _tpmod.status(),
        "control_status": _tcmod.status(),
        "explore_status": _temod.status(),
        "emission_status": _tmmod.status(),
        "wall_s": round(wall, 1),
        "games": games,
        "n_games": len(games),
        "mean_actions": round(sum(g["actions"] for g in games) / n, 2),
        "mean_levels": round(sum(g["levels_completed"] for g in games) / n, 3),
        "mean_score": round(sum(float(g["final_score"] or 0.0) for g in games) / n, 4),
        "zero_level_games": sum(1 for g in games if g["levels_completed"] == 0),
        "mean_turns": round(sum(float(g["turns"] or 0) for g in games) / n, 1),
        "metrics_delta": delta,
        "prefix_hit_rate": round(hits / queries, 4) if queries else None,
        "prefill_per_gen_token": round(prompt / gen, 2) if gen else None,
        "gen_tok_s": round(gen / wall, 1) if wall else None,
    }
    with _tel_lock:
        TP_PHASES.append(rec)
    try:
        _TEL_PATH.write_text(_tel_json.dumps(
            {"phases": TP_PHASES, "phase_errors": TP_PHASE_ERRORS}, indent=1, default=str),
            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(f"[tp-tel] phase {phase} end: games={rec['n_games']} mean_actions={rec['mean_actions']} "
          f"mean_levels={rec['mean_levels']} mean_score={rec['mean_score']} "
          f"zero_level={rec['zero_level_games']} turns={rec['mean_turns']} "
          f"prefix_hit={rec['prefix_hit_rate']} prefill/gen={rec['prefill_per_gen_token']} "
          f"gen_tok_s={rec['gen_tok_s']} wall={rec['wall_s']}s", flush=True)


# Session exit seam: record turns + token counters per game for the phase.
_inner_play = _solver_mod._HarnessGameSession.play


def _tel_play(self):
    try:
        return _inner_play(self)
    finally:
        try:
            gid = getattr(getattr(self.game, "game_run", None), "game_id", "?")
            an = self.analyzer
            rec = {
                "turns": int(getattr(self, "analysis_step", 0) or 0),
                "total_tokens": int(getattr(an, "_session_total_tokens", 0) or 0),
                "generated_tokens": int(getattr(an, "_session_generated_tokens", 0) or 0),
                "history_messages": len(getattr(an, "_history_messages", []) or []),
                "context_budget_tokens": getattr(an, "_context_budget_tokens", None),
                "yield_seconds": getattr(an, "_yield_seconds", None),
                "tool_steps": getattr(an, "_tool_steps", None),
            }
            with _tel_lock:
                TP_SESSIONS.setdefault(_TP_CURRENT.get("phase") or "?", {})[gid] = rec
        except Exception:  # noqa: BLE001
            pass


if not getattr(_solver_mod._HarnessGameSession.play, "_tp_tel", False):
    _tel_play._tp_tel = True
    _solver_mod._HarnessGameSession.play = _tel_play

print("[tp-tel] installed; metrics url =", _metrics_url(), flush=True)
'''

REPORT_CELL = r'''# ---- tp smoke final report (grep for TP SMOKE / PHASE / READ) ----
print("=" * 78)
print("TP SMOKE RESULTS (arm", ARM_NAME, "read rule", READ_RULE, ")")
print("install verdict:", _tp_status)
by = {p["phase"]: p for p in TP_PHASES}
for name in [p[0] for p in SMOKE_PHASES]:
    p = by.get(name)
    if not p:
        print(f"PHASE {name}: MISSING")
        continue
    print(f"PHASE {name}: games={p['n_games']} mean_actions={p['mean_actions']} "
          f"mean_levels={p['mean_levels']} mean_score={p['mean_score']} "
          f"zero_level={p['zero_level_games']} mean_turns={p['mean_turns']} "
          f"prefix_hit={p['prefix_hit_rate']} prefill/gen={p['prefill_per_gen_token']} "
          f"gen_tok_s={p['gen_tok_s']} wall={p['wall_s']}s")
    for g in sorted(p["games"], key=lambda g: g["game_id"]):
        print(f"  {g['game_id']}: levels={g['levels_completed']}/{g['number_of_levels']} "
              f"score={g['final_score']} actions={g['actions']} turns={g['turns']} "
              f"gen_tok={g['session_generated_tokens']} state={g['state']}")

verdict = "UNREADABLE"
detail = ""
_names = [p[0] for p in SMOKE_PHASES]
if len(_names) == 2 and all(n in by and by[n]["n_games"] for n in _names):
    s, t = by[_names[0]], by[_names[1]]
    ratio = (t["mean_actions"] / s["mean_actions"]) if s["mean_actions"] else float("inf")
    dlev = t["mean_levels"] - s["mean_levels"]
    dzero = t["zero_level_games"] - s["zero_level_games"]
    hit = t["prefix_hit_rate"] or 0.0
    detail = (f"actions x{ratio:.2f} levels {dlev:+.3f} zero_level {dzero:+d} "
              f"prefix_hit {hit:.2f} score {t['mean_score'] - s['mean_score']:+.3f}")
    if READ_RULE == "throughput":
        if ratio >= 2.0 and dlev >= -0.15 and hit >= 0.5:
            verdict = "PASS"
        elif ratio >= 1.5 and dlev >= -0.15:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAIL"
    else:  # control: fewer zero-level games without losing depth
        if dzero <= -3 and dlev >= -0.1:
            verdict = "PASS"
        elif dlev < -0.15 or dzero > 2:
            verdict = "FAIL"
        else:
            verdict = "INCONCLUSIVE"
print(f"TP SMOKE READ: {verdict} ({detail}) phase_errors={TP_PHASE_ERRORS}")
results = {
    "arm": ARM_NAME,
    "install_verdict": _tp_status,
    "phases": TP_PHASES,
    "phase_errors": TP_PHASE_ERRORS,
    "verdict": verdict,
    "detail": detail,
}
(WORKING_DIR / "tp_smoke_results.json").write_text(
    json.dumps(results, indent=1, default=str), encoding="utf-8")
print("wrote", WORKING_DIR / "tp_smoke_results.json")
'''


def public_games() -> list[str]:
    env_root = ROOT / "environment_files"
    names = []
    for game_dir in sorted(p for p in env_root.iterdir() if p.is_dir()):
        hashes = sorted(p.name for p in game_dir.iterdir() if p.is_dir())
        assert len(hashes) == 1, (game_dir, hashes)
        names.append(f"{game_dir.name}-{hashes[0]}")
    assert len(names) == 25, len(names)
    return names


def main(arm: str = "tp") -> None:
    spec = ARMS[arm]
    slug = spec["slug"]
    nb = json.loads(BASE_NB.read_text())
    graft_src = GRAFT_PY.read_text()
    graft2_src = GRAFT2_PY.read_text()
    graft4_src = GRAFT4_PY.read_text()
    fe_src = EXPLORER_PY.read_text()
    graft5_src = GRAFT5_PY.read_text()
    assert "def install() -> str:" in graft5_src
    assert "def install() -> str:" in graft_src and "def install() -> str:" in graft2_src
    assert "def install() -> str:" in graft4_src and "class FrontierExplorer" in fe_src
    assert "def time_guard_per_game_s" in graft_src
    assert "def run_probe" in graft2_src
    games = public_games()

    def idx_of(marker: str) -> int:
        hits = [i for i, c in enumerate(nb["cells"])
                if c["cell_type"] == "code" and marker in "".join(c["source"])]
        assert len(hits) == 1, (marker, len(hits))
        return hits[0]

    def code_cell(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": text.splitlines(keepends=True)}

    nb["cells"].insert(idx_of(MARK_IMPORTS) + 1, code_cell(GPU_ASSERT_CELL))
    nb["cells"][idx_of(MARK_SMOKE)]["source"] = _smoke_cell(games, spec["phases"]).splitlines(keepends=True)

    run_idx = idx_of(MARK_RUN)
    run_src = "".join(nb["cells"][run_idx]["source"])
    assert run_src.count(RUN_BLOCK_OLD) == 1, "base run cell drifted"
    run_src = run_src.replace(RUN_BLOCK_OLD, RUN_BLOCK_NEW)
    nb["cells"][run_idx]["source"] = run_src.splitlines(keepends=True)

    graft_cell_text = (GRAFT_CELL_HEAD + repr(graft_src) + "\n_TP2_SOURCE = " + repr(graft2_src)
                       + "\n_FE_SOURCE = " + repr(fe_src) + "\n_TP4_SOURCE = " + repr(graft4_src)
                       + "\n_TP5_SOURCE = " + repr(graft5_src)
                       + "\nARM_NAME = " + repr(arm) + "\nREAD_RULE = " + repr(spec["read"]) + "\n"
                       + GRAFT_CELL_TAIL)
    for src in (graft_src, graft2_src, graft4_src, fe_src, graft5_src):
        assert repr(src) in graft_cell_text
    nb["cells"].insert(run_idx, code_cell(graft_cell_text))
    nb["cells"].insert(run_idx + 1, code_cell(TELEMETRY_CELL))
    nb["cells"].insert(idx_of(MARK_RUN) + 1, code_cell(REPORT_CELL))

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined
    assert joined.index("GPU misbind") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index("SMOKE_PHASES = [")
    assert joined.index('os.environ["TP_ENABLE"] = "1"') < joined.index(MARK_RUN)
    assert joined.count("SMOKE_PHASES = [") == 1 and all(n in joined for n, _ in spec["phases"])
    assert joined.index("[tp-tel] installed") < joined.index(MARK_RUN)
    assert joined.index(MARK_RUN) < joined.index("TP SMOKE RESULTS")
    assert 'EFFORT_MEDIUM"] = "1"' not in joined
    for name in games:
        assert joined.count(name) == 1, name

    out_dir = HERE if arm == "tp" else HERE / arm
    out_dir.mkdir(parents=True, exist_ok=True)
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
    print("built", slug, "arm", arm, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest()[:16])
    print("cells:", len(nb["cells"]), "games:", len(games), "notebook bytes:",
          (out_dir / f"{slug}.ipynb").stat().st_size)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "tp")
