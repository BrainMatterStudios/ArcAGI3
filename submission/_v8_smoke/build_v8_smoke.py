#!/usr/bin/env python3
"""build_v8_smoke.py — the EXPLORER V8 GPU SMOKE: the gate that makes v8
flight-eligible for a scored slot.

ARM (2026-08-26): duck38-v12 base (Aug-07 anim bundle + Qwen3.8-27B-FP8
repacked, boot-attested) + ONE graft, stated explicitly:

  submission/_explorer_v8/graft_explorer_v8.py with EXPLORER_V8=1 — the live
  SearchCore graft (v7 substrate: trigger, self-harm gate, semaphore and
  envelope guards VERBATIM; v8 replaces only the grind ALGORITHM with the
  measured SearchCore portfolio + specialist tier + banking, and enforces
  every guard on EVERY engine call).

  effort_medium is DELIBERATELY NOT INSTALLED. It was REVERTED as harmful
  live (pooled 1.12 vs 1.55, commit 6a8e11f); this smoke keeps it off.

Offline provenance being tested live (docs/ENVELOPE-2026-08-26-v8.md §4):
  ft09  6 levels FULL CRACK via spec:ft09_gf2, 1233 engine actions (9.5 s at
        the live 130 act/s), banked in a 75-action replay
  tu93  9 levels FULL CRACK via nbfs_macros, 84687 actions (651 s), banked
        in 187 actions
48 offline tests pass (submission/_explorer_v8/test_graft_explorer_v8.py).

GEOMETRY — TWO PHASES inside ONE vLLM boot (one bm.run per phase; teardown
runs once, after both):
  PHASE A  ft09-0d8bbf25 alone, 3600 s box.  ft09 is THE read (the specialist
           crack target).  Running it alone guarantees it wins the run-wide
           grind semaphore (BoundedSemaphore(1)) — in a shared phase a
           competing game's 600-1500 s engagement can hold the gate and
           starve the primary read, and the 2700 s cumulative budget could be
           spent before ft09 ever engages.
  PHASE B  dc22-fdcac232 + vc33-5430563c + sk48-d8078629, 3600 s box — the
           regression read vs the v12 comparators AND the contended-gate /
           cumulative-budget envelope read (3 games sharing one semaphore).
  Global soft end 12600 s (3.5 h) => whole kernel <= 4 h.

DIAGNOSTICS (the read; printed AND persisted to v8_smoke_results.json /
v8_smoke_telemetry.json):
  (a) specialists.detect — did it fire, on which game, with which class
  (b) FULL CRACK — did any game reach engine WIN under the grinder
  (c) banking — did bank_crack fire, replay action count, verdict, and the
      score the banked play projects (per-level minimal actions vs the
      engine's own base_actions_per_level)
  (d) EVERY envelope guard's OBSERVED maximum vs its cap: cumulative grind
      wall, per-engagement wall (generic 600 s / owned 1500 s), per-engagement
      engine actions, absolute run ceiling, bank-replay cap
  (e) per-game levels + score vs the v12 comparators

PRE-REGISTERED PASS BARS:
  1. specialist fires on ft09
  2. ft09 completes >= 3 levels (offline: 6)
  3. no envelope guard exceeded
  4. no crash
  5. no regression > 1 level on vc33 / dc22 vs their v12 comparators

Usage:
  .venv/bin/python submission/_v8_smoke/build_v8_smoke.py
  cd submission/_v8_smoke && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
BASE_NB = SUB / "_duck38_v12" / "arc3-duck38-v12.ipynb"
KERNEL_SLUG = "arc3-v8-smoke"

# The five files the v8 bundle needs flat in one directory (envelope doc §9).
BUNDLE_FILES = {
    "graft_explorer_v8.py": SUB / "_explorer_v8" / "graft_explorer_v8.py",
    "graft_explorer.py": SUB / "_explorer_floor" / "graft_explorer.py",
    "graft_bank.py": SUB / "_duck38_v12_bank" / "graft_bank.py",
    "search_core.py": SUB / "_search_core" / "search_core.py",
    "specialists.py": SUB / "_search_core" / "specialists.py",
}

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
        # v8 smoke: TWO phases inside ONE vLLM boot. The base cell's single
        # bm.run is replaced by a loop; teardown still runs once, in the
        # outer finally, so phase B still has a live server. The scored-rerun
        # path takes exactly one pass with the competition games (the phase
        # list collapses to one entry when run_as_submission).
        for _phase_name, _phase_games, _phase_cap in SMOKE_PHASES:
            if not run_as_submission:
                bm.games = [GameAPI(env_name=_n, arcade_spec=_spec) for _n in _phase_games]
                bm.n_passes = 1
                bm.game_weights = None
                bm.label = "v8-smoke-" + _phase_name
                bm.solver.max_runtime_s_per_game = float(_phase_cap)
                print(f"=== PHASE {_phase_name}: games={_phase_games} "
                      f"per_game_cap={_phase_cap}s ===", flush=True)
            try:
                await bm.run(
                    soft_end_time=soft_end,
                    runtime_environment=target,
                    minimal_diagnostics=run_as_submission,
                )
            except Exception as _phase_exc:  # noqa: BLE001
                import traceback

                V8_PHASE_ERRORS.append(f"{_phase_name}: {type(_phase_exc).__name__}: {_phase_exc}")
                print(f"PHASE {_phase_name} RAISED {type(_phase_exc).__name__}: {_phase_exc}",
                      flush=True)
                traceback.print_exc()
            try:
                _v8_snapshot()
            except Exception:  # noqa: BLE001
                pass
            if run_as_submission:
                break
'''

GPU_ASSERT_CELL = r'''# Fail-fast GPU assert: metadata machine_shape + --accelerator alone can still
# bind P100 (3 wasted pushes on serving-lab proved it; the competition source
# attachment is the real RTX Pro 6000 gate). Die here, before any setup cost.
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

SMOKE_CELL = r'''# Smoke/eval hook: a NORMAL COMMIT runs the EXPLORER V8 SMOKE on the scored
# GPU class. The scored rerun path (KAGGLE_IS_COMPETITION_RERUN) never enters
# this branch.
#
# TWO PHASES, one vLLM boot (the run cell loops over SMOKE_PHASES):
#   A: ft09 ALONE — the specialist crack target and THE read of this smoke.
#      Alone because the grind semaphore is run-wide BoundedSemaphore(1): a
#      competing game's 600-1500 s engagement could hold the gate (or spend
#      the 2700 s cumulative budget) before ft09 ever engages.
#   B: dc22 + vc33 + sk48 — the regression read vs the v12 comparators and
#      the contended-gate / cumulative-budget envelope read.
SMOKE_PHASES = [
    ("A-ft09", ["ft09-0d8bbf25"], 3600),
    ("B-panel", ["dc22-fdcac232", "vc33-5430563c", "sk48-d8078629"], 3600),
]
V8_PHASE_ERRORS = []

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
    # Phase A's games are set here so the run cell has a valid benchmark even
    # if the phase loop is ever bypassed; the loop re-sets them per phase.
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in SMOKE_PHASES[0][1]]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = "v8-smoke-" + SMOKE_PHASES[0][0]
    bm.solver.max_runtime_s_per_game = float(SMOKE_PHASES[0][2])
    # Global backstop 3.5 h from notebook start => whole kernel <= 4 h.
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=12600)
    print(f"smoke hook: phases={[(p[0], p[1]) for p in SMOKE_PHASES]} "
          f"env_dir={_env_dir} per_game_cap={bm.solver.max_runtime_s_per_game}s "
          f"soft_end={soft_end}")
else:
    print("scored rerun: smoke hook inert — full competition games")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''

GRAFT_CELL_HEAD = r'''# Graft install — ONE graft: explorer v8 (EXPLORER_V8=1, default-off flag
# explicitly enabled). effort_medium is DELIBERATELY ABSENT: it was REVERTED
# as harmful live (pooled 1.12 vs 1.55, commit 6a8e11f).
#
# The five-file flat bundle is written to WORKING_DIR/v8_bundle and resolved
# through EXPLORER_V8_CORE_DIR — the exact layout the envelope doc §9 says
# was verified end-to-end offline.
os.environ["EXPLORER"] = "1"
os.environ["EXPLORER_V8"] = "1"
for _stale in ("EFFORT_MEDIUM", "EFFORT_DEAD_RETRY", "YIELD_CARRYOVER", "YIELD_SLICE_CAP"):
    os.environ.pop(_stale, None)

'''

GRAFT_CELL_TAIL = r'''
_V8_DIR = WORKING_DIR / "v8_bundle"
_V8_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in _V8_BUNDLE.items():
    (_V8_DIR / _name).write_text(_src, encoding="utf-8")
os.environ["EXPLORER_V8_CORE_DIR"] = str(_V8_DIR)
if str(_V8_DIR) not in sys.path:
    sys.path.insert(0, str(_V8_DIR))

import importlib as _importlib

_v8mod = _importlib.import_module("graft_explorer_v8")
_v8_status = _v8mod.install()
print("[explorer-v8]", _v8_status)
# A smoke that silently measures v7 (or stock) is worse than one that dies:
# hard-gate BOTH halves of the install verdict.
assert "explorer: OK" in _v8_status, "v7 substrate must be live, got: " + repr(_v8_status)
assert "v8: OK" in _v8_status, "v8 grind must be live, got: " + repr(_v8_status)
print("[explorer-v8] flags:", {k: v for k, v in os.environ.items() if k.startswith("EXPLORER")})
'''

TELEMETRY_CELL = r'''# ---- v8 smoke telemetry: THE READ (installed OUTSIDE the graft, wrapping the
# already-grafted seams). Five questions, each answered by observation:
#   (a) did specialists.detect fire, on which game, with which class
#   (b) did any game reach a FULL CRACK (engine WIN under the grinder)
#   (c) did bank_crack fire, in how many replay actions, with what verdict,
#       and what score does the banked play project
#   (d) every envelope guard's OBSERVED maximum vs its cap
#   (e) per-game levels/score (the report cell, from bm.game_runs)
import json as _tel_json
import threading as _tel_threading
import time as _tel_time

import graft_bank as _bankm  # noqa: F401 — presence proves the kill switch is importable
import graft_explorer as _v7m
import graft_explorer_v8 as _v8m
import search_core as _scm
import specialists as _specm

_tel_lock = _tel_threading.Lock()
_TEL_TLS = _tel_threading.local()
_TEL_PATH = WORKING_DIR / "v8_smoke_telemetry.json"


def _cap(name, default):
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


CAPS = {
    "engagement_wall_generic_s": _cap("EXPLORER_GRIND_TIME_S", 600),
    "engagement_wall_owned_s": _cap("EXPLORER_OWNED_TIME_S", 1500),
    "engagement_actions": min(_cap("EXPLORER_GRIND_BUDGET", 500000),
                              int(_cap("EXPLORER_OWNED_TIME_S", 1500)
                                  * _v8m.GATEWAY_ACT_PER_S * 1.5)),
    "run_grind_budget_s": _cap("EXPLORER_RUN_GRIND_BUDGET_S", 2700),
    "run_start_cutoff_s": _cap("EXPLORER_RUN_CUTOFF_S", 18000),
    "run_hard_cutoff_s": int(_v8m._hard_cutoff_s()),
    "bank_max_actions": _cap("EXPLORER_V8_BANK_MAX_ACTIONS", 2000),
    "grinds_per_level": _cap("EXPLORER_GRIND_MAX_PER_LEVEL", 1),
}

OBS = {
    "max_engagement_wall_generic_s": 0.0,
    "max_engagement_wall_owned_s": 0.0,
    "max_engagement_actions": 0,
    "max_cumulative_grind_s": 0.0,
    "max_run_elapsed_at_grind_s": 0.0,
    "max_bank_replay_actions": 0,
    "abort_reasons": {},
    "engagements": 0,
    "engine_actions_total": 0,
}
GAMES = {}
EVENTS = []


def _game_rec(game_id):
    rec = GAMES.get(game_id)
    if rec is None:
        rec = {
            "game_id": game_id,
            "engagements": 0,
            "specialist": None,
            "detect_calls": 0,
            "detect_actions": 0,
            "specialist_solve_calls": 0,
            "specialist_solves": 0,
            "lane_calls": {},
            "lane_solves": {},
            "levels_unlocked_by_grinder": 0,
            "grind_unlocked_levels": [],
            "full_crack": False,
            "bank_fired": False,
            "bank_ok": False,
            "bank_reason": None,
            "bank_plan_actions": None,
            "bank_plan_per_level": None,
            "bank_projected_score": None,
            "stop_reasons": [],
            "engagement_walls_s": [],
            "engagement_actions": [],
            "grinder_actions": 0,
        }
        GAMES[game_id] = rec
    return rec


def _snapshot_dict():
    with _tel_lock:
        return {
            "caps": CAPS,
            "observed": _tel_json.loads(_tel_json.dumps(OBS)),
            "games": _tel_json.loads(_tel_json.dumps(GAMES)),
            "events": list(EVENTS[-200:]),
        }


def _v8_snapshot():
    try:
        _TEL_PATH.write_text(_tel_json.dumps(_snapshot_dict(), indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001 — telemetry must never break the run
        pass


def _event(text):
    stamp = round(_tel_time.monotonic() - _TEL_T0, 1)
    with _tel_lock:
        EVENTS.append(f"[{stamp}s] {text}")
    print("[v8-tel]", text, flush=True)


_TEL_T0 = _tel_time.monotonic()

# --- (d) envelope observation: GuardedEnv.check runs on EVERY engine call ---
_inner_check = _v8m.GuardedEnv.check


def _observe(genv):
    try:
        wall = genv.wall()
        owned = bool(genv._xs.get("grind_unlocked_levels"))
        with _v7m._RUN_LOCK:
            spent = _v7m._GRIND_WALL_SPENT[0]
            run_t0 = _v7m._RUN_T0
        run_elapsed = (_tel_time.monotonic() - run_t0) if run_t0 is not None else 0.0
        with _tel_lock:
            key = "max_engagement_wall_owned_s" if owned else "max_engagement_wall_generic_s"
            OBS[key] = max(OBS[key], round(wall, 1))
            OBS["max_engagement_actions"] = max(OBS["max_engagement_actions"], genv.actions)
            OBS["max_cumulative_grind_s"] = max(OBS["max_cumulative_grind_s"],
                                                round(spent + wall, 1))
            OBS["max_run_elapsed_at_grind_s"] = max(OBS["max_run_elapsed_at_grind_s"],
                                                    round(run_elapsed, 1))
    except Exception:  # noqa: BLE001
        pass


def _tel_check(self):
    try:
        if self.actions % 16 == 0:
            _observe(self)
    except Exception:  # noqa: BLE001
        pass
    try:
        _inner_check(self)
    except _v8m._GrindAbort as _ab:
        with _tel_lock:
            OBS["abort_reasons"][_ab.reason] = OBS["abort_reasons"].get(_ab.reason, 0) + 1
        raise


_v8m.GuardedEnv.check = _tel_check

# --- (a) specialist detection + solve ---
_inner_detect = _specm.detect
_inner_spec_solve = _specm.solve_level


def _tel_detect(core):
    game_id = getattr(_TEL_TLS, "game_id", "?")
    rec = _game_rec(game_id)
    with _tel_lock:
        rec["detect_calls"] += 1
    a0 = getattr(core, "env", None)
    before = getattr(a0, "actions", None)
    out = _inner_detect(core)
    after = getattr(a0, "actions", None)
    with _tel_lock:
        rec["specialist"] = out
        if before is not None and after is not None:
            rec["detect_actions"] = int(after) - int(before)
    _event(f"{game_id}: specialists.detect -> {out!r} "
           f"({rec['detect_actions']} probe actions)")
    return out


def _tel_spec_solve(core, name, target, budget_s):
    game_id = getattr(_TEL_TLS, "game_id", "?")
    rec = _game_rec(game_id)
    with _tel_lock:
        rec["specialist_solve_calls"] += 1
    res = _inner_spec_solve(core, name, target, budget_s)
    solved = bool(res.get("solved"))
    with _tel_lock:
        if solved:
            rec["specialist_solves"] += 1
    _event(f"{game_id}: spec:{name} level {target} solved={solved} "
           f"reason={res.get('reason')!r}")
    return res


_specm.detect = _tel_detect
_specm.solve_level = _tel_spec_solve

# --- generic lane accounting ---
_inner_solve_with = _scm.solve_with


def _tel_solve_with(core, algo, target, budget_s, *args, **kwargs):
    game_id = getattr(_TEL_TLS, "game_id", "?")
    rec = _game_rec(game_id)
    with _tel_lock:
        rec["lane_calls"][algo] = rec["lane_calls"].get(algo, 0) + 1
    res = _inner_solve_with(core, algo, target, budget_s, *args, **kwargs)
    if res.get("solved"):
        with _tel_lock:
            rec["lane_solves"][algo] = rec["lane_solves"].get(algo, 0) + 1
    return res


_scm.solve_with = _tel_solve_with

# --- (c) banking ---
_inner_bank = _v8m.bank_crack


def _projected_score(per_level, baselines, n_levels):
    """The score the banked play projects: the harness's own formula
    (taaf.game.GameRun._compute_final_score) applied to the minimal replay's
    per-level action counts against the engine's published baselines."""
    if not baselines or not n_levels:
        return None
    total, weights, max_weights = 0.0, 0, 0
    for idx in range(int(n_levels)):
        weight = idx + 1
        weights += weight
        acts = per_level.get(idx + 1, 0)
        base = baselines[idx] if idx < len(baselines) else None
        score = min(115.0, (float(base) / acts) ** 2 * 100) if (acts and base) else 0.0
        if score > 0:
            max_weights += weight
        total += score * weight
    if not weights:
        return None
    return round(min(total / weights, max_weights / weights * 100), 4)


def _tel_bank(env, level_seqs, **kwargs):
    game_id = getattr(_TEL_TLS, "game_id", "?")
    rec = _game_rec(game_id)
    per_level = {int(k): len(v) for k, v in (level_seqs or {}).items()}
    plan_len = sum(per_level.values())
    t0 = _tel_time.monotonic()
    ok, why = _inner_bank(env, level_seqs, **kwargs)
    game = getattr(getattr(_TEL_TLS, "session", None), "game", None)
    baselines = getattr(game, "base_actions_per_level", None)
    n_levels = getattr(game, "number_of_levels", None)
    with _tel_lock:
        rec["bank_fired"] = True
        rec["bank_ok"] = bool(ok)
        rec["bank_reason"] = why
        rec["bank_plan_actions"] = plan_len
        rec["bank_plan_per_level"] = per_level
        rec["bank_wall_s"] = round(_tel_time.monotonic() - t0, 2)
        rec["bank_baselines"] = list(baselines) if baselines else None
        rec["bank_projected_score"] = _projected_score(per_level, baselines, n_levels)
        OBS["max_bank_replay_actions"] = max(OBS["max_bank_replay_actions"], plan_len)
    _event(f"{game_id}: bank_crack ok={ok} plan={plan_len} actions "
           f"per_level={per_level} reason={why!r} "
           f"projected_score={rec['bank_projected_score']}")
    return ok, why


_v8m.bank_crack = _tel_bank

# --- engagement accounting: wrap the INSTALLED v8 grind ---
_inner_grind = _v7m._grind
assert getattr(_inner_grind, "_v8", False), "the v8 grind must be installed first"


def _tel_grind(session, xs, level):
    game_id = getattr(getattr(getattr(session, "game", None), "game_run", None),
                      "game_id", "?")
    _TEL_TLS.game_id = game_id
    _TEL_TLS.session = session
    rec = _game_rec(game_id)
    with _tel_lock:
        rec["engagements"] += 1
        OBS["engagements"] += 1
    _event(f"{game_id}: ENGAGE level {level}")
    t0 = _tel_time.monotonic()
    try:
        return _inner_grind(session, xs, level)
    finally:
        wall = _tel_time.monotonic() - t0
        try:
            diag = xs.get("diag", {})
            with _v7m._RUN_LOCK:
                spent = _v7m._GRIND_WALL_SPENT[0]
            owned = bool(xs.get("grind_unlocked_levels"))
            with _tel_lock:
                rec["engagement_walls_s"].append(round(wall, 1))
                rec["levels_unlocked_by_grinder"] = int(diag.get("levels_unlocked_by_grinder", 0))
                rec["grind_unlocked_levels"] = sorted(int(x) for x in xs.get("grind_unlocked_levels", ()))
                rec["full_crack"] = bool(diag.get("games_won_by_grinder", 0))
                rec["grinder_actions"] = int(diag.get("grinder_actions", 0))
                rec["specialist"] = diag.get("v8_specialist", rec["specialist"])
                rec["detect_actions"] = int(diag.get("v8_detect_actions", rec["detect_actions"] or 0))
                rec["first_unlock_actions"] = diag.get("v8_first_unlock_actions")
                rec["v8_engagements"] = int(diag.get("v8_engagements", 0))
                rec["v8_bank_result"] = diag.get("v8_bank_result")
                rec["games_banked_by_grinder"] = int(diag.get("games_banked_by_grinder", 0))
                key = "max_engagement_wall_owned_s" if owned else "max_engagement_wall_generic_s"
                OBS[key] = max(OBS[key], round(wall, 1))
                OBS["max_cumulative_grind_s"] = max(OBS["max_cumulative_grind_s"],
                                                    round(spent + wall, 1))
                OBS["engine_actions_total"] = sum(
                    int(g.get("grinder_actions", 0)) for g in GAMES.values())
            _event(f"{game_id}: DISENGAGE after {wall:.0f}s — "
                   f"unlocked={rec['grind_unlocked_levels']} "
                   f"crack={rec['full_crack']} banked={rec.get('games_banked_by_grinder')} "
                   f"cumulative_grind={spent + wall:.0f}s/{CAPS['run_grind_budget_s']}s")
        except Exception:  # noqa: BLE001
            pass
        _v8_snapshot()


_tel_grind._v8 = True  # keep the v8 idempotence marker intact
_v7m._grind = _tel_grind

print("[v8-tel] counters installed; caps =", CAPS, flush=True)
'''

REPORT_CELL = r'''# ---- v8 smoke final report (grep for V8 SMOKE / GAME / BAR / ENVELOPE) ----
print("=" * 78)
print("V8 SMOKE RESULTS (EXPLORER_V8=1, effort_medium OFF)")
print("install verdict:", _v8_status)

# (e) per-game levels + score vs the v12 comparators. NOTE the harness's own
# levels/score count ONLY LLM-executed actions: the grinder steps the raw
# engine, so a grinder crack does NOT show up in game_run unless the LLM
# subsequently observes it. The grinder read is the telemetry block below.
COMPARATORS = {
    "ft09": {"levels": 1, "score": None,
             "note": "v12 lanes: 0-1 levels, no specialist tier"},
    "vc33": {"levels": 2, "score": 10.71,
             "note": "v12 comparator (task); effort-smoke 08-23 L2/4.73"},
    "dc22": {"levels": 1, "score": None,
             "note": "same-rig 8v8 A/B: v12 2/8 nonzero => L0-L1"},
    "sk48": {"levels": 0, "score": 0.0,
             "note": "0 levels in the effort + carryover smokes"},
}

games_out = []
for game_run in bm.game_runs:
    actions = sum(game_run.actions_per_level) if game_run.actions_per_level else len(game_run.history)
    stem = str(game_run.game_id).split("-")[0]
    row = {
        "game_id": game_run.game_id,
        "stem": stem,
        "state": game_run.state,
        "levels_completed": game_run.levels_completed,
        "number_of_levels": game_run.number_of_levels,
        "final_score": game_run.final_score,
        "llm_actions": actions,
        "wallclock_s": game_run.final_wallclock_seconds,
        "base_actions_per_level": list(game_run.base_actions_per_level or []),
        "comparator": COMPARATORS.get(stem),
    }
    games_out.append(row)
    comp = COMPARATORS.get(stem) or {}
    print(f"GAME {row['game_id']}: state={row['state']} "
          f"levels={row['levels_completed']}/{row['number_of_levels']} "
          f"score={row['final_score']} llm_actions={row['llm_actions']} "
          f"wall={row['wallclock_s']}s | v12 comparator levels={comp.get('levels')} "
          f"score={comp.get('score')}")

snap = _snapshot_dict()
tel_games = snap["games"]
obs = snap["observed"]

print("-" * 78)
print("GRINDER TELEMETRY (per game)")
for gid, rec in sorted(tel_games.items()):
    print(f"  {gid}: engagements={rec['engagements']} "
          f"specialist={rec['specialist']!r} detect_actions={rec['detect_actions']} "
          f"lanes={rec['lane_calls']} solves={rec['lane_solves']} "
          f"spec_solves={rec['specialist_solves']}/{rec['specialist_solve_calls']}")
    print(f"    unlocked={rec['grind_unlocked_levels']} "
          f"levels_unlocked={rec['levels_unlocked_by_grinder']} "
          f"FULL_CRACK={rec['full_crack']} engine_actions={rec['grinder_actions']} "
          f"walls={rec['engagement_walls_s']}")
    print(f"    BANK fired={rec['bank_fired']} ok={rec['bank_ok']} "
          f"plan={rec['bank_plan_actions']} per_level={rec['bank_plan_per_level']} "
          f"projected_score={rec['bank_projected_score']} reason={rec['bank_reason']!r}")

print("-" * 78)
print("ENVELOPE OBSERVATION (observed maximum vs cap — validated by observation)")
ENVELOPE_ROWS = [
    ("per-engagement wall, generic", obs["max_engagement_wall_generic_s"],
     CAPS["engagement_wall_generic_s"], "s"),
    ("per-engagement wall, grind-owned", obs["max_engagement_wall_owned_s"],
     CAPS["engagement_wall_owned_s"], "s"),
    ("per-engagement engine actions", obs["max_engagement_actions"],
     CAPS["engagement_actions"], "acts"),
    ("cumulative grind wall, whole run", obs["max_cumulative_grind_s"],
     CAPS["run_grind_budget_s"], "s"),
    ("absolute grind ceiling (run clock)", obs["max_run_elapsed_at_grind_s"],
     CAPS["run_hard_cutoff_s"], "s"),
    ("bank replay actions", obs["max_bank_replay_actions"],
     CAPS["bank_max_actions"], "acts"),
]
envelope_rows = []
envelope_ok = True
for name, seen, cap, unit in ENVELOPE_ROWS:
    ok = float(seen) <= float(cap)
    envelope_ok = envelope_ok and ok
    envelope_rows.append({"guard": name, "observed": seen, "cap": cap,
                          "unit": unit, "within": ok})
    print(f"  {'OK ' if ok else 'OVER'} {name}: observed {seen} {unit} / cap {cap} {unit}")
print(f"  abort reasons observed: {obs['abort_reasons']}")
print(f"  engagements={obs['engagements']} total grinder engine actions={obs['engine_actions_total']}")

# ---- pre-registered bars ----
ft09_rec = next((r for g, r in tel_games.items() if g.startswith("ft09")), None)
ft09_row = next((r for r in games_out if r["stem"] == "ft09"), None)
ft09_grind_levels = int(ft09_rec["levels_unlocked_by_grinder"]) if ft09_rec else 0
ft09_levels_any = max(ft09_grind_levels,
                      int(ft09_row["levels_completed"]) if ft09_row else 0)
bar1 = bool(ft09_rec and ft09_rec["specialist"])
bar2 = ft09_levels_any >= 3
bar3 = envelope_ok
crashed = [r["game_id"] for r in games_out if r["state"] in ("crashed",)]
bar4 = not crashed and not V8_PHASE_ERRORS
regressions = []
for stem in ("vc33", "dc22"):
    row = next((r for r in games_out if r["stem"] == stem), None)
    comp = COMPARATORS[stem]["levels"]
    if row is None:
        regressions.append(f"{stem}: NOT RUN")
    elif int(row["levels_completed"]) < comp - 1:
        regressions.append(f"{stem}: {row['levels_completed']} < {comp}-1")
bar5 = not regressions

BARS = [
    ("1. specialist fires on ft09", bar1,
     f"detected={ft09_rec['specialist']!r}" if ft09_rec else "no ft09 engagement"),
    ("2. ft09 completes >= 3 levels", bar2,
     f"grinder_levels={ft09_grind_levels} harness_levels="
     f"{ft09_row['levels_completed'] if ft09_row else 'n/a'}"),
    ("3. no envelope guard exceeded", bar3,
     "all observed maxima <= caps" if bar3 else "see ENVELOPE table"),
    ("4. no crash", bar4, f"crashed={crashed} phase_errors={V8_PHASE_ERRORS}"),
    ("5. no regression > 1 level on vc33/dc22", bar5,
     "; ".join(regressions) if regressions else "within 1 level of comparators"),
]
print("-" * 78)
for name, ok, detail in BARS:
    print(f"BAR {'PASS' if ok else 'FAIL'} — {name} ({detail})")
verdict = all(ok for _, ok, _ in BARS)
print(f"V8 SMOKE VERDICT: {'PASS' if verdict else 'FAIL'} "
      f"({sum(1 for _, ok, _ in BARS if ok)}/{len(BARS)} bars)")

results = {
    "arm": "v8-smoke",
    "install_verdict": _v8_status,
    "flags": {k: v for k, v in os.environ.items() if k.startswith("EXPLORER")},
    "phases": [[p[0], p[1], p[2]] for p in SMOKE_PHASES],
    "phase_errors": V8_PHASE_ERRORS,
    "games": games_out,
    "telemetry": snap,
    "envelope": envelope_rows,
    "bars": [{"bar": n, "pass": bool(ok), "detail": d} for n, ok, d in BARS],
    "verdict": "PASS" if verdict else "FAIL",
    "comparators": COMPARATORS,
    "offline_reference": {
        "ft09": {"levels": 6, "crack": True, "lane": "spec:ft09_gf2",
                 "engine_actions": 1233, "bank_actions": 75},
        "tu93": {"levels": 9, "crack": True, "lane": "nbfs_macros",
                 "engine_actions": 84687, "bank_actions": 187},
    },
}
(WORKING_DIR / "v8_smoke_results.json").write_text(
    json.dumps(results, indent=1), encoding="utf-8")
_v8_snapshot()
print("wrote", WORKING_DIR / "v8_smoke_results.json")
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    sources = {}
    for name, path in BUNDLE_FILES.items():
        text = path.read_text()
        assert text.strip(), name
        sources[name] = text
    assert "def install() -> str:" in sources["graft_explorer_v8.py"]
    assert "def _grind_v8" in sources["graft_explorer_v8.py"]
    assert "class _BankingKillSwitch" in sources["graft_bank.py"]

    def idx_of(marker: str) -> int:
        hits = [
            i
            for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and marker in "".join(c["source"])
        ]
        assert len(hits) == 1, (marker, len(hits))
        return hits[0]

    def code_cell(text: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }

    # 1. GPU fail-fast right after the imports cell.
    nb["cells"].insert(idx_of(MARK_IMPORTS) + 1, code_cell(GPU_ASSERT_CELL))

    # 2. Replace the base smoke hook with the two-phase v8 geometry.
    nb["cells"][idx_of(MARK_SMOKE)]["source"] = SMOKE_CELL.splitlines(keepends=True)

    # 3. Rewrite the run cell's single bm.run into the phase loop (everything
    #    else in that cell — including the scored-rerun branch — untouched).
    run_idx = idx_of(MARK_RUN)
    run_src = "".join(nb["cells"][run_idx]["source"])
    assert run_src.count(RUN_BLOCK_OLD) == 1, "base run cell drifted"
    run_src = run_src.replace(RUN_BLOCK_OLD, RUN_BLOCK_NEW)
    nb["cells"][run_idx]["source"] = run_src.splitlines(keepends=True)

    # 4+5. Graft install (5-file flat bundle) + telemetry, immediately before
    #      the run cell.
    bundle_lines = ["_V8_BUNDLE = {\n"]
    for name in ("graft_explorer.py", "graft_bank.py", "search_core.py",
                 "specialists.py", "graft_explorer_v8.py"):
        bundle_lines.append(f"    {name!r}: {sources[name]!r},\n")
    bundle_lines.append("}\n")
    graft_cell_text = GRAFT_CELL_HEAD + "".join(bundle_lines) + GRAFT_CELL_TAIL
    # The graft sources must ride the notebook byte-identically: the cell
    # embeds them via !r, and Python str repr round-trips by language guarantee.
    for name, text in sources.items():
        assert repr(text) in graft_cell_text, name
    nb["cells"].insert(run_idx, code_cell(graft_cell_text))
    nb["cells"].insert(run_idx + 1, code_cell(TELEMETRY_CELL))

    # 6. Final report after the run cell.
    nb["cells"].insert(idx_of(MARK_RUN) + 1, code_cell(REPORT_CELL))

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined
    assert joined.index("GPU misbind") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index("SMOKE_PHASES = [")
    assert joined.index('os.environ["EXPLORER_V8"] = "1"') < joined.index(MARK_RUN)
    assert joined.index("[v8-tel] counters installed") < joined.index(MARK_RUN)
    assert joined.index(MARK_RUN) < joined.index("V8 SMOKE RESULTS")
    assert "EFFORT_MEDIUM\"] = \"1\"" not in joined, "effort_medium must stay OFF"
    for stem in ("ft09-0d8bbf25", "dc22-fdcac232", "vc33-5430563c", "sk48-d8078629"):
        assert joined.count(stem) == 1, stem

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(
        json.dumps(
            {
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
                # LAW: the competition source is what gates the RTX Pro 6000
                # pool — machine_shape + --accelerator alone bind P100.
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "model_sources": [
                    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
                ],
            },
            indent=2,
        )
        + "\n"
    )

    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    print("built", KERNEL_SLUG, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(nb["cells"]), "notebook bytes:",
          (HERE / f"{KERNEL_SLUG}.ipynb").stat().st_size)


if __name__ == "__main__":
    main()
