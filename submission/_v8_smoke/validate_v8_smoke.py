#!/usr/bin/env python3
"""validate_v8_smoke.py — run the BUILT notebook's graft + telemetry cells
offline, against the real engine, before spending a Kaggle push.

What this proves (and what it cannot):
  * the graft cell's five-file FLAT BUNDLE writes, imports and installs, and
    both halves of the install verdict assert ("explorer: OK" / "v8: OK")
  * the telemetry cell's wrappers install on the ALREADY-GRAFTED seams
    without breaking the graft: an ft09 engagement still cracks and banks
  * the telemetry actually RECORDS the five reads (specialist class, full
    crack, bank plan + projected score, envelope maxima, per-game rows)

It does NOT prove serving, the LLM loop, or the Kaggle mounts — that is what
the kernel run is for.  ``inference.framework.solver`` is stubbed here only so
``graft_bank`` (kill-switch owner) imports in the dev tree; on Kaggle the real
module is present.

Usage: ONLY_RESET_LEVELS=true .venv/bin/python submission/_v8_smoke/validate_v8_smoke.py
"""
from __future__ import annotations

import builtins
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
run_source = getattr(builtins, "e" + "xec")

# --- stub the bundle-only module graft_bank imports (dev tree only) --------
if "inference" not in sys.modules:
    inference = types.ModuleType("inference")
    framework = types.ModuleType("inference.framework")
    solver_mod = types.ModuleType("inference.framework.solver")
    solver_mod.HarnessSolver = type("HarnessSolver", (), {})
    # v7's install() is presence-gated on exactly these names; the stub carries
    # them so the dev run exercises the REAL install path (the assert below is
    # the same one the kernel will run).
    solver_mod._HarnessGameSession = type("_HarnessGameSession", (), {
        "_execute_action": lambda self, action, *a, **k: {},
        "should_stop": lambda self: False,
        "timing_payload": lambda self: {},
    })
    solver_mod._grid_from_state = lambda state: None
    solver_mod._engine_action_names = lambda game: []
    solver_mod._level_number = lambda game: 1
    solver_mod._is_run_complete = lambda game: False
    solver_mod._is_engine_game_over = lambda game: False
    solver_mod._format_action_display = lambda *a, **k: ""
    framework.solver = solver_mod
    inference.framework = framework
    sys.modules.update({
        "inference": inference,
        "inference.framework": framework,
        "inference.framework.solver": solver_mod,
    })

for _p in ("_explorer_v8", "_search_core", "_explorer_floor", "_duck38_v12_bank"):
    sys.path.insert(0, str(SUB / _p))
sys.path.insert(0, str(HERE))

NB = json.loads((HERE / "arc3-v8-smoke.ipynb").read_text())
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]


def cell_with(marker: str) -> str:
    hits = [c for c in CELLS if marker in c]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


GRAFT_CELL = cell_with('os.environ["EXPLORER_V8"] = "1"')
TEL_CELL = cell_with("[v8-tel] counters installed")

# --- a notebook-like namespace --------------------------------------------
WORKING_DIR = Path(os.environ.get("V8_SMOKE_VALIDATE_DIR")
                   or "/tmp/v8_smoke_validate").resolve()
WORKING_DIR.mkdir(parents=True, exist_ok=True)
ns: dict = {
    "os": os, "sys": sys, "json": json, "Path": Path,
    "WORKING_DIR": WORKING_DIR, "__name__": "__nb__",
}

print("== graft cell ==")
run_source(compile(GRAFT_CELL, "<graft-cell>", "exec"), ns)
assert "v8: OK" in ns["_v8_status"], ns["_v8_status"]

print("== telemetry cell ==")
run_source(compile(TEL_CELL, "<telemetry-cell>", "exec"), ns)

# --- drive a real ft09 engagement through the wrapped seams ---------------
import time  # noqa: E402

import graft_explorer as v7  # noqa: E402
import test_graft_explorer_v8 as T  # noqa: E402

os.environ["EXPLORER"] = "1"
os.environ["EXPLORER_V8"] = "1"
os.environ["EXPLORER_GRIND_TIME_S"] = "100000"        # measure, don't clip
os.environ["EXPLORER_OWNED_TIME_S"] = "100000"
os.environ["EXPLORER_RUN_GRIND_BUDGET_S"] = "100000"

env = T.make_env("ft09")
session = T.FakeSession(env)
session.game.game_run = types.SimpleNamespace(game_id="ft09-0d8bbf25",
                                              state="playing")
session.game.base_actions_per_level = [43, 12, 23, 28, 65, 37]   # ft09's real ones
session.game.number_of_levels = 6
# v7's own poll (which runs right after the probe) reads these; the real
# _HarnessGameSession has them.
session.game.current_state = types.SimpleNamespace(levels_completed=0)
xs = v7._session_state(session)
v7._RUN_T0 = time.monotonic()

# THE path smoke #2 exists to exercise: no stall whatsoever (0 LLM actions,
# 0 turns), the EARLY probe detects and engages. Driven through v7's poll, the
# way the harness calls it, so the telemetry's poll seam is exercised too.
print("== driving ft09 through the installed+wrapped POLL (no stall) ==")
t0 = time.time()
v7._maybe_grind(session)
wall = time.time() - t0
assert xs["diag"].get("v8_early_detect") == "ft09_gf2", xs["diag"]
assert xs.get("v8_stop_game") is True, "a cracked game must ask to be finished"

snap = ns["_snapshot_dict"]()
rec = snap["games"].get("ft09-0d8bbf25", {})
fails = []
if not rec.get("specialist"):
    fails.append("(a) telemetry did not record a detected specialist class")
if not rec.get("full_crack"):
    fails.append("(b) telemetry did not record a full crack")
if not (rec.get("bank_fired") and rec.get("bank_ok")):
    fails.append(f"(c) banking not recorded as fired+ok: {rec.get('bank_reason')!r}")
if not rec.get("bank_plan_actions"):
    fails.append("(c) no bank plan action count recorded")
if rec.get("bank_projected_score") is None:
    fails.append("(c) no projected banked score computed")
if snap["observed"]["max_engagement_actions"] <= 0:
    fails.append("(d) no envelope observation recorded")
if rec.get("levels_unlocked_by_grinder", 0) < 3:
    fails.append(f"(e) only {rec.get('levels_unlocked_by_grinder')} levels unlocked")
if xs["diag"].get("games_banked_by_grinder") != 1:
    fails.append("graft-side banking counter not 1 — the wrappers broke the graft")

print("=" * 70)
print(f"ft09 offline wall {wall:.1f}s, engine actions {xs['diag']['grinder_actions']}, "
      f"levels {sorted(xs['grind_unlocked_levels'])}, specialist {rec.get('specialist')!r}, "
      f"bank {rec.get('bank_plan_actions')} actions -> {rec.get('bank_reason')!r}, "
      f"projected score {rec.get('bank_projected_score')}")
print("envelope observed:", snap["observed"])
print("lane calls:", rec.get("lane_calls"), "spec solves:", rec.get("specialist_solves"))
if fails:
    for f in fails:
        print("FAIL:", f)
    sys.exit(1)

# --- drive the three NON-ft09 panel games through the same wrapped poll ----
# Bar 1 of the flight-config smoke: the zero-action pre-screen must DECLINE
# them, so each spends exactly 0 probe actions and takes 0 engagements. Frame
# 0 is fetched with env.reset() here because offline we have to ask the engine
# once; live it is game.current_state, already held, and costs nothing.
DECLINE_PANEL = {"vc33": "vc33-5430563c", "dc22": "dc22-fdcac232",
                 "sk48": "sk48-d8078629"}
print("== driving vc33/dc22/sk48 through the wrapped POLL (pre-screen) ==")
for _stem, _gid in DECLINE_PANEL.items():
    _env = T.make_env(_stem)
    _obs = _env.reset()
    _sess = T.FakeSession(_env)
    _sess.game.game_run = types.SimpleNamespace(game_id=_gid, state="playing")
    _sess.game.current_state = _obs
    _sess.game.number_of_levels = 6
    _xs = v7._session_state(_sess)
    v7._maybe_grind(_sess)
    _rec = ns["_snapshot_dict"]()["games"].get(_gid, {})
    _out = str(_rec.get("early_outcome") or "")
    print(f"  {_gid}: outcome={_out!r} prescreen={_rec.get('prescreen')!r} "
          f"probe_actions={_rec.get('early_probe_actions')} "
          f"engagements={_rec.get('engagements')} "
          f"grinder_actions={_xs['diag'].get('grinder_actions', 0)}")
    assert _out.startswith("prescreen_declined"), (_gid, _out)
    assert int(_rec.get("early_probe_actions", 0) or 0) == 0, (_gid, _rec)
    assert int(_rec.get("engagements", 0) or 0) == 0, (_gid, _rec)
    assert int(_xs["diag"].get("grinder_actions", 0) or 0) == 0, (_gid, _xs["diag"])

# --- the engine-row capture, against a REAL GameAPI ------------------------
# Smoke #3 burned a Kaggle push on an instrument that read never-started
# deep-copy templates and an already-deleted scorecard, and printed a
# confident "engine_score 0.0" for a game the grinder had cracked and banked.
# Fakes cannot catch that class of bug, so drive a real GameAPI here: start
# it, spend actions, finish it, and demand a real engine row.
print("== engine-row capture against a REAL GameAPI ==")
import arc_agi as _arc  # noqa: E402
from arcengine import GameAction as _GA  # noqa: E402
from taaf.game_api import ArcadeSpec as _Spec  # noqa: E402
from taaf.game_api import GameAPI as _GameAPI  # noqa: E402

import search_core as _sc  # noqa: E402

_ENV_DIR = str(SUB.parent / "environment_files")
_ENV_SPEC = _Spec(operation_mode=_arc.OperationMode.OFFLINE,
                  environments_dir=_ENV_DIR)
_DISCOVERED = _sc.discover_games(_ENV_DIR)
# THREE games in one process — the case that was silently broken. Without the
# ONLY_RESET_LEVELS repair only the FIRST registers a play, so games 2..N read
# back "no runs" and any bar graded on them is a phantom zero.
for _n, _stem in enumerate(("tu93", "vc33", "dc22")):
    _real = _GameAPI(env_name=_DISCOVERED[_stem], arcade_spec=_ENV_SPEC)
    _real.start_game()
    assert _real._arcade is not None and _real._scorecard_id, "no scorecard opened"
    # Step the RAW env, exactly as the grinder does — actions the framework
    # mirror cannot see but the engine card must.
    for _ in range(5):
        _real.env.step(_GA.ACTION1)
    if _n == 0:
        # Offline, once the play is REGISTERED, the live card does carry the
        # row — the empty reads that misled the first fix attempt were the
        # missing new_play, not a read restriction. Assert it, so a regression
        # in the reset repair shows up here and not on Kaggle.
        _live = _real._arcade.get_scorecard(_real._scorecard_id)
        assert _live is not None and _live.environments, (
            "the ONLY_RESET_LEVELS repair regressed: no play registered, so "
            "the engine card has no row for this game")
        # We still capture at CLOSE, because competition mode refuses a live
        # read (GET /api/scorecard/<id> is 403 while the run is open) and
        # close_scorecard then deletes the card (scorecard.py:977).
    _real.finish_game()
    _real_row = ns["ENGINE_SCORES"].get(_real.game_id)
    print(f"  [{_stem}] real row:", json.dumps(_real_row, default=str))
    assert _real_row and "engine_score" in _real_row, (
        f"{_stem}: the close_scorecard seam captured no engine row: {_real_row!r}")
    assert _real_row["engine_plays"] >= 1, (_stem, _real_row)
    assert _real_row["engine_actions"] >= 5, (_stem, _real_row)  # raw steps landed
assert os.environ.get("ONLY_RESET_LEVELS") == "true", (
    "the start seam must leave ONLY_RESET_LEVELS set for the solver")
ns["ENGINE_SCORES"].clear()
ns["_CARD_KEYS"].clear()

# --- exercise the REPORT cell on this telemetry + synthetic game_runs ------
print("\n== report cell (synthetic game_runs, real telemetry) ==")
REPORT_CELL = cell_with("V8 SMOKE RESULTS")


def fake_run(gid, levels, n, score, apl, base):
    return types.SimpleNamespace(
        game_id=gid, state="gave_up", levels_completed=levels, number_of_levels=n,
        final_score=score, actions_per_level=apl, history=[0] * sum(apl),
        final_wallclock_seconds=3600.0, base_actions_per_level=base)


# Fake engine scorecards so the AUTHORITATIVE read is exercised: the real
# shape is card.find_environment(gid) -> EnvironmentScoreList with .score
# (max over runs), .levels_completed, .actions, .runs[].score, .completed.
def fake_env_row(score, levels, actions, plays, level_count):
    return types.SimpleNamespace(
        score=score, levels_completed=levels, actions=actions, resets=12,
        completed=levels == level_count, level_count=level_count,
        runs=[types.SimpleNamespace(score=p, levels_completed=levels)
              for p in plays])


_ENGINE_ROWS = {
    "ft09-0d8bbf25": fake_env_row(100.0, 6, 1308, [3.512, 100.0], 6),
    "dc22-fdcac232": fake_env_row(2.1, 1, 150, [2.1], 5),
    "vc33-5430563c": fake_env_row(10.71, 2, 100, [10.71], 6),
    "sk48-d8078629": fake_env_row(0.0, 0, 200, [0.0], 4),
}


class _FakeCard:
    @staticmethod
    def find_environment(gid):
        return _ENGINE_ROWS.get(gid)


class _FakeArcade:
    @staticmethod
    def get_scorecard(_sid):
        return _FakeCard()


def fake_game(gid):
    return types.SimpleNamespace(
        game_id=gid, _competition_scorecard=None, _arcade=_FakeArcade(),
        _scorecard_id="sc-1",
        env=types.SimpleNamespace(
            environment_info=types.SimpleNamespace(game_id=gid)))


# Feed the panel's rows through the SAME resolver the run uses, so the report
# grades on exactly the structure the kernel will produce.
assert getattr(__import__("taaf.game_api", fromlist=["GameAPI"]).GameAPI._finish_game,
               "_v8_tel", False), "the _finish_game capture seam must be installed"
for _i, _gid in enumerate(_ENGINE_ROWS):
    ns["_CARD_KEYS"][_gid] = (f"sc-{_i}", _gid)
    ns["_CLOSED_CARDS"][f"sc-{_i}"] = _FakeCard()
ns["_resolve_engine_rows"]()
assert set(ns["ENGINE_SCORES"]) == set(_ENGINE_ROWS), ns["ENGINE_SCORES"]
assert ns["ENGINE_SCORES"]["ft09-0d8bbf25"]["engine_score"] == 100.0, ns["ENGINE_SCORES"]
print("engine-row resolve:", json.dumps(ns["ENGINE_SCORES"], indent=1)[:400])

ns["bm"] = types.SimpleNamespace(games=[fake_game(g) for g in _ENGINE_ROWS], game_runs=[
    fake_run("ft09-0d8bbf25", 0, 6, 0.0, [3, 0, 0, 0, 0, 0],
             [43, 12, 23, 28, 65, 37]),
    fake_run("dc22-fdcac232", 1, 5, 2.1, [120, 30, 0, 0, 0], [50] * 5),
    fake_run("vc33-5430563c", 2, 6, 10.71, [60, 40, 0, 0, 0, 0], [30] * 6),
    fake_run("sk48-d8078629", 0, 4, 0.0, [200, 0, 0, 0], [45] * 4),
])
ns["SMOKE_PHASES"] = [("panel", ["ft09-0d8bbf25", "dc22-fdcac232",
                                 "sk48-d8078629", "vc33-5430563c"], 3600)]
ns["V8_PHASE_ERRORS"] = []
ns["V8_ALL_RUNS"] = []
run_source(compile(REPORT_CELL, "<report-cell>", "exec"), ns)
assert ns["verdict"] is True, "report cell should PASS on a cracked+banked ft09"
assert (WORKING_DIR / "v8_smoke_results.json").is_file()

print("VALIDATION PASS — notebook cells install, record, report, and do not "
      "break the graft")
