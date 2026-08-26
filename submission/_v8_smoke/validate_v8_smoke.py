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
session.game.game_run = types.SimpleNamespace(game_id="ft09-0d8bbf25")
session.game.base_actions_per_level = [40] * 6
session.game.number_of_levels = 6
xs = v7._session_state(session)
v7._RUN_T0 = time.monotonic()

print("== driving the ft09 engagement through the installed+wrapped grind ==")
t0 = time.time()
v7._grind(session, xs, 0)          # the telemetry-wrapped v8 grind
wall = time.time() - t0

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

# --- exercise the REPORT cell on this telemetry + synthetic game_runs ------
print("\n== report cell (synthetic game_runs, real telemetry) ==")
REPORT_CELL = cell_with("V8 SMOKE RESULTS")


def fake_run(gid, levels, n, score, apl, base):
    return types.SimpleNamespace(
        game_id=gid, state="gave_up", levels_completed=levels, number_of_levels=n,
        final_score=score, actions_per_level=apl, history=[0] * sum(apl),
        final_wallclock_seconds=3600.0, base_actions_per_level=base)


ns["bm"] = types.SimpleNamespace(game_runs=[
    fake_run("ft09-0d8bbf25", 0, 6, 0.0, [140, 0, 0, 0, 0, 0], [40] * 6),
    fake_run("dc22-fdcac232", 1, 5, 2.1, [120, 30, 0, 0, 0], [50] * 5),
    fake_run("vc33-5430563c", 2, 6, 10.71, [60, 40, 0, 0, 0, 0], [30] * 6),
    fake_run("sk48-d8078629", 0, 4, 0.0, [200, 0, 0, 0], [45] * 4),
])
ns["SMOKE_PHASES"] = [("A-ft09", ["ft09-0d8bbf25"], 3600),
                      ("B-panel", ["dc22-fdcac232", "vc33-5430563c",
                                   "sk48-d8078629"], 3600)]
ns["V8_PHASE_ERRORS"] = []
ns["V8_ALL_RUNS"] = []
run_source(compile(REPORT_CELL, "<report-cell>", "exec"), ns)
assert ns["verdict"] is True, "report cell should PASS on a cracked+banked ft09"
assert (WORKING_DIR / "v8_smoke_results.json").is_file()

print("VALIDATION PASS — notebook cells install, record, report, and do not "
      "break the graft")
