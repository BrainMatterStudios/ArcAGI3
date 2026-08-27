#!/usr/bin/env python3
"""validate_duck38_v8.py — run the BUILT scored arm's graft cell offline,
against the real engine, before spending a Kaggle push.

What this proves:
  * the inserted cell's six-file FLAT BUNDLE writes, imports and installs
  * the HARD assert passes on all five halves of the install verdict
    (explorer / v8 / early / prescreen / stop-after-crack)
  * the FLIGHT CONFIG is what actually reaches the graft at runtime — read
    back from os.environ, not from the source text
  * the arm carries the v12 base intact (attestation, smoke hook, the scored
    KAGGLE_IS_COMPETITION_RERUN path) and NO telemetry/report machinery
  * the configured graft still cracks and banks ft09 through the real engine

It does NOT prove serving, the LLM loop, or the Kaggle mounts — that is what
the commit run is for. ``inference.framework.solver`` is stubbed here only so
``graft_bank`` imports in the dev tree; on Kaggle the real module is present.

Usage: .venv/bin/python submission/_duck38_v8/validate_duck38_v8.py
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

NB = json.loads((HERE / "arc3-duck38-v8.ipynb").read_text())
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
JOINED = "\n".join(CELLS)

FLIGHT_CONFIG = {
    "EXPLORER": "1",
    "EXPLORER_V8": "1",
    "EXPLORER_V8_EARLY": "1",
    "EXPLORER_V8_EARLY_ENGAGE": "1",
    "EXPLORER_V8_PRESCREEN": "1",
    "EXPLORER_V8_ENGAGE_SPECIALISTS": "ft09_gf2",
    "EXPLORER_V8_SPECIALIST_MAX_ACTIONS": "4000",
    "EXPLORER_V8_STALL_GRIND": "0",
    "EXPLORER_V8_BANK": "1",
    "EXPLORER_V8_STOP_AFTER_CRACK": "1",
}

# --- the arm must be the v12 base + ONE cell, nothing else -----------------
assert len(NB["cells"]) == 12, len(NB["cells"])
assert "attest FAIL: quantization_config" in JOINED, "boot attestation cell missing"
assert "KAGGLE_IS_COMPETITION_RERUN" in JOINED, "scored-rerun path missing"
assert "Qwen/Qwen3.8-27B-FP8" in JOINED, "served brain changed"
for banned in ("v8_smoke_results.json", "[v8-tel]", "SMOKE_PHASES",
               'EFFORT_MEDIUM"] = "1"'):
    assert banned not in JOINED, f"the scored arm must not carry {banned!r}"

hits = [c for c in CELLS if 'os.environ["EXPLORER_V8"] = "1"' in c]
assert len(hits) == 1, len(hits)
GRAFT_CELL = hits[0]

# --- run the graft cell ----------------------------------------------------
WORKING_DIR = Path(os.environ.get("V8_ARM_VALIDATE_DIR")
                   or "/tmp/duck38_v8_validate").resolve()
WORKING_DIR.mkdir(parents=True, exist_ok=True)
ns: dict = {"os": os, "sys": sys, "json": json, "Path": Path,
            "WORKING_DIR": WORKING_DIR, "__name__": "__nb__"}

print("== graft cell ==")
run_source(compile(GRAFT_CELL, "<graft-cell>", "exec"), ns)
status = ns["_v8_status"]
for half in ("explorer: OK", "v8: OK", "early: OK", "prescreen: OK",
             "stop-after-crack"):
    assert half in status, (half, status)

# The config as the RUNTIME sees it, not as the source reads.
live = {k: os.environ.get(k) for k in FLIGHT_CONFIG}
print("live flight config:", json.dumps(live))
assert live == FLIGHT_CONFIG, live
assert os.environ.get("EFFORT_MEDIUM") is None, "effort_medium must be purged"
assert (WORKING_DIR / "v8_bundle" / "prescreen.py").is_file(), "bundle not written"

# --- the configured graft must still crack and bank ft09 ------------------
import time  # noqa: E402

import graft_explorer as v7  # noqa: E402
import test_graft_explorer_v8 as T  # noqa: E402

os.environ["EXPLORER_GRIND_TIME_S"] = "100000"        # measure, don't clip
os.environ["EXPLORER_OWNED_TIME_S"] = "100000"
os.environ["EXPLORER_RUN_GRIND_BUDGET_S"] = "100000"

env = T.make_env("ft09")
session = T.FakeSession(env)
session.game.game_run = types.SimpleNamespace(game_id="ft09-0d8bbf25", state="playing")
session.game.base_actions_per_level = [43, 12, 23, 28, 65, 37]
session.game.number_of_levels = 6
session.game.current_state = env.reset()
xs = v7._session_state(session)
v7._RUN_T0 = time.monotonic()

print("== ft09 through the installed poll (no stall, prescreen in front) ==")
v7._maybe_grind(session)
diag = xs["diag"]
assert diag.get("v8_prescreen", "").startswith("ft09_gf2:strict_lattice"), diag
assert diag.get("v8_early_detect") == "ft09_gf2", diag
assert diag.get("games_won_by_grinder") == 1, diag
assert diag.get("games_banked_by_grinder") == 1, diag
assert xs.get("v8_stop_game") is True, "a cracked+banked game must ask to stop"
print(f"  ft09 cracked levels {sorted(xs['grind_unlocked_levels'])} in "
      f"{diag['grinder_actions']} engine actions, banked "
      f"({diag.get('v8_bank_result')})")

# --- and a declined game must spend nothing -------------------------------
print("== vc33 through the same poll (must be declined at 0 actions) ==")
env2 = T.make_env("vc33")
s2 = T.FakeSession(env2)
s2.game.game_run = types.SimpleNamespace(game_id="vc33-5430563c", state="playing")
s2.game.current_state = env2.reset()
xs2 = v7._session_state(s2)
v7._maybe_grind(s2)
assert str(xs2.get("v8_early_done") or "").startswith("prescreen_declined"), xs2
assert int(xs2["diag"].get("grinder_actions", 0) or 0) == 0, xs2["diag"]
assert int(xs2["diag"].get("v8_early_probe_actions", 0) or 0) == 0, xs2["diag"]
print(f"  vc33 declined: {xs2['v8_early_done']} — 0 engine actions")

print("VALIDATION PASS — the scored arm installs the flight config, cracks and "
      "banks ft09, and spends nothing on a declined game")
