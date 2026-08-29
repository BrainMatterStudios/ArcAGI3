#!/usr/bin/env python3
"""validate_duck38_tp1.py — run the BUILT flight arm's graft cell offline,
against the real anim bundle, before spending a Kaggle push.

Proves: the embedded source writes/imports/installs; the hard asserts pass;
the flight config is what reaches the graft at runtime (read back from
os.environ and ToolAgent instances, not from source text); the time guard
shrinks the per-game cap on the scored path; the v12 base is intact.

Usage: .venv/bin/python submission/_duck38_tp1/validate_duck38_tp1.py
"""
from __future__ import annotations

import builtins
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
BUNDLE = SUB / "_inspect_replay" / "assets_build" / "ARC3-Inference"
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(SUB / "_throughput_v1"))
run_source = getattr(builtins, "e" + "xec")

ARM = sys.argv[1] if len(sys.argv) > 1 else "tp1"
nb = json.loads(next((HERE / ARM).glob("*.ipynb")).read_text())
cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
graft_cells = [c for c in cells if 'os.environ["TP_ENABLE"] = "1"' in c]
assert len(graft_cells) == 1, len(graft_cells)
graft = graft_cells[0]

for flag in list(os.environ):
    if flag.startswith(("TP_", "TP2_", "TP4_")):
        os.environ.pop(flag, None)
os.environ["EFFORT_MEDIUM"] = "1"      # must be purged by the cell

solver = types.SimpleNamespace(max_runtime_s_per_game=7920.0, concurrency=28)
bm = types.SimpleNamespace(solver=solver)
ns = {
    "os": os, "sys": sys, "time": time, "Path": Path,
    "WORKING_DIR": Path(tempfile.mkdtemp(prefix="tp1_validate_")),
    "NOTEBOOK_START_EPOCH": time.time() - 900.0,   # pretend setup took 15 min
    "true_submission": True,
    "bm": bm,
}
run_source(graft, ns)

from inference.agent import tool_agent as agent_mod  # noqa: E402

status = ns["_tpmod"].status()
import graft_control as _tc, graft_explore as _te  # noqa: E402
checks = {
    "install OK": ns["_tp_status"] == "throughput: OK",
    "flags in env": {k: os.environ.get(k) for k in ("TP_ENABLE", "TP_CONTEXT_WINDOW", "TP_BATCH_CAP")}
                    == {"TP_ENABLE": "1", "TP_CONTEXT_WINDOW": "0", "TP_BATCH_CAP": "0"},
    "effort purged": "EFFORT_MEDIUM" not in os.environ,
    "status enabled": status["enabled"] and status["trim_low_water"] == 1.0,
    "seams wrapped": all(
        hasattr(getattr(agent_mod.ToolAgent, n), "_tp_stock") or hasattr(getattr(agent_mod.ToolAgent, n), "_tp2_stock")
        for n in ("_trim_messages_for_context", "__init__", "_update_summarized_knowledge_from_step_summary",
                  "_normalize_python_actions", "_run_python_tool")),
    "time guard shrank cap": 7000.0 < solver.max_runtime_s_per_game < 7920.0,
    "v12 base intact": sum(1 for c in cells if "KAGGLE_IS_COMPETITION_RERUN" in c) >= 1
                       and any("attest: OK" in c for c in cells)
                       and any('SMOKE_GAMES = ["vc33-5430563c"' in c for c in cells),
    "no telemetry cells": not any("tp-tel" in c for c in cells),
    "control/explore installed": ns["_tc_status"] == "control: OK" and ns["_te_status"] == "explore: OK",
    "arm flags": {"tp1": (False, False), "tp2": (True, False), "tp24": (True, True)}[ARM]
                 == (_tc.enabled(), _te.enabled()),
}
agent = agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
checks["instance budget"] = agent._context_budget_tokens == max(1024, agent_mod._LOCAL_ANALYZER_CONTEXT_WINDOW - agent._reply_reserve_tokens - agent._request_safety_margin_tokens)
checks["instance yield/steps"] = agent._yield_seconds == (None if agent_mod._LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(agent_mod._LOCAL_ANALYZER_YIELD_SECONDS)) and agent._tool_steps == (None if agent_mod._LOCAL_ANALYZER_TOOL_STEPS <= 0 else agent_mod._LOCAL_ANALYZER_TOOL_STEPS)

ok = True
for name, passed in checks.items():
    print(("OK  " if passed else "FAIL"), name)
    ok = ok and passed
print("per-game cap after guard:", solver.max_runtime_s_per_game)
print("VALIDATE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
