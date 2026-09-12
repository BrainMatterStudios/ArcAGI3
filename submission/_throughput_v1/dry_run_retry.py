#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the fresh-mind level retry graft on the REAL
engine (arcengine over environment_files, three public games) with the mock
brain from dry_run.py.

Two modes:

  --mode retry-only (default)
      june_stock agent bytes (the bundle the offkaggle runner plays) + the
      sha-pinned taaf framework, ONLY graft_retry installed, thresholds
      shrunk (RETRY_K=1, RETRY_ABS=15, cooldown 10, max 2) so the trigger
      fires inside a 60-action game. Proves: baselines are reachable from
      the agent at runtime; the RESET goes through the session's normal
      action path (history / actions_per_level / runtime state); the
      [RETRY] marker lands in the harness transcript; the FRESH MIND block
      rides exactly one prompt per retry; games end normally.

  --mode all-packs
      graft_retry installed FIRST, then dry_run.main() installs Packs 1-10
      on top with its own checks — proves no interaction regression with the
      full graft stack (anim bundle, as dry_run.py is written).

Run:  .venv/bin/python submission/_throughput_v1/dry_run_retry.py [--mode all-packs]
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TOOLKIT = REPO / "reference/arc-agi-toolkit"
JUNE_STOCK = REPO / "scratchpad/bundles/june_stock/src/ARC3-Inference"
TAAF_PINNED = REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"
RETRY_ENV = {"RETRY_ENABLE": "1", "RETRY_K": "1", "RETRY_ABS": "15", "RETRY_COOLDOWN": "10", "RETRY_MAX": "2"}
# --wipe (2026-09-12): the no-RESET dose on the turns trigger — 3 analyze() turns on a level, max 2 wipes/level
WIPE_ENV = {**RETRY_ENV, "RETRY_MODE": "wipe", "RETRY_TURNS": "3"}
# dry_run.MOCK_TOOL_CODE emits ACTION7 on sb26, which june_stock's action names
# reject (0 actions in 240 s); this picker sticks to the model-facing names.
MOCK_TOOL_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')]\n"
    "step = len(history)\n"
    "if prefs:\n"
    "    action(prefs[step % len(prefs)])\n"
    "elif 'MOUSE' in [str(v).upper() for v in valid_actions]:\n"
    "    action({'action': 'MOUSE', 'row': (step * 7) % 64, 'col': (step * 11) % 64})\n"
    "else:\n"
    "    action(valid_actions[0])\n"
)


def _retry_only(wipe: bool = False) -> int:
    sys.path.insert(0, str(HERE))
    import dry_run  # noqa: PLC0415 - MockBrain + game list, no side effects at import

    for p in (TOOLKIT, JUNE_STOCK, TAAF_PINNED):
        assert p.is_dir(), f"missing {p}"
    sys.path.insert(0, str(TOOLKIT))
    sys.path.insert(0, str(JUNE_STOCK))
    sys.path.insert(0, str(TAAF_PINNED))

    workroot = REPO / "scratchpad" / "tp_dry_run" / ("retry-" + time.strftime("%H%M%S"))
    workroot.mkdir(parents=True, exist_ok=True)
    dry_run.MOCK_TOOL_CODE = MOCK_TOOL_CODE          # MockBrain reads the module global per request
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), dry_run.MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[retry-dry] mock brain at {base_url}")

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b", "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true", "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(workroot / "server_recording"),
        "LOCAL_ANALYZER_YIELD_SECONDS": "60", "LOCAL_ANALYZER_TOOL_STEPS": "8",
        **(WIPE_ENV if wipe else RETRY_ENV),
    })

    import arc_agi  # noqa: PLC0415
    import graft_retry as tr  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415
    from taaf.benchmark import Benchmark  # noqa: PLC0415
    from taaf.game_api import ArcadeSpec, GameAPI  # noqa: PLC0415

    assert Path(agent_mod.__file__).resolve().is_relative_to(JUNE_STOCK.resolve()), agent_mod.__file__
    print("[retry-dry]", tr.install())
    assert tr._STATE["installed"]

    counters = {"resets_via_execute_action": 0, "prompts_with_fresh_mind": 0, "prompts": 0}

    inner_prompt = agent_mod.ToolAgent._build_user_prompt

    def prompt(self, *a, **k):
        text = inner_prompt(self, *a, **k)
        counters["prompts"] += 1
        counters["prompts_with_fresh_mind"] += "FRESH MIND" in text
        return text

    agent_mod.ToolAgent._build_user_prompt = prompt

    inner_execute_action = solver_mod._HarnessGameSession._execute_action

    def execute_action_counted(self, action, *a, **k):
        if action.id.name == "RESET":
            counters["resets_via_execute_action"] += 1
        return inner_execute_action(self, action, *a, **k)

    solver_mod._HarnessGameSession._execute_action = execute_action_counted

    spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
    solver = solver_mod.HarnessSolver(label="retry-dry", model="mock-27b", analyzer_timeout=30,
                                      max_actions_per_game=60, max_runtime_s_per_game=240.0, concurrency=3)
    games = [GameAPI(env_name=g, arcade_spec=spec) for g in dry_run._game_names()]
    bm = Benchmark(label="retry-dry", games=games, solver=solver, job_dir=workroot)
    t0 = time.monotonic()
    asyncio.run(bm.run(minimal_diagnostics=True))
    wall = time.monotonic() - t0

    rows = []
    for run in bm.game_runs:
        apl = list(run.actions_per_level or [])
        rows.append({"game": run.game_id, "state": run.state, "levels": run.levels_completed,
                     "actions": sum(apl) if apl else len(run.history), "actions_per_level": apl,
                     "baselines": list(run.base_actions_per_level) if run.base_actions_per_level else None,
                     "history_len": len(run.history), "resets_in_history": sum(1 for r in run.history if r.action.id.name == "RESET"),
                     "note": run.solver_note})
    print("[retry-dry] runs:", json.dumps(rows, indent=1))
    st = tr.status()
    print("[retry-dry] retry status:", json.dumps({k: st[k] for k in ("retries_fired", "levels_cleared_after_retry",
                                                                       "retry_log", "clear_log", "skips")}, indent=1))
    transcripts = sorted((workroot / "transcripts").glob("*.txt"))
    marker_lines = 0
    clear_lines = 0
    section_lines = 0
    for path in transcripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        marker_lines += sum(1 for line in text.splitlines() if line.startswith("[RETRY] game="))
        clear_lines += sum(1 for line in text.splitlines() if line.startswith("[RETRY-CLEAR] game="))
        section_lines += text.count("[HARNESS RETRY]\n")
    print("[retry-dry] counters:", counters, f"transcripts={len(transcripts)} [RETRY]={marker_lines} "
          f"[RETRY-CLEAR]={clear_lines} sections={section_lines} tool_posts={dry_run.MockBrain.tool_posts} wall={wall:.0f}s")

    fired = st["retries_fired"]
    if wipe:
        graft_resets = sum(1 for r in st["retry_log"] if r.get("mode") != "wipe")
        checks = {
            "3 games played": len(rows) == 3 and all(r["actions"] > 0 for r in rows),
            "no crash": all(r["state"] != "crashed" for r in rows) and all("error" not in str(r["note"]) for r in rows),
            "wipe fired in every game": fired >= 3 and {r["game"] for r in st["retry_log"]} == {r["game"] for r in rows},
            "every fire was a wipe (no graft RESET)": graft_resets == 0 and all(r.get("mode") == "wipe" for r in st["retry_log"]),
            "[RETRY] marker per wipe": marker_lines == fired and section_lines == marker_lines + clear_lines,
            "FRESH MIND block once per wipe": counters["prompts_with_fresh_mind"] == fired,
            "bucket invariant": all(sum(r["actions_per_level"]) == r["history_len"] for r in rows),
            "turns trigger obeyed (threshold = RETRY_TURNS)": all(r["threshold"] == int(WIPE_ENV["RETRY_TURNS"]) for r in st["retry_log"]),
            "max wipes per level": all(r["retry"] <= 2 for r in st["retry_log"]),
        }
        ok = True
        for name, passed in checks.items():
            print(("OK  " if passed else "FAIL"), name)
            ok = ok and passed
        print("[retry-dry] WIPE", "PASS" if ok else "FAIL", "artifacts under", workroot)
        return 0 if ok else 1
    checks = {
        "3 games played": len(rows) == 3 and all(r["actions"] > 0 for r in rows),
        "no crash": all(r["state"] != "crashed" for r in rows) and all("error" not in str(r["note"]) for r in rows),
        "baselines reachable at runtime (offline)": all(isinstance(r["baselines"], list) and r["baselines"] for r in rows),
        "retry fired in every game": fired >= 3 and {r["game"] for r in st["retry_log"]} == {r["game"] for r in rows},
        "RESET went through _execute_action": counters["resets_via_execute_action"] >= fired,
        "RESET recorded in game history": sum(r["resets_in_history"] for r in rows) >= fired,
        "[RETRY] marker per retry": marker_lines == fired and section_lines == marker_lines + clear_lines,
        "FRESH MIND block once per retry": counters["prompts_with_fresh_mind"] == fired,
        "bucket invariant": all(sum(r["actions_per_level"]) == r["history_len"] for r in rows),
        "trigger obeyed threshold": all(r["actions"] >= r["threshold"] for r in st["retry_log"]),
        "max retries per level": all(r["retry"] <= 2 for r in st["retry_log"]),
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    print("[retry-dry]", "PASS" if ok else "FAIL", "artifacts under", workroot)
    return 0 if ok else 1


def _all_packs() -> int:
    sys.path.insert(0, str(HERE))
    import dry_run  # noqa: PLC0415

    sys.path.insert(0, str(dry_run.TOOLKIT))
    sys.path.insert(0, str(dry_run.INFERENCE))
    sys.path.insert(0, str(dry_run.TAAF))
    os.environ.update(RETRY_ENV)
    import graft_retry as tr  # noqa: PLC0415
    print("[retry-dry] installing graft_retry FIRST:", tr.install())
    rc = dry_run.main()
    st = tr.status()
    print("[retry-dry] retry status after full-stack dry run:",
          json.dumps({k: st[k] for k in ("retries_fired", "levels_cleared_after_retry", "retry_log", "skips")}, indent=1))
    fired_ok = st["retries_fired"] >= 1
    print(("OK  " if fired_ok else "FAIL"), "retry fired under the full graft stack")
    print("[retry-dry] all-packs", "PASS" if rc == 0 and fired_ok else "FAIL")
    return 0 if rc == 0 and fired_ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("retry-only", "all-packs", "wipe"), default="retry-only")
    args = p.parse_args()
    if args.mode == "wipe":
        return _retry_only(wipe=True)
    return _retry_only() if args.mode == "retry-only" else _all_packs()


if __name__ == "__main__":
    sys.exit(main())
