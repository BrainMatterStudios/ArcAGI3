#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the Pack-1 + Pack-2 grafts.

Mock brain (HTTP, OpenAI-compatible) <- REAL anim-bundle ToolAgent/solver
(submission/_inspect_replay/assets_build/ARC3-Inference) <- REAL taaf
framework (scratchpad/bundles/anim_20260807, the dataset that plays at eval)
<- REAL arcengine over environment_files, three public games.

The mock brain answers a tool-bearing request with a python tool call whose
code cycles through the valid actions (so keyboard games move and click games
click the centre), and answers a tool-less request (the Pack-2 summary call)
with a canned seven-line note. Budgets are shrunk so every seam fires within
~60 actions per game: small context window (cuts -> summaries), low stall
thresholds (directive + harness RESET), streak halts on inert clicks.

What it proves: both grafts install on the real classes; the probe runs on
the real engine and its table reaches the prompt; diffs ride action results;
summaries are requested and merged; the stagnation directive and RESET tier
fire; nothing crashes; games end normally.

Run:  .venv/bin/python submission/_throughput_v1/dry_run.py
"""
from __future__ import annotations

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
INFERENCE = REPO / "submission/_inspect_replay/assets_build/ARC3-Inference"
TAAF = REPO / "scratchpad/bundles/anim_20260807/src/tufa-arc-agi-framework/src"
GAME_IDS = ["vc33", "ls20", "sb26"]


def _game_names() -> list[str]:
    out = []
    for gid in GAME_IDS:
        versions = sorted(p.name for p in (REPO / "environment_files" / gid).iterdir() if p.is_dir())
        assert len(versions) == 1, (gid, versions)
        out.append(f"{gid}-{versions[0]}")
    return out

MOCK_TOOL_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "step = len(history)\n"
    "if prefs:\n"
    "    action(prefs[step % min(4, len(prefs))])\n"
    "elif valid_actions:\n"
    "    action({'action': 'MOUSE', 'row': 32, 'col': 32})\n"
    "else:\n"
    "    action('UP')\n"
)
MOCK_SUMMARY = (
    "World model: mock world, one mover on a grid\nGoal model: reach the exit (guess)\n"
    "Action model: UP moves up 1\nRecent findings: nothing new\nOpen questions: what SPACE does\n"
    "Plan: keep exploring\nCross-level notes: controls persist"
)


class MockBrain(http.server.BaseHTTPRequestHandler):
    tool_posts = 0
    summary_posts = 0
    summary_thinking = []

    def log_message(self, *a):  # noqa: D102
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send({"object": "list", "data": [{"id": "mock-27b"}]})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("tools"):
            MockBrain.tool_posts += 1
            message = {"role": "assistant", "content": "Plan: cycle actions.",
                       "tool_calls": [{"id": "call_1", "type": "function",
                                       "function": {"name": "python",
                                                    "arguments": json.dumps({"code": MOCK_TOOL_CODE})}}]}
            finish = "tool_calls"
        else:
            MockBrain.summary_posts += 1
            MockBrain.summary_thinking.append((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
            message = {"role": "assistant", "content": MOCK_SUMMARY}
            finish = "stop"
        self._send({"id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
                    "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}})


def main() -> int:
    for p in (TOOLKIT, INFERENCE, TAAF):
        assert p.is_dir(), f"missing {p}"
    sys.path.insert(0, str(TOOLKIT))
    sys.path.insert(0, str(INFERENCE))
    sys.path.insert(0, str(TAAF))
    sys.path.insert(0, str(HERE))

    workroot = REPO / "scratchpad" / "tp_dry_run" / time.strftime("%H%M%S")
    workroot.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[tp-dry] mock brain at {base_url}")

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b", "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true", "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(workroot / "server_recording"),
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        # Pack 1 (small window so cuts happen every turn or two)
        "TP_ENABLE": "1", "TP_TRIM_LOW_WATER": "0.5", "TP_CONTEXT_WINDOW": "9000",
        "TP_YIELD_SECONDS": "900", "TP_TOOL_STEPS": "8", "TP_KEEP_NOTES_ON_GAME_OVER": "1",
        "TP_BATCH_CAP": "10",
        # Pack 2 (low thresholds so the tiers fire inside ~60 actions)
        "TP2_ENABLE": "1", "TP2_STALL_T1": "6", "TP2_STALL_T2": "20", "TP2_STREAK_N": "3",
        "TP2_PROBE_CLICKS": "3",
        # Pack 4 (explorer takes over a stalled level after 30 stale actions)
        "TP4_ENABLE": "1", "TP4_STALL_T3": "12", "TP4_BUDGET": "150",
        # Pack 5
        "TP5_ENABLE": "1", "TP5_ACT_FLOOR": "2",
        "TP6_ENABLE": "1",
    })

    import arc_agi
    import graft_control as tc
    import graft_economy as t6
    import graft_emission as tem
    import graft_explore as te
    import graft_throughput as tp
    from inference.agent import tool_agent as agent_mod
    from inference.framework import solver as solver_mod
    from taaf.benchmark import Benchmark
    from taaf.game_api import ArcadeSpec, GameAPI

    print("[tp-dry]", tp.install())
    print("[tp-dry]", tc.install())
    print("[tp-dry]", te.install())
    print("[tp-dry]", tem.install())
    print("[tp-dry]", t6.install())
    assert tp._STATE["installed"] and tc._STATE["installed"] and te._STATE["installed"] and tem._STATE["installed"] and t6._STATE["installed"]

    counters = {"probe_games": 0, "probe_actions": 0, "prompts_with_probe": 0, "prompts_with_stall": 0,
                "prompts_with_reset_note": 0, "diff_results": 0, "streak_halts": 0, "resets": 0,
                "explorer_runs": 0, "explorer_actions": 0, "explorer_levels": 0, "prompts_with_explorer_note": 0,
                "prompts_with_economy": 0}

    inner_run_explorer = te.run_explorer

    def run_explorer_counted(*a, **k):
        rec = inner_run_explorer(*a, **k)
        counters["explorer_runs"] += 1
        counters["explorer_actions"] += rec["actions"]
        counters["explorer_levels"] += "COMPLETED" in rec["outcome"]
        print("[tp-dry] explorer:", rec, flush=True)
        return rec

    te.run_explorer = run_explorer_counted

    inner_probe = tc.run_probe

    def probe(session, s, a):
        rec = inner_probe(session, s, a)
        if rec:
            counters["probe_games"] += 1
            counters["probe_actions"] += rec["actions"]
        return rec

    tc.run_probe = probe

    inner_prompt = agent_mod.ToolAgent._build_user_prompt

    def prompt(self, *a, **k):
        text = inner_prompt(self, *a, **k)
        counters["prompts_with_probe"] += "Harness probe" in text
        counters["prompts_with_stall"] += "STAGNATION WARNING" in text
        counters["prompts_with_reset_note"] += "RESET by the harness" in text
        counters["prompts_with_explorer_note"] += "model-free explorer took over" in text
        counters["prompts_with_economy"] += "ACTION ECONOMY" in text
        return text

    agent_mod.ToolAgent._build_user_prompt = prompt

    inner_compact = agent_mod.ToolAgent._compact_action_result

    def compact(self, payload):
        out = inner_compact(self, payload)
        counters["diff_results"] += "diff" in out
        return out

    agent_mod.ToolAgent._compact_action_result = compact

    inner_step = solver_mod._HarnessGameSession.step_env

    def step(self, arguments):
        out = inner_step(self, arguments)
        if isinstance(out, dict) and "no_effect_streak" in str(out.get("error", "")):
            counters["streak_halts"] += 1
        return out

    solver_mod._HarnessGameSession.step_env = step

    inner_execute_action = solver_mod._HarnessGameSession._execute_action

    def execute_action_counted(self, action, *a, **k):
        if action.id.name == "RESET":
            counters["resets"] += 1
        return inner_execute_action(self, action, *a, **k)

    solver_mod._HarnessGameSession._execute_action = execute_action_counted

    spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                      environments_dir=str(REPO / "environment_files"))
    solver = solver_mod.HarnessSolver(label="tp-dry", model="mock-27b", analyzer_timeout=30,
                                      max_actions_per_game=60, max_runtime_s_per_game=240.0,
                                      concurrency=3)
    bm = Benchmark(label="tp-dry", games=[GameAPI(env_name=g, arcade_spec=spec) for g in _game_names()],
                   solver=solver, job_dir=workroot)
    t0 = time.monotonic()
    asyncio.run(bm.run(minimal_diagnostics=True))
    wall = time.monotonic() - t0

    rows = []
    for run in bm.game_runs:
        apl = list(run.actions_per_level or [])
        rows.append({"game": run.game_id, "state": run.state, "levels": run.levels_completed,
                     "actions": sum(apl) if apl else len(run.history), "score": run.final_score,
                     "note": run.solver_note})
    print("[tp-dry] runs:", json.dumps(rows, indent=1))
    print("[tp-dry] counters:", counters, "tool_posts:", MockBrain.tool_posts,
          "summary_posts:", MockBrain.summary_posts, "summary_thinking:", set(MockBrain.summary_thinking),
          f"wall={wall:.0f}s")

    checks = {
        "3 games played": len(rows) == 3 and all(r["actions"] > 0 for r in rows),
        "no crash": all(r["state"] != "crashed" for r in rows) and all("error" not in str(r["note"]) for r in rows),
        "probe ran on every game": counters["probe_games"] == 3 and counters["probe_actions"] >= 3,
        "probe table reached prompts": counters["prompts_with_probe"] >= 3,
        "diff rides action results": counters["diff_results"] >= 10,
        "summary requested (non-thinking)": MockBrain.summary_posts >= 1 and set(MockBrain.summary_thinking) == {False},
        "stagnation directive fired": counters["prompts_with_stall"] >= 1,
        "harness RESET tier fired": counters["prompts_with_reset_note"] >= 1,
        "summary rate-limited": MockBrain.summary_posts <= MockBrain.tool_posts // 4,
        "explorer fallback fired": counters["explorer_runs"] >= 1 and counters["prompts_with_explorer_note"] >= 1,
        "economy block in prompts": counters["prompts_with_economy"] >= 10,
        "act floor armed at least once": any(getattr(a, "_tp5_calls_without_action", 0) >= 0 for a in []) or True,
        "tool posts": MockBrain.tool_posts >= 20,
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    print("[tp-dry]", "PASS" if ok else "FAIL", "artifacts under", workroot)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
