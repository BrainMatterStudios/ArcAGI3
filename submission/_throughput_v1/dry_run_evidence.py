#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the evidence-integrity aid (graft_evidence)
and the hypothesis rule (graft_hypo) on the REAL engine (arcengine over
environment_files) with the june_stock agent bytes + the sha-pinned taaf
framework, driven by a loopback mock brain (dry_run_retry.py pattern).

Games: sb26 (the mock REPLAYS the recorded win trace
docs/test-artifacts-2026-08-02/sb26_win_trace.json in 4-action batches, so
level clears land INSIDE a batch and the LEVEL CLEARED split is exercised on
real frames), vc33 and ls20 (cycling 3-action batches, no clear expected).
The mock knows the game from the request's `model` field (one ToolAgent per
game, model id "mock-<game_id>", built by an analyzer_factory).

Proves: the "[EVID]" block rides the tool result the model sees and lands in
the transcript under [TOOL RESULT: python] (marker count == graft counter ==
executed-action tool results); LEVEL CLEARED flags == real level clears that
happened inside a batch; per-action TRACE on batched calls; the "[HYPO]"
block rides every analyzer prompt exactly once (count == turn headers on
uncleared levels == graft counter); games end normally; no crash.

Run:  .venv/bin/python submission/_throughput_v1/dry_run_evidence.py [--graft evid|hypo|both]
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TOOLKIT = REPO / "reference/arc-agi-toolkit"
JUNE_STOCK = REPO / "scratchpad/bundles/june_stock/src/ARC3-Inference"
TAAF_PINNED = REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"
SB26_TRACE = REPO / "docs/test-artifacts-2026-08-02/sb26_win_trace.json"
GAME_IDS = ("sb26", "vc33", "ls20")
BATCH = 4


def _game_names() -> list[str]:
    out = []
    for gid in GAME_IDS:
        versions = sorted(p.name for p in (REPO / "environment_files" / gid).iterdir() if p.is_dir())
        assert len(versions) == 1, (gid, versions)
        out.append(f"{gid}-{versions[0]}")
    return out


def _sb26_actions() -> list:
    """Engine trace (ACTION6 x/y, ACTION5) -> model actions (MOUSE row/col, SPACE)."""
    out = []
    for rec in json.loads(SB26_TRACE.read_text()):
        name = rec["name"]
        if name == "ACTION6":
            out.append({"action": "MOUSE", "row": int(rec["y"]), "col": int(rec["x"])})
        elif name == "ACTION5":
            out.append("SPACE")
        else:
            out.append({"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}[name])
    return out


REPLAY_CODE = (
    "TRACE = __TRACE__\n"
    "i = current_frame.step\n"
    "batch = TRACE[i:i + __BATCH__]\n"
    "if batch:\n"
    "    # two action() calls in one snippet: the harness stops a single call at a level clear,\n"
    "    # so frames AFTER a clear inside one tool result only arise from a second call\n"
    "    r = action(batch[:2])\n"
    "    if batch[2:]:\n"
    "        r = action(batch[2:])\n"
    "    print('replay', i, r.get('executed_count'), 'lvl', r.get('level'), 'cleared', r.get('level_completed'))\n"
    "else:\n"
    "    r = action(['SPACE'])\n"
    "    print('trace exhausted')\n"
)
CYCLE_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')]\n"
    "step = current_frame.step\n"
    "if prefs:\n"
    "    r = action([prefs[(step + k) % len(prefs)] for k in range(3)])\n"
    "elif 'MOUSE' in [str(v).upper() for v in valid_actions]:\n"
    "    r = action([{'action': 'MOUSE', 'row': (step * 7) % 64, 'col': (step * 11) % 64}])\n"
    "else:\n"
    "    r = action([valid_actions[0]])\n"
    "print('cycle', step, r.get('executed_count'), r.get('board_changed'))\n"
)


class MockBrain(http.server.BaseHTTPRequestHandler):
    tool_posts = 0
    replay_code = ""

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
        self._send({"object": "list", "data": [{"id": "mock"}]})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        MockBrain.tool_posts += 1
        model = str(body.get("model") or "")
        code = MockBrain.replay_code if "sb26" in model else CYCLE_CODE
        message = {"role": "assistant", "content": "Plan: run the batch.",
                   "tool_calls": [{"id": "call_1", "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}
        self._send({"id": "cmpl-mock", "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graft", choices=("evid", "hypo", "both"), default="both")
    args = p.parse_args()
    want_evid = args.graft in ("evid", "both")
    want_hypo = args.graft in ("hypo", "both")

    for path in (TOOLKIT, JUNE_STOCK, TAAF_PINNED, SB26_TRACE):
        assert path.exists(), f"missing {path}"
    sys.path.insert(0, str(TOOLKIT))
    sys.path.insert(0, str(JUNE_STOCK))
    sys.path.insert(0, str(TAAF_PINNED))
    sys.path.insert(0, str(HERE))

    workroot = REPO / "scratchpad" / "tp_dry_run" / (f"evid-{args.graft}-" + time.strftime("%H%M%S"))
    workroot.mkdir(parents=True, exist_ok=True)
    MockBrain.replay_code = REPLAY_CODE.replace("__TRACE__", json.dumps(_sb26_actions())).replace("__BATCH__", str(BATCH))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[evid-dry] mock brain at {base_url}; grafts: evid={want_evid} hypo={want_hypo}")

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock", "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true", "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(workroot / "server_recording"),
        "LOCAL_ANALYZER_YIELD_SECONDS": "60", "LOCAL_ANALYZER_TOOL_STEPS": "8",
        "EVID_ENABLE": "1" if want_evid else "0", "HYPO_ENABLE": "1" if want_hypo else "0",
    })

    import arc_agi  # noqa: PLC0415
    import graft_evidence as ev  # noqa: PLC0415
    import graft_hypo as hy  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415
    from taaf.benchmark import Benchmark  # noqa: PLC0415
    from taaf.game_api import ArcadeSpec, GameAPI  # noqa: PLC0415

    assert Path(agent_mod.__file__).resolve().is_relative_to(JUNE_STOCK.resolve()), agent_mod.__file__
    statuses = {}
    if want_evid:
        statuses["graft_evidence"] = ev.install()
        assert ev._STATE["installed"], statuses
    if want_hypo:
        statuses["graft_hypo"] = hy.install()
        assert hy._STATE["installed"], statuses
    print("[evid-dry] installed:", statuses)

    counters = {"prompts": 0, "prompts_with_hypo": 0, "tool_results_with_action": 0, "tool_results_with_evid": 0}
    inner_prompt = agent_mod.ToolAgent._build_user_prompt

    def prompt(self, *a, **k):
        text = inner_prompt(self, *a, **k)
        counters["prompts"] += 1
        counters["prompts_with_hypo"] += "[HYPO]" in text
        return text

    agent_mod.ToolAgent._build_user_prompt = prompt
    inner_dispatch = agent_mod.ToolAgent._dispatch_tool

    def dispatch(self, *a, **k):
        res = inner_dispatch(self, *a, **k)
        if res.step_executed:
            counters["tool_results_with_action"] += 1
            counters["tool_results_with_evid"] += "[EVID]" in res.content
        return res

    agent_mod.ToolAgent._dispatch_tool = dispatch

    spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
    solver = solver_mod.HarnessSolver(label="evid-dry", model="mock", analyzer_timeout=30,
                                      max_actions_per_game=60, max_runtime_s_per_game=240.0, concurrency=3)

    def factory(game, index):
        gid = getattr(getattr(game, "game_run", None), "game_id", None) or f"index-{index}"
        return agent_mod.ToolAgent(model=f"mock-{gid}", timeout=30, save_request_logs=False)

    solver.analyzer_factory = factory
    games = [GameAPI(env_name=g, arcade_spec=spec) for g in _game_names()]
    bm = Benchmark(label="evid-dry", games=games, solver=solver, job_dir=workroot)
    t0 = time.monotonic()
    asyncio.run(bm.run(minimal_diagnostics=True))
    wall = time.monotonic() - t0

    rows = []
    for run in bm.game_runs:
        apl = list(run.actions_per_level or [])
        rows.append({"game": run.game_id, "state": run.state, "levels": run.levels_completed,
                     "actions": len(run.history), "actions_per_level": apl, "note": run.solver_note})
    print("[evid-dry] runs:", json.dumps(rows, indent=1))
    est = ev.status()
    hst = hy.status()
    print("[evid-dry] evidence status:", json.dumps({k: est[k] for k in ("diffs_emitted", "level_flags", "chars_added",
                                                                            "traces_emitted", "errors", "skips", "per_game")}, indent=1))
    print("[evid-dry] hypo status:", json.dumps({k: hst[k] for k in ("blocks_injected", "errors", "skips", "per_game")}, indent=1))

    transcripts = sorted((workroot / "transcripts").glob("*.txt"))
    evid_lines = hypo_lines = turn_headers = level_flag_lines = tool_result_sections = evid_in_tool_result = 0
    sample = ""
    per_game_flags: dict[str, int] = {}
    for path in transcripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        evid_lines += sum(1 for ln in text.splitlines() if ln.startswith("[EVID] harness object diff"))
        hypo_lines += sum(1 for ln in text.splitlines() if ln.startswith("[HYPO] Hypothesis discipline"))
        turn_headers += len(re.findall(r"^--- analysis_step=\d+ \| action=\d+ \| ", text, re.M))
        flags = sum(1 for ln in text.splitlines() if ln.startswith("LEVEL CLEARED after action"))
        level_flag_lines += flags
        per_game_flags[path.stem] = flags
        # a tool-result section runs until the next harness section header ([EVID]/[HYPO] are not headers)
        sections = re.findall(r"\[TOOL RESULT: python\]\n(.*?)(?=\n\n\[(?!EVID|HYPO)[A-Z ]+[:\]])", text, re.S)
        tool_result_sections += len(sections)
        evid_in_tool_result += sum(1 for s in sections if "[EVID] harness object diff" in s)
        if not sample and "sb26" in path.name:
            hits = [s for s in sections if "LEVEL CLEARED" in s]
            sample = hits[0] if hits else (sections[0] if sections else "")
    print("[evid-dry] counters:", counters,
          f"transcripts={len(transcripts)} [EVID]={evid_lines} [HYPO]={hypo_lines} turns={turn_headers} "
          f"level_flags={level_flag_lines} {per_game_flags} tool_results={tool_result_sections} "
          f"tool_posts={MockBrain.tool_posts} wall={wall:.0f}s")
    if sample:
        (workroot / "sample_evid_block.txt").write_text(sample)
        print("[evid-dry] sample tool result (sb26, with LEVEL CLEARED):\n" + sample)

    sb26 = next((r for r in rows if r["game"].startswith("sb26")), None)
    checks = {
        "3 games played": len(rows) == 3 and all(r["actions"] > 0 for r in rows),
        "no crash": all(r["state"] != "crashed" for r in rows) and all("error" not in str(r["note"]) for r in rows),
        "sb26 replay cleared >= 2 levels": bool(sb26) and sb26["levels"] >= 2,
    }
    if want_evid:
        checks.update({
            "[EVID] on every executed-action tool result": counters["tool_results_with_evid"] == counters["tool_results_with_action"] > 0,
            "[EVID] transcript lines == graft diffs_emitted": evid_lines == est["diffs_emitted"] == counters["tool_results_with_evid"],
            "LEVEL CLEARED flags == sb26 clears inside batches": level_flag_lines == est["level_flags"] >= 2
                                                                  and per_game_flags.get(next((p.stem for p in transcripts if "sb26" in p.name), ""), 0) == level_flag_lines,
            "traces on batched calls": est["traces_emitted"] > 0,
            "no graft errors": est["errors"] == 0,
            "block within cap": all(len(s) <= 4200 for s in [sample]) and est["chars_added"] > 0,
        })
    else:
        checks["no [EVID] when disabled"] = evid_lines == 0 and est["diffs_emitted"] == 0
    if want_hypo:
        checks.update({
            "[HYPO] once per analyzer turn": hypo_lines == hst["blocks_injected"] == counters["prompts_with_hypo"] == counters["prompts"] == turn_headers,
            "no hypo errors": hst["errors"] == 0,
        })
    else:
        checks["no [HYPO] when disabled"] = hypo_lines == 0 and hst["blocks_injected"] == 0
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    print("[evid-dry]", "PASS" if ok else "FAIL", "artifacts under", workroot)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
