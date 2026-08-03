#!/usr/bin/env python3
"""Reproduce duck-patched v5's EXACT hook-cell path locally.

Differences from _ab_wmr/dry_run.py (which passed pre-submission):
  * FULL apply_all() with notebook semantics: the patch source is
    exec'd WITHOUT __file__, exactly like a notebook cell, so patch6 must FAIL
    with the same NameError as the commit log.
  * All env kill-switches at their defaults (all three WMR patches live), plus
    grid-burner + prompts patches active (v5 shipped them; A/B excluded them).
  * TAAF_RUN_AS_SUBMISSION=1 and minimal_diagnostics=True, like a real rerun.
  * Competition-mode arcade (shared scorecard), 3 games.

Pass criterion: every game run executes actions (>0) and the benchmark
completes; also print levels + scorecard state so a zeroing shows up.
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
from types import SimpleNamespace

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
WORKDIR = Path(__file__).resolve().parent / "repro_work" / time.strftime("%H%M%S")

MOCK_REPLY = {
    "id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
    "choices": [{
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "Plan: probe with UP.",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "python",
                             "arguments": json.dumps({"code": "action('UP')"})},
            }],
        },
        "logprobs": {"content": [
            {"token": "Plan", "logprob": -0.02, "top_logprobs": []},
        ]},
    }],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
}


class MockBrain(http.server.BaseHTTPRequestHandler):
    n_posts = 0

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": "mock-27b"}]})
        else:
            self._send({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        _ = self.rfile.read(n)
        MockBrain.n_posts += 1
        self._send(MOCK_REPLY)


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[repro] mock brain at {base}", flush=True)

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base,
        "OPENAI_BASE_URL": base,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b",
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true",
        "TAAF_RUN_AS_SUBMISSION": "1",
        "TAAF_MINIMAL_DIAGNOSTICS": "1",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(WORKDIR / "server_recording"),
    })

    for p in (TAAF_ROOT / "src/ARC3-Inference", TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))

    # --- the hook cell, verbatim semantics: exec the file source WITHOUT __file__
    src = (REPO / "submission/_duck_patched/duck_patches.py").read_text()
    cellns: dict = {"__name__": "__main__"}
    exec(compile(src, "<hook-cell>", "exec"), cellns)
    results = cellns["apply_all"]()
    fails = [r for r in results if "FAIL" in r]
    print(f"[repro] patch results: {len(results)} lines, fails={fails}", flush=True)

    # sandbox liveness after the FULL patch set
    from inference.agent import python_tool_sandbox as ptx
    sbx = ptx.run_sandboxed_python(
        code="print('sandbox-alive')", timeout_seconds=20,
        initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
        action_handler=lambda actions: {"result": [], "state": {}})
    alive = "sandbox-alive" in str(sbx.get("stdout", ""))
    print(f"[repro] sandbox alive after full apply_all: {alive} ({str(sbx.get('stderr',''))[:200]})", flush=True)

    # vision path with the grid burner live (the exact call the analyzer makes)
    from inference.agent import vision_context
    frame = SimpleNamespace(grid=[[(r + c) % 10 for c in range(64)] for r in range(64)])
    try:
        url = vision_context.frame_to_png_data_url(frame)
        print(f"[repro] grid-burner render OK ({len(url)} chars)", flush=True)
    except Exception as exc:
        print(f"[repro] grid-burner render FAILED: {exc!r}", flush=True)

    # --- run 3 games through the REAL benchmark against a competition arcade ---
    import taaf.benchmark
    import taaf.competition_arcade as _ca
    import taaf.game_api
    from inference.framework import solver as duck_solver

    env_dir = str(REPO / "environment_files")
    import arc_agi
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    by_stem = {e.game_id.split("-")[0]: e.game_id for e in arcade.available_environments}
    chosen = [by_stem[g] for g in ("ft09", "tu93", "ls20") if g in by_stem]
    csrv = _ca.CompetitionArcadeServer(
        game_ids=tuple(chosen), total_runs=len(chosen), environments_dir=env_dir).start()
    try:
        clones = list(csrv.exposed_game_ids)
        games = [taaf.game_api.GameAPI(env_name=c, arcade_spec=csrv.arcade_spec) for c in clones]
        solver = duck_solver.HarnessSolver(
            label="repro-v5", model="mock-27b", analyzer_timeout=30,
            max_actions_per_game=8, max_runtime_s_per_game=90.0, concurrency=3)
        bm = taaf.benchmark.Benchmark(label="repro_v5", games=games, solver=solver, n_passes=1)
        asyncio.run(bm.run(soft_end_time=None, runtime_environment=None,
                           minimal_diagnostics=True))
        rows = []
        for gr in (getattr(bm, "game_runs", None) or []):
            hist = len(getattr(gr, "history", []) or [])
            rows.append({
                "game": getattr(gr, "game_id", None),
                "levels": int(getattr(gr, "levels_completed", 0) or 0),
                "actions": hist,
                "state": str(getattr(gr, "state", None)),
            })
        print(f"[repro] game rows: {json.dumps(rows)}", flush=True)
        print(f"[repro] mock-brain calls: {MockBrain.n_posts}", flush=True)
        zero = [r for r in rows if r["actions"] == 0]
        if zero or not rows:
            print(f"[repro] *** ZERO-ACTION RUNS: {zero} ***", flush=True)
            return 1
        print("[repro] PASS - duck executes actions with the full v5 patch set", flush=True)
        return 0
    finally:
        csrv.stop()


if __name__ == "__main__":
    raise SystemExit(main())
