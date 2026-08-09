#!/usr/bin/env python3
"""GPU-free 28-clone dry run of patches 21+22 (structural plan channel).

Mirrors the package screen's dry-run rig (mock brain <- REAL duck harness on
the scored-ref bundle <- REAL CompetitionArcadeServer at the full 28-clone
geometry) with one twist: the mock brain is PROMPT-SENSITIVE -- it reads each
request's system prompt and obeys whatever action contract it states. When the
PLAN CONTRACT is present it submits 5-action plans; otherwise it acts one
action per turn, exactly like the package-screen 27B did.

Headline metric: realized actions-per-turn (scored actions / analyzer calls),
struct arm vs baseline arm. Bar: > 3 with TAAF_STRUCT=1, ~1 baseline.

Measured 2026-08-09 (commit 4bb431d bytes): struct 4.65 plan-actions/turn
(131/131 turns emitted 5-action plans; PLAN REPORT in 103/131 prompts =
every turn after the first; PHASE line 131/131; contract in the tool
schema 131/131; 42 soft brake flags; 12 phase transitions; 28 wiggle
batteries) vs base 1.0 model-actions/turn (0 plans, delta-pure).

Run:  .venv/bin/python submission/_duck_patched/dry_run_struct.py
"""
from __future__ import annotations

import asyncio
import http.server
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
TOOLKIT = REPO / "reference/arc-agi-toolkit"
PATCHES = REPO / "submission/_duck_patched/duck_patches.py"
WORKROOT = REPO / "scratchpad" / "struct_dry_run" / time.strftime("%H%M%S")

GEOMETRY = {"clones": 28, "per_game_s": 90, "concurrency": 28}
MAX_ACTIONS = 25

SINGLE_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "if prefs:\n"
    "    action(prefs[0])\n"
    "elif valid_actions:\n"
    "    action({'action': 'MOUSE', 'row': 32, 'col': 32})\n"
    "else:\n"
    "    action('UP')\n"
)

PLAN_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "if prefs:\n"
    "    plan = (list(prefs) * 5)[:5]\n"
    "elif valid_actions:\n"
    "    plan = [{'action': 'MOUSE', 'row': r, 'col': c}\n"
    "            for r, c in [(20, 20), (32, 32), (44, 44), (20, 44), (44, 20)]]\n"
    "else:\n"
    "    plan = ['UP']\n"
    "action(plan)\n"
)

BARE_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "if prefs:\n"
    "    action(prefs[0])\n"
    "elif valid_actions:\n"
    "    action({'action': 'MOUSE', 'row': 32, 'col': 32})\n"
    "else:\n"
    "    action('UP')\n"
)

STATS = {
    "posts": 0,
    "plan_turns": 0,
    "bare_turns": 0,
    "single_turns": 0,
    "plan_report_seen": 0,
    "phase_seen": 0,
    "brake_seen": 0,
    "contract_in_schema": 0,
    "example_seen": 0,
    "coach_seen": 0,
}


def mock_reply(body: bytes):
    try:
        payload = json.loads(body.decode() or "{}")
    except Exception:
        payload = {}
    messages = payload.get("messages") or []
    system_text = str((messages[0] or {}).get("content", "")) if messages else ""
    last_user = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            last_user = content if isinstance(content, str) else json.dumps(content)
            break
    tools_text = json.dumps(payload.get("tools") or [])
    if "ordered plan of 1-20" in tools_text:
        STATS["contract_in_schema"] += 1
    if "PLAN REPORT" in last_user:
        STATS["plan_report_seen"] += 1
    if "PHASE:" in last_user:
        STATS["phase_seen"] += 1
    if "BRAKE" in last_user:
        STATS["brake_seen"] += 1
    if "EXAMPLE — your wiggle battery" in last_user:
        STATS["example_seen"] += 1
    if "Batch 2-20 actions" in last_user or "COMMIT phase: proven" in last_user:
        STATS["coach_seen"] += 1
    if "PLAN CONTRACT" in system_text:
        # scripted 27B stand-in: its FIRST acting turn is a bare single (the
        # observed dominant failure mode); coached turns batch 5-action plans
        if "PLAN REPORT" in last_user:
            STATS["plan_turns"] += 1
            code = PLAN_CODE
        else:
            STATS["bare_turns"] += 1
            code = BARE_CODE
    else:
        STATS["single_turns"] += 1
        code = SINGLE_CODE
    return {
        "id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "Plan: probe per the contract the prompt states.",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "python",
                                 "arguments": json.dumps({"code": code})},
                }],
            },
        }],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 30,
                  "total_tokens": 1230},
    }


class MockBrain(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": "mock-27b"}]})
        else:
            self._send({})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n)
        STATS["posts"] += 1
        self._send(mock_reply(body))


def setup():
    WORKROOT.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[struct-dry] mock brain at {base_url}", flush=True)

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b",
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true",
        "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(WORKROOT / "server_recording"),
    })

    sys.path.insert(0, str(TOOLKIT))
    for p in (TAAF_ROOT / "src/ARC3-Inference",
              TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))

    # load duck_patches exactly as the kernel does: source exec, no __file__
    source = PATCHES.read_text()
    mod = types.ModuleType("duck_patches")
    mod.__dict__["__name__"] = "duck_patches"
    exec(compile(source, "<inlined duck_patches>", "exec"), mod.__dict__)
    sys.modules["duck_patches"] = mod
    results = mod.apply_all()
    bad = [l for l in results if "FAIL" in l]
    assert not bad, f"patch layer failed: {bad}"
    return mod


async def run_arm(dp, arm: str, arm_env: dict) -> dict:
    import arc_agi
    import taaf.benchmark
    import taaf.game_api
    import taaf.competition_arcade as _ca
    from inference.framework import solver as duck_solver

    os.environ.update(arm_env)
    for key in STATS:
        STATS[key] = 0
    diag_before = json.loads(json.dumps(dp.STRUCT_DIAGNOSTICS))
    wiggle_before = dict(dp.WIGGLE_DIAGNOSTICS)

    env_dir = str(REPO / "environment_files")
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=env_dir)
    official = sorted(e.game_id for e in arcade.available_environments)
    assert len(official) == 25, official
    srv = _ca.CompetitionArcadeServer(
        game_ids=tuple(official), total_runs=GEOMETRY["clones"],
        environments_dir=env_dir).start()
    try:
        clones = list(srv.exposed_game_ids)
        assert len(clones) == GEOMETRY["clones"], len(clones)
        games = [taaf.game_api.GameAPI(env_name=c, arcade_spec=srv.arcade_spec)
                 for c in clones]
        solver = duck_solver.HarnessSolver(
            label=f"struct-dry-{arm}", model="mock-27b", analyzer_timeout=30,
            max_actions_per_game=MAX_ACTIONS,
            max_runtime_s_per_game=float(GEOMETRY["per_game_s"]),
            concurrency=GEOMETRY["concurrency"])
        bench = taaf.benchmark.Benchmark(
            label=f"struct_{arm}", games=games, solver=solver, n_passes=1)
        t0 = time.time()
        await bench.run(soft_end_time=None, runtime_environment=None,
                        minimal_diagnostics=False)
        elapsed = time.time() - t0
    finally:
        srv.stop()

    actions_total = 0
    levels = 0
    for gr in (getattr(bench, "game_runs", None) or []):
        actions_total += len(getattr(gr, "history", []) or [])
        levels += int(getattr(gr, "levels_completed", 0) or 0)
    diag_after = json.loads(json.dumps(dp.STRUCT_DIAGNOSTICS))
    delta = {}
    for key, val in diag_after.items():
        if isinstance(val, dict):
            delta[key] = {k: v - diag_before.get(key, {}).get(k, 0)
                          for k, v in val.items()
                          if v - diag_before.get(key, {}).get(k, 0)}
        else:
            delta[key] = val - diag_before.get(key, 0)
    wiggle_delta = {k: v - wiggle_before.get(k, 0)
                    for k, v in dp.WIGGLE_DIAGNOSTICS.items()}
    posts = STATS["posts"]
    row = {
        "arm": arm,
        "arm_env": arm_env,
        "elapsed_s": round(elapsed, 1),
        "clones": len(games),
        "actions_total": actions_total,
        "levels_completed": levels,
        "analyzer_posts": posts,
        "mock": dict(STATS),
        "struct_diag_delta": delta,
        "wiggle_delta": wiggle_delta,
        "actions_per_turn_raw": round(actions_total / posts, 2) if posts else None,
        "plan_actions_per_turn": (
            round(delta["plan_actions"] / posts, 2) if posts else None),
    }
    print(f"[struct-dry] arm={arm} done in {elapsed:.0f}s: "
          f"actions={actions_total} posts={posts} "
          f"raw a/t={row['actions_per_turn_raw']} "
          f"plan a/t={row['plan_actions_per_turn']} "
          f"plans={delta.get('plans')} lengths={delta.get('plan_lengths')}",
          flush=True)
    return row


def main():
    dp = setup()
    base_env = {"TAAF_STRUCT": "0", "TAAF_DIFF_LINES": "0", "TAAF_WIGGLE": "0",
                "TAAF_RUN_PROBE": "0", "TAAF_DISPATCH": "0", "TAAF_VERIFY": "0"}
    struct_env = {"TAAF_STRUCT": "1", "TAAF_DIFF_LINES": "1", "TAAF_WIGGLE": "1",
                  "TAAF_RUN_PROBE": "1", "TAAF_DISPATCH": "1", "TAAF_VERIFY": "0"}
    rows = []
    rows.append(asyncio.run(run_arm(dp, "base", base_env)))
    rows.append(asyncio.run(run_arm(dp, "struct", struct_env)))
    out = WORKROOT / "struct_dry_result.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"[struct-dry] wrote {out}")

    base, struct = rows
    # -- hard checks (assert, don't hope) ------------------------------------
    assert base["struct_diag_delta"]["plans"] == 0, base["struct_diag_delta"]
    assert base["mock"]["plan_turns"] == 0, base["mock"]
    # Raw history/posts is inflated by LLM-free actions (auto-RESET on
    # game_over, wiggle presses); the model-attributable baseline is exactly
    # 1 action per turn: the mock emitted one single-action call per post.
    assert base["mock"]["single_turns"] == base["analyzer_posts"], base["mock"]
    base["model_actions_per_turn"] = 1.0
    sd = struct["struct_diag_delta"]
    assert sd["plans"] >= 28, sd  # every clone submitted plans
    assert struct["mock"]["plan_turns"] > 0 and struct["mock"]["single_turns"] == 0
    assert struct["mock"]["contract_in_schema"] > 0, "tool schema lost the contract"
    assert struct["mock"]["plan_report_seen"] > 0, "PLAN REPORT never reached a prompt"
    assert struct["mock"]["phase_seen"] > 0, "PHASE line never reached a prompt"
    assert sd["reports_injected"] > 0
    # 2026-08-09 adoption levers, observed end-to-end at 28-clone geometry
    assert sd["bare_singles"] >= 20, sd  # the scripted first-turn singles
    assert sd["examples_shown"] >= 1, sd  # lever 1 seeded from real batteries
    assert struct["mock"]["example_seen"] >= 1, struct["mock"]
    assert sd["coach_lines"] >= 1, sd  # lever 2 fired after bare singles
    assert struct["mock"]["coach_seen"] >= 1, struct["mock"]
    assert struct["plan_actions_per_turn"] is not None
    assert struct["plan_actions_per_turn"] > 3.0, (
        f"ADOPTION BAR MISSED: {struct['plan_actions_per_turn']} plan-actions/turn")
    print("[struct-dry] ALL CHECKS PASSED -- adoption bar met: "
          f"{struct['plan_actions_per_turn']} plan-actions/turn (struct) vs "
          f"1.0 model-actions/turn (base); raw "
          f"{struct['actions_per_turn_raw']} vs {base['actions_per_turn_raw']}")


if __name__ == "__main__":
    main()
