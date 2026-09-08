#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the harness-enforced probe discipline
(graft_probe) on the REAL engine (arcengine over environment_files, three
public games) with the june_stock agent bytes + the sha-pinned taaf
framework, driven by a loopback mock brain (dry_run_retry.py pattern).

The mock brain reads its position inside the turn from the request itself
(the number of tool messages after the turn's own user prompt) and plays two
scripts per game:

  phase 1 (the game's first turn, until the harness tells it the previous
           turn executed nothing): ONLY analysis-only snippets. With
           LOCAL_ANALYZER_TOOL_STEPS=6 the turn is A A R R A A and ends
           with "No action(...) call was captured" -> a NOACT turn; the
           next prompt must carry the one-line "Previous turn executed no
           action ..." notice, which flips the mock to phase 2.
  phase 2 (every later turn): analysis, analysis, a SENTINEL analysis-only
           snippet (must be REFUSED, never executed), then an action.

Proves on the real engine: the refusal text is the tool message the model
receives (the mock counts requests that carry it) and lands in the transcript
under [TOOL RESULT: python] right after the [PROBE-REFUSE] marker; the refused
snippet is never executed (no SENTINEL in any tool result); the very next
call after every refusal executes an action (graft counter == refusals);
the refusal cap holds (2 per turn, then the stock loop proceeds); a NOACT
turn is marked once and its notice rides exactly the next prompt; games end
normally; no crash.

Run:  .venv/bin/python submission/_throughput_v1/dry_run_probe.py
"""
from __future__ import annotations

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
PROBE_ENV = {"PROBE_ENABLE": "1", "PROBE_MAX_ANALYSIS": "2", "PROBE_MAX_PROBE": "5", "PROBE_MAX_REFUSALS": "2",
             "PROBE_NOTE_LINES": "3"}
REFUSAL_HEAD = "Analysis budget for this turn is spent"
NOACT_HEAD = "Previous turn executed no action"
SENTINEL = "SENTINEL-REFUSED-SNIPPET"

ANALYSIS_CODE = "seg = current_frame.segmentation\nprint('analysis', len(history), len(seg) if seg else 0)\n"
SENTINEL_CODE = f"print('{SENTINEL}')\n"
ACTION_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')]\n"
    "step = len(history)\n"
    "if prefs:\n"
    "    r = action(prefs[step % len(prefs)])\n"
    "elif 'MOUSE' in [str(v).upper() for v in valid_actions]:\n"
    "    r = action({'action': 'MOUSE', 'row': (step * 7) % 64, 'col': (step * 11) % 64})\n"
    "else:\n"
    "    r = action(valid_actions[0])\n"
    "print('acted', r.get('executed'))\n"
)
NOTE = "Open questions: does SPACE open the door; is the edge bar a timer. Plan: test it."


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def turn_position(messages: list[dict]) -> tuple[int, str]:
    """(python calls already made in this turn, the turn's user prompt text)."""
    last_user = None
    for i, m in enumerate(messages):
        if m.get("role") == "user" and "Current state: step" in _text_of(m):
            last_user = i
    if last_user is None:
        return 0, ""
    k = sum(1 for m in messages[last_user + 1:] if m.get("role") == "tool")
    return k, _text_of(messages[last_user])


class MockBrain(http.server.BaseHTTPRequestHandler):
    lock = threading.Lock()
    stats = {"posts": 0, "refusal_seen": 0, "noact_seen": 0, "analysis": 0, "sentinel": 0, "act": 0}

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
        messages = body.get("messages") or []
        k, prompt = turn_position(messages)
        tool_texts = [str(m.get("content")) for m in messages if m.get("role") == "tool"]
        noact = NOACT_HEAD in prompt
        with MockBrain.lock:
            MockBrain.stats["posts"] += 1
            MockBrain.stats["refusal_seen"] += any(REFUSAL_HEAD in t for t in tool_texts[-1:])   # the newest tool result
            MockBrain.stats["noact_seen"] += noact and k == 0      # the notice rides the turn's prompt: count the turn once
        first_turn = "Current state: step 1," in prompt and not noact
        if first_turn:
            code, kind = ANALYSIS_CODE, "analysis"          # phase 1: never act -> NOACT turn
        elif k < 2:
            code, kind = ANALYSIS_CODE, "analysis"
        elif k == 2:
            code, kind = SENTINEL_CODE, "sentinel"          # the 3rd analysis-only call: must be refused
        else:
            code, kind = ACTION_CODE, "act"
        with MockBrain.lock:
            MockBrain.stats[kind] += 1
        message = {"role": "assistant", "content": NOTE,
                   "tool_calls": [{"id": f"call_{MockBrain.stats['posts']}", "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}
        self._send({"id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}})


def main() -> int:
    sys.path.insert(0, str(HERE))
    import dry_run  # noqa: PLC0415 - game list only

    for p in (TOOLKIT, JUNE_STOCK, TAAF_PINNED):
        assert p.is_dir(), f"missing {p}"
    sys.path.insert(0, str(TOOLKIT))
    sys.path.insert(0, str(JUNE_STOCK))
    sys.path.insert(0, str(TAAF_PINNED))

    workroot = REPO / "scratchpad" / "tp_dry_run" / ("probe-" + time.strftime("%H%M%S"))
    workroot.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[probe-dry] mock brain at {base_url}")

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b", "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true", "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(workroot / "server_recording"),
        "LOCAL_ANALYZER_YIELD_SECONDS": "900", "LOCAL_ANALYZER_TOOL_STEPS": "6",
        **PROBE_ENV,
    })

    import arc_agi  # noqa: PLC0415
    import graft_probe as pr  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415
    from taaf.benchmark import Benchmark  # noqa: PLC0415
    from taaf.game_api import ArcadeSpec, GameAPI  # noqa: PLC0415

    assert Path(agent_mod.__file__).resolve().is_relative_to(JUNE_STOCK.resolve()), agent_mod.__file__
    print("[probe-dry]", pr.install())
    assert pr._STATE["installed"]

    spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
    solver = solver_mod.HarnessSolver(label="probe-dry", model="mock-27b", analyzer_timeout=30,
                                      max_actions_per_game=24, max_runtime_s_per_game=240.0, concurrency=3)
    games = [GameAPI(env_name=g, arcade_spec=spec) for g in dry_run._game_names()]
    bm = Benchmark(label="probe-dry", games=games, solver=solver, job_dir=workroot)
    t0 = time.monotonic()
    asyncio.run(bm.run(minimal_diagnostics=True))
    wall = time.monotonic() - t0

    rows = []
    for run in bm.game_runs:
        apl = list(run.actions_per_level or [])
        rows.append({"game": run.game_id, "state": run.state, "levels": run.levels_completed,
                     "actions": sum(apl) if apl else len(run.history), "note": run.solver_note})
    print("[probe-dry] runs:", json.dumps(rows, indent=1))
    st = pr.status()
    print("[probe-dry] probe status:", json.dumps({k: st[k] for k in (
        "refusals", "turns_with_refusal", "noact_turns", "analysis_calls_total", "acting_calls_total",
        "calls_after_refusal", "acting_calls_after_refusal", "turns_total", "turns_ge3_analysis", "errors", "skips")}))
    print("[probe-dry] per game:", json.dumps(st["per_game"]))
    print("[probe-dry] mock:", MockBrain.stats, f"wall={wall:.0f}s")

    transcripts = sorted((workroot / "transcripts").glob("*.txt"))
    refuse_marks = noact_marks = refusal_results = sentinel_results = notices = turns = 0
    refusal_then_act = 0
    sample = None
    for path in transcripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        refuse_marks += len(re.findall(r"^\[HARNESS PROBE\]\n\[PROBE-REFUSE\] game=\S+ turn=\d+ analysis_calls=\d+ refusal=\d/2$", text, re.M))
        noact_marks += len(re.findall(r"^\[PROBE-NOACT\] game=\S+ turn=\d+ analysis_calls=\d+ refusals=\d+ reason=\w+$", text, re.M))
        refusal_results += text.count("[TOOL RESULT: python]\n" + REFUSAL_HEAD)
        sentinel_results += text.count("[TOOL RESULT: python]\n" + SENTINEL)
        notices += text.count("[USER PROMPT]\n" + NOACT_HEAD)
        turns += len(re.findall(r"^--- analysis_step=\d+ ", text, re.M))
        for turn_text in re.split(r"\n(?=--- analysis_step=)", text):
            i = turn_text.find("[TOOL RESULT: python]\n" + REFUSAL_HEAD)
            if i >= 0 and "[TOOL RESULT: python]\nacted True" in turn_text[i:]:
                refusal_then_act += 1
                if sample is None:
                    j = turn_text.find("[TOOL RESULT: python]\nacted True", i)
                    sample = (path.name, turn_text[max(0, i - 400): j + 40])
    print(f"[probe-dry] transcripts={len(transcripts)} turns={turns} [PROBE-REFUSE]={refuse_marks} [PROBE-NOACT]={noact_marks} "
          f"refusal results={refusal_results} sentinel results={sentinel_results} notices={notices} "
          f"refusal->act turns={refusal_then_act}")
    if sample:
        print(f"[probe-dry] sample ({sample[0]}):\n" + "\n".join("    | " + ln for ln in sample[1].splitlines()))

    n_games = len(rows)
    checks = {
        "3 games played": n_games == 3 and all(r["actions"] > 0 for r in rows),
        "no crash": all(r["state"] != "crashed" for r in rows) and all("error" not in str(r["note"]) for r in rows),
        "refusal fired in every game": st["refusals"] >= n_games and all(v["refusals"] >= 1 for v in st["per_game"].values()),
        "[PROBE-REFUSE] marker per refusal": refuse_marks == st["refusals"],
        "refusal text is the tool result the model reads": refusal_results == st["refusals"] and MockBrain.stats["refusal_seen"] == st["refusals"],
        "refused snippet never executed": sentinel_results == 0 and MockBrain.stats["sentinel"] >= n_games,
        "next call after a refusal acted (phase 2)": st["acting_calls_after_refusal"] >= n_games
            and refusal_then_act == st["acting_calls_after_refusal"],
        "refusal cap held on the NOACT turns (2 refusals, then the loop proceeded)":
            st["noact_turns"] == n_games and st["skips"].get("refusal_cap", 0) >= n_games and st["turns_ge3_analysis"] >= n_games,
        "one NOACT turn per game, notice on exactly the next prompt": noact_marks == n_games == notices == MockBrain.stats["noact_seen"],
        "graft counters == transcript": st["turns_total"] == turns and st["errors"] == 0,
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    print("[probe-dry]", "PASS" if ok else "FAIL", "artifacts under", workroot)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
