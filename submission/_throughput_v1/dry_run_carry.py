#!/usr/bin/env python3
"""GPU-free end-to-end dry run of compaction-instead-of-eviction (graft_carry)
on the REAL engine (arcengine over environment_files, three public games) with
the june_stock agent bytes + the sha-pinned taaf framework, driven by a loopback
mock brain (dry_run_probe.py pattern).

The mock brain answers every analyzer call with ~2,500 chars of unique reasoning,
a short note and a python snippet that executes one valid action, so the
context window (shrunk to LOCAL_ANALYZER_CONTEXT_WINDOW=14000 -> budget 12,976
estimated tokens) fills every 2-3 turns and the trimmer must evict. It answers
every COMPACTION request (no tools; system prompt = graft_carry.COMPACT_SYSTEM)
with a summary block carrying a sentinel and the running compaction count, and
records (a) whether the dropped turns' reasoning reached the compactor, (b)
whether the previous block was handed back for merging, (c) whether later
regular requests carry the block in their system message, and (d) the stock's
own token estimate of every request (must never exceed the budget).

Proves on the real engine: compactions fire in every game (>= 1), never fail,
the block rides every later request exactly once, dropped reasoning is fed to
the compactor, the previous block is merged, the window is never exceeded,
markers == graft counters == mock counts, games end normally, no crash.

Run:  .venv/bin/python submission/_throughput_v1/dry_run_carry.py
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
CARRY_ENV = {"CARRY_ENABLE": "1", "CARRY_TARGET_FRACTION": "0.5", "CARRY_SUMMARY_CHARS": "4800", "CARRY_INPUT_CHARS": "48000",
             "CARRY_COMPACT_MAX_TOKENS": "1500", "CARRY_COMPACT_THINKING": "0", "CARRY_MIN_DROP_MSGS": "2"}
CONTEXT_WINDOW = 14000
SENTINEL = "DRYRUN-BLOCK-SENTINEL"
REASONING_TAG = "DRYRUN-REASONING"

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
NOTE = "World model: a movable block and walls. Plan: cycle the movement actions and watch the diff."
COUNTER_KEYS = ("calls_total", "calls_with_summary", "reasoning_msgs_total", "reasoning_chars_total", "prompt_tokens_total",
                "prompt_tokens_max", "prompt_over_window", "compactions", "compaction_failures", "dropped_msgs_total",
                "dropped_chars_total", "summary_chars_total", "compaction_prompt_tokens", "compaction_completion_tokens",
                "turns_total", "errors")


def _estimate(payload: dict) -> int:
    """The stock ToolAgent._estimate_request_input_tokens on the request the mock received."""
    sub = {"messages": payload.get("messages")}
    if payload.get("tools"):
        sub["tools"] = payload["tools"]
        sub["tool_choice"] = payload.get("tool_choice")
    rendered = json.dumps(sub, ensure_ascii=True, sort_keys=True, default=str)
    return max(1, (len(rendered) + 2) // 3)


class MockBrain(http.server.BaseHTTPRequestHandler):
    lock = threading.Lock()
    stats = {"posts": 0, "regular": 0, "compactions": 0, "compact_saw_reasoning": 0, "compact_saw_previous_block": 0,
             "compact_first_none_yet": 0, "regular_with_block": 0, "regular_block_dupes": 0, "max_estimate": 0,
             "regular_reasoning_msgs": 0, "compact_had_tools": 0, "compact_thinking_on": 0, "compact_max_tokens": None}
    compact_head = ""
    summary_intro = ""

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
        first = messages[0] if messages else {}
        est = _estimate(body)
        with MockBrain.lock:
            MockBrain.stats["posts"] += 1
            MockBrain.stats["max_estimate"] = max(MockBrain.stats["max_estimate"], est)
            k = MockBrain.stats["posts"]
        is_compaction = (not body.get("tools") and first.get("role") == "system"
                         and str(first.get("content", "")).startswith(MockBrain.compact_head))
        if is_compaction:
            user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
            with MockBrain.lock:
                MockBrain.stats["compactions"] += 1
                c = MockBrain.stats["compactions"]
                MockBrain.stats["compact_saw_reasoning"] += REASONING_TAG in user
                MockBrain.stats["compact_saw_previous_block"] += SENTINEL in user
                MockBrain.stats["compact_first_none_yet"] += "(none yet)" in user
                MockBrain.stats["compact_had_tools"] += bool(body.get("tools"))
                MockBrain.stats["compact_thinking_on"] += bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
                MockBrain.stats["compact_max_tokens"] = body.get("max_tokens")
            text = (f"MECHANICS VERIFIED\n- {SENTINEL} #{c}: UP/DOWN/LEFT/RIGHT move the block one cell when free.\n"
                    "HYPOTHESES REFUTED\n- SPACE did nothing on this level.\nCURRENT PLAN\n- keep cycling the movement actions.")
            self._send({"id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                        "usage": {"prompt_tokens": est, "completion_tokens": 60, "total_tokens": est + 60}})
            return
        sys_content = str(first.get("content", "")) if first.get("role") == "system" else ""
        with MockBrain.lock:
            MockBrain.stats["regular"] += 1
            MockBrain.stats["regular_with_block"] += SENTINEL in sys_content
            MockBrain.stats["regular_block_dupes"] += sys_content.count(MockBrain.summary_intro) > 1
            MockBrain.stats["regular_reasoning_msgs"] += sum(1 for m in messages if m.get("role") == "assistant" and m.get("reasoning"))
        reasoning = (f"{REASONING_TAG} call {k}: " + "I compare the segmentation before and after the last action and look for the moved object. ") * 22
        message = {"role": "assistant", "content": NOTE, "reasoning": reasoning,
                   "tool_calls": [{"id": f"call_{k}", "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": ACTION_CODE})}}]}
        self._send({"id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                    "usage": {"prompt_tokens": est, "completion_tokens": 700, "total_tokens": est + 700}})


def _read_transcripts(workroot: Path) -> dict:
    out = {"files": 0, "turns": 0, "compact_marks": 0, "compact_ok": 0, "call_marks": 0, "metas": 0, "blocks_in_transcript": 0,
           "per_file": {}, "sample": None}
    for path in sorted((workroot / "transcripts").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        out["files"] += 1
        marks = re.findall(r"^\[HARNESS CARRY\]\n\[CARRY-COMPACT\] game=\S+ turn=\d+ dropped_msgs=(\d+) input_chars=\d+ "
                           r"summary_chars=\d+ prompt_tokens=\S+ completion_tokens=\S+ e2e_s=\S+ ok=([01])", text, re.M)
        out["compact_marks"] += len(marks)
        out["compact_ok"] += sum(1 for _, ok in marks if ok == "1")
        out["per_file"][path.name] = len(marks)
        out["call_marks"] += len(re.findall(r"^\[HARNESS CARRY\]\n\[CARRY-CALL\] game=\S+ turn=\d+ req=\d+ msgs=\d+ ", text, re.M))
        out["metas"] += text.count("[MODEL RESPONSE META]")
        out["blocks_in_transcript"] += text.count("SUMMARY:\nMECHANICS VERIFIED\n- " + SENTINEL)
        out["turns"] += len(re.findall(r"^--- analysis_step=\d+ ", text, re.M))
        if out["sample"] is None and marks:
            i = text.find("[CARRY-COMPACT]")
            out["sample"] = (path.name, text[max(0, i - 60): i + 420])
    return out


def _play(label: str, workroot: Path, *, tool_steps: int, per_game_s: float, max_actions: int, game_names: list[str]) -> list[dict]:
    import arc_agi  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415
    os.environ["LOCAL_ANALYZER_TOOL_STEPS"] = str(tool_steps)
    agent_mod._LOCAL_ANALYZER_TOOL_STEPS = int(tool_steps)
    from taaf.benchmark import Benchmark  # noqa: PLC0415
    from taaf.game_api import ArcadeSpec, GameAPI  # noqa: PLC0415

    spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
    solver = solver_mod.HarnessSolver(label=label, model="mock-27b", analyzer_timeout=30,
                                      max_actions_per_game=max_actions, max_runtime_s_per_game=per_game_s, concurrency=3)
    games = [GameAPI(env_name=g, arcade_spec=spec) for g in game_names]
    bm = Benchmark(label=label, games=games, solver=solver, job_dir=workroot)
    asyncio.run(bm.run(minimal_diagnostics=True))
    rows = []
    for run in bm.game_runs:
        apl = list(run.actions_per_level or [])
        rows.append({"game": run.game_id, "state": run.state, "levels": run.levels_completed,
                     "actions": sum(apl) if apl else len(run.history), "note": run.solver_note})
    return rows


def main() -> int:
    sys.path.insert(0, str(HERE))
    import dry_run  # noqa: PLC0415 - game list only

    for p in (TOOLKIT, JUNE_STOCK, TAAF_PINNED):
        assert p.is_dir(), f"missing {p}"
    sys.path.insert(0, str(TOOLKIT))
    sys.path.insert(0, str(JUNE_STOCK))
    sys.path.insert(0, str(TAAF_PINNED))

    stamp = time.strftime("%H%M%S")
    work = REPO / "scratchpad" / "tp_dry_run" / f"carry-{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[carry-dry] mock brain at {base_url}")

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b", "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true", "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(work / "server_recording"),
        "LOCAL_ANALYZER_YIELD_SECONDS": "900",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str(CONTEXT_WINDOW),
        "LOCAL_ANALYZER_MAX_OUTPUT": "0",
        "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "4",
        **CARRY_ENV,
    })

    import graft_carry as cr  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

    assert Path(agent_mod.__file__).resolve().is_relative_to(JUNE_STOCK.resolve()), agent_mod.__file__
    assert agent_mod._LOCAL_ANALYZER_CONTEXT_WINDOW == CONTEXT_WINDOW
    MockBrain.compact_head = cr.COMPACT_SYSTEM_HEAD
    MockBrain.summary_intro = cr.SUMMARY_INTRO
    print("[carry-dry]", cr.install())
    assert cr._STATE["installed"]
    games = dry_run._game_names()
    n = len(games)
    budget = CONTEXT_WINDOW - 512 - 512

    t0 = time.monotonic()
    rows = _play("carry-dry", work, tool_steps=4, per_game_s=240.0, max_actions=30, game_names=games)
    wall = time.monotonic() - t0
    st = cr.status()
    mock = dict(MockBrain.stats)
    tr = _read_transcripts(work)
    print("[carry-dry] runs:", json.dumps(rows))
    print("[carry-dry] status:", json.dumps({k: st[k] for k in COUNTER_KEYS}), "skips", st["skips"])
    print("[carry-dry] mock:", mock, f"wall={wall:.0f}s")
    print("[carry-dry] transcripts:", {k: v for k, v in tr.items() if k != "sample"})
    if tr["sample"]:
        print(f"[carry-dry] sample ({tr['sample'][0]}):\n" + "\n".join("    | " + ln for ln in tr["sample"][1].splitlines()))

    checks = {
        "3 games played, actions executed": len(rows) == n and all(r["actions"] > 0 for r in rows),
        "no crash": all(r["state"] != "crashed" for r in rows) and all("error" not in str(r["note"]) for r in rows),
        "compaction fired in every game (>= 1), none failed": (
            st["compactions"] >= n and all(v["compactions"] >= 1 for v in st["per_game"].values()) and st["compaction_failures"] == 0),
        "markers == graft counters == mock compactions": (
            tr["compact_marks"] == tr["compact_ok"] == st["compactions"] == mock["compactions"] == tr["blocks_in_transcript"]),
        "one [CARRY-CALL] per model call": tr["call_marks"] == st["calls_total"] == tr["metas"] == mock["regular"],
        "compaction request: no tools, thinking off, max_tokens 1500": (
            mock["compact_had_tools"] == 0 and mock["compact_thinking_on"] == 0 and mock["compact_max_tokens"] == 1500),
        "dropped turns' reasoning reached the compactor every time": mock["compact_saw_reasoning"] == mock["compactions"],
        "first compaction per game had no previous block; later ones merged it": (
            mock["compact_first_none_yet"] == n and mock["compact_saw_previous_block"] == mock["compactions"] - n),
        "the block rides later requests exactly once (no duplicate append)": (
            mock["regular_with_block"] >= st["compactions"] and mock["regular_block_dupes"] == 0
            and st["calls_with_summary"] == mock["regular_with_block"]),
        "window never exceeded (stock estimate of every request <= budget)": mock["max_estimate"] <= budget,
        "reasoning is carried by the stock (assistant messages with reasoning in requests)": (
            mock["regular_reasoning_msgs"] > 0 and st["reasoning_msgs_total"] == mock["regular_reasoning_msgs"]),
        "no graft errors": st["errors"] == 0,
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    print("[carry-dry]", "PASS" if ok else "FAIL", "artifacts under", work)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
