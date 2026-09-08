#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the harness-enforced probe discipline
(graft_probe) on the REAL engine (arcengine over environment_files, three
public games) with the june_stock agent bytes + the sha-pinned taaf
framework, driven by a loopback mock brain (dry_run_retry.py pattern).

Two phases, two Benchmark runs in one process (the graft's module counters
are diffed between them):

  COMPLIANT brain (tool steps 4, 240 s / 24 actions per game):
    turn 1 of a game: analysis-only snippets only -> A A R R, "No action(...)
      call was captured" -> a NOACT turn (refusals 2 of 4, both settled as
      non-acting follow-ups: R after R, then the turn ended on a refusal);
    the turn that carries the "Previous turn executed no action" notice:
      the brain acts at once (the carried span closes);
    every later turn: analysis, analysis, a SENTINEL analysis-only snippet
      (must be REFUSED, never executed), then an action.
  STUBBORN brain (tool steps 6, 15 s per game): analysis-only forever.
    turn 1 = A A R R R R (the 4-refusal cap, the last refusal ends the turn);
    every later turn on the same level CARRIES the span: the cap is already
    reached, so the six snippets run (leak_cap_lifted once per span) and the
    turn is NOACT again; no action is ever executed; the run ends on its
    runtime cap.

Proves on the real engine: the refusal text is the tool message the model
receives and lands under [TOOL RESULT: python] right after the [PROBE-REFUSE]
marker; the refused snippet is never executed; the very next call after a
refusal acts on every compliant pattern turn; the 4-refusal cap holds and
carries across NOACT turns (exactly 4 refusals per stubborn game, however
many turns it burns); NOACT turns are marked once each and their notice
rides exactly the next prompt; games end normally; no crash.

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
PROBE_ENV = {"PROBE_ENABLE": "1", "PROBE_MAX_ANALYSIS": "2", "PROBE_MAX_PROBE": "5", "PROBE_MAX_REFUSALS": "4",
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
COUNTER_KEYS = ("refusals", "turns_with_refusal", "noact_turns", "carried_turns", "analysis_calls_total",
                "acting_calls_total", "calls_after_refusal", "acting_calls_after_refusal", "first_refusal_followups",
                "acted_after_first_refusal", "refusal_turn_ending", "turns_total", "turns_ge3_analysis",
                "leak_cap_lifted", "leak_dead_branch", "leak_unparsable", "errors")


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
    stubborn = False
    stats = {"posts": 0, "refusal_seen": 0, "noact_seen": 0, "analysis": 0, "sentinel": 0, "act": 0}

    @classmethod
    def reset(cls, *, stubborn: bool) -> None:
        with cls.lock:
            cls.stubborn = stubborn
            for k in cls.stats:
                cls.stats[k] = 0

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
            stubborn = MockBrain.stubborn
        first_turn = "Current state: step 1," in prompt and not noact
        if stubborn or first_turn:
            code, kind = ANALYSIS_CODE, "analysis"          # never act
        elif noact:
            code, kind = ACTION_CODE, "act"                 # told the previous turn did nothing: act now
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


def _read_transcripts(workroot: Path) -> dict:
    out = {"files": 0, "turns": 0, "refuse_marks": 0, "noact_marks": 0, "refusal_results": 0, "sentinel_results": 0,
           "notices": 0, "refusal_then_act": 0, "cap_marks": 0, "max_noact_analysis": 0, "sample": None,
           "refusals_per_file": {}}
    for path in sorted((workroot / "transcripts").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        out["files"] += 1
        marks = re.findall(r"^\[HARNESS PROBE\]\n\[PROBE-REFUSE\] game=\S+ turn=\d+ analysis_calls=\d+ refusal=(\d)/4$", text, re.M)
        out["refuse_marks"] += len(marks)
        out["refusals_per_file"][path.name] = len(marks)
        out["cap_marks"] += marks.count("4")
        noacts = re.findall(r"^\[PROBE-NOACT\] game=\S+ turn=\d+ analysis_calls=(\d+) refusals=\d+ reason=\w+$", text, re.M)
        out["noact_marks"] += len(noacts)
        out["max_noact_analysis"] = max([out["max_noact_analysis"], *[int(x) for x in noacts]])
        out["refusal_results"] += text.count("[TOOL RESULT: python]\n" + REFUSAL_HEAD)
        out["sentinel_results"] += text.count("[TOOL RESULT: python]\n" + SENTINEL)
        out["notices"] += text.count("[USER PROMPT]\n" + NOACT_HEAD)
        out["turns"] += len(re.findall(r"^--- analysis_step=\d+ ", text, re.M))
        for turn_text in re.split(r"\n(?=--- analysis_step=)", text):
            i = turn_text.find("[TOOL RESULT: python]\n" + REFUSAL_HEAD)
            if i >= 0 and "[TOOL RESULT: python]\nacted True" in turn_text[i:]:
                out["refusal_then_act"] += 1
                if out["sample"] is None:
                    j = turn_text.find("[TOOL RESULT: python]\nacted True", i)
                    out["sample"] = (path.name, turn_text[max(0, i - 400): j + 40])
    return out


def _play(label: str, workroot: Path, *, tool_steps: int, per_game_s: float, max_actions: int, game_names: list[str]) -> list[dict]:
    import arc_agi  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415
    # the stock reads LOCAL_ANALYZER_TOOL_STEPS at import; each ToolAgent copies the module constant at
    # construction (one agent per game per Benchmark run), so set the constant per phase
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
    root = REPO / "scratchpad" / "tp_dry_run"
    work1 = root / f"probe-{stamp}-compliant"
    work2 = root / f"probe-{stamp}-stubborn"
    for w in (work1, work2):
        w.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[probe-dry] mock brain at {base_url}")

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b", "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true", "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(work1 / "server_recording"),
        "LOCAL_ANALYZER_YIELD_SECONDS": "900",
        **PROBE_ENV,
    })

    import graft_probe as pr  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

    assert Path(agent_mod.__file__).resolve().is_relative_to(JUNE_STOCK.resolve()), agent_mod.__file__
    print("[probe-dry]", pr.install())
    assert pr._STATE["installed"]
    games = dry_run._game_names()
    n = len(games)

    # ---- phase 1: compliant brain -------------------------------------------------------------
    MockBrain.reset(stubborn=False)
    t0 = time.monotonic()
    rows1 = _play("probe-dry-compliant", work1, tool_steps=4, per_game_s=240.0, max_actions=24, game_names=games)
    wall1 = time.monotonic() - t0
    st1 = pr.status()
    mock1 = dict(MockBrain.stats)
    tr1 = _read_transcripts(work1)
    print("[probe-dry] compliant runs:", json.dumps(rows1))
    print("[probe-dry] compliant status:", json.dumps({k: st1[k] for k in COUNTER_KEYS}), "skips", st1["skips"])
    print("[probe-dry] compliant mock:", mock1, f"wall={wall1:.0f}s")
    print("[probe-dry] compliant transcripts:", {k: v for k, v in tr1.items() if k != "sample"})
    if tr1["sample"]:
        print(f"[probe-dry] sample ({tr1['sample'][0]}):\n" + "\n".join("    | " + ln for ln in tr1["sample"][1].splitlines()))

    # ---- phase 2: stubborn brain ---------------------------------------------------------------
    MockBrain.reset(stubborn=True)
    os.environ["RECORDINGS_DIR"] = str(work2 / "server_recording")
    t0 = time.monotonic()
    rows2 = _play("probe-dry-stubborn", work2, tool_steps=6, per_game_s=15.0, max_actions=24, game_names=games)
    wall2 = time.monotonic() - t0
    st2 = pr.status()
    d = {k: st2[k] - st1[k] for k in COUNTER_KEYS}
    d["skips_refusal_cap"] = st2["skips"].get("refusal_cap", 0) - st1["skips"].get("refusal_cap", 0)
    mock2 = dict(MockBrain.stats)
    tr2 = _read_transcripts(work2)
    print("[probe-dry] stubborn runs:", json.dumps(rows2))
    print("[probe-dry] stubborn status delta:", json.dumps(d))
    print("[probe-dry] stubborn mock:", mock2, f"wall={wall2:.0f}s")
    print("[probe-dry] stubborn transcripts:", {k: v for k, v in tr2.items() if k != "sample"})

    checks = {
        "3 games played (compliant)": len(rows1) == n and all(r["actions"] > 0 for r in rows1),
        "no crash": all(r["state"] != "crashed" for r in rows1 + rows2) and all("error" not in str(r["note"]) for r in rows1 + rows2),
        "compliant: refusal fired in every game": st1["refusals"] >= n and all(v["refusals"] >= 1 for v in st1["per_game"].values()),
        "compliant: [PROBE-REFUSE] marker per refusal": tr1["refuse_marks"] == st1["refusals"],
        "compliant: refusal text is the tool result the model reads": tr1["refusal_results"] == st1["refusals"] == mock1["refusal_seen"],
        "compliant: refused sentinel never executed": tr1["sentinel_results"] == 0 and mock1["sentinel"] >= n,
        "compliant: next call after every pattern refusal acted": (
            tr1["refusal_then_act"] == st1["acting_calls_after_refusal"] >= n
            and st1["acting_calls_after_refusal"] == st1["acting_calls_total"] - n),
        "compliant: turn 1 = A A R R -> NOACT, refusals settled non-acting, span carried, then closed by the action": (
            st1["noact_turns"] == n == st1["carried_turns"] == st1["refusal_turn_ending"]
            # every later pattern turn is its own span: first refusal -> action; turn 1's first refusal -> refusal
            and st1["first_refusal_followups"] == n + st1["acting_calls_after_refusal"]
            and st1["acted_after_first_refusal"] == st1["acting_calls_after_refusal"]
            and st1["turns_ge3_analysis"] == 0 and st1["skips"].get("refusal_cap", 0) == 0),
        "compliant: one NOACT per game, notice on exactly the next prompt": tr1["noact_marks"] == n == tr1["notices"] == mock1["noact_seen"],
        "compliant: graft counters == transcript": st1["turns_total"] == tr1["turns"] and st1["errors"] == 0,
        "stubborn: no action ever executed": all(r["actions"] == 0 for r in rows2) and d["acting_calls_total"] == 0 and mock2["act"] == 0,
        "stubborn: exactly 4 refusals per game across all its turns (cap carried over NOACT turns)": (
            d["refusals"] == 4 * n and all(v == 4 for v in tr2["refusals_per_file"].values()) and tr2["cap_marks"] == n),
        "stubborn: the cap lifted once per game (leak_cap_lifted), carried turns burned analysis calls": (
            d["leak_cap_lifted"] == n == d["turns_ge3_analysis"] and d["carried_turns"] >= n
            and d["noact_turns"] >= 2 * n and tr2["max_noact_analysis"] >= 8 and d["skips_refusal_cap"] >= 4 * n),
        "stubborn: first-refusal follow-ups all non-acting; 4th refusal ended turn 1": (
            d["first_refusal_followups"] == n and d["acted_after_first_refusal"] == 0 and d["refusal_turn_ending"] == n
            and d["calls_after_refusal"] == 4 * n and d["acting_calls_after_refusal"] == 0),
        "stubborn: every NOACT turn but the last per game got its notice": (
            tr2["noact_marks"] == d["noact_turns"] and tr2["notices"] == mock2["noact_seen"]
            and d["noact_turns"] - n <= tr2["notices"] <= d["noact_turns"]),
        "stubborn: no errors": d["errors"] == 0,
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    print("[probe-dry]", "PASS" if ok else "FAIL", "artifacts under", work1, "and", work2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
