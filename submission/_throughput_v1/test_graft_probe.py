"""Offline validation for the harness-enforced probe discipline (PROBE).

Run:  GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference \
          .venv/bin/python submission/_throughput_v1/test_graft_probe.py
      (default bundle: the anim bundle under submission/_inspect_replay)

The python tool runs in the REAL sandbox subprocess (python_tool_sandbox), so
"executed / not executed" is the harness's own payload read, not a mock.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(os.environ.get(
    "GRAFT_TEST_BUNDLE",
    str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference"),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_probe as pr  # noqa: E402

_FLAGS = ("PROBE_ENABLE", "PROBE_MAX_ANALYSIS", "PROBE_MAX_PROBE", "PROBE_MAX_REFUSALS", "PROBE_NOTE_LINES")
_GRID = tuple(tuple((r * 7 + c) % 5 for c in range(8)) for r in range(8))

A_CODE = "n = len(history)\nprint('analysis', n)\n"                    # analysis-only: no action() anywhere
A_SENTINEL = "print('SENTINEL-MUST-NOT-RUN')\n"
X_CODE = "r = action(['UP'])\nprint('acted', r.get('executed'))\n"     # acts
BAD_CODE = "r = action(['BOGUS'])\nprint('bogus', r)\n"                 # action() that fails -> executed False
ERR_CODE = "x = 1 / 0\n"


def _clear() -> None:
    for name in _FLAGS:
        os.environ.pop(name, None)


def _reset_counters() -> None:
    with pr._LOCK:
        for key in pr._PER_GAME_KEYS:
            pr._STATE[key] = 0
        pr._STATE["errors"] = 0
        pr._STATE["skips"] = {}
        pr._STATE["per_game"] = {}


def _tool_call(code: str, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}


class _FakeSession:
    """Stand-in for _HarnessGameSession: step_env executes UP/DOWN/LEFT/RIGHT and
    returns the solver's error payload (executed False) for anything else."""

    def __init__(self, runtime_mod, state_path: Path, *, game_id="tu93-test"):
        self.runtime_mod = runtime_mod
        self.state_path = state_path
        self.game = types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=game_id, state="playing"),
                                          current_state=types.SimpleNamespace(won=False))
        self.history_entries = [runtime_mod.HistoryEntry(action="", frame=runtime_mod.Frame(grid=_GRID, step=0, level=1))]
        self.executed: list[str] = []
        self._write()

    def _write(self):
        self.runtime_mod.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame,
                                             history=self.history_entries)

    def step_env(self, arguments):
        acts = arguments.get("actions") or []
        names = [str(a.get("action", "")).upper() for a in acts]
        if any(n not in ("UP", "DOWN", "LEFT", "RIGHT") for n in names):
            return {"executed": False, "error": f"{names[0]} is not valid right now.", "valid_actions": ["UP", "DOWN"]}
        for name in names:
            step = len(self.history_entries)
            self.executed.append(name)
            self.history_entries.append(self.runtime_mod.HistoryEntry(
                action=name, frame=self.runtime_mod.Frame(grid=_GRID, step=step, level=1)))
        self._write()
        return {"executed": True, "action_num": len(self.history_entries) - 1, "level": 1, "score": 0, "reward": 0.0,
                "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"], "board_changed": True,
                "done": False, "level_completed": False, "game_over": False, "run_complete": False,
                "action_display": names[-1], "executed_actions": names, "requested_count": len(names),
                "executed_count": len(names), "stopped_early": False, "batched": len(names) > 1}


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.agent import runtime_state as runtime_mod  # noqa: PLC0415
        cls.agent_mod, cls.runtime_mod = agent_mod, runtime_mod
        cls.status = pr.install()

    def setUp(self) -> None:
        _clear()
        _reset_counters()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "artifacts" / "tu93-test_p0_state.json"
        self.transcript = self.root / "transcripts" / "tu93-test_p0.txt"
        self.state_path.parent.mkdir(parents=True)
        self.transcript.parent.mkdir(parents=True)
        self.transcript.touch()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- helpers ----------------------------------------------------------
    def _agent(self):
        return self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def _session(self, **kw):
        return _FakeSession(self.runtime_mod, self.state_path, **kw)

    def _reply(self, code: str | None, cid: str = "c1", content=None):
        if code is None:
            return self.agent_mod._ChatCompletionResult(message={"role": "assistant", "content": content or "World model: hmm."},
                                                        finish_reason="stop", usage={})
        return self.agent_mod._ChatCompletionResult(
            message={"role": "assistant", "content": content, "tool_calls": [_tool_call(code, cid)]},
            finish_reason="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    def _analyze(self, agent, sess, replies, *, step=1, action_num=0, transcript=None):
        it = iter(replies)
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=lambda *a, **k: next(it)):
            return agent.analyze(self.state_path, action_num, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"],
                                 step_env=sess.step_env, transcript_path=transcript or self.transcript,
                                 analysis_step=step)

    def _open_turn(self, agent, sess, turn=1):
        """Arm a fake in-flight turn so _run_python_tool can be driven directly."""
        st = pr._pstate(agent, self.state_path)
        st.agent_mod = self.agent_mod
        st.transcript_path = self.transcript
        st.game = "tu93-test_p0"
        st.turn = turn
        st.reset_turn()
        st.in_turn = True
        agent._ensure_session(self.state_path)
        agent._step_env_callback = sess.step_env
        agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        return st

    def _run(self, agent, code):
        return agent._run_python_tool(self.state_path, {"code": code})

    # ------------------------------------------------------------ 01 install
    def test_01_install(self) -> None:
        self.assertIn(self.status, {"probe: OK", "probe: SKIP (already applied)"})
        self.assertTrue(pr._STATE["installed"])
        for name in ("analyze", "_run_python_tool", "_build_user_prompt"):
            self.assertTrue(hasattr(getattr(self.agent_mod.ToolAgent, name), "_probe_stock"), name)
        self.assertEqual(set(pr._STOCK), {"analyze", "run_python_tool", "build_user_prompt"})

    # ------------------------------------------------- 02 flag-off passthrough
    def test_02_flag_off_is_byte_identical_to_stock(self) -> None:
        """Stock analyze vs the wrapper with PROBE_ENABLE=0 on a turn shaped A A A X
        (a refusal would fire on the 3rd call if the flag were on): same transcript
        bytes, same prompts, same persisted history, no state object, zero counters."""
        os.environ["PROBE_ENABLE"] = "0"
        transcripts, prompts, histories = [], [], []
        for variant in ("stock", "wrapped"):
            root = self.root / variant
            state_path = root / "artifacts" / "g_p0_state.json"
            transcript = root / "transcripts" / "g_p0.txt"
            state_path.parent.mkdir(parents=True)
            transcript.parent.mkdir(parents=True)
            transcript.touch()
            sess = _FakeSession(self.runtime_mod, state_path)
            agent = self._agent()
            seen = []
            inner = pr._STOCK["build_user_prompt"]

            def spy(s, an, *a, **k):  # noqa: ANN001
                text = inner(s, an, *a, **k)
                seen.append(text)
                return text

            stock_analyze = pr._STOCK["analyze"]
            stock_run = pr._STOCK["run_python_tool"]
            with mock.patch.object(self.agent_mod.time, "strftime", return_value="00:00:00"), \
                    mock.patch.dict(pr._STOCK, {"build_user_prompt": spy}):
                if variant == "stock":
                    patches = [mock.patch.object(self.agent_mod.ToolAgent, "analyze", stock_analyze),
                               mock.patch.object(self.agent_mod.ToolAgent, "_run_python_tool", stock_run),
                               mock.patch.object(self.agent_mod.ToolAgent, "_build_user_prompt", spy)]
                else:
                    patches = []
                for p in patches:
                    p.start()
                try:
                    for turn in (1, 2):
                        replies = [self._reply(A_CODE, "c1"), self._reply(A_CODE, "c2"),
                                   self._reply(A_SENTINEL, "c3"), self._reply(X_CODE, "c4")]
                        it = iter(replies)
                        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion",
                                               side_effect=lambda *a, **k: next(it)):
                            res = agent.analyze(state_path, turn - 1, valid_actions=["UP", "DOWN"], step_env=sess.step_env,
                                                transcript_path=transcript, analysis_step=turn)
                        self.assertTrue(res.step_executed)
                finally:
                    for p in patches:
                        p.stop()
            self.assertFalse(hasattr(agent, "_probe"))
            transcripts.append(transcript.read_bytes())
            prompts.append(seen)
            histories.append(json.dumps(agent._history_messages, sort_keys=True))
            self.assertEqual(sess.executed, ["UP", "UP"])
        self.assertEqual(transcripts[0], transcripts[1])
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(histories[0], histories[1])
        self.assertIn(b"[TOOL RESULT: python]\nSENTINEL-MUST-NOT-RUN", transcripts[1])   # flag off: the 3rd snippet ran
        self.assertNotIn(b"[PROBE-REFUSE]", transcripts[1])
        self.assertNotIn(b"Analysis budget", transcripts[1])
        self.assertNotIn(b"Previous turn executed no action", transcripts[1])
        st = pr.status()
        self.assertEqual((st["refusals"], st["analysis_calls_total"], st["turns_total"]), (0, 0, 0))

    # ----------------------------------------- 03 analysis-only from the payload
    def test_03_analysis_only_detected_from_the_sandbox_payload(self) -> None:
        agent = self._agent()
        sess = self._session()
        st = self._open_turn(agent, sess)
        r = self._run(agent, X_CODE)                                   # acts
        self.assertTrue(r.step_executed)
        self.assertEqual((st.acting_calls, st.analysis_calls), (1, 0))
        r = self._run(agent, A_CODE)                                   # pure analysis
        self.assertFalse(r.step_executed)
        self.assertIn('"stdout": "analysis 2\\n"', r.content)          # it ran
        self.assertEqual((st.acting_calls, st.analysis_calls), (1, 1))
        os.environ["PROBE_MAX_ANALYSIS"] = "9"                        # keep counting, no refusal here
        r = self._run(agent, BAD_CODE)                                 # action() that FAILED: executed False in the payload
        self.assertFalse(r.step_executed)
        self.assertIn("is not valid right now", r.content)
        self.assertEqual((st.acting_calls, st.analysis_calls), (1, 2))  # counted from the payload, not the code
        r = self._run(agent, ERR_CODE)                                 # python error before any action
        self.assertFalse(r.step_executed)
        self.assertIn("ZeroDivisionError", r.content)
        self.assertEqual((st.acting_calls, st.analysis_calls), (1, 3))
        self.assertEqual(sess.executed, ["UP"])
        s = pr.status()
        self.assertEqual((s["analysis_calls_total"], s["acting_calls_total"], s["refusals"]), (3, 1, 0))
        self.assertEqual(s["turns_ge3_analysis"], 1)                   # 3 executed analysis-only calls = a leak
        # the static predicate used BEFORE running
        self.assertIs(pr.code_calls_action(A_CODE), False)
        self.assertIs(pr.code_calls_action(X_CODE), True)
        self.assertIs(pr.code_calls_action(BAD_CODE), True)
        self.assertIs(pr.code_calls_action("def go():\n    action('UP')\n"), True)   # defined but never called: run it
        self.assertIs(pr.code_calls_action("print('call action( later')\n"), False)  # text is not a call
        self.assertIs(pr.code_calls_action("def broken(:\n"), None)

    # ---------------------------------------- 04 refusal at the 3rd call only
    def test_04_refusal_fires_exactly_at_the_third_analysis_only_call(self) -> None:
        agent = self._agent()
        sess = self._session()
        st = self._open_turn(agent, sess)                              # (_ensure_session resets the carried note)
        agent._summarized_knowledge["open_questions"] = ("Does SPACE toggle the door? Is the red bar a timer; "
                                                         "maybe the key must be carried to the exit")
        r1 = self._run(agent, A_CODE)
        r2 = self._run(agent, A_CODE)
        self.assertFalse(r1.step_executed or r2.step_executed)
        self.assertNotIn("Analysis budget", r1.content + r2.content)
        self.assertEqual(st.refusals, 0)
        self.assertNotIn("[PROBE-REFUSE]", self.transcript.read_text())
        r3 = self._run(agent, A_SENTINEL)                              # the 3rd analysis-only call
        self.assertFalse(r3.step_executed)
        self.assertEqual(st.refusals, 1)
        self.assertNotIn("SENTINEL", r3.content)                       # NOT executed
        self.assertNotIn('"stdout"', r3.content)
        payload = json.loads(r3.content)
        self.assertEqual(set(payload), {"tool", "error"})
        text = payload["error"]
        self.assertTrue(text.startswith("Analysis budget for this turn is spent (2 analysis-only calls). "
                                        "Only a snippet that executes a game action is accepted now: run a <=5-action "
                                        "test of your leading hypothesis with action([...]) and read the result. "
                                        "Untested hypotheses in your notes:"), text)
        self.assertIn("\n- Does SPACE toggle the door?", text)
        self.assertIn("\n- Is the red bar a timer", text)
        self.assertIn("\n- maybe the key must be carried to the exit", text)
        self.assertEqual(self.transcript.read_text().count(
            "[HARNESS PROBE]\n[PROBE-REFUSE] game=tu93-test_p0 turn=1 analysis_calls=2 refusal=1/2\n"), 1)
        # the transcript display of an error-only payload is the bare text (stock _render_tool_result_display)
        self.assertEqual(self.agent_mod._render_tool_result_display(r3.content), text)
        # a 3rd call that DOES call action() is never refused
        r4 = self._run(agent, X_CODE)
        self.assertTrue(r4.step_executed)
        self.assertEqual(sess.executed, ["UP"])
        s = pr.status()
        self.assertEqual((s["refusals"], s["turns_with_refusal"], s["calls_after_refusal"], s["acting_calls_after_refusal"]),
                         (1, 1, 1, 1))
        self.assertEqual(s["acting_after_refusal_share"], 1.0)
        self.assertEqual((st.analysis_calls, st.acting_calls), (2, 1))  # a refused call is not an executed analysis call

    # --------------------------------------------------------- 05 deadlock cap
    def test_05_refusal_cap_never_deadlocks_a_turn(self) -> None:
        agent = self._agent()
        sess = self._session()
        st = self._open_turn(agent, sess)
        self._run(agent, A_CODE)
        self._run(agent, A_CODE)
        r3 = self._run(agent, A_SENTINEL)
        r4 = self._run(agent, A_SENTINEL)
        self.assertIn("Analysis budget", r3.content)
        self.assertIn("Analysis budget", r4.content)
        self.assertEqual(st.refusals, 2)
        r5 = self._run(agent, "print('FIFTH RAN')\n")                  # cap reached: the stock loop proceeds
        self.assertNotIn("Analysis budget", r5.content)
        self.assertIn("FIFTH RAN", r5.content)
        self.assertEqual(st.refusals, 2)
        s = pr.status()
        self.assertEqual(s["skips"].get("refusal_cap"), 1)
        self.assertEqual(s["refusals"], 2)
        self.assertEqual(s["turns_with_refusal"], 1)                   # counted once per turn
        self.assertEqual((s["calls_after_refusal"], s["acting_calls_after_refusal"]), (2, 0))
        self.assertEqual(s["turns_ge3_analysis"], 1)                   # the 5th call = 3rd executed analysis-only
        self.assertEqual(self.transcript.read_text().count("[PROBE-REFUSE]"), 2)
        self.assertIn("refusal=2/2", self.transcript.read_text())
        # PROBE_MAX_REFUSALS=0 disables refusing entirely; PROBE_MAX_ANALYSIS=0 refuses the very first analysis call
        os.environ["PROBE_MAX_REFUSALS"] = "0"
        st = self._open_turn(agent, sess, turn=2)
        for _ in range(4):
            self.assertNotIn("Analysis budget", self._run(agent, A_CODE).content)
        os.environ["PROBE_MAX_REFUSALS"] = "2"
        os.environ["PROBE_MAX_ANALYSIS"] = "0"
        st = self._open_turn(agent, sess, turn=3)
        self.assertIn("Analysis budget for this turn is spent (0 analysis-only calls)", self._run(agent, A_CODE).content)
        self.assertEqual(st.refusals, 1)

    # ---------------------------------------- 06 through analyze(): per-turn reset
    def test_06_analyze_end_to_end_and_per_turn_reset(self) -> None:
        """Turn 1 = A A A(refused) X: the refusal text is the tool message the model gets,
        lands under [TOOL RESULT: python] after the marker, and the action still executes.
        Turn 2 = A A X: fresh budget, no refusal."""
        agent = self._agent()
        sess = self._session()
        requests = []
        stock_chat = self.agent_mod.ToolAgent._chat_completion  # noqa: F841 (documenting the seam)
        replies1 = [self._reply(A_CODE, "c1"), self._reply(A_CODE, "c2"), self._reply(A_SENTINEL, "c3"),
                    self._reply(X_CODE, "c4")]
        it = iter(replies1)

        def chat(self_, messages, **kwargs):
            requests.append(json.loads(json.dumps(messages)))
            return next(it)

        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", chat):
            res = agent.analyze(self.state_path, 0, valid_actions=["UP", "DOWN"], step_env=sess.step_env,
                                transcript_path=self.transcript, analysis_step=1)
        self.assertTrue(res.step_executed)
        self.assertEqual(sess.executed, ["UP"])
        text = self.transcript.read_text()
        self.assertEqual(text.count("[PROBE-REFUSE]"), 1)
        # order in the file: the refused TOOL CALL, the marker, then the TOOL RESULT carrying the text
        i_call = text.index("SENTINEL-MUST-NOT-RUN")
        i_mark = text.index("[HARNESS PROBE]\n[PROBE-REFUSE] game=tu93-test_p0 turn=1 analysis_calls=2 refusal=1/2")
        i_res = text.index("[TOOL RESULT: python]\nAnalysis budget for this turn is spent (2 analysis-only calls).")
        self.assertLess(i_call, i_mark)
        self.assertLess(i_mark, i_res)
        # the sentinel is quoted in the META and TOOL CALL sections (the model's request) but never in a result
        self.assertNotIn("[TOOL RESULT: python]\nSENTINEL-MUST-NOT-RUN", text)
        self.assertEqual(text.count("SENTINEL-MUST-NOT-RUN"), 2)
        self.assertIn("[TOOL RESULT: python]\nacted True", text)          # the 4th call acted
        # the 4th request (the one after the refusal) carried the refusal as a tool message
        tool_msgs = [m for m in requests[3] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 3)
        self.assertIn("Analysis budget for this turn is spent", tool_msgs[2]["content"])
        self.assertEqual(json.loads(tool_msgs[2]["content"])["tool"], "python")
        # persisted for later turns like any tool result
        kept = [m for m in agent._history_messages if m.get("role") == "tool"]
        self.assertTrue(any("Analysis budget" in m["content"] for m in kept))
        s = pr.status()
        self.assertEqual((s["refusals"], s["turns_with_refusal"], s["analysis_calls_total"], s["acting_calls_total"]), (1, 1, 2, 1))
        self.assertEqual((s["calls_after_refusal"], s["acting_calls_after_refusal"], s["noact_turns"], s["turns_total"]), (1, 1, 0, 1))
        # turn 2: the budget is per turn
        res = self._analyze(agent, sess, [self._reply(A_CODE, "d1"), self._reply(A_CODE, "d2"), self._reply(X_CODE, "d3")],
                            step=2, action_num=1)
        self.assertTrue(res.step_executed)
        self.assertEqual(self.transcript.read_text().count("[PROBE-REFUSE]"), 1)
        self.assertEqual(sess.executed, ["UP", "UP"])
        s = pr.status()
        self.assertEqual((s["refusals"], s["turns_total"], s["analysis_calls_total"]), (1, 2, 4))
        self.assertEqual(s["per_game"]["tu93-test_p0"]["refusals"], 1)   # keyed by the run stem

    # --------------------------------------------- 07 NOACT line once, next turn
    def test_07_noact_prefix_once_on_the_next_turn_only(self) -> None:
        agent = self._agent()
        agent._tool_steps = 2                                          # the turn ends after 2 calls, nothing executed
        sess = self._session()
        res = self._analyze(agent, sess, [self._reply(A_CODE, "c1"), self._reply(A_CODE, "c2")], step=1)
        self.assertFalse(res.step_executed)
        text = self.transcript.read_text()
        self.assertIn("[HARNESS PROBE]\n[PROBE-NOACT] game=tu93-test_p0 turn=1 analysis_calls=2 refusals=0 reason=no_capture\n", text)
        self.assertNotIn("Previous turn executed no action", text)      # not on the turn itself
        self.assertEqual(pr.status()["noact_turns"], 1)
        agent._tool_steps = None
        prompts = []
        inner = pr._STOCK["build_user_prompt"]

        def spy(s, an, *a, **k):  # noqa: ANN001
            out = inner(s, an, *a, **k)
            prompts.append(out)
            return out

        with mock.patch.dict(pr._STOCK, {"build_user_prompt": spy}):
            res = self._analyze(agent, sess, [self._reply(X_CODE, "d1")], step=2)
            self.assertTrue(res.step_executed)
            res = self._analyze(agent, sess, [self._reply(X_CODE, "e1")], step=3, action_num=1)
        text = self.transcript.read_text()
        self.assertEqual(text.count("Previous turn executed no action after 2 analysis calls."), 1)
        # it is the FIRST line of turn 2's user prompt, and turn 3 has none
        turn2 = text.split("--- analysis_step=2 ")[1].split("--- analysis_step=3 ")[0]
        turn3 = text.split("--- analysis_step=3 ")[1]
        self.assertIn("[USER PROMPT]\nPrevious turn executed no action after 2 analysis calls.\n", turn2)
        self.assertNotIn("Previous turn executed no action", turn3)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(pr.status()["noact_turns"], 1)
        # a yielded turn is a NOACT with reason=yield (0 analysis calls when the yield hits before any call)
        agent._yield_seconds = 0.0
        res = self._analyze(agent, sess, [], step=4, action_num=2)
        self.assertTrue(res.yielded_control)
        self.assertIn("[PROBE-NOACT] game=tu93-test_p0 turn=4 analysis_calls=0 refusals=0 reason=yield", self.transcript.read_text())
        agent._yield_seconds = None
        res = self._analyze(agent, sess, [self._reply(X_CODE, "f1")], step=5, action_num=2)
        self.assertIn("[USER PROMPT]\nPrevious turn executed no action after 0 analysis calls.\n",
                      self.transcript.read_text().split("--- analysis_step=5 ")[1])
        # a stop_requested yield (the run is ending) is NOT a NOACT turn either
        agent._probe.noact_pending = None
        it = iter([])
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=lambda *a, **k: next(it)):
            res = agent.analyze(self.state_path, 3, valid_actions=["UP"], step_env=sess.step_env,
                                transcript_path=self.transcript, analysis_step=6, should_stop=lambda: True)
        self.assertTrue(res.yielded_control)
        self.assertEqual(pr.status()["noact_turns"], 2)
        self.assertEqual(pr.status()["skips"].get("stop_requested"), 1)
        self.assertIsNone(agent._probe.noact_pending)
        # a request error is NOT a NOACT turn (the solver retries the same step)
        import requests as _rq  # noqa: PLC0415
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=_rq.RequestException("boom")):
            res = agent.analyze(self.state_path, 3, valid_actions=["UP"], step_env=sess.step_env,
                                transcript_path=self.transcript, analysis_step=6)
        self.assertTrue(res.retryable_failure)
        self.assertEqual(pr.status()["noact_turns"], 2)
        self.assertIsNone(agent._probe.noact_pending)

    # ------------------------------------------------------- 08 status / counters
    def test_08_status_shape_and_per_game(self) -> None:
        s = pr.status()
        for key in ("installed", "enabled", "max_analysis", "max_probe", "max_refusals", "note_lines", "refusals",
                    "turns_with_refusal", "noact_turns", "analysis_calls_total", "acting_calls_total",
                    "calls_after_refusal", "acting_calls_after_refusal", "turns_total", "turns_ge3_analysis",
                    "acting_after_refusal_share", "errors", "skips", "per_game"):
            self.assertIn(key, s)
        self.assertEqual((s["max_analysis"], s["max_probe"], s["max_refusals"], s["note_lines"]), (2, 5, 2, 3))
        self.assertIsNone(s["acting_after_refusal_share"])
        os.environ.update({"PROBE_MAX_ANALYSIS": "garbage", "PROBE_MAX_PROBE": "3", "PROBE_MAX_REFUSALS": "-4",
                           "PROBE_NOTE_LINES": "1"})
        s = pr.status()
        self.assertEqual((s["max_analysis"], s["max_probe"], s["max_refusals"], s["note_lines"]), (2, 3, 0, 1))
        _clear()
        agent = self._agent()
        sess = self._session()
        self._open_turn(agent, sess)
        for code in (A_CODE, A_CODE, A_CODE, X_CODE):
            self._run(agent, code)
        s = pr.status()
        pg = s["per_game"]["tu93-test_p0"]
        self.assertEqual(pg["refusals"], 1)
        self.assertEqual(pg["acting_calls_after_refusal"], 1)
        self.assertEqual(pg["analysis_calls_total"], 2)
        self.assertEqual(s["acting_after_refusal_share"], 1.0)

    # ------------------------------------------------------ 09 exception fallback
    def test_09_exception_inside_the_graft_returns_the_stock_result(self) -> None:
        agent = self._agent()
        sess = self._session()
        self._open_turn(agent, sess)
        self._run(agent, A_CODE)
        self._run(agent, A_CODE)
        with mock.patch.object(pr, "code_calls_action", side_effect=RuntimeError("boom")):
            r = self._run(agent, "print('RAN ANYWAY')\n")
        self.assertIn("RAN ANYWAY", r.content)                         # stock ran the snippet
        self.assertGreaterEqual(pr.status()["errors"], 1)
        with mock.patch.object(pr, "render_noact_line", side_effect=RuntimeError("boom")):
            agent._probe.noact_pending = {"analysis_calls": 1, "refusals": 0}
            text = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertEqual(text, pr._STOCK["build_user_prompt"](agent, 1, valid_actions=["UP"]))

    # --------------------------------------------------- 10 refusal / noact text
    def test_10_text_rendering_and_caps(self) -> None:
        self.assertEqual(pr.note_hypotheses({}), [])
        self.assertEqual(pr.note_hypotheses({"cross_level_notes": "kept but never quoted"}), [])
        notes = pr.note_hypotheses({"open_questions": "1) what does SPACE do? 2) is the bar a timer; 3) door needs key. 4) extra one",
                                    "current_plan": "Next: test UP twice."})
        self.assertEqual(notes, ["what does SPACE do?", "is the bar a timer", "door needs key."])
        self.assertEqual(pr.note_hypotheses({"current_plan": "Try the lever - then the door"}, limit=2), ["Try the lever", "then the door"])
        self.assertEqual(pr.note_hypotheses({"world_model": "Hypothesis text lives here after the stock mapping"}, limit=1),
                         ["Hypothesis text lives here after the stock mapping"])
        long = pr.note_hypotheses({"open_questions": "q" * 500}, limit=3)
        self.assertEqual(len(long), 1)
        self.assertLessEqual(len(long[0]), pr.NOTE_LINE_CHARS)
        text = pr.render_refusal(2, {"open_questions": "w " * 400 + "; " + "x " * 400 + "; " + "y " * 400})
        self.assertLessEqual(len(text), pr.REFUSAL_CHARS)
        self.assertIn("(none recorded - state one now and test it)", pr.render_refusal(2, {}))
        os.environ["PROBE_MAX_PROBE"] = "3"
        self.assertIn("run a <=3-action test", pr.render_refusal(2, None))
        os.environ["PROBE_NOTE_LINES"] = "0"
        self.assertIn("(none recorded", pr.render_refusal(2, {"open_questions": "something"}))
        _clear()
        self.assertEqual(pr.render_noact_line({"analysis_calls": 5, "refusals": 0}),
                         "Previous turn executed no action after 5 analysis calls.")
        line = pr.render_noact_line({"analysis_calls": 5, "refusals": 2})
        self.assertEqual(line, "Previous turn executed no action after 5 analysis calls (2 refused).")
        self.assertLessEqual(len(line), pr.NOACT_LINE_CHARS)

    # ------------------------------------------------ 11 game change resets state
    def test_11_new_game_gets_fresh_state(self) -> None:
        agent = self._agent()
        sess = self._session()
        self._analyze(agent, sess, [self._reply(A_CODE, "c1"), self._reply(A_CODE, "c2"), self._reply(A_CODE, "c3"),
                                    self._reply(X_CODE, "c4")], step=1)
        self.assertEqual(agent._probe.refusals, 1)
        other_state = self.root / "artifacts2" / "vc33-test_p0_state.json"
        other_state.parent.mkdir()
        other_tr = self.root / "transcripts" / "vc33-test_p0.txt"
        other_tr.touch()
        sess2 = _FakeSession(self.runtime_mod, other_state, game_id="vc33-test")
        agent._tool_steps = 1
        it = iter([self._reply(A_CODE, "z1")])
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=lambda *a, **k: next(it)):
            agent.analyze(other_state, 0, valid_actions=["UP"], step_env=sess2.step_env, transcript_path=other_tr, analysis_step=1)
        self.assertEqual(agent._probe.game, "vc33-test_p0")
        self.assertEqual(agent._probe.refusals, 0)
        self.assertIn("[PROBE-NOACT] game=vc33-test_p0 turn=1 analysis_calls=1", other_tr.read_text())
        self.assertEqual(set(pr.status()["per_game"]), {"tu93-test_p0", "vc33-test_p0"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
