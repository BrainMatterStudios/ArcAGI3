"""Offline validation for compaction-instead-of-eviction (CARRY).

Run:  GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference \
          .venv/bin/python submission/_throughput_v1/test_graft_carry.py
      (default bundle: the anim bundle under submission/_inspect_replay)

The python tool runs in the REAL sandbox subprocess; the model is a fake
`_chat_completion` (patched INSIDE the graft's _STOCK so the wrapper stays
active) and the compaction call hits a fake `requests.post`.
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

import graft_carry as cr  # noqa: E402

_FLAGS = ("CARRY_ENABLE", "CARRY_TARGET_FRACTION", "CARRY_SUMMARY_CHARS", "CARRY_INPUT_CHARS",
          "CARRY_COMPACT_MAX_TOKENS", "CARRY_COMPACT_THINKING", "CARRY_MIN_DROP_MSGS", "CARRY_TIMEOUT_S",
          "CARRY_WINDOW_TOKENS")
_GRID = tuple(tuple((r * 7 + c) % 5 for c in range(8)) for r in range(8))
X_CODE = "r = action(['UP'])\nprint('acted', r.get('executed'))\n"
A_CODE = "n = len(history)\nprint('analysis', n)\n"
SUMMARY_SENTINEL = "SUMMARY-SENTINEL"
SUMMARY_TEXT = (f"MECHANICS VERIFIED\n- UP moves the blue block one row ({SUMMARY_SENTINEL}).\n"
                "HYPOTHESES REFUTED\n- SPACE does nothing.\nCURRENT PLAN\n- test DOWN next.")


def _clear() -> None:
    for name in _FLAGS:
        os.environ.pop(name, None)


def _reset_counters() -> None:
    with cr._LOCK:
        for key in cr._PER_GAME_KEYS:
            cr._STATE[key] = 0
        cr._STATE["errors"] = 0
        cr._STATE["skips"] = {}
        cr._STATE["per_game"] = {}


def _tool_call(code: str, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}


def _long_reasoning(tag: str, n: int = 3000) -> str:
    body = f"Thinking about the board ({tag}). "
    return (body * (n // len(body) + 1))[:n]


class _FakeResponse:
    def __init__(self, status: int = 200, body: dict | None = None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


class _FakeSession:
    """Stand-in for _HarnessGameSession (graft_probe tests): step_env executes UP/DOWN/LEFT/RIGHT."""

    def __init__(self, runtime_mod, state_path: Path, *, game_id="tu93-test"):
        self.runtime_mod = runtime_mod
        self.state_path = state_path
        self.game = types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=game_id, state="playing"),
                                          current_state=types.SimpleNamespace(won=False, levels_completed=0))
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


class CarryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.agent import runtime_state as runtime_mod  # noqa: PLC0415
        cls.agent_mod, cls.runtime_mod = agent_mod, runtime_mod
        cls.status = cr.install()

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
        self.requests: list[list[dict]] = []          # messages of every regular model call
        self.compactions: list[dict] = []             # payloads of every compaction post

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- helpers ----------------------------------------------------------
    def _agent(self, budget: int | None = None):
        agent = self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
        if budget is not None:
            agent._context_budget_tokens = budget
        return agent

    def _session(self, **kw):
        return _FakeSession(self.runtime_mod, self.state_path, **kw)

    def _reply(self, code: str, cid: str = "c1", reasoning: str = "", content=None, prompt_tokens: int = 10):
        return self.agent_mod._ChatCompletionResult(
            message={"role": "assistant", "content": content, "reasoning": reasoning, "tool_calls": [_tool_call(code, cid)]},
            finish_reason="tool_calls",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": 5, "total_tokens": prompt_tokens + 5})

    def _fake_chat(self, replies):
        it = iter(replies)

        def fake(agent, messages, **kwargs):  # noqa: ANN001
            self.requests.append(json.loads(json.dumps(messages)))
            return next(it)
        return fake

    def _fake_post(self, *, summary: str = SUMMARY_TEXT, status: int = 200, raise_exc: Exception | None = None,
                   finish: str = "stop"):
        def post(url, headers=None, json=None, timeout=None, **kw):  # noqa: A002
            self.compactions.append({"url": url, "payload": json, "timeout": timeout, "headers": headers})
            if raise_exc is not None:
                raise raise_exc
            if status >= 400:
                return _FakeResponse(status, {"error": "boom"})
            return _FakeResponse(200, {"choices": [{"index": 0, "finish_reason": finish,
                                                     "message": {"role": "assistant", "content": summary}}],
                                       "usage": {"prompt_tokens": 2500, "completion_tokens": 240, "total_tokens": 2740}})
        return post

    def _turn(self, agent, sess, replies, *, step, post=None, request_timeout=None):
        with mock.patch.dict(cr._STOCK, {"chat_completion": self._fake_chat(replies)}), \
                mock.patch.object(self.agent_mod.requests, "post", post or self._fake_post()):
            return agent.analyze(self.state_path, step - 1, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"],
                                 step_env=sess.step_env, transcript_path=self.transcript, analysis_step=step,
                                 request_timeout_seconds=request_timeout)

    def _budget_for(self, agent, extra: int) -> int:
        """A budget = system + tools + `extra` estimated tokens, so eviction fires after a few turns."""
        sysmsg = {"role": "system", "content": agent._system_prompt}
        return agent._estimate_request_input_tokens([sysmsg], tools=agent._tools(self.state_path)) + extra

    # ------------------------------------------------------------ 01 install
    def test_01_install(self) -> None:
        self.assertIn(self.status, {"carry: OK", "carry: SKIP (already applied)"})
        self.assertTrue(cr._STATE["installed"])
        for name in ("analyze", "_trim_messages_for_context", "_chat_completion"):
            self.assertTrue(hasattr(getattr(self.agent_mod.ToolAgent, name), "_carry_stock"), name)
        self.assertEqual(set(cr._STOCK), {"analyze", "trim", "chat_completion"})
        self.assertTrue(cr.COMPACT_SYSTEM.startswith(cr.COMPACT_SYSTEM_HEAD))

    # ------------------------------------------------- 02 flag-off passthrough
    def test_02_flag_off_is_byte_identical_to_stock(self) -> None:
        """Stock vs the wrapper with CARRY_ENABLE=0, on a budget so small that the stock trimmer evicts
        during the run: same transcript bytes, same request messages, same persisted history, no state
        object, zero counters, no compaction post."""
        os.environ["CARRY_ENABLE"] = "0"
        transcripts, requests_seen, histories, evicted = [], [], [], []
        for variant in ("stock", "wrapped"):
            root = self.root / variant
            state_path = root / "artifacts" / "g_p0_state.json"
            transcript = root / "transcripts" / "g_p0.txt"
            state_path.parent.mkdir(parents=True)
            transcript.parent.mkdir(parents=True)
            transcript.touch()
            sess = _FakeSession(self.runtime_mod, state_path)
            agent = self._agent()
            agent._context_budget_tokens = self._budget_for(agent, 5200)
            self.requests = []
            self.compactions = []
            fake = self._fake_chat([self._reply(X_CODE, f"c{t}", reasoning=_long_reasoning(f"t{t}")) for t in range(1, 7)])
            patches = []
            if variant == "stock":
                patches = [mock.patch.object(self.agent_mod.ToolAgent, "analyze", cr._STOCK["analyze"]),
                           mock.patch.object(self.agent_mod.ToolAgent, "_trim_messages_for_context", cr._STOCK["trim"]),
                           mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", fake)]
            else:
                patches = [mock.patch.dict(cr._STOCK, {"chat_completion": fake})]
            with mock.patch.object(self.agent_mod.time, "strftime", return_value="00:00:00"), \
                    mock.patch.object(self.agent_mod.requests, "post", self._fake_post()):
                for p in patches:
                    p.start()
                try:
                    for turn in range(1, 7):
                        res = agent.analyze(state_path, turn - 1, valid_actions=["UP", "DOWN"], step_env=sess.step_env,
                                            transcript_path=transcript, analysis_step=turn)
                        self.assertTrue(res.step_executed)
                finally:
                    for p in patches:
                        p.stop()
            self.assertFalse(hasattr(agent, "_carry"))
            self.assertEqual(self.compactions, [])
            transcripts.append(transcript.read_bytes())
            requests_seen.append(json.dumps(self.requests, sort_keys=True))
            histories.append(json.dumps(agent._history_messages, sort_keys=True))
            # the trimmer DID evict: the last request holds fewer turns than were played
            n_assistant = sum(1 for m in self.requests[-1] if m.get("role") == "assistant")
            evicted.append(n_assistant)
            self.assertEqual(sess.executed, ["UP"] * 6)
        self.assertLess(evicted[0], 5, evicted)
        self.assertEqual(transcripts[0], transcripts[1])
        self.assertEqual(requests_seen[0], requests_seen[1])
        self.assertEqual(histories[0], histories[1])
        self.assertNotIn(b"[HARNESS CARRY]", transcripts[1])
        self.assertNotIn(cr.SUMMARY_INTRO.encode(), transcripts[1])
        st = cr.status()
        self.assertEqual((st["calls_total"], st["compactions"], st["turns_total"]), (0, 0, 0))

    # ---------------------------------- 03 reasoning reaches the next request
    def test_03_prior_turn_reasoning_is_in_the_next_request_and_measured(self) -> None:
        """The stock keeps `reasoning` on persisted assistant messages; the wrapper measures it."""
        agent = self._agent()
        sess = self._session()
        self._turn(agent, sess, [self._reply(X_CODE, "c1", reasoning="R1-SENTINEL thinking " * 20, prompt_tokens=4000)], step=1)
        self._turn(agent, sess, [self._reply(X_CODE, "c2", reasoning="R2 thinking", prompt_tokens=5200)], step=2)
        second = self.requests[1]
        carried = [m for m in second if m.get("role") == "assistant" and "R1-SENTINEL" in str(m.get("reasoning", ""))]
        self.assertEqual(len(carried), 1, [m.get("role") for m in second])
        self.assertEqual(second[0]["role"], "system")
        self.assertNotIn(cr.SUMMARY_INTRO, second[0]["content"])           # nothing compacted yet
        st = cr.status()
        self.assertEqual(st["calls_total"], 2)
        self.assertEqual(st["reasoning_msgs_total"], 1)                     # request 2 carried one reasoning message
        self.assertEqual(st["reasoning_chars_total"], len(("R1-SENTINEL thinking " * 20).strip()))   # stock strips it
        self.assertEqual((st["prompt_tokens_total"], st["prompt_tokens_max"], st["prompt_over_window"]), (9200, 5200, 0))
        self.assertEqual(st["calls_with_summary"], 0)
        text = self.transcript.read_text()
        self.assertIn("[HARNESS CARRY]\n[CARRY-CALL] game=tu93-test_p0 turn=1 req=1 msgs=2 reasoning_msgs=0 reasoning_chars=0 "
                      "summary_chars=0 prompt_tokens=4000 completion_tokens=5", text)
        self.assertIn("[CARRY-CALL] game=tu93-test_p0 turn=2 req=1 msgs=5 reasoning_msgs=1 reasoning_chars="
                      f"{len(('R1-SENTINEL thinking ' * 20).strip())} summary_chars=0 prompt_tokens=5200", text)
        # the marker lands BEFORE the call's [MODEL RESPONSE META] (extractor adjacency META -> THINKING intact)
        i_mark = text.index("[CARRY-CALL] game=tu93-test_p0 turn=1")
        i_meta = text.index("[MODEL RESPONSE META]")
        self.assertLess(i_mark, i_meta)
        self.assertEqual(st["per_game"]["tu93-test_p0"]["calls_total"], 2)

    # ------------------------------------------------------ 04 compaction fires
    def test_04_compaction_replaces_eviction(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = self._budget_for(agent, 5200)
        sess = self._session()
        for turn in range(1, 7):
            res = self._turn(agent, sess, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"DROPPED-SENTINEL-{turn}"))],
                             step=turn)
            self.assertTrue(res.step_executed)
        st = cr.status()
        self.assertGreaterEqual(st["compactions"], 1, st)
        self.assertEqual(st["compaction_failures"], 0)
        self.assertEqual(st["errors"], 0)
        self.assertEqual(len(self.compactions), st["compactions"])
        first = self.compactions[0]
        payload = first["payload"]
        self.assertTrue(cr.is_compaction_payload(payload))
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 1500)
        self.assertEqual(payload["model"], "test-model")
        self.assertTrue(first["url"].endswith("/v1/chat/completions"))
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertTrue(payload["messages"][0]["content"].startswith(cr.COMPACT_SYSTEM_HEAD))
        user = payload["messages"][1]["content"]
        self.assertIn("DROPPED-SENTINEL-1", user)                            # the dropped turn's reasoning reached the compactor
        self.assertIn("(none yet)", user)                                    # first compaction: no previous block
        self.assertIn("Game level in play: 1", user)
        self.assertIn("[TURN STATE]", user)
        self.assertIn("[YOUR CODE]", user)
        self.assertIn("[TOOL RESULT]", user)
        self.assertNotIn("Only tool: `python`", user)                        # boilerplate stripped
        # the request AFTER the compaction carries the block once, in the system message, and not the dropped turn
        k = next(i for i, msgs in enumerate(self.requests) if cr.SUMMARY_INTRO in msgs[0]["content"])
        after = self.requests[k]
        self.assertEqual(after[0]["role"], "system")
        self.assertEqual(after[0]["content"].count(cr.SUMMARY_INTRO), 1)
        self.assertIn(SUMMARY_SENTINEL, after[0]["content"])
        self.assertTrue(after[0]["content"].startswith(agent._system_prompt))
        self.assertFalse(any("DROPPED-SENTINEL-1" in str(m.get("reasoning", "")) for m in after))
        self.assertTrue(all(m.get("role") != "system" for m in agent._history_messages))   # never persisted
        # later compactions see the previous block
        if len(self.compactions) > 1:
            self.assertIn(SUMMARY_SENTINEL, self.compactions[1]["payload"]["messages"][1]["content"])
            self.assertNotIn("(none yet)", self.compactions[1]["payload"]["messages"][1]["content"])
        # transcript
        text = self.transcript.read_text()
        self.assertEqual(text.count("[CARRY-COMPACT] game=tu93-test_p0 "), st["compactions"])
        self.assertIn(" ok=1\nSUMMARY:\n" + SUMMARY_TEXT, text)
        self.assertIn(f"dropped_msgs={first['payload'] and st['per_game']['tu93-test_p0']['dropped_msgs_total'] if st['compactions'] == 1 else ''}"
                      if st["compactions"] == 1 else "dropped_msgs=", text)
        self.assertIn("prompt_tokens=2500 completion_tokens=240", text)
        self.assertGreaterEqual(st["calls_with_summary"], 1)
        self.assertEqual(st["summary_chars_total"], len(SUMMARY_TEXT) * st["compactions"])
        self.assertEqual(st["compaction_prompt_tokens"], 2500 * st["compactions"])
        self.assertGreaterEqual(st["dropped_msgs_total"], 2)
        self.assertEqual(sess.executed, ["UP"] * 6)
        # token accounting includes the compaction call
        self.assertGreaterEqual(agent.total_tokens, 2740)

    # ------------------------------------------------ 05 window never exceeded
    def test_05_window_is_never_exceeded_and_summary_is_capped(self) -> None:
        os.environ["CARRY_SUMMARY_CHARS"] = "600"
        agent = self._agent()
        agent._context_budget_tokens = self._budget_for(agent, 5200)
        sess = self._session()
        huge = ("MECHANICS VERIFIED\n- " + "x" * 50 + "\n") * 200
        post = self._fake_post(summary=huge)
        for turn in range(1, 8):
            self._turn(agent, sess, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"t{turn}"))], step=turn, post=post)
        st = cr.status()
        self.assertGreaterEqual(st["compactions"], 1)
        tools = agent._tools(self.state_path)
        for msgs in self.requests:
            self.assertLessEqual(agent._estimate_request_input_tokens(msgs, tools=tools), agent._context_budget_tokens)
        self.assertLessEqual(len(agent._carry.summary), 600)
        self.assertTrue(agent._carry.summary.endswith("…"))
        self.assertLessEqual(st["summary_chars_total"], 600 * st["compactions"])
        self.assertLessEqual(st["prompt_tokens_max"], cr.window_tokens())

    # ---------------------------------------------------- 06 failure modes
    def test_06_compaction_failure_falls_back_to_eviction(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = self._budget_for(agent, 5200)
        sess = self._session()
        exc = self.agent_mod.requests.ConnectionError("boom")
        for turn in range(1, 7):
            res = self._turn(agent, sess, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"t{turn}"))],
                             step=turn, post=self._fake_post(raise_exc=exc))
            self.assertTrue(res.step_executed)                               # the turn is never lost
        st = cr.status()
        self.assertGreaterEqual(st["compaction_failures"], 1)
        self.assertEqual(st["compactions"], 0)
        self.assertEqual(agent._carry.summary, "")
        tools = agent._tools(self.state_path)
        for msgs in self.requests:                                          # eviction still enforced the budget
            self.assertLessEqual(agent._estimate_request_input_tokens(msgs, tools=tools), agent._context_budget_tokens)
        text = self.transcript.read_text()
        self.assertIn(" ok=0 err=ConnectionError", text)
        self.assertNotIn("SUMMARY:", text)
        self.assertEqual(st["errors"], 0)
        # HTTP 500 and an empty completion are failures too
        _reset_counters()
        agent2 = self._agent()
        agent2._context_budget_tokens = self._budget_for(agent2, 5200)
        root2 = self.root / "g2"
        (root2 / "artifacts").mkdir(parents=True)
        (root2 / "transcripts").mkdir(parents=True)
        self.state_path = root2 / "artifacts" / "g2_p0_state.json"
        self.transcript = root2 / "transcripts" / "g2_p0.txt"
        self.transcript.touch()
        sess2 = self._session()
        posts = iter([self._fake_post(status=500), self._fake_post(summary="")] + [self._fake_post()] * 10)
        seen = []

        def switching_post(*a, **k):
            fn = next(posts)
            seen.append(fn)
            return fn(*a, **k)
        for turn in range(1, 13):                                            # compaction fires every ~3 turns here
            self._turn(agent2, sess2, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"t{turn}"))], step=turn,
                       post=switching_post)
        text2 = self.transcript.read_text()
        self.assertIn("err=http_500", text2)
        self.assertIn("err=empty", text2)
        st2 = cr.status()
        self.assertGreaterEqual(st2["compaction_failures"], 2)
        self.assertGreaterEqual(st2["compactions"], 1)                       # recovered on a later compaction
        self.assertEqual(st2["errors"], 0)

    # ---------------------------------------------------------- 07 skips
    def test_07_small_drop_overflow_path_and_no_time_skip_the_call(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = self._budget_for(agent, 5200)
        sess = self._session()
        os.environ["CARRY_MIN_DROP_MSGS"] = "99"
        for turn in range(1, 6):
            self._turn(agent, sess, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"t{turn}"))], step=turn)
        st = cr.status()
        self.assertEqual(self.compactions, [])
        self.assertGreaterEqual(st["skips"].get("small_drop", 0), 1)
        self.assertEqual(st["compactions"], 0)
        os.environ.pop("CARRY_MIN_DROP_MSGS")
        # overflow-recovery path (extra_safety_tokens > 0): eviction only
        msgs = [{"role": "system", "content": agent._system_prompt}, *agent._history_messages,
                {"role": "user", "content": "x"}]
        with mock.patch.object(self.agent_mod.requests, "post", self._fake_post()):
            out = agent._trim_messages_for_context(msgs, tools=agent._tools(self.state_path), extra_safety_tokens=512)
        self.assertEqual(self.compactions, [])
        self.assertGreaterEqual(cr.status()["skips"].get("overflow_path", 0), 1)
        self.assertLessEqual(agent._estimate_request_input_tokens(out, tools=agent._tools(self.state_path)),
                             agent._context_budget_tokens - 512)
        # no time left in the game: eviction only
        agent._carry.request_timeout = 5.0
        big = [{"role": "system", "content": agent._system_prompt}]
        for t in range(6):
            big.append({"role": "user", "content": f"Current state: step {t}, level 1."})
            big.append({"role": "assistant", "content": None, "reasoning": _long_reasoning(f"z{t}"), "tool_calls": [_tool_call(X_CODE, f"z{t}")]})
            big.append({"role": "tool", "tool_call_id": f"z{t}", "content": "{}"})
        with mock.patch.object(self.agent_mod.requests, "post", self._fake_post()):
            out = agent._trim_messages_for_context(big, tools=agent._tools(self.state_path))
        self.assertEqual(self.compactions, [])
        self.assertGreaterEqual(cr.status()["skips"].get("no_time", 0), 1)
        self.assertLessEqual(agent._estimate_request_input_tokens(out, tools=agent._tools(self.state_path)), agent._context_budget_tokens)

    # -------------------------------------------------- 08 new game resets
    def test_08_new_game_gets_a_fresh_block(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = self._budget_for(agent, 5200)
        sess = self._session()
        for turn in range(1, 7):
            self._turn(agent, sess, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"t{turn}"))], step=turn)
        self.assertGreaterEqual(cr.status()["compactions"], 1)
        self.assertTrue(agent._carry.summary)
        root2 = self.root / "game2"
        (root2 / "artifacts").mkdir(parents=True)
        (root2 / "transcripts").mkdir(parents=True)
        self.state_path = root2 / "artifacts" / "vc33-test_p0_state.json"
        self.transcript = root2 / "transcripts" / "vc33-test_p0.txt"
        self.transcript.touch()
        sess2 = self._session(game_id="vc33-test")
        self._turn(agent, sess2, [self._reply(X_CODE, "n1", reasoning="fresh")], step=1)
        self.assertEqual(agent._carry.summary, "")
        self.assertEqual(agent._carry.game, "vc33-test_p0")
        self.assertNotIn(cr.SUMMARY_INTRO, self.requests[-1][0]["content"])
        self.assertIn("tu93-test_p0", cr.status()["per_game"])
        self.assertIn("vc33-test_p0", cr.status()["per_game"])

    # ------------------------------------------------ 09 exception safety
    def test_09_exceptions_inside_the_graft_return_the_stock_result(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = self._budget_for(agent, 5200)
        sess = self._session()
        with mock.patch.object(cr, "system_with_summary", side_effect=RuntimeError("bang")), \
                mock.patch.object(cr, "reasoning_stats", side_effect=RuntimeError("bang")):
            for turn in range(1, 5):
                res = self._turn(agent, sess, [self._reply(X_CODE, f"c{turn}", reasoning=_long_reasoning(f"t{turn}"))], step=turn)
                self.assertTrue(res.step_executed)
        st = cr.status()
        self.assertGreater(st["errors"], 0)
        self.assertEqual(self.compactions, [])
        tools = agent._tools(self.state_path)
        for msgs in self.requests:
            self.assertLessEqual(agent._estimate_request_input_tokens(msgs, tools=tools), agent._context_budget_tokens)
        self.assertEqual(sess.executed, ["UP"] * 4)

    # ------------------------------------------------ 10 status + env parsing
    def test_10_status_shape_and_env_parsing(self) -> None:
        st = cr.status()
        for key in cr._PER_GAME_KEYS:
            self.assertIn(key, st)
        for key in ("installed", "enabled", "target_fraction", "summary_chars_cap", "reasoning_msgs_per_call",
                    "calls_with_summary_share", "summary_chars_mean", "compaction_e2e_s_mean", "errors", "skips", "per_game"):
            self.assertIn(key, st)
        self.assertEqual((st["target_fraction"], st["summary_chars_cap"], st["input_chars_cap"], st["compact_max_tokens"],
                          st["compact_thinking"], st["min_drop_msgs"], st["window_tokens"]), (0.5, 4800, 48000, 1500, False, 2, 32768))
        os.environ.update({"CARRY_TARGET_FRACTION": "garbage", "CARRY_SUMMARY_CHARS": "-5", "CARRY_COMPACT_THINKING": "1",
                           "CARRY_MIN_DROP_MSGS": "0", "CARRY_ENABLE": "off"})
        self.assertEqual(cr.target_fraction(), 0.5)
        self.assertEqual(cr.summary_chars(), 200)
        self.assertTrue(cr.compact_thinking())
        self.assertEqual(cr.min_drop_msgs(), 1)
        self.assertFalse(cr.enabled())
        os.environ["CARRY_TARGET_FRACTION"] = "0.05"
        self.assertEqual(cr.target_fraction(), 0.2)

    # ------------------------------------------------ 11 rendering of dropped turns
    def test_11_render_dropped_turns(self) -> None:
        agent = self._agent()
        agent._ensure_session(self.state_path)
        prompt = agent._build_user_prompt(3, valid_actions=["UP", "DOWN"],
                                          previous_step_summary={"executed_count": 2, "executed_actions": ["UP", "UP"], "level": 1})
        agent._summarized_knowledge["world_model"] = "blue block moves; red wall blocks"
        prompt_with_note = agent._build_user_prompt(3, valid_actions=["UP", "DOWN"],
                                                    previous_step_summary={"executed_count": 2, "executed_actions": ["UP", "UP"], "level": 1})
        slim = cr.slim_user_prompt(prompt_with_note)
        self.assertIn("The code executed 2 actions in the previous sequence.", slim)
        self.assertIn("Current state: step 4, level 1.", slim)
        self.assertIn("- World model: blue block moves; red wall blocks", slim)
        self.assertNotIn("Only tool", slim)
        self.assertNotIn("Revise any item", slim)
        self.assertNotIn("Note carried at that time", cr.slim_user_prompt(prompt))
        self.assertEqual(cr.slim_user_prompt("You have not acted yet. Investigate first. blah"),
                         "[harness: no tool call in the previous reply; asked to act]")
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": prompt_with_note}, {"type": "image_url", "image_url": {"url": "data:x"}}]},
            {"role": "assistant", "content": "World model: hmm", "reasoning": "R" * 9000, "tool_calls": [_tool_call("print(1)\n" * 400, "a")]},
            {"role": "tool", "tool_call_id": "a", "content": json.dumps({"returncode": 0, "stdout": "1\n" * 2000})},
        ]
        text = cr.render_dropped(self.agent_mod, msgs)
        self.assertIn("[TURN STATE]\nThe code executed 2 actions", text)
        self.assertIn("[YOUR REASONING]\n" + "R" * 3500 + "\n[... 4000 chars cut ...]\n" + "R" * 1500, text)
        self.assertIn("[YOUR NOTE]\nWorld model: hmm", text)
        self.assertIn("[YOUR CODE]\n" + "print(1)\n" * 166, text)                   # 1500-char head of the code
        code_block = text.split("[YOUR CODE]")[1].split("[TOOL RESULT]")[0]
        self.assertIn("[... 2100 chars cut]", code_block)
        self.assertLessEqual(len(code_block), 1500 + 40)
        self.assertIn("[TOOL RESULT]\n1\n1", text)
        self.assertLessEqual(len(text.split("[TOOL RESULT]\n")[1]), 1000 + 40)
        capped = cr.render_dropped(self.agent_mod, msgs * 20, cap=5000)
        self.assertLessEqual(len(capped), 5000 + 60)
        self.assertIn("chars cut ...]", capped)
        cm = cr.compaction_messages(2, "", "rendered", 6)
        self.assertEqual(cm[0]["role"], "system")
        self.assertIn(f"at most {4800 // 6} words", cm[0]["content"])
        self.assertIn("Game level in play: 2.", cm[1]["content"])
        self.assertIn("(none yet)", cm[1]["content"])
        self.assertIn("(6 messages, oldest first)", cm[1]["content"])
        self.assertTrue(cr.is_compaction_payload({"messages": cm}))
        self.assertFalse(cr.is_compaction_payload({"messages": cm, "tools": [{"x": 1}]}))
        self.assertFalse(cr.is_compaction_payload({"messages": [{"role": "system", "content": "You are a coding agent"}]}))
        self.assertEqual(cr.system_with_summary("base", ""), "base")
        self.assertEqual(cr.summary_in_messages([{"role": "system", "content": cr.system_with_summary("base", "abc")}]), 3)
        self.assertEqual(cr.summary_in_messages([{"role": "system", "content": "base"}]), 0)
        self.assertEqual(cr.reasoning_stats([{"role": "assistant", "reasoning": "ab"}, {"role": "assistant", "reasoning_content": "c"},
                                             {"role": "assistant", "content": "x"}, {"role": "user", "reasoning": "zz"}]), (2, 3))

    # ------------------------------- 13 every request keeps a real user message
    def test_13_never_returns_a_request_without_a_user_message(self) -> None:
        """The chat template answers 400 `No user query found in messages.` unless a role=="user"
        message survives (tool results ride as role "tool"). The UNMODIFIED stock reaches that state
        when one turn's own assistant+tool pairs exceed the budget; our bigger system message (it
        carries the block) reaches it sooner, so the wrapper must recover. Live evidence: 1 such 400
        in the 09-08 keith_carry kill test, 0 in three prior stock waves."""
        def turn_messages(agent, n, rlen, tlen):
            msgs = [{"role": "system", "content": agent._system_prompt},
                    {"role": "user", "content": "Current state: step 5, level 1."}]
            for i in range(n):
                msgs.append({"role": "assistant", "content": None, "reasoning": "R" * rlen,
                             "tool_calls": [_tool_call("print(1)", f"c{i}")]})
                msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "O" * tlen})
            return msgs

        # the stock defect, on the stock trimmer itself (no graft state involved)
        stock_agent = self._agent()
        heavy = turn_messages(stock_agent, 10, 8000, 4000)
        stock_out = cr._STOCK["trim"](stock_agent, heavy, tools=None)
        self.assertFalse(cr.has_user_message(stock_out))                    # documents the stock behaviour
        # the graft recovers on every severity, keeps the block, and stays inside the budget
        for n, rlen, tlen in ((8, 16000, 12000), (10, 8000, 4000), (8, 4000, 3000)):
            agent = self._agent()
            st = cr._cstate(agent, self.state_path)
            st.game, st.summary, st.request_timeout = "tu93-test_p0", SUMMARY_TEXT + "Z" * 3000, None
            msgs = turn_messages(agent, n, rlen, tlen)
            with mock.patch.object(self.agent_mod.requests, "post", self._fake_post()):
                out = agent._trim_messages_for_context(msgs, tools=None)
            self.assertTrue(cr.has_user_message(out), (n, rlen, tlen))
            self.assertLessEqual(agent._estimate_request_input_tokens(out, tools=None), agent._context_budget_tokens)
            self.assertEqual(out[0]["role"], "system")
            self.assertIn(cr.SUMMARY_INTRO, out[0]["content"])               # the block survives the recovery
            self.assertEqual(out[-1]["role"] in ("user", "tool"), True)
        skips = cr.status()["skips"]
        self.assertGreaterEqual(skips.get("keep_last_user", 0), 1)           # the drop-to-target loop stopped in time
        self.assertGreaterEqual(skips.get("rebuilt_from_last_user", 0), 1)   # the last-resort rebuild fired
        self.assertEqual(cr.status()["errors"], 0)
        # helper semantics: tool messages are NOT user queries
        self.assertFalse(cr.has_user_message([{"role": "system"}, {"role": "tool"}, {"role": "assistant"}]))
        self.assertTrue(cr.has_user_message([{"role": "tool"}, {"role": "user"}]))

    # ---------------------------------------------- 12 window counters
    def test_12_prompt_token_counters_and_over_window(self) -> None:
        os.environ["CARRY_WINDOW_TOKENS"] = "5000"
        agent = self._agent()
        sess = self._session()
        self._turn(agent, sess, [self._reply(X_CODE, "c1", prompt_tokens=4900)], step=1)
        self._turn(agent, sess, [self._reply(X_CODE, "c2", prompt_tokens=5100)], step=2)
        st = cr.status()
        self.assertEqual((st["prompt_tokens_total"], st["prompt_tokens_max"], st["prompt_over_window"]), (10000, 5100, 1))
        self.assertEqual(st["prompt_tokens_per_call"], 5000.0)
        self.assertEqual(st["per_game"]["tu93-test_p0"]["prompt_tokens_max"], 5100)
        # a call outside analyze() is a pass-through
        with mock.patch.dict(cr._STOCK, {"chat_completion": lambda a, m, **k: self._reply(X_CODE, "z", prompt_tokens=1)}):
            agent._chat_completion([{"role": "system", "content": "s"}], tools=None)
        self.assertEqual(cr.status()["calls_total"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
