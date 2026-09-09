"""Offline validation for the persistent workspace + transition log + verifier (WS / A2).

Run:  GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference \
          .venv/bin/python submission/_throughput_v1/test_graft_workspace.py

The python tool runs in the REAL sandbox subprocess and the verifier in its own
subprocess; only the model is faked.
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
_BUNDLE = Path(os.environ.get("GRAFT_TEST_BUNDLE",
                              str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference")))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_workspace as ws  # noqa: E402

_FLAGS = ("WS_ENABLE", "WS_MAX_FILES", "WS_MAX_FILE_CHARS", "WS_BACKTEST_TIMEOUT", "WS_LOG_MAX",
          "WS_PREAMBLE", "WS_PREAMBLE_MAX_CHARS", "WS_MISMATCH_CELLS", "WS_DIRECT_ENABLE",
          "WS_DIRECT_AFTER_ACTIONS", "WS_DIRECT_MAX_CALLS", "WS_DIRECT_MAX_PER_GAME", "WS_DIRECT_MAX_TRANSITIONS")
_GRID = tuple(tuple(0 for _ in range(8)) for _ in range(8))
X_CODE = "r = action(['UP'])\nprint('acted', r.get('executed'))\n"


def _clear():
    for f in _FLAGS:
        os.environ.pop(f, None)


def _reset():
    with ws._LOCK:
        for k in ws._PER_GAME_KEYS:
            ws._STATE[k] = 0
        ws._STATE["errors"] = 0
        ws._STATE["skips"] = {}
        ws._STATE["per_game"] = {}


class _FakeSession:
    """step_env moves a '5' marker right by one column per UP; grids are real tuples."""

    def __init__(self, runtime_mod, state_path, game_id="dc22-test"):
        self.runtime_mod = runtime_mod
        self.state_path = state_path
        self.game = types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=game_id, state="playing"),
                                          current_state=types.SimpleNamespace(won=False, levels_completed=0))
        self.pos = 0
        self.history_entries = [runtime_mod.HistoryEntry(action="", frame=runtime_mod.Frame(grid=self._grid(), step=0, level=1))]
        self.executed = []
        self._write()

    def _grid(self):
        return tuple(tuple(5 if (r == 0 and c == self.pos) else 0 for c in range(8)) for r in range(8))

    def _write(self):
        self.runtime_mod.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame,
                                             history=self.history_entries)

    def step_env(self, arguments):
        names = [str(a.get("action", "")).upper() for a in (arguments.get("actions") or [])]
        for nm in names:
            self.pos = min(7, self.pos + 1)
            self.executed.append(nm)
            self.history_entries.append(self.runtime_mod.HistoryEntry(
                action=nm, frame=self.runtime_mod.Frame(grid=self._grid(), step=len(self.history_entries), level=1)))
        self._write()
        return {"executed": True, "action_num": len(self.history_entries) - 1, "level": 1, "score": 0, "reward": 0.0,
                "state": "NOT_FINISHED", "valid_actions": ["UP"], "board_changed": True, "done": False,
                "level_completed": False, "game_over": False, "run_complete": False, "action_display": names[-1],
                "action_name": names[-1], "executed_actions": names, "requested_count": len(names),
                "executed_count": len(names), "stopped_early": False}


class WsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.agent import runtime_state as runtime_mod  # noqa: PLC0415
        cls.agent_mod, cls.runtime_mod = agent_mod, runtime_mod
        cls.status = ws.install()

    def setUp(self):
        _clear(); _reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "artifacts" / "dc22-test_p0_state.json"
        self.transcript = self.root / "transcripts" / "dc22-test_p0.txt"
        self.state_path.parent.mkdir(parents=True)
        self.transcript.parent.mkdir(parents=True)
        self.transcript.touch()

    def tearDown(self):
        self.tmp.cleanup()

    def _agent(self):
        return self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def _armed(self, agent, sess):
        agent._ensure_session(self.state_path)
        agent._step_env_callback = sess.step_env
        agent._current_valid_actions = ["UP"]
        st = ws._wstate(agent, self.state_path)
        st.transcript_path = self.transcript      # in production analyze() sets this from the solver's kwarg
        return st

    # 01 -----------------------------------------------------------------
    def test_01_install(self):
        self.assertIn(self.status, {"workspace: OK", "workspace: SKIP (already applied)"})
        for n in ("analyze", "_tools", "_dispatch_tool", "_run_python_tool", "_compact_action_result",
                  "_build_user_prompt"):
            self.assertTrue(hasattr(getattr(self.agent_mod.ToolAgent, n), "_ws_stock"), n)

    def test_01b_analyze_supplies_the_real_transcript_path(self):
        """Regression: the runtime stem is '<run>_tool_runtime_state', so guessing the transcript path
        from state_path silently wrote every marker into the wrong file (found in the first dry run)."""
        agent = self._agent()
        sess = _FakeSession(self.runtime_mod, self.state_path)
        reply = self.agent_mod._ChatCompletionResult(message={"role": "assistant", "content": "hi"},
                                                     finish_reason="stop", usage={})
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", return_value=reply):
            agent.analyze(self.state_path, 0, valid_actions=["UP"], step_env=sess.step_env,
                          transcript_path=self.transcript, analysis_step=1)
        self.assertEqual(agent._ws.transcript_path, self.transcript)
        self.assertEqual(agent._ws.game, "dc22-test_p0")   # run stem: run_regime_wave keys per-run telemetry by it

    def test_01c_prompt_matches_the_tool_schema(self):
        """The first live kill test got 181 python calls and ZERO backtest/workspace calls because the stock
        prompts assert python is the only tool - in the system prompt once and in the USER prompt every turn
        (44x in a 60-call run). With WS on, both must name all three tools; with WS off, both stay stock."""
        agent = self._agent()
        self.assertNotIn("The only tool is", agent._system_prompt)
        self.assertIn("Three tools:", agent._system_prompt)
        up = agent._build_user_prompt(3, valid_actions=["UP"])
        self.assertNotIn("Only tool: `python`.", up)
        self.assertIn("backtest", up)
        self.assertIn("workspace", up)
        self.assertIn("`python`. It receives", up)      # the stock sentence that follows still reads correctly
        os.environ["WS_ENABLE"] = "0"
        stock_up = agent._build_user_prompt(3, valid_actions=["UP"])
        self.assertIn("Only tool: `python`.", stock_up)
        self.assertEqual(ws.status()["errors"], 0)

    # 02 -----------------------------------------------------------------
    def test_02_flag_off_is_stock(self):
        os.environ["WS_ENABLE"] = "0"
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        self._armed(agent, sess)
        tools = agent._tools(self.state_path)
        self.assertEqual([t["function"]["name"] for t in tools], ["python"])
        r = agent._run_python_tool(self.state_path, {"code": X_CODE})
        self.assertTrue(r.step_executed)
        self.assertEqual(ws.status()["python_calls"], 0)
        self.assertEqual(ws.status()["transitions_logged"], 0)
        d = agent._dispatch_tool(self.state_path, "backtest", {"code": "x"})
        self.assertIn("Unknown tool", d.content)          # stock behaviour for an unknown name

    # 03 -----------------------------------------------------------------
    def test_03_tools_exposed(self):
        agent = self._agent()
        names = [t["function"]["name"] for t in agent._tools(self.state_path)]
        self.assertEqual(names, ["python", "backtest", "workspace"])
        bt = [t for t in agent._tools(self.state_path) if t["function"]["name"] == "backtest"][0]
        self.assertEqual(bt["function"]["parameters"]["required"], ["code"])
        self.assertIn("init_state", bt["function"]["description"])
        self.assertIn("stdlib", bt["function"]["description"])     # sandbox caveat is stated

    # 04 -----------------------------------------------------------------
    def test_04_transition_log_captured(self):
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        st = self._armed(agent, sess)
        for _ in range(3):
            agent._run_python_tool(self.state_path, {"code": X_CODE})
        self.assertEqual(len(st.log), 3)
        self.assertEqual(sess.executed, ["UP"] * 3)
        t = st.log[0]
        self.assertEqual((t["action"], t["level_before"], t["level_up"], t["dead"]), (-1, 1, False, False))
        self.assertEqual(len(t["grid"]), 8)
        self.assertTrue(all(x["changed_cells"] == 2 for x in st.log))   # marker moved: 2 cells differ
        self.assertIn(1, st.entry_by_level)
        self.assertEqual(ws.status()["transitions_logged"], 3)

    # 05 -----------------------------------------------------------------
    def test_05_backtest_tool_end_to_end(self):
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        st = self._armed(agent, sess)
        for _ in range(3):
            agent._run_python_tool(self.state_path, {"code": X_CODE})
        good = ("def step(grid, action, x, y):\n"
                "    g = [list(r) for r in grid]\n"
                "    for c in range(8):\n"
                "        if g[0][c] == 5:\n"
                "            g[0][c] = 0\n"
                "            g[0][min(7, c + 1)] = 5\n"
                "            break\n"
                "    return g, {}\n")
        d = agent._dispatch_tool(self.state_path, "backtest", {"code": good})
        res = json.loads(d.content)
        self.assertEqual((res["matched"], res["total"], res["green"]), (3, 3, True))
        self.assertIn("note", res)
        self.assertFalse(d.step_executed)
        bad = "def step(grid, action, x, y):\n    return grid, {}\n"
        res2 = json.loads(agent._dispatch_tool(self.state_path, "backtest", {"code": bad}).content)
        self.assertFalse(res2["green"])
        self.assertEqual(res2["first_mismatch"]["kind"], "grid")
        self.assertEqual(res2["best_so_far"], "3/3")            # best is remembered across calls
        text = self.transcript.read_text()
        self.assertIn("[WS-BACKTEST] game=dc22-test_p0 level=1 matched=3/3 green=1", text)
        s = ws.status()
        self.assertEqual((s["backtests"], s["backtests_green"]), (2, 1))

    # 06 -----------------------------------------------------------------
    def test_06_workspace_ops_and_caps(self):
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        self._armed(agent, sess)
        d = lambda a: json.loads(agent._dispatch_tool(self.state_path, "workspace", a).content)  # noqa: E731
        self.assertEqual(d({"op": "save", "name": "m.py", "content": "print(1)"})["saved"], "m.py")
        self.assertEqual(d({"op": "load", "name": "m.py"})["content"], "print(1)")
        self.assertEqual(d({"op": "list"})["files"], {"m.py": 8})
        self.assertIn("error", d({"op": "load", "name": "nope"}))
        self.assertIn("error", d({"op": "save", "name": "x", "content": "y" * 999999}))
        self.assertIn("error", d({"op": "bogus"}))
        self.assertEqual(d({"op": "delete", "name": "m.py"})["files"], [])
        os.environ["WS_MAX_FILES"] = "1"
        d({"op": "save", "name": "a", "content": "1"})
        self.assertIn("error", d({"op": "save", "name": "b", "content": "2"}))
        self.assertIn("[WS-SAVE] game=dc22-test_p0 name=m.py chars=8", self.transcript.read_text())

    # 07 -----------------------------------------------------------------
    def test_07_preamble_injected(self):
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        self._armed(agent, sess)
        agent._dispatch_tool(self.state_path, "workspace", {"op": "save", "name": "note.py", "content": "SENTINEL=42"})
        agent._run_python_tool(self.state_path, {"code": X_CODE})          # builds the log
        seen = {}
        stock = ws._STOCK["run_python_tool"]

        def spy(self_, sp, args):
            seen["code"] = args.get("code", "")
            return stock(self_, sp, args)
        with mock.patch.dict(ws._STOCK, {"run_python_tool": spy}):
            agent._run_python_tool(self.state_path, {"code": "print('body')\n"})
        self.assertIn("WORKSPACE = ", seen["code"])
        self.assertIn("SENTINEL=42", seen["code"])
        self.assertIn("TRANSITIONS = ", seen["code"])
        self.assertTrue(seen["code"].rstrip().endswith("print('body')"))
        # the injected literals are usable inside the REAL sandbox
        r = agent._run_python_tool(self.state_path, {"code": "print('files', sorted(WORKSPACE), 'n', len(TRANSITIONS))\n"})
        self.assertIn("files ['note.py']", r.content)
        os.environ["WS_PREAMBLE"] = "0"
        with mock.patch.dict(ws._STOCK, {"run_python_tool": spy}):
            agent._run_python_tool(self.state_path, {"code": "print('body2')\n"})
        self.assertNotIn("WORKSPACE = ", seen["code"])

    # 08 -----------------------------------------------------------------
    def test_08_new_game_resets(self):
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        st = self._armed(agent, sess)
        agent._dispatch_tool(self.state_path, "workspace", {"op": "save", "name": "a", "content": "1"})
        agent._run_python_tool(self.state_path, {"code": X_CODE})
        self.assertTrue(st.files and st.log)
        root2 = self.root / "g2"
        (root2 / "artifacts").mkdir(parents=True); (root2 / "transcripts").mkdir(parents=True)
        sp2 = root2 / "artifacts" / "vc33-test_p0_state.json"
        sess2 = _FakeSession(self.runtime_mod, sp2, game_id="vc33-test")
        agent._ensure_session(sp2); agent._step_env_callback = sess2.step_env
        st2 = ws._wstate(agent, sp2)
        self.assertEqual((st2.files, st2.log), ({}, []))

    # 09 -----------------------------------------------------------------
    def test_09_exception_safety(self):
        agent = self._agent(); sess = _FakeSession(self.runtime_mod, self.state_path)
        self._armed(agent, sess)
        with mock.patch.object(ws, "build_preamble", side_effect=RuntimeError("bang")), \
                mock.patch.object(ws, "action_to_int", side_effect=RuntimeError("bang")):
            agent._dispatch_tool(self.state_path, "workspace", {"op": "save", "name": "a", "content": "1"})
            r = agent._run_python_tool(self.state_path, {"code": X_CODE})
        self.assertTrue(r.step_executed)                        # the turn still acts
        self.assertGreater(ws.status()["errors"], 0)
        with mock.patch.object(ws, "run_backtest", side_effect=RuntimeError("bang")):
            res = json.loads(agent._dispatch_tool(self.state_path, "backtest", {"code": "def step(g,a,x,y): return g,{}"}).content)
        self.assertEqual(res["error"], "RuntimeError")

    # 10 -----------------------------------------------------------------
    def test_10_status_and_env(self):
        s = ws.status()
        for k in ws._PER_GAME_KEYS:
            self.assertIn(k, s)
        for k in ("installed", "enabled", "max_files", "backtest_timeout", "green_share", "errors", "skips", "per_game"):
            self.assertIn(k, s)
        os.environ.update({"WS_MAX_FILES": "garbage", "WS_LOG_MAX": "-4", "WS_ENABLE": "off"})
        self.assertEqual(ws.max_files(), 12)
        self.assertEqual(ws.log_max(), 1)
        self.assertFalse(ws.enabled())
        self.assertEqual(ws.action_to_int({"action_name": "ACTION6", "action_display": "MOUSE(row=21, col=19)"}), (6, 19, 21))
        self.assertEqual(ws.action_to_int({"action_name": "ACTION3"}), (3, None, None))
        self.assertEqual(ws.action_to_int({"action_name": "RESET"}), (0, None, None))

class DirectTests(unittest.TestCase):
    """WS_DIRECT_*: the harness runs the Stage-1 procedure itself when a run is stuck at a wall."""

    @classmethod
    def setUpClass(cls):
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.agent import runtime_state as runtime_mod  # noqa: PLC0415
        cls.agent_mod, cls.runtime_mod = agent_mod, runtime_mod
        ws.install()

    def setUp(self):
        _clear(); _reset()
        os.environ["WS_DIRECT_ENABLE"] = "1"
        os.environ["WS_DIRECT_AFTER_ACTIONS"] = "3"
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "artifacts" / "dc22-test_p0_state.json"
        self.transcript = self.root / "transcripts" / "dc22-test_p0.txt"
        self.state_path.parent.mkdir(parents=True); self.transcript.parent.mkdir(parents=True)
        self.transcript.touch()

    def tearDown(self):
        self.tmp.cleanup()

    def _stuck_agent(self, n=4):
        agent = self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
        sess = _FakeSession(self.runtime_mod, self.state_path)
        agent._ensure_session(self.state_path); agent._step_env_callback = sess.step_env
        st = ws._wstate(agent, self.state_path)
        st.transcript_path = self.transcript; st.game = "dc22-test_p0"; st.agent_mod = self.agent_mod
        st.entry_by_level[1] = [list(r) for r in sess._grid()]
        for i in range(n):                              # marker at (0,i) -> (0,i+1), the mechanic to model
            before = [list(r) for r in sess._grid()]
            sess.pos += 1
            st.log.append({"index": i, "action": 1, "x": None, "y": None,
                           "grid": [list(r) for r in sess._grid()], "level_up": False, "dead": False,
                           "win": False, "level_before": 1, "changed_cells": 2})
        return agent, st

    def _reply(self, text):
        return self.agent_mod._ChatCompletionResult(message={"role": "assistant", "content": text},
                                                    finish_reason="stop", usage={})

    GOOD = ("```python\ndef step(grid, action, x, y):\n    g = [list(r) for r in grid]\n"
            "    for c in range(8):\n        if g[0][c] == 5:\n            g[0][c] = 0\n"
            "            g[0][min(7, c + 1)] = 5\n            break\n    return g, {}\n```")
    BAD = "```python\ndef step(grid, action, x, y):\n    return grid, {}\n```"

    def test_direct_builds_verifies_saves_and_notifies(self):
        agent, st = self._stuck_agent()
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", return_value=self._reply(self.GOOD)):
            out = ws.run_directed_build(agent, self.agent_mod, st, 1)
        self.assertTrue(out["green"] and out["saved"])
        self.assertEqual(out["calls"], 1)                       # green on the first attempt: no wasted calls
        self.assertIn(ws.DIRECT_MODEL_FILE, st.files)
        self.assertIn("def step", st.files[ws.DIRECT_MODEL_FILE])
        self.assertIn("VERIFIED", st.direct_notice)
        self.assertIn(ws.DIRECT_MODEL_FILE, st.direct_notice)
        text = self.transcript.read_text()
        self.assertIn("[WS-DIRECT] game=dc22-test_p0 level=1 attempt=1 matched=4/4 green=1", text)
        s = ws.status()
        self.assertEqual((s["direct_green"], s["direct_attempts"], s["backtests_green"]), (1, 1, 1))
        # the notice rides exactly the next user prompt, once
        up1 = agent._build_user_prompt(3, valid_actions=["UP"])
        self.assertIn("VERIFIED", up1)
        self.assertNotIn("VERIFIED", agent._build_user_prompt(4, valid_actions=["UP"]))

    def test_direct_iterates_on_a_wrong_model_then_gives_up_cleanly(self):
        agent, st = self._stuck_agent()
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", return_value=self._reply(self.BAD)):
            out = ws.run_directed_build(agent, self.agent_mod, st, 1)
        self.assertFalse(out["green"])
        self.assertEqual(out["calls"], ws.direct_max_calls())    # used its budget, no more
        self.assertNotIn(ws.DIRECT_MODEL_FILE, st.files)
        self.assertEqual(st.direct_notice, "")                  # nothing claimed to the play loop
        self.assertIn(1, st.direct_done)                        # never retried on this level
        self.assertEqual(ws.status()["errors"], 0)

    def test_direct_survives_a_model_that_returns_no_code_or_raises(self):
        agent, st = self._stuck_agent()
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", return_value=self._reply("no fence here")):
            out = ws.run_directed_build(agent, self.agent_mod, st, 1)
        self.assertFalse(out["green"]) and self.assertEqual(out["calls"], ws.direct_max_calls())
        self.assertIn("no_code", self.transcript.read_text())
        _reset()
        agent2, st2 = self._stuck_agent()
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=RuntimeError("boom")):
            out2 = ws.run_directed_build(agent2, self.agent_mod, st2, 1)
        self.assertFalse(out2["green"])
        self.assertIn("call_failed=RuntimeError", self.transcript.read_text())
        self.assertEqual(ws.status()["errors"], 0)              # a failed model call is not a graft error

    def test_direct_is_off_by_default_and_budget_capped(self):
        _clear()
        self.assertFalse(ws.direct_enabled())                   # opt-in: stock arms are untouched
        os.environ.update({"WS_DIRECT_ENABLE": "1", "WS_DIRECT_MAX_PER_GAME": "0"})
        agent, st = self._stuck_agent()
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", return_value=self._reply(self.GOOD)):
            agent.analyze(self.state_path, 0, valid_actions=["UP"], step_env=agent._step_env_callback,
                          transcript_path=self.transcript, analysis_step=1)
        self.assertEqual(ws.status()["direct_attempts"], 0)     # cap 0 => never fires

    def test_render_transitions_is_changed_cells_not_full_grids(self):
        entry = [[0, 0], [0, 0]]
        trans = [{"index": 0, "action": 6, "x": 1, "y": 0, "grid": [[0, 5], [0, 0]],
                  "level_up": False, "dead": False, "win": False}]
        text = ws.render_transitions(entry, trans)
        self.assertIn("ENTRY GRID (2 rows x 2 cols)", text)
        self.assertIn("HEX DIGIT", text)
        self.assertIn("\n00\n00\n", text)                        # hex rows, one char per cell
        # the encoding must stay cheap: space-separated ints put a live prompt at 27,999 tokens and left
        # 4,769 for output, so no model could be emitted. A full 64x64 grid + 24 transitions must stay small.
        big = [[(r * 7 + c) % 16 for c in range(64)] for r in range(64)]
        many = [{"index": i, "action": 1, "x": None, "y": None, "grid": big, "level_up": False,
                 "dead": False, "win": False} for i in range(24)]
        self.assertLess(len(ws.render_transitions(big, many)), 12000)
        self.assertIn("[0] action=6 at x=1 y=0 changed: r0c1:0->5", text)
        self.assertNotIn("5\n0 0", text.split("TRANSITIONS")[1])   # the after-grid is not dumped in full
        t2 = [{"index": 0, "action": 1, "x": None, "y": None, "grid": entry, "level_up": True, "dead": False, "win": False}]
        self.assertIn("flags=level_up", ws.render_transitions(entry, t2))
        self.assertIn("(no cell changed)", ws.render_transitions(entry, t2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
