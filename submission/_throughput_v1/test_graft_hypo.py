"""Offline validation for the hypothesis-enumeration + probe rule (HYPO).

Run:  GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference \
          .venv/bin/python submission/_throughput_v1/test_graft_hypo.py
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

import graft_hypo as hy  # noqa: E402

_GRID = tuple(tuple((r * 7 + c) % 5 for c in range(8)) for r in range(8))


def _reset() -> None:
    os.environ.pop("HYPO_ENABLE", None)
    with hy._LOCK:
        hy._STATE["blocks_injected"] = 0
        hy._STATE["errors"] = 0
        hy._STATE["skips"] = {}
        hy._STATE["per_game"] = {}


class _FakeSession:
    def __init__(self, runtime_mod, state_path: Path, *, game_id="lf52-test"):
        self.runtime_mod = runtime_mod
        self.state_path = state_path
        self.history_entries = [runtime_mod.HistoryEntry(action="", frame=runtime_mod.Frame(grid=_GRID, step=0, level=1))]
        self.game = types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=game_id, state="playing"),
                                          current_state=types.SimpleNamespace(won=False))
        runtime_mod.write_runtime_state(state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)

    def step_env(self, arguments):
        acts = arguments.get("actions") or []
        step = len(self.history_entries)
        self.history_entries.append(self.runtime_mod.HistoryEntry(action="UP", frame=self.runtime_mod.Frame(grid=_GRID, step=step, level=1)))
        self.runtime_mod.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)
        return {"executed": True, "action_num": step, "level": 1, "score": 0, "reward": 0.0, "state": "NOT_FINISHED",
                "valid_actions": ["UP", "DOWN"], "board_changed": False, "done": False, "level_completed": False,
                "game_over": False, "run_complete": False, "action_display": "UP", "executed_actions": ["UP"] * len(acts),
                "requested_count": len(acts), "executed_count": len(acts), "stopped_early": False, "batched": len(acts) > 1}


class HypoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset()
        from inference.agent import runtime_state as runtime_mod
        from inference.agent import tool_agent as agent_mod
        cls.runtime_mod = runtime_mod
        cls.agent_mod = agent_mod
        cls.stock = {"prompt": agent_mod.ToolAgent._build_user_prompt}
        os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "test-model")
        cls.install_status = hy.install()

    def setUp(self):
        _reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / "artifacts" / "lf52-test_p0_tool_runtime_state.json"
        self.state_path.parent.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()
        _reset()

    def _agent(self, sess=None):
        agent = self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
        agent._ensure_session(self.state_path)
        if sess is not None:
            agent._step_env_callback = sess.step_env
        return agent

    def test_01_install(self):
        self.assertEqual(self.install_status, "hypo: OK")
        self.assertTrue(hy._STATE["installed"])
        self.assertIs(self.agent_mod.ToolAgent._build_user_prompt._hypo_stock, self.stock["prompt"])
        self.assertEqual(hy.install(), "hypo: SKIP (already applied)")

    def test_02_flag_off_identical(self):
        os.environ["HYPO_ENABLE"] = "0"
        agent = self._agent(_FakeSession(self.runtime_mod, self.state_path))
        frame = self.runtime_mod.Frame(grid=_GRID, step=3, level=1)
        kwargs = dict(valid_actions=["UP", "DOWN"], current_frame=frame, history_entries=[],
                      previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 1})
        self.assertEqual(agent._build_user_prompt(3, **kwargs), self.stock["prompt"](agent, 3, **kwargs))
        self.assertEqual(hy.status()["blocks_injected"], 0)

    def test_03_block_once_per_prompt_on_uncleared_level(self):
        agent = self._agent(_FakeSession(self.runtime_mod, self.state_path))
        frame = self.runtime_mod.Frame(grid=_GRID, step=3, level=1)
        kwargs = dict(valid_actions=["UP"], current_frame=frame, history_entries=[],
                      previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 1})
        text = agent._build_user_prompt(3, **kwargs)
        stock = self.stock["prompt"](agent, 3, **kwargs)
        self.assertEqual(text, stock + "\n" + hy.block_text())
        self.assertEqual(text.count("[HYPO]"), 1)
        self.assertTrue(text.endswith(hy.block_text()))
        # first turn of a game (no summary) also carries it
        text0 = agent._build_user_prompt(0, valid_actions=["UP"], current_frame=frame, history_entries=[], previous_step_summary=None)
        self.assertEqual(text0.count("[HYPO]"), 1)
        # the turn right after a level clear keeps it (the new level is the next wall)
        text1 = agent._build_user_prompt(9, valid_actions=["UP"], current_frame=frame, history_entries=[],
                                         previous_step_summary={"executed_count": 2, "level": 2, "level_transition": True})
        self.assertEqual(text1.count("[HYPO]"), 1)
        self.assertEqual(hy.status()["blocks_injected"], 3)
        self.assertEqual(hy.status()["per_game"]["lf52-test"], 3)

    def test_04_absent_after_win_or_not_playing(self):
        sess = _FakeSession(self.runtime_mod, self.state_path)
        agent = self._agent(sess)
        frame = self.runtime_mod.Frame(grid=_GRID, step=3, level=1)
        base = dict(valid_actions=["UP"], current_frame=frame, history_entries=[])
        t = agent._build_user_prompt(3, **base, previous_step_summary={"executed_count": 1, "run_complete": True, "level": 3})
        self.assertNotIn("[HYPO]", t)
        sess.game.game_run.state = "won"
        t = agent._build_user_prompt(3, **base, previous_step_summary={"executed_count": 1, "level": 1})
        self.assertNotIn("[HYPO]", t)
        sess.game.game_run.state = "playing"
        sess.game.current_state.won = True
        t = agent._build_user_prompt(3, **base, previous_step_summary={"executed_count": 1, "level": 1})
        self.assertNotIn("[HYPO]", t)
        st = hy.status()
        self.assertEqual(st["blocks_injected"], 0)
        self.assertEqual(st["skips"], {"run_complete": 1, "not_playing": 1, "won": 1})
        # no session at all: the summary decides
        agent2 = self._agent()
        self.assertIn("[HYPO]", agent2._build_user_prompt(3, **base, previous_step_summary={"executed_count": 1, "level": 1}))
        self.assertNotIn("[HYPO]", agent2._build_user_prompt(3, **base, previous_step_summary={"executed_count": 1, "run_complete": True}))

    def test_05_caps_and_content(self):
        text = hy.block_text()
        self.assertLessEqual(len(text), 900)
        self.assertTrue(text.startswith("[HYPO]"))
        for needle in (">=3 candidate", "<=3 actions", "falsified", "re-derive the goal", "action([...])"):
            self.assertIn(needle, text)
        self.assertEqual(hy.status()["block_chars"], len(text))
        with mock.patch.object(hy, "HYPO_BLOCK", "[HYPO] " + "x" * 2000):
            self.assertEqual(len(hy.block_text()), 900)

    def test_06_exception_fallback(self):
        agent = self._agent(_FakeSession(self.runtime_mod, self.state_path))
        frame = self.runtime_mod.Frame(grid=_GRID, step=3, level=1)
        kwargs = dict(valid_actions=["UP"], current_frame=frame, history_entries=[], previous_step_summary=None)
        with mock.patch.object(hy, "should_inject", side_effect=RuntimeError("boom")):
            text = agent._build_user_prompt(3, **kwargs)
        self.assertEqual(text, self.stock["prompt"](agent, 3, **kwargs))
        self.assertEqual(hy.status()["errors"], 1)

    def test_07_analyze_end_to_end_once_per_call(self):
        """Through analyze(): a text-only reply triggers the inline follow-up prompt, which must NOT
        repeat the block; the block is in the transcript's first [USER PROMPT] exactly once."""
        sess = _FakeSession(self.runtime_mod, self.state_path)
        agent = self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
        replies = iter([
            self.agent_mod._ChatCompletionResult(message={"role": "assistant", "content": "World model: unsure."},
                                                 finish_reason="stop", usage={}),
            self.agent_mod._ChatCompletionResult(message={"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "action(['UP'])"})}}]},
                finish_reason="tool_calls", usage={}),
        ])
        transcript = Path(self.tmp.name) / "transcripts" / "lf52-test_p0.txt"
        transcript.parent.mkdir(parents=True)
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=lambda *a, **k: next(replies)):
            result = agent.analyze(self.state_path, 0, valid_actions=["UP", "DOWN"], step_env=sess.step_env,
                                   transcript_path=transcript, analysis_step=1)
        self.assertTrue(result.step_executed)
        text = transcript.read_text()
        self.assertEqual(text.count("[HYPO]"), 1)
        self.assertEqual(text.count("[USER PROMPT]"), 2)          # the initial prompt + the "not acted yet" follow-up
        self.assertEqual(hy.status()["blocks_injected"], 1)
        users = [m for m in agent._history_messages if m.get("role") == "user"]
        self.assertIn("[HYPO]", str(users[0]["content"]))
        self.assertNotIn("[HYPO]", str(users[1]["content"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
