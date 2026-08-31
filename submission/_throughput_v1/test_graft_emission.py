"""Offline validation for the emission graft (Pack 5). No GPU/network.

Run:  .venv/bin/python submission/_throughput_v1/test_graft_emission.py
"""
from __future__ import annotations

import json
import os
import sys
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

import graft_emission as tem  # noqa: E402
import graft_throughput as tp  # noqa: E402


def _clear_flags() -> None:
    for name in list(os.environ):
        if name.startswith(("TP_", "TP2_", "TP4_", "TP5_")):
            os.environ.pop(name, None)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _completion(message, finish="stop"):
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"total_tokens": 10, "completion_tokens": 5}}


class EmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.utils import openai_compat as compat_mod  # noqa: PLC0415
        cls.agent_mod, cls.compat_mod = agent_mod, compat_mod
        tp.install()
        cls.status = tem.install()

    def setUp(self) -> None:
        _clear_flags()

    def _agent(self):
        return self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def _chat(self, agent, message):
        with mock.patch("requests.post", return_value=_FakeResponse(_completion(message))) as post:
            result = agent._chat_completion([{"role": "system", "content": "s"},
                                             {"role": "user", "content": "u"}], tools=self._tools())
        return result, post

    def _tools(self):
        return [{"type": "function", "function": {"name": "python", "parameters": {}}}]

    def test_01_install(self) -> None:
        self.assertIn(self.status, {"emission: OK", "emission: SKIP (already applied)"})
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._chat_completion, "_tp5_stock"))
        self.assertTrue(hasattr(self.compat_mod.build_chat_payload, "_tp5_stock"))
        self.assertTrue(hasattr(self.agent_mod.build_chat_payload, "_tp5_stock"))

    def test_02_wm_harvested_from_reasoning_when_content_empty(self) -> None:
        agent = self._agent()
        message = {"role": "assistant", "content": "",
                   "reasoning": "Let me think.\nWorld model: door needs key\nPlan: get the key first"}
        self._chat(agent, message)
        self.assertEqual(agent._summarized_knowledge["world_model"], "door needs key")
        self.assertEqual(agent._summarized_knowledge["current_plan"], "get the key first")

    def test_02b_reasoning_intent_line_never_glues(self) -> None:
        # J10 re-verify (iv): a Next: line in the reasoning channel must not be
        # absorbed into stock fields by the labeled-block continuation rule.
        agent = self._agent()
        message = {"role": "assistant", "content": "",
                   "reasoning": "World model: door needs key\nNext: try the red door"}
        self._chat(agent, message)
        wm = agent._summarized_knowledge.get("world_model", "")
        self.assertEqual(wm, "door needs key")
        self.assertNotIn("try the red door", wm)

    def test_03_assistant_channel_wins_when_present(self) -> None:
        agent = self._agent()
        message = {"role": "assistant", "content": "World model: from content",
                   "reasoning": "World model: from reasoning"}
        self._chat(agent, message)
        # the harvest itself happens in analyze(); the graft must NOT override
        # from reasoning when content parses
        self.assertNotEqual(agent._summarized_knowledge.get("world_model"), "from reasoning")

    def test_04_act_floor_forces_tool_choice_and_act_line(self) -> None:
        os.environ["TP5_ACT_FLOOR"] = "2"
        agent = self._agent()
        message = {"role": "assistant", "content": "thinking aloud"}
        # two analysis-only calls build the streak
        self._chat(agent, message)
        self._chat(agent, message)
        result, post = self._chat(agent, message)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload.get("tool_choice"), tem.FORCED_TOOL_CHOICE)
        self.assertIn("without executing any action", payload["messages"][-1]["content"])

    def test_05_streak_resets_on_executed_action(self) -> None:
        os.environ["TP5_ACT_FLOOR"] = "2"
        agent = self._agent()
        agent._tp5_calls_without_action = 5
        fake = mock.Mock()
        fake.step_executed = True
        with mock.patch.dict(tem._RUN_STOCK, {"fn": lambda *a, **k: fake}):
            self.agent_mod.ToolAgent._run_python_tool(agent, Path("/nonexistent"), {"code": ""})
        self.assertEqual(agent._tp5_calls_without_action, 0)
        message = {"role": "assistant", "content": "x"}
        _, post = self._chat(agent, message)
        self.assertNotEqual(post.call_args.kwargs["json"].get("tool_choice"), tem.FORCED_TOOL_CHOICE)

    def test_06_disabled_is_passthrough(self) -> None:
        os.environ["TP5_ENABLE"] = "0"
        agent = self._agent()
        agent._tp5_calls_without_action = 99
        message = {"role": "assistant", "content": "",
                   "reasoning": "World model: should not be harvested"}
        _, post = self._chat(agent, message)
        self.assertNotEqual(agent._summarized_knowledge.get("world_model"), "should not be harvested")
        self.assertNotEqual(post.call_args.kwargs["json"].get("tool_choice"), tem.FORCED_TOOL_CHOICE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
