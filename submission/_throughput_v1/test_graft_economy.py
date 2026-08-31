"""Offline validation for the economy graft (TP6).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_economy.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(os.environ.get(
    "GRAFT_TEST_BUNDLE",
    str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference"),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_economy as t6  # noqa: E402


def _clear() -> None:
    os.environ.pop("TP6_ENABLE", None)


class EconomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        cls.status = t6.install()

    def setUp(self) -> None:
        _clear()

    def _agent(self):
        return self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def test_01_install(self) -> None:
        self.assertIn(self.status, {"economy: OK", "economy: SKIP (already applied)"})
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._build_user_prompt, "_tp6_stock"))

    def test_02_block_appended_with_counter(self) -> None:
        agent = self._agent()
        agent._last_step_summary = {"end_action_num": 37, "executed_count": 2, "level": 1}
        text = agent._build_user_prompt(37, valid_actions=["UP"])
        self.assertIn("ACTION ECONOMY", text)
        self.assertIn("You have executed 37 actions", text)
        self.assertIn("one action([...]) batch", text)

    def test_03_disabled_is_passthrough(self) -> None:
        os.environ["TP6_ENABLE"] = "0"
        agent = self._agent()
        text = agent._build_user_prompt(5, valid_actions=["UP"])
        self.assertNotIn("ACTION ECONOMY", text)

    def test_04_fail_open_on_bad_summary(self) -> None:
        agent = self._agent()
        agent._last_step_summary = {"end_action_num": object()}
        text = agent._build_user_prompt(9, valid_actions=["UP"])
        self.assertIn("ACTION ECONOMY", text)   # falls back to action_num


if __name__ == "__main__":
    unittest.main(verbosity=2)
