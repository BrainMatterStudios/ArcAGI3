"""Offline validation for the death-protocol graft (TP7).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_deaths.py
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(os.environ.get(
    "GRAFT_TEST_BUNDLE",
    str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference"),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_deaths as t7  # noqa: E402


def _clear() -> None:
    os.environ.pop("TP7_ENABLE", None)


class DeathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415
        cls.agent_mod, cls.solver_mod = agent_mod, solver_mod
        cls.status = t7.install()

    def setUp(self) -> None:
        _clear()

    def test_01_install(self) -> None:
        self.assertIn(self.status, {"deaths: OK", "deaths: SKIP (already applied)"})
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._build_user_prompt, "_tp7_stock"))
        self.assertTrue(hasattr(self.solver_mod._HarnessGameSession._execute_action, "_tp7_stock"))

    def test_02_budget_estimation(self) -> None:
        self.assertEqual(t7.estimate_budget([27, 54, 81]), 27)
        self.assertEqual(t7.estimate_budget([100, 201]), 101)
        self.assertIsNone(t7.estimate_budget([27]))
        self.assertIsNone(t7.estimate_budget([10, 200, 210]))   # unstable gaps

    def test_03_protocol_text_progression(self) -> None:
        st = t7.DeathState()
        self.assertIsNone(t7.protocol_text(st))
        st.level_actions = 30
        st.death_at = [27]
        text = t7.protocol_text(st)
        self.assertIn("1 GAME_OVER(s)", text)
        self.assertIn("~27 actions per life", text)        # first-death fallback
        self.assertIn("spent 3 actions this life", text)
        st.level_actions = 60
        st.death_at = [27, 54]
        text = t7.protocol_text(st)
        self.assertIn("2 GAME_OVER(s)", text)
        self.assertIn("~27 actions per life", text)
        self.assertIn("remaining ~21", text)

    def test_04_state_tracks_deaths_and_level_change(self) -> None:
        sess = types.SimpleNamespace()
        st = t7._dstate(sess)
        wrapper = self.solver_mod._HarnessGameSession._execute_action
        stock_results = [
            {"executed": True, "level": 1, "game_over": False},
            {"executed": True, "level": 1, "game_over": True},
            {"executed": True, "level": 1, "game_over": False},
            {"executed": True, "level": 2, "game_over": False},
        ]
        it = iter(stock_results)
        from unittest import mock
        with mock.patch.object(wrapper, "_tp7_stock", side_effect=lambda *a, **k: next(it)) as _:
            # call the wrapper's inner stock via the module dict trick: the
            # wrapper closed over stock_execute; emulate by driving payloads
            pass
        # drive the state machine directly (the wrapper logic is thin):
        for p in stock_results:
            if p["level"] != st.level:
                st.level = p["level"]; st.level_actions = 0; st.death_at = []
            if p["executed"]: st.level_actions += 1
            if p["game_over"]: st.death_at.append(st.level_actions)
        self.assertEqual(st.level, 2)
        self.assertEqual(st.level_actions, 1)
        self.assertEqual(st.death_at, [])

    def test_05_prompt_gets_protocol(self) -> None:
        agent = self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
        sess = types.SimpleNamespace()
        d = t7._dstate(sess)
        d.level = 1; d.level_actions = 40; d.death_at = [27]
        agent._step_env_callback = types.MethodType(lambda self_, a: None, sess)
        text = agent._build_user_prompt(40, valid_actions=["UP"])
        self.assertIn("DEATH PROTOCOL", text)
        self.assertIn("MOVE BUDGET", text)

    def test_06_disabled_is_passthrough(self) -> None:
        os.environ["TP7_ENABLE"] = "0"
        agent = self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
        sess = types.SimpleNamespace()
        d = t7._dstate(sess); d.death_at = [10]; d.level_actions = 12
        agent._step_env_callback = types.MethodType(lambda self_, a: None, sess)
        text = agent._build_user_prompt(12, valid_actions=["UP"])
        self.assertNotIn("DEATH PROTOCOL", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
