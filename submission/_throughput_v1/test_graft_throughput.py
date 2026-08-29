"""Offline validation for the throughput graft (no GPU/network/Kaggle).

Runs against the anim bundle that actually plays at eval
(submission/_inspect_replay/assets_build/ARC3-Inference, byte-identical in
inference/ to the jakobbrggen/taaf-kaggle-source-anim-20260807-anim dataset).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_throughput.py
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

import graft_throughput as tp  # noqa: E402

FLAGS = ("TP_ENABLE", "TP_TRIM_LOW_WATER", "TP_CONTEXT_WINDOW", "TP_YIELD_SECONDS",
         "TP_TOOL_STEPS", "TP_KEEP_NOTES_ON_GAME_OVER", "TP_BATCH_CAP")


def _clear_flags() -> None:
    for name in FLAGS:
        os.environ.pop(name, None)


class _AgentMixin:
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        cls.install_status = tp.install()

    def setUp(self) -> None:
        _clear_flags()

    def _agent(self):
        return self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")

    @staticmethod
    def _turn(i: int, chars: int = 600) -> list[dict]:
        return [
            {"role": "user", "content": f"u{i} " + ("x" * chars)},
            {"role": "assistant", "content": f"a{i} " + ("y" * chars),
             "tool_calls": [{"id": f"c{i}", "function": {"name": "python", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "t" * chars},
        ]

    def _messages(self, turns: int) -> list[dict]:
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(turns):
            msgs.extend(self._turn(i))
        return msgs


class InstallTests(_AgentMixin, unittest.TestCase):
    def test_01_install_ok_then_idempotent(self) -> None:
        # Other test classes may have installed first (alphabetical order).
        self.assertIn(self.install_status, {"throughput: OK", "throughput: SKIP (already applied)"})
        self.assertTrue(tp._STATE["installed"])
        self.assertEqual(tp.install(), "throughput: SKIP (already applied)")

    def test_02_defaults(self) -> None:
        self.assertTrue(tp.enabled())
        self.assertEqual(tp.trim_low_water(), 0.5)
        self.assertEqual(tp.context_window(), 24576)
        self.assertEqual(tp.yield_seconds(), 900.0)
        self.assertEqual(tp.tool_steps(), 8)
        self.assertTrue(tp.keep_notes_on_game_over())
        self.assertEqual(tp.batch_cap(), 10)

    def test_03_master_switch_off(self) -> None:
        os.environ["TP_ENABLE"] = "0"
        self.assertFalse(tp.enabled())
        self.assertEqual(tp.trim_low_water(), 1.0)
        self.assertEqual(tp.context_window(), 0)
        self.assertEqual(tp.yield_seconds(), -1.0)
        self.assertEqual(tp.tool_steps(), -1)
        self.assertFalse(tp.keep_notes_on_game_over())
        self.assertEqual(tp.batch_cap(), 0)

    def test_04_every_seam_wrapped_with_stock_handle(self) -> None:
        cls = self.agent_mod.ToolAgent
        for name in ("_trim_messages_for_context", "__init__",
                     "_update_summarized_knowledge_from_step_summary",
                     "_normalize_python_actions", "_run_python_tool"):
            self.assertTrue(hasattr(getattr(cls, name), "_tp_stock"), name)


class TrimTests(_AgentMixin, unittest.TestCase):
    def test_10_under_budget_is_untouched(self) -> None:
        agent = self._agent()
        msgs = self._messages(2)
        self.assertEqual(agent._trim_messages_for_context(msgs), msgs)

    def test_11_over_budget_cuts_to_low_water(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = 4000
        msgs = self._messages(12)
        out = agent._trim_messages_for_context(msgs)
        est = agent._estimate_request_input_tokens(out)
        self.assertLessEqual(est, 2000 + 700)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["role"], "user")
        self.assertEqual(out[-1], msgs[-1])

    def test_12_stable_prefix_across_turns(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = 4000
        msgs = agent._trim_messages_for_context(self._messages(12))
        first_user = msgs[1]
        stable_checks = 0
        for i in range(12, 15):
            msgs = agent._trim_messages_for_context(msgs + self._turn(i))
            if agent._estimate_request_input_tokens(msgs) <= 4000 and msgs[1] is first_user:
                stable_checks += 1
        self.assertGreaterEqual(stable_checks, 1)

    def test_13_stock_when_disabled(self) -> None:
        os.environ["TP_ENABLE"] = "0"
        agent = self._agent()
        agent._context_budget_tokens = 4000
        out = agent._trim_messages_for_context(self._messages(12))
        est = agent._estimate_request_input_tokens(out)
        self.assertGreater(est, 3000)
        self.assertLessEqual(est, 4000)

    def test_14_preserve_recent_and_extra_safety_honoured(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = 4000
        msgs = self._messages(12)
        out = agent._trim_messages_for_context(msgs, preserve_recent=3, extra_safety_tokens=1000)
        self.assertLessEqual(agent._estimate_request_input_tokens(out), 3000)
        self.assertEqual(out[-3:], msgs[-3:])


class InitOverrideTests(_AgentMixin, unittest.TestCase):
    def test_20_defaults_applied(self) -> None:
        agent = self._agent()
        self.assertEqual(
            agent._context_budget_tokens,
            24576 - agent._reply_reserve_tokens - agent._request_safety_margin_tokens,
        )
        self.assertEqual(agent._yield_seconds, 900.0)
        self.assertEqual(agent._tool_steps, 8)

    def test_21_stock_when_disabled(self) -> None:
        os.environ["TP_ENABLE"] = "0"
        agent = self._agent()
        stock_window = self.agent_mod._LOCAL_ANALYZER_CONTEXT_WINDOW
        self.assertEqual(
            agent._context_budget_tokens,
            max(1024, stock_window - agent._reply_reserve_tokens - agent._request_safety_margin_tokens),
        )
        stock_yield = self.agent_mod._LOCAL_ANALYZER_YIELD_SECONDS
        self.assertEqual(agent._yield_seconds, None if stock_yield <= 0 else float(stock_yield))

    def test_22_zero_tool_steps_means_unlimited(self) -> None:
        os.environ["TP_TOOL_STEPS"] = "0"
        self.assertIsNone(self._agent()._tool_steps)

    def test_23_yield_zero_disables(self) -> None:
        os.environ["TP_YIELD_SECONDS"] = "0"
        self.assertIsNone(self._agent()._yield_seconds)


class NotesTests(_AgentMixin, unittest.TestCase):
    def _agent_with_notes(self):
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "walls block moves"
        agent._summarized_knowledge["current_plan"] = "go right"
        agent._summarized_knowledge["cross_level_notes"] = "keep"
        return agent

    def test_30_game_over_keeps_notes(self) -> None:
        agent = self._agent_with_notes()
        agent._last_step_summary = {"game_over": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "walls block moves")
        self.assertEqual(agent._summarized_knowledge["current_plan"], "go right")

    def test_31_level_transition_still_wipes(self) -> None:
        agent = self._agent_with_notes()
        agent._last_step_summary = {"level_transition": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")
        self.assertEqual(agent._summarized_knowledge["cross_level_notes"], "keep")

    def test_32_stock_when_disabled(self) -> None:
        os.environ["TP_KEEP_NOTES_ON_GAME_OVER"] = "0"
        agent = self._agent_with_notes()
        agent._last_step_summary = {"game_over": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")


class BatchCapTests(_AgentMixin, unittest.TestCase):
    def test_40_cap_truncates_within_one_tool_call(self) -> None:
        agent = self._agent()
        tp.begin_tool_call()
        out = agent._normalize_python_actions(["UP"] * 25)
        self.assertEqual(len(out), 10)
        with self.assertRaises(ValueError) as ctx:
            agent._normalize_python_actions(["UP"])
        self.assertIn("action batch cap reached", str(ctx.exception))

    def test_41_budget_resets_per_tool_call(self) -> None:
        agent = self._agent()
        tp.begin_tool_call()
        agent._normalize_python_actions(["UP"] * 10)
        tp.begin_tool_call()
        self.assertEqual(len(agent._normalize_python_actions(["DOWN"] * 3)), 3)

    def test_42_unlimited_when_disabled(self) -> None:
        os.environ["TP_BATCH_CAP"] = "0"
        agent = self._agent()
        tp.begin_tool_call()
        self.assertEqual(len(agent._normalize_python_actions(["UP"] * 25)), 25)

    def test_43_run_python_tool_resets_budget(self) -> None:
        agent = self._agent()
        tp.begin_tool_call()
        agent._normalize_python_actions(["UP"] * 10)
        agent._run_python_tool(Path("/nonexistent/state.json"), {"code": ""})
        self.assertEqual(len(agent._normalize_python_actions(["UP"] * 2)), 2)


class TimeGuardTests(unittest.TestCase):
    def test_50_no_change_when_margin_holds(self) -> None:
        self.assertEqual(
            tp.time_guard_per_game_s(7920.0, setup_elapsed_s=630.0, games=110, concurrency=28,
                                     margin_s=0.0), 7920.0)

    def test_50b_default_margin_removes_the_90s_cliff(self) -> None:
        # R4: 4 x 7920 + 630 s setup = 32,310 s, only 90 s under the 9 h box.
        got = tp.time_guard_per_game_s(7920.0, setup_elapsed_s=630.0, games=110, concurrency=28)
        self.assertAlmostEqual(got, (32400.0 - 630.0 - 240.0) / 4, places=1)

    def test_51_shrinks_when_setup_ate_margin(self) -> None:
        got = tp.time_guard_per_game_s(7920.0, setup_elapsed_s=1200.0, games=110, concurrency=28)
        self.assertLess(got, 7920.0)
        self.assertAlmostEqual(got, (32400.0 - 1200.0 - 240.0) / 4, places=1)

    def test_52_never_grows(self) -> None:
        self.assertEqual(
            tp.time_guard_per_game_s(3600.0, setup_elapsed_s=0.0, games=4, concurrency=28), 3600.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
