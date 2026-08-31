"""Offline validation for the durable-timeout graft (TP8).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_durable.py
"""
from __future__ import annotations

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

import graft_durable as t8  # noqa: E402


def _clear() -> None:
    os.environ.pop("TP8_ENABLE", None)


class DurableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        cls.status = t8.install()

    def setUp(self) -> None:
        _clear()

    def test_01_install(self) -> None:
        self.assertIn(self.status, {"durable: OK", "durable: SKIP (already applied)"})
        self.assertTrue(hasattr(self.agent_mod.run_sandboxed_python, "_tp8_stock"))

    def test_02_timeout_with_actions_gets_recap(self) -> None:
        raw = {"error": "Tool timed out after 30s.", "stdout": "partial",
               "action_results": [
                   {"executed": True, "level": 2, "state": "NOT_FINISHED", "board_changed": True,
                    "score": 1, "level_completed": False, "game_over": False},
                   {"executed": True, "level": 2, "state": "NOT_FINISHED", "board_changed": False,
                    "score": 1, "level_completed": False, "game_over": False}]}
        with mock.patch.dict(t8._STOCK, {"fn": lambda *a, **k: raw}):
            out = self.agent_mod.run_sandboxed_python(
                code="x", timeout_seconds=30, initial_state={}, action_handler=lambda a: {})
        self.assertIn("2 action(s) DID execute", out["error"])
        self.assertIn("level=2", out["error"])
        self.assertIn("re-observe it", out["error"])

    def test_03_timeout_without_actions_untouched(self) -> None:
        raw = {"error": "Tool timed out after 30s.", "stdout": "", "action_results": []}
        with mock.patch.dict(t8._STOCK, {"fn": lambda *a, **k: raw}):
            out = self.agent_mod.run_sandboxed_python(
                code="x", timeout_seconds=30, initial_state={}, action_handler=lambda a: {})
        self.assertEqual(out["error"], "Tool timed out after 30s.")

    def test_04_non_timeout_error_untouched(self) -> None:
        raw = {"error": "Python syntax error: bad", "action_results": [
            {"executed": True, "level": 1}]}
        with mock.patch.dict(t8._STOCK, {"fn": lambda *a, **k: raw}):
            out = self.agent_mod.run_sandboxed_python(
                code="x", timeout_seconds=30, initial_state={}, action_handler=lambda a: {})
        self.assertEqual(out["error"], "Python syntax error: bad")

    def test_05_disabled_passthrough(self) -> None:
        os.environ["TP8_ENABLE"] = "0"
        raw = {"error": "Tool timed out after 30s.", "action_results": [{"executed": True}]}
        with mock.patch.dict(t8._STOCK, {"fn": lambda *a, **k: raw}):
            out = self.agent_mod.run_sandboxed_python(
                code="x", timeout_seconds=30, initial_state={}, action_handler=lambda a: {})
        self.assertEqual(out["error"], "Tool timed out after 30s.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
