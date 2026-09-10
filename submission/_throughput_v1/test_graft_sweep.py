#!/usr/bin/env python3
"""Tests for graft_sweep. The sweep fires REAL game actions, so the safety tests matter
more than the happy path: a sweep that runs past a level completion or keeps firing after
a refusal would spend a game's actions for nothing."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graft_sweep as gs  # noqa: E402


def payload(changed=True, level=1, **kw):
    out = {"executed_count": 1, "board_changed": changed, "level": level,
           "level_completed": False, "game_over": False, "run_complete": False, "done": False}
    out.update(kw)
    return out


class SweepTests(unittest.TestCase):
    def setUp(self):
        gs._RESULTS.clear(); gs._STATE["done"] = set(); gs._STATE["skips"].clear()
        gs._STATE["disabled"] = False

    def test_fires_each_action_once_then_repeats_the_first(self):
        seen = []
        def step(arg):
            seen.append(arg["actions"][0]["action"]); return payload()
        gs.run_sweep(step, ["UP", "DOWN", "LEFT"], "k")
        self.assertEqual(seen, ["UP", "DOWN", "LEFT", "UP"])

    def test_never_fires_RESET_or_MOUSE(self):
        seen = []
        def step(arg):
            seen.append(arg["actions"][0]["action"]); return payload()
        gs.run_sweep(step, ["UP", "RESET", "MOUSE", "DOWN"], "k")
        self.assertNotIn("RESET", seen)
        self.assertFalse(any(a.startswith("MOUSE") for a in seen))

    def test_stops_on_level_completion(self):
        seen = []
        def step(arg):
            seen.append(arg["actions"][0]["action"])
            return payload(level_completed=len(seen) == 2)
        gs.run_sweep(step, ["UP", "DOWN", "LEFT", "SPACE"], "k")
        self.assertEqual(len(seen), 2, "must stop the instant a level completes")

    def test_stops_on_game_over(self):
        seen = []
        def step(arg):
            seen.append(arg["actions"][0]["action"])
            return payload(game_over=len(seen) == 1)
        gs.run_sweep(step, ["UP", "DOWN", "LEFT"], "k")
        self.assertEqual(len(seen), 1)

    def test_stops_when_an_action_is_refused(self):
        seen = []
        def step(arg):
            seen.append(arg["actions"][0]["action"])
            return {"executed": False, "error": "nope"}
        gs.run_sweep(step, ["UP", "DOWN", "LEFT"], "k")
        self.assertEqual(len(seen), 1)
        self.assertIn("not_executed", gs._STATE["skips"])

    def test_step_env_exception_is_swallowed(self):
        def step(arg):
            raise RuntimeError("boom")
        gs.run_sweep(step, ["UP", "DOWN"], "k")
        self.assertIn("step_env_failed", gs._STATE["skips"])

    def test_respects_the_action_cap(self):
        seen = []
        def step(arg):
            seen.append(1); return payload()
        import os
        os.environ["SWEEP_MAX_ACTIONS"] = "2"
        try:
            gs.run_sweep(step, ["A", "B", "C", "D", "E"], "k")
            self.assertLessEqual(len(seen), 3)   # 2 capped + 1 repeat
        finally:
            del os.environ["SWEEP_MAX_ACTIONS"]

    def test_block_reports_determinism(self):
        def step(arg):
            return payload(changed=arg["actions"][0]["action"] != "DOWN")
        gs.run_sweep(step, ["UP", "DOWN"], "k")
        b = gs.block("k")
        self.assertIn("UP", b)
        self.assertIn("did NOT change the board", b)
        self.assertIn("same result both times", b)

    def test_block_is_empty_without_a_sweep(self):
        self.assertEqual(gs.block("never"), "")

    def test_install_is_gated_off_by_default(self):
        gs._STATE["installed"] = False
        self.assertIn("SKIP", gs.install())


if __name__ == "__main__":
    unittest.main(verbosity=2)
