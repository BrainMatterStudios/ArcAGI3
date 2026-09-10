#!/usr/bin/env python3
"""Tests for graft_effects. The load-bearing ones use the REAL stock segmentation and
the REAL runtime_state dataclasses, so a signature or shape change in the stock bundle
breaks a test instead of silently producing an empty block at wave time."""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scratchpad/bundles/june_stock/src/ARC3-Inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graft_effects as ge  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402


def grid(rows=8, cols=8, blocks=()):
    g = [[0] * cols for _ in range(rows)]
    for (r, c, v) in blocks:
        g[r][c] = v
    return tuple(tuple(r) for r in g)


def frame(blocks, step=0, level=1):
    return Frame(grid=grid(blocks=blocks), step=step, level=level)


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        ge._RUNS.clear(); ge._STATE["skips"].clear(); ge._STATE["disabled"] = False

    def test_real_segmentation_gives_hashes(self):
        n = ge._nodes(frame([(2, 2, 3), (2, 3, 3)]).grid, {})
        self.assertIsNotNone(n)
        self.assertTrue(all(x["hash"] for x in n))

    def test_cache_segments_each_frame_once(self):
        cache = {}
        a = frame([(2, 2, 3)]).grid
        ge._nodes(a, cache); ge._nodes(a, cache)
        self.assertEqual(len(cache), 1)

    def test_translation_invariant_hash_detects_a_move(self):
        cache = {}
        before = ge._nodes(frame([(2, 2, 3), (2, 3, 3)]).grid, cache)
        after = ge._nodes(frame([(2, 4, 3), (2, 5, 3)]).grid, cache)
        d = ge._diff(before, after)
        self.assertTrue(d["changed"])
        self.assertEqual([(dr, dc) for _h, dr, dc in d["moved"]], [(0, 2)])

    def test_appearance_and_vanishing(self):
        cache = {}
        b = ge._nodes(frame([(1, 1, 3)]).grid, cache)
        a = ge._nodes(frame([(1, 1, 3), (5, 5, 4)]).grid, cache)
        self.assertGreaterEqual(ge._diff(b, a)["appeared"], 1)
        self.assertGreaterEqual(ge._diff(a, b)["vanished"], 1)

    def test_identical_frames_report_no_change(self):
        cache = {}
        f = ge._nodes(frame([(3, 3, 5)]).grid, cache)
        self.assertFalse(ge._diff(f, f)["changed"])


class AccumulateTests(unittest.TestCase):
    def setUp(self):
        ge._RUNS.clear(); ge._STATE["skips"].clear(); ge._STATE["disabled"] = False

    def _history(self, moves):
        """moves = list of (action, col) building a single object sliding right."""
        hist = [HistoryEntry(action="", frame=frame([(2, 2, 3)]))]
        for act, col in moves:
            hist.append(HistoryEntry(action=act, frame=frame([(2, col, 3)])))
        return hist

    def test_observe_builds_a_consistent_effect_table(self):
        run = ge._run("k")
        ge._observe(run, self._history([("RIGHT", 3), ("RIGHT", 4), ("RIGHT", 5)]))
        self.assertEqual(len(run["effects"]["RIGHT"]), 3)
        block = ge._block(run)
        self.assertIn("RIGHT x3", block)
        self.assertIn("(+0,+1)", block)
        self.assertIn("[3/3]", block)

    def test_observe_is_incremental_not_quadratic(self):
        run = ge._run("k")
        h = self._history([("RIGHT", 3), ("RIGHT", 4)])
        ge._observe(run, h)
        self.assertEqual(run["seen"], len(h))
        h2 = h + [HistoryEntry(action="RIGHT", frame=frame([(2, 5, 3)]))]
        ge._observe(run, h2)
        self.assertEqual(len(run["effects"]["RIGHT"]), 3)   # not 5

    def test_a_no_op_action_is_reported_as_such(self):
        run = ge._run("k")
        hist = [HistoryEntry(action="", frame=frame([(2, 2, 3)]))]
        for _ in range(3):
            hist.append(HistoryEntry(action="SPACE", frame=frame([(2, 2, 3)])))
        ge._observe(run, hist)
        self.assertIn("no board change [3/3]", ge._block(run))

    def test_block_is_capped(self):
        run = ge._run("k")
        for i in range(20):
            run["effects"][f"ACT{i}"] = [{"moved": [], "appeared": 0, "vanished": 0, "changed": False}]
        block = ge._block(run)
        self.assertLessEqual(block.count("\n- "), ge._cfg()["max_actions"])
        self.assertLessEqual(len(block), ge._cfg()["max_chars"] + 200)

    def test_empty_table_yields_no_block(self):
        self.assertEqual(ge._block(ge._run("empty")), "")

    def test_history_entries_missing_frames_do_not_raise(self):
        run = ge._run("k")
        ge._observe(run, [HistoryEntry(action="", frame=None),
                          HistoryEntry(action="LEFT", frame=None)])
        self.assertEqual(ge._block(run), "")


class InstallTests(unittest.TestCase):
    def setUp(self):
        ge._RUNS.clear(); ge._STATE["installed"] = False; ge._STATE["disabled"] = False

    def test_install_binds_both_methods_and_is_idempotent(self):
        from inference.agent import tool_agent as mod
        before = mod.ToolAgent.analyze
        msg = ge.install()
        self.assertTrue(msg.endswith(": OK"), msg)   # the runner enforces this contract
        self.assertIsNot(mod.ToolAgent.analyze, before)
        self.assertIn("SKIP", ge.install())
        mod.ToolAgent.analyze = before   # leave the class as we found it

    def test_prompt_wrapper_is_pass_through_when_disabled(self):
        ge._STATE["disabled"] = True
        self.assertEqual(ge._block(ge._run("x")), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
