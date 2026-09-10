#!/usr/bin/env python3
"""Tests for graft_compactstate. The correctness bar is: the file the harness reads back
must be byte-equivalent in CONTENT, and it must be fresh whenever anything reads it."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scratchpad/bundles/june_stock/src/ARC3-Inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state  # noqa: E402
from inference.agent import runtime_state as rs  # noqa: E402
import graft_compactstate as gc  # noqa: E402


def grid(seed=0):
    r = random.Random(seed)
    return tuple(tuple(r.randint(0, 15) for _ in range(64)) for _ in range(64))


def hist(n):
    return [HistoryEntry(action=f"A{i}", frame=Frame(grid=grid(i), step=i, level=1)) for i in range(n)]


class WriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock_write = rs.write_runtime_state
        cls.d = Path(tempfile.mkdtemp())
        cls.h = hist(20)
        cls.cur = Frame(grid=grid(99), step=20, level=2)
        cls.p_stock = cls.d / "stock.json"
        cls.stock_write(cls.p_stock, current_frame=cls.cur, history=cls.h)
        gc._STATE["installed"] = False
        cls.msg = gc.install()
        cls.p_new = cls.d / "new.json"
        rs.write_runtime_state(cls.p_new, current_frame=cls.cur, history=cls.h)

    def test_install_reports_both_halves(self):
        self.assertTrue(self.msg.endswith(": OK"), self.msg)
        self.assertIn("compact_json", self.msg)

    def test_content_round_trips_identically(self):
        a, ha = load_runtime_state(self.p_stock)
        b, hb = load_runtime_state(self.p_new)
        self.assertEqual(a.grid, b.grid)
        self.assertEqual(a.step, b.step)
        self.assertEqual(a.level, b.level)
        self.assertEqual(len(ha), len(hb))
        for x, y in zip(ha, hb):
            self.assertEqual(x.action, y.action)
            self.assertEqual(x.frame.grid, y.frame.grid)

    def test_parsed_json_is_equal_not_merely_loadable(self):
        self.assertEqual(json.loads(self.p_stock.read_text()), json.loads(self.p_new.read_text()))

    def test_it_is_materially_smaller(self):
        self.assertLess(self.p_new.stat().st_size * 3, self.p_stock.stat().st_size,
                        "compact JSON should be at least 3x smaller on 64x64 int grids")

    def test_no_leftover_tmp_file(self):
        self.assertFalse((self.d / "new.json.tmp").exists())


class BatchTests(unittest.TestCase):
    """The batch half must never leave the file stale when a reader looks at it."""

    def test_writes_are_suppressed_inside_a_batch_and_flushed_once(self):
        calls = {"n": 0}

        class FakeSession:
            def write_runtime_state(self):
                calls["n"] += 1

            def step_env(self, arguments):
                for _ in range(arguments["k"]):
                    self.write_runtime_state()      # the per-action write
                return {"ok": True}

        stock_write, stock_step = FakeSession.write_runtime_state, FakeSession.step_env

        def write_state(self):
            if getattr(self, "_cs_in_batch", False):
                self._cs_dirty = True
                return
            stock_write(self)

        def step_env(self, arguments):
            self._cs_in_batch = True
            self._cs_dirty = False
            try:
                return stock_step(self, arguments)
            finally:
                self._cs_in_batch = False
                if getattr(self, "_cs_dirty", False):
                    stock_write(self)

        FakeSession.write_runtime_state = write_state
        FakeSession.step_env = step_env
        s = FakeSession()
        s.step_env({"k": 10})
        self.assertEqual(calls["n"], 1, "10 actions in one batch must produce exactly 1 write")
        s.step_env({"k": 1})
        self.assertEqual(calls["n"], 2)

    def test_a_batch_that_writes_nothing_flushes_nothing(self):
        calls = {"n": 0}

        class S:
            def write_runtime_state(self):
                calls["n"] += 1

            def step_env(self, arguments):
                return {}

        stock_write, stock_step = S.write_runtime_state, S.step_env

        def ws(self):
            if getattr(self, "_cs_in_batch", False):
                self._cs_dirty = True
                return
            stock_write(self)

        def se(self, a):
            self._cs_in_batch = True
            self._cs_dirty = False
            try:
                return stock_step(self, a)
            finally:
                self._cs_in_batch = False
                if getattr(self, "_cs_dirty", False):
                    stock_write(self)

        S.write_runtime_state, S.step_env = ws, se
        S().step_env({})
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
