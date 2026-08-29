"""Unit tests for the frontier explorer (pure Python, no engine).

Run:  .venv/bin/python submission/_throughput_v1/test_frontier_explorer.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import frontier_explorer as fe  # noqa: E402


def grid(cells: dict[tuple[int, int], int], h: int = 64, w: int = 64, bg: int = 0):
    g = [[bg] * w for _ in range(h)]
    for (r, c), v in cells.items():
        g[r][c] = v
    return tuple(tuple(row) for row in g)


class FrameTests(unittest.TestCase):
    def test_segments_and_groups(self) -> None:
        cells = {(10, 10): 9, (10, 11): 9, (11, 10): 9, (11, 11): 9}          # salient medium -> tier 0
        cells.update({(30, 30): 3, (30, 31): 3, (31, 30): 3, (31, 31): 3})   # non-salient medium -> tier 1
        cells.update({(50, 50): 12})                                          # salient 1-cell -> tier 2
        g = grid(cells)
        segs = fe.segments(g)
        self.assertEqual(len(segs), 4)   # background + 3 blobs
        groups = sorted(fe.group_of(s, False) for s in segs if s["color"] != 0)
        self.assertEqual(groups, [0, 1, 2])

    def test_status_bar_mask_bar_and_twins(self) -> None:
        cells = {(0, c): 5 for c in range(0, 40)}                              # top bar, ratio 40
        for k in range(3):                                                    # three twins on the bottom edge
            cells.update({(62, 5 + 4 * k): 7, (62, 6 + 4 * k): 7, (63, 5 + 4 * k): 7, (63, 6 + 4 * k): 7})
        cells.update({(30, 30): 7, (30, 31): 7, (31, 30): 7, (31, 31): 7})   # interior twin-shaped blob: NOT masked
        g = grid(cells)
        mask = fe.status_bar_mask(g, fe.segments(g))
        self.assertIn((0, 3), mask)
        self.assertIn((62, 5), mask)
        self.assertNotIn((30, 30), mask)

    def test_hash_ignores_masked_cells(self) -> None:
        a = grid({(0, 0): 1, (20, 20): 4})
        b = grid({(0, 0): 2, (20, 20): 4})
        self.assertNotEqual(fe.frame_hash(a, set()), fe.frame_hash(b, set()))
        self.assertEqual(fe.frame_hash(a, {(0, 0)}), fe.frame_hash(b, {(0, 0)}))


class GraphTests(unittest.TestCase):
    def test_explore_then_travel_then_tier_advance(self) -> None:
        ex = fe.FrontierExplorer(seed=1)
        g0 = grid({(10, 10): 9, (10, 11): 9})
        g1 = grid({(10, 12): 9, (10, 13): 9})
        k0 = ex.observe(g0, [1, 2])
        # two keyboard candidates in tier 0
        i, why = ex.choose(k0)
        self.assertIn("untested", why)
        k1 = ex.observe(g1, [1, 2])
        ex.record(k0, i, k1)
        self.assertEqual(ex.nodes[k0].result[i], 1)
        # from k1 exhaust both keys as no-ops -> k1 closed; k0 still has one untested -> travel
        for j in list(range(2)):
            ex.record(k1, j, k1)
        self.assertEqual(ex.nodes[k1].open_in(0), [])
        # k1 has no path back (no edge k1->k0), so the explorer advances tiers then goes random
        i2, why2 = ex.choose(k1)
        self.assertTrue("exhausted" in why2 or "towards" in why2)
        # now give k0 an edge from k1 and check travel
        ex.rev.setdefault(k0, set()).add((k1, 0))
        ex.nodes[k1].target[0] = k0
        ex.nodes[k1].result[0] = 1
        ex.active = 0
        ex._rebuild()
        i3, why3 = ex.choose(k1)
        self.assertEqual(i3, 0)
        self.assertIn("towards frontier", why3)

    def test_click_candidates_and_masked_tier(self) -> None:
        ex = fe.FrontierExplorer(seed=2)
        cells = {(0, c): 5 for c in range(0, 40)}
        cells.update({(30, 30): 9, (30, 31): 9, (31, 30): 9, (31, 31): 9})
        g = grid(cells)
        k = ex.observe(g, [6])
        node = ex.nodes[k]
        kinds = Counter = {}
        for cand in node.candidates:
            kinds[cand.group] = kinds.get(cand.group, 0) + 1
        self.assertEqual(sum(kinds.values()), 3)          # background, bar, blob
        self.assertEqual(kinds.get(4), 1)                 # the bar is tier 4
        self.assertEqual(kinds.get(0), 1)                 # the salient blob is tier 0
        i, why = ex.choose(k)
        self.assertEqual(node.candidates[i].group, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
