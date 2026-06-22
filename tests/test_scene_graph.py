"""Tests for the scene-graph extractor (symbolic goal perception, no learning)."""
import numpy as np

from arcagi3 import scene_graph as SG


def test_symmetry_detected():
    g = np.zeros((16, 16), dtype=np.int8)
    g[2:5, 2:5] = 3
    g[2:5, 11:14] = 3          # mirror image across vertical axis
    s = SG.extract(g, bg=0)
    assert s["symmetry"]["vertical"] >= 0.9


def test_framed_target_detected():
    g = np.zeros((16, 16), dtype=np.int8)
    g[4:8, 4:8] = 5            # gray box
    g[5:7, 5:7] = 9            # maroon inside (framed) -> a target
    s = SG.extract(g, bg=0)
    assert any(t["color"] == 9 for t in s["target_candidates"])


def test_collectibles_grouped():
    g = np.zeros((16, 16), dtype=np.int8)
    for (r, c) in [(1, 1), (1, 5), (5, 1), (9, 9)]:
        g[r, c] = 7            # 4 scattered single-cell color-7 -> collectibles
    s = SG.extract(g, bg=0)
    assert any(col["color"] == 7 and col["count"] >= 3 for col in s["collectibles"])


def test_hypotheses_rank_symmetry_game():
    g = np.zeros((16, 16), dtype=np.int8)
    g[2:6, 1:4] = 2
    g[2:6, 12:15] = 2          # symmetric
    hyp = SG.goal_hypotheses(SG.extract(g, bg=0))
    assert hyp and hyp[0]["goal"] == "complete_symmetry"


def test_empty_grid_no_hypotheses():
    g = np.zeros((16, 16), dtype=np.int8)
    assert SG.goal_hypotheses(SG.extract(g, bg=0)) == []
