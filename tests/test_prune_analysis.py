# tests/test_prune_analysis.py
import numpy as np
from arcagi3 import prune_analysis as A


def _steps():
    # level 0: detour a->d->a, then path a->b->c; c is the level-up (reward) state.
    # fields: idx, level, from_key, action, reward, tier
    return [
        A.Step(0, 0, b"a", ("S", 4), 0.0, 0),
        A.Step(1, 0, b"d", ("S", 1), 0.0, 9),
        A.Step(2, 0, b"a", ("S", 1), 0.0, 0),
        A.Step(3, 0, b"b", ("S", 2), 0.0, 0),
        A.Step(4, 0, b"c", ("S", 3), 1.0, 0),   # reward -> level up
        A.Step(5, 1, b"z", ("S", 1), 0.0, 0),   # first state of level 1
    ]


def test_build_edges_excludes_reset_and_levelup():
    edges, first_seen = A.build_edges(_steps())
    assert set(edges[b"a"]) == {(("S", 4), b"d"), (("S", 1), b"b")}
    assert edges[b"b"] == [(("S", 2), b"c")]
    assert b"c" not in edges  # step 4 is a reward edge -> excluded
    assert first_seen[b"a"] == 0 and first_seen[b"d"] == 1
    assert first_seen[b"b"] == 3 and first_seen[b"c"] == 4
