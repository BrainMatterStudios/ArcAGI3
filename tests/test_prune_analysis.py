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

def test_segment_and_ceilings():
    steps = _steps()
    edges, first_seen = A.build_edges(steps)
    segs = A.segment_levels(steps, first_seen)
    assert set(segs) == {0}                       # only level 0 ends in a reward here
    seg = segs[0]
    assert seg.start_key == b"a" and seg.target_key == b"c"
    assert seg.member_keys == {b"a", b"b", b"c", b"d"}
    assert seg.actual_actions == 4                # end_idx 4 - start_idx 0

    path = A.shortest_path(edges, seg.start_key, seg.target_key, seg.member_keys)
    assert path == [b"a", b"b", b"c"]

    c = A.ceilings(seg, path)
    assert c["reachable"] is True
    assert c["discovered_states"] == 4 and c["path_states"] == 3
    assert c["ceiling_states"] == 0.25
    assert c["path_actions"] == 2 and c["ceiling_actions"] == 0.5


def test_shortest_path_unreachable_returns_none():
    edges = {b"a": [(("S", 1), b"b")]}
    assert A.shortest_path(edges, b"a", b"x", {b"a", b"b"}) is None
    c = A.ceilings(A.LevelSeg(0, b"a", b"x", {b"a", b"b"}, 0, 3), None)
    assert c["reachable"] is False and c["ceiling_actions"] is None
