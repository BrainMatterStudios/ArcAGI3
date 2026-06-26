import numpy as np

from arcagi3.object_interaction import interaction_signature


def _grid(rows):
    return np.array(rows, dtype=np.int64)


def test_no_change_returns_none_signature():
    g = _grid([[0, 0], [0, 0]])
    sig = interaction_signature(g, ("S", 1), g.copy(), bg=0)
    assert sig == ("NONE",)


def test_object_appears_is_a_distinct_signature():
    prev = _grid([[0, 0], [0, 0]])
    cur = _grid([[0, 3], [0, 0]])  # color 3 appears
    sig = interaction_signature(prev, ("S", 2), cur, bg=0)
    assert sig == ("S", 2, "appeared", frozenset({3}))


def test_object_vanishes_under_click_uses_target_color():
    prev = _grid([[0, 5], [0, 0]])
    cur = _grid([[0, 0], [0, 0]])  # color 5 vanishes; click target is that cell (row0,col1)
    sig = interaction_signature(prev, ("C", 1, 0), cur, bg=0)  # x=col1, y=row0
    assert sig == ("C", 5, "vanished", frozenset({5}))


def test_recolor_is_distinct_from_appeared():
    prev = _grid([[2, 0]])
    cur = _grid([[4, 0]])  # 2 -> 4 (neither is bg)
    sig = interaction_signature(prev, ("S", 1), cur, bg=0)
    assert sig == ("S", 1, "recolored", frozenset({2, 4}))
