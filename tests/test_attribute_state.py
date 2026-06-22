import numpy as np
from arcagi3.attribute_state import agent_attributes, AttrVec


def _shape_grid(color, orient):
    # an L-tromino so rotation is detectable; placed in a 5x5 patch
    g = np.zeros((5, 5), dtype=np.int8)
    cells = {0: [(0,0),(1,0),(2,0),(2,1)], 1: [(0,0),(0,1),(0,2),(1,0)]}[orient]
    for r, c in cells:
        g[r, c] = color
    return g, [(r, c) for r, c in cells]


def test_attr_captures_color():
    g, cells = _shape_grid(9, 0)
    a = agent_attributes(g, cells)
    assert a.color == 9


def test_attr_distinguishes_rotation():
    g0, c0 = _shape_grid(9, 0)
    g1, c1 = _shape_grid(9, 1)
    assert agent_attributes(g0, c0).shape_sig != agent_attributes(g1, c1).shape_sig


def test_attr_equal_for_same_object():
    g, cells = _shape_grid(7, 0)
    assert agent_attributes(g, cells) == agent_attributes(g.copy(), cells)
