import numpy as np
from arcagi3.model_guide import ModelGuide

def test_suggest_none_before_any_observation():
    g = ModelGuide()
    grid = np.zeros((8, 8), dtype=np.int8); grid[3, 3] = 5
    assert g.suggest(grid, [1, 2, 3, 4]) is None   # no model yet

def test_observe_then_suggest_returns_simple_action():
    g = ModelGuide()
    a = np.zeros((8, 8), dtype=np.int8); a[3, 3] = 5; a[3, 6] = 9   # agent + a distinct target
    b = np.zeros((8, 8), dtype=np.int8); b[3, 4] = 5; b[3, 6] = 9   # agent moved right (action 4)
    g.observe(prev_grid=a, prev_action=("S", 4), grid=b, level=0)
    c = np.zeros((8, 8), dtype=np.int8); c[3, 5] = 5; c[3, 6] = 9   # moved right again
    g.observe(prev_grid=b, prev_action=("S", 4), grid=c, level=0)
    s = g.suggest(c, [1, 2, 3, 4])
    assert s is None or s in (1, 2, 3, 4)   # a valid simple action id, or None if no plan

def test_level_change_does_not_crash():
    g = ModelGuide()
    a = np.zeros((6, 6), dtype=np.int8); a[0, 0] = 5
    b = np.zeros((6, 6), dtype=np.int8); b[0, 1] = 5
    g.observe(prev_grid=a, prev_action=("S", 4), grid=b, level=0)
    g.observe(prev_grid=b, prev_action=("S", 4), grid=b, level=1)  # level increment
    assert g.suggest(b, [1, 2, 3, 4]) in (None, 1, 2, 3, 4)
