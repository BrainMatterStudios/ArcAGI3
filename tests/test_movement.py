import numpy as np

from arcagi3 import movement as M


def _g():
    return np.zeros((16, 16), dtype=np.int8)


def test_infer_translation_single_object():
    b = _g()
    b[5, 5] = 7  # avatar
    b[10, 10] = 3  # static decoration
    a = b.copy()
    a[5, 5] = 0
    a[5, 6] = 7  # moved right by (0,+1)
    res = M.infer_translation(b, a, background=0)
    assert res == (7, 0, 1)


def test_infer_translation_prefers_smallest_mover():
    b = _g()
    b[2, 2] = 5  # small avatar (1 cell)
    b[8:12, 8:12] = 9  # big block also "moves" -- but smaller one preferred
    a = b.copy()
    a[2, 2] = 0
    a[1, 2] = 5  # avatar up
    res = M.infer_translation(b, a, background=0)
    assert res[0] == 5 and res[1:] == (-1, 0)


def test_infer_translation_none_when_static():
    b = _g()
    b[5, 5] = 7
    res = M.infer_translation(b, b.copy(), background=0)
    assert res is None


def test_avatar_centroid():
    mm = M.MotionModel(avatar_color=4, deltas={1: (-1, 0)})
    g = _g()
    g[3, 7] = 4
    assert mm.ok
    assert mm.avatar_centroid(g) == (3.0, 7.0)
