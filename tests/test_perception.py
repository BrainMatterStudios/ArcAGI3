import numpy as np

from arcagi3 import perception as P


def _grid(fill=0):
    return np.full((P.GRID, P.GRID), fill, dtype=np.int8)


def test_to_grid_takes_last_subframe():
    stack = np.stack([_grid(1), _grid(2), _grid(3)])
    g = P.to_grid(stack)
    assert g.shape == (64, 64)
    assert (g == 3).all()


def test_to_grid_handles_2d():
    assert P.to_grid(_grid(5)).shape == (64, 64)


def test_detect_background_is_most_common():
    g = _grid(7)
    g[0, 0] = 3
    assert P.detect_background(g) == 7


def test_connected_components_basic():
    g = _grid(0)
    # two separate single-cell objects of color 4, one 2x2 block of color 9
    g[1, 1] = 4
    g[10, 10] = 4
    g[20:22, 20:22] = 9
    objs = P.connected_components(g, background=0)
    assert len(objs) == 3
    sizes = sorted(o.size for o in objs)
    assert sizes == [1, 1, 4]
    block = [o for o in objs if o.size == 4][0]
    assert block.color == 9
    assert block.bbox == (20, 20, 21, 21)
    assert block.centroid == (20.5, 20.5)


def test_connected_4connectivity_diagonal_separate():
    g = _grid(0)
    g[0, 0] = 5
    g[1, 1] = 5  # diagonal -> separate under 4-connectivity
    objs = P.connected_components(g, background=0)
    assert len(objs) == 2


def test_state_hash_distinguishes_and_masks():
    a = _grid(0)
    b = _grid(0)
    b[5, 5] = 1
    assert P.state_hash(a) != P.state_hash(b)
    # if the differing cell is masked, hashes match
    mask = np.zeros((64, 64), dtype=bool)
    mask[5, 5] = True
    assert P.state_hash(a, mask) == P.state_hash(b, mask)


def test_volatility_tracker_flags_counter_cell():
    vt = P.VolatilityTracker(threshold=0.9, min_steps=4)
    rng = np.random.default_rng(0)
    for i in range(12):
        g = _grid(0)
        g[0, 0] = i % 16  # cell that changes every step -> volatile
        g[30, 30] = 1 if i < 3 else 1  # stable cell
        vt.update(g)
    m = vt.mask()
    assert m[0, 0]  # counter cell flagged
    assert not m[30, 30]


def test_salient_targets_prioritizes_small_objects():
    g = _grid(0)
    g[2, 2] = 4  # tiny object -> priority 0
    g[40:50, 40:50] = 9  # big object -> lower priority
    targets = P.salient_click_targets(g, background=0)
    assert targets[0][2] == 0  # first target is highest priority
    # tiny object centroid is (2,2) -> (x=2,y=2)
    assert (2, 2, 0) in targets
