from arcagi3.motion_lattice import lattice_pitch, snap


def test_lattice_pitch_from_deltas():
    assert lattice_pitch({1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}) == (5, 5)


def test_lattice_pitch_zero_axis():
    # only vertical motion -> column pitch 0 (no snapping on that axis)
    assert lattice_pitch({1: (-3, 0), 2: (3, 0)}) == (3, 0)


def test_snap_to_nearest_lattice_cell():
    # origin (10,10), pitch (5,5): (12,18) -> row 10+round(0.4)*5=10, col 10+round(1.6)*5=20
    assert snap((12, 18), (10, 10), (5, 5), (64, 64)) == (10, 20)


def test_snap_identity_on_lattice():
    assert snap((20, 25), (10, 10), (5, 5), (64, 64)) == (20, 25)


def test_snap_clamps_to_bounds():
    # row would snap to 100 -> clamp to 19; col (3-10)/5=-1.4 -> round -1 -> 5
    assert snap((100, 3), (10, 10), (5, 5), (20, 20)) == (19, 5)


def test_snap_zero_pitch_keeps_origin_on_that_axis():
    # col pitch 0 -> column stays at origin's column (10)
    assert snap((12, 18), (10, 10), (5, 0), (64, 64)) == (10, 10)
