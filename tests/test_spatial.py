from arcagi3.spatial import OccupancyMap

# arrow deltas: 1=up,2=down,3=left,4=right with step 5 (e.g. ls20)
DELTAS = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}


def test_usable_and_step():
    om = OccupancyMap(DELTAS)
    assert om.usable
    assert om.step == 5


def test_quantize_lattice():
    om = OccupancyMap(DELTAS)
    assert om.quantize((10.0, 10.0)) == (0, 0)  # origin
    assert om.quantize((15.0, 10.0)) == (1, 0)
    assert om.quantize((10.0, 20.0)) == (0, 2)


def test_astar_straight_line():
    om = OccupancyMap(DELTAS)
    om.quantize((0.0, 0.0))  # set origin
    path = om.astar((0.0, 0.0), (0.0, 15.0))  # 3 steps right
    assert path == [4, 4, 4]


def test_astar_routes_around_wall():
    om = OccupancyMap(DELTAS)
    om.origin = (0.0, 0.0)
    # wall directly to the right at (0,1); must detour
    om.blocked.add((0, 1))
    path = om.astar((0.0, 0.0), (0.0, 10.0))  # want to reach (0,2)
    assert path is not None
    # verify the path avoids stepping into the blocked cell
    pos = (0, 0)
    units = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    for a in path:
        ur, uc = units[a]
        pos = (pos[0] + ur, pos[1] + uc)
        assert pos not in om.blocked
    assert pos == (0, 2)


def test_observe_move_learns_free_and_blocked():
    om = OccupancyMap(DELTAS)
    om.origin = (0.0, 0.0)
    # moved right successfully
    om.observe_move((0.0, 0.0), 4, (0.0, 5.0))
    assert (0, 1) in om.free
    # tried to move right but blocked (stayed)
    om.observe_move((0.0, 5.0), 4, (0.0, 5.0))
    assert (0, 2) in om.blocked


def test_not_usable_when_only_one_axis():
    om = OccupancyMap({1: (-5, 0), 2: (5, 0)})  # only vertical
    assert not om.usable
    assert om.astar((0.0, 0.0), (0.0, 10.0)) is None
