from arcagi3.factored_model import FactoredState, InducedModel, plan
from arcagi3.transform_induction import OnEnterCycle, TerminalPredicate


def _toy_model():
    # 1x5 corridor; cell at col 2 cycles "rot" attr through (0,1,2,3); slot at col 4 needs rot=0.
    # agent starts col 0 rot=3. Deltas: action 4 = +1 col (right), action 3 = -1 col (left).
    deltas = {3: (0, -1), 4: (0, 1)}
    walls = set()                      # open corridor
    tiles = {(0, 2): OnEnterCycle(tile_color=4, attribute="rot", order=(0, 1, 2, 3))}
    return InducedModel(deltas=deltas, walls=walls, tiles=tiles, width=5, height=1,
                        terminal=TerminalPredicate())


def test_plan_reaches_attr_match_slot():
    model = _toy_model()
    start = FactoredState(pos=(0, 0), attrs={"rot": 3}, completed=frozenset())
    slots = [{"pos": (0, 4), "attr_req": {"rot": 0}, "done": False}]
    actions = plan(model, start, slots, max_nodes=5000)
    assert actions is not None
    assert actions[-1] == 4
    assert 0 < len(actions) <= 12


def _open_model(width=5, height=1, tiles=None, walls=None):
    return InducedModel(deltas={3: (0, -1), 4: (0, 1)}, walls=walls or set(),
                        tiles=tiles or {}, width=width, height=height, terminal=TerminalPredicate())


def test_already_at_goal_returns_empty_plan():
    model = _open_model()
    start = FactoredState(pos=(0, 4), attrs={"rot": 0}, completed=frozenset())
    slots = [{"pos": (0, 4), "attr_req": {"rot": 0}, "done": False}]
    assert plan(model, start, slots, max_nodes=5000) == []


def test_empty_slots_returns_empty_plan():
    model = _open_model()
    start = FactoredState(pos=(0, 0), attrs={"rot": 0}, completed=frozenset())
    assert plan(model, start, [], max_nodes=5000) == []


def test_no_solution_returns_none():
    # slot requires rot=1 but there is no cycler tile anywhere -> unreachable
    model = _open_model()
    start = FactoredState(pos=(0, 0), attrs={"rot": 0}, completed=frozenset())
    slots = [{"pos": (0, 4), "attr_req": {"rot": 1}, "done": False}]
    assert plan(model, start, slots, max_nodes=5000) is None


def test_walls_block_movement():
    # wall at col 2 blocks the corridor; slot beyond it is unreachable
    model = _open_model(walls={(0, 2)})
    start = FactoredState(pos=(0, 0), attrs={"rot": 0}, completed=frozenset())
    slots = [{"pos": (0, 4), "attr_req": {"rot": 0}, "done": False}]
    assert plan(model, start, slots, max_nodes=5000) is None


def test_factored_state_attr_order_normalised():
    # same mapping, different construction -> equal & same hash
    a = FactoredState(pos=(0, 0), attrs={"x": 1, "y": 2}, completed=frozenset())
    b = FactoredState(pos=(0, 0), attrs=(("y", 2), ("x", 1)), completed=frozenset())
    assert a == b and hash(a) == hash(b)
