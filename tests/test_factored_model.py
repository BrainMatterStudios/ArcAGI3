from arcagi3.factored_model import FactoredState, InducedModel, plan, plan_painted_set
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


def _corridor_painted(max_paints):
    # 1x5 corridor; cols 1,2,3 are PAINTABLE (must be painted to traverse); slot at col 4.
    return dict(
        deltas={3: (0, -1), 4: (0, 1)},
        true_walls=set(),
        paintable_cells={(0, 1), (0, 2), (0, 3)},
        tiles={},
        width=5, height=1,
        start_pos=(0, 0), start_attrs={},
        slots=[{"pos": (0, 4), "attr_req": {}, "done": False}],
        max_paints=max_paints, max_nodes=5000,
    )

def test_a2_solves_when_budget_allows():
    actions, status = plan_painted_set(**_corridor_painted(max_paints=3))
    assert status == "solved" and actions[-1] == 4 and len(actions) == 4

def test_a2_no_solution_when_budget_too_small():
    actions, status = plan_painted_set(**_corridor_painted(max_paints=2))
    assert status == "no_solution" and actions is None

def test_a2_reports_intractable_on_node_cap():
    cfg = _corridor_painted(max_paints=3); cfg["max_nodes"] = 1
    actions, status = plan_painted_set(**cfg)
    assert status == "intractable"


def test_a2_tile_on_paintable_cell_cycles_attr_and_paints():
    # A paintable cell that ALSO holds a rotation cycler: entering it both paints it and cycles attr.
    actions, status = plan_painted_set(
        deltas={4: (0, 1)}, true_walls=set(), paintable_cells={(0, 1)},
        tiles={(0, 1): OnEnterCycle(tile_color=11, attribute="rot", order=(0, 1, 2, 3))},
        width=3, height=1, start_pos=(0, 0), start_attrs={"rot": 3},
        slots=[{"pos": (0, 2), "attr_req": {"rot": 0}, "done": False}],
        max_paints=None, max_nodes=5000)
    # step onto (0,1): paint it AND rot 3->0; step to (0,2): matches rot=0 -> solved
    assert status == "solved" and actions == [4, 4]


def test_a2_multi_slot_accumulates_completion():
    # two slots in an open row (cols 1 and 3); agent must visit BOTH (order-free) to win.
    actions, status = plan_painted_set(
        deltas={3: (0, -1), 4: (0, 1)}, true_walls=set(), paintable_cells=set(),
        tiles={}, width=4, height=1, start_pos=(0, 0), start_attrs={},
        slots=[{"pos": (0, 1), "attr_req": {}, "done": False},
               {"pos": (0, 3), "attr_req": {}, "done": False}],
        max_paints=None, max_nodes=5000)
    assert status == "solved" and actions[-1] == 4  # ends by reaching the far slot


def test_a2_start_cell_paintable_costs_budget_on_reentry():
    # start cell is paintable; with max_paints=1, re-entering the start after leaving would need a
    # 2nd paint -> but a direct forward path needs only 1 paint, so it still solves at budget 1.
    actions, status = plan_painted_set(
        deltas={3: (0, -1), 4: (0, 1)}, true_walls=set(), paintable_cells={(0, 0), (0, 1)},
        tiles={}, width=3, height=1, start_pos=(0, 0), start_attrs={},
        slots=[{"pos": (0, 2), "attr_req": {}, "done": False}],
        max_paints=1, max_nodes=5000)
    # path: (0,0)->(0,1)[paint #1]->(0,2). Start cell (0,0) not pre-painted but never re-entered.
    assert status == "solved" and actions == [4, 4]
