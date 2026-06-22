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
