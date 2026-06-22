import numpy as np
from arcagi3.transform_induction import induce_on_enter_cycles, OnEnterCycle


def test_induces_color_cycle_on_tile_contact():
    # Each time the agent enters a cell that WAS color 4, its color advances; color-7 tile does nothing.
    triples = [
        {"entered_color": 4, "attr": "color", "before": 9, "after": 10},
        {"entered_color": 4, "attr": "color", "before": 10, "after": 11},
        {"entered_color": 7, "attr": "color", "before": 9, "after": 9},   # no change
    ]
    rules = induce_on_enter_cycles(triples)
    assert any(r.tile_color == 4 and r.attribute == "color" for r in rules)
    assert all(r.tile_color != 7 for r in rules)  # non-effecting tiles are not rules
