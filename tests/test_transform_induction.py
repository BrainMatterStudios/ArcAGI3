import numpy as np
from arcagi3.transform_induction import induce_on_enter_cycles, OnEnterCycle, induce_terminal, TerminalPredicate


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


def test_terminal_is_position_and_attr_match():
    prewin = {"agent_pos": (3, 5), "agent_attrs": {"color": 9, "shape": "S0"},
              "slots": [{"pos": (3, 5), "attr_req": {"color": 9, "shape": "S0"}, "done": False}]}
    pred = induce_terminal(prewin)
    assert pred.kind == "attr_match_at_slot"
    # holds when agent on slot with matching attrs
    assert pred.holds(agent_pos=(3, 5), agent_attrs={"color": 9, "shape": "S0"},
                      slots=[{"pos": (3, 5), "attr_req": {"color": 9, "shape": "S0"}, "done": False}])
    # fails when attrs don't match
    assert not pred.holds(agent_pos=(3, 5), agent_attrs={"color": 9, "shape": "S1"},
                          slots=[{"pos": (3, 5), "attr_req": {"color": 9, "shape": "S0"}, "done": False}])


def test_terminal_all_slots_must_be_satisfied():
    pred = TerminalPredicate()
    # two slots, agent satisfies slot 0 but slot 1 not done and agent not on it -> not terminal
    assert not pred.holds(agent_pos=(0, 0), agent_attrs={"color": "c", "shape": "s"},
                          slots=[{"pos": (0, 0), "attr_req": {"color": "c", "shape": "s"}, "done": False},
                                 {"pos": (9, 9), "attr_req": {"color": "c", "shape": "s"}, "done": False}])
    # slot 1 already done, agent on slot 0 with match -> terminal
    assert pred.holds(agent_pos=(0, 0), agent_attrs={"color": "c", "shape": "s"},
                      slots=[{"pos": (0, 0), "attr_req": {"color": "c", "shape": "s"}, "done": False},
                             {"pos": (9, 9), "attr_req": {"color": "c", "shape": "s"}, "done": True}])
