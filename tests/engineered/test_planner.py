"""Planner unit tests: frontier reachability, RESET cost accounting,
win-edge replay, prior-priced frontier choice, fatal retry."""
from __future__ import annotations

import numpy as np

from engineered.graph import LevelGraph, RESET_ACTION, simple_key
from engineered.planner import dijkstra, plan

A1, A2, A3 = simple_key(1), simple_key(2), simple_key(3)
F = np.zeros((4, 4), dtype=np.int16)


def chain_graph() -> LevelGraph:
    """r -A1-> a -A1-> b, root r."""
    g = LevelGraph("test", 0)
    for k in ("r", "a", "b"):
        g.ensure_node(k, F, [1, 2])
    g.set_root("r")
    g.observe("r", A1, "a")
    g.observe("a", A1, "b")
    return g


def test_planner_reaches_known_frontier() -> None:
    """From r the nearest untried pair on a fully-mapped chain is found and
    the path + expected successors are exact."""
    g = chain_graph()
    g.observe("b", A1, "r")  # close the cycle: A1 tried everywhere
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}
    p = plan(g, "b", lambda k: menu[k], greedy_rank_max=-1)
    assert p.kind == "frontier"
    assert p.actions == [A1, A2]      # walk b->r, then probe A2
    assert p.expected == ["r"]        # verified prefix only
    assert p.target == "r"


def test_greedy_fast_path_probes_current_node() -> None:
    g = chain_graph()
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}
    p = plan(g, "r", lambda k: menu[k])
    assert p.kind == "frontier" and p.actions == [A2] and p.expected == []


def test_reset_cost_accounting() -> None:
    """Stranded at b (self-loop only), the way home is the virtual RESET
    edge at cost exactly 1 scored action."""
    g = chain_graph()
    g.observe("b", A1, "b")  # strand: b's only action self-loops
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}
    dist, parent = dijkstra(g, "b")
    assert dist["r"] == 1.0           # RESET is 1 action, not free
    assert parent["r"] == ("b", RESET_ACTION)
    p = plan(g, "b", lambda k: menu[k], prior_penalty=0.3, greedy_rank_max=-1)
    assert p.actions == [RESET_ACTION, A2]
    assert p.cost == 1 + 1 + 0.3      # reset + probe + rank-1 prior


def test_observed_reset_edge_preempts_virtual() -> None:
    g = chain_graph()
    g.observe("b", RESET_ACTION, "r")  # real reset observed from b
    dist, parent = dijkstra(g, "b")
    assert dist["r"] == 1.0 and parent["r"] == ("b", RESET_ACTION)


def test_win_edge_takes_priority_over_frontier() -> None:
    g = chain_graph()
    g.observe("a", A2, None, level_up=True)  # known level-up at (a, A2)
    menu = {"r": [A1, A2], "a": [A1, A2], "b": [A1]}
    p = plan(g, "r", lambda k: menu[k], greedy_rank_max=-1)
    assert p.kind == "win"
    assert p.actions == [A1, A2]      # r->a then the winning action
    assert p.expected == ["a"]


def test_prior_prices_distance_against_rank() -> None:
    """A rank-0 probe two steps away beats a rank-9 probe here (2+1+0 <
    1+2.7); with a flat prior the local probe wins."""
    g = chain_graph()
    menu = {"r": [A1] + [simple_key(9)] * 0 + [A2], "a": [A1], "b": [A1]}
    # current r: untried A2 at rank 9 vs untried at b? craft menus:
    long_menu_r = [A1] + [simple_key(i + 10) for i in range(8)] + [A2]
    menu = {"r": long_menu_r, "a": [A1], "b": [A1, A3]}
    # untried: r offers rank 1..9 (A2 last); b offers A3 at rank 1, dist 2
    p = plan(g, "r", lambda k: menu[k], prior_penalty=0.3, greedy_rank_max=-1)
    # best at r: rank-1 action cost 1+0.3; at b: dist2+1+0.3 = 3.3
    assert p.target == "r" and p.cost == 1 + 0.3
    p2 = plan(g, "r", lambda k: menu[k], prior_penalty=2.0, greedy_rank_max=-1)
    # heavy prior: rank-1 here costs 3.0 vs b's rank-1 at 2+1+2 = 5 — still r
    assert p2.target == "r"


def test_fatal_pair_retried_only_when_frontier_exhausted() -> None:
    g = chain_graph()
    g.observe("b", A1, "r")           # cycle closed
    g.observe("r", A2, None, game_over=True)  # r's other action killed once
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}
    p = plan(g, "r", lambda k: menu[k], greedy_rank_max=-1)
    assert p.kind == "frontier" and p.actions == [A2]  # retry the fatal pair
    g.observe("r", A2, None, game_over=True)  # second death: at the cap
    p2 = plan(g, "r", lambda k: menu[k], greedy_rank_max=-1)
    assert p2.kind == "none"


def test_no_reachable_frontier_returns_none() -> None:
    g = chain_graph()
    g.observe("b", A1, "r")
    menu = {"r": [A1], "a": [A1], "b": [A1]}  # everything tried
    p = plan(g, "r", lambda k: menu[k])
    assert p.kind == "none"
