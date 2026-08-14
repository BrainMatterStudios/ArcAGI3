"""LevelGraph unit tests: determinism, Markov-violation escalation,
GAME_OVER persistence, fatal-retry semantics."""
from __future__ import annotations

import numpy as np
import pytest

from engineered.graph import LevelGraph, RESET_ACTION, click_key, simple_key

A1, A2 = simple_key(1), simple_key(2)
F = np.zeros((4, 4), dtype=np.int16)


def make_graph() -> LevelGraph:
    g = LevelGraph("test", 0)
    for k in ("r", "a", "b"):
        g.ensure_node(k, F, [1, 2, 6])
    g.set_root("r")
    return g


def test_deterministic_edge_accumulates_without_violation() -> None:
    g = make_graph()
    for _ in range(5):
        res = g.observe("r", A1, "a")
        assert not res.violation
    assert g.deterministic_successor("r", A1) == "a"
    assert g.edges[("r", A1)]["a"] == 5
    assert g.violation_events == 0 and not g.violated


def test_untried_respects_tried_and_menu_order() -> None:
    g = make_graph()
    menu = [A1, A2, click_key(3, 3)]
    assert g.untried("r", menu) == menu
    g.observe("r", A1, "a")
    assert g.untried("r", menu) == [A2, click_key(3, 3)]


def test_markov_violation_flags_and_escalates() -> None:
    """Same (state, action) -> different successor must flag the state,
    purge its plain-key evidence, and split future identity by context."""
    g = make_graph()
    g.observe("r", A1, "a")
    res = g.observe("r", A1, "b")  # contradiction
    assert res.violation
    assert "r" in g.violated and g.violation_events == 1
    # plain-key outgoing evidence purged: unusable for exact planning
    assert g.deterministic_successor("r", A1) is None
    assert ("r", A1) not in g.edges
    # escalated identity separates contexts…
    k1 = g.effective_key("r", ("p1", A2))
    k2 = g.effective_key("r", ("p2", A2))
    assert k1 != k2 and k1[0] == "ctx"
    # …and each context is deterministic on its own
    assert not g.observe(k1, A1, "a").violation
    assert not g.observe(k2, A1, "b").violation
    assert g.deterministic_successor(k1, A1) == "a"
    assert g.deterministic_successor(k2, A1) == "b"


def test_escalation_context_never_nests() -> None:
    """Composite contexts must collapse to their raw key — nesting exploded
    46 real tu93 states into 771 never-merging nodes."""
    g = make_graph()
    g.observe("r", A1, "a")
    g.observe("r", A1, "b")
    ctx_prev = ("ctx", "p0", A2, "q")  # arriving FROM an escalated state
    k = g.effective_key("r", (ctx_prev, A1))
    assert k == ("ctx", "q", A1, "r")  # prev collapsed to its raw component


def test_escalation_purges_incoming_stale_edges() -> None:
    """Edges INTO an escalated state are stale (future arrivals are
    context-keyed); leaving them cascades violations backwards."""
    g = make_graph()
    g.observe("r", A1, "a")      # r -> a
    g.observe("a", A2, "b")
    g.observe("a", A2, "r")      # violation: 'a' escalates
    assert "a" in g.violated
    # the r -A1-> a edge must be gone, and (r, A1) re-opened as frontier
    assert ("r", A1) not in g.edges
    assert A1 in g.untried("r", [A1, A2])
    # a re-observation of r -A1-> a(ctx) must NOT count as a violation
    a_ctx = g.effective_key("a", ("r", A1))
    assert not g.observe("r", A1, a_ctx).violation


def test_game_over_is_evidence_not_violation() -> None:
    """Timer-driven deaths (masked budget bar) must not shatter identity:
    no escalation, pair stays out of planning only while never-succeeded."""
    g = make_graph()
    g.observe("r", A1, "a")
    res = g.observe("r", A1, None, game_over=True)  # dies where it moved
    assert not res.violation          # NOT an identity violation
    assert g.violation_events == 0 and not g.violated
    assert g.death_contradictions == 1
    assert g.fatal[("r", A1)] == 1
    # the successful successor is retained and still plannable
    assert g.deterministic_successor("r", A1) == "a"
    assert any(a == A1 for a, _ in g.adjacency().get("r", []))


def test_graph_survives_game_over() -> None:
    """GAME_OVER-persistent memory: knowledge accumulates across deaths."""
    g = make_graph()
    g.observe("r", A1, "a")
    g.observe("a", A1, "b")
    g.observe("b", A2, None, game_over=True)
    assert g.fatal[("b", A2)] == 1
    # nothing else was wiped
    assert g.deterministic_successor("r", A1) == "a"
    assert g.deterministic_successor("a", A1) == "b"
    assert len(g.nodes) == 3
    # a pure-fatal pair (no success ever) never enters the adjacency
    assert not any(a == A2 for a, _ in g.adjacency().get("b", []))


def test_win_edge_recorded() -> None:
    g = make_graph()
    g.observe("r", A1, "a")
    res = g.observe("a", A2, None, level_up=True)
    assert res.new_edge and ("a", A2) in g.win_edges


def test_reset_edge_is_ordinary_evidence() -> None:
    g = make_graph()
    g.observe("r", A1, "a")
    g.observe("a", RESET_ACTION, "r")  # observed voluntary reset
    assert g.deterministic_successor("a", RESET_ACTION) == "r"
