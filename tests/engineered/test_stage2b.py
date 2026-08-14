"""Stage-2b unit tests: blocked-move family, win predicate, audit planning.

All synthetic except the ft09 regression pin, which needs the offline engine
(see TestFt09Regression — it replays the exact failure the Stage-2a run
measured: L1 win starved behind gated click_null colors).
"""
from __future__ import annotations

import numpy as np
import pytest

from engineered.effects import (
    EffectConfig,
    EffectEngine,
    MoveBlockedRule,
    WinRule,
)
from engineered.graph import LevelGraph, click_key, simple_key
from engineered.perception import Perception
from engineered.planner import plan

A1, A2, A3, A5 = simple_key(1), simple_key(2), simple_key(3), simple_key(5)


def _perception() -> Perception:
    return Perception("test", [])


def _blob_frame(x0: int, y0: int = 10, color: int = 5,
                decoration: tuple[int, int, int] | None = None) -> np.ndarray:
    f = np.zeros((64, 64), dtype=np.int16)
    f[y0:y0 + 2, x0:x0 + 2] = color
    if decoration is not None:
        y, x, c = decoration
        f[y, x] = c
    return f


# ---------------------------------------------------------------------------
# MoveBlockedRule: legality is memory, not layout
# ---------------------------------------------------------------------------

class TestMoveBlocked:
    def _trained_engine(self, nulls_at_fence: int = 14) -> EffectEngine:
        """A3 moves the blob right except at x0=30 (invisible fence)."""
        eng = EffectEngine(_perception())
        for x in range(2, 30):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        f30 = _blob_frame(30)
        for _ in range(nulls_at_fence):
            eng.observe(f30, A3, f30)
        return eng

    def test_learns_blocked_position(self) -> None:
        eng = self._trained_engine()
        rule = eng.rules[("move_blocked", "A3")]
        assert isinstance(rule, MoveBlockedRule)
        assert rule.blocked[(10, 30)] >= 3
        # the fence position predicts null...
        assert np.array_equal(rule.try_predict(_blob_frame(30), A3),
                              _blob_frame(30))
        # ...an unblocked position declines (BlobRelocate's job, not ours)
        assert rule.try_predict(_blob_frame(20), A3) is None

    def test_generalizes_over_board_state(self) -> None:
        """The whole point: blocked at (10,30) predicts null from a board
        state never seen (a decoration changed elsewhere) — T0 exact memory
        cannot do this, it would re-probe the fence at the new state."""
        eng = self._trained_engine()
        rule = eng.rules[("move_blocked", "A3")]
        f_new = _blob_frame(30, decoration=(50, 50, 7))
        assert np.array_equal(rule.try_predict(f_new, A3), f_new)

    def test_gates_prequentially(self) -> None:
        """The fence nulls after min_fit score as correct predictions; the
        window gate opens like every other family."""
        eng = self._trained_engine(nulls_at_fence=16)
        rule = eng.rules[("move_blocked", "A3")]
        assert rule.gated
        assert rule.accuracy is not None and rule.accuracy >= 0.9

    def test_fence_memory_outranks_translation(self) -> None:
        """A gated TranslationRule predicts a move through the fence (the
        frame shows clear cells ahead); the engine must serve the blocked
        rule's null instead (priority ordering). Translation is re-gated
        after the fence nulls degrade it (fresh successful moves at other
        rows — the fence nulls scored against it, which is prequential
        honesty, not the conflict under test)."""
        eng = self._trained_engine(nulls_at_fence=16)
        for x in range(2, 44):  # re-gate translation on a different row
            eng.observe(_blob_frame(x, y0=40), A3, _blob_frame(x + 1, y0=40))
        assert eng.rules[("translation", "A3")].gated  # the conflict exists
        assert eng.rules[("move_blocked", "A3")].gated
        p = eng.predict("s", _blob_frame(30), A3)
        assert p is not None and p.kind == "null"

    def test_observed_move_unblocks(self) -> None:
        """A door opening (successful move from a 'blocked' position) stops
        the prediction immediately."""
        eng = self._trained_engine()
        rule = eng.rules[("move_blocked", "A3")]
        eng.observe(_blob_frame(30), A3, _blob_frame(31))
        assert rule.try_predict(_blob_frame(30), A3) is None

    def test_declines_on_ambiguous_mover(self) -> None:
        eng = self._trained_engine()
        rule = eng.rules[("move_blocked", "A3")]
        f = _blob_frame(30)
        f[40:42, 30:32] = 5  # second blob of mover color and size
        assert rule.try_predict(f, A3) is None


# ---------------------------------------------------------------------------
# WinRule: cross-level win predicate
# ---------------------------------------------------------------------------

def _win_frame(cells: dict[tuple[int, int], int]) -> np.ndarray:
    f = np.zeros((64, 64), dtype=np.int16)
    for (y, x), c in cells.items():
        f[y, x] = c
    return f


class TestWinRule:
    def _gated_click_rule(self) -> tuple[EffectEngine, WinRule]:
        """Two observed level wins, both clicking color 7."""
        eng = EffectEngine(_perception())
        f_l0 = _win_frame({(5, 5): 7, (9, 9): 3})
        f_l1 = _win_frame({(20, 20): 7, (9, 9): 3, (2, 2): 4})
        eng.observe_win(f_l0, click_key(5, 5))
        eng.observe_win(f_l1, click_key(20, 20))
        rule = eng.rules[("win_sig", "A6")]
        assert isinstance(rule, WinRule)
        return eng, rule

    def test_gates_on_consistent_signature(self) -> None:
        eng, rule = self._gated_click_rule()
        assert rule.gated and rule.signature_color == 7
        assert eng.has_win_rules

    def test_one_win_does_not_gate(self) -> None:
        eng = EffectEngine(_perception())
        eng.observe_win(_win_frame({(5, 5): 7}), click_key(5, 5))
        assert not eng.rules[("win_sig", "A6")].gated

    def test_inconsistent_click_color_does_not_gate(self) -> None:
        eng = EffectEngine(_perception())
        eng.observe_win(_win_frame({(5, 5): 7}), click_key(5, 5))
        eng.observe_win(_win_frame({(5, 5): 9}), click_key(5, 5))
        assert not eng.rules[("win_sig", "A6")].gated

    def test_candidates_target_signature_color(self) -> None:
        eng, _ = self._gated_click_rule()
        f = _win_frame({(30, 31): 7, (9, 9): 3})
        menu = [click_key(31, 30), click_key(9, 9), click_key(1, 1)]
        cands = eng.win_candidates(f, (6,), menu)
        assert cands == [click_key(31, 30)]  # only the color-7 cell

    def test_click_precondition_tolerates_novel_color(self) -> None:
        """vc33 lesson: levels introduce fresh colors, so a novel color must
        NOT block a click predicate (the signature color localizes it)."""
        eng, _ = self._gated_click_rule()
        f = _win_frame({(30, 31): 7, (9, 9): 3, (2, 2): 12})  # 12 is novel
        assert eng.win_candidates(f, (6,), [click_key(31, 30)]) \
            == [click_key(31, 30)]

    def test_click_precondition_requires_always_present(self) -> None:
        """A color present at EVERY win source must be present."""
        eng, _ = self._gated_click_rule()
        f = _win_frame({(30, 31): 7})  # color 3 (present at both wins) missing
        assert eng.win_candidates(f, (6,), [click_key(31, 30)]) == []

    def test_failure_cap_kills_predicate(self) -> None:
        eng, rule = self._gated_click_rule()
        for _ in range(EffectConfig().max_win_failures):
            eng.win_attempt_result(click_key(1, 1), success=False)
        assert rule.dead and not rule.gated and not eng.has_win_rules

    def test_success_keeps_predicate_alive(self) -> None:
        eng, rule = self._gated_click_rule()
        eng.win_attempt_result(click_key(1, 1), success=True)
        assert rule.gated and rule.attempt_successes == 1

    def test_nonclick_signature(self) -> None:
        """A5-style wins: candidates propose the bare action id."""
        eng = EffectEngine(_perception())
        eng.observe_win(_win_frame({(5, 5): 3}), A5)
        eng.observe_win(_win_frame({(6, 6): 3}), A5)
        f = _win_frame({(7, 7): 3})
        assert eng.win_candidates(f, (5, 6), []) == [A5]
        # not offered when ACTION5 is unavailable
        assert eng.win_candidates(f, (6,), []) == []
        # bare-action predicates KEEP the conservative no-novel-color test
        f_novel = _win_frame({(7, 7): 3, (1, 1): 9})
        assert eng.win_candidates(f_novel, (5, 6), []) == []


# ---------------------------------------------------------------------------
# Planner: audit mode (reorder-not-prune) + predicted-win goal
# ---------------------------------------------------------------------------

F = np.zeros((4, 4), dtype=np.int16)


def _cycle_graph() -> LevelGraph:
    """r -A1-> a -A1-> b -A1-> r; A2 untried at r."""
    g = LevelGraph("test", 0)
    for k in ("r", "a", "b"):
        g.ensure_node(k, F, [1, 2])
    g.set_root("r")
    g.observe("r", A1, "a")
    g.observe("a", A1, "b")
    g.observe("b", A1, "r")
    return g


class _NullPort:
    """Fake effects port: predicts 'null' for A2 at r, nothing else."""

    active = True
    has_win = False
    max_pred_depth = 3
    max_pred_nodes = 100

    class _P:
        kind = "null"
        key = None
        frame = None
        penalty = 0.0

    def classify(self, key, akey):
        return self._P() if (key, akey) == ("r", A2) else None

    def chain(self, key, frame, avail):
        return []

    def win_candidates(self, key, menu):
        return []


def test_audit_probes_predicted_pair() -> None:
    """Reorder, never prune: the predicted-null pair is deferred by a normal
    plan (goal 2c only) but an audit plan probes it outright."""
    g = _cycle_graph()
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}
    port = _NullPort()
    normal = plan(g, "b", lambda k: menu[k], effects=port)
    audited = plan(g, "b", lambda k: menu[k], effects=port, audit=True)
    # both must reach the SAME pair here (it is the only untried pair —
    # completeness), but only the audit plan flags it
    assert normal.kind == "frontier" and normal.actions == [A1, A2]
    assert not normal.audit
    assert audited.kind == "frontier" and audited.actions == [A1, A2]
    assert audited.audit and audited.target == "r"


def test_audit_prefers_predicted_over_unpredicted() -> None:
    """With BOTH an unpredicted and a predicted untried pair on offer, the
    normal plan takes the unpredicted one and the audit plan the predicted
    one — the guaranteed 1/N share that unstarves ft09-class wins."""
    g = _cycle_graph()
    g.ensure_node("c", F, [1, 2, 3])
    g.observe("r", A2, "c")  # A2 tried at r now; c has untried A1,A2,A3
    menu = {"r": [A1, A2, A3], "a": [A1], "b": [A1], "c": [A1, A2, A3]}

    class Port(_NullPort):
        def classify(self, key, akey):
            # A3 at r is predicted-null; everything else unknown
            return self._P() if (key, akey) == ("r", A3) else None

    port = Port()
    normal = plan(g, "r", lambda k: menu[k], effects=port, greedy_rank_max=-1)
    audited = plan(g, "r", lambda k: menu[k], effects=port, audit=True)
    assert normal.kind == "frontier" and normal.actions[-1] != A3
    assert audited.audit and audited.actions == [A3]


def test_win_pred_goal_preempts_frontier() -> None:
    """A gated win predicate's candidate is targeted before exploration and
    the plan is marked kind='win_pred' for on-execution verification."""
    g = _cycle_graph()
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}

    class Port(_NullPort):
        has_win = True

        def win_candidates(self, key, menu):
            return [A2] if key == "r" else []

    p = plan(g, "b", lambda k: menu[k], effects=Port())
    assert p.kind == "win_pred"
    assert p.actions == [A1, A2] and p.target == "r"


def test_win_pred_skips_tried_pairs() -> None:
    g = _cycle_graph()
    g.observe("r", A2, "a")  # the candidate pair has been tried: no attempt
    menu = {"r": [A1, A2], "a": [A1], "b": [A1]}

    class Port(_NullPort):
        has_win = True

        def win_candidates(self, key, menu):
            return [A2] if key == "r" else []

    p = plan(g, "b", lambda k: menu[k], effects=Port(), greedy_rank_max=-1)
    assert p.kind != "win_pred"


def test_observed_win_edge_preempts_predicted_win() -> None:
    g = _cycle_graph()
    g.observe("a", A2, None, level_up=True)  # real win edge at a
    menu = {"r": [A1, A2], "a": [A1, A2], "b": [A1]}

    class Port(_NullPort):
        has_win = True

        def win_candidates(self, key, menu):
            return [A2] if key == "r" else []

    p = plan(g, "b", lambda k: menu[k], effects=Port())
    assert p.kind == "win"  # goal 1 wins over goal 1.5


# ---------------------------------------------------------------------------
# ft09 regression pin (offline engine; the Stage-2b deliverable #1)
# ---------------------------------------------------------------------------

class TestFt09Regression:
    """Stage-2a measured: effects ON starved the ft09 L1 win behind gated
    click_null colors (L0 done in 268 actions, then 3732 actions on L1 with
    ZERO of the demoted clicks ever probed — goal 2c is unreachable while
    c=9 clicks keep spawning fresh states). T0-control found the win. The
    audit share must recover it: same budget, >= 2 levels."""

    def test_ft09_recovers_l2_with_effects_on(
            self, arcade: "object", gid_of: dict[str, str]) -> None:
        from engineered.agent import AgentConfig, EngineeredAgent

        env = arcade.make(game_id=gid_of["ft09"], scorecard_id="t-s2b-ft09")
        agent = EngineeredAgent(AgentConfig(budget=4000, wall_s=900,
                                            effects_enabled=True))
        report = agent.play(env, gid_of["ft09"])
        assert report.levels_completed >= 2, (
            f"ft09 regression: {report.levels_completed} levels, "
            f"end={report.end_reason}, audits={report.audit_plans}")
        # the fix must not cost L0's efficiency win (s2a: 268 vs t0c 1984)
        assert report.per_level_actions[0] <= 1000
