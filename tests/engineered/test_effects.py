"""Stage-2a unit tests: rule learning, accuracy gates, demotion, planning.

All synthetic — no engine needed. The e2e efficiency test lives in
test_agent_e2e.py (it needs the offline engine).
"""
from __future__ import annotations

import numpy as np
import pytest

from engineered.effects import (
    EffectConfig,
    EffectEngine,
    Prediction,
)
from engineered.graph import LevelGraph, click_key, simple_key
from engineered.perception import Perception
from engineered.planner import plan

A1, A2, A3 = simple_key(1), simple_key(2), simple_key(3)


def _perception() -> Perception:
    return Perception("test", [])


def _blob_frame(x0: int, y0: int = 10, color: int = 5) -> np.ndarray:
    f = np.zeros((64, 64), dtype=np.int16)
    f[y0:y0 + 2, x0:x0 + 2] = color
    return f


# ---------------------------------------------------------------------------
# Rule learning on synthetic transition sets
# ---------------------------------------------------------------------------

class TestTranslationLearning:
    def test_learns_and_generalizes(self) -> None:
        eng = EffectEngine(_perception())
        for x in range(2, 40):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        rule = eng.rules[("translation", "A3")]
        assert rule.gated
        assert rule.accuracy is not None and rule.accuracy >= 0.9
        # generalizes to a position never seen during training
        p = eng.predict("s", _blob_frame(50), A3)
        assert p is not None and p.kind == "edge"
        assert np.array_equal(p.frame, _blob_frame(51))

    def test_declines_when_blocked(self) -> None:
        eng = EffectEngine(_perception())
        for x in range(2, 40):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        f = _blob_frame(30)
        f[10:12, 32] = 7  # obstacle directly in the path
        assert eng.predict("s", f, A3) is None

    def test_gate_needs_min_evals(self) -> None:
        eng = EffectEngine(_perception())
        for x in range(2, 8):  # too few held-out outcomes
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        assert not eng.rules[("translation", "A3")].gated
        assert eng.predict("s", _blob_frame(50), A3) is None


class TestRecolorLearning:
    def test_learns_global_toggle(self) -> None:
        eng = EffectEngine(_perception())
        f_a = np.zeros((64, 64), dtype=np.int16)
        f_a[5:9, 5:9] = 3
        f_a[20:24, 20:24] = 7
        f_b = f_a.copy()
        f_b[f_a == 3] = 7
        f_b[f_a == 7] = 3
        for _ in range(20):  # 3<->7 toggle back and forth
            eng.observe(f_a, A1, f_b)
            eng.observe(f_b, A1, f_a)
        rule = eng.rules[("colormap", "A1")]
        assert rule.gated and rule.mapping in ({3: 7, 7: 3},)
        # generalizes to an unseen board with the same palette
        g = np.zeros((64, 64), dtype=np.int16)
        g[40:42, 40:42] = 3
        p = eng.predict("s", g, A1)
        assert p is not None and p.kind == "edge"
        expect = g.copy()
        expect[g == 3] = 7
        assert np.array_equal(p.frame, expect)


class TestClickMappingLearning:
    """T2: the generalization over ACTION6 coordinates."""

    def _tile_frame(self) -> np.ndarray:
        f = np.zeros((64, 64), dtype=np.int16)
        for i in range(6):
            for j in range(6):
                f[2 + 6 * i:6 + 6 * i, 2 + 6 * j:6 + 6 * j] = 4
        return f

    def test_click_recolor_generalizes_across_coordinates(self) -> None:
        eng = EffectEngine(_perception())
        f = self._tile_frame()
        # click 15 different tiles; each clicked tile recolors 4 -> 9
        for k in range(15):
            i, j = divmod(k, 6)
            x, y = 3 + 6 * j, 3 + 6 * i
            dst = f.copy()
            dst[2 + 6 * i:6 + 6 * i, 2 + 6 * j:6 + 6 * j] = 9
            eng.observe(f, click_key(x, y), dst)
        rule = eng.rules[("click_recolor", "click:c=4")]
        assert rule.gated and rule.to_color == 9
        # predicts for a tile that was NEVER clicked (coordinate transfer)
        p = eng.predict("s", f, click_key(3 + 6 * 5, 3 + 6 * 5))
        assert p is not None and p.kind == "edge"
        expect = f.copy()
        expect[2 + 30:6 + 30, 2 + 30:6 + 30] = 9
        assert np.array_equal(p.frame, expect)

    def test_click_null_feeds_menu_not_edges(self) -> None:
        eng = EffectEngine(_perception())
        f = self._tile_frame()
        for i in range(15):  # clicking background never does anything
            eng.observe(f, click_key(0, i), f)
        rule = eng.rules[("click_null", "click:c=0")]
        assert rule.gated
        p = eng.predict("s", f, click_key(0, 40))
        assert p is not None and p.kind == "null"


# ---------------------------------------------------------------------------
# Accuracy-gate enforcement: an 85% rule never feeds the planner
# ---------------------------------------------------------------------------

class TestGateEnforcement:
    def test_85_percent_rule_never_gates(self) -> None:
        cfg = EffectConfig()
        eng = EffectEngine(_perception(), cfg)
        # 40 transitions; every 7th breaks the translation (~85.7% correct)
        for k, x in enumerate(range(2, 42)):
            src = _blob_frame(x)
            if k % 7 == 3:
                dst = src.copy()  # a null where the rule expects a move
            else:
                dst = _blob_frame(x + 1)
            eng.observe(src, A3, dst)
            if k % 7 == 3:
                # re-anchor so subsequent pairs stay clean translations
                pass
        rule = eng.rules[("translation", "A3")]
        assert rule.window_accuracy is not None
        assert rule.window_accuracy < cfg.gate_acc
        assert not rule.gated
        assert eng.predict("s", _blob_frame(50), A3) is None

    def test_degradation_ungates(self) -> None:
        eng = EffectEngine(_perception())
        for x in range(2, 30):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        rule = eng.rules[("translation", "A3")]
        assert rule.gated
        # mechanic changes: the action stops working; window decays
        f = _blob_frame(40)
        for _ in range(30):
            eng.observe(f, A3, f)
        assert not rule.gated
        assert eng.predict("sZ", _blob_frame(50), A3) is None


# ---------------------------------------------------------------------------
# Misprediction demotion
# ---------------------------------------------------------------------------

class TestMispredictionDemotion:
    def _gated_engine(self) -> EffectEngine:
        eng = EffectEngine(_perception())
        for x in range(2, 40):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        assert eng.rules[("translation", "A3")].gated
        return eng

    def test_live_mispredict_demotes_and_flags_state(self) -> None:
        eng = self._gated_engine()
        p = eng.predict("sK", _blob_frame(50), A3)
        assert p is not None and p.kind == "edge"
        eng.live_mispredict("sK", A3)
        rule = eng.rules[("translation", "A3")]
        assert not rule.gated                      # gate closed immediately
        assert rule.live_mispredictions == 1
        assert "sK" in eng.suspect
        # the flagged state never gets predictions again, even after the
        # rule re-earns its gate
        for x in range(2, 20):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        assert eng.rules[("translation", "A3")].gated
        assert eng.predict("sK", _blob_frame(50), A3) is None
        assert eng.predict("sOther", _blob_frame(50), A3) is not None

    def test_reearn_requires_fresh_window(self) -> None:
        eng = self._gated_engine()
        assert eng.predict("sK", _blob_frame(50), A3) is not None  # issued
        eng.live_mispredict("sK", A3)
        rule = eng.rules[("translation", "A3")]
        assert len(rule.window) == 0               # must re-earn from scratch
        for x in range(2, 2 + rule.cfg.min_evals):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        assert rule.gated

    def test_level_up_probation(self) -> None:
        """On level-up every rule keeps its hypothesis but must re-earn its
        gate from the new level's own transitions; suspect flags (level-local
        state keys) are cleared."""
        eng = self._gated_engine()
        eng.suspect.add("sOld")
        eng.on_level_up()
        rule = eng.rules[("translation", "A3")]
        assert not rule.gated and len(rule.window) == 0
        assert not eng.suspect
        assert eng.predict("s", _blob_frame(50), A3) is None
        # hypothesis survived: min_evals clean transitions re-open the gate
        for x in range(2, 2 + rule.cfg.min_evals):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        assert rule.gated
        assert eng.predict("s", _blob_frame(50), A3) is not None

    def test_persistent_live_failure_kills_rule(self) -> None:
        eng = self._gated_engine()
        rule = eng.rules[("translation", "A3")]
        cap = rule.cfg.max_live_mispred
        for i in range(cap):
            for x in range(2, 2 + rule.cfg.min_evals):  # re-earn the gate
                eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
            eng.predict(f"s{i}", _blob_frame(50), A3)
            eng.live_mispredict(f"s{i}", A3)
        assert rule.dead
        for x in range(2, 40):
            eng.observe(_blob_frame(x), A3, _blob_frame(x + 1))
        assert not rule.gated                      # dead rules stay out


# ---------------------------------------------------------------------------
# Planner integration: predicted edges + commit mode
# ---------------------------------------------------------------------------

class _FakeEffects:
    """Scriptable EffectsPort."""

    active = True
    max_pred_depth = 4
    max_pred_nodes = 100

    def __init__(self) -> None:
        self.edges: dict[tuple, tuple] = {}   # (key, akey) -> (dst, penalty)
        self.nulls: set[tuple] = set()
        self.chains: dict[object, list] = {}

    def classify(self, key, akey):
        if (key, akey) in self.nulls:
            return Prediction(kind="null")
        hit = self.edges.get((key, akey))
        if hit is None:
            return None
        dst, pen = hit
        return Prediction(kind="edge", key=dst, frame=None, penalty=pen)

    def chain(self, key, frame, avail):
        return self.chains.get(key, [])


def _line_graph(b_has_a2: bool = True) -> tuple[LevelGraph, dict]:
    """Cycle r -a1-> a -a1-> b -a1-> r; a2 untried at r (and b if enabled)."""
    g = LevelGraph("t", 0)
    f = np.zeros((2, 2))
    for k in ("r", "a", "b"):
        g.ensure_node(k, f, [1, 2])
    g.set_root("r")
    g.observe("r", A1, "a")
    g.observe("a", A1, "b")
    g.observe("b", A1, "r")
    menu = {"r": [A1, A2], "a": [A1], "b": [A1, A2] if b_has_a2 else [A1]}
    return g, menu


class TestPlannerWithEffects:
    def test_unpredicted_probe_preferred_over_predicted(self) -> None:
        g, menu = _line_graph()
        fx = _FakeEffects()
        # a2@b predicted to land on the KNOWN state a -> no longer frontier;
        # the unpredicted a2@r becomes the probe even though it is farther
        fx.edges[("b", A2)] = ("a", 0.25)
        p = plan(g, "b", lambda k: menu[k], effects=fx)
        assert p.kind == "frontier" and p.target == "r"
        assert p.actions == [A1, A2]

    def test_predicted_null_demoted_from_frontier(self) -> None:
        g, menu = _line_graph()
        fx = _FakeEffects()
        fx.nulls.add(("b", A2))
        p = plan(g, "b", lambda k: menu[k], effects=fx)
        assert p.kind == "frontier" and p.target == "r"

    def test_novel_predicted_state_becomes_target(self) -> None:
        g, menu = _line_graph()
        fx = _FakeEffects()
        fx.edges[("b", A2)] = ("NOVEL", 0.25)   # never-seen state
        fx.nulls.add(("r", A2))                 # nothing unpredicted left
        p = plan(g, "b", lambda k: menu[k], effects=fx)
        assert p.kind == "frontier" and p.target == "NOVEL"
        assert p.actions == [A2]
        assert p.predicted == [True]

    def test_all_predicted_pairs_still_probed_eventually(self) -> None:
        """Completeness: predictions reorder the sweep, never truncate it."""
        g, menu = _line_graph()
        fx = _FakeEffects()
        fx.nulls.add(("b", A2))
        fx.edges[("r", A2)] = ("a", 0.25)  # predicted -> known state
        p = plan(g, "b", lambda k: menu[k], effects=fx)
        # no unpredicted pair and no novel state -> goal 2c probes the
        # cheapest predicted pair anyway
        assert p.kind == "frontier"
        assert p.target in ("r", "b")


class TestCommitMode:
    def test_commit_switches_when_win_observed(self) -> None:
        """Before the win edge is known: explore. The moment a level-up
        transition is observed: every plan is the shortest win replay and
        exploration stops, even with untried frontier left (including a
        rank-0 greedy probe at the CURRENT node, which must be preempted)."""
        g, menu = _line_graph()
        p = plan(g, "b", lambda k: menu[k])
        assert p.kind == "frontier"            # exploring before the win
        g.observe("a", A2, None, level_up=True)
        menu["a"] = [A1, A2]
        p2 = plan(g, "b", lambda k: menu[k])
        assert p2.kind == "win"
        assert p2.actions == [A1, A1, A2]      # b -> r -> a, then the win
        # untried frontier still exists (a2@r, a2@b) but is NOT chosen
        assert g.untried("r", menu["r"]) and g.untried("b", menu["b"])

    def test_commit_path_may_ride_predicted_edges(self) -> None:
        """A win source unreachable via observed edges is reached through a
        gated-rule predicted edge; the predicted step is flagged for
        verification."""
        g = LevelGraph("t", 0)
        f = np.zeros((2, 2))
        for k in ("r", "w"):
            g.ensure_node(k, f, [1, 2])
        g.set_root("r")
        g.observe("w", A2, None, level_up=True)   # win known at w
        menu = {"r": [A1, A2], "w": [A1, A2]}
        p0 = plan(g, "r", lambda k: menu[k])
        assert p0.kind == "frontier"              # w unreachable: no commit
        fx = _FakeEffects()
        fx.edges[("r", A1)] = ("w", 0.25)
        p = plan(g, "r", lambda k: menu[k], effects=fx)
        assert p.kind == "win"
        assert p.actions == [A1, A2]
        assert p.predicted == [True, False]

    def test_commit_prefers_shortest_path(self) -> None:
        g, menu = _line_graph()
        g.observe("b", A2, None, level_up=True)   # win right here at b
        p = plan(g, "b", lambda k: menu[k])
        assert p.kind == "win" and p.actions == [A2] and p.cost == 1


def test_planner_without_effects_is_stage1_t0() -> None:
    """effects=None keeps the exact Stage-1 behavior (regression guard)."""
    g, menu = _line_graph(b_has_a2=False)
    p = plan(g, "b", lambda k: menu[k])
    assert p.kind == "frontier" and p.actions == [A1, A2]
    # greedy fast path still fires when the current node has cheap frontier
    g2, menu2 = _line_graph(b_has_a2=True)
    p2 = plan(g2, "b", lambda k: menu2[k])
    assert p2.kind == "frontier" and p2.actions == [A2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
