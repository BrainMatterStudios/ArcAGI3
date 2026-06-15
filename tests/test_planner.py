"""Unit + integration tests for C6 v2 — the forward-model search planner (planner.py).

The two prior C6 attempts were INERT because they gated on occupancy (occ.usable needs
both movement axes). These tests pin the redesign's two load-bearing properties:

  * UNIT: the planner finds a plan over the C4 forward model with a SINGLE learned axis
    (the exact configuration that killed the occupancy planner), and it NEVER consults
    occupancy.
  * INTEGRATION (the acceptance gate): with ARCAGI3_PLANNER ON, WorldModelPolicy emits
    > 0 planner actions AND reaches the goal-test (wins) on >= 3 local games.
  * NO-REGRESSION: planner ON does not break the slide-physics / delivery games it cannot
    plan (they still win via the proven fallback), and push stays at its baseline action
    count (the delivery guard declines, control falls through unchanged).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ARC_API_KEY", "local-dev")

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")


# ---------------------------------------------------------------------------
# Unit: plan over the forward model with a SINGLE learned axis (no occupancy)
# ---------------------------------------------------------------------------

def _stub_models(deltas, table):
    """Build a MotionModel + StubAffordanceModel for direct planner construction."""
    from arcagi3 import forward_model as FM
    from arcagi3.movement import MotionModel

    mm = MotionModel(avatar_color=1, deltas=dict(deltas), avatar_colors=frozenset({1}))

    class _Aff:
        """Minimal C3-shaped affordance model: .affordance(color) -> Verdict-like."""

        def __init__(self, t):
            self._t = t  # color -> (effect_value, reliable)

        def affordance(self, color):
            from dataclasses import dataclass

            @dataclass
            class V:
                effect: object
                confidence: float
                support: int
                _rel: bool

                @property
                def reliable(self):
                    return self._rel

            ev, rel = self._t.get(int(color), (None, False))
            if ev is None:
                from arcagi3.affordance import Effect
                return V(Effect.NONE, 0.0, 0, False)
            return V(ev, 0.95, 5, rel)

    return mm, _Aff(table)


def test_planner_plans_with_single_axis_no_occupancy():
    """A single learned axis is enough: the planner reaches a GOAL cell to the right.

    deltas = {4: (0, 1)} -> ONLY the +column axis is known. The legacy occupancy planner
    refused here (occ.usable needs both axes); the forward-model planner must still produce
    a multi-step plan. The planner code path also never imports/builds an OccupancyMap.
    """
    from arcagi3.affordance import Effect
    from arcagi3.planner import ForwardPlanner

    mm, aff = _stub_models({4: (0, 1)}, {3: (Effect.GOAL, True)})
    planner = ForwardPlanner(mm, aff)

    bg = 0
    grid = np.zeros((1, 8), dtype=np.int8)
    grid[0, 1] = 1   # avatar (color 1) at col 1
    grid[0, 5] = 3   # GOAL (color 3) at col 5

    plan = planner.plan(grid, bg, goal=None, gi=None, available=[4])
    assert plan is not None, "planner returned no plan on a single-axis reachable GOAL"
    assert len(plan) > 0
    assert all(tok == ("S", 4) for tok in plan), plan
    # forward-model win was modelled (reward_pred>0 path), not a blind reach
    assert planner.stats.plans_found == 1
    assert planner.stats.declines["ok"] == 1


def test_planner_does_not_use_occupancy(monkeypatch):
    """Hard guarantee: planning never constructs spatial.OccupancyMap (the prior failure)."""
    import arcagi3.spatial as spatial
    from arcagi3.affordance import Effect
    from arcagi3.planner import ForwardPlanner

    calls = {"n": 0}
    orig = spatial.OccupancyMap.__init__

    def _tracking_init(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(spatial.OccupancyMap, "__init__", _tracking_init)

    mm, aff = _stub_models({4: (0, 1)}, {3: (Effect.GOAL, True)})
    planner = ForwardPlanner(mm, aff)
    grid = np.zeros((1, 8), dtype=np.int8)
    grid[0, 1] = 1
    grid[0, 4] = 3
    planner.plan(grid, 0, goal=None, gi=None, available=[4])
    assert calls["n"] == 0, "planner constructed an OccupancyMap — must not gate on occupancy"


def test_planner_inert_without_motion():
    """No motion model -> a clean decline (NO_MOTION), never a crash."""
    from arcagi3.movement import MotionModel
    from arcagi3.planner import Decline, ForwardPlanner

    class _Aff:
        def affordance(self, color):
            from arcagi3.affordance import Effect

            class V:
                effect = Effect.NONE
                confidence = 0.0
                support = 0
                reliable = False

            return V()

    mm = MotionModel()  # not ok (no deltas)
    planner = ForwardPlanner(mm, _Aff())
    grid = np.zeros((1, 8), dtype=np.int8)
    assert planner.plan(grid, 0) is None
    assert planner.stats.declines[Decline.NO_MOTION] == 1


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def _make_env(game_id: str):
    import logging

    from arc_agi import Arcade, OperationMode

    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=GAMES_DIR, logger=logging.getLogger("t"))
    return client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")


def _run_planner(game_id: str, budget: int):
    """Drive WorldModelPolicy with the forward planner ON; return (won, stats, actions)."""
    from arcengine import GameAction, GameState

    from arcagi3 import perception as P
    from arcagi3.wm_policy import WMConfig, WorldModelPolicy

    cfg = WMConfig(enabled=True, use_planner=True, use_fwd_planner=True,
                   use_goal_inference=True, min_delegate_actions=0)
    pol = WorldModelPolicy(cfg=cfg, seed=0)
    env = _make_env(game_id)
    obs = env.reset()
    n = 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(
            grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=int(obs.levels_completed or 0),
            available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        n += 1
    won = obs.state == GameState.WIN
    stats = pol.planner.stats if pol.planner is not None else None
    return won, stats, n


# ---------------------------------------------------------------------------
# ACCEPTANCE: planner emits >0 plan actions AND reaches the goal-test (wins)
# on >= 3 local games. (C6_v2.md hard requirement #1.)
# ---------------------------------------------------------------------------

# Games where the C4 forward model is valid (fixed per-action delta) and a goal/subgoal is
# reachable. maze/switchdoor are slide-until-wall (variable stride, single learned axis) and
# are deliberately EXCLUDED from the non-inert assertion — a fixed-delta model cannot path
# them; they are covered by the no-regression test instead.
PLANNER_GAMES = ("collect", "navg", "navgc")


@pytest.mark.parametrize("game_id", PLANNER_GAMES)
def test_planner_not_inert_and_reaches_goal(game_id):
    """Per-game: the planner emits > 0 plan actions and the game is WON (goal-test reached)."""
    won, stats, n = _run_planner(game_id, budget=1500)
    assert stats is not None, f"{game_id}: planner was never built"
    assert stats.actions_emitted > 0, (
        f"{game_id}: planner emitted 0 actions (INERT) — declines={dict(stats.declines)}"
    )
    assert stats.plans_found > 0, f"{game_id}: planner found 0 plans"
    assert won, f"{game_id}: planner-ON run did not reach the goal-test (win) in {n} actions"


def test_planner_not_inert_on_at_least_three_games():
    """Aggregate gate: > 0 plan actions on >= 3 distinct local games (the redesign's headline
    promise vs the prior 0-everywhere planner)."""
    non_inert = []
    for g in PLANNER_GAMES:
        won, stats, _ = _run_planner(g, budget=1200)
        if stats is not None and stats.actions_emitted > 0:
            non_inert.append((g, stats.actions_emitted))
    assert len(non_inert) >= 3, f"planner non-inert on fewer than 3 games: {non_inert}"


# ---------------------------------------------------------------------------
# NO-REGRESSION: planner ON still wins the games it cannot plan; push stays at
# its baseline action count (delivery guard declines -> proven fallback unchanged).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ("maze", "switchdoor"))
def test_planner_on_slide_games_still_win(game_id):
    """Slide-physics games (no plannable fixed-delta model) still WIN via the fallback."""
    won, _stats, _n = _run_planner(game_id, budget=1500)
    assert won, f"{game_id}: planner ON broke a game the fallback should still solve"


def test_planner_on_push_no_regression():
    """push (sokoban) must still win 3/3 at its baseline action count with the planner ON.

    The delivery guard means the planner never commits a wrong block-chase plan, so the
    proven graph fallback solves push exactly as it does flag-OFF (baseline 3051 actions).
    """
    won, _stats, n = _run_planner("push", budget=4000)
    assert won, "push did not win with planner ON"
    assert n == 3051, f"push action count regressed: {n} != 3051 baseline"
