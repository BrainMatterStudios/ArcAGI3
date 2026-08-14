"""End-to-end agent smokes on the offline engine.

The pre-registered Stage-1b smoke expected sb26 L1 "well inside budget" on
the strength of its known 9-action human solution. Measured reality
(2026-08-14): sb26 L0 is a combinatorial assignment puzzle — ~3 100 masked
states (4 slots x palette colors x selection), and the win needs the exact
target config + ACTION5. Blind T0 sweep tried ACTION5 at 203/216 discovered
states in 4 000 actions and did not finish at 20 000 either; the 9-action
solve uses pattern inference T0 deliberately does not do. The sb26 test
below is therefore an xfail with teeth (strict): it documents the measured
negative and will flag if a later layer (T1/T2) unlocks it. tu93 is the
passing end-to-end smoke: the agent must complete L1 well inside budget.
"""
from __future__ import annotations

from typing import Any

import pytest

from engineered.agent import AgentConfig, EngineeredAgent


def _play(arcade: Any, gid_of: dict[str, str], stem: str,
          **cfg: Any) -> Any:
    env = arcade.make(game_id=gid_of[stem], scorecard_id=f"t-e2e-{stem}")
    agent = EngineeredAgent(AgentConfig(**cfg))
    return agent.play(env, gid_of[stem]), agent


def test_tu93_completes_l1_inside_budget(arcade: Any, gid_of: dict[str, str]) -> None:
    report, _ = _play(arcade, gid_of, "tu93", budget=1500, wall_s=120)
    assert report.levels_completed >= 1
    assert report.actions_total <= 1500


def test_agent_respects_action_budget(arcade: Any, gid_of: dict[str, str]) -> None:
    report, agent = _play(arcade, gid_of, "r11l", budget=200, wall_s=60)
    assert report.actions_total <= 200
    assert report.end_reason in ("budget", "win", "frontier_exhausted")
    # every executed action is attributed to a level
    assert sum(report.per_level_actions.values()) == report.actions_total


def test_graph_knowledge_survives_death(arcade: Any, gid_of: dict[str, str]) -> None:
    """tu93 kills regularly; the level-0 graph must keep growing across
    deaths rather than restarting (GAME_OVER-persistent memory)."""
    report, agent = _play(arcade, gid_of, "tu93", budget=600, wall_s=60)
    g0 = agent.graphs[0]
    assert len(g0.fatal) >= 1 or report.levels_completed >= 1
    assert g0.n_transitions > 100          # one graph accumulated everything
    assert len(agent.graphs) >= 1


def test_stop_loss_ends_game(arcade: Any, gid_of: dict[str, str]) -> None:
    """A tight stop-loss (1x human apl on a game we cannot finish in that
    many actions) must end the run with reason 'stop_loss'."""
    report, _ = _play(arcade, gid_of, "sb26", budget=4000, wall_s=60,
                      level_stop_multiplier=1.0, human_apl={"sb26": 23.5})
    assert report.end_reason == "stop_loss"
    assert report.actions_total < 200


def test_r11l_efficiency_with_effects_on(arcade: Any, gid_of: dict[str, str]) -> None:
    """Stage-2a regression guard: r11l is the one game T0 already did right
    (Stage-1b: 37 actions for L1 vs human median 34.5 = 1.07x). With the
    effect model ON it must stay <= 1.5x human median for its first level."""
    report, _ = _play(arcade, gid_of, "r11l", budget=600, wall_s=120)
    assert report.levels_completed >= 1
    assert report.per_level_actions[0] <= 1.5 * 34.5


@pytest.mark.xfail(
    strict=True,
    reason="Measured 2026-08-14: sb26 L0 is a combinatorial assignment "
    "puzzle out of blind-T0 reach (0 levels at 20k actions; ACTION5 swept "
    "at 203/216 states). Flip expected on a T1/T2 unlock.",
)
def test_sb26_completes_l1_inside_budget(arcade: Any, gid_of: dict[str, str]) -> None:
    report, _ = _play(arcade, gid_of, "sb26", budget=4000, wall_s=120)
    assert report.levels_completed >= 1
