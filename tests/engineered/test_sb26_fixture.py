"""The sb26 Stage-0 ground-truth fixture, plus battery budget/resume tests.

Stage-0 debug-sweep ground truth (STAGE0-REPORT.md finding 5): at L0 the ONLY
click-reactive region on sb26 is the answer strip, y 56-60 / x 18-45;
ACTION5/ACTION7 change nothing at L0 except the row-53 budget bar.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from engineered.battery import BatteryConfig, BatteryState, ProbeBattery

STRIP_Y = slice(56, 61)  # y 56-60 inclusive
STRIP_X = slice(18, 46)  # x 18-45 inclusive


@pytest.fixture(scope="module")
def sb26_run(arcade: Any, gid_of: dict[str, str]) -> tuple:
    env = arcade.make(game_id=gid_of["sb26"], scorecard_id="t-sb26-fix")
    return ProbeBattery(BatteryConfig()).run(env)


def test_battery_respects_budget(sb26_run: tuple) -> None:
    profile, state = sb26_run
    assert state.actions_spent <= 16
    assert profile.actions_spent == state.actions_spent


def test_sb26_learns_row53_hud(sb26_run: tuple) -> None:
    """The row-53 budget bar ticks ONLY on ACTION5 — the aux phase is what
    makes it learnable at all."""
    profile, _ = sb26_run
    assert ("row", 53) in profile.hud_lines


def test_sb26_reactive_clicks_hit_the_answer_strip(sb26_run: tuple) -> None:
    profile, _ = sb26_run
    reactive = [c for c in profile.clicks if c["changed"]]
    assert reactive, "battery found no reactive clicks on sb26"
    for c in reactive:
        assert 56 <= c["y"] <= 60 and 18 <= c["x"] <= 45, (
            f"reactive click at ({c['y']},{c['x']}) outside the answer strip"
        )


def test_sb26_reactive_mask_overlaps_the_strip(sb26_run: tuple) -> None:
    profile, _ = sb26_run
    assert profile.reactive_mask[STRIP_Y, STRIP_X].any()
    # and the strip must not be classified dead where it reacted
    assert not (profile.dead_mask & profile.reactive_mask).any()


def test_sb26_archetype_is_click(sb26_run: tuple) -> None:
    profile, _ = sb26_run
    assert profile.archetype == "click"
    assert profile.self_mask.sum() == 0


def test_battery_resume_equals_single_shot(
    arcade: Any, gid_of: dict[str, str], sb26_run: tuple
) -> None:
    """Budget-capped incremental use: suspending after every action and
    resuming (state round-tripped through JSON) must reproduce the
    single-shot battery byte for byte."""
    single_profile, single_state = sb26_run
    env = arcade.make(game_id=gid_of["sb26"], scorecard_id="t-sb26-resume")
    battery = ProbeBattery(BatteryConfig())
    profile, state = battery.run(env, max_actions=1)
    while state.phase != "done" and state.actions_spent < battery.config.budget:
        state = BatteryState.from_json(state.to_json())  # exercise round-trip
        profile, state = battery.run(env, state=state, max_actions=1)
    assert state.actions_spent == single_state.actions_spent
    assert [s.diff for s in state.steps] == [s.diff for s in single_state.steps]
    assert profile.to_json() == single_profile.to_json()
