"""Regression tests: the general agent should solve the local dev games.

These spin up the real arcengine offline, so they double as integration tests for the
toolkit wiring. Budgets are generous; the point is correctness, not efficiency (efficiency
is tracked separately by the runner).
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("ARC_API_KEY", "local-dev")

GAMES_DIR = str(Path(__file__).parent.parent / "src" / "arcagi3" / "games")


def _run(game_id, budget, agent_name="hybrid"):
    from arcagi3.runner import run_game

    return run_game(game_id, GAMES_DIR, budget, agent_name=agent_name)


def test_navg_wins_efficiently():
    # motion model + coordinate navigation must solve nav cheaply (not blind BFS)
    r = _run("navg", 4000)
    assert r.won and r.levels_completed == r.win_levels
    assert r.actions < 600


def test_btnc_wins_efficiently():
    r = _run("btnc", 2000)
    assert r.won
    assert r.actions < 200  # object-centric click proposal must be efficient


def test_push_makes_progress():
    # sokoban planning not yet implemented; ensure no regression below 1 level
    r = _run("push", 6000)
    assert r.levels_completed >= 1


def test_explorer_baseline_still_solves_clicks():
    r = _run("btnc", 2000, agent_name="explorer")
    assert r.won


# --- reactive policy (the submission-shaped, one-action-per-call interface) ---

def test_reactive_solves_all_local_games():
    for gid in ("navg", "btnc", "push"):
        r = _run(gid, 4000, agent_name="reactive")
        assert r.won, f"reactive failed to win {gid}: {r}"


def test_reactive_navg_efficient():
    r = _run("navg", 4000, agent_name="reactive")
    assert r.actions < 600
