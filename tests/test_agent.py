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


def _run(game_id, budget):
    from arcagi3.runner import run_game

    return run_game(game_id, GAMES_DIR, budget)


def test_navg_wins():
    r = _run("navg", 30000)
    assert r.won and r.levels_completed == r.win_levels


def test_btnc_wins_efficiently():
    r = _run("btnc", 2000)
    assert r.won
    assert r.actions < 200  # object-centric click proposal must be efficient


def test_push_makes_progress():
    # blind baseline clears at least the first level; efficiency improved later
    r = _run("push", 6000)
    assert r.levels_completed >= 1
