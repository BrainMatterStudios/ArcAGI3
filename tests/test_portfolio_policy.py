"""Tests for PortfolioPolicy — strategy delegation + the new-play SAFETY abort."""
from __future__ import annotations

import numpy as np

from arcagi3.portfolio_policy import PortfolioPolicy


class StubStrat:
    """Minimal strategy: returns a fixed simple action; never completes a level."""

    def __init__(self, action_id=1):
        self.action_id = action_id

        class _GS:
            wm = {}
        self.gs = _GS()

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        return ("S", self.action_id)


def _grid():
    return np.zeros((64, 64), dtype=np.int8)


def _strats(n=3):
    return [(f"s{i}", (lambda a: (lambda seed: StubStrat(a)))(i + 1)) for i in range(n)]


def test_delegates_to_current_strategy():
    p = PortfolioPolicy(strategies=_strats(2))
    assert p.decide(_grid(), False, False, 0, [1, 2]) == ("S", 1)  # strategy 0


def test_transitions_to_next_strategy_after_level_stall():
    p = PortfolioPolicy(strategies=_strats(2), level_stall_limit=3)
    toks = []
    for _ in range(8):
        p._last_full_reset = True             # engine creates new plays -> safety satisfied
        toks.append(p.decide(_grid(), False, False, 0, [1, 2]))
    assert ("reset",) in toks                 # a play transition was issued
    assert p.idx == 1                          # advanced to strategy 1 (safety not tripped)
    assert p._multiplay_broken is False


def test_safety_aborts_when_no_full_reset_observed():
    # if the engine never flags full_reset during the transition, multiplay is "broken" ->
    # revert to strategy 0 and stop transitioning (cannot regress below the banked best).
    p = PortfolioPolicy(strategies=_strats(3), level_stall_limit=2)
    p._last_full_reset = False                 # engine never creates a new play
    for _ in range(12):
        p.decide(_grid(), False, False, 0, [1, 2])
    assert p._multiplay_broken is True
    assert p.idx == 0                          # reverted to the safe banked strategy


def test_safety_not_tripped_when_full_reset_seen():
    p = PortfolioPolicy(strategies=_strats(3), level_stall_limit=2)
    # simulate the engine flagging full_reset during the reset injections
    for i in range(12):
        p._last_full_reset = (i % 2 == 1)      # alternating -> some True during transitions
        p.decide(_grid(), False, False, 0, [1, 2])
    assert p._multiplay_broken is False
