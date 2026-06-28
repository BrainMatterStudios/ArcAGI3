"""Tests for StochasticGooseExplorer — the faithful standalone CNN-RL policy.

The CNN is dependency-injected (FakeModel) so the policy LOGIC (masked-key dedup, cold-start
order, model-guided sampling, cross-level model carry, stuck-handling, determinism) is tested
without torch. A separate light test exercises the real torch-backed model on CPU.
"""
from __future__ import annotations

import numpy as np

from arcagi3.stochastic_goose_explorer import StochasticGooseExplorer


def _grid(fill=0):
    return np.full((64, 64), fill, dtype=np.int8)


def _grid_with_obj(color=3, at=(10, 10)):
    g = _grid(0)
    r, c = at
    g[r:r + 2, c:c + 2] = color
    return g


class FakeModel:
    """Injectable stand-in for the torch effect-CNN. `scores` returns a fixed per-action prob."""

    def __init__(self, ready=False, favor=None, prob=0.99):
        self._ready = ready
        self.favor = favor          # action tuple to score highest
        self.prob = prob
        self.observes = []          # (action, changed) log
        self.reset_level_calls = 0
        self.reset_game_calls = 0

    def ready(self):
        return self._ready

    def scores(self, one_hot, cands):
        return {a: (self.prob if a == self.favor else 0.01) for a in cands}

    def observe(self, one_hot, action, changed):
        self.observes.append((action, changed))

    def reset_level(self):
        self.reset_level_calls += 1

    def reset_game(self):
        self.reset_game_calls += 1


def _decide(pol, grid, levels=0, terminal=False, notplayed=False, available=(1, 2)):
    return pol.decide(grid, gstate_terminal=terminal, gstate_notplayed=notplayed,
                      levels=levels, available=list(available))


def test_terminal_returns_reset():
    pol = StochasticGooseExplorer(seed=0, model=FakeModel())
    assert _decide(pol, _grid(), terminal=True) == ("reset",)


def test_notplayed_returns_reset():
    pol = StochasticGooseExplorer(seed=0, model=FakeModel())
    assert _decide(pol, _grid(), notplayed=True) == ("reset",)


def test_cold_start_returns_an_available_simple_action():
    pol = StochasticGooseExplorer(seed=0, model=FakeModel(ready=False))
    tok = _decide(pol, _grid_with_obj(), available=(1, 2))
    assert tok[0] == "S" and tok[1] in (1, 2)


def test_dedup_does_not_repeat_a_tried_action_at_same_state():
    # cold-start, model not ready; two simple actions both tier 0; the SAME grid each call
    pol = StochasticGooseExplorer(seed=0, model=FakeModel(ready=False))
    g = _grid_with_obj()
    a1 = _decide(pol, g, available=(1, 2))
    a2 = _decide(pol, g, available=(1, 2))   # same key -> must pick the OTHER untried action
    assert a1[0] == "S" and a2[0] == "S"
    assert a1[1] != a2[1]


def test_model_guided_picks_highest_effect_untried_action():
    fav = ("S", 2)
    pol = StochasticGooseExplorer(seed=0, temperature=0.0,  # greedy
                                  model=FakeModel(ready=True, favor=fav))
    tok = _decide(pol, _grid_with_obj(), available=(1, 2))
    assert tok == fav


def test_model_is_carried_across_levels_not_reset():
    fm = FakeModel(ready=True, favor=("S", 1))
    pol = StochasticGooseExplorer(seed=0, model=fm)
    _decide(pol, _grid_with_obj(at=(5, 5)), levels=0, available=(1, 2))
    _decide(pol, _grid_with_obj(at=(7, 7)), levels=0, available=(1, 2))
    _decide(pol, _grid_with_obj(at=(9, 9)), levels=1, available=(1, 2))  # LEVEL-UP
    _decide(pol, _grid_with_obj(at=(11, 11)), levels=1, available=(1, 2))
    # faithful StochasticGoose retrains BETWEEN levels (carries the model); never resets it
    assert fm.reset_level_calls == 0
    assert fm.reset_game_calls == 0
    assert len(fm.observes) >= 2  # transitions kept flowing into the same model


def test_observe_records_change_label_from_key_delta():
    fm = FakeModel(ready=True, favor=("S", 1))
    pol = StochasticGooseExplorer(seed=0, model=fm)
    _decide(pol, _grid_with_obj(at=(5, 5)), available=(1, 2))      # take action at state A
    _decide(pol, _grid_with_obj(at=(40, 40)), available=(1, 2))   # state changed -> changed=True
    assert fm.observes and fm.observes[0][1] is True


def test_stuck_returns_valid_available_action_not_crash():
    # only one candidate; exhaust it, then the next call is "stuck" -> still returns a valid action
    pol = StochasticGooseExplorer(seed=0, model=FakeModel(ready=False))
    g = _grid_with_obj()
    _decide(pol, g, available=(1,))
    tok = _decide(pol, g, available=(1,))   # all untried exhausted at this key
    assert tok == ("S", 1)                  # valid available action, never ("reset",) off-terminal


def test_deterministic_under_fixed_seed():
    def run():
        pol = StochasticGooseExplorer(seed=7, model=FakeModel(ready=True, favor=("S", 2)))
        return [tuple(_decide(pol, _grid_with_obj(at=(i, i)), available=(1, 2)))
                for i in range(5)]
    assert run() == run()


def test_real_torch_model_scores_are_probabilities():
    # light integration: the real effect-CNN builds and returns per-action probs in [0,1] on CPU
    import os
    os.environ.setdefault("ARCAGI3_ALLOW_CPU", "1")
    from arcagi3.stochastic_goose_explorer import GooseEffectModel
    m = GooseEffectModel(min_train=0, allow_cpu=True)
    oh = np.zeros((16, 64, 64), dtype=np.float32)
    cands = [("S", 1), ("C", 10, 12)]
    sc = m.scores(oh, cands)
    assert set(sc) == set(cands)
    assert all(0.0 <= v <= 1.0 for v in sc.values())
