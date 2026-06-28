"""Tests for RewardRLLearner — the deep-net, scale-tuned reward-driven RL learner.

Covers the faithful StochasticGoose behaviours: 4-layer net forward, model CARRIED across levels (not
reset), abstain-before-any-reward (coverage preserved pre-first-levelup), and reward back-labelling.
"""
from __future__ import annotations

import numpy as np

from arcagi3.reward_rl_learner import RewardRLLearner, _build_net_deep


class FakeNode:
    def __init__(self, cands):
        self.cands = list(cands)
        self.edges = {}


def _grid(fill=0):
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:12, 10:12] = 3
    return g + fill * 0


def test_deep_net_has_four_conv_layers_and_correct_output_shapes():
    import torch
    net = _build_net_deep(torch)
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len([c for c in convs if c.kernel_size == (3, 3)]) == 4   # 4 feature conv layers
    click, simple = net(torch.zeros(1, 16, 64, 64))
    assert tuple(click.shape) == (1, 64, 64)
    assert tuple(simple.shape) == (1, 5)


def test_abstains_before_any_reward_seen():
    lrn = RewardRLLearner(min_reward=1)
    lrn.see(_grid())
    node = FakeNode([("S", 1), ("S", 2)])
    assert lrn.act(_grid(), node, b"k") is None   # n_rewards=0 -> abstain -> coverage backbone runs


def test_model_carried_across_levels_not_reset():
    lrn = RewardRLLearner()
    lrn.see(_grid())
    # drive one reward so a net is built and the buffer is populated
    lrn.observe(b"a", ("S", 1), b"b", 0.0)
    lrn.see(_grid())
    lrn.observe(b"b", ("S", 2), b"c", 1.0)        # level-up -> back-label + n_rewards=1
    net_before = lrn._net_ready()
    buf_before = len(lrn._buf)
    lrn.reset_level()                              # faithful: keep net + buffer across levels
    assert lrn._net is net_before                  # SAME net object (not rebuilt)
    assert len(lrn._buf) == buf_before             # buffer preserved
    assert lrn._traj == []                         # only the in-progress trajectory resets


def test_reward_backlabels_trajectory_with_gamma_discount():
    lrn = RewardRLLearner(gamma=0.9, horizon=48)
    lrn.see(_grid()); lrn.observe(b"a", ("S", 1), b"b", 0.0)
    lrn.see(_grid()); lrn.observe(b"b", ("S", 2), b"c", 0.0)
    lrn.see(_grid()); lrn.observe(b"c", ("S", 3), b"d", 1.0)   # reward here
    # three steps back-labelled: closest-to-reward gets gamma^0=1.0, then 0.9, then 0.81
    targets = sorted(t for _oh, _a, t in lrn._buf)
    assert targets[-1] == 1.0
    assert abs(targets[-2] - 0.9) < 1e-6
    assert abs(targets[-3] - 0.81) < 1e-6


def test_reset_game_clears_rewards_and_net():
    lrn = RewardRLLearner()
    lrn.see(_grid()); lrn.observe(b"a", ("S", 1), b"b", 1.0)
    assert lrn._n_rewards == 1
    lrn.reset_game()                               # full game restart -> fresh model
    assert lrn._n_rewards == 0
    assert lrn._net is None
