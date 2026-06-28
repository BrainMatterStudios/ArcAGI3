"""RewardRLLearner — the faithful StochasticGoose-class reward-driven RL learner at SCALE.

This is ValueExploitLearner (reward back-labelled gamma^d, kept across levels, confidence-gated
abstain-default exploit) but with StochasticGoose's DEEPER 4-layer CNN and scale-tuned hyperparameters,
purpose-built for the T4 eval path. The campaign killed ValueExploitLearner at CPU/2-layer/~8k-action
scale because its shallow-level value was confidently WRONG on deeper levels (lp85 L5->L2, tu93 L5->L3).
The OPEN QUESTION this learner exists to settle: is that deep-level divergence a CAPACITY/SCALE artifact
of a tiny 2-layer net trained briefly on CPU, or fundamental? StochasticGoose (4-layer CNN + off-policy
RL on reward + retrain between levels, at T4 scale) is the leaders' actual recipe; this reproduces it
faithfully so a T4 run can answer the question definitively.

Slots into the LearnedExplorer harness (propose mode, T4 fail-safe, abstain-default firewall) exactly like
ValueExploitLearner — enable_learn=False / no working CUDA op -> byte-identical to banked TransferExplorer.
"""
from __future__ import annotations

from .value_exploit_learner import ValueExploitLearner


def _build_net_deep(torch):
    """StochasticGoose-faithful 4-conv-layer effect/value net. Same output interface as
    learned_dynamics._build_net (per-cell click map + 5 simple-action logits) so it is a drop-in."""
    nn = torch.nn

    class _DeepNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(16, 32, 3, padding=1)
            self.c2 = nn.Conv2d(32, 32, 3, padding=1)
            self.c3 = nn.Conv2d(32, 16, 3, padding=1)
            self.c4 = nn.Conv2d(16, 8, 3, padding=1)
            self.click = nn.Conv2d(8, 1, 1)
            self.simple = nn.Linear(8, 5)

        def forward(self, x):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            h = torch.relu(self.c3(h))
            h = torch.relu(self.c4(h))
            click = self.click(h).squeeze(1)
            simple = self.simple(h.mean(dim=(2, 3)))
            return click, simple

    return _DeepNet()


class RewardRLLearner(ValueExploitLearner):
    """Deep-net, scale-tuned reward-driven RL learner. Defaults lean toward the T4 regime: a longer reward
    horizon and a larger replay buffer than the killed CPU baseline, so the gamma-discounted value has more
    context and the off-policy buffer holds more reward-bearing trajectories."""

    def __init__(self, gamma: float = 0.95, horizon: int = 48, min_reward: int = 1,
                 confidence: float = 0.6, train_every: int = 4, batch: int = 32, lr: float = 1e-3,
                 buffer: int = 20000) -> None:
        super().__init__(gamma=gamma, horizon=horizon, min_reward=min_reward, confidence=confidence,
                         train_every=train_every, batch=batch, lr=lr, buffer=buffer)

    def _net_ready(self):
        if self._net is None:
            torch = self._ensure_torch()
            self._net = _build_net_deep(torch)
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg["lr"])
        return self._net
