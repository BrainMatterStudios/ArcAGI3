"""ValueRankerExplorer — per-game distance-to-milestone value ranking (Blind-Squirrel recipe).

Corrects the specific error in our killed CNN prune-probe (which failed CROSS-GAME, predicting
on-path-to-GOAL). This is the reproducible mid-cluster construction instead:
  - PER-GAME (not cross-game): a small value net trained online on THIS game only.
  - Labels = graph-distance to a reward we ALREADY hit (BFS back-label from each level-up over
    the observed graph). ONE reward -> hundreds of (state, distance) labels, defeating the
    "<5 positive labels" sparsity objection (that was reward-CLASSIFICATION; this is regression).
  - Used only to RANK within-tier ties (prefer frontiers the value net thinks are closer to the
    next milestone), never to prune coverage. Primary nearest-first order is preserved. Gated:
    until the first reward the net is untrained -> pure SalienceExplorer (0.33 floor preserved).

Why it can beat the wall where transfer/struct didn't: it does NOT identify the goal; it learns
a SMOOTH distance-to-the-found-reward that generalizes to the next level (escalating levels share
mechanics) even when shapes/colors change (so it fires where discrete color/shape transfer went
inert). Fail-safe: no torch / GPU-only at eval -> behaves as v6.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import perception as P
from .salience_explorer import SalienceExplorer

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


if _TORCH:
    class _ValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            return self.fc(self.net(x).flatten(1)).squeeze(1)


class ValueRankerExplorer(SalienceExplorer):
    def __init__(self, *args, enable_value: bool = True, train_steps: int = 200, **kwargs) -> None:
        self.enable_value = bool(enable_value) and _TORCH
        self.train_steps = int(train_steps)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.grid_by_key: dict = {}
        self.value_net = None
        self._val_cache: dict = {}
        self._cur_grid = None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_value and not gstate_terminal and not gstate_notplayed:
            if self.bg is None:
                self.bg = P.detect_background(grid)
            self._cur_grid = grid
            self.grid_by_key.setdefault(self._key(grid), grid)
            if self.prev_action is not None and levels > self.prev_levels and self.prev_key is not None:
                self._train_on_milestone(self.prev_key)
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _train_on_milestone(self, milestone_key):
        """BFS-back over observed edges from the milestone; label states by -distance; fit V."""
        radj: dict = {}
        for k, node in self.nodes.items():
            for a, (nk, _r) in node.edges.items():
                radj.setdefault(nk, []).append(k)
        dist = {milestone_key: 0}
        q = deque([milestone_key])
        while q:
            k = q.popleft()
            for prev in radj.get(k, []):
                if prev not in dist:
                    dist[prev] = dist[k] + 1
                    q.append(prev)
        X, y = [], []
        for k, d in dist.items():
            g = self.grid_by_key.get(k)
            if g is None:
                continue
            X.append(P.encode_onehot(g).astype(np.float32))
            y.append(-float(d))   # closer to milestone = higher value
        if len(X) < 8:
            return
        X = np.array(X); y = np.array(y, dtype=np.float32)
        y = (y - y.mean()) / (y.std() + 1e-6)
        if self.value_net is None:
            self.value_net = _ValueNet()
            self._opt = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        Xt = torch.tensor(X); yt = torch.tensor(y)
        self.value_net.train(True)
        loss_fn = nn.MSELoss()
        for _ in range(self.train_steps):
            idx = np.random.permutation(len(Xt))[:64]
            self._opt.zero_grad()
            loss = loss_fn(self.value_net(Xt[idx]), yt[idx])
            loss.backward(); self._opt.step()
        self._val_cache = {}

    def _value(self, key):
        if self.value_net is None:
            return None
        if key in self._val_cache:
            return self._val_cache[key]
        g = self.grid_by_key.get(key)
        if g is None:
            return None
        self.value_net.train(False)
        with torch.no_grad():
            v = float(self.value_net(torch.tensor(P.encode_onehot(g).astype(np.float32))[None])[0])
        self._val_cache[key] = v
        return v

    def _path_to_frontier(self, start, p):
        base = super()._path_to_frontier(start, p)
        if not self.enable_value or self.value_net is None or not base:
            return base
        if start not in self.nodes or self.nodes[start].has_untried_le(p):
            return base
        seen = {start}; q = deque([(start, [])])
        best, best_depth, best_v = base, len(base), -1e9
        while q:
            k, path = q.popleft()
            if len(path) > best_depth:
                break
            node = self.nodes.get(k)
            if not node:
                continue
            for a, (nk, _r) in node.edges.items():
                if nk in seen:
                    continue
                seen.add(nk); np_ = path + [a]
                nn_ = self.nodes.get(nk)
                if nn_ is not None and nn_.has_untried_le(p):
                    v = self._value(nk)
                    if v is not None and len(np_) == best_depth and v > best_v:
                        best, best_v = np_, v
                else:
                    q.append((nk, np_))
        return best
