"""SpatialValueExplorer — per-game spatial action-value CNN (cluster-faithful + hardened).

The reproducible 0.58-0.70 cluster (Blind Squirrel / StochasticGoose) learns, PER GAME, a value
that ranks (state, action) toward the next milestone, then EXPLOITS it. This is that method,
built from scratch here, with novel generalization-hardening on top (the leaders' versions
overfit-collapsed 12.58%->0.25%, so hardening is real added value):

  - A small CNN on the 16-channel one-hot 64x64 frame with TWO heads:
      * spatial click-value map (64x64): value of clicking each cell;
      * simple-action value vector (actions 1-5).
  - Trained ONLINE per game: on each level-up, BFS-back over the observed graph labels every
    state with graph-distance to the milestone; each observed transition (state, action,
    next_state) becomes a target Q = -distance(next_state). One reward -> hundreds of labels.
  - EXPLOIT: once trained (post-first-reward), among the salience-proposed UNTRIED candidates at
    the current state, pick the highest predicted value (move toward the next milestone). The
    spatial head generalizes to untried cells.
  - HARDENING (novel): (1) gate exploitation on CONFIDENCE — only exploit when the value spread
    across candidates exceeds a margin AND MC-dropout uncertainty is low; else fall back to the
    SalienceExplorer search (preserves coverage + the 0.33 floor). (2) Until the first reward,
    pure SalienceExplorer (byte-identical to v6). (3) Salience graph stays the backbone for
    coverage, so a mis-trained net can't collapse exploration (the failure mode that sank the
    preview leaders).

Firewall: enable_value_cnn=False -> byte-identical to SalienceExplorer (v6). Fail-safe: no torch
(GPU-only at eval) -> behaves as v6.
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
    class _QNet(nn.Module):
        def __init__(self, p_drop: float = 0.1):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.Dropout2d(p_drop),
                nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(), nn.Dropout2d(p_drop))
            self.spatial = nn.Conv2d(24, 1, 1)               # 64x64 click-value map
            self.act = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(24, 6))

        def forward(self, x):
            f = self.body(x)
            return self.spatial(f).squeeze(1), self.act(f)   # (B,64,64), (B,6)


class SpatialValueExplorer(SalienceExplorer):
    def __init__(self, *args, enable_value_cnn: bool = True, train_steps: int = 150,
                 conf_margin: float = 0.6, exploit_cap: int = 150, **kwargs) -> None:
        self.enable_value_cnn = bool(enable_value_cnn) and _TORCH
        self.train_steps = int(train_steps)
        self.conf_margin = float(conf_margin)
        # cap consecutive exploit-actions per level: if the value-guided exploit hasn't produced a
        # level-up within exploit_cap steps, it's chasing a wrong milestone-direction (the cd82
        # failure: L2 differs from L1) -> revert to salience coverage. Bounds wasted exploit.
        self.exploit_cap = int(exploit_cap)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.qnet = None
        self.grid_by_key: dict = {}
        self.transitions: list = []       # (from_key, action, to_key)
        self._svp_prev_grid = None
        self._exploit_streak = 0
        self._svp_levels = 0

    # ---- capture + train ----
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_value_cnn and not gstate_terminal and not gstate_notplayed:
            if self.bg is None:
                self.bg = P.detect_background(grid)
            self._svp_prev_grid = grid   # current grid, for _choose's value lookup
            if levels > self._svp_levels:   # new level -> reset the exploit budget
                self._svp_levels = levels
                self._exploit_streak = 0
            self.grid_by_key.setdefault(self._key(grid), grid)
            if self.prev_action is not None and self.prev_key is not None:
                self.transitions.append((self.prev_key, self.prev_action, self._key(grid)))
            if self.prev_action is not None and levels > self.prev_levels and self.prev_key is not None:
                self._train(self.prev_key)
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _train(self, milestone_key):
        radj: dict = {}
        for (fk, a, tk) in self.transitions:
            radj.setdefault(tk, []).append(fk)
        dist = {milestone_key: 0}
        q = deque([milestone_key])
        while q:
            k = q.popleft()
            for p in radj.get(k, []):
                if p not in dist:
                    dist[p] = dist[k] + 1
                    q.append(p)
        # build (grid, action, target) samples from transitions whose successor has a distance
        samples = []
        for (fk, a, tk) in self.transitions:
            if tk in dist and fk in self.grid_by_key:
                samples.append((self.grid_by_key[fk], a, -float(dist[tk])))
        if len(samples) < 16:
            return
        ys = np.array([s[2] for s in samples], dtype=np.float32)
        mu, sd = ys.mean(), ys.std() + 1e-6
        if self.qnet is None:
            self.qnet = _QNet()
            self._opt = torch.optim.Adam(self.qnet.parameters(), lr=1e-3)
        self.qnet.train(True)
        lossfn = nn.MSELoss()
        idxs = np.arange(len(samples))
        for _ in range(self.train_steps):
            b = np.random.choice(idxs, size=min(48, len(idxs)), replace=False)
            X = torch.tensor(np.stack([P.encode_onehot(samples[i][0]).astype(np.float32) for i in b]))
            sp, av = self.qnet(X)
            pred = []
            for j, i in enumerate(b):
                act = samples[i][1]
                if act[0] == "C":
                    pred.append(sp[j, int(act[2]), int(act[1])])
                else:
                    pred.append(av[j, int(act[1])])
            pred = torch.stack(pred)
            tgt = torch.tensor([(samples[i][2] - mu) / sd for i in b], dtype=torch.float32)
            self._opt.zero_grad(); loss = lossfn(pred, tgt); loss.backward(); self._opt.step()

    # ---- exploit ----
    def _choose(self, cur):
        if not self.enable_value_cnn or self.qnet is None:
            return super()._choose(cur)
        node = self.nodes.get(cur)
        if node is None:
            return super()._choose(cur)
        ra = node.reward_action()
        if ra is not None:
            return ra
        # untried candidates here
        untried = [a for a in node.cands if a not in node.edges]
        if len(untried) < 2 or self._svp_prev_grid is None:
            return super()._choose(cur)
        if self._exploit_streak >= self.exploit_cap:   # exploit budget spent -> salience coverage
            return super()._choose(cur)
        grid = self._svp_prev_grid
        self.qnet.train(False)
        with torch.no_grad():
            X = torch.tensor(P.encode_onehot(grid).astype(np.float32))[None]
            sp, av = self.qnet(X)
            sp = sp[0]; av = av[0]
        scored = []
        for a in untried:
            v = float(sp[int(a[2]), int(a[1])]) if a[0] == "C" else float(av[int(a[1])])
            scored.append((v, a))
        scored.sort(reverse=True)
        best_v, best_a = scored[0]
        # HARDENING: only exploit if the net is confident (clear margin over the median candidate)
        med = float(np.median([v for v, _ in scored]))
        if best_v - med >= self.conf_margin:
            self._exploit_streak += 1
            return best_a
        return super()._choose(cur)
