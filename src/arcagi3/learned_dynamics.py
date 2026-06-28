"""DynamicsLearner — a REAL per-game online-trained (test-time-training) neural learner for the
LearnedExplorer harness. A small CNN predicts each action's EFFECT (will it change the masked state key?)
from the 16-channel one-hot board; trained online on the game's own interaction stream (StochasticGoose-
class self-supervised signal), reset per level. In act() it proposes the highest-confidence EFFECTIVE
untried action and ABSTAINS otherwise -> the base explorer runs, so coverage is preserved (the abstain-
default harness is the hardening the prior reranker/primary online-CNNs lacked, which catastrophically
regressed tu93 9->0). Torch is imported lazily; the harness only activates this on a verified-working GPU
(T4) and otherwise falls back to banked TransferExplorer, so the 0.33 floor is never at risk.

The net has no dropout/batchnorm, so it is mode-agnostic (no train/eval toggling needed).
"""
from __future__ import annotations

from collections import deque

from . import perception as P
from .learned_explorer import Learner

SIMPLE_IDS = [1, 2, 3, 4, 5]


def _build_net(torch):
    nn = torch.nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(16, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 8, 3, padding=1)
            self.click = nn.Conv2d(8, 1, 1)          # (B,1,64,64) per-cell click-effect logit
            self.simple = nn.Linear(8, 5)            # 5 simple-action effect logits (global-pooled)

        def forward(self, x):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            click = self.click(h).squeeze(1)          # (B,64,64)
            simple = self.simple(h.mean(dim=(2, 3)))  # (B,5)
            return click, simple

    return _Net()


class DynamicsLearner(Learner):
    def __init__(self, train_every: int = 8, batch: int = 16, min_train: int = 64,
                 confidence: float = 0.6, noop_thresh: float = 0.05, lr: float = 1e-3,
                 buffer: int = 4000) -> None:
        self.cfg = dict(train_every=train_every, batch=batch, min_train=min_train,
                        confidence=confidence, noop_thresh=noop_thresh, lr=lr, buffer=buffer)
        self._torch = None
        self.reset_game()

    # -- lifecycle --------------------------------------------------------------------------------
    def _ensure_torch(self):
        if self._torch is None:
            import torch  # noqa: PLC0415
            self._torch = torch
        return self._torch

    def reset_game(self):
        self._buf: deque = deque(maxlen=self.cfg["buffer"])
        self._net = None
        self._opt = None
        self._steps = 0
        self._n_seen = 0
        self._last_oh = None        # one-hot tensor of the grid act() last saw (for observe pairing)

    def reset_level(self):
        self.reset_game()           # mirror the recipe: fresh model+buffer per level

    def _net_ready(self):
        if self._net is None:
            torch = self._ensure_torch()
            self._net = _build_net(torch)
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg["lr"])
        return self._net

    def _encode(self, grid):
        torch = self._ensure_torch()
        oh = P.encode_onehot(grid, 16)                 # (16,64,64) float32
        return torch.from_numpy(oh).unsqueeze(0)       # (1,16,64,64)

    def see(self, grid):
        self._last_oh = self._encode(grid)             # cache once per step for observe/act/noop_set

    def _click_probs(self):
        torch = self._torch
        net = self._net_ready()
        with torch.no_grad():
            click, _ = net(self._last_oh)
            return torch.sigmoid(click)[0]             # (64,64)

    # -- predict (act) ----------------------------------------------------------------------------
    def act(self, grid, node, key):
        if self._last_oh is None:
            self._last_oh = self._encode(grid)
        if node is None or self._n_seen < self.cfg["min_train"]:
            return None
        torch = self._torch
        net = self._net_ready()
        with torch.no_grad():
            click, simple = net(self._last_oh)
            click = torch.sigmoid(click)[0]            # (64,64)
            simple = torch.sigmoid(simple)[0]          # (5,)
        best, best_p = None, 0.0
        for a in node.cands:
            if a in node.edges:                        # already tried here
                continue
            if a[0] == "S" and a[1] in SIMPLE_IDS:
                p = float(simple[SIMPLE_IDS.index(a[1])])
            elif a[0] == "C":
                p = float(click[int(a[2]), int(a[1])])  # (row, col)
            else:
                continue
            if p > best_p:
                best, best_p = a, p
        return best if best_p >= self.cfg["confidence"] else None   # abstain unless confident

    # -- prune (safe class): confident no-op clicks to demote ------------------------------------
    def noop_set(self, grid, click_cands):
        if self._last_oh is None or self._n_seen < self.cfg["min_train"] or not click_cands:
            return set()
        probs = self._click_probs()
        thr = self.cfg["noop_thresh"]
        return {a for a in click_cands
                if a[0] == "C" and float(probs[int(a[2]), int(a[1])]) < thr}

    # -- learn (observe) --------------------------------------------------------------------------
    def observe(self, key, action, next_key, reward):
        if self._last_oh is None or action[0] not in ("S", "C"):
            return
        label = 1.0 if next_key != key else 0.0
        self._buf.append((self._last_oh, action, label))
        self._n_seen += 1
        self._steps += 1
        if self._n_seen >= self.cfg["min_train"] and self._steps % self.cfg["train_every"] == 0:
            self._train_step()

    def _train_step(self):
        torch = self._ensure_torch()
        net = self._net_ready()
        import random  # noqa: PLC0415  (sampling only; not in the decision path)
        idx = random.sample(range(len(self._buf)), min(self.cfg["batch"], len(self._buf)))
        logits, labels = [], []
        for i in idx:
            oh, action, label = self._buf[i]
            click, simple = net(oh)
            if action[0] == "S" and action[1] in SIMPLE_IDS:
                logits.append(simple[0, SIMPLE_IDS.index(action[1])])
            else:
                logits.append(click[0, int(action[2]), int(action[1])])
            labels.append(label)
        logit = torch.stack(logits)
        target = torch.tensor(labels, dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
