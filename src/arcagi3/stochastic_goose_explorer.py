"""StochasticGooseExplorer — a faithful STANDALONE reimplementation of the StochasticGoose family
(Tufa Labs, 1st place ARC-AGI-3 preview, 12.58%): a CNN action-EFFECT predictor that drives action
selection directly, with state-hash dedup and online (test-time) training carried across levels.

WHY THIS FILE EXISTS. The live Kaggle leaderboard (2026-06-28, no-internet eval) shows offline methods
reach 0.5-1.21 while this repo banks 0.33 — so 0.33 is the ceiling of THIS codebase, not the offline
paradigm. Every prior learned attempt here (GraphRanker / primary_model_explorer / EffectLearner /
DynamicsLearner / online_explorer) bolted a CNN onto the SalienceExplorer graph as a reranker/proposer/
pruner, which "breaks the graph." The documented winning family runs the CNN STANDALONE as the whole
policy. That standalone architecture has never been built here; this is it.

Difference from the graph explorers: NO navigation graph, NO shortest-path replay. A hash table records
which actions were tried at each (masked) state key purely to avoid repeating them. The effect-CNN picks
the next action by predicted P(state-change) — spending budget on state-changing actions instead of the
~51% no-op clicks blind coverage wastes. Reactive one-action-per-call interface, drops into run_reactive.

Firewall: brand-new file; nothing in the banked submission is touched. Torch is imported lazily inside
GooseEffectModel; the policy logic is torch-free and the model is dependency-injectable for testing.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import perception as P
from .learned_dynamics import _build_net

SIMPLE_IDS = [1, 2, 3, 4, 5]
MAX_TIER = 9


class GooseEffectModel:
    """The real torch-backed effect predictor: a small CNN (16-ch one-hot -> per-cell click-effect map
    + 5 simple-action effect logits) trained ONLINE on the game's own (state, action, changed?) stream.
    Carried across levels (no per-level reset) — faithful "retrain between levels". Torch is lazy."""

    def __init__(self, min_train: int = 64, train_every: int = 8, batch: int = 16,
                 lr: float = 1e-3, buffer: int = 4000, allow_cpu: bool = False) -> None:
        self.min_train = int(min_train)
        self.train_every = int(train_every)
        self.batch = int(batch)
        self.lr = float(lr)
        self.allow_cpu = bool(allow_cpu)
        self._torch = None
        self._net = None
        self._opt = None
        self._dev = None
        self._buf: deque = deque(maxlen=int(buffer))
        self._n = 0
        self._steps = 0

    def _t(self):
        if self._torch is None:
            import torch  # noqa: PLC0415
            self._torch = torch
        return self._torch

    def _device(self):
        if self._dev is None:
            torch = self._t()
            self._dev = "cpu" if (self.allow_cpu or not torch.cuda.is_available()) else "cuda"
        return self._dev

    def _net_ready(self):
        if self._net is None:
            torch = self._t()
            self._net = _build_net(torch).to(self._device())
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        return self._net

    def ready(self) -> bool:
        return self._n >= self.min_train

    def _to_t(self, one_hot):
        torch = self._t()
        return torch.from_numpy(np.ascontiguousarray(one_hot)).unsqueeze(0).to(self._device())

    def scores(self, one_hot, cands) -> dict:
        torch = self._t()
        net = self._net_ready()
        with torch.no_grad():
            click, simple = net(self._to_t(one_hot))
            click = torch.sigmoid(click)[0]          # (64,64)
            simple = torch.sigmoid(simple)[0]        # (5,)
        out = {}
        for a in cands:
            if a[0] == "S" and a[1] in SIMPLE_IDS:
                out[a] = float(simple[SIMPLE_IDS.index(a[1])])
            elif a[0] == "C":
                out[a] = float(click[int(a[2]), int(a[1])])   # [row=y, col=x]
            else:
                out[a] = 0.0
        return out

    def observe(self, one_hot, action, changed) -> None:
        if action[0] not in ("S", "C"):
            return
        self._buf.append((one_hot, action, 1.0 if changed else 0.0))
        self._n += 1
        self._steps += 1
        if self._n >= self.min_train and self.train_every > 0 and self._steps % self.train_every == 0:
            self._train_step()

    def _train_step(self):
        torch = self._t()
        net = self._net_ready()
        import random  # noqa: PLC0415
        idx = random.sample(range(len(self._buf)), min(self.batch, len(self._buf)))
        logits, labels = [], []
        for i in idx:
            oh, action, label = self._buf[i]
            click, simple = net(self._to_t(oh))
            if action[0] == "S" and action[1] in SIMPLE_IDS:
                logits.append(simple[0, SIMPLE_IDS.index(action[1])])
            else:
                logits.append(click[0, int(action[2]), int(action[1])])
            labels.append(label)
        logit = torch.stack(logits)
        target = torch.tensor(labels, dtype=torch.float32, device=self._device())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()


class StochasticGooseExplorer:
    def __init__(self, seed: int = 0, model=None, temperature: float = 0.5,
                 max_click_targets: int = 96, coarse_grid_step: int = 8, border_mask: int = 2,
                 min_train: int = 64, allow_cpu: bool = False, stuck_mode: str = "random",
                 max_stuck_resets: int = 200) -> None:
        self.rng = np.random.default_rng(seed)
        self.temperature = float(temperature)
        self.max_click_targets = int(max_click_targets)
        self.coarse_grid_step = int(coarse_grid_step)
        self.border_mask = max(0, int(border_mask))
        self._model = model
        self._min_train = int(min_train)
        self._allow_cpu = bool(allow_cpu)
        # stuck_mode: "random" = take a random available action on local-exhaustion (v1);
        # "reset" = Go-Explore-style bounce to root + re-explore a new branch via dedup.
        self.stuck_mode = stuck_mode
        self.max_stuck_resets = int(max_stuck_resets)
        self.reset_all()

    # duck-typing for the runner's states_seen (len(pol.gs.wm))
    @property
    def gs(self):
        return self

    @property
    def wm(self):
        return self.tried

    def __len__(self):
        return len(self.tried)

    @property
    def model(self):
        if self._model is None:
            self._model = GooseEffectModel(min_train=self._min_train, allow_cpu=self._allow_cpu)
        return self._model

    def reset_all(self):
        self.vt = P.VolatilityTracker()
        self.bg: int | None = None
        self.tried: dict[bytes, set] = {}
        self.prev_key: bytes | None = None
        self.prev_action = None
        self.prev_levels = 0
        self._last_oh = None
        self.expect_reset = False
        self.stuck_resets = 0

    # -- keying (reuses the proven SalienceExplorer masking) --------------------------------------
    def _key(self, grid):
        m = self.vt.mask()
        bm = self._border_mask()
        if bm is not None:
            m = m | bm
        if m.any():
            grid = grid.copy()
            grid[m] = self.bg if self.bg is not None else 0
        return P.object_state_key(grid, background=self.bg)

    def _border_mask(self):
        b = self.border_mask
        if b <= 0 or self.vt.changes is None or self.vt.steps < self.vt.min_steps:
            return None
        edge = np.zeros(self.vt.shape, dtype=bool)
        edge[:b] = edge[-b:] = True
        edge[:, :b] = edge[:, -b:] = True
        return edge & (self.vt.changes > 0)

    def _candidates(self, grid, available):
        cands = []
        for aid in SIMPLE_IDS:
            if aid in available:
                cands.append((("S", aid), 0))
        if 6 in available:
            for x, y, prio in P.salient_click_targets(grid, max_targets=self.max_click_targets,
                                                      coarse_grid_step=self.coarse_grid_step):
                cands.append((("C", int(x), int(y)), int(prio)))
        return cands

    def _encode(self, grid):
        return P.encode_onehot(grid, 16)   # (16,64,64) float32, torch-free

    # -- decision loop -----------------------------------------------------------------------------
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        self.vt.update(grid)
        if self.bg is None:
            self.bg = P.detect_background(grid)
        cur = self._key(grid)

        if gstate_terminal or gstate_notplayed:
            self.prev_action = None
            self.expect_reset = True
            return ("reset",)

        if self.expect_reset:
            self.expect_reset = False
            self.prev_action = None

        # record the previous transition's change-label into the online model (test-time training)
        if self.prev_action is not None and self.prev_key is not None and self._last_oh is not None:
            changed = cur != self.prev_key
            self.model.observe(self._last_oh, self.prev_action, changed)

        self.prev_levels = levels
        cands = self._candidates(grid, available)
        one_hot = self._encode(grid)
        action = self._choose(cur, cands, one_hot)
        self._last_oh = one_hot
        self.prev_key = cur
        self.prev_action = action
        return action

    def _choose(self, cur, cands, one_hot):
        tried = self.tried.setdefault(cur, set())
        untried = [(a, p) for (a, p) in cands if a not in tried]
        if not untried:
            # stuck: all local actions tried here.
            if self.stuck_mode == "reset" and self.stuck_resets < self.max_stuck_resets:
                # Go-Explore: bounce to root; dedup makes the re-exploration cover a new branch.
                self.stuck_resets += 1
                self.expect_reset = True
                return ("reset",)
            # default "random": take a random available action (accept a revisit). NEVER ("reset",)
            # off-terminal in this mode: env.reset() returns to L0 (verified), costly on deep levels.
            if not cands:
                return ("S", 1)
            return cands[int(self.rng.integers(0, len(cands)))][0]
        if self.model.ready():
            acts = [a for a, _ in untried]
            action = self._sample(acts, self.model.scores(one_hot, acts))
        else:
            action = self._cold_pick(untried)
        tried.add(action)
        return action

    def _cold_pick(self, untried):
        """Before the model warms up: salience-tier order (lowest priority first), rng tie-break."""
        mp = min(p for _, p in untried)
        choices = [a for a, p in untried if p == mp]
        return choices[int(self.rng.integers(0, len(choices)))]

    def _sample(self, acts, scores):
        s = np.array([scores.get(a, 0.0) for a in acts], dtype=float)
        if self.temperature <= 1e-9:
            return acts[int(np.argmax(s))]
        z = s / self.temperature
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        return acts[int(self.rng.choice(len(acts), p=w))]
