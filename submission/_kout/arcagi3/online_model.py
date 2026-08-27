"""Phase A online action-effect model (GraphRanker) — a small CNN trained ONLINE, per game,
to predict which actions/clicks cause a frame change, used ONLY to re-rank the explorer's
already-sanctioned equal-tier tie-break candidates (see online_explorer.py).

Design (from rebuild/PHASE_A_PLAN.md, judge-gated):
- Input: 16-channel one-hot 64x64 (perception.encode_onehot).
- Backbone: 4 conv layers + GroupNorm (batch-size-independent; NOT BatchNorm).
- Heads: action-effect head -> P(frame-change) for ACTION1-5; a 1x1-conv click head -> a
  64x64 per-pixel P(frame-change) map for ACTION6 (spatially faithful, NOT flattened).
- Online training: int8-grid replay buffer, hash-dedup of (state,action), BCE every N steps;
  label = the explorer's MASKED object_state_key changed (not raw pixels -> ignores HUD noise).
  Buffer + model reset between levels.

TORCH-OPTIONAL + HARDENED FAIL-SAFE (eval image may give a P100 cap-6.0 that torch can't use,
or no torch at all): the model is usable ONLY if torch imports AND a tiny op on the chosen
device actually SUCCEEDS. Otherwise `usable` is False and every method is a no-op, so the
caller degrades to the pure SalienceExplorer (the 0.33 floor). Nothing here ever raises to
the caller.
"""
from __future__ import annotations

import numpy as np

from . import perception as P

NUM_COLORS = 16
SIMPLE_IDS = [1, 2, 3, 4, 5]


def _select_device(allow_cpu: bool = False):
    """Return (torch, device) if a USABLE compute device exists, else (torch_or_None, None).

    Hardened GPU gate: cuda.is_available() is NOT trusted alone — we run a tiny matmul on the
    GPU and require it to SUCCEED (a P100/cap-6.0 with an incompatible torch build reports
    available but errors on ops). GPU-ONLY by default: if no usable CUDA device, return None so
    the model disables and the agent runs the pure SalienceExplorer at full speed — because at
    eval the budget is WALL-CLOCK (12h) and CPU training means fewer actions = a net regression.
    allow_cpu=True (local dev only) permits a CPU device to validate the model logic.
    """
    try:
        import torch
    except Exception:
        return None, None
    try:
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            _x = torch.zeros((8, 8), device=dev)
            float((_x @ _x).sum().item())  # forces a real kernel launch
            return torch, dev
    except Exception:
        pass
    if allow_cpu:
        # LOCAL DEV ONLY (allow_cpu is never set at eval). Prefer the Apple Metal GPU if present
        # so the GPU model path can be exercised on this hardware; fall back to CPU. This does NOT
        # change eval behavior — at eval allow_cpu=False, so the path is still cuda-or-disabled.
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                dev = torch.device("mps")
                _x = torch.zeros((8, 8), device=dev)
                float((_x @ _x).sum().item())  # forces a real Metal kernel launch
                return torch, dev
        except Exception:
            pass
        try:
            dev = torch.device("cpu")
            _x = torch.zeros((4, 4), device=dev)
            float((_x @ _x).sum().item())
            return torch, dev
        except Exception:
            return torch, None
    return torch, None  # GPU-only: no usable CUDA -> disabled (pure SalienceExplorer)


def _build_net(torch, device):
    nn = torch.nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            def block(ci, co):
                return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1),
                                     nn.GroupNorm(8, co), nn.ReLU())
            self.backbone = nn.Sequential(block(NUM_COLORS, 32), block(32, 48),
                                          block(48, 64), block(64, 64))
            self.action_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                             nn.Linear(64, len(SIMPLE_IDS)))
            self.click_head = nn.Conv2d(64, 1, 1)  # 1x1 conv -> per-pixel logit map

        def forward(self, x):
            f = self.backbone(x)
            return self.action_head(f), self.click_head(f).squeeze(1)  # (B,5), (B,H,W)

    return _Net().to(device)


class OnlineActionEffectModel:
    """Online frame-change predictor. `usable` gates everything; no method ever raises."""

    def __init__(self, seed: int = 0, train_every: int = 8, max_buffer: int = 30000,
                 batch_size: int = 64, conf_threshold: float = 0.0,
                 allow_cpu: bool = False) -> None:
        self.train_every = train_every
        self.max_buffer = max_buffer
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.usable = False
        self.device_kind = "none"
        self._torch = None
        self._dev = None
        self._net = None
        self._opt = None
        try:
            torch, dev = _select_device(allow_cpu=allow_cpu)
            if torch is not None and dev is not None:
                torch.manual_seed(seed)
                self._torch = torch
                self._dev = dev
                self._net = _build_net(torch, dev)
                self._opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
                self.usable = True
                self.device_kind = dev.type
        except Exception:
            self.usable = False
        self._buf: dict[bytes, tuple] = {}  # (state_bytes,action_id,x,y) -> (grid_int8, label)
        self._steps = 0

    # --- lifecycle ---
    def reset_level(self):
        """Reset buffer + model between levels (StochasticGoose recipe)."""
        if not self.usable:
            return
        try:
            self._net = _build_net(self._torch, self._dev)
            self._opt = self._torch.optim.Adam(self._net.parameters(), lr=1e-3)
            self._buf.clear()
            self._steps = 0
        except Exception:
            self.usable = False

    # --- training data ---
    def observe(self, grid_before, action, changed: bool):
        """Record a (state, action) -> frame-changed? sample (hash-deduped, latest-wins)."""
        if not self.usable:
            return
        try:
            key = _sample_key(grid_before, action)
            self._buf[key] = (np.asarray(grid_before, dtype=np.int8), 1.0 if changed else 0.0)
            if len(self._buf) > self.max_buffer:
                # drop an arbitrary oldest-ish entry (dict preserves insertion order)
                self._buf.pop(next(iter(self._buf)))
            self._steps += 1
            if self._steps % self.train_every == 0:
                self._train_step()
        except Exception:
            self.usable = False

    def _train_step(self):
        if not self.usable or len(self._buf) < 8:
            return
        torch = self._torch
        try:
            items = list(self._buf.items())
            idx = np.random.default_rng(self._steps).integers(0, len(items),
                                                              size=min(self.batch_size, len(items)))
            grids, a_idx, xs, ys, labels, is_click = [], [], [], [], [], []
            for j in idx:
                (sb, aid, x, y), (g, lab) = items[int(j)]
                grids.append(P.encode_onehot(g))
                labels.append(lab)
                if aid == 6:
                    is_click.append(True); xs.append(x); ys.append(y); a_idx.append(0)
                else:
                    is_click.append(False); xs.append(0); ys.append(0); a_idx.append(SIMPLE_IDS.index(aid))
            X = torch.as_tensor(np.stack(grids), dtype=torch.float32, device=self._dev)
            lab = torch.as_tensor(labels, dtype=torch.float32, device=self._dev)
            act_logits, click_map = self._net(X)
            preds = []
            for i in range(len(idx)):
                if is_click[i]:
                    preds.append(click_map[i, ys[i], xs[i]])
                else:
                    preds.append(act_logits[i, a_idx[i]])
            pred = torch.stack(preds)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, lab)
            self._opt.zero_grad(); loss.backward(); self._opt.step()
            return float(loss.item())
        except Exception:
            self.usable = False
            return None

    # --- inference (the re-ranker) ---
    def best(self, grid, choices):
        """Return the candidate in `choices` with the highest predicted frame-change prob,
        or None if not usable / not confident / error (caller then keeps its own pick).
        `choices` are action tuples ("S",aid) | ("C",x,y)."""
        if not self.usable or self._steps < self.train_every or len(choices) < 2:
            return None
        torch = self._torch
        try:
            with torch.no_grad():
                X = torch.as_tensor(P.encode_onehot(grid)[None], dtype=torch.float32,
                                    device=self._dev)
                act_logits, click_map = self._net(X)
                act_p = torch.sigmoid(act_logits[0])
                click_p = torch.sigmoid(click_map[0])
                best_c, best_v = None, -1.0
                for c in choices:
                    if c[0] == "S":
                        v = float(act_p[SIMPLE_IDS.index(c[1])]) if c[1] in SIMPLE_IDS else 0.0
                    else:
                        v = float(click_p[int(c[2]), int(c[1])])  # map[y, x]
                    if v > best_v:
                        best_v, best_c = v, c
            if best_v < self.conf_threshold:
                return None
            return best_c
        except Exception:
            self.usable = False
            return None


def _sample_key(grid, action):
    sb = np.asarray(grid, dtype=np.int8).tobytes()
    if action[0] == "S":
        return (sb, action[1], 0, 0)
    return (sb, 6, int(action[1]), int(action[2]))
