"""BatchValueExplorer — coverage-SAFE within-game exploitation (the one form not yet tried).

Every prior exploitation lever failed the same way: it changed WHICH state/frontier the agent
explores (value-CNN committed to a predicted candidate; value-ranker reranked frontiers; CausalProbe
promoted classes across tiers) -> corrupted the coverage order that deep games (tu93) depend on.

This one CANNOT corrupt coverage by construction: it only reorders the equal-tier untried batch at
the CURRENT node (salience's step-3 choice, normally uniform-random). Every action in that batch is
still tried over successive visits — the SET explored is identical; only the ORDER changes. So a
level-up trigger hidden in a large untried batch is surfaced EARLIER (within-level efficiency) with
zero risk of skipping coverage.

Value signal (online, witnessed-this-game only — no goal guess): among the batch, prefer action
classes that have caused a level-up (proof certificate) > a structural delta (big change) > any
change, over never-productive ones. Before any certificate is earned, the order is the original
uniform-random (byte-identical to v6).

Backbone: TransferExplorer (v13). Firewall: enable_batch=False -> byte-identical to v13.
"""

from __future__ import annotations

import numpy as np

from . import perception as P
from .mechanic_certificates import MechanicCertificateStore
from .transfer_explorer import TransferExplorer


class BatchValueExplorer(TransferExplorer):
    def __init__(self, *args, enable_batch: bool = True, **kwargs) -> None:
        self.enable_batch = bool(enable_batch)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.bv_certs = MechanicCertificateStore()
        self._bv_prev_grid = None
        self._bv_cur_grid = None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_batch and not gstate_terminal and not gstate_notplayed:
            if self.bg is None:
                self.bg = P.detect_background(grid)
            if self.prev_action is not None and self._bv_prev_grid is not None:
                level_up = levels > self.prev_levels
                self.bv_certs.observe(self.prev_action, self._bv_prev_grid, grid, level_up)
            self._bv_prev_grid = grid
            self._bv_cur_grid = grid          # grid at the current node, for value lookup
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _action_value(self, a) -> float:
        if self._bv_cur_grid is None:
            return 0.0
        key = self.bv_certs.action_class(a, self._bv_cur_grid)
        if key is None:
            return 0.0
        if self.bv_certs.is_level_trigger(key):
            return 3.0
        if self.bv_certs.is_structural(key):
            return 2.0
        c = self.bv_certs.certs.get(key)
        if c is not None and c.change_rate > 0.5:
            return 1.0
        return 0.0

    def _pick_from_batch(self, choices, node):
        if not self.enable_batch:
            return super()._pick_from_batch(choices, node)
        vals = [self._action_value(a) for a in choices]
        best = max(vals)
        if best <= 0.0:
            return super()._pick_from_batch(choices, node)   # no signal -> v6 random (byte-identical)
        top = [a for a, v in zip(choices, vals) if v == best]
        return top[int(self.rng.integers(0, len(top)))]      # random among the highest-value class
