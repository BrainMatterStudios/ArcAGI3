"""CausalProbeExplorer — proof-gated structural exploitation on the v13 backbone.

The causal-probe architecture's first behavioral increment. Built on the survey finding that the
wall games are signal-RICH navigation-goal games (avatar moves on ~every action) — the wall is
reaching the goal CONFIG, not lack of signal. The lever for those games is progress toward
STRUCTURAL change (collect / transform / spawn — the object color-multiset changes), as opposed
to the positional noise that drowns salience.

Design (coverage-preserving, per the two-lane principle):
  - Lane A: TransferExplorer's exact frontier order, UNTOUCHED (preserves tu93 deep reach + the
    0.33 floor; reordering is what cratered every prior lever).
  - Online proof layer: MechanicCertificateStore observes every witnessed transition. We
    additionally flag action-CLASSES that caused a STRUCTURAL delta (color-multiset change, not a
    pure avatar move) — a witnessed progress proxy, never a goal guess.
  - Lane B (the only added behavior): PROMOTE a structurally-certified action class to tier 0 when
    it is an untried candidate at the current node — exactly transfer's safe promote mechanism
    (promote, never remove; full coverage preserved), but keyed on witnessed structural progress
    instead of only the level-up reward signature. Fires only AFTER the class is witnessed causing
    structure to change in THIS game (proof certificate).

Firewall: enable_causal=False -> byte-identical to TransferExplorer (== v13). The structural
promotion is strictly additive (a tier-0 promotion of an already-present candidate), so when no
structural certificate has been earned yet, behavior is byte-identical to transfer.
"""

from __future__ import annotations

import numpy as np

from . import perception as P
from .mechanic_certificates import MechanicCertificateStore
from .transfer_explorer import TransferExplorer


class CausalProbeExplorer(TransferExplorer):
    def __init__(self, *args, enable_causal: bool = True, struct_min_cells: float = 1.0,
                 **kwargs) -> None:
        self.enable_causal = bool(enable_causal)
        self.struct_min_cells = float(struct_min_cells)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.certs = MechanicCertificateStore()
        self.struct_triggers: set = set()   # action classes witnessed to change object structure
        self._cp_prev_grid = None

    def _color_hist(self, grid):
        """Multiset of non-background colors (object structure, position-invariant)."""
        vals, counts = np.unique(grid, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts) if v != self.bg}

    def _struct_changed(self, g0, g1) -> bool:
        """Object color-multiset changed (collect/transform/spawn) vs pure positional move."""
        if g0.shape != g1.shape:
            return True
        return self._color_hist(g0) != self._color_hist(g1)

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_causal and not gstate_terminal and not gstate_notplayed:
            if self.bg is None:
                self.bg = P.detect_background(grid)
            if self.prev_action is not None and self._cp_prev_grid is not None:
                level_up = levels > self.prev_levels
                key = self.certs.observe(self.prev_action, self._cp_prev_grid, grid, level_up)
                if key is not None and self._struct_changed(self._cp_prev_grid, grid):
                    self.struct_triggers.add(key)
            self._cp_prev_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)   # transfer's reward promotion already applied
        if not self.enable_causal or not self.struct_triggers:
            return cands
        out = []
        for (act, tier) in cands:
            key = self.certs.action_class(act, grid)
            # promote structurally-certified classes (but never override transfer's reward promotion
            # being already tier 0, and never demote anything — additive coverage-preserving)
            if tier > 0 and key in self.struct_triggers:
                out.append((act, 0))
            else:
                out.append((act, tier))
        return out
