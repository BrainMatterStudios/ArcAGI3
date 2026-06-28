"""ObjectCuriosityExplorer — graph-only, reward-free object-interaction curiosity.

Re-orders the explorer's equal-tier untried-action batch toward descriptors (clicked object-type
or simple-action id) with the rarest object-interaction history. Reward-free, so it is non-trivial
on the wall games (ls20/re86) where reward-based learning (RVH No-T) is mute. Graph-only: it only
re-orders real untried actions via the coverage-safe _pick_from_batch hook; it never invents an
edge. Firewall: enable_object_curiosity=False -> byte-identical to TransferExplorer.
"""

from __future__ import annotations

from collections import Counter

from .object_interaction import NONE_SIGNATURE, interaction_signature
from .transfer_explorer import TransferExplorer

UNSEEN_BONUS = 1e6


class ObjectCuriosityExplorer(TransferExplorer):
    def __init__(self, *args, enable_object_curiosity: bool = True, **kwargs) -> None:
        self.enable_object_curiosity = bool(enable_object_curiosity)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.descriptor_apps: Counter = Counter()
        self.descriptor_sigs: dict = {}
        self.sig_counts: Counter = Counter()
        self._cur_grid = None

    # --- signature accounting -------------------------------------------------
    def _descriptor_for(self, action, grid):
        if action[0] == "S":
            return ("S", int(action[1]))
        color = self._cell_to_sig(grid).get((action[2], action[1]))
        return ("C", color[0] if color is not None else -1)

    def _count_interaction(self, prev_grid, action, cur_grid):
        desc = self._descriptor_for(action, prev_grid)
        self.descriptor_apps[desc] += 1
        sig = interaction_signature(prev_grid, action, cur_grid, self.bg)
        if sig != NONE_SIGNATURE:
            self.sig_counts[sig] += 1
            self.descriptor_sigs.setdefault(desc, set()).add(sig)

    def _curiosity_score(self, desc):
        if self.descriptor_apps.get(desc, 0) == 0:
            return UNSEEN_BONUS
        sigs = self.descriptor_sigs.get(desc)
        if not sigs:
            return 0.0
        return max(1.0 / (1.0 + self.sig_counts.get(s, 0)) for s in sigs)

    # --- hooks ----------------------------------------------------------------
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if (self.enable_object_curiosity and self.prev_action is not None
                and not gstate_terminal and not gstate_notplayed
                and self._prev_grid is not None):
            self._count_interaction(self._prev_grid, self.prev_action, grid)
        self._cur_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _pick_from_batch(self, choices, node):
        if (not self.enable_object_curiosity or self._cur_grid is None
                or not self.descriptor_apps):
            return super()._pick_from_batch(choices, node)
        scored = [(self._curiosity_score(self._descriptor_for(a, self._cur_grid)), a)
                  for a in choices]
        best = max(s for s, _a in scored)
        top = [a for s, a in scored if s == best]
        return top[int(self.rng.integers(0, len(top)))]
