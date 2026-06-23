"""HybridTransferExplorer — TransferExplorer + best-effort model-based action promotion.

The salience explorer stays the floor that clears levels; the ModelGuide's suggested goal-directed
action is promoted to tier 0 in the candidate ordering (same mechanism TransferExplorer uses for the
reward signature). Firewall: enable_model_guidance=False -> byte-identical to TransferExplorer.
"""
from __future__ import annotations

from .model_guide import ModelGuide
from .transfer_explorer import TransferExplorer


class HybridTransferExplorer(TransferExplorer):
    def __init__(self, *args, enable_model_guidance: bool = True, **kwargs) -> None:
        self.enable_model_guidance = bool(enable_model_guidance)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self._guide = ModelGuide()
        self._guide_prev_grid = None
        self._guide_prev_token = None
        self._guide_suggestion = None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if (self.enable_model_guidance and self._guide_prev_grid is not None
                and self._guide_prev_token is not None and self._guide_prev_token[0] == "S"):
            try:
                self._guide.observe(self._guide_prev_grid, self._guide_prev_token, grid, levels)
            except Exception:  # noqa: BLE001  best-effort; never crash the floor
                pass
        self._guide_suggestion = None
        if self.enable_model_guidance and not gstate_terminal and not gstate_notplayed:
            try:
                self._guide_suggestion = self._guide.suggest(grid, available)
            except Exception:  # noqa: BLE001
                self._guide_suggestion = None
        token = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._guide_prev_grid, self._guide_prev_token = grid, token
        return token

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        if not self.enable_model_guidance or self._guide_suggestion is None:
            return cands
        sug = ("S", self._guide_suggestion)
        return [(act, 0 if act == sug else tier) for (act, tier) in cands]
