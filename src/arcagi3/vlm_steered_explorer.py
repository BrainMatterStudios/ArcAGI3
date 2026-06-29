"""VLMSteeredExplorer — TransferExplorer + a per-level GOAL-REGION prior (the VLM's job) that ADDITIVELY
orders the click exploration toward the goal. Coverage-safe by construction: it ONLY reorders the equal-tier
untried CLICK batch (the SalienceExplorer._pick_from_batch contract — every action is still tried over
successive visits; only the ORDER shifts toward the goal). Firewall: no target / no provider -> byte-identical
to TransferExplorer (the banked 0.33 floor).

Motivation (memory arcagi3-goal-legibility-finding): P0 proved a local VLM can READ a goal region from a
frame; P1 proved that steering clicks to the right region reaches lp85 L0 in ~15 vs ~368 blind actions (24x).
The VLM is called ONCE PER LEVEL (cheap: ~60-100 calls/eval), returns a target (row,col), and the fast
explorer captures the efficiency. This is the read-only "steering" slice (NOT the dead VLM-planner); its
value is EFFICIENCY on goal-legible interact-with-region games, which the squared RHAE metric rewards.
"""
from __future__ import annotations

from .transfer_explorer import TransferExplorer


class VLMSteeredExplorer(TransferExplorer):
    def __init__(self, *args, target_provider=None, **kwargs) -> None:
        # target_provider: callable(grid) -> (row, col) goal point, or None to abstain.
        self.target_provider = target_provider
        self._cur_grid = None
        self._target = None
        self._target_level = -1
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self._cur_grid = None
        self._target = None
        self._target_level = -1

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        self._cur_grid = grid
        # refresh the goal-region target ONCE per level (cheap periodic VLM call)
        if (self.target_provider is not None and not (gstate_terminal or gstate_notplayed)
                and levels != self._target_level):
            try:
                self._target = self.target_provider(grid)
            except Exception:  # noqa: BLE001  VLM failure -> abstain -> pure TransferExplorer
                self._target = None
            self._target_level = levels
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _pick_from_batch(self, choices, node):
        """Among the equal-tier untried batch, try the CLICK nearest the goal target first (additive,
        coverage-safe). Abstain (no target / no clicks in batch) -> base behaviour."""
        if self._target is None:
            return super()._pick_from_batch(choices, node)
        tr, tc = self._target
        clicks = [a for a in choices if a[0] == "C"]
        if not clicks:
            return super()._pick_from_batch(choices, node)
        # a = ("C", x=col, y=row); distance to target (row,col)
        return min(clicks, key=lambda a: (a[2] - tr) ** 2 + (a[1] - tc) ** 2)
