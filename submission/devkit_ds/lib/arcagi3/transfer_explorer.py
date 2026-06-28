"""TransferExplorer — within-game cross-level reward-signature transfer.

Subclass of SalienceExplorer. When a level is solved, the rewarding action is a REAL
labeled positive of the game's mechanic. Levels of a game share escalating mechanics, so
the rewarding-action SIGNATURE (a simple-action id, or a clicked object's color/shape)
usually recurs on later levels. We learn it and re-prioritise the next levels' candidates
toward the matching class (promote matches to tier 0, demote the rest), so each new level
tries the historically-rewarding action FIRST -> far fewer actions before the deep,
heavily-level-weighted level-ups.

This reorders WITH a real reward signal (the previous levels' level-up edges), unlike the
killed signal-free frontier reorderings; full coverage is preserved (only tier ORDER
changes, every candidate is still reachable).

Firewall: enable_transfer=False -> byte-identical action trace to SalienceExplorer (v6).
"""

from __future__ import annotations

from . import perception as P
from .salience_explorer import MAX_TIER, SalienceExplorer


def _size_bucket(n: int) -> int:
    if n <= 4:
        return 0
    if n <= 16:
        return 1
    if n <= 64:
        return 2
    return 3


class TransferExplorer(SalienceExplorer):
    def __init__(self, *args, enable_transfer: bool = True, transfer_demote: int = 3,
                 sig_mode: str = "color", **kwargs) -> None:
        # set before super().__init__ (which calls reset_all)
        self.enable_transfer = bool(enable_transfer)
        self.transfer_demote = int(transfer_demote)
        self.sig_mode = sig_mode  # "color" (robust, transferable) or "colorshape"
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.reward_simple: set = set()      # learned ("S", aid) signatures
        self.reward_click_sig: set = set()   # learned click signatures
        self._prev_grid = None

    # --- signature extraction ------------------------------------------------
    def _click_sig(self, o):
        if o is None:
            return None
        if self.sig_mode == "color":
            return (int(o.color),)
        return (int(o.color), _size_bucket(o.size))

    def _cell_to_sig(self, grid):
        """Map (row, col) -> click signature for every object cell on this grid."""
        m = {}
        for o in P.connected_components(grid, background=self.bg):
            s = self._click_sig(o)
            for (rr, cc) in o.cells:
                m[(rr, cc)] = s
        return m

    def _learn(self, grid, action):
        if action[0] == "S":
            self.reward_simple.add(action)
        elif action[0] == "C":
            # action = ("C", x=col, y=row)
            sig = self._cell_to_sig(grid).get((action[2], action[1]))
            if sig is not None:
                self.reward_click_sig.add(sig)

    # --- hooks ---------------------------------------------------------------
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if (self.enable_transfer and self.prev_action is not None
                and not gstate_terminal and not gstate_notplayed
                and levels > self.prev_levels and self._prev_grid is not None):
            self._learn(self._prev_grid, self.prev_action)
        action = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._prev_grid = grid
        return action

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        if not self.enable_transfer or not (self.reward_simple or self.reward_click_sig):
            return cands
        cell2sig = self._cell_to_sig(grid) if self.reward_click_sig else {}
        out = []
        for (act, tier) in cands:
            if act[0] == "S":
                out.append((act, 0 if act in self.reward_simple else tier))
            else:
                sig = cell2sig.get((act[2], act[1]))
                if sig is not None and sig in self.reward_click_sig:
                    out.append((act, 0))                       # promote rewarding class
                elif self.reward_click_sig:
                    out.append((act, min(MAX_TIER, tier + self.transfer_demote)))  # demote rest
                else:
                    out.append((act, tier))
        return out
