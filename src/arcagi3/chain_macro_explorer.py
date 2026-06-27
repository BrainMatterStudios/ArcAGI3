"""ChainMacroExplorer — within-game SOLUTION-SEQUENCE replay (geodesic-replay / CausalChainMacro).

The RHAE headroom oracle (scripts/rhae_headroom.py) measured 35x MEDIAN (100-800x on L2+) recoverable
within-game efficiency: the explorer re-discovers the same mechanic from scratch every level. TransferExplorer
captures only a sliver of it -- it promotes the rewarding action's CLASS to tier 0 but the explorer still
EXPLORES to find where to apply it, in arbitrary order. The headroom is the multi-step SOLUTION SEQUENCE.

This subclass records, per level, the ordered sequence of EFFECTIVE clicks (a click that changed the state
key -> a real mechanic operation, not a no-op) keyed by the same transferable signature TransferExplorer
learns (object colour). On the next level it REPLAYS that chain as a directed EXPLOIT: for each chain step it
clicks an as-yet-unclicked object whose signature matches, in order. For CLICK/recolor games (where the
headroom is largest: lf52/vc33/cd82/ar25) the matching object is clickable from ANY state, so the chain
replays with NO navigation -- sidestepping the spatial-nav wall that killed every prior planner. If the chain
can't progress (no matching object, or it runs out without a level-up) it hands straight back to the base
explorer, so coverage is preserved and a misfire only costs the (short) chain length.

This is EXPLOIT-toward-the-known-reward-chain (driven by the previous levels' real level-up edges), not a
signal-free frontier reorder -- the W3-safe class. enable_macro=False -> byte-identical to TransferExplorer
(the banked submission), the firewall. See docs .../2026-06-21-history-augmented-state-design.md and the
memory arcagi3-rhae-headroom.
"""
from __future__ import annotations

from . import perception as P
from .transfer_explorer import TransferExplorer


class ChainMacroExplorer(TransferExplorer):
    def __init__(self, *args, enable_macro: bool = True, max_macro_misses: int = 0, **kwargs) -> None:
        self.enable_macro = bool(enable_macro)
        # how many extra non-clickable-progress steps the macro tolerates before giving up (0 = strict).
        self.max_macro_misses = int(max_macro_misses)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.macro: list = []            # the learned ordered chain (click signatures)
        self.macro_pos: int = 0
        self.in_macro: bool = False
        self._level_ops: list = []       # effective-click signatures THIS level, in order
        self._macro_clicked: set = set()  # cells already clicked during the current macro replay
        self._cur_grid = None

    # --- learn the chain: effective clicks (state-changing) in order -----------------------------
    def _record(self, key, action, next_key, reward, cands, terminal):
        super()._record(key, action, next_key, reward, cands, terminal)
        if (self.enable_macro and action[0] == "C" and next_key != key and reward == 0
                and self._prev_grid is not None):
            sig = self._cell_to_sig(self._prev_grid).get((action[2], action[1]))
            if sig is not None and (not self._level_ops or self._level_ops[-1] != sig):
                self._level_ops.append(sig)

    # --- on level-up: bank the chain and arm replay for the next level ---------------------------
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_macro:
            if gstate_terminal or gstate_notplayed:
                self.in_macro = False
                self._level_ops = []
            elif levels > self.prev_levels:
                if self._level_ops:
                    self.macro = list(self._level_ops)   # this level's solution sequence
                self._level_ops = []
                self.macro_pos = 0
                self._macro_clicked = set()
                self.in_macro = bool(self.macro)
            self._cur_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _choose(self, cur):
        if self.enable_macro and self.in_macro and self.macro_pos < len(self.macro):
            a = self._macro_pick()
            if a is not None:
                return a
            self.in_macro = False   # chain can't progress here -> hand back to the base explorer
        return super()._choose(cur)

    def _macro_pick(self):
        """Click an as-yet-unclicked object matching the current chain step's signature (directed
        exploit). Returns a ("C", x=col, y=row) token, or None if no match (caller falls back)."""
        if self._cur_grid is None or self.macro_pos >= len(self.macro):
            return None
        target = self.macro[self.macro_pos]
        best = None
        for o in P.connected_components(self._cur_grid, background=self.bg):
            if self._click_sig(o) != target:
                continue
            cell = (int(round(o.centroid[0])), int(round(o.centroid[1])))   # (row, col)
            if cell not in self._macro_clicked:
                best = cell
                break
        if best is None:
            return None
        self._macro_clicked.add(best)
        self.macro_pos += 1
        return ("C", int(best[1]), int(best[0]))   # token is (col, row)
