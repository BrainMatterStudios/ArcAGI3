"""CAIPruneExplorer — prune proven-no-op click candidates (safe efficiency cut).

The Phase-E re-traversal audit found ~51% of actions are redundant no-op click-probes and
concluded there was "no safe cut because telling no-op from productive needs a transition
model." The physics-lens exploration pointed out: the transition model is FREE — it's the
edges we already record. A click whose outcome leaves the state unchanged has zero causal
action influence (CAI=0). So: track, per object COLOR, how often clicking it is a no-op vs
causes a change; once a color is confidently no-op (>= k consecutive no-ops, no change ever),
STOP proposing clicks on that color. This is PRUNING (removing provably-dead candidates),
NOT frontier reranking — so it can't corrupt the load-bearing nearest-first coverage that
sank every reorder lever (walk-reduction/struct/value-ranker). It only saves the actions the
explorer would otherwise waste probing dead colors (sk48-style no-op clicks).

Firewall: enable_cai=False -> byte-identical to SalienceExplorer (v6).
"""

from __future__ import annotations

from collections import defaultdict

from . import perception as P
from .salience_explorer import SalienceExplorer


class CAIPruneExplorer(SalienceExplorer):
    def __init__(self, *args, enable_cai: bool = True, noop_k: int = 8, **kwargs) -> None:
        self.enable_cai = bool(enable_cai)
        self.noop_k = int(noop_k)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.color_noop: dict = defaultdict(int)   # color -> consecutive no-op clicks
        self.color_active: set = set()             # colors a click ever changed the state
        self.pruned_colors: set = set()
        self._cai_prev_grid = None

    def _clicked_color(self, grid, x, y):
        for o in P.connected_components(grid, background=self.bg):
            if (y, x) in o.cells:
                return int(o.color)
        return None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        # judge the previous click: did the state change? (no-op vs causal)
        if (self.enable_cai and self.prev_action is not None and self.prev_action[0] == "C"
                and self._cai_prev_grid is not None and not gstate_terminal and not gstate_notplayed):
            col = self._clicked_color(self._cai_prev_grid, self.prev_action[1], self.prev_action[2])
            if col is not None:
                changed = self._key(grid) != self._key(self._cai_prev_grid)
                if changed:
                    self.color_active.add(col)
                    self.color_noop[col] = 0
                else:
                    self.color_noop[col] += 1
                    if (self.color_noop[col] >= self.noop_k and col not in self.color_active):
                        self.pruned_colors.add(col)
        action = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._cai_prev_grid = grid
        return action

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)
        if not self.enable_cai or not self.pruned_colors:
            return cands
        # drop click candidates whose cell color is a proven-no-op color
        cell_color = {}
        for o in P.connected_components(grid, background=self.bg):
            for (rr, cc) in o.cells:
                cell_color[(rr, cc)] = int(o.color)
        out = []
        for (act, tier) in cands:
            if act[0] == "C" and cell_color.get((act[2], act[1])) in self.pruned_colors:
                continue  # provably dead -> prune
            out.append((act, tier))
        return out
