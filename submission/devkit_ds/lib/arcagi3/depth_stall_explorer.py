"""DepthStallExplorer — Phase 1 of the multi-session build: reach the first reward on navigation
walls faster (user-approved).

Diagnosis (2026-06-22 survey + records): the navigation wall games are signal-rich and their L1 is
REACHABLE but too slow (sk48 L1 ~25k actions) — they are an exponential exploration tree, and
salience's nearest-frontier search keeps re-descending shallow branches (drowns in shallow
positional novelty) instead of driving deep toward the goal config.

Lever (untested): on STALL (no level-up for stall_trigger actions), flip the frontier selection
from NEAREST (breadth) to FARTHEST (depth) — navigate to the deepest known untried frontier,
driving exploration into the deep part of the tree where a navigation goal config lives. Local
untried actions are still tried first (handled upstream in _choose), so coverage is preserved;
only the cross-tree frontier CHOICE changes, and only while stalled.

Backbone: TransferExplorer (v13). Firewall: enable_depth=False OR not-stalled -> byte-identical to
v13. The stall trigger is the known risk (it can fire on slow-but-progressing games like tu93 whose
inter-level gap reaches 3223) — measured empirically, not assumed.
"""

from __future__ import annotations

from collections import deque

from .transfer_explorer import TransferExplorer


class DepthStallExplorer(TransferExplorer):
    def __init__(self, *args, enable_depth: bool = True, stall_trigger: int = 2000,
                 **kwargs) -> None:
        self.enable_depth = bool(enable_depth)
        self.stall_trigger = int(stall_trigger)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self._ds_since_levelup = 0
        self._ds_levels = 0

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_depth and not gstate_terminal and not gstate_notplayed:
            if levels > self._ds_levels:
                self._ds_levels = levels
                self._ds_since_levelup = 0
            else:
                self._ds_since_levelup += 1
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _stalled(self) -> bool:
        return self.enable_depth and self._ds_since_levelup >= self.stall_trigger

    def _path_to_frontier(self, start, p):
        if not self._stalled():
            return super()._path_to_frontier(start, p)
        # DEPTH mode: among all reachable frontiers with an untried action <= p, navigate to the
        # FARTHEST one (max path length) instead of the nearest -> drive into the deep tree.
        if start not in self.nodes:
            return None
        if self.nodes[start].has_untried_le(p):
            return []                      # still exhaust the current node first (coverage)
        seen = {start}
        q = deque([(start, [])])
        best = None
        while q:
            k, path = q.popleft()
            node = self.nodes.get(k)
            if not node:
                continue
            for a, (nk, _r) in node.edges.items():
                if nk in seen:
                    continue
                seen.add(nk)
                np_ = path + [a]
                nn = self.nodes.get(nk)
                if nn is not None and nn.has_untried_le(p):
                    if best is None or len(np_) > len(best):
                        best = np_
                q.append((nk, np_))
        return best
