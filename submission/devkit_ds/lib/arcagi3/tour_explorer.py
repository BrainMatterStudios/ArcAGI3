"""TourExplorer — frontier-ordering A/B variants of the banked SalienceExplorer (Phase E
Increment 2). Tests whether reordering WHICH frontier the explorer walks to cuts re-traversal
walk distance without losing coverage. SalienceExplorer is UNTOUCHED — variants live here in a
subclass selected by frontier_mode, so the banked 0.33 agent cannot regress. frontier_mode="v6"
is a byte-identical passthrough (firewall-tested). See
docs/superpowers/specs/2026-06-21-phase-e-walk-reduction-design.md.
"""
from __future__ import annotations

from collections import deque

from .salience_explorer import SalienceExplorer


class TourExplorer(SalienceExplorer):
    def __init__(self, *args, frontier_mode: str = "v6", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if frontier_mode not in ("v6", "yield", "dfs"):
            raise ValueError(f"unknown frontier_mode: {frontier_mode}")
        self.frontier_mode = frontier_mode
        self._disc_order: dict = {}      # key -> first-seen counter (side-channel, no behaviour effect)
        self._disc_counter = 0

    def _observe(self, key, cands, terminal=False):
        n = super()._observe(key, cands, terminal=terminal)
        if key not in self._disc_order:
            self._disc_order[key] = self._disc_counter
            self._disc_counter += 1
        return n

    def _path_to_frontier(self, start, p):
        if self.frontier_mode == "v6":
            return super()._path_to_frontier(start, p)
        if start not in self.nodes:
            return None
        if self.nodes[start].has_untried_le(p):
            return []
        # Full BFS: shortest action-path (+depth) to every reachable node; collect frontiers.
        frontiers = []  # (key, path, depth)
        seen = {start}
        q = deque([(start, [])])
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
                    frontiers.append((nk, np_, len(np_)))
                q.append((nk, np_))
        if not frontiers:
            return None
        if self.frontier_mode == "yield":
            d_min = min(f[2] for f in frontiers)
            near = [f for f in frontiers if f[2] == d_min]
            # most untried at the nearest depth; deterministic tie-break by earliest discovery
            return max(near, key=lambda f: (len(self.nodes[f[0]].untried_le(p)),
                                            -self._disc_order[f[0]]))[1]
        # dfs: most-recently-discovered frontier (disc_order is unique per node)
        return max(frontiers, key=lambda f: self._disc_order[f[0]])[1]
