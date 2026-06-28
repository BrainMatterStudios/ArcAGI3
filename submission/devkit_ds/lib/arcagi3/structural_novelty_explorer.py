"""StructuralNoveltyExplorer — intrinsic structural-novelty exploration (goal-free).

Sidesteps the proven goal-identification wall: instead of trying to learn WHICH state is the
goal, bias exploration toward states that introduce a NEW object CONFIGURATION (color-count
multiset). Progress events (collect a key -> a count drops; open a door -> a region recolors;
new object appears) change the multiset; mere avatar wandering does NOT. So structural novelty
is a discriminating, goal-free intrinsic signal — unlike the killed go-explore progress flag
(which fired on ~98% of states because maze cells recolor as the avatar passes).

Applied conservatively as a TIE-BREAK among EQUIDISTANT frontiers (preserving the load-bearing
nearest-first ordering that the walk-reduction A/B showed is essential), the agent, when two
frontiers are equally close, heads toward the one whose neighborhood has historically produced
more structural novelty. enable_struct=False -> byte-identical to SalienceExplorer (v6).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import perception as P
from .salience_explorer import MAX_TIER, SalienceExplorer


class StructuralNoveltyExplorer(SalienceExplorer):
    def __init__(self, *args, enable_struct: bool = True, **kwargs) -> None:
        self.enable_struct = bool(enable_struct)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.seen_sigs: set = set()
        self.sig_by_key: dict = {}
        self.node_yield: dict = {}   # key -> [novel_count, tried_count]

    def _struct_sig(self, grid):
        objs = P.connected_components(grid, background=self.bg)
        counts: dict[int, int] = {}
        for o in objs:
            counts[o.color] = counts.get(o.color, 0) + 1
        return frozenset(counts.items())

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.enable_struct and not gstate_terminal and not gstate_notplayed:
            if self.bg is None:
                self.bg = P.detect_background(grid)
            cur = self._key(grid)
            if cur not in self.sig_by_key:
                sig = self._struct_sig(grid)
                self.sig_by_key[cur] = sig
                novel = sig not in self.seen_sigs
                self.seen_sigs.add(sig)
                # credit the node we just acted from with whether this led somewhere structurally new
                if self.prev_key is not None and self.prev_action is not None:
                    y = self.node_yield.setdefault(self.prev_key, [0, 0])
                    y[0] += 1 if novel else 0
                    y[1] += 1
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _yield_score(self, key):
        y = self.node_yield.get(key)
        return (y[0] / y[1]) if y and y[1] else 0.0

    def _path_to_frontier(self, start, p):
        if not self.enable_struct:
            return super()._path_to_frontier(start, p)
        if start not in self.nodes:
            return None
        if self.nodes[start].has_untried_le(p):
            return []
        # BFS, but among all frontiers found at the SHALLOWEST depth, pick the highest-yield one
        seen = {start}
        q = deque([(start, [])])
        best_path, best_depth, best_y = None, None, -1.0
        while q:
            k, path = q.popleft()
            if best_depth is not None and len(path) > best_depth:
                break  # only consider the shallowest frontier layer (preserve nearest-first)
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
                    ys = self._yield_score(nk)
                    if best_depth is None or (len(np_) == best_depth and ys > best_y):
                        best_path, best_depth, best_y = np_, len(np_), ys
                else:
                    q.append((nk, np_))
        return best_path
