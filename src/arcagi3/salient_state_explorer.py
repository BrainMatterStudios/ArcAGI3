"""SalientStateExplorer — rare-object-exact / common-object-merged state key.

The wall games' state graph explodes because the full object hash distinguishes every
positional variant of COMMON objects (walls, floor, counters) — behaviourally-irrelevant
noise — while the goal lives in a few RARE objects (avatar, key, door, button). The lossy
relational hash (quantize EVERYTHING) cratered because it merged the goal-relevant objects
too. This keys state on:
  - RARE objects (color appears on <= rare_max objects): EXACT geometry (color,bbox,size)
  - COMMON objects: merged to a per-color (count, total_size) multiset (positions discarded)
So positional noise in walls/floor collapses (smaller graph, faster to the goal) while the
avatar/key/door/button stay precise (goal distinctions preserved). Dynamics-preserving in the
spirit of bisimulation: it only merges detail that the goal can't depend on.

Firewall: enable_salient=False -> byte-identical to SalienceExplorer (v6). This is a research
policy measured on the eval harness; the submission default stays TransferExplorer.
"""

from __future__ import annotations

import numpy as np

from . import perception as P
from .salience_explorer import SalienceExplorer


class SalientStateExplorer(SalienceExplorer):
    def __init__(self, *args, enable_salient: bool = True, rare_max: int = 2, **kwargs) -> None:
        self.enable_salient = bool(enable_salient)
        self.rare_max = int(rare_max)
        super().__init__(*args, **kwargs)

    def _key(self, grid):
        if not self.enable_salient:
            return super()._key(grid)
        # replicate base masking (volatile + dynamic border)
        m = self.vt.mask()
        bm = self._border_mask()
        if bm is not None:
            m = m | bm
        if m.any():
            grid = grid.copy()
            grid[m] = self.bg if self.bg is not None else 0
        objs = P.connected_components(grid, background=self.bg)
        counts: dict[int, int] = {}
        for o in objs:
            counts[o.color] = counts.get(o.color, 0) + 1
        rare_parts = []          # exact geometry for rare (likely interactive/goal) objects
        common: dict[int, list] = {}   # color -> [count, total_size]
        for o in objs:
            if counts[o.color] <= self.rare_max:
                r0, c0, r1, c1 = o.bbox
                rare_parts.append((int(o.color), r0, c0, r1, c1, o.size))
            else:
                agg = common.setdefault(int(o.color), [0, 0])
                agg[0] += 1
                agg[1] += o.size
        rare_parts.sort()
        common_parts = sorted((c, n, s) for c, (n, s) in common.items())
        return repr(("S", tuple(rare_parts), tuple(common_parts))).encode()
