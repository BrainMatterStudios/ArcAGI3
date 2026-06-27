"""HistoryAugmentedExplorer — folds mod-N visit-counters of the static objects the avatar steps on
into the node key, so render-invisible cyclic hidden state (e.g. ls20's rotation) becomes part of the
node identity. Self-gating: no walkable special tiles -> no counters -> byte-identical to the banked
SalienceExplorer (==TransferExplorer key). augment=False is the byte-identical firewall. See
docs/superpowers/specs/2026-06-21-history-augmented-state-design.md
"""
from __future__ import annotations

import numpy as np

from . import perception as P
from .movement import infer_translation
from .salience_explorer import SalienceExplorer

OBJ_MAX_SIZE = 16   # only track small glyphs/tiles as visit targets (not big regions)


class HistoryAugmentedExplorer(SalienceExplorer):
    def __init__(self, *args, augment: bool = False, counter_mod: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = bool(augment)
        self.counter_mod = int(counter_mod)
        self._counts: dict = {}          # object signature -> cumulative visit count
        self._obj_locations: dict = {}   # signature -> frozenset of (r,c) cells (last seen visible)
        self._avatar_colors: set = set()
        self._prev_grid = None
        self._prev_overlaps: set = set()

    def _key(self, grid):
        base = super()._key(grid)
        if not self.augment or not self._counts:
            return base
        aug = tuple(sorted((sig, c % self.counter_mod) for sig, c in self._counts.items()))
        return base + b"|H|" + repr(aug).encode()

    def _avatar_cells(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        if self._prev_grid is not None and self._prev_grid.shape == grid.shape:
            tr = infer_translation(self._prev_grid, grid, self.bg)
            if tr is not None:
                self._avatar_colors.add(int(tr[0]))
        if not self._avatar_colors:
            return set()
        return {(int(r), int(c))
                for r, c in np.argwhere(np.isin(grid, list(self._avatar_colors)))}

    def _update_history(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        avatar = self._avatar_cells(grid)
        # refresh remembered locations of small static glyphs that are CURRENTLY visible and
        # NOT the avatar (avatar colors excluded so the avatar isn't a visit target)
        for o in P.connected_components(grid, background=self.bg):
            if o.size <= OBJ_MAX_SIZE and int(o.color) not in self._avatar_colors:
                self._obj_locations[(int(o.color),) + tuple(o.bbox)] = \
                    frozenset((int(r), int(c)) for r, c in o.cells)
        # edge-triggered visit: avatar enters a remembered object's cells now but did not last step
        overlaps_now = set()
        for sig, cells in self._obj_locations.items():
            if avatar & cells:
                overlaps_now.add(sig)
                if sig not in self._prev_overlaps:
                    self._counts[sig] = self._counts.get(sig, 0) + 1
        self._prev_overlaps = overlaps_now
        self._prev_grid = grid
