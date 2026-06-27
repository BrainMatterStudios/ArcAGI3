"""HistoryAugmentedExplorer — folds mod-N visit-counters of the static objects the avatar steps on
into the node key, so render-invisible cyclic hidden state (e.g. ls20's rotation) becomes part of the
node identity. Self-gating: no walkable special tiles -> no counters -> byte-identical to the banked
SalienceExplorer (==TransferExplorer key). augment=False is the byte-identical firewall. See
docs/superpowers/specs/2026-06-21-history-augmented-state-design.md
"""
from __future__ import annotations

import numpy as np

from collections import Counter

from . import perception as P
from .movement import infer_all_translations
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
            # A multi-color avatar (e.g. ls20 head+body) moves RIGIDLY AS ONE: its colors share a
            # single translation delta. infer_translation returns only the smallest single mover, so
            # it would miss the body; use infer_all_translations and take the delta shared by the most
            # colors (the avatar), which also excludes independent animations (each has its own delta).
            movers = infer_all_translations(self._prev_grid, grid, self.bg)  # {color: (dr,dc)}
            if movers:
                by_delta = Counter(movers.values())
                known = [movers[c] for c in self._avatar_colors if c in movers]
                dom = (max(known, key=lambda d: by_delta[d]) if known
                       else by_delta.most_common(1)[0][0])
                for color, delta in movers.items():
                    if delta == dom:
                        self._avatar_colors.add(int(color))
        if not self._avatar_colors:
            return set()
        return {(int(r), int(c))
                for r, c in np.argwhere(np.isin(grid, list(self._avatar_colors)))}

    def _update_history(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        avatar = self._avatar_cells(grid)
        # purge any remembered location whose color turned out to be an avatar color (the avatar may
        # be learned only after a first move, so its colors can be recorded as targets at step 0).
        for sig in [s for s in self._obj_locations if s[0] in self._avatar_colors]:
            del self._obj_locations[sig]
            self._counts.pop(sig, None)
            self._prev_overlaps.discard(sig)
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
