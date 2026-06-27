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
