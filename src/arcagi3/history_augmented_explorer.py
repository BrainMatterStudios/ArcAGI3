"""HistoryAugmentedExplorer — folds mod-N visit-counters of static glyphs the avatar steps on into the
node key, so render-invisible cyclic hidden state (e.g. ls20's rotation) becomes part of the node
identity. Self-gating: no occludable special glyphs -> no counters -> byte-identical to the banked
SalienceExplorer (==TransferExplorer key). augment=False is the byte-identical firewall.

VISIT DETECTION = DISAPPEARANCE/OCCLUSION EVENTS (not avatar overlap). A "visit" is a CONFIRMED-STATIC
glyph cluster (seen at the same cells for >=2 consecutive frames) that was visible last frame and is
gone this frame -- it can only vanish because the avatar stepped onto/over it (occlusion). This needs
NO avatar segmentation (the ls20 blocker: the avatar body color is shared with the maze), because the
moving avatar's own parts never repeat a cell-set and so never confirm static. Cluster signatures are
color-aware, so "occluded" (avatar color now on those cells) reads as absent.
See docs/superpowers/specs/2026-06-21-history-augmented-state-design.md
"""
from __future__ import annotations

from . import perception as P
from .salience_explorer import SalienceExplorer

OBJ_MAX_SIZE = 16   # only track small glyphs/tiles (not big regions/walls-as-one)


class HistoryAugmentedExplorer(SalienceExplorer):
    def __init__(self, *args, augment: bool = False, counter_mod: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = bool(augment)
        self.counter_mod = int(counter_mod)
        self._counts: dict = {}          # cluster signature -> cumulative occlusion (visit) count
        self._prev_clusters: set = set()  # cluster sigs visible last frame
        self._seen_count: dict = {}       # sig -> consecutive frames seen (>=2 => confirmed static)
        self._hist_levels = 0             # last-seen levels, to detect a level-up (engine resets state)

    def _reset_history(self):
        # The engine resets the hidden cyclic state on life-loss / level-restart; mirror that so the
        # counter tracks the state WITHIN a life rather than accumulating across resets.
        self._counts = {}
        self._prev_clusters = set()
        self._seen_count = {}

    def _key(self, grid):
        base = super()._key(grid)
        if not self.augment or not self._counts:
            return base
        aug = tuple(sorted((sig, c % self.counter_mod) for sig, c in self._counts.items()))
        return base + b"|H|" + repr(aug).encode()

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.augment:
            if gstate_terminal or gstate_notplayed or levels != self._hist_levels:
                self._reset_history()        # life-loss / reset / level-up -> engine resets state
                self._hist_levels = levels
            else:
                self._update_history(grid)   # updates self._counts BEFORE super().decide -> self._key
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _cluster_sigs(self, grid, bg):
        """Color-aware signatures of spatially-clustered small non-background objects.

        Groups adjacent small-object cells (8-connectivity) so a multi-cell glyph (e.g. ls20's arrow,
        3 cells of colors 0/1) is ONE cluster. sig = sorted tuple of (r, c, color) -> stable for a
        static glyph, distinct from whatever occludes it.
        """
        small: dict = {}
        for o in P.connected_components(grid, background=bg):
            if o.size <= OBJ_MAX_SIZE:
                for r, c in o.cells:
                    small[(int(r), int(c))] = int(o.color)
        sigs, seen = set(), set()
        for cell in small:
            if cell in seen:
                continue
            comp, stack = [cell], [cell]
            seen.add(cell)
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nb = (r + dr, c + dc)
                        if nb in small and nb not in seen:
                            seen.add(nb)
                            comp.append(nb)
                            stack.append(nb)
            sigs.add(tuple(sorted((r, c, small[(r, c)]) for r, c in comp)))
        return sigs

    def _update_history(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        present = self._cluster_sigs(grid, self.bg)
        # visit = a confirmed-static cluster present last frame, gone now (occluded by the avatar).
        for sig in self._prev_clusters - present:
            if self._seen_count.get(sig, 0) >= 2:
                self._counts[sig] = self._counts.get(sig, 0) + 1
        self._seen_count = {sig: self._seen_count.get(sig, 0) + 1 for sig in present}
        self._prev_clusters = present
