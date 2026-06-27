"""HistoryAugmentedExplorer — folds mod-N visit-counters of static glyphs the avatar steps on into the
node key, so render-invisible cyclic hidden state (e.g. ls20's rotation) becomes part of the node
identity. Self-gating: no occludable special glyphs -> no counters -> byte-identical to the banked
SalienceExplorer (==TransferExplorer key). augment=False is the byte-identical firewall.

VISIT DETECTION = DISAPPEARANCE/OCCLUSION EVENTS, with MOBILITY MEMORY. A "visit" is a CONFIRMED-STATIC
glyph cluster (seen at the same cells for >=2 consecutive frames) that was visible last frame and is
gone this frame -- it can only vanish because the avatar stepped onto/over it (occlusion).

The ls20 trap: stability ALONE cannot distinguish a real static glyph from a PAUSED avatar (the avatar
sits >=2 frames whenever a move is blocked by a wall), so the prior detector counted the avatar's own
blocks. The only discriminator is HISTORY: the avatar moved BEFORE it paused. So we track each small
connected component across frames (per-color nearest-centroid match) and carry an ``ever_moved`` flag;
any component that has ever translated is EXCLUDED from glyph clusters forever (even while paused). This
needs no color-based avatar segmentation (the body color 9 is shared with the maze) because matching is
per-COMPONENT: the moving avatar-body component is mobile while static maze components of the same color
are not. Clusters are built by 8-adjacency over STATIC components only, so the multi-component arrow
groups into one unit and the avatar can never merge into it.
See docs/superpowers/specs/2026-06-21-history-augmented-state-design.md (Redesign v3, mobility memory).
"""
from __future__ import annotations

from . import perception as P
from .salience_explorer import SalienceExplorer

OBJ_MAX_SIZE = 16    # only track small glyphs/tiles (not big regions/walls-as-one)
MOVE_EPS = 0.5       # centroid manhattan delta above which a matched component counts as "moved"
MATCH_GATE = 6.0     # max centroid manhattan distance to match a component to the previous frame


class HistoryAugmentedExplorer(SalienceExplorer):
    def __init__(self, *args, augment: bool = False, counter_mod: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = bool(augment)
        self.counter_mod = int(counter_mod)
        self._counts: dict = {}          # cluster signature -> cumulative occlusion (visit) count
        self._prev_clusters: set = set()  # cluster sigs visible last frame
        self._seen_count: dict = {}       # sig -> consecutive frames seen (>=2 => confirmed static)
        self._tracks: list = []           # per-component identity: list of (color, centroid, ever_moved)
        self._hist_levels = 0             # last-seen levels, to detect a level-up (engine resets state)

    def _reset_history(self):
        # The engine resets the hidden cyclic state on life-loss / level-restart; mirror that so the
        # counter tracks the state WITHIN a life rather than accumulating across resets.
        self._counts = {}
        self._prev_clusters = set()
        self._seen_count = {}
        self._tracks = []

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

    def _flag_mobility(self, comps):
        """Match each small component to the previous frame (per-color, global nearest-centroid greedy)
        and return a parallel list of ``ever_moved`` flags. Mobility is sticky: once a component has
        translated it stays mobile, so a paused avatar is excluded for the rest of the life. Also
        rebuilds ``self._tracks`` for the next frame.
        """
        prev = self._tracks
        triples = []  # (dist, comp_idx, prev_idx) for same-color pairs within the match gate
        for ci, c in enumerate(comps):
            for pi, (pcolor, pcent, _) in enumerate(prev):
                if pcolor != c.color:
                    continue
                d = abs(c.centroid[0] - pcent[0]) + abs(c.centroid[1] - pcent[1])
                if d <= MATCH_GATE:
                    triples.append((d, ci, pi))
        triples.sort()
        match: dict = {}            # comp_idx -> prev_idx
        used_c, used_p = set(), set()
        for d, ci, pi in triples:
            if ci in used_c or pi in used_p:
                continue
            used_c.add(ci)
            used_p.add(pi)
            match[ci] = pi
        flags, new_tracks = [], []
        for ci, c in enumerate(comps):
            if ci in match:
                pcolor, pcent, pmoved = prev[match[ci]]
                d = abs(c.centroid[0] - pcent[0]) + abs(c.centroid[1] - pcent[1])
                moved = pmoved or (d > MOVE_EPS)
            else:
                moved = False       # freshly appeared -- innocent until it translates
            flags.append(moved)
            new_tracks.append((c.color, c.centroid, moved))
        self._tracks = new_tracks
        return flags

    def _cluster_static(self, comps, flags):
        """8-adjacency cluster signatures over the cells of STATIC (not ``ever_moved``) components only.

        Excluding mobile components both removes the avatar (so it can never confirm static) and stops
        it merging into a glyph cluster. sig = sorted (r, c, color) -> stable for a static glyph,
        distinct from whatever occludes it.
        """
        small: dict = {}
        for c, moved in zip(comps, flags):
            if moved:
                continue
            for r, col in c.cells:
                small[(int(r), int(col))] = int(c.color)
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
        comps = [o for o in P.connected_components(grid, background=self.bg) if o.size <= OBJ_MAX_SIZE]
        flags = self._flag_mobility(comps)
        present = self._cluster_static(comps, flags)
        mobile_cells = {(int(r), int(c))
                        for comp, moved in zip(comps, flags) if moved for r, c in comp.cells}
        # visit = a confirmed-static cluster gone this frame AND now covered by a MOVING object: only a
        # mobile occluder (the avatar stepping on) counts. A glyph that vanishes into background (avatar
        # leaving spawn) or is replaced by another static thing (a HUD digit changing) is NOT a visit.
        for sig in self._prev_clusters - present:
            if self._seen_count.get(sig, 0) >= 2 and any((r, c) in mobile_cells for r, c, _ in sig):
                self._counts[sig] = self._counts.get(sig, 0) + 1
        self._seen_count = {sig: self._seen_count.get(sig, 0) + 1 for sig in present}
        self._prev_clusters = present
