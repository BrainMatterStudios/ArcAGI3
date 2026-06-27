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
from .salience_explorer import MAX_TIER, SalienceExplorer

OBJ_MAX_SIZE = 16    # only track small glyphs/tiles (not big regions/walls-as-one)
MOVE_EPS = 0.5       # centroid manhattan delta above which a matched component counts as "moved"
MATCH_GATE = 6.0     # max centroid manhattan distance to match a component to the previous frame


class HistoryAugmentedExplorer(SalienceExplorer):
    """PHASE-GATED RETRY (v4): the hidden cyclic phase NEVER enters the node key (so navigation/pathing
    is byte-identical to banked -- the v2/v3 lesson: changing keys mid-exploration fragments AND breaks
    the graph). Instead, a node's BLOCKED action -- a self-loop (next_key == key, no reward), i.e. a move
    the avatar could not make -- is FREED (made untried again) when the hidden phase changes, so the
    explorer re-tries it at a phase it has not yet tried there. Crucially the re-opened move is
    deprioritised to the LOWEST tier (MAX_TIER): the explorer only chases it after exhausting every real
    frontier, so a productive game (which always has higher-tier frontiers) is never diverted into
    re-bumping walls, while ls20 -- whose finite maze gets fully explored -- climbs to the lowest tier
    and retries the blocked goal-entry until the winning rotation. A move blocked at ALL counter_mod
    phases is a confirmed static wall and is never re-opened (cap). Real navigation edges (next_key !=
    key) are never touched, and the node key never changes, so navigation is byte-identical to banked
    (the v2/v3 lesson). Self-gating: no occludable cyclic glyph -> phase never changes -> nothing is ever
    freed -> behaviour is banked. ``augment=False`` is the firewall.
    """

    def __init__(self, *args, augment: bool = False, counter_mod: int = 4,
                 retry_tier: int = MAX_TIER, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = bool(augment)
        self.counter_mod = int(counter_mod)
        # Priority tier a re-opened blocked move is given. Measured tradeoff (see the spec's v4 result):
        #   0  = same priority as real moves -> cracks ls20 (+1 solve, faster) but DIVERTS productive
        #        games (tu93 L5->L0): net dev regression.
        #   1  = above simple-move frontiers -> worst of both (regresses lp85/lf52, still no crack).
        #   MAX_TIER = last resort -> zero dev regression but the retry fires too late to crack ls20.
        # No single tier is both cracking AND non-regressing, because the explorer cannot tell the
        # outcome-gating GATE from an ordinary WALL (both are blocked self-loops). DEFAULT = MAX_TIER
        # (safe / non-regressing). The crack needs SEMANTIC gate-identification (re-open only blocked
        # moves into a salient goal-like object), a follow-up sub-project this mechanism is substrate for.
        self.retry_tier = max(0, min(int(retry_tier), MAX_TIER))
        self._counts: dict = {}          # cluster signature -> cumulative occlusion (visit) count
        self._prev_clusters: set = set()  # cluster sigs visible last frame
        self._seen_count: dict = {}       # sig -> consecutive frames seen (>=2 => confirmed static)
        self._tracks: list = []           # per-component identity: list of (color, centroid, ever_moved)
        self._blocked_phases: dict = {}   # (key, action) -> set of phases this self-loop move was blocked at
        self._prev_phase: int = 0         # phase last step, to detect a phase change -> re-open stale blocks
        self._hist_levels = 0             # last-seen levels, to detect a level-up (engine resets state)

    def _reset_history(self):
        # The engine resets the hidden cyclic state on life-loss / level-restart; mirror that so the
        # counter tracks the state WITHIN a life rather than accumulating across resets.
        self._counts = {}
        self._prev_clusters = set()
        self._seen_count = {}
        self._tracks = []
        self._blocked_phases = {}
        self._prev_phase = 0

    def _phase(self) -> int:
        return sum(self._counts.values()) % self.counter_mod

    def _free_stale_blocks_global(self):
        """The phase changed: re-open each BLOCKED self-loop not yet confirmed blocked at the new phase,
        across all nodes, and DEPRIORITISE it to MAX_TIER so the explorer only re-tries it once every
        real frontier is exhausted (this is what keeps productive games undisturbed). A move blocked at
        ALL ``counter_mod`` phases is a confirmed static wall and is never re-opened. Real navigation
        edges (next != key) are never removed."""
        ph = self._phase()
        for (key, a), phases in self._blocked_phases.items():
            if ph in phases:
                continue                     # already known blocked at this phase -> wall-safe, skip
            node = self.nodes.get(key)
            if node is not None:
                e = node.edges.get(a)
                if e is not None and e[1] == 0 and e[0] == key:
                    del node.edges[a]              # not yet tried at this phase -> re-open ...
                    node.tier[a] = self.retry_tier  # ... at a low priority so it never diverts a productive game

    def _record(self, key, action, next_key, reward, cands, terminal):
        super()._record(key, action, next_key, reward, cands, terminal)
        if self.augment and reward == 0 and next_key == key:
            self._blocked_phases.setdefault((key, action), set()).add(self._phase())

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.augment:
            if gstate_terminal or gstate_notplayed or levels != self._hist_levels:
                self._reset_history()        # life-loss / reset / level-up -> engine resets state
                self._hist_levels = levels
            else:
                self._update_history(grid)   # updates self._counts (the phase) BEFORE super().decide
                ph = self._phase()
                if ph != self._prev_phase:
                    self._free_stale_blocks_global()   # phase changed -> re-open blocked moves (low tier)
                    self._prev_phase = ph
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
