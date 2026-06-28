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

# v5 semantic gate-identification (gate_only=True). A "gate move" is a blocked move INTO a goal-like
# object: a static component that is medium-sized (a discrete target, not the floor/walls-as-one nor a
# speck), interior (not HUD/chrome at the grid edge), and a colour that is neither background, the floor
# colour, nor an avatar colour. Verified on real ls20 frames (scripts/ls20_gate_diag.py): with these
# bounds the only such object the avatar is ever adjacent to is the framed colour-5 goal — fires UP at
# the goal-entry and nowhere else along the solution path.
GATE_MIN_SIZE = 20   # the goal is a multi-tile STRUCTURE, not a glyph/cell: excludes specks (ls20's
# rotation-tile arrow is size 3) AND ordinary game cells/tiles (tu93's maze cells are size 8-9, which
# at GATE_MIN_SIZE=6 were spuriously flagged as goal-like and — tu93 having a cyclic glyph that flips
# the phase ~hundreds of times — got eagerly retried into an L5->L2 regression). ls20's framed goal
# fill is size ~38, comfortably above this floor. Raising the floor only ever REMOVES gate flags, so it
# cannot introduce a regression on a game that was byte-identical; the only risk is under-cracking ls20
# (re-verified: still fires UP at the goal and still cracks).
GATE_MAX_SIZE = 64   # exclude the floor mega-object and large chrome strips
GATE_GAP = 3         # max cell gap between the avatar bbox and the goal object in the move direction
GATE_EDGE = 3        # components whose bbox lies within this many cells of any grid edge are chrome/HUD


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
                 retry_tier: int = MAX_TIER, gate_only: bool = False,
                 stall_trigger: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = bool(augment)
        self.counter_mod = int(counter_mod)
        # gate_only (v5): record a blocked self-loop into _blocked_phases ONLY when its move points at a
        # goal-like object (see _compute_gate_actions). Walls are then never recorded -> never re-opened
        # -> navigation stays banked-identical even at retry_tier=0, so the eager retry that cracks ls20
        # no longer re-bumps walls across the dev suite (the v4 crack-vs-regression tradeoff). Off by
        # default (records every blocked self-loop, the v4 behaviour); the crack config is
        # gate_only=True + retry_tier=0.
        self.gate_only = bool(gate_only)
        # stall_trigger (v5.1): with gate_only + EAGER retry_tier=0, the gate retry cracks ls20 but, on
        # a game with a cyclic glyph AND a still-productive frontier (lf52), the eager retry diverts it.
        # A config-only fix fails: at retry_tier=MAX_TIER the gate retry is diluted among the tier-9 click
        # lattice and never cracks ls20. Fix: keep retry_tier=0 (so it cracks) but ACTIVATE the retry only
        # once the game has LEVEL-STALLED -- no level-up for stall_trigger consecutive actions. ls20 is
        # stuck at L0 (never levels up) so it stalls and fires -> crack; lf52 levels up periodically (the
        # counter resets each level-up) so it never stalls during its productive climb to L2 -> byte-
        # identical -> no regression. (State-saturation was the wrong signal: ls20's wandering transform
        # tiles keep minting new nodes so it never saturates.) 0 = disabled (retry always active = v5).
        self.stall_trigger = max(0, int(stall_trigger))
        self._since_levelup = 0
        self._prev_stall_levels = 0
        self._stalled = False
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
        self._gate_actions: set = set()   # actions pointing at a goal-like object THIS frame (gate_only)

    def _reset_history(self):
        # The engine resets the hidden cyclic state on life-loss / level-restart; mirror that so the
        # counter tracks the state WITHIN a life rather than accumulating across resets.
        self._counts = {}
        self._prev_clusters = set()
        self._seen_count = {}
        self._tracks = []
        self._blocked_phases = {}
        self._prev_phase = 0
        self._gate_actions = set()

    def _phase(self) -> int:
        return sum(self._counts.values()) % self.counter_mod

    def _retry_active(self) -> bool:
        """Whether a phase change is allowed to re-open recorded gate moves. With stall_trigger>0 the
        eager retry is held back until the explorer has saturated state-discovery (walled)."""
        return self.stall_trigger <= 0 or self._stalled

    def _update_stall(self, levels):
        """Track consecutive actions with no level-up; once it reaches stall_trigger the game has
        level-stalled (walled) and the eager gate retry becomes active. Any level-up resets the counter,
        so a still-progressing game never activates the retry during its productive window."""
        if levels > self._prev_stall_levels:
            self._since_levelup = 0
        else:
            self._since_levelup += 1
        self._prev_stall_levels = levels
        self._stalled = self.stall_trigger > 0 and self._since_levelup >= self.stall_trigger

    def _free_stale_blocks_global(self):
        """The phase changed: re-open each recorded BLOCKED self-loop not yet confirmed blocked at the
        new phase, across all nodes, at priority ``retry_tier``. A move blocked at ALL ``counter_mod``
        phases is a confirmed static wall and is never re-opened. Real navigation edges (next != key) are
        never removed. What stays in ``_blocked_phases`` is controlled by ``_record``: with gate_only it
        is ONLY gate moves (into a goal-like object), so an EAGER retry_tier=0 re-opens just the gate and
        never re-bumps walls; without it, every blocked self-loop is recorded and the dev suite is kept
        safe instead by retry_tier=MAX_TIER (re-tried only after every real frontier is exhausted)."""
        ph = self._phase()
        for (key, a), phases in self._blocked_phases.items():
            if ph in phases:
                continue                     # already known blocked at this phase -> wall-safe, skip
            node = self.nodes.get(key)
            if node is not None:
                e = node.edges.get(a)
                if e is not None and e[1] == 0 and e[0] == key:
                    del node.edges[a]                # not yet tried at this phase -> re-open ...
                    node.tier[a] = self.retry_tier   # ... at the configured retry priority

    def _record(self, key, action, next_key, reward, cands, terminal):
        super()._record(key, action, next_key, reward, cands, terminal)
        # blocked self-loop (no reward, avatar didn't move). In gate_only mode only record it if the move
        # points at a goal-like object this frame -- so walls are never recorded and the eager retry
        # never re-bumps them (v5). Otherwise record every blocked self-loop (v4).
        if self.augment and reward == 0 and next_key == key:
            if not self.gate_only or action in self._gate_actions:
                self._blocked_phases.setdefault((key, action), set()).add(self._phase())

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.augment:
            if gstate_terminal or gstate_notplayed or levels != self._hist_levels:
                self._reset_history()        # life-loss / reset / level-up -> engine resets state
                self._hist_levels = levels
            else:
                self._update_history(grid)   # updates self._counts (the phase) BEFORE super().decide
                self._update_stall(levels)   # track level-stall -> gates the eager retry
                ph = self._phase()
                if ph != self._prev_phase:
                    if self._retry_active():
                        self._free_stale_blocks_global()   # phase changed + walled -> re-open gate moves
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

    def _compute_gate_actions(self, grid, all_comps, small, flags):
        """Actions whose direction points the avatar at a GOAL-LIKE object this frame. Goal-like = a
        static component that is medium-sized (GATE_MIN..GATE_MAX), interior (not within GATE_EDGE of a
        grid edge -> excludes HUD/chrome), and a colour that is neither background, the floor colour
        (largest component), nor an avatar colour. The avatar is the mobile components (``flags``).
        Returns simple-action tuples ("S", aid) for up/down/left/right (1/2/3/4)."""
        avatar_cells, avatar_colors = [], set()
        for c, moved in zip(small, flags):
            if moved:
                avatar_cells.extend((int(r), int(cc)) for r, cc in c.cells)
                avatar_colors.add(int(c.color))
        if not avatar_cells:
            return set()
        ar0 = min(r for r, _ in avatar_cells); ar1 = max(r for r, _ in avatar_cells)
        ac0 = min(c for _, c in avatar_cells); ac1 = max(c for _, c in avatar_cells)
        floor = max(all_comps, key=lambda o: o.size).color if all_comps else None
        excl = {self.bg, int(floor) if floor is not None else None} | avatar_colors
        h, w = grid.shape
        out: set = set()
        for o in all_comps:
            if not (GATE_MIN_SIZE <= o.size <= GATE_MAX_SIZE) or int(o.color) in excl:
                continue
            r0, c0, r1, c1 = o.bbox
            if r0 < GATE_EDGE or c0 < GATE_EDGE or r1 >= h - GATE_EDGE or c1 >= w - GATE_EDGE:
                continue                                    # HUD / chrome at the grid edge
            if not (c1 < ac0 or c0 > ac1):                  # column overlap -> a vertical move can reach it
                if r0 < ar0 and ar0 - r1 <= GATE_GAP:
                    out.add(("S", 1))                       # object above -> up
                if r1 > ar1 and r0 - ar1 <= GATE_GAP:
                    out.add(("S", 2))                       # object below -> down
            if not (r1 < ar0 or r0 > ar1):                  # row overlap -> a horizontal move can reach it
                if c0 < ac0 and ac0 - c1 <= GATE_GAP:
                    out.add(("S", 3))                       # object left -> left
                if c1 > ac1 and c0 - ac1 <= GATE_GAP:
                    out.add(("S", 4))                       # object right -> right
        return out

    def _update_history(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        all_comps = P.connected_components(grid, background=self.bg)
        comps = [o for o in all_comps if o.size <= OBJ_MAX_SIZE]
        flags = self._flag_mobility(comps)
        if self.gate_only:
            self._gate_actions = self._compute_gate_actions(grid, all_comps, comps, flags)
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
