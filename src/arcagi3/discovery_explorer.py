"""DiscoveryExplorer — active model-discovery loop on a black-box game (Exp #43, T8).

A STATEFUL phase machine implementing the standard engine `decide()` contract: it returns ONE
action per call and the harness applies it. The loop is:

    PROBE_MOVEMENT  -> identify the agent + fit action_id -> (dr,dc); find candidate walls
    PROBE_TRANSFORMS-> graph-scaffolded coverage; record attribute-change triples on entry
    INDUCE          -> build an InducedModel from deltas + on_enter_cycles + walls; set terminal
    PLAN            -> enumerate candidate slot attr-specs; BFS-plan to satisfy them; cache plan
    EXECUTE         -> emit cached plan actions one per decide()
    REFINE          -> on a misfire/exhausted plan, log the surprise and re-INDUCE

On a `levels` increment the GRAMMAR (deltas, tiles, terminal form) is KEPT while the LAYOUT
(walls, plan, seen-hashes) is flushed and PROBE_TRANSFORMS is re-entered — the cross-level
transfer lever. All primitives are reused from perception/movement/attribute_state/
transform_induction/factored_model/scene_graph; nothing here is game-specific.
"""
from __future__ import annotations

import numpy as np

from arcagi3 import perception as P, movement as Mv, scene_graph as SG
from arcagi3.attribute_state import agent_attributes
from arcagi3.motion_lattice import lattice_pitch, snap
from arcagi3.transform_induction import (
    induce_on_enter_cycles, induce_terminal,
    induce_recolor_on_move, induce_collect_on_contact,
)
from arcagi3.factored_model import FactoredState, InducedModel, plan, plan_painted_set

# Bounded coverage so a black-box game can't trap the explorer in a phase forever.
_MAX_TRANSFORM_STEPS = 400
# Movement-probe is bounded too: try each directional action this many rounds, then proceed to
# induction with WHATEVER deltas were found. Without this cap, a game where only a subset of
# directions move the avatar (1-axis movement, blocked directions) never satisfies
# _deltas_fully_probed and the probe re-tries the dead directions forever, never reaching PLAN.
_MAX_PROBE_ROUNDS = 2
_MAX_TARGET_OBJ_SIZE = 64   # objects this small (non-agent, non-bg) can be collect targets


class DiscoveryExplorer:
    def __init__(self, seed: int = 0, planner_backend: str = "traversal"):
        self.seed = seed
        self._backend = planner_backend  # "traversal" = A1 (path BFS) | "painted_set" = A2
        self.reset_all()

    # ------------------------------------------------------------------ lifecycle
    def reset_all(self):
        self._phase = "PROBE_MOVEMENT"
        self._deltas: dict[int, tuple[int, int]] = {}
        self._walls: set = set()
        self._tiles: dict = {}
        self._cycles: list = []
        self._triples: list[dict] = []
        self._last_induced_len = (-1, -1)
        self._plan: list = []
        self._goal_pos = None           # the candidate position the current plan targets
        self._tried_goals: set = set()  # candidate positions already reached without a level-up
        self._seen_hashes: set = set()
        self._last_level = 0
        self._prev_grid = None
        self._prev_token = None
        self._agent_color = None
        self._probe_queue: list = []
        self._probe_rounds = 0  # times the movement-probe queue has been refilled (bounded by _MAX_PROBE_ROUNDS)
        self._bg = None
        self._terminal = None
        self._transform_steps = 0
        self._prev_attr = None
        # Multi-color avatar: a rigid sprite can be made of several colors that translate together
        # (e.g. a "head" color over a "body" color). We track the WHOLE set so attribute extraction
        # (orientation/shape) sees the full sprite, not one fragment, and so the avatar's own colors
        # are not mistaken for a world recolor/paint. `_agent_anchor` localizes the avatar cluster
        # (its colors also appear elsewhere — goals, UI — so we can't take all cells globally).
        self._agent_colors: set = set()
        self._agent_anchor = None
        self._visited: set = set()  # avatar anchors visited (for frontier-seeking coverage)
        self._wall_order: list = []  # walls in the order learned (for boxed-in escape)
        self._world_obs: list[dict] = []
        # World-transform rules induced from _world_obs (Task 6/7):
        self._recolors: list = []        # RecolorOnMove(from_color, to_color) — paint
        self._collects: list = []        # CollectOnContact(color) — collect/vanish
        self._paint_colors: set = set()  # from_color of each recolor rule (paintable cells)

    def on_level_change(self, new_level: int):
        """KEEP the induced grammar (deltas/tiles/terminal); FLUSH the layout state."""
        self._last_level = new_level
        self._walls = set()
        self._plan = []
        self._goal_pos = None
        self._tried_goals = set()
        self._seen_hashes = set()
        self._probe_queue = []
        self._probe_rounds = 0
        self._transform_steps = 0
        self._prev_attr = None
        self._prev_grid = None
        self._agent_anchor = None  # avatar colors (grammar) kept; its location (layout) reset
        self._visited = set()
        self._wall_order = []
        # grammar known -> jump straight to re-mapping the new layout's transforms; else re-probe.
        self._phase = "PROBE_TRANSFORMS" if self._deltas else "PROBE_MOVEMENT"

    # ------------------------------------------------------------------ contract
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if levels != self._last_level:
            self.on_level_change(levels)
        if gstate_terminal:
            self._plan = []
            return ("reset",)
        tok = self._decide_inner(grid, available)
        self._prev_grid, self._prev_token = grid, tok
        return tok

    # ------------------------------------------------------------------ helpers
    def _simple_actions(self, available):
        return [a for a in available if a in (1, 2, 3, 4)]

    def _avatar_colors(self) -> set:
        """All colors that translate together as the avatar; falls back to the single _agent_color."""
        cols = set(self._agent_colors)
        if self._agent_color is not None:
            cols.add(int(self._agent_color))
        if self._bg is not None:
            cols.discard(int(self._bg))  # the background is never part of the avatar
        return cols

    def _agent_cells(self, grid):
        """Cells of the identified avatar — the connected cluster of avatar-colored cells nearest
        the avatar's last known anchor.

        A multi-color avatar (e.g. ls20's color-12 head over a color-9 body) translates as ONE
        rigid sprite, so we group ALL avatar colors. But those same colors also appear elsewhere on
        the board (goal markers, UI), so we must NOT take every cell of those colors globally — we
        take the single 8-connected component (over the union of avatar colors) closest to the
        avatar's last anchor. This keeps attribute extraction (orientation) on the real sprite and
        prevents distant same-color decorations from polluting the agent footprint."""
        cols = self._avatar_colors()
        if not cols or self._bg is None:
            return ()
        import numpy as _np
        mask = _np.isin(grid, list(cols))
        if not mask.any():
            return ()
        # 8-connected components of the avatar-color mask
        comps = self._connected_mask_components(mask)
        if not comps:
            return ()
        anchor = self._agent_anchor
        if anchor is None:
            # no prior anchor: pick the largest cluster (the avatar sprite is a solid block, whereas
            # stray same-color decorations are typically small/scattered).
            best = max(comps, key=len)
        else:
            def _cdist(cells):
                rs = [r for r, _ in cells]; cs = [c for _, c in cells]
                cy = sum(rs) / len(rs); cx = sum(cs) / len(cs)
                return abs(cy - anchor[0]) + abs(cx - anchor[1])
            best = min(comps, key=_cdist)
        return tuple(best)

    @staticmethod
    def _connected_mask_components(mask):
        """8-connected components of a boolean mask -> list of cell-tuple lists."""
        import numpy as _np
        h, w = mask.shape
        seen = _np.zeros_like(mask, dtype=bool)
        out: list = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]; seen[r, c] = True; comp = []
                while stack:
                    cr, cc = stack.pop(); comp.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True; stack.append((nr, nc))
                out.append(comp)
        return out

    @staticmethod
    def _grow_within(allowed: set, seed: set) -> set:
        """8-connected flood fill from `seed`, staying within `allowed` cells."""
        out = set(); stack = [c for c in seed if c in allowed]
        out.update(stack)
        while stack:
            cr, cc = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (cr + dr, cc + dc)
                    if n in allowed and n not in out:
                        out.add(n); stack.append(n)
        return out

    def _agent_pos(self, grid):
        cells = self._agent_cells(grid)
        if not cells:
            return None
        # representative position = the (min-row, min-col) anchor of the agent cell-set
        return (min(r for r, _ in cells), min(c for _, c in cells))

    def _update_anchor(self, grid):
        """Refresh the avatar's tracked centroid so _agent_cells can follow the right cluster."""
        cols = self._avatar_colors()
        if not cols or self._bg is None:
            return
        cells = self._agent_cells(grid)
        if cells:
            cy = sum(r for r, _ in cells) / len(cells)
            cx = sum(c for _, c in cells) / len(cells)
            self._agent_anchor = (cy, cx)

    def _ingest_movement(self, grid):
        """From prev_grid -> grid, find the moving color (the agent) and fit the action's delta.

        A simple action that produced NO motion marks the attempted direction as a candidate
        wall in front of the agent. Returns True iff the previous simple action FAILED to move the
        agent (a confirmed wall bump) — the caller uses this to abandon a stalled plan during
        EXECUTE so it can re-route with the new wall knowledge (Fix 1).
        """
        # TODO(phase2): consolidate agent identification with movement.MotionModel /
        # reach_target_test.identify_agent (multi-color avatars) instead of this inline heuristic.
        if self._bg is None:
            return False
        if self._prev_grid is None or self._prev_token is None:
            return False
        if self._prev_token[0] != "S":
            return False
        action = self._prev_token[1]
        movers = Mv.infer_all_translations(self._prev_grid, grid, self._bg)
        if movers:
            # The avatar is a rigid sprite: ALL its colors translate by the SAME vector. Independent
            # animations/counters translate by a DIFFERENT vector (or none). So group the colors
            # sharing the single most-common translation as the multi-color avatar, rather than
            # picking the smallest single mover (which would grab only a fragment, e.g. a sprite's
            # head, fragmenting orientation/attribute reads).
            from collections import Counter
            vote = Counter(movers.values())
            best_delta, _ = vote.most_common(1)[0]
            # Colors whose ENTIRE footprint translated by best_delta (rigid global movers). A
            # multi-color sprite's body color often FAILS this test, because that color also paints
            # static decor/goals that stay put — so the global color set isn't a rigid translation.
            rigid_cols = {int(c) for c, d in movers.items() if d == best_delta}
            self._deltas[action] = best_delta
            # Recover the FULL avatar footprint: the connected (8-neighbour) cluster of non-background
            # cells in the CURRENT frame that contains a rigid mover. A multi-color sprite's body
            # color often FAILS the global per-color rigid test (it also paints static decor/goals),
            # so we grow OUT from the rigid-mover cells through the contiguous sprite. To avoid the
            # cluster bleeding into a large maze/background FILL the moving sprite merely passes over,
            # we exclude dominant-area colors from the walkable set.
            avatar_cols = set(rigid_cols)
            cap = 256  # generous sprite-size ceiling; dominant fills (maze interior) are far larger
            non_dom = {int(c) for c in np.unique(grid)
                       if int(c) != int(self._bg) and int(np.count_nonzero(grid == c)) <= cap}
            walkable = {(int(r), int(c)) for r, c in np.argwhere(np.isin(grid, list(non_dom)))} \
                if non_dom else set()
            seed = {(int(r), int(c)) for r, c in np.argwhere(np.isin(grid, list(rigid_cols)))}
            seed &= walkable
            cluster = set()
            if seed:
                cluster = self._grow_within(walkable, seed)
                for cell in cluster:
                    avatar_cols.add(int(grid[cell]))
            avatar_cols.discard(int(self._bg))  # never let the background join the avatar
            self._agent_colors |= avatar_cols
            # Anchor DIRECTLY to the cluster that actually moved this step — NOT to the nearest
            # same-color cluster (which can snap onto a static same-colored decoration the avatar
            # passes near, desyncing the tracked agent from the real one). This keeps _agent_cells
            # locked onto the genuine moving sprite.
            if cluster:
                cy = sum(r for r, _ in cluster) / len(cluster)
                cx = sum(c for _, c in cluster) / len(cluster)
                self._agent_anchor = (cy, cx)
            # primary color (for single-color callers) = the cleanest rigid mover (smallest footprint
            # among rigid_cols tends to be the unique head color), else largest avatar color.
            self._agent_color = (min(rigid_cols, key=lambda c: int(np.count_nonzero(grid == c)))
                                 if rigid_cols else max(
                avatar_cols, key=lambda c: int(np.count_nonzero(grid == c))))
            return False
        # no motion: if we knew where the agent was, the cell it tried to enter is a wall.
        # This fires on EVERY step (probe OR execute) where a known simple action produced no
        # agent displacement, so the maze structure is discovered as the agent traverses it.
        if self._agent_color is not None and action in self._deltas:
            pos = self._agent_pos(self._prev_grid)
            if pos is not None:
                dr, dc = self._deltas[action]
                w = (pos[0] + dr, pos[1] + dc)
                if w not in self._walls:
                    self._walls.add(w)
                    self._wall_order.append(w)
                return True  # confirmed wall bump
        return False

    def _ingest_transform(self, grid):
        """If the agent's attribute vector changed this step, record an on-enter triple.

        entered_color is the color of the cell the agent MOVED ONTO, read from the PREVIOUS
        frame (self._prev_grid) at the agent's current (destination) representative position.
        This is the tile color that caused the transformation, NOT the agent's old color.
        """
        if self._agent_color is None:
            return
        cells = self._agent_cells(grid)
        attr = agent_attributes(grid, cells)
        # Sentinel guard: if the agent isn't located this frame, agent_attributes returns
        # AttrVec(color=-1, shape_sig=()). Don't record a triple or pollute _prev_attr with it.
        if not cells or attr.color == -1:
            return
        prev = self._prev_attr
        self._prev_attr = attr
        if prev is None:
            return

        # Determine what tile the agent stepped onto: the color at the agent's current
        # position in the PREVIOUS frame.  Fall back to None if prev_grid is unavailable
        # or the position is out of bounds.
        dest_pos = self._agent_pos(grid)
        tile_color: int | None = None
        if dest_pos is not None and self._prev_grid is not None:
            dr, dc = dest_pos
            h, w = self._prev_grid.shape
            if 0 <= dr < h and 0 <= dc < w:
                tile_color = int(self._prev_grid[dr, dc])

        if attr.color != prev.color:
            # Only record if we have a valid tile color; skip rather than log garbage.
            if tile_color is not None:
                self._triples.append({
                    "entered_color": tile_color,
                    "attr": "color", "before": prev.color, "after": attr.color,
                })
        if attr.shape_sig != prev.shape_sig:
            if tile_color is not None:
                self._triples.append({
                    "entered_color": tile_color,
                    "attr": "shape", "before": prev.shape_sig, "after": attr.shape_sig,
                })

    def _ingest_world_delta(self, grid):
        """Record cells the agent's own movement transformed this step (paint / collect signal).

        Fix 3 — TRAIL-ONLY paint induction. A recolor/collect rule should be induced only from the
        agent's OWN trail — cells the agent just vacated or entered (or immediately adjacent to that
        path) — NOT arbitrary world cells or static structure. ls20's color-3 maze wall is present
        at frame 0 and the agent never successfully occupies it, so it must be classified as wall,
        not paint; only the cells the agent actually steps on/off form its painted trail. We
        therefore restrict observations to changes co-located with the agent's just-vacated/entered
        footprint (and its 4-neighborhood, to catch a paint that lands on the cell behind the move).
        Changes ON the agent's current cell or that involve the agent color are still skipped (that
        is the avatar itself, not a world transform). vanished == cell became background.
        """
        if self._bg is None or self._prev_grid is None or grid.shape != self._prev_grid.shape:
            return
        cur_cells = set(self._agent_cells(grid))
        prev_cells = set(self._agent_cells(self._prev_grid))
        # Trail = cells the agent occupied last frame and no longer occupies (just vacated), plus a
        # 1-cell halo around the agent's path. A painted trail manifests at the vacated cell.
        trail = (prev_cells - cur_cells)
        halo: set = set()
        for r, c in (prev_cells | cur_cells):
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                halo.add((r + dr, c + dc))
        considered = (trail | halo) - cur_cells  # never the agent's current footprint
        changed = np.argwhere(grid != self._prev_grid)
        for r, c in changed:
            cell = (int(r), int(c))
            if cell not in considered:
                continue  # only the agent's own trail counts as paint/collect evidence
            f = int(self._prev_grid[r, c]); t = int(grid[r, c])
            # Skip ANY transition touching an avatar color: a multi-color sprite vacating/entering a
            # cell flips that cell between its own colors and the background (head<->body<->bg). That
            # is the avatar moving, NOT a world recolor/paint — counting it hallucinates a paint
            # mechanic (ls20: the color-9 body over color-3 background looks like a 9<->3 "recolor").
            avatar = self._avatar_colors()
            if f in avatar or t in avatar:
                continue
            self._world_obs.append({"from_color": f, "to_color": t, "vanished": t == self._bg})

    # ------------------------------------------------------------------ phase machine
    def _decide_inner(self, grid, available):
        if self._bg is None:
            self._bg = P.detect_background(grid)
        simple = self._simple_actions(available)

        # Fix 1 — learn walls from EVERY step, including while EXECUTING a plan. If the previous
        # simple action produced no agent displacement, that cell is a wall; if we were executing
        # a plan when the move failed, the plan is routing through newly-discovered maze structure,
        # so abandon it and drop to REFINE to re-route with the wall now known.
        bumped = self._ingest_movement(grid)
        if bumped and self._phase == "EXECUTE":
            self._plan = []
            # a failed planned move is genuine new evidence: don't blacklist the goal (the route
            # was wrong, not the target). Clear _goal_pos so REFINE's blacklist add is skipped,
            # and reset the induce guard so we re-plan around the freshly-found wall.
            self._goal_pos = None
            self._last_induced_len = (-1, -1)
            self._phase = "REFINE"

        # PROBE_MOVEMENT: try each simple action once, fitting deltas/walls from the transition.
        if self._phase == "PROBE_MOVEMENT":
            if (not self._probe_queue and not self._deltas_fully_probed(simple)
                    and self._probe_rounds < _MAX_PROBE_ROUNDS):
                self._probe_queue = list(simple)
                self._probe_rounds += 1
            if self._probe_queue:
                return ("S", self._probe_queue.pop(0))
            self._phase = "PROBE_TRANSFORMS"
            self._prev_attr = agent_attributes(grid, self._agent_cells(grid)) if self._agent_color else None

        # PROBE_TRANSFORMS: graph-scaffolded coverage; log attribute-change triples on entry.
        # (Walls/deltas are refined by the top-of-call _ingest_movement above.)
        if self._phase == "PROBE_TRANSFORMS":
            self._ingest_transform(grid)
            self._ingest_world_delta(grid)
            self._transform_steps += 1
            h = P.state_hash(grid)
            covered = h in self._seen_hashes
            self._seen_hashes.add(h)
            if self._transform_steps < _MAX_TRANSFORM_STEPS and (not covered or self._transform_steps < len(simple) * 4):
                a = self._next_coverage_action(grid, simple)
                if a is not None:
                    return ("S", a)
            self._phase = "INDUCE"

        # REFINE: log the surprise (the state where the plan ran out) and re-induce. Placed
        # BEFORE INDUCE so REFINE -> INDUCE -> PLAN -> EXECUTE all run within this one decide()
        # and EXECUTE can emit the first replanned action — REFINE costs no wasted, un-ingested
        # action (Fix E).
        if self._phase == "REFINE":
            self._ingest_transform(grid)  # capture any unexpected attribute change as a new triple
            # A plan that reached its goal candidate WITHOUT a level-up means that candidate is not
            # the real goal (or not the right attr): blacklist it so the disjunctive sweep advances
            # to the NEXT candidate instead of re-selecting the same nearest one (Exp-46).
            if self._goal_pos is not None:
                self._tried_goals.add(self._goal_pos)
                self._goal_pos = None
            self._phase = "INDUCE"

        # INDUCE: assemble the factored model from the fitted grammar + discovered layout.
        if self._phase == "INDUCE":
            # Guard against re-inducing every call when nothing changed (Fix F): if the last
            # induction already failed to yield a plan and no new evidence has arrived since,
            # don't loop back through INDUCE — emit a coverage/fallback action to gather more.
            # The guard key includes the tried-goal count: when a candidate was just blacklisted in
            # REFINE, that IS new planning evidence (the sweep will now try a DIFFERENT candidate),
            # so don't short-circuit — fall through and re-PLAN against the remaining candidates.
            ev = (len(self._triples), len(self._world_obs), len(self._tried_goals))
            if not self._plan and ev == self._last_induced_len:
                if simple:
                    return ("S", self._next_coverage_action(grid, simple) or
                            simple[self._transform_steps % len(simple)])
                return ("S", available[0] if available else 1)
            self._build_model(grid)
            self._last_induced_len = ev
            self._phase = "PLAN"

        # PLAN: enumerate candidate slot attr-specs, BFS-plan, cache the action list.
        if self._phase == "PLAN":
            self._build_plan(grid)
            self._phase = "EXECUTE" if self._plan else "REFINE"

        # EXECUTE: emit one cached action per call; exhaustion without level-up -> REFINE.
        if self._phase == "EXECUTE":
            if self._plan:
                return ("S", self._plan.pop(0))
            self._phase = "REFINE"

        # Fallback: keep moving (and discovering) rather than stalling.
        if self._plan:
            return ("S", self._plan.pop(0))
        if simple:
            return ("S", simple[self._transform_steps % len(simple)])
        return ("S", available[0] if available else 1)

    def _deltas_fully_probed(self, simple) -> bool:
        return all(a in self._deltas for a in simple) and bool(simple)

    def _next_coverage_action(self, grid, simple):
        """Pick an action that advances exploration. First preference: a one-step move into a cell
        that is neither a known wall nor already visited. If every immediate neighbour is walled or
        visited (the agent is in an explored pocket), FRONTIER-SEEK: BFS over the known-passable
        lattice to the nearest UNVISITED cell and step toward it — this escapes local pockets so the
        agent can reach distant transformer tiles/goals it would never hit by local round-robin.
        Falls back to round-robin if no frontier is reachable. General; no game constants."""
        if not simple:
            return None
        pos = self._agent_pos(grid)
        if pos is None or not self._deltas:
            return simple[self._transform_steps % len(simple)]
        self._visited.add(pos)
        # 0) BOXED-IN ESCAPE. If EVERY immediate neighbour is a known wall, the agent has trapped
        # itself — but a solvable level never fully boxes the avatar, so some of those "walls" are
        # not static (e.g. an orientation/attribute-GATED cell that bumped while mis-oriented, or a
        # mis-registered bump). Forget the walls immediately around this position so the agent can
        # re-probe and leave the pocket. General: triggered purely by the impossible all-walled state.
        neigh = []
        for a in simple:
            if a in self._deltas:
                dr, dc = self._deltas[a]
                neigh.append((a, (pos[0] + dr, pos[1] + dc)))
        if neigh and all(n in self._walls for _, n in neigh):
            for _, n in neigh:
                self._walls.discard(n)
                if n in self._wall_order:
                    self._wall_order.remove(n)
            # re-probe a freshly-unblocked direction this very step
            return neigh[self._transform_steps % len(neigh)][0]
        # 1) immediate unvisited, unwalled neighbour
        for a in simple:
            if a not in self._deltas:
                return a
            dr, dc = self._deltas[a]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt not in self._walls and nxt not in self._visited:
                return a
        # 2) frontier BFS: route to the nearest unvisited, non-wall lattice cell.
        a = self._frontier_step(grid, pos, simple)
        if a is not None:
            return a
        # 3) any unwalled neighbour (revisit is OK to traverse toward unexplored regions)
        for a in simple:
            dr, dc = self._deltas.get(a, (0, 0))
            if (pos[0] + dr, pos[1] + dc) not in self._walls:
                return a
        return simple[self._transform_steps % len(simple)]

    def _frontier_step(self, grid, pos, simple):
        """BFS over the known-passable lattice (4-neighbour via _deltas) from `pos` to the nearest
        cell not in _visited; return the first action along that path, or None if no frontier."""
        from collections import deque
        H, W = grid.shape
        act_deltas = [(a, self._deltas[a]) for a in simple if a in self._deltas]
        if not act_deltas:
            return None
        q = deque([(pos, None)])
        seen = {pos}
        while q:
            cur, first = q.popleft()
            for a, (dr, dc) in act_deltas:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < H and 0 <= nxt[1] < W):
                    continue
                if nxt in self._walls or nxt in seen:
                    continue
                fa = a if first is None else first
                if nxt not in self._visited:
                    return fa  # reached a frontier cell; step the first action of the route
                seen.add(nxt)
                q.append((nxt, fa))
        return None

    def _build_model(self, grid):
        """Build InducedModel: deltas + on_enter cycles + walls + terminal predicate."""
        # Induce cycles ONCE and cache them, so the tile map and the attainable-value
        # enumeration (which reuses self._cycles) can't diverge.
        self._cycles = induce_on_enter_cycles(self._triples)
        # tiles: map every cell currently showing a cycling tile-color to its OnEnterCycle.
        self._tiles = {}
        cycle_by_color = {c.tile_color: c for c in self._cycles}
        if cycle_by_color and self._bg is not None:
            for o in P.connected_components(grid, background=self._bg):
                cyc = cycle_by_color.get(int(o.color))
                if cyc is not None:
                    for cell in o.cells:
                        self._tiles[cell] = cyc
        self._terminal = induce_terminal({})
        # World-transform rules (Task 7): paint (recolor-on-move) + collect (vanish-on-contact),
        # induced from the world-delta observations gathered during PROBE_TRANSFORMS.
        self._recolors = induce_recolor_on_move(self._world_obs)
        self._collects = induce_collect_on_contact(self._world_obs)
        self._paint_colors = {r.from_color for r in self._recolors}

    def _attainable_attr_values(self):
        """The small set of attribute values the agent can take, from observed cycle orders."""
        vals = {"color": set(), "shape": set()}
        for c in self._cycles:  # reuse the cached induction (see _build_model)
            vals.setdefault(c.attribute, set()).update(c.order)
        return vals

    def _build_plan(self, grid):
        """Read candidate goal slots from the scene graph, enumerate attr-specs, BFS-plan."""
        self._plan = []
        if self._bg is None or not self._deltas:
            return
        H, W = grid.shape
        start_pos = self._agent_pos(grid)
        if start_pos is None:
            return
        attr = agent_attributes(grid, self._agent_cells(grid))
        start_attrs = {"color": attr.color, "shape": attr.shape_sig}

        scene = SG.extract(grid, self._bg)
        cands = scene.get("target_candidates", [])
        # candidate goal positions: rounded centroids of framed/rare-static targets
        raw_positions = []
        for t in cands:
            cy, cx = t["centroid"]
            raw_positions.append((int(round(cy)), int(round(cx))))
        # Broadened goal extraction (Fix 4). The framed scene `target_candidates` sometimes MISS the
        # real goal cell (ls20 L1's goal is not framed). So ALSO consider small, distinct-color,
        # non-agent, non-wall objects as candidate goals and ADD them to the disjunction sweep — the
        # nearest-first sweep tries each candidate and the env tells us (via level-up) which is real.
        # "Distinct" = a rare color (small total pixel footprint, not the dominant maze/wall colour);
        # objects sitting on confirmed-wall cells are skipped. General: size/rarity, no colour const.
        # Nearest-first ordering keeps framed scene targets preferred when they ARE present, so games
        # with correct framed targets are unaffected.
        extra_positions: list = []
        comps = P.connected_components(grid, background=self._bg)
        # per-color total footprint, to gauge distinctness/rarity (dominant colors are structure).
        color_area: dict = {}
        for o in comps:
            color_area[int(o.color)] = color_area.get(int(o.color), 0) + int(o.size)
        for o in comps:
            col = int(o.color)
            if col == int(self._agent_color):
                continue
            if o.size > _MAX_TARGET_OBJ_SIZE:
                continue
            # a goal object should be small AND a rare color (not the bulk maze/wall material).
            if color_area.get(col, 0) > _MAX_TARGET_OBJ_SIZE:
                continue
            if set(o.cells) & self._walls:   # don't aim at a confirmed wall
                continue
            cy, cx = o.centroid
            extra_positions.append((int(round(cy)), int(round(cx))))
        # de-dup while preserving order: framed scene candidates first, then broadened ones.
        raw_positions = list(dict.fromkeys(raw_positions + extra_positions))
        if not raw_positions:
            return
        # Footprint map: each non-agent, non-bg object indexed by the cells it occupies. Used so
        # that lattice-snapping doesn't land a target OFF the object (Fix 2 / Exp-45 defect 2):
        # we clamp the snapped lattice cell into the candidate object's bbox so the planned cell is
        # actually ON the goal object's footprint, not an adjacent empty cell.
        footprints = []  # (set_of_cells, bbox=(r0,c0,r1,c1))
        for o in P.connected_components(grid, background=self._bg):
            if int(o.color) == int(self._agent_color):
                continue
            footprints.append((set(o.cells), o.bbox))

        def _object_for(pos):
            """Object whose footprint contains pos, else the nearest object (by centroid)."""
            best = None; best_d = None
            for cells, bbox in footprints:
                if pos in cells:
                    return (cells, bbox)
                r0, c0, r1, c1 = bbox
                cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0
                d = abs(pos[0] - cy) + abs(pos[1] - cx)
                if best_d is None or d < best_d:
                    best_d, best = d, (cells, bbox)
            return best

        # Snap targets to the agent's motion lattice (pitch from _deltas, origin = start_pos) so
        # they are reachable by the planner's pitch-sized steps (Phase-3 fix for Exp-44). Then
        # footprint-align: if the snap landed off the candidate object, clamp the snapped cell into
        # the object's bbox so the planned cell sits ON the goal footprint (Fix 2).
        pitch = lattice_pitch(self._deltas)
        # clamp tolerance: a snap is "off the object" only if the candidate object is within ~1
        # lattice pitch of the centroid — otherwise we'd drag a centroid onto an unrelated far object.
        tol = max(pitch[0], pitch[1], 1)

        def _lattice_cells_in_bbox(bbox):
            """All motion-lattice cells (anchored at start_pos, given pitch) inside bbox, clamped
            to the grid. Used to find a target cell that is BOTH on the object footprint AND
            reachable by the agent's pitch-sized steps."""
            r0, c0, r1, c1 = bbox
            pr, pc = pitch
            sr, sc = start_pos
            rows = [sr] if pr == 0 else range(sr - ((sr - r0) // pr + 1) * pr, r1 + pr, pr)
            cols = [sc] if pc == 0 else range(sc - ((sc - c0) // pc + 1) * pc, c1 + pc, pc)
            out = []
            for rr in rows:
                if r0 <= rr <= r1 and 0 <= rr < H:
                    for cc in cols:
                        if c0 <= cc <= c1 and 0 <= cc < W:
                            out.append((int(rr), int(cc)))
            return out

        snapped = []
        for p in raw_positions:
            s = snap(p, start_pos, pitch, (H, W))
            obj = _object_for(p)
            if obj is not None and s not in obj[0]:
                r0, c0, r1, c1 = obj[1]
                near = (r0 - tol <= p[0] <= r1 + tol) and (c0 - tol <= p[1] <= c1 + tol)
                in_bbox = r0 <= s[0] <= r1 and c0 <= s[1] <= c1
                if near and not in_bbox:
                    # Fix 2: align onto the object footprint, but ONLY to a cell that is ALSO on the
                    # motion lattice (else the target is on the object yet unreachable). Pick the
                    # in-bbox lattice cell nearest the centroid; if NONE exists, keep the plain
                    # lattice snap `s` (reachable) rather than an off-lattice bbox cell (unreachable).
                    lat = _lattice_cells_in_bbox(obj[1])
                    if lat:
                        s = min(lat, key=lambda q: abs(q[0] - p[0]) + abs(q[1] - p[1]))
            snapped.append(s)
        slot_positions = list(dict.fromkeys(snapped))

        # Paintable cells (Task 7): cells whose color is a known paint `from_color`, EXCEPT any
        # cell the agent confirmed is a wall (Fix 2 below). Under A1 a paintable cell that is not a
        # confirmed wall is passable (not added to true_walls); under A2 it must be painted to be
        # traversed. Computed from the same connected-components view used everywhere else.
        paintable_cells: set = set()
        if self._paint_colors and self._bg is not None:
            for o in P.connected_components(grid, background=self._bg):
                if int(o.color) in self._paint_colors:
                    paintable_cells.update(o.cells)
        # Fix 2 — WALLS OVERRIDE PAINT. A cell the agent confirmed impassable (bumped) stays a wall
        # even if its color matches a paint from_color: static maze structure (e.g. ls20's color-3
        # wall) is often the same color as the agent's painted trail, but a confirmed bump proves
        # it is solid. So keep ALL confirmed walls in true_walls, and strip them from paintable so
        # the planner can't try to "paint through" a known wall.
        paintable_cells -= self._walls
        true_walls = set(self._walls)

        # Enumerate single-attribute requirements per slot over the small attainable set.
        # Our attribute vector is AttrVec(color, shape_sig); shape_sig is rotation-SENSITIVE,
        # so an orientation goal (e.g. ls20 level 1) manifests as a target "shape" value, NOT a
        # color. We therefore try reach-only first, then each attainable color value, then each
        # attainable shape value. Keys ("color"/"shape") match FactoredState.attrs, the cycle
        # `attribute` field, and InducedModel.step's tile.attribute, so the goal is reachable
        # end-to-end. Bounded (no cross-product) so the black-box game can't blow up the search.
        attainable = self._attainable_attr_values()
        reqs: list[dict] = [{}]  # reach-only
        for v in sorted(attainable.get("color", set())):
            reqs.append({"color": v})
        for v in sorted(attainable.get("shape", set())):
            reqs.append({"shape": v})
        # A1 (traversal): InducedModel with paintable cells removed from walls (passable);
        # the existing path-BFS plan() satisfies the slots.
        # A2 (painted_set): plan_painted_set uses start_pos/start_attrs directly — no model/start.
        # Build model and start only for A1 to avoid constructing an unused InducedModel for A2.
        if self._backend != "painted_set":
            model = InducedModel(deltas=self._deltas, walls=true_walls, tiles=self._tiles,
                                 width=W, height=H, terminal=self._terminal)
            start = FactoredState(pos=start_pos, attrs=start_attrs, completed=frozenset())

        def _try(slots):
            """Plan to satisfy `slots`; return a non-empty action list or None."""
            if self._backend == "painted_set":
                # A2: paintable cells must be painted (entered) to be traversed.
                actions, status = plan_painted_set(
                    deltas=self._deltas, true_walls=true_walls,
                    paintable_cells=paintable_cells, tiles=self._tiles,
                    width=W, height=H, start_pos=start_pos, start_attrs=start_attrs,
                    slots=slots, max_paints=None, max_nodes=200_000,
                )
                return list(actions) if (status == "solved" and actions) else None
            actions = plan(model, start, slots)
            return list(actions) if actions else None

        # Fix 1 — DISJUNCTION over candidates (Exp-45 ls20 blocker). A reach-target game has ONE
        # real goal slot among ~9 scene candidates; requiring ALL satisfied (a conjunction) made
        # plan() return None whenever ANY candidate was unreachable (on a wall / the agent start /
        # no path). So FIRST sweep each candidate INDIVIDUALLY, nearest-first, and accept the first
        # req x candidate that yields a non-empty plan — "reach candidate C with attr-req R". The
        # execute/REFINE loop retries other candidates if the env doesn't level-up. The full
        # all-candidates conjunction is kept as a FALLBACK below (collect's reach-all relies on it).
        ordered = sorted(slot_positions,
                         key=lambda p: abs(p[0] - start_pos[0]) + abs(p[1] - start_pos[1]))
        # Skip candidates already reached without a level-up (blacklisted in REFINE). When EVERY
        # candidate has been tried, clear the blacklist so the sweep re-attempts from the current
        # position rather than stalling forever — a reach-only candidate may level-up once we hold a
        # different attr, and the layout/attrs evolve as we move.
        untried = [p for p in ordered if p != start_pos and p not in self._tried_goals]
        if not untried:
            self._tried_goals = set()
            untried = [p for p in ordered if p != start_pos]
        for pos in untried:
            for req in reqs:
                actions = _try([{"pos": pos, "attr_req": req, "done": False}])
                if actions:
                    self._plan = actions
                    self._goal_pos = pos   # remember the goal so REFINE can blacklist it on failure
                    return

        # Fallback: the all-candidates CONJUNCTION (reach-all) — needed for collect-all, where the
        # goal genuinely is to reach EVERY small object. Tried after the single-candidate sweep so a
        # reach-ONE goal (ls20) is preferred and a poison candidate can't kill the single-target plan.
        for req in reqs:
            slots = [{"pos": pos, "attr_req": req, "done": False} for pos in slot_positions]
            actions = _try(slots)
            if actions:
                self._plan = actions
                return
