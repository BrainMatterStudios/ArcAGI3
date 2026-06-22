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

import itertools

import numpy as np

from arcagi3 import perception as P, movement as Mv, scene_graph as SG
from arcagi3.attribute_state import agent_attributes
from arcagi3.transform_induction import induce_on_enter_cycles, induce_terminal
from arcagi3.factored_model import FactoredState, InducedModel, plan

# Bounded coverage so a black-box game can't trap the explorer in a phase forever.
_MAX_TRANSFORM_STEPS = 400


class DiscoveryExplorer:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset_all()

    # ------------------------------------------------------------------ lifecycle
    def reset_all(self):
        self._phase = "PROBE_MOVEMENT"
        self._deltas: dict[int, tuple[int, int]] = {}
        self._walls: set = set()
        self._tiles: dict = {}
        self._triples: list[dict] = []
        self._plan: list = []
        self._seen_hashes: set = set()
        self._last_level = 0
        self._prev_grid = None
        self._prev_token = None
        self._agent_color = None
        self._probe_queue: list = []
        self._bg = None
        self._terminal = None
        self._transform_steps = 0
        self._prev_attr = None

    def on_level_change(self, new_level: int):
        """KEEP the induced grammar (deltas/tiles/terminal); FLUSH the layout state."""
        self._last_level = new_level
        self._walls = set()
        self._plan = []
        self._seen_hashes = set()
        self._transform_steps = 0
        self._prev_attr = None
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

    def _agent_cells(self, grid):
        """Cells of the identified agent (its connected component of `self._agent_color`)."""
        if self._agent_color is None or self._bg is None:
            return ()
        objs = P.connected_components(grid, background=self._bg)
        cells: list = []
        for o in objs:
            if int(o.color) == int(self._agent_color):
                cells.extend(o.cells)
        return tuple(cells)

    def _agent_pos(self, grid):
        cells = self._agent_cells(grid)
        if not cells:
            return None
        # representative position = the (min-row, min-col) anchor of the agent cell-set
        return (min(r for r, _ in cells), min(c for _, c in cells))

    def _ingest_movement(self, grid):
        """From prev_grid -> grid, find the moving color (the agent) and fit the action's delta.

        A simple action that produced NO motion marks the attempted direction as a candidate
        wall in front of the agent.
        """
        if self._prev_grid is None or self._prev_token is None:
            return
        if self._prev_token[0] != "S":
            return
        action = self._prev_token[1]
        movers = Mv.infer_all_translations(self._prev_grid, grid, self._bg)
        if movers:
            # smallest mover = agent (decorations/counters move as a block but rarely smallest)
            color = min(movers, key=lambda c: int(np.count_nonzero(self._prev_grid == c)))
            self._agent_color = int(color)
            self._deltas[action] = movers[color]
        else:
            # no motion: if we knew where the agent was, the cell it tried to enter is a wall
            if self._agent_color is not None and action in self._deltas:
                pos = self._agent_pos(self._prev_grid)
                if pos is not None:
                    dr, dc = self._deltas[action]
                    self._walls.add((pos[0] + dr, pos[1] + dc))

    def _ingest_transform(self, grid):
        """If the agent's attribute vector changed this step, record an on-enter triple."""
        if self._agent_color is None:
            return
        attr = agent_attributes(grid, self._agent_cells(grid))
        prev = self._prev_attr
        self._prev_attr = attr
        if prev is None:
            return
        if attr.color != prev.color:
            self._triples.append({
                "entered_color": int(prev.color),  # color cycles -> tile color is what we left
                "attr": "color", "before": prev.color, "after": attr.color,
            })
        if attr.shape_sig != prev.shape_sig:
            self._triples.append({
                "entered_color": int(attr.color),
                "attr": "shape", "before": prev.shape_sig, "after": attr.shape_sig,
            })

    # ------------------------------------------------------------------ phase machine
    def _decide_inner(self, grid, available):
        if self._bg is None:
            self._bg = P.detect_background(grid)
        simple = self._simple_actions(available)

        # PROBE_MOVEMENT: try each simple action once, fitting deltas/walls from the transition.
        if self._phase == "PROBE_MOVEMENT":
            self._ingest_movement(grid)
            if not self._probe_queue and not self._deltas_fully_probed(simple):
                self._probe_queue = list(simple)
            if self._probe_queue:
                return ("S", self._probe_queue.pop(0))
            self._phase = "PROBE_TRANSFORMS"
            self._prev_attr = agent_attributes(grid, self._agent_cells(grid)) if self._agent_color else None

        # PROBE_TRANSFORMS: graph-scaffolded coverage; log attribute-change triples on entry.
        if self._phase == "PROBE_TRANSFORMS":
            self._ingest_movement(grid)   # keep refining walls/deltas while wandering
            self._ingest_transform(grid)
            self._transform_steps += 1
            h = P.state_hash(grid)
            covered = h in self._seen_hashes
            self._seen_hashes.add(h)
            if self._transform_steps < _MAX_TRANSFORM_STEPS and (not covered or self._transform_steps < len(simple) * 4):
                a = self._next_coverage_action(grid, simple)
                if a is not None:
                    return ("S", a)
            self._phase = "INDUCE"

        # INDUCE: assemble the factored model from the fitted grammar + discovered layout.
        if self._phase == "INDUCE":
            self._build_model(grid)
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

        # REFINE: log the surprise (the state where the plan ran out) and re-induce.
        if self._phase == "REFINE":
            self._ingest_transform(grid)  # capture any unexpected attribute change as a new triple
            self._phase = "INDUCE"
            # fall through to a safe action this step; next call re-enters INDUCE

        # Fallback: keep moving (and discovering) rather than stalling.
        if self._plan:
            return ("S", self._plan.pop(0))
        if simple:
            return ("S", simple[self._transform_steps % len(simple)])
        return ("S", available[0] if available else 1)

    def _deltas_fully_probed(self, simple) -> bool:
        return all(a in self._deltas for a in simple) and bool(simple)

    def _next_coverage_action(self, grid, simple):
        """Prefer an action whose resulting cell is unseen/unblocked; else round-robin."""
        if not simple:
            return None
        pos = self._agent_pos(grid)
        if pos is not None and self._deltas:
            # try actions that lead to a not-yet-walled cell, biasing toward exploration
            for a in simple:
                if a not in self._deltas:
                    return a
                dr, dc = self._deltas[a]
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt not in self._walls:
                    return a
        return simple[self._transform_steps % len(simple)]

    def _build_model(self, grid):
        """Build InducedModel: deltas + on_enter cycles + walls + terminal predicate."""
        cycles = induce_on_enter_cycles(self._triples)
        # tiles: map every cell currently showing a cycling tile-color to its OnEnterCycle.
        self._tiles = {}
        cycle_by_color = {c.tile_color: c for c in cycles}
        if cycle_by_color and self._bg is not None:
            for o in P.connected_components(grid, background=self._bg):
                cyc = cycle_by_color.get(int(o.color))
                if cyc is not None:
                    for cell in o.cells:
                        self._tiles[cell] = cyc
        self._terminal = induce_terminal({})

    def _attainable_attr_values(self):
        """The small set of attribute values the agent can take, from observed cycle orders."""
        vals = {"color": set(), "shape": set()}
        for c in induce_on_enter_cycles(self._triples):
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
        slot_positions = []
        for t in cands:
            cy, cx = t["centroid"]
            slot_positions.append((int(round(cy)), int(round(cx))))
        if not slot_positions:
            return

        model = InducedModel(deltas=self._deltas, walls=self._walls, tiles=self._tiles,
                             width=W, height=H, terminal=self._terminal)
        start = FactoredState(pos=start_pos, attrs=start_attrs, completed=frozenset())

        # Enumerate attribute requirements per slot over the small attainable set. The simplest
        # spec (no attr requirement = just reach the slot) is tried first; then single-attr reqs.
        attainable = self._attainable_attr_values()
        color_opts = [None] + sorted(attainable.get("color", set()))
        for color_req in color_opts:
            slots = []
            for pos in slot_positions:
                req = {} if color_req is None else {"color": color_req}
                slots.append({"pos": pos, "attr_req": req, "done": False})
            actions = plan(model, start, slots)
            if actions:
                self._plan = list(actions)
                return
