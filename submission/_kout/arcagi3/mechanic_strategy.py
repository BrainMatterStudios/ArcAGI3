"""MechanicSolverStrategy — a reactive (decide()-interface) archetype solver that slots into PortfolioPolicy
as an ADDITIVE max-over-plays play. On its play it recognizes an archetype and emits the solving action
tokens; if it does not recognize the game (or finishes), it ABSTAINS by emitting benign actions so its play
simply scores whatever it achieved (max-over-plays keeps the banked coverage play). By construction it can
only ADD levels on archetype-matching games and never regress the coverage floor.

Currently implements the PATTERN-MATCH archetype (pure clicks: place palette tiles into slots to match a
visible answer key, then submit). Reactive form of scripts/research_2026_07_01/pattern_match.py; self-contained
here so it embeds in the submission notebook.

Token format (matches my_agent.choose_action): ("reset",) | ("S", action_id) | (x, y) for ACTION6 clicks.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from arcagi3 import perception as P


def _square(o):
    w = o.bbox[3] - o.bbox[1] + 1
    h = o.bbox[2] - o.bbox[0] + 1
    return h > 0 and 0.5 <= w / h <= 2.0 and o.size >= 8


def perceive_patternmatch(grid):
    bg = P.detect_background(grid)
    objs = P.connected_components(grid, background=bg)
    H = grid.shape[0]
    tiles = sorted((o.centroid[1], o.color, o.centroid)
                   for o in objs if (o.bbox[0] + o.bbox[2]) / 2 > H * 0.78 and o.color != bg and _square(o))
    tile_colors = {c for _, c, _ in tiles}
    ak = [(o.centroid[1], o.color) for o in objs
          if (o.bbox[0] + o.bbox[2]) / 2 < H * 0.22 and o.color in tile_colors]
    ak = [c for _, c in sorted(ak)]
    mid = [(o.centroid[0], o.centroid[1], o.centroid) for o in objs
           if H * 0.30 < (o.bbox[0] + o.bbox[2]) / 2 < H * 0.62 and o.size <= 12]
    slots = []
    if mid:
        rows = Counter(round(r / 3) for r, _, _ in mid)
        best_row = rows.most_common(1)[0][0]
        slots = sorted((c, cen) for r, c, cen in mid if round(r / 3) == best_row)
    return ak, tiles, slots


class MechanicSolverStrategy:
    """Reactive archetype solver for PortfolioPolicy. Abstains (benign action) when no archetype matches."""

    def __init__(self, seed: int = 0):
        self._queue: list = []          # pending action tokens for the current level
        self._planned_level = -1        # level we built the queue for
        self._abstain = False
        self.gs = None                  # PortfolioPolicy exposes pols[idx].gs (guarded, .wm) -> None is safe

    def _benign(self, available):
        # a harmless action for abstain / when the queue is empty and nothing to do
        if 5 in available:
            return ("S", 5)
        if available:
            return ("S", int(available[0]))
        return ("reset",)

    def _build_plan(self, grid):
        ak, tiles, slots = perceive_patternmatch(grid)
        if not ak or not tiles or len(slots) < len(ak):
            return None
        tokens = []
        used = set()
        for i, target in enumerate(ak):
            ti = next((k for k, (_, col, _) in enumerate(tiles) if col == target and k not in used), None)
            if ti is None:
                return None
            used.add(ti)
            tcen = tiles[ti][2]; scen = slots[i][1]
            # click token format matches my_agent/SalienceExplorer: ("C", x, y) with x=col, y=row
            tokens.append(("C", int(round(tcen[1])), int(round(tcen[0]))))   # click tile
            tokens.append(("C", int(round(scen[1])), int(round(scen[0]))))   # click slot
        tokens.append(("S", 5))                                          # submit
        return tokens

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        if gstate_terminal:
            self._queue = []; self._planned_level = -1
            return ("reset",)
        # re-plan when we reach a new level (or first time)
        if levels != self._planned_level and not self._queue:
            self._planned_level = levels
            self._abstain = False
            plan = self._build_plan(grid)
            if plan is None:
                self._abstain = True
            else:
                self._queue = plan
        if self._queue:
            return self._queue.pop(0)
        return self._benign(available)
