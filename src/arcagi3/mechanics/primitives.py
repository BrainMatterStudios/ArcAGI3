"""Executable mechanic primitives + goal hypotheses (winning/mechanic-model-search, Phase 2).

Two kinds of object:
  * TRANSITION primitives — how the world changes on an action. Each can fit() its parameters from
    observed transitions and apply() a predicted cell-level change. Movement + paint + collect reuse
    the validated inducers in transform_induction; push is new (the wa30/sokoban gap from Phase 1).
  * GOAL hypotheses — candidate win-conditions, treated as HYPOTHESES not truth (the model beam
    plans to each; the env's level-up confirms the real one). satisfied()/target_cells() read the
    CURRENT frame, so a moving goal (ls20's color-9) is handled by re-reading + replanning, not by
    modelling its dynamics.

State for prediction/scoring is the raw 64x64 palette grid; planning uses a compact factored state
(see world_model.py). Everything is general — no per-game constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arcagi3.transform_induction import (
    CollectOnContact, RecolorOnMove, induce_collect_on_contact, induce_recolor_on_move,
)


# --------------------------------------------------------------------- transition primitives
@dataclass
class MoveAgent:
    """Agent translates by a per-action delta unless the target is a wall/edge (blocked -> stay)."""
    deltas: dict = field(default_factory=dict)   # action_id -> (dr, dc)
    walls: set = field(default_factory=set)

    def num_params(self):
        return len(self.deltas) + len(self.walls)


@dataclass
class PaintOnMove:
    """Entering a cell of from_color repaints it to_color (ls20 trail / path-opening)."""
    rules: list = field(default_factory=list)    # list[RecolorOnMove]

    def map(self):
        return {r.from_color: r.to_color for r in self.rules}

    def num_params(self):
        return len(self.rules)


@dataclass
class CollectContact:
    """Contacting a cell of `color` removes it (vanish -> background)."""
    colors: set = field(default_factory=set)

    def num_params(self):
        return len(self.colors)


@dataclass
class PushObject:
    """A rigid non-agent blob of `block_color` translates one cell in the agent's move direction
    when the agent moves into it (sokoban). Fitted from rigid-translation evidence."""
    block_colors: set = field(default_factory=set)

    def num_params(self):
        return len(self.block_colors)


def fit_push(transitions, agent_colors, bg):
    """Detect block colors that rigidly translate by a small vector in correlation with agent moves."""
    blocks = set()
    for prev, _aid, grid in transitions:
        if prev.shape != grid.shape:
            continue
        for col in (set(np.unique(prev)) & set(np.unique(grid))) - set(agent_colors) - {bg}:
            a = np.argwhere(prev == col); b = np.argwhere(grid == col)
            if len(a) == len(b) >= 3:
                shift = b.min(0) - a.min(0)
                if 0 < int(abs(shift).sum()) <= 2 and np.array_equal(
                        np.sort(a - a.min(0), 0), np.sort(b - b.min(0), 0)):
                    blocks.add(int(col))
    return PushObject(block_colors=blocks)


# --------------------------------------------------------------------- goal hypotheses
class Goal:
    """Base goal hypothesis. target_cells/satisfied read the CURRENT grid (handles moving goals)."""
    kind = "goal"

    def target_cells(self, grid, agent_colors):
        return []

    def satisfied(self, grid, agent_pos, agent_colors):
        return False


@dataclass
class ReachColor(Goal):
    color: int
    kind = "reach_color"

    def target_cells(self, grid, agent_colors):
        return [tuple(p) for p in np.argwhere(grid == self.color)]

    def satisfied(self, grid, agent_pos, agent_colors):
        if agent_pos is None:
            return False
        r, c = agent_pos
        H, W = grid.shape
        return any(0 <= r + dr < H and 0 <= c + dc < W and grid[r + dr, c + dc] == self.color
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


@dataclass
class CollectAll(Goal):
    color: int
    kind = "collect_all"

    def target_cells(self, grid, agent_colors):
        return [tuple(p) for p in np.argwhere(grid == self.color)]

    def satisfied(self, grid, agent_pos, agent_colors):
        return not np.any(grid == self.color)


@dataclass
class SymmetrySatisfied(Goal):
    axis: str = "v"          # "v" mirror left/right, "h" mirror top/bottom
    kind = "symmetry"

    def _err(self, grid):
        m = np.fliplr(grid) if self.axis == "v" else np.flipud(grid)
        return float(np.mean(grid != m))

    def target_cells(self, grid, agent_colors):
        return []

    def satisfied(self, grid, agent_pos, agent_colors):
        return self._err(grid) < 0.02


def enumerate_goals(grid, agent_colors, bg, objects_colors):
    """Generate candidate goal hypotheses from the scene (colors present, small rare objects)."""
    goals: list[Goal] = []
    present = [int(c) for c in np.unique(grid) if int(c) not in agent_colors and int(c) != bg]
    counts = {c: int(np.count_nonzero(grid == c)) for c in present}
    # reach-color: rarest non-bg colors are the most goal-like markers
    for c in sorted(present, key=lambda c: counts[c])[:6]:
        goals.append(ReachColor(c))
    # collect-all: colors that appear as several small same-size blobs read as collectibles
    for c in present:
        if 2 <= counts[c] <= 60:
            goals.append(CollectAll(c))
    goals.append(SymmetrySatisfied("v"))
    goals.append(SymmetrySatisfied("h"))
    # de-dup by (kind, param)
    seen, out = set(), []
    for g in goals:
        key = (g.kind, getattr(g, "color", getattr(g, "axis", None)))
        if key not in seen:
            seen.add(key); out.append(g)
    return out
