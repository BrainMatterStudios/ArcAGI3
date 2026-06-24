"""Goal induction from the reward/level-up event (winning branch — the unexploited signal).

Every prior approach GUESSED the goal from the scene, and that is the wall. The environment hands
you a free, bespoke-proof label of the goal every time you level up. This module LEARNS the goal
PREDICATE from the winning transition (contrast it against ordinary states), so the next level can
be cleared by planning to satisfy the SAME learned predicate with the accurate transition model —
closing the loop Exp 40 opened (ls20 goal = reach color-9, recovered from the level-up) but never
transferred. The score is level-weighted and levels repeat structure, so the one expensive first
win unlocks cheap deep levels.
"""
from __future__ import annotations

import numpy as np

from arcagi3.mechanics.primitives import CollectAll, ReachColor, SymmetrySatisfied


def _adjacent_colors(grid, pos):
    if pos is None:
        return set()
    H, W = grid.shape
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < H and 0 <= c < W:
                out.add(int(grid[r, c]))
    return out


class RewardGoalLearner:
    """Induce a transferable goal predicate from the first level-up event."""

    def __init__(self):
        self.learned = None
        self._recent: list = []       # (grid, agent_pos) baseline of ORDINARY states

    def note(self, grid, agent_pos):
        self._recent.append((grid, agent_pos))
        if len(self._recent) > 60:
            self._recent.pop(0)

    def _contrast(self, goal, agent_colors):
        """How DISTINCTIVE the goal is: rarely satisfied in ordinary recent states (0 = generic)."""
        if not self._recent:
            return 1.0
        hits = sum(goal.satisfied(g, ap, agent_colors) for g, ap in self._recent)
        return 1.0 - hits / len(self._recent)

    def on_levelup(self, prev_grid, win_grid, agent_colors, bg, win_pos):
        """Contrast the winning transition vs ordinary states; pick the most distinctive predicate."""
        cands = []
        # ReachColor: a color the agent is adjacent to AT the win that is non-bg, non-agent
        for c in _adjacent_colors(win_grid, win_pos) - set(agent_colors) - {bg}:
            cands.append(ReachColor(int(c)))
        # CollectAll: a color that was present before and is now gone (all collected)
        for c in set(np.unique(prev_grid)) - set(np.unique(win_grid)):
            if int(c) not in agent_colors and int(c) != bg:
                cands.append(CollectAll(int(c)))
        # SymmetrySatisfied: only a SUBSTANTIVE symmetry (not a trivially-symmetric near-empty board,
        # which would spuriously fire at every collect/clear level-up)
        nonbg = float(np.mean(win_grid != bg))
        if nonbg > 0.15:
            for axis in ("v", "h"):
                g = SymmetrySatisfied(axis)
                if g.satisfied(win_grid, win_pos, agent_colors):
                    cands.append(g)
        if not cands:
            return None
        # keep the candidate that holds at the win but is rarest in ordinary states
        scored = [(self._contrast(g, agent_colors), g) for g in cands
                  if g.satisfied(win_grid, win_pos, agent_colors)]
        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        self.learned = scored[0][1]
        return self.learned

    def goal_key(self):
        g = self.learned
        if g is None:
            return None
        return (g.kind, getattr(g, "color", getattr(g, "axis", None)))
