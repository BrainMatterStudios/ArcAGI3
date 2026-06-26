"""Pure interaction-signature extractor for the OC-WM object-curiosity explorer.

Given two consecutive grids and the action taken between them, produce a hashable signature
describing the object interaction. Reward-free and deterministic. ("NONE",) means no object
change was attributable to the action.
"""

from __future__ import annotations

import numpy as np

NONE_SIGNATURE = ("NONE",)


def _action_descriptor(action, prev_grid, bg):
    """('S', aid) for simple actions; ('C', color_at_target) for clicks."""
    if action[0] == "S":
        return ("S", int(action[1]))
    # click ("C", x=col, y=row): the target color on the PREVIOUS grid
    x, y = int(action[1]), int(action[2])
    h, w = prev_grid.shape
    color = int(prev_grid[y, x]) if 0 <= y < h and 0 <= x < w else -1
    return ("C", color)


def _outcome(prev_colors, cur_colors, bg):
    """Classify the change at the changed cells (sets of colors before/after)."""
    prev_obj = prev_colors - ({bg} if bg is not None else set())
    cur_obj = cur_colors - ({bg} if bg is not None else set())
    if not prev_obj and cur_obj:
        return "appeared"
    if prev_obj and not cur_obj:
        return "vanished"
    return "recolored"


def interaction_signature(prev_grid, action, cur_grid, bg):
    prev_grid = np.asarray(prev_grid)
    cur_grid = np.asarray(cur_grid)
    if prev_grid.shape != cur_grid.shape:
        return NONE_SIGNATURE
    changed = prev_grid != cur_grid
    if not changed.any():
        return NONE_SIGNATURE
    prev_colors = set(int(c) for c in np.unique(prev_grid[changed]))
    cur_colors = set(int(c) for c in np.unique(cur_grid[changed]))
    outcome = _outcome(prev_colors, cur_colors, bg)
    involved = frozenset(
        (prev_colors | cur_colors) - ({bg} if bg is not None else set())
    )
    desc = _action_descriptor(action, prev_grid, bg)
    return (desc[0], desc[1], outcome, involved)
