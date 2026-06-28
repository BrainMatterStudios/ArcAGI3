"""SymbolicForwardModel — induced-rule next-frame predictor (rule-synthesis foundation).

The goal-signal-learning paradigm is proven closed (CNN prune-probe AUC~chance). The one
non-refuted paradigm is rule SYNTHESIS: induce the game's transition rules as executable code,
which makes the goal known/plannable instead of learned. This is the foundation — a forward
model that predicts the next frame from induced rules. Unlike the old fixed-delta forward
model (C4, broke on slides), this implements SLIDE-to-wall movement (the real mechanic on
ls20/re86), so it can be accurate where the fixed-delta one wasn't.

Rules implemented (induced by mechanic_inference): the avatar (a set of colors) moves in the
action's unit direction, sliding until the next cell is blocked (non-background, non-avatar) or
the grid edge. Everything else is static. predict() returns the predicted next grid.

This is a PREDICTOR only (no agent behavior) — validated for accuracy before any planner is
built on top. If slide-rule prediction is accurate on a slide game, the paradigm has legs.
"""

from __future__ import annotations

import numpy as np


class SymbolicForwardModel:
    def __init__(self, avatar_cols: set[int], deltas: dict, background: int) -> None:
        self.avatar_cols = set(avatar_cols)
        self.deltas = dict(deltas)            # action_id -> (dr, dc) unit direction
        self.bg = int(background)

    def _avatar_mask(self, grid):
        return np.isin(grid, list(self.avatar_cols))

    def predict(self, grid, action_id):
        """Predict the next grid after a slide move. action_id is a simple action (1-4)."""
        if action_id not in self.deltas:
            return grid.copy()
        dr, dc = self.deltas[action_id]
        h, w = grid.shape
        mask = self._avatar_mask(grid)
        cells = np.argwhere(mask)
        if len(cells) == 0:
            return grid.copy()
        # slide the whole avatar rigidly in (dr,dc) until the leading edge is blocked
        obstacle = (~mask) & (grid != self.bg)     # non-bg, non-avatar = walls/objects
        steps = 0
        cur = cells.copy()
        while True:
            nxt = cur + np.array([dr, dc])
            # out of bounds?
            if (nxt[:, 0] < 0).any() or (nxt[:, 0] >= h).any() or \
               (nxt[:, 1] < 0).any() or (nxt[:, 1] >= w).any():
                break
            # would any avatar cell move into an obstacle (that isn't itself avatar)?
            blocked = False
            nxt_set = set(map(tuple, nxt.tolist()))
            for (r, c) in nxt:
                if obstacle[r, c] and (r, c) not in set(map(tuple, cur.tolist())):
                    blocked = True
                    break
            if blocked:
                break
            cur = nxt
            steps += 1
            if steps > h + w:
                break
        out = grid.copy()
        if steps > 0:
            # clear old avatar footprint to background, paint at new position (keep per-cell color)
            colors = grid[cells[:, 0], cells[:, 1]]
            out[cells[:, 0], cells[:, 1]] = self.bg
            out[cur[:, 0], cur[:, 1]] = colors
        return out
