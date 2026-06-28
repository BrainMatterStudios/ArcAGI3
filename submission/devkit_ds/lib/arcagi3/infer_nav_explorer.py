"""InferNavExplorer — robust-detection goal-directed slide nav (Exp #10).

The slide-nav family delegated on ls20 because its crude 24-step cycle-arrows probe can't
detect the avatar that the full mechanic_inference (rigid-translation over a diverse
trajectory) finds cleanly. Fix: BOOTSTRAP by delegating to SalienceExplorer for N steps while
capturing the trajectory, run mechanic_inference for robust avatar+deltas, then engage
goal-directed slide nav (inherits the slide-graph + goal steering from GoalDirectedSlideExplorer).

If inference can't find a clean axis-aligned avatar with both axes and no push -> keep
delegating (== v6, firewall). This isolates the genuine question: with the avatar reliably
detected, does goal-directed avatar-cell nav reach deeper levels on a wall game (ls20) than
the board-hash graph explorer?
"""

from __future__ import annotations

import numpy as np

from . import perception as P
from .goal_slide_explorer import GoalDirectedSlideExplorer
from .mechanic_inference import infer_mechanics


class InferNavExplorer(GoalDirectedSlideExplorer):
    def __init__(self, *args, bootstrap_steps: int = 2000, **kwargs) -> None:
        self.bootstrap_steps = int(bootstrap_steps)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.mode = "bootstrap" if self.enable_slide else "delegate"
        self._boot_traj = []
        self._boot_prev = None   # (grid, token)
        self._boot_n = 0

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        self._avail = [a for a in (1, 2, 3, 4) if a in available]
        if not self.enable_slide or 6 in available or not self._avail:
            return self.fallback.decide(grid, gstate_terminal, gstate_notplayed, levels, available)

        if self.mode == "bootstrap":
            # capture the transition produced by the previous delegated action
            if self._boot_prev is not None and not gstate_terminal and not gstate_notplayed:
                pg, ptok = self._boot_prev
                r = float(levels - self.prev_levels)
                self._boot_traj.append((pg, ptok, grid, r))
            self.prev_levels = levels
            tok = self.fallback.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
            self._boot_prev = (grid, tok) if tok[0] != "reset" else None
            self._boot_n += 1
            if self._boot_n >= self.bootstrap_steps:
                self._acquire_from_inference()
            return tok

        # nav / delegate handled by the parent (GoalDirectedSlideExplorer -> SlideNavExplorer)
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _acquire_from_inference(self):
        rep = infer_mechanics(self._boot_traj, self.bg)
        if rep.avatar_color is None or "push" in rep.interactions:
            self.mode = "delegate"
            return
        # convert inferred (dr,dc) deltas to unit axis-aligned directions
        deltas = {}
        for a, d in rep.avatar_action_deltas.items():
            if d is None:
                continue
            dr, dc = d
            if abs(dr) >= abs(dc) and dr != 0:
                deltas[a] = (1 if dr > 0 else -1, 0)
            elif dc != 0:
                deltas[a] = (0, 1 if dc > 0 else -1)
        has_r = any(v[0] for v in deltas.values())
        has_c = any(v[1] for v in deltas.values())
        if not (has_r and has_c):
            self.mode = "delegate"
            return
        self.avatar_cols = {rep.avatar_color}
        self.deltas = deltas
        # goal-color is learned on the first nav-phase level-up (parent _gd tracking)
        self.mode = "nav"
        self.prev_action = None
        self.prev_cell = None
