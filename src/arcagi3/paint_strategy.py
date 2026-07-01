"""PaintStrategy — reactive (decide()-interface) SOURCE-FREE paint-to-stencil solver (re86 archetype),
embeddable as an additive PortfolioPolicy play. Pieces = single-color movable shapes; target cells = same-
color pixels surrounded by the frame color (4); active piece carries the color-0 cursor. For each target
color: select that piece (ACTION5 cycle), then drag the cursor-head over the target so its trail paints the
stencil. Abstains (benign) on non-matching games -> can only ADD, never regress the floor.

Token format: ("S", id) for moves/ACTION5; ("reset",).
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from arcagi3 import perception as P

FRAME_COLOR = 4
CURSOR = 0


def _frame_adjacent(grid):
    fm = (grid == FRAME_COLOR); adj = np.zeros_like(fm)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            adj |= np.roll(np.roll(fm, dr, axis=0), dc, axis=1)
    return adj


def _perceive(grid):
    bg = P.detect_background(grid); adj = _frame_adjacent(grid)
    rows = np.arange(grid.shape[0])[:, None] * np.ones((1, grid.shape[1]))
    pieces, targets = {}, {}
    for c in range(16):
        if c in (bg, FRAME_COLOR, CURSOR):
            continue
        cmask = (grid == c) & (rows < 60)
        if not cmask.any():
            continue
        tmask = cmask & adj; pmask = cmask & ~adj
        if tmask.sum() > 0:
            ys, xs = np.where(tmask); targets[c] = (ys.mean(), xs.mean())
        if pmask.sum() >= 8:
            ys, xs = np.where(pmask); pieces[c] = (ys.mean(), xs.mean())
    return pieces, targets


def _cursor(grid):
    ys, xs = np.where(grid == CURSOR)
    return (ys.mean(), xs.mean()) if len(ys) else None


def _active_color(grid):
    cur = _cursor(grid)
    if cur is None:
        return None
    pieces, _ = _perceive(grid)
    if not pieces:
        return None
    return min(pieces, key=lambda c: abs(pieces[c][0] - cur[0]) + abs(pieces[c][1] - cur[1]))


class PaintStrategy:
    def __init__(self, seed: int = 0):
        self.gs = None
        self._reset()

    def _reset(self):
        self._order = None
        self._ci = 0
        self._phase = "init"     # init -> select -> drag -> abstain
        self._seltries = 0
        self._planned_level = -1

    def _benign(self, available):
        if 5 in available:
            return ("S", 5)
        if available:
            return ("S", int(available[0]))
        return ("reset",)

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        if gstate_terminal:
            self._reset(); return ("reset",)
        if levels != self._planned_level and self._phase != "init":
            self._reset()
        if self._phase == "abstain":
            return self._benign(available)

        if self._phase == "init":
            self._planned_level = levels
            if 5 not in available:
                self._phase = "abstain"; return self._benign(available)
            pieces, targets = _perceive(grid)
            self._order = [c for c in targets if c in pieces]
            self._targets = targets
            if not self._order:
                self._phase = "abstain"; return self._benign(available)
            self._ci = 0; self._phase = "select"; self._seltries = 0

        if self._ci >= len(self._order):
            self._phase = "abstain"; return self._benign(available)
        color = self._order[self._ci]

        if self._phase == "select":
            if _active_color(grid) == color:
                self._phase = "drag"
            else:
                self._seltries += 1
                if self._seltries > 5:      # can't select -> skip this color
                    self._ci += 1; self._seltries = 0
                    return self.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
                return ("S", 5)

        if self._phase == "drag":
            ty, tx = self._targets[color]
            cur = _cursor(grid)
            if cur is None or abs(tx - cur[1]) + abs(ty - cur[0]) < 2:
                self._ci += 1; self._phase = "select"; self._seltries = 0
                return self.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
            dy, dx = ty - cur[0], tx - cur[1]
            a = (4 if dx > 0 else 3) if abs(dx) >= abs(dy) else (2 if dy > 0 else 1)
            return ("S", a)

        return self._benign(available)
