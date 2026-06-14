"""Motion model: learn the controllable object (avatar) and how actions move it.

Most ARC-AGI-3 games (and interactive games generally) have an avatar the player moves
with simple actions. If we can identify it and learn each action's displacement vector,
we can navigate in coordinate space (cheap, goal-directed) instead of blind exploration
over hashed states. This is fully general — no per-game knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import perception as P


def colored_cells(grid: np.ndarray, background: int) -> dict[int, np.ndarray]:
    """Map color -> boolean mask of its cells (excluding background)."""
    out = {}
    for c in np.unique(grid):
        if int(c) == background:
            continue
        out[int(c)] = grid == c
    return out


def infer_translation(before: np.ndarray, after: np.ndarray, background: int):
    """If exactly one color's region translated by a constant vector, return (color, dr, dc).

    Returns None if the change isn't a clean single-object translation.
    """
    if before.shape != after.shape:
        return None
    changed_colors = []
    for c in set(np.unique(before)).union(np.unique(after)):
        c = int(c)
        if c == background:
            continue
        b = before == c
        a = after == c
        if not np.array_equal(b, a):
            changed_colors.append(c)
    # The avatar is a color whose mask moved. Static decorations don't change.
    best = None
    for c in changed_colors:
        b = np.argwhere(before == c)
        a = np.argwhere(after == c)
        if len(b) == 0 or len(a) == 0 or len(b) != len(a):
            continue
        # candidate translation = centroid shift
        db = b.mean(axis=0)
        da = a.mean(axis=0)
        dr, dc = da - db, None
        shift = (da - db)
        # verify it's a rigid translation: shifting before-cells by round(shift) == after-cells
        sr, sc = int(round(shift[0])), int(round(shift[1]))
        shifted = b + np.array([sr, sc])
        if set(map(tuple, shifted.tolist())) == set(map(tuple, a.tolist())):
            if (sr, sc) != (0, 0):
                # prefer the smallest moving object (likely the avatar)
                if best is None or len(b) < best[3]:
                    best = (c, sr, sc, len(b))
    if best is None:
        return None
    return (best[0], best[1], best[2])


def infer_all_translations(before: np.ndarray, after: np.ndarray, background: int) -> dict:
    """Return {color: (dr, dc)} for every non-background color that rigidly translated.

    Unlike infer_translation (single best mover), this reports all movers so the caller
    can distinguish the avatar (motion varies with the action) from independent
    animations/counters (motion is constant regardless of the action).
    """
    out: dict[int, tuple[int, int]] = {}
    if before.shape != after.shape:
        return out
    for c in set(np.unique(before)).union(np.unique(after)):
        c = int(c)
        if c == background:
            continue
        b = np.argwhere(before == c)
        a = np.argwhere(after == c)
        if len(b) == 0 or len(a) == 0 or len(b) != len(a):
            continue
        shift = a.mean(axis=0) - b.mean(axis=0)
        sr, sc = int(round(shift[0])), int(round(shift[1]))
        if (sr, sc) == (0, 0):
            continue
        shifted = b + np.array([sr, sc])
        if set(map(tuple, shifted.tolist())) == set(map(tuple, a.tolist())):
            out[c] = (sr, sc)
    return out


@dataclass
class MotionModel:
    avatar_color: int | None = None
    deltas: dict[int, tuple[int, int]] = field(default_factory=dict)  # action_id -> (dr,dc)

    @property
    def ok(self) -> bool:
        return self.avatar_color is not None and len(self.deltas) > 0

    def avatar_centroid(self, grid: np.ndarray):
        if self.avatar_color is None:
            return None
        cells = np.argwhere(grid == self.avatar_color)
        if len(cells) == 0:
            return None
        return tuple(cells.mean(axis=0))  # (row, col) float

    def avatar_cells(self, grid: np.ndarray) -> np.ndarray:
        return np.argwhere(grid == self.avatar_color)
