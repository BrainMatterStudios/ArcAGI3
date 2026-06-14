"""Perception: turn a raw ARC-AGI-3 frame into an object-centric state.

A frame from the engine is an int8 array of shape (N, 64, 64) holding one or more
sub-frames (animation/transition steps) with color values 0-15. The agent reasons over
the final settled sub-frame plus a temporally-derived mask of "volatile" cells (status
bars / counters) that must be ignored when deciding whether two states are the same.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

GRID = 64


def to_grid(frame) -> np.ndarray:
    """Return the final settled 64x64 sub-frame as an int8 ndarray.

    Accepts a FrameData, a list, or an ndarray of shape (N,64,64) / (64,64).
    """
    if hasattr(frame, "frame"):  # a FrameData
        frame = frame.frame
    arr = np.asarray(frame, dtype=np.int8)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"unexpected frame shape {arr.shape}")
    return arr


def grid_stack(frame) -> np.ndarray:
    """Return all sub-frames as (N,64,64); animation across a single step."""
    if hasattr(frame, "frame"):
        frame = frame.frame
    arr = np.asarray(frame, dtype=np.int8)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr


def detect_background(grid: np.ndarray) -> int:
    """Most frequent color = presumed background."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


@dataclass
class Obj:
    """A connected region of a single color (4-connectivity)."""

    color: int
    cells: tuple[tuple[int, int], ...]  # (row, col) pairs
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    size: int
    centroid: tuple[float, float]

    @property
    def top_left(self) -> tuple[int, int]:
        return (self.bbox[0], self.bbox[1])

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1


def connected_components(
    grid: np.ndarray,
    background: int | None = None,
    include_background: bool = False,
) -> list[Obj]:
    """4-connectivity connected components of equal color.

    Background color components are skipped unless include_background is True.
    """
    if background is None:
        background = detect_background(grid)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objs: list[Obj] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if not include_background and color == background:
                seen[r, c] = True
                continue
            # BFS flood fill
            cells: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            sr = sc = 0
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                sr += cr
                sc += cc
                r0, r1 = min(r0, cr), max(r1, cr)
                c0, c1 = min(c0, cc), max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            n = len(cells)
            objs.append(
                Obj(
                    color=color,
                    cells=tuple(cells),
                    bbox=(r0, c0, r1, c1),
                    size=n,
                    centroid=(sr / n, sc / n),
                )
            )
    return objs


def state_hash(grid: np.ndarray, mask: np.ndarray | None = None) -> bytes:
    """Exact hash of the grid, optionally zeroing masked (volatile) cells first."""
    if mask is not None:
        g = grid.copy()
        g[mask] = -1
        return g.tobytes()
    return np.ascontiguousarray(grid).tobytes()


class VolatilityTracker:
    """Tracks which cells change frequently across steps to mask status bars/counters.

    Cells that change on (almost) every step regardless of effect are likely step
    counters or animated decorations and should be excluded from the state key so the
    state graph doesn't explode.
    """

    def __init__(self, threshold: float = 0.9, min_steps: int = 8) -> None:
        self.threshold = threshold
        self.min_steps = min_steps
        self.changes = np.zeros((GRID, GRID), dtype=np.int32)
        self.steps = 0
        self._prev: np.ndarray | None = None

    def update(self, grid: np.ndarray) -> None:
        if self._prev is not None:
            self.changes += (grid != self._prev).astype(np.int32)
            self.steps += 1
        self._prev = grid.copy()

    def mask(self) -> np.ndarray:
        """Boolean mask of cells to ignore (True = volatile)."""
        if self.steps < self.min_steps:
            return np.zeros((GRID, GRID), dtype=bool)
        return (self.changes / max(self.steps, 1)) >= self.threshold


def salient_click_targets(
    grid: np.ndarray, background: int | None = None, max_targets: int = 64,
    coarse_grid_step: int = 0,
) -> list[tuple[int, int, int]]:
    """Propose (x, y, priority) click targets from object geometry.

    Object-centric instead of brute-forcing all 4096 pixels. Priority is a salience
    tier (lower = try first): small distinct objects and their corners are most likely
    interactive. Returns (x=col, y=row, priority).

    If coarse_grid_step > 0, also add a coarse lattice of low-priority fallback targets
    (every `coarse_grid_step` pixels) so large click action-spaces (e.g. ft09's ~4096
    positions) where the goal cell isn't an object centroid are still reachable.
    """
    if background is None:
        background = detect_background(grid)
    objs = connected_components(grid, background=background)
    targets: list[tuple[int, int, int]] = []
    for o in objs:
        r, c = o.centroid
        cr, cc = int(round(r)), int(round(c))
        # smaller objects are likely buttons/agents/items -> higher priority (lower num)
        if o.size <= 2:
            prio = 0
        elif o.size <= 8:
            prio = 1
        elif o.size <= 32:
            prio = 2
        else:
            prio = 3
        targets.append((cc, cr, prio))
        # corners of larger objects (handles/edges)
        if o.size > 4:
            r0, c0, r1, c1 = o.bbox
            for (yy, xx) in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                targets.append((xx, yy, prio + 1))
    # coarse lattice fallback for large click spaces (lowest priority)
    if coarse_grid_step and coarse_grid_step > 0:
        h, w = grid.shape
        off = coarse_grid_step // 2
        for yy in range(off, h, coarse_grid_step):
            for xx in range(off, w, coarse_grid_step):
                targets.append((xx, yy, 9))
    # dedup keeping best (lowest) priority
    best: dict[tuple[int, int], int] = {}
    for x, y, p in targets:
        k = (x, y)
        if k not in best or p < best[k]:
            best[k] = p
    out = [(x, y, p) for (x, y), p in best.items()]
    out.sort(key=lambda t: t[2])
    return out[:max_targets]
