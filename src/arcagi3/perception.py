"""Perception: turn a raw ARC-AGI-3 frame into an object-centric state.

A frame from the engine is an int8 array of shape (N, 64, 64) holding one or more
sub-frames (animation/transition steps) with color values 0-15. The agent reasons over
the final settled sub-frame plus a temporally-derived mask of "volatile" cells (status
bars / counters) that must be ignored when deciding whether two states are the same.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

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
    """4-connectivity connected components of equal color (memoized per grid).

    Background color components are skipped unless include_background is True. Results are
    cached on the raw grid bytes: a single decision step calls this several times on the
    SAME grid (state hashing, click targets, nav targeting), so memoizing is a pure
    speedup (identical results) that buys more actions/sec — i.e. more levels at eval.
    """
    if background is None:
        background = detect_background(grid)
    return _connected_components_cached(
        np.ascontiguousarray(grid).tobytes(), grid.shape, int(background), include_background
    )


@lru_cache(maxsize=16)
def _connected_components_cached(grid_bytes, shape, background, include_background) -> list[Obj]:
    grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape(shape)
    h, w = shape
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


def _object_tuples(objs: list[Obj], ignore_colors: set[int] | None = None) -> list[tuple]:
    """Sorted [(color, r0, c0, r1, c1, size)] summary of connected components.

    Single source of truth for the object-level state summary, shared by
    ``object_state_key`` and ``forward_model.Scene.key`` so the two are byte-identical
    (the C4 linchpin: a correct prediction's key must equal the real next key exactly).
    """
    ignore = ignore_colors or set()
    parts = []
    for o in objs:
        if o.color in ignore:
            continue
        r0, c0, r1, c1 = o.bbox
        parts.append((o.color, r0, c0, r1, c1, o.size))
    parts.sort()
    return parts


def object_state_key(grid: np.ndarray, background: int | None = None,
                     ignore_colors: set[int] | None = None) -> bytes:
    """Coarse, robust state key from OBJECT structure (not raw pixels).

    Each non-background, non-ignored connected component is summarised as
    (color, r0, c0, r1, c1, size). Sorting + serialising these is far more stable than a
    pixel hash: it collapses within-object jitter and irrelevant single-pixel noise that
    would otherwise explode the state graph on real games, while still distinguishing
    object moves, appearances/disappearances, and shape changes. Matches the SOTA's
    object-segmentation approach.
    """
    if background is None:
        background = detect_background(grid)
    parts = _object_tuples(connected_components(grid, background=background), ignore_colors)
    return repr(parts).encode()


class VolatilityTracker:
    """Tracks which cells change frequently across steps to mask status bars/counters.

    Cells that change on (almost) every step regardless of effect are likely step
    counters or animated decorations and should be excluded from the state key so the
    state graph doesn't explode.
    """

    def __init__(self, threshold: float = 0.9, min_steps: int = 8) -> None:
        self.threshold = threshold
        self.min_steps = min_steps
        self.changes: np.ndarray | None = None  # lazily sized to the actual frame
        self.shape: tuple[int, int] = (GRID, GRID)
        self.steps = 0
        self._prev: np.ndarray | None = None

    def update(self, grid: np.ndarray) -> None:
        # Lazily adopt the real frame shape; reset if it ever changes (defensive).
        if self.changes is None or grid.shape != self.shape:
            self.shape = grid.shape
            self.changes = np.zeros(self.shape, dtype=np.int32)
            self.steps = 0
            self._prev = None
        if self._prev is not None and self._prev.shape == grid.shape:
            self.changes += (grid != self._prev).astype(np.int32)
            self.steps += 1
        self._prev = grid.copy()

    def mask(self) -> np.ndarray:
        """Boolean mask of cells to ignore (True = volatile)."""
        if self.changes is None or self.steps < self.min_steps:
            return np.zeros(self.shape, dtype=bool)
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
    h, w = grid.shape
    objs = connected_components(grid, background=background)
    # color rarity: rarer colors are more likely interactive (buttons/items)
    color_counts: dict[int, int] = {}
    for o in objs:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
    targets: list[tuple[int, int, int]] = []
    for o in objs:
        r, c = o.centroid
        cr, cc = int(round(r)), int(round(c))
        r0, c0, r1, c1 = o.bbox
        # SOTA-style 5 salience tiers (lower = try first): small + rare-color objects are
        # the most likely interactive elements; wide flat edge-hugging blobs (status bars)
        # go last.
        is_status_bar = (o.height <= 2 or o.width <= 2) and (o.width >= w * 0.6 or o.height >= h * 0.6)
        rare = color_counts.get(o.color, 9) <= 2
        if is_status_bar:
            prio = 4
        elif o.size <= 4:
            prio = 0 if rare else 1
        elif o.size <= 16:
            prio = 1 if rare else 2
        elif o.size <= 64:
            prio = 2 if rare else 3
        else:
            prio = 3
        targets.append((cc, cr, prio))
        # corners of larger objects (handles/edges), one tier lower
        if o.size > 8 and not is_status_bar:
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
