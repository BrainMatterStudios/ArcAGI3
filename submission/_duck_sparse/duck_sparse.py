"""duck_sparse.py — Track B Sparse Model + Inlined arc3kit.py REPL sandbox primitives.

Self-contained module that inlines arc3kit spatial/graph primitives directly into builtins
for the Python REPL sandbox, configures environmental parameters, and applies harness patches.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import sys
from typing import Any
import numpy as np

# Ensure standard context window and ratio to avoid prefill latency penalty
os.environ["TAAF_HUD_MASK"] = "1"
os.environ["TAAF_WIN_REPLAY"] = "1"
os.environ["TAAF_WATCHDOG"] = "1"
os.environ["TAAF_GRAPH"] = "0"
os.environ["TAAF_COMPACT"] = "0"
os.environ["TAAF_PLAYBOOK"] = "0"

GRID_SIZE = 64


@dataclass
class Obj:
    """A connected component region of a single color."""

    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
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


def detect_background(grid: np.ndarray) -> int:
    """Return the most frequent color in the 64x64 grid as background."""
    arr = np.asarray(grid, dtype=np.int8)
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def connected_components(
    grid: np.ndarray,
    background: int | None = None,
    include_background: bool = False,
) -> list[Obj]:
    """Extract 4-connectivity connected components of equal color."""
    arr = np.asarray(grid, dtype=np.int8)
    if arr.ndim == 3:
        arr = arr[-1]

    if background is None:
        background = detect_background(arr)

    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    objs: list[Obj] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(arr[r, c])
            if not include_background and color == background:
                seen[r, c] = True
                continue

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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(arr[nr, nc]) == color:
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


def find_path_bfs(
    grid: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    obstacle_colors: set[int] | list[int] | None = None,
) -> list[tuple[int, int]] | None:
    """Find shortest path from start (r,c) to target (r,c) avoiding obstacle colors."""
    arr = np.asarray(grid, dtype=np.int8)
    if arr.ndim == 3:
        arr = arr[-1]

    h, w = arr.shape
    obstacles = set(obstacle_colors) if obstacle_colors else set()

    sr, sc = start
    tr, tc = target

    if not (0 <= sr < h and 0 <= sc < w and 0 <= tr < h and 0 <= tc < w):
        return None

    q = deque([(sr, sc)])
    visited = {(sr, sc): None}

    while q:
        cr, cc = q.popleft()
        if (cr, cc) == (tr, tc):
            path = []
            curr = (tr, tc)
            while curr is not None:
                path.append(curr)
                curr = visited[curr]
            path.reverse()
            return path

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                cell_color = int(arr[nr, nc])
                if cell_color not in obstacles or (nr, nc) == (tr, tc):
                    visited[(nr, nc)] = (cr, cc)
                    q.append((nr, nc))

    return None


def detect_avatar_motion(
    prev_grid: np.ndarray,
    curr_grid: np.ndarray,
    background: int | None = None,
) -> dict | None:
    """Identify controllable avatar displacement between two consecutive frames."""
    p_arr = np.asarray(prev_grid, dtype=np.int8)
    c_arr = np.asarray(curr_grid, dtype=np.int8)

    if p_arr.ndim == 3:
        p_arr = p_arr[-1]
    if c_arr.ndim == 3:
        c_arr = c_arr[-1]

    diff_mask = p_arr != c_arr
    changed_cells = int(np.sum(diff_mask))

    if changed_cells == 0:
        return {"moved": False, "delta": (0, 0)}

    p_objs = connected_components(p_arr, background=background)
    c_objs = connected_components(c_arr, background=background)

    p_by_color = {o.color: o for o in p_objs}
    c_by_color = {o.color: o for o in c_objs}

    for color, p_o in p_by_color.items():
        if color in c_by_color:
            c_o = c_by_color[color]
            if p_o.size == c_o.size and p_o.bbox != c_o.bbox:
                dr = int(round(c_o.centroid[0] - p_o.centroid[0]))
                dc = int(round(c_o.centroid[1] - p_o.centroid[1]))
                return {
                    "moved": True,
                    "color": color,
                    "from": p_o.top_left,
                    "to": c_o.top_left,
                    "delta": (dr, dc),
                }

    return {"moved": True, "changed_cells": changed_cells, "delta": (0, 0)}


def frame_diff_summary(prev_grid: np.ndarray, curr_grid: np.ndarray) -> dict:
    """Compute summary of changes between two consecutive 64x64 grid frames."""
    p_arr = np.asarray(prev_grid, dtype=np.int8)
    c_arr = np.asarray(curr_grid, dtype=np.int8)

    if p_arr.ndim == 3:
        p_arr = p_arr[-1]
    if c_arr.ndim == 3:
        c_arr = c_arr[-1]

    diff = p_arr != c_arr
    num_changed = int(np.sum(diff))

    if num_changed == 0:
        return {"changed": False, "num_changed_cells": 0}

    rows, cols = np.where(diff)
    r0, r1 = int(np.min(rows)), int(np.max(rows))
    c0, c1 = int(np.min(cols)), int(np.max(cols))

    return {
        "changed": True,
        "num_changed_cells": num_changed,
        "bbox": (r0, c0, r1, c1),
        "changed_colors": [int(x) for x in np.unique(c_arr[diff])],
    }


def inject_arc3kit_primitives() -> str:
    """Pre-seed arc3kit primitives into python builtins for REPL availability."""
    import builtins
    
    builtins.connected_components = connected_components
    builtins.find_path_bfs = find_path_bfs
    builtins.detect_avatar_motion = detect_avatar_motion
    builtins.frame_diff_summary = frame_diff_summary
    
    return "arc3kit REPL pre-seeding: OK (connected_components, find_path_bfs, detect_avatar_motion, frame_diff_summary)"


def apply_sparse_patches() -> str:
    """Apply Track B sparse model patches and inlined arc3kit sandbox pre-seeding."""
    r1 = inject_arc3kit_primitives()
    return f"Track B Sparse Patches Applied!\n{r1}"


if __name__ == "__main__":
    status = apply_sparse_patches()
    print(status)
