"""duck_patches_sparse.py — Harness patches for Track B arc3kit Python Tool Sandbox REPL Injection.

Directly monkey-patches `inference.agent.python_tool_sandbox._SANDBOX_BOOTSTRAP` so that
`connected_components`, `find_path_bfs`, `detect_avatar_motion`, `frame_diff_summary` are:
  1. Injected directly into the sandboxed subprocess text bootstrap.
  2. Registered in `runtime_globals` and `runtime_globals["__builtins__"]` inside the sandbox.
  3. Formally advertised in the LLM System Prompt (`prompts.PYTHON_ADDENDUM` & `tool_agent.PYTHON_ADDENDUM`).
"""

from __future__ import annotations

import inspect
import os
import textwrap
from typing import Any

ARC3KIT_SANDBOX_BOOTSTRAP_CODE = textwrap.dedent(
    """
    # --- ARC3KIT INLINED SANDBOX PRIMITIVES ---
    from collections import deque as _deque
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class Obj:
        color: int
        cells: tuple
        bbox: tuple
        size: int
        centroid: tuple

        @property
        def top_left(self):
            return (self.bbox[0], self.bbox[1])

        @property
        def width(self):
            return self.bbox[3] - self.bbox[1] + 1

        @property
        def height(self):
            return self.bbox[2] - self.bbox[0] + 1


    def _grid_to_2d_list(grid):
        if grid is None:
            return []
        if hasattr(grid, "_grid"):
            grid = getattr(grid, "_grid", None)
        elif hasattr(grid, "grid"):
            grid = getattr(grid, "grid", None)
        elif hasattr(grid, "ascii"):
            ascii_val = getattr(grid, "ascii", "")
            if ascii_val:
                lines = [line.strip() for line in str(ascii_val).strip().splitlines() if line.strip()]
                grid = [[ord(ch) - 65 if 'A' <= ch <= 'Z' else (int(ch) if ch.isdigit() else 0) for ch in line] for line in lines]

        if grid is None:
            return []
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            return [[int(c) for c in row] for row in grid]
        return []


    def detect_background(grid):
        arr = _grid_to_2d_list(grid)
        if not arr:
            return 0
        counts = {}
        for row in arr:
            for val in row:
                counts[val] = counts.get(val, 0) + 1
        return max(counts.keys(), key=lambda k: counts[k])


    def connected_components(grid, background=None, include_background=False):
        arr = _grid_to_2d_list(grid)
        if not arr or not arr[0]:
            return []

        if background is None:
            background = detect_background(arr)

        h = len(arr)
        w = len(arr[0])
        seen = [[False] * w for _ in range(h)]
        objs = []

        for r in range(h):
            for c in range(w):
                if seen[r][c]:
                    continue
                color = arr[r][c]
                if not include_background and color == background:
                    seen[r][c] = True
                    continue

                cells = []
                q = _deque([(r, c)])
                seen[r][c] = True
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
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and arr[nr][nc] == color:
                            seen[nr][nc] = True
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


    def find_path_bfs(grid, start, target, obstacle_colors=None):
        arr = _grid_to_2d_list(grid)
        if not arr or not arr[0]:
            return None

        h = len(arr)
        w = len(arr[0])
        obstacles = {int(x) for x in obstacle_colors} if obstacle_colors else set()

        sr, sc = start
        tr, tc = target

        if not (0 <= sr < h and 0 <= sc < w and 0 <= tr < h and 0 <= tc < w):
            return None

        q = _deque([(sr, sc)])
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
                    cell_color = arr[nr][nc]
                    if cell_color not in obstacles or (nr, nc) == (tr, tc):
                        visited[(nr, nc)] = (cr, cc)
                        q.append((nr, nc))

        return None


    def detect_avatar_motion(prev_grid, curr_grid, background=None):
        if prev_grid is None or curr_grid is None:
            return {"moved": False, "delta": (0, 0)}

        p_arr = _grid_to_2d_list(prev_grid)
        c_arr = _grid_to_2d_list(curr_grid)

        if not p_arr or not c_arr:
            return {"moved": False, "delta": (0, 0)}

        changed_cells = 0
        h, w = len(p_arr), len(p_arr[0])
        for r in range(h):
            for c in range(w):
                if p_arr[r][c] != c_arr[r][c]:
                    changed_cells += 1

        if changed_cells == 0:
            return {"moved": False, "delta": (0, 0)}

        p_objs = connected_components(p_arr, background=background)
        c_objs = connected_components(c_arr, background=background)

        for p_o in p_objs:
            for c_o in c_objs:
                if p_o.color == c_o.color and p_o.size == c_o.size and p_o.bbox != c_o.bbox:
                    dr = int(round(c_o.centroid[0] - p_o.centroid[0]))
                    dc = int(round(c_o.centroid[1] - p_o.centroid[1]))
                    if abs(dr) <= 2 and abs(dc) <= 2:
                        return {
                            "moved": True,
                            "color": p_o.color,
                            "from": p_o.top_left,
                            "to": c_o.top_left,
                            "delta": (dr, dc),
                        }

        return {"moved": True, "changed_cells": changed_cells, "delta": (0, 0)}


    def frame_diff_summary(prev_grid, curr_grid):
        if prev_grid is None or curr_grid is None:
            return {"changed": False, "num_changed_cells": 0}

        p_arr = _grid_to_2d_list(prev_grid)
        c_arr = _grid_to_2d_list(curr_grid)

        if not p_arr or not c_arr:
            return {"changed": False, "num_changed_cells": 0}

        h, w = len(p_arr), len(p_arr[0])
        changed_coords = []
        changed_colors = set()

        for r in range(h):
            for c in range(w):
                if p_arr[r][c] != c_arr[r][c]:
                    changed_coords.append((r, c))
                    changed_colors.add(c_arr[r][c])

        if not changed_coords:
            return {"changed": False, "num_changed_cells": 0}

        rows = [r for r, _ in changed_coords]
        cols = [c for _, c in changed_coords]

        return {
            "changed": True,
            "num_changed_cells": len(changed_coords),
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
            "changed_colors": sorted(list(changed_colors)),
        }
    """
)

ARC3KIT_PROMPT_ADDENDUM = (
    "\n\nPre-seeded spatial and graph helper functions available directly in Python (NO IMPORTS NEEDED):\n"
    "- `connected_components(grid, background=None)`: returns list of `Obj(color, cells, bbox, size, centroid)` for all 4-connected same-color objects.\n"
    "- `find_path_bfs(grid, start_pos, target_pos, obstacle_colors)`: returns shortest path list of `(r, c)` coordinates avoiding `obstacle_colors`, or `None` if blocked.\n"
    "- `detect_avatar_motion(prev_grid, curr_grid)`: returns `dict(moved=True/False, color=c, delta=(dr, dc))` describing avatar translation.\n"
    "- `frame_diff_summary(prev_grid, curr_grid)`: returns `dict(changed=True/False, num_changed_cells=N, bbox=(r0,c0,r1,c1))`.\n"
    "USE THESE HELPER FUNCTIONS to avoid writing custom BFS or connected component flood-fills!\n"
)


def patch_sandbox_arc3kit() -> str:
    """Patch python_tool_sandbox._SANDBOX_BOOTSTRAP to inject arc3kit into the sandbox subshell."""
    from inference.agent import python_tool_sandbox

    if getattr(python_tool_sandbox, "_arc3kit_sandbox_patched", False):
        return "patch_sandbox_arc3kit: SKIP (already applied)"

    bootstrap = python_tool_sandbox._SANDBOX_BOOTSTRAP

    # 1. Append ARC3KIT_SANDBOX_BOOTSTRAP_CODE into bootstrap
    if 'SAFE_MODULES = {' in bootstrap:
        bootstrap = bootstrap.replace(
            'SAFE_MODULES = {',
            f'{ARC3KIT_SANDBOX_BOOTSTRAP_CODE}\n\nSAFE_MODULES = {{',
            1,
        )
    else:
        raise RuntimeError("SAFE_MODULES marker not found in _SANDBOX_BOOTSTRAP")

    # 2. Add runtime_globals mapping inside bootstrap main()
    runtime_globals_target = '    runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
    runtime_globals_insert = (
        '    runtime_globals["_grid_to_2d_list"] = _grid_to_2d_list\n'
        '    runtime_globals["detect_background"] = detect_background\n'
        '    runtime_globals["Obj"] = Obj\n'
        '    runtime_globals["connected_components"] = connected_components\n'
        '    runtime_globals["find_path_bfs"] = find_path_bfs\n'
        '    runtime_globals["detect_avatar_motion"] = detect_avatar_motion\n'
        '    runtime_globals["frame_diff_summary"] = frame_diff_summary\n'
        '    runtime_globals["__builtins__"]["connected_components"] = connected_components\n'
        '    runtime_globals["__builtins__"]["find_path_bfs"] = find_path_bfs\n'
        '    runtime_globals["__builtins__"]["detect_avatar_motion"] = detect_avatar_motion\n'
        '    runtime_globals["__builtins__"]["frame_diff_summary"] = frame_diff_summary\n'
    )
    if runtime_globals_target in bootstrap:
        bootstrap = bootstrap.replace(
            runtime_globals_target,
            runtime_globals_target + runtime_globals_insert,
            1,
        )
    else:
        raise RuntimeError("runtime_globals_target marker not found in _SANDBOX_BOOTSTRAP")

    python_tool_sandbox._SANDBOX_BOOTSTRAP = bootstrap
    python_tool_sandbox._arc3kit_sandbox_patched = True
    return "patch_sandbox_arc3kit: OK (primitives inlined & registered in _SANDBOX_BOOTSTRAP)"


def patch_prompts_arc3kit() -> str:
    """Patch inference.agent.prompts.PYTHON_ADDENDUM and tool_agent.PYTHON_ADDENDUM to advertise arc3kit."""
    from inference.agent import prompts, tool_agent

    if getattr(prompts, "_arc3kit_prompts_patched", False):
        return "patch_prompts_arc3kit: SKIP (already applied)"

    prompts.PYTHON_ADDENDUM = prompts.PYTHON_ADDENDUM + ARC3KIT_PROMPT_ADDENDUM
    tool_agent.PYTHON_ADDENDUM = tool_agent.PYTHON_ADDENDUM + ARC3KIT_PROMPT_ADDENDUM
    prompts._arc3kit_prompts_patched = True
    return "patch_prompts_arc3kit: OK (advertised in prompts & tool_agent System Prompt)"


def apply_sparse_sandbox_patches() -> str:
    """Apply all Track B sandbox injection and prompt advertisement patches."""
    r1 = patch_sandbox_arc3kit()
    r2 = patch_prompts_arc3kit()
    return f"{r1}\n{r2}"
