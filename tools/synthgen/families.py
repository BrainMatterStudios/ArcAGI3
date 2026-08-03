"""Samplers + mechanical reference solvers for the 3 synthetic mechanic families.

Families (distinct skill classes):
    nav   -- navigate/avoid with state-gated doors (keys and one-shot switches)
    click -- click-target selection by visual property, distractors, optional
             state-gated arm button
    push  -- transform/arrange: push boxes onto target pads (Sokoban-lite,
             order-sensitive)

Every sampled game is a plain-dict "spec" (JSON-serializable) that fully
determines the generated game file AND is rich enough for the mechanical
reference solver to produce a ground-truth action trace without ever running
an LLM. Solvers model the exact semantics implemented by the templates in
render_game.py; validate.py replays the trace through the real engine to prove
the two agree.

Action encoding in traces: {"id": "ACTION1"} .. {"id": "ACTION5"} or
{"id": "ACTION6", "x": <display col 0-63>, "y": <display row 0-63>}.
Direction convention (matches public games, e.g. ls20):
    ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right.

Mapping to the 2026-08-03 unlock-failure diagnosis (scratchpad/unlock_diag/
NOTES.md ranked synthetic families F1-F7 by what blocks the 9 never-unlocked
public games — dominant blocker: mechanic/goal inference):
    nav   ~ F1 precondition-gate mazes (keys/switches arm the goal) — partial:
            our gates are visually explicit, F1 wants subtle indicator changes.
    click ~ F3 select/state-gated clicks (arm button) — partial: F3 also wants
            select-reveal-execute (peg-jump) and button->remote-toggle pairs.
    push  ~ F6 arrange/order-sensitive transforms — partial: F6 wants matching
            against a rendered legend/partner markers.
Not covered yet (v2 candidates): F2 record-replay tools, F4 grab-carry with
facing, F5 coupled multi-avatar, F7 symbolic rule tables, and SCoRe-shaped
wrong-hypothesis->revision trajectories (needs a deliberately imperfect teacher).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

FAMILIES = ("nav", "click", "push")

# family -> 2-char game-id prefix. No public game id starts with 'x', so the
# synthetic namespace can never collide with a real one.
FAMILY_PREFIX = {"nav": "xn", "click": "xk", "push": "xp"}

DIR_ACTIONS = {(-1, 0): "ACTION1", (1, 0): "ACTION2", (0, -1): "ACTION3", (0, 1): "ACTION4"}
DIRS = list(DIR_ACTIONS.keys())

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def game_id_for(family: str, index: int, seed: int) -> str:
    """4-char base id + 8-hex version, e.g. 'xn03-1a2b3c4d'.

    Uses a stable hash (not builtin hash(), which is salted per process) so
    game ids are reproducible across runs.
    """
    import hashlib

    base = f"{FAMILY_PREFIX[family]}{index % 100:02d}"
    version = hashlib.md5(f"synthgen-{family}-{index}-{seed}".encode()).hexdigest()[:8]
    return f"{base}-{version}"


def _pick_colors(rng: random.Random, n: int, exclude: set[int] | None = None) -> list[int]:
    """n distinct palette colors 0-15, avoiding `exclude`."""
    pool = [c for c in range(16) if c not in (exclude or set())]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError("palette exhausted")
    return pool[:n]


def _display_scale_offset(width: int, height: int) -> tuple[int, int, int]:
    """Replicates Camera._calculate_scale_and_offset for a 64x64 display."""
    scale = min(64 // width, 64 // height)
    return scale, (64 - width * scale) // 2, (64 - height * scale) // 2


def _sprite_masks(rng: random.Random, size: int) -> list[list[int]]:
    """A small library of size x size 0/1 masks for player/object shapes."""
    full = [[1] * size for _ in range(size)]
    ring = [[1 if r in (0, size - 1) or c in (0, size - 1) else 0 for c in range(size)] for r in range(size)]
    plus = [[1 if r == size // 2 or c == size // 2 else 0 for c in range(size)] for r in range(size)]
    diamond = [
        [1 if abs(r - size // 2) + abs(c - size // 2) <= size // 2 else 0 for c in range(size)]
        for r in range(size)
    ]
    notch = [row[:] for row in full]
    notch[0][0] = notch[0][size - 1] = 0
    tee = [[1 if r == 0 or c == size // 2 else 0 for c in range(size)] for r in range(size)]
    u_shape = [[1 if c in (0, size - 1) or r == size - 1 else 0 for c in range(size)] for r in range(size)]
    return rng.choice([full, ring, plus, diamond, notch, tee, u_shape])


SHAPE_NAMES = ("full", "ring", "plus", "diamond", "notch", "tee", "u")


def _named_mask(name: str, size: int) -> list[list[int]]:
    half = size // 2
    if name == "full":
        return [[1] * size for _ in range(size)]
    if name == "ring":
        return [[1 if r in (0, size - 1) or c in (0, size - 1) else 0 for c in range(size)] for r in range(size)]
    if name == "plus":
        return [[1 if r == half or c == half else 0 for c in range(size)] for r in range(size)]
    if name == "diamond":
        return [[1 if abs(r - half) + abs(c - half) <= half else 0 for c in range(size)] for r in range(size)]
    if name == "notch":
        m = [[1] * size for _ in range(size)]
        m[0][0] = m[0][size - 1] = 0
        return m
    if name == "tee":
        return [[1 if r == 0 or c == half else 0 for c in range(size)] for r in range(size)]
    if name == "u":
        return [[1 if c in (0, size - 1) or r == size - 1 else 0 for c in range(size)] for r in range(size)]
    raise ValueError(name)


# ---------------------------------------------------------------------------
# family: nav
# ---------------------------------------------------------------------------

# map chars: '#' wall, '.' floor, 'P' player, 'G' goal, 'x' hazard,
# 'a'/'b' keys, 'A'/'B' key-doors, 's' switch, 'S' switch-door
_GATE_TOKENS = ["a", "b", "s"]


def _nav_bfs(grid: list[str], tokens: list[str]) -> list[tuple[int, int]] | None:
    """BFS over (r, c, opened-mask). Returns cell path (incl. start) or None.

    Semantics mirrored by the nav template: doors block until their token is
    opened; keys/switches open their token when stepped on; hazards are fatal
    (never entered); goal ends the level.
    """
    rows, cols = len(grid), len(grid[0])
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "P":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    assert start and goal
    tok_bit = {t: 1 << i for i, t in enumerate(tokens)}
    init = (start[0], start[1], 0)
    prev: dict[tuple[int, int, int], tuple[int, int, int] | None] = {init: None}
    q = deque([init])
    while q:
        r, c, mask = q.popleft()
        if (r, c) == goal:
            path = []
            node: tuple[int, int, int] | None = (r, c, mask)
            while node is not None:
                path.append((node[0], node[1]))
                node = prev[node]
            return path[::-1]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            ch = grid[nr][nc]
            if ch == "#" or ch == "x":
                continue
            if ch in ("A", "B", "S") and not mask & tok_bit.get(ch.lower(), 0):
                continue
            nmask = mask
            if ch in ("a", "b", "s"):
                nmask |= tok_bit[ch]
            nstate = (nr, nc, nmask)
            if nstate not in prev:
                prev[nstate] = (r, c, mask)
                q.append(nstate)
    return None


def _nav_reachable_no_gates(grid: list[str], start: tuple[int, int]) -> set[tuple[int, int]]:
    """Cells reachable while treating every door as a wall (gate bypass check)."""
    rows, cols = len(grid), len(grid[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                if grid[nr][nc] not in ("#", "x", "A", "B", "S"):
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return seen


def _sample_nav_level(rng: random.Random, rows: int, cols: int, n_gates: int, n_hazards: int) -> list[str]:
    """One nav map. Gates are wall-lines with a single door gap, so they can
    never be bypassed; keys/switches sit in the band before their door."""
    for _attempt in range(400):
        g = [["." for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            g[r][0] = g[r][cols - 1] = "#"
        for c in range(cols):
            g[0][c] = g[rows - 1][c] = "#"

        # gate wall-lines split the interior into vertical bands.
        # Two lines need >= 3 cols between them; narrow grids get one gate.
        n_gates_eff = min(n_gates, 2 if cols >= 12 else 1)
        line_cols: list[int] = []
        if n_gates_eff > 0:
            usable = list(range(3, cols - 3))
            if n_gates_eff == 1:
                if not usable:
                    continue
                line_cols = [rng.choice(usable)]
            else:
                if len(usable) < 4:
                    continue
                j1 = rng.choice(usable[: len(usable) // 2])
                right = [j for j in usable if j >= j1 + 3]
                if not right:
                    continue
                line_cols = [j1, rng.choice(right)]
        gate_kinds = []
        for _ in line_cols:
            kind = rng.choice(["key", "key", "switch"])
            if kind == "switch" and "switch" in gate_kinds:
                kind = "key"  # only one switch token exists; two 'S' doors would share it
            gate_kinds.append(kind)
        key_letters = iter(["a", "b"])
        door_cells: list[tuple[int, int]] = []
        for j, kind in zip(line_cols, gate_kinds):
            for r in range(1, rows - 1):
                g[r][j] = "#"
            gap = rng.randrange(1, rows - 1)
            g[gap][j] = "S" if kind == "switch" else next(key_letters).upper()
            door_cells.append((gap, j))

        # random interior clutter walls (never on gate lines)
        density = rng.uniform(0.08, 0.2)
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if g[r][c] == "." and c not in line_cols and rng.random() < density:
                    g[r][c] = "#"

        bands = [0] + line_cols + [cols - 1]

        def band_floor(i: int) -> list[tuple[int, int]]:
            return [
                (r, c)
                for r in range(1, rows - 1)
                for c in range(bands[i] + 1, bands[i + 1])
                if g[r][c] == "."
            ]

        floors0 = band_floor(0)
        floors_last = band_floor(len(bands) - 2)
        if not floors0 or not floors_last:
            continue
        pr, pc = rng.choice(floors0)
        g[pr][pc] = "P"
        goal_choices = [cell for cell in floors_last if cell != (pr, pc)]
        if not goal_choices:
            continue
        gr, gc = rng.choice(goal_choices)
        g[gr][gc] = "G"

        # key/switch for gate i lives in band i (before its door)
        ok = True
        key_letters2 = iter(["a", "b"])
        for i, kind in enumerate(gate_kinds):
            cells = band_floor(i)
            if not cells:
                ok = False
                break
            r, c = rng.choice(cells)
            if kind == "switch":
                g[r][c] = "s"
            else:
                g[r][c] = next(key_letters2)
        if not ok:
            continue

        # hazards on leftover floor
        floor_cells = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] == "."]
        rng.shuffle(floor_cells)
        for r, c in floor_cells[:n_hazards]:
            g[r][c] = "x"

        grid = ["".join(row) for row in g]
        path = _nav_bfs(grid, _GATE_TOKENS)
        if path is None or len(path) < max(6, (rows + cols) // 3):
            continue
        # every gate must actually gate: goal unreachable with doors closed
        if line_cols and (gr, gc) in _nav_reachable_no_gates(grid, (pr, pc)):
            continue
        return grid
    raise RuntimeError("nav level sampling failed")


def sample_nav(rng: random.Random) -> dict[str, Any]:
    cell_px = rng.choice([3, 4])
    max_cells = 64 // cell_px
    rows = rng.randint(9, min(14, max_cells))
    cols = rng.randint(9, min(14, max_cells))
    n_levels = rng.randint(3, 6)
    gates_ramp = [0, 1, 1, 2, 2, 2]
    levels = []
    for li in range(n_levels):
        n_gates = min(gates_ramp[li] + rng.choice([0, 0, 1]), 2)
        n_hazards = rng.randint(0, 2 + li)
        levels.append({"map": _sample_nav_level(rng, rows, cols, n_gates, n_hazards)})

    bg, wall, player, goal, hazard, key_a, key_b, door_a, door_b, sw, sw_on, letterbox = _pick_colors(rng, 12)
    hud = rng.random() < 0.6
    return {
        "family": "nav",
        "cell_px": cell_px,
        "rows": rows,
        "cols": cols,
        "hud": hud,
        "colors": {
            "bg": bg,
            "wall": wall,
            "player": player,
            "goal": goal,
            "hazard": hazard,
            "key_a": key_a,
            "key_b": key_b,
            "door_a": door_a,
            "door_b": door_b,
            "switch": sw,
            "switch_on": sw_on,
            "letterbox": letterbox,
            "hud_fill": wall,
            "hud_empty": bg,
        },
        "player_shape": _sprite_masks(rng, cell_px),
        "levels": levels,
        "available_actions": [1, 2, 3, 4],
    }


def solve_nav(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-level ground-truth action lists."""
    out = []
    for level in spec["levels"]:
        path = _nav_bfs(level["map"], _GATE_TOKENS)
        assert path is not None, "unsolvable nav level escaped sampling"
        actions = []
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            actions.append({"id": DIR_ACTIONS[(r1 - r0, c1 - c0)]})
        out.append(actions)
    return out


# ---------------------------------------------------------------------------
# family: click
# ---------------------------------------------------------------------------


def _place_objects(
    rng: random.Random, n: int, size: int, grid: int, forbidden: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]] | None:
    """Non-overlapping (x, y) top-left positions with a 1px margin, or None if
    the layout does not fit (caller retries with fewer objects)."""
    placed: list[tuple[int, int]] = []
    lo, hi = 1, grid - size - 2
    for _ in range(4000):
        if len(placed) == n:
            return placed
        x, y = rng.randint(lo, hi), rng.randint(lo, hi)
        box = (x - 1, y - 1, size + 2, size + 2)
        clash = any(
            box[0] < fx + fw and fx < box[0] + box[2] and box[1] < fy + fh and fy < box[1] + box[3]
            for fx, fy, fw, fh in forbidden
        ) or any(abs(x - px) < size + 2 and abs(y - py) < size + 2 for px, py in placed)
        if not clash:
            placed.append((x, y))
    return None


def sample_click(rng: random.Random) -> dict[str, Any]:
    grid = rng.choice([32, 64, 64])
    obj_size = rng.choice([4, 5]) if grid == 32 else rng.choice([5, 6, 7])
    rule = rng.choice(["match_color", "match_shape", "odd_one_out"])
    n_levels = rng.randint(3, 6)
    gate = rng.random() < 0.5
    hud = rng.random() < 0.5

    bg, legend_frame, cue_neutral, gate_off, gate_on, life = _pick_colors(rng, 6)
    obj_pool = [c for c in range(16) if c not in {bg, legend_frame, cue_neutral, gate_off, gate_on, life}]

    legend_needed = rule in ("match_color", "match_shape")
    legend_rect = (0, 0, obj_size + 6, obj_size + 6) if legend_needed else None
    gate_px = max(4, obj_size - 1)
    # gate sprite top-left (single source of truth for sampler/solver/template)
    gate_xy = (grid - gate_px - 2, grid - gate_px - 2)
    gate_rect = (gate_xy[0] - 1, gate_xy[1] - 1, gate_px + 2, gate_px + 2) if gate else None
    lives_rect = (grid - 10, 0, 10, 4)

    levels = []
    for li in range(n_levels):
        n_objects = min(5 + li + rng.randint(0, 2), 6 if grid == 32 else 12)
        forbidden = [r for r in (legend_rect, gate_rect, lives_rect) if r]
        forbidden.append((0, grid - 3, grid, 3))  # HUD strip
        positions = None
        while positions is None:
            positions = _place_objects(rng, n_objects, obj_size, grid, forbidden)
            if positions is None:
                if n_objects <= 4:
                    raise RuntimeError("click layout infeasible even with 4 objects")
                n_objects -= 1  # crowded layout: retry with fewer objects
        if rule == "odd_one_out":
            n_targets = 1
        else:
            n_targets = rng.randint(2, min(4, n_objects - 2))

        target_color = rng.choice(obj_pool)
        distract_colors = [c for c in obj_pool if c != target_color]
        target_shape = rng.choice(SHAPE_NAMES)
        distract_shapes = [s for s in SHAPE_NAMES if s != target_shape]

        objects = []
        for i, (x, y) in enumerate(positions):
            is_target = i < n_targets
            if rule == "match_color":
                color = target_color if is_target else rng.choice(distract_colors)
                shape = rng.choice(SHAPE_NAMES)
            elif rule == "match_shape":
                shape = target_shape if is_target else rng.choice(distract_shapes)
                color = rng.choice(obj_pool)
            else:  # odd_one_out: all same shape+color except the target's color
                shape = target_shape
                color = target_color if is_target else distract_colors[li % len(distract_colors)]
            objects.append({"x": x, "y": y, "shape": shape, "color": color, "target": is_target})
        rng.shuffle(objects)

        cue = None
        if rule == "match_color":
            cue = {"kind": "color", "color": target_color}
        elif rule == "match_shape":
            cue = {"kind": "shape", "shape": target_shape, "color": cue_neutral}
        levels.append({"objects": objects, "cue": cue})

    return {
        "family": "click",
        "grid": grid,
        "obj_size": obj_size,
        "rule": rule,
        "gate": gate,
        "gate_px": gate_px,
        "gate_xy": list(gate_xy),
        "shape_masks": {n: _named_mask(n, obj_size) for n in SHAPE_NAMES},
        "hud": hud,
        "lives": 3,
        "colors": {
            "bg": bg,
            "legend_frame": legend_frame,
            "cue_neutral": cue_neutral,
            "gate_off": gate_off,
            "gate_on": gate_on,
            "life": life,
            "letterbox": bg,
            "hud_fill": legend_frame,
            "hud_empty": bg,
        },
        "levels": levels,
        "available_actions": [6],
    }


def _click_display_point(spec: dict[str, Any], gx: int, gy: int) -> tuple[int, int]:
    scale, ox, oy = _display_scale_offset(spec["grid"], spec["grid"])
    return gx * scale + ox, gy * scale + oy


def solve_click(spec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    s = spec["obj_size"]
    for level in spec["levels"]:
        actions = []
        if spec["gate"]:
            gp = spec["gate_px"]
            gx, gy = spec["gate_xy"]
            dx, dy = _click_display_point(spec, gx + gp // 2, gy + gp // 2)
            actions.append({"id": "ACTION6", "x": dx, "y": dy})
        targets = sorted(
            (o for o in level["objects"] if o["target"]), key=lambda o: (o["y"], o["x"])
        )
        for o in targets:
            mask = _named_mask(o["shape"], s)
            # click a pixel that is actually part of the shape (center may be a hole)
            best = min(
                ((r, c) for r in range(s) for c in range(s) if mask[r][c]),
                key=lambda rc: abs(rc[0] - s // 2) + abs(rc[1] - s // 2),
            )
            dx, dy = _click_display_point(spec, o["x"] + best[1], o["y"] + best[0])
            actions.append({"id": "ACTION6", "x": dx, "y": dy})
        out.append(actions)
    return out


# ---------------------------------------------------------------------------
# family: push
# ---------------------------------------------------------------------------


def _push_bfs(
    walls: set[tuple[int, int]],
    rows: int,
    cols: int,
    player: tuple[int, int],
    boxes: tuple[tuple[int, int], ...],
    pads: frozenset[tuple[int, int]],
    max_states: int = 400_000,
) -> list[tuple[int, int]] | None:
    """BFS over (player, boxes). Returns list of moves (dr, dc) or None."""
    start = (player, tuple(sorted(boxes)))
    prev: dict[Any, Any] = {start: None}
    q = deque([start])
    n = 0
    while q:
        state = q.popleft()
        (pr, pc), bxs = state
        if pads.issubset(bxs):
            moves = []
            node = state
            while prev[node] is not None:
                node, mv = prev[node]
                moves.append(mv)
            return moves[::-1]
        n += 1
        if n > max_states:
            return None
        bset = set(bxs)
        for dr, dc in DIRS:
            nr, nc = pr + dr, pc + dc
            if (nr, nc) in walls or not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in bset:
                br, bc = nr + dr, nc + dc
                if (br, bc) in walls or (br, bc) in bset or not (0 <= br < rows and 0 <= bc < cols):
                    continue
                nbxs = tuple(sorted(bset - {(nr, nc)} | {(br, bc)}))
            else:
                nbxs = bxs
            nstate = ((nr, nc), nbxs)
            if nstate not in prev:
                prev[nstate] = (state, (dr, dc))
                q.append(nstate)
    return None


def _walk_reachable(
    walls: set[tuple[int, int]], rows: int, cols: int, start: tuple[int, int], boxes: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls and (nr, nc) not in boxes:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return seen


def _sample_push_level(
    rng: random.Random, rows: int, cols: int, n_boxes: int, n_pulls: int
) -> dict[str, Any]:
    for _attempt in range(400):
        walls = set()
        for r in range(rows):
            walls |= {(r, 0), (r, cols - 1)}
        for c in range(cols):
            walls |= {(0, c), (rows - 1, c)}
        interior = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
        for cell in interior:
            if rng.random() < 0.08:
                walls.add(cell)
        floor = [c for c in interior if c not in walls]
        if len(floor) < n_boxes * 4:
            continue
        pads = rng.sample(floor, n_boxes)
        boxes = set(pads)
        # player next to some box
        adj = [
            (r + dr, c + dc)
            for (r, c) in boxes
            for dr, dc in DIRS
            if (r + dr, c + dc) not in walls and (r + dr, c + dc) not in boxes
            and 0 < r + dr < rows - 1 and 0 < c + dc < cols - 1
        ]
        if not adj:
            continue
        player = rng.choice(adj)

        # reverse play: pulls (with connectivity-checked repositioning walks)
        pulled = 0
        for _ in range(n_pulls * 6):
            if pulled >= n_pulls:
                break
            box = rng.choice(sorted(boxes))
            dr, dc = rng.choice(DIRS)
            hand = (box[0] + dr, box[1] + dc)      # player stands here to pull
            dest = (box[0] + 2 * dr, box[1] + 2 * dc)  # player retreats here
            if any(
                p in walls or p in boxes or not (0 < p[0] < rows - 1 and 0 < p[1] < cols - 1)
                for p in (hand, dest)
            ):
                continue
            if hand not in _walk_reachable(walls, rows, cols, player, boxes):
                continue
            boxes.remove(box)
            boxes.add(hand)
            player = dest
            pulled += 1
        if pulled < max(2, n_pulls // 3):
            continue
        if set(pads).issubset(boxes):
            continue  # already solved
        # the map encodes the player as 'P', which would overwrite a pad cell:
        # walk the player off any pad before serializing
        if player in pads:
            offs = [
                (player[0] + dr, player[1] + dc)
                for dr, dc in DIRS
                if (player[0] + dr, player[1] + dc) not in walls
                and (player[0] + dr, player[1] + dc) not in boxes
                and (player[0] + dr, player[1] + dc) not in pads
                and 0 < player[0] + dr < rows - 1
                and 0 < player[1] + dc < cols - 1
            ]
            if not offs:
                continue
            player = rng.choice(offs)
        moves = _push_bfs(walls, rows, cols, player, tuple(boxes), frozenset(pads))
        if moves is None or len(moves) < 3 * n_boxes:
            continue
        g = [["." for _ in range(cols)] for _ in range(rows)]
        for r, c in walls:
            g[r][c] = "#"
        for r, c in pads:
            g[r][c] = "T"
        for r, c in boxes:
            g[r][c] = "O" if (r, c) in pads else "B"
        g[player[0]][player[1]] = "P"
        return {"map": ["".join(row) for row in g], "_moves": moves}
    raise RuntimeError("push level sampling failed")


def sample_push(rng: random.Random) -> dict[str, Any]:
    cell_px = rng.choice([4, 5, 6])
    max_cells = 64 // cell_px
    rows = rng.randint(7, min(10, max_cells))
    cols = rng.randint(7, min(10, max_cells))
    n_levels = rng.randint(3, 6)
    levels = []
    for li in range(n_levels):
        n_boxes = min(1 + li // 2 + rng.choice([0, 1]), 3)
        n_pulls = 6 + 3 * li + rng.randint(0, 6)
        levels.append(_sample_push_level(rng, rows, cols, n_boxes, n_pulls))

    bg, wall, player, box, box_on, pad, letterbox = _pick_colors(rng, 7)
    return {
        "family": "push",
        "cell_px": cell_px,
        "rows": rows,
        "cols": cols,
        "hud": rng.random() < 0.6,
        "colors": {
            "bg": bg,
            "wall": wall,
            "player": player,
            "box": box,
            "box_on": box_on,
            "pad": pad,
            "letterbox": letterbox,
            "hud_fill": wall,
            "hud_empty": bg,
        },
        "player_shape": _sprite_masks(rng, cell_px),
        "levels": levels,
        "available_actions": [1, 2, 3, 4],
    }


def solve_push(spec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for level in spec["levels"]:
        moves = level.get("_moves")
        if moves is None:
            grid = level["map"]
            rows, cols = len(grid), len(grid[0])
            walls, pads, boxes, player = set(), set(), set(), None
            for r in range(rows):
                for c in range(cols):
                    ch = grid[r][c]
                    if ch == "#":
                        walls.add((r, c))
                    elif ch == "T":
                        pads.add((r, c))
                    elif ch == "B":
                        boxes.add((r, c))
                    elif ch == "O":
                        pads.add((r, c))
                        boxes.add((r, c))
                    elif ch == "P":
                        player = (r, c)
            moves = _push_bfs(walls, rows, cols, player, tuple(boxes), frozenset(pads))
            assert moves is not None
        out.append([{"id": DIR_ACTIONS[m]} for m in moves])
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_SAMPLERS = {"nav": sample_nav, "click": sample_click, "push": sample_push}
_SOLVERS = {"nav": solve_nav, "click": solve_click, "push": solve_push}


def sample_game(family: str, index: int, seed: int) -> dict[str, Any]:
    """Sample one game spec. Deterministic in (family, index, seed)."""
    rng = random.Random(f"synthgen-{family}-{index}-{seed}")
    spec = _SAMPLERS[family](rng)
    spec["game_id"] = game_id_for(family, index, seed)
    spec["index"] = index
    spec["seed"] = seed
    per_level = _SOLVERS[family](spec)
    # budget: generous per-level action budgets derived from the solution
    spec["budget"] = [max(20, len(a) * 4) for a in per_level]
    spec["solution_levels"] = per_level
    for level in spec["levels"]:
        level.pop("_moves", None)
    return spec


def solution_trace(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat ground-truth trace (excluding the initial RESET)."""
    return [a for level in spec["solution_levels"] for a in level]
