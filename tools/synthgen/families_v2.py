"""v2 mechanic families mapped to the 2026-08-03 unlock-failure diagnosis.

New families (all via arcengine primitives, mechanical reference solvers):
    replay -- F2 record-replay tool (g50t-class): ACTION5 banks the walked path
              as a ghost that replays it move-for-move; the ghost parks on a
              pressure plate that holds a door open. The bank LOOKS like a
              punishing reset (avatar teleports to start).
    carry  -- F4 grab-carry with facing (wa30-class): arrows set facing even
              when blocked; ACTION5 grabs the faced block (it vanishes into
              carry) or releases it into the faced cell. Win: every block on a
              zone cell, hands empty.
    mirror -- F5 coupled avatars (m0r0-class): every arrow moves BOTH dots, one
              axis negated for the twin; walls block each independently (the
              only way to change the offset). Win: both on the same cell.
    rules  -- F7 symbolic rule table (tr87-class): legend of LHS=RHS glyph
              rules; cursor+cycle keys edit a working row until
              working[i] == rule(target[i]) for all cells.

Each family ships a pure-python simulator (XxxSim) whose semantics are
mirrored EXACTLY by the rendered game template (render_v2.py); solvers and the
scripted-teacher episodes (episodes.py) run against the sim, and validate
replays prove sim == engine.

v1 families (nav/click/push) remain available through sample_game_v2 unchanged
apart from v2-style episodes; see episodes.py.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from families import (
    DIR_ACTIONS,
    DIRS,
    _named_mask,
    _pick_colors,
    _sprite_masks,
    sample_game as sample_game_v1,
)

V2_FAMILIES = ("replay", "carry", "mirror", "rules")
V1_FAMILIES = ("nav", "click", "push")
ALL_FAMILIES = V1_FAMILIES + V2_FAMILIES

FAMILY_PREFIX_V2 = {"replay": "xr", "carry": "xg", "mirror": "xm", "rules": "xs"}

DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
DIR_NAME = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
ACTION_OF_DELTA = {v: k for k, v in DELTA.items()}

GLYPH_ALPHABET = ("full", "ring", "plus", "diamond", "notch", "tee", "u")


def game_id_v2(family: str, index: int, seed: int) -> str:
    import hashlib

    base = f"{FAMILY_PREFIX_V2[family]}{index % 100:02d}"
    version = hashlib.md5(f"synthgen2-{family}-{index}-{seed}".encode()).hexdigest()[:8]
    return f"{base}-{version}"


def _find(grid: list[str], ch: str) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == ch]


def _bfs_path(
    passable, start: tuple[int, int], goal: tuple[int, int], rows: int, cols: int
) -> list[tuple[int, int]] | None:
    """Plain grid BFS; `passable(cell)` decides walkability. Path incl. start."""
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if not (0 <= n[0] < rows and 0 <= n[1] < cols) or n in prev:
                continue
            if not passable(n):
                continue
            prev[n] = (r, c)
            if n == goal:
                path = [n]
                while path[-1] is not None:
                    path.append(prev[path[-1]])
                return path[-2::-1] if path[-1] is None else path[::-1]
            q.append(n)
    return None


def _path_actions(path: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {"id": DIR_ACTIONS[(r1 - r0, c1 - c0)]}
        for (r0, c0), (r1, c1) in zip(path, path[1:])
    ]


# ===========================================================================
# family: replay (F2, g50t-class)
# ===========================================================================


class ReplaySim:
    """Exact model of the rendered replay game (see render_v2._REPLAY_BODY).

    map chars: '#' wall, '.' floor, 'P' start, 'L' plate, 'D' door, 'G' goal.
    Semantics:
      * arrows move the avatar; walls block; the door cell blocks unless the
        plate is occupied (by avatar or ghost) at decision time.
      * successful pre-bank moves are recorded.
      * ACTION5 (once, with a non-empty recording): banks the recording as a
        ghost at the start cell and teleports the avatar back to start.
      * on every subsequent action (arrow OR no-op) the ghost first advances
        one step along its recorded path, then parks forever at its end.
      * win: avatar reaches 'G'.
    """

    def __init__(self, grid: list[str]) -> None:
        self.grid = grid
        self.rows, self.cols = len(grid), len(grid[0])
        self.start = _find(grid, "P")[0]
        self.plate = _find(grid, "L")[0]
        self.goal = _find(grid, "G")[0]
        self.doors = set(_find(grid, "D"))
        self.pos = self.start
        self.recorded: list[tuple[int, int]] = []
        self.banked = False
        self.ghost_path: list[tuple[int, int]] = []
        self.ghost_pos: tuple[int, int] | None = None
        self.ghost_idx = 0
        self.won = False

    def _wall(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return True
        return self.grid[r][c] == "#"

    def plate_occupied(self) -> bool:
        return self.pos == self.plate or (self.banked and self.ghost_pos == self.plate)

    def door_open(self) -> bool:
        return self.plate_occupied()

    def step(self, action_id: int) -> dict[str, Any]:
        ev: dict[str, Any] = {"moved": False, "banked": False, "blocked_by": None}
        if self.won:
            return ev
        if action_id == 5:
            if not self.banked and self.recorded:
                self.banked = True
                self.ghost_path = list(self.recorded)
                self.ghost_pos = self.start
                self.ghost_idx = 0
                self.pos = self.start
                ev["banked"] = True
            return ev
        d = DELTA.get(action_id)
        if d is None:
            return ev
        if self.banked and self.ghost_idx < len(self.ghost_path):
            gd = self.ghost_path[self.ghost_idx]
            self.ghost_pos = (self.ghost_pos[0] + gd[0], self.ghost_pos[1] + gd[1])
            self.ghost_idx += 1
            ev["ghost_moved"] = True
        t = (self.pos[0] + d[0], self.pos[1] + d[1])
        if self._wall(t):
            ev["blocked_by"] = "wall"
            return ev
        if t in self.doors and not self.plate_occupied():
            ev["blocked_by"] = "door"
            return ev
        self.pos = t
        ev["moved"] = True
        if not self.banked:
            self.recorded.append(d)
        if t == self.goal:
            self.won = True
            ev["won"] = True
        return ev


def _sample_replay_level(rng: random.Random, rows: int, cols: int) -> list[str]:
    for _ in range(600):
        g = [["." for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            g[r][0] = g[r][cols - 1] = "#"
        for c in range(cols):
            g[0][c] = g[rows - 1][c] = "#"
        wcol = rng.randint(cols - 5, cols - 4)
        door_row = rng.randint(1, rows - 2)
        for r in range(1, rows - 1):
            g[r][wcol] = "#"
        g[door_row][wcol] = "D"
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if c == wcol or g[r][c] != ".":
                    continue
                if rng.random() < (0.10 if c < wcol else 0.05):
                    g[r][c] = "#"
        left = [(r, c) for r in range(1, rows - 1) for c in range(1, wcol) if g[r][c] == "."]
        right = [
            (r, c) for r in range(1, rows - 1) for c in range(wcol + 1, cols - 1) if g[r][c] == "."
        ]
        if len(left) < 8 or not right:
            continue
        rng.shuffle(left)
        start = left[0]
        door = (door_row, wcol)
        plates = [
            p
            for p in left[1:]
            if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= 3
            and abs(p[0] - door[0]) + abs(p[1] - door[1]) >= 2
        ]
        if not plates:
            continue
        plate = rng.choice(plates)
        goal = rng.choice(right)
        g[start[0]][start[1]] = "P"
        g[plate[0]][plate[1]] = "L"
        g[goal[0]][goal[1]] = "G"
        grid = ["".join(row) for row in g]

        blocked = {"#", "D"}

        def passable(cell):
            return grid[cell[0]][cell[1]] not in blocked

        # reachability: start->plate, start->left-door-neighbour, door->goal
        if _bfs_path(passable, start, plate, rows, cols) is None:
            continue
        dn_left = (door_row, wcol - 1)
        dn_right = (door_row, wcol + 1)
        if grid[dn_left[0]][dn_left[1]] == "#" or grid[dn_right[0]][dn_right[1]] == "#":
            continue
        if _bfs_path(passable, start, dn_left, rows, cols) is None:
            continue
        if _bfs_path(passable, dn_right, goal, rows, cols) is None:
            continue
        # a free non-door neighbour of start for filler wiggles
        if not any(
            grid[start[0] + dr][start[1] + dc] in ".L"
            for dr, dc in DIRS
        ):
            continue
        return grid
    raise RuntimeError("replay level sampling failed")


def solve_replay_level(grid: list[str]) -> list[dict[str, Any]]:
    """Clean reference solution: record start->plate, bank, (fillers), walk to
    goal through the ghost-held door. Verified against ReplaySim."""
    sim = ReplaySim(grid)
    rows, cols = sim.rows, sim.cols

    def passable_closed(cell):
        return grid[cell[0]][cell[1]] not in ("#", "D")

    def passable_open(cell):
        return grid[cell[0]][cell[1]] != "#"

    p1 = _bfs_path(passable_closed, sim.start, sim.plate, rows, cols)
    assert p1 is not None
    actions = _path_actions(p1)
    actions.append({"id": "ACTION5"})
    ghost_len = len(p1) - 1

    p2 = _bfs_path(passable_open, sim.start, sim.goal, rows, cols)
    assert p2 is not None
    door_cell = next(iter(sim.doors))
    k = p2.index(door_cell)  # actions taken when ENTERING the door cell
    fillers = max(0, ghost_len - k)
    if fillers % 2:
        fillers += 1
    filler_dir = next(
        (dr, dc)
        for dr, dc in DIRS
        if grid[sim.start[0] + dr][sim.start[1] + dc] in ".L"
    )
    back = (-filler_dir[0], -filler_dir[1])
    for i in range(fillers):
        actions.append({"id": DIR_ACTIONS[filler_dir if i % 2 == 0 else back]})
    actions.extend(_path_actions(p2))

    # verify against the sim
    check = ReplaySim(grid)
    for a in actions:
        check.step(5 if a["id"] == "ACTION5" else int(a["id"][-1]))
    assert check.won, "replay reference solution failed in sim"
    return actions


def sample_replay(rng: random.Random) -> dict[str, Any]:
    rows = rng.randint(9, 12)
    cols = rng.randint(12, 14)
    cell_px = 64 // max(rows, cols)
    cell_px = min(cell_px, 5)
    n_levels = rng.randint(2, 3)
    levels = [{"map": _sample_replay_level(rng, rows, cols)} for _ in range(n_levels)]
    bg, wall, player, ghost, plate, door, goal, letterbox = _pick_colors(rng, 8)
    return {
        "family": "replay",
        "cell_px": cell_px,
        "rows": rows,
        "cols": cols,
        "hud": rng.random() < 0.5,
        "colors": {
            "bg": bg,
            "wall": wall,
            "player": player,
            "ghost": ghost,
            "plate": plate,
            "door": door,
            "goal": goal,
            "letterbox": letterbox,
            "hud_fill": wall,
            "hud_empty": bg,
        },
        "player_shape": _sprite_masks(rng, cell_px),
        "levels": levels,
        "available_actions": [1, 2, 3, 4, 5],
    }


def solve_replay(spec: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [solve_replay_level(lvl["map"]) for lvl in spec["levels"]]


# ===========================================================================
# family: carry (F4, wa30-class)
# ===========================================================================


class CarrySim:
    """Exact model of the rendered carry game.

    map chars: '#' wall, '.' floor, 'P' avatar, 'B' block, 'Z' zone cell.
    Semantics:
      * arrows: facing := direction ALWAYS (even when blocked); move if the
        target cell is not wall/out-of-bounds/another (un-carried) block.
      * ACTION5 empty-handed: if the faced-adjacent cell holds a block, the
        block vanishes into carry.
      * ACTION5 carrying: if the faced-adjacent cell is free floor (no wall,
        no block), the block reappears there.
      * win: every block sits on a zone cell and nothing is carried.
    """

    def __init__(self, grid: list[str]) -> None:
        self.grid = grid
        self.rows, self.cols = len(grid), len(grid[0])
        self.pos = _find(grid, "P")[0]
        self.blocks = _find(grid, "B")
        self.zone = set(_find(grid, "Z"))
        self.facing: tuple[int, int] | None = None
        self.carrying: int | None = None
        self.won = False

    def _wall(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return True
        return self.grid[r][c] == "#"

    def _block_at(self, cell: tuple[int, int]) -> int | None:
        for i, b in enumerate(self.blocks):
            if i != self.carrying and b == cell:
                return i
        return None

    def _check_win(self) -> None:
        if self.carrying is None and all(
            b in self.zone for i, b in enumerate(self.blocks)
        ):
            self.won = True

    def step(self, action_id: int) -> dict[str, Any]:
        ev: dict[str, Any] = {"moved": False, "grabbed": None, "released": None, "blocked_by": None}
        if self.won:
            return ev
        d = DELTA.get(action_id)
        if d is not None:
            self.facing = d
            t = (self.pos[0] + d[0], self.pos[1] + d[1])
            if self._wall(t):
                ev["blocked_by"] = "wall"
            elif self._block_at(t) is not None:
                ev["blocked_by"] = "block"
            else:
                self.pos = t
                ev["moved"] = True
            return ev
        if action_id == 5 and self.facing is not None:
            t = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
            if self.carrying is None:
                b = self._block_at(t)
                if b is not None:
                    self.carrying = b
                    ev["grabbed"] = b
            else:
                if not self._wall(t) and self._block_at(t) is None:
                    self.blocks[self.carrying] = t
                    ev["released"] = self.carrying
                    self.carrying = None
                    self._check_win()
                    if self.won:
                        ev["won"] = True
        return ev


def _carry_reach(sim: CarrySim, start: tuple[int, int]) -> dict[str, Any]:
    obstacles = {b for i, b in enumerate(sim.blocks) if i != sim.carrying}

    def passable(cell):
        return not sim._wall(cell) and cell not in obstacles

    return passable


def solve_carry_from(sim: CarrySim) -> list[dict[str, Any]]:
    """Deliver every un-zoned block into a free zone cell, from the sim's
    CURRENT state (used both for clean solves and post-misstep recovery)."""
    actions: list[dict[str, Any]] = []

    def emit(a):
        actions.append(a)
        sim.step(5 if a["id"] == "ACTION5" else int(a["id"][-1]))

    for _guard in range(16):
        pending = [
            i
            for i, b in enumerate(sim.blocks)
            if i != sim.carrying and b not in sim.zone
        ]
        if sim.carrying is None and not pending:
            break
        if sim.carrying is None:
            # grab the closest pending block
            passable = _carry_reach(sim, sim.pos)
            best = None
            for i in pending:
                b = sim.blocks[i]
                for dr, dc in DIRS:
                    y = (b[0] - dr, b[1] - dc)  # stand here, face (dr,dc) at b
                    if sim._wall(y) or sim._block_at(y) is not None:
                        continue
                    path = _bfs_path(passable, sim.pos, y, sim.rows, sim.cols)
                    if path is None:
                        continue
                    if best is None or len(path) < len(best[0]):
                        best = (path, (dr, dc))
            assert best is not None, "carry solver: no reachable block"
            path, d = best
            for a in _path_actions(path):
                emit(a)
            emit({"id": DIR_ACTIONS[d]})  # bump into the block: sets facing
            emit({"id": "ACTION5"})  # grab
            assert sim.carrying is not None
        else:
            # release into a free zone cell via arrive-facing-it
            passable = _carry_reach(sim, sim.pos)
            free_zone = [
                z for z in sorted(sim.zone) if sim._block_at(z) is None
            ]
            best = None
            for z in free_zone:
                for dr, dc in DIRS:
                    y2 = (z[0] - dr, z[1] - dc)
                    p2 = (z[0] - 2 * dr, z[1] - 2 * dc)
                    if any(sim._wall(c) or sim._block_at(c) is not None for c in (y2, p2)):
                        continue
                    path = _bfs_path(passable, sim.pos, p2, sim.rows, sim.cols)
                    if path is None:
                        continue
                    if best is None or len(path) < len(best[0]):
                        best = (path, (dr, dc))
            assert best is not None, "carry solver: no reachable release slot"
            path, d = best
            for a in _path_actions(path):
                emit(a)
            emit({"id": DIR_ACTIONS[d]})  # step into y2, facing z
            emit({"id": "ACTION5"})  # release into z
            assert sim.carrying is None
    assert sim.won, "carry solver did not reach WIN"
    return actions


def _sample_carry_level(rng: random.Random, rows: int, cols: int, n_blocks: int) -> list[str]:
    for _ in range(600):
        g = [["." for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            g[r][0] = g[r][cols - 1] = "#"
        for c in range(cols):
            g[0][c] = g[rows - 1][c] = "#"
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if rng.random() < 0.08:
                    g[r][c] = "#"
        # zone: small rectangle, area >= n_blocks, in a random quadrant
        zr = rng.choice([1, 2])
        zc = max(n_blocks // zr + (n_blocks % zr > 0), 1)
        r0 = rng.randint(1, rows - 1 - zr)
        c0 = rng.randint(1, cols - 1 - zc)
        zone = [(r0 + i, c0 + j) for i in range(zr) for j in range(zc)]
        for r, c in zone:
            g[r][c] = "Z"
        floor = [
            (r, c)
            for r in range(1, rows - 1)
            for c in range(1, cols - 1)
            if g[r][c] == "."
        ]
        rng.shuffle(floor)
        far = [
            p
            for p in floor
            if min(abs(p[0] - z[0]) + abs(p[1] - z[1]) for z in zone) >= 3
        ]
        if len(far) < n_blocks + 1:
            continue
        blocks = far[:n_blocks]
        player = far[n_blocks]
        for r, c in blocks:
            g[r][c] = "B"
        g[player[0]][player[1]] = "P"
        grid = ["".join(row) for row in g]
        try:
            sim = CarrySim(grid)
            sol = solve_carry_from(sim)
        except (AssertionError, RuntimeError):
            continue
        if len(sol) < 6:
            continue
        return grid
    raise RuntimeError("carry level sampling failed")


def sample_carry(rng: random.Random) -> dict[str, Any]:
    rows = rng.randint(8, 11)
    cols = rng.randint(8, 12)
    cell_px = min(64 // max(rows, cols), 5)
    n_levels = rng.randint(2, 4)
    levels = []
    for li in range(n_levels):
        n_blocks = min(1 + li, 3)
        levels.append({"map": _sample_carry_level(rng, rows, cols, n_blocks)})
    bg, wall, player, nose, block, zone, letterbox = _pick_colors(rng, 7)
    return {
        "family": "carry",
        "cell_px": cell_px,
        "rows": rows,
        "cols": cols,
        "hud": rng.random() < 0.5,
        "colors": {
            "bg": bg,
            "wall": wall,
            "player": player,
            "nose": nose,
            "block": block,
            "zone": zone,
            "letterbox": letterbox,
            "hud_fill": wall,
            "hud_empty": bg,
        },
        "levels": levels,
        "available_actions": [1, 2, 3, 4, 5],
    }


def solve_carry(spec: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [solve_carry_from(CarrySim(lvl["map"])) for lvl in spec["levels"]]


# ===========================================================================
# family: mirror (F5, m0r0-class)
# ===========================================================================


class MirrorSim:
    """Exact model of the rendered mirror game.

    map chars: '#' wall, '.' floor, 'A' dot A, 'B' dot B.
    Every arrow moves BOTH dots: dot A by the plain delta, dot B by the delta
    with the mirrored axis negated ('x': columns negated, 'y': rows negated).
    Each dot moves only if its own target is free; a blocked dot stays (wall
    asymmetry is the only way to change the offset). No dot-dot collision.
    Win: both dots on the same cell after an action.
    """

    def __init__(self, grid: list[str], axis: str) -> None:
        self.grid = grid
        self.axis = axis
        self.rows, self.cols = len(grid), len(grid[0])
        self.a = _find(grid, "A")[0]
        self.b = _find(grid, "B")[0]
        self.won = False

    def _wall(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return True
        return self.grid[r][c] == "#"

    def twin_delta(self, d: tuple[int, int]) -> tuple[int, int]:
        return (d[0], -d[1]) if self.axis == "x" else (-d[0], d[1])

    def step(self, action_id: int) -> dict[str, Any]:
        ev: dict[str, Any] = {"a_moved": False, "b_moved": False}
        if self.won:
            return ev
        d = DELTA.get(action_id)
        if d is None:
            return ev
        db = self.twin_delta(d)
        ta = (self.a[0] + d[0], self.a[1] + d[1])
        tb = (self.b[0] + db[0], self.b[1] + db[1])
        if not self._wall(ta):
            self.a = ta
            ev["a_moved"] = True
        if not self._wall(tb):
            self.b = tb
            ev["b_moved"] = True
        if self.a == self.b:
            self.won = True
            ev["won"] = True
        return ev


def solve_mirror_from(sim: MirrorSim) -> list[dict[str, Any]] | None:
    """BFS over (a, b) joint states; returns the action list or None."""
    start = (sim.a, sim.b)
    prev: dict[Any, Any] = {start: None}
    q = deque([start])
    while q:
        state = q.popleft()
        a, b = state
        if a == b:
            acts = []
            node = state
            while prev[node] is not None:
                node, aid = prev[node]
                acts.append({"id": f"ACTION{aid}"})
            return acts[::-1]
        for aid, d in DELTA.items():
            db = sim.twin_delta(d)
            ta = (a[0] + d[0], a[1] + d[1])
            tb = (b[0] + db[0], b[1] + db[1])
            na = a if sim._wall(ta) else ta
            nb = b if sim._wall(tb) else tb
            ns = (na, nb)
            if ns not in prev:
                prev[ns] = (state, aid)
                q.append(ns)
    return None


def _sample_mirror_level(rng: random.Random, rows: int, cols: int, axis: str) -> list[str]:
    for _ in range(600):
        g = [["." for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            g[r][0] = g[r][cols - 1] = "#"
        for c in range(cols):
            g[0][c] = g[rows - 1][c] = "#"
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if rng.random() < 0.14:
                    g[r][c] = "#"
        floor = [
            (r, c)
            for r in range(1, rows - 1)
            for c in range(1, cols - 1)
            if g[r][c] == "."
        ]
        if len(floor) < 10:
            continue
        a, b = rng.sample(floor, 2)
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 4:
            continue
        g[a[0]][a[1]] = "A"
        g[b[0]][b[1]] = "B"
        grid = ["".join(row) for row in g]
        sol = solve_mirror_from(MirrorSim(grid, axis))
        if sol is None or not (6 <= len(sol) <= 40):
            continue
        return grid
    raise RuntimeError("mirror level sampling failed")


def sample_mirror(rng: random.Random) -> dict[str, Any]:
    rows = rng.randint(8, 12)
    cols = rng.randint(8, 12)
    cell_px = min(64 // max(rows, cols), 5)
    axis = rng.choice(["x", "x", "y"])  # x-negation is the m0r0 form
    n_levels = rng.randint(2, 4)
    levels = [{"map": _sample_mirror_level(rng, rows, cols, axis)} for _ in range(n_levels)]
    bg, wall, dot, letterbox = _pick_colors(rng, 4)
    return {
        "family": "mirror",
        "cell_px": cell_px,
        "rows": rows,
        "cols": cols,
        "axis": axis,
        "hud": rng.random() < 0.5,
        "colors": {
            "bg": bg,
            "wall": wall,
            "dot": dot,
            "letterbox": letterbox,
            "hud_fill": wall,
            "hud_empty": bg,
        },
        "levels": levels,
        "available_actions": [1, 2, 3, 4],
    }


def solve_mirror(spec: dict[str, Any]) -> list[list[dict[str, Any]]]:
    out = []
    for lvl in spec["levels"]:
        sol = solve_mirror_from(MirrorSim(lvl["map"], spec["axis"]))
        assert sol is not None
        out.append(sol)
    return out


# ===========================================================================
# family: rules (F7, tr87-class)
# ===========================================================================


class RulesSim:
    """Exact model of the rendered rules game.

    State: cursor index + working row of glyph names. ACTION3/4 move the
    cursor left/right (clamped); ACTION1/2 cycle the selected cell's glyph
    forward/backward through the fixed alphabet. A cell shows a mark when
    working[i] == rules[target[i]]. Win: all cells marked.
    """

    def __init__(self, rules: dict[str, str], target: list[str], working: list[str]) -> None:
        self.rules = rules
        self.target = list(target)
        self.working = list(working)
        self.cursor = 0
        self.n = len(target)
        self.won = False

    def correct(self, i: int) -> bool:
        return self.working[i] == self.rules[self.target[i]]

    def marks(self) -> list[bool]:
        return [self.correct(i) for i in range(self.n)]

    def step(self, action_id: int) -> dict[str, Any]:
        ev: dict[str, Any] = {"cursor": self.cursor, "changed": False}
        if self.won:
            return ev
        if action_id == 3:
            self.cursor = max(0, self.cursor - 1)
        elif action_id == 4:
            self.cursor = min(self.n - 1, self.cursor + 1)
        elif action_id in (1, 2):
            i = GLYPH_ALPHABET.index(self.working[self.cursor])
            i = (i + 1) % 7 if action_id == 1 else (i - 1) % 7
            self.working[self.cursor] = GLYPH_ALPHABET[i]
            ev["changed"] = True
        ev["cursor"] = self.cursor
        if all(self.marks()):
            self.won = True
            ev["won"] = True
        return ev


def _cycle_actions(cur: str, want: str) -> list[dict[str, Any]]:
    ci, wi = GLYPH_ALPHABET.index(cur), GLYPH_ALPHABET.index(want)
    up = (wi - ci) % 7
    if up <= 3:
        return [{"id": "ACTION1"}] * up
    return [{"id": "ACTION2"}] * (7 - up)


def solve_rules_from(sim: RulesSim) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    def emit(a):
        actions.append(a)
        sim.step(int(a["id"][-1]))

    for i in range(sim.n):
        while sim.cursor < i:
            emit({"id": "ACTION4"})
        while sim.cursor > i:
            emit({"id": "ACTION3"})
        for a in _cycle_actions(sim.working[i], sim.rules[sim.target[i]]):
            emit(a)
    assert sim.won, "rules solver did not reach WIN"
    return actions


def sample_rules(rng: random.Random) -> dict[str, Any]:
    n_rules = rng.randint(3, 4)
    lhs = rng.sample(GLYPH_ALPHABET, n_rules)
    # injective, identity-free mapping
    for _ in range(200):
        rhs = rng.sample(GLYPH_ALPHABET, n_rules)
        if all(a != b for a, b in zip(lhs, rhs)):
            break
    rules = dict(zip(lhs, rhs))
    n_levels = rng.randint(2, 3)
    levels = []
    for li in range(n_levels):
        n_cells = rng.randint(4, 5)
        target = [rng.choice(lhs) for _ in range(n_cells)]
        working = []
        for i, t in enumerate(target):
            forbidden = {rules[t]}
            if li == 0 and i == 0:
                forbidden.add(t)  # keep the copy-the-target misstep observable
            working.append(rng.choice([g for g in GLYPH_ALPHABET if g not in forbidden]))
        levels.append({"target": target, "working": working})
    colors = _pick_colors(rng, 7)
    bg, tcol, wcol, cursor, mark, sep, letterbox = colors
    return {
        "family": "rules",
        "grid": 64,
        "alphabet": list(GLYPH_ALPHABET),
        "rules": rules,
        "glyph_masks": {n: _named_mask(n, 5) for n in GLYPH_ALPHABET},
        "hud": False,
        "colors": {
            "bg": bg,
            "target_glyph": tcol,
            "working_glyph": wcol,
            "cursor": cursor,
            "mark": mark,
            "separator": sep,
            "letterbox": letterbox,
        },
        "levels": levels,
        "available_actions": [1, 2, 3, 4],
    }


def solve_rules(spec: dict[str, Any]) -> list[list[dict[str, Any]]]:
    out = []
    for lvl in spec["levels"]:
        sim = RulesSim(spec["rules"], lvl["target"], lvl["working"])
        out.append(solve_rules_from(sim))
    return out


# ===========================================================================
# public API
# ===========================================================================

_SAMPLERS_V2 = {
    "replay": sample_replay,
    "carry": sample_carry,
    "mirror": sample_mirror,
    "rules": sample_rules,
}
_SOLVERS_V2 = {
    "replay": solve_replay,
    "carry": solve_carry,
    "mirror": solve_mirror,
    "rules": solve_rules,
}


def sample_game_v2(family: str, index: int, seed: int) -> dict[str, Any]:
    """Sample one v2 game spec (deterministic in family/index/seed).
    v1 families delegate to the v1 sampler unchanged."""
    if family in V1_FAMILIES:
        return sample_game_v1(family, index, seed)
    rng = random.Random(f"synthgen2-{family}-{index}-{seed}")
    spec = _SAMPLERS_V2[family](rng)
    spec["game_id"] = game_id_v2(family, index, seed)
    spec["index"] = index
    spec["seed"] = seed
    per_level = _SOLVERS_V2[family](spec)
    # generous budgets: episodes include scripted missteps on top of the solve
    spec["budget"] = [max(80, len(a) * 6) for a in per_level]
    spec["solution_levels"] = per_level
    return spec


def make_sim(spec: dict[str, Any], level_idx: int):
    """Fresh simulator for one level of a v2-family spec."""
    family = spec["family"]
    lvl = spec["levels"][level_idx]
    if family == "replay":
        return ReplaySim(lvl["map"])
    if family == "carry":
        return CarrySim(lvl["map"])
    if family == "mirror":
        return MirrorSim(lvl["map"], spec["axis"])
    if family == "rules":
        return RulesSim(spec["rules"], lvl["target"], lvl["working"])
    raise ValueError(family)
