import numpy as np

def _copy(g):
    return [row[:] for row in g]

def init_state(entry_grid):
    return {"step": 0}

def _count3(g):
    return sum(1 for x in range(64) if g[63][x] == 3)

def _set_count(g, n):
    for x in range(64):
        g[63][x] = 3 if x < n else 0

def _region_all(g, r0, r1, c0, c1, v):
    return all(g[r][c] == v for r in range(r0, r1) for c in range(c0, c1))

def _fill_rect(g, r0, r1, c0, c1, v):
    for r in range(r0, r1):
        for c in range(c0, c1):
            g[r][c] = v

def _toggle8(g):
    # rectangle A: rows 30-33 cols 12-17 (4x6), rectangle B: rows 24-29 cols 18-21 (6x4)
    if _region_all(g, 30, 34, 12, 18, 8):
        _fill_rect(g, 30, 34, 12, 18, 4)
        _fill_rect(g, 24, 30, 18, 22, 8)
    else:
        _fill_rect(g, 24, 30, 18, 22, 4)
        _fill_rect(g, 30, 34, 12, 18, 8)

def _fill_moats(g):
    for r in range(16, 23):
        for c in range(41, 56):
            if g[r][c] == 0:
                g[r][c] = 5
    for r in range(33, 40):
        for c in range(41, 56):
            if g[r][c] == 0:
                g[r][c] = 5

def _toggle9(g):
    # S: solid block rows 20-23 cols 18-21 ; C: checker rows 34-37 cols 8-11
    if _region_all(g, 20, 24, 18, 22, 9):
        for r in range(20, 24):
            for c in range(18, 22):
                if (r + c) % 2 == 0:
                    g[r][c] = 4
        _fill_rect(g, 34, 38, 8, 12, 9)
    else:
        _fill_rect(g, 20, 24, 18, 22, 9)
        for r in range(34, 38):
            for c in range(8, 12):
                if (r + c) % 2 == 0:
                    g[r][c] = 4
                else:
                    g[r][c] = 9

def predict(state, grid, action, x=None, y=None):
    g = _copy(grid)
    step = state.get("step", 0)

    moat_filled = all(g[r][c] != 0 for r in range(16, 23) for c in range(41, 56))
    inc = moat_filled or (step % 2 == 0)

    if action == 0:
        _set_count(g, 0)
        return g, {"level_up": False, "dead": False, "win": False}, {"step": 0}

    if action in (1, 2, 3, 4):
        dr = {1: -2, 2: 2, 3: 0, 4: 0}[action]
        dc = {1: 0, 2: 0, 3: -2, 4: 2}[action]
        cells = [(r, c) for r in range(64) for c in range(64) if g[r][c] == 14]
        if cells:
            ok = True
            for (r, c) in cells:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 64 and 0 <= nc < 64):
                    ok = False
                    break
                if g[nr][nc] not in (2, 14):
                    ok = False
                    break
            if ok:
                for (r, c) in cells:
                    g[r][c] = 2
                for (r, c) in cells:
                    g[r + dr][c + dc] = 14

    elif action == 6:
        col = g[y][x]
        if col in (0, 2, 5, 8, 9):
            inc = True
        if 16 <= y <= 22 and 41 <= x <= 55:
            _toggle8(g)
            _fill_moats(g)
        elif 33 <= y <= 39 and 41 <= x <= 55:
            _toggle9(g)

    if inc:
        _set_count(g, _count3(g) + 1)

    return g, {"level_up": False, "dead": False, "win": False}, {"step": step + 1}

def step(grid, action, x=None, y=None):
    st = {"step": _count3(grid) * 0}
    g, flags, _ = predict(st, grid, action, x, y)
    return g, flags

def is_goal(state, grid):
    return False
