import copy

# ---------------------------------------------------------------------------
# dc22L2 level 0 world model.
#
# The visible grid contains:
#   * a 2x2 player sprite (color 14 / 'e') that moves in steps of 2,
#   * terrain colors (2 floor, 8/9/d/7 walkable, 4/3/5 walls/panels, 6, 11, 13),
#   * two toggle switches in the right panel clicked at (52,22) and (52,40),
#   * a progress bar drawn on row 63 (color 3 fills from the left).
#
# We keep a "base" terrain grid (the grid with the player sprite removed) so
# that the cells the player vacates are restored to their true color, exactly
# as the recordings show.
# ---------------------------------------------------------------------------

WALKABLE = {2, 6, 7, 8, 9, 11, 13}


def _find_player(g):
    for y in range(63):
        for x in range(63):
            if g[y][x] == 14:
                return (x, y)
    return (6, 30)


def _infer_under(g, x, y):
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, 0), (0, -1), (1, 0), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < 64 and 0 <= ny < 64 and g[ny][nx] != 14:
            return g[ny][nx]
    return 2


def init_state(entry_grid):
    g = [list(r) for r in entry_grid]
    px, py = _find_player(g)
    base = [list(r) for r in g]
    u = _infer_under(g, px, py)
    for yy in (py, py + 1):
        for xx in (px, px + 1):
            if 0 <= yy < 64 and 0 <= xx < 64:
                base[yy][xx] = u
    # switch states read from the grid itself
    tA = base[32][16] == 7
    tB = base[28][8] == 4
    cnt = 0
    for c in range(64):
        if base[63][c] == 3:
            cnt += 1
    return {"base": base, "px": px, "py": py, "tA": tA, "tB": tB, "cnt": cnt}


def _set_rect(g, y0, y1, x0, x1, v):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if 0 <= y < 64 and 0 <= x < 64:
                g[y][x] = v


def _toggleA_on(g):
    _set_rect(g, 40, 43, 8, 15, 4)
    _set_rect(g, 40, 43, 20, 27, 4)
    _set_rect(g, 32, 39, 16, 19, 7)
    _set_rect(g, 44, 51, 16, 19, 7)


def _toggleA_off(g):
    _set_rect(g, 32, 39, 16, 19, 4)
    _set_rect(g, 44, 51, 16, 19, 4)
    _set_rect(g, 40, 43, 8, 15, 7)
    _set_rect(g, 40, 43, 20, 27, 7)


def _toggleB_on(g):
    for y in range(28, 32):
        if (y - 28) % 2 == 0:
            g[y][8], g[y][9], g[y][10], g[y][11] = 4, 9, 4, 9
        else:
            g[y][8], g[y][9], g[y][10], g[y][11] = 9, 4, 9, 4
    _set_rect(g, 32, 39, 4, 7, 9)


def _toggleB_off(g):
    _set_rect(g, 28, 31, 8, 11, 9)
    for y in range(32, 40):
        if (y - 32) % 2 == 0:
            g[y][4], g[y][5], g[y][6], g[y][7] = 4, 9, 4, 9
        else:
            g[y][4], g[y][5], g[y][6], g[y][7] = 9, 4, 9, 4


def predict(state, grid, action, x=None, y=None):
    st = {k: (copy.deepcopy(v) if isinstance(v, list) else v) for k, v in state.items()}
    base = st["base"]
    inc = False

    if action == 6:
        if x == 52 and y == 22:
            st["tA"] = not st["tA"]
            if st["tA"]:
                _toggleA_on(base)
                inc = True
            else:
                _toggleA_off(base)
        elif x == 52 and y == 40:
            was = st["tB"]
            st["tB"] = not st["tB"]
            if st["tB"]:
                _toggleB_on(base)
            else:
                _toggleB_off(base)
            if was or st["cnt"] < 2:
                inc = True
        else:
            # clicking a collectible tile (color 8) in the right / bottom areas
            if 0 <= y < 64 and 0 <= x < 64 and base[y][x] == 8 and (x >= 24 or y >= 44):
                inc = True
    elif action in (1, 2, 3, 4):
        dx = {1: 0, 2: 0, 3: -2, 4: 2}[action]
        dy = {1: -2, 2: 2, 3: 0, 4: 0}[action]
        nx, ny = st["px"] + dx, st["py"] + dy
        ok = nx >= 0 and ny >= 0 and nx + 1 < 64 and ny + 1 < 64
        if ok:
            for yy in (ny, ny + 1):
                for xx in (nx, nx + 1):
                    if base[yy][xx] not in WALKABLE:
                        ok = False
        if ok:
            st["px"], st["py"] = nx, ny
            land = base[ny][nx]
            if land == 8 and nx in (12, 18):
                inc = True
            elif land == 9 and ny == 28:
                inc = True

    if inc:
        st["cnt"] += 1

    out = [list(r) for r in base]
    px, py = st["px"], st["py"]
    for yy in (py, py + 1):
        for xx in (px, px + 1):
            if 0 <= yy < 64 and 0 <= xx < 64:
                out[yy][xx] = 14
    for c in range(64):
        out[63][c] = 3 if c < st["cnt"] else 0

    return out, {"level_up": False, "dead": False, "win": False}, st


def step(grid, action, x=None, y=None):
    s = init_state(grid)
    return predict(s, grid, action, x, y)
