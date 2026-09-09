# World model for cn04 level 0.
# Player object = 4-connected component of {0,8} cells (excluding static 8s) containing 0s.
# Static 8s = 8s 4-connected to the goal frame (color 14); they are obstacles and never move.
# Actions: 1=up3, 2=down3, 3=left3, 4=right3, 5=rotate 90 CW about bbox top-left.
# Blocked rotation -> level_up. Budget bar: hidden timer, tick%7 in {1,3,5} consumes leftmost row-0 '4'.

def _static8(g):
    static = set()
    frontier = []
    for y in range(64):
        for x in range(64):
            if g[y][x] == 8:
                for ny, nx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                    if 0 <= ny < 64 and 0 <= nx < 64 and g[ny][nx] == 14:
                        static.add((y, x))
                        frontier.append((y, x))
                        break
    while frontier:
        cy, cx = frontier.pop()
        for ny, nx in ((cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)):
            if 0 <= ny < 64 and 0 <= nx < 64 and g[ny][nx] == 8 and (ny, nx) not in static:
                static.add((ny, nx))
                frontier.append((ny, nx))
    return static

def _piece(g, static):
    seen = set()
    best = None
    best0 = -1
    for y in range(64):
        for x in range(64):
            if (g[y][x] in (0, 8) and (y, x) not in static and (y, x) not in seen):
                comp = set()
                stack = [(y, x)]
                seen.add((y, x))
                while stack:
                    cy, cx = stack.pop()
                    comp.add((cy, cx))
                    for ny, nx in ((cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)):
                        if 0 <= ny < 64 and 0 <= nx < 64 and (ny, nx) not in seen \
                                and (ny, nx) not in static and g[ny][nx] in (0, 8):
                            seen.add((ny, nx))
                            stack.append((ny, nx))
                n0 = sum(1 for (yy, xx) in comp if g[yy][xx] == 0)
                if n0 > best0:
                    best0 = n0
                    best = comp
    return best if best is not None else set()

def init_state(entry_grid):
    return {"t": 0}

def predict(state, grid, action, x=None, y=None):
    t = state.get("t", 0)
    state = {"t": t + 1}
    flags = {"level_up": False, "dead": False, "win": False}
    g = [row[:] for row in grid]

    static = _static8(g)
    shape = _piece(g, static)
    if shape:
        blocked = set()
        for y in range(64):
            for xx in range(64):
                if (y, xx) in shape:
                    continue
                v = g[y][xx]
                if v == 14 or (y, xx) in static:
                    blocked.add((y, xx))

        ys = [p[0] for p in shape]
        xs = [p[1] for p in shape]
        r0, c0 = min(ys), min(xs)
        H = max(ys) - r0 + 1

        targets = []
        if action == 1:
            targets = [(yy-3, xx, g[yy][xx]) for (yy, xx) in shape]
        elif action == 2:
            targets = [(yy+3, xx, g[yy][xx]) for (yy, xx) in shape]
        elif action == 3:
            targets = [(yy, xx-3, g[yy][xx]) for (yy, xx) in shape]
        elif action == 4:
            targets = [(yy, xx+3, g[yy][xx]) for (yy, xx) in shape]
        elif action == 5:
            for (yy, xx) in shape:
                i = yy - r0
                j = xx - c0
                targets.append((r0 + j, c0 + (H - 1 - i), g[yy][xx]))
        else:
            targets = [(yy, xx, g[yy][xx]) for (yy, xx) in shape]

        ok = True
        for (ny, nx, v) in targets:
            if not (0 <= ny < 64 and 0 <= nx < 64) or (ny, nx) in blocked:
                ok = False
                break

        if ok:
            for (yy, xx) in shape:
                g[yy][xx] = 10
            for (ny, nx, v) in targets:
                g[ny][nx] = v
        else:
            if action == 5:
                flags["level_up"] = True

    if (t % 7) in (1, 3, 5):
        for xx in range(64):
            if g[0][xx] == 4:
                g[0][xx] = 0
                break

    return g, flags, state

def is_goal(state, grid):
    return False
