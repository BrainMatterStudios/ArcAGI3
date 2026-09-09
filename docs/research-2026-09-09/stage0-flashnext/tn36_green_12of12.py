import numpy as np

_cache = {}

def _analyze():
    if 'done' in _cache:
        return _cache['res']
    g = ENTRY_GRID
    H, W = len(g), len(g[0])
    top = []
    for y in range(H):
        x = 0
        while x < W:
            if g[y][x] == 1:
                x2 = x
                while x2 + 1 < W and g[y][x2 + 1] == 1:
                    x2 += 1
                if x2 - x + 1 == 3:
                    top.append((y, x, x2))
                x = x2 + 1
            else:
                x += 1
    bot = []
    for x in range(W):
        y = 0
        while y < H:
            if g[y][x] == 1:
                y2 = y
                while y2 + 1 < H and g[y2 + 1][x] == 1:
                    y2 += 1
                if y2 - y + 1 >= 3:
                    bot.append((x, y, y2))
                y = y2 + 1
            else:
                y += 1
    bar = None
    for y in range(H):
        if y < 40 and 9 in g[y]:
            bar = y
            break
    gy = [yy for yy in range(H) for xx in range(W) if yy >= 40 and g[yy][xx] == 9]
    gx = [xx for yy in range(H) for xx in range(W) if yy >= 40 and g[yy][xx] == 9]
    if gy:
        goal = (min(gy), max(gy), min(gx), max(gx))
    else:
        goal = None
    res = {'top': top, 'bot': bot, 'bar': bar, 'goal': goal}
    _cache['done'] = True
    _cache['res'] = res
    return res

def step(grid, action, x=None, y=None):
    ng = [row[:] for row in grid]
    flags = {"level_up": False, "dead": False, "win": False}
    A = _analyze()
    H, W = len(ng), len(ng[0])

    # budget bar: rightmost remaining 9 in the bar row becomes 3 on every action
    by = A['bar']
    if by is not None:
        for cx in range(W - 1, -1, -1):
            if ng[by][cx] == 9:
                ng[by][cx] = 3
                break

    if action == 6 and x is not None and y is not None:
        toggled = False
        # top bars: horizontal 3-run slots, clickable one row above/below
        for (ry, x1, x2) in A['top']:
            if x1 - 1 <= x <= x2 + 1 and ry - 1 <= y <= ry + 1:
                for cx in range(x1, x2 + 1):
                    v = ng[ry][cx]
                    if v == 1:
                        ng[ry][cx] = 5
                    elif v == 5:
                        ng[ry][cx] = 1
                toggled = True
                break
        if not toggled:
            # bottom bars: vertical 3-run slots, clickable inside
            for (cx, y1, y2) in A['bot']:
                if cx - 1 <= x <= cx + 1 and y1 <= y <= y2:
                    for cy in range(y1, y2 + 1):
                        v = ng[cy][cx]
                        if v == 1:
                            ng[cy][cx] = 5
                        elif v == 5:
                            ng[cy][cx] = 1
                    break
        # win: all slot cells collected (no 1s) and click inside goal circle
        remaining = False
        for (ry, x1, x2) in A['top']:
            for cx in range(x1, x2 + 1):
                if ng[ry][cx] == 1:
                    remaining = True
        for (cx, y1, y2) in A['bot']:
            for cy in range(y1, y2 + 1):
                if ng[cy][cx] == 1:
                    remaining = True
        if not remaining and A['goal'] is not None:
            gy0, gy1, gx0, gx1 = A['goal']
            if gy0 <= y <= gy1 and gx0 <= x <= gx1:
                flags['level_up'] = True

    return ng, flags

def is_goal(state, grid):
    return False
