import copy

def init_state(entry_grid):
    return {"p": 0}

def _toggle9patterns(g):
    processed = set()
    for yy in range(10, 53):
        for xx in range(0, 28):
            cells = [(yy + a, xx + b) for a in range(4) for b in range(4)]
            if any(c in processed for c in cells):
                continue
            vals = [g[a][b] for a, b in cells]
            if 9 not in vals:
                continue
            if any(v not in (4, 9) for v in vals):
                continue
            nines = [(a, b) for a, b in cells if g[a][b] == 9]
            # solid block
            if len(nines) == 16:
                for a, b in cells:
                    if (a + b) % 2 == 0:
                        g[a][b] = 4
                for c_ in cells:
                    processed.add(c_)
                continue
            # checkerboard block: 8 nines all of one parity
            if len(nines) == 8:
                par = set((a + b) % 2 for a, b in nines)
                if len(par) == 1:
                    for a, b in cells:
                        g[a][b] = 9
                    for c_ in cells:
                        processed.add(c_)
                    continue

def predict(state, grid, action, x=None, y=None):
    g = copy.deepcopy(grid)
    flags = {"level_up": False, "dead": False, "win": False}
    p = state.get("p", 0)
    tick = (p == 0)
    p ^= 1
    effect = False

    if action in (1, 2, 3, 4):
        pc = [(yy, xx) for yy in range(64) for xx in range(64) if g[yy][xx] == 14]
        if pc:
            py = min(c[0] for c in pc); px = min(c[1] for c in pc)
            dy = -2 if action == 1 else (2 if action == 2 else 0)
            dx = -2 if action == 3 else (2 if action == 4 else 0)
            ny, nx = py + dy, px + dx
            ok = all(0 <= ny + iy < 64 and 0 <= nx + ix < 64 and g[ny + iy][nx + ix] == 2
                     for iy in (0, 1) for ix in (0, 1))
            if ok:
                for iy in (0, 1):
                    for ix in (0, 1):
                        g[py + iy][px + ix] = 2
                        g[ny + iy][nx + ix] = 14
    elif action == 6 and x is not None and y is not None:
        col = g[y][x]
        if x >= 32 and col == 8:
            effect = True
            # fill 0-outlines adjacent to 8/9 objects in the right panel
            seeds = []
            for yy in range(64):
                for xx in range(32, 64):
                    if g[yy][xx] == 0:
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            a, b = yy + dy, xx + dx
                            if 0 <= a < 64 and 0 <= b < 64 and g[a][b] in (8, 9):
                                seeds.append((yy, xx)); break
            if seeds:
                seen = set(seeds)
                stack = list(seeds)
                while stack:
                    yy, xx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = yy + dy, xx + dx
                        if 0 <= a < 64 and 32 <= b < 64 and g[a][b] == 0 and (a, b) not in seen:
                            seen.add((a, b)); stack.append((a, b))
                for yy, xx in seen:
                    g[yy][xx] = 5
            # toggle left-panel 8 block between the two docks
            eights = [(yy, xx) for yy in range(64) for xx in range(32) if g[yy][xx] == 8]
            if eights:
                r0 = min(c[0] for c in eights)
                if r0 <= 29:
                    for yy in range(24, 30):
                        for xx in range(18, 22):
                            g[yy][xx] = 4
                    for yy in range(30, 34):
                        for xx in range(12, 18):
                            g[yy][xx] = 8
                else:
                    for yy in range(30, 34):
                        for xx in range(12, 18):
                            g[yy][xx] = 4
                    for yy in range(24, 30):
                        for xx in range(18, 22):
                            g[yy][xx] = 8
        elif x >= 32 and col == 9:
            effect = True
            _toggle9patterns(g)

    if effect:
        p ^= 1
    if tick:
        n = sum(1 for xx in range(64) if g[63][xx] == 3)
        if n < 64:
            g[63][n] = 3
    return g, flags, {"p": p}
