import copy

def _find_head(g):
    for r in range(63):
        for c in range(63):
            if g[r][c] == 14 and g[r][c+1] == 14 and g[r+1][c] == 14 and g[r+1][c+1] == 14:
                # make sure it's the 2x2 head (not part of bigger run) - accept first found
                return (r, c)
    return None

def init_state(entry_grid):
    st = {"n": 0, "under": {}, "head": None}
    h = _find_head(entry_grid)
    if h:
        st["head"] = h
        for r in range(h[0], h[0]+2):
            for c in range(h[1], h[1]+2):
                st["under"][(r, c)] = 2
    return st

def _bar_count(g):
    return sum(1 for c in range(64) if g[63][c] == 3)

def predict(state, grid, action, x=None, y=None):
    g = copy.deepcopy(grid)
    st = {"n": state.get("n", 0) + 1, "under": dict(state.get("under", {})), "head": None}
    head = _find_head(g)
    st["head"] = head
    if head is not None:
        # re-sync underlying for current head cells if missing
        for r in range(head[0], head[0]+2):
            for c in range(head[1], head[1]+2):
                st["under"].setdefault((r, c), 2)

    if action == 6 and x is not None:
        if x == 52 and y == 22:
            # toggle 7-cross orientation
            horiz = (g[40][8] == 7)
            if horiz:
                for r in range(40, 44):
                    for c in list(range(8, 16)) + list(range(20, 28)):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2):
                            continue
                        g[r][c] = 4
                for r in list(range(32, 40)) + list(range(44, 52)):
                    for c in range(16, 20):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2):
                            continue
                        g[r][c] = 7
            else:
                for r in list(range(32, 40)) + list(range(44, 52)):
                    for c in range(16, 20):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2):
                            continue
                        g[r][c] = 4
                for r in range(40, 44):
                    for c in list(range(8, 16)) + list(range(20, 28)):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2):
                            continue
                        g[r][c] = 7
        elif x == 52 and y == 40:
            # toggle 9 solid/checker swap between A(rows28-31,c8-11) and B(rows32-39,c4-7)
            a_solid = (g[28][8] == 9 and g[28][9] == 9)
            def setA_solid():
                for r in range(28, 32):
                    for c in range(8, 12):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2): continue
                        g[r][c] = 9
            def setA_checker():
                for r in range(28, 32):
                    for c in range(8, 12):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2): continue
                        on = (g[r][c] != 14)
                        if (r - 28) % 2 == 0:
                            g[r][c] = 9 if c % 2 == 1 else 4
                        else:
                            g[r][c] = 9 if c % 2 == 0 else 4
            def setB_solid():
                for r in range(32, 40):
                    for c in range(4, 8):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2): continue
                        g[r][c] = 9
            def setB_checker():
                for r in range(32, 40):
                    for c in range(4, 8):
                        if head and (head[0] <= r < head[0]+2 and head[1] <= c < head[1]+2): continue
                        if (r - 32) % 2 == 0:
                            g[r][c] = 9 if c % 2 == 1 else 4
                        else:
                            g[r][c] = 9 if c % 2 == 0 else 4
            if a_solid:
                setA_checker(); setB_solid()
            else:
                setA_solid(); setB_checker()
    elif action in (1, 2, 3, 4) and head is not None:
        dr, dc = {1: (-2, 0), 2: (2, 0), 3: (0, -2), 4: (0, 2)}[action]
        nr, nc = head[0] + dr, head[1] + dc
        if 8 <= nr and nr + 2 <= 56 and 0 <= nc and nc + 2 <= 44:
            # record underlying under new cells
            new_under = {}
            for r in range(nr, nr + 2):
                for c in range(nc, nc + 2):
                    new_under[(r, c)] = g[r][c]
            # restore old head cells
            for r in range(head[0], head[0] + 2):
                for c in range(head[1], head[1] + 2):
                    g[r][c] = st["under"].get((r, c), 2)
            for (r, c), col in new_under.items():
                st["under"][(r, c)] = col
            for r in range(nr, nr + 2):
                for c in range(nc, nc + 2):
                    g[r][c] = 14
            st["head"] = (nr, nc)
        else:
            # blocked/clamped: head stays; underlying unchanged
            pass

    # progress bar checkpoints
    if st["n"] - 1 in (0, 1, 4, 6, 8, 11, 14, 16, 18):
        k = _bar_count(g)
        if k < 64:
            g[63][k] = 3

    flags = {"level_up": False, "dead": False, "win": False}
    return g, flags, st

def step(grid, action, x=None, y=None):
    st = init_state(grid)
    g, f, _ = predict(st, grid, action, x, y)
    return g, f
