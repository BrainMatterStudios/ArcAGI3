def init_state(entry_grid):
    H = len(entry_grid)
    W = len(entry_grid[0])
    visited = [[False] * W for _ in range(H)]
    slots = []
    for y in range(H):
        for x in range(W):
            if entry_grid[y][x] == 1 and not visited[y][x]:
                comp = []
                stack = [(x, y)]
                visited[y][x] = True
                while stack:
                    cx, cy = stack.pop()
                    comp.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H and not visited[ny][nx] and entry_grid[ny][nx] == 1:
                            visited[ny][nx] = True
                            stack.append((nx, ny))
                slots.append(comp)
    return slots


def predict(state, grid, action, x=None, y=None):
    H = len(grid)
    W = len(grid[0])
    new_grid = [row[:] for row in grid]
    flags = {"level_up": False, "dead": False, "win": False}

    # does the previous (input) grid still contain any 1?
    has_ones = False
    for yy in range(H):
        row = grid[yy]
        for xx in range(W):
            if row[xx] == 1:
                has_ones = True
                break
        if has_ones:
            break

    if action == 6 and x is not None and y is not None:
        # find the slot whose (bounding box expanded by 1) contains the click
        best = None
        for slot in state:
            xs = [c[0] for c in slot]
            ys = [c[1] for c in slot]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmin - 1 <= x <= xmax + 1 and ymin - 1 <= y <= ymax + 1:
                key = (ymin, xmin)
                if best is None or key < best[0]:
                    best = (key, slot)
        if best is not None:
            slot = best[1]
            is_on = False
            for cx, cy in slot:
                if grid[cy][cx] == 1:
                    is_on = True
                    break
            val = 5 if is_on else 1
            for cx, cy in slot:
                new_grid[cy][cx] = val

        # budget bar: rightmost 9 in row 1 becomes 3
        if H > 1:
            bar = new_grid[1]
            for c in range(W - 1, -1, -1):
                if bar[c] == 9:
                    bar[c] = 3
                    break

        # win: a click made while the board is already clear
        if not has_ones:
            flags["level_up"] = True

    return new_grid, flags, state


def is_goal(state, grid):
    for yy in range(len(grid)):
        for xx in range(len(grid[0])):
            if grid[yy][xx] == 1:
                return False
    return True
