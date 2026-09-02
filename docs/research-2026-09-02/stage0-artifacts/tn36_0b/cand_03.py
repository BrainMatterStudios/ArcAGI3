def init_state(entry_grid):
    grid = entry_grid
    H, W = len(grid), len(grid[0])
    seen = [[False] * W for _ in range(H)]
    bars = []
    for y in range(H):
        for x in range(W):
            if grid[y][x] == 1 and not seen[y][x]:
                comp = []
                stack = [(y, x)]
                seen[y][x] = True
                while stack:
                    cy, cx = stack.pop()
                    comp.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and grid[ny][nx] == 1:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                comp.sort()
                bars.append({'cells': comp, 'center': comp[len(comp) // 2]})
    return {'bars': bars}

def _has_ones(grid):
    for row in grid:
        for v in row:
            if v == 1:
                return True
    return False

def predict(state, grid, action, x=None, y=None):
    if state is None:
        state = init_state(grid)
    new_grid = [list(row) for row in grid]
    flags = {"level_up": False, "dead": False, "win": False}

    if action == 6 and x is not None and y is not None and 0 <= x < 64 and 0 <= y < 64:
        clicked_val = new_grid[y][x]
        # Budget bar: rightmost remaining 9 in row 1 becomes 3
        for cx in range(63, -1, -1):
            if new_grid[1][cx] == 9:
                new_grid[1][cx] = 3
                break
        # Toggle the bar whose center is within Chebyshev distance 1 of the click
        for bar in state['bars']:
            cy, cx = bar['center']
            if abs(y - cy) <= 1 and abs(x - cx) <= 1:
                for (by, bx) in bar['cells']:
                    new_grid[by][bx] = 5 if new_grid[by][bx] == 1 else 1
                break
        # Win: click the goal band (9-cells below the structure) once all bars are cleared
        if clicked_val == 9 and y >= 48 and not _has_ones(new_grid):
            flags['level_up'] = True

    return new_grid, flags, state

def is_goal(state, grid):
    return not _has_ones(grid)
