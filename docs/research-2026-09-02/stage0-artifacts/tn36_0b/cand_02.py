import copy

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
                bars.append({'cells': comp, 'present': True})
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
    new_state = {'bars': [dict(b) for b in state.get('bars', [])]}

    if action == 6 and x is not None and y is not None and 0 <= x < 64 and 0 <= y < 64:
        # Budget: rightmost remaining 9 in row 1 becomes 3
        for cx in range(63, -1, -1):
            if new_grid[1][cx] == 9:
                new_grid[1][cx] = 3
                break
        # Toggle clicked bar (1 <-> 5)
        for b in new_state['bars']:
            if not b['present']:
                continue
            if (y, x) in b['cells']:
                for (cy, cx) in b['cells']:
                    new_grid[cy][cx] = 5 if new_grid[cy][cx] == 1 else 1
                b['present'] = False
        # Goal: click a gem (0) when no bars remain
        if new_grid[y][x] == 0 and not _has_ones(new_grid):
            flags['level_up'] = True
    return new_grid, flags, new_state

def is_goal(state, grid):
    return not _has_ones(grid)
