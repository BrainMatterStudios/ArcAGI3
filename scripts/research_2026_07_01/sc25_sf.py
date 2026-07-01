"""sc25 SOURCE-FREE glyph-cast solver (frame perception only). Reads the target glyph from the spell-icon
(the minority color inside the icon forms the 3x3 pattern), distinguishes avatar (smaller) from goal (larger)
among the same-colored objects, perceives the maze walls, then: prime (click icon) -> toggle the glyph slots
-> cast (avatar shrinks) -> BFS-navigate the small avatar to the goal.

Slot grid + icon are located by layout (bottom UI), which generalizes to the glyph-game family.
"""
from __future__ import annotations
from collections import deque, Counter
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

UI_ROW = 45   # gameplay above this row; UI (icon, slot grid) below


def read_glyph(grid):
    """find the spell-icon (bottom-left non-bg region) and read its 3x3 pattern (minority color = 'on')."""
    bg = P.detect_background(grid)
    ys, xs = np.where((grid[UI_ROW:, :22] != bg))
    if len(ys) == 0:
        return None, None
    r0, r1 = ys.min() + UI_ROW, ys.max() + UI_ROW
    c0, c1 = xs.min(), xs.max()
    reg = grid[r0:r1 + 1, c0:c1 + 1]
    vals = Counter(int(v) for v in reg.flatten())
    if len(vals) < 2:
        return None, None
    on_color = min(vals, key=lambda k: vals[k])   # glyph cells = minority color
    h, w = reg.shape; gh, gw = max(1, h // 3), max(1, w // 3)
    glyph = [[1 if (reg[i*gh:(i+1)*gh, j*gw:(j+1)*gw] == on_color).any() else 0 for j in range(3)] for i in range(3)]
    return glyph, ((r0 + r1) // 2, (c0 + c1) // 2)   # glyph, icon center (row,col)


def slot_centers(grid):
    """the 3x3 toggle grid bottom-center (cols ~22-42; exclude the right-edge progress bar)."""
    bg = P.detect_background(grid)
    ys, xs = np.where(grid[UI_ROW:, 22:43] != bg)
    if len(ys) == 0:
        return None
    r0, r1 = ys.min() + UI_ROW, ys.max() + UI_ROW
    c0, c1 = xs.min() + 22, xs.max() + 22
    centers = {}
    for i in range(3):
        for j in range(3):
            cy = r0 + (r1 - r0) * (2 * i + 1) // 6
            cx = c0 + (c1 - c0) * (2 * j + 1) // 6
            centers[(i, j)] = (cy, cx)
    return centers


def _cc_agnostic(grid, bg, exclude=()):
    """color-agnostic connected components of the non-bg mask in the gameplay area (merges border+fill),
    excluding wall colors so actors aren't merged into the maze."""
    mask = (grid != bg)
    for c in exclude:
        mask &= (grid != c)
    mask[UI_ROW:, :] = False
    seen = np.zeros_like(mask); objs = []
    H, W = grid.shape
    for r in range(UI_ROW):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < UI_ROW and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
                objs.append((len(cells), (min(ys), min(xs), max(ys), max(xs))))
    return objs


def perceive_actors(grid):
    """avatar + goal are multi-color (border+fill) objects; avatar is SMALLER, goal LARGER. Excludes the
    dominant wall color, then merges border+fill (color-agnostic). Returns (avatar_bbox, goal_bbox)."""
    bg = P.detect_background(grid)
    play = grid[:UI_ROW]
    counts = Counter(int(v) for v in play.flatten() if v != bg)
    if not counts:
        return None, None
    # walls = the most-common non-bg colors (the maze structure); actors = small distinct colors
    wall_colors = [c for c, n in counts.items() if n >= 40]
    objs = [o for o in _cc_agnostic(grid, bg, exclude=wall_colors) if 3 <= o[0] <= 80]
    if len(objs) < 2:
        return None, None
    objs.sort(key=lambda o: o[0])
    return objs[0][1], objs[-1][1]


def solve(verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("sc25"))
    env = client.make(game_id=gid, scorecard_id="sc25-sf")
    obs = env.reset()

    def grid():
        return P.to_grid(obs.frame)
    def click(x, y):
        nonlocal obs
        obs = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

    g = grid()
    glyph, icon = read_glyph(g)
    slots = slot_centers(g)
    if glyph is None or slots is None:
        print("  glyph/slots not perceived -> abstain"); return 0
    if verbose:
        print(f"read glyph={glyph} icon@{icon}")

    # STAGE 1: prime (click icon), then toggle the 'on' glyph cells
    click(icon[1], icon[0])                    # x=col, y=row
    for i in range(3):
        for j in range(3):
            if glyph[i][j]:
                cy, cx = slots[(i, j)]
                click(cx, cy)
    av = perceive_actors(grid())[0]
    if verbose and av is not None:
        print(f"after cast: avatar bbox {av} (h={av[2]-av[0]+1} w={av[3]-av[1]+1})")

    # STAGE 2: navigate the (shrunk) avatar to the goal
    g = grid()
    avatar, goal = perceive_actors(g)
    if avatar is None or goal is None:
        print("  avatar/goal not perceived after cast"); return 0
    ar0, ac0, ar1, ac1 = avatar
    w = ac1 - ac0 + 1; h = ar1 - ar0 + 1
    ax, ay = ac0, ar0                                  # top-left (col,row)
    gy0, gx0, gy1, gx1 = goal                          # (r0,c0,r1,c1)
    step = max(2, w)
    bg = P.detect_background(g)
    walls = set()
    rows = np.arange(g.shape[0])[:, None] * np.ones((1, g.shape[1]))
    ys, xs = np.where((g != bg) & (rows < UI_ROW))
    for r, c in zip(ys.tolist(), xs.tolist()):
        if (ar0 <= r <= ar1 and ac0 <= c <= ac1) or (gy0 <= r <= gy1 and gx0 <= c <= gx1):
            continue                                   # skip avatar/goal cells
        walls.add((c, r))

    def free(x, y):
        return all(0 <= x < 64 and 0 <= y < UI_ROW and (x + a, y + b) not in walls
                   for a in range(w) for b in range(h))
    def at_goal(x, y):
        return x < gx1 + 1 and x + w > gx0 and y < gy1 + 1 and y + h > gy0

    DIRS = {1: (0, -step), 2: (0, step), 3: (-step, 0), 4: (step, 0)}
    start = (ax, ay); seen = {start}; q = deque([(start, [])]); plan = None
    while q and plan is None:
        (x, y), path = q.popleft()
        for a, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if at_goal(nx, ny):
                plan = path + [a]; break
            if (nx, ny) not in seen and free(nx, ny):
                seen.add((nx, ny)); q.append(((nx, ny), path + [a]))
    if plan is None:
        print(f"  no nav plan (avatar {w}x{h} reached {len(seen)} cells)"); return 0
    if verbose:
        print(f"  nav plan: {len(plan)} moves")
    lv = int(obs.levels_completed or 0)
    for a in plan:
        obs = env.step(GameAction.from_id(a))
        if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
            print("  *** sc25 L0 SOLVED SOURCE-FREE ***"); return int(obs.levels_completed or 0)
        if obs.state == GameState.GAME_OVER:
            print("  GAME_OVER"); break
    print(f"  did not solve (levels {obs.levels_completed})"); return int(obs.levels_completed or 0)


if __name__ == "__main__":
    solve()
