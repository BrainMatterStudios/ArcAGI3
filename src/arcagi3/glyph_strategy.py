"""GlyphStrategy — reactive (decide()-interface) SOURCE-FREE glyph-cast solver (sc25 archetype), embeddable
as an additive PortfolioPolicy play. Reads the target glyph from the spell-icon (minority color), toggles the
matching slots to cast (avatar shrinks), then BFS-navigates the small avatar to the goal. Abstains (benign) on
non-matching games -> can only ADD, never regress the floor.

Token format: ("C", x, y) for clicks; ("S", id) for moves; ("reset",).
"""
from __future__ import annotations
from collections import deque, Counter
import numpy as np
from arcagi3 import perception as P

UI_ROW = 45


def _read_glyph(grid):
    bg = P.detect_background(grid)
    ys, xs = np.where(grid[UI_ROW:, :22] != bg)
    if len(ys) == 0:
        return None, None
    r0, r1 = ys.min() + UI_ROW, ys.max() + UI_ROW
    c0, c1 = xs.min(), xs.max()
    reg = grid[r0:r1 + 1, c0:c1 + 1]
    vals = Counter(int(v) for v in reg.flatten())
    if len(vals) < 2:
        return None, None
    on = min(vals, key=lambda k: vals[k])
    h, w = reg.shape; gh, gw = max(1, h // 3), max(1, w // 3)
    glyph = [[1 if (reg[i*gh:(i+1)*gh, j*gw:(j+1)*gw] == on).any() else 0 for j in range(3)] for i in range(3)]
    return glyph, ((r0 + r1) // 2, (c0 + c1) // 2)


def _slots(grid):
    bg = P.detect_background(grid)
    ys, xs = np.where(grid[UI_ROW:, 22:43] != bg)
    if len(ys) == 0:
        return None
    r0, r1 = ys.min() + UI_ROW, ys.max() + UI_ROW
    c0, c1 = xs.min() + 22, xs.max() + 22
    return {(i, j): (r0 + (r1 - r0) * (2*i+1)//6, c0 + (c1 - c0) * (2*j+1)//6) for i in range(3) for j in range(3)}


def _cc(grid, bg, exclude):
    mask = (grid != bg)
    for c in exclude:
        mask &= (grid != c)
    mask[UI_ROW:, :] = False
    seen = np.zeros_like(mask); objs = []; W = grid.shape[1]
    for r in range(UI_ROW):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < UI_ROW and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
                objs.append((len(cells), (min(ys), min(xs), max(ys), max(xs))))
    return objs


def _actors(grid):
    bg = P.detect_background(grid)
    counts = Counter(int(v) for v in grid[:UI_ROW].flatten() if v != bg)
    if not counts:
        return None, None
    walls = [c for c, n in counts.items() if n >= 40]
    objs = [o for o in _cc(grid, bg, walls) if 3 <= o[0] <= 80]
    if len(objs) < 2:
        return None, None
    objs.sort(key=lambda o: o[0])
    return objs[0][1], objs[-1][1]


def _nav_plan(grid, avatar, goal):
    ar0, ac0, ar1, ac1 = avatar; gy0, gx0, gy1, gx1 = goal
    w = ac1 - ac0 + 1; h = ar1 - ar0 + 1; step = max(2, w)
    bg = P.detect_background(grid)
    walls = set()
    rows = np.arange(grid.shape[0])[:, None] * np.ones((1, grid.shape[1]))
    ys, xs = np.where((grid != bg) & (rows < UI_ROW))
    for r, c in zip(ys.tolist(), xs.tolist()):
        if (ar0 <= r <= ar1 and ac0 <= c <= ac1) or (gy0 <= r <= gy1 and gx0 <= c <= gx1):
            continue
        walls.add((c, r))
    def free(x, y):
        return all(0 <= x < 64 and 0 <= y < UI_ROW and (x+a, y+b) not in walls for a in range(w) for b in range(h))
    def at_goal(x, y):
        return x < gx1 + 1 and x + w > gx0 and y < gy1 + 1 and y + h > gy0
    D = {1: (0, -step), 2: (0, step), 3: (-step, 0), 4: (step, 0)}
    start = (ac0, ar0); seen = {start}; q = deque([(start, [])])
    while q:
        (x, y), path = q.popleft()
        for a, (dx, dy) in D.items():
            nx, ny = x + dx, y + dy
            if at_goal(nx, ny):
                return path + [a]
            if (nx, ny) not in seen and free(nx, ny):
                seen.add((nx, ny)); q.append(((nx, ny), path + [a]))
    return None


class GlyphStrategy:
    def __init__(self, seed: int = 0):
        self.gs = None
        self._reset()

    def _reset(self):
        self._phase = "init"     # init -> draw -> nav -> abstain
        self._queue = []
        self._planned_level = -1

    def _benign(self, available):
        if available:
            return ("S", int(available[0]))
        return ("reset",)

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        if gstate_terminal:
            self._reset(); return ("reset",)
        if levels != self._planned_level and self._phase != "init":
            self._reset()
        if self._phase == "abstain":
            return self._benign(available)

        if self._phase == "init":
            self._planned_level = levels
            if 6 not in available:
                self._phase = "abstain"; return self._benign(available)
            glyph, icon = _read_glyph(grid); slots = _slots(grid)
            if glyph is None or slots is None or not any(any(r) for r in glyph):
                self._phase = "abstain"; return self._benign(available)
            q = [("C", int(icon[1]), int(icon[0]))]          # prime (click icon)
            for i in range(3):
                for j in range(3):
                    if glyph[i][j]:
                        cy, cx = slots[(i, j)]
                        q.append(("C", int(cx), int(cy)))
            self._queue = q; self._phase = "draw"

        if self._phase == "draw":
            if self._queue:
                return self._queue.pop(0)
            # cast done -> plan navigation
            avatar, goal = _actors(grid)
            if avatar is None or goal is None:
                self._phase = "abstain"; return self._benign(available)
            plan = _nav_plan(grid, avatar, goal)
            if not plan:
                self._phase = "abstain"; return self._benign(available)
            self._queue = [("S", a) for a in plan]; self._phase = "nav"

        if self._phase == "nav":
            if self._queue:
                return self._queue.pop(0)
            self._phase = "abstain"; return self._benign(available)

        return self._benign(available)
