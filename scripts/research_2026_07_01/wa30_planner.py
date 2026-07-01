"""wa30 grab-drag PLANNER — first Lever-B mechanic solver.

wa30 is a deterministic 16x16 grid puzzle (cell=4px). Avatar moves 1 cell (facing = last move dir);
ACTION5 grabs the faced-adjacent block or releases; while held, avatar+block move rigidly if both target
cells are free; WIN = every block anchor on a goal cell and none held. Fully reverse-engineered from source.

Approach: perceive the logical state from the frame, then plan ONE block at a time via BFS over an exact
forward model (other blocks + borders = walls), execute, repeat. Perception + forward model are validated
against engine ground truth in the accompanying test.

Coords are engine (x,y) = (col_pixel, row_pixel), multiples of CELL=4. Facing: 0=up,90=right,180=down,270=left.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from arcagi3 import perception as P

CELL = 4
AVATAR_COLOR = 14
BLOCK_COLOR = 4
GOAL_BORDER_COLOR = 9   # goal pad border (small c9 comps are block-markers, filtered by size)
DELTAS = {1: (0, -CELL), 2: (0, CELL), 3: (-CELL, 0), 4: (CELL, 0)}  # action -> (dx,dy)


def facing_of(dx, dy):
    if dy < 0: return 0
    if dx > 0: return 90
    if dy > 0: return 180
    return 270


def faced_cell(ax, ay, facing):
    if facing == 0: return (ax, ay - CELL)
    if facing == 180: return (ax, ay + CELL)
    if facing == 90: return (ax + CELL, ay)
    return (ax - CELL, ay)


# ---------- perception: frame -> logical anchors ----------
def _anchors(grid, color, size_lo=6, size_hi=40):
    """top-left (x=col, y=row) of each compact component of `color`, excluding HUD (row>=60)."""
    out = []
    for o in P.connected_components(grid, background=P.detect_background(grid)):
        if o.color != color or not (size_lo <= o.size <= size_hi):
            continue
        r0, c0, r1, c1 = o.bbox
        if r0 >= 60:  # bottom HUD bar
            continue
        # snap to the 4px grid: a facing-marker/anti-alias pixel can offset the top-left by 1
        out.append(((c0 // CELL) * CELL, (r0 // CELL) * CELL))
    return sorted(set(out))


def _goal_corners(grid):
    """3+ goal slots = the LARGE c9 goal-strip component(s) tiled into 4px cells. Small c9 comps are
    per-block markers and are excluded by size."""
    bg = P.detect_background(grid)
    corners = []
    for o in P.connected_components(grid, background=bg):
        if o.color != GOAL_BORDER_COLOR or o.size < 16:
            continue
        r0, c0, r1, c1 = o.bbox
        for x in range((c0 // CELL) * CELL, c1 + 1, CELL):
            for y in range((r0 // CELL) * CELL, r1 + 1, CELL):
                corners.append((x, y))
    return sorted(set(corners))


def perceive(grid):
    av = _anchors(grid, AVATAR_COLOR)
    avatar = av[0] if av else None
    blocks = sorted(_anchors(grid, BLOCK_COLOR))
    pads = _goal_corners(grid)
    return avatar, blocks, pads


# ---------- exact forward model (single block + avatar; other blocks = walls) ----------
class Model:
    def __init__(self, walls: set):
        self.walls = walls  # cells the avatar/block cannot occupy (borders + other blocks)

    def step(self, state, action):
        """state = (ax, ay, bx, by, facing, held) -> next state (deterministic)."""
        ax, ay, bx, by, facing, held = state
        if action == 5:
            if held:
                return (ax, ay, bx, by, facing, False)
            if faced_cell(ax, ay, facing) == (bx, by):
                return (ax, ay, bx, by, facing, True)
            return state
        dx, dy = DELTAS[action]
        if not held:
            nf = facing_of(dx, dy)
            tgt = (ax + dx, ay + dy)
            if tgt not in self.walls and tgt != (bx, by):
                return (tgt[0], tgt[1], bx, by, nf, held)
            return (ax, ay, bx, by, nf, held)  # blocked, but facing updated
        # held: rigid drag, offset preserved
        offx, offy = bx - ax, by - ay
        nav = (ax + dx, ay + dy)
        nbl = (bx + dx, by + dy)
        ok = ((nav not in self.walls or nav == (bx, by)) and
              (nbl not in self.walls or nbl == (ax, ay)))
        if ok:
            return (nav[0], nav[1], nbl[0], nbl[1], facing, held)
        return state


def bfs(model, start, goal_block_xy):
    """A* (Manhattan heuristic on the block) for the action list that lands the block on goal_block_xy
    and releases it. A* over BFS avoids the ~500k-state blowup that makes plain BFS intractable."""
    import heapq
    gx, gy = goal_block_xy

    def h(s):
        return (abs(s[2] - gx) + abs(s[3] - gy)) // CELL  # cells the block still must travel

    def is_win(s):
        return (s[2], s[3]) == goal_block_xy and not s[5]
    if is_win(start):
        return []
    counter = 0
    pq = [(h(start), 0, counter, start, [])]
    best = {start: 0}
    while pq:
        f, g, _, s, path = heapq.heappop(pq)
        if g > best.get(s, 1 << 30):
            continue
        for a in (1, 2, 3, 4, 5):
            ns = model.step(s, a)
            ng = g + 1
            if ng >= best.get(ns, 1 << 30):
                continue
            if is_win(ns):
                return path + [a]
            best[ns] = ng
            counter += 1
            heapq.heappush(pq, (ng + h(ns), ng, counter, ns, path + [a]))
    return None


def border_walls():
    w = set()
    for i in range(0, 64, CELL):
        w |= {(-CELL, i), (64, i), (i, -CELL), (i, 64)}
    return w


def _plan_sequence(avatar, block_pad_pairs, all_blocks):
    """plan the given ordered (block, pad) pairs; other not-yet-placed blocks are walls. Min-actions per
    block via A*. Returns (actions, total) or None if any leg is unreachable."""
    borders = border_walls()
    ax, ay, facing = avatar[0], avatar[1], 0
    placed, full = [], []
    pending = [b for (b, _) in block_pad_pairs]
    for i, (b, pad) in enumerate(block_pad_pairs):
        pending = [bb for (bb, _) in block_pad_pairs[i + 1:]]
        walls = borders | set(pending) | set(placed)
        model = Model(walls)
        path = bfs(model, (ax, ay, b[0], b[1], facing, False), pad)
        if path is None:
            return None
        s = (ax, ay, b[0], b[1], facing, False)
        for a in path:
            s = model.step(s, a)
        ax, ay, facing = s[0], s[1], s[4]
        placed.append(pad)
        full.extend(path)
    return full


def plan_all(avatar, blocks, pads):
    """Minimize total actions: search over block->pad assignments and placement orders (feasible for the
    small block counts in wa30), keep the shortest plan that fits."""
    from itertools import permutations
    n = len(blocks)
    pads = pads[:n] if len(pads) >= n else pads
    best = None
    # try each assignment of pads to blocks, and each placement order
    for pad_perm in permutations(pads, n):
        pairs0 = list(zip(blocks, pad_perm))
        for order in permutations(range(n)):
            pairs = [pairs0[i] for i in order]
            plan = _plan_sequence(avatar, pairs, blocks)
            if plan is not None and (best is None or len(plan) < len(best)):
                best = plan
    return best
