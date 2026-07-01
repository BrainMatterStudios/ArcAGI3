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


def plan_joint(avatar, blocks, goal_cells, max_nodes=800_000):
    """Globally-optimal (min-action) JOINT plan via A* over the full state
    (avatar, all block positions, facing, held-block-index). Heuristic = sum over blocks of
    Manhattan-to-nearest-goal (cells) — admissible. Fits tight step budgets that the sequential
    planner overruns. Returns action list or None."""
    import heapq
    borders = border_walls()
    goal = set(goal_cells)
    blocks = tuple(sorted(blocks))
    n = len(blocks)

    def h(bpos):
        tot = 0
        for (bx, by) in bpos:
            tot += min((abs(bx - gx) + abs(by - gy)) for (gx, gy) in goal) // CELL
        return tot

    def is_win(bpos, held):
        return held == -1 and all(b in goal for b in bpos)

    start = (avatar[0], avatar[1], blocks, 0, -1)  # ax, ay, block-tuple, facing, held_idx
    if is_win(start[2], start[4]):
        return []
    counter = 0
    pq = [(h(blocks), 0, counter, start, [])]
    best = {(start[0], start[1], start[2], start[3], start[4]): 0}
    nodes = 0
    while pq and nodes < max_nodes:
        f, g, _, s, path = heapq.heappop(pq)
        ax, ay, bpos, facing, held = s
        key = (ax, ay, bpos, facing, held)
        if g > best.get(key, 1 << 30):
            continue
        nodes += 1
        for a in (1, 2, 3, 4, 5):
            nax, nay, nb, nfac, nheld = ax, ay, bpos, facing, held
            if a == 5:
                if held != -1:
                    nheld = -1
                else:
                    fc = faced_cell(ax, ay, facing)
                    hit = next((i for i, b in enumerate(bpos) if b == fc), -1)
                    if hit == -1:
                        continue
                    nheld = hit
            else:
                dx, dy = DELTAS[a]
                if held == -1:
                    nfac = facing_of(dx, dy)
                    tgt = (ax + dx, ay + dy)
                    if tgt in borders or tgt in bpos:
                        nax, nay = ax, ay  # blocked; only facing changes
                    else:
                        nax, nay = tgt
                else:
                    hb = bpos[held]
                    nav = (ax + dx, ay + dy)
                    nbl = (hb[0] + dx, hb[1] + dy)
                    others = tuple(b for i, b in enumerate(bpos) if i != held)
                    if (nav in borders or nav in others) or (nbl in borders or nbl in others):
                        continue  # drag blocked
                    nax, nay = nav
                    nb = tuple(nbl if i == held else b for i, b in enumerate(bpos))
            ns = (nax, nay, nb, nfac, nheld)
            nkey = (nax, nay, nb, nfac, nheld)
            ng = g + 1
            if ng >= best.get(nkey, 1 << 30):
                continue
            if is_win(nb, nheld):
                return path + [a]
            best[nkey] = ng
            counter += 1
            heapq.heappush(pq, (ng + h(nb), ng, counter, ns, path + [a]))
    return None


def _greedy_assign(blocks, pads):
    """assign each block a distinct pad minimizing total Manhattan (greedy over sorted pair distances)."""
    pairs = sorted(((abs(b[0]-p[0])+abs(b[1]-p[1]), bi, pi)
                    for bi, b in enumerate(blocks) for pi, p in enumerate(pads)))
    assign = {}; used_p = set()
    for _, bi, pi in pairs:
        if bi in assign or pi in used_p:
            continue
        assign[bi] = pads[pi]; used_p.add(pi)
    return [assign[i] for i in range(len(blocks))]


def plan_all(avatar, blocks, pads):
    """For few blocks (<=3) use the globally-optimal JOINT A*. For more, use assignment + order search:
    greedy block->pad assignment, then min-total over placement orders (each leg = optimal per-block A*)."""
    from itertools import permutations
    n = len(blocks)
    if n <= 3:
        joint = plan_joint(avatar, blocks, pads)
        if joint is not None:
            return joint
    pads = pads[:max(n, len(pads))]
    assign = _greedy_assign(blocks, pads)
    best = None
    orders = permutations(range(n)) if n <= 7 else [sorted(range(n), key=lambda i: abs(blocks[i][0]-avatar[0])+abs(blocks[i][1]-avatar[1]))]
    for order in orders:
        pairs = [(blocks[i], assign[i]) for i in order]
        plan = _plan_sequence(avatar, pairs, blocks)
        if plan is not None and (best is None or len(plan) < len(best)):
            best = plan
    return best
