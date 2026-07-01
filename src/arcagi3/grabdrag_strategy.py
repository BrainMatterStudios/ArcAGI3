"""GrabDragStrategy — a reactive (decide()-interface) SOURCE-FREE grab-drag solver that slots into
PortfolioPolicy as an additive max-over-plays play. Self-contained (embeddable in the submission notebook).

Grab-drag mechanic (generalized from wa30): a deterministic grid where the avatar moves 1 cell (facing = last
move dir); ACTION5 grabs the faced-adjacent block or releases; while held, avatar+block move rigidly if both
target cells are free; win = every block anchor on a goal cell and none held. The CELL size + avatar are
LEARNED online by probing; goal color is a ranked hypothesis disambiguated by the engine reward. On a
non-matching game the strategy ABSTAINS (benign action) so it can never regress the coverage floor.

Token format matches my_agent: ("reset",) | ("S", id) | ("C", x, y).
"""
from __future__ import annotations
import heapq
from collections import deque, Counter
import numpy as np
from arcagi3 import perception as P


# ---------------- forward model (parameterized by cell) ----------------
def _facing_of(dx, dy):
    if dy < 0: return 0
    if dx > 0: return 90
    if dy > 0: return 180
    return 270


def _faced(ax, ay, facing, cell):
    if facing == 0: return (ax, ay - cell)
    if facing == 180: return (ax, ay + cell)
    if facing == 90: return (ax + cell, ay)
    return (ax - cell, ay)


class _Model:
    def __init__(self, walls, cell):
        self.walls = walls; self.cell = cell
        self.D = {1: (0, -cell), 2: (0, cell), 3: (-cell, 0), 4: (cell, 0)}

    def step(self, s, a):
        ax, ay, bx, by, facing, held = s
        if a == 5:
            if held:
                return (ax, ay, bx, by, facing, False)
            if _faced(ax, ay, facing, self.cell) == (bx, by):
                return (ax, ay, bx, by, facing, True)
            return s
        dx, dy = self.D[a]
        if not held:
            nf = _facing_of(dx, dy); tgt = (ax + dx, ay + dy)
            if tgt not in self.walls and tgt != (bx, by):
                return (tgt[0], tgt[1], bx, by, nf, held)
            return (ax, ay, bx, by, nf, held)
        offx, offy = bx - ax, by - ay
        nav = (ax + dx, ay + dy); nbl = (bx + dx, by + dy)
        if ((nav not in self.walls or nav == (bx, by)) and (nbl not in self.walls or nbl == (ax, ay))):
            return (nav[0], nav[1], nbl[0], nbl[1], facing, held)
        return s


def _astar(model, start, goal_xy, cell):
    gx, gy = goal_xy
    def h(s): return (abs(s[2] - gx) + abs(s[3] - gy)) // cell
    def win(s): return (s[2], s[3]) == goal_xy and not s[5]
    if win(start): return []
    pq = [(h(start), 0, 0, start, [])]; best = {start: 0}; c = 0
    while pq:
        f, g, _, s, path = heapq.heappop(pq)
        if g > best.get(s, 1 << 30): continue
        for a in (1, 2, 3, 4, 5):
            ns = model.step(s, a); ng = g + 1
            if ng >= best.get(ns, 1 << 30): continue
            if win(ns): return path + [a]
            best[ns] = ng; c += 1
            heapq.heappush(pq, (ng + h(ns), ng, c, ns, path + [a]))
    return None


def _borders(cell):
    w = set()
    for i in range(0, 64, cell):
        w |= {(-cell, i), (64, i), (i, -cell), (i, 64)}
    return w


def _plan(avatar, blocks, goals, cell):
    """assign each block to a distinct goal (greedy), plan placement order for min total actions."""
    from itertools import permutations
    n = len(blocks)
    if n == 0 or len(goals) < n:
        return None
    goals = goals[:n]
    # greedy assignment by nearest
    pairs = sorted((abs(b[0]-p[0])+abs(b[1]-p[1]), bi, pi) for bi, b in enumerate(blocks) for pi, p in enumerate(goals))
    assign = {}; used = set()
    for _, bi, pi in pairs:
        if bi in assign or pi in used: continue
        assign[bi] = goals[pi]; used.add(pi)
    orders = permutations(range(n)) if n <= 6 else [tuple(sorted(range(n), key=lambda i: abs(blocks[i][0]-avatar[0])+abs(blocks[i][1]-avatar[1])))]
    best = None
    borders = _borders(cell)
    for order in orders:
        ax, ay, facing = avatar[0], avatar[1], 0
        placed, full, ok = [], [], True
        for k, i in enumerate(order):
            pend = [blocks[order[m]] for m in range(k + 1, n)]
            walls = borders | set(pend) | set(placed)
            model = _Model(walls, cell)
            path = _astar(model, (ax, ay, blocks[i][0], blocks[i][1], facing, False), assign[i], cell)
            if path is None: ok = False; break
            s = (ax, ay, blocks[i][0], blocks[i][1], facing, False)
            for a in path: s = model.step(s, a)
            ax, ay, facing = s[0], s[1], s[4]; placed.append(assign[i]); full.extend(path)
        if ok and (best is None or len(full) < len(best)):
            best = full
    return best


# ---------------- source-free role perception ----------------
def _anchors(grid, color, cell, lo=6, hi=40):
    out = []
    for o in P.connected_components(grid, background=P.detect_background(grid)):
        if o.color != color or not (lo <= o.size <= hi): continue
        r0, c0, r1, c1 = o.bbox
        if r0 >= 60: continue
        out.append(((c0 // cell) * cell, (r0 // cell) * cell))
    return sorted(set(out))


def _roles(grid, avatar_color):
    bg = P.detect_background(grid)
    comps = {}
    for o in P.connected_components(grid, background=bg):
        comps.setdefault(o.color, []).append(o)
    block_c, best = None, None
    for c, os in comps.items():
        if c in (bg, avatar_color): continue
        small = [o for o in os if o.size <= 30]
        if len(small) >= 2:
            spread = max(o.size for o in small) - min(o.size for o in small)
            sc = (len(small), -spread)
            if best is None or sc > best[0]: best = (sc, c)
    if best: block_c = best[1]
    block_cells = set((round(o.centroid[0]), round(o.centroid[1])) for o in comps.get(block_c, []))
    cands = []
    for c, os in comps.items():
        if c in (bg, avatar_color, block_c): continue
        big = max((o.size for o in os), default=0)
        if big < 12: continue
        marks = sum(1 for o in os if o.size <= 6 and any(abs(o.centroid[0]-bc[0])+abs(o.centroid[1]-bc[1]) < 12 for bc in block_cells))
        cands.append((marks, big, c))
    cands.sort(reverse=True)
    return block_c, [c for (_, _, c) in cands]


def _goal_cells(grid, goal_color, cell):
    bg = P.detect_background(grid); out = []
    for o in P.connected_components(grid, background=bg):
        if o.color != goal_color or o.size < 12: continue
        r0, c0, r1, c1 = o.bbox
        for x in range((c0 // cell) * cell, c1 + 1, cell):
            for y in range((r0 // cell) * cell, r1 + 1, cell):
                out.append((x, y))
    return sorted(set(out))


def _centroids(grid):
    d = {}
    for c in range(16):
        ys, xs = np.where(grid == c)
        if len(ys): d[c] = (ys.mean(), xs.mean(), len(ys))
    return d


# ---------------- reactive strategy ----------------
class GrabDragStrategy:
    EXPECT = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(self, seed: int = 0):
        self.gs = None
        self._reset_state()

    def _reset_state(self):
        self._phase = "learn"       # learn -> plan -> exec -> abstain
        self._moves = None          # available move actions to probe
        self._pi = 0                # probe index (0..2*len(moves)-1)
        self._prev = None           # (grid, action) awaiting its result
        self._disp = {}             # action -> {color: (dr,dc)} accumulated
        self._queue = []
        self._planned_level = -1
        self._avatar = None; self._cell = None; self._goalhyps = None; self._ghi = 0

    def _benign(self, available):
        if 5 in available: return ("S", 5)
        if available: return ("S", int(available[0]))
        return ("reset",)

    def _finish_learn(self, grid):
        bg = P.detect_background(grid); cen = _centroids(grid)
        colors = [c for c in cen if c != bg]
        score = {}
        for c in colors:
            s = 0
            for a in self._moves:
                dr, dc = self._disp.get(a, {}).get(c, (0, 0))
                if abs(dr) + abs(dc) > 0.3:
                    er, ec = self.EXPECT[a]
                    if (er == 0 or dr * er > 0) and (ec == 0 or dc * ec > 0): s += 1
            score[c] = s
        cand = sorted(colors, key=lambda c: (-score[c], cen[c][2]))
        avatar = next((c for c in cand if score[c] >= 2), cand[0] if cand else None)
        if avatar is None:
            self._phase = "abstain"; return
        deltas = []
        for a in self._moves:
            dr, dc = self._disp.get(a, {}).get(avatar, (0, 0))
            deltas.append(abs(round(dr)) + abs(round(dc)))
        nz = [d for d in deltas if d > 0]
        self._cell = min(nz) if nz else 4
        self._avatar = avatar
        block_c, goalhyps = _roles(grid, avatar)
        if block_c is None or not goalhyps:
            self._phase = "abstain"; return
        self._block_c = block_c; self._goalhyps = goalhyps; self._ghi = 0
        self._phase = "plan"

    def _build_queue(self, grid):
        # try the current goal-color hypothesis
        while self._ghi < len(self._goalhyps):
            go_c = self._goalhyps[self._ghi]
            avatar = _anchors(grid, self._avatar, self._cell, 3, 60)
            blocks = _anchors(grid, self._block_c, self._cell)
            goals = _goal_cells(grid, go_c, self._cell)
            if avatar and blocks and goals:
                plan = _plan(avatar[0], blocks, goals, self._cell)
                if plan:
                    self._queue = [("S", a) if a in (1, 2, 3, 4, 5) else a for a in plan]
                    return True
            self._ghi += 1
        return False

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        if gstate_terminal:
            self._reset_state(); return ("reset",)
        # relearn on a new level
        if levels != self._planned_level and self._phase not in ("learn",):
            self._planned_level = levels
            self._reset_state()

        if self._phase == "abstain":
            return self._benign(available)

        if self._phase == "learn":
            self._planned_level = levels
            if self._moves is None:
                self._moves = [a for a in available if a in (1, 2, 3, 4)]
                if not self._moves:
                    self._phase = "abstain"; return self._benign(available)
            # record the result of the previous probe action
            if self._prev is not None:
                pg, pa = self._prev
                cb = _centroids(pg); ca = _centroids(grid)
                for c in cb:
                    if c in ca:
                        dr, dc = ca[c][0] - cb[c][0], ca[c][1] - cb[c][1]
                        pr, pc = self._disp.setdefault(pa, {}).get(c, (0.0, 0.0))
                        self._disp[pa][c] = (pr + dr, pc + dc)
            # probe each move action (once is enough for direction)
            if self._pi < len(self._moves):
                a = self._moves[self._pi]; self._pi += 1
                self._prev = (grid, a)
                return ("S", a)
            # done probing
            self._prev = None
            self._finish_learn(grid)
            if self._phase == "plan":
                if not self._build_queue(grid):
                    self._phase = "abstain"; return self._benign(available)
                self._phase = "exec"
            return self.decide(grid, gstate_terminal, gstate_notplayed, levels, available)

        if self._phase == "exec":
            if self._queue:
                return self._queue.pop(0)
            # plan exhausted; try next goal hypothesis else abstain
            self._ghi += 1
            if self._build_queue(grid):
                return self._queue.pop(0)
            self._phase = "abstain"
            return self._benign(available)

        return self._benign(available)
