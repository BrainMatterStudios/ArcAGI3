"""GHP — Goal-Hypothesis Planner. A GENERAL, source-free agent that (1) learns the action model online
(which object is the avatar + per-action displacement; whether ACTION5/6 cause changes), (2) hypothesizes
the goal as a salient visible target (Exp-2: zero-game goals are visible), (3) plans an exact path to each
goal hypothesis and tries interaction there, keeping whichever hypothesis triggers a reward.

This is the offline-legal, generalizable core of Lever B: unlike the wa30 planner it reads NO source — it
would run identically on a hidden game. Deliberately general so we can test what it cracks across many games.
"""
from __future__ import annotations
import heapq
from collections import Counter, deque
import numpy as np
from arcengine import GameAction, GameState
from arcagi3 import perception as P


def centroids_by_color(grid):
    d = {}
    for c in range(16):
        ys, xs = np.where(grid == c)
        if len(ys):
            d[c] = (ys.mean(), xs.mean(), len(ys))
    return d


class GHP:
    def __init__(self, seed=0):
        self.deltas = {}          # action -> (dr, dc) avatar displacement
        self.avatar_color = None
        self.can5 = self.can6 = False
        self.cell = 1

    # expected avatar displacement SIGN per action: A1 up(row-), A2 down(row+), A3 left(col-), A4 right(col+)
    EXPECT = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    # ---- learn the action model by probing (directional + camera-aware) ----
    def learn(self, env, obs):
        avail = [a for a in (obs.available_actions or [])]
        self.can5 = 5 in avail
        self.can6 = 6 in avail
        moves = [a for a in avail if a in (1, 2, 3, 4)]
        # measure per-color displacement vector under each move (average over 2 probes for stability)
        disp = {a: {} for a in moves}   # action -> {color: (dr,dc)}
        seqobs = obs
        for a in moves:
            for _ in range(2):
                b = centroids_by_color(P.to_grid(seqobs.frame))
                seqobs = env.step(GameAction.from_id(a))
                n = centroids_by_color(P.to_grid(seqobs.frame))
                for c in b:
                    if c in n:
                        dr, dc = n[c][0]-b[c][0], n[c][1]-b[c][1]
                        pr, pc = disp[a].get(c, (0.0, 0.0))
                        disp[a][c] = (pr+dr, pc+dc)
        g = P.to_grid(seqobs.frame); bg = P.detect_background(g); sizes = centroids_by_color(g)
        # camera detection: if under some action MOST colors share a common nonzero displacement, that's a pan
        colors = [c for c in sizes if c != bg]
        # directional score: for each color, count actions where its displacement sign matches EXPECT
        score = {}
        for c in colors:
            s = 0; moved = 0
            for a in moves:
                dr, dc = disp[a].get(c, (0, 0))
                if abs(dr) + abs(dc) > 0.3:
                    moved += 1
                    er, ec = self.EXPECT[a]
                    if (er == 0 or (dr * er) > 0) and (ec == 0 or (dc * ec) > 0):
                        s += 1
            score[c] = (s, moved)
        # avatar = color with best directional match (prefer >=2 matching actions), tie-break smaller size
        cand = sorted(colors, key=lambda c: (-score[c][0], sizes[c][2]))
        self.avatar_color = None
        for c in cand:
            if score[c][0] >= 2:
                self.avatar_color = c; break
        if self.avatar_color is None and cand:
            self.avatar_color = cand[0]
        # per-action delta for the chosen avatar (average of the 2 probes)
        if self.avatar_color is not None:
            for a in moves:
                dr, dc = disp[a].get(self.avatar_color, (0, 0))
                self.deltas[a] = (round(dr/2), round(dc/2))
            nz = [abs(v[0])+abs(v[1]) for v in self.deltas.values() if v != (0, 0)]
            self.cell = min(nz) if nz else 1
        self.camera = self._camera_panning(disp, moves, bg, colors)
        return seqobs

    def _camera_panning(self, disp, moves, bg, colors):
        """heuristic: many colors sharing a common displacement under a move => camera follows the avatar."""
        for a in moves:
            vecs = [disp[a].get(c, (0, 0)) for c in colors]
            vecs = [(round(r), round(c)) for (r, c) in vecs if abs(r)+abs(c) > 0.3]
            if len(vecs) >= 4 and len(set(vecs)) <= 2:
                return True
        return False

    def avatar_pos(self, grid):
        c = centroids_by_color(grid).get(self.avatar_color)
        return (c[0], c[1]) if c else None

    # ---- goal hypotheses: salient distinct static objects, ranked ----
    def goal_hypotheses(self, grid):
        bg = P.detect_background(grid)
        objs = [o for o in P.connected_components(grid, background=bg)
                if o.color not in (bg, self.avatar_color)]
        col_counts = Counter(int(v) for v in grid.flatten())
        scored = []
        for o in objs:
            rarity = 1.0 / (col_counts[o.color] + 1)
            compact = 1.0 / (o.size + 1)
            scored.append((rarity + 0.3 * compact, o.centroid, o.color, o.size))
        scored.sort(reverse=True)
        # dedup by rounded centroid
        seen = set(); hyps = []
        for s, cen, col, sz in scored:
            k = (round(cen[0]), round(cen[1]))
            if k in seen:
                continue
            seen.add(k); hyps.append((cen, col, sz))
        return hyps[:8]

    # ---- A* over free cells using learned deltas ----
    def plan_to(self, grid, target, max_steps=4000):
        start = self.avatar_pos(grid)
        if start is None or not self.deltas:
            return None
        cell = self.cell
        obst = self._obstacles(grid)
        sr, sc = int(round(start[0])), int(round(start[1]))
        tr, tc = int(round(target[0])), int(round(target[1]))
        # snap to avatar grid
        def h(r, c): return (abs(r-tr)+abs(c-tc)) / max(cell, 1)
        pq = [(h(sr, sc), 0, sr, sc, [])]
        best = {(sr, sc): 0}
        while pq:
            f, gg, r, c, path = heapq.heappop(pq)
            if abs(r-tr)+abs(c-tc) <= cell:
                return path
            if gg > best.get((r, c), 1e9) or len(path) > 200:
                continue
            for a, (dr, dc) in self.deltas.items():
                if (dr, dc) == (0, 0):
                    continue
                nr, nc = r+dr, c+dc
                if not (0 <= nr < 64 and 0 <= nc < 64) or (nr, nc) in obst:
                    continue
                ng = gg+1
                if ng < best.get((nr, nc), 1e9):
                    best[(nr, nc)] = ng
                    heapq.heappush(pq, (ng + h(nr, nc), ng, nr, nc, path+[a]))
        return None

    def _obstacles(self, grid):
        # coarse: non-background, non-avatar static cells are walls (blocks avatar)
        bg = P.detect_background(grid)
        obst = set()
        ys, xs = np.where((grid != bg) & (grid != (self.avatar_color if self.avatar_color is not None else -1)))
        for r, c in zip(ys.tolist(), xs.tolist()):
            obst.add((r, c))
        return obst
