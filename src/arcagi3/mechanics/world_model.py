"""Executable WorldModel (winning/mechanic-model-search, Phase 2 / Task 2).

A compact, executable transition model fit from observed (state, action, next_state) triples:

    model.predict_move(grid, agent_pos, action) -> next_pos        # agent dynamics
    model.score_transition(prev, action, grid)  -> float in [0,1]  # held-out fidelity
    model.applicable_actions(grid)              -> list[int]
    model.plan_to(grid, goal, max_nodes)        -> [action_id, ...] | None  # BFS over the model

Planning is grid-cell BFS that moves the agent by the fitted per-action delta, treats walls as
blocked EXCEPT paintable cells (paint opens paths — A1 traversal), and snaps the goal's observed
target cells onto the agent's motion lattice (start + k*delta). The goal's position is read fresh
from the current grid each call, so a MOVING goal is handled by replanning, not by modelling it.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from arcagi3.mechanics.primitives import (
    CollectContact, MoveAgent, PaintOnMove, PushObject, fit_push,
)
from arcagi3.transform_induction import induce_collect_on_contact, induce_recolor_on_move


def _centroid(grid, colors):
    mask = np.isin(grid, list(colors)) if colors else np.zeros(grid.shape, bool)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (int(round(ys.mean())), int(round(xs.mean())))


def _world_deltas(prev, grid, agent_colors):
    out = []
    for r, c in np.argwhere(prev != grid):
        f, t = int(prev[r, c]), int(grid[r, c])
        if f in agent_colors or t in agent_colors:
            continue
        out.append((f, t))
    return out


@dataclass
class WorldModel:
    agent_colors: set
    bg: int
    move: MoveAgent
    paint: PaintOnMove
    collect: CollectContact
    push: PushObject
    width: int = 64
    height: int = 64
    transition_accuracy: float = 0.0          # held-out (movement + world), set at fit time
    move_accuracy: float = 0.0                # held-out MOVEMENT-only — the gate for planning
    start_pos: tuple | None = None            # motion-lattice anchor

    # ---- prediction / scoring -------------------------------------------------
    def agent_pos(self, grid):
        return _centroid(grid, self.agent_colors)

    def _blocked(self, grid, nr, nc):
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return True
        if (nr, nc) in self.move.walls:
            return int(grid[nr, nc]) not in self.paint.map()  # paintable walls are passable
        return False

    def predict_move(self, grid, agent_pos, action):
        d = self.move.deltas.get(action)
        if d is None or agent_pos is None:
            return agent_pos
        nr, nc = agent_pos[0] + d[0], agent_pos[1] + d[1]
        return agent_pos if self._blocked(grid, nr, nc) else (nr, nc)

    def score_transition(self, prev, action, grid):
        """Fraction of the observed change explained: agent displacement + world deltas."""
        hit = tot = 0
        ap, an = self.agent_pos(prev), self.agent_pos(grid)
        if action in self.move.deltas and ap and an:
            actual = (an[0] - ap[0], an[1] - ap[1])
            if actual != (0, 0):
                tot += 1
                if actual == self.move.deltas[action]:
                    hit += 1
        pm, coll = self.paint.map(), self.collect.colors
        for f, t in _world_deltas(prev, grid, self.agent_colors):
            tot += 1
            if (t == self.bg and f in coll) or pm.get(f) == t:
                hit += 1
        return hit / tot if tot else (1.0 if action in self.move.deltas else 0.0)

    def score_move(self, prev, action, grid):
        """Movement-only fidelity for one transition: 1/0 on real moves, None when no move to test."""
        ap, an = self.agent_pos(prev), self.agent_pos(grid)
        if action in self.move.deltas and ap and an:
            actual = (an[0] - ap[0], an[1] - ap[1])
            if actual != (0, 0):
                return 1.0 if actual == self.move.deltas[action] else 0.0
        return None

    def applicable_actions(self, grid):
        return sorted(self.move.deltas)

    def num_primitives(self):
        return sum(bool(p.num_params()) for p in (self.move, self.paint, self.collect, self.push))

    def num_params(self):
        return sum(p.num_params() for p in (self.move, self.paint, self.collect, self.push))

    # ---- planning -------------------------------------------------------------
    def _snap(self, target, start):
        """Snap an observed target cell onto the lattice start + k*delta (per-axis pitch)."""
        prs = [abs(d[0]) for d in self.move.deltas.values() if d[0]]
        pcs = [abs(d[1]) for d in self.move.deltas.values() if d[1]]
        pr = min(prs) if prs else 1
        pc = min(pcs) if pcs else 1
        sr = start[0] + round((target[0] - start[0]) / pr) * pr
        sc = start[1] + round((target[1] - start[1]) / pc) * pc
        return (sr, sc)

    def plan_to(self, grid, goal, max_nodes=120_000):
        """BFS over agent position to reach (any snapped target cell of) `goal`. Returns actions."""
        start = self.agent_pos(grid)
        if start is None:
            return None
        targets = goal.target_cells(grid, self.agent_colors)
        if not targets:
            return None
        snapped = {self._snap(t, start) for t in targets}
        snapped.discard(start)
        if not snapped:
            return None
        seen = {start}
        q = deque([(start, [])])
        nodes = 0
        while q and nodes < max_nodes:
            pos, path = q.popleft()
            nodes += 1
            for a, d in self.move.deltas.items():
                nr, nc = pos[0] + d[0], pos[1] + d[1]
                if self._blocked(grid, nr, nc):
                    continue
                npos = (nr, nc)
                # reach if adjacent-or-on a snapped target (goal.satisfied uses 8-neighbourhood)
                if npos in snapped or any(abs(npos[0] - s[0]) <= abs(d[0] or 1) and
                                          abs(npos[1] - s[1]) <= abs(d[1] or 1) for s in snapped):
                    return path + [a]
                if npos not in seen:
                    seen.add(npos)
                    q.append((npos, path + [a]))
        return None


def fit_world_model(transitions, width=64, height=64):
    """Fit a WorldModel from observed transitions. Identifies the agent by action-correlated motion,
    fits per-action deltas, learns blocked (wall) cells, and induces paint/collect/push."""
    if not transitions:
        return None
    bg = int(Counter(int(c) for _p, _a, g in transitions for c in np.unique(g)).most_common(1)[0][0])
    # background = most common color across frames (cheap, robust)
    bg = int(Counter(int(transitions[-1][2][r, c]) for r in range(0, height, 4)
                     for c in range(0, width, 4)).most_common(1)[0][0])

    # agent = the color whose per-action displacement is DIRECTIONAL and DISCRIMINATIVE — a real
    # avatar moves DIFFERENTLY per action. A paint-trail/drift color shifts the SAME way for every
    # action (e.g. all (0,1)); that must be rejected, not selected (the ls20 agent-ID bug).
    colors = set()
    for _p, _a, g in transitions:
        colors |= {int(c) for c in np.unique(g)}
    cand = [c for c in colors if c != bg]
    best_col, best_score, best_deltas = None, -1.0, {}
    for col in cand:
        shifts: dict = {}
        for prev, aid, grid in transitions:
            if aid is None:
                continue
            cp, cg = _centroid(prev, {col}), _centroid(grid, {col})
            if cp and cg:
                s = (cg[0] - cp[0], cg[1] - cp[1])
                if s != (0, 0):
                    shifts.setdefault(aid, Counter())[s] += 1
        if not shifts:
            continue
        per_action = {a: cnt.most_common(1)[0][0] for a, cnt in shifts.items()}
        distinct = len(set(per_action.values()))
        if distinct < 2:
            continue   # same shift for every action -> drift/trail, not an avatar
        consistency = float(np.mean([cnt.most_common(1)[0][1] / sum(cnt.values()) for cnt in shifts.values()]))
        n_moves = sum(sum(c.values()) for c in shifts.values())
        size = float(np.mean([np.count_nonzero(g == col) for _p, _a, g in transitions]))
        # reward: discriminative (distinct directions) x consistent x enough evidence; prefer small
        score = distinct * consistency * min(1.0, n_moves / 8.0) / (1.0 + size / 50.0)
        if score > best_score:
            best_score, best_col, best_deltas = score, col, per_action
    if best_col is None:
        return None
    agent_colors = {best_col}

    # walls: cells the agent was adjacent-to and a move toward them produced no displacement
    walls = set()
    for prev, aid, grid in transitions:
        if aid is None or aid not in best_deltas:
            continue
        cp, cg = _centroid(prev, agent_colors), _centroid(grid, agent_colors)
        if cp and cg and (cg[0] - cp[0], cg[1] - cp[1]) == (0, 0):
            d = best_deltas[aid]
            walls.add((cp[0] + d[0], cp[1] + d[1]))

    world_obs = [{"from_color": f, "to_color": t, "vanished": t == bg}
                 for prev, _a, grid in transitions for f, t in _world_deltas(prev, grid, agent_colors)]
    paint = PaintOnMove(rules=induce_recolor_on_move(world_obs))
    collect = CollectContact(colors={c.color for c in induce_collect_on_contact(world_obs)})
    push = fit_push(transitions, agent_colors, bg)

    model = WorldModel(agent_colors=agent_colors, bg=bg, move=MoveAgent(best_deltas, walls),
                       paint=paint, collect=collect, push=push, width=width, height=height,
                       start_pos=_centroid(transitions[0][0], agent_colors))
    return model
