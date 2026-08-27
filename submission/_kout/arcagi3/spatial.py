"""Spatial scene model: learn an occupancy map of the avatar's world and A*-pathfind.

Part of the structured world-model rebuild. The agent's avatar moves on a lattice (each
simple action shifts its centroid by a roughly-constant delta). By recording which lattice
positions the avatar successfully entered (free) versus tried-and-was-blocked (wall), we
build an occupancy map and plan optimal paths to any target with A* — instead of greedy
navigation that stalls at the first obstacle. Unknown cells are treated as free
(optimistic), so the planner routes around *known* walls and probes the unknown.

Assumes axis-aligned movement (the dominant ARC-AGI-3 control scheme). If deltas aren't
axis-aligned/consistent, the caller should not use this and fall back to graph exploration.
"""

from __future__ import annotations

import heapq
from math import gcd


class OccupancyMap:
    def __init__(self, deltas: dict[int, tuple[int, int]]):
        # keep only nonzero, axis-aligned deltas (one axis zero)
        self.deltas = {
            a: (dr, dc) for a, (dr, dc) in deltas.items()
            if (dr, dc) != (0, 0) and (dr == 0 or dc == 0)
        }
        mags = [abs(dr) or abs(dc) for dr, dc in self.deltas.values()]
        self.step = _gcd_list(mags) if mags else 1
        self.origin: tuple[float, float] | None = None
        self.free: set[tuple[int, int]] = set()
        self.blocked: set[tuple[int, int]] = set()

    @property
    def usable(self) -> bool:
        # need axis-aligned moves covering both axes to pathfind in 2D
        haves_row = any(dr != 0 for dr, dc in self.deltas.values())
        haves_col = any(dc != 0 for dr, dc in self.deltas.values())
        return self.step > 0 and haves_row and haves_col

    def quantize(self, centroid: tuple[float, float]) -> tuple[int, int]:
        if self.origin is None:
            self.origin = centroid
        r0, c0 = self.origin
        return (round((centroid[0] - r0) / self.step), round((centroid[1] - c0) / self.step))

    def _step_units(self, dr: int, dc: int) -> tuple[int, int]:
        return (int(round(dr / self.step)), int(round(dc / self.step)))

    def observe_move(self, before: tuple[float, float], action_id: int,
                     after: tuple[float, float]) -> None:
        """Record the outcome of a simple move for occupancy learning."""
        if action_id not in self.deltas:
            return
        qb = self.quantize(before)
        qa = self.quantize(after)
        self.free.add(qb)
        dr, dc = self.deltas[action_id]
        ur, uc = self._step_units(dr, dc)
        target = (qb[0] + ur, qb[1] + uc)
        if qa == qb:
            # avatar didn't move -> the target cell is blocked (wall/boundary)
            self.blocked.add(target)
        else:
            self.free.add(qa)

    def astar(self, start: tuple[float, float], goal: tuple[float, float]) -> list[int] | None:
        """Return a list of action_ids moving the avatar from start to goal, or None.

        Plans over the lattice: known-blocked cells are walls; unknown cells are assumed
        free (optimistic). Goal is matched at lattice resolution.
        """
        if not self.usable:
            return None
        qs = self.quantize(start)
        qg = self.quantize(goal)
        if qs == qg:
            return []
        moves = [(a, self._step_units(dr, dc)) for a, (dr, dc) in self.deltas.items()]

        def h(p):
            return abs(p[0] - qg[0]) + abs(p[1] - qg[1])

        openh = [(h(qs), 0, qs, [])]
        seen = {qs: 0}
        bound = 4 * (abs(qs[0] - qg[0]) + abs(qs[1] - qg[1]) + 4)  # avoid runaway in open space
        while openh:
            f, g, pos, path = heapq.heappop(openh)
            if pos == qg:
                return path
            if g > bound:
                continue
            for a, (ur, uc) in moves:
                npos = (pos[0] + ur, pos[1] + uc)
                if npos in self.blocked:
                    continue
                ng = g + 1
                if npos in seen and seen[npos] <= ng:
                    continue
                seen[npos] = ng
                heapq.heappush(openh, (ng + h(npos), ng, npos, path + [a]))
        return None


def _gcd_list(xs: list[int]) -> int:
    g = 0
    for x in xs:
        g = gcd(g, int(x))
    return g or 1
