"""GoalDirectedSlideExplorer — slide-nav that STEERS toward a learned goal-object (Exp #9).

The slide-nav kill was about UNDIRECTED coverage. But mechanic inference reliably finds the
avatar + slide model, and the level-up frame reveals what the avatar reached. So: explore
until the first level-up, detect the GOAL-OBJECT COLOR (the rare object the avatar was nearest
to when the level flipped), then on later levels navigate the avatar DIRECTLY to that color
instead of blindly covering the maze. If a game's deeper levels reuse the same goal-object,
this can reach L2+ where undirected exploration walls at L1.

Subclasses SlideNavExplorer (inherits avatar probe, slide-graph, push-safety gate, stall
trigger). Goal-direction only kicks in AFTER a goal color is learned; before that it is plain
slide-nav. enable_slide=False -> delegate (== v6 firewall).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import perception as P
from .slide_nav_explorer import SlideNavExplorer


class GoalDirectedSlideExplorer(SlideNavExplorer):
    def reset_all(self):
        super().reset_all()
        self.goal_color = None
        self._gd_prev_grid = None
        self._gd_prev_cell = None
        self._gd_prev_levels = 0

    def _rare_color_near(self, grid, cell):
        """Color of the nearest rare (count<=2) non-avatar object to `cell`."""
        objs = P.connected_components(grid, background=self.bg)
        counts = {}
        for o in objs:
            counts[o.color] = counts.get(o.color, 0) + 1
        best, bestd = None, 1e9
        for o in objs:
            if o.color in self.avatar_cols or counts[o.color] > 2:
                continue
            d = abs(o.centroid[0] - cell[0]) + abs(o.centroid[1] - cell[1])
            if d < bestd:
                best, bestd = int(o.color), d
        return best

    def _goal_target_cells(self, grid):
        """Lattice cells of the avatar that sit adjacent to a goal-colored object."""
        if self.goal_color is None:
            return set()
        targets = set()
        cells = np.argwhere(grid == self.goal_color)
        for (r, c) in cells:
            targets.add((int(r), int(c)))
        return targets

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        # detect the goal object at the moment of a level-up (pre-flip frame + avatar position)
        if (self.enable_slide and self.goal_color is None and levels > self._gd_prev_levels
                and self._gd_prev_grid is not None and self._gd_prev_cell is not None
                and self.avatar_cols):
            self.goal_color = self._rare_color_near(self._gd_prev_grid, self._gd_prev_cell)
        self._gd_prev_levels = levels
        # remember this step's grid + avatar cell for the next-step level-up attribution
        if self.avatar_cols:
            cm = self._avatar_centroid(grid)
            self._gd_prev_cell = self._quant(cm) if cm is not None else self._gd_prev_cell
        self._gd_prev_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _bfs_to_frontier(self, start):
        # once a goal color is known, steer toward avatar cells on/adjacent to a goal object;
        # fall back to plain frontier exploration if no goal route is known yet.
        if self.goal_color is not None and self._cur_grid is not None:
            goal_cells = self._goal_target_cells(self._cur_grid)
            if goal_cells:
                path = self._bfs_to_targets(start, goal_cells)
                if path:
                    return path
        return super()._bfs_to_frontier(start)

    def _bfs_to_targets(self, start, goal_cells):
        """BFS over learned slide-edges toward any avatar-cell whose (r,c) is in goal_cells."""
        q = deque([(start, [])])
        seen = {start}
        while q:
            s, path = q.popleft()
            for a in self._avail:
                if a not in self.deltas:
                    continue
                nxt = self.edges.get((s, a))
                if nxt is None or nxt in seen:
                    continue
                seen.add(nxt)
                npath = path + [a]
                if (nxt[0], nxt[1]) in goal_cells:   # reached a goal-adjacent cell
                    return npath
                q.append((nxt, npath))
        return None
