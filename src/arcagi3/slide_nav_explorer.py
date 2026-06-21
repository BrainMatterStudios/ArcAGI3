"""SlideNavExplorer — inference-gated slide-aware spatial explorer (Increment 2).

Increment 1 proved wall games like ls20 have a CLEAN slide-to-wall avatar (color 12, 5-cell
slide). The graph explorer keys state on the full board hash, so it re-discovers spatially-
equivalent avatar positions and re-walks from root — expensive on big mazes. This explorer
collapses state to (avatar lattice-cell, #collectibles-remaining): far fewer states, so it
covers the maze's reachable avatar positions in far fewer actions, and reaches the goal
faster. The collectible count is the minimal board augmentation that distinguishes
before/after a key pickup (so key->door progress isn't lost to over-merging).

GATED for safety: a short online probe identifies the avatar (the action-dependent mover) and
checks for a clean axis-aligned mover with walls and NO push (a non-avatar object moving with
the avatar). If the game isn't a clean nav game, it DELEGATES every decision to an internal
SalienceExplorer (== v6, firewall) — this is exactly the gate the killed un-gated spatial.py
lacked (it pushed sokoban blocks wrongly). enable_slide=False also forces pure delegation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import perception as P
from .salience_explorer import SalienceExplorer

SIMPLE = [1, 2, 3, 4]
# action_id -> unit direction (row, col). Matches the inferred ls20 mapping (1=up,2=down,3=left,4=right).
_DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


class SlideNavExplorer:
    def __init__(self, seed: int = 0, trust_threshold: int = 3, border_mask: int = 2,
                 enable_slide: bool = True, probe_steps: int = 24, stall_trigger: int = 1500,
                 coarse_grid_step: int = 8, max_click_targets: int = 96) -> None:
        self.enable_slide = bool(enable_slide)
        self.probe_steps = int(probe_steps)
        # STALL-TRIGGERED engagement (additive-by-construction): delegate to SalienceExplorer by
        # default; only switch to spatial nav after the explorer has gone stall_trigger actions
        # with no level-up. Cheap/progressing games (tr87 L1@212, tu93) never trigger it -> no
        # regression; genuinely-stuck mazes (ls20) get the slide-nav rescue. 0 disables (engage
        # immediately, the un-gated behaviour).
        self.stall_trigger = int(stall_trigger)
        self.fallback = SalienceExplorer(seed=seed, trust_threshold=trust_threshold,
                                         border_mask=border_mask, coarse_grid_step=coarse_grid_step,
                                         max_click_targets=max_click_targets)
        self.reset_all()

    # duck-typing for the runner's states_seen
    @property
    def gs(self):
        return self.fallback

    def reset_all(self):
        if hasattr(self.fallback, "reset_all"):
            self.fallback.reset_all()
        self.bg = None
        # watch: delegate to fallback + watch for a stall; probe: identify avatar; nav: spatial; delegate: pure fallback
        self.mode = "watch" if self.stall_trigger > 0 else "probe"
        self._since_level = 0
        self._watch_levels = 0
        self._tried_probe = False
        self.avatar_cols: set[int] = set()
        self.deltas: dict[int, tuple[int, int]] = {}    # action_id -> dominant (dr,dc)
        self._probe_hist: list = []  # (action_id, prev_centroid, centroid)
        self._probe_prev_cm = None
        self._probe_n = 0
        self._avail = SIMPLE
        # nav state
        self.visited: set = set()                # (qr,qc,nobj) lattice states seen
        self.tried: dict = {}                    # (qr,qc,nobj) -> set(action_id) tried from here
        self.edges: dict = {}                    # ((qr,qc,nobj),action) -> (qr,qc,nobj)
        self.origin = None
        self.step = 1
        self.plan: list = []
        self.prev_levels = 0
        self.prev_cell = None
        self.prev_action = None

    # ---- avatar geometry ----
    def _avatar_centroid(self, grid):
        if not self.avatar_cols:
            return None
        cells = np.argwhere(np.isin(grid, list(self.avatar_cols)))
        return tuple(cells.mean(axis=0)) if len(cells) else None

    def _quant(self, centroid):
        # slide stops at arbitrary wall-adjacent cells -> key on the raw integer cell, no lattice
        return (int(round(centroid[0])), int(round(centroid[1])))

    def _ncollect(self, grid):
        """Coarse board augmentation: # of non-bg, non-avatar objects (drops when one is collected)."""
        objs = P.connected_components(grid, background=self.bg)
        return sum(1 for o in objs if o.color not in self.avatar_cols)

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        self._avail = [a for a in SIMPLE if a in available]
        # delegate path (gate failed / disabled / not a pure-arrow game)
        if not self.enable_slide or self.mode == "delegate" or 6 in available or not self._avail:
            return self.fallback.decide(grid, gstate_terminal, gstate_notplayed, levels, available)

        # WATCH: run the fallback explorer, count actions since the last level-up; only switch to
        # spatial nav once it has clearly STALLED (additive — games it handles never trigger us).
        if self.mode == "watch":
            if levels > self._watch_levels:
                self._watch_levels = levels
                self._since_level = 0
            if not gstate_terminal and not gstate_notplayed:
                self._since_level += 1
            if (self._since_level >= self.stall_trigger and not self._tried_probe
                    and not gstate_terminal and not gstate_notplayed):
                self.mode = "probe"          # stalled -> try to identify the avatar
                self._tried_probe = True
                self.prev_action = None
            else:
                return self.fallback.decide(grid, gstate_terminal, gstate_notplayed, levels, available)

        if gstate_terminal or gstate_notplayed:
            self.prev_action = None
            self.plan = []
            return ("reset",)

        if self.mode == "probe":
            return self._probe(grid, levels)
        return self._nav(grid, levels)

    # ---- phase 1: identify the avatar online ----
    def _probe(self, grid, levels):
        # record outcome of the previous probe action
        if self.prev_action is not None and self._probe_prev_cm is not None:
            cm = self._all_centroids(grid)
            self._probe_hist.append((self.prev_action, self._probe_prev_cm, cm))
        self._probe_n += 1
        if self._probe_n > self.probe_steps:
            self._finish_probe()
            if self.mode == "delegate":
                return self.fallback.decide(grid, False, False, levels, self._avail)
            return self._nav(grid, levels)
        a = self._avail[self._probe_n % len(self._avail)]
        self._probe_prev_cm = self._all_centroids(grid)
        self.prev_action = a
        return ("S", a)

    def _all_centroids(self, grid):
        out = {}
        for o in P.connected_components(grid, background=self.bg):
            out.setdefault(int(o.color), []).append(o)
        # centroid per color over all its cells
        res = {}
        for col, objs in out.items():
            cells = np.vstack([np.array(o.cells) for o in objs])
            res[col] = (cells[:, 0].mean(), cells[:, 1].mean())
        return res

    def _finish_probe(self):
        # per color: map action -> set of deltas; avatar = action-dependent axis-aligned mover
        from collections import defaultdict
        moves = defaultdict(lambda: defaultdict(list))  # color -> action -> [(dr,dc)]
        movers_per_step = []  # set of colors that moved each step (for push detection)
        for (a, before, after) in self._probe_hist:
            moved = set()
            for col in set(before) & set(after):
                dr = after[col][0] - before[col][0]
                dc = after[col][1] - before[col][1]
                if abs(dr) < 0.5 and abs(dc) < 0.5:
                    continue
                moves[col][a].append((dr, dc))
                moved.add(col)
            movers_per_step.append(moved)
        # avatar candidate: most-moving color whose direction depends on action and is axis-aligned
        best, best_n = None, 0
        for col, by_a in moves.items():
            n = sum(len(v) for v in by_a.values())
            dirs = set()
            for a, ds in by_a.items():
                mr = np.median([d[0] for d in ds]); mc = np.median([d[1] for d in ds])
                if abs(mr) >= abs(mc):
                    dirs.add(("r", 1 if mr > 0 else -1))
                else:
                    dirs.add(("c", 1 if mc > 0 else -1))
            if len(by_a) >= 2 and len(dirs) >= 2 and n > best_n:
                best, best_n = col, n
        if best is None:
            self.mode = "delegate"
            return
        # GATE: if a non-avatar object ever moved together with the avatar -> push mechanic
        # (sokoban). Spatial nav would push blocks wrongly (the killed spatial.py failure) -> delegate.
        if any(len(m - {best}) > 0 for m in movers_per_step if best in m):
            self.mode = "delegate"
            return
        self.avatar_cols = {best}
        # derive per-action unit direction + slide step (median movement magnitude)
        by_a = moves[best]
        mags = []
        for a, ds in by_a.items():
            mr = np.median([d[0] for d in ds]); mc = np.median([d[1] for d in ds])
            if abs(mr) >= abs(mc):
                self.deltas[a] = (1 if mr > 0 else -1, 0)
                mags.append(abs(mr))
            else:
                self.deltas[a] = (0, 1 if mc > 0 else -1)
                mags.append(abs(mc))
        # need both axes to navigate in 2D
        if not (any(d[0] for d in self.deltas.values()) and any(d[1] for d in self.deltas.values())):
            self.mode = "delegate"; return
        self.step = max(1, int(round(np.median(mags)))) if mags else 1
        self.mode = "nav"
        self.origin = None
        self.prev_action = None
        self.prev_cell = None

    # ---- phase 2: slide-graph exploration over (avatar-cell, ncollect) ----
    def _state(self, grid):
        cm = self._avatar_centroid(grid)
        if cm is None:
            return None
        q = self._quant(cm)
        return (q[0], q[1], self._ncollect(grid))

    def _nav(self, grid, levels):
        st = self._state(grid)
        if st is None:
            self.mode = "delegate"
            return self.fallback.decide(grid, False, False, levels, self._avail)
        # record edge from previous step
        if self.prev_cell is not None and self.prev_action is not None:
            self.edges[(self.prev_cell, self.prev_action)] = st
            self.tried.setdefault(self.prev_cell, set()).add(self.prev_action)
        self.visited.add(st)
        self.tried.setdefault(st, set())
        if levels > self.prev_levels:
            self.plan = []   # new level: re-explore fresh
        self.prev_levels = levels

        # follow an active plan
        if self.plan:
            a = self.plan.pop(0)
            self.prev_cell, self.prev_action = st, a
            return ("S", a)
        # untried direction here?
        untried = [a for a in self._avail if a in self.deltas and a not in self.tried[st]]
        if untried:
            a = untried[0]
            self.prev_cell, self.prev_action = st, a
            return ("S", a)
        # navigate to nearest state with an untried direction (BFS over known edges)
        path = self._bfs_to_frontier(st)
        if path:
            self.plan = path[1:]
            a = path[0]
            self.prev_cell, self.prev_action = st, a
            return ("S", a)
        # nothing reachable to explore -> hand off to fallback
        self.mode = "delegate"
        return self.fallback.decide(grid, False, False, levels, self._avail)

    def _bfs_to_frontier(self, start):
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
                if any(x not in self.tried.get(nxt, set()) for x in self._avail if x in self.deltas):
                    return npath
                q.append((nxt, npath))
        return None

    def __len__(self):
        return len(self.visited) or len(self.fallback)
