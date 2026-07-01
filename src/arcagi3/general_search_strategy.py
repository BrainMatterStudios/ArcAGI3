"""GeneralSearchStrategy — deploys the search-replay paradigm over the reactive decide() interface (the eval
path), zero per-game code. It drives an affordance-pruned BFS by emitting reset/step tokens (a DIRTY search
play), and once a winning sequence is found it caches it and REPLAYS it (a CLEAN play). Under max-over-runs
scoring the clean replay is what scores; the dirty search play is discarded.

Macros searched: the available simple actions (1-5) + a click at each salient object cell (ACTION6 games). The
frame-state (HUD-masked) is the BFS dedup key. Abstains (benign) when the search space is exhausted -> floor-safe.

Token format: ("reset",) | ("S", id) | ("C", x, y).
"""
from __future__ import annotations
from collections import deque
import numpy as np
from arcagi3 import perception as P


def _salient(grid, max_n=16):
    bg = P.detect_background(grid)
    from collections import Counter
    comps = [o for o in P.connected_components(grid, background=bg) if o.color != bg and o.size >= 2]
    cc = Counter(o.color for o in comps)
    out = []
    for o in sorted(comps, key=lambda o: (cc[o.color], -o.size)):
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        if 0 <= cy < 64 and 0 <= cx < 64 and grid[cy, cx] == o.color and all(
                abs(cx - x) + abs(cy - y) > 2 for (x, y) in out):
            out.append((cx, cy))
        if len(out) >= max_n:
            break
    return out


class GeneralSearchStrategy:
    def __init__(self, seed: int = 0, max_candidates: int = 4000):
        self.gs = None
        self.max_candidates = max_candidates
        self._reset_all()

    def _reset_all(self):
        self.macros = None
        self.queue = deque([[]])
        self.seen = set()
        self.cur_seq = None
        self.cur_flat = []
        self.pending = []
        self.awaiting_eval = False
        self.candidates = 0
        self.mode = "search"          # search -> replay -> done
        self.solution = None
        self.rp = 0
        self.planned_level = -1

    def _benign(self, available):
        for a in (5, 1, 2, 3, 4):
            if a in available:
                return ("S", a)
        return ("reset",)

    def _init(self, grid, available):
        macros = [[("S", a)] for a in available if a in (1, 2, 3, 4, 5)]
        if 6 in available:
            macros += [[("C", cx, cy)] for (cx, cy) in _salient(grid)]
        self.macros = macros or [[("S", a)] for a in (available or [1])]

    def decide(self, grid, gstate_terminal=False, gstate_notplayed=False, levels=0, available=()):
        available = list(available or [])
        if self.macros is None:
            self._init(grid, available); self.planned_level = levels

        # a level-up during SEARCH means the current candidate wins -> cache + switch to clean replay
        if self.mode == "search" and levels > self.planned_level:
            self.solution = list(self.cur_flat)
            self.mode = "replay"; self.rp = 0
            return ("reset",)                       # open a fresh (clean) play, then replay

        if self.mode == "replay":
            if self.rp < len(self.solution):
                t = self.solution[self.rp]; self.rp += 1
                return t
            self.mode = "done"
            return self._benign(available)
        if self.mode == "done":
            return self._benign(available)

        # SEARCH mode
        if gstate_terminal and self.pending:
            self.pending = []                       # candidate died; drop it and move on
        if self.pending:
            return self.pending.pop(0)

        # a candidate just finished executing -> evaluate its result frame + expand the frontier
        if self.awaiting_eval:
            self.awaiting_eval = False
            if not gstate_terminal:
                st = hash(grid[:56, :56].tobytes())
                if st not in self.seen:
                    self.seen.add(st)
                    for m in self.macros:
                        self.queue.append(self.cur_seq + [m])

        if not self.queue or self.candidates >= self.max_candidates:
            self.mode = "done"
            return self._benign(available)

        self.cur_seq = self.queue.popleft()
        self.cur_flat = [tok for m in self.cur_seq for tok in m]
        self.pending = [("reset",)] + list(self.cur_flat)
        self.candidates += 1
        self.awaiting_eval = True
        return self.pending.pop(0)
