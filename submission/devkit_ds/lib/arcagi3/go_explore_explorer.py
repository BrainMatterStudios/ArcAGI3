"""GoExploreExplorer — cell-archive exploration with deterministic replay (Go-Explore, Nature 2021).

The canonical method for sparse-reward + deterministic + huge-state (= our exact wall). Its two
named failure modes (detachment = forgetting how to reach a promising state; derailment = failing
to reliably return) are precisely our re-traversal audit's findings. Key idea we tested WRONG
before: the cell function is DELIBERATELY lossy at an INTERMEDIATE granularity (we tried exact =
explodes, and relational = craters; never count-based intermediate coarsening).

cell = block-downsampled dominant-color signature of the (HUD-masked) grid. Archive maps each
cell -> the SHORTEST action trace that reaches it (from reset) + a visit count. Loop: sample a
cell (favoring rarely-visited), RETURN by reset+replaying its trace (deterministic), then EXPLORE
a few actions; archive any newly-reached cells (overwrite on shorter trace = anti-detachment).
On level-up the achieving trace is the solution. (Resets/replay DO count toward the score, so the
edge is better coverage-per-action via cell selection + intermediate coarsening, not free replay.)

Firewall: enable_goexplore=False -> delegates to SalienceExplorer (== v6).
"""

from __future__ import annotations

import numpy as np

from . import perception as P
from .salience_explorer import SalienceExplorer

SIMPLE = [1, 2, 3, 4, 5]


class GoExploreExplorer:
    def __init__(self, seed: int = 0, trust_threshold: int = 3, border_mask: int = 2,
                 enable_goexplore: bool = True, block: int = 8, explore_steps: int = 8,
                 coarse_grid_step: int = 8, max_click_targets: int = 96) -> None:
        self.enable_goexplore = bool(enable_goexplore)
        self.block = int(block)
        self.explore_steps = int(explore_steps)
        self.rng = np.random.default_rng(seed)
        self.fallback = SalienceExplorer(seed=seed, trust_threshold=trust_threshold,
                                         border_mask=border_mask, coarse_grid_step=coarse_grid_step,
                                         max_click_targets=max_click_targets)
        self.reset_all()

    @property
    def gs(self):
        return self

    @property
    def wm(self):
        return self.archive

    def __len__(self):
        return len(self.archive)

    def reset_all(self):
        if hasattr(self.fallback, "reset_all"):
            self.fallback.reset_all()
        self.bg = None
        self.archive: dict = {}          # cell -> (trace tuple, visits)
        self.cur_trace: list = []        # actions since last reset
        self.mode = "explore"            # explore | return
        self.replay: list = []           # remaining actions to replay (return mode)
        self._avail = SIMPLE
        self._since_explore = 0
        self._cands = None
        self.prev_levels = 0

    def _cell(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        b = self.block
        h, w = grid.shape
        sig = []
        for r in range(0, h, b):
            for c in range(0, w, b):
                blk = grid[r:r+b, c:c+b]
                vals, counts = np.unique(blk, return_counts=True)
                sig.append(int(vals[int(counts.argmax())]))
        return tuple(sig)

    def _archive_cell(self, cell):
        cur = tuple(self.cur_trace)
        if cell not in self.archive or len(cur) < len(self.archive[cell][0]):
            self.archive[cell] = (cur, self.archive.get(cell, (None, 0))[1])
        t, v = self.archive[cell]
        self.archive[cell] = (t, v + 1)

    def _sample_cell(self):
        cells = list(self.archive)
        if not cells:
            return None
        # favor rarely-visited cells (count-based)
        w = np.array([1.0 / np.sqrt(self.archive[c][1] + 1) for c in cells])
        w = w / w.sum()
        return cells[int(self.rng.choice(len(cells), p=w))]

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if not self.enable_goexplore:
            return self.fallback.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._avail = [a for a in SIMPLE if a in available]
        click_ok = 6 in available
        if self.bg is None:
            self.bg = P.detect_background(grid)

        if gstate_terminal or gstate_notplayed:
            self.cur_trace = []
            return ("reset",)

        # level-up -> keep exploring the new level fresh
        if levels > self.prev_levels:
            self.prev_levels = levels
            self.archive = {}; self.cur_trace = []; self.mode = "explore"; self.replay = []

        cell = self._cell(grid)
        self._archive_cell(cell)

        # RETURN mode: replay the stored trace toward a sampled cell
        if self.mode == "return":
            if self.replay:
                a = self.replay.pop(0)
                self.cur_trace.append(a)
                return self._emit(a, grid, click_ok)
            self.mode = "explore"; self._since_explore = 0

        # EXPLORE mode: take a few actions, then go-to a fresh cell
        if self._since_explore >= self.explore_steps:
            target = self._sample_cell()
            if target is not None and self.archive[target][0]:
                self.replay = list(self.archive[target][0])
                self.mode = "return"; self.cur_trace = []
                return ("reset",)   # return starts from a clean reset
            self._since_explore = 0

        self._since_explore += 1
        a = self._explore_action(grid, click_ok)
        if a[0] != "reset":
            self.cur_trace.append(a)
        return a

    def _explore_action(self, grid, click_ok):
        # mix of simple actions and a salient click (object-centric), random among available
        opts = list(self._avail)
        if click_ok:
            opts.append(6)
        if not opts:
            return ("S", 1)
        a = opts[int(self.rng.integers(0, len(opts)))]
        if a == 6:
            tgts = P.salient_click_targets(grid, max_targets=64)
            if tgts:
                x, y, _ = tgts[int(self.rng.integers(0, len(tgts)))]
                return ("C", int(x), int(y))
            return ("C", int(self.rng.integers(0, 64)), int(self.rng.integers(0, 64)))
        return ("S", a)

    def _emit(self, a, grid, click_ok):
        return a
