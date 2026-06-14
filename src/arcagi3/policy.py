"""Reactive hybrid policy: one action per call, state persisted on the object.

This is the submission-shaped form of the agent. The Kaggle eval drives agents via the
official `Agent.choose_action(frames, latest_frame) -> GameAction` interface (one action
at a time, result observed on the next call). HybridPolicy implements the same strategy
as HybridAgent (motion model + coordinate navigation + graph-exploration fallback) but as
an incremental state machine, so it works both through the official framework and through
our own reactive runner / offline env.

Action tokens: ("reset",) | ("S", id) | ("C", x, y) — the caller maps these to GameAction.
"""

from __future__ import annotations

import numpy as np

from . import movement as MV
from . import perception as P
from .agent import ACT, RESET, STOP, GraphStrategy, candidates_for
from .world_model import Action


class HybridPolicy:
    def __init__(self, use_clicks: bool = True, max_click_targets: int = 96,
                 nav_step_cap: int = 200, seed: int = 0) -> None:
        self.use_clicks = use_clicks
        self.max_click_targets = max_click_targets
        self.nav_step_cap = nav_step_cap
        self.rng = np.random.default_rng(seed)
        self.reset_all()

    def reset_all(self) -> None:
        self.vt = P.VolatilityTracker()
        self.root_key: bytes | None = None
        self.gs: GraphStrategy | None = None
        self.bg: int | None = None
        self.prev_key: bytes | None = None
        self.prev_action: Action | None = None
        self.prev_levels = 0
        self.level = -1
        self.expect_reset = False
        # motion / phase
        self.phase = "probe"
        self.mm: MV.MotionModel | None = None
        self._votes: dict[int, dict[int, tuple[int, int]]] = {}
        self._changed_colors: set[int] = set()
        self.distractor_colors: set[int] = set()  # animated/counter colors to mask + ignore
        self._probe_queue: list[int] | None = None
        self._probe_before: np.ndarray | None = None
        self._probe_aid: int | None = None
        # navigation
        self.target: tuple[int, int] | None = None
        self.tried_targets: set[tuple[int, int]] = set()
        self.nav_steps = 0
        self.nav_stale = 0
        self.nav_last: tuple[float, float] | None = None

    def _cands(self, grid, available):
        return candidates_for(grid, available, self.use_clicks, self.max_click_targets, False)

    def _key(self, grid: np.ndarray) -> bytes:
        """Object-structure state key (robust to pixel noise), ignoring animated distractors.

        Object-level hashing collapses irrelevant per-pixel jitter that would otherwise
        explode the state graph on real games; animated-distractor colors are excluded.
        """
        return P.object_state_key(grid, background=self.bg, ignore_colors=self.distractor_colors)

    def _new_level(self, levels: int) -> None:
        self.level = levels
        self.phase = "probe"
        self.mm = None
        self._votes = {}
        self._changed_colors = set()
        self.distractor_colors = set()
        self.bg = None
        self._probe_queue = None
        self._probe_before = None
        self._probe_aid = None
        self.target = None
        self.tried_targets = set()

    # main entry: given the latest observation, return the next action token
    def decide(self, grid: np.ndarray, gstate_terminal: bool, gstate_notplayed: bool,
               levels: int, available: list[int]) -> Action:
        self.vt.update(grid)
        if self.bg is None:
            self.bg = P.detect_background(grid)
        cur_key = self._key(grid)

        # terminal / not-played -> RESET
        if gstate_terminal or gstate_notplayed:
            if gstate_terminal and self.prev_action is not None and self.prev_key is not None and self.gs:
                self.gs.update(self.prev_key, self.prev_action, cur_key, 0.0,
                               self._cands(grid, available), terminal=True)
            self.prev_action = None
            self.expect_reset = True
            if self.gs:
                self.gs.plan = []
            return ("reset",)

        # first real frame (after initial reset) -> establish root
        if self.root_key is None:
            self.root_key = cur_key
            self.gs = GraphStrategy(self.root_key)
            self.gs.wm.observe(self.root_key, self._cands(grid, available))
            self._new_level(levels)
            self.bg = P.detect_background(grid)

        if self.expect_reset:
            self.expect_reset = False
            self.prev_action = None  # don't record across reset

        # record outcome of the previous action
        if self.prev_action is not None and self.prev_key is not None and self.gs is not None:
            reward = float(levels - self.prev_levels)
            self.gs.update(self.prev_key, self.prev_action, cur_key, reward,
                           self._cands(grid, available), terminal=False)
            if self.phase == "probe" and self._probe_before is not None and self._probe_aid is not None:
                trans = MV.infer_all_translations(self._probe_before, grid, self.bg)
                for color, (dr, dc) in trans.items():
                    self._votes.setdefault(color, {})[self._probe_aid] = (dr, dc)
                # any non-background color whose cells changed this step
                for c in set(np.unique(self._probe_before)).union(np.unique(grid)):
                    c = int(c)
                    if c != self.bg and not np.array_equal(self._probe_before == c, grid == c):
                        self._changed_colors.add(c)

        # new level -> relearn
        if levels != self.level:
            self._new_level(levels)

        self.prev_levels = levels
        action = self._choose(grid, cur_key, available)
        self.prev_key = cur_key
        self.prev_action = None if action[0] == "reset" else action
        return action

    def _choose(self, grid, cur_key, available, _depth: int = 0) -> Action:
        if _depth > 3:
            return self._random_action(grid, available)
        simple_avail = [a for a in (1, 2, 3, 4, 5) if a in available]

        # ----- PROBE: learn motion model -----
        if self.phase == "probe":
            if self._probe_queue is None:
                self._probe_queue = list(simple_avail)
            if self._probe_queue:
                aid = self._probe_queue.pop(0)
                self._probe_before = grid
                self._probe_aid = aid
                return ("S", aid)
            # finished probing: the avatar is the object whose motion CORRELATES with the
            # action (most distinct delta vectors); counters/animations move constantly.
            if self._votes:
                def _score(c):
                    deltas = self._votes[c]
                    return (len(set(deltas.values())), len(deltas))
                color = max(self._votes, key=_score)
                self.mm = MV.MotionModel(avatar_color=color, deltas=self._votes[color])
                # Animated distractor = a color that RIGIDLY TRANSLATES with a constant
                # delta regardless of the action (a counter/animation), NOT merely a color
                # whose cells changed (that also flags structural cells the avatar moves
                # over, e.g. maze walls — which would blind us to doors opening).
                self.distractor_colors = {
                    c for c, d in self._votes.items()
                    if c != color and c != self.bg
                    and len(d) >= 2 and len(set(d.values())) == 1
                }
                self.phase = "navigate"
                self.target = None
            else:
                self.phase = "graph"
            return self._choose(grid, cur_key, available, _depth + 1)

        # ----- NAVIGATE avatar to candidate goal objects -----
        if self.phase == "navigate" and self.mm is not None and self.mm.ok:
            if self.target is None:
                t = self._next_target(grid)
                if t is None:
                    self.phase = "graph"
                    return self._choose(grid, cur_key, available, _depth + 1)
                self.target = t
                self.tried_targets.add(t)
                self.nav_steps = 0
                self.nav_stale = 0
                self.nav_last = None
            act = self._nav_step(grid)
            if act is None:
                self.target = None
                return self._choose(grid, cur_key, available, _depth + 1)
            return act

        # ----- GRAPH fallback -----
        if self.gs is not None:
            kind, a = self.gs.decide(cur_key)
            if kind == RESET:
                self.expect_reset = True
                self.gs.plan = []
                return ("reset",)
            if kind == STOP:
                return self._random_action(grid, available)
            return a
        return self._random_action(grid, available)

    def _next_target(self, grid):
        objs = P.connected_components(grid, background=self.bg)
        ac = self.mm.avatar_centroid(grid)
        if ac is None:
            return None
        cands = []
        for o in objs:
            if o.color == self.mm.avatar_color or o.color in self.distractor_colors:
                continue
            r, c = int(round(o.centroid[0])), int(round(o.centroid[1]))
            if (r, c) in self.tried_targets:
                continue
            cands.append((abs(r - ac[0]) + abs(c - ac[1]), (r, c)))
        if not cands:
            return None
        cands.sort()
        return cands[0][1]

    def _nav_step(self, grid):
        ac = self.mm.avatar_centroid(grid)
        if ac is None:
            return None
        cr, cc = ac
        tr, tc = self.target
        if abs(cr - tr) < 1 and abs(cc - tc) < 1:
            return None  # arrived
        if self.nav_steps >= self.nav_step_cap:
            return None
        # detect stalled avatar (didn't move since last nav action)
        if self.nav_last is not None and abs(self.nav_last[0] - cr) < 0.5 and abs(self.nav_last[1] - cc) < 0.5:
            self.nav_stale += 1
            if self.nav_stale >= 2:
                return None
        else:
            self.nav_stale = 0
        cur_d = abs(cr - tr) + abs(cc - tc)
        best_a, best_d = None, None
        for aid, (dr, dc) in self.mm.deltas.items():
            nd = abs(cr + dr - tr) + abs(cc + dc - tc)
            if best_d is None or nd < best_d:
                best_d, best_a = nd, aid
        if best_a is None or best_d >= cur_d:
            return None
        self.nav_last = (cr, cc)
        self.nav_steps += 1
        return ("S", best_a)

    def _random_action(self, grid, available) -> Action:
        cands = self._cands(grid, available)
        if not cands:
            return ("S", available[0]) if available else ("reset",)
        i = int(self.rng.integers(0, len(cands)))
        return cands[i]
