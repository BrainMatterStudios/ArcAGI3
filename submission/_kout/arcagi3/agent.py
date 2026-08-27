"""Agents for ARC-AGI-3.

- GraphStrategy: graph-based exploration/exploitation over a WorldModel (frontier search
  + shortest-path replay + reward exploitation). Reusable decision policy.
- ExplorerAgent: pure graph explorer (baseline).
- HybridAgent: learns a motion model (controllable avatar + per-action displacement) and
  navigates in coordinate space to candidate goal objects; falls back to GraphStrategy
  when no avatar is found or motion progress stalls.

All training-free and game-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from arcengine import GameAction, GameState

from . import movement as MV
from . import perception as P
from .world_model import Action, WorldModel

logger = logging.getLogger("arcagi3.agent")

SIMPLE_IDS = [1, 2, 3, 4, 5, 7]  # RESET(0)/ACTION6(click) handled separately


@dataclass
class PlayResult:
    game_id: str
    levels_completed: int
    win_levels: int
    actions: int
    won: bool
    states_seen: int
    reason: str = ""


def to_game_action(a: Action) -> tuple[GameAction, dict]:
    if a[0] == "S":
        return GameAction.from_id(a[1]), {}
    if a[0] == "C":
        return GameAction.ACTION6, {"x": a[1], "y": a[2]}
    raise ValueError(a)


def candidates_for(grid: np.ndarray, available: list[int], use_clicks: bool,
                   max_click_targets: int, use_undo: bool) -> tuple[Action, ...]:
    cands: list[Action] = []
    for aid in SIMPLE_IDS:
        if aid in available and (aid != 7 or use_undo):
            cands.append(("S", aid))
    if use_clicks and 6 in available:
        for x, y, _prio in P.salient_click_targets(
            grid, max_targets=max_click_targets, coarse_grid_step=8
        ):
            cands.append(("C", int(x), int(y)))
    return tuple(cands)


# --- decisions the strategy can return to the driving loop ---
ACT = "act"
RESET = "reset"
STOP = "stop"


class GraphStrategy:
    """Stateful graph exploration policy over a shared WorldModel."""

    def __init__(self, root_key: bytes, max_stuck_resets: int = 50) -> None:
        self.wm = WorldModel()
        self.root_key = root_key
        self.plan: list[Action] = []
        self._expect: bytes | None = None
        self.stuck_resets = 0
        self.max_stuck_resets = max_stuck_resets

    def decide(self, cur_key: bytes) -> tuple[str, Action | None]:
        node = self.wm.nodes.get(cur_key)
        if node is None:
            return (STOP, None)

        # 1) exploit a known reward-producing action
        r_act = self.wm.reward_action(cur_key)
        if r_act is not None:
            return (ACT, r_act)

        # 2) continue an active plan (replay), aborting on divergence
        if self.plan:
            if self._expect is not None and cur_key != self._expect:
                self.plan = []
            else:
                return (ACT, self.plan.pop(0))

        # 3) untried candidate here
        if node.untried():
            return (ACT, node.untried()[0])

        # 4) navigate to nearest frontier
        path = self.wm.path_to_frontier(cur_key)
        if path is None:
            if cur_key != self.root_key and self.stuck_resets < self.max_stuck_resets:
                self.stuck_resets += 1
                return (RESET, None)
            path = self.wm.path_to_frontier(self.root_key)
            if path is None:
                return (STOP, None)
        if path:
            self.plan = path
            return (ACT, self.plan.pop(0))
        return (STOP, None)

    def update(self, cur_key: bytes, action: Action, next_key: bytes, reward: float,
               candidates: tuple[Action, ...], terminal: bool) -> None:
        self.wm.record(cur_key, action, next_key, reward)
        self.wm.observe(next_key, candidates, terminal=terminal)
        self._expect = next_key if self.plan else None


class ExplorerAgent:
    """Pure graph-based explorer (baseline)."""

    def __init__(self, max_actions: int = 4000, use_clicks: bool = True,
                 max_click_targets: int = 96, use_undo: bool = False, seed: int = 0) -> None:
        self.max_actions = max_actions
        self.use_clicks = use_clicks
        self.max_click_targets = max_click_targets
        self.use_undo = use_undo

    def _cands(self, grid, available):
        return candidates_for(grid, available, self.use_clicks, self.max_click_targets, self.use_undo)

    def play(self, env, game_id: str = "?") -> PlayResult:
        vt = P.VolatilityTracker()
        obs = env.reset()
        grid = P.to_grid(obs.frame)
        vt.update(grid)
        root_key = P.state_hash(grid, vt.mask())
        gs = GraphStrategy(root_key)
        gs.wm.observe(root_key, self._cands(grid, obs.available_actions))
        cur_key = root_key
        actions = 0
        prev_levels = int(obs.levels_completed or 0)
        win_levels = int(obs.win_levels or 0)
        reason = "budget"

        while actions < self.max_actions:
            if obs.state == GameState.WIN:
                reason = "win"
                break
            if obs.state == GameState.GAME_OVER:
                n = gs.wm.nodes.get(cur_key)
                if n:
                    n.terminal = True
                obs = env.reset(); actions += 1
                grid = P.to_grid(obs.frame); vt.update(grid); cur_key = root_key; gs.plan = []
                continue

            kind, action = gs.decide(cur_key)
            if kind == STOP:
                reason = "exhausted"; break
            if kind == RESET:
                obs = env.reset(); actions += 1
                grid = P.to_grid(obs.frame); vt.update(grid); cur_key = root_key; gs.plan = []
                continue

            obs, grid, cur_key, prev_levels = self._step(env, action, gs, vt, cur_key, prev_levels)
            actions += 1

        return PlayResult(game_id, prev_levels, win_levels, actions,
                          obs.state == GameState.WIN, len(gs.wm), reason)

    def _step(self, env, action, gs, vt, cur_key, prev_levels):
        ga, data = to_game_action(action)
        obs = env.step(ga, data=data) if data else env.step(ga)
        ngrid = P.to_grid(obs.frame); vt.update(ngrid)
        nkey = P.state_hash(ngrid, vt.mask())
        nlevels = int(obs.levels_completed or 0)
        terminal = obs.state == GameState.GAME_OVER
        gs.update(cur_key, action, nkey, float(nlevels - prev_levels),
                  self._cands(ngrid, obs.available_actions), terminal)
        return obs, ngrid, nkey, nlevels


class HybridAgent:
    """Motion-first agent: learn the avatar + per-action displacement, navigate to goal
    objects in coordinate space; fall back to graph exploration when stalled."""

    def __init__(self, max_actions: int = 4000, use_clicks: bool = True,
                 max_click_targets: int = 96, nav_step_cap: int = 200, seed: int = 0) -> None:
        self.max_actions = max_actions
        self.use_clicks = use_clicks
        self.max_click_targets = max_click_targets
        self.nav_step_cap = nav_step_cap
        self.rng = np.random.default_rng(seed)

    def _cands(self, grid, available):
        return candidates_for(grid, available, self.use_clicks, self.max_click_targets, False)

    def play(self, env, game_id: str = "?") -> PlayResult:
        vt = P.VolatilityTracker()
        obs = env.reset()
        grid = P.to_grid(obs.frame); vt.update(grid)
        root_key = P.state_hash(grid, vt.mask())
        gs = GraphStrategy(root_key)
        gs.wm.observe(root_key, self._cands(grid, obs.available_actions))
        cur_key = root_key
        actions = 0
        prev_levels = int(obs.levels_completed or 0)
        win_levels = int(obs.win_levels or 0)
        reason = "budget"

        bg = P.detect_background(grid)
        mm: MV.MotionModel | None = None
        level_of_model = -1
        tried_targets: set[tuple[int, int]] = set()
        motion_dead = False  # avatar strategy gave up for this level

        def record(action, obs_new):
            nonlocal cur_key, prev_levels, grid, actions
            ngrid = P.to_grid(obs_new.frame); vt.update(ngrid)
            nkey = P.state_hash(ngrid, vt.mask())
            nlevels = int(obs_new.levels_completed or 0)
            terminal = obs_new.state == GameState.GAME_OVER
            gs.update(cur_key, action, nkey, float(nlevels - prev_levels),
                      self._cands(ngrid, obs_new.available_actions), terminal)
            cur_key, prev_levels, grid = nkey, nlevels, ngrid
            actions += 1

        while actions < self.max_actions:
            if obs.state == GameState.WIN:
                reason = "win"; break
            if obs.state == GameState.GAME_OVER:
                n = gs.wm.nodes.get(cur_key)
                if n:
                    n.terminal = True
                obs = env.reset(); actions += 1
                grid = P.to_grid(obs.frame); vt.update(grid); cur_key = root_key; gs.plan = []
                mm = None; motion_dead = False; tried_targets.clear()
                continue

            # New level -> relearn motion
            if prev_levels != level_of_model:
                mm = None; motion_dead = False; tried_targets.clear()
                level_of_model = prev_levels

            simple_avail = [a for a in (1, 2, 3, 4, 5) if a in obs.available_actions]

            # ---- learn motion model by probing simple actions ----
            if mm is None and simple_avail and not motion_dead:
                mm = self._learn_motion(env, obs, grid, bg, simple_avail, record_fn=record)
                obs = self._last_obs
                if mm is None or not mm.deltas:
                    motion_dead = True
                continue

            # ---- navigate avatar to a candidate goal object ----
            if mm is not None and mm.ok and not motion_dead:
                target = self._next_target(grid, bg, mm, tried_targets)
                if target is None:
                    motion_dead = True
                    continue
                tried_targets.add(target)
                obs = self._navigate(env, mm, target, record_fn=record, start_levels=prev_levels)
                if obs.state == GameState.WIN:
                    reason = "win"; break
                continue

            # ---- graph fallback ----
            kind, action = gs.decide(cur_key)
            if kind == STOP:
                # last resort: if motion existed, reset and let motion retry fresh
                reason = "exhausted"; break
            if kind == RESET:
                obs = env.reset(); actions += 1
                grid = P.to_grid(obs.frame); vt.update(grid); cur_key = root_key; gs.plan = []
                mm = None; motion_dead = False; tried_targets.clear(); level_of_model = prev_levels
                continue
            ga, data = to_game_action(action)
            obs = env.step(ga, data=data) if data else env.step(ga)
            record(action, obs)

        return PlayResult(game_id, prev_levels, win_levels, actions,
                          obs.state == GameState.WIN, len(gs.wm), reason)

    # ----- motion learning -----
    def _learn_motion(self, env, obs, grid, bg, simple_avail, record_fn) -> MV.MotionModel | None:
        """Try each simple action once; detect the avatar (consistently-translating color)."""
        votes: dict[int, dict[int, tuple[int, int]]] = {}  # color -> {action: (dr,dc)}
        cur_grid = grid
        last_obs = obs
        for aid in simple_avail:
            before = cur_grid
            action = ("S", aid)
            ga, _ = to_game_action(action)
            o = env.step(ga)
            record_fn(action, o)
            last_obs = o
            after = P.to_grid(o.frame)
            res = MV.infer_translation(before, after, bg)
            if res is not None:
                color, dr, dc = res
                votes.setdefault(color, {})[aid] = (dr, dc)
            cur_grid = after
            if o.state in (GameState.WIN, GameState.GAME_OVER):
                break
        self._last_obs = last_obs
        if not votes:
            return None
        # avatar = color that moved for the most actions
        avatar_color = max(votes, key=lambda c: len(votes[c]))
        return MV.MotionModel(avatar_color=avatar_color, deltas=votes[avatar_color])

    def _next_target(self, grid, bg, mm: MV.MotionModel, tried) -> tuple[int, int] | None:
        """Pick the nearest untried non-avatar object centroid to navigate to."""
        objs = P.connected_components(grid, background=bg)
        ac = mm.avatar_centroid(grid)
        if ac is None:
            return None
        cands = []
        for o in objs:
            if o.color == mm.avatar_color:
                continue
            r, c = int(round(o.centroid[0])), int(round(o.centroid[1]))
            if (r, c) in tried:
                continue
            d = abs(r - ac[0]) + abs(c - ac[1])
            cands.append((d, (r, c)))
        if not cands:
            return None
        cands.sort()
        return cands[0][1]

    def _navigate(self, env, mm: MV.MotionModel, target, record_fn, start_levels):
        """Greedily drive the avatar toward target using learned deltas. Returns last obs."""
        obs = self._last_obs
        tr, tc = target
        steps = 0
        stale = 0
        while steps < self.nav_step_cap:
            grid = P.to_grid(obs.frame)
            ac = mm.avatar_centroid(grid)
            if ac is None:
                break
            cr, cc = ac
            if abs(cr - tr) < 1 and abs(cc - tc) < 1:
                break  # arrived
            # choose action minimizing post-move distance
            best_a, best_d = None, None
            cur_d = abs(cr - tr) + abs(cc - tc)
            for aid, (dr, dc) in mm.deltas.items():
                nd = abs(cr + dr - tr) + abs(cc + dc - tc)
                if best_d is None or nd < best_d:
                    best_d, best_a = nd, aid
            if best_a is None or best_d >= cur_d:
                break  # no improving move (greedy stuck)
            action = ("S", best_a)
            ga, _ = to_game_action(action)
            before_levels = int(obs.levels_completed or 0)
            o = env.step(ga)
            record_fn(action, o)
            obs = o
            self._last_obs = o
            steps += 1
            if o.state in (GameState.WIN, GameState.GAME_OVER):
                break
            if int(o.levels_completed or 0) > before_levels:
                break  # reward!
            # detect blocked (avatar didn't move) -> stop to avoid spin
            ng = P.to_grid(o.frame)
            nac = mm.avatar_centroid(ng)
            if nac is not None and abs(nac[0] - cr) < 0.5 and abs(nac[1] - cc) < 0.5:
                stale += 1
                if stale >= 2:
                    break
            else:
                stale = 0
        return obs
