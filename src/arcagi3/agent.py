"""ExplorerAgent: a general, training-free interactive agent.

Strategy (graph-based exploration + exploitation):
  1. Perceive the frame -> object-centric state key (volatile/counter cells masked).
  2. If the current state has a known action that produced reward (level-up), take it.
  3. Else if the current state has untried candidate actions, try the next one
     (simple actions first; clicks proposed object-centrically by salience).
  4. Else navigate to the nearest frontier state (known state with untried actions)
     by replaying the shortest known action path; abort/replan on divergence.
  5. On GAME_OVER, RESET (escape) and avoid the offending transition. When nothing is
     reachable, RESET to root; when even root is exhausted, stop.

Generalises across arrow / click / mixed games with no per-game knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from arcengine import GameAction, GameState

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


class ExplorerAgent:
    def __init__(
        self,
        max_actions: int = 4000,
        use_clicks: bool = True,
        max_click_targets: int = 48,
        use_undo: bool = False,
        seed: int = 0,
    ) -> None:
        self.max_actions = max_actions
        self.use_clicks = use_clicks
        self.max_click_targets = max_click_targets
        self.use_undo = use_undo
        self.rng = np.random.default_rng(seed)

    # ----- action token <-> GameAction -----
    @staticmethod
    def _to_game_action(a: Action) -> tuple[GameAction, dict]:
        if a[0] == "S":
            return GameAction.from_id(a[1]), {}
        if a[0] == "C":
            return GameAction.ACTION6, {"x": a[1], "y": a[2]}
        raise ValueError(a)

    def _candidates(self, grid: np.ndarray, available: list[int]) -> tuple[Action, ...]:
        cands: list[Action] = []
        for aid in SIMPLE_IDS:
            if aid in available and (aid != 7 or self.use_undo):
                cands.append(("S", aid))
        if self.use_clicks and 6 in available:
            for x, y, _prio in P.salient_click_targets(grid, max_targets=self.max_click_targets):
                cands.append(("C", int(x), int(y)))
        return tuple(cands)

    # ----- env helpers -----
    def _key(self, grid: np.ndarray, mask) -> bytes:
        return P.state_hash(grid, mask)

    def play(self, env, game_id: str = "?") -> PlayResult:
        wm = WorldModel()
        vt = P.VolatilityTracker()

        obs = env.reset()
        grid = P.to_grid(obs.frame)
        vt.update(grid)
        root_key = self._key(grid, vt.mask())
        cur_key = root_key
        wm.observe(cur_key, self._candidates(grid, obs.available_actions))

        actions = 0
        prev_levels = int(obs.levels_completed or 0)
        win_levels = int(obs.win_levels or 0)
        plan: list[Action] = []
        plan_expect: bytes | None = None
        stuck_resets = 0
        reason = "budget"

        while actions < self.max_actions:
            state = obs.state
            if state == GameState.WIN:
                reason = "win"
                break

            if state == GameState.GAME_OVER:
                # mark terminal, must RESET
                node = wm.nodes.get(cur_key)
                if node:
                    node.terminal = True
                obs = env.reset()
                actions += 1
                grid = P.to_grid(obs.frame)
                vt.update(grid)
                cur_key = root_key  # reset returns to root
                plan = []
                continue

            node = wm.nodes.get(cur_key)
            if node is None:
                node = wm.observe(cur_key, self._candidates(grid, obs.available_actions))

            # 1) exploit known reward action
            chosen: Action | None = None
            r_act = wm.reward_action(cur_key)
            if r_act is not None:
                chosen = r_act
            # 2) follow active plan
            elif plan:
                # verify we are where the plan expects
                if plan_expect is not None and cur_key != plan_expect:
                    plan = []  # divergence -> replan
                else:
                    chosen = plan.pop(0)
            # 3) untried action at current state
            if chosen is None and node.untried():
                chosen = node.untried()[0]
            # 4) plan toward nearest frontier
            if chosen is None:
                path = wm.path_to_frontier(cur_key)
                if path is None:
                    # nothing reachable from here; try from root via reset
                    if cur_key != root_key and stuck_resets < 50:
                        stuck_resets += 1
                        obs = env.reset()
                        actions += 1
                        grid = P.to_grid(obs.frame)
                        vt.update(grid)
                        cur_key = root_key
                        plan = []
                        continue
                    path = wm.path_to_frontier(root_key)
                    if path is None:
                        reason = "exhausted"
                        break
                if path:
                    plan = path
                    chosen = plan.pop(0)
                else:
                    # frontier is current node but untried() was empty -> safety
                    reason = "exhausted"
                    break

            # execute chosen action
            ga, data = self._to_game_action(chosen)
            obs = env.step(ga, data=data) if data else env.step(ga)
            actions += 1
            ngrid = P.to_grid(obs.frame)
            vt.update(ngrid)
            mask = vt.mask()
            nkey = self._key(ngrid, mask)
            nlevels = int(obs.levels_completed or 0)
            reward = float(nlevels - prev_levels)

            wm.record(cur_key, chosen, nkey, reward)
            terminal = obs.state == GameState.GAME_OVER
            wm.observe(nkey, self._candidates(ngrid, obs.available_actions), terminal=terminal)

            # set up plan-expectation for next iteration
            plan_expect = nkey if plan else None

            prev_levels = nlevels
            cur_key = nkey
            grid = ngrid

        return PlayResult(
            game_id=game_id,
            levels_completed=prev_levels,
            win_levels=win_levels,
            actions=actions,
            won=(obs.state == GameState.WIN),
            states_seen=len(wm),
            reason=reason,
        )
