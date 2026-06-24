"""WinningExplorer — TransferExplorer + executable mechanic model-search (winning branch, Phase 6).

The bet: discover compact executable transition rules online (a model BEAM), enumerate goal
hypotheses, plan through the model, execute a SHORT prefix, verify, and replan — falling back
SAFELY to TransferExplorer. v13 TransferExplorer (public 0.33) is never modified.

Per-game state machine inside `_model_decide`:
    PROBE   active information-gain directional probes to fit the model beam (bounded budget)
    PLAN    once a model is confident (held-out transition acc >= MIN_CONF), plan to a goal
            hypothesis and execute a CHUNK; a moving goal is chased by re-planning each chunk
    GIVE-UP probe budget spent with no confident plan -> defer to TransferExplorer for this game

Firewall (absolute, regression-tested):
    * enable_model_search=False  -> byte-identical to TransferExplorer
    * no directional actions (click-only games like lp85)  -> never acts; byte-identical
    * model never produces a confident plan -> defers to TransferExplorer
Model search activates ONLY on movement games (directional actions present) — disjoint from
transfer's click-signature domain (Exp 46), which protects the transfer-critical click games.
"""
from __future__ import annotations

import os

from arcagi3.mechanics.model_search import ModelSearch
from arcagi3.mechanics.reward_goal import RewardGoalLearner
from arcagi3.planning.model_beam_planner import MIN_CONF, best_plan
from arcagi3.transfer_explorer import TransferExplorer

CHUNK = 12          # execute at most this many plan actions before re-observing + replanning
MAX_PROBE = 250     # active-probe budget per game before giving up to transfer
MAX_COVERAGE_ROUNDS = 8  # re-cover the reachable space this many times (paint/goal motion grows it)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


class WinningExplorer(TransferExplorer):
    def __init__(self, *args, enable_model_search: bool | None = None, **kwargs) -> None:
        self.enable_model_search = (
            _env_flag("ENABLE_MODEL_SEARCH", True) if enable_model_search is None
            else bool(enable_model_search))
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self._model_fires = 0
        self._fallback_actions = 0
        self._model_plan_aborts = 0
        self._model_plan_starts = 0
        self._reset_model_state()

    def _reset_model_state(self):
        self._ms = ModelSearch()
        self._goal_learner = RewardGoalLearner()
        self._prev_grid = None
        self._prev_action = None
        self._win_frame = None          # last frame before a level-up (for goal induction)
        self._last_emit = None          # (grid, action) of the last model-emitted move (winning action)
        self._plan: list = []
        self._plan_goal = None
        self._tried_goals: set = set()
        self._probe_used = 0
        self._gave_up = False
        self._level = 0
        self._step = 0
        self._visited: set = set()       # agent cells physically occupied (model-based coverage)
        self._coverage_rounds = 0        # re-exploration rounds (paint/goal motion grows the space)

    def on_level_change(self, new_level: int):
        if hasattr(super(), "on_level_change"):
            super().on_level_change(new_level)
        # LEARN the goal from the level-up: contrast the last pre-win frame against ordinary states,
        # then transfer the learned predicate to clear the next level (the unexploited reward signal).
        m = self._ms.best()
        if new_level > self._level and m is not None and self._last_emit is not None:
            wgrid, waction = self._last_emit
            ap = m.agent_pos(wgrid)
            learned = None
            if ap is not None and waction in m.move.deltas:           # robust: contact-cell goal
                dr, dc = m.move.deltas[waction]
                er, ec = ap[0] + dr, ap[1] + dc
                if 0 <= er < wgrid.shape[0] and 0 <= ec < wgrid.shape[1]:
                    learned = self._goal_learner.learn_from_contact(int(wgrid[er, ec]), m.agent_colors, m.bg)
            if learned is None and self._win_frame is not None:        # fallback: contrastive
                learned = self._goal_learner.on_levelup(
                    self._prev_grid if self._prev_grid is not None else self._win_frame,
                    self._win_frame, m.agent_colors, m.bg, m.agent_pos(self._win_frame))
            if learned is not None:
                self._ms.learned_goal = learned
        # a level-up resets the puzzle — keep the fitted model + learned goal, reset plan bookkeeping
        self._plan = []
        self._plan_goal = None
        self._tried_goals = set()
        self._prev_grid = None
        self._prev_action = None

    # ------------------------------------------------------------------ model loop
    def _goal_key(self, goal):
        return (goal.kind, getattr(goal, "color", getattr(goal, "axis", None)))

    def _model_decide(self, grid, available):
        """Return a ('S', action) token from the model loop, or None to defer to TransferExplorer.
        Always observes the latest transition so the model keeps improving even while deferring."""
        if not self.enable_model_search:
            return None
        if self._prev_grid is not None:
            self._ms.observe(self._prev_grid, self._prev_action, grid)
        self._prev_grid = grid
        self._prev_action = None

        simple = [a for a in available if a in (1, 2, 3, 4)]
        m = self._ms.best()
        if m is not None:
            ap = m.agent_pos(grid)
            self._goal_learner.note(grid, ap)                  # ordinary-state baseline for contrast
            if ap is not None:
                self._visited.add(ap)                          # mark the reachable cell visited
        if not simple or self._gave_up:
            return None  # click-only / given up -> transfer's domain (firewall holds)

        # EXECUTE: emit the next action of the current chunk
        if self._plan:
            a = self._plan.pop(0)
            self._prev_action = a; self._last_emit = (grid, a)
            return ("S", a)

        # PLAN: if a model is confident, (re)plan to a goal hypothesis and start a chunk
        self._ms.fit()
        if self._ms.confidence() >= MIN_CONF:
            bp = best_plan(self._ms, grid)
            if bp is not None and self._goal_key(bp.goal) not in self._tried_goals:
                self._plan = list(bp.plan[:CHUNK])
                self._plan_goal = bp.goal
                self._model_plan_starts += 1
                if self._plan:
                    a = self._plan.pop(0)
                    self._prev_action = a; self._last_emit = (grid, a)
                    return ("S", a)
            # a confident model but no fresh/feasible goal: blacklist the satisfied one
            if bp is not None:
                self._tried_goals.add(self._goal_key(bp.goal))
            # FRONTIER: systematic model-based coverage — visit the nearest unvisited reachable cell.
            # This covers the reachable state space (paint opens paths) to STUMBLE the first win,
            # which the reward-goal learner then captures and transfers (the requested swing).
            fplan, _target = m.frontier_plan(grid, self._visited)
            if fplan:
                self._plan = list(fplan[:CHUNK])
                self._model_plan_starts += 1
                a = self._plan.pop(0)
                self._prev_action = a; self._last_emit = (grid, a)
                return ("S", a)
            # reachable space covered with no win: re-explore (paint/goal motion may have grown it)
            if self._coverage_rounds < MAX_COVERAGE_ROUNDS:
                self._coverage_rounds += 1
                self._visited = set()
                self._tried_goals = set()
                return ("S", simple[self._coverage_rounds % len(simple)])  # nudge, restart coverage

        # PROBE: active information-gain directional probe (bounded), before a model is confident
        if self._ms.confidence() < MIN_CONF and self._probe_used < MAX_PROBE:
            a = self._ms.probe_action(grid, available)
            if a is not None:
                self._probe_used += 1
                self._prev_action = a; self._last_emit = (grid, a)
                return ("S", a)

        # confident model + reachable space fully covered N rounds (or never confident) -> defer
        self._gave_up = (self._ms.confidence() >= MIN_CONF and self._coverage_rounds >= MAX_COVERAGE_ROUNDS) \
            or (self._ms.confidence() < MIN_CONF and self._probe_used >= MAX_PROBE)
        return None

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if levels != self._level:
            self.on_level_change(levels)      # learns from the win using the OLD self._level
            self._level = levels
        if self.enable_model_search and not gstate_terminal and not gstate_notplayed:
            token = self._model_decide(grid, available)
            self._win_frame = grid            # latest frame = pre-win frame for the next level-up
            if token is not None:
                self._model_fires += 1
                return token
            self._fallback_actions += 1
            self._prev_grid = grid            # keep observing transfer's frames on deferral
            self._prev_action = None
            return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._fallback_actions += 1
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
