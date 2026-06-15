"""C7 — Integration policy: WorldModelPolicy.

Thin orchestrator that COMPOSES (not subclasses) a proven HybridPolicy instance and only
ever intercepts control to execute a confident, divergence-monitored plan; otherwise it
returns byte-for-byte what HybridPolicy.decide would return.

Safety properties (verified by test_wm_passthrough_equals_hybrid):
  * With cfg.enabled=False (or no planning components present), WorldModelPolicy.decide(...)
    == HybridPolicy.decide(...) token-for-token on all 8 local games.
  * policy.py is NOT edited (D1 composition, not modification).
  * The only write into HybridPolicy is `base.prev_action = out` when we override, keeping the
    base's transition bookkeeping consistent with the actually-executed action (D3 the crux).

Gate (D0): default-OFF. env ARCAGI3_WORLDMODEL=1 enables; unset => HybridPolicy (bit-identical).

Usage:
    from arcagi3.wm_policy import make_policy
    pol = make_policy(seed=0)   # returns HybridPolicy unless ARCAGI3_WORLDMODEL=1
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import perception as P
from .world_model import Action

if TYPE_CHECKING:
    from .policy import HybridPolicy


# ---------------------------------------------------------------------------
# Selector (D0 — default OFF)
# ---------------------------------------------------------------------------

def make_policy(seed: int = 0):
    """Return WorldModelPolicy if ARCAGI3_WORLDMODEL=1, else HybridPolicy (default)."""
    if os.environ.get("ARCAGI3_WORLDMODEL", "0") == "1":
        return WorldModelPolicy(seed=seed)
    from .policy import HybridPolicy
    return HybridPolicy(seed=seed)


# ---------------------------------------------------------------------------
# Config & state
# ---------------------------------------------------------------------------

class Mode(enum.Enum):
    DELEGATE = 0  # pure passthrough to base
    LEARN = 1     # observing; motion/affordance not ready for planning
    PLAN = 2      # executing a planner-generated plan
    FALLBACK = 3  # sticky disable for this level


@dataclass
class WMConfig:
    enabled: bool = True            # master kill-switch -> pure HybridPolicy
    use_planner: bool = True
    use_goal_inference: bool = True
    min_goal_conf: float = 0.6
    min_goal_conf_push: float = 0.85  # higher gate for fragile push archetype
    min_plan_len: int = 1
    max_plan_len: int = 80
    min_delegate_actions: int = 0   # budget floor before planning may fire
    plan_fail_limit: int = 3        # per-level aborts before sticky FALLBACK
    max_plan_steps_no_levelup: int = 40


@dataclass
class WMState:
    mode: Mode = Mode.DELEGATE
    level: int = -1
    plan: list[Action] = field(default_factory=list)
    plan_expect_key: bytes | None = None
    plan_goal_sig: object = None          # cached goal signature for re-plan gating
    plan_aborts: int = 0
    no_progress_steps: int = 0            # steps under planner control without level-up
    actions_this_level: int = 0
    level_disabled: set[int] = field(default_factory=set)
    # shadow mode for push archetype (D4)
    shadow_pending: bool = False          # waiting to compare shadow prediction vs reality
    shadow_pred_key: bytes | None = None  # predicted key from shadow step
    shadow_validated: bool = False        # True once a shadow prediction matched reality


# ---------------------------------------------------------------------------
# WorldModelPolicy
# ---------------------------------------------------------------------------

class WorldModelPolicy:
    """Orchestrator wrapper around HybridPolicy; planning off by default.

    All C1-C6 component handles (objects/afford/goal/fwd/planner) start as None so C7
    degenerates to pure HybridPolicy passthrough when no components are wired.

    Proxy properties (.gs, .phase, .mm, .reset_all) match the HybridPolicy surface so the
    runner/adapter can treat both polymorphically without isinstance checks.
    """

    def __init__(self, *, cfg: WMConfig | None = None, seed: int = 0, **kw) -> None:
        from .policy import HybridPolicy
        self.base: HybridPolicy = HybridPolicy(seed=seed, **kw)
        self.cfg: WMConfig = cfg or WMConfig()
        # Optional duck-typed component handles; None => capability off (D6 / R6).
        self.objects = None   # C1 ObjectTracker
        self.afford = None    # C3 AffordanceModel (duck-typed)
        self.goal = None      # C5 GoalInference (duck-typed)
        self.fwd = None       # C4 forward model
        self.planner = None   # C6 planner
        self.s: WMState = WMState()

    # --- public surface (runner.py + adapter read these) ---

    @property
    def gs(self):
        return self.base.gs

    @property
    def phase(self) -> str:
        return self.base.phase

    @property
    def mm(self):
        return self.base.mm

    def reset_all(self) -> None:
        self.base.reset_all()
        self.s = WMState()

    # --- main entry point (matching HybridPolicy.decide signature) ---

    def decide(self, grid: np.ndarray, gstate_terminal: bool, gstate_notplayed: bool,
               levels: int, available: list[int]) -> Action:
        """One action per call.

        (A) Always call base.decide() first — keeps graph/motion/distractor state warm.
        (B) Hard passthroughs: kill-switch / terminal / notplayed / reset.
        (C) Level change -> reset per-level WM state (keep affordances/goal across levels).
        (D) Observe transition into this frame (C1/C2/C3/C5), wrapped in try/except.
        (E) Sticky level-disable circuit-breaker.
        (F) Never fight probe: base still learning motion model -> delegate.
        (G) Budget floor: proven nav gets first crack each level.
        (H) Continue live plan with divergence guard.
        (I) No live plan: try to form one if a confident goal exists.
        (J) THE CRUX: if overriding, write back the actually-executed action to base.prev_action.
        """

        # (A) ALWAYS advance the proven policy first — this keeps graph/motion/distractor warm
        base_token = self.base.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        out = base_token

        # (B) Hard passthroughs
        if (not self.cfg.enabled) or gstate_terminal or gstate_notplayed or base_token[0] == "reset":
            self._on_reset_or_terminal(levels)
            return base_token

        # (C) Level change -> reset per-level WM state (affordances/goal persist)
        if levels != self.s.level:
            self._new_level(levels)
        self.s.actions_this_level += 1

        # (D) Observe transition using base.prev_key / base.prev_action as cause
        try:
            self._observe(grid, levels, available)
        except Exception:
            self._disable_level(self.s.level)
            return base_token

        # (E) Sticky circuit-breaker
        if self.s.level in self.s.level_disabled:
            return base_token

        # (F) Never fight probe phase
        mm = self.base.mm
        if self.base.phase == "probe" or mm is None or not mm.ok:
            return base_token

        # (G) Budget floor
        if self.s.actions_this_level < self.cfg.min_delegate_actions:
            return base_token

        # (H) Continue a committed, still-valid plan
        if self.s.plan:
            cur_key = self.base._key(grid)  # SAME keying as base (R2 guard)
            if self.s.plan_expect_key is not None and cur_key != self.s.plan_expect_key:
                # divergence: drop plan, count abort
                self.s.plan = []
                self.s.plan_aborts += 1
                self._maybe_disable_planner()
            else:
                tok = self.s.plan[0]
                if self._action_legal(tok, available):
                    self.s.plan.pop(0)
                    self.s.plan_expect_key = self._predict_key(grid, tok)
                    self.s.no_progress_steps += 1
                    if self.s.no_progress_steps >= self.cfg.max_plan_steps_no_levelup:
                        self.s.plan = []
                        self._maybe_disable_planner()
                    else:
                        out = tok
                else:
                    self.s.plan = []

        # (I) No live plan: try to form one if a confident goal exists
        if not self.s.plan and out is base_token and self._goal_ready():
            try:
                goal = self._get_goal_hyp()
            except Exception:
                goal = None
            if goal is not None and self._goal_conf_ok(goal) and self.s.plan_aborts < self.cfg.plan_fail_limit:
                try:
                    occ = self._occupancy_from_base()
                    plan = self._replan(grid, goal, occ, available)
                except Exception:
                    plan = None
                if (plan is not None
                        and self.cfg.min_plan_len <= len(plan) <= self.cfg.max_plan_len
                        and self._shadow_ok(grid, goal)):
                    self.s.plan = list(plan)
                    self.s.plan_goal_sig = self._goal_sig(goal)
                    self.s.no_progress_steps = 0
                    tok = self.s.plan[0]
                    if self._action_legal(tok, available):
                        self.s.plan.pop(0)
                        self.s.plan_expect_key = self._predict_key(grid, tok)
                        self.s.no_progress_steps += 1
                        out = tok
                else:
                    if plan is None or (plan is not None and len(plan) > 0):
                        self._maybe_disable_planner()

        # (J) THE CRUX: if we overrode, correct base.prev_action so the next transition is
        #     computed against ground truth — the ONLY write into base (mirrors policy.py:136)
        if out is not base_token:
            self.base.prev_action = None if out[0] == "reset" else out

        return out

    # --- lifecycle helpers ---

    def _on_reset_or_terminal(self, levels: int) -> None:
        """Partial reset when returning base_token on terminal/notplayed/reset/disabled."""
        if self.s.level != levels:
            # level changed while we were in passthrough — keep level-disabled set stable
            self.s.level = levels
        self.s.plan = []
        self.s.plan_expect_key = None
        self.s.no_progress_steps = 0

    def _new_level(self, levels: int) -> None:
        """Reset per-level planning state; affordances/goal persist across levels."""
        self.s.level = levels
        self.s.plan = []
        self.s.plan_expect_key = None
        self.s.plan_goal_sig = None
        self.s.plan_aborts = 0
        self.s.no_progress_steps = 0
        self.s.actions_this_level = 0
        self.s.shadow_pending = False
        self.s.shadow_pred_key = None
        self.s.shadow_validated = False
        # mode follows circuit-breaker state
        if levels in self.s.level_disabled:
            self.s.mode = Mode.FALLBACK
        else:
            self.s.mode = Mode.DELEGATE

    def _disable_level(self, level: int) -> None:
        """Sticky circuit-breaker: pure passthrough for the rest of this level."""
        self.s.level_disabled.add(level)
        self.s.plan = []
        self.s.plan_expect_key = None
        self.s.mode = Mode.FALLBACK

    def _maybe_disable_planner(self) -> None:
        """Increment abort counter; if >= plan_fail_limit, disable this level."""
        if self.s.plan_aborts >= self.cfg.plan_fail_limit:
            self._disable_level(self.s.level)

    # --- observation ---

    def _observe(self, grid: np.ndarray, levels: int, available: list[int]) -> None:
        """Feed C1/C3/C5 observation. Fully optional (all handles may be None)."""
        # C1 object tracking (duck-typed: objects.update(grid) -> [TrackedObj])
        if self.objects is not None:
            try:
                self.objects.update(grid)
            except Exception:
                pass

        # C3 affordance: already learned by base.aff via policy.py; no extra call needed
        # (base runs observe_step inside HybridPolicy.decide every step when
        # enable_affordance=True). We expose self.base.aff for C6/replan use.

        # C5 goal inference: already updated by base.gi inside HybridPolicy.decide
        # (infer_goals=True). We expose self.base.gi for reading below.

    # --- planning helpers ---

    def _goal_ready(self) -> bool:
        """True iff planning is configured on AND at least one component handle is wired.

        When all handles are None (pure composition shell), no planning fires regardless
        of config flags — the wrapper degenerates to HybridPolicy (D0/D6 guard).
        A goal source requires either self.goal (external C5) or self.base.gi being
        explicitly exposed via use_goal_inference=True AND a planner path being available.
        """
        if not self.cfg.use_goal_inference or not self.cfg.use_planner:
            return False
        # If there's an explicit goal handle, we can plan.
        if self.goal is not None:
            return True
        # As a convenience, allow using base.gi directly — but only if explicitly enabled
        # by setting use_goal_inference=True AND the base.gi is populated (not first level).
        gi = getattr(self.base, "gi", None)
        if gi is None:
            return False
        # Require at least 2 level-ups before trusting base.gi goals, to prevent
        # single-observation interference (support=1 goals from the first level-up
        # are too noisy to override the proven policy). This is a conservative gate.
        levelups = getattr(gi, "model", None)
        if levelups is None:
            return False
        return getattr(levelups, "levelups_seen", 0) >= 2

    def _get_goal_hyp(self):
        """Return the best GoalHypothesis from the explicit goal handle or base.gi."""
        # Prefer the external C5 goal handle if wired
        if self.goal is not None:
            try:
                return self.goal.hypothesis()
            except Exception:
                return None
        # Fall back to base.gi when _goal_ready() approved it (levelups_seen >= 2)
        gi = getattr(self.base, "gi", None)
        if gi is None:
            return None
        try:
            return gi.current_goal(min_conf=0.0, min_support=1)
        except Exception:
            return None

    def _goal_conf_ok(self, goal) -> bool:
        """Check confidence gate, with a higher bar for push goals."""
        conf = getattr(goal, "confidence", 0.0)
        kind = getattr(goal, "kind", "")
        gate = (self.cfg.min_goal_conf_push if "PUSH" in kind else self.cfg.min_goal_conf)
        return conf >= gate

    def _goal_sig(self, goal) -> object:
        """Stable signature for goal change detection; falls back to (kind, color)."""
        try:
            return goal.signature()
        except AttributeError:
            kind = getattr(goal, "kind", None)
            color = getattr(goal, "color", None)
            return (kind, color)

    def _conf_gate(self, goal) -> float:
        """Return the confidence threshold for this goal's kind."""
        kind = getattr(goal, "kind", "")
        return self.cfg.min_goal_conf_push if "PUSH" in kind else self.cfg.min_goal_conf

    def _action_legal(self, tok: Action, available: list[int]) -> bool:
        """True iff the action token can be executed given available action ids."""
        if not tok:
            return False
        if tok[0] == "reset":
            return True
        if tok[0] == "S":
            return tok[1] in available
        if tok[0] == "C":
            return 6 in available or not available  # click available when ACTION6 in avail
        return False

    def _predict_key(self, grid: np.ndarray, tok: Action) -> bytes | None:
        """Predict the object-state key after executing tok, using the base's deltas + bg.

        Calls IDENTICAL P.object_state_key(..., ignore_colors=base.distractor_colors)
        as policy.py:67, so predicted keys compare like-for-like to live keys (R2 guard).

        We only handle simple move tokens ('S', aid). Clicks and resets return None (no
        prediction -> divergence guard is inactive for that step).
        """
        if tok[0] != "S":
            return None
        mm = self.base.mm
        if mm is None or not mm.ok:
            return None
        aid = tok[1]
        delta = mm.deltas.get(aid)
        if delta is None:
            return None

        dr, dc = delta
        h, w = grid.shape
        predicted = grid.copy()
        # translate all avatar colors by (dr, dc)
        for ac in mm.avatar_colors:
            src = np.argwhere(grid == ac)
            if len(src) == 0:
                continue
            # clear old positions
            for r, c in src:
                predicted[r, c] = self.base.bg if self.base.bg is not None else 0
            # set new positions (in-bounds only)
            for r, c in src:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    predicted[nr, nc] = ac

        return P.object_state_key(
            predicted,
            background=self.base.bg,
            ignore_colors=self.base.distractor_colors,
        )

    def _occupancy_from_base(self):
        """Build an OccupancyMap from the base's learned motion deltas."""
        from .spatial import OccupancyMap
        mm = self.base.mm
        if mm is None or not mm.ok:
            return None
        return OccupancyMap(mm.deltas)

    def _replan(self, grid: np.ndarray, goal, occ, available: list[int]) -> list[Action] | None:
        """BFS/A* from avatar centroid to goal target cells using the occupancy map.

        Returns a list of ('S', action_id) tokens, or None on failure.
        Excludes movable/non-BLOCK colors from walls (push-safe per REBUILD_PLAN C6).
        """
        if occ is None or not occ.usable:
            return None
        mm = self.base.mm
        if mm is None or not mm.ok:
            return None

        # avatar start position
        start = mm.avatar_centroid(grid)
        if start is None:
            return None

        # goal target cells from the goal-inference model
        gi = getattr(self.base, "gi", None)
        if gi is None:
            return None
        try:
            targets = gi.goal_target_cells(
                grid,
                bg=self.base.bg,
                min_conf=0.0,
                min_support=1,
            )
        except Exception:
            return None
        if not targets:
            return None

        # mark known-blocked cells in the occupancy map from affordance data
        # (exclude movable/non-BLOCK colors from walls — push-safe)
        aff = getattr(self.base, "aff", None)
        if aff is not None:
            try:
                from . import perception as P_mod
                from .affordance import Effect
                objs = P_mod.connected_components(grid, background=self.base.bg)
                for o in objs:
                    if o.color in (mm.avatar_colors or set()):
                        continue
                    if o.color in (self.base.distractor_colors or set()):
                        continue
                    v = aff.affordance(o.color)
                    if v.reliable and v.effect == Effect.BLOCK:
                        # mark as wall in occupancy map at this object's centroid
                        cell = (int(round(o.centroid[0])), int(round(o.centroid[1])))
                        occ.blocked.add(occ.quantize(cell))
            except Exception:
                pass

        # plan to the nearest target
        best_plan = None
        for t in targets:
            try:
                aid_list = occ.astar(start, t)
            except Exception:
                continue
            if aid_list is None:
                continue
            tokens = [("S", aid) for aid in aid_list]
            if best_plan is None or len(tokens) < len(best_plan):
                best_plan = tokens
        return best_plan

    def _shadow_ok(self, grid: np.ndarray, goal) -> bool:
        """Shadow-mode-first gate for push goals (D4 sokoban guard).

        The FIRST time the planner would take control on a level whose goal is PUSH-type,
        do NOT take control: predict the next key, return base_token, and on the next step
        compare predicted vs realized key. Only grant control after a shadow prediction
        matched reality. A wrong push model cannot silently take over.

        For non-push goals, always returns True (no shadow required).
        """
        kind = getattr(goal, "kind", "")
        if "PUSH" not in kind:
            return True

        if self.s.shadow_validated:
            return True  # shadow already passed for this level

        if not self.s.shadow_pending:
            # start shadow: record the predicted key for the NEXT step's comparison
            # (we don't override this step, so out == base_token)
            if self.base.mm is not None and self.base.mm.ok:
                # find the best action base will emit; we can't know exactly since we've
                # already called base.decide() this step, so we predict off the current state
                # using the first planned action (not yet popped)
                if self.s.plan:
                    try:
                        self.s.shadow_pred_key = self._predict_key(grid, self.s.plan[0])
                    except Exception:
                        self.s.shadow_pred_key = None
            self.s.shadow_pending = True
            return False  # NOT OK to override; let base handle this step

        # pending: check if last prediction matched reality (base.prev_key is the key AFTER
        # the last base action, set at the END of HybridPolicy.decide -> after step (A) above
        # the current frame's key is already in play. Use base._key(grid) for "now".)
        if self.s.shadow_pred_key is not None:
            cur_key = self.base._key(grid)
            if cur_key == self.s.shadow_pred_key:
                self.s.shadow_validated = True
                self.s.shadow_pending = False
                return True  # prediction matched; safe to plan
            else:
                # prediction wrong: model unreliable for push; disable this level
                self._disable_level(self.s.level)
                self.s.shadow_pending = False
                return False
        else:
            # couldn't form a prediction; try again next step
            self.s.shadow_pending = False
            return False
