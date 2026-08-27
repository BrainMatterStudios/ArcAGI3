"""Phase A GraphRanker — OnlineLearningExplorer composes (never modifies) SalienceExplorer.

The explorer's object-graph, BFS-to-frontier, suspicion filter, HUD mask, exploit-reward, and
reset logic remain the sole execution backbone and memory. An online per-game CNN
(online_model.OnlineActionEffectModel) does exactly ONE thing: when the explorer's pick is a
provable equal-tier RANDOM tie-break among >1 untried candidates, the model re-orders that
*already-sanctioned* set by predicted frame-change probability and substitutes its top pick.

It never picks a tier, never overrides reward_action / a plan replay / a reset / BFS, never
invents a candidate, never plans over predictions. On ANY model error or when the model is not
usable (no torch / unusable GPU / untrained / not confident) it returns the base token verbatim
-> the agent degrades to the pure SalienceExplorer (the banked 0.33 floor). Seam mirrors
wm_policy.py (base.decide first; write back base.prev_action on override).
"""
from __future__ import annotations

import numpy as np

from .salience_explorer import SalienceExplorer
from .online_model import OnlineActionEffectModel


class OnlineLearningExplorer:
    def __init__(self, seed: int = 0, trust_threshold: int = 3, border_mask: int = 2,
                 conf_threshold: float = 0.8, breaker_limit: int = 12,
                 novelty_limit: int = 15, allow_cpu: bool = False, **model_kw) -> None:
        self.base = SalienceExplorer(seed=seed, trust_threshold=trust_threshold,
                                     border_mask=border_mask)
        self.model = OnlineActionEffectModel(seed=seed, conf_threshold=conf_threshold,
                                             allow_cpu=allow_cpu, **model_kw)
        self.breaker_limit = breaker_limit
        self.novelty_limit = novelty_limit
        self._ov_stall = 0       # consecutive overrides that discovered no new state
        self._last_grid = None
        self._last_action = None       # action actually executed into the last frame
        self._last_key = None          # base key the last action was taken from
        self._last_was_override = False
        self._level = 0
        self._level_disabled = False
        self._no_change_overrides = 0
        # telemetry (A.1 redundancy proof / A.4 analysis)
        self.n_overrides = 0
        self.n_tiebreaks = 0

    # duck-type the runner's states_seen (pol.gs.wm)
    @property
    def gs(self):
        return self.base.gs

    def __len__(self):
        return len(self.base)

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        try:
            grid = np.asarray(grid, dtype=np.int8)  # match engine frames; defends wrong dtype
            return self._decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        except Exception:
            # absolute fail-safe (AC-4): never raise; act safely.
            if gstate_terminal or gstate_notplayed:
                return ("reset",)
            for a in (available or [1, 2, 3, 4, 5]):
                if a != 6:
                    return ("S", int(a))
            return ("S", 1)

    def _decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        # snapshot the cause-state BEFORE base mutates it
        key_prev = self.base.prev_key
        action_prev = self._last_action
        plan_empty_before = not self.base.plan
        size_before = len(self.base.nodes)

        base_token = self.base.decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        new_state = len(self.base.nodes) > size_before  # did THIS frame add a graph node?

        # terminal/notplayed/reset: passthrough; drop cross-episode training link
        if gstate_terminal or gstate_notplayed or base_token[0] == "reset":
            self._last_grid = None
            self._last_action = None
            self._last_key = None
            self._last_was_override = False
            return base_token

        # per-level reset (mirror: model + buffer reset between levels)
        if levels != self._level:
            self._level = levels
            self._level_disabled = False
            self._no_change_overrides = 0
            self._ov_stall = 0
            try:
                self.model.reset_level()
            except Exception:
                pass

        cur_key = self.base.prev_key  # base just set this to key(grid)

        # train on the PREVIOUS action's observed outcome (changed = masked key changed)
        if (action_prev is not None and key_prev is not None and self._last_grid is not None):
            changed = cur_key != key_prev
            if self._last_was_override and not changed:
                self._no_change_overrides += 1
                if self._no_change_overrides >= self.breaker_limit:
                    self._level_disabled = True
            elif changed:
                self._no_change_overrides = 0
            # novelty-stall breaker: overrides that change the frame but discover NO new state
            # (cycling in a tiny loop — the tu93 9->2 collapse) -> disable the model this level.
            if self._last_was_override:
                if new_state:
                    self._ov_stall = 0
                else:
                    self._ov_stall += 1
                    if self._ov_stall >= self.novelty_limit:
                        self._level_disabled = True
            try:
                self.model.observe(self._last_grid, action_prev, bool(changed))
            except Exception:
                pass

        out = base_token
        is_override = False
        if not self._level_disabled and plan_empty_before and not self.base.plan:
            choices = self._tiebreak_choices(cur_key, base_token)
            if choices is not None:
                self.n_tiebreaks += 1
                try:
                    best = self.model.best(grid, choices)
                except Exception:
                    best = None
                if best is not None and best != base_token:
                    out = best
                    is_override = True
                    self.n_overrides += 1
                    self.base.prev_action = best  # write-back (the seam crux)

        self._last_grid = np.asarray(grid, dtype=np.int8).copy()
        self._last_action = out
        self._last_key = cur_key
        self._last_was_override = is_override
        return out

    def _tiebreak_choices(self, cur_key, base_token):
        """Return the equal-lowest-tier untried candidate set IFF base_token was a random
        tie-break pick among >1 of them (step-3), else None. No reconstruction of _choose's
        branching — just verifies the conditions on the returned token + node state."""
        node = self.base.nodes.get(cur_key)
        if node is None or node.reward_action() is not None:
            return None
        g = self.base.active_group
        local = node.untried_le(g)
        if not local:
            return None
        mp = min(node.tier.get(a, 0) for a in local)
        choices = [a for a in local if node.tier.get(a, 0) == mp]
        if len(choices) < 2 or base_token not in choices:
            return None
        return choices
