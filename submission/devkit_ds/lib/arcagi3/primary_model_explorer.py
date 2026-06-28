"""Phase A' — PrimaryModelExplorer: the online action-effect model as the PRIMARY frontier
action selector (StochasticGoose-faithful), NOT a tie-break re-ranker (that was GraphRanker,
killed at the A.4 gate for corrupting tu93).

Subclasses SalienceExplorer (does NOT modify it) and overrides ONLY:
  - __init__: attach an OnlineActionEffectModel.
  - decide: stash the live grid (so _choose can featurize) + drive online training on each
    observed transition outcome (masked-key change), via the _record hook.
  - _choose: when the model is usable, at the frontier pick the untried candidate (across ALL
    tiers) with the highest predicted frame-change prob; fall back to the base tier order when
    the model is not confident. Everything else (exploit-reward, plan replay, BFS-to-frontier,
    reset/bounce) is inherited unchanged.

Firewall: model is GPU-only by default (online_model capability gate) -> if no usable CUDA the
model is unusable and EVERY override path falls through to super()._choose => byte-identical to
SalienceExplorer (the banked 0.33). A per-level novelty-stall guard reverts to the base selector
if model-driven selection stops discovering new states. Nothing here raises to the caller.
"""
from __future__ import annotations

import numpy as np

from .salience_explorer import SalienceExplorer, MAX_TIER
from .online_model import OnlineActionEffectModel


class PrimaryModelExplorer(SalienceExplorer):
    def __init__(self, seed: int = 0, trust_threshold: int = 3, border_mask: int = 2,
                 conf_threshold: float = 0.5, novelty_limit: int = 40,
                 allow_cpu: bool = False, **model_kw) -> None:
        super().__init__(seed=seed, trust_threshold=trust_threshold, border_mask=border_mask)
        self.model = OnlineActionEffectModel(seed=seed, conf_threshold=conf_threshold,
                                             allow_cpu=allow_cpu, **model_kw)
        self.novelty_limit = novelty_limit
        self._cur_grid = None
        self._prev_grid = None       # grid the last-recorded transition was taken from
        self._model_level = 0
        self._model_off_level = False
        self._stall = 0
        self.n_model_picks = 0

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        try:
            grid = np.asarray(grid, dtype=np.int8)
        except Exception:
            grid = grid
        self._cur_grid = grid
        if levels != self._model_level:
            self._model_level = levels
            self._model_off_level = False
            self._stall = 0
            try:
                self.model.reset_level()
            except Exception:
                pass
        token = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        # remember the grid this action was taken from (grid_before for the NEXT transition)
        if token[0] == "reset" or gstate_terminal or gstate_notplayed:
            self._prev_grid = None
        else:
            self._prev_grid = grid
        return token

    def _record(self, key, action, next_key, reward, cands, terminal):
        # base records the (key,action)->next_key edge; we additionally train the model on the
        # observed outcome (did the MASKED state key change?). grid_before == self._prev_grid.
        super()._record(key, action, next_key, reward, cands, terminal)
        if self._prev_grid is not None:
            changed = key != next_key
            try:
                self.model.observe(self._prev_grid, action, bool(changed))
            except Exception:
                pass

    def _choose(self, cur):
        if (not self.model.usable) or self._model_off_level:
            return super()._choose(cur)
        node = self.nodes.get(cur)
        if node is None:
            return super()._choose(cur)
        # 1) exploit reward (inherited semantics)
        ra = node.reward_action()
        if ra is not None:
            return ra
        # 2) active backtrack plan
        if self.plan:
            if self._expect is not None and cur != self._expect:
                self.plan = []
            else:
                return self.plan.pop(0)
        # 3) model-driven frontier pick over ALL untried candidates (tier-agnostic)
        untried = node.untried_le(MAX_TIER)
        if untried:
            self._note_growth(node)
            pick = None
            try:
                pick = self.model.best(self._cur_grid, untried)
            except Exception:
                pick = None
            if pick is None or pick not in untried:
                # not confident / single candidate -> base order (lowest tier, random)
                mp = min(node.tier.get(a, 0) for a in untried)
                choices = [a for a in untried if node.tier.get(a, 0) == mp]
                pick = choices[int(self.rng.integers(0, len(choices)))]
            else:
                self.n_model_picks += 1
            return pick
        # 4) BFS to nearest node with ANY untried candidate
        path = self._path_to_frontier(cur, MAX_TIER)
        if path:
            self.plan = path
            return self.plan.pop(0)
        # 5) bounce off root / random (inherited semantics)
        if (self.root_key is not None and cur != self.root_key
                and self.stuck_resets < self.max_stuck_resets):
            self.stuck_resets += 1
            self.expect_reset = True
            self.plan = []
            return ("reset",)
        return self._random(cur)

    def _note_growth(self, node):
        """Novelty-stall guard: if the graph stops growing while model-driven, revert to base."""
        sz = len(self.nodes)
        if sz > getattr(self, "_last_sz", 0):
            self._stall = 0
        else:
            self._stall += 1
            if self._stall >= self.novelty_limit:
                self._model_off_level = True
        self._last_sz = sz
