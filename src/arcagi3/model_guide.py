"""ModelGuide — advisory wrapper over the discovery induction/planner.

Maintains an induced model from the observed (prev_grid, action, grid) stream and suggests a
goal-directed action, WITHOUT emitting actions itself. Used by HybridTransferExplorer to bias the
salience search. Reuses DiscoveryExplorer's induction internals (no duplication); if a level's
model can't yield a plan, suggest() returns None and the explorer falls back to salience.
"""
from __future__ import annotations

from arcagi3 import perception as P
from arcagi3.discovery_explorer import DiscoveryExplorer


class ModelGuide:
    def __init__(self):
        self._d = DiscoveryExplorer(seed=0)
        self._d.reset_all()
        self._obs_count = 0
        self._built_at = -1
        self._plan_cache: list = []

    def observe(self, prev_grid, prev_action, grid, level) -> None:
        d = self._d
        if d._bg is None:
            d._bg = P.detect_background(grid)
        if level != d._last_level:
            d.on_level_change(level)
        d._prev_grid, d._prev_token = prev_grid, prev_action
        d._ingest_movement(grid)
        d._ingest_transform(grid)
        d._ingest_world_delta(grid)
        self._obs_count += 1

    def suggest(self, grid, available):
        d = self._d
        if d._bg is None or not d._deltas:
            return None
        if self._obs_count != self._built_at:
            d._build_model(grid)
            d._plan = []
            d._build_plan(grid)
            self._plan_cache = list(d._plan)
            self._built_at = self._obs_count
        if self._plan_cache:
            a = self._plan_cache[0]
            if isinstance(a, int) and a in available:
                return a
        return None
