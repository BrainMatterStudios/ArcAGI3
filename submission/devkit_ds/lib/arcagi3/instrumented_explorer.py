"""InstrumentedExplorer — read-only subclass of the banked SalienceExplorer that tags which
_choose branch fired, for the Phase E re-traversal audit. _choose is a VERBATIM copy of the
parent's logic with ONLY counters added: same decisions, same RNG, byte-identical action trace
(enforced by tests/test_instrumented_explorer.py). See
docs/superpowers/specs/2026-06-21-phase-e-traversal-efficiency-measurement-design.md.
"""
from __future__ import annotations

from collections import Counter

from .salience_explorer import MAX_TIER, SalienceExplorer


class InstrumentedExplorer(SalienceExplorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__ (which calls reset_all); these persist across the run.
        self.tags: Counter = Counter()
        self.plan_invalidations: int = 0
        self.new_plan_len: list = []

    def _choose(self, cur):
        node = self.nodes.get(cur)
        if node is None:
            self.tags["random_no_node"] += 1
            return self._random(cur)
        # 1) exploit reward
        ra = node.reward_action()
        if ra is not None:
            self.tags["exploit"] += 1
            return ra
        # 2) active plan replay
        if self.plan:
            if self._expect is not None and cur != self._expect:
                self.plan_invalidations += 1
                self.plan = []
            else:
                self.tags["plan_replay_walk"] += 1
                return self.plan.pop(0)
        # 3) hierarchical tier exploration
        g = self.active_group
        while g <= MAX_TIER:
            local = node.untried_le(g)
            if local:
                self.tags["fresh_local_test"] += 1
                self.active_group = g
                mp = min(node.tier.get(a, 0) for a in local)
                choices = [a for a in local if node.tier.get(a, 0) == mp]
                return choices[int(self.rng.integers(0, len(choices)))]
            path = self._path_to_frontier(cur, g)
            if path:
                self.tags["new_plan_to_frontier"] += 1
                self.new_plan_len.append(len(path))
                self.active_group = g
                self.plan = path
                return self.plan.pop(0)
            g += 1
        # 4) exhausted from here -> bounce off root
        if (self.root_key is not None and cur != self.root_key
                and self.stuck_resets < self.max_stuck_resets):
            self.tags["reset_bounce"] += 1
            self.stuck_resets += 1
            self.expect_reset = True
            self.plan = []
            return ("reset",)
        self.tags["random_exhausted"] += 1
        return self._random(cur)
