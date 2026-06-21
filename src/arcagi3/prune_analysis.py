# src/arcagi3/prune_analysis.py
"""Pure, API-free analysis for the Phase 0a prune-capability oracle probe.

Consumes a captured banked-v6 trajectory (list[Step]) and answers, per completed level:
  Tier-1 (ceilings): how much exploration was OFF the shortest path to the level-up edge.
  Tier-2 (logo_auc): are on-path vs off-path states separable by decision-time features.
numpy only; no live API, no torch. See docs/superpowers/specs/2026-06-21-prune-oracle-probe-design.md
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from . import perception as P

RESET = ("reset",)


@dataclass
class Step:
    idx: int          # action index
    level: int        # levels_completed BEFORE this action
    from_key: bytes   # state acted from (pol.prev_key after decide)
    action: tuple     # ("S", id) | ("C", x, y) | ("reset",)
    reward: float     # levels_after - levels_before for this action
    tier: int         # salience tier of `action` at from_key


def build_edges(steps):
    """Global directed edges from consecutive steps, EXCLUDING reset actions and
    level-up (reward > 0) transitions (those cross into the next level/board).

    Returns (edges, first_seen):
      edges: dict[from_key] -> list[(action, to_key)]   (to_key = next step's from_key)
      first_seen: dict[key] -> idx of first step where it appears as from_key.
    """
    edges = defaultdict(list)
    first_seen = {}
    for i, s in enumerate(steps):
        if s.from_key not in first_seen:
            first_seen[s.from_key] = s.idx
        if i + 1 < len(steps) and s.action != RESET and s.reward <= 0:
            edges[s.from_key].append((s.action, steps[i + 1].from_key))
    return dict(edges), first_seen
