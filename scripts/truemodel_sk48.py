"""True-model helpers for sk48 — used by bakeoff_metrics to compute A_h.

sk48 mechanics (reverse-engineered from environment_files/sk48/d8078629/sk48.py):
  Class       : Sk48 (subclass of ARCBaseGame, game_id="sk48")
  Actions     : [1, 2, 3, 4, 6, 7]
                  1-4 = directional (up/down/left/right)
                  6   = ACTION6 click (switches active snake)
                  7   = ACTION7 undo
  Effective BFS moves: ACTION1-4 only.
    - ACTION6 (switch): all levels have exactly 1 above-line snake so switching
      never produces a different state.
    - ACTION7 (undo): never on an optimal path.

  State space: a single above-line snake slides on rails.
    - Segments are always contiguous at fixed 6-pixel offsets from the head:
        seg[i] = (head.x + i*6, head.y)  for rotation=0 (rightward)
        seg[i] = (head.x, head.y + i*6)  for rotation=90 (downward)
      (verified empirically: head position + segment count fully determines the state)
    - The below-line "row snake" is a stationary target — it never moves.
    - Step budget (qiercdohl) counts down with each directional move; reaching 0 is a loss.
    - Win: above-line snake's segments must align with colored target blocks in the same
      color order as the row snake's segments.

  State key: (head_x, head_y, num_segments, step_budget)
    Compact and collision-free: fully determines the win predicate and all future states.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from arcengine import GameAction

SK48_PATH = Path(__file__).resolve().parent.parent / "environment_files/sk48/d8078629/sk48.py"

# Only directional moves are useful for BFS; ACTION6 and ACTION7 are never optimal.
SK48_MOVES = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]


def load_sk48_class():
    """Load and return the Sk48 game class from the environment source."""
    spec = importlib.util.spec_from_file_location("sk48_env", SK48_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Sk48


def sk48_key(g) -> tuple:
    """Compact BFS dedup key for sk48.

    Captures the above-line snake's head position, segment count, and remaining
    step budget — enough to fully determine all future reachable states and the
    win predicate.  The below-line row snake never moves so it is omitted.
    """
    fzjeqdahvs = 53  # the dividing line (y < 53 = above; matches sk48.py constant)
    active_head = g.vzvypfsnt
    # mwfajkguqx contains ALL snake heads; filter to the above-line one
    above_segs = [
        segs for head, segs in g.mwfajkguqx.items()
        if head.y < fzjeqdahvs
    ]
    if above_segs:
        head = next(h for h in g.mwfajkguqx if h.y < fzjeqdahvs)
        num_segs = len(g.mwfajkguqx[head])
        return (head.x, head.y, num_segs, g.qiercdohl)
    # Fallback (should not happen in normal play): use active head
    return (active_head.x, active_head.y, len(g.mwfajkguqx.get(active_head, [])), g.qiercdohl)
