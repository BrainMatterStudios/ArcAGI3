"""Additive, stall-gated MECHANIC-SOLVER scaffold — the regression-safe way to wire a mechanic planner into
the portfolio. It runs as a SEPARATE play (max-over-plays): recognize whether the current game matches a
known archetype; if so, plan+execute to solve it; if NOT, ABSTAIN (emit nothing) so the portfolio keeps the
banked coverage play. By construction it can only ADD levels on games it recognizes and can never regress the
0.33 floor.

Currently registered archetype: wa30-style grab-drag (avatar + movable blocks + a goal pad). Adding a new
archetype = register (recognize, plan) — the same perceive->forward-model->plan pattern per mechanic. HONEST
caveat: this only improves the EVAL score if a hidden game reuses a registered archetype (unmeasurable
offline; the mechanics are adversarially varied by design).
"""
from __future__ import annotations
import numpy as np
from arcengine import GameAction, GameState
from arcagi3 import perception as P
import wa30_planner as W


def recognize_grabdrag(grid):
    """wa30-style: an avatar (c14), >=1 movable block (c4), and a goal strip (large c9 comp)."""
    avatar, blocks, pads = W.perceive(grid)
    return avatar is not None and len(blocks) >= 1 and len(pads) >= 1


ARCHETYPES = [("grabdrag", recognize_grabdrag, W.perceive, W.plan_all)]


def try_solve(env, obs, budget=4000):
    """Attempt to solve the current game with a recognized archetype. Returns (levels_gained, actions_used)
    or None if no archetype recognizes it (ABSTAIN -> portfolio keeps the coverage play)."""
    grid = P.to_grid(obs.frame)
    for name, recog, perceive, plan_fn in ARCHETYPES:
        if not recog(grid):
            continue
        used = 0
        start_lv = int(obs.levels_completed or 0)
        while used < budget:
            grid = P.to_grid(obs.frame)
            avatar, blocks, pads = perceive(grid)
            if avatar is None or not blocks or not pads:
                break
            plan = plan_fn(avatar, blocks, pads)
            if not plan:
                break
            progressed = False
            for a in plan:
                obs = env.step(GameAction.ACTION5 if a == 5 else GameAction.from_id(a))
                used += 1
                if int(obs.levels_completed or 0) > start_lv or obs.state == GameState.WIN:
                    progressed = True
                    start_lv = int(obs.levels_completed or 0)
                    break
                if obs.state == GameState.GAME_OVER:
                    return (start_lv - int(obs.levels_completed or 0), used) if False else (start_lv, used)
            if not progressed:
                break
        return (start_lv, used)  # levels reached via this archetype, actions used
    return None  # abstain
