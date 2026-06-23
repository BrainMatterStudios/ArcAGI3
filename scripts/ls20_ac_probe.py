"""Exp 50b — ls20 A/C disambiguation: does disjunction-fixed discovery clear ls20?

The conjunctive-slot poison (Exp-45 Wall C) is ALREADY fixed in discovery_explorer._build_plan
(commit dca97e4: disjunctive candidate sweep + footprint-aligned snap). But ls20 was never
re-run end-to-end with all fixes in place (Exp 49 tested other holdouts). This probe runs the
current DiscoveryExplorer on ls20 with phase/plan instrumentation:

  - if it CLEARS L1  -> Wall C alone was the residual blocker; the fix works.
  - if it does NOT   -> confirms Wall A (the moving-goal ontology gap, Exp-40): even with a correct
                        planner + the paint primitive, the static-slot DSL can't express ls20's
                        moving color-9 goal / strategic path-opening.

ls20 source is touched ONLY by the A_h grader. Live API (NORMAL); keep the budget modest.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/ls20_ac_probe.py [budget] [backend]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `scripts` package

from scripts.discovery_bakeoff import make_game
from arcagi3 import perception as P
from arcagi3.discovery_explorer import DiscoveryExplorer
from arcengine import GameAction, GameState


def run(budget: int, backend: str):
    eng = DiscoveryExplorer(seed=0, planner_backend=backend)
    eng.reset_all()
    env = make_game("ls20")
    obs = env.reset()
    cur_level = int(obs.levels_completed or 0)
    n = 0
    phase_counts: dict[str, int] = {}
    plans_built = 0
    last_phase = None
    induced_logged = False
    print(f"=== ls20 A/C probe — backend={backend} budget={budget} ===", flush=True)
    while n < budget and cur_level < 2:
        if obs.state == GameState.WIN:
            print("  WIN", flush=True)
            break
        grid = P.to_grid(obs.frame)
        token = eng.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=cur_level,
            available=list(obs.available_actions or []),
        )
        ph = getattr(eng, "_phase", "?")
        phase_counts[ph] = phase_counts.get(ph, 0) + 1
        if ph != last_phase:
            # log phase transitions and, once a model is induced, its summary
            if ph == "PLAN" and not induced_logged and getattr(eng, "_deltas", None):
                print(f"  [induced @ n={n}] deltas={eng._deltas} "
                      f"paint_colors={sorted(getattr(eng, '_paint_colors', set()))} "
                      f"walls={len(getattr(eng, '_walls', set()))} "
                      f"tiles={len(getattr(eng, '_tiles', {}))}", flush=True)
                induced_logged = True
            last_phase = ph
        # detect a freshly built plan reaching EXECUTE
        if ph == "EXECUTE" and getattr(eng, "_plan", None) and getattr(eng, "_goal_pos", None):
            gp = eng._goal_pos
            if not hasattr(run, "_seen_goals"):
                run._seen_goals = set()
            if gp not in run._seen_goals:
                run._seen_goals.add(gp)
                plans_built += 1
                if plans_built <= 25:
                    print(f"  [plan {plans_built} @ n={n}] goal={gp} len={len(eng._plan)} "
                          f"tried={len(getattr(eng, '_tried_goals', set()))}", flush=True)

        if token[0] == "reset":
            obs = env.reset()
        elif token[0] == "S":
            obs = env.step(GameAction.from_id(token[1])); n += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]}); n += 1

        new_level = int(obs.levels_completed or 0)
        if new_level > cur_level:
            print(f"  *** LEVEL UP {cur_level}->{new_level} at n={n} actions ***", flush=True)
            cur_level = new_level

    print(f"\n--- result backend={backend}: levels_cleared={cur_level}  actions={n}  "
          f"distinct_goals_tried={plans_built} ---", flush=True)
    print(f"    phase histogram: {phase_counts}", flush=True)
    return cur_level


if __name__ == "__main__":
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    backend = sys.argv[2] if len(sys.argv) > 2 else "painted_set"
    run(budget, backend)
