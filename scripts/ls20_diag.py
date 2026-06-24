"""ls20 engine diagnostic — why did WinningExplorer start only 9 plans on the 0.996-accurate model?

Probe ls20 live, fit the model beam, then report: agent, deltas, walls, confidence over time, the
goals enumerated, and for EACH goal whether plan_to() returns a path. Pinpoints whether ls20's
0-clear is the goal-wall or a fixable engine bug (low confidence / bad lattice-snap / goal missing).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv; load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.discovery_bakeoff import make_game
from arcagi3 import perception as P
from arcagi3.mechanics.model_search import ModelSearch
from arcagi3.planning.model_beam_planner import best_plan
from arcagi3.transfer_explorer import TransferExplorer
from arcengine import GameAction, GameState

logging.basicConfig(level=logging.ERROR)


def main():
    probe = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    ms = ModelSearch()
    drv = TransferExplorer(seed=0); drv.reset_all()  # let transfer drive; model just observes+probes
    env = make_game("ls20"); obs = env.reset()
    prev = paid = None
    conf_trace = []
    n = 0
    while n < probe and obs is not None and obs.state != GameState.WIN and np.asarray(obs.frame).size:
        grid = P.to_grid(obs.frame)
        if prev is not None:
            ms.observe(prev, paid, grid)
        # round-robin directional probe to build the movement model fast
        simple = [a for a in (obs.available_actions or []) if a in (1, 2, 3, 4)]
        a = ms.probe_action(grid, list(obs.available_actions or [])) if simple else None
        if n % 50 == 0 and n > 0:
            ms.fit()
            conf_trace.append((n, round(ms.confidence(), 3), len(ms.beam)))
        prev = grid; paid = a
        if a is not None:
            obs = env.step(GameAction.from_id(a)); n += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": 32, "y": 32}); n += 1

    ms.fit()
    m = ms.best()
    print(f"=== ls20 diagnostic after {n} probe actions ===")
    print(f"confidence trace (n, conf, beam): {conf_trace}")
    if m is None:
        print("NO MODEL FIT — agent not identified"); return
    print(f"agent_colors={m.agent_colors} bg={m.bg}")
    print(f"deltas={m.move.deltas}  walls={len(m.move.walls)}  paint={m.paint.map()} collect={m.collect.colors}")
    print(f"held-out transition_acc={round(m.transition_accuracy,3)} MOVE_acc={round(m.move_accuracy,3)} (gate 0.85)")
    grid = prev
    ap = m.agent_pos(grid)
    print(f"agent_pos={ap}")
    goals = ms.goals(grid)
    print(f"\nenumerated {len(goals)} goals; plan_to() per goal:")
    for g in goals:
        key = (g.kind, getattr(g, "color", getattr(g, "axis", None)))
        tcells = g.target_cells(grid, m.agent_colors)
        snapped = {m._snap(t, ap) for t in tcells[:8]} if (ap and tcells) else set()
        plan = m.plan_to(grid, g)
        print(f"  {str(key):28} targets={len(tcells):3} snapped~{list(snapped)[:3]} "
              f"-> plan={'len ' + str(len(plan)) if plan else 'None'}")
    bp = best_plan(ms, grid)
    print(f"\nbest_plan -> {None if bp is None else (bp.goal.kind, len(bp.plan), 'conf', round(bp.confidence,3))}")


if __name__ == "__main__":
    main()
