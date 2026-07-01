"""MONOTONE-INVARIANT / MANUFACTURED-REWARD PROBE (proposed months ago, never run).
Question: is there a general SCALAR function of the frame that monotonically tracks level-progress? If yes,
it is a dense reward we can hill-climb (attacks W2 reward-starvation) WITHOUT knowing the mechanic.

Method: drive the coverage explorer; at each action record level and a battery of frame-scalars. For each
completed level, measure whether each scalar RISES over the actions leading INTO the level-up (Spearman-like
monotonicity of the last K frames before each reward vs a random-window baseline).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 python scripts/research_2026_07_01/scalar_reward_probe.py [budget] [games]
"""
import logging, sys
from collections import defaultdict
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
logging.basicConfig(level=logging.ERROR)

GAMES = ["vc33", "cd82", "tu93", "lp85", "lf52", "ar25", "su15", "m0r0"]


def scalars(grid):
    bg = P.detect_background(grid)
    objs = P.connected_components(grid, background=bg)
    nonbg = int((grid != bg).sum())
    ncolors = len(set(int(v) for v in grid.flatten()))
    # symmetry defect (lower = more symmetric)
    sym_lr = int((grid != grid[:, ::-1]).sum())
    sym_ud = int((grid != grid[::-1, :]).sum())
    # largest-object size, count
    sizes = [o.size for o in objs]
    return {
        "nonbg_cells": nonbg,
        "n_objects": len(objs),
        "n_colors": ncolors,
        "sym_lr_match": -sym_lr,      # higher = more L-R symmetric
        "sym_ud_match": -sym_ud,
        "max_obj_size": max(sizes) if sizes else 0,
        "mean_obj_size": float(np.mean(sizes)) if sizes else 0,
    }


def run(game, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"scal-{game}")
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset(); n = 0; prev = 0
    traj = []  # (action_idx, level, {scalars})
    levelups = []
    while n < budget:
        if obs.state == GameState.WIN:
            break
        g = P.to_grid(obs.frame)
        traj.append((n, int(obs.levels_completed or 0), scalars(g)))
        tok = pol.decide(g, obs.state == GameState.GAME_OVER, obs.state == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        if int(obs.levels_completed or 0) > prev:
            levelups.append(n); prev = int(obs.levels_completed or 0)
    return traj, levelups


def monotonicity(vals):
    """fraction of consecutive steps that are non-decreasing (1.0 = perfectly rising)."""
    if len(vals) < 2:
        return 0.5
    ups = sum(1 for i in range(1, len(vals)) if vals[i] >= vals[i-1])
    return ups / (len(vals) - 1)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    games = sys.argv[2].split(",") if len(sys.argv) > 2 else GAMES
    K = 40  # window of actions before a level-up
    agg = defaultdict(list)      # scalar -> list of pre-levelup monotonicity
    base = defaultdict(list)     # scalar -> random-window monotonicity (baseline)
    keys = None
    for gm in games:
        traj, lus = run(gm, budget)
        if not lus:
            continue
        idx = {t[0]: t[2] for t in traj}
        order = [t[0] for t in traj]
        for lu in lus:
            pre = [a for a in order if lu - K <= a < lu]
            if len(pre) < 5:
                continue
            for sk in pre[0] and idx[pre[0]]:
                keys = idx[pre[0]].keys()
                break
            for sk in idx[pre[0]].keys():
                agg[sk].append(monotonicity([idx[a][sk] for a in pre]))
        # baseline: random windows not ending in a level-up
        import random
        for _ in range(len(lus) * 3):
            if len(order) <= K + 1:
                break
            s = order[(len(order)//2) % max(1, len(order)-K)]
            win = [a for a in order if s <= a < s + K]
            if len(win) >= 5:
                for sk in idx[win[0]].keys():
                    base[sk].append(monotonicity([idx[a][sk] for a in win]))
    print(f"MANUFACTURED-REWARD PROBE (budget {budget}, {len(games)} games). "
          f"pre-levelup monotonicity vs random-window baseline:\n")
    print(f"{'scalar':>16} {'pre-LU rise':>12} {'baseline':>10} {'lift':>8}")
    for sk in sorted(agg, key=lambda k: -(np.mean(agg[k]) - np.mean(base[k] or [0.5]))):
        pu = float(np.mean(agg[sk])); bl = float(np.mean(base[sk] or [0.5]))
        print(f"{sk:>16} {pu:>12.3f} {bl:>10.3f} {pu-bl:>+8.3f}")
    print("\nRead: a scalar with pre-levelup rise >> baseline is a DENSE PROGRESS SIGNAL to hill-climb "
          "(manufactured reward, attacks W2). Lift ~0 => no general dense signal in these scalars.")


if __name__ == "__main__":
    main()
