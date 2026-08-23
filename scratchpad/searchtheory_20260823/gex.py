"""Go-Explore probe (Ecoffet et al. 2021, exploit-determinism phase 1),
snapshot-return variant, on the offline engine. Scratch only.

Archive: cell = HUD-masked frame hash (exact; the 64x64x16 grid IS already a
factored low-dim state, unlike Atari pixels). Each cell stores env snapshot,
depth, visit count. Loop: select cell with weight 1/sqrt(visits+1), restore
snapshot, roll out K random actions with 90% repeat-momentum (clicks drawn
from dynamic component centroids), archive novel cells, stop on level-up.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
ROOT = "/Users/ahmed/Documents/ArcAGI3"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scratchpad/ideas"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_agi import Arcade, OperationMode          # noqa: E402
from arcengine import GameState                    # noqa: E402
from hud_mask import frame_key                     # noqa: E402
from probe import GAMES, actions_for, apply_tok, lv, settled  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)

K_ROLLOUT = 30
P_REPEAT = 0.90


def explore_level(root_env, stem, target, budget_s, max_cells=30000):
    t0 = time.time()
    obs = root_env.reset()
    if lv(obs) >= target:
        return root_env, obs, 0, dict(depth=0, cells=1, steps=0, wall=0.0)
    archive = {}  # key -> [env, obs, depth, visits]
    k0 = frame_key(settled(obs), stem)
    archive[k0] = [root_env, obs, 0, 0]
    steps = 0
    max_depth = 0
    while time.time() - t0 < budget_s and len(archive) < max_cells:
        cells = list(archive.values())
        ws = np.array([1.0 / np.sqrt(c[3] + 1) for c in cells])
        cell = cells[rng.choices(range(len(cells)), weights=ws)[0]]
        cell[3] += 1
        env = copy.deepcopy(cell[0])
        obs, depth = cell[1], cell[2]
        prev_tok = None
        for _ in range(K_ROLLOUT):
            if time.time() - t0 > budget_s:
                break
            acts = actions_for(obs, stem)
            if not acts:
                break
            if prev_tok in acts and rng.random() < P_REPEAT:
                tok = prev_tok
            else:
                tok = rng.choice(acts)
            prev_tok = tok
            obs = apply_tok(env, tok)
            steps += 1
            depth += 1
            if obs is None:
                break
            if lv(obs) >= target:
                return env, obs, depth, dict(
                    depth=depth, cells=len(archive), steps=steps,
                    wall=round(time.time() - t0, 1))
            if obs.state == GameState.GAME_OVER:
                break
            k = frame_key(settled(obs), stem)
            hit = archive.get(k)
            if hit is None:
                archive[k] = [copy.deepcopy(env), obs, depth, 0]
                max_depth = max(max_depth, depth)
            elif depth < hit[2]:
                archive[k] = [copy.deepcopy(env), obs, depth, hit[3]]
    return None, None, None, dict(cells=len(archive), steps=steps,
                                  wall=round(time.time() - t0, 1),
                                  max_depth=max_depth)


def run_game(stem, budget_s):
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(ROOT, "environment_files"))
    env = client.make(GAMES[stem])
    env.reset()
    t_all = time.time()
    levels = []
    won = 0
    while time.time() - t_all < budget_s:
        remain = budget_s - (time.time() - t_all)
        win_env, win_obs, depth, st = explore_level(env, stem, won + 1, remain)
        if win_env is None:
            levels.append(dict(level=won + 1, solved=False, **st))
            break
        levels.append(dict(level=won + 1, solved=True, **st))
        won += 1
        env = win_env
        if win_obs is not None and win_obs.state == GameState.WIN:
            levels.append(dict(game_won=True))
            break
    return dict(game=stem, algo="gex", levels_won=won, budget_s=budget_s,
                wall=round(time.time() - t_all, 1), levels=levels)


if __name__ == "__main__":
    budget = int(sys.argv[1])
    stems = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(GAMES)
    for stem in stems:
        try:
            res = run_game(stem, budget)
        except Exception as e:  # noqa: BLE001
            res = dict(game=stem, algo="gex", error=repr(e))
        with open(os.path.join(OUT, f"{stem}_gex.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "levels"}),
              flush=True)
        for L in res.get("levels", []):
            print("   ", L, flush=True)
