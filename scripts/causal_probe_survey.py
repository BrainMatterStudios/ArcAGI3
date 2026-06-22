"""Causal-probe survey (logging-only, ZERO behavior change) — increment 2 of the causal-probe
architecture. Runs the v13 backbone (TransferExplorer) and LOGS, per transition, the observed
causal delta. Answers the central design risk BEFORE building the controller:

  Do the scored / wall games emit any probe-able causal signal at all?

For each game we measure: what fraction of actions produce a frame delta, how many produce a
level-up, and a per-action-CLASS effect table (simple action id -> effect; clicked color ->
effect). A game where ~0% of actions ever change the frame is signal-STARVED (probes can't help
it); a game with deltas we're not exploiting is signal-RICH (the proof-gated exploit gate can).

This builds the empirical "mechanic certificate store" contents the real CausalProbeExplorer will
maintain online. It changes NO behavior — it only observes a normal TransferExplorer run.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/causal_probe_survey.py [budget]
"""
from __future__ import annotations
import json
import logging
import sys
import time
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("survey"))


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"    [retry {i+1}/{tries}] {type(e).__name__}", flush=True)
            time.sleep(delay * (i + 1))


def survey_game(gid: str, budget: int):
    card = client.open_scorecard(tags=["survey"])
    env = client.make(game_id=gid, scorecard_id=card)
    # v13 shipping config: dense click lattice
    pol = TransferExplorer(seed=0, trust_threshold=3, border_mask=2,
                           coarse_grid_step=4, max_click_targets=256)
    obs = _retry(env.reset)
    prev_grid = P.to_grid(obs.frame)
    prev_levels = int(obs.levels_completed or 0)

    n = 0
    n_changed = 0
    n_levelup = 0
    simple = defaultdict(lambda: [0, 0, 0])          # aid -> [tried, changed, levelup]
    click_color = defaultdict(lambda: [0, 0, 0, 0])  # color -> [tried, changed, cells, levelup]
    avail_union = set()

    while n < budget:
        st = obs.state
        avail = list(obs.available_actions or [])
        avail_union.update(a.value if hasattr(a, "value") else int(a) for a in avail)
        tok = pol.decide(prev_grid,
                         gstate_terminal=(st == GameState.GAME_OVER),
                         gstate_notplayed=(st == GameState.NOT_PLAYED),
                         levels=prev_levels,
                         available=[a.value if hasattr(a, "value") else int(a) for a in avail])
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = _retry(env.reset)
            prev_grid = P.to_grid(obs.frame)
            prev_levels = int(obs.levels_completed or 0)
            continue
        clicked_color = None
        if tok[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(tok[1])))
        else:
            col, row = int(tok[1]), int(tok[2])
            if 0 <= row < prev_grid.shape[0] and 0 <= col < prev_grid.shape[1]:
                clicked_color = int(prev_grid[row, col])
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": col, "y": row}))

        new_grid = P.to_grid(obs.frame)
        new_levels = int(obs.levels_completed or 0)
        cells = int(np.count_nonzero(new_grid != prev_grid)) if new_grid.shape == prev_grid.shape else -1
        changed = cells != 0
        levelup = new_levels > prev_levels
        n += 1
        n_changed += int(changed)
        n_levelup += int(levelup)
        if tok[0] == "S":
            rec = simple[tok[1]]
            rec[0] += 1; rec[1] += int(changed); rec[2] += int(levelup)
        elif clicked_color is not None:
            rec = click_color[clicked_color]
            rec[0] += 1; rec[1] += int(changed); rec[2] += max(cells, 0); rec[3] += int(levelup)

        prev_grid = new_grid
        prev_levels = new_levels

    return {
        "actions": n,
        "pct_changed": round(n_changed / max(n, 1), 4),
        "n_changed": n_changed,
        "n_levelup": n_levelup,
        "levels_reached": prev_levels,
        "available_actions": sorted(avail_union),
        "simple_effects": {k: v for k, v in sorted(simple.items())},
        "click_color_effects": {k: v for k, v in sorted(click_color.items())},
    }


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    t0 = time.time()
    gids = sorted(e.game_id for e in client.get_environments())
    out = {}
    print(f"surveying {len(gids)} games @ budget={budget}\n", flush=True)
    for gid in gids:
        key = gid[:4]
        try:
            r = survey_game(gid, budget)
        except Exception as e:  # noqa: BLE001
            print(f"  {key}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        out[key] = r
        # classify: starved if almost nothing ever changes the frame
        tag = "STARVED" if r["pct_changed"] < 0.02 else ("rich" if r["pct_changed"] > 0.3 else "mid")
        print(f"  {key}: L{r['levels_reached']} pct_changed={r['pct_changed']:.3f} "
              f"levelups={r['n_levelup']} avail={r['available_actions']} [{tag}]", flush=True)
    path = f"/tmp/causal_survey_{budget}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nelapsed {time.time()-t0:.0f}s  saved {path}", flush=True)


if __name__ == "__main__":
    main()
