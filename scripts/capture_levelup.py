"""Mechanic induction, step 0: CAPTURE the transition that actually causes a level-up.

User-directed pivot (2026-06-22): stop asking 'what is the goal?' and discover 'what are the
transition laws?'. The most direct probe: drive salience until a level-up, then log the EXACT
(prev_state, action, delta) that triggered it — plus the per-color/per-action effect grammar leading
up to it. This empirically reveals each game's real mechanic (answers: hidden rule? wrong perceived
goal? missing causal abstraction?).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/capture_levelup.py [game] [budget]
"""
from __future__ import annotations
import logging, sys, time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("cap"))


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def gridsafe(obs):
    try:
        return P.to_grid(obs.frame)
    except Exception:
        return None


def colhist(g, bg):
    v, ct = np.unique(g, return_counts=True)
    return {int(a): int(b) for a, b in zip(v, ct) if a != bg}


def describe_delta(g0, g1, bg):
    if g0 is None or g1 is None or g0.shape != g1.shape:
        return "shape-change/terminal"
    h0, h1 = colhist(g0, bg), colhist(g1, bg)
    cols = set(h0) | set(h1)
    changes = {c: (h0.get(c, 0), h1.get(c, 0)) for c in cols if h0.get(c, 0) != h1.get(c, 0)}
    ncells = int((g0 != g1).sum())
    return f"cells={ncells} colorcount_changes={changes}"


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["cap"]))
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2,
                           coarse_grid_step=4, max_click_targets=256)
    obs = _retry(env.reset)
    bg = P.detect_background(gridsafe(obs))
    prev_grid = gridsafe(obs)
    prev_levels = 0
    prev_tok = None
    n = 0
    t0 = time.time()
    print(f"{game}: bg={bg} capturing level-up transitions (budget={budget})", flush=True)
    while n < budget:
        g = gridsafe(obs)
        st = obs.state
        tok = pol.decide(g if g is not None else prev_grid,
                         st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0),
                         [a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])])
        if st == GameState.WIN:
            print(f"  WIN at {n}", flush=True); break
        prev_grid_for_step = g
        if tok[0] == "reset":
            obs = _retry(env.reset)
        elif tok[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(tok[1])))
        else:
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])}))
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > prev_levels:
            ng = gridsafe(obs)
            print(f"\n  *** LEVEL-UP {prev_levels}->{lv} at action {n} ***", flush=True)
            print(f"      triggering action: {tok}", flush=True)
            print(f"      delta: {describe_delta(prev_grid_for_step, ng, bg)}", flush=True)
            prev_levels = lv
            if lv >= 1:
                print(f"\n  reached L{lv} in {n} actions, {time.time()-t0:.0f}s", flush=True)
                break
        prev_grid = g if g is not None else prev_grid
    if prev_levels == 0:
        print(f"  no level-up in {budget} actions", flush=True)


if __name__ == "__main__":
    main()
