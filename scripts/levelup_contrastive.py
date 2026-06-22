"""Level-Up Contrastive analysis (user-directed 2026-06-22): which state variables form the
completion predicate?

Not 'what is the goal' and not 'what are the mechanics' (largely known) — but: what STRUCTURALLY
distinguishes the near-win state from the thousands of ordinary states? Drive salience to L1 while
logging a structural feature vector each step; then contrast the pre-win state against the full
run distribution to see which feature was extremal at completion (all painted? all collected?
coverage threshold? component count?).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/levelup_contrastive.py [game] [budget]
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
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("lc"))


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


def features(g, bg):
    """Structural feature vector: per-color counts + n_components + coverage-ish stats."""
    f = {}
    v, ct = np.unique(g, return_counts=True)
    for a, b in zip(v, ct):
        f[f"c{int(a)}"] = int(b)
    try:
        objs = P.connected_components(g, background=bg)
        f["ncomp"] = len(objs)
    except Exception:
        f["ncomp"] = -1
    f["nonbg"] = int((g != bg).sum())
    return f


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["lc"]))
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2,
                           coarse_grid_step=4, max_click_targets=256)
    obs = _retry(env.reset)
    bg = P.detect_background(gridsafe(obs))
    traj = []          # feature dicts over the run
    prewin = None
    prev_levels = 0
    prev_feat = None
    n = 0
    while n < budget:
        g = gridsafe(obs)
        st = obs.state
        if g is not None and st not in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            ft = features(g, bg)
            traj.append(ft)
            prev_feat = ft
        tok = pol.decide(g if g is not None else np.zeros((64, 64), np.int8),
                         st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0),
                         [a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])])
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = _retry(env.reset)
        elif tok[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(tok[1])))
        else:
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])}))
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > prev_levels:
            prewin = prev_feat   # the state just before the triggering action
            break

    if prewin is None:
        print(f"{game}: no level-up in {budget}"); return
    # contrast pre-win features vs the run distribution
    keys = sorted(set().union(*[set(d) for d in traj]))
    print(f"{game}: reached L1 at action {n}; {len(traj)} states logged. Contrast (pre-win vs run dist):", flush=True)
    print(f"  {'feat':<8}{'prewin':>9}{'min':>9}{'max':>9}{'mean':>9}{'  extremal?'}", flush=True)
    for k in keys:
        vals = np.array([d.get(k, 0) for d in traj], dtype=float)
        pw = prewin.get(k, 0)
        lo, hi, mu = vals.min(), vals.max(), vals.mean()
        tag = ""
        if pw <= lo + 0.001:
            tag = "<-- MIN (depleted?)"
        elif pw >= hi - 0.001:
            tag = "<-- MAX (filled?)"
        print(f"  {k:<8}{pw:>9.0f}{lo:>9.0f}{hi:>9.0f}{mu:>9.1f}  {tag}", flush=True)


if __name__ == "__main__":
    main()
