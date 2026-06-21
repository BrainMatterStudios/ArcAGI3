# scripts/retraversal_audit.py
"""Phase E Increment 1 — re-traversal audit across the 16 tune+holdout games.

Runs the behavior-identical InstrumentedExplorer (banked v6 config) on each game and reports the
action anatomy: frontier-walk %, redundant-probe %, discovery %, exploit %, reset %, plus the
pre-registered proceed/kill check (a pool >=30% of actions on >=10/16 games + a structurally-safe
cut => Increment 2; else KILL).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/retraversal_audit.py [budget]
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3.instrumented_explorer import InstrumentedExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("audit"))

TUNE = ["vc33", "cd82", "sc25", "lp85", "lf52", "tu93", "ar25", "sp80"]
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]


def audit_game(prefix, budget):
    try:
        gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    except StopIteration:
        return None
    card = client.open_scorecard(tags=["retraversal-audit"])
    env = client.make(game_id=gid, scorecard_id=card)
    pol = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset()
    n, best = 0, 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame), st == GameState.GAME_OVER,
                         st == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        best = max(best, int(obs.levels_completed or 0))
        n += 1
    t = pol.tags
    walk = t["plan_replay_walk"] + t["new_plan_to_frontier"]
    local = t["fresh_local_test"]
    distinct = len(pol.nodes)                      # discoveries (only local tests discover)
    redundant = max(0, local - distinct)           # local probes that hit an already-known state
    exploit = t["exploit"]
    resets = t["reset_bounce"]
    tot = max(n, 1)
    wl = np.array(pol.new_plan_len) if pol.new_plan_len else np.array([0])
    return {"gid": gid, "best_level": best, "total_actions": n, "distinct_states": distinct,
            "pct_walk": round(walk / tot, 3), "pct_redundant_probe": round(redundant / tot, 3),
            "pct_discovery": round(distinct / tot, 3), "pct_exploit": round(exploit / tot, 3),
            "pct_reset": round(resets / tot, 3), "plan_invalidations": pol.plan_invalidations,
            "walk_count": len(pol.new_plan_len), "walk_mean_len": round(float(wl.mean()), 1),
            "walk_max_len": int(wl.max())}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    games = sys.argv[2:] if len(sys.argv) > 2 else (TUNE + HOLDOUT)
    t0 = time.time()
    rows = {}
    for g in games:
        r = audit_game(g, budget)
        if r is None:
            print(f"  {g}: NOT IN DEV SET", flush=True)
            continue
        rows[g] = r
        print(f"  {g} L{r['best_level']} acts={r['total_actions']}: "
              f"walk={r['pct_walk']} redundant={r['pct_redundant_probe']} "
              f"discovery={r['pct_discovery']} exploit={r['pct_exploit']} reset={r['pct_reset']} "
              f"invalid={r['plan_invalidations']} (walks={r['walk_count']} mean={r['walk_mean_len']} "
              f"max={r['walk_max_len']})", flush=True)
    if rows:
        n = len(rows)
        for pool in ("pct_walk", "pct_redundant_probe"):
            vals = [r[pool] for r in rows.values()]
            n_ge30 = sum(1 for v in vals if v >= 0.30)
            print(f"== {pool}: action-weighted mean={np.mean(vals):.3f}  games>=30%: {n_ge30}/{n}",
                  flush=True)
        print("== PRE-REGISTERED: a pool >=30% on >=10/16 games + a structurally-safe cut "
              "=> Increment 2; else KILL the efficiency lever.", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)
    out = "/tmp/retraversal_audit.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
