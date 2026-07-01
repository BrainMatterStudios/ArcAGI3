"""FRONT 2+3: parallel end-to-end run of the unified general agent across ALL 25 dev games.

Demonstrates the compute-as-search lever (games are independent -> embarrassingly parallel; on Kaggle T4x2's
CPUs this turns the 12h budget into search depth) and produces the honest aggregate: coverage + which play
solved each + plan length vs human proxy. Each worker builds its own env (fork-safe: no shared engine state).
"""
from __future__ import annotations
import sys, time, os
from concurrent.futures import ProcessPoolExecutor, as_completed

ALL = ["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0",
       "r11l","re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]


def _one(game, budget):
    import logging; logging.basicConfig(level=logging.ERROR)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from unified_agent import unified
    t = time.time()
    try:
        r = unified(game, budget=budget, verbose=False)
    except Exception as e:  # noqa: BLE001
        return dict(game=game, solved=False, via=f"ERR:{type(e).__name__}", secs=time.time()-t)
    r["secs"] = time.time() - t
    return r


def main(budget=15000, workers=6):
    from click_affordance_probe import HUMAN_PROXY
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, g, budget): g for g in ALL}
        for f in as_completed(futs):
            r = f.result(); results[r["game"]] = r
            tag = "OK " if r["solved"] else "   "
            print(f"  {tag}{r['game']:>6}  {r.get('via','-'):>14}  "
                  f"plan={r.get('plan_len','-')!s:>4}  {r['secs']:.0f}s")
    res = [results[g] for g in ALL if g in results]
    solved = [r for r in res if r["solved"]]
    from collections import Counter
    print(f"\n=== {len(solved)}/{len(res)} solved zero per-game code  (wall {time.time()-t0:.0f}s, {workers} workers) ===")
    print("  solved:", ", ".join(r["game"] for r in solved))
    print("  by play:", dict(Counter(r["via"] for r in solved)))
    prox = [(r["game"], r["plan_len"], HUMAN_PROXY[r["game"]]) for r in solved
            if r.get("plan_len") and HUMAN_PROXY.get(r["game"])]
    if prox:
        print("  efficiency vs human:", ", ".join(f"{g}={p/h:.1f}x" for g, p, h in prox))


if __name__ == "__main__":
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(f"Parallel unified-agent run (budget={budget}, workers={workers}):\n")
    main(budget, workers)
