"""UNIFIED general agent — all validated capabilities in ONE agent, as a max-over-plays portfolio (legitimate:
scoring is max over runs). For each game it tries, in order, independent PLAYS and keeps the first that wins:
  PLAY A  goal-directed DRAG   (drag_solve): induce actor+deltas+A5-selection+targets, drag movables to targets.
  PLAY B  two-phase SELECT->APPLY + search (select_apply_probe): strong compound-affordance detection + search.
  PLAY C  single-click affordance + search (general_solve): the base click/search path.
Each play is frames-only, ZERO per-game code. Measures the true combined coverage across all 25 dev games.
"""
from __future__ import annotations
import sys, time
from drag_solve import solve as drag_solve
from select_apply_probe import run as sa_run
from general_solve import solve as gen_solve

ALL = ["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0",
       "r11l","re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]


def unified(game, budget=15000, verbose=True):
    t0 = time.time()
    # PLAY A: goal-directed drag (cheap, covers move/paint/carry with a movable->target read)
    try:
        if drag_solve(game, verbose=False).get("solved"):
            if verbose: print(f"{game:>6}: SOLVED via DRAG ({time.time()-t0:.0f}s)")
            return dict(game=game, solved=True, via="drag")
    except Exception:
        pass
    # PLAY B: strong two-phase select->apply + search
    try:
        r = sa_run(game, verbose=False, max_nodes=budget)
        if r.get("solved"):
            if verbose: print(f"{game:>6}: SOLVED via SELECT->APPLY plan={r.get('plan_len')} ({time.time()-t0:.0f}s)")
            return dict(game=game, solved=True, via="select-apply", plan_len=r.get("plan_len"))
    except Exception:
        pass
    # PLAY C: single-click affordance + search
    try:
        r = gen_solve(game, max_nodes=budget, verbose=False)
        if r.get("solved"):
            if verbose: print(f"{game:>6}: SOLVED via SEARCH plan={r.get('plan_len')} ({time.time()-t0:.0f}s)")
            return dict(game=game, solved=True, via="search", plan_len=r.get("plan_len"))
    except Exception:
        pass
    if verbose: print(f"{game:>6}: unsolved ({time.time()-t0:.0f}s)")
    return dict(game=game, solved=False, via="-")


if __name__ == "__main__":
    games = sys.argv[1:] or ALL
    budget = 15000
    print(f"UNIFIED agent (drag + select->apply + search, budget={budget}), zero per-game code:\n")
    res = [unified(g, budget=budget) for g in games]
    s = [r for r in res if r["solved"]]
    from collections import Counter
    via = Counter(r["via"] for r in s)
    print(f"\n=== AGGREGATE: {len(s)}/{len(res)} solved zero-code ===")
    print("  solved:", ", ".join(r["game"] for r in s))
    print("  by path:", dict(via))
