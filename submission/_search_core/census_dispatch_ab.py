"""A/B the AVATAR dispatch order on the LIVE cost model — does nbfs-first make
more cracks affordable?

WHY (measured 2026-08-28).

`DISPATCH_ORDER['AVATAR'] = ['nbfs_macros', 'nbfs', 'goexplore']`, so the
shipped grinder auditions `nbfs_macros` first on every AVATAR game. On tu93 —
the only game in the 25 that the GENERIC lane cracks — that choice is what
makes the crack unflyable:

    nbfs_macros   84,681 offline actions  x2.66 guard tax -> 1,733 s  vs 1,500 s cap: +15.5% FAIL
    nbfs          73,800 offline actions  x2.66            -> 1,510 s              : +0.7%

Both crack all 9 levels. The research doc records THREE attempts to close that
16% gap — a depth bound (0 pruned), a path cache (9 actions of 58,522), and
skipping the portfolio audition (4 actions) — and concludes the deployable
crack inventory is 1. All three optimised WITHIN the macros lane. None
cost-tested the plain `nbfs` lane, which closes 94% of the gap on its own.

The ordering is defensible on its own terms: macros win more LEVELS across the
corpus. But the crack lane does not care about level count on games it cannot
crack; it cares about total actions on games it can. Optimising the wrong
objective is the "actions are not levels" law pointing the other way.

WHAT THIS MEASURES. The full 25-game census on `reset_replay`, with AVATAR
reordered to nbfs-first, reported next to the shipped order. The numbers that
decide anything:

  * cracks, and for each crack its LIVE seconds (offline actions x2.66 / 130)
    against the 1,500 s owned engagement cap;
  * total levels won, because if nbfs-first cracks the same games but loses
    levels elsewhere it may still be a net loss for the LLM-assist path.

Run:  ONLY_RESET_LEVELS=true .venv/bin/python \
          submission/_search_core/census_dispatch_ab.py [budget_s] [jobs]
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("ONLY_RESET_LEVELS", "true")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
ROOT = os.path.dirname(os.path.dirname(_HERE))

GUARD_TAX = 2.66          # MEASURED on CompetitionArcadeServer, ENVELOPE §D1
ACT_PER_S = 130.0
OWNED_CAP_S = 1500.0


def live_seconds(offline_actions: int) -> float:
    """Offline engine actions -> live seconds, via the measured guard tax.
    ENVELOPE-2026-08-26-v8 §D1: the competition path cost 2.66x the offline
    backend on tu93 (84,687 -> 225,511 actions), because the competition guard
    swallows resets issued at _action_count == 0 (api.py:316-334)."""
    return offline_actions * GUARD_TAX / ACT_PER_S


def one(stem: str, budget_s: float, order: list[str] | None):
    import logging

    logging.disable(logging.CRITICAL)
    import search_core as sc

    if order is not None:
        sc.DISPATCH_ORDER["AVATAR"] = list(order)
    t0 = time.time()
    try:
        r = sc.run_game(stem, budget_s, backend="reset_replay",
                        algo="portfolio")
    except Exception as exc:  # noqa: BLE001
        return {"game": stem, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "game": stem, "levels_won": r["levels_won"], "cracked": r["cracked"],
        "actions": r["actions_spent"], "winner": r.get("winner"),
        "archetype": r.get("archetype"), "wall": round(time.time() - t0, 1),
        "live_s": round(live_seconds(r["actions_spent"]), 1),
        "fits_cap": live_seconds(r["actions_spent"]) <= OWNED_CAP_S,
    }


def _worker(args):
    return one(*args)


def run(order, label, games, budget_s, jobs):
    import multiprocessing as mp

    print(f"\n=== {label}: AVATAR order = {order} ===", flush=True)
    with mp.Pool(jobs) as pool:
        rows = pool.map(_worker, [(g, budget_s, order) for g in games])
    rows.sort(key=lambda r: r["game"])
    print(f"{'game':6} {'lvls':>4} {'crack':>5} {'actions':>9} "
          f"{'live_s':>8} {'fits':>5}  winner")
    for r in rows:
        if "error" in r:
            print(f"{r['game']:6} ERROR {r['error'][:60]}")
            continue
        print(f"{r['game']:6} {r['levels_won']:4} "
              f"{'YES' if r['cracked'] else '':>5} {r['actions']:9} "
              f"{r['live_s']:8.1f} {'ok' if r['fits_cap'] else 'OVER':>5}  "
              f"{r.get('winner')}")
    ok = [r for r in rows if "error" not in r]
    cracks = [r for r in ok if r["cracked"]]
    flyable = [r for r in cracks if r["fits_cap"]]
    print(f"  TOTAL levels {sum(r['levels_won'] for r in ok)}   "
          f"cracks {len(cracks)}   FLYABLE cracks {len(flyable)} "
          f"{[r['game'] for r in flyable]}")
    return rows


def main() -> int:
    budget_s = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    env_dir = os.path.join(ROOT, "environment_files")
    games = sorted(d for d in os.listdir(env_dir)
                   if os.path.isdir(os.path.join(env_dir, d)))
    t0 = time.time()
    shipped = run(["nbfs_macros", "nbfs", "goexplore"], "SHIPPED",
                  games, budget_s, jobs)
    nbfs_first = run(["nbfs", "nbfs_macros", "goexplore"], "NBFS-FIRST",
                     games, budget_s, jobs)
    out = os.path.join(_HERE, "results", "census_dispatch_ab.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"budget_s": budget_s, "guard_tax": GUARD_TAX,
                   "owned_cap_s": OWNED_CAP_S,
                   "shipped": shipped, "nbfs_first": nbfs_first,
                   "wall_s": time.time() - t0}, fh, indent=2)
    print(f"\nwrote {out}  ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
