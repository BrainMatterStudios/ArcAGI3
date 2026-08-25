"""Specialist-tier tests (stage 6).

1. DETECTOR MATRIX — every detector against all 25 fixtures' warmed-up
   roots: each must fire on its own game and on NONE of the other 24 (a
   wrong fire burns a real lane budget slice, worse than no specialist).
2. END-TO-END — each specialist solves its dev game through the SearchCore
   portfolio (engine-verified level-ups via the backend handle chain).

Run:  .venv/bin/python submission/_search_core/test_specialists.py matrix
      .venv/bin/python submission/_search_core/test_specialists.py e2e
      .venv/bin/python submission/_search_core/test_specialists.py   (both)
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ["ONLY_RESET_LEVELS"] = "true"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ROOT = os.path.dirname(os.path.dirname(_HERE))

EXPECT = {
    "ft09": "ft09_gf2",
    "tn36": "tn36_program",
    "sc25": "sc25_glyph",
    "wa30": "wa30_grabdrag",
}

# engine-verified floors (2026-08-25 prototypes): ft09 clue-solver L1-L4,
# tn36 register enumeration L1, sc25 glyph-cast+BFS L1, wa30 A* L1-L2
E2E_MIN_LEVELS = {"ft09": 4, "tn36": 1, "sc25": 1, "wa30": 2}
E2E_BUDGET = {"ft09": 420, "tn36": 420, "sc25": 420, "wa30": 900}


def make_core(stem: str):
    import logging

    logging.disable(logging.CRITICAL)
    from search_core import SearchCore, discover_games

    from arc_agi import Arcade, OperationMode

    environments_dir = os.path.join(ROOT, "environment_files")
    games = discover_games(environments_dir)
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=environments_dir)
    env = client.make(games[stem])
    core = SearchCore(env, backend="snapshot")
    core.warmup_and_freeze()
    return core


def run_matrix() -> bool:
    import specialists

    from search_core import ROOT as _sc_root  # noqa: F401

    stems = sorted(os.listdir(os.path.join(ROOT, "environment_files")))
    stems = [s for s in stems
             if os.path.isdir(os.path.join(ROOT, "environment_files", s))]
    names = [n for n, _ in specialists.DETECTORS]
    rows = []
    ok = True
    t_all = time.time()
    for stem in stems:
        t0 = time.time()
        try:
            core = make_core(stem)
            verdicts = specialists.detect_matrix(core)
        except Exception as e:  # noqa: BLE001
            verdicts = {n: False for n in names}
            print(f"{stem}: ERROR {e!r}")
        row = {"game": stem, **verdicts,
               "wall": round(time.time() - t0, 1)}
        rows.append(row)
        expected = EXPECT.get(stem)
        for n in names:
            want = (n == expected)
            if verdicts.get(n, False) != want:
                ok = False
                print(f"  MISMATCH {stem}/{n}: got {verdicts.get(n)} "
                      f"want {want}")
        fired = [n for n in names if verdicts.get(n)]
        print(f"{stem}: {fired or '-'} ({row['wall']}s)", flush=True)
    # render matrix
    print("\n| game | " + " | ".join(n.split('_')[0] for n in names) + " |")
    print("|------|" + "|".join(["------"] * len(names)) + "|")
    for r in rows:
        cells = " | ".join("X" if r.get(n) else "." for n in names)
        print(f"| {r['game']} | {cells} |")
    out = os.path.join(_HERE, "results",
                       f"detector_matrix_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nmatrix {'PASS' if ok else 'FAIL'} "
          f"({round(time.time() - t_all, 1)}s) -> {out}")
    return ok


def run_e2e() -> bool:
    from search_core import run_game

    ok = True
    for stem, min_levels in E2E_MIN_LEVELS.items():
        t0 = time.time()
        res = run_game(stem, E2E_BUDGET[stem], backend="snapshot",
                       algo="portfolio")
        spec_levels = [L for L in res["levels"]
                       if str(L.get("algo", "")).startswith("spec:")
                       and L.get("solved")]
        # solved level entries only exist for FAILED lanes; solved ones are
        # recorded via bank() without algo -> count levels_won instead and
        # require the specialist lane to have been detected
        won = res["levels_won"]
        det = res.get("specialist")
        line = (f"{stem}: specialist={det} levels={won} "
                f"cracked={res.get('cracked')} wall={res['wall']}s "
                f"actions={res.get('actions_spent')}")
        if det != EXPECT[stem] or won < min_levels:
            ok = False
            line += f"   FAIL (need detect={EXPECT[stem]}, >={min_levels})"
        print(line, flush=True)
        _ = spec_levels, t0
    print(f"e2e {'PASS' if ok else 'FAIL'}")
    return ok


# pytest entry points -------------------------------------------------------

def test_detector_matrix():
    assert run_matrix()


def test_e2e_specialists():
    assert run_e2e()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    good = True
    if which in ("matrix", "both"):
        good &= run_matrix()
    if which in ("e2e", "both"):
        good &= run_e2e()
    sys.exit(0 if good else 1)
