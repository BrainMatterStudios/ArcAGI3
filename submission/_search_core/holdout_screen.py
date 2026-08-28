"""Does the ft09_gf2 frame-0 screen generalise OFF the development corpus?

WHY THIS EXISTS (2026-08-28, after the v8 null draw).

The 08-28 crack-or-nothing arm scored 1.65 — a null, zero cracks banked. The
pre-registered reading says a null is evidence about hidden-set composition
rather than against the mechanism, but that reading only holds if we can tell
the two worlds apart:

  world A  the screen never fired on a hidden game  -> no ft09-class game was
           drawn; the ticket simply did not come up, and the lane is untested.
  world B  the screen fired and the solver did not crack -> the detector
           generalises but the specialist does not; the lane is dead.

**Neither is observable from a scored rerun.** Kaggle exposes no output files
for a competition rerun (``kaggle competitions`` has no submission-output
endpoint; ``logs`` is simulation-episodes only), so nothing the notebook writes
in the scored run can ever be read back. Adding logging to the shipped path
cannot answer this question. It has to be answered offline, and it can be.

WHAT THIS MEASURES, AND WHAT IT DOES NOT.

The screen's headline number — 1 TP / 0 FN / 0 FP — was measured on the 25
public games, **every one of which it was developed against**. That number is
therefore an in-sample fit, not a generalisation estimate. This sweep re-runs
the identical screen over ``scratchpad/holdout_arcint`` — 13 arc-interactive
games that took no part in the screen's design.

  * MEASURABLE HERE: **specificity on unseen games** (the false-positive rate).
    Every admitted holdout game is a game the live arm would have paid the full
    ~2.7-3.8 point probe tax on for nothing. This is the number that decides
    whether the arm's "measured zero floor" survives contact with unseen games,
    and it is the dominant term in EV (it is paid on ~96% of games).

  * NOT MEASURABLE HERE unless a holdout game happens to be of the class: the
    **true-positive rate on an unseen same-class game**. Our corpus contains
    exactly one GF2 game (ft09) and the screen was built on it. If no holdout
    game admits, the TPR stays unmeasured and the handoff's honest caveat
    stands untouched — this sweep cannot retire it.

So: an admit here is either a false positive (bad: the floor is not zero) or a
same-class discovery (very good: the first TPR datapoint we could ever get).
Both outcomes are worth more than another slot.

MARGINS. A binary verdict wastes most of the signal. For every game this also
reports how far it sat from admission on each branch, so a corpus that declines
everywhere by a hair reads very differently from one that declines by a mile.

THE FALSE-NEGATIVE CHECK (``--detect``). Specificity is the cheap half. The
half that can destroy the value case is a **false negative**: a genuinely
engageable game the screen throws away for free. ``--detect`` runs the real
``specialists.detect_matrix`` over the same 13 games and cross-tabs it against
the screen verdict. Offline the detector's ~40 confirmation clicks are free, so
this is still zero-slot. Three outcomes:

  detector fires, screen ADMITTED  -> the first out-of-sample TPR datapoint.
  detector fires, screen DECLINED  -> FALSE NEGATIVE. The screen must be
                                      loosened before this arm flies again.
  detector never fires             -> no same-class game here; TPR stays
                                      unmeasured and the caveat stands.

Run:  ONLY_RESET_LEVELS=true .venv/bin/python \
          submission/_search_core/holdout_screen.py [--detect]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

os.environ.setdefault("ONLY_RESET_LEVELS", "true")  # law 4: before arcengine

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
ROOT = os.path.dirname(os.path.dirname(_HERE))

import prescreen  # noqa: E402

DEV_DIR = os.path.join(ROOT, "environment_files")
HOLDOUT_DIR = os.path.join(ROOT, "scratchpad/holdout_arcint/environment_files")
RESULTS = os.path.join(_HERE, "results")

# the one in-sample positive, for the contrast row
GF2_GAMES = {"ft09"}


def frame0(env_dir: str, stem: str):
    """The observation the harness already holds at game start.

    Offline we must ask the engine for it once; live it is
    ``game.current_state`` and costs zero engine actions."""
    logging.disable(logging.CRITICAL)
    from arc_agi import Arcade, OperationMode

    from search_core import discover_games

    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=env_dir)
    env = client.make(discover_games(env_dir)[stem])
    return env.reset()


def margins(f: dict) -> dict:
    """How far this frame sits from each admission branch.

    Negative = short of the bar by that much. ``branch`` is the *closest*
    approach across the two branches, in units of the binding shortfall."""
    if not f.get("usable"):
        return {"a_strict": None, "a_lat": None,
                "b_blocks": None, "b_lat": None, "b_frac": None,
                "b_clues": None, "closest": None}
    sl, bl = f["strict_lattice"], f["block_lattice"]
    a = {
        "a_strict": f["n_strict"] - prescreen.MIN_STRICT,
        "a_lat": sl["n"] - prescreen.MIN_STRICT_LATTICE,
    }
    b = {
        "b_blocks": f["n_blocks"] - prescreen.FRAC_MIN_TILES,
        "b_lat": bl["n"] - prescreen.FRAC_MIN_LATTICE,
        "b_frac": round(f["block_frac"] - prescreen.FRAC_MIN, 3),
        "b_clues": bl["clues"] - 1,
    }
    # a branch fires only when every one of its terms is >= 0; the branch's
    # distance from firing is therefore its WORST term.
    out = dict(a, **b)
    out["closest"] = max(min(a.values()), min(b.values()))
    return out


def sweep(env_dir: str, label: str, verbose: bool = True) -> list[dict]:
    from search_core import discover_games

    rows = []
    for stem in sorted(discover_games(env_dir)):
        t1 = time.time()
        obs = frame0(env_dir, stem)
        admit, note = prescreen.screen(obs, obs, ("ft09_gf2",))
        f = prescreen.ft09_gf2_features(obs, obs)
        m = margins(f)
        rows.append({
            "corpus": label, "game": stem, "admit": bool(admit), "note": note,
            "in_sample": label == "dev",
            "usable": bool(f.get("usable")),
            "click": bool(f.get("click")) if f.get("usable") else None,
            "n_blocks": f.get("n_blocks"), "n_strict": f.get("n_strict"),
            "block_frac": round(f["block_frac"], 3) if f.get("usable") else None,
            "clues": f["block_lattice"]["clues"] if f.get("usable") else None,
            "margins": m, "ms": (time.time() - t1) * 1000.0,
        })
    if verbose:
        print(f"\n=== {label} corpus ({len(rows)} games) ===")
        print(f"{'game':6} {'admit':>5} {'blk':>4} {'strict':>6} {'frac':>5} "
              f"{'clue':>4} {'closest':>7}  note")
        for r in rows:
            c = r["margins"]["closest"]
            print(f"{r['game']:6} {str(r['admit']):>5} "
                  f"{str(r['n_blocks']):>4} {str(r['n_strict']):>6} "
                  f"{str(r['block_frac']):>5} {str(r['clues']):>4} "
                  f"{str(c):>7}  {r['note']}")
    return rows


def detect_sweep(env_dir: str, verbose: bool = True) -> dict[str, dict]:
    """The REAL detectors on unseen games. Offline the confirmation clicks are
    free, so this stays zero-slot. Answers the false-negative question that the
    frame-0 screen alone cannot."""
    import logging

    logging.disable(logging.CRITICAL)
    from search_core import SearchCore, discover_games

    from arc_agi import Arcade, OperationMode

    import specialists

    out: dict[str, dict] = {}
    for stem in sorted(discover_games(env_dir)):
        t0 = time.time()
        try:
            client = Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=env_dir)
            core = SearchCore(client.make(discover_games(env_dir)[stem]),
                              backend="snapshot")
            core.warmup_and_freeze()
            v = specialists.detect_matrix(core)
        except Exception as exc:  # noqa: BLE001
            v = {n: False for n, _ in specialists.DETECTORS}
            v["_error"] = f"{type(exc).__name__}: {exc}"
        v["_s"] = round(time.time() - t0, 1)
        out[stem] = v
        if verbose:
            hits = [k for k, val in v.items()
                    if not k.startswith("_") and val]
            print(f"  {stem:6} {'HIT ' + ','.join(hits) if hits else 'none':28}"
                  f" {v['_s']:6.1f}s"
                  + (f"  [{v['_error']}]" if "_error" in v else ""))
    return out


def main() -> int:
    t0 = time.time()
    want_detect = "--detect" in sys.argv
    dev = sweep(DEV_DIR, "dev")
    hold = sweep(HOLDOUT_DIR, "holdout")

    admits = [r for r in hold if r["admit"]]
    n = len(hold)
    fpr = len(admits) / n if n else float("nan")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"dev corpus (IN-SAMPLE, screen was built on these): "
          f"{sum(1 for r in dev if r['admit'])}/{len(dev)} admitted "
          f"— expected exactly {sorted(GF2_GAMES)}")
    print(f"holdout corpus (UNSEEN): {len(admits)}/{n} admitted "
          f"=> false-positive rate {fpr:.1%}" if not admits else
          f"holdout corpus (UNSEEN): {len(admits)}/{n} admitted "
          f"=> {[r['game'] for r in admits]}")
    if not admits:
        print("\n  Screen declined every unseen game at ZERO engine actions.")
        print("  => specificity holds off the development corpus; the arm's")
        print("     zero floor survives contact with games it never saw.")
        print("  => TPR on an unseen same-class game REMAINS UNMEASURED:")
        print("     no holdout game is of the class, so this sweep cannot")
        print("     distinguish world A from world B. The caveat stands.")
    else:
        print("\n  Admitted games are EITHER false positives (the floor is not")
        print("  zero on unseen games) OR the first same-class games we have")
        print("  ever seen outside ft09. Next step is to run detect_ft09 on")
        print("  them: a detector hit makes this the TPR datapoint the lane")
        print("  has never had; a miss makes it a measured false positive.")

    margin_vals = [r["margins"]["closest"] for r in hold
                   if r["margins"]["closest"] is not None]
    if margin_vals:
        print(f"\n  holdout closest-approach to admission: "
              f"max {max(margin_vals)}, median "
              f"{sorted(margin_vals)[len(margin_vals) // 2]} "
              f"(0 = would have admitted; more negative = safer)")

    det = None
    if want_detect:
        print("\n" + "=" * 72)
        print("FALSE-NEGATIVE CHECK — real detectors on the 13 unseen games")
        print("=" * 72)
        det = detect_sweep(HOLDOUT_DIR)
        screen_of = {r["game"]: r["admit"] for r in hold}
        fn_games, tp_games = [], []
        for stem, v in det.items():
            fired = [k for k, val in v.items()
                     if not k.startswith("_") and val]
            if not fired:
                continue
            (tp_games if screen_of.get(stem) else fn_games).append(
                (stem, fired))
        if fn_games:
            print("\n  *** FALSE NEGATIVE — screen declined an engageable "
                  "game ***")
            for stem, fired in fn_games:
                print(f"      {stem}: detector(s) {fired} but screen DECLINED")
            print("  The screen must be loosened before this arm flies again;")
            print("  a false negative destroys the entire value case.")
        elif tp_games:
            print("\n  Out-of-sample TRUE POSITIVE(s): "
                  f"{[s for s, _ in tp_games]}")
            print("  This is the TPR datapoint the lane has never had.")
        else:
            print("\n  No detector fires on ANY of the 13 unseen games.")
            print("  => no same-class game in the holdout corpus, so the")
            print("     screen had nothing to be wrong about in either")
            print("     direction. TPR remains unmeasured; world A vs world B")
            print("     stays undecidable from this corpus.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "holdout_screen.json")
    with open(out, "w") as fh:
        json.dump({"dev": dev, "holdout": hold,
                   "holdout_admitted": [r["game"] for r in admits],
                   "holdout_fpr": fpr, "detect": det,
                   "wall_s": time.time() - t0}, fh, indent=2)
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
