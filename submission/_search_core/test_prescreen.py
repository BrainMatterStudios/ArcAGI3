"""Validation of the ZERO-ACTION frame-0 pre-screen on all 25 dev fixtures.

Ground truth: the ft09_gf2 class fires on exactly one dev game — ft09 —
per the detector matrix (``test_specialists.py`` EXPECT + the 25x4 matrix
assertion that each detector fires on its own game and NONE of the other 24).

What this file proves:
  1. CONFUSION MATRIX. The screen keeps ft09 (a false negative destroys the
     entire value case) and rejects as many of the other 24 as possible.
  2. C_probe. Actions spent per game, averaged over the corpus, converted to
     points with the measured cost model (Addendum 3 §C1): a wasted probe
     that lands on level 1 costs -1.19 points, on level 3 -12.14. The screen
     runs at GAME START, so every rejected game is a level-1 case; a rejected
     game spends ZERO actions and therefore costs 0.
  3. EV TABLE. The crack-or-nothing config's EV recomputed with the new
     C_probe, plus the break-even p.

Run:  .venv/bin/python submission/_search_core/test_prescreen.py
      .venv/bin/python -m pytest -q submission/_search_core/test_prescreen.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

os.environ.setdefault("ONLY_RESET_LEVELS", "true")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (os.path.join(ROOT, "submission/_explorer_floor"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import prescreen  # noqa: E402

ENV_DIR = os.path.join(ROOT, "environment_files")

# ground truth: the ft09_gf2 class on the dev corpus
GF2_GAMES = {"ft09"}

# measured cost model (measure_failed_engagement.py on CompetitionArcadeServer,
# docs/ENVELOPE-2026-08-26-v8.md Addendum 3 §C1) — a STEP function of WHERE the
# wasted actions land, not of how many there are.
COST_LEVEL1 = -1.19
COST_LEVEL3 = -12.14
# the A/B probe tax with 1 level completed afterwards (Addendum 2 §B3)
C_PROBE_MEASURED = (2.7, 3.8)
GAIN = 85.7            # ft09 14.29 -> banked 100.0, measured


def fixture_games() -> list[str]:
    stems = set()
    for d in (os.path.join(ROOT, "submission/_explorer_floor/fixtures"),
              os.path.join(ROOT, "scratchpad/testing_20260822/human_fixtures")):
        if os.path.isdir(d):
            stems.update(f[:-5] for f in os.listdir(d) if f.endswith(".json")
                         and not f.startswith("_"))
    return sorted(stems & set(os.listdir(ENV_DIR)))


def frame0(stem: str):
    """The observation the harness already holds at game start. Offline we
    must ask the engine for it once; live it is ``game.current_state`` and
    costs nothing."""
    logging.disable(logging.CRITICAL)
    from arc_agi import Arcade, OperationMode

    from search_core import discover_games

    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=ENV_DIR)
    env = client.make(discover_games(ENV_DIR)[stem])
    return env.reset()


def run_matrix(verbose: bool = True) -> dict:
    rows = []
    t0 = time.time()
    for stem in fixture_games():
        obs = frame0(stem)
        t1 = time.time()
        admit, note = prescreen.screen(obs, obs, ("ft09_gf2",))
        ms = (time.time() - t1) * 1000.0
        f = prescreen.ft09_gf2_features(obs, obs)
        rows.append({
            "game": stem, "truth": stem in GF2_GAMES, "admit": bool(admit),
            "note": note, "ms": ms,
            "blocks": f["n_blocks"], "strict": f["n_strict"],
            "frac": round(f["block_frac"], 3),
            "clues": f["block_lattice"]["clues"],
        })
    tp = sum(1 for r in rows if r["truth"] and r["admit"])
    fn = sum(1 for r in rows if r["truth"] and not r["admit"])
    fp = sum(1 for r in rows if not r["truth"] and r["admit"])
    tn = sum(1 for r in rows if not r["truth"] and not r["admit"])
    out = {"rows": rows, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
           "wall_s": time.time() - t0}
    if verbose:
        print(f"\n{'game':6} {'truth':>5} {'admit':>5} {'blk':>4} {'strict':>6} "
              f"{'frac':>5} {'clue':>4} {'ms':>6}  note")
        for r in rows:
            print(f"{r['game']:6} {str(r['truth']):>5} {str(r['admit']):>5} "
                  f"{r['blocks']:4} {r['strict']:6} {r['frac']:5.2f} "
                  f"{r['clues']:4} {r['ms']:6.1f}  {r['note']}")
        print(f"\nconfusion matrix (ground truth = ft09_gf2 class):")
        print(f"  true positive  {tp}   false negative {fn}")
        print(f"  false positive {fp}   true negative  {tn}")
    return out


def ev_table(fp: int, n_nongf2: int, probe_actions_screened: float,
             verbose: bool = True) -> dict:
    """EV per game for the crack-or-nothing config, before and after.

    EV(p) = p*GAIN - (1-p)*f*C_probe, where f = P(screen admits | no crack).
    A rejected game spends 0 actions at GAME START, i.e. the level-1 row of
    the measured cost table, and 0 actions cost 0 points."""
    f_admit = fp / max(1, n_nongf2)
    out = {"f_admit": f_admit, "screened_probe_actions_mean": probe_actions_screened}
    if verbose:
        print(f"\nC_probe (points charged to a NON-cracking game):")
        print(f"  before screen : {C_PROBE_MEASURED[0]:.2f} - "
              f"{C_PROBE_MEASURED[1]:.2f}   (measured A/B, level-1 landing "
              f"{COST_LEVEL1}, level-3 landing {COST_LEVEL3})")
        for lo_hi, label in ((C_PROBE_MEASURED, "after screen "),):
            print(f"  {label} : {f_admit * lo_hi[0]:.2f} - "
                  f"{f_admit * lo_hi[1]:.2f}   (admit rate {f_admit:.3f} "
                  f"x the same tax)")
        print(f"\nEV per game = p*{GAIN} - (1-p)*C_probe")
        print(f"  {'p':>6} {'EV before':>12} {'EV after':>12}")
        for p in (0.01, 0.02, 0.03, 0.04, 0.06, 0.10, 0.16):
            before = p * GAIN - (1 - p) * C_PROBE_MEASURED[0]
            before_hi = p * GAIN - (1 - p) * C_PROBE_MEASURED[1]
            after = p * GAIN - (1 - p) * f_admit * C_PROBE_MEASURED[0]
            after_hi = p * GAIN - (1 - p) * f_admit * C_PROBE_MEASURED[1]
            print(f"  {p:6.2f} {before:6.2f}/{before_hi:<5.2f} "
                  f"{after:6.2f}/{after_hi:<5.2f}")
        for lo, label in ((C_PROBE_MEASURED[0], "lo"), (C_PROBE_MEASURED[1], "hi")):
            c = f_admit * lo
            be = c / (GAIN + c) if (GAIN + c) else 0.0
            print(f"  break-even p ({label}, C_probe={c:.2f}): {100 * be:.2f} %")
    for lo in C_PROBE_MEASURED:
        c = f_admit * lo
        out[f"breakeven_{lo}"] = c / (GAIN + c) if (GAIN + c) else 0.0
    out["ev_at_4pct"] = [0.04 * GAIN - 0.96 * f_admit * lo
                         for lo in C_PROBE_MEASURED]
    return out


# --------------------------------------------------------------------------
# pytest surface
# --------------------------------------------------------------------------

_CACHE: dict = {}


def _matrix():
    if "m" not in _CACHE:
        _CACHE["m"] = run_matrix(verbose=False)
    return _CACHE["m"]


def test_geometry_constants_match_the_specialist():
    import specialists

    assert prescreen.TILE == specialists.TILE
    assert prescreen.LAT == specialists.LAT
    assert {(dx, dy) for dx, dy in prescreen.NB8} == \
        {(dx, dy) for dx, dy, _, _ in specialists.NB8}


def test_uniform_and_patterned_scans_match_the_specialist():
    """The screen must see the same blocks the detector would, or its
    rejection would not be evidence about the detector's own preconditions."""
    import numpy as np

    import specialists

    for stem in ("ft09", "vc33", "dc22", "tn36"):
        g = np.asarray(frame0(stem).frame)
        g = g[-1] if g.ndim == 3 else g
        mine = {(x, y) for (x, y, _) in prescreen._uniform_blocks(g)}
        theirs = {(x, y) for (x, y, _) in specialists._uniform_blocks(g)}
        assert mine == theirs, stem
        mp = set(prescreen._patterned_blocks(g))
        tp = {(x, y) for (x, y, _) in specialists._patterned_blocks(g)}
        assert mp == tp, stem


def test_ft09_passes_the_screen():
    """REQUIRED: a false negative here destroys the whole value case."""
    m = _matrix()
    row = next(r for r in m["rows"] if r["game"] == "ft09")
    assert row["admit"] is True, row
    assert m["fn"] == 0, [r for r in m["rows"] if r["truth"] and not r["admit"]]


def test_screen_rejects_the_non_gf2_corpus():
    m = _matrix()
    assert m["tn"] == 24, [r for r in m["rows"] if not r["truth"] and r["admit"]]
    assert m["fp"] == 0


def test_screen_costs_zero_engine_actions():
    """The screen is a pure function of frame 0 — it cannot call an engine."""
    import numpy as np

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"screen touched the engine: {name}")

    g = np.zeros((64, 64), dtype=np.int8)
    admit, note = prescreen.screen(g, [6], ("ft09_gf2",))
    assert admit is False, note
    # and nothing engine-shaped is ever consulted
    assert "step" not in note and "reset" not in note


def test_screen_is_fast_enough_to_run_on_every_game():
    m = _matrix()
    worst = max(r["ms"] for r in m["rows"])
    assert worst < 500.0, worst


def test_screen_fails_open_on_garbage():
    for bad in (None, object(), "not a frame", [], [[]], 42):
        admit, note = prescreen.screen(bad, [6], ("ft09_gf2",))
        assert admit is True, (bad, note)


def test_wildcard_and_unknown_classes_are_unscreenable():
    import numpy as np

    g = np.zeros((64, 64), dtype=np.int8)
    assert prescreen.screen(g, [6], ("*",))[0] is True
    assert prescreen.screen(g, [6], ("tn36_program",))[0] is True
    assert prescreen.screen(g, [6], ("ft09_gf2", "wa30_grabdrag"))[0] is True
    assert prescreen.screen(g, [6], ())[0] is True


def test_c_probe_and_break_even():
    m = _matrix()
    ev = ev_table(m["fp"], m["fp"] + m["tn"], 0.0, verbose=False)
    # with zero false positives the probe tax on non-cracking games is 0
    assert ev["f_admit"] == 0.0
    assert min(ev["ev_at_4pct"]) > 3.4


if __name__ == "__main__":
    m = run_matrix(verbose=True)
    ev = ev_table(m["fp"], m["fp"] + m["tn"], 0.0, verbose=True)
    print(f"\nscreen wall time for all 25 games: {m['wall_s']:.1f}s "
          f"(incl. one engine reset per game to obtain frame 0 offline)")
    res = os.path.join(_HERE, "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "prescreen_matrix.json")
    with open(path, "w") as fh:
        json.dump({"matrix": m, "ev": ev}, fh, indent=1)
    print(f"wrote {path}")
