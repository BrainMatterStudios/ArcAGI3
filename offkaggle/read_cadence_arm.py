#!/usr/bin/env python3
"""Evaluate the CADENCE arm against its pre-registration, mechanically.

Written 2026-09-09 while the wave was still running, BEFORE any of its data existed,
so the read cannot be tuned to the result. Gates are transcribed from
offkaggle/REGIME_WAVE_STATUS.md "PRE-REGISTERED — keith at CADENCE geometry".

Usage: .venv/bin/python offkaggle/read_cadence_arm.py <run_dir>
       (run_dir = the .../<stamp>-regime-keith directory holding results.json)
"""
import json
import statistics as st
import sys
from pathlib import Path

# ---- pre-registered constants (do not edit after the wave lands) -------------
POOLED_BASE_LEVELS = 39.33      # 25-game pooled base
POOLED_BASE_SD = 2.34
STEP, REDRAW_LO = 48, 45        # >=48 step candidate; 45-47 redraw; <=44 dead
E2E_MAX = 60.0                  # the turn budget the arm must get under
CALLS_PER_TURN_MIN = 1.5        # live base 1.02; conc-3 reference 1.66
CALLS_PER_GAME_MIN = 40         # guards the underfed-GPU confound
WALLS_TARGET = 3
# the 12 six-draw never-passed walls (REGIME_WAVE_STATUS.md:616-617)
WALLS = {"bp35": 2, "dc22": 2, "g50t": 2, "lf52": 2, "lp85": 6, "ls20": 2,
         "r11l": 3, "sb26": 2, "sp80": 2, "tn36": 3, "vc33": 4, "wa30": 2}


def main(run_dir):
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results.json").read_text())
    tel = {}
    tp = run_dir / "telemetry.json"
    if tp.exists():
        tel = json.loads(tp.read_text()).get("aggregate", {})
    games = res["games"]

    geo = res.get("geometry", {})
    print("=" * 78)
    print(f"CADENCE ARM READ — {run_dir.name}")
    print(f"  arm {res.get('arm')} | grafts {res.get('grafts', {}).get('installed')}"
          f" | conc {geo.get('concurrency')} | per_game_s {geo.get('max_runtime_s_per_game')}")
    print("=" * 78)

    # ---- ENGAGEMENT (checked FIRST; failure => VOID, not dead) ---------------
    _e = tel.get("client_e2e_s_pooled") or tel.get("e2e") or {}
    e2e = _e.get("median") or _e.get("mean")
    cpt = tel.get("calls_per_turn")
    cpg = tel.get("calls_per_game")
    checks = [
        ("median e2e per call < 60 s", e2e, lambda v: v < E2E_MAX, f"< {E2E_MAX}"),
        ("calls per turn >= 1.5", cpt, lambda v: v >= CALLS_PER_TURN_MIN, f">= {CALLS_PER_TURN_MIN}"),
        ("calls per game >= 40", cpg, lambda v: v >= CALLS_PER_GAME_MIN, f">= {CALLS_PER_GAME_MIN}"),
    ]
    print("\nENGAGEMENT GATE")
    engaged = True
    for name, val, ok, want in checks:
        if val is None:
            print(f"  ?? {name:<34} MISSING (want {want})")
            engaged = False
            continue
        good = ok(val)
        engaged &= good
        print(f"  {'PASS' if good else 'FAIL'} {name:<34} {val:>8.2f}  (want {want})")

    # ---- PRIMARY ------------------------------------------------------------
    levels = sum(g["levels_completed"] for g in games)
    n = len(games)
    z = (levels - POOLED_BASE_LEVELS) / POOLED_BASE_SD
    print(f"\nPRIMARY — levels on {n} games")
    print(f"  levels {levels}  vs pooled base {POOLED_BASE_LEVELS} (sd {POOLED_BASE_SD})"
          f"  = {levels - POOLED_BASE_LEVELS:+.2f} = {z:+.2f} sd")

    # ---- CO-PRIMARY: the 12 never-passed walls ------------------------------
    passed, present = [], []
    for g in games:
        stem = g["game_id"].split("-")[0]
        if stem in WALLS:
            present.append(stem)
            if g["levels_completed"] >= WALLS[stem]:
                passed.append(f"{stem} L{WALLS[stem]}")
    print(f"\nCO-PRIMARY — the 12 six-draw never-passed walls")
    print(f"  present in this wave: {len(present)}/12")
    print(f"  PASSED: {len(passed)}  (target >= {WALLS_TARGET})"
          + (f"  -> {', '.join(passed)}" if passed else ""))

    # ---- SAFETY / diagnosis -------------------------------------------------
    zero = sum(1 for g in games if g["levels_completed"] == 0)
    wall = [g.get("wallclock_s") for g in games if g.get("wallclock_s")]
    print("\nSAFETY / DIAGNOSIS")
    print(f"  levels/game        {levels / n:.2f}")
    print(f"  zero-level games   {zero} / {n}")
    print(f"  mean score/game    {st.mean(g['score'] for g in games):.2f}")
    print(f"  actions/game       {st.mean(sum(g.get('actions_per_level') or [0]) for g in games):.0f}")
    print(f"  wallclock/game     {st.mean(wall):.0f}s" if wall else "  wallclock/game  n/a")
    for k in ("yields_share", "no_tool_call_share", "actions_per_call", "retries_per_game"):
        if k in tel:
            print(f"  {k:<18} {tel[k]}")

    # ---- VERDICT ------------------------------------------------------------
    print("\n" + "-" * 78)
    if not engaged:
        print("VERDICT: **VOID** — the mechanism never ran. Not evidence about cadence.")
        print("         Pre-registered remedy: re-run at conc 4 or conc 3.")
    elif levels >= STEP:
        print(f"VERDICT: **STEP CANDIDATE** ({levels} >= {STEP}).")
        print("         Next per pre-registration: propose a live flight. AHMED'S GO REQUIRED.")
    elif levels >= REDRAW_LO:
        print(f"VERDICT: **POSITIVE, REDRAW** ({REDRAW_LO} <= {levels} <= {STEP - 1}).")
        print("         Counterbalanced redraw in a fresh session before any claim.")
    else:
        print(f"VERDICT: **DEAD by rule** ({levels} <= {STEP - 3}), engaged or not.")
        print("         This would be the sixth engaged-and-flat replication.")
    print(f"         Co-primary walls {len(passed)}/{WALLS_TARGET} required for a step claim.")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
