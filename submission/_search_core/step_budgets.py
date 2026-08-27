"""step_budgets — the per-level move limit the games impose, and what it means.

FINDING (2026-08-27). 10 of the 25 public games carry a per-level
``StepCounter`` in their level data, and the engine renders it as the edge bar
the prompt warns the model not to click on:

    wa30.py:794  render_interface  -> a 64-wide bar of current/total steps
    wa30.py:784  pfakmupgbr        -> decrements once per move; at 0 the
                                      level's attempt is over

So on those games a level is not "solve it however long it takes" — it is
"solve it in at most B moves", and B is READABLE FROM THE FRAME.

Two consequences the search core does not currently exploit:

1. **B is a hard depth bound.** Any node deeper than B cannot lead to a scored
   solution on that level. Under the reset-replay cost model, reaching depth d
   costs 1 reset + d replay actions, so pruning above B removes the most
   expensive nodes in the tree. Today the core learns a volatility mask that
   MASKS the bar out as HUD and then searches past the bound it encodes.

2. **The human baselines are multi-attempt.** On wa30, 5 of 9 levels have a
   human baseline LARGER than the level's own budget (L3: baseline 183, budget
   100). A single attempt cannot exceed the budget, so the published baseline
   counts several attempts. That resolves the wa30 specialist's open note
   ("L3's optimal plan (169 acts) exceeds the level's step budget"): a
   within-budget solution must exist, and 169 is our planner being ~1.7x
   worse than the level allows, not the level being unsolvable.

This module reads the budgets straight from the engine (no frame parsing, no
guessing) so the search core and the specialists can take them as bounds.

Usage:  python step_budgets.py
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Level-data keys that carry a per-level move limit, in priority order.
BUDGET_KEYS = ("StepCounter", "MaxSteps", "steps")


def budgets_for(game, max_levels: int = 20) -> list[int | None]:
    """Per-level move limit, or None where the level does not impose one."""
    out: list[int | None] = []
    idx = 0
    while idx < max_levels:
        try:
            game.set_level(idx)
        except Exception:  # noqa: BLE001 — past the last level
            break
        val = None
        for key in BUDGET_KEYS:
            try:
                v = game.current_level.get_data(key)
            except Exception:  # noqa: BLE001 — key absent on this game
                v = None
            if isinstance(v, int) and v > 0:
                val = v
                break
        out.append(val)
        idx += 1
    return out


def main() -> int:
    from arc_agi import Arcade, OperationMode

    envdir = os.path.join(ROOT, "environment_files")
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=envdir)
    base = {e.game_id.split("-")[0]: e.baseline_actions for e in arc.get_environments()}

    rows = []
    for stem in sorted(os.listdir(envdir)):
        try:
            env = arc.make(stem)
            env.reset()
            b = budgets_for(env._game)  # noqa: SLF001 — engine object by design
        except Exception as exc:  # noqa: BLE001
            print(f"{stem}: SKIP {type(exc).__name__}")
            continue
        if not any(x is not None for x in b):
            continue
        rows.append((stem, b, base.get(stem, [])))

    print(f"games with a per-level move budget: {len(rows)}/25\n")
    print(f"{'game':6s} {'lvl':>4} {'budget':>7} {'human':>7} {'ratio':>7}  read")
    tight = multi = 0
    for stem, b, hb in rows:
        for i, (bud, hum) in enumerate(zip(b, hb), start=1):
            if bud is None:
                continue
            if not hum:
                print(f"{stem:6s} {i:>4} {bud:>7} {'—':>7} {'—':>7}")
                continue
            ratio = hum / bud
            note = ""
            if ratio > 1.0:
                note = "human baseline > budget => baseline spans RETRIES"
                multi += 1
            elif ratio > 0.8:
                note = "tight"
                tight += 1
            print(f"{stem:6s} {i:>4} {bud:>7} {hum:>7} {ratio:>7.2f}  {note}")
    print(f"\nlevels whose human baseline EXCEEDS the level budget: {multi}")
    print(f"levels where the human uses >80% of the budget       : {tight}")
    print("\nUSE: budget B is a hard depth bound for that level. Under reset-replay,"
          "\nreaching depth d costs 1 reset + d replay actions, so pruning d > B"
          "\nremoves the most expensive nodes in the tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
