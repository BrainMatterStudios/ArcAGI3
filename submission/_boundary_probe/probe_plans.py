"""probe_plans — why patch 21's extra actions did not become levels.

Follow-up to docs/RESEARCH-2026-08-27 §9b, which measured the outcome: the
mandatory plan channel bought x1.40 actions/game on ft09 and cleared 25%
FEWER levels (36 -> 27 across 28 clones per arm).

The accounting of WHERE the extra actions went is already decisive:

                        base        struct
  actions total         2463        3565     (+1102)
  burned on the level
    it never cleared    1537        2504      (+967)

so 88% of every extra action the channel bought was spent on a level that was
never cleared. It did not buy progress; it bought deeper failure.

THIS probe asks the mechanism question: inside a committed plan, does the plan
stop being right after its first step? A plan of k actions is chosen from ONE
observation. If the board answers step 1 in a way the model did not expect,
steps 2..k are spent against a state it has never seen.

MEASURED, per plan-step position:
  - did the step change the board at all
  - was it an immediate repeat of the previous action
  - was it a direct reversal (UP after DOWN, LEFT after RIGHT)

If effectiveness decays with position, the fix is not shorter plans but CHEAP
REVALIDATION: keep the committed plan, abort it the moment an observed step
contradicts the prediction that justified it.

Only the struct arm emits per-step tables, so this reads that arm alone.

Usage:  python probe_plans.py
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The step table is a list of dicts inside the inline Result payload. Parsed
# with targeted regex rather than a literal evaluator: transcripts truncate
# long dicts mid-line, so a tolerant scan reads more of the corpus than a
# strict parse would, and never executes anything from the file.
STEPS_BLOCK = re.compile(r"'steps': \[(.*?)\](?:, '\w+':|\}\s*$)", re.S)
STEP_ENTRY = re.compile(
    r"\{'action': '([^']*)'.*?'board_changed': (True|False)", re.S)
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def kind(action: str) -> str:
    return action.split("(")[0].strip()


def plans_in(text: str) -> list[list[tuple[str, bool]]]:
    """Every parsed plan as [(action, board_changed), ...]."""
    plans = []
    for m in STEPS_BLOCK.finditer(text):
        steps = [(a, c == "True") for a, c in STEP_ENTRY.findall(m.group(1))]
        if steps:
            plans.append(steps)
    return plans


def analyse(paths: list[str]) -> dict:
    by_pos = defaultdict(lambda: [0, 0])          # position -> [changed, total]
    plans, repeats, reversals, steps_total = [], 0, 0, 0
    first_dud, wasted_after_dud = [], 0
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        for steps in plans_in(text):
            plans.append(len(steps))
            prev, dud_at = None, None
            for i, (a, changed) in enumerate(steps, start=1):
                steps_total += 1
                by_pos[i][1] += 1
                by_pos[i][0] += int(changed)
                if prev is not None:
                    if a == prev:
                        repeats += 1
                    elif kind(a) in OPPOSITE and OPPOSITE[kind(a)] == kind(prev):
                        reversals += 1
                if not changed and dud_at is None:
                    dud_at = i
                prev = a
            if dud_at is not None:
                first_dud.append(dud_at)
                wasted_after_dud += len(steps) - dud_at
    return dict(by_pos=by_pos, plans=plans, repeats=repeats, reversals=reversals,
                steps_total=steps_total, first_dud=first_dud,
                wasted_after_dud=wasted_after_dud)


def main() -> int:
    paths = sorted(glob.glob(os.path.join(
        ROOT, "scratchpad/ft09_ablation/ft09-struct/taaf_harness_artifacts/transcripts/*.txt")))
    if not paths:
        print("no struct transcripts found")
        return 1
    r = analyse(paths)
    pl = r["plans"]
    print(f"struct transcripts : {len(paths)}")
    print(f"plans parsed       : {len(pl)}   steps: {r['steps_total']}")
    if not pl:
        print("no per-step tables parsed — cannot answer.")
        return 2
    print(f"plan length        : median {statistics.median(pl):.0f}  mean {statistics.mean(pl):.1f}  max {max(pl)}")

    print("\n--- does a committed plan stay right after step 1? ---")
    print(f"{'position':>9} {'steps':>7} {'changed the board':>19}")
    rows = sorted(r["by_pos"].items())
    for pos, (ch, tot) in rows:
        if tot < 15 or pos > 12:
            continue
        print(f"{pos:>9} {tot:>7} {100*ch/tot:>18.1f}%  {'#' * round(20*ch/tot)}")
    head = r["by_pos"].get(1)
    tail = [(ch, tot) for pos, (ch, tot) in rows if pos >= 4]
    if head and tail and head[1]:
        t_ch = sum(c for c, _ in tail)
        t_tot = sum(t for _, t in tail)
        if t_tot:
            print(f"\n  step 1   : {100*head[0]/head[1]:.1f}% of steps changed the board")
            print(f"  steps 4+ : {100*t_ch/t_tot:.1f}%")
            print(f"  decay    : {100*(head[0]/head[1] - t_ch/t_tot):+.1f} points")

    print("\n--- what the wasted steps look like ---")
    st = r["steps_total"]
    print(f"  immediate repeats of the previous action : {r['repeats']:>5} / {st}  ({100*r['repeats']/st:.1f}%)")
    print(f"  direct reversals of the previous action  : {r['reversals']:>5} / {st}  ({100*r['reversals']/st:.1f}%)")
    fd = r["first_dud"]
    if fd:
        c = Counter(fd)
        print(f"  plans containing a no-op step            : {len(fd)} / {len(pl)}  ({100*len(fd)/len(pl):.1f}%)")
        print(f"  first no-op arrives at step (median)     : {statistics.median(fd):.0f}")
        print(f"    at step 1: {c.get(1,0)}   step 2: {c.get(2,0)}   step 3+: {sum(v for k,v in c.items() if k>=3)}")
        w = r["wasted_after_dud"]
        print(f"\n  steps executed AFTER a plan's first no-op: {w} of {st}  ({100*w/st:.1f}%)")
        print("  -> the budget an abort-on-contradiction rule reclaims, without")
        print("     shortening a single plan that was going well.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
