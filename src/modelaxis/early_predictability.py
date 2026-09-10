#!/usr/bin/env python3
"""early_predictability.py — is a game's final depth predictable from its first K actions?

WHY THIS EXISTS: the budget-controller family (ReD / CLEAR / ZEBRA / BAGEN) is the
best-evidenced idea in the 09-10 harness sweep, but every one of those results assumes
you can tell early which tasks are worth more budget. Published AUCs for that judgement
range from 0.86 (TextCraft) down to 0.59 (WebShop) and below 0.60 for deep research --
i.e. it is domain-dependent and OUR domain is unmeasured. This settles it offline on
recorded runs, at zero rig cost, BEFORE any harness work.

HARD CONSTRAINT ON THE FEATURES: `baseline_actions` is excluded from the eval API
(arc_agi/api.py:60), so anything baseline-relative is unusable live. Every feature here is
computable from what the agent can actually see at eval time.

The three published triage results also say the decision must NOT live in the model, so
this is a code-side signal by construction.

Usage: .venv/bin/python src/modelaxis/early_predictability.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGETS = (20, 40, 60, 80)          # actions elapsed when the decision would be taken
TARGETS = (2, 3)                    # predict "final levels_completed >= T"


def load_runs() -> list[dict]:
    runs = []
    for pat in ("offkaggle/results/*/results.json", "offkaggle/results/*/*/results.json"):
        for rj in glob.glob(str(REPO / pat)):
            try:
                res = json.loads(Path(rj).read_text())
            except json.JSONDecodeError:
                continue
            if res.get("dry_run"):
                continue
            art = Path(rj).parent / "artifacts"
            for g in res.get("games", []):
                if not isinstance(g, dict) or "run_stem" not in g:
                    continue
                ev = art / f"{g['run_stem']}_events.jsonl"
                if not ev.exists():
                    continue
                events = []
                for line in ev.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                if events:
                    runs.append({"game": g["run_stem"].split("-")[0],
                                 "final_levels": g.get("levels_completed") or 0,
                                 "events": events})
    return runs


def features_at(events: list[dict], k: int) -> dict | None:
    """Everything observable after k real actions. No baselines: they are hidden at eval."""
    seen = [e for e in events if (e.get("action_num") or 0) <= k]
    if len(seen) < 3:
        return None
    levels = [e.get("level") or 1 for e in seen]
    lvl_now = max(levels)
    # actions spent on the CURRENT level so far (the "am I stuck" signal)
    first_at_lvl = next((e.get("action_num") or 0 for e in seen if (e.get("level") or 1) == lvl_now), 0)
    return {
        "levels_cleared_by_k": lvl_now - 1,
        "actions_on_current_level": k - first_at_lvl,
        "game_overs": sum(1 for e in seen if str(e.get("state", "")).upper() == "GAME_OVER"),
        "distinct_actions": len({str(e.get("action_display") or "")[:12] for e in seen}),
    }


def auc(pairs: list[tuple[float, int]]) -> float:
    """Rank-based AUC. pairs = (score, label)."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> int:
    runs = load_runs()
    print(f"runs with event logs: {len(runs)}")
    print(f"final-level distribution: {dict(sorted(Counter(r['final_levels'] for r in runs).items()))}\n")

    for target in TARGETS:
        print(f"=== predicting final levels_completed >= {target} ===")
        print(f"{'after':>6} {'n':>5} {'base rate':>10} " +
              " ".join(f"{f:>26}" for f in ("levels_cleared_by_k", "actions_on_current_level")))
        for k in BUDGETS:
            rows = []
            for r in runs:
                f = features_at(r["events"], k)
                if f is None:
                    continue
                rows.append((f, 1 if r["final_levels"] >= target else 0))
            if not rows:
                continue
            base = sum(y for _f, y in rows) / len(rows)
            cells = []
            for feat, sign in (("levels_cleared_by_k", 1), ("actions_on_current_level", -1)):
                a = auc([(sign * f[feat], y) for f, y in rows])
                cells.append(f"{a:26.3f}")
            print(f"{k:>5}a {len(rows):>5} {base:>10.2f} " + " ".join(cells))
        print()
    print("READ: AUC 0.5 = the signal is worthless; the published range for this kind of")
    print("early-abort judgement is 0.59 (WebShop) to 0.86 (TextCraft). Below ~0.7 the")
    print("budget-controller family is not worth building on our data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
