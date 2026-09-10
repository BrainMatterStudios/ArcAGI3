#!/usr/bin/env python3
"""triage_simulation.py — what would an external budget controller have actually BOUGHT?

The AUC result (early_predictability.py) says final depth is predictable from the first
40-60 actions at AUC 0.88-0.94, the top of the published range. That licenses the question
this script answers: on the runs we actually recorded, what does abandoning the predicted-
hopeless games and redistributing their budget do to the SCORE?

Two halves, and only one of them is measured:
  LOSS  (measured exactly) — levels a game cleared AFTER the decision point are forfeited.
        Scored properly: score is depth-weighted, min(weighted mean, completion-share cap),
        so losing a deep level costs far more than losing a shallow one.
  GAIN  (estimated, stated as such) — freed budget goes to survivors. We have one measured
        elasticity for that conversion: the KV10 wave bought +47 % calls for +19 % levels,
        so ~0.4. Anything here is only as good as that number.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ELASTICITY = 0.4          # measured: KV10 gave +47% calls -> +19% levels
DECISION_POINTS = (40, 60, 80)


def game_score(levels_cleared: int, n_levels: int) -> float:
    """The real rule at baseline parity: 100 * sum(cleared weights) / sum(all weights)."""
    if not n_levels:
        return 0.0
    return 100.0 * (levels_cleared * (levels_cleared + 1)) / (n_levels * (n_levels + 1))


def load():
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
                if not events:
                    continue
                runs.append({
                    "final_levels": g.get("levels_completed") or 0,
                    "n_levels": g.get("number_of_levels") or len(g.get("baselines") or []) or 7,
                    "total_actions": max((e.get("action_num") or 0) for e in events),
                    "events": events,
                })
    return runs


def levels_at(events, k):
    seen = [e.get("level") or 1 for e in events if (e.get("action_num") or 0) <= k]
    return (max(seen) - 1) if seen else 0


def main() -> int:
    runs = load()
    n = len(runs)
    base_score = sum(game_score(r["final_levels"], r["n_levels"]) for r in runs) / n
    base_lv = sum(r["final_levels"] for r in runs) / n
    print(f"runs {n} | baseline: {base_lv:.2f} levels/game, score {base_score:.2f}/100\n")
    print("POLICY: at action K, abandon every game that has cleared 0 levels so far.")
    print("        Freed actions are redistributed to the survivors.\n")
    print(f"{'K':>4} {'abandoned':>10} {'levels lost':>12} {'score lost':>11} "
          f"{'budget freed':>13} {'survivor gain':>14} {'NET score':>10}")
    for k in DECISION_POINTS:
        abandoned = [r for r in runs if levels_at(r["events"], k) == 0]
        survivors = [r for r in runs if levels_at(r["events"], k) > 0]
        if not abandoned or not survivors:
            continue
        # LOSS: measured. abandoned games keep only what they had cleared by K (which is 0).
        lost_score = sum(game_score(r["final_levels"], r["n_levels"]) for r in abandoned) / n
        lost_lv = sum(r["final_levels"] for r in abandoned) / n
        # BUDGET FREED: the actions those games would have spent after K
        freed = sum(max(0, r["total_actions"] - k) for r in abandoned)
        surv_actions = sum(r["total_actions"] for r in survivors)
        uplift = (freed / surv_actions) if surv_actions else 0.0
        # GAIN: estimated via the measured elasticity, applied to survivors' level counts
        gain_score = 0.0
        for r in survivors:
            lv_new = r["final_levels"] * (1 + ELASTICITY * uplift)
            gain_score += game_score(int(lv_new), r["n_levels"]) + \
                (lv_new - int(lv_new)) * (game_score(int(lv_new) + 1, r["n_levels"])
                                          - game_score(int(lv_new), r["n_levels"])) \
                - game_score(r["final_levels"], r["n_levels"])
        gain_score /= n
        print(f"{k:>4} {len(abandoned):>10} {lost_lv:>12.3f} {lost_score:>11.2f} "
              f"{f'+{100*uplift:.0f}%':>13} {gain_score:>14.2f} {gain_score - lost_score:>+10.2f}")
    print()
    print("The LOSS column is measured exactly from the recorded runs.")
    print(f"The GAIN column assumes the single measured elasticity ({ELASTICITY}); it is an estimate,")
    print("and it is the whole basis of the result, so treat a small positive NET as noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
