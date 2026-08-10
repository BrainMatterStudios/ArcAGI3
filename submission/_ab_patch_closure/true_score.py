"""The TRUE scored objective, aggregated from a wave result.

WHY THIS MODULE EXISTS
    Every A/B verdict this campaign recorded was decided on an unweighted,
    ft09-excluded level COUNT. Measured across the eight banked waves, that
    statistic correlates with the actual objective at **r = -0.009** — it is not
    a biased proxy, it is noise. The wave with the most levels (closure
    candidate, 17) had the LOWEST true score (0.313); base with 11 levels scored
    1.475.

    Meanwhile `pc_driver.pc_env_score` has been computing the real objective per
    row all along and storing it at `rows[].score`, under a header reading
    "informative only; the classifier reads levels".

    This module aggregates that stored value the way the leaderboard does:
        per game  = MAX over clones      (scorecard.py:239-241)
        per wave  = MEAN over games      (scorecard.py:613-618)

DELIBERATELY NOT DONE HERE
    This does not change any ADVANCE/STOP threshold. New bars must be
    pre-registered before a wave runs, not invented afterwards on data already
    seen — that is precisely how the verdicts this module exists to correct were
    produced. The job here is to make the real number impossible to omit from a
    report; choosing bars on it is a separate, pre-registered decision.
"""
from __future__ import annotations

from typing import Any


def per_game_true_score(result: dict[str, Any]) -> dict[str, float]:
    """Best-over-clones true score for each source game in a wave result."""
    best: dict[str, float] = {}
    for row in (result.get("rows") or []):
        game = row.get("source_game")
        score = row.get("score")
        if game is None or score is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if value > best.get(game, float("-inf")):
            best[game] = value
    return best


def true_score_metrics(result: dict[str, Any], excluded: tuple[str, ...] = ()) -> dict[str, Any]:
    """The objective-aligned metrics block to embed in any screen report.

    `excluded` mirrors whatever the legacy level statistic excludes, so a reader
    can see BOTH the honest all-games number and the one comparable to the old
    verdicts. ft09 was excluded from the level count on a stated rationale
    ("8 levels") that is factually wrong — metadata and every banked row say 6.
    """
    best = per_game_true_score(result)
    if not best:
        return {
            "true_score_all_games": None,
            "true_score_excl": None,
            "true_score_n_games": 0,
            "true_score_per_game": {},
            "note": "no rows carried a stored score — cannot compute the objective",
        }
    all_games = sorted(best)
    kept = [g for g in all_games if g not in excluded]
    mean_all = sum(best[g] for g in all_games) / len(all_games)
    mean_kept = (sum(best[g] for g in kept) / len(kept)) if kept else None
    top = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "true_score_all_games": round(mean_all, 4),
        "true_score_excl": None if mean_kept is None else round(mean_kept, 4),
        "true_score_excluded_games": list(excluded),
        "true_score_n_games": len(all_games),
        "true_score_top_contributors": [[g, round(v, 3)] for g, v in top],
        "true_score_per_game": {g: round(best[g], 4) for g in all_games},
        "objective": ("min(115, 100*(baseline/actions)^2) weighted by (level_index+1), "
                      "capped by completion share; per game MAX over clones, "
                      "per wave MEAN over games"),
        "warning": ("levels-excl-ft09 correlates with this at r=-0.009 across the eight "
                    "banked waves — do not read a verdict off the level count"),
    }
