"""Regression tests for the objective-aligned screen metric.

These exist because the campaign spent seven weeks deciding A/B verdicts on a
statistic that correlates with the actual objective at r = -0.009. The tests below
lock three things: the aggregation matches the leaderboard's (max over clones,
mean over games), it reproduces an INDEPENDENT reimplementation on the real banked
waves, and the report can never silently omit it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from true_score import per_game_true_score, true_score_metrics  # noqa: E402

BANKED = REPO / "scratchpad/banked_waves_20260809"
WAVES = ["pc_base", "w2_base", "pc_cand", "w2_cand", "pkg", "struct", "struct2", "struct3"]


def _load(stem: str) -> dict:
    return json.loads((BANKED / f"{stem}.json").read_text())


def test_aggregation_is_max_over_clones():
    result = {"rows": [
        {"source_game": "aa11", "score": 1.0},
        {"source_game": "aa11", "score": 7.5},   # best clone must win
        {"source_game": "bb22", "score": 2.0},
    ]}
    assert per_game_true_score(result) == {"aa11": 7.5, "bb22": 2.0}
    m = true_score_metrics(result)
    assert m["true_score_all_games"] == pytest.approx((7.5 + 2.0) / 2)


def test_exclusion_reports_both_numbers():
    result = {"rows": [
        {"source_game": "ft09", "score": 14.286},
        {"source_game": "aa11", "score": 1.0},
    ]}
    m = true_score_metrics(result, excluded=("ft09",))
    assert m["true_score_all_games"] == pytest.approx((14.286 + 1.0) / 2)
    assert m["true_score_excl"] == pytest.approx(1.0)
    # the honest all-games number must always be present, not replaced
    assert m["true_score_excluded_games"] == ["ft09"]


def test_missing_scores_degrade_loudly_not_silently():
    m = true_score_metrics({"rows": [{"source_game": "aa11"}]})
    assert m["true_score_all_games"] is None
    assert "note" in m


@pytest.mark.skipif(not BANKED.is_dir(), reason="banked waves not present")
@pytest.mark.parametrize("stem", WAVES)
def test_matches_independent_reimplementation_on_real_waves(stem):
    """Cross-check against scripts/rescore_waves.py, which recomputes the objective
    from actions_per_level + baselines rather than reading rows[].score."""
    from rescore_waves import score_wave

    best, _, _, _, mismatches = score_wave(stem)
    assert mismatches == 0, "the independent rescorer disagrees with the stored score"
    reference = sum(best.values()) / len(best)
    got = true_score_metrics(_load(stem))["true_score_all_games"]
    assert got == pytest.approx(reference, abs=5e-4)


@pytest.mark.skipif(not BANKED.is_dir(), reason="banked waves not present")
def test_level_count_does_not_predict_the_objective():
    """The finding this module exists for. If this ever starts passing with a high
    correlation, the level count became meaningful and the doctrine should be revisited."""
    import statistics as st

    xs, ys = [], []
    for stem in WAVES:
        res = _load(stem)
        best = per_game_true_score(res)
        levels = {}
        for row in res.get("rows") or []:
            g = row.get("source_game")
            lv = int(row.get("levels_completed") or 0)
            levels[g] = max(levels.get(g, 0), lv)
        xs.append(sum(v for k, v in levels.items() if k != "ft09"))
        ys.append(sum(best.values()) / len(best))
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (n - 1)
    r = cov / (st.stdev(xs) * st.stdev(ys))
    assert abs(r) < 0.4, f"level count now correlates with the objective at r={r:.3f} — revisit the doctrine"
