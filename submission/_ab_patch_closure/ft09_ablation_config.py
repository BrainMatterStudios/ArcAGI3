"""FT09 ABLATION — pre-registered 2026-08-09, before any wave was launched.

WHY
    Re-scoring the eight banked waves on the TRUE objective (scripts/rescore_waves.py,
    validated against pc_driver's stored score on 224/224 rows) produced two results:

      1. corr(raw levels excl-ft09, true score) across 8 waves = -0.009. The statistic
         every recorded verdict was decided on is UNCORRELATED with the objective.
      2. ft09 alone accounts for 102% of the gap between the closure BASE arm
         (mean 1.3395) and every other arm (mean 0.8541). Excluding ft09 the other
         arms are slightly AHEAD.

    ft09 detail: BASE clears level 1 in 7-14 actions and is capped at the completion
    share (3/21 = 14.2857, i.e. both completed levels sit at the 115 ceiling). Every
    other arm needs 41-1522 actions on level 1 and mostly fails outright. The two arms
    carrying DIFF_LINES + DISPATCH + WIGGLE are 0/4 on ft09.

    But the full-25 geometry gives ONE clone per game, so every cell above is n=1
    (BASE's ft09 result is n=2). A 12-point swing on the single largest score term,
    resting on two samples, is not a finding yet.

THE FIX
    geometry["games"] = ("ft09",) spreads all 28 clones over one game, so a single
    wave yields n=28 instead of n=1.

PRE-REGISTERED READING (fixed before launch; do not revise after seeing results)
    Stage 1 (confirmation), arms FT09_BASE vs FT09_STRUCT, n=28 each:
      * CONFIRMED   iff mean(base ft09 score) - mean(struct ft09 score) >= 5.0 AND
                    a Mann-Whitney U one-sided test gives p < 0.01.
      * REFUTED     iff the difference is < 2.0 or the sign reverses.
      * AMBIGUOUS   otherwise -> stop, do not proceed to stage 2.
      With n=28 per arm and the observed spread, a 12-point effect is enormous; if it
      does not clear this bar it was an n=1 artifact and the ft09 story is dead.

    Stage 2 (attribution) runs ONLY on CONFIRMED. Single-flag arms over BASE_ENV
    isolate which switch causes it. Expected: whichever arm reproduces BASE's
    ~14.29 is innocent; whichever collapses to ~0 is the culprit.

    Secondary readouts on every arm: actions-on-level-1 distribution, levels
    completed, wallclock, and whether the stall watchdog fired ("killed": "stall").
    NOTE a confound to check in stage 1: BASE's two banked ft09 rows were both
    watchdog stall-kills at ~3700-4240s, while the other arms ran the full 7920s
    box. If early termination is what PRESERVES the efficiency score, the causal
    story is about the watchdog, not the patch set.

SCOPE LIMIT (state this in any conclusion)
    This measures ONE public game. It cannot establish that the effect generalises to
    the 55 hidden games. What it CAN do is identify a mechanism, which is then judged
    on whether it is game-specific or general.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from patch_closure_config import BASE_ENV  # noqa: E402

FT09_HYPOTHESIS = "ft09-ablation-2026-08-09"

# 28 clones, ALL on ft09. per_game_s/concurrency stay at the eval-shaped values so
# the only change from the frozen geometry is which games the clones cover.
FT09_GEOMETRY = {"clones": 28, "per_game_s": 7920, "concurrency": 28, "games": ("ft09",)}


def _with(**overrides: str) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(overrides)
    return env


# Stage 1 — confirmation.
FT09_ARMS: dict[str, dict[str, str]] = {
    "ft09_base": dict(BASE_ENV),
    "ft09_struct": _with(TAAF_DIFF_LINES="1", TAAF_DISPATCH="1",
                         TAAF_WIGGLE="1", TAAF_STRUCT="1"),
}

# Stage 2 — attribution. Single-flag deltas over BASE_ENV. Note patch19's own log
# line says its scaffolds need TAAF_WIGGLE=1, so ft09_dispatch may be inert by
# construction; that is itself informative and is why it is a separate arm.
FT09_STAGE2_ARMS: dict[str, dict[str, str]] = {
    "ft09_wiggle": _with(TAAF_WIGGLE="1"),
    "ft09_diff": _with(TAAF_DIFF_LINES="1"),
    "ft09_dispatch": _with(TAAF_DISPATCH="1", TAAF_WIGGLE="1"),
    "ft09_structonly": _with(TAAF_STRUCT="1"),
}

ALL_ARMS = {**FT09_ARMS, **FT09_STAGE2_ARMS}

FT09_SLUGS = {name: f"arc-agi-3-{name.replace('_', '-')}" for name in ALL_ARMS}

FT09_READING = {
    "hypothesis": (
        "The closure BASE arm's ft09 advantage (14.2857 vs ~0, n=2 vs 6 in the banked "
        "waves) is real and caused by one of DIFF_LINES / DISPATCH / WIGGLE / STRUCT."
    ),
    "stage1_confirmed": "mean(base) - mean(struct) >= 5.0 AND Mann-Whitney one-sided p < 0.01",
    "stage1_refuted": "difference < 2.0 or sign reverses",
    "stage2_gate": "runs only if stage 1 is CONFIRMED",
    "confound_to_check": (
        "BASE's banked ft09 rows were watchdog stall-kills at ~3700-4240s while other "
        "arms ran the full 7920s. Early termination may be what preserves the score."
    ),
    "scope_limit": "one public game; does not establish transfer to the 55 hidden games",
    "objective": "arc_agi scorecard: min(115,100*(baseline/actions)^2) * (level+1), capped by completion share",
}
