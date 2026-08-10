"""Frozen arm contract + pre-registered classifier for the patch-closure gate.

Two commit kernels, identical bytes except for the arm env mapping baked in by
build_patch_closure.py:

  base       — byte-identical patch layer, env-pinned to reconstruct the SCORED
               v7 behaviour: watchdog stall 900s, animation OFF, frontier
               graph (grinder/narration/veto) OFF. The v7 values are written
               out EXPLICITLY here — never inferred from current code defaults
               (the code default for TAAF_WATCHDOG_STALL_S is already 600).
  candidate  — the approved three-mechanism delta ONLY:
               TAAF_WATCHDOG_STALL_S=600, TAAF_ANIMATION=1, TAAF_GRAPH=1.
               Grinder parameters are pinned to their audited values so a
               later default change cannot silently move this experiment.

NOTE (2026-08-07 amendment): the refill-aware HUD unmask guard (commit f0ec605)
is part of duck_patches.py itself and therefore SHARED BASE for both arms — it
is intentionally NOT part of the candidate delta and must not appear in it.

`classify_result(base, candidate)` is the pre-registered analysis: it takes the
two `patch_closure_result.json` payloads and emits a frozen verdict —
GO / NO_GO / INFRA_FAILURE / INVALID — with every input metric echoed so the
verdict is auditable from the artifact alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1
HYPOTHESIS = "patch-closure-2026-08-04"

# The submission-shaped eval geometry (see submission/_rig/build_rig.py: one
# concurrency wave of 28 clones at the real 7920s per-game box).
GEOMETRY = {"clones": 28, "per_game_s": 7920, "concurrency": 28}

# The 25 official public games (environment_files/), by stem.
OFFICIAL_GAMES = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
)

# Never-unlocked target games (the 2026-08-02 research: the gap to 1.8 is 100%
# unlock-limited). A candidate-only first unlock here is the GO currency.
TARGET_GAMES = ("wa30", "m0r0", "g50t", "dc22")
# Unlock-sensitive continuity games watched for regression.
REGRESSION_GAMES = ("su15", "tu93")
# ft09 is excluded from level sums: its deep level count (8 levels) dominates
# any sum and drowns the unlock signal the gate exists to detect.
LEVELS_EXCLUDED_GAMES = ("ft09",)

# --- the frozen arm environments ------------------------------------------------

BASE_ENV = {
    "TAAF_WATCHDOG": "1", "TAAF_WATCHDOG_STALL_S": "900",
    "TAAF_HUD_MASK": "1", "TAAF_WIN_REPLAY": "1",
    "TAAF_ANTIFREEZE": "1", "TAAF_ANIMATION": "0",
    "TAAF_GRAPH": "0", "TAAF_GRAPH_GRIND_AGE_ACTIONS": "120",
    "TAAF_GRAPH_GRIND_AGE_TURNS": "10",
    "TAAF_GRAPH_GRIND_MAX_PER_LEVEL": "2",
    "TAAF_COMPACT": "0", "TAAF_PLAYBOOK": "0",
    "TAAF_GRID_BURNER": "0",
}
CANDIDATE_ENV = {
    **BASE_ENV,
    "TAAF_WATCHDOG_STALL_S": "600",
    "TAAF_ANIMATION": "1",
    "TAAF_GRAPH": "1",
}
ARM_ENV = {"base": BASE_ENV, "candidate": CANDIDATE_ENV}


@dataclass(frozen=True)
class GateThresholds:
    """Pre-registered decision boundaries. Editing these after the GPU data
    lands is goalpost-moving and must show up as a diff to this frozen file."""

    # GO requires at least this many candidate-only FIRST unlocks on the
    # never-unlocked target games (base levels == 0, candidate levels >= 1).
    min_new_target_unlocks: int = 2
    # NO_GO if any regression-watch game loses at least this many levels.
    regression_levels: int = 2
    # Fewer probe-observed actions than this means the arms never really
    # played (behav_probe.assert_observed's floor) -> INFRA_FAILURE.
    min_behavior_actions: int = 50


_REQUIRED_KEYS = (
    "schema_version", "hypothesis", "arm", "arm_env", "source_base_sha256",
    "patch_sha256", "geometry", "identity", "rows", "rows_by_source",
    "behavior", "patch_diagnostics",
)


def _levels(result: dict, stem: str) -> int:
    row = (result.get("rows_by_source") or {}).get(stem) or {}
    try:
        return int(row.get("levels", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _diag(result: dict, *path: str, default: int = 0) -> int:
    node: Any = result.get("patch_diagnostics") or {}
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    try:
        return int(node or 0)
    except (TypeError, ValueError):
        return default


def classify_result(
    base: dict, candidate: dict, thresholds: GateThresholds = GateThresholds()
) -> dict:
    """Pre-registered classification of a base/candidate result pair.

    Returns a dict whose "state" is one of:
      INVALID       — the pair violates the frozen contract (wrong schema, arm
                      env drift, hash mismatch, delta impurity). The comparison
                      is meaningless; nothing can be read from it.
      INFRA_FAILURE — contract OK but at least one run did not complete as an
                      instrument (error, short rows, dead probe). Re-run.
      NO_GO         — valid measurement, candidate did not clear the bar (or
                      tripped the regression veto).
      GO            — valid measurement, candidate cleared the pre-registered
                      unlock bar with no regression veto.
    """
    reasons: list[str] = []
    invalid: list[str] = []
    infra: list[str] = []

    pair = {"base": base, "candidate": candidate}
    for arm, result in pair.items():
        if not isinstance(result, dict):
            invalid.append(f"{arm}: result is not an object")
            continue
        missing = [k for k in _REQUIRED_KEYS if k not in result]
        if missing:
            invalid.append(f"{arm}: missing required keys {missing}")
            continue
        if result["schema_version"] != SCHEMA_VERSION:
            invalid.append(f"{arm}: schema_version {result['schema_version']!r} != {SCHEMA_VERSION}")
        if result["hypothesis"] != HYPOTHESIS:
            invalid.append(f"{arm}: hypothesis {result['hypothesis']!r} != {HYPOTHESIS!r}")
        if result["arm"] != arm:
            invalid.append(f"{arm}: arm label is {result['arm']!r}")
        if result["arm_env"] != ARM_ENV[arm]:
            drift = {
                k: (ARM_ENV[arm].get(k), (result["arm_env"] or {}).get(k))
                for k in set(ARM_ENV[arm]) | set(result["arm_env"] or {})
                if ARM_ENV[arm].get(k) != (result["arm_env"] or {}).get(k)
            }
            invalid.append(f"{arm}: arm_env drifted from the frozen mapping: {drift}")
        if not isinstance(result["identity"], dict) or not result["identity"]:
            invalid.append(f"{arm}: identity proof is missing or empty")

    if not invalid:
        for key in ("source_base_sha256", "patch_sha256", "geometry"):
            if base[key] != candidate[key]:
                invalid.append(
                    f"arms disagree on {key}: {base[key]!r} != {candidate[key]!r}")
        both_dry = bool(base.get("dry_run")) and bool(candidate.get("dry_run"))
        if base["geometry"] != GEOMETRY and not both_dry:
            invalid.append(
                f"geometry {base['geometry']!r} != registered {GEOMETRY!r} "
                "and the pair is not a dry run")
        # Delta purity: the base arm must show ZERO candidate-mechanism
        # activity. Any animation delivery or grinder motion in base means the
        # env pins leaked and this was not the registered comparison.
        base_anim = _diag(base, "animation", "payload_deliveries")
        base_grind = _diag(base, "graph", "grinder_engagements")
        base_veto = _diag(base, "graph", "vetoes_issued")
        if base_anim or base_grind or base_veto:
            invalid.append(
                "delta impurity: base arm shows candidate-mechanism activity "
                f"(animation_deliveries={base_anim}, grinder_engagements={base_grind}, "
                f"vetoes_issued={base_veto})")

    if invalid:
        return {
            "state": "INVALID",
            "reasons": invalid,
            "thresholds": asdict(thresholds),
            "hypothesis": HYPOTHESIS,
        }

    # --- instrument health (INFRA_FAILURE) -------------------------------------
    for arm, result in pair.items():
        if result.get("error") is not None:
            infra.append(f"{arm}: run recorded an error: {str(result['error'])[:300]}")
        stage = result.get("stage")
        if stage is not None and stage != "done":
            infra.append(f"{arm}: run stopped at stage {stage!r}")
        n_rows = len(result.get("rows") or [])
        clones = int((result.get("geometry") or {}).get("clones") or 0)
        if n_rows < clones:
            infra.append(f"{arm}: only {n_rows} rows for {clones} clones")
        actions = int(
            ((result.get("behavior") or {}).get("corpus") or {}).get("actions", 0) or 0)
        if actions < thresholds.min_behavior_actions:
            infra.append(
                f"{arm}: probe observed only {actions} actions "
                f"(< {thresholds.min_behavior_actions}) — the arms never really played")
        watch = TARGET_GAMES + REGRESSION_GAMES + LEVELS_EXCLUDED_GAMES
        absent = [g for g in watch if g not in (result.get("rows_by_source") or {})]
        if absent:
            infra.append(f"{arm}: rows_by_source is missing watched games {absent}")
        if (result.get("identity") or {}).get("toggles_ok") is False:
            infra.append(f"{arm}: identity proof reports realized toggles != expected")

    # --- pre-registered metrics -------------------------------------------------
    def level_sum(result: dict) -> int:
        return sum(
            _levels(result, g) for g in (result.get("rows_by_source") or {})
            if g not in LEVELS_EXCLUDED_GAMES
        )

    new_target_unlocks = [
        g for g in TARGET_GAMES
        if _levels(base, g) == 0 and _levels(candidate, g) >= 1
    ]
    regressions = {
        g: _levels(base, g) - _levels(candidate, g)
        for g in REGRESSION_GAMES
        if _levels(base, g) - _levels(candidate, g) >= thresholds.regression_levels
    }
    # THE OBJECTIVE, reported FIRST for both arms. The level deltas beneath it
    # are uncorrelated with the real score (r = -0.009 across the eight banked
    # waves), so a verdict read off them is not a verdict.
    #
    # READ THIS BEFORE INTERPRETING ANY DELTA HERE: the "base" arm is the
    # SETTLED PATCHED config (WATCHDOG/HUD_MASK/WIN_REPLAY/ANTIFREEZE on), NOT
    # the config we actually ship. duck-base v2 — the live pinned submission —
    # contains no duck_patches at all. On the live ledger the patched family
    # averages 0.788 (n=5) against base's 0.965 (n=10), so this baseline may
    # itself underperform what we ship by ~0.18. Every delta below is relative
    # to an unshipped reference.
    from true_score import true_score_metrics  # local import: avoids a cycle

    excl = tuple(LEVELS_EXCLUDED_GAMES)
    metrics = {
        "true_score": {
            "base": true_score_metrics(base, excluded=excl),
            "candidate": true_score_metrics(candidate, excluded=excl),
        },
        "baseline_caveat": ("'base' here is the settled PATCHED arm, not the shipped "
                            "duck-base v2 (which carries no patches at all)"),
        "levels_excl_ft09": {
            "base": level_sum(base),
            "candidate": level_sum(candidate),
            "delta": level_sum(candidate) - level_sum(base),
        },
        "new_target_unlocks": new_target_unlocks,
        "regressions": regressions,
        "animation_uptake": {
            "base_payload_deliveries": _diag(base, "animation", "payload_deliveries"),
            "candidate_payload_deliveries": _diag(candidate, "animation", "payload_deliveries"),
            "candidate_frames_delivered": _diag(candidate, "animation", "frames_delivered"),
        },
        "grinder": {
            "engagements": _diag(candidate, "graph", "grinder_engagements"),
            "age_triggers": _diag(candidate, "graph", "grinder_age_triggers"),
            "levels_unlocked_by_grinder": _diag(candidate, "graph", "levels_unlocked_by_grinder"),
            "narrations_injected": _diag(candidate, "graph", "narrations_injected"),
            "vetoes_issued": _diag(candidate, "graph", "vetoes_issued"),
        },
        "stale_closes": {
            "base": _diag(base, "watchdog", "stall_kills"),
            "candidate": _diag(candidate, "watchdog", "stall_kills"),
        },
    }

    if infra:
        return {
            "state": "INFRA_FAILURE",
            "reasons": infra,
            "metrics": metrics,
            "thresholds": asdict(thresholds),
            "hypothesis": HYPOTHESIS,
        }

    # --- the frozen decision ----------------------------------------------------
    if regressions:
        state = "NO_GO"
        reasons.append(
            f"regression veto: {regressions} (>= {thresholds.regression_levels} "
            "levels lost on a regression-watch game)")
    elif len(new_target_unlocks) >= thresholds.min_new_target_unlocks:
        state = "GO"
        reasons.append(
            f"candidate-only first unlocks on never-unlocked targets: "
            f"{new_target_unlocks} (>= {thresholds.min_new_target_unlocks})")
    else:
        state = "NO_GO"
        reasons.append(
            f"only {len(new_target_unlocks)} new target unlock(s) "
            f"({new_target_unlocks}) — bar is {thresholds.min_new_target_unlocks}; "
            "no regression veto tripped")

    return {
        "state": state,
        "reasons": reasons,
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "hypothesis": HYPOTHESIS,
    }
