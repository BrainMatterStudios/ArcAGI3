"""STRUCTURAL SCREEN — frozen arm env, reading contract and pre-registered
classifier (final kernel of the 2026-08-09 sprint).

One candidate-only kernel (slug arc-agi-3-struct-screen) at the closure
geometry. Arm env = closure BASE_ENV (v7 pins) PLUS
{TAAF_DIFF_LINES, TAAF_WIGGLE, TAAF_DISPATCH, TAAF_STRUCT} = "1".
TAAF_RUN_PROBE stays UNSET: the patch-21 plan channel supersedes run_probe
(its advertisement is replaced by the plan contract; the sandbox global merely
remains available). TAAF_VERIFY stays UNSET: one variable set at a time.

The package screen's autopsy (db5bc44: STOP — ADOPTION failure, not
capability; 1238/1238 turns never used run_probe because the tool schema and
runtime-globals sentence omitted it) motivates patch 21/22: the plan channel
is STRUCTURAL — every plain action() call routes through plan_execute, the
contract lives in the tool schema itself, and the enforced brake/phase gates
ship the offline-validated mechanisms. The single most important number this
run produces is LIVE ACTIONS-PER-TURN on the real 27B (the adoption metric:
dry-run mock adoption 4.65 actions/turn vs 1.0 base at cfeb92a).

`classify_struct(result)` emits ADVANCE / STOP / INFRA_FAILURE / INVALID.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    GEOMETRY,
    LEVELS_EXCLUDED_GAMES,
    SCHEMA_VERSION,
    TARGET_GAMES,
)

STRUCT_HYPOTHESIS = "struct-screen-2026-08-09"

STRUCT_FLAGS = {
    "TAAF_DIFF_LINES": "1",
    "TAAF_WIGGLE": "1",
    "TAAF_DISPATCH": "1",
    "TAAF_STRUCT": "1",
}
STRUCT_ENV = {**BASE_ENV, **STRUCT_FLAGS}

# Banked references this screen is read against (same geometry and machinery).
BANKED = {
    "base_pair": {
        "kernel": "ahmedmobasher86/arc-agi-3-patch-closure-base",
        "levels_excl_ft09_by_wave": [11, 12],
        "target_first_unlocks": 0,
        "actions_per_turn": 1.0,
    },
    "package_screen": {
        "kernel": "ahmedmobasher86/arc-agi-3-package-screen",
        "levels_excl_ft09": 10,
        "verdict": "STOP (adoption failure, not capability — db5bc44)",
    },
    "struct_waves": {
        "kernel": "ahmedmobasher86/arc-agi-3-struct-screen",
        "adoption_by_wave": [2.01, 2.59],
        "levels_excl_ft09_by_wave": [17, 12],
        "notes": ("w1: 17 excl-ft09 + g50t unlock (6ea2f22); w2: level spike "
                  "did not replicate, adoption did (4a51db5) — hence the "
                  "f151409 adoption iteration this wave tests"),
    },
}


@dataclass(frozen=True)
class StructThresholds:
    """Pre-registered bars. Editing after the data lands = goalpost-moving."""

    # PRIMARY success criterion for the f151409 adoption-iteration wave
    # (approved 2026-08-09): live plan-actions per LLM deliberation.
    primary_min_plan_actions_per_llm_turn: float = 3.5
    # Secondary (levels) bars — unchanged from the first struct waves.
    advance_min_levels_excl_ft09: int = 18
    advance_min_new_target_unlocks: int = 2
    min_behavior_actions: int = 50


STRUCT_READING = {
    "question": (
        "Does the STRUCTURAL plan channel (patch 21: every action() call IS a "
        "1-20 step plan submission, contract in the tool schema; patch 22: "
        "enforced A-not-B brake + SCOUT/COMMIT phase gate) fix the adoption "
        "failure the package screen exposed, and does adopted planning move "
        "real play at 27B on top of the v7 pins?"
    ),
    "primary_criterion": (
        "PRIMARY (approved 2026-08-09, this wave): live adoption "
        "result.adoption.plan_actions_per_llm_turn > 3.5. The first two "
        "struct waves measured 2.01 then 2.59 with singles still ~2/3 of "
        "deliberations; f151409's levers (worked battery-replay example, "
        "yield-aware nudges, post-single coaching, COMMIT length floor) "
        "exist to close exactly that gap, so this wave succeeds or fails on "
        "that number. Levels are SECONDARY at the unchanged bars below."
    ),
    "advance_bars": (
        "SECONDARY (unchanged): ADVANCE to a paired wave if the struct arm "
        "reaches >= 18 levels excl-ft09 OR >= 2 first unlocks among "
        "wa30/m0r0/g50t/dc22 (never unlocked in the banked base pair). "
        "Otherwise STOP."
    ),
    "adoption": (
        "Definition (cfeb92a): plan-actions per LLM DELIBERATION — "
        "result.adoption.plan_actions_per_llm_turn, where llm_turns counts "
        "ToolAgent.analyze calls (the kernel-side equivalent of the mock "
        "dry run's POST count; the behavioural probe's actions_per_turn "
        "reads 1.0 by construction under TAAF_STRUCT because every plan step "
        "executes as its own single-action batch). Banked: base pair ~1.0; "
        "struct w1 2.01, w2 2.59; mock dry-run 4.65 (raw 5.34 incl. wiggle "
        "presses). Read it with the STRUCT_DIAGNOSTICS plan_lengths "
        "histogram and the new lever counters (examples_shown, coach_lines, "
        "commit_floor_lines, yield_nudges, bare_singles vs "
        "explicit_single_plans): a 27B that still submits 1-plans did NOT "
        "adopt, regardless of score."
    ),
    "mechanism_engagement": (
        "Report (not gate): STRUCT_DIAGNOSTICS (plans, plan_actions, "
        "plan_lengths histogram, wrapped_singles, invalid_dropped, nudges, "
        "brake_strips + menu_strips, scout/cap truncations, score_flushes, "
        "phase_transitions, reports_injected) plus the carried package "
        "mechanisms (wiggle batteries + repaired verdict modes, diff-lines "
        "reports, dispatch swaps/escapes)."
    ),
    "protocol_health": (
        "The 27B must still parse and act: compare actions-per-game against "
        "the banked base pair and the package screen (10 levels excl-ft09). "
        "A collapse means the plan contract hurts the protocol — a finding, "
        "not noise. Watch nudges and turns_without_plan for contract "
        "rejection."
    ),
    "not_a_readout": (
        "Score means at 1 wave are noise (A/A floor RMS 0.707 "
        "levels/game-run). One wave resolves only large effects; ADVANCE "
        "buys a paired wave, it ships nothing by itself."
    ),
    "banked": BANKED,
    "excluded_variables": (
        "TAAF_RUN_PROBE unset (superseded by the struct channel), "
        "TAAF_VERIFY unset (one variable set at a time), TAAF_ANIMATION/"
        "TAAF_GRAPH remain 0 (v7 pins)."
    ),
}

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


def _diag(result: dict, *path: str):
    node = result.get("patch_diagnostics") or {}
    for key in path:
        if not isinstance(node, dict):
            return 0
        node = node.get(key)
    if isinstance(node, dict):
        return node
    try:
        return int(node or 0)
    except (TypeError, ValueError):
        return 0


def classify_struct(
    result: dict, thresholds: StructThresholds = StructThresholds()
) -> dict:
    """Pre-registered read of the struct-screen result. States:
    ADVANCE / STOP / INFRA_FAILURE / INVALID."""
    invalid: list[str] = []
    infra: list[str] = []

    if not isinstance(result, dict):
        return {"state": "INVALID", "reasons": ["result is not an object"],
                "thresholds": asdict(thresholds), "hypothesis": STRUCT_HYPOTHESIS}
    missing = [k for k in _REQUIRED_KEYS if k not in result]
    if missing:
        invalid.append(f"missing required keys {missing}")
    else:
        if result["schema_version"] != SCHEMA_VERSION:
            invalid.append(f"schema_version {result['schema_version']!r} != {SCHEMA_VERSION}")
        if result["hypothesis"] != STRUCT_HYPOTHESIS:
            invalid.append(f"hypothesis {result['hypothesis']!r} != {STRUCT_HYPOTHESIS!r}")
        if result["arm"] != "struct":
            invalid.append(f"arm label is {result['arm']!r}")
        if result["arm_env"] != STRUCT_ENV:
            drift = {k: (STRUCT_ENV.get(k), (result["arm_env"] or {}).get(k))
                     for k in set(STRUCT_ENV) | set(result["arm_env"] or {})
                     if STRUCT_ENV.get(k) != (result["arm_env"] or {}).get(k)}
            invalid.append(f"arm_env drifted from the frozen mapping: {drift}")
        if not isinstance(result["identity"], dict) or not result["identity"]:
            invalid.append("identity proof is missing or empty")
        if result["geometry"] != GEOMETRY and not result.get("dry_run"):
            invalid.append(f"geometry {result['geometry']!r} != registered {GEOMETRY!r} "
                           "and the result is not a dry run")
    if invalid:
        return {"state": "INVALID", "reasons": invalid,
                "thresholds": asdict(thresholds), "hypothesis": STRUCT_HYPOTHESIS}

    if result.get("error") is not None:
        infra.append(f"run recorded an error: {str(result['error'])[:300]}")
    if result.get("stage") not in (None, "done"):
        infra.append(f"run stopped at stage {result.get('stage')!r}")
    clones = int((result.get("geometry") or {}).get("clones") or 0)
    if len(result.get("rows") or []) < clones:
        infra.append(f"only {len(result.get('rows') or [])} rows for {clones} clones")
    corpus = ((result.get("behavior") or {}).get("corpus") or {})
    actions_observed = int(corpus.get("actions", 0) or 0)
    if actions_observed < thresholds.min_behavior_actions:
        infra.append(f"probe observed only {actions_observed} actions "
                     f"(< {thresholds.min_behavior_actions})")
    watch = TARGET_GAMES + LEVELS_EXCLUDED_GAMES
    absent = [g for g in watch if g not in (result.get("rows_by_source") or {})]
    if absent:
        infra.append(f"rows_by_source is missing watched games {absent}")
    if (result.get("identity") or {}).get("toggles_ok") is False:
        infra.append("identity proof reports realized toggles != expected")
    struct = _diag(result, "struct")
    if not isinstance(struct, dict) or not struct:
        infra.append("STRUCT_DIAGNOSTICS missing from patch_diagnostics — the "
                     "channel this screen exists to measure was not recorded")
        struct = {}

    levels = sum(_levels(result, g) for g in (result.get("rows_by_source") or {})
                 if g not in LEVELS_EXCLUDED_GAMES)
    unlocks = [g for g in TARGET_GAMES if _levels(result, g) >= 1]
    rows = result.get("rows") or []
    acts = [int(r.get("actions_total") or 0) for r in rows]
    plans = int(struct.get("plans", 0) or 0)
    adoption_rec = result.get("adoption") or {}
    if not adoption_rec:
        infra.append("adoption block missing from the result — the headline "
                     "metric of this screen was not recorded")
    metrics = {
        "levels_excl_ft09": levels,
        "banked": BANKED,
        "new_target_unlocks": unlocks,
        "adoption": {
            # THE headline (cfeb92a definition): plan-actions per LLM deliberation
            "plan_actions_per_llm_turn": adoption_rec.get("plan_actions_per_llm_turn"),
            "actions_per_llm_turn_raw": adoption_rec.get("actions_per_llm_turn"),
            "llm_turns": adoption_rec.get("llm_turns"),
            "banked_base_actions_per_turn": BANKED["base_pair"]["actions_per_turn"],
            # probe-batch view: 1.0 by construction under TAAF_STRUCT (each
            # plan step is its own batch) — recorded for continuity, not a readout
            "probe_actions_per_turn": corpus.get("actions_per_turn"),
            "plan_lengths": struct.get("plan_lengths") or {},
            "plans": plans,
            "wrapped_singles": int(struct.get("wrapped_singles", 0) or 0),
            "multi_step_plan_share": round(
                1.0 - (int(struct.get("wrapped_singles", 0) or 0) / plans), 4)
                if plans else None,
        },
        "engagement": {
            "struct": struct,
            "wiggle": _diag(result, "wiggle"),
            "wiggle_modes_assigned": (result.get("patch_diagnostics") or {}).get(
                "wiggle_modes_assigned") or {},
            "diff_lines": _diag(result, "diff_lines"),
            "dispatch": _diag(result, "dispatch"),
        },
        "protocol_health": {
            "mean_actions_per_game": round(sum(acts) / max(len(acts), 1), 1),
            "games_with_zero_actions": sum(1 for a in acts if a == 0),
            "nudges": int(struct.get("nudges", 0) or 0),
            "turns_without_plan": int(struct.get("turns_without_plan", 0) or 0),
            "note": "compare against the banked base pair and package screen",
        },
    }

    if infra:
        return {"state": "INFRA_FAILURE", "reasons": infra, "metrics": metrics,
                "thresholds": asdict(thresholds), "hypothesis": STRUCT_HYPOTHESIS}

    # PRIMARY (this wave): the adoption bar. Reported as its own verdict block
    # so the read cannot bury it; the ADVANCE/STOP state stays level-driven
    # (the unchanged secondary bars).
    adoption_value = adoption_rec.get("plan_actions_per_llm_turn")
    primary_adoption = {
        "bar": thresholds.primary_min_plan_actions_per_llm_turn,
        "value": adoption_value,
        "met": (adoption_value is not None
                and adoption_value > thresholds.primary_min_plan_actions_per_llm_turn),
    }

    reasons = [
        ("PRIMARY adoption criterion "
         f"{'MET' if primary_adoption['met'] else 'NOT MET'}: "
         f"plan_actions_per_llm_turn={adoption_value} vs bar "
         f">{thresholds.primary_min_plan_actions_per_llm_turn} "
         "(banked: base ~1.0, struct waves 2.01/2.59)")
    ]
    level_reasons = []
    if levels >= thresholds.advance_min_levels_excl_ft09:
        state = "ADVANCE"
        level_reasons.append(
            f"levels excl-ft09 {levels} >= {thresholds.advance_min_levels_excl_ft09}")
    if len(unlocks) >= thresholds.advance_min_new_target_unlocks:
        state = "ADVANCE"
        level_reasons.append(f"first unlocks on never-unlocked targets: {unlocks} "
                             f">= {thresholds.advance_min_new_target_unlocks}")
    if not level_reasons:
        state = "STOP"
        level_reasons.append(
            f"levels excl-ft09 {levels} < {thresholds.advance_min_levels_excl_ft09} "
            f"and only {len(unlocks)} target unlock(s) ({unlocks}) — bar is "
            f"{thresholds.advance_min_new_target_unlocks}")
    reasons.extend(f"secondary: {r}" for r in level_reasons)

    return {"state": state, "primary_adoption": primary_adoption,
            "reasons": reasons, "metrics": metrics,
            "thresholds": asdict(thresholds), "hypothesis": STRUCT_HYPOTHESIS,
            "pre_registered_reading_present": "pre_registered_reading" in result}
