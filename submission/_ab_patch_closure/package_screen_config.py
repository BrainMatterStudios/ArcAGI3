"""Probe-infrastructure PACKAGE SCREEN — frozen arm env, reading contract and
pre-registered classifier (Ahmed-approved 2026-08-09).

One candidate-only kernel (slug arc-agi-3-package-screen) at the closure
geometry, read against the BANKED closure base pair (two COMPLETE base-arm
waves at identical geometry/kernels: 11 and 12 levels excl-ft09). Arm env =
the closure BASE_ENV (v7 pins: stall 900, animation 0, graph 0) PLUS the four
package flags — patches 16-19, all default-OFF in duck_patches.py and
call-time gated:

  TAAF_DIFF_LINES=1  patch16 action-locked structured change lines
  TAAF_WIGGLE=1      patch17 LLM-free opening wiggle battery -> masks + mode
  TAAF_RUN_PROBE=1   patch18 run_probe probe-battery macro-action
  TAAF_DISPATCH=1    patch19 archetype-dispatch prompt scaffolds

`classify_screen(package)` emits ADVANCE / STOP / INFRA_FAILURE / INVALID
with the mechanism-engagement and protocol-health metrics echoed.
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

PACKAGE_HYPOTHESIS = "package-screen-2026-08-09"

PACKAGE_FLAGS = {
    "TAAF_DIFF_LINES": "1",
    "TAAF_WIGGLE": "1",
    "TAAF_RUN_PROBE": "1",
    "TAAF_DISPATCH": "1",
}
PACKAGE_ENV = {**BASE_ENV, **PACKAGE_FLAGS}

# The banked reference this screen is read against (closure base arm, same
# 28-clone / 7920s / concurrency-28 geometry, COMPLETE runs).
BANKED_BASE = {
    "kernel": "ahmedmobasher86/arc-agi-3-patch-closure-base",
    "geometry": dict(GEOMETRY),
    "levels_excl_ft09_by_wave": [11, 12],
    "target_first_unlocks": 0,  # wa30/m0r0/g50t/dc22 never unlocked in either wave
}


@dataclass(frozen=True)
class ScreenThresholds:
    """Pre-registered bars. Editing after the data lands = goalpost-moving."""

    advance_min_levels_excl_ft09: int = 18
    advance_min_new_target_unlocks: int = 2
    min_behavior_actions: int = 50


PACKAGE_READING = {
    "question": (
        "Does the probe-infrastructure package (patches 16-19: structured "
        "change lines + wiggle controllability masks + run_probe macro-action "
        "+ archetype dispatch) move real play at 27B on top of the v7 pins? "
        "Candidate-only screen against the banked closure base pair "
        f"({BANKED_BASE['levels_excl_ft09_by_wave']} levels excl-ft09 at the "
        "same geometry and kernel machinery)."
    ),
    "advance_bars": (
        "ADVANCE to a paired wave if the package arm reaches >= 18 levels "
        "excl-ft09 OR >= 2 first unlocks among wa30/m0r0/g50t/dc22 (never "
        "unlocked in either banked base wave). Otherwise STOP."
    ),
    "mechanism_engagement": (
        "Report (not gate): wiggle batteries run + presses, run_probe calls / "
        "probe actions / refusals, dispatch modes assigned (per-session wiggle "
        "mode histogram) + scaffold swaps/escapes, diff-lines reports "
        "injected. Zero engagement on an enabled mechanism means the screen "
        "did not test that mechanism — say so in the read."
    ),
    "protocol_health": (
        "The 27B must still parse and act under the package prompts: compare "
        "actions-per-game and behav actions_per_turn against the banked base "
        "pair. A collapse in actions-per-game vs base means the package "
        "prompts are hurting the protocol — that is a FINDING, not noise; "
        "report it alongside the verdict."
    ),
    "not_a_readout": (
        "Score means at 1 wave are noise (A/A floor RMS 0.707 "
        "levels/game-run). One wave resolves only large effects; ADVANCE "
        "buys a paired wave, it ships nothing by itself."
    ),
    "banked_base": BANKED_BASE,
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


def _diag(result: dict, *path: str) -> int:
    node = result.get("patch_diagnostics") or {}
    for key in path:
        if not isinstance(node, dict):
            return 0
        node = node.get(key)
    try:
        return int(node or 0)
    except (TypeError, ValueError):
        return 0


def classify_screen(
    package: dict, thresholds: ScreenThresholds = ScreenThresholds()
) -> dict:
    """Pre-registered read of the package-screen result against the banked
    base pair. States: ADVANCE / STOP / INFRA_FAILURE / INVALID."""
    invalid: list[str] = []
    infra: list[str] = []

    if not isinstance(package, dict):
        return {"state": "INVALID", "reasons": ["result is not an object"],
                "thresholds": asdict(thresholds), "hypothesis": PACKAGE_HYPOTHESIS}
    missing = [k for k in _REQUIRED_KEYS if k not in package]
    if missing:
        invalid.append(f"missing required keys {missing}")
    else:
        if package["schema_version"] != SCHEMA_VERSION:
            invalid.append(f"schema_version {package['schema_version']!r} != {SCHEMA_VERSION}")
        if package["hypothesis"] != PACKAGE_HYPOTHESIS:
            invalid.append(f"hypothesis {package['hypothesis']!r} != {PACKAGE_HYPOTHESIS!r}")
        if package["arm"] != "package":
            invalid.append(f"arm label is {package['arm']!r}")
        if package["arm_env"] != PACKAGE_ENV:
            drift = {k: (PACKAGE_ENV.get(k), (package["arm_env"] or {}).get(k))
                     for k in set(PACKAGE_ENV) | set(package["arm_env"] or {})
                     if PACKAGE_ENV.get(k) != (package["arm_env"] or {}).get(k)}
            invalid.append(f"arm_env drifted from the frozen mapping: {drift}")
        if not isinstance(package["identity"], dict) or not package["identity"]:
            invalid.append("identity proof is missing or empty")
        if package["geometry"] != GEOMETRY and not package.get("dry_run"):
            invalid.append(f"geometry {package['geometry']!r} != registered {GEOMETRY!r} "
                           "and the result is not a dry run")
    if invalid:
        return {"state": "INVALID", "reasons": invalid,
                "thresholds": asdict(thresholds), "hypothesis": PACKAGE_HYPOTHESIS}

    if package.get("error") is not None:
        infra.append(f"run recorded an error: {str(package['error'])[:300]}")
    if package.get("stage") not in (None, "done"):
        infra.append(f"run stopped at stage {package.get('stage')!r}")
    clones = int((package.get("geometry") or {}).get("clones") or 0)
    if len(package.get("rows") or []) < clones:
        infra.append(f"only {len(package.get('rows') or [])} rows for {clones} clones")
    actions_observed = int(
        ((package.get("behavior") or {}).get("corpus") or {}).get("actions", 0) or 0)
    if actions_observed < thresholds.min_behavior_actions:
        infra.append(f"probe observed only {actions_observed} actions "
                     f"(< {thresholds.min_behavior_actions})")
    watch = TARGET_GAMES + LEVELS_EXCLUDED_GAMES
    absent = [g for g in watch if g not in (package.get("rows_by_source") or {})]
    if absent:
        infra.append(f"rows_by_source is missing watched games {absent}")
    if (package.get("identity") or {}).get("toggles_ok") is False:
        infra.append("identity proof reports realized toggles != expected")

    levels = sum(_levels(package, g) for g in (package.get("rows_by_source") or {})
                 if g not in LEVELS_EXCLUDED_GAMES)
    unlocks = [g for g in TARGET_GAMES if _levels(package, g) >= 1]
    rows = package.get("rows") or []
    acts = [int(r.get("actions_total") or 0) for r in rows]
    # THE OBJECTIVE, reported FIRST. The level count beneath it is uncorrelated
    # with the real score (r = -0.009 across the eight banked waves). Note also
    # that the banked "base" comparator is the settled PATCHED arm, not the
    # shipped duck-base v2 (which carries no patches at all).
    from true_score import true_score_metrics  # local import: avoids a cycle

    metrics = {
        **true_score_metrics(package, excluded=tuple(LEVELS_EXCLUDED_GAMES)),
        "levels_excl_ft09": levels,
        "banked_base_levels_excl_ft09": BANKED_BASE["levels_excl_ft09_by_wave"],
        "new_target_unlocks": unlocks,
        "engagement": {
            "wiggle": {
                "batteries": _diag(package, "wiggle", "batteries"),
                "battery_presses": _diag(package, "wiggle", "battery_presses"),
                "reprobes": _diag(package, "wiggle", "reprobes"),
            },
            "run_probe": {
                "calls": _diag(package, "run_probe", "calls"),
                "actions": _diag(package, "run_probe", "actions"),
                "refusals": _diag(package, "run_probe", "refusals"),
            },
            "dispatch": {
                "modes_assigned": (package.get("patch_diagnostics") or {}).get(
                    "wiggle_modes_assigned") or {},
                "swaps": _diag(package, "dispatch", "swaps"),
                "escapes": _diag(package, "dispatch", "escapes"),
            },
            "diff_lines": {
                "reports": _diag(package, "diff_lines", "reports"),
                "actions": _diag(package, "diff_lines", "actions"),
            },
        },
        "protocol_health": {
            "mean_actions_per_game": round(sum(acts) / max(len(acts), 1), 1),
            "games_with_zero_actions": sum(1 for a in acts if a == 0),
            "behav_actions_per_turn": (
                ((package.get("behavior") or {}).get("corpus") or {}).get("actions_per_turn")),
            "note": "compare against the banked base pair's rows before reading",
        },
    }

    if infra:
        return {"state": "INFRA_FAILURE", "reasons": infra, "metrics": metrics,
                "thresholds": asdict(thresholds), "hypothesis": PACKAGE_HYPOTHESIS}

    reasons = []
    if levels >= thresholds.advance_min_levels_excl_ft09:
        state = "ADVANCE"
        reasons.append(f"levels excl-ft09 {levels} >= {thresholds.advance_min_levels_excl_ft09}")
    if len(unlocks) >= thresholds.advance_min_new_target_unlocks:
        state = "ADVANCE"
        reasons.append(f"first unlocks on never-unlocked targets: {unlocks} "
                       f">= {thresholds.advance_min_new_target_unlocks}")
    if not reasons:
        state = "STOP"
        reasons.append(
            f"levels excl-ft09 {levels} < {thresholds.advance_min_levels_excl_ft09} "
            f"and only {len(unlocks)} target unlock(s) ({unlocks}) — bar is "
            f"{thresholds.advance_min_new_target_unlocks}")

    return {"state": state, "reasons": reasons, "metrics": metrics,
            "thresholds": asdict(thresholds), "hypothesis": PACKAGE_HYPOTHESIS,
            "pre_registered_reading_present": "pre_registered_reading" in package}
