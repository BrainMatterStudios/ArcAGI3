"""SHIPPED-CONFIG SCREEN — frozen arm contract, pre-registration and classifier.

THE QUESTION (HANDOFF-2026-08-10 §3/§5.1). Every A/B this campaign ran used a
`base` arm running WITH patches (BASE_ENV: WATCHDOG, HUD_MASK, WIN_REPLAY,
ANTIFREEZE ON), but the actually-shipped live submission — duck-base v2, the
pinned config behind the whole n=10 base distribution — contains NO
duck_patches at all. The live ledger hints the unpatched config ships BETTER:
patched family mean 0.788 (n=5, heterogeneous) vs base 0.965 (n=10). So every
rig delta was measured against an unshipped baseline. This screen asks the
question directly, offline, on the true objective.

THE ARMS.
  shipped — duck-base with NO patch inline, NO apply_all, NO arm env pins
            beyond what duck-base itself sets. The hook cell carries the
            MIRROR of APPLY_BLOCK: it hard-asserts patch symbols are ABSENT.
            Instrumentation (behavioural probe + pc_driver) is byte-identical
            to every other arm.
  base    — the EXISTING, UNCHANGED closure base kernel (BASE_ENV, patches
            applied and hard-asserted). A fresh wave from the same session is
            the primary comparator; the two banked base waves at identical
            geometry are additional controls (BANKED_BASE_CONTROLS below).

`classify_shipped(shipped, base)` is the pre-registered analysis. States:
INVALID / INFRA_FAILURE / SHIPPED_AHEAD / BASE_AHEAD / INDISTINGUISHABLE.
"""
from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    GEOMETRY,
    HYPOTHESIS as CLOSURE_HYPOTHESIS,
    LEVELS_EXCLUDED_GAMES,
    OFFICIAL_GAMES,
    SCHEMA_VERSION,
)

SHIPPED_HYPOTHESIS = "shipped-config-screen-2026-08-10"

# The shipped arm pins NOTHING: duck-base v2's own cells set only
# TAAF_RUN_AS_SUBMISSION / TAAF_MINIMAL_DIAGNOSTICS / ONLY_RESET_LEVELS /
# MPLBACKEND and the kaggle bundle plumbing (verified against
# submission/_duck_base/duck-base.ipynb cell 2). An empty mapping here IS the
# contract — its fingerprint (sha256 of "{}") cannot collide with BASE_ENV's
# or CANDIDATE_ENV's non-empty mappings.
SHIPPED_ENV: dict[str, str] = {}

# Stamped into the contract and the result where the inlined-patch sha256
# would otherwise be. Deliberately not hex so it can never be mistaken for a
# real hash.
UNPATCHED_SENTINEL = "UNPATCHED"

# Additional controls: the two COMPLETE banked closure-base waves. Geometry
# verified identical to GEOMETRY on 2026-08-10 (both files carry
# {'clones': 28, 'per_game_s': 7920, 'concurrency': 28}, stage=done, 28 rows,
# 25 sources). True scores computed with true_score.true_score_metrics.
BANKED_BASE_CONTROLS = {
    "path": "scratchpad/banked_waves_20260809/",
    "files": ["pc_base.json", "w2_base.json"],
    "kernel": "ahmedmobasher86/arc-agi-3-patch-closure-base",
    "geometry": dict(GEOMETRY),
    "geometry_verified": ("2026-08-10: both waves are full-25-game, 28-clone, "
                          "7920s, concurrency-28, stage=done"),
    "true_score_all_games_by_wave": [1.4751, 1.2039],
    "true_score_excl_ft09_by_wave": [0.9414, 0.6588],
}

# The banked base pair's own between-wave spread on the decision statistic:
# |1.4751 - 1.2039| = 0.2712. That is the best same-arm repeat spread we hold
# at this geometry, and it is LARGER than the live hint (0.18).
_BANKED_SPREAD = round(abs(BANKED_BASE_CONTROLS["true_score_all_games_by_wave"][0]
                           - BANKED_BASE_CONTROLS["true_score_all_games_by_wave"][1]), 4)


@dataclass(frozen=True)
class ShippedThresholds:
    """Pre-registered decision boundaries. Editing these after the GPU data
    lands is goalpost-moving and must show up as a diff to this frozen file."""

    # Directional call (SHIPPED_AHEAD / BASE_AHEAD) requires |delta| at least
    # this large on true_score_all_games; below it the verdict is
    # INDISTINGUISHABLE. Set to the banked base pair's own repeat spread
    # (0.2712) — an effect smaller than the instrument's same-arm wobble is
    # not a directional finding at one wave per arm.
    min_effect_true_score: float = _BANKED_SPREAD
    # Fewer probe-observed actions than this means an arm never really played.
    min_behavior_actions: int = 50


SHIPPED_READING = {
    "hypothesis": (
        "The shipped unpatched config (duck-base v2, no duck_patches) differs "
        "from BASE_ENV (the settled patched rig baseline) on the true "
        "objective; live-ledger hint of the magnitude: 0.18 (patched family "
        "0.788 n=5 heterogeneous vs shipped base 0.965 n=10)."
    ),
    "reading_rule": (
        "Decision statistic: true_score_all_games (per game MAX over clones, "
        "per wave MEAN over the full 25 games; rows[].score as stored by "
        "pc_driver) for the shipped arm MINUS the fresh base arm from the "
        "SAME session at the frozen geometry (28 clones, 7920s, concurrency "
        "28, full 25 games). Directional verdict only if |delta| >= 0.2712 "
        "(the banked base pair's own repeat spread); otherwise "
        "INDISTINGUISHABLE. Secondary (report, never gate): "
        "true_score_excl_ft09, per-game deltas, levels."
    ),
    "banked_controls": (
        "Two COMPLETE banked base waves at IDENTICAL geometry exist "
        "(scratchpad/banked_waves_20260809/pc_base.json, w2_base.json; "
        "true_score_all_games 1.4751 and 1.2039) and serve as additional "
        "base-side controls: report where the shipped wave falls relative to "
        "the banked band [1.2039, 1.4751] alongside the primary same-session "
        "delta."
    ),
    "power_honesty": (
        "One wave per arm CANNOT resolve the 0.18 live hint: the banked base "
        "pair's same-arm spread is 0.2712 at this exact geometry. This screen "
        "can only detect a LARGE offline difference or band the effect; a "
        "null here (INDISTINGUISHABLE) does NOT refute the live-ledger hint. "
        "The live ledger comparison itself is heterogeneous (patched n=5 "
        "spans several configs) — treat 0.18 as a hint, not a measurement."
    ),
    "not_a_readout": (
        "Do not read levels: the level count correlates with the objective at "
        "r = -0.009 across the eight banked waves. Do not promote either "
        "config off this screen alone; a slot decision needs the live "
        "distribution (base n=10 mean 0.965 sd 0.208, one-slot MDE 0.61)."
    ),
    "banked_base_controls": BANKED_BASE_CONTROLS,
}

_REQUIRED_KEYS = (
    "schema_version", "hypothesis", "arm", "arm_env", "source_base_sha256",
    "patch_sha256", "geometry", "identity", "rows", "rows_by_source",
    "behavior", "patch_diagnostics",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


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


def _levels_sum(result: dict, excluded: tuple[str, ...]) -> int:
    total = 0
    for stem, row in (result.get("rows_by_source") or {}).items():
        if stem in excluded:
            continue
        try:
            total += int((row or {}).get("levels", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def classify_shipped(
    shipped: dict, base: dict, thresholds: ShippedThresholds = ShippedThresholds()
) -> dict:
    """Pre-registered classification of a shipped/base result pair.

    INVALID        — contract violation (wrong arm labels/env/hashes, patch
                     contamination in the shipped arm, or an UNPATCHED base
                     arm — either way the registered comparison did not run).
    INFRA_FAILURE  — contract OK but an arm did not complete as an instrument.
    SHIPPED_AHEAD / BASE_AHEAD — |delta| >= min_effect_true_score.
    INDISTINGUISHABLE — valid measurement, delta below the pre-registered
                     minimum effect (NOT evidence of equality; see
                     power_honesty in SHIPPED_READING).
    """
    invalid: list[str] = []
    infra: list[str] = []
    pair = {"shipped": shipped, "base": base}

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
        if result["arm"] != arm:
            invalid.append(f"{arm}: arm label is {result['arm']!r}")
        if not isinstance(result["identity"], dict) or not result["identity"]:
            invalid.append(f"{arm}: identity proof is missing or empty")

    if not invalid:
        # -- per-arm frozen contracts ------------------------------------------
        if shipped["hypothesis"] != SHIPPED_HYPOTHESIS:
            invalid.append(
                f"shipped: hypothesis {shipped['hypothesis']!r} != {SHIPPED_HYPOTHESIS!r}")
        if shipped["arm_env"] != SHIPPED_ENV:
            invalid.append(
                f"shipped: arm_env must be EMPTY (no pins), got {shipped['arm_env']!r}")
        if shipped["patch_sha256"] != UNPATCHED_SENTINEL:
            invalid.append(
                f"shipped: patch_sha256 {shipped['patch_sha256']!r} != {UNPATCHED_SENTINEL!r} "
                "— the shipped arm must not carry inlined patch bytes")
        if base["hypothesis"] != CLOSURE_HYPOTHESIS:
            invalid.append(
                f"base: hypothesis {base['hypothesis']!r} != {CLOSURE_HYPOTHESIS!r} "
                "(the comparator is the EXISTING closure base kernel, unchanged)")
        if base["arm_env"] != BASE_ENV:
            drift = {k: (BASE_ENV.get(k), (base["arm_env"] or {}).get(k))
                     for k in set(BASE_ENV) | set(base["arm_env"] or {})
                     if BASE_ENV.get(k) != (base["arm_env"] or {}).get(k)}
            invalid.append(f"base: arm_env drifted from the frozen BASE_ENV: {drift}")
        if not _HEX64.match(str(base["patch_sha256"] or "")):
            invalid.append(
                f"base: patch_sha256 {base['patch_sha256']!r} is not a real inlined-patch hash")

        # -- pair coherence ----------------------------------------------------
        if shipped["source_base_sha256"] != base["source_base_sha256"]:
            invalid.append(
                "arms disagree on source_base_sha256: "
                f"{shipped['source_base_sha256']!r} != {base['source_base_sha256']!r}")
        if shipped["geometry"] != base["geometry"]:
            invalid.append(
                f"arms disagree on geometry: {shipped['geometry']!r} != {base['geometry']!r}")
        both_dry = bool(shipped.get("dry_run")) and bool(base.get("dry_run"))
        if shipped["geometry"] != GEOMETRY and not both_dry:
            invalid.append(
                f"geometry {shipped['geometry']!r} != registered {GEOMETRY!r} "
                "and the pair is not a dry run")

        # -- purity: the shipped arm must show ZERO patch-mechanism activity ---
        proof = (shipped.get("identity") or {}).get("patch_proof") or {}
        if proof.get("shipped_no_patch_markers") is not True:
            invalid.append(
                "shipped: identity.patch_proof.shipped_no_patch_markers is not True "
                "— the absence proof did not run (or patches leaked in)")
        contamination = {
            "animation_payload_deliveries": _diag(shipped, "animation", "payload_deliveries"),
            "graph_sessions": _diag(shipped, "graph", "sessions_with_graph_state"),
            "watchdog_stall_kills": _diag(shipped, "watchdog", "stall_kills"),
            "watchdog_recovery_resets": _diag(shipped, "watchdog", "recovery_resets"),
        }
        stalls = ((shipped.get("patch_diagnostics") or {}).get("watchdog")
                  or {}).get("stall_s_observed") or []
        if any(contamination.values()) or stalls:
            invalid.append(
                f"shipped: patch-mechanism activity detected: {contamination} "
                f"stall_s_observed={stalls} — this is not the unpatched shipped config")
        # -- and the base arm must be genuinely PATCHED ------------------------
        base_proof = (base.get("identity") or {}).get("patch_proof") or {}
        if base_proof.get("watchdog_should_stop_patched") is not True:
            invalid.append(
                "base: identity.patch_proof.watchdog_should_stop_patched is not True "
                "— an unpatched 'base' arm would make this an A/A, not the "
                "registered shipped-vs-BASE_ENV comparison")

    if invalid:
        return {"state": "INVALID", "reasons": invalid,
                "thresholds": asdict(thresholds), "hypothesis": SHIPPED_HYPOTHESIS}

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
            infra.append(f"{arm}: probe observed only {actions} actions "
                         f"(< {thresholds.min_behavior_actions})")
        absent = [g for g in OFFICIAL_GAMES if g not in (result.get("rows_by_source") or {})]
        if absent:
            infra.append(f"{arm}: rows_by_source is missing games {absent}")
        if (result.get("identity") or {}).get("toggles_ok") is False:
            infra.append(f"{arm}: identity proof reports realized toggles != expected")

    # --- pre-registered metrics -------------------------------------------------
    from true_score import per_game_true_score, true_score_metrics  # local: avoids a cycle

    excl = tuple(LEVELS_EXCLUDED_GAMES)
    ts_shipped = true_score_metrics(shipped, excluded=excl)
    ts_base = true_score_metrics(base, excluded=excl)
    per_shipped = per_game_true_score(shipped)
    per_base = per_game_true_score(base)
    per_game_delta = {
        g: round(per_shipped.get(g, 0.0) - per_base.get(g, 0.0), 4)
        for g in sorted(set(per_shipped) | set(per_base))
    }
    s_all, b_all = ts_shipped["true_score_all_games"], ts_base["true_score_all_games"]
    delta_all = (round(s_all - b_all, 4)
                 if s_all is not None and b_all is not None else None)
    s_ex, b_ex = ts_shipped["true_score_excl"], ts_base["true_score_excl"]
    banked = BANKED_BASE_CONTROLS["true_score_all_games_by_wave"]
    metrics = {
        "true_score": {"shipped": ts_shipped, "base": ts_base},
        "delta_all_games_shipped_minus_base": delta_all,
        "delta_excl_ft09_shipped_minus_base": (
            round(s_ex - b_ex, 4) if s_ex is not None and b_ex is not None else None),
        "per_game_delta": per_game_delta,
        "banked_base_band": {
            "true_score_all_games_by_wave": banked,
            "shipped_vs_band": (
                None if s_all is None else
                ("above" if s_all > max(banked) else
                 "below" if s_all < min(banked) else "inside")),
        },
        "levels_excl_ft09_secondary": {
            "shipped": _levels_sum(shipped, excl),
            "base": _levels_sum(base, excl),
            "warning": "r = -0.009 with the objective — never a verdict input",
        },
        "shipped_purity": {
            "no_patch_markers": True,
            "stall_s_observed": [],
        },
    }

    if infra:
        return {"state": "INFRA_FAILURE", "reasons": infra, "metrics": metrics,
                "thresholds": asdict(thresholds), "hypothesis": SHIPPED_HYPOTHESIS}

    # --- the frozen decision ----------------------------------------------------
    reasons: list[str] = []
    if delta_all is None:
        return {"state": "INFRA_FAILURE",
                "reasons": ["no rows carried a stored score — cannot compute the objective"],
                "metrics": metrics, "thresholds": asdict(thresholds),
                "hypothesis": SHIPPED_HYPOTHESIS}
    if abs(delta_all) >= thresholds.min_effect_true_score:
        state = "SHIPPED_AHEAD" if delta_all > 0 else "BASE_AHEAD"
        reasons.append(
            f"|delta true_score_all_games| = {abs(delta_all)} >= "
            f"{thresholds.min_effect_true_score} (shipped {s_all} vs base {b_all})")
    else:
        state = "INDISTINGUISHABLE"
        reasons.append(
            f"|delta true_score_all_games| = {abs(delta_all)} < "
            f"{thresholds.min_effect_true_score} (shipped {s_all} vs base {b_all}) "
            "— below the banked same-arm repeat spread; NOT evidence of equality")

    return {"state": state, "reasons": reasons, "metrics": metrics,
            "thresholds": asdict(thresholds), "hypothesis": SHIPPED_HYPOTHESIS,
            "pre_registered_reading_present": "pre_registered_reading" in shipped}
