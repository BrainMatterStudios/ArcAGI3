"""QUANT-SWAP SCREEN — frozen arm contract, pre-registration and classifier.

THE QUESTION (submission/_quant_probe/UPLOAD_PLAN.md, Step 3). Every scored
submission and every rig arm serves the vrfai FP8 quant of Qwen3.6-27B
(driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot). sonpham's public 5-way quant
A/B found the RedHatAI FP8 quant (RedHatAI/Qwen3.6-27B-FP8) best among the
quants he tested — an EXTERNAL, unreplicated result on his own harness/games:
a hypothesis source, never evidence. RedHatAI-vs-vrfai is UNMEASURED for us.
This screen measures it offline on the true objective.

THE ARMS.
  quant   — the SHIPPED config (duck-base v2, NO duck_patches, NO apply_all)
            with ONE treatment, applied at build time by build_quant_screen.py:
            the served FP8 snapshot is swapped vrfai -> RedHatAI. Three seams,
            all hard-asserted: (1) kernel-metadata dataset mount, (2) duck-base
            cell 6's DATASET_SOURCES literal (TAAF_KAGGLE_INPUT_PATHS mapping),
            (3) an in-memory rewrite of setup_commands.json cmd 0's three serve
            anchors (MODEL_OWNER / MODEL_SLUG / SERVED_MODEL_NAME) immediately
            before the setup loop runs them — anchor drift aborts BEFORE the
            serve. SERVED_MODEL_NAME flows to LOCAL_ANALYZER_MODEL_ID, which
            pc_driver._pc_serving_probe verifies against /models, so a
            wrong-model serve is caught before any game minute.
  shipped — the EXISTING, UNCHANGED shipped-screen kernel
            (arc-agi-3-patch-closure-shipped). A wave from the same session is
            the comparator.

`classify_quant(quant, shipped)` is the pre-registered analysis. States:
INVALID / INFRA_FAILURE / QUANT_AHEAD / SHIPPED_AHEAD / INDISTINGUISHABLE.
"""
from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from patch_closure_config import (  # noqa: E402
    GEOMETRY,
    LEVELS_EXCLUDED_GAMES,
    OFFICIAL_GAMES,
    SCHEMA_VERSION,
)
from shipped_screen_config import (  # noqa: E402
    BANKED_BASE_CONTROLS,
    SHIPPED_ENV,
    SHIPPED_HYPOTHESIS,
    UNPATCHED_SENTINEL,
    ShippedThresholds,
)

QUANT_HYPOTHESIS = "quant-swap-screen-2026-08-14"

# The quant arm pins NOTHING behavioural. PC_QUANT_SNAPSHOT is an INERT
# identity pin (grep-verified 2026-08-14: no code in submission/ or the scored
# ref reads any PC_QUANT* name) whose only job is to make this arm's env
# fingerprint distinct from shipped's empty mapping — the classifier and the
# driver's env verification both hold it to exactly this value.
QUANT_ENV: dict[str, str] = {
    "PC_QUANT_SNAPSHOT": "redhatai-qwen3-6-27b-fp8-hf-snapshot",
}

# --- the swap, single source of truth --------------------------------------------
# Dataset refs (kernel-metadata dataset_sources + duck-base cell 6 literal).
VRFAI_DATASET = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
QUANT_DATASET = "ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot"

# Served-model aliases (--served-model-name -> /models -> LOCAL_ANALYZER_MODEL_ID).
VRFAI_SERVED_MODEL = "vrfai/Qwen3.6-27B-FP8"
QUANT_SERVED_MODEL = "RedHatAI/Qwen3.6-27B-FP8"

# The three serve anchors inside setup_commands.json cmd 0 (mounted read-only
# from the taaf-src bundle; verified byte-exact against
# scratchpad/taaf_scored_ref/setup_commands.json, one occurrence each,
# 2026-08-14). The quant run cell rewrites the command string IN-MEMORY with
# these pairs before subprocess.run and hard-aborts unless every anchor fired
# exactly once — anchor drift must never silently serve vrfai.
SERVE_ANCHOR_REWRITES: tuple[tuple[str, str], ...] = (
    ("MODEL_OWNER = 'driessmit1'",
     "MODEL_OWNER = 'ahmedmobasher86'"),
    ("MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'",
     "MODEL_SLUG = 'redhatai-qwen3-6-27b-fp8-hf-snapshot'"),
    ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
     "SERVED_MODEL_NAME = 'RedHatAI/Qwen3.6-27B-FP8'"),
)


@dataclass(frozen=True)
class QuantThresholds:
    """Pre-registered decision boundaries. Editing these after the GPU data
    lands is goalpost-moving and must show up as a diff to this frozen file."""

    # Directional call (QUANT_AHEAD / SHIPPED_AHEAD) requires |delta| at least
    # this large on true_score_all_games; below it the verdict is
    # INDISTINGUISHABLE. Same bar as the shipped screen: the banked base
    # pair's own same-arm repeat spread (0.2712) at this exact geometry.
    min_effect_true_score: float = ShippedThresholds().min_effect_true_score
    # Fewer probe-observed actions than this means an arm never really played.
    min_behavior_actions: int = 50


QUANT_READING = {
    "hypothesis": (
        "The RedHatAI FP8 quant of Qwen3.6-27B (RedHatAI/Qwen3.6-27B-FP8, "
        "mounted as ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot) "
        "differs on the true objective from the vrfai FP8 quant every scored "
        "submission has served (driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot). "
        "Source: sonpham's public 5-way quant A/B found the RedHatAI quant "
        "best — EXTERNAL and unreplicated, on his own harness and games; "
        "treat it as a hypothesis source ONLY, not evidence, and expect it "
        "may not transfer."
    ),
    "reading_rule": (
        "Decision statistic: true_score_all_games (per game MAX over clones, "
        "per wave MEAN over the full 25 games; rows[].score as stored by "
        "pc_driver) for the quant arm MINUS the shipped comparator wave from "
        "the SAME session at the frozen geometry (28 clones, 7920s, "
        "concurrency 28, full 25 games). Directional verdict only if |delta| "
        ">= 0.2712 (the banked base pair's own repeat spread); otherwise "
        "INDISTINGUISHABLE. Secondary (report, never gate): "
        "true_score_excl_ft09, per-game deltas, levels."
    ),
    "treatment": (
        "EXACTLY one treatment vs the shipped arm, applied at build time: the "
        "served FP8 snapshot swaps vrfai -> RedHatAI (kernel dataset mount + "
        "duck-base cell 6 DATASET_SOURCES + in-memory rewrite of "
        "setup_commands.json cmd 0's MODEL_OWNER / MODEL_SLUG / "
        "SERVED_MODEL_NAME anchors, each hard-asserted to fire exactly once "
        "BEFORE any setup command runs). SERVED_MODEL_NAME flows to "
        "LOCAL_ANALYZER_MODEL_ID, so the serving probe proves the swap "
        "against /models before any game minute."
    ),
    "power_honesty": (
        "One wave per arm can only detect a LARGE offline difference: the "
        "same-arm repeat spread at this geometry is 0.2712. A null "
        "(INDISTINGUISHABLE) does NOT establish quant equivalence, and a "
        "directional result here is an offline screen, not a slot decision — "
        "that needs the live distribution (base n=10 mean 0.965 sd 0.208, "
        "one-slot MDE 0.61)."
    ),
    "not_a_readout": (
        "Do not read levels: the level count correlates with the objective at "
        "r = -0.009 across the eight banked waves. Do not promote either "
        "quant off this screen alone."
    ),
    "banked_controls_context": (
        "The two banked closure-base waves (true_score_all_games 1.4751 and "
        "1.2039, PATCHED BASE_ENV arms at identical geometry) are context "
        "only: report where the quant wave falls relative to [1.2039, 1.4751] "
        "but decide ONLY on the same-session quant-minus-shipped delta."
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


def _purity(result: dict, arm: str, proof_key: str) -> list[str]:
    """Both arms are UNPATCHED by contract: absence proof present + zero
    patch-mechanism activity. Any hit means the registered comparison
    (quant-vs-shipped on the SAME unpatched config) did not run."""
    problems: list[str] = []
    proof = (result.get("identity") or {}).get("patch_proof") or {}
    if proof.get(proof_key) is not True:
        problems.append(
            f"{arm}: identity.patch_proof.{proof_key} is not True "
            "— the absence proof did not run (or patches leaked in)")
    contamination = {
        "animation_payload_deliveries": _diag(result, "animation", "payload_deliveries"),
        "graph_sessions": _diag(result, "graph", "sessions_with_graph_state"),
        "watchdog_stall_kills": _diag(result, "watchdog", "stall_kills"),
        "watchdog_recovery_resets": _diag(result, "watchdog", "recovery_resets"),
    }
    stalls = ((result.get("patch_diagnostics") or {}).get("watchdog")
              or {}).get("stall_s_observed") or []
    if any(contamination.values()) or stalls:
        problems.append(
            f"{arm}: patch-mechanism activity detected: {contamination} "
            f"stall_s_observed={stalls} — this is not the unpatched config")
    return problems


def _serving_identity(result: dict, arm: str, expected: str, forbidden: str) -> list[str]:
    """The treatment IS the served snapshot, so serving identity is a
    contract, not a diagnostic: the wrong (or an unproven) serve makes the
    registered comparison meaningless."""
    serving = (result.get("identity") or {}).get("serving")
    if not isinstance(serving, dict):
        return [f"{arm}: identity.serving is missing — the serve was never proven"]
    problems: list[str] = []
    if serving.get("expected_id") != expected:
        problems.append(
            f"{arm}: served model {serving.get('expected_id')!r} != {expected!r}")
    if serving.get("ok") is not True:
        problems.append(f"{arm}: serving probe did not report ok: {serving!r}")
    served_ids = serving.get("served_ids") or []
    if forbidden in served_ids:
        problems.append(
            f"{arm}: /models serves the forbidden alias {forbidden!r}: {served_ids}")
    if expected not in served_ids:
        problems.append(
            f"{arm}: /models does not serve {expected!r}: {served_ids}")
    return problems


def classify_quant(
    quant: dict, shipped: dict, thresholds: QuantThresholds = QuantThresholds()
) -> dict:
    """Pre-registered classification of a quant/shipped result pair.

    INVALID        — contract violation (wrong arm labels/env/hashes, patch
                     contamination in either arm, or a wrong/unproven served
                     model — either way the registered comparison did not run).
    INFRA_FAILURE  — contract OK but an arm did not complete as an instrument.
    QUANT_AHEAD / SHIPPED_AHEAD — |delta| >= min_effect_true_score.
    INDISTINGUISHABLE — valid measurement, delta below the pre-registered
                     minimum effect (NOT evidence of equality; see
                     power_honesty in QUANT_READING).
    """
    invalid: list[str] = []
    infra: list[str] = []
    pair = {"quant": quant, "shipped": shipped}

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
        if quant["hypothesis"] != QUANT_HYPOTHESIS:
            invalid.append(
                f"quant: hypothesis {quant['hypothesis']!r} != {QUANT_HYPOTHESIS!r}")
        if quant["arm_env"] != QUANT_ENV:
            invalid.append(
                f"quant: arm_env must be exactly {QUANT_ENV!r} (the inert "
                f"fingerprint pin, nothing else), got {quant['arm_env']!r}")
        if quant["patch_sha256"] != UNPATCHED_SENTINEL:
            invalid.append(
                f"quant: patch_sha256 {quant['patch_sha256']!r} != {UNPATCHED_SENTINEL!r} "
                "— the quant arm must not carry inlined patch bytes")
        if shipped["hypothesis"] != SHIPPED_HYPOTHESIS:
            invalid.append(
                f"shipped: hypothesis {shipped['hypothesis']!r} != {SHIPPED_HYPOTHESIS!r} "
                "(the comparator is the EXISTING shipped-screen kernel, unchanged)")
        if shipped["arm_env"] != SHIPPED_ENV:
            invalid.append(
                f"shipped: arm_env must be EMPTY (no pins), got {shipped['arm_env']!r}")
        if shipped["patch_sha256"] != UNPATCHED_SENTINEL:
            invalid.append(
                f"shipped: patch_sha256 {shipped['patch_sha256']!r} != "
                f"{UNPATCHED_SENTINEL!r} — a patched comparator is not the "
                "shipped config")

        # -- pair coherence ----------------------------------------------------
        if quant["source_base_sha256"] != shipped["source_base_sha256"]:
            invalid.append(
                "arms disagree on source_base_sha256: "
                f"{quant['source_base_sha256']!r} != {shipped['source_base_sha256']!r}")
        if quant["geometry"] != shipped["geometry"]:
            invalid.append(
                f"arms disagree on geometry: {quant['geometry']!r} != {shipped['geometry']!r}")
        both_dry = bool(quant.get("dry_run")) and bool(shipped.get("dry_run"))
        if quant["geometry"] != GEOMETRY and not both_dry:
            invalid.append(
                f"geometry {quant['geometry']!r} != registered {GEOMETRY!r} "
                "and the pair is not a dry run")

        # -- purity: BOTH arms must show ZERO patch-mechanism activity ---------
        invalid += _purity(quant, "quant", "quant_no_patch_markers")
        invalid += _purity(shipped, "shipped", "shipped_no_patch_markers")

        # -- serving identity: the treatment itself ----------------------------
        # (skipped only when BOTH results are dry runs — no vLLM exists there)
        if not both_dry:
            invalid += _serving_identity(
                quant, "quant", QUANT_SERVED_MODEL, VRFAI_SERVED_MODEL)
            invalid += _serving_identity(
                shipped, "shipped", VRFAI_SERVED_MODEL, QUANT_SERVED_MODEL)

    if invalid:
        return {"state": "INVALID", "reasons": invalid,
                "thresholds": asdict(thresholds), "hypothesis": QUANT_HYPOTHESIS}

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
    ts_quant = true_score_metrics(quant, excluded=excl)
    ts_shipped = true_score_metrics(shipped, excluded=excl)
    per_quant = per_game_true_score(quant)
    per_shipped = per_game_true_score(shipped)
    per_game_delta = {
        g: round(per_quant.get(g, 0.0) - per_shipped.get(g, 0.0), 4)
        for g in sorted(set(per_quant) | set(per_shipped))
    }
    q_all, s_all = ts_quant["true_score_all_games"], ts_shipped["true_score_all_games"]
    delta_all = (round(q_all - s_all, 4)
                 if q_all is not None and s_all is not None else None)
    q_ex, s_ex = ts_quant["true_score_excl"], ts_shipped["true_score_excl"]
    banked = BANKED_BASE_CONTROLS["true_score_all_games_by_wave"]
    metrics = {
        "true_score": {"quant": ts_quant, "shipped": ts_shipped},
        "delta_all_games_quant_minus_shipped": delta_all,
        "delta_excl_ft09_quant_minus_shipped": (
            round(q_ex - s_ex, 4) if q_ex is not None and s_ex is not None else None),
        "per_game_delta": per_game_delta,
        "serving_identity": {
            "quant": (quant.get("identity") or {}).get("serving"),
            "shipped": (shipped.get("identity") or {}).get("serving"),
        },
        "banked_base_band_context": {
            "note": "patched BASE_ENV waves — context only, never a verdict input",
            "true_score_all_games_by_wave": banked,
            "quant_vs_band": (
                None if q_all is None else
                ("above" if q_all > max(banked) else
                 "below" if q_all < min(banked) else "inside")),
        },
        "levels_excl_ft09_secondary": {
            "quant": _levels_sum(quant, excl),
            "shipped": _levels_sum(shipped, excl),
            "warning": "r = -0.009 with the objective — never a verdict input",
        },
    }

    if infra:
        return {"state": "INFRA_FAILURE", "reasons": infra, "metrics": metrics,
                "thresholds": asdict(thresholds), "hypothesis": QUANT_HYPOTHESIS}

    # --- the frozen decision ----------------------------------------------------
    reasons: list[str] = []
    if delta_all is None:
        return {"state": "INFRA_FAILURE",
                "reasons": ["no rows carried a stored score — cannot compute the objective"],
                "metrics": metrics, "thresholds": asdict(thresholds),
                "hypothesis": QUANT_HYPOTHESIS}
    if abs(delta_all) >= thresholds.min_effect_true_score:
        state = "QUANT_AHEAD" if delta_all > 0 else "SHIPPED_AHEAD"
        reasons.append(
            f"|delta true_score_all_games| = {abs(delta_all)} >= "
            f"{thresholds.min_effect_true_score} (quant {q_all} vs shipped {s_all})")
    else:
        state = "INDISTINGUISHABLE"
        reasons.append(
            f"|delta true_score_all_games| = {abs(delta_all)} < "
            f"{thresholds.min_effect_true_score} (quant {q_all} vs shipped {s_all}) "
            "— below the banked same-arm repeat spread; NOT evidence of equality")

    return {"state": state, "reasons": reasons, "metrics": metrics,
            "thresholds": asdict(thresholds), "hypothesis": QUANT_HYPOTHESIS,
            "pre_registered_reading_present": "pre_registered_reading" in quant}
