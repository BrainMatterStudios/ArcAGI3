"""MEM-STACK SCREEN — frozen arm contract, pre-registration and classifier.

THE QUESTION. The duck-mem P1-P4 "anti-waste" stack (submission/_duck_mem/
harness_mem.py: token estimator //4, middle-drop history trimmer, prompt
own-goal neutralization, repeated-no-effect batch guard) was built for a live
slot on top of the SHIPPED duck-base v2 config. It has never been measured
offline against that same shipped config on the true objective. This screen
runs the pair in the rig.

THE ARMS.
  mem     — duck-base with the harness_mem.py stack inlined and applied via
            apply_all(BUNDLE_DIR), P5/P6 gated OFF (DUCK_MEM_P5/P6 pinned "0"),
            exactly as the duck-mem notebook hook applies it. NO duck_patches.
            The hook hard-asserts duck_patches symbols ABSENT (the shipped
            screen's mirror guard) AND the four [duck-mem] P1-P4 markers
            PRINTED. Instrumentation (behavioural probe + pc_driver) is
            byte-identical to every other arm, plus two observe-only counters
            (guard fires, trims) surfaced at patch_diagnostics.mem.
  shipped — the EXISTING, UNCHANGED shipped-screen kernel
            (arc-agi-3-patch-closure-shipped): duck-base v2, no patches, no
            env pins. A fresh wave from the same session is the primary
            comparator; the banked base waves band the instrument.

`classify_mem(mem, shipped)` is the pre-registered analysis. States:
INVALID / INFRA_FAILURE / MEM_AHEAD / SHIPPED_AHEAD / INDISTINGUISHABLE.

READING EMPHASIS (pre-registered, not a post-hoc gloss): the PRIMARY statistic
(true_score delta) is nearly blind at one wave per arm — the banked same-arm
repeat spread is 0.2712. The SECONDARY BEHAVIOURAL readouts are the sensitive
instrument for whether the stack DID anything: LLM turns per game, trim
counts, guard fires (each fire is one stop_reason=repeated_no_effect batch
stop), gen_tokens. They are reported with emphasis and NEVER gate the primary
verdict.
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
)

MEM_HYPOTHESIS = "mem-stack-screen-2026-08-13"

# The mem arm pins ONLY the two duck-mem gates, both OFF — the default P1-P4
# stack, exactly what the duck-mem notebook ships. Deliberately non-empty so
# the arm-env fingerprint can never collide with SHIPPED_ENV ({}) or the
# closure arms' TAAF_* mappings.
MEM_ENV: dict[str, str] = {"DUCK_MEM_P5": "0", "DUCK_MEM_P6": "0"}

# The exact lines harness_mem.apply_all prints for the default (P1-P4) stack.
# The hook captures apply_all's stdout and hard-asserts every one of these —
# a silently-degraded patch (harness_mem is fail-safe by design) can never
# masquerade as the measured arm.
MEM_MARKER_LINES = (
    "[duck-mem] P1 estimator //4: OK",
    "[duck-mem] P2 middle-drop trimmer: OK",
    "[duck-mem] P3 prompt own-goal neutralized (both namespaces): OK",
    "[duck-mem] P4 repeated-no-effect guard: OK",
    "[duck-mem] P5 yield: GATED OFF",
    "[duck-mem] P6 memory expansion: GATED OFF",
    "[duck-mem] patch stack applied: 4/4",
)

# identity.patch_proof keys the mem prelude must report True (pc_main aborts
# on any False boolean; the classifier re-asserts them from the artifact).
MEM_PROOF_KEYS = (
    "mem_no_duck_patches_markers",
    "mem_p1_estimator",
    "mem_p2_middle_drop",
    "mem_p3_prompt_neutralized",
    "mem_p4_no_effect_guard",
)

# The banked base pair's own between-wave spread on the decision statistic
# (|1.4751 - 1.2039| = 0.2712) — the same floor the shipped screen registered.
_BANKED_SPREAD = round(abs(BANKED_BASE_CONTROLS["true_score_all_games_by_wave"][0]
                           - BANKED_BASE_CONTROLS["true_score_all_games_by_wave"][1]), 4)


@dataclass(frozen=True)
class MemThresholds:
    """Pre-registered decision boundaries. Editing these after the GPU data
    lands is goalpost-moving and must show up as a diff to this frozen file."""

    # Directional call (MEM_AHEAD / SHIPPED_AHEAD) requires |delta| at least
    # this large on true_score_all_games; below it the verdict is
    # INDISTINGUISHABLE. Set to the banked base pair's own repeat spread.
    min_effect_true_score: float = _BANKED_SPREAD
    # Fewer probe-observed actions than this means an arm never really played.
    min_behavior_actions: int = 50


MEM_READING = {
    "hypothesis": (
        "The duck-mem P1-P4 anti-waste stack applied on top of the shipped "
        "duck-base v2 config changes behaviour (context use, batch waste, "
        "exploration) versus the unmodified shipped config; direction on the "
        "true objective is UNKNOWN — no prior measurement exists."
    ),
    "reading_rule": (
        "PRIMARY decision statistic: true_score_all_games (per game MAX over "
        "clones, per wave MEAN over the full 25 games; rows[].score as stored "
        "by pc_driver) for the mem arm MINUS the shipped arm from the SAME "
        "session at the frozen geometry (28 clones, 7920s, concurrency 28). "
        "Directional verdict only if |delta| >= 0.2712 (the banked base "
        "pair's own repeat spread); otherwise INDISTINGUISHABLE."
    ),
    "behavioral_emphasis": (
        "SECONDARY AND EMPHASIZED: at one wave per arm the primary statistic "
        "is nearly blind (same-arm spread 0.2712), so the behavioural "
        "readouts are the sensitive instrument for whether the stack DID "
        "anything: (a) LLM turns per game (adoption.llm_turns) and actions "
        "per LLM turn; (b) trim count (patch_diagnostics.mem.trims — every "
        "P2 middle-drop eviction); (c) guard fires "
        "(patch_diagnostics.mem.guard_fires — each fire is exactly one "
        "stop_reason=repeated_no_effect batch stop; stop_reason itself is "
        "not persisted in rows, the counter is its 1:1 observer); "
        "(d) gen_tokens per turn. These NEVER gate the primary verdict."
    ),
    "power_honesty": (
        "One wave per arm CANNOT resolve a small objective effect: the "
        "banked base pair's same-arm spread is 0.2712 at this exact "
        "geometry. This screen can only detect a LARGE offline difference "
        "or band the effect; INDISTINGUISHABLE here is NOT evidence the "
        "stack is inert — read the behavioural block for that."
    ),
    "not_a_readout": (
        "Do not read levels: the level count correlates with the objective "
        "at r = -0.009 across the eight banked waves. Do not promote either "
        "config off this screen alone; a slot decision needs the live "
        "distribution (base n=10 mean 0.965 sd 0.208, one-slot MDE 0.61). "
        "Note the live duck-mem slot result (if any) is a SINGLE draw from "
        "a 0.208-sd distribution — neither it nor this screen alone settles "
        "the stack."
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


def _duck_patches_contamination(result: dict) -> tuple[dict, list]:
    """Zero-activity proof that NO duck_patches mechanism ran in an arm.
    Both arms of this screen must be clean (neither carries duck_patches)."""
    contamination = {
        "animation_payload_deliveries": _diag(result, "animation", "payload_deliveries"),
        "graph_sessions": _diag(result, "graph", "sessions_with_graph_state"),
        "watchdog_stall_kills": _diag(result, "watchdog", "stall_kills"),
        "watchdog_recovery_resets": _diag(result, "watchdog", "recovery_resets"),
    }
    stalls = ((result.get("patch_diagnostics") or {}).get("watchdog")
              or {}).get("stall_s_observed") or []
    return contamination, stalls


def _gen_tokens_total(result: dict) -> int:
    total = 0
    for row in (result.get("rows") or []):
        try:
            total += max(0, int(row.get("gen_tokens", 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def _actions_total(result: dict) -> int:
    total = 0
    for row in (result.get("rows") or []):
        try:
            total += max(0, int(row.get("actions_total", 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def _behavioral_block(mem: dict, shipped: dict) -> dict:
    """The EMPHASIZED secondary readouts (see MEM_READING.behavioral_emphasis).
    Report-only: nothing here feeds the primary verdict."""
    out = {"emphasis": ("SECONDARY AND EMPHASIZED — the sensitive readouts at "
                        "this geometry; they never gate the primary verdict")}
    per_arm = {}
    for name, result in (("mem", mem), ("shipped", shipped)):
        adoption = result.get("adoption") or {}
        n_games = len(result.get("rows_by_source") or {}) or None
        llm_turns = adoption.get("llm_turns")
        gen_tokens = _gen_tokens_total(result)
        actions = _actions_total(result)
        corpus = (result.get("behavior") or {}).get("corpus") or {}
        per_arm[name] = {
            "llm_turns": llm_turns,
            "llm_turns_per_game": (round(llm_turns / n_games, 2)
                                   if llm_turns is not None and n_games else None),
            "actions_per_llm_turn": adoption.get("actions_per_llm_turn"),
            "actions_total": actions,
            "gen_tokens_total": gen_tokens,
            "gen_tokens_per_llm_turn": (round(gen_tokens / llm_turns, 1)
                                        if llm_turns else None),
            "probe_actions_per_turn": corpus.get("actions_per_turn"),
        }
    mem_diag = (mem.get("patch_diagnostics") or {}).get("mem") or {}
    guard_fires = mem_diag.get("guard_fires")
    mem_actions = per_arm["mem"]["actions_total"]
    out["per_arm"] = per_arm
    out["mem_stack_activity"] = {
        "trims": mem_diag.get("trims"),
        "guard_fires": guard_fires,
        "guard_fires_per_1000_actions": (
            round(1000.0 * guard_fires / mem_actions, 2)
            if isinstance(guard_fires, int) and mem_actions else None),
        "note": ("guard_fires is the 1:1 observer of "
                 "stop_reason=repeated_no_effect batch stops (P4); trims counts "
                 "P2 middle-drop evictions. Zero activity on the shipped arm by "
                 "construction — it carries no counters."),
    }
    deltas = {}
    for key in ("llm_turns", "llm_turns_per_game", "actions_per_llm_turn",
                "actions_total", "gen_tokens_total", "gen_tokens_per_llm_turn"):
        a, b = per_arm["mem"].get(key), per_arm["shipped"].get(key)
        deltas[key] = (round(a - b, 2)
                       if isinstance(a, (int, float)) and isinstance(b, (int, float))
                       else None)
    out["delta_mem_minus_shipped"] = deltas
    return out


def classify_mem(
    mem: dict, shipped: dict, thresholds: MemThresholds = MemThresholds()
) -> dict:
    """Pre-registered classification of a mem/shipped result pair.

    INVALID        — contract violation (wrong arm labels/env/hashes, a
                     duck_patches leak in either arm, a mem arm whose P1-P4
                     proof is not all-True — the registered comparison did
                     not run).
    INFRA_FAILURE  — contract OK but an arm did not complete as an
                     instrument (including absent mem counters).
    MEM_AHEAD / SHIPPED_AHEAD — |delta| >= min_effect_true_score.
    INDISTINGUISHABLE — valid measurement, delta below the pre-registered
                     minimum effect (NOT evidence of equality or inertness;
                     see power_honesty / behavioral_emphasis in MEM_READING).
    """
    invalid: list[str] = []
    infra: list[str] = []
    pair = {"mem": mem, "shipped": shipped}

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
        if mem["hypothesis"] != MEM_HYPOTHESIS:
            invalid.append(f"mem: hypothesis {mem['hypothesis']!r} != {MEM_HYPOTHESIS!r}")
        if mem["arm_env"] != MEM_ENV:
            invalid.append(f"mem: arm_env must be the frozen gate pins {MEM_ENV}, "
                           f"got {mem['arm_env']!r}")
        if not _HEX64.match(str(mem["patch_sha256"] or "")):
            invalid.append(
                f"mem: patch_sha256 {mem['patch_sha256']!r} is not a real inlined "
                "harness_mem hash — the mem arm must carry the inlined stack bytes")
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
                f"{UNPATCHED_SENTINEL!r} — the shipped arm must not carry patch bytes")

        # -- pair coherence ----------------------------------------------------
        if mem["source_base_sha256"] != shipped["source_base_sha256"]:
            invalid.append(
                "arms disagree on source_base_sha256: "
                f"{mem['source_base_sha256']!r} != {shipped['source_base_sha256']!r}")
        if mem["geometry"] != shipped["geometry"]:
            invalid.append(
                f"arms disagree on geometry: {mem['geometry']!r} != {shipped['geometry']!r}")
        both_dry = bool(mem.get("dry_run")) and bool(shipped.get("dry_run"))
        if mem["geometry"] != GEOMETRY and not both_dry:
            invalid.append(
                f"geometry {mem['geometry']!r} != registered {GEOMETRY!r} "
                "and the pair is not a dry run")

        # -- mem arm: duck_patches ABSENT, duck-mem P1-P4 PRESENT --------------
        proof = (mem.get("identity") or {}).get("patch_proof") or {}
        bad_proof = [k for k in MEM_PROOF_KEYS if proof.get(k) is not True]
        if bad_proof:
            invalid.append(
                f"mem: identity.patch_proof keys not True: {bad_proof} — the "
                "P1-P4 presence/absence proof did not run clean")
        contamination, stalls = _duck_patches_contamination(mem)
        if any(contamination.values()) or stalls:
            invalid.append(
                f"mem: duck_patches mechanism activity detected: {contamination} "
                f"stall_s_observed={stalls} — this is not the mem-stack-only config")
        # -- shipped arm purity (same checks the shipped screen registers) -----
        shipped_proof = (shipped.get("identity") or {}).get("patch_proof") or {}
        if shipped_proof.get("shipped_no_patch_markers") is not True:
            invalid.append(
                "shipped: identity.patch_proof.shipped_no_patch_markers is not True "
                "— the absence proof did not run (or patches leaked in)")
        s_contamination, s_stalls = _duck_patches_contamination(shipped)
        if any(s_contamination.values()) or s_stalls:
            invalid.append(
                f"shipped: patch-mechanism activity detected: {s_contamination} "
                f"stall_s_observed={s_stalls} — this is not the unpatched config")

    if invalid:
        return {"state": "INVALID", "reasons": invalid,
                "thresholds": asdict(thresholds), "hypothesis": MEM_HYPOTHESIS}

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
    mem_diag = (mem.get("patch_diagnostics") or {}).get("mem")
    if not isinstance(mem_diag, dict) or not {"guard_fires", "trims"} <= set(mem_diag):
        infra.append(
            "mem: patch_diagnostics.mem counters absent — the observe-only "
            "instrumentation (guard_fires/trims) did not install; the emphasized "
            "behavioural readouts cannot be trusted")

    # --- pre-registered metrics -------------------------------------------------
    from true_score import per_game_true_score, true_score_metrics  # local: avoids a cycle

    excl = tuple(LEVELS_EXCLUDED_GAMES)
    ts_mem = true_score_metrics(mem, excluded=excl)
    ts_shipped = true_score_metrics(shipped, excluded=excl)
    per_mem = per_game_true_score(mem)
    per_shipped = per_game_true_score(shipped)
    per_game_delta = {
        g: round(per_mem.get(g, 0.0) - per_shipped.get(g, 0.0), 4)
        for g in sorted(set(per_mem) | set(per_shipped))
    }
    m_all, s_all = ts_mem["true_score_all_games"], ts_shipped["true_score_all_games"]
    delta_all = (round(m_all - s_all, 4)
                 if m_all is not None and s_all is not None else None)
    m_ex, s_ex = ts_mem["true_score_excl"], ts_shipped["true_score_excl"]
    banked = BANKED_BASE_CONTROLS["true_score_all_games_by_wave"]
    metrics = {
        # EMPHASIZED secondary block FIRST — the sensitive readouts.
        "behavioral_EMPHASIZED": _behavioral_block(mem, shipped),
        "true_score": {"mem": ts_mem, "shipped": ts_shipped},
        "delta_all_games_mem_minus_shipped": delta_all,
        "delta_excl_ft09_mem_minus_shipped": (
            round(m_ex - s_ex, 4) if m_ex is not None and s_ex is not None else None),
        "per_game_delta": per_game_delta,
        "banked_base_band": {
            "true_score_all_games_by_wave": banked,
            "mem_vs_band": (
                None if m_all is None else
                ("above" if m_all > max(banked) else
                 "below" if m_all < min(banked) else "inside")),
        },
    }

    if infra:
        return {"state": "INFRA_FAILURE", "reasons": infra, "metrics": metrics,
                "thresholds": asdict(thresholds), "hypothesis": MEM_HYPOTHESIS}

    # --- the frozen decision ----------------------------------------------------
    reasons: list[str] = []
    if delta_all is None:
        return {"state": "INFRA_FAILURE",
                "reasons": ["no rows carried a stored score — cannot compute the objective"],
                "metrics": metrics, "thresholds": asdict(thresholds),
                "hypothesis": MEM_HYPOTHESIS}
    if abs(delta_all) >= thresholds.min_effect_true_score:
        state = "MEM_AHEAD" if delta_all > 0 else "SHIPPED_AHEAD"
        reasons.append(
            f"|delta true_score_all_games| = {abs(delta_all)} >= "
            f"{thresholds.min_effect_true_score} (mem {m_all} vs shipped {s_all})")
    else:
        state = "INDISTINGUISHABLE"
        reasons.append(
            f"|delta true_score_all_games| = {abs(delta_all)} < "
            f"{thresholds.min_effect_true_score} (mem {m_all} vs shipped {s_all}) "
            "— below the banked same-arm repeat spread; NOT evidence of equality. "
            "Read metrics.behavioral_EMPHASIZED for the sensitive readouts.")

    return {"state": state, "reasons": reasons, "metrics": metrics,
            "thresholds": asdict(thresholds), "hypothesis": MEM_HYPOTHESIS,
            "pre_registered_reading_present": "pre_registered_reading" in mem}
