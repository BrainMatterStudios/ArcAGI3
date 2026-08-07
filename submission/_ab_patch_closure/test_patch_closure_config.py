"""Frozen-contract tests for the patch-closure gate (Task 1 of the 2026-08-04 plan).

The arm mapping and the classifier decision boundaries are PRE-REGISTERED here:
these tests are the registration. Changing a threshold after the GPU data lands
means editing a frozen test — visible in the diff, deliberate, never silent.

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_patch_closure_config.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from patch_closure_config import (  # noqa: E402
    ARM_ENV,
    BASE_ENV,
    CANDIDATE_ENV,
    GEOMETRY,
    HYPOTHESIS,
    OFFICIAL_GAMES,
    SCHEMA_VERSION,
    classify_result,
)


# --- the arm contract (frozen) --------------------------------------------------


def test_candidate_delta_is_exactly_the_approved_three_mechanisms():
    changed = {k for k in BASE_ENV | CANDIDATE_ENV if BASE_ENV.get(k) != CANDIDATE_ENV.get(k)}
    assert changed == {"TAAF_WATCHDOG_STALL_S", "TAAF_ANIMATION", "TAAF_GRAPH"}


def test_base_reconstructs_v7_changed_defaults():
    assert BASE_ENV["TAAF_WATCHDOG_STALL_S"] == "900"
    assert BASE_ENV["TAAF_ANIMATION"] == "0"
    assert BASE_ENV["TAAF_GRAPH"] == "0"


def test_candidate_pins_grinder_parameters():
    assert CANDIDATE_ENV["TAAF_GRAPH_GRIND_AGE_ACTIONS"] == "120"
    assert CANDIDATE_ENV["TAAF_GRAPH_GRIND_AGE_TURNS"] == "10"
    assert CANDIDATE_ENV["TAAF_GRAPH_GRIND_MAX_PER_LEVEL"] == "2"


def test_arm_env_maps_both_arms_to_the_frozen_dicts():
    assert ARM_ENV == {"base": BASE_ENV, "candidate": CANDIDATE_ENV}


# --- decision boundaries (pre-registered) --------------------------------------


class _Pair:
    def __init__(self, base: dict, candidate: dict) -> None:
        self.base = base
        self.candidate = candidate


def _valid_result(arm: str) -> dict:
    """A complete, contract-conforming result with EQUAL (zero-unlock) levels —
    the same shape the kernel and dry run emit. Default verdict: NO_GO."""
    rows = []
    rows_by_source: dict[str, dict] = {}
    for i in range(GEOMETRY["clones"]):
        stem = OFFICIAL_GAMES[i % len(OFFICIAL_GAMES)]
        levels = 4 if stem == "ft09" else (2 if stem in ("su15", "tu93") else 0)
        rows.append({
            "clone_id": f"k{i:03d}", "source_game": stem,
            "levels_completed": levels, "levels_total": 6,
            "actions_total": 80, "state": "GameRunState.DONE",
        })
        slot = rows_by_source.setdefault(stem, {"levels": 0, "levels_total": 6, "n_clones": 0})
        slot["levels"] = max(slot["levels"], levels)
        slot["n_clones"] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": HYPOTHESIS,
        "arm": arm,
        "arm_env": dict(ARM_ENV[arm]),
        "source_base_sha256": "a" * 64,
        "patch_sha256": "b" * 64,
        "geometry": dict(GEOMETRY),
        "identity": {
            "arm": arm, "toggles_ok": True,
            "watchdog_stall_s_observed": [900.0 if arm == "base" else 600.0],
            "patch_proof": {"watchdog_should_stop_patched": True},
        },
        "rows": rows,
        "rows_by_source": rows_by_source,
        "behavior": {"corpus": {"actions": 1600, "noop_masked": 0.4}},
        "patch_diagnostics": {
            "animation": {
                "payload_deliveries": 0 if arm == "base" else 12,
                "frames_delivered": 0 if arm == "base" else 60,
            },
            "graph": {
                "grinder_engagements": 0 if arm == "base" else 1,
                "grinder_age_triggers": 0 if arm == "base" else 1,
                "levels_unlocked_by_grinder": 0,
                "narrations_injected": 0 if arm == "base" else 3,
                "vetoes_issued": 0 if arm == "base" else 2,
            },
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0, "recovery_resets": 1},
            "antifreeze": {"triggers": 0},
        },
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


@pytest.fixture
def valid_pair() -> _Pair:
    return _Pair(_valid_result("base"), _valid_result("candidate"))


def test_valid_equal_pair_defaults_to_no_go(valid_pair):
    out = classify_result(valid_pair.base, valid_pair.candidate)
    assert out["state"] == "NO_GO"
    assert out["metrics"]["new_target_unlocks"] == []


def test_go_on_two_new_target_unlocks(valid_pair):
    valid_pair.candidate["rows_by_source"]["dc22"]["levels"] = 1
    valid_pair.candidate["rows_by_source"]["m0r0"]["levels"] = 1
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "GO"


def test_one_unlock_is_not_enough(valid_pair):
    valid_pair.candidate["rows_by_source"]["dc22"]["levels"] = 1
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "NO_GO"


def test_regression_veto_beats_unlocks(valid_pair):
    valid_pair.candidate["rows_by_source"]["dc22"]["levels"] = 1
    valid_pair.candidate["rows_by_source"]["m0r0"]["levels"] = 1
    valid_pair.base["rows_by_source"]["tu93"]["levels"] = 3
    valid_pair.candidate["rows_by_source"]["tu93"]["levels"] = 0
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "NO_GO"


def test_no_go_when_su15_regresses_two_levels(valid_pair):
    valid_pair.base["rows_by_source"]["su15"]["levels"] = 3
    valid_pair.candidate["rows_by_source"]["su15"]["levels"] = 1
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "NO_GO"


def test_one_level_regression_does_not_veto(valid_pair):
    valid_pair.candidate["rows_by_source"]["dc22"]["levels"] = 1
    valid_pair.candidate["rows_by_source"]["m0r0"]["levels"] = 1
    valid_pair.base["rows_by_source"]["su15"]["levels"] = 3
    valid_pair.candidate["rows_by_source"]["su15"]["levels"] = 2
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "GO"


def test_invalid_when_identity_proof_is_missing(valid_pair):
    del valid_pair.candidate["identity"]
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INVALID"


def test_invalid_when_arm_env_drifts(valid_pair):
    valid_pair.candidate["arm_env"]["TAAF_GRAPH_GRIND_AGE_ACTIONS"] = "60"
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INVALID"


def test_invalid_when_base_shows_candidate_mechanism_activity(valid_pair):
    valid_pair.base["patch_diagnostics"]["animation"]["payload_deliveries"] = 3
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INVALID"


def test_invalid_when_arms_disagree_on_source_hash(valid_pair):
    valid_pair.candidate["source_base_sha256"] = "c" * 64
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INVALID"


def test_invalid_on_non_dry_run_geometry_drift(valid_pair):
    for r in (valid_pair.base, valid_pair.candidate):
        r["geometry"] = {"clones": 28, "per_game_s": 600, "concurrency": 28}
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INVALID"


def test_dry_run_pair_may_use_reduced_geometry(valid_pair):
    for r in (valid_pair.base, valid_pair.candidate):
        r["geometry"] = {"clones": 28, "per_game_s": 45, "concurrency": 28}
        r["dry_run"] = True
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "NO_GO"


def test_infra_failure_on_recorded_error(valid_pair):
    valid_pair.candidate["error"] = "Traceback: boom"
    valid_pair.candidate["stage"] = "run"
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INFRA_FAILURE"


def test_infra_failure_on_short_rows(valid_pair):
    valid_pair.base["rows"] = valid_pair.base["rows"][:5]
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INFRA_FAILURE"


def test_infra_failure_when_probe_saw_no_play(valid_pair):
    valid_pair.candidate["behavior"]["corpus"]["actions"] = 3
    assert classify_result(valid_pair.base, valid_pair.candidate)["state"] == "INFRA_FAILURE"


def test_metrics_echo_mechanism_uptake(valid_pair):
    out = classify_result(valid_pair.base, valid_pair.candidate)
    assert out["metrics"]["animation_uptake"]["candidate_payload_deliveries"] == 12
    assert out["metrics"]["grinder"]["engagements"] == 1
    assert out["metrics"]["stale_closes"] == {"base": 0, "candidate": 0}


def test_classify_cli_round_trip(tmp_path, valid_pair, capsys):
    import json

    import classify

    b = tmp_path / "base.json"
    c = tmp_path / "candidate.json"
    b.write_text(json.dumps(valid_pair.base))
    c.write_text(json.dumps(valid_pair.candidate))
    rc = classify.main([str(b), str(c)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["state"] == "NO_GO"

    del valid_pair.candidate["identity"]
    c.write_text(json.dumps(valid_pair.candidate))
    rc = classify.main([str(b), str(c)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2 and out["state"] == "INVALID"
