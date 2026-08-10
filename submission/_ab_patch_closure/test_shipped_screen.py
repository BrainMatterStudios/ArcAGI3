"""Frozen-contract + builder + classifier + dry-run tests for the SHIPPED
screen (the arm that reproduces duck-base v2: NO patches, NO env pins).

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_shipped_screen.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_patch_closure import build_arm, notebook_contract  # noqa: E402
from build_shipped_screen import (  # noqa: E402
    SHIPPED_HOOK,
    SHIPPED_PRELUDE,
    SHIPPED_SLUG,
    build_shipped,
)
from package_screen_config import PACKAGE_HYPOTHESIS  # noqa: E402
from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    CANDIDATE_ENV,
    GEOMETRY,
    HYPOTHESIS as CLOSURE_HYPOTHESIS,
    OFFICIAL_GAMES,
)
from shipped_screen_config import (  # noqa: E402
    BANKED_BASE_CONTROLS,
    SHIPPED_ENV,
    SHIPPED_HYPOTHESIS,
    SHIPPED_READING,
    UNPATCHED_SENTINEL,
    ShippedThresholds,
    classify_shipped,
)

REPO = Path(__file__).resolve().parents[2]

# Byte sequences that exist ONLY if duck_patches source was inlined. The guard
# and prelude may NAME symbols (e.g. in an absence check), so definition
# markers are what a contaminated build cannot avoid carrying.
PATCH_DEFINITION_MARKERS = (
    "def apply_all(",
    "def patch_action7(",
    "def patch_watchdog(",
    "def patch_win_replay(",
    "def patch_hud_board_identity(",
    "def patch_frontier_graph(",
    "class HudMaskTracker",
    "_pc_patch_results = apply_all()",   # the patched arms' APPLY_BLOCK line
)


def _nb_text(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    return "\n@@CELL@@\n".join(
        "".join(c.get("source", [])) for c in nb["cells"])


def _env_fp(env: dict) -> str:
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:12]


# --- the frozen shipped contract --------------------------------------------------


def test_shipped_env_is_empty_and_hypothesis_distinct():
    assert SHIPPED_ENV == {}
    assert SHIPPED_HYPOTHESIS not in (CLOSURE_HYPOTHESIS, PACKAGE_HYPOTHESIS)
    assert UNPATCHED_SENTINEL == "UNPATCHED"


def test_env_fingerprint_cannot_collide_with_base_or_candidate():
    fps = {_env_fp(SHIPPED_ENV), _env_fp(BASE_ENV), _env_fp(CANDIDATE_ENV)}
    assert len(fps) == 3


def test_preregistration_content():
    assert "0.18" in SHIPPED_READING["hypothesis"]
    assert "true_score_all_games" in SHIPPED_READING["reading_rule"]
    assert "25 games" in SHIPPED_READING["reading_rule"]
    assert "0.2712" in SHIPPED_READING["reading_rule"]
    # power honesty: one wave per arm cannot resolve the live hint
    assert "CANNOT resolve" in SHIPPED_READING["power_honesty"]
    assert SHIPPED_READING["banked_base_controls"] is BANKED_BASE_CONTROLS


def test_banked_controls_match_the_frozen_geometry():
    assert BANKED_BASE_CONTROLS["geometry"] == GEOMETRY
    assert BANKED_BASE_CONTROLS["files"] == ["pc_base.json", "w2_base.json"]
    assert BANKED_BASE_CONTROLS["true_score_all_games_by_wave"] == [1.4751, 1.2039]
    # threshold is exactly the banked pair's own spread
    assert ShippedThresholds().min_effect_true_score == pytest.approx(0.2712)


def test_banked_control_true_scores_recompute_from_the_artifacts():
    """The frozen control numbers must recompute from the banked files (guards
    against a typo'd pre-registration). Skips if the artifacts are absent
    (they live in scratchpad, not git)."""
    from true_score import true_score_metrics

    root = REPO / "scratchpad/banked_waves_20260809"
    if not root.is_dir():
        pytest.skip("banked wave artifacts not present on this machine")
    for fname, expected in zip(
            BANKED_BASE_CONTROLS["files"],
            BANKED_BASE_CONTROLS["true_score_all_games_by_wave"]):
        wave = json.loads((root / fname).read_text())
        assert wave["geometry"] == GEOMETRY, fname
        got = true_score_metrics(wave)["true_score_all_games"]
        assert got == pytest.approx(expected), (fname, got, expected)


# --- classifier -------------------------------------------------------------------


def _rows(score: float, levels: int = 0) -> tuple[list, dict]:
    rows, by_source = [], {}
    for i in range(GEOMETRY["clones"]):
        stem = OFFICIAL_GAMES[i % len(OFFICIAL_GAMES)]
        rows.append({"clone_id": f"k{i:03d}", "source_game": stem,
                     "levels_completed": levels, "levels_total": 6,
                     "actions_total": 80, "score": score,
                     "state": "GameRunState.DONE"})
        slot = by_source.setdefault(stem, {"levels": 0, "levels_total": 6, "n_clones": 0})
        slot["levels"] = max(slot["levels"], levels)
        slot["n_clones"] += 1
    return rows, by_source


def _valid_shipped(score: float = 1.0) -> dict:
    rows, by_source = _rows(score)
    return {
        "schema_version": 1,
        "hypothesis": SHIPPED_HYPOTHESIS,
        "arm": "shipped",
        "arm_env": {},
        "source_base_sha256": "a" * 64,
        "patch_sha256": UNPATCHED_SENTINEL,
        "geometry": dict(GEOMETRY),
        "identity": {"arm": "shipped", "toggles_ok": True,
                     "patch_proof": {"shipped_no_patch_markers": True}},
        "rows": rows,
        "rows_by_source": by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "patch_diagnostics": {
            "animation": {"payload_deliveries": 0, "frames_delivered": 0},
            "graph": {"sessions_with_graph_state": 0},
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                         "recovery_resets": 0, "stall_s_observed": []},
        },
        "pre_registered_reading": dict(SHIPPED_READING),
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


def _valid_base(score: float = 1.0) -> dict:
    rows, by_source = _rows(score)
    return {
        "schema_version": 1,
        "hypothesis": CLOSURE_HYPOTHESIS,
        "arm": "base",
        "arm_env": dict(BASE_ENV),
        "source_base_sha256": "a" * 64,
        "patch_sha256": "b" * 64,
        "geometry": dict(GEOMETRY),
        "identity": {"arm": "base", "toggles_ok": True,
                     "patch_proof": {"watchdog_should_stop_patched": True},
                     "watchdog_stall_s_observed": [900.0]},
        "rows": rows,
        "rows_by_source": by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "patch_diagnostics": {
            "animation": {"payload_deliveries": 0, "frames_delivered": 0},
            "graph": {"sessions_with_graph_state": 28},
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                         "recovery_resets": 0, "stall_s_observed": [900.0]},
        },
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


def test_equal_scores_are_indistinguishable():
    out = classify_shipped(_valid_shipped(1.0), _valid_base(1.0))
    assert out["state"] == "INDISTINGUISHABLE"
    assert out["metrics"]["delta_all_games_shipped_minus_base"] == 0.0
    assert "NOT evidence of equality" in out["reasons"][0]


def test_shipped_ahead_at_the_preregistered_effect():
    out = classify_shipped(_valid_shipped(1.30), _valid_base(1.0))
    assert out["state"] == "SHIPPED_AHEAD"
    assert out["metrics"]["delta_all_games_shipped_minus_base"] == pytest.approx(0.30)


def test_base_ahead_at_the_preregistered_effect():
    out = classify_shipped(_valid_shipped(1.0), _valid_base(1.30))
    assert out["state"] == "BASE_AHEAD"


def test_sub_threshold_delta_is_indistinguishable():
    # 0.18 — exactly the live hint — is BELOW the banked repeat spread
    out = classify_shipped(_valid_shipped(1.18), _valid_base(1.0))
    assert out["state"] == "INDISTINGUISHABLE"


def test_banked_band_position_is_reported():
    out = classify_shipped(_valid_shipped(1.0), _valid_base(1.0))
    assert out["metrics"]["banked_base_band"]["shipped_vs_band"] == "below"
    out2 = classify_shipped(_valid_shipped(1.3), _valid_base(1.3))
    assert out2["metrics"]["banked_base_band"]["shipped_vs_band"] == "inside"


def test_invalid_on_shipped_env_pins():
    shipped = _valid_shipped()
    shipped["arm_env"] = {"TAAF_WATCHDOG": "1"}
    assert classify_shipped(shipped, _valid_base())["state"] == "INVALID"


def test_invalid_on_shipped_patch_contamination():
    shipped = _valid_shipped()
    shipped["patch_diagnostics"]["watchdog"]["stall_s_observed"] = [900.0]
    assert classify_shipped(shipped, _valid_base())["state"] == "INVALID"

    shipped2 = _valid_shipped()
    shipped2["identity"]["patch_proof"] = {}
    assert classify_shipped(shipped2, _valid_base())["state"] == "INVALID"

    shipped3 = _valid_shipped()
    shipped3["patch_sha256"] = "c" * 64
    assert classify_shipped(shipped3, _valid_base())["state"] == "INVALID"


def test_invalid_on_unpatched_base_arm():
    base = _valid_base()
    base["identity"]["patch_proof"] = {"shipped_no_patch_markers": True}
    out = classify_shipped(_valid_shipped(), base)
    assert out["state"] == "INVALID"
    assert any("A/A" in r for r in out["reasons"])


def test_invalid_on_source_hash_or_geometry_mismatch():
    base = _valid_base()
    base["source_base_sha256"] = "f" * 64
    assert classify_shipped(_valid_shipped(), base)["state"] == "INVALID"

    shipped, base2 = _valid_shipped(), _valid_base()
    shipped["geometry"] = {"clones": 4, "per_game_s": 90, "concurrency": 4}
    base2["geometry"] = dict(shipped["geometry"])
    assert classify_shipped(shipped, base2)["state"] == "INVALID"  # not a dry pair
    shipped["dry_run"] = base2["dry_run"] = True
    assert classify_shipped(shipped, base2)["state"] == "INDISTINGUISHABLE"


def test_infra_failure_on_error_short_rows_or_dead_probe():
    shipped = _valid_shipped()
    shipped["error"] = "Traceback: boom"
    assert classify_shipped(shipped, _valid_base())["state"] == "INFRA_FAILURE"

    shipped2 = _valid_shipped()
    shipped2["rows"] = shipped2["rows"][:5]
    assert classify_shipped(shipped2, _valid_base())["state"] == "INFRA_FAILURE"

    base = _valid_base()
    base["behavior"]["corpus"]["actions"] = 3
    assert classify_shipped(_valid_shipped(), base)["state"] == "INFRA_FAILURE"


def test_classify_shipped_cli_round_trip(tmp_path, capsys):
    import classify_shipped as cli

    s, b = tmp_path / "shipped.json", tmp_path / "base.json"
    s.write_text(json.dumps(_valid_shipped(1.0)))
    b.write_text(json.dumps(_valid_base(1.0)))
    rc = cli.main([str(s), str(b)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["state"] == "INDISTINGUISHABLE"


# --- builder ----------------------------------------------------------------------


def test_shipped_notebook_contains_no_patch_bytes(tmp_path):
    text = _nb_text(build_shipped(output_root=tmp_path))
    for marker in PATCH_DEFINITION_MARKERS:
        assert marker not in text, f"patch bytes leaked into the shipped arm: {marker!r}"


def test_shipped_notebook_carries_probe_guard_and_prelude(tmp_path):
    nb_path = build_shipped(output_root=tmp_path)
    text = _nb_text(nb_path)
    # behavioural probe, byte-identical machinery to the other arms
    assert "def install() -> bool" in text and "dead_reissue" in text
    assert "behav_assert = assert_observed" in text
    # anti-apply guard + patch-independent sandbox liveness in the hook
    assert "ANTI-APPLY GUARD" in text
    assert "sandbox-alive" in text
    # driver shims in the run cell, AFTER the driver and BEFORE pc_main
    run_src = next(
        "".join(c["source"]) for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code" and "await pc_main" in "".join(c["source"]))
    assert "shipped_no_patch_markers" in run_src
    assert run_src.index("async def pc_main") < run_src.index("shipped_no_patch_markers")
    assert run_src.index("shipped_no_patch_markers") < run_src.index("await pc_main")
    # the standard machinery is still present
    assert "arc_agi_3_wheels" in text                       # install cell
    assert "force the serve" in text                        # serve forced
    assert "banked_controls" in run_src and "reading=" in run_src


def test_shipped_contract_and_metadata(tmp_path):
    nb_path = build_shipped(output_root=tmp_path)
    contract = notebook_contract(nb_path)
    assert contract["arm"] == "shipped"
    assert contract["arm_env"] == {}
    assert contract["hypothesis"] == SHIPPED_HYPOTHESIS
    assert contract["geometry"] == GEOMETRY          # full 25 games, frozen box
    assert contract["patch_sha256"] == UNPATCHED_SENTINEL
    assert len(contract["source_base_sha256"]) == 64

    build_shipped()  # regenerate the repo copy (idempotent)
    meta = json.loads(
        (REPO / "submission/_ab_patch_closure/shipped/kernel-metadata.json").read_text())
    assert meta["id"] == f"ahmedmobasher86/{SHIPPED_SLUG}"
    assert meta["code_file"] == "patch-closure-shipped.ipynb"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")
    # a NEW slug: it can never overwrite the existing arms
    assert SHIPPED_SLUG not in ("arc-agi-3-patch-closure-base",
                                "arc-agi-3-patch-closure-candidate",
                                "arc-agi-3-package-screen",
                                "arc-agi-3-struct-screen",
                                "arc-agi-3-duck-patched")


def test_shipped_build_is_deterministic(tmp_path):
    a = build_shipped(output_root=tmp_path / "a").read_bytes()
    b = build_shipped(output_root=tmp_path / "b").read_bytes()
    assert a == b


def test_hook_and_prelude_sources_are_what_the_kernel_gets(tmp_path):
    text = _nb_text(build_shipped(output_root=tmp_path))
    # the whole guard and shim sources land verbatim (modulo nothing)
    for chunk in (SHIPPED_HOOK, SHIPPED_PRELUDE):
        for line in chunk.splitlines():
            if line.strip():
                assert line in text, f"missing built line: {line!r}"


def test_base_and_candidate_arms_are_untouched(tmp_path):
    """The patched arms must still inline the patch layer + APPLY_BLOCK and
    carry NONE of the shipped machinery (byte-identity itself was verified by
    rebuilding base/candidate before and after the builder edit — hashes
    28a8be18… / da8bd5a5… unchanged, 2026-08-10)."""
    for arm in ("base", "candidate"):
        nb_path = build_arm(arm, output_root=tmp_path)
        text = _nb_text(nb_path)
        assert "def apply_all(" in text and "_pc_patch_results = apply_all()" in text
        assert "ANTI-APPLY GUARD" not in text
        assert "shipped_no_patch_markers" not in text
        contract = notebook_contract(nb_path)
        assert len(contract["patch_sha256"]) == 64
        assert contract["patch_sha256"] != UNPATCHED_SENTINEL


# --- GPU-free end-to-end ----------------------------------------------------------


def test_shipped_dry_run_end_to_end(tmp_path):
    """Runs dry_run_shipped.py in a SUBPROCESS (same isolation rationale as
    the closure/package dry-run tests: patches are process-global). The script
    hard-asserts the shipped arm completes with NO patch layer, the absence
    proof + purity fields, the patched base arm afterwards, and an
    INDISTINGUISHABLE verdict from classify_shipped."""
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "dry_run_shipped.py")],
        env={**os.environ, "PC_DRY_WORKDIR": str(tmp_path)},
        capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (
        f"shipped dry run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    assert "state=INDISTINGUISHABLE" in proc.stdout, proc.stdout[-2000:]

    workroot = next(p for p in sorted(tmp_path.iterdir()) if p.is_dir())
    shipped = json.loads(
        (workroot / "shipped" / "patch_closure_result.json").read_text())
    assert shipped["arm"] == "shipped" and shipped["stage"] == "done"
    assert shipped["arm_env"] == {} and shipped["patch_sha256"] == UNPATCHED_SENTINEL
    assert shipped["identity"]["patch_proof"]["shipped_no_patch_markers"] is True
    base = json.loads((workroot / "base" / "patch_closure_result.json").read_text())
    assert base["arm"] == "base" and base["stage"] == "done"
    assert base["identity"]["patch_proof"]["watchdog_should_stop_patched"] is True
