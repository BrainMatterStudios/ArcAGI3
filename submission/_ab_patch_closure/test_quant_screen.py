"""Frozen-contract + builder + classifier tests for the QUANT-SWAP screen
(the shipped arm with the served FP8 snapshot swapped vrfai -> RedHatAI).

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_quant_screen.py -q
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
from build_quant_screen import (  # noqa: E402
    _DS_LINE_NEW,
    _DS_LINE_OLD,
    _SETUP_LOOP_OLD,
    QUANT_HOOK,
    QUANT_PRELUDE,
    QUANT_SLUG,
    build_quant,
)
from build_shipped_screen import build_shipped  # noqa: E402
from package_screen_config import PACKAGE_HYPOTHESIS  # noqa: E402
from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    CANDIDATE_ENV,
    GEOMETRY,
    HYPOTHESIS as CLOSURE_HYPOTHESIS,
    OFFICIAL_GAMES,
)
from quant_screen_config import (  # noqa: E402
    QUANT_DATASET,
    QUANT_ENV,
    QUANT_HYPOTHESIS,
    QUANT_READING,
    QUANT_SERVED_MODEL,
    SERVE_ANCHOR_REWRITES,
    VRFAI_DATASET,
    VRFAI_SERVED_MODEL,
    QuantThresholds,
    classify_quant,
)
from shipped_screen_config import (  # noqa: E402
    SHIPPED_ENV,
    SHIPPED_HYPOTHESIS,
    UNPATCHED_SENTINEL,
)

REPO = Path(__file__).resolve().parents[2]

# Byte sequences that exist ONLY if duck_patches source was inlined (same
# rationale as test_shipped_screen.PATCH_DEFINITION_MARKERS).
PATCH_DEFINITION_MARKERS = (
    "def apply_all(",
    "def patch_action7(",
    "def patch_watchdog(",
    "def patch_win_replay(",
    "def patch_hud_board_identity(",
    "def patch_frontier_graph(",
    "class HudMaskTracker",
    "_pc_patch_results = apply_all()",
)


def _nb_text(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    return "\n@@CELL@@\n".join(
        "".join(c.get("source", [])) for c in nb["cells"])


def _env_fp(env: dict) -> str:
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:12]


# --- the frozen quant contract ----------------------------------------------------


def test_quant_env_is_the_inert_pin_and_hypothesis_distinct():
    assert QUANT_ENV == {"PC_QUANT_SNAPSHOT": "redhatai-qwen3-6-27b-fp8-hf-snapshot"}
    # inert: no behavioural TAAF_* toggle may ride along
    assert not any(k.startswith("TAAF_") for k in QUANT_ENV)
    assert QUANT_HYPOTHESIS not in (
        CLOSURE_HYPOTHESIS, PACKAGE_HYPOTHESIS, SHIPPED_HYPOTHESIS)


def test_env_fingerprint_distinct_from_every_other_arm():
    fps = {_env_fp(QUANT_ENV), _env_fp(SHIPPED_ENV), _env_fp(BASE_ENV),
           _env_fp(CANDIDATE_ENV)}
    assert len(fps) == 4


def test_swap_constants_are_coherent():
    assert VRFAI_DATASET == "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
    assert QUANT_DATASET == "ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot"
    assert QUANT_SERVED_MODEL == "RedHatAI/Qwen3.6-27B-FP8"
    # every rewrite goes vrfai-family -> redhatai-family, never the reverse
    olds, news = zip(*SERVE_ANCHOR_REWRITES)
    assert all("driessmit1" in o or "vrfai" in o for o in olds)
    assert not any("vrfai" in n for n in news)
    assert len(SERVE_ANCHOR_REWRITES) == 3
    anchors = " ".join(olds)
    for name in ("MODEL_OWNER", "MODEL_SLUG", "SERVED_MODEL_NAME"):
        assert name in anchors


def test_preregistration_content():
    assert "true_score_all_games" in QUANT_READING["reading_rule"]
    assert "25 games" in QUANT_READING["reading_rule"]
    assert "0.2712" in QUANT_READING["reading_rule"]
    # sonpham is a hypothesis source ONLY — the pre-registration must say so
    assert "sonpham" in QUANT_READING["hypothesis"]
    assert "hypothesis source" in QUANT_READING["hypothesis"]
    assert "not evidence" in QUANT_READING["hypothesis"]
    assert "0.2712" in QUANT_READING["power_honesty"]
    assert QuantThresholds().min_effect_true_score == pytest.approx(0.2712)


def test_serve_anchors_match_the_scored_ref_bytes():
    """Each FROM anchor must appear exactly once in setup_commands cmd 0 and
    no TO anchor may pre-exist (skips when the scratchpad ref is absent)."""
    ref = REPO / "scratchpad/taaf_scored_ref/setup_commands.json"
    if not ref.is_file():
        pytest.skip("scored-ref setup_commands.json not present on this machine")
    commands = json.loads(ref.read_text())
    joined = "\n".join(commands)
    for old, new in SERVE_ANCHOR_REWRITES:
        assert joined.count(old) == 1, old
        assert new not in joined, new


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


def _unpatched_diags() -> dict:
    return {
        "animation": {"payload_deliveries": 0, "frames_delivered": 0},
        "graph": {"sessions_with_graph_state": 0},
        "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                     "recovery_resets": 0, "stall_s_observed": []},
    }


def _valid_quant(score: float = 1.0) -> dict:
    rows, by_source = _rows(score)
    return {
        "schema_version": 1,
        "hypothesis": QUANT_HYPOTHESIS,
        "arm": "quant",
        "arm_env": dict(QUANT_ENV),
        "source_base_sha256": "a" * 64,
        "patch_sha256": UNPATCHED_SENTINEL,
        "geometry": dict(GEOMETRY),
        "identity": {"arm": "quant", "toggles_ok": True,
                     "patch_proof": {"quant_no_patch_markers": True},
                     "serving": {"base_url": "http://127.0.0.1:1234/v1",
                                 "served_ids": [QUANT_SERVED_MODEL],
                                 "expected_id": QUANT_SERVED_MODEL, "ok": True}},
        "rows": rows,
        "rows_by_source": by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "patch_diagnostics": _unpatched_diags(),
        "pre_registered_reading": dict(QUANT_READING),
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


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
                     "patch_proof": {"shipped_no_patch_markers": True},
                     "serving": {"base_url": "http://127.0.0.1:1234/v1",
                                 "served_ids": [VRFAI_SERVED_MODEL],
                                 "expected_id": VRFAI_SERVED_MODEL, "ok": True}},
        "rows": rows,
        "rows_by_source": by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "patch_diagnostics": _unpatched_diags(),
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


def test_equal_scores_are_indistinguishable():
    out = classify_quant(_valid_quant(1.0), _valid_shipped(1.0))
    assert out["state"] == "INDISTINGUISHABLE"
    assert out["metrics"]["delta_all_games_quant_minus_shipped"] == 0.0
    assert "NOT evidence of equality" in out["reasons"][0]


def test_quant_ahead_at_the_preregistered_effect():
    out = classify_quant(_valid_quant(1.30), _valid_shipped(1.0))
    assert out["state"] == "QUANT_AHEAD"
    assert out["metrics"]["delta_all_games_quant_minus_shipped"] == pytest.approx(0.30)


def test_shipped_ahead_at_the_preregistered_effect():
    out = classify_quant(_valid_quant(1.0), _valid_shipped(1.30))
    assert out["state"] == "SHIPPED_AHEAD"


def test_sub_threshold_delta_is_indistinguishable():
    out = classify_quant(_valid_quant(1.18), _valid_shipped(1.0))
    assert out["state"] == "INDISTINGUISHABLE"


def test_invalid_when_quant_served_the_vrfai_model():
    quant = _valid_quant()
    quant["identity"]["serving"] = {
        "base_url": "http://127.0.0.1:1234/v1",
        "served_ids": [VRFAI_SERVED_MODEL],
        "expected_id": VRFAI_SERVED_MODEL, "ok": True}
    out = classify_quant(quant, _valid_shipped())
    assert out["state"] == "INVALID"
    assert any("served model" in r or "forbidden alias" in r for r in out["reasons"])


def test_invalid_when_serving_identity_is_missing():
    quant = _valid_quant()
    del quant["identity"]["serving"]
    out = classify_quant(quant, _valid_shipped())
    assert out["state"] == "INVALID"
    assert any("never proven" in r for r in out["reasons"])


def test_invalid_when_shipped_comparator_served_redhatai():
    shipped = _valid_shipped()
    shipped["identity"]["serving"]["served_ids"] = [QUANT_SERVED_MODEL]
    shipped["identity"]["serving"]["expected_id"] = QUANT_SERVED_MODEL
    assert classify_quant(_valid_quant(), shipped)["state"] == "INVALID"


def test_invalid_on_quant_env_drift():
    quant = _valid_quant()
    quant["arm_env"] = {}
    assert classify_quant(quant, _valid_shipped())["state"] == "INVALID"

    quant2 = _valid_quant()
    quant2["arm_env"] = {**QUANT_ENV, "TAAF_WATCHDOG": "1"}
    assert classify_quant(quant2, _valid_shipped())["state"] == "INVALID"


def test_invalid_on_patch_contamination_in_either_arm():
    quant = _valid_quant()
    quant["patch_diagnostics"]["watchdog"]["stall_s_observed"] = [900.0]
    assert classify_quant(quant, _valid_shipped())["state"] == "INVALID"

    quant2 = _valid_quant()
    quant2["identity"]["patch_proof"] = {}
    assert classify_quant(quant2, _valid_shipped())["state"] == "INVALID"

    quant3 = _valid_quant()
    quant3["patch_sha256"] = "c" * 64
    assert classify_quant(quant3, _valid_shipped())["state"] == "INVALID"

    shipped = _valid_shipped()
    shipped["patch_sha256"] = "b" * 64  # a PATCHED comparator is not shipped
    assert classify_quant(_valid_quant(), shipped)["state"] == "INVALID"


def test_invalid_on_source_hash_or_geometry_mismatch():
    shipped = _valid_shipped()
    shipped["source_base_sha256"] = "f" * 64
    assert classify_quant(_valid_quant(), shipped)["state"] == "INVALID"

    quant, shipped2 = _valid_quant(), _valid_shipped()
    quant["geometry"] = {"clones": 4, "per_game_s": 90, "concurrency": 4}
    shipped2["geometry"] = dict(quant["geometry"])
    assert classify_quant(quant, shipped2)["state"] == "INVALID"  # not a dry pair
    quant["dry_run"] = shipped2["dry_run"] = True
    # a dry pair has no vLLM: serving checks are skipped as well
    del quant["identity"]["serving"], shipped2["identity"]["serving"]
    assert classify_quant(quant, shipped2)["state"] == "INDISTINGUISHABLE"


def test_infra_failure_on_error_short_rows_or_dead_probe():
    quant = _valid_quant()
    quant["error"] = "Traceback: boom"
    assert classify_quant(quant, _valid_shipped())["state"] == "INFRA_FAILURE"

    quant2 = _valid_quant()
    quant2["rows"] = quant2["rows"][:5]
    assert classify_quant(quant2, _valid_shipped())["state"] == "INFRA_FAILURE"

    shipped = _valid_shipped()
    shipped["behavior"]["corpus"]["actions"] = 3
    assert classify_quant(_valid_quant(), shipped)["state"] == "INFRA_FAILURE"


def test_classify_quant_cli_round_trip(tmp_path, capsys):
    import classify_quant as cli

    q, s = tmp_path / "quant.json", tmp_path / "shipped.json"
    q.write_text(json.dumps(_valid_quant(1.0)))
    s.write_text(json.dumps(_valid_shipped(1.0)))
    rc = cli.main([str(q), str(s)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["state"] == "INDISTINGUISHABLE"


# --- builder ----------------------------------------------------------------------


def test_quant_notebook_contains_no_patch_bytes(tmp_path):
    text = _nb_text(build_quant(output_root=tmp_path))
    for marker in PATCH_DEFINITION_MARKERS:
        assert marker not in text, f"patch bytes leaked into the quant arm: {marker!r}"


def test_quant_notebook_swaps_the_dataset_and_serve_anchors(tmp_path):
    nb_path = build_quant(output_root=tmp_path)
    text = _nb_text(nb_path)
    # cell 6: the swapped literal is in, the vrfai literal is gone
    assert _DS_LINE_NEW in text
    assert _DS_LINE_OLD not in text
    assert f'"{QUANT_DATASET}"' in text
    assert f'"{VRFAI_DATASET}"' not in text
    # cell 8: the plain loop is gone, the anchor-rewrite loop is in
    assert _SETUP_LOOP_OLD not in text
    assert "_pc_quant_anchors" in text and "_pc_quant_rewritten" in text
    assert "refusing to run any setup command" in text
    for old, new in SERVE_ANCHOR_REWRITES:
        assert repr(old) in text and repr(new) in text
    # the serve is still forced and the original execution line survives
    assert "force the serve" in text
    assert ("subprocess.run(command, shell=True, check=True, "
            "cwd=WORKING_DIR, env=env)") in text
    # kernel metadata mounts RedHatAI, not vrfai
    meta = json.loads((nb_path.parent / "kernel-metadata.json").read_text())
    assert QUANT_DATASET in meta["dataset_sources"]
    assert VRFAI_DATASET not in meta["dataset_sources"]
    assert "driessmit1/arc3-vllm-h100-wheelhouse-v3" in meta["dataset_sources"]
    assert "ahmedmobasher86/taaf-src-hybrid" in meta["dataset_sources"]


def test_quant_notebook_carries_probe_guard_and_prelude(tmp_path):
    nb_path = build_quant(output_root=tmp_path)
    text = _nb_text(nb_path)
    # behavioural probe, byte-identical machinery to the other arms
    assert "def install() -> bool" in text and "dead_reissue" in text
    assert "behav_assert = assert_observed" in text
    # anti-apply guard + patch-independent sandbox liveness in the hook
    assert "ANTI-APPLY GUARD" in text
    assert "sandbox-alive" in text
    # arm identity: quant, with the inert env pin exported to the environment
    assert 'PC_ARM = "quant"' in text
    assert "_pc_arm_os.environ.update(PC_ARM_ENV)" in text
    # driver shims in the run cell, AFTER the driver and BEFORE pc_main
    run_src = next(
        "".join(c["source"]) for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code" and "await pc_main" in "".join(c["source"]))
    assert "quant_no_patch_markers" in run_src
    assert "shipped_no_patch_markers" not in run_src
    assert run_src.index("async def pc_main") < run_src.index("quant_no_patch_markers")
    assert run_src.index("quant_no_patch_markers") < run_src.index("await pc_main")
    # pre-registration baked verbatim into the run cell
    assert "arc_agi_3_wheels" in text                       # install cell
    assert "sonpham" in run_src and "reading=" in run_src
    assert "0.2712" in run_src


def test_quant_contract_and_metadata(tmp_path):
    nb_path = build_quant(output_root=tmp_path)
    contract = notebook_contract(nb_path)
    assert contract["arm"] == "quant"
    assert contract["arm_env"] == QUANT_ENV
    assert contract["hypothesis"] == QUANT_HYPOTHESIS
    assert contract["geometry"] == GEOMETRY          # full 25 games, frozen box
    assert contract["patch_sha256"] == UNPATCHED_SENTINEL
    assert len(contract["source_base_sha256"]) == 64
    assert contract["quant_swap"]["dataset"] == {
        "from": VRFAI_DATASET, "to": QUANT_DATASET}
    assert contract["quant_swap"]["serve_anchor_rewrites"] == [
        list(pair) for pair in SERVE_ANCHOR_REWRITES]

    build_quant()  # regenerate the repo copy (idempotent)
    meta = json.loads(
        (REPO / "submission/_ab_patch_closure/quant/kernel-metadata.json").read_text())
    assert meta["id"] == f"ahmedmobasher86/{QUANT_SLUG}"
    assert meta["code_file"] == "patch-closure-quant.ipynb"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")
    # a NEW slug: it can never overwrite the existing arms
    assert QUANT_SLUG not in ("arc-agi-3-patch-closure-base",
                              "arc-agi-3-patch-closure-candidate",
                              "arc-agi-3-patch-closure-shipped",
                              "arc-agi-3-package-screen",
                              "arc-agi-3-struct-screen",
                              "arc-agi-3-duck-patched")


def test_quant_build_is_deterministic(tmp_path):
    a = build_quant(output_root=tmp_path / "a").read_bytes()
    b = build_quant(output_root=tmp_path / "b").read_bytes()
    assert a == b


def test_hook_and_prelude_sources_are_what_the_kernel_gets(tmp_path):
    text = _nb_text(build_quant(output_root=tmp_path))
    for chunk in (QUANT_HOOK, QUANT_PRELUDE):
        for line in chunk.splitlines():
            if line.strip():
                assert line in text, f"missing built line: {line!r}"


def test_other_arms_are_untouched(tmp_path):
    """shipped/base/candidate rebuilds carry NONE of the quant machinery (the
    builders share build_kernel, which this change did not touch)."""
    for nb_path in (build_shipped(output_root=tmp_path / "shipped"),
                    build_arm("base", output_root=tmp_path / "closure"),
                    build_arm("candidate", output_root=tmp_path / "closure")):
        text = _nb_text(nb_path)
        assert "_pc_quant_anchors" not in text
        assert "PC_QUANT_SNAPSHOT" not in text
        assert QUANT_DATASET not in text
        meta = json.loads((nb_path.parent / "kernel-metadata.json").read_text())
        assert VRFAI_DATASET in meta["dataset_sources"]
        assert QUANT_DATASET not in meta["dataset_sources"]
