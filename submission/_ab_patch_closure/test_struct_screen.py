"""Frozen-contract + builder + dry-run tests for the STRUCT screen.

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_struct_screen.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_patch_closure import notebook_contract  # noqa: E402
from build_struct_screen import STRUCT_SLUG, build_struct  # noqa: E402
from package_screen_config import PACKAGE_HYPOTHESIS  # noqa: E402
from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    GEOMETRY,
    HYPOTHESIS,
    OFFICIAL_GAMES,
)
from struct_screen_config import (  # noqa: E402
    BANKED,
    STRUCT_ENV,
    STRUCT_FLAGS,
    STRUCT_HYPOTHESIS,
    STRUCT_READING,
    classify_struct,
)

REPO = Path(__file__).resolve().parents[2]


# --- the frozen struct contract -------------------------------------------------


def test_struct_env_is_base_env_plus_exactly_the_four_flags():
    changed = {k for k in BASE_ENV | STRUCT_ENV if BASE_ENV.get(k) != STRUCT_ENV.get(k)}
    assert changed == {"TAAF_DIFF_LINES", "TAAF_WIGGLE", "TAAF_DISPATCH", "TAAF_STRUCT"}
    assert all(STRUCT_ENV[k] == "1" for k in changed)
    assert STRUCT_FLAGS == {k: "1" for k in changed}


def test_struct_keeps_v7_pins_and_excludes_superseded_variables():
    assert STRUCT_ENV["TAAF_WATCHDOG_STALL_S"] == "900"
    assert STRUCT_ENV["TAAF_ANIMATION"] == "0"
    assert STRUCT_ENV["TAAF_GRAPH"] == "0"
    # run_probe superseded by the plan channel; verify = one variable at a time
    assert "TAAF_RUN_PROBE" not in STRUCT_ENV
    assert "TAAF_VERIFY" not in STRUCT_ENV


def test_struct_hypothesis_and_banked_references():
    assert STRUCT_HYPOTHESIS not in (HYPOTHESIS, PACKAGE_HYPOTHESIS)
    assert BANKED["base_pair"]["levels_excl_ft09_by_wave"] == [11, 12]
    assert BANKED["package_screen"]["levels_excl_ft09"] == 10
    assert "18 levels" in STRUCT_READING["advance_bars"]
    assert "ACTIONS-PER-TURN" in STRUCT_READING["adoption"]


# --- decision boundaries --------------------------------------------------------


def _valid_struct_result() -> dict:
    rows = []
    rows_by_source: dict[str, dict] = {}
    for i in range(GEOMETRY["clones"]):
        stem = OFFICIAL_GAMES[i % len(OFFICIAL_GAMES)]
        levels = 4 if stem == "ft09" else (2 if stem in ("su15", "tu93") else 0)
        rows.append({"clone_id": f"k{i:03d}", "source_game": stem,
                     "levels_completed": levels, "levels_total": 6,
                     "actions_total": 110, "state": "GameRunState.DONE"})
        slot = rows_by_source.setdefault(stem, {"levels": 0, "levels_total": 6, "n_clones": 0})
        slot["levels"] = max(slot["levels"], levels)
        slot["n_clones"] += 1
    return {
        "schema_version": 1,
        "hypothesis": STRUCT_HYPOTHESIS,
        "arm": "struct",
        "arm_env": dict(STRUCT_ENV),
        "source_base_sha256": "a" * 64,
        "patch_sha256": "b" * 64,
        "geometry": dict(GEOMETRY),
        "identity": {"arm": "struct", "toggles_ok": True,
                     "watchdog_stall_s_observed": [900.0]},
        "rows": rows,
        "rows_by_source": rows_by_source,
        "behavior": {"corpus": {"actions": 3000, "actions_per_turn": 3.8}},
        "patch_diagnostics": {
            "struct": {
                "plans": 700, "plan_actions": 2700, "wrapped_singles": 120,
                "plan_lengths": {"1": 120, "4": 300, "8": 200, "20": 80},
                "invalid_dropped": 14, "reset_deduped": 2, "cap_truncations": 3,
                "scout_truncations": 40, "score_flushes": 25,
                "brake_strips": 60, "brake_soft_flags": 12, "menu_strips": 9,
                "phase_transitions": 55, "reports_injected": 690,
                "nudges": 4, "turns_without_plan": 6,
            },
            "wiggle": {"batteries": 30, "battery_presses": 200,
                       "reprobes": 20, "reprobe_presses": 40},
            "diff_lines": {"reports": 700, "actions": 3000},
            "dispatch": {"swaps": 10, "escapes": 2},
            "run_probe": {"calls": 0, "actions": 0, "refusals": 0},
            "wiggle_modes_assigned": {"CLICK": 12, "AVATAR": 8, "HERD": 3,
                                      "CURSOR": 3, "UNCLEAR": 2},
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                         "recovery_resets": 0, "stall_s_observed": [900.0]},
        },
        "adoption": {
            "llm_turns": 790, "actions_total": 3080,
            "actions_per_llm_turn": 3.9,
            "plan_actions": 2700, "plan_actions_per_llm_turn": 3.42,
        },
        "pre_registered_reading": dict(STRUCT_READING),
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


@pytest.fixture
def struct_result() -> dict:
    return _valid_struct_result()


def test_default_struct_verdict_is_stop(struct_result):
    out = classify_struct(struct_result)
    assert out["state"] == "STOP"
    assert out["metrics"]["levels_excl_ft09"] == 4
    assert out["metrics"]["new_target_unlocks"] == []


def test_advance_on_18_levels_excl_ft09(struct_result):
    for stem in ("cd82", "cn04", "sp80", "sk48", "re86", "s5i5", "ka59"):
        struct_result["rows_by_source"][stem]["levels"] = 2
    out = classify_struct(struct_result)
    assert out["metrics"]["levels_excl_ft09"] == 18
    assert out["state"] == "ADVANCE"


def test_seventeen_levels_is_not_enough(struct_result):
    for stem in ("cd82", "cn04", "sp80", "sk48", "re86", "s5i5"):
        struct_result["rows_by_source"][stem]["levels"] = 2
    struct_result["rows_by_source"]["ka59"]["levels"] = 1
    out = classify_struct(struct_result)
    assert out["metrics"]["levels_excl_ft09"] == 17
    assert out["state"] == "STOP"


def test_advance_on_two_target_first_unlocks(struct_result):
    struct_result["rows_by_source"]["wa30"]["levels"] = 1
    struct_result["rows_by_source"]["g50t"]["levels"] = 1
    out = classify_struct(struct_result)
    assert out["state"] == "ADVANCE"
    assert set(out["metrics"]["new_target_unlocks"]) == {"wa30", "g50t"}


def test_one_target_unlock_is_not_enough(struct_result):
    struct_result["rows_by_source"]["m0r0"]["levels"] = 1
    assert classify_struct(struct_result)["state"] == "STOP"


def test_invalid_on_arm_env_drift(struct_result):
    struct_result["arm_env"]["TAAF_RUN_PROBE"] = "1"
    assert classify_struct(struct_result)["state"] == "INVALID"


def test_invalid_on_missing_identity(struct_result):
    del struct_result["identity"]
    assert classify_struct(struct_result)["state"] == "INVALID"


def test_infra_failure_on_error_or_missing_struct_diag(struct_result):
    struct_result["error"] = "Traceback: boom"
    assert classify_struct(struct_result)["state"] == "INFRA_FAILURE"
    fresh = _valid_struct_result()
    del fresh["patch_diagnostics"]["struct"]
    out = classify_struct(fresh)
    assert out["state"] == "INFRA_FAILURE"
    assert any("STRUCT_DIAGNOSTICS" in r for r in out["reasons"])


def test_infra_failure_when_adoption_block_missing(struct_result):
    del struct_result["adoption"]
    out = classify_struct(struct_result)
    assert out["state"] == "INFRA_FAILURE"
    assert any("adoption block missing" in r for r in out["reasons"])


def test_metrics_echo_adoption_and_engagement(struct_result):
    out = classify_struct(struct_result)
    adoption = out["metrics"]["adoption"]
    assert adoption["plan_actions_per_llm_turn"] == 3.42
    assert adoption["actions_per_llm_turn_raw"] == 3.9
    assert adoption["probe_actions_per_turn"] == 3.8
    assert adoption["banked_base_actions_per_turn"] == 1.0
    assert adoption["plan_lengths"] == {"1": 120, "4": 300, "8": 200, "20": 80}
    assert adoption["multi_step_plan_share"] == round(1.0 - 120 / 700, 4)
    eng = out["metrics"]["engagement"]
    assert eng["struct"]["brake_strips"] == 60
    assert eng["struct"]["phase_transitions"] == 55
    assert eng["wiggle_modes_assigned"]["HERD"] == 3
    health = out["metrics"]["protocol_health"]
    assert health["nudges"] == 4 and health["turns_without_plan"] == 6
    assert out["pre_registered_reading_present"] is True


def test_classify_struct_cli_round_trip(tmp_path, struct_result, capsys):
    import classify_struct

    p = tmp_path / "struct.json"
    p.write_text(json.dumps(struct_result))
    rc = classify_struct.main([str(p)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["state"] == "STOP"


# --- builder --------------------------------------------------------------------


def test_struct_kernel_contract_and_metadata(tmp_path):
    nb_path = build_struct(output_root=tmp_path)
    contract = notebook_contract(nb_path)
    assert contract["arm"] == "struct"
    assert contract["arm_env"] == STRUCT_ENV
    assert contract["hypothesis"] == STRUCT_HYPOTHESIS
    assert contract["geometry"] == GEOMETRY

    build_struct()  # regenerate the repo copy (idempotent)
    meta = json.loads(
        (REPO / "submission/_ab_patch_closure/struct/kernel-metadata.json").read_text())
    assert meta["id"] == f"ahmedmobasher86/{STRUCT_SLUG}"
    assert meta["code_file"] == "struct-screen.ipynb"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")
    assert STRUCT_SLUG != "arc-agi-3-duck-patched"

    nb = json.loads(nb_path.read_text())
    hook = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "PC_ARM_ENV" in "".join(c["source"]))
    assert 'PC_ARM = "struct"' in hook
    for key, value in STRUCT_FLAGS.items():
        assert f'"{key}": "{value}"' in hook
    assert '"TAAF_RUN_PROBE"' not in hook.split("PC_ARM_ENV = {")[1].split("}")[0]
    run = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "await pc_main" in "".join(c["source"]))
    assert "reading=" in run and "ACTIONS-PER-TURN" in run
    # the parallel-load probe must NOT ride on this kernel (already measured)
    assert not any("PARALLEL LOAD PROBE" in "".join(c["source"])
                   for c in nb["cells"] if c["cell_type"] == "code")
    install = [c for c in nb["cells"] if c["cell_type"] == "code"
               and "arc_agi_3_wheels" in "".join(c["source"])]
    assert len(install) == 1
    assert '"/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"' in "".join(install[0]["source"])


def test_struct_shares_source_hashes_with_the_other_kernels(tmp_path):
    from build_package_screen import build_package

    struct = notebook_contract(build_struct(output_root=tmp_path))
    package = notebook_contract(build_package(output_root=tmp_path))
    assert struct["source_base_sha256"] == package["source_base_sha256"]
    assert struct["patch_sha256"] == package["patch_sha256"]


# --- GPU-free end-to-end --------------------------------------------------------


def test_struct_dry_run_plan_channel_fires(tmp_path):
    """Runs dry_run_struct.py in a SUBPROCESS (same isolation rationale as the
    closure dry-run test). The script hard-asserts both plan routes (multi-step
    + auto-wrapped single), actions-per-turn > 1.0, quiet run_probe, carried
    package mechanisms, the embedded reading, and a STOP verdict."""
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "dry_run_struct.py")],
        env={**os.environ, "PC_DRY_WORKDIR": str(tmp_path)},
        capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"struct dry run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    assert "state=STOP" in proc.stdout, proc.stdout[-2000:]

    workroot = next(p for p in sorted(tmp_path.iterdir()) if p.is_dir())
    result = json.loads((workroot / "struct" / "patch_closure_result.json").read_text())
    assert result["arm"] == "struct" and result["stage"] == "done"
    assert result["pre_registered_reading"]["banked"]["package_screen"]["levels_excl_ft09"] == 10
    st = result["patch_diagnostics"]["struct"]
    assert st["plans"] >= 1 and st["plan_actions"] >= 1
    assert any(int(k) >= 2 for k in (st.get("plan_lengths") or {}))
    assert result["adoption"]["plan_actions_per_llm_turn"] > 1.0
    out = classify_struct(result)
    assert out["state"] == "STOP"
    assert out["metrics"]["adoption"]["plan_actions_per_llm_turn"] > 1.0
