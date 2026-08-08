"""Frozen-contract + builder + dry-run tests for the package screen.

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_package_screen.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_package_screen import PACKAGE_SLUG, build_package  # noqa: E402
from build_patch_closure import notebook_contract  # noqa: E402
from package_screen_config import (  # noqa: E402
    BANKED_BASE,
    PACKAGE_ENV,
    PACKAGE_FLAGS,
    PACKAGE_HYPOTHESIS,
    PACKAGE_READING,
    classify_screen,
)
from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    GEOMETRY,
    HYPOTHESIS,
    OFFICIAL_GAMES,
)

REPO = Path(__file__).resolve().parents[2]


# --- the frozen package contract ------------------------------------------------


def test_package_env_is_base_env_plus_exactly_the_four_flags():
    changed = {k for k in BASE_ENV | PACKAGE_ENV if BASE_ENV.get(k) != PACKAGE_ENV.get(k)}
    assert changed == {"TAAF_DIFF_LINES", "TAAF_WIGGLE", "TAAF_RUN_PROBE", "TAAF_DISPATCH"}
    assert all(PACKAGE_ENV[k] == "1" for k in changed)
    assert PACKAGE_FLAGS == {k: "1" for k in changed}


def test_package_keeps_the_v7_pins():
    assert PACKAGE_ENV["TAAF_WATCHDOG_STALL_S"] == "900"
    assert PACKAGE_ENV["TAAF_ANIMATION"] == "0"
    assert PACKAGE_ENV["TAAF_GRAPH"] == "0"


def test_package_hypothesis_is_distinct_from_closure():
    assert PACKAGE_HYPOTHESIS != HYPOTHESIS
    assert BANKED_BASE["levels_excl_ft09_by_wave"] == [11, 12]
    assert "18 levels" in PACKAGE_READING["advance_bars"]
    assert "wa30/m0r0/g50t/dc22" in PACKAGE_READING["advance_bars"]


# --- decision boundaries --------------------------------------------------------


def _valid_package_result() -> dict:
    rows = []
    rows_by_source: dict[str, dict] = {}
    for i in range(GEOMETRY["clones"]):
        stem = OFFICIAL_GAMES[i % len(OFFICIAL_GAMES)]
        levels = 4 if stem == "ft09" else (2 if stem in ("su15", "tu93") else 0)
        rows.append({"clone_id": f"k{i:03d}", "source_game": stem,
                     "levels_completed": levels, "levels_total": 6,
                     "actions_total": 80, "state": "GameRunState.DONE"})
        slot = rows_by_source.setdefault(stem, {"levels": 0, "levels_total": 6, "n_clones": 0})
        slot["levels"] = max(slot["levels"], levels)
        slot["n_clones"] += 1
    return {
        "schema_version": 1,
        "hypothesis": PACKAGE_HYPOTHESIS,
        "arm": "package",
        "arm_env": dict(PACKAGE_ENV),
        "source_base_sha256": "a" * 64,
        "patch_sha256": "b" * 64,
        "geometry": dict(GEOMETRY),
        "identity": {"arm": "package", "toggles_ok": True,
                     "watchdog_stall_s_observed": [900.0]},
        "rows": rows,
        "rows_by_source": rows_by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "patch_diagnostics": {
            "diff_lines": {"reports": 120, "actions": 900},
            "wiggle": {"batteries": 9, "battery_presses": 70,
                       "reprobes": 4, "reprobe_presses": 8},
            "run_probe": {"calls": 30, "actions": 90, "refusals": 2},
            "dispatch": {"swaps": 12, "escapes": 1},
            "wiggle_modes_assigned": {"CLICK": 14, "AVATAR": 5, "UNCLEAR": 9},
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                         "recovery_resets": 0, "stall_s_observed": [900.0]},
        },
        "pre_registered_reading": dict(PACKAGE_READING),
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


@pytest.fixture
def package_result() -> dict:
    return _valid_package_result()


def test_default_screen_verdict_is_stop(package_result):
    out = classify_screen(package_result)
    assert out["state"] == "STOP"
    # 24 non-ft09 games in rows: su15+tu93 at 2 each + 3 duplicate clones at 0
    assert out["metrics"]["levels_excl_ft09"] == 4
    assert out["metrics"]["new_target_unlocks"] == []


def test_advance_on_18_levels_excl_ft09(package_result):
    for stem in ("cd82", "cn04", "sp80", "sk48", "re86", "s5i5", "ka59"):
        package_result["rows_by_source"][stem]["levels"] = 2
    out = classify_screen(package_result)
    assert out["metrics"]["levels_excl_ft09"] == 18
    assert out["state"] == "ADVANCE"


def test_seventeen_levels_is_not_enough(package_result):
    for stem in ("cd82", "cn04", "sp80", "sk48", "re86", "s5i5"):
        package_result["rows_by_source"][stem]["levels"] = 2
    package_result["rows_by_source"]["ka59"]["levels"] = 1
    out = classify_screen(package_result)
    assert out["metrics"]["levels_excl_ft09"] == 17
    assert out["state"] == "STOP"


def test_advance_on_two_target_first_unlocks(package_result):
    package_result["rows_by_source"]["m0r0"]["levels"] = 1
    package_result["rows_by_source"]["dc22"]["levels"] = 1
    out = classify_screen(package_result)
    assert out["state"] == "ADVANCE"
    assert set(out["metrics"]["new_target_unlocks"]) == {"m0r0", "dc22"}


def test_one_target_unlock_is_not_enough(package_result):
    package_result["rows_by_source"]["wa30"]["levels"] = 1
    assert classify_screen(package_result)["state"] == "STOP"


def test_invalid_on_arm_env_drift(package_result):
    package_result["arm_env"]["TAAF_WIGGLE"] = "0"
    assert classify_screen(package_result)["state"] == "INVALID"


def test_invalid_on_missing_identity(package_result):
    del package_result["identity"]
    assert classify_screen(package_result)["state"] == "INVALID"


def test_infra_failure_on_error_or_short_rows(package_result):
    package_result["error"] = "Traceback: boom"
    assert classify_screen(package_result)["state"] == "INFRA_FAILURE"
    fresh = _valid_package_result()
    fresh["rows"] = fresh["rows"][:5]
    assert classify_screen(fresh)["state"] == "INFRA_FAILURE"


def test_metrics_echo_engagement_and_health(package_result):
    out = classify_screen(package_result)
    eng = out["metrics"]["engagement"]
    assert eng["wiggle"]["batteries"] == 9
    assert eng["run_probe"]["calls"] == 30
    assert eng["dispatch"]["modes_assigned"] == {"CLICK": 14, "AVATAR": 5, "UNCLEAR": 9}
    assert eng["diff_lines"]["reports"] == 120
    assert out["metrics"]["protocol_health"]["mean_actions_per_game"] == 80.0
    assert out["pre_registered_reading_present"] is True


def test_classify_package_cli_round_trip(tmp_path, package_result, capsys):
    import classify_package

    p = tmp_path / "package.json"
    p.write_text(json.dumps(package_result))
    rc = classify_package.main([str(p)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["state"] == "STOP"


# --- builder --------------------------------------------------------------------


def test_package_kernel_contract_and_metadata(tmp_path):
    nb_path = build_package(output_root=tmp_path)
    contract = notebook_contract(nb_path)
    assert contract["arm"] == "package"
    assert contract["arm_env"] == PACKAGE_ENV
    assert contract["hypothesis"] == PACKAGE_HYPOTHESIS
    assert contract["geometry"] == GEOMETRY

    build_package()  # regenerate the repo copy (idempotent)
    meta = json.loads(
        (REPO / "submission/_ab_patch_closure/package/kernel-metadata.json").read_text())
    assert meta["id"] == f"ahmedmobasher86/{PACKAGE_SLUG}"
    assert meta["code_file"] == "package-screen.ipynb"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")
    assert PACKAGE_SLUG != "arc-agi-3-duck-patched"

    nb = json.loads(nb_path.read_text())
    hook = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "PC_ARM_ENV" in "".join(c["source"]))
    assert 'PC_ARM = "package"' in hook
    for key, value in PACKAGE_FLAGS.items():
        assert f'"{key}": "{value}"' in hook
    run = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "await pc_main" in "".join(c["source"]))
    assert "reading=" in run and "advance_bars" in run
    install = [c for c in nb["cells"] if c["cell_type"] == "code"
               and "arc_agi_3_wheels" in "".join(c["source"])]
    assert len(install) == 1
    assert '"/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"' in "".join(install[0]["source"])


def test_parallel_load_probe_rides_after_the_run_cell(tmp_path):
    """The probe must be a SEPARATE cell strictly AFTER the run cell (the
    result JSON is on disk before it starts), hard-capped, fully wrapped, and
    absent from the closure arms."""
    import ast

    from build_patch_closure import build_arm

    nb = json.loads(build_package(output_root=tmp_path).read_text())
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    run_idx = next(i for i, s in enumerate(srcs) if "await pc_main" in s)
    probe_idx = next(i for i, s in enumerate(srcs) if "PARALLEL LOAD PROBE" in s)
    assert probe_idx == run_idx + 1, (run_idx, probe_idx)
    probe = srcs[probe_idx]
    ast.parse(probe)
    assert "HARD_CAP_S = 300.0" in probe
    assert "parallel_load.json" in probe
    assert '("A", 28)' in probe and '("B", 56)' in probe
    assert "except Exception" in probe and "the probe must NEVER fail the kernel" in probe
    assert '"enable_thinking": False' in probe

    for arm in ("base", "candidate"):
        nb_arm = json.loads(build_arm(arm, output_root=tmp_path).read_text())
        assert not any("PARALLEL LOAD PROBE" in "".join(c["source"])
                       for c in nb_arm["cells"] if c["cell_type"] == "code"), arm


def test_package_shares_source_hashes_with_closure_build(tmp_path):
    from build_patch_closure import build_arm

    pkg = notebook_contract(build_package(output_root=tmp_path))
    base = notebook_contract(build_arm("base", output_root=tmp_path))
    assert pkg["source_base_sha256"] == base["source_base_sha256"]
    assert pkg["patch_sha256"] == base["patch_sha256"]


# --- GPU-free end-to-end --------------------------------------------------------


def test_package_dry_run_all_four_mechanisms_fire(tmp_path):
    """Runs dry_run_package.py in a SUBPROCESS (same isolation rationale as the
    closure dry-run test). The script itself hard-asserts wiggle batteries,
    run_probe execution, diff-lines reports, dispatch mode assignment, the
    embedded reading contract, and a STOP verdict."""
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "dry_run_package.py")],
        env={**os.environ, "PC_DRY_WORKDIR": str(tmp_path)},
        capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"package dry run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    assert "state=STOP" in proc.stdout, proc.stdout[-2000:]

    workroot = next(p for p in sorted(tmp_path.iterdir()) if p.is_dir())
    result = json.loads((workroot / "package" / "patch_closure_result.json").read_text())
    assert result["arm"] == "package" and result["stage"] == "done"
    assert result["pre_registered_reading"]["banked_base"]["levels_excl_ft09_by_wave"] == [11, 12]
    pd = result["patch_diagnostics"]
    assert pd["wiggle"]["batteries"] >= 1
    assert pd["run_probe"]["calls"] >= 1
    assert pd["diff_lines"]["reports"] >= 1
    assert (pd.get("wiggle_modes_assigned") or {}).get("CLICK", 0) >= 1
    out = classify_screen(result)
    assert out["state"] == "STOP"
