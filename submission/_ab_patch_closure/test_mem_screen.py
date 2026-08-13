"""Frozen-contract + builder + classifier + dry-run tests for the MEM screen
(duck-mem P1-P4 stack on the shipped duck-base v2 config, vs the shipped arm).

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_mem_screen.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_mem_screen import (  # noqa: E402
    HARNESS_MEM,
    MEM_COUNTERS_BLOCK,
    MEM_GUARD,
    MEM_PRELUDE,
    MEM_SLUG,
    _mem_patch_bytes,
    build_mem,
    mem_hook_cell,
)
from build_patch_closure import build_arm, notebook_contract  # noqa: E402
from build_shipped_screen import build_shipped  # noqa: E402
from mem_screen_config import (  # noqa: E402
    MEM_ENV,
    MEM_HYPOTHESIS,
    MEM_MARKER_LINES,
    MEM_PROOF_KEYS,
    MEM_READING,
    MemThresholds,
    classify_mem,
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
    SHIPPED_HYPOTHESIS,
    UNPATCHED_SENTINEL,
)

REPO = Path(__file__).resolve().parents[2]

# Byte sequences that exist ONLY if duck_patches source was inlined. NOTE:
# "def apply_all(" is deliberately NOT here — harness_mem legitimately defines
# its own apply_all; the duck_patches-specific definitions below cannot appear
# in a clean mem build.
DUCK_PATCHES_DEFINITION_MARKERS = (
    "def patch_action7(",
    "def patch_watchdog(",
    "def patch_win_replay(",
    "def patch_hud_board_identity(",
    "def patch_frontier_graph(",
    "class HudMaskTracker",
    "_pc_patch_results = apply_all()",   # the patched arms' APPLY_BLOCK line
)

# Definitions a faithful harness_mem inline MUST carry.
MEM_DEFINITION_MARKERS = (
    "def patch_token_estimator(",
    "def patch_middle_drop(",
    "def patch_prompt_own_goal(",
    "def patch_repeated_no_effect_guard(",
    "apply_all(BUNDLE_DIR)",
)


def _nb_text(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    return "\n@@CELL@@\n".join("".join(c.get("source", [])) for c in nb["cells"])


def _env_fp(env: dict) -> str:
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:12]


# --- the frozen mem contract ------------------------------------------------------


def test_mem_env_and_hypothesis_are_distinct():
    assert MEM_ENV == {"DUCK_MEM_P5": "0", "DUCK_MEM_P6": "0"}
    assert MEM_HYPOTHESIS not in (
        CLOSURE_HYPOTHESIS, PACKAGE_HYPOTHESIS, SHIPPED_HYPOTHESIS)
    fps = {_env_fp(MEM_ENV), _env_fp({}), _env_fp(BASE_ENV), _env_fp(CANDIDATE_ENV)}
    assert len(fps) == 4


def test_marker_lines_match_harness_mem_source():
    """The asserted [duck-mem] marker lines must exist verbatim in
    harness_mem.py (a drifted print would turn the hard assert into a
    permanent build failure — catch it here, not on the GPU)."""
    src = HARNESS_MEM.read_text()
    for line in MEM_MARKER_LINES:
        if line == "[duck-mem] patch stack applied: 4/4":
            continue  # composed at runtime from ARM_MARKER + counts
        tail = line.replace("[duck-mem] ", "")
        assert tail in src, f"marker line not in harness_mem.py: {line!r}"
    assert 'ARM_MARKER = "[duck-mem]"' in src
    assert "patch stack applied:" in src


def test_preregistration_content():
    assert "true_score_all_games" in MEM_READING["reading_rule"]
    assert "0.2712" in MEM_READING["reading_rule"]
    assert MEM_READING["behavioral_emphasis"].startswith("SECONDARY AND EMPHASIZED")
    for needle in ("llm_turns", "trims", "guard_fires", "repeated_no_effect"):
        assert needle in MEM_READING["behavioral_emphasis"], needle
    assert "CANNOT resolve" in MEM_READING["power_honesty"]
    assert MemThresholds().min_effect_true_score == pytest.approx(0.2712)


def test_prelude_reports_every_registered_proof_key():
    for key in MEM_PROOF_KEYS:
        assert key in MEM_PRELUDE, f"prelude never reports {key}"
    # the P3 anchor line in the prelude must match harness_mem's anchor
    assert "- Optimize for as few in-game actions as possible while " in MEM_PRELUDE


def test_mem_patch_bytes_are_head_clean_and_hashable():
    data = _mem_patch_bytes()
    assert b"repeated_no_effect" in data and b"[duck-mem]" in data
    assert len(hashlib.sha256(data).hexdigest()) == 64


# --- classifier -------------------------------------------------------------------


def _rows(score: float, levels: int = 0) -> tuple[list, dict]:
    rows, by_source = [], {}
    for i in range(GEOMETRY["clones"]):
        stem = OFFICIAL_GAMES[i % len(OFFICIAL_GAMES)]
        rows.append({"clone_id": f"k{i:03d}", "source_game": stem,
                     "levels_completed": levels, "levels_total": 6,
                     "actions_total": 80, "gen_tokens": 900, "score": score,
                     "state": "GameRunState.DONE"})
        slot = by_source.setdefault(stem, {"levels": 0, "levels_total": 6, "n_clones": 0})
        slot["levels"] = max(slot["levels"], levels)
        slot["n_clones"] += 1
    return rows, by_source


def _valid_mem(score: float = 1.0) -> dict:
    rows, by_source = _rows(score)
    return {
        "schema_version": 1,
        "hypothesis": MEM_HYPOTHESIS,
        "arm": "mem",
        "arm_env": dict(MEM_ENV),
        "source_base_sha256": "a" * 64,
        "patch_sha256": "d" * 64,
        "geometry": dict(GEOMETRY),
        "identity": {"arm": "mem", "toggles_ok": True,
                     "patch_proof": {k: True for k in MEM_PROOF_KEYS}},
        "rows": rows,
        "rows_by_source": by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "adoption": {"llm_turns": 500, "actions_total": 2240,
                     "actions_per_llm_turn": 4.48},
        "patch_diagnostics": {
            "animation": {"payload_deliveries": 0, "frames_delivered": 0},
            "graph": {"sessions_with_graph_state": 0},
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                         "recovery_resets": 0, "stall_s_observed": []},
            "mem": {"guard_fires": 12, "trims": 30},
        },
        "pre_registered_reading": dict(MEM_READING),
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
                     "patch_proof": {"shipped_no_patch_markers": True}},
        "rows": rows,
        "rows_by_source": by_source,
        "behavior": {"corpus": {"actions": 1600, "actions_per_turn": 1.4}},
        "adoption": {"llm_turns": 550, "actions_total": 2240,
                     "actions_per_llm_turn": 4.07},
        "patch_diagnostics": {
            "animation": {"payload_deliveries": 0, "frames_delivered": 0},
            "graph": {"sessions_with_graph_state": 0},
            "watchdog": {"stall_kills": 0, "wall_cap_kills": 0,
                         "recovery_resets": 0, "stall_s_observed": []},
        },
        "error": None,
        "stage": "done",
        "dry_run": False,
    }


def test_equal_scores_are_indistinguishable_and_emphasize_behavior():
    out = classify_mem(_valid_mem(1.0), _valid_shipped(1.0))
    assert out["state"] == "INDISTINGUISHABLE"
    assert out["metrics"]["delta_all_games_mem_minus_shipped"] == 0.0
    assert "NOT evidence of equality" in out["reasons"][0]
    behav = out["metrics"]["behavioral_EMPHASIZED"]
    assert behav["mem_stack_activity"]["guard_fires"] == 12
    assert behav["mem_stack_activity"]["trims"] == 30
    assert behav["mem_stack_activity"]["guard_fires_per_1000_actions"] == pytest.approx(
        1000 * 12 / (28 * 80), abs=0.01)
    assert behav["per_arm"]["mem"]["llm_turns"] == 500
    assert behav["delta_mem_minus_shipped"]["llm_turns"] == -50
    # the emphasized block leads the metrics
    assert next(iter(out["metrics"])) == "behavioral_EMPHASIZED"


def test_directional_calls_at_the_preregistered_effect():
    assert classify_mem(_valid_mem(1.30), _valid_shipped(1.0))["state"] == "MEM_AHEAD"
    assert classify_mem(_valid_mem(1.0), _valid_shipped(1.30))["state"] == "SHIPPED_AHEAD"
    assert classify_mem(_valid_mem(1.18), _valid_shipped(1.0))["state"] == "INDISTINGUISHABLE"


def test_invalid_on_mem_contract_violations():
    mem = _valid_mem()
    mem["arm_env"] = {}
    assert classify_mem(mem, _valid_shipped())["state"] == "INVALID"

    mem2 = _valid_mem()
    mem2["patch_sha256"] = UNPATCHED_SENTINEL
    assert classify_mem(mem2, _valid_shipped())["state"] == "INVALID"

    mem3 = _valid_mem()
    mem3["identity"]["patch_proof"]["mem_p3_prompt_neutralized"] = False
    out = classify_mem(mem3, _valid_shipped())
    assert out["state"] == "INVALID"
    assert any("mem_p3_prompt_neutralized" in r for r in out["reasons"])


def test_invalid_on_duck_patches_contamination_in_either_arm():
    mem = _valid_mem()
    mem["patch_diagnostics"]["watchdog"]["stall_s_observed"] = [900.0]
    assert classify_mem(mem, _valid_shipped())["state"] == "INVALID"

    shipped = _valid_shipped()
    shipped["patch_diagnostics"]["graph"]["sessions_with_graph_state"] = 3
    assert classify_mem(_valid_mem(), shipped)["state"] == "INVALID"

    shipped2 = _valid_shipped()
    shipped2["identity"]["patch_proof"] = {}
    assert classify_mem(_valid_mem(), shipped2)["state"] == "INVALID"


def test_invalid_on_source_hash_or_geometry_mismatch():
    shipped = _valid_shipped()
    shipped["source_base_sha256"] = "f" * 64
    assert classify_mem(_valid_mem(), shipped)["state"] == "INVALID"

    mem, shipped2 = _valid_mem(), _valid_shipped()
    mem["geometry"] = {"clones": 4, "per_game_s": 90, "concurrency": 4}
    shipped2["geometry"] = dict(mem["geometry"])
    assert classify_mem(mem, shipped2)["state"] == "INVALID"  # not a dry pair
    mem["dry_run"] = shipped2["dry_run"] = True
    assert classify_mem(mem, shipped2)["state"] == "INDISTINGUISHABLE"


def test_infra_failure_on_error_short_rows_dead_probe_or_missing_counters():
    mem = _valid_mem()
    mem["error"] = "Traceback: boom"
    assert classify_mem(mem, _valid_shipped())["state"] == "INFRA_FAILURE"

    mem2 = _valid_mem()
    mem2["rows"] = mem2["rows"][:5]
    assert classify_mem(mem2, _valid_shipped())["state"] == "INFRA_FAILURE"

    shipped = _valid_shipped()
    shipped["behavior"]["corpus"]["actions"] = 3
    assert classify_mem(_valid_mem(), shipped)["state"] == "INFRA_FAILURE"

    mem3 = _valid_mem()
    del mem3["patch_diagnostics"]["mem"]
    out = classify_mem(mem3, _valid_shipped())
    assert out["state"] == "INFRA_FAILURE"
    assert any("counters absent" in r for r in out["reasons"])


def test_classify_mem_cli_round_trip(tmp_path, capsys):
    import classify_mem as cli

    m, s = tmp_path / "mem.json", tmp_path / "shipped.json"
    m.write_text(json.dumps(_valid_mem(1.0)))
    s.write_text(json.dumps(_valid_shipped(1.0)))
    rc = cli.main([str(m), str(s)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["state"] == "INDISTINGUISHABLE"


# --- builder ----------------------------------------------------------------------


def test_mem_notebook_carries_stack_but_no_duck_patches_bytes(tmp_path):
    text = _nb_text(build_mem(output_root=tmp_path))
    for marker in DUCK_PATCHES_DEFINITION_MARKERS:
        assert marker not in text, f"duck_patches bytes leaked into the mem arm: {marker!r}"
    for marker in MEM_DEFINITION_MARKERS:
        assert marker in text, f"harness_mem bytes missing from the mem arm: {marker!r}"
    # the inlined stack is byte-identical to the source file
    assert HARNESS_MEM.read_text() in text


def test_mem_notebook_carries_guard_counters_probe_and_prelude(tmp_path):
    nb_path = build_mem(output_root=tmp_path)
    text = _nb_text(nb_path)
    # behavioural probe, byte-identical machinery to the other arms
    assert "def install() -> bool" in text and "dead_reissue" in text
    assert "behav_assert = assert_observed" in text
    # anti-apply guard + captured-apply asserts + counters in the hook
    assert "ANTI-APPLY GUARD" in text
    assert "sandbox-alive" in text
    for line in MEM_MARKER_LINES:
        assert repr(line) in text or line in text, f"marker not asserted: {line!r}"
    assert "MEM_DIAGNOSTICS" in text and "_mem_install_trim_counter" in text
    # driver shims in the run cell, AFTER the driver and BEFORE pc_main
    run_src = next(
        "".join(c["source"]) for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code" and "await pc_main" in "".join(c["source"]))
    assert "mem_no_duck_patches_markers" in run_src
    assert run_src.index("async def pc_main") < run_src.index("mem_no_duck_patches_markers")
    assert run_src.index("mem_no_duck_patches_markers") < run_src.index("await pc_main")
    assert "collect_patch_diagnostics" in run_src
    # the standard machinery is still present
    assert "arc_agi_3_wheels" in text                       # install cell
    assert "force the serve" in text                        # serve forced
    assert "behavioral_emphasis" in run_src and "reading=" in run_src


def test_mem_contract_and_metadata(tmp_path):
    nb_path = build_mem(output_root=tmp_path)
    contract = notebook_contract(nb_path)
    assert contract["arm"] == "mem"
    assert contract["arm_env"] == MEM_ENV
    assert contract["hypothesis"] == MEM_HYPOTHESIS
    assert contract["geometry"] == GEOMETRY          # full 25 games, frozen box
    assert contract["patch_sha256"] == hashlib.sha256(_mem_patch_bytes()).hexdigest()
    assert contract["patch_sha256"] != UNPATCHED_SENTINEL
    assert len(contract["source_base_sha256"]) == 64

    build_mem()  # regenerate the repo copy (idempotent)
    meta = json.loads(
        (REPO / "submission/_ab_patch_closure/mem/kernel-metadata.json").read_text())
    assert meta["id"] == f"ahmedmobasher86/{MEM_SLUG}"
    assert meta["code_file"] == "patch-closure-mem.ipynb"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")
    # a NEW slug: it can never overwrite the existing arms
    assert MEM_SLUG not in ("arc-agi-3-patch-closure-base",
                            "arc-agi-3-patch-closure-candidate",
                            "arc-agi-3-patch-closure-shipped",
                            "arc-agi-3-package-screen",
                            "arc-agi-3-struct-screen",
                            "arc-agi-3-duck-patched",
                            "arc-agi-3-duck-mem")


def test_mem_build_is_deterministic(tmp_path):
    a = build_mem(output_root=tmp_path / "a").read_bytes()
    b = build_mem(output_root=tmp_path / "b").read_bytes()
    assert a == b


def test_hook_guard_counters_and_prelude_land_verbatim(tmp_path):
    text = _nb_text(build_mem(output_root=tmp_path))
    for chunk in (MEM_GUARD, MEM_COUNTERS_BLOCK, MEM_PRELUDE):
        for line in chunk.splitlines():
            if line.strip():
                assert line in text, f"missing built line: {line!r}"
    # and the assembled hook is exactly what the builder emits
    for line in mem_hook_cell().splitlines():
        if line.strip():
            assert line in text, f"missing hook line: {line!r}"


def test_other_arms_are_untouched(tmp_path):
    """base/candidate must still inline duck_patches + APPLY_BLOCK; shipped
    must still carry NO stack at all; none may carry mem machinery."""
    for arm in ("base", "candidate"):
        text = _nb_text(build_arm(arm, output_root=tmp_path))
        assert "def apply_all(" in text and "_pc_patch_results = apply_all()" in text
        assert "MEM_DIAGNOSTICS" not in text
        assert "[duck-mem]" not in text
    shipped_text = _nb_text(build_shipped(output_root=tmp_path))
    assert "MEM_DIAGNOSTICS" not in shipped_text
    assert "[duck-mem]" not in shipped_text
    assert "def patch_token_estimator(" not in shipped_text


# --- GPU-free end-to-end ----------------------------------------------------------


def test_mem_dry_run_end_to_end(tmp_path):
    """Runs dry_run_mem.py in a SUBPROCESS (same isolation rationale as the
    closure/package/shipped dry-run tests: patches are process-global). The
    script hard-asserts the shipped arm on clean classes, the mem arm's
    captured [duck-mem] markers + P1-P4 proof + exact {guard_fires: 1,
    trims: 1} counters, and an INDISTINGUISHABLE verdict from classify_mem."""
    import os
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "dry_run_mem.py")],
        env={**os.environ, "PC_DRY_WORKDIR": str(tmp_path)},
        capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (
        f"mem dry run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    assert "state=INDISTINGUISHABLE" in proc.stdout, proc.stdout[-2000:]

    workroot = next(p for p in sorted(tmp_path.iterdir()) if p.is_dir())
    mem = json.loads((workroot / "mem" / "patch_closure_result.json").read_text())
    assert mem["arm"] == "mem" and mem["stage"] == "done"
    assert mem["arm_env"] == MEM_ENV
    assert mem["patch_diagnostics"]["mem"] == {"guard_fires": 1, "trims": 1}
    for key in MEM_PROOF_KEYS:
        assert mem["identity"]["patch_proof"][key] is True, key
    shipped = json.loads(
        (workroot / "shipped" / "patch_closure_result.json").read_text())
    assert shipped["arm"] == "shipped" and shipped["stage"] == "done"
    assert shipped["patch_sha256"] == UNPATCHED_SENTINEL
