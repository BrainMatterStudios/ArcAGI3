#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the PACKAGE-SCREEN arm (probe infrastructure).

Reuses dry_run.py's machinery (mock brain, scored-ref bundle, kernel-identical
duck_patches load, real CompetitionArcadeServer at the full 28-clone geometry,
the real pc_driver) with two package-specific twists:

  * the mock brain's sandbox code calls run_probe(...) once per turn before
    acting, so patch18's host-side probe executor runs for real;
  * the solver allows a few more actions per game so click-only games get
    turns after patch17's LLM-free opening wiggle battery has spent its
    presses on the directional games.

All four package mechanisms must fire NATURALLY and land in the result:
  patch16 diff-lines  -> reports/actions counters (prompt injection per turn)
  patch17 wiggle      -> batteries + presses (directional games), per-session
                         modes (click-only games classify CLICK at zero cost)
  patch18 run_probe   -> calls/actions (mock calls it each turn)
  patch19 dispatch    -> modes assigned histogram + swap counter

Expected verdict: STOP (mock play unlocks nothing and completes no levels —
far below the ADVANCE bars; no forced ADVANCE is encoded anywhere).

Run:  .venv/bin/python submission/_ab_patch_closure/dry_run_package.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dry_run as base_dry  # noqa: E402  (shares _setup: mock brain, bundle, patches)
from package_screen_config import (  # noqa: E402
    PACKAGE_ENV,
    PACKAGE_HYPOTHESIS,
    PACKAGE_READING,
    classify_screen,
)

DRY_MAX_ACTIONS = 6

# Mock policy: probe first (exercises patch18's real host-side executor), then
# act. Deterministic and state-driven, like the closure mock.
PKG_MOCK_TOOL_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "plan = [prefs[0]] if prefs else [{'action': 'MOUSE', 'row': 32, 'col': 32}]\n"
    "try:\n"
    "    run_probe(plan)\n"
    "except Exception as exc:\n"
    "    print(f'run_probe unavailable: {exc!r}')\n"
    "if prefs:\n"
    "    action(prefs[0])\n"
    "elif valid_actions:\n"
    "    action({'action': 'MOUSE', 'row': 32, 'col': 32})\n"
    "else:\n"
    "    action('UP')\n"
)


def dry_run_package(output_dir: Path) -> dict:
    """Run the package arm end-to-end; returns the result dict (same schema a
    GPU run writes, plus the embedded pre-registered reading)."""
    state = base_dry._setup()
    base_dry.MOCK_TOOL_CODE = PKG_MOCK_TOOL_CODE  # mock_reply reads it per call
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update(PACKAGE_ENV)
    drv = state["drv"]
    from inference.framework import solver as duck_solver
    solver = duck_solver.HarnessSolver(
        label="pc-dry-package", model="mock-27b", analyzer_timeout=30,
        max_actions_per_game=DRY_MAX_ACTIONS,
        max_runtime_s_per_game=float(base_dry.DRY_GEOMETRY["per_game_s"]),
        concurrency=base_dry.DRY_GEOMETRY["concurrency"])
    result = asyncio.run(drv.pc_main(
        bm=SimpleNamespace(solver=solver), target=None, working_dir=output_dir,
        arm="package", arm_env=PACKAGE_ENV,
        hypothesis=PACKAGE_HYPOTHESIS, geometry=dict(base_dry.DRY_GEOMETRY),
        source_base_sha256=state["source_hash"], patch_sha256=state["patch_hash"],
        behav_report=state["behav_probe"].report,
        behav_assert=state["behav_probe"].assert_observed,
        dry_run=True, reading=PACKAGE_READING))

    # -- hard artifact checks ---------------------------------------------------
    assert result["error"] is None, f"package arm recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert result["schema_version"] == 1 and result["hypothesis"] == PACKAGE_HYPOTHESIS
    assert result["arm"] == "package" and result["arm_env"] == PACKAGE_ENV
    assert result["pre_registered_reading"] == PACKAGE_READING
    assert len(result["rows"]) == base_dry.DRY_GEOMETRY["clones"], len(result["rows"])
    identity = result["identity"]
    assert identity["toggles_ok"] is True, identity
    assert identity["animation_prompt_probe"]["present"] is False, identity  # v7 pins
    assert identity["watchdog_stall_s_observed"] == [900.0], identity       # v7 pins

    pd = result["patch_diagnostics"]
    # patch17 wiggle: the opening battery must have run on the directional games
    assert pd["wiggle"]["batteries"] >= 1, pd["wiggle"]
    assert pd["wiggle"]["battery_presses"] >= 1, pd["wiggle"]
    # patch18 run_probe: the mock calls it every turn — real host-side execution
    assert pd["run_probe"]["calls"] >= 1, pd["run_probe"]
    assert pd["run_probe"]["actions"] >= 1, pd["run_probe"]
    # patch16 diff-lines: per-action capture + prompt injection
    assert pd["diff_lines"]["actions"] >= 1, pd["diff_lines"]
    assert pd["diff_lines"]["reports"] >= 1, pd["diff_lines"]
    # patch19 dispatch: modes assigned per session (click-only games -> CLICK)
    modes = pd.get("wiggle_modes_assigned") or {}
    assert sum(modes.values()) >= 1, modes
    assert modes.get("CLICK", 0) >= 1, modes
    assert "dispatch" in pd, sorted(pd)
    rows_with_wiggle = [r for r in result["rows"] if "wiggle" in r]
    assert rows_with_wiggle, "no per-row wiggle diagnostics landed"
    behav = result["behavior"]
    assert behav and behav["corpus"]["actions"] >= 50, behav
    on_disk = json.loads((output_dir / "patch_closure_result.json").read_text())
    assert on_disk["arm"] == "package" and on_disk["stage"] == "done"
    assert on_disk["pre_registered_reading"] == PACKAGE_READING

    print(f"[pkg-dry] 28 rows; wiggle={pd['wiggle']} run_probe={pd['run_probe']} "
          f"diff_lines={pd['diff_lines']} dispatch={pd['dispatch']} modes={modes}")
    return result


def main() -> int:
    state = base_dry._setup()
    root = state["workroot"]
    result = dry_run_package(root / "package")
    verdict = classify_screen(result)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"state={verdict['state']}")
    assert verdict["state"] == "STOP", (
        "package dry-run verdict must be STOP (no unlocks, no levels — the "
        f"screen encodes no forced ADVANCE): {verdict['state']} — {verdict['reasons']}")
    assert base_dry.MockBrain.n_posts > 0
    print(f"\n[pkg-dry] PASS — artifacts under {root / 'package'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
