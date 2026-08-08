#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the STRUCT-SCREEN arm (plan channel).

Reuses dry_run.py's machinery (mock brain, scored-ref bundle, kernel-identical
duck_patches load, real CompetitionArcadeServer at the full 28-clone geometry,
the real pc_driver). The mock policy alternates deterministically between a
2-step plan (`action([a, a])`) and a bare single (`action(a)`) so both plan
channel routes execute for real: multi-step plans AND the auto-wrapped
1-plan migration path.

Verified naturally (assertions on the emitted result):
  patch21 struct   -> plans, plan_actions, wrapped_singles, plan_lengths
                      histogram with a multi-step bucket, reports injected;
                      LIVE actions-per-turn > 1.0 (the adoption metric)
  patch22 gates    -> brake/phase counters present (values are board-dependent)
  patch17 wiggle   -> batteries + presses + repaired-verdict modes
  patch16 diff     -> reports/actions counters
  patch19 dispatch -> modes histogram + swap/escape counters
  run_probe        -> ZERO calls (superseded channel stays quiet; the mock
                      never calls it and nothing advertises it)

Expected verdict: STOP (mock play unlocks nothing — no forced ADVANCE).

Run:  .venv/bin/python submission/_ab_patch_closure/dry_run_struct.py
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
from struct_screen_config import (  # noqa: E402
    STRUCT_ENV,
    STRUCT_HYPOTHESIS,
    STRUCT_READING,
    classify_struct,
)

DRY_MAX_ACTIONS = 6

# Deterministic alternation keyed off the sandbox `history` global: even-length
# history -> submit a 2-step plan; odd -> a bare single (auto-wrap route).
STRUCT_MOCK_TOOL_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "a = prefs[0] if prefs else ({'action': 'MOUSE', 'row': 32, 'col': 32}\n"
    "                            if valid_actions else 'UP')\n"
    "if len(history) % 2 == 0:\n"
    "    action([a, a])\n"
    "else:\n"
    "    action(a)\n"
)


def dry_run_struct(output_dir: Path) -> dict:
    """Run the struct arm end-to-end; returns the result dict (same schema a
    GPU run writes, plus the embedded pre-registered reading)."""
    state = base_dry._setup()
    base_dry.MOCK_TOOL_CODE = STRUCT_MOCK_TOOL_CODE  # mock_reply reads it per call
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update(STRUCT_ENV)
    drv = state["drv"]
    from inference.framework import solver as duck_solver
    solver = duck_solver.HarnessSolver(
        label="pc-dry-struct", model="mock-27b", analyzer_timeout=30,
        max_actions_per_game=DRY_MAX_ACTIONS,
        max_runtime_s_per_game=float(base_dry.DRY_GEOMETRY["per_game_s"]),
        concurrency=base_dry.DRY_GEOMETRY["concurrency"])
    result = asyncio.run(drv.pc_main(
        bm=SimpleNamespace(solver=solver), target=None, working_dir=output_dir,
        arm="struct", arm_env=STRUCT_ENV,
        hypothesis=STRUCT_HYPOTHESIS, geometry=dict(base_dry.DRY_GEOMETRY),
        source_base_sha256=state["source_hash"], patch_sha256=state["patch_hash"],
        behav_report=state["behav_probe"].report,
        behav_assert=state["behav_probe"].assert_observed,
        dry_run=True, reading=STRUCT_READING))

    # -- hard artifact checks ---------------------------------------------------
    assert result["error"] is None, f"struct arm recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert result["schema_version"] == 1 and result["hypothesis"] == STRUCT_HYPOTHESIS
    assert result["arm"] == "struct" and result["arm_env"] == STRUCT_ENV
    assert result["pre_registered_reading"] == STRUCT_READING
    assert len(result["rows"]) == base_dry.DRY_GEOMETRY["clones"], len(result["rows"])
    identity = result["identity"]
    assert identity["toggles_ok"] is True, identity
    assert identity["animation_prompt_probe"]["present"] is False, identity  # v7 pins
    assert identity["watchdog_stall_s_observed"] == [900.0], identity       # v7 pins

    pd = result["patch_diagnostics"]
    st = pd.get("struct") or {}
    # patch21: both plan routes executed for real
    assert st.get("plans", 0) >= 1, st
    assert st.get("plan_actions", 0) >= 1, st
    assert st.get("wrapped_singles", 0) >= 1, st
    lengths = st.get("plan_lengths") or {}
    assert any(int(k) >= 2 and v >= 1 for k, v in lengths.items()), lengths
    assert st.get("reports_injected", 0) >= 1, st
    assert st.get("nudges", 0) == 0, st  # the mock always submits a valid plan
    for key in ("brake_strips", "menu_strips", "phase_transitions",
                "scout_truncations", "score_flushes", "invalid_dropped"):
        assert key in st, (key, sorted(st))
    # THE adoption metric (cfeb92a definition): plan-actions per LLM
    # deliberation must exceed the banked base pair's 1.0 (mock alternates
    # 2-plans and singles -> expected ~1.5). The probe's actions_per_turn is
    # 1.0 by construction under TAAF_STRUCT (each step is its own batch).
    adoption = result["adoption"]
    assert adoption["llm_turns"] >= 1, adoption
    assert adoption["plan_actions_per_llm_turn"] is not None, adoption
    assert adoption["plan_actions_per_llm_turn"] > 1.0, adoption
    corpus = result["behavior"]["corpus"]
    # superseded channel stays quiet
    assert (pd.get("run_probe") or {}).get("calls", 0) == 0, pd.get("run_probe")
    # carried package mechanisms
    assert pd["wiggle"]["batteries"] >= 1, pd["wiggle"]
    assert pd["diff_lines"]["reports"] >= 1, pd["diff_lines"]
    assert "dispatch" in pd and (pd.get("wiggle_modes_assigned") or {}), pd.get("dispatch")
    assert corpus["actions"] >= 50, corpus
    on_disk = json.loads((output_dir / "patch_closure_result.json").read_text())
    assert on_disk["arm"] == "struct" and on_disk["stage"] == "done"
    assert on_disk["pre_registered_reading"] == STRUCT_READING

    print(f"[struct-dry] 28 rows; struct={st} adoption={adoption} "
          f"modes={pd.get('wiggle_modes_assigned')}")
    return result


def main() -> int:
    state = base_dry._setup()
    root = state["workroot"]
    result = dry_run_struct(root / "struct")
    verdict = classify_struct(result)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"state={verdict['state']}")
    assert verdict["state"] == "STOP", (
        "struct dry-run verdict must be STOP (no unlocks, no levels — the "
        f"screen encodes no forced ADVANCE): {verdict['state']} — {verdict['reasons']}")
    assert base_dry.MockBrain.n_posts > 0
    print(f"\n[struct-dry] PASS — artifacts under {root / 'struct'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
