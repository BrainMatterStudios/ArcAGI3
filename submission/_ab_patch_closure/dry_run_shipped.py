#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the SHIPPED-vs-BASE pair.

Mock brain <- REAL duck harness (scratchpad/taaf_scored_ref bundle) <- REAL
CompetitionArcadeServer at the full 28-clone geometry <- the REAL pc_driver.
Same machinery as dry_run.py with the arm ORDER INVERTED for a hard reason:
patches are process-global monkey-patches, so the SHIPPED arm must run FIRST
on verifiably clean classes; only then is duck_patches loaded exactly as the
base kernel loads it, BASE_ENV pinned, and the base arm run.

The shipped arm executes the EXACT SHIPPED_PRELUDE source the builder inlines
(compiled and executed into the driver module's namespace — the
notebook-equivalent seam), then the original driver helpers are RESTORED for
the base arm. This proves:
  * pc_main completes end-to-end with NO patch layer (the two shims suffice;
    nothing else in the driver assumes patches);
  * the absence proof lands in identity.patch_proof and the purity fields
    (zero animation deliveries, zero graph sessions, empty stall_s_observed);
  * the base arm still runs as the settled patched config afterwards;
  * classify_shipped reads the pair: INDISTINGUISHABLE expected (identical
    mock policy in both arms — the dry run does not encode a forced verdict).

Run:  .venv/bin/python submission/_ab_patch_closure/dry_run_shipped.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
TOOLKIT = REPO / "reference/arc-agi-toolkit"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dry_run import (  # noqa: E402  (mock brain + kernel-shaped patch loader)
    DRY_GEOMETRY,
    DRY_MAX_ACTIONS,
    MockBrain,
    load_duck_patches_like_the_kernel,
)
from patch_closure_config import ARM_ENV, HYPOTHESIS as CLOSURE_HYPOTHESIS  # noqa: E402
from shipped_screen_config import (  # noqa: E402
    SHIPPED_ENV,
    SHIPPED_HYPOTHESIS,
    SHIPPED_READING,
    UNPATCHED_SENTINEL,
    classify_shipped,
)


def _setup() -> dict:
    """Clean-process environment: mock brain, bundle paths, probe, driver.
    NO patch layer — the shipped arm must see virgin classes."""
    import http.server
    import threading

    workroot = Path(os.environ.get(
        "PC_DRY_WORKDIR", str(REPO / "scratchpad" / "shipped_dry_run"))) / time.strftime("%H%M%S")
    workroot.mkdir(parents=True, exist_ok=True)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[shipped-dry] mock brain at {base_url}")

    # The shipped config carries NO TAAF_* arm pins; scrub any inherited ones
    # so the arm genuinely runs on duck-base defaults (duck-base itself sets
    # only TAAF_RUN_AS_SUBMISSION / TAAF_MINIMAL_DIAGNOSTICS / ONLY_RESET_LEVELS).
    for key in [k for k in os.environ if k.startswith("TAAF_")]:
        del os.environ[key]
    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b",
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true",
        "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(workroot / "server_recording"),
    })

    assert TOOLKIT.is_dir(), f"missing {TOOLKIT}"
    sys.path.insert(0, str(TOOLKIT))
    for p in (TAAF_ROOT / "src/ARC3-Inference", TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))
    sys.path.insert(0, str(REPO / "submission/_rig"))

    # -- the shipped hook's anti-apply guard, process-level equivalent ---------
    assert "duck_patches" not in sys.modules
    from inference.framework import solver as duck_solver
    cls = duck_solver._HarnessGameSession
    for name in ("should_stop", "step_env", "_execute_action", "play"):
        markers = [a for a in vars(getattr(cls, name)) if a.endswith("_patched")]
        assert not markers, f"clean-process guard: {name} carries {markers}"
    print("[shipped-dry] anti-apply guard: patch layer verifiably ABSENT")

    from inference.agent import python_tool_sandbox as ptx
    sbx = ptx.run_sandboxed_python(
        code="print('sandbox-alive')", timeout_seconds=20,
        initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
        action_handler=lambda actions: {"result": [], "state": {}})
    assert "sandbox-alive" in str(sbx.get("stdout", "")), f"sandbox DEAD: {sbx}"
    print("[shipped-dry] UNPATCHED sandbox liveness: OK")

    import behav_probe
    assert behav_probe.install(), "probe failed to install"

    spec = importlib.util.spec_from_file_location("pc_driver", HERE / "pc_driver.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    import build_patch_closure as bld
    source_hash, patch_hash = bld._source_hashes()

    solver = duck_solver.HarnessSolver(
        label="shipped-dry", model="mock-27b", analyzer_timeout=30,
        max_actions_per_game=DRY_MAX_ACTIONS,
        max_runtime_s_per_game=float(DRY_GEOMETRY["per_game_s"]),
        concurrency=DRY_GEOMETRY["concurrency"])

    return {"workroot": workroot, "drv": drv, "behav_probe": behav_probe,
            "solver": solver, "source_hash": source_hash, "patch_hash": patch_hash}


def run_shipped_arm(state: dict) -> dict:
    """The shipped arm on clean classes, with the builder's EXACT prelude."""
    from build_shipped_screen import SHIPPED_PRELUDE

    drv = state["drv"]
    saved = {"_pc_patch_proof": drv._pc_patch_proof,
             "_pc_install_animation_counter": drv._pc_install_animation_counter}
    # The notebook seam: the prelude source runs in the driver namespace, so
    # pc_main resolves the rebound helpers exactly as the kernel will.
    prelude_code = compile(SHIPPED_PRELUDE, "<shipped-prelude>", "exec")
    exec(prelude_code, drv.__dict__)  # noqa: S102 - fixed repo-owned source
    out_dir = state["workroot"] / "shipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(drv.pc_main(
            bm=SimpleNamespace(solver=state["solver"]), target=None,
            working_dir=out_dir, arm="shipped", arm_env=dict(SHIPPED_ENV),
            hypothesis=SHIPPED_HYPOTHESIS, geometry=dict(DRY_GEOMETRY),
            source_base_sha256=state["source_hash"],
            patch_sha256=UNPATCHED_SENTINEL,
            reading=dict(SHIPPED_READING),
            behav_report=state["behav_probe"].report,
            behav_assert=state["behav_probe"].assert_observed,
            dry_run=True))
    finally:
        for name, fn in saved.items():
            setattr(drv, name, fn)

    assert result["error"] is None, f"shipped arm recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert result["arm"] == "shipped" and result["arm_env"] == {}
    assert result["patch_sha256"] == UNPATCHED_SENTINEL
    assert len(result["rows"]) == DRY_GEOMETRY["clones"], len(result["rows"])
    assert len(result["rows_by_source"]) == 25, sorted(result["rows_by_source"])
    proof = result["identity"]["patch_proof"]
    assert proof.get("shipped_no_patch_markers") is True, proof
    assert result["identity"]["animation_prompt_probe"]["present"] is False
    pd = result["patch_diagnostics"]
    assert pd["animation"]["payload_deliveries"] == 0, pd["animation"]
    assert pd["graph"]["sessions_with_graph_state"] == 0, pd["graph"]
    assert pd["watchdog"]["stall_s_observed"] == [], pd["watchdog"]
    # Aggregate, not per-row: without ANTIFREEZE this arm CAN legitimately burn
    # a whole mock box deliberating on one game without acting (observed once
    # on su15, state=gave_up, 71790 analyzer tokens, 0 actions — that is
    # shipped-config behaviour, not an instrument failure).
    played = [r for r in result["rows"] if int(r["actions_total"] or 0) > 0]
    assert len(played) >= DRY_GEOMETRY["clones"] - 2, (
        f"only {len(played)}/{len(result['rows'])} clones ever acted")
    for row in played:
        assert row["gen_tokens"] > 0, f"gen_tokens zero on a played row: {row}"
    for row in result["rows"]:
        assert "watchdog" not in row and "graph" not in row, sorted(row)
    assert result["behavior"]["corpus"]["actions"] >= 50
    assert result["pre_registered_reading"]["hypothesis"].startswith(
        "The shipped unpatched config")
    on_disk = json.loads((out_dir / "patch_closure_result.json").read_text())
    assert on_disk["arm"] == "shipped" and on_disk["stage"] == "done"
    print(f"[shipped-dry] shipped arm: {len(result['rows'])} rows, absence proof OK, "
          f"purity fields clean")
    return result


def run_base_arm(state: dict) -> dict:
    """AFTER the shipped arm: load + apply the patch layer exactly as the
    (unchanged) base kernel does, pin BASE_ENV, run with the ORIGINAL driver
    helpers — the settled patched comparator."""
    os.environ.update(ARM_ENV["base"])
    dp = load_duck_patches_like_the_kernel()
    results = dp.apply_all()
    expected_skips = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
    bad = [line for line in results
           if "FAIL" in line or "REVIEW" in line
           or ("SKIP" in line and not line.startswith(expected_skips))]
    assert not bad, f"patch layer did not fully apply: {bad}"
    print("[shipped-dry] base arm: apply_all() OK (expected SKIPs verified)")

    drv = state["drv"]
    out_dir = state["workroot"] / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(drv.pc_main(
        bm=SimpleNamespace(solver=state["solver"]), target=None,
        working_dir=out_dir, arm="base", arm_env=ARM_ENV["base"],
        hypothesis=CLOSURE_HYPOTHESIS, geometry=dict(DRY_GEOMETRY),
        source_base_sha256=state["source_hash"], patch_sha256=state["patch_hash"],
        behav_report=state["behav_probe"].report,
        behav_assert=state["behav_probe"].assert_observed,
        dry_run=True))
    assert result["error"] is None, f"base arm recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert result["identity"]["patch_proof"]["watchdog_should_stop_patched"] is True
    assert result["identity"]["watchdog_stall_s_observed"] == [900.0]
    print(f"[shipped-dry] base arm: {len(result['rows'])} rows, patched identity OK")
    return result


def main() -> int:
    state = _setup()
    shipped = run_shipped_arm(state)
    base = run_base_arm(state)
    verdict = classify_shipped(shipped, base)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"state={verdict['state']}")
    assert verdict["state"] == "INDISTINGUISHABLE", (
        "dry-run verdict must be INDISTINGUISHABLE (identical mock policy): "
        f"{verdict['state']} — {verdict['reasons']}")
    assert MockBrain.n_posts > 0
    print(f"\n[shipped-dry] PASS — {MockBrain.n_posts} mock-brain calls, "
          f"artifacts under {state['workroot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
