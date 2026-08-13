#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the MEM-vs-SHIPPED pair.

Mock brain <- REAL duck harness (scratchpad/taaf_scored_ref bundle) <- REAL
CompetitionArcadeServer at the full 28-clone geometry <- the REAL pc_driver.
Arm ORDER is fixed for a hard reason: the duck-mem stack is a process-global
monkey-patch, so the SHIPPED arm must run FIRST on verifiably clean classes
(reusing dry_run_shipped's setup + shipped arm verbatim); only then is
harness_mem loaded exactly as the mem kernel inlines it, apply_all run with
captured-and-asserted [duck-mem] markers, the counters installed, and the mem
arm run with the builder's EXACT MEM_PRELUDE source.

Deterministic mechanism exercises (via pc_main's dry_exercise seam, inside
the run window so they land in patch_diagnostics.mem):
  * ONE P4 guard fire through the COUNTING _should_guard_no_effect wrapper
    (plus two non-fire probes that must NOT count);
  * ONE P2 middle-drop trim through the counting wrapper on the live class;
  * live-class checks: the sessions' step_env is the mem guard copy.
The mock policy issues single-action batches, so the guard can never fire
organically and the artifact's counters are exactly {guard_fires: 1, trims: 1}.

Verdict: classify_mem must read INDISTINGUISHABLE (identical mock policy in
both arms — the dry run does not encode a forced verdict).

Run:  .venv/bin/python submission/_ab_patch_closure/dry_run_mem.py
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dry_run import DRY_GEOMETRY, MockBrain  # noqa: E402
from dry_run_shipped import _setup, run_shipped_arm  # noqa: E402
from mem_screen_config import (  # noqa: E402
    MEM_ENV,
    MEM_HYPOTHESIS,
    MEM_MARKER_LINES,
    MEM_PROOF_KEYS,
    MEM_READING,
    classify_mem,
)


def load_harness_mem_like_the_kernel() -> types.ModuleType:
    """Exec harness_mem.py source into a fresh namespace, as the inlined hook
    cell does. Sourced through build_mem_screen._mem_patch_bytes() so the dry
    run exercises EXACTLY the bytes the builder inlines (MEM_PATCHES_REF
    honored; dirty working bytes refused)."""
    import build_mem_screen as bmem

    source = bmem._mem_patch_bytes().decode("utf-8")
    mod = types.ModuleType("duck_mem_inline")
    exec(compile(source, "<inlined harness_mem>", "exec"), mod.__dict__)  # noqa: S102
    return mod


def run_mem_arm(state: dict, shipped_hash_check: str) -> dict:
    """AFTER the shipped arm: apply the duck-mem stack exactly as the mem
    kernel does (captured apply_all + marker asserts + counters), then run
    pc_main with the builder's EXACT prelude."""
    import build_mem_screen as bmem

    # The kernel pins the arm env immediately before apply_all.
    os.environ.update(MEM_ENV)

    # -- the hook cell, dry-run seam: inline stack + captured apply ------------
    mem_ns = load_harness_mem_like_the_kernel()
    bundle_dir = state["workroot"] / "mem_bundle"  # no setup_commands.json here;
    bundle_dir.mkdir(parents=True, exist_ok=True)  # P6 is gated OFF anyway
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        mem_ns.apply_all(str(bundle_dir))
    apply_out = cap.getvalue()
    print(apply_out, end="")
    missing = [line for line in MEM_MARKER_LINES if line not in apply_out]
    assert not missing, f"duck-mem stack did not fully apply — missing: {missing}"
    print("[mem-dry] duck-mem P1-P4 marker lines all present; P5/P6 gated off")

    # -- the counters block, exec'd into the SAME namespace (the notebook seam:
    # _step_env resolves _should_guard_no_effect from its defining globals) ----
    exec(compile(bmem.MEM_COUNTERS_BLOCK, "<mem-counters>", "exec"),  # noqa: S102
         mem_ns.__dict__)
    assert mem_ns.MEM_DIAGNOSTICS == {"guard_fires": 0, "trims": 0}

    # -- the run cell, dry-run seam: EXACT MEM_PRELUDE in the driver namespace -
    drv = state["drv"]
    saved = {"_pc_patch_proof": drv._pc_patch_proof,
             "_pc_install_animation_counter": drv._pc_install_animation_counter,
             "collect_patch_diagnostics": drv.collect_patch_diagnostics}
    drv.MEM_DIAGNOSTICS = mem_ns.MEM_DIAGNOSTICS  # notebook globals equivalent
    prelude_code = compile(bmem.MEM_PRELUDE, "<mem-prelude>", "exec")
    exec(prelude_code, drv.__dict__)  # noqa: S102 - fixed repo-owned source

    def exercise(sessions):
        # (1) ONE deterministic P4 guard fire through the COUNTING wrapper —
        # the same route a real repeated_no_effect batch stop takes — plus two
        # probes that must NOT count.
        before = dict(mem_ns.MEM_DIAGNOSTICS)
        assert mem_ns._should_guard_no_effect(False, "UP", "UP") is True
        assert mem_ns._should_guard_no_effect(True, "UP", "UP") is False
        assert mem_ns._should_guard_no_effect(False, "UP", "DOWN") is False
        assert mem_ns.MEM_DIAGNOSTICS["guard_fires"] == before["guard_fires"] + 1
        # (2) ONE real middle-drop trim through the counting wrapper on the
        # live class (drops the head user+assistant block, keeps the tail).
        from inference.agent import tool_agent as ta
        hist = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
        dropped = ta.ToolAgent._drop_oldest_history_block(None, hist, preserve_recent=2)
        assert dropped is True and len(hist) == 2, (dropped, hist)
        assert mem_ns.MEM_DIAGNOSTICS["trims"] == before["trims"] + 1
        # (3) the live sessions actually run the mem step_env guard copy.
        assert sessions, "no sessions captured by the registry"
        import inference.framework.solver as sv
        assert getattr(sv._HarnessGameSession.step_env, "_mem", False)

    out_dir = state["workroot"] / "mem"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(drv.pc_main(
            bm=SimpleNamespace(solver=state["solver"]), target=None,
            working_dir=out_dir, arm="mem", arm_env=dict(MEM_ENV),
            hypothesis=MEM_HYPOTHESIS, geometry=dict(DRY_GEOMETRY),
            source_base_sha256=state["source_hash"],
            patch_sha256=hashlib.sha256(bmem._mem_patch_bytes()).hexdigest(),
            reading=dict(MEM_READING),
            behav_report=state["behav_probe"].report,
            behav_assert=state["behav_probe"].assert_observed,
            dry_run=True, dry_exercise=exercise))
    finally:
        for name, fn in saved.items():
            setattr(drv, name, fn)

    assert result["error"] is None, f"mem arm recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert result["arm"] == "mem" and result["arm_env"] == MEM_ENV
    assert result["source_base_sha256"] == shipped_hash_check
    assert len(result["rows"]) == DRY_GEOMETRY["clones"], len(result["rows"])
    assert len(result["rows_by_source"]) == 25, sorted(result["rows_by_source"])
    proof = result["identity"]["patch_proof"]
    for key in MEM_PROOF_KEYS:
        assert proof.get(key) is True, (key, proof)
    assert result["identity"]["animation_prompt_probe"]["present"] is False
    pd = result["patch_diagnostics"]
    assert pd["mem"] == {"guard_fires": 1, "trims": 1}, pd["mem"]
    assert pd["animation"]["payload_deliveries"] == 0, pd["animation"]
    assert pd["graph"]["sessions_with_graph_state"] == 0, pd["graph"]
    assert pd["watchdog"]["stall_s_observed"] == [], pd["watchdog"]
    played = [r for r in result["rows"] if int(r["actions_total"] or 0) > 0]
    assert len(played) >= DRY_GEOMETRY["clones"] - 2, (
        f"only {len(played)}/{len(result['rows'])} clones ever acted")
    assert result["behavior"]["corpus"]["actions"] >= 50
    assert result["pre_registered_reading"]["behavioral_emphasis"].startswith(
        "SECONDARY AND EMPHASIZED")
    on_disk = json.loads((out_dir / "patch_closure_result.json").read_text())
    assert on_disk["arm"] == "mem" and on_disk["stage"] == "done"
    print(f"[mem-dry] mem arm: {len(result['rows'])} rows, P1-P4 proof OK, "
          f"mem counters {pd['mem']}, duck_patches purity clean")
    return result


def main() -> int:
    state = _setup()
    shipped = run_shipped_arm(state)
    mem = run_mem_arm(state, shipped_hash_check=shipped["source_base_sha256"])
    verdict = classify_mem(mem, shipped)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"state={verdict['state']}")
    assert verdict["state"] == "INDISTINGUISHABLE", (
        "dry-run verdict must be INDISTINGUISHABLE (identical mock policy): "
        f"{verdict['state']} — {verdict['reasons']}")
    behav = verdict["metrics"]["behavioral_EMPHASIZED"]
    assert behav["mem_stack_activity"]["guard_fires"] == 1
    assert behav["mem_stack_activity"]["trims"] == 1
    assert MockBrain.n_posts > 0
    print(f"\n[mem-dry] PASS — {MockBrain.n_posts} mock-brain calls, "
          f"artifacts under {state['workroot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
