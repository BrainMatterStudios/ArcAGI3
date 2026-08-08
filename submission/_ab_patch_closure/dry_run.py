#!/usr/bin/env python3
"""GPU-free end-to-end dry run of BOTH patch-closure arms (Task 3).

Mock brain (fixed `python` tool call -> action('UP'), usage counts so token
accounting is provable) <- REAL duck harness (scratchpad/taaf_scored_ref
bundle, the 2026-07-26 audit law) <- REAL CompetitionArcadeServer over the
repo's environment_files at the FULL 28-clone geometry <- the REAL pc_driver
(imported from this directory, the same file build_patch_closure.py inlines).

duck_patches.py is loaded EXACTLY as the kernel loads it: source exec'd into a
module namespace with NO __file__, so patch6 declines for want of
/kaggle/input/arcagi3-agent here too and the dry run exercises the kernel's
true patch surface (patch9 SKIP included).

arc_agi comes from reference/arc-agi-toolkit (inserted ahead of the venv's
0.9.1): the scored bundle's competition path references
OperationMode.COMPETITION, which the venv package predates.

Arm order matters and is fixed: BASE first, then CANDIDATE. Module-level
mechanism counters are cumulative in-process; running base first proves its
delta purity (zero animation deliveries / zero grinder motion) on a genuinely
clean counter.

Deterministic mechanism exercises (via pc_main's dry_exercise seam, inside the
run window so they land in patch_diagnostics):
  * animation delivery through the REAL patched run_sandboxed_python route
    with a seeded _ANIM_TLS frame — counted in the candidate arm, provably
    NOT delivered in the base arm (env gate);
  * ONE level-age grinder engagement recorded on a real session's graph state
    (schema-level: the age-trigger MECHANISM itself is proven end-to-end by
    submission/_duck_patched/test_frontier_graph.py, which runs green);
  * per-arm watchdog stall identity (900 base / 600 candidate) is checked by
    the driver itself from the sessions' realized _watchdog_state.

Equal levels are EXPECTED (identical mock policy in both arms), so the
default verdict is NO_GO — the dry run does not encode a forced GO.

Run:  .venv/bin/python submission/_ab_patch_closure/dry_run.py
"""
from __future__ import annotations

import asyncio
import http.server
import importlib.util
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
TOOLKIT = REPO / "reference/arc-agi-toolkit"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from patch_closure_config import ARM_ENV, HYPOTHESIS  # noqa: E402

# Reduced per-game box for the mock regime; clones/concurrency stay at the
# registered 28 so the row geometry is production-shaped. The classifier
# accepts the reduced box only because both results carry dry_run=True.
DRY_GEOMETRY = {"clones": 28, "per_game_s": 90, "concurrency": 28}
DRY_MAX_ACTIONS = 4

MOCK_CONTENT = (
    "World model: mock static world for the patch-closure dry run.\n"
    "Plan: probe with the first valid action."
)

# The panel spans all 25 games, 10 of which are click-only (measured on the
# first dry run: a fixed action('UP') leaves them at 0 actions for the whole
# box). The sandbox knows each game's valid_actions, so pick there — a key
# action when one exists, a fixed centre click otherwise. Deterministic and
# identical in both arms.
MOCK_TOOL_CODE = (
    "prefs = [v for v in valid_actions if str(v).upper() not in ('RESET', 'MOUSE')]\n"
    "if prefs:\n"
    "    action(prefs[0])\n"
    "elif valid_actions:\n"
    "    action({'action': 'MOUSE', 'row': 32, 'col': 32})\n"
    "else:\n"
    "    action('UP')\n"
)


def mock_reply():
    return {
        "id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": MOCK_CONTENT,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "python",
                                 "arguments": json.dumps({"code": MOCK_TOOL_CODE})},
                }],
            },
        }],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
    }


class MockBrain(http.server.BaseHTTPRequestHandler):
    n_posts = 0

    def log_message(self, *a):  # noqa: D102
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": "mock-27b"}]})
        else:
            self._send({})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        _ = self.rfile.read(n)
        MockBrain.n_posts += 1
        self._send(mock_reply())


def load_duck_patches_like_the_kernel() -> types.ModuleType:
    """Exec duck_patches.py source with NO __file__, as the inlined cell does.

    Sourced through build_patch_closure._patches_bytes() so the dry run
    exercises EXACTLY the bytes the builder inlines (PC_PATCHES_REF honored;
    dirty working bytes refused)."""
    import build_patch_closure as bld

    source = bld._patches_bytes().decode("utf-8")
    mod = types.ModuleType("duck_patches")
    mod.__dict__["__name__"] = "duck_patches"
    assert "__file__" not in mod.__dict__
    exec(compile(source, "<inlined duck_patches>", "exec"), mod.__dict__)
    sys.modules["duck_patches"] = mod
    return mod


_STATE: dict = {}


def _setup() -> dict:
    """One-time environment: mock brain, bundle paths, patch layer, probe,
    driver, solver. Memoized — both arms share it, exactly as two waves do."""
    if _STATE:
        return _STATE

    workroot = Path(os.environ.get(
        "PC_DRY_WORKDIR", str(REPO / "scratchpad" / "pc_dry_run"))) / time.strftime("%H%M%S")
    workroot.mkdir(parents=True, exist_ok=True)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[pc-dry] mock brain at {base_url}")

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
    sys.path.insert(0, str(TOOLKIT))  # arc_agi with OperationMode.COMPETITION
    for p in (TAAF_ROOT / "src/ARC3-Inference", TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))
    sys.path.insert(0, str(REPO / "submission/_rig"))

    # -- the patch layer, applied EXACTLY as the hook cell applies it ----------
    # (arm env is pinned per-arm at call time in dry_run_arm; every patch reads
    # its switches at call time, so one apply_all serves both arms — the
    # _ab_wmr wave precedent)
    os.environ.update(ARM_ENV["base"])
    dp = load_duck_patches_like_the_kernel()
    results = dp.apply_all()
    expected_skips = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
    bad = [line for line in results
           if "FAIL" in line or "REVIEW" in line
           or ("SKIP" in line and not line.startswith(expected_skips))]
    assert not bad, f"patch layer did not fully apply: {bad}"
    for prefix in expected_skips:
        line = next((l for l in results if l.startswith(prefix)), "")
        assert "SKIP" in line, f"expected {prefix} SKIP on this bundle, got {line!r}"
    print("[pc-dry] patch layer = apply_all(); expected SKIPs verified")

    from inference.agent import python_tool_sandbox as ptx
    sbx = ptx.run_sandboxed_python(
        code="print('sandbox-alive')", timeout_seconds=20,
        initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
        action_handler=lambda actions: {"result": [], "state": {}})
    assert "sandbox-alive" in str(sbx.get("stdout", "")), f"sandbox DEAD: {sbx}"
    print("[pc-dry] sandbox liveness: OK")

    import behav_probe
    assert behav_probe.install(), "probe failed to install"

    spec = importlib.util.spec_from_file_location("pc_driver", HERE / "pc_driver.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    import build_patch_closure as bld
    source_hash, patch_hash = bld._source_hashes()

    from inference.framework import solver as duck_solver
    solver = duck_solver.HarnessSolver(
        label="pc-dry", model="mock-27b", analyzer_timeout=30,
        max_actions_per_game=DRY_MAX_ACTIONS,
        max_runtime_s_per_game=float(DRY_GEOMETRY["per_game_s"]),
        concurrency=DRY_GEOMETRY["concurrency"])

    _STATE.update({
        "workroot": workroot, "dp": dp, "drv": drv, "ptx": ptx,
        "behav_probe": behav_probe, "solver": solver,
        "source_hash": source_hash, "patch_hash": patch_hash,
        "arms_run": [],
    })
    return _STATE


def _dry_exercise(state: dict, arm: str):
    """Deterministic mechanism exercises, run inside pc_main's collection
    window so their effects land in the arm's patch_diagnostics."""
    dp, drv, ptx = state["dp"], state["drv"], state["ptx"]

    def exercise(sessions):
        # (1) animation delivery through the REAL patched sandbox route.
        # Candidate: the seeded frame must be DELIVERED (counter +>=1).
        # Base: TAAF_ANIMATION=0 must gate it to zero deliveries.
        dp._ANIM_TLS.frames = [{
            "ascii": "0 0\n0 0", "step": 1, "level": 1,
            "shape": [2, 2], "grid": [[0, 0], [0, 0]],
        }]
        before = dict(drv.PC_ANIMATION_UPTAKE)
        out = ptx.run_sandboxed_python(
            code="print('anim-probe')", timeout_seconds=20,
            initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
            action_handler=lambda actions: {"result": [], "state": {}})
        assert "anim-probe" in str(out.get("stdout", "")), out
        delivered = drv.PC_ANIMATION_UPTAKE["payload_deliveries"] - before["payload_deliveries"]
        if arm == "candidate":
            assert delivered >= 1, (
                f"candidate arm: seeded animation frame was NOT delivered ({delivered})")
        else:
            assert delivered == 0, (
                f"base arm: animation delivered {delivered} times despite TAAF_ANIMATION=0")
        dp._ANIM_TLS.frames = []

        # (2) ONE level-age grinder engagement on a real session's graph state
        # (candidate only; schema-level — the age-trigger mechanism itself is
        # proven by test_frontier_graph.py's engine-driven e2e tests).
        if arm == "candidate":
            assert sessions, "no sessions captured by the registry"
            gs = dp._graph_session_state(sessions[0])
            gs["diag"]["grinder_engagements"] += 1
            gs["diag"]["grinder_age_triggers"] += 1
        else:
            leaked = [s for s in sessions if getattr(s, "_graph_state", None) is not None]
            assert not leaked, f"base arm: {len(leaked)} sessions grew graph state"

    return exercise


def dry_run_arm(arm: str, output_dir: Path) -> dict:
    """Run one arm end-to-end (mock brain, real harness, 28 clones) and return
    the result dict — the same schema a GPU run writes."""
    assert arm in ARM_ENV, arm
    state = _setup()
    if arm == "candidate":
        assert "base" in state["arms_run"], (
            "run the base arm first: module counters must be provably clean "
            "when base's delta purity is recorded")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update(ARM_ENV[arm])
    drv = state["drv"]
    bm = SimpleNamespace(solver=state["solver"])
    result = asyncio.run(drv.pc_main(
        bm=bm, target=None, working_dir=output_dir,
        arm=arm, arm_env=ARM_ENV[arm],
        hypothesis=HYPOTHESIS, geometry=dict(DRY_GEOMETRY),
        source_base_sha256=state["source_hash"], patch_sha256=state["patch_hash"],
        behav_report=state["behav_probe"].report,
        behav_assert=state["behav_probe"].assert_observed,
        dry_run=True, dry_exercise=_dry_exercise(state, arm)))

    # -- hard artifact checks (the _ab_wmr dry-run law: assert, don't hope) ----
    assert result["error"] is None, f"arm {arm} recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert result["schema_version"] == 1 and result["hypothesis"] == HYPOTHESIS
    assert result["arm"] == arm and result["arm_env"] == ARM_ENV[arm]
    assert len(result["rows"]) == DRY_GEOMETRY["clones"], len(result["rows"])
    assert len(result["rows_by_source"]) == 25, sorted(result["rows_by_source"])
    for row in result["rows"]:
        assert row["source_game"] and row["actions_total"] > 0, row
        assert row["gen_tokens"] > 0, f"gen_tokens zero: {row}"
        assert "watchdog" in row and "trace_len" in row, sorted(row)
        if arm == "base":
            assert "graph" not in row, f"graph state leaked into base: {row}"
    identity = result["identity"]
    assert identity["toggles_ok"] is True, identity
    assert identity["animation_prompt_probe"]["present"] is (arm == "candidate"), identity
    expected_stall = 900.0 if arm == "base" else 600.0
    assert identity["watchdog_stall_s_observed"] == [expected_stall], identity
    pd = result["patch_diagnostics"]
    if arm == "candidate":
        assert pd["animation"]["payload_deliveries"] >= 1, pd["animation"]
        assert pd["graph"]["grinder_engagements"] == 1, pd["graph"]
        assert pd["graph"]["grinder_age_triggers"] == 1, pd["graph"]
    else:
        assert pd["animation"]["payload_deliveries"] == 0, pd["animation"]
        assert pd["graph"]["grinder_engagements"] == 0, pd["graph"]
    assert pd["watchdog"]["stall_s_observed"] == [expected_stall], pd["watchdog"]
    behav = result["behavior"]
    assert behav and behav["corpus"]["actions"] >= 50, behav
    on_disk = json.loads((output_dir / "patch_closure_result.json").read_text())
    assert on_disk["arm"] == arm and on_disk["stage"] == "done"

    state["arms_run"].append(arm)
    print(f"[pc-dry] arm {arm}: {len(result['rows'])} rows, "
          f"animation={pd['animation']} grinder_engagements={pd['graph']['grinder_engagements']} "
          f"stall_s={pd['watchdog']['stall_s_observed']}")
    return result


def main() -> int:
    from patch_closure_config import classify_result

    state = _setup()
    root = state["workroot"]
    base = dry_run_arm("base", root / "base")
    candidate = dry_run_arm("candidate", root / "candidate")
    verdict = classify_result(base, candidate)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"state={verdict['state']}")
    assert verdict["state"] == "NO_GO", (
        "dry-run verdict must be NO_GO (equal levels, no forced GO): "
        f"{verdict['state']} — {verdict['reasons']}")
    assert MockBrain.n_posts > 0
    print(f"\n[pc-dry] PASS — {MockBrain.n_posts} mock-brain calls, artifacts under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
