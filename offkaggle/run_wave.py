#!/usr/bin/env python3
"""Off-Kaggle wave runner — ONE patch-closure arm at the frozen eval geometry
against a REMOTE exact-fidelity vLLM endpoint (offkaggle/modal_vllm_serve.py).

This is submission/_ab_patch_closure/dry_run_shipped.py with the mock brain
replaced by the remote endpoint and the mock geometry replaced by the
registered one: the REAL duck harness (scratchpad/taaf_scored_ref bundle), the
REAL CompetitionArcadeServer over environment_files, the REAL pc_driver, the
REAL pickled scored solver (benchmark_initial.pkl: concurrency 28,
max_runtime_s_per_game 7920, analyzer_timeout 900, max_actions_per_game None),
and the REAL behavioural probe. Results are pc_driver's own
patch_closure_result.json, written under offkaggle/results/<ts>-<arm>/ so
classify_shipped.py reads them unchanged.

Arms (one PROCESS per arm — patches are process-global monkey-patches, so the
shipped arm must run in a process that never loads duck_patches):
  shipped — duck-base v2 semantics: NO patch layer, NO TAAF_* pins. The
            builder's exact SHIPPED_PRELUDE is exec'd into the driver
            namespace (the notebook seam) and the clean-process anti-apply
            guard runs first.
  base    — the settled patched comparator: duck_patches loaded EXACTLY as
            the kernel loads it (source exec, no __file__), apply_all with
            the expected-SKIP assertions, BASE_ENV pinned.

Auth: the duck harness natively sends `Authorization: Bearer <key>` on every
analyzer call when LOCAL_ANALYZER_API_KEY is set (tool_agent._headers ->
build_headers). pc_driver's serving probe hits /models WITHOUT auth — the
Modal proxy exempts GET /v1/models for exactly that reason.

Run:
    .venv/bin/python offkaggle/run_wave.py --arm shipped \
        --base-url https://<you>--arc3-vllm-serve.modal.run/v1 \
        --token "$ARC3_VLLM_TOKEN"
    .venv/bin/python offkaggle/run_wave.py --arm base ...  # fresh process

Then classify the pair:
    .venv/bin/python submission/_ab_patch_closure/classify_shipped.py \
        offkaggle/results/<ts>-shipped/patch_closure_result.json \
        offkaggle/results/<ts>-base/patch_closure_result.json

NOTE ON DERATING: --concurrency/--per-game-s/--clones exist for diagnostics,
but the recorded geometry is always the REALIZED one — classify_shipped
refuses any pair whose geometry differs from the registered
{28 clones, 7920 s, concurrency 28} (unless both arms are dry runs). A derated
wave is a plumbing check, never a certification.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
AB = REPO / "submission/_ab_patch_closure"
RIG = REPO / "submission/_rig"
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
TOOLKIT = REPO / "reference/arc-agi-toolkit"
DEFAULT_OUT = REPO / "offkaggle/results"

RESULT_FILENAME = "patch_closure_result.json"  # pc_driver's own artifact name
ARMS = ("shipped", "base")

# ARC3_WAVE_SERVED_MODEL overrides for candidate-brain waves (e.g. the
# Qwen3.8 arm on arc3-vllm38, 2026-08-15). Default = the scored contract;
# test_offkaggle.py pins the default, not the override.
SERVED_MODEL_NAME = os.environ.get(
    "ARC3_WAVE_SERVED_MODEL", "vrfai/Qwen3.6-27B-FP8")

# ---------------------------------------------------------------------------
# The scored analyzer env block, copied FAITHFULLY from the setup_env dict in
# `git show HEAD:scratchpad/taaf_scored_ref/setup_commands.json` (test-pinned
# by test_offkaggle.py against that blob). Two deliberate deviations, both
# structural, zero behavioural:
#   * PYTHONPATH is omitted — in the blob it prepends the Kaggle wheelhouse
#     target dir; locally the .venv provides the packages.
#   * LOCAL_ANALYZER_BASE_URL / OPENAI_BASE_URL are set at runtime from
#     --base-url (in the blob they point at the kernel-local vLLM).
# ---------------------------------------------------------------------------
SETUP_ENV = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}
# Keys of the blob's setup_env this runner fills at runtime instead:
OVERRIDDEN_AT_RUNTIME = ("LOCAL_ANALYZER_BASE_URL", "OPENAI_BASE_URL")
# Keys of the blob's setup_env deliberately not reproduced locally:
SETUP_ENV_EXCLUDED = ("PYTHONPATH",)


def result_dir(out_root: Path | str, arm: str, ts: str) -> Path:
    """<out_root>/<timestamp>-<arm>/ — pc_main writes RESULT_FILENAME inside."""
    return Path(out_root) / f"{ts}-{arm}"


def arm_contract(arm: str) -> dict:
    """The frozen per-arm contract, resolved from the rig's own config modules
    (never re-declared here — the rig files are the source of truth)."""
    if str(AB) not in sys.path:
        sys.path.insert(0, str(AB))
    from patch_closure_config import ARM_ENV, HYPOTHESIS
    from shipped_screen_config import (
        SHIPPED_ENV, SHIPPED_HYPOTHESIS, SHIPPED_READING, UNPATCHED_SENTINEL)

    if arm == "shipped":
        return {"arm_env": dict(SHIPPED_ENV), "hypothesis": SHIPPED_HYPOTHESIS,
                "reading": dict(SHIPPED_READING),
                "patch_sha256": UNPATCHED_SENTINEL}
    if arm == "base":
        return {"arm_env": dict(ARM_ENV["base"]), "hypothesis": HYPOTHESIS,
                "reading": None, "patch_sha256": None}  # None -> real hash later
    raise ValueError(f"unknown arm {arm!r} (choose from {ARMS})")


# --- environment ------------------------------------------------------------


def _install_env(arm: str, base_url: str, token: str, workdir: Path) -> None:
    """Full process env BEFORE any harness import (tool_agent reads its
    _LOCAL_ANALYZER_* constants at import time)."""
    # Patch bytes must be traceable to a commit: pin the builder's source ref
    # so _patches_bytes()/_source_hashes() read HEAD's blob, not working-tree
    # bytes (the 2026-08-09 uncommitted-patch20 near-miss).
    os.environ.setdefault("PC_PATCHES_REF", "HEAD")

    # The shipped config carries NO TAAF_* pins; scrub inherited ones so both
    # arms start from a clean slate (dry_run_shipped._setup's scrub).
    for key in [k for k in os.environ if k.startswith("TAAF_")]:
        del os.environ[key]

    os.environ.update(SETUP_ENV)
    os.environ.update({
        # runtime halves of the blob's setup_env:
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_BASE_URL": base_url,
        # bearer for every analyzer call (tool_agent._headers chain):
        "LOCAL_ANALYZER_API_KEY": token,
        # duck-base cell-2 mirror for a non-submission run:
        "MPLBACKEND": "Agg",
        "TAAF_RUN_AS_SUBMISSION": "0",
        "TAAF_MINIMAL_DIAGNOSTICS": "0",
        "ONLY_RESET_LEVELS": "true",
        # local competition-sim plumbing (dry_run mirror):
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(workdir / "server_recording"),
    })
    if arm == "base":
        from patch_closure_config import ARM_ENV  # AB on path via arm_contract
        os.environ.update(ARM_ENV["base"])


def _install_paths() -> None:
    for p in (AB, RIG):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    assert TOOLKIT.is_dir(), f"missing {TOOLKIT}"
    sys.path.insert(0, str(TOOLKIT))  # arc_agi with OperationMode.COMPETITION
    for p in (TAAF_ROOT / "src/ARC3-Inference",
              TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))


# --- preflight --------------------------------------------------------------


def _preflight(base_url: str, token: str, deadline_s: float) -> None:
    """Fail in seconds, not after a 2 h wave: wake the endpoint, check the
    served id UNauthenticated (exactly what pc_driver's probe will do), then
    prove the bearer works with one real chat completion."""
    base = base_url.rstrip("/")
    deadline = time.monotonic() + deadline_s
    last: Exception | None = None
    print(f"[wave] preflight: waking {base} (cold start can take ~10-15 min)",
          flush=True)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/models", timeout=600) as r:
                ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            if SERVED_MODEL_NAME not in ids:
                raise RuntimeError(f"endpoint serves {ids}, "
                                   f"expected {SERVED_MODEL_NAME!r}")
            break
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            print(f"[wave] endpoint not ready yet ({e!r}); retrying...", flush=True)
            time.sleep(15)
    else:
        raise TimeoutError(f"endpoint never became ready: {last!r}")
    print("[wave] /models OK (unauthenticated, as pc_driver probes it)", flush=True)

    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with the word: ok"}],
            "temperature": 0.0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    content = body["choices"][0]["message"].get("content", "")
    print(f"[wave] authenticated completion OK: {content.strip()!r}", flush=True)


# --- arm setup (dry_run_shipped machinery, remote brain) --------------------


def _setup_arm(arm: str) -> dict:
    from dry_run import load_duck_patches_like_the_kernel  # kernel-shaped loader

    if arm == "shipped":
        # Clean-process anti-apply guard: the shipped arm must see virgin
        # classes (dry_run_shipped._setup's guard, verbatim semantics).
        assert "duck_patches" not in sys.modules, (
            "duck_patches already loaded — the shipped arm needs a fresh process")
        from inference.framework import solver as duck_solver
        cls = duck_solver._HarnessGameSession
        for name in ("should_stop", "step_env", "_execute_action", "play"):
            markers = [a for a in vars(getattr(cls, name)) if a.endswith("_patched")]
            assert not markers, f"clean-process guard: {name} carries {markers}"
        print("[wave] anti-apply guard: patch layer verifiably ABSENT", flush=True)
    else:
        dp = load_duck_patches_like_the_kernel()
        results = dp.apply_all()
        expected_skips = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
        bad = [line for line in results
               if "FAIL" in line or "REVIEW" in line
               or ("SKIP" in line and not line.startswith(expected_skips))]
        assert not bad, f"patch layer did not fully apply: {bad}"
        for prefix in expected_skips:
            line = next((l for l in results if l.startswith(prefix)), "")
            assert "SKIP" in line, f"expected {prefix} SKIP, got {line!r}"
        print("[wave] base arm: apply_all() OK (expected SKIPs verified)", flush=True)

    from inference.agent import python_tool_sandbox as ptx
    sbx = ptx.run_sandboxed_python(
        code="print('sandbox-alive')", timeout_seconds=20,
        initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
        action_handler=lambda actions: {"result": [], "state": {}})
    assert "sandbox-alive" in str(sbx.get("stdout", "")), f"sandbox DEAD: {sbx}"
    print("[wave] sandbox liveness: OK", flush=True)

    import behav_probe
    assert behav_probe.install(), "behavioural probe failed to install"

    spec = importlib.util.spec_from_file_location("pc_driver", AB / "pc_driver.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    import build_patch_closure as bld
    source_hash, patch_hash = bld._source_hashes()

    # The SCORED solver, not a reconstruction: the same pickles the GPU
    # kernels restore (concurrency 28, 7920 s box, analyzer_timeout 900,
    # max_actions_per_game None). pc_main re-pins per_game_s/concurrency from
    # the geometry argument, exactly as the kernels do.
    with open(TAAF_ROOT / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    with open(TAAF_ROOT / "deploy_target.pkl", "rb") as f:
        target = pickle.load(f)
    target.actual_run_as_submission = False   # kernel cell 10 with
    target.is_competition_rerun = False       # TRUE_SUBMISSION=False
    print(f"[wave] scored solver restored: label={bm.solver.label} "
          f"model={bm.solver.model} analyzer_timeout={bm.solver.analyzer_timeout} "
          f"max_actions_per_game={bm.solver.max_actions_per_game}", flush=True)

    if arm == "shipped":
        # The notebook seam: the builder's EXACT prelude rebinds the two
        # driver helpers that would otherwise hard-assert a patch layer.
        from build_shipped_screen import SHIPPED_PRELUDE
        exec(compile(SHIPPED_PRELUDE, "<shipped-prelude>", "exec"), drv.__dict__)  # noqa: S102

    return {"drv": drv, "behav_probe": behav_probe, "bm": bm, "target": target,
            "source_hash": source_hash, "patch_hash": patch_hash}


def _post_asserts(arm: str, result: dict, geometry: dict) -> None:
    assert result["error"] is None, f"{arm} arm recorded an error:\n{result['error']}"
    assert result["stage"] == "done", result["stage"]
    assert len(result["rows"]) == geometry["clones"], len(result["rows"])
    assert len(result["rows_by_source"]) == 25, sorted(result["rows_by_source"])
    pd = result["patch_diagnostics"]
    if arm == "shipped":
        proof = result["identity"]["patch_proof"]
        assert proof.get("shipped_no_patch_markers") is True, proof
        assert pd["animation"]["payload_deliveries"] == 0, pd["animation"]
        assert pd["graph"]["sessions_with_graph_state"] == 0, pd["graph"]
        assert pd["watchdog"]["stall_s_observed"] == [], pd["watchdog"]
    else:
        assert result["identity"]["patch_proof"]["watchdog_should_stop_patched"] is True
        stalls = set(pd["watchdog"]["stall_s_observed"])
        assert stalls == {900.0}, (
            f"base arm watchdog stall observed {stalls}, expected {{900.0}}")


# --- main -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible base URL incl. /v1, e.g. "
                             "https://<you>--arc3-vllm-serve.modal.run/v1")
    parser.add_argument("--token", default=os.environ.get("ARC3_VLLM_TOKEN", ""),
                        help="bearer token (default: $ARC3_VLLM_TOKEN)")
    parser.add_argument("--clones", type=int, default=28)
    parser.add_argument("--per-game-s", type=int, default=7920)
    parser.add_argument("--concurrency", type=int, default=28,
                        help="derate if the endpoint saturates — but a derated "
                             "wave cannot certify (geometry contract)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preflight-timeout", type=float, default=3600.0)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.token:
        parser.error("--token or $ARC3_VLLM_TOKEN is required")
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        parser.error(f"--base-url must end in /v1, got {args.base_url!r}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = result_dir(args.out, args.arm, ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = arm_contract(args.arm)          # also puts AB on sys.path
    _install_env(args.arm, base_url, args.token, out_dir)
    _install_paths()

    from patch_closure_config import GEOMETRY  # the registered contract
    geometry = {"clones": args.clones, "per_game_s": args.per_game_s,
                "concurrency": args.concurrency}
    if geometry != GEOMETRY:
        print(f"[wave] WARNING: geometry {geometry} != registered {GEOMETRY} — "
              "this wave is a PLUMBING CHECK; classify_shipped will refuse the "
              "pair (INVALID) at anything but the frozen geometry.", flush=True)

    if not args.skip_preflight:
        _preflight(base_url, args.token, args.preflight_timeout)

    state = _setup_arm(args.arm)
    patch_sha = contract["patch_sha256"] or state["patch_hash"]

    print(f"[wave] arm={args.arm} geometry={geometry} out={out_dir}", flush=True)
    t0 = time.time()
    result = asyncio.run(state["drv"].pc_main(
        bm=SimpleNamespace(solver=state["bm"].solver),
        target=state["target"],
        working_dir=out_dir,
        arm=args.arm,
        arm_env=contract["arm_env"],
        hypothesis=contract["hypothesis"],
        geometry=geometry,
        source_base_sha256=state["source_hash"],
        patch_sha256=patch_sha,
        reading=contract["reading"],
        behav_report=state["behav_probe"].report,
        behav_assert=state["behav_probe"].assert_observed))

    _post_asserts(args.arm, result, geometry)
    artifact = out_dir / RESULT_FILENAME
    assert artifact.is_file(), f"driver did not write {artifact}"
    print(f"\n[wave] DONE arm={args.arm} in {round((time.time() - t0) / 60, 1)} min")
    print(f"[wave] artifact: {artifact}")
    print("[wave] classify the pair with:\n"
          "  .venv/bin/python submission/_ab_patch_closure/classify_shipped.py "
          "<shipped>/patch_closure_result.json <base>/patch_closure_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
