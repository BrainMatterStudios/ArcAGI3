#!/usr/bin/env python3
"""Build the two patch-closure arm kernels (Task 2 of the 2026-08-04 plan).

One notebook per arm from the SAME duck-base source notebook and the SAME
duck_patches.py bytes; the arm environment mapping (patch_closure_config
BASE_ENV / CANDIDATE_ENV) is the only behavioural delta, pinned into the hook
cell IMMEDIATELY BEFORE apply_all(). NOT a competition submission: each arm is
a private GPU commit kernel on the submission/_rig mechanism (serve forced,
run cell replaced by pc_driver.pc_main at the frozen eval geometry — 28
competition-sim clones, 7920s per game, concurrency 28).

The base arm byte-reconstructs the SCORED v7 behaviour by explicit env pins
(stall 900 / animation 0 / graph 0) — never by trusting current code defaults.
The candidate arm's delta is exactly {TAAF_WATCHDOG_STALL_S=600,
TAAF_ANIMATION=1, TAAF_GRAPH=1}. The f0ec605 refill-aware HUD unmask guard is
part of duck_patches.py and therefore SHARED BASE in both arms.

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_patch_closure.py --all
    kaggle kernels push -p submission/_ab_patch_closure/base --accelerator NvidiaRtxPro6000
    kaggle kernels push -p submission/_ab_patch_closure/candidate --accelerator NvidiaRtxPro6000
(the accelerator flag is required — metadata machine_shape does NOT select the
RTX Pro 6000; July-proven on the sft kernels)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from patch_closure_config import ARM_ENV, GEOMETRY, HYPOTHESIS  # noqa: E402

BASE_NB = REPO / "submission/_duck_base/duck-base.ipynb"
PATCHES = REPO / "submission/_duck_patched/duck_patches.py"
PROBE = REPO / "submission/_rig/behav_probe.py"
DRIVER = HERE / "pc_driver.py"

ARM_SLUGS = {
    "base": "arc-agi-3-patch-closure-base",
    "candidate": "arc-agi-3-patch-closure-candidate",
}

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
SERVE_GUARD = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it"
RUN_MARKER = "# Build the live competition game list from the gateway's available environments."

APPLY_BLOCK = '''
# --- full patch application (both arms, identical bytes) ------------------------
# The arm env above is already pinned, so every call-time-gated patch reads its
# arm's switches. Hard gate (rig pack law: an unpatched arm comparison is
# meaningless): any FAIL/REVIEW aborts, and SKIP is allowed ONLY for the two
# lines this bundle is EXPECTED to skip —
#   patch9: the scored bundle's sandbox bootstrap lacks state_hash/diff_frames
#           (patch9 self-detects that and declines);
#   patch6: arcagi3-agent is deliberately not attached to these kernels.
# Both are asserted POSITIVELY: if either unexpectedly applied, the bundle under
# test is not the one this experiment was designed for — abort.
_pc_patch_results = apply_all()
_PC_EXPECTED_SKIPS = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
_pc_bad = [line for line in _pc_patch_results
           if "FAIL" in line or "REVIEW" in line
           or ("SKIP" in line and not line.startswith(_PC_EXPECTED_SKIPS))]
if _pc_bad:
    raise RuntimeError(f"[pc] patch layer did not fully apply: {_pc_bad}")
for _prefix in _PC_EXPECTED_SKIPS:
    _line = next((l for l in _pc_patch_results if l.startswith(_prefix)), "")
    if "SKIP" not in _line:
        raise RuntimeError(f"[pc] expected {_prefix} SKIP on this bundle, got: {_line!r}")
print("[pc] patch layer = apply_all(); expected SKIPs verified", flush=True)
# Prove the sandbox is ALIVE after patching (a dead sandbox = two zero-action arms).
from inference.agent import python_tool_sandbox as _pc_ptx
_pc_sbx = _pc_ptx.run_sandboxed_python(
    code="print('sandbox-alive')", timeout_seconds=20,
    initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
    action_handler=lambda actions: {"result": [], "state": {}})
if "sandbox-alive" not in str(_pc_sbx.get("stdout", "")):
    raise RuntimeError(f"[pc] python sandbox is DEAD after patching: {_pc_sbx}")
print("[pc] sandbox liveness: OK", flush=True)
'''

PROBE_BLOCK = '''
if not install():
    raise RuntimeError("[pc] behavioural probe failed to install")
behav_report = report
behav_assert = assert_observed
print("[pc] behavioural probe ACTIVE (identical in both arms)", flush=True)
'''


def _source_hashes() -> tuple[str, str]:
    return (
        hashlib.sha256(BASE_NB.read_bytes()).hexdigest(),
        hashlib.sha256(PATCHES.read_bytes()).hexdigest(),
    )


def _arm_pin_block(arm: str) -> str:
    env_literal = json.dumps(ARM_ENV[arm], indent=4)
    return (
        "\n# --- ARM ENVIRONMENT: pinned IMMEDIATELY BEFORE apply_all() ---------------------\n"
        "# (frozen in patch_closure_config.py; the classifier rejects any drift)\n"
        f'PC_ARM = "{arm}"\n'
        f"PC_ARM_ENV = {env_literal}\n"
        "import os as _pc_env_os\n"
        "_pc_env_os.environ.update(PC_ARM_ENV)\n"
        'print(f"[pc] arm={PC_ARM}: env pinned before apply_all(): {PC_ARM_ENV}", flush=True)\n'
    )


def hook_cell(arm: str) -> str:
    return (
        "# ============================================================================\n"
        f"# Patch-closure arm: {arm.upper()}. Inlined from\n"
        "# submission/_duck_patched/duck_patches.py and submission/_rig/behav_probe.py\n"
        "# by build_patch_closure.py — edit those and rebuild.\n"
        "# ============================================================================\n"
        f"{PATCHES.read_text()}\n"
        f"{_arm_pin_block(arm)}\n"
        f"{APPLY_BLOCK}\n"
        "# --- behavioural probe: identical in both arms, observes only ---\n"
        f"{PROBE.read_text()}\n"
        f"{PROBE_BLOCK}"
    )


def run_cell(arm: str, source_hash: str, patch_hash: str) -> str:
    return (
        "# patch-closure: competition-simulated single-arm run. Replaces duck-base's\n"
        "# submission cell (its gateway poll cannot succeed in a commit run).\n"
        "# NOT a submission; no submission.parquet.\n"
        'print((BUNDLE_DIR / "preamble.txt").read_text())\n'
        'os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))\n'
        "\n"
        f"{DRIVER.read_text()}\n"
        "\n"
        "_pc_result = await pc_main(\n"
        "    bm=bm, target=target, working_dir=WORKING_DIR,\n"
        "    arm=PC_ARM, arm_env=PC_ARM_ENV,\n"
        f"    hypothesis={HYPOTHESIS!r},\n"
        f"    geometry={json.dumps(GEOMETRY)},\n"
        f"    source_base_sha256={source_hash!r},\n"
        f"    patch_sha256={patch_hash!r},\n"
        "    behav_report=behav_report, behav_assert=behav_assert)\n"
        "if _pc_result[\"error\"] is not None:\n"
        "    raise RuntimeError(f\"[pc] arm run FAILED: {_pc_result['error']}\")\n"
    )


def build_arm(arm: str, output_root: Path | None = None) -> Path:
    """Build one arm's notebook + kernel metadata; returns the notebook path."""
    if arm not in ARM_SLUGS:
        raise ValueError(f"unknown arm {arm!r}")
    source_hash, patch_hash = _source_hashes()
    hook_src = hook_cell(arm)
    run_src = run_cell(arm, source_hash, patch_hash)
    # ast.parse, not compile(): IPython executes cells per-statement, so the
    # inlined modules' mid-cell `from __future__` lines are runtime-legal (the
    # COMPLETE ab-wmr kernels carry the same byte pattern) but a strict module
    # compile would reject them. Top-level await is notebook-only: strip it.
    ast.parse(hook_src)
    ast.parse(run_src.replace("await pc_main", "_ = pc_main"))

    nb = json.loads(BASE_NB.read_text())
    seen = {"serve": False, "hook": False, "run": False}
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            if SERVE_GUARD in src:
                src = src.replace(
                    SERVE_GUARD,
                    "if True:  # patch-closure: force the serve — a commit run must serve Qwen")
                seen["serve"] = True
            elif HOOK_MARKER in src:
                src = hook_src
                seen["hook"] = True
            elif RUN_MARKER in src:
                src = run_src
                seen["run"] = True
        out = dict(cell)
        out["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out["execution_count"] = None
            out["outputs"] = []
        cells.append(out)

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise SystemExit(f"never found anchors: {missing}")
    for i, (before, after) in enumerate(zip(nb["cells"], cells)):
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            src_i = "".join(cell["source"])
            for marker in ("await pc_main", "await bm.run_and_score", "await bm.run"):
                src_i = src_i.replace(marker, "_ = " + marker.split(" ", 1)[1])
            try:
                ast.parse(src_i)
            except SyntaxError:
                # duck-base cells may use other top-level awaits; fall back to
                # the interactive-await grammar before declaring breakage.
                compile(src_i, f"{arm}-cell{i}", "exec",
                        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)

    nb["cells"] = cells
    nb.setdefault("metadata", {})["patch_closure_contract"] = {
        "hypothesis": HYPOTHESIS,
        "arm": arm,
        "arm_env": dict(ARM_ENV[arm]),
        "source_base_sha256": source_hash,
        "patch_sha256": patch_hash,
        "geometry": dict(GEOMETRY),
    }

    arm_dir = (Path(output_root) if output_root is not None else HERE) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    nb_path = arm_dir / f"patch-closure-{arm}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))
    (arm_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{ARM_SLUGS[arm]}",
        "title": ARM_SLUGS[arm],
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        # NOTE: machine_shape is INERT — the accelerator is set by the CLI flag
        # at push time: kaggle kernels push -p <dir> --accelerator NvidiaRtxPro6000
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "ahmedmobasher86/taaf-src-hybrid",
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2) + "\n")
    return nb_path


def notebook_contract(nb_path: Path) -> dict:
    """The immutable build contract embedded in a built notebook's metadata."""
    return json.loads(Path(nb_path).read_text())["metadata"]["patch_closure_contract"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="build both arms")
    parser.add_argument("--arm", choices=sorted(ARM_SLUGS), help="build one arm")
    args = parser.parse_args(argv)
    arms = sorted(ARM_SLUGS) if (args.all or not args.arm) else [args.arm]
    fingerprints = {}
    for arm in arms:
        nb_path = build_arm(arm)
        contract = notebook_contract(nb_path)
        env_fp = hashlib.sha256(
            json.dumps(contract["arm_env"], sort_keys=True).encode()).hexdigest()[:12]
        fingerprints[arm] = env_fp
        print(f"wrote {nb_path}")
        print(f"  base_sha256={contract['source_base_sha256'][:12]}... "
              f"patch_sha256={contract['patch_sha256'][:12]}... arm_env_fp={env_fp}")
        print(f"  push with: kaggle kernels push -p {nb_path.parent} "
              f"--accelerator NvidiaRtxPro6000")
    if len(fingerprints) == 2 and len(set(fingerprints.values())) != 2:
        raise SystemExit("arm env fingerprints are NOT distinct — the arms are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
