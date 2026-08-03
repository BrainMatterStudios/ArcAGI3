#!/usr/bin/env python3
"""Build submission/_ab_wmr/ab-wmr.ipynb — ROUND 2: marginal value of patch11
(frontier graph) and patch12 (compaction + plan queue) over the submitted v6
config. Round 1 (WMR trio A/B, waves A,B,B,A) is preserved at commit 3c21726;
its results live in docs/test-artifacts-2026-08-02/AB-WMR-ANALYSIS-2026-08-03.md.

NOT a competition submission. A plain GPU commit kernel on the _rig mechanism
(submission/_rig/build_rig.py): duck-base notebook, serve forced, run cell
replaced by the A/B wave driver (ab_wave_driver.py). One vLLM session; arms
toggle ONLY env pins between counterbalanced waves B,C,D,D,C,B:
    B = v6 as submitted (WMR trio on, TAAF_GRAPH=0, TAAF_COMPACT=0)
    C = B + TAAF_GRAPH=1        D = B + TAAF_COMPACT=1

Every arm gets the IDENTICAL patch layer the submitted v6 kernel installs:
apply_all() from the CURRENT duck_patches.py (patches 11/12, the
marker-forwarding fix, grid-burner default OFF, the patch6 no-__file__ fix).
Expected SKIPs on this bundle, positively asserted by the apply gate:
  * patch9 hud-sandbox — the scored bundle's sandbox bootstrap has no
    state_hash/diff_frames; patch9 now detects that and declines (round 1 had
    to exclude it by hand for the same defect).
  * patch6 tool_agent_analyze — the arcagi3-agent dataset is deliberately NOT
    attached to this kernel (its heuristic prober would own the first 200
    actions of every game and confound all three arms against round 1).
Anything else SKIP/FAIL/REVIEW aborts before a GPU-minute is spent on games.

Round-2 instrumentation (see ab_wave_driver.py): fixed gen_tokens accounting
(round 1 read final_generated_tokens, which this bundle's solver never sets;
the real counts live on run.history records), per-game graph diagnostics in C
waves, COMPACT_DIAGNOSTICS wave deltas + per-game queue state in D waves.

Output artifact: /kaggle/working/ab_result.json (written after every wave and
on failure) + a compact summary table at the end of the log.

Usage:
    .venv/bin/python submission/_ab_wmr/build_ab_wmr.py
    kaggle kernels push -p submission/_ab_wmr
Env overrides at build time (baked into nothing — the driver reads them at
RUN time, so on Kaggle the defaults in ab_wave_driver.py apply):
    AB_GAMES, AB_WAVES, AB_BUDGET, AB_DEADLINE_S  (see ab_wave_driver.py)
"""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
PATCHES = REPO / "submission/_duck_patched/duck_patches.py"
PROBE = REPO / "submission/_rig/behav_probe.py"
DRIVER = Path(__file__).parent / "ab_wave_driver.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "ab-wmr.ipynb"
SLUG = "arc-agi-3-ab-wmr"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
SERVE_GUARD = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it"
RUN_MARKER = "# Build the live competition game list from the gateway's available environments."

APPLY_BLOCK = '''
# --- full v6 patch application (ALL arms, identical) ----------------------------
# Round 2 installs the SAME layer the submitted v6 kernel installs: apply_all().
# Arms differ ONLY in env pins the driver sets per wave (each patch reads its
# switch at call time). Hard gate (rig pack law: an unpatched arm comparison is
# meaningless): any FAIL/REVIEW aborts, and SKIP is allowed ONLY for the two
# lines this bundle is EXPECTED to skip —
#   patch9: the scored bundle's sandbox bootstrap lacks state_hash/diff_frames
#           (patch9 self-detects the round-1 measured defect and declines);
#   patch6: arcagi3-agent is deliberately not attached (its prober would own
#           the first 200 actions and confound all arms vs round 1).
# Both are asserted POSITIVELY: if either unexpectedly applied, the bundle
# under test is not the one this experiment was designed for — abort.
_ab_patch_results = apply_all()
_AB_EXPECTED_SKIPS = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
_ab_bad = [line for line in _ab_patch_results
           if "FAIL" in line or "REVIEW" in line
           or ("SKIP" in line and not line.startswith(_AB_EXPECTED_SKIPS))]
if _ab_bad:
    raise RuntimeError(f"[ab] patch layer did not fully apply: {_ab_bad}")
for _prefix in _AB_EXPECTED_SKIPS:
    _line = next((l for l in _ab_patch_results if l.startswith(_prefix)), "")
    if "SKIP" not in _line:
        raise RuntimeError(
            f"[ab] expected {_prefix} SKIP on this bundle, got: {_line!r}")
print("[ab] patch layer = v6 apply_all(); expected SKIPs verified "
      "(patch9 sandbox-defect decline, patch6 prober not mounted)", flush=True)
# Belt and suspenders: prove the sandbox is ALIVE after patching. patch9 (had
# it been applied) kills every sandbox subprocess on this bundle; a dead
# sandbox turns the whole A/B into two zero-action arms.
from inference.agent import python_tool_sandbox as _ab_ptx
_ab_sbx = _ab_ptx.run_sandboxed_python(
    code="print('sandbox-alive')", timeout_seconds=20,
    initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
    action_handler=lambda actions: {"result": [], "state": {}})
if "sandbox-alive" not in str(_ab_sbx.get("stdout", "")):
    raise RuntimeError(f"[ab] python sandbox is DEAD after patching: {_ab_sbx}")
print("[ab] sandbox liveness: OK", flush=True)
'''

PROBE_BLOCK = '''
if not install():
    raise RuntimeError("[ab] behavioural probe failed to install")
behav_report = report


def behav_raw():
    """Raw cumulative probe counters (diffable per wave offline)."""
    return {stem: {k: v for k, v in s.items() if not k.startswith("_")}
            for stem, s in _G.items()}


print("[ab] behavioural probe ACTIVE (identical in both arms)", flush=True)
'''


def hook_cell() -> str:
    return (
        "# ============================================================================\n"
        "# A/B patch layer. Inlined from submission/_duck_patched/duck_patches.py and\n"
        "# submission/_rig/behav_probe.py by build_ab_wmr.py — edit those and rebuild.\n"
        "# ============================================================================\n"
        f"{PATCHES.read_text()}\n"
        f"{APPLY_BLOCK}\n"
        "# --- behavioural probe: identical in every arm, observes only ---\n"
        f"{PROBE.read_text()}\n"
        f"{PROBE_BLOCK}"
    )


def run_cell() -> str:
    return (
        "# rig-style A/B run. Replaces duck-base's submission cell (its gateway poll\n"
        "# cannot succeed in a commit run). NOT a submission; no submission.parquet.\n"
        "print((BUNDLE_DIR / \"preamble.txt\").read_text())\n"
        "os.environ.setdefault(\"RECORDINGS_DIR\", str(WORKING_DIR / \"server_recording\"))\n"
        "\n"
        f"{DRIVER.read_text()}\n"
        "\n"
        "_ab_result = await ab_main(bm=bm, target=target, working_dir=WORKING_DIR,\n"
        "                           notebook_start=NOTEBOOK_START_EPOCH,\n"
        "                           behav_report=behav_report, behav_raw=behav_raw)\n"
    )


def main() -> None:
    # The hook/run cells must be valid python on their own (they contain two
    # inlined modules) — catch syntax breakage at build time, not on the GPU.
    ast.parse(hook_cell())
    run_src = run_cell()
    ast.parse(run_src.replace("await ab_main", "_ = ab_main"))  # top-level await is notebook-only

    nb = json.loads(BASE.read_text())
    seen = {"serve": False, "hook": False, "run": False}
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            if SERVE_GUARD in src:
                src = src.replace(
                    SERVE_GUARD,
                    "if True:  # ab-wmr: force the serve — a commit run must serve Qwen")
                seen["serve"] = True
            elif HOOK_MARKER in src:
                src = hook_cell()
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

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{SLUG}",
        "title": SLUG,
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
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
    print(f"wrote {OUT} ({len(nb['cells'])} cells, anchors {sorted(seen)})")
    print(f"push with: kaggle kernels push -p {OUT_DIR}")


if __name__ == "__main__":
    main()
