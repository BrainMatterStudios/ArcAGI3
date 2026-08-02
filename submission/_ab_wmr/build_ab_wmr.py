#!/usr/bin/env python3
"""Build submission/_ab_wmr/ab-wmr.ipynb — behavioral A/B for the three shipped
duck patches (watchdog f591392, HUD mask b28569c, replay-at-WIN a1d0378).

NOT a competition submission. A plain GPU commit kernel on the _rig mechanism
(submission/_rig/build_rig.py): duck-base notebook, serve forced, run cell
replaced by the A/B wave driver (ab_wave_driver.py). One vLLM session; arms
toggle ONLY the env kill switches between counterbalanced waves A,B,B,A.

Both arms get the SAME curated patch subset applied (hard-gated):
    patch_action7, patch_animation_producer, patch_animation_metadata,
    verify_reset_already_handled, patch_watchdog, patch_hud_board_identity,
    patch_win_replay
The experimental grid-burner / prompt / transfer-explorer patches in
duck_patches.py are deliberately EXCLUDED: they never scored, patch6's
heuristic prober would dominate the first 200 actions, and the pinned base
distribution (the 0.929 +- 0.195 yardstick) was measured without them. Arm A
is therefore pinned-behavior-plus-ACTION7/animation with the three patches
installed-but-disabled; arm B flips only the three switches.

patch_hud_sandbox is EXCLUDED for a measured defect (see APPLY_BLOCK comment
and dry_run.py): on the scored bundle it NameErrors every sandbox process
(no state_hash in that bundle's bootstrap) and the duck executes 0 actions.

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
# --- curated patch application (BOTH arms, identical) ---------------------------
# The three patches under test are installed here and env-gated at CALL time;
# grid-burner/prompt/transfer-explorer are excluded (never scored, would
# confound the A/B — see build_ab_wmr.py docstring).
#
# patch_hud_sandbox is ALSO excluded — measured defect (dry_run.py, 2026-08-02):
# its helper block does `_hud_raw_state_hash = state_hash` at sandbox-module
# level, but the scored bundle's _SANDBOX_BOOTSTRAP (taaf-src-hybrid, Jul-03)
# defines NO state_hash/diff_frames anywhere, so every sandbox process dies
# with NameError at startup and the duck executes ZERO actions. Its unit tests
# pass only against the drifted _adopt tree. On this bundle there is nothing
# for patch9 to make HUD-aware, so excluding it loses nothing; patch8
# (board_changed masking) is the semantic core of the HUD arm and stays.
_AB_PATCH_FNS = [
    patch_action7,
    patch_animation_producer,
    patch_animation_metadata,
    verify_reset_already_handled,
    patch_watchdog,
    patch_hud_board_identity,
    patch_win_replay,
]
_ab_patch_results = {}
for _fn in _AB_PATCH_FNS:
    _line = _fn()
    _ab_patch_results[_fn.__name__] = _line
    print(f"[ab-patch] {_line}", flush=True)
_ab_bad = {k: v for k, v in _ab_patch_results.items()
           if "FAIL" in v or "REVIEW" in v}
if _ab_bad:
    # Hard gate: an unpatched arm comparison is meaningless (rig pack law).
    raise RuntimeError(f"[ab] patch layer did not fully apply: {_ab_bad}")
print("[ab] patch layer applied; excluded by design: patch_dynamic_grid_burner, "
      "patch_prompts, patch_tool_agent_analyze; excluded for the measured "
      "sandbox NameError defect: patch_hud_sandbox", flush=True)
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
