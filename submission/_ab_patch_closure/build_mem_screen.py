#!/usr/bin/env python3
"""Build the MEM-STACK SCREEN kernel (duck-mem P1-P4 on the shipped config).

One commit kernel on the patch-closure machinery that runs the duck-mem
anti-waste stack (submission/_duck_mem/harness_mem.py) on top of the SHIPPED
(unpatched) duck-base v2 config — exactly as the duck-mem notebook hook does:
inline the harness_mem.py bytes, call apply_all(BUNDLE_DIR), P5/P6 gated OFF.
NO duck_patches anywhere. The hook cell carries:
  * the shipped screen's ANTI-APPLY guard (duck_patches symbols hard-asserted
    ABSENT — globals, sys.modules, live-class "_patched" markers);
  * the inlined harness_mem stack + apply_all with CAPTURED stdout — every
    [duck-mem] P1-P4 marker line is hard-asserted (harness_mem degrades to
    warnings by design; a silently-degraded stack must abort, not measure);
  * state-level P1-P4 asserts on the live objects (belt and braces);
  * two observe-only counters (P4 guard fires = stop_reason=repeated_no_effect
    batch stops; P2 trims) surfaced at patch_diagnostics.mem via a driver shim;
  * the same sandbox-liveness check and behavioural probe as every other arm.

Compared against the EXISTING shipped-screen kernel
(arc-agi-3-patch-closure-shipped, unchanged) with classify_mem
(mem_screen_config.py): primary true_score delta at threshold 0.2712;
SECONDARY AND EMPHASIZED behavioural readouts (turns, trims, guard fires,
gen_tokens) — the sensitive instrument at one wave per arm.

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_mem_screen.py
    kaggle kernels push -p submission/_ab_patch_closure/mem --accelerator NvidiaRtxPro6000
    kaggle kernels push -p submission/_ab_patch_closure/shipped --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_patch_closure import (  # noqa: E402
    PROBE,
    PROBE_BLOCK,
    build_kernel,
    notebook_contract,
)
from mem_screen_config import (  # noqa: E402
    MEM_ENV,
    MEM_HYPOTHESIS,
    MEM_MARKER_LINES,
    MEM_READING,
)

MEM_SLUG = "arc-agi-3-patch-closure-mem"
HARNESS_MEM = REPO / "submission/_duck_mem/harness_mem.py"


def _mem_patch_bytes() -> bytes:
    """harness_mem.py bytes for inlining — SOURCE-PIN AWARE (the same law as
    build_patch_closure._patches_bytes: a kernel must inline bytes traceable
    to a commit).
      * MEM_PATCHES_REF=<git ref>  -> inline that ref's blob (e.g. HEAD);
      * unset + working file matches HEAD -> working file (status quo);
      * unset + working file DIRTY -> hard abort naming both shas.
    """
    import os
    import subprocess

    rel = "submission/_duck_mem/harness_mem.py"
    ref = os.environ.get("MEM_PATCHES_REF", "").strip()
    if ref:
        return subprocess.check_output(["git", "-C", str(REPO), "show", f"{ref}:{rel}"])
    working = HARNESS_MEM.read_bytes()
    try:
        head = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"])
    except Exception:  # noqa: BLE001 - no git (e.g. exported tree): trust the file
        return working
    if working != head:
        raise SystemExit(
            "harness_mem.py in the working tree differs from HEAD "
            f"(working sha256 {hashlib.sha256(working).hexdigest()[:12]}, "
            f"HEAD {hashlib.sha256(head).hexdigest()[:12]}). Refusing to inline "
            "uncommitted stack bytes into a kernel: commit them, or pin the "
            "source explicitly with MEM_PATCHES_REF=HEAD (or another ref).")
    return working


# ---------------------------------------------------------------------------
# Hook section 1 — the shipped screen's anti-apply guard, duck_patches only.
# "apply_all" is deliberately NOT in the forbidden-globals list: harness_mem
# (inlined ABOVE this guard in the same cell) legitimately defines its own
# apply_all. The duck_patches-specific definition names below cannot collide
# with anything harness_mem defines (test_mem_screen.py pins this).
# ---------------------------------------------------------------------------
MEM_GUARD = '''\
# --- ANTI-APPLY GUARD: duck_patches must be verifiably ABSENT ---------------
# (mirror of the patched arms' APPLY_BLOCK, minus "apply_all" — the duck-mem
# stack inlined above owns that name. A build regression that leaks the
# duck_patches layer into this arm must die HERE, before any GPU minute.)
import sys as _pc_guard_sys
_pc_guard_forbidden = [
    _sym for _sym in (
        "patch_action7", "patch_watchdog", "patch_hud_board_identity",
        "patch_win_replay", "patch_frontier_graph", "HudMaskTracker",
        "ANTIFREEZE_DIAGNOSTICS")
    if _sym in globals()]
if _pc_guard_forbidden or "duck_patches" in _pc_guard_sys.modules:
    raise RuntimeError(
        "[pc] MEM arm is contaminated with duck_patches symbols: "
        f"globals={_pc_guard_forbidden} "
        f"duck_patches_module={'duck_patches' in _pc_guard_sys.modules}")
from inference.framework import solver as _pc_guard_sv
_pc_guard_cls = _pc_guard_sv._HarnessGameSession
_pc_guard_markers = {
    _name: [_attr for _attr in vars(_fn) if _attr.endswith("_patched")]
    for _name, _fn in (
        ("should_stop", _pc_guard_cls.should_stop),
        ("step_env", _pc_guard_cls.step_env),
        ("_execute_action", _pc_guard_cls._execute_action),
        ("play", _pc_guard_cls.play))}
_pc_guard_markers = {k: v for k, v in _pc_guard_markers.items() if v}
if _pc_guard_markers:
    raise RuntimeError(
        f"[pc] MEM arm: live classes carry duck_patches markers: {_pc_guard_markers}")
print("[pc] MEM guard: duck_patches layer verifiably ABSENT "
      "(globals, sys.modules, live-class markers)", flush=True)
'''

# ---------------------------------------------------------------------------
# Hook section 2 — apply the stack the way the duck-mem notebook hook does
# (apply_all(BUNDLE_DIR)), but with stdout CAPTURED and every expected
# [duck-mem] marker line hard-asserted, then state-level P1-P4 asserts.
# harness_mem is fail-safe by design (degrades to printed warnings); an arm
# measuring a silently-degraded stack would be a phantom arm.
# ---------------------------------------------------------------------------
_MARKER_LINES_LITERAL = "(\n" + "".join(
    f"    {line!r},\n" for line in MEM_MARKER_LINES) + ")"

MEM_APPLY_BLOCK = f'''\
# --- APPLY the duck-mem stack: captured, asserted, then re-printed ----------
import contextlib as _mem_ctl
import io as _mem_io
from inference.agent import tool_agent as _mem_ta_pre
_mem_yield_before = float(_mem_ta_pre._LOCAL_ANALYZER_YIELD_SECONDS)
_mem_cap = _mem_io.StringIO()
with _mem_ctl.redirect_stdout(_mem_cap):
    apply_all(BUNDLE_DIR)
_mem_apply_out = _mem_cap.getvalue()
print(_mem_apply_out, end="", flush=True)
_MEM_EXPECTED_LINES = {_MARKER_LINES_LITERAL}
_mem_missing = [_l for _l in _MEM_EXPECTED_LINES if _l not in _mem_apply_out]
if _mem_missing:
    raise RuntimeError(
        f"[pc] duck-mem stack did NOT fully apply — missing marker lines: "
        f"{{_mem_missing}}; full output:\\n{{_mem_apply_out}}")
print("[pc] duck-mem P1-P4 marker lines all present; P5/P6 gated off", flush=True)

# --- state-level P1-P4 asserts on the live objects (belt and braces) --------
from inference.agent import prompts as _mem_pr
from inference.agent import tool_agent as _mem_ta
import inference.framework.solver as _mem_sv
_mem_state = {{
    "p1_estimator": bool(getattr(_mem_ta._estimate_tokens, "_mem", False)),
    "p2_middle_drop": bool(getattr(_mem_ta.ToolAgent, "_mem_middle_drop", False)),
    "p3_prompt_neutralized": (_OLD_OPTIMIZE_LINE not in _mem_pr.GAME_OVERVIEW_ADDENDUM
                              and _OLD_OPTIMIZE_LINE not in _mem_ta.GAME_OVERVIEW_ADDENDUM
                              and _NEW_OPTIMIZE_LINE in _mem_ta.GAME_OVERVIEW_ADDENDUM),
    "p4_no_effect_guard": bool(
        getattr(_mem_sv._HarnessGameSession, "_mem_no_effect_guard", False)
        and getattr(_mem_sv._HarnessGameSession.step_env, "_mem", False)),
    "p5_yield_untouched": float(_mem_ta._LOCAL_ANALYZER_YIELD_SECONDS) == _mem_yield_before,
}}
_mem_bad_state = [k for k, v in _mem_state.items() if not v]
if _mem_bad_state:
    raise RuntimeError(f"[pc] duck-mem state asserts FAILED: {{_mem_bad_state}}")
print(f"[pc] duck-mem state asserts: {{_mem_state}}", flush=True)
'''

# ---------------------------------------------------------------------------
# Hook section 3 — observe-only counters. P4's stop_reason=repeated_no_effect
# is NOT persisted in the driver's rows, so count it at its source:
# _step_env resolves `_should_guard_no_effect` from its defining namespace at
# call time, so rebinding the name here (same namespace the stack was inlined
# into) makes every True return — exactly one per repeated_no_effect batch
# stop — increment MEM_DIAGNOSTICS. Likewise every successful P2 middle-drop
# eviction increments trims via a counting wrapper on the live class.
# ---------------------------------------------------------------------------
MEM_COUNTERS_BLOCK = '''\
# --- mem observability counters (observe-only; feed patch_diagnostics.mem) --
MEM_DIAGNOSTICS = {"guard_fires": 0, "trims": 0}

_mem_orig_guard_fn = _should_guard_no_effect


def _should_guard_no_effect(board_changed, action_display, next_display):  # noqa: F811
    fired = _mem_orig_guard_fn(board_changed, action_display, next_display)
    if fired:
        MEM_DIAGNOSTICS["guard_fires"] += 1
    return fired


def _mem_install_trim_counter():
    from inference.agent import tool_agent as _mem_tc_ta
    _mem_tc_cls = _mem_tc_ta.ToolAgent
    _mem_tc_inner = _mem_tc_cls._drop_oldest_history_block
    if getattr(_mem_tc_inner, "_mem_trim_counted", False):
        return True
    if not getattr(_mem_tc_inner, "_mem", False):
        raise RuntimeError(
            "[pc] trim counter: _drop_oldest_history_block is not the duck-mem "
            "middle-drop (P2 not applied?)")

    def _mem_counting_drop(self, history, *, preserve_recent):
        dropped = _mem_tc_inner(self, history, preserve_recent=preserve_recent)
        if dropped:
            MEM_DIAGNOSTICS["trims"] += 1
        return dropped

    _mem_counting_drop._mem = True
    _mem_counting_drop._mem_trim_counted = True
    _mem_tc_cls._drop_oldest_history_block = _mem_counting_drop
    return True


if not _mem_install_trim_counter():
    raise RuntimeError("[pc] mem trim counter failed to install")
print("[pc] mem counters installed (guard_fires, trims)", flush=True)
'''

MEM_SANDBOX_BLOCK = '''\
# --- ARM IDENTITY banner ----------------------------------------------------
print("[pc] arm=mem: duck-mem P1-P4 stack on the shipped duck-base v2 config "
      "(NO duck_patches; P5/P6 gated off) under the standard rig "
      "instrumentation", flush=True)

# --- sandbox liveness (patch-INDEPENDENT: run_sandboxed_python is base duck
# code; every other arm runs the same check) ---------------------------------
from inference.agent import python_tool_sandbox as _pc_ptx
_pc_sbx = _pc_ptx.run_sandboxed_python(
    code="print('sandbox-alive')", timeout_seconds=20,
    initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
    action_handler=lambda actions: {"result": [], "state": {}})
if "sandbox-alive" not in str(_pc_sbx.get("stdout", "")):
    raise RuntimeError(f"[pc] python sandbox is DEAD after patching: {_pc_sbx}")
print("[pc] sandbox liveness: OK", flush=True)
'''


def _arm_pin_block() -> str:
    env_literal = json.dumps(MEM_ENV, indent=4)
    return (
        "\n# --- ARM ENVIRONMENT: pinned IMMEDIATELY BEFORE apply_all() -----------------\n"
        "# (frozen in mem_screen_config.py; the classifier rejects any drift)\n"
        'PC_ARM = "mem"\n'
        f"PC_ARM_ENV = {env_literal}\n"
        "import os as _pc_env_os\n"
        "_pc_env_os.environ.update(PC_ARM_ENV)\n"
        'print(f"[pc] arm={PC_ARM}: env pinned before apply_all(): {PC_ARM_ENV}", flush=True)\n'
    )


def mem_hook_cell() -> str:
    """The full mem hook: inlined harness_mem stack FIRST (its module
    docstring + `from __future__` stay position-legal after the comment
    header), then the duck_patches absence guard, arm pins, the captured
    apply, the counters, sandbox liveness, and the SAME probe section every
    other arm carries (identical bytes and identical trailing PROBE_BLOCK)."""
    return (
        "# ============================================================================\n"
        "# Patch-closure machinery, arm: MEM. duck-mem P1-P4 anti-waste stack on the\n"
        "# SHIPPED duck-base v2 config (NO duck_patches). Stack inlined from\n"
        "# submission/_duck_mem/harness_mem.py by build_mem_screen.py — edit there\n"
        "# and rebuild. Applied via apply_all(BUNDLE_DIR), exactly as the duck-mem\n"
        "# notebook hook applies it (P5/P6 gated OFF).\n"
        "# ============================================================================\n"
        f"{_mem_patch_bytes().decode('utf-8')}\n"
        f"\n{MEM_GUARD}\n"
        f"{_arm_pin_block()}\n"
        f"{MEM_APPLY_BLOCK}\n"
        f"{MEM_COUNTERS_BLOCK}\n"
        f"{MEM_SANDBOX_BLOCK}\n"
        "# --- behavioural probe: identical in every arm, observes only ---\n"
        f"{PROBE.read_text()}\n"
        f"{PROBE_BLOCK}"
    )


# ---------------------------------------------------------------------------
# Run-cell prelude: rebinds the two driver helpers that hard-assert a
# duck_patches layer (same seam as the shipped screen) and wraps the
# diagnostics collector so MEM_DIAGNOSTICS lands at patch_diagnostics.mem.
# Executed after the inlined pc_driver source and before pc_main, in the same
# notebook globals, so pc_main resolves the rebound names at call time.
# ---------------------------------------------------------------------------
MEM_PRELUDE = '''\
# --- MEM-arm driver shims ---------------------------------------------------
def _pc_patch_proof():
    """duck_patches ABSENT + duck-mem P1-P4 PRESENT on the live objects.
    pc_main aborts on any False boolean in this dict, so every entry below is
    load-bearing (mem_screen_config.MEM_PROOF_KEYS re-asserts them from the
    artifact)."""
    from inference.agent import prompts as _pr
    from inference.agent import tool_agent as _ta
    from inference.framework import solver as _sv
    cls = _sv._HarnessGameSession
    leaked = {name: [a for a in vars(fn) if a.endswith("_patched")]
              for name, fn in (("should_stop", cls.should_stop),
                               ("step_env", cls.step_env),
                               ("_execute_action", cls._execute_action),
                               ("play", cls.play))}
    leaked = {k: v for k, v in leaked.items() if v}
    if leaked:
        raise RuntimeError(
            f"[pc] MEM arm: duck_patches markers on live classes: {leaked}")
    _old_line = ("- Optimize for as few in-game actions as possible while "
                 "still being reliable.\\n")
    return {
        "mem_no_duck_patches_markers": True,
        "mem_p1_estimator": bool(getattr(_ta._estimate_tokens, "_mem", False)),
        "mem_p2_middle_drop": bool(getattr(_ta.ToolAgent, "_mem_middle_drop", False)),
        "mem_p3_prompt_neutralized": (
            _old_line not in _pr.GAME_OVERVIEW_ADDENDUM
            and _old_line not in _ta.GAME_OVERVIEW_ADDENDUM),
        "mem_p4_no_effect_guard": bool(
            getattr(cls, "_mem_no_effect_guard", False)
            and getattr(cls.step_env, "_mem", False)),
        "execute_chain_repr": repr(cls._execute_action),
        "play_chain_repr": repr(cls.play),
    }


def _pc_install_animation_counter():
    """No duck_patches module exists in this arm; PC_ANIMATION_UPTAKE stays
    zero by design and the classifier asserts exactly that (purity check)."""
    return True


_pc_collect_unwrapped = collect_patch_diagnostics


def collect_patch_diagnostics(sessions, animation_delta, antifreeze_delta,  # noqa: F811
                              compact_delta, package_delta=None):
    """Standard collector + the mem stack's observe-only counters at
    patch_diagnostics.mem (guard_fires = stop_reason=repeated_no_effect batch
    stops, 1:1; trims = P2 middle-drop evictions)."""
    out = _pc_collect_unwrapped(sessions, animation_delta, antifreeze_delta,
                                compact_delta, package_delta=package_delta)
    out["mem"] = dict(MEM_DIAGNOSTICS)
    return out
'''


def build_mem(output_root: Path | None = None) -> Path:
    """Build the mem-screen kernel; returns the notebook path."""
    return build_kernel(
        arm="mem",
        slug=MEM_SLUG,
        arm_env=MEM_ENV,
        hypothesis=MEM_HYPOTHESIS,
        reading=MEM_READING,
        code_stem="patch-closure-mem",
        output_root=output_root,
        hook_override=mem_hook_cell(),
        run_prelude=MEM_PRELUDE,
        patch_sha256_override=hashlib.sha256(_mem_patch_bytes()).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    nb_path = build_mem()
    contract = notebook_contract(nb_path)
    env_fp = hashlib.sha256(
        json.dumps(contract["arm_env"], sort_keys=True).encode()).hexdigest()[:12]
    print(f"wrote {nb_path}")
    print(f"  base_sha256={contract['source_base_sha256'][:12]}... "
          f"patch_sha256={contract['patch_sha256'][:12]}... (harness_mem) "
          f"arm={contract['arm']} arm_env_fp={env_fp}")
    print(f"  push with: kaggle kernels push -p {nb_path.parent} "
          f"--accelerator NvidiaRtxPro6000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
