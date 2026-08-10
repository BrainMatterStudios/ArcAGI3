#!/usr/bin/env python3
"""Build the SHIPPED-CONFIG SCREEN kernel (HANDOFF-2026-08-10 §5.1).

One commit kernel on the patch-closure machinery that reproduces the LIVE
pinned submission config (duck-base v2): NO duck_patches inlined, NO
apply_all, NO arm env pins beyond what duck-base itself sets. The hook cell
carries the MIRROR IMAGE of APPLY_BLOCK — it hard-asserts patch symbols are
ABSENT (globals, sys.modules, and live-class markers), keeps the
patch-independent sandbox-liveness check, and installs the same behavioural
probe as every other arm. The run cell is the standard pc_driver at the frozen
eval geometry (full 25 games, 28 clones, 7920s, concurrency 28) with a small
prelude that rebinds the two driver checks which hard-assert patch PRESENCE
(_pc_patch_proof, _pc_install_animation_counter) to their absence-proof
mirrors; everything else in the driver runs unchanged.

Read with classify_shipped.py against a FRESH base wave from the same session
(the existing, unchanged closure base kernel); the two banked base waves at
identical geometry are additional controls. Pre-registration:
shipped_screen_config.SHIPPED_READING (baked verbatim into the run cell and
the result artifact).

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_shipped_screen.py
    kaggle kernels push -p submission/_ab_patch_closure/shipped --accelerator NvidiaRtxPro6000
    kaggle kernels push -p submission/_ab_patch_closure/base --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_patch_closure import (  # noqa: E402
    PROBE,
    PROBE_BLOCK,
    build_kernel,
    notebook_contract,
)
from shipped_screen_config import (  # noqa: E402
    SHIPPED_ENV,
    SHIPPED_HYPOTHESIS,
    SHIPPED_READING,
    UNPATCHED_SENTINEL,
)

SHIPPED_SLUG = "arc-agi-3-patch-closure-shipped"

# ---------------------------------------------------------------------------
# The shipped hook cell: the mirror image of the patched arms' hook. Where
# they inline duck_patches + APPLY_BLOCK (which hard-asserts patches APPLIED),
# this cell hard-asserts patches ABSENT — a build regression can never
# silently ship a patched "shipped" arm. NOTE: symbol names are referenced as
# plain strings only; the notebook must never contain "def apply_all(" or any
# other patch-definition bytes (test_shipped_screen.py pins this).
# ---------------------------------------------------------------------------
SHIPPED_HOOK = '''\
# ============================================================================
# Patch-closure machinery, arm: SHIPPED. Reproduces the LIVE pinned submission
# config (duck-base v2): NO duck_patches inlined, NO apply_all, NO arm env
# pins beyond duck-base's own. Only the run-cell driver and the behavioural
# probe (observe-only, byte-identical in every arm) are added.
# Built by build_shipped_screen.py — edit there and rebuild.
# ============================================================================
# --- ANTI-APPLY GUARD: the mirror image of the patched arms' APPLY_BLOCK ----
# A build regression that leaks the patch layer into this arm must die HERE,
# before any GPU minute is spent.
import sys as _pc_guard_sys
_pc_guard_forbidden = [
    _sym for _sym in (
        "apply_all", "patch_action7", "patch_watchdog", "patch_hud_board_identity",
        "patch_win_replay", "patch_frontier_graph", "HudMaskTracker",
        "ANTIFREEZE_DIAGNOSTICS")
    if _sym in globals()]
if _pc_guard_forbidden or "duck_patches" in _pc_guard_sys.modules:
    raise RuntimeError(
        "[pc] SHIPPED arm is contaminated with patch symbols: "
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
        f"[pc] SHIPPED arm: live classes carry patch markers: {_pc_guard_markers}")
print("[pc] SHIPPED guard: patch layer verifiably ABSENT "
      "(globals, sys.modules, live-class markers)", flush=True)

# --- ARM IDENTITY: no env pins — the live duck-base v2 config ---------------
PC_ARM = "shipped"
PC_ARM_ENV = {}
print("[pc] arm=shipped: NO patch layer, NO env pins — the live duck-base v2 "
      "config under the standard rig instrumentation", flush=True)

# --- sandbox liveness (patch-INDEPENDENT: run_sandboxed_python is base duck
# code; the patched arms run the same check after apply_all) -----------------
from inference.agent import python_tool_sandbox as _pc_ptx
_pc_sbx = _pc_ptx.run_sandboxed_python(
    code="print('sandbox-alive')", timeout_seconds=20,
    initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
    action_handler=lambda actions: {"result": [], "state": {}})
if "sandbox-alive" not in str(_pc_sbx.get("stdout", "")):
    raise RuntimeError(f"[pc] python sandbox is DEAD: {_pc_sbx}")
print("[pc] sandbox liveness: OK", flush=True)
'''

# ---------------------------------------------------------------------------
# Run-cell prelude: rebinds the ONLY two driver helpers that hard-assert a
# patch layer this arm by design does not carry. Executed after the inlined
# pc_driver source and before pc_main, in the same notebook globals, so
# pc_main resolves the rebound names. Everything else — env verification
# (trivially clean on an empty mapping), serving probe, session registry,
# LLM-turn counter, behavioural probe, row/score collection — runs unchanged.
# ---------------------------------------------------------------------------
SHIPPED_PRELUDE = '''\
# --- SHIPPED-arm driver shims (absence-proof mirrors) -----------------------
def _pc_patch_proof():
    """Mirror of the patched arms' proof: assert every patch marker ABSENT on
    the live classes and return an all-True absence proof (pc_main aborts on
    any False boolean, so a leaked patch flips this to a hard failure)."""
    from inference.framework import solver as _sv
    cls = _sv._HarnessGameSession
    surfaces = {"should_stop": cls.should_stop, "step_env": cls.step_env,
                "_execute_action": cls._execute_action, "play": cls.play}
    leaked = {name: [a for a in vars(fn) if a.endswith("_patched")]
              for name, fn in surfaces.items()}
    leaked = {k: v for k, v in leaked.items() if v}
    if leaked:
        raise RuntimeError(
            f"[pc] SHIPPED arm: patch markers on live classes: {leaked}")
    return {"shipped_no_patch_markers": True,
            "execute_chain_repr": repr(cls._execute_action),
            "play_chain_repr": repr(cls.play)}


def _pc_install_animation_counter():
    """No patch module exists in this arm; PC_ANIMATION_UPTAKE stays zero by
    design and the classifier asserts exactly that (purity check)."""
    return True
'''


def shipped_hook_cell() -> str:
    """The full shipped hook: anti-apply guard + arm identity + sandbox
    liveness, then the SAME probe section the patched arms' hook carries
    (identical bytes and identical trailing PROBE_BLOCK)."""
    return (
        SHIPPED_HOOK
        + "\n# --- behavioural probe: identical in every arm, observes only ---\n"
        + f"{PROBE.read_text()}\n"
        + PROBE_BLOCK
    )


def build_shipped(output_root: Path | None = None) -> Path:
    """Build the shipped-screen kernel; returns the notebook path."""
    return build_kernel(
        arm="shipped",
        slug=SHIPPED_SLUG,
        arm_env=SHIPPED_ENV,
        hypothesis=SHIPPED_HYPOTHESIS,
        reading=SHIPPED_READING,
        code_stem="patch-closure-shipped",
        output_root=output_root,
        hook_override=shipped_hook_cell(),
        run_prelude=SHIPPED_PRELUDE,
        patch_sha256_override=UNPATCHED_SENTINEL,
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    nb_path = build_shipped()
    contract = notebook_contract(nb_path)
    env_fp = hashlib.sha256(
        json.dumps(contract["arm_env"], sort_keys=True).encode()).hexdigest()[:12]
    print(f"wrote {nb_path}")
    print(f"  base_sha256={contract['source_base_sha256'][:12]}... "
          f"patch_sha256={contract['patch_sha256']!r} arm={contract['arm']} "
          f"arm_env_fp={env_fp}")
    print(f"  push with: kaggle kernels push -p {nb_path.parent} "
          f"--accelerator NvidiaRtxPro6000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
