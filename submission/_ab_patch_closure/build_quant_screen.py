#!/usr/bin/env python3
"""Build the QUANT-SWAP SCREEN kernel (submission/_quant_probe/UPLOAD_PLAN.md
Step 3).

The quant arm IS the shipped arm (duck-base v2: NO duck_patches, NO apply_all,
same absence-proof machinery) plus ONE treatment, applied at build time: the
served FP8 snapshot is swapped vrfai -> RedHatAI. Three seams, every one
hard-asserted so anchor drift aborts the BUILD (and, for the serve command
itself, the RUN — before any setup command executes):

  1. kernel-metadata `dataset_sources`: driessmit1/vrfai-...-snapshot is
     replaced by ahmedmobasher86/redhatai-...-snapshot (the mount).
  2. duck-base cell 6's baked DATASET_SOURCES list literal: same swap, so the
     TAAF_KAGGLE_INPUT_PATHS mapping resolves the new mount (belt and braces;
     resolve_kaggle_dataset_path falls back to /kaggle/input/<slug> anyway).
  3. duck-base cell 8's setup loop: the loop is rewritten so every command
     from the read-only setup_commands.json is rewritten IN-MEMORY with the
     three serve anchors (MODEL_OWNER / MODEL_SLUG / SERVED_MODEL_NAME,
     quant_screen_config.SERVE_ANCHOR_REWRITES) BEFORE subprocess.run, with a
     hard assert that each anchor fired exactly once. SERVED_MODEL_NAME flows
     to LOCAL_ANALYZER_MODEL_ID, so pc_driver's serving probe then proves the
     swap against /models before any game minute.

The hook cell and run-cell prelude are DERIVED from the shipped screen's via
exact-match substitutions (each asserted to fire exactly once), so a change to
the shipped machinery cannot silently diverge from this arm.

Read with classify_quant.py against a shipped wave from the same session.
Pre-registration: quant_screen_config.QUANT_READING (baked verbatim into the
run cell and the result artifact); primary = true_score_all_games delta at
the 0.2712 banked-spread threshold. sonpham's external quant A/B is the
hypothesis source ONLY — not evidence.

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_quant_screen.py
    kaggle kernels push -p submission/_ab_patch_closure/quant --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_patch_closure import build_kernel, notebook_contract  # noqa: E402
from build_shipped_screen import (  # noqa: E402
    SHIPPED_HOOK,
    SHIPPED_PRELUDE,
    shipped_hook_cell,
)
from quant_screen_config import (  # noqa: E402
    QUANT_DATASET,
    QUANT_ENV,
    QUANT_HYPOTHESIS,
    QUANT_READING,
    SERVE_ANCHOR_REWRITES,
    VRFAI_DATASET,
)
from shipped_screen_config import UNPATCHED_SENTINEL  # noqa: E402

QUANT_SLUG = "arc-agi-3-patch-closure-quant"


def _derive(text: str, subs: tuple[tuple[str, str], ...], what: str) -> str:
    """Exact-match substitution where EVERY pattern must fire exactly once —
    upstream drift aborts the build instead of silently shipping a stale or
    half-renamed arm."""
    for old, new in subs:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"quant build: {what}: expected exactly 1 occurrence of "
                f"{old!r}, found {n} — upstream drift, refusing to build")
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Hook cell: the shipped hook with the arm identity renamed to quant. The
# anti-apply guard, sandbox liveness and probe bytes are inherited verbatim
# from the shipped screen (single source: build_shipped_screen.SHIPPED_HOOK).
# ---------------------------------------------------------------------------
_HOOK_SUBS = (
    (
        "# Patch-closure machinery, arm: SHIPPED. Reproduces the LIVE pinned submission\n"
        "# config (duck-base v2): NO duck_patches inlined, NO apply_all, NO arm env\n"
        "# pins beyond duck-base's own. Only the run-cell driver and the behavioural\n"
        "# probe (observe-only, byte-identical in every arm) are added.\n"
        "# Built by build_shipped_screen.py — edit there and rebuild.",
        "# Patch-closure machinery, arm: QUANT. The SHIPPED config (duck-base v2: NO\n"
        "# duck_patches inlined, NO apply_all) with ONE build-time treatment: the\n"
        "# served FP8 snapshot is swapped vrfai -> RedHatAI (dataset mount + cell-6\n"
        "# DATASET_SOURCES + in-memory serve-anchor rewrite in the setup loop).\n"
        "# Built by build_quant_screen.py — edit there and rebuild.",
    ),
    (
        '        "[pc] SHIPPED arm is contaminated with patch symbols: "',
        '        "[pc] QUANT arm is contaminated with patch symbols: "',
    ),
    (
        'f"[pc] SHIPPED arm: live classes carry patch markers: {_pc_guard_markers}")',
        'f"[pc] QUANT arm: live classes carry patch markers: {_pc_guard_markers}")',
    ),
    (
        'print("[pc] SHIPPED guard: patch layer verifiably ABSENT "',
        'print("[pc] QUANT guard: patch layer verifiably ABSENT "',
    ),
    (
        '# --- ARM IDENTITY: no env pins — the live duck-base v2 config ---------------\n'
        'PC_ARM = "shipped"\n'
        "PC_ARM_ENV = {}\n"
        'print("[pc] arm=shipped: NO patch layer, NO env pins — the live duck-base v2 "\n'
        '      "config under the standard rig instrumentation", flush=True)',
        '# --- ARM IDENTITY: inert fingerprint pin only; behaviour is duck-base v2 ----\n'
        '# (PC_QUANT_SNAPSHOT is read by NOTHING — it exists so this arm\'s env\n'
        '# fingerprint is distinct from shipped\'s empty mapping)\n'
        'PC_ARM = "quant"\n'
        f"PC_ARM_ENV = {json.dumps(QUANT_ENV)}\n"
        "import os as _pc_arm_os\n"
        "_pc_arm_os.environ.update(PC_ARM_ENV)\n"
        'print("[pc] arm=quant: NO patch layer; served snapshot swapped to RedHatAI "\n'
        '      "at the serve seam; PC_QUANT_SNAPSHOT is an inert identity pin", flush=True)',
    ),
)

QUANT_HOOK = _derive(SHIPPED_HOOK, _HOOK_SUBS, "hook")

# ---------------------------------------------------------------------------
# Run-cell prelude: the shipped absence-proof shims with the proof key renamed
# so the artifact carries this arm's own identity (classify_quant reads
# identity.patch_proof.quant_no_patch_markers).
# ---------------------------------------------------------------------------
_PRELUDE_SUBS = (
    ("# --- SHIPPED-arm driver shims (absence-proof mirrors) -----------------------",
     "# --- QUANT-arm driver shims (absence-proof mirrors) -------------------------"),
    ('f"[pc] SHIPPED arm: patch markers on live classes: {leaked}")',
     'f"[pc] QUANT arm: patch markers on live classes: {leaked}")'),
    ('return {"shipped_no_patch_markers": True,',
     'return {"quant_no_patch_markers": True,'),
)

QUANT_PRELUDE = _derive(SHIPPED_PRELUDE, _PRELUDE_SUBS, "prelude")


def quant_hook_cell() -> str:
    """The full quant hook: derived guard + arm identity, then the SAME probe
    section every arm carries (inherited from shipped_hook_cell so the probe
    bytes cannot diverge)."""
    shipped_cell = shipped_hook_cell()
    if not shipped_cell.startswith(SHIPPED_HOOK):
        raise SystemExit("quant build: shipped_hook_cell no longer starts with "
                         "SHIPPED_HOOK — upstream drift, refusing to build")
    return QUANT_HOOK + shipped_cell[len(SHIPPED_HOOK):]


# ---------------------------------------------------------------------------
# The three-seam snapshot swap, applied to the built notebook + kernel
# metadata. Every seam is exact-match and hard-asserted.
# ---------------------------------------------------------------------------

# Seam 2 — duck-base cell 6's baked DATASET_SOURCES literal, replaced as ONE
# exact line so any upstream reordering/renaming aborts instead of half-firing.
_DS_LINE_OLD = ('DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", '
                '"driessmit1/arc3-vllm-h100-wheelhouse-v3", '
                f'"{VRFAI_DATASET}"]')
_DS_LINE_NEW = ('DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", '
                '"driessmit1/arc3-vllm-h100-wheelhouse-v3", '
                f'"{QUANT_DATASET}"]')

# Seam 3 — duck-base cell 8's setup loop (AFTER build_kernel already forced
# the serve guard to `if True:`). Exact bytes of the loop being replaced:
_SETUP_LOOP_OLD = (
    '    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
    '        print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
    "        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n"
    "        env = _command_env()\n"
    "        os.environ.update(env)\n"
)


def _quant_setup_loop() -> str:
    """The rewritten setup loop: anchor-rewrite every command IN-MEMORY, abort
    on drift BEFORE any command runs, then run the rewritten commands through
    the original (byte-identical) execution lines."""
    anchors = "".join(
        f"        ({old!r},\n         {new!r}),\n" for old, new in SERVE_ANCHOR_REWRITES)
    return (
        "    # QUANT arm (build_quant_screen.py): swap the served FP8 snapshot\n"
        "    # IN-MEMORY before any setup command runs. setup_commands.json is\n"
        "    # mounted read-only and hardcodes the vrfai snapshot; each anchor must\n"
        "    # fire exactly once or we abort HERE — this arm must never silently\n"
        "    # serve vrfai (the whole treatment would vanish unmeasured).\n"
        "    _pc_quant_anchors = [\n"
        f"{anchors}"
        "    ]\n"
        '    _pc_quant_commands = json.loads((BUNDLE_DIR / "setup_commands.json").read_text())\n'
        "    _pc_quant_fired = {old: 0 for old, _new in _pc_quant_anchors}\n"
        "    _pc_quant_rewritten = []\n"
        "    for command in _pc_quant_commands:\n"
        "        for _old, _new in _pc_quant_anchors:\n"
        "            _pc_quant_fired[_old] += command.count(_old)\n"
        "            command = command.replace(_old, _new)\n"
        "        _pc_quant_rewritten.append(command)\n"
        "    if any(n != 1 for n in _pc_quant_fired.values()):\n"
        "        raise RuntimeError(\n"
        '            "[pc] QUANT serve-anchor rewrite did NOT fire exactly once per "\n'
        '            f"anchor: {_pc_quant_fired} — anchor drift in setup_commands.json; "\n'
        '            "refusing to run any setup command (a silent vrfai serve "\n'
        '            "measures nothing)")\n'
        '    print("[pc] QUANT serve anchors rewritten vrfai -> RedHatAI "\n'
        '          f"(fired: {_pc_quant_fired})", flush=True)\n'
        "    for command in _pc_quant_rewritten:\n"
        '        print(f"taaf.kaggle: setup command: {command}", flush=True)\n'
        "        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)\n"
        "        env = _command_env()\n"
        "        os.environ.update(env)\n"
    )


def _swap_once(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(
            f"quant build: {what}: expected exactly 1 occurrence, found {n} "
            "— anchor drift, refusing to build")
    return src.replace(old, new)


def _apply_quant_swap(nb_path: Path) -> None:
    """Post-build pass over the freshly built notebook + kernel metadata:
    seams 2 and 3 in the notebook, seam 1 in kernel-metadata.json. Each seam
    must fire exactly once across the whole notebook."""
    nb = json.loads(nb_path.read_text())
    fired = {"dataset_sources_literal": 0, "setup_loop": 0}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        changed = False
        if _DS_LINE_OLD in src:
            src = _swap_once(src, _DS_LINE_OLD, _DS_LINE_NEW,
                            "cell-6 DATASET_SOURCES literal")
            fired["dataset_sources_literal"] += 1
            changed = True
        if _SETUP_LOOP_OLD in src:
            src = _swap_once(src, _SETUP_LOOP_OLD, _quant_setup_loop(),
                            "cell-8 setup loop")
            fired["setup_loop"] += 1
            changed = True
        if changed:
            ast.parse(src)  # neither swapped cell carries top-level await
            cell["source"] = src.splitlines(keepends=True)
    bad = {k: v for k, v in fired.items() if v != 1}
    if bad:
        raise SystemExit(
            f"quant build: swap seams did not fire exactly once each: {fired} "
            "— duck-base drift, refusing to build")

    # Stamp the swap into the immutable build contract (audit trail).
    nb["metadata"]["patch_closure_contract"]["quant_swap"] = {
        "dataset": {"from": VRFAI_DATASET, "to": QUANT_DATASET},
        "serve_anchor_rewrites": [list(pair) for pair in SERVE_ANCHOR_REWRITES],
    }
    nb_path.write_text(json.dumps(nb, indent=1))

    # Seam 1 — the kernel dataset mount.
    meta_path = nb_path.parent / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    sources = meta["dataset_sources"]
    if sources.count(VRFAI_DATASET) != 1:
        raise SystemExit(
            f"quant build: kernel-metadata dataset_sources {sources!r} does not "
            f"contain {VRFAI_DATASET!r} exactly once — refusing to build")
    meta["dataset_sources"] = [
        QUANT_DATASET if s == VRFAI_DATASET else s for s in sources]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def build_quant(output_root: Path | None = None) -> Path:
    """Build the quant-screen kernel; returns the notebook path."""
    nb_path = build_kernel(
        arm="quant",
        slug=QUANT_SLUG,
        arm_env=QUANT_ENV,
        hypothesis=QUANT_HYPOTHESIS,
        reading=QUANT_READING,
        code_stem="patch-closure-quant",
        output_root=output_root,
        hook_override=quant_hook_cell(),
        run_prelude=QUANT_PRELUDE,
        patch_sha256_override=UNPATCHED_SENTINEL,
    )
    _apply_quant_swap(nb_path)
    return nb_path


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    nb_path = build_quant()
    contract = notebook_contract(nb_path)
    env_fp = hashlib.sha256(
        json.dumps(contract["arm_env"], sort_keys=True).encode()).hexdigest()[:12]
    print(f"wrote {nb_path}")
    print(f"  base_sha256={contract['source_base_sha256'][:12]}... "
          f"patch_sha256={contract['patch_sha256']!r} arm={contract['arm']} "
          f"arm_env_fp={env_fp}")
    print(f"  dataset swap: {contract['quant_swap']['dataset']}")
    print(f"  push with: kaggle kernels push -p {nb_path.parent} "
          f"--accelerator NvidiaRtxPro6000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
