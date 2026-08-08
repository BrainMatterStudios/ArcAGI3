#!/usr/bin/env python3
"""Build the STRUCTURAL SCREEN kernel (final kernel of the 2026-08-09 sprint).

One candidate-only commit kernel on the patch-closure machinery: duck-base +
the committed duck_patches.py (patches 1-22 via the source-pin guard), arm
env = closure BASE_ENV (v7 pins) + {TAAF_DIFF_LINES, TAAF_WIGGLE,
TAAF_DISPATCH, TAAF_STRUCT} = "1" (TAAF_RUN_PROBE unset — superseded by the
plan channel; TAAF_VERIFY unset — one variable at a time). Closure geometry,
contract embedding, docker pin, dual-mount install cell. NO parallel-load
probe cell (already measured on the package screen: +41%/+36%).

The pre-registered reading contract (struct_screen_config.STRUCT_READING) is
baked into the run cell and lands in patch_closure_result.json verbatim. Read
with classify_struct.py against the banked base pair (11/12 excl-ft09) and
the package screen (10): ADVANCE iff >= 18 excl-ft09 OR >= 2 target first
unlocks; the headline diagnostic is LIVE ACTIONS-PER-TURN (adoption).

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_struct_screen.py
    kaggle kernels push -p submission/_ab_patch_closure/struct --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_patch_closure import build_kernel, notebook_contract  # noqa: E402
from struct_screen_config import (  # noqa: E402
    STRUCT_ENV,
    STRUCT_HYPOTHESIS,
    STRUCT_READING,
)

STRUCT_SLUG = "arc-agi-3-struct-screen"


def build_struct(output_root: Path | None = None) -> Path:
    """Build the struct-screen kernel; returns the notebook path."""
    return build_kernel(
        arm="struct",
        slug=STRUCT_SLUG,
        arm_env=STRUCT_ENV,
        hypothesis=STRUCT_HYPOTHESIS,
        reading=STRUCT_READING,
        code_stem="struct-screen",
        output_root=output_root,
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    nb_path = build_struct()
    contract = notebook_contract(nb_path)
    print(f"wrote {nb_path}")
    print(f"  base_sha256={contract['source_base_sha256'][:12]}... "
          f"patch_sha256={contract['patch_sha256'][:12]}... arm={contract['arm']}")
    print(f"  push with: kaggle kernels push -p {nb_path.parent} "
          f"--accelerator NvidiaRtxPro6000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
