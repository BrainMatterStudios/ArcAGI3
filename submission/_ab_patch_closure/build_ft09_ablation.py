#!/usr/bin/env python3
"""Build the ft09-ablation kernels (see ft09_ablation_config.py for the pre-registration).

Same machinery as the closure/struct screens — identical duck_patches.py bytes,
arm env pinned immediately before apply_all(), docker pin, dual-mount install —
with one change: geometry names a FOCUS SUBSET of ("ft09",), so all 28 clones
land on that one game and a wave yields n=28 instead of n=1.

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_ft09_ablation.py --stage 1
    kaggle kernels push -p submission/_ab_patch_closure/ft09_base      --accelerator NvidiaRtxPro6000
    kaggle kernels push -p submission/_ab_patch_closure/ft09_struct    --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_patch_closure import build_kernel, notebook_contract  # noqa: E402
from ft09_ablation_config import (  # noqa: E402
    ALL_ARMS,
    FT09_ARMS,
    FT09_GEOMETRY,
    FT09_HYPOTHESIS,
    FT09_READING,
    FT09_SLUGS,
    FT09_STAGE2_ARMS,
)


def build_arm(name: str, output_root: Path | None = None) -> Path:
    if name not in ALL_ARMS:
        raise ValueError(f"unknown ft09 arm {name!r}; known: {sorted(ALL_ARMS)}")
    return build_kernel(
        arm=name,
        slug=FT09_SLUGS[name],
        arm_env=ALL_ARMS[name],
        hypothesis=FT09_HYPOTHESIS,
        reading=FT09_READING,
        code_stem=name.replace("_", "-"),
        output_root=output_root,
        geometry=FT09_GEOMETRY,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--arm", help="build a single named arm instead of a stage")
    args = ap.parse_args(argv)

    names = [args.arm] if args.arm else sorted(FT09_ARMS if args.stage == 1 else FT09_STAGE2_ARMS)
    for name in names:
        nb = build_arm(name)
        contract = notebook_contract(nb)
        print(f"wrote {nb}")
        print(f"  slug={FT09_SLUGS[name]}  base_sha256={contract['source_base_sha256'][:12]}… "
              f"patch_sha256={contract['patch_sha256'][:12]}…")
        print(f"  geometry={contract['geometry']}")
        delta = {k: v for k, v in ALL_ARMS[name].items()
                 if k not in ALL_ARMS["ft09_base"] or ALL_ARMS["ft09_base"][k] != v}
        print(f"  delta vs BASE_ENV: {delta or '(none — this IS the base arm)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
