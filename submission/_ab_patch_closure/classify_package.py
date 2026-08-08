#!/usr/bin/env python3
"""Thin JSON CLI over package_screen_config.classify_screen.

Usage:
    .venv/bin/python submission/_ab_patch_closure/classify_package.py \
        scratchpad/package_screen/patch_closure_result.json

Exit code 0 for a valid screen read (ADVANCE or STOP), 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from package_screen_config import classify_screen  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path,
                        help="package-screen patch_closure_result.json")
    args = parser.parse_args(argv)
    result = classify_screen(json.loads(args.package.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {"ADVANCE", "STOP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
