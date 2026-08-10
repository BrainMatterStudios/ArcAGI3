#!/usr/bin/env python3
"""Thin JSON CLI over shipped_screen_config.classify_shipped.

Usage:
    .venv/bin/python submission/_ab_patch_closure/classify_shipped.py \
        scratchpad/shipped_screen/shipped/patch_closure_result.json \
        scratchpad/shipped_screen/base/patch_closure_result.json

Argument order is shipped FIRST, base SECOND. Exit code 0 for a valid read
(SHIPPED_AHEAD / BASE_AHEAD / INDISTINGUISHABLE), 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from shipped_screen_config import classify_shipped  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shipped", type=Path,
                        help="shipped-arm patch_closure_result.json")
    parser.add_argument("base", type=Path,
                        help="base-arm patch_closure_result.json (fresh wave)")
    args = parser.parse_args(argv)
    result = classify_shipped(
        json.loads(args.shipped.read_text()), json.loads(args.base.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {
        "SHIPPED_AHEAD", "BASE_AHEAD", "INDISTINGUISHABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
