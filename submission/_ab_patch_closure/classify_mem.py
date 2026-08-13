#!/usr/bin/env python3
"""Thin JSON CLI over mem_screen_config.classify_mem.

Usage:
    .venv/bin/python submission/_ab_patch_closure/classify_mem.py \
        scratchpad/mem_screen/mem/patch_closure_result.json \
        scratchpad/mem_screen/shipped/patch_closure_result.json

Argument order is mem FIRST, shipped SECOND. Exit code 0 for a valid read
(MEM_AHEAD / SHIPPED_AHEAD / INDISTINGUISHABLE), 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from mem_screen_config import classify_mem  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mem", type=Path,
                        help="mem-arm patch_closure_result.json")
    parser.add_argument("shipped", type=Path,
                        help="shipped-arm patch_closure_result.json (fresh wave)")
    args = parser.parse_args(argv)
    result = classify_mem(
        json.loads(args.mem.read_text()), json.loads(args.shipped.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {
        "MEM_AHEAD", "SHIPPED_AHEAD", "INDISTINGUISHABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
