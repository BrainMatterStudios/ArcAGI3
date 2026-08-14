#!/usr/bin/env python3
"""Thin JSON CLI over quant_screen_config.classify_quant.

Usage:
    .venv/bin/python submission/_ab_patch_closure/classify_quant.py \
        scratchpad/quant_screen/quant/patch_closure_result.json \
        scratchpad/quant_screen/shipped/patch_closure_result.json

Argument order is quant FIRST, shipped SECOND. Exit code 0 for a valid read
(QUANT_AHEAD / SHIPPED_AHEAD / INDISTINGUISHABLE), 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from quant_screen_config import classify_quant  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quant", type=Path,
                        help="quant-arm patch_closure_result.json")
    parser.add_argument("shipped", type=Path,
                        help="shipped-arm patch_closure_result.json (same-session comparator)")
    args = parser.parse_args(argv)
    result = classify_quant(
        json.loads(args.quant.read_text()), json.loads(args.shipped.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {
        "QUANT_AHEAD", "SHIPPED_AHEAD", "INDISTINGUISHABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
