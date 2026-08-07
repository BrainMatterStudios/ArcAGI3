#!/usr/bin/env python3
"""Thin JSON CLI over patch_closure_config.classify_result.

Usage:
    .venv/bin/python submission/_ab_patch_closure/classify.py \
        scratchpad/patch_closure/base/patch_closure_result.json \
        scratchpad/patch_closure/candidate/patch_closure_result.json

Exit code 0 for a valid measurement (GO or NO_GO), 2 for INVALID/INFRA_FAILURE.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from patch_closure_config import classify_result  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="base arm patch_closure_result.json")
    parser.add_argument("candidate", type=Path, help="candidate arm patch_closure_result.json")
    args = parser.parse_args(argv)
    result = classify_result(
        json.loads(args.base.read_text()),
        json.loads(args.candidate.read_text()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {"GO", "NO_GO"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
