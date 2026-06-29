#!/bin/bash
# Submit the PortfolioPolicy kernel to ARC-AGI-3.
#
# IMPORTANT (2026-06-29): the geodesic double-reset FIX (commit df9b96b) lives in notebook.ipynb's embedded
# modules. The Kaggle kernel v22 was built with the BUGGED no-op geodesic -> it MUST be re-pushed first, or
# this submits transfer-only (~0.33, no efficiency gain).
#
# WORKFLOW:
#   1) Re-push the fixed notebook (creates a NEW version, e.g. v23) and verify it runs:
#        cd /Users/ahmed/Documents/ArcAGI3/submission && kaggle kernels push -p .
#      then watch the run + confirm the [verify] cell prints "OK: PortfolioPolicy will run":
#        kaggle kernels status ahmedmobasher86/arc-agi-3-hybrid-explorer
#        kaggle kernels output ahmedmobasher86/arc-agi-3-hybrid-explorer -p /tmp/k && grep -i verify /tmp/k/*
#   2) Set VERSION below to the freshly-pushed version number, then run this script AT/AFTER the daily reset
#      (00:05 UTC / 02:05 CEST on 2026-06-30).
set -e
cd /Users/ahmed/Documents/ArcAGI3
VERSION="${1:?usage: submit_portfolio.sh <kernel_version>  (the version pushed AFTER the df9b96b fix)}"
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k ahmedmobasher86/arc-agi-3-hybrid-explorer -v "$VERSION" -f submission.parquet \
  -m "PortfolioPolicy: geodesic double-reset fix (7-109x per-play efficiency) + max-over-plays strict-superset portfolio"
echo "submitted v$VERSION; check: kaggle competitions submissions arc-prize-2026-arc-agi-3"
