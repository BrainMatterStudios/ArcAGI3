#!/bin/bash
# Submit the PortfolioPolicy kernel (v20, efficiency portfolio) to ARC-AGI-3. Run this AT/AFTER the daily reset
# (00:05 UTC / 02:05 CEST on 2026-06-30) — today's slot is already used.
# v20 is COMPLETE + verified ([verify] agent _Policy = PortfolioPolicy).
set -e
cd /Users/ahmed/Documents/ArcAGI3
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k ahmedmobasher86/arc-agi-3-hybrid-explorer -v 20 -f submission.parquet \
  -m "PortfolioPolicy: max-over-plays multi-strategy portfolio (strict-superset of banked transfer, full_reset safety floor)"
echo "submitted v20; check: kaggle competitions submissions arc-prize-2026-arc-agi-3"
