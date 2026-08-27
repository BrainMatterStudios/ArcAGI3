#!/bin/bash
cd /Users/ahmed/Documents/ArcAGI3
MSG="PURE Tufa duck reproduction (base Qwen3.6-27B, NO geodesic post-pass) — test if dropping our geodesic reaches the field 1.2 tier (dev-mean 1.78)"
for i in $(seq 1 60); do
  kaggle competitions submit arc-prize-2026-arc-agi-3 -k ahmedmobasher86/arc-agi-3-duck-reproduction -v 1 -f submission.parquet -m "$MSG" >/dev/null 2>&1
  sleep 25
  # verify: did a NEW submission with our pure-duck message appear?
  if kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -qi "PURE Tufa duck"; then
    echo "[attempt $i] SUBMITTED_OK"; kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -vE "Warning" | head -3; break
  fi
  echo "[attempt $i] not yet (slot closed) — retry in 5m"; sleep 275
done
