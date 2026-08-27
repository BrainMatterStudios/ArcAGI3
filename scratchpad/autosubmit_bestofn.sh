#!/bin/bash
cd /Users/ahmed/Documents/ArcAGI3
MSG="Floor-safe sequential best-of-N: pure Tufa duck (Qwen3.6-27B) + additive replay of each game as new plays (max-over-plays) when budget remains after pass 0. Overlay-verified applied; pass-0 = reproduction floor."
for i in $(seq 1 60); do
  kaggle competitions submit arc-prize-2026-arc-agi-3 -k ahmedmobasher86/arc-agi-3-duck-bestofn -v 2 -f submission.parquet -m "$MSG" >/dev/null 2>&1
  sleep 25
  if kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -qi "Floor-safe sequential best-of-N"; then
    echo "[attempt $i] BESTOFN_SUBMITTED_OK"; kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -vE "Warning" | head -3; break
  fi
  echo "[attempt $i] slot closed, retry in 5m"; sleep 275
done
