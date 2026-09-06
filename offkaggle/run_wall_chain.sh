#!/bin/zsh
# 09-06 3-wall instrument: four arms in sequence on one server session.
cd /Users/ahmed/Documents/ArcAGI3 || exit 1
BASE=https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1
STAMP=$(date -u +%Y%m%dT%H%M)
for ARM in keith keith_evid keith_hypo keith_up8; do
  OUT=offkaggle/results/${STAMP}-wall-${ARM}
  echo "=== $(date -u +%H:%MZ) ARM $ARM -> $OUT ==="
  .venv/bin/python offkaggle/run_regime_wave.py --arm $ARM --base-url $BASE --games cd82,dc22,lf52 --draws 2 --concurrency 3 --max-calls 60 --per-game-s 1500 --out "$OUT"
  echo "=== $(date -u +%H:%MZ) ARM $ARM rc=$? ==="
done
echo "=== CHAIN DONE $(date -u +%H:%MZ) ==="
