#!/bin/bash
# Nightly pinned duck-base v2 farm draws #6-#9 (slots of Jul 28, 29, 30, 31 UTC).
# Standing authorization from Ahmed 2026-07-27 ("Yes" to arming through Jul 31).
# One submission per 00:00 UTC window; verifies each by unique marker; logs all.
cd /Users/ahmed/Documents/ArcAGI3 || exit 1

DRAW=6
while [ "$DRAW" -le 9 ]; do
  # wait for the next 23:58 UTC window
  while true; do
    H=$(date -u +%H); M=$(date -u +%M)
    if [ "$H" = "23" ] && [ "$M" -ge 58 ]; then break; fi
    sleep 120
  done
  MARKER="farm draw #$DRAW"
  MSG="Pinned duck-base v2 farm draw #$DRAW (identical bytes; n=5 to date: mean 0.918 sd 0.149; nightly yardstick series through Jul 31)."
  echo "[night] window for draw #$DRAW reached at $(date -u '+%F %H:%M:%S UTC')"

  OK=0
  for i in $(seq 1 40); do
    kaggle competitions submit arc-prize-2026-arc-agi-3 \
      -k ahmedmobasher86/arc-agi-3-duck-base -v 2 \
      -f submission.parquet -m "$MSG" >/dev/null 2>&1
    sleep 30
    if kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -qi "$MARKER"; then
      echo "[draw#$DRAW attempt $i] SUBMITTED_OK at $(date -u '+%F %H:%M:%S UTC')"
      OK=1; break
    fi
    echo "[draw#$DRAW attempt $i] not confirmed yet ($(date -u '+%H:%M:%S UTC')) — retry in 5m"
    sleep 270
  done
  if [ "$OK" != "1" ]; then
    echo "[FAIL] draw #$DRAW never confirmed — continuing to next night anyway"
  fi
  DRAW=$((DRAW + 1))
  # sleep well past the current window before re-arming (so we don't double-fire)
  sleep 7200
done
echo "[done] nightly series complete (draws 6-9 attempted) at $(date -u '+%F %H:%M:%S UTC')"
