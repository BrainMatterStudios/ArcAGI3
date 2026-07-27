#!/bin/bash
# Arm the Jul-27 00:00 UTC slot: 5th pinned duck-base v2 identical-bytes draw.
# Authorized by Ahmed 2026-07-26 ("Arm it"). Waits for the slot, submits,
# verifies by message marker, exits on confirmation.
cd /Users/ahmed/Documents/ArcAGI3 || exit 1
MARKER="farm draw #5"
MSG="Pinned duck-base v2 farm draw #5 (identical bytes; distribution to date: 0.92, 1.14, 0.82, 0.75; yardstick for the pre-registered debut stop-rule)."

# wait until 23:58 UTC before first attempt
while true; do
  H=$(date -u +%H); M=$(date -u +%M)
  if [ "$H" = "23" ] && [ "$M" -ge 58 ]; then break; fi
  if [ "$H" = "00" ] || [ "$H" = "01" ]; then break; fi
  sleep 60
done
echo "[armed] window reached at $(date -u '+%H:%M:%S UTC') — begin attempts"

for i in $(seq 1 40); do
  kaggle competitions submit arc-prize-2026-arc-agi-3 \
    -k ahmedmobasher86/arc-agi-3-duck-base -v 2 \
    -f submission.parquet -m "$MSG" >/dev/null 2>&1
  sleep 30
  if kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -qi "$MARKER"; then
    echo "[attempt $i] SUBMITTED_OK at $(date -u '+%H:%M:%S UTC')"
    kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -v Warning | head -3
    exit 0
  fi
  echo "[attempt $i] slot not open yet ($(date -u '+%H:%M:%S UTC')) — retry in 5m"
  sleep 270
done
echo "[FAIL] never confirmed after 40 attempts"
exit 1
