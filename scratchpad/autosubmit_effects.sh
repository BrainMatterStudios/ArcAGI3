#!/bin/bash
# Submit duck-effects v2 (B1 grounded effect memory) in the 2026-08-01 00:00 UTC slot.
#
# v2 is the CORRECTED build: effect_memory.py is inlined into the notebook's
# customization-hook cell and its application is hard-verified. The Kaggle commit run
# for v2 logged "[effects] ARM ACTIVE — EFFECT_MEMORY=1" plus all four verify lines,
# so the arm is proven live before the slot is spent. (v1 only set an env var no shipped
# code reads — it would have scored as plain base duck under an effect-memory label.)
#
# Authorized by Ahmed 2026-07-31 as a deliberate deviation from B-TRACK-PROTOCOL
# §1/§3 (unmeasured arm, A-track selection does not yet exist). Recorded as a §6
# amendment; the result is EXPLORATORY, not confirmatory, and a single draw cannot
# separate the arm from base (base n=8: mean 0.929, sd 0.195).
cd /Users/ahmed/Documents/ArcAGI3 || exit 1

MARKER="B1-effects-exploratory"
MSG="Duck-Effects v2: base duck + B1 grounded effect memory (harness-measured action/click effect table + productivity grid in every user prompt; system line binds CONFIRMED to observed transitions). Patch application hard-verified in the commit log. [$MARKER] EXPLORATORY per B-TRACK-PROTOCOL amendment 6.1 — single draw, base n=8 mean 0.929 sd 0.195, not a confirmatory measurement."

# Wait for the 00:00 UTC window.
while true; do
  H=$(date -u +%H); M=$(date -u +%M)
  if [ "$H" = "00" ] && [ "$M" -ge 5 ]; then break; fi
  sleep 120
done
echo "[effects] slot window reached at $(date -u '+%F %H:%M:%S UTC')"

# Re-confirm the kernel is still COMPLETE before spending the slot.
STATUS=$(kaggle kernels status ahmedmobasher86/arc-agi-3-duck-effects 2>/dev/null | tail -1)
echo "[effects] kernel status: $STATUS"
case "$STATUS" in
  *COMPLETE*) ;;
  *) echo "[effects] ABORT — kernel not COMPLETE, refusing to submit"; exit 1;;
esac

OK=0
for i in $(seq 1 40); do
  kaggle competitions submit arc-prize-2026-arc-agi-3 \
    -k ahmedmobasher86/arc-agi-3-duck-effects -v 2 \
    -f submission.parquet -m "$MSG" >/dev/null 2>&1
  sleep 30
  # Confirm by unique marker rather than trusting the exit code.
  if kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -q "$MARKER"; then
    echo "[effects] SUBMITTED_OK at $(date -u '+%F %H:%M:%S UTC') (attempt $i)"
    OK=1; break
  fi
  echo "[effects] attempt $i not confirmed ($(date -u '+%H:%M:%S UTC')) — retry in 5m"
  sleep 270
done
[ "$OK" = "1" ] || echo "[FAIL] duck-effects never confirmed in the 2026-08-01 slot"
echo "[effects] done at $(date -u '+%F %H:%M:%S UTC')"
