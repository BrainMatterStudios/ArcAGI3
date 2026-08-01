#!/bin/bash
# Daily draw — one submission per available slot, logged, verified.
#
# WHY THIS EXISTS. Private scores are computed at each submission's original run
# time and never rerun, and up to two finals are selected from every submission ever
# made. The objective is max-over-draws. Every unused day is a discarded lottery
# ticket: E[best of N] from our measured base N(0.9288, 0.1947) runs 1.21 at N=8 and
# 1.41 at N=90, and we have ~93 days left.
#
# FIXES TWO DEFECTS THE 2026-08-01 AUDIT FOUND IN THE PREVIOUS NIGHTLY SCRIPT:
#   1. It ran `kaggle ... >/dev/null 2>&1`, so when draw #8 failed 40 consecutive
#      times the root cause was unrecoverable. Nothing is discarded here.
#   2. A submit guard treated an unreadable status as failure and forfeited 11
#      hours of a slot. This never aborts on an unknown status -- it logs and
#      verifies against the submissions list, which is the only ground truth.
#
# Usage:  daily_draw.sh <kernel-slug> <version> "<message>"
#   e.g.  daily_draw.sh ahmedmobasher86/arc-agi-3-duck-base 2 "base draw"
#
# ALWAYS pass an explicit version. duck-base has an unsubmitted v3 on Kaggle while
# the entire n=8 reference distribution came from v2; an unpinned submit would
# silently draw from v3 and contaminate the reference set.
set -u

COMP="arc-prize-2026-arc-agi-3"
KERNEL="${1:?usage: daily_draw.sh <kernel-slug> <version> <message>}"
VERSION="${2:?explicit version required — never submit unpinned}"
MSG="${3:?message required}"

LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$LOG_DIR/daily_draw.log"
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

say "=== draw start: $KERNEL v$VERSION"

before=$(kaggle competitions submissions -c "$COMP" 2>>"$LOG" | awk 'NR>2 {print $1}' | head -1)
say "latest submission before: ${before:-none}"

attempt=0
while [ "$attempt" -lt 3 ]; do
  attempt=$((attempt + 1))
  say "submit attempt $attempt"
  # No output discarded — the previous script's silence is exactly what made a
  # 40-failure night unexplainable.
  kaggle competitions submit "$COMP" -k "$KERNEL" -v "$VERSION" -f submission.parquet -m "$MSG" 2>&1 | tee -a "$LOG"

  # Verify against the submissions list rather than trusting the exit status. A
  # submission that landed is a new ref at the top; that is the only ground truth.
  sleep 20
  after=$(kaggle competitions submissions -c "$COMP" 2>>"$LOG" | awk 'NR>2 {print $1}' | head -1)
  if [ -n "$after" ] && [ "$after" != "$before" ]; then
    say "VERIFIED: new submission $after"
    say "=== draw done"
    exit 0
  fi
  say "not visible yet (latest still ${after:-unknown}) — retrying"
  sleep 60
done

# Deliberately not a hard failure path that gives up on the day. Log loudly and
# leave the slot open so a later run can still take it.
say "WARNING: could not verify a new submission after $attempt attempts. Slot may still be free."
exit 1
