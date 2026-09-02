#!/bin/zsh
# Retry `kaggle kernels push` of the attested keithtyser V14 byte-copy until the weekly GPU quota
# resets (expected Sat 2026-09-05 00:00 UTC). Commit run only — NOT a submission.
cd /Users/ahmed/Documents/ArcAGI3 || exit 1
LOG=logs/push_keith_copy.log
n=0
while true; do
  n=$((n+1))
  out=$(kaggle kernels push -p submission/_keith_copy/push 2>&1)
  if echo "$out" | grep -q "successfully pushed"; then
    echo "[push-when-free $(date -u +%Y-%m-%dT%H:%M:%SZ)] attempt $n: PUSHED — $out" >> $LOG
    exit 0
  fi
  echo "[push-when-free $(date -u +%Y-%m-%dT%H:%M:%SZ)] attempt $n: $(echo "$out" | tail -1 | cut -c1-120) — retry in 1800s" >> $LOG
  sleep 1800
done
