#!/bin/zsh
cd /Users/ahmed/Documents/ArcAGI3 || exit 1
BASE=https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1
OUT1=offkaggle/results/$(date -u +%Y%m%dT%H%M)-halfconc-keith
echo "=== $(date -u +%H:%MZ) ARM halfconc -> $OUT1 ==="
.venv/bin/python offkaggle/run_regime_wave.py --arm keith --base-url $BASE --games all --concurrency 14 --per-game-s 3960 --out "$OUT1"
echo "=== $(date -u +%H:%MZ) ARM halfconc rc=$? ==="
OUT2=offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_yield900
echo "=== $(date -u +%H:%MZ) ARM keith_yield900 -> $OUT2 ==="
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_yield900 --base-url $BASE --games all --out "$OUT2"
echo "=== $(date -u +%H:%MZ) ARM keith_yield900 rc=$? ==="
echo "=== EVENING CHAIN DONE $(date -u +%H:%MZ) ==="
