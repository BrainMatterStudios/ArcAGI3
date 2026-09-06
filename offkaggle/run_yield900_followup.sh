#!/bin/zsh
cd /Users/ahmed/Documents/ArcAGI3 || exit 1
LOG=/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/regime_halfconc.log
until grep -q 'REGIME WAVE' "$LOG"; do sleep 60; done
OUT=offkaggle/results/$(date -u +%Y%m%dT%H%M)-keith_yield900
echo "=== $(date -u +%H:%MZ) ARM keith_yield900 -> $OUT ==="
.venv/bin/python offkaggle/run_regime_wave.py --arm keith_yield900 --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 --games all --out "$OUT"
echo "=== $(date -u +%H:%MZ) ARM keith_yield900 rc=$? ==="
echo "=== Y900 DONE $(date -u +%H:%MZ) ==="
