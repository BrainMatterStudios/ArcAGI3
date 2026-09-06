#!/bin/zsh
# Disambiguation arm: same queue-free geometry (conc 3, 60 calls) but the FULL 7920 s clock shown to the model.
cd /Users/ahmed/Documents/ArcAGI3 || exit 1
LOG=/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/wall_chain.log
until grep -q 'CHAIN DONE' "$LOG"; do sleep 60; done
OUT=offkaggle/results/$(date -u +%Y%m%dT%H%M)-wall-keith-longclock
echo "=== $(date -u +%H:%MZ) ARM keith-longclock -> $OUT ==="
.venv/bin/python offkaggle/run_regime_wave.py --arm keith --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 --games cd82,dc22,lf52 --draws 2 --concurrency 3 --max-calls 60 --per-game-s 7920 --out "$OUT"
echo "=== $(date -u +%H:%MZ) ARM keith-longclock rc=$? ==="
echo "=== FOLLOWUP DONE $(date -u +%H:%MZ) ==="
