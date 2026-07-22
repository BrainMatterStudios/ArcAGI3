# duck-tplite — base duck + ACTION7 fix + KV retune + adaptive budget

The decomposed best-guess config after sub 54887708 (full patch bundle: ACTION7 +
animation metadata) scored **0.81** — below every clean base-duck draw (0.92 / 0.98 /
1.10 / 1.26). One draw is suggestive, not proof; but the plausible harm mechanism is the
animation metadata's per-action token tax (many games animate constantly), not the
ACTION7 fix (which only converts impossible error-turns into real probes on the ~6/25
games offering ACTION7). So: keep the bug fix, park the metadata, add the big lever.

## Contents

1. **ACTION7 round-trip fix** (from `_duck_patched`, patch 1 only). Confirmed bug:
   the bundle's action map covers ACTION1-6+RESET, so a legal ACTION7 shown to the
   model could never be executed. Animation metadata is **deliberately NOT applied**.
2. **KV retune** (from `_duck_throughput`): `--max-model-len` 65536 → analyzer window
   + 8192 = 40960. The KV cache supports ~10 concurrent max-length requests while the
   solver runs 28; the analyzer caps prompt+completion at 32768, so half of every
   sequence's reservation was pure waste. Fail-safe: any anchor mismatch serves
   unchanged.
3. **Adaptive per-game budget** (from `_duck_throughput`): raise-only, derived from
   `len(bm.games)` at runtime. 110 games → byte-for-byte current behaviour; 55 games →
   claims the ~4.6 idle hours.

Model, quantization, sampling, concurrency, n_passes, game list, submission path:
unchanged.

## Verification

Patch cell executed against the real harness:
```
[duck-patch] patch1 action7: OK
[duck-patch] patch3 reset: NOT NEEDED (RESET stripped from choices + auto-reset on GAME_OVER)
[duck-patch] animation metadata: PARKED (not applied in tplite)
ACTION7 round trip: ACTION7 | animation NOT wired: True
[duck-tplite] max-model-len 65536 -> 40960
```
All cells compile; cell keys (incl. attachments) preserved; anchors asserted.

## Build / push

```
.venv/bin/python submission/_duck_tplite/build_duck_tplite.py
kaggle kernels push -p submission/_duck_tplite --accelerator NvidiaRtxPro6000
```
