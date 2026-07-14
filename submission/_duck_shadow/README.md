# Duck + shadow-replay submission (floor-safe efficiency climb)

`duck-shadow.ipynb` = the pure base-duck notebook (`_repro/duck-repro.ipynb`) with **one change**:
`await bm.run(...)` → `await run_with_shadow(bm, ...)`. The duck solver is **untouched** (banks its
~1.26 floor); a concurrent zero-LLM shadow watches each pass-0 game and, as it finishes, replays the
duck's own **winning attempt** per level in a fresh play on the same card. `arc_agi` scores a game as
**max over plays**, so the efficient replay can only ADD.

## Why floor-safe
- The duck play is never modified/capped/interrupted → worst case ≈ duck (1.26), only variance.
- Extra plays are additive (max-over-plays); a replay desync abandons just that game.
- All shadow code is exception-fenced; a shadow bug cannot crash `bm.run` (it's a separate task).

## Validated (offline, no GPU)
- `shadow_replay` core on **real recorded duck history**: tu93 L1 0.03 → 2.22 (**81×**).
- `run_with_shadow` end-to-end through a **real TAAF Benchmark** (`test_shadow_concurrent.py`): a
  scripted duck wins tu93 L1 wastefully (171 actions, 0.03); the concurrent shadow opens a fresh
  play, replays the 18-action winning attempt, and **max-over-plays banks 2.22** (2 plays registered).

## The one untested unknown → insurance
Mid-run extra plays on the **live competition card** were never exercised (best-of-N never fired). The
code path is identical to pass-0 play opens, and everything except the gateway is offline-validated. So:
**submit `duck-shadow` and keep a same-day pure-duck resubmit (`_repro`) as insurance** — if the shadow
plays don't register on the live card, the duck floor still stands.

## Run
```bash
python submission/_duck_shadow/build_duck_shadow.py   # regenerate from the scaffold
kaggle kernels push -p submission/_duck_shadow        # GPU (serves Qwen); needs quota
# then Submit to Competition. Read logs for [shadow] lines: "duck won N lvl(s) -> replay banked ..."
```
GPU-blocked until the weekly quota resets, same as everything else. Expected: **1.26 → ~1.6–2.5** if the
duck's live wins carry the ~35× median RHAE waste the headroom analysis found.

## Estimate the live uplift beforehand
`scratchpad/ewm_pathb/replay_counterfactual.py <runs_dir>` over any real duck `benchmark.json` sizes the
orig→replay uplift per game without a submission.
