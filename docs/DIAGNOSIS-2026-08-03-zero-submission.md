# DIAGNOSIS 2026-08-03 — the duck-patched v5 0.00 (submission 55197631)

**Verdict: push→submit race. The hidden games were never played.** The patches were
not the cause; the submission procedure was. Repro scripts and captured logs:
`docs/test-artifacts-2026-08-02/zero_diag/`.

## The two submissions being compared

| sub | kernel | content | score |
|---|---|---|---|
| 55187670 (0.78) | arc-agi-3-duck-depth | duck-base.ipynb + depth_pack.py, bundle `ahmedmobasher86/taaf-src-hybrid` | genuine run |
| 55197631 (0.00) | arc-agi-3-duck-patched v5 | duck-repro.ipynb + duck_patches.py apply_all, bundle `jeroencottaar/taaf-kaggle-source-share` | never played |

The bundles are byte-identical content (taaf-src-hybrid is a Jul-3 copy of the Jun-12
share bundle; every listed file size matches — setup_commands.json 10359,
tool_agent.py 89299, vision_context.py 2311, pkls 3281/823 — and both commit logs
print identical git_status lines). The bundle swap is NOT a variable.

## Root cause

- Submitted 2026-08-03 **00:07:39 UTC**; **COMPLETE with publicScore 0.00 by ≤ 00:23
  UTC**. A real run takes ~9 h (110 games, 27B, concurrency 28). Sixteen minutes to a
  scored COMPLETE means the scored rerun never played the games.
- Procedure delta vs every previously scored submission: `submit_wmr.py` **pushed v5
  at 00:05 UTC and submitted it ~90–150 s later with no `kaggle kernels
  status`==COMPLETE gate**. autosubmit_sft.sh has such a gate; the depth sub was
  submitted ~2 h after its push; farm draws submitted a days-old version.

## The 3411-byte never-played fingerprint

`total_bytes` of the scored submission file, from the Kaggle submissions API (the
CLI's CSV does not expose it):

| submission | total_bytes | outcome |
|---|---|---|
| all genuinely-played duck runs | **3642–3710** | real scores |
| 55197631 (this zero) | **3411** | 0.00 |
| 54847434 (EWM probe, Jul 20) | **3411** | 0.00 |
| 54344660 (best-of-3 crater, Jul 5) | **3411** | 0.00 |

3411 bytes = the canonical "empty scorecard / all games zero-play" parquet. It is
**gateway-produced**, not our CPU-safe commit dummy (that parquet is 2648 bytes).
Any future submission at ≤ 3500 bytes was never played — `scripts/submit_gated.py`
alarms on exactly this.

## The patch-defect hypothesis — refuted

- **patch6 `__file__` NameError** (the only FAIL in the v5 commit log): fires while
  building `possible_paths`, **before any mutation** → clean no-op. `apply_all`
  catches per-patch; patches 7–10 still applied. The same failure occurs in ANY
  notebook-exec context and patch6 was never part of a scored config. It cannot zero
  a run. (Fixed anyway on 2026-08-03: resolves without `__file__`, presence-gates
  arcagi3, SKIPs with an accurate message.)
- **Full-stack local repro** (`repro_v5_full.py`): exact hook-cell semantics (exec
  without `__file__`), full `apply_all`, `TAAF_RUN_AS_SUBMISSION=1`,
  competition-mode arcade, the real scored bundle bytes
  (`scratchpad/taaf_scored_ref`), mock brain — **behaviourally identical to the
  no-patch control** (`repro_control.py`): same action counts, benchmark completes,
  scorecard closes, sandbox alive. (ft09 banks 0 actions under the mock in patched
  AND control — a mock artifact, not a patch effect.)
- **ab-wmr kernel** (real GPU, real 27B, curated patch subset incl.
  watchdog/HUD-mask/replay): banked ~1 level/game-run in all 4 waves, 3 watchdog
  recoveries, 0 kills. The patch trio works live.

## Scorecard trap (for the record; NOT the cause here)

arc_agi `scorecard.py::_calculate_score`: `len(env_info.baseline_actions) <
len(card.actions_by_level[idx])` → score 0.0 "Human baseline actions size
mismatch". `Card.set_levels_completed` appends on ANY levels change, including
decreases — a RESET-at-WIN that does NOT open a new play would re-append level
entries and zero that game. Verified to open a new play on the prod API and the
local sim, never verified on the Kaggle gateway. Per-game risk only; requires play
to have happened, so it cannot explain a 3411-byte file.

## Side findings

1. **v5 did not contain the depth pack.** Its description said "WMR pack on
   depth-pack v1"; duck_patches.py had no cross_level_notes / death-safe code and
   build_duck_patched.py inlines only duck_patches.py. The submission message lied
   about its own content → submit_gated.py's builder-honesty gate greps the
   notebook for the markers of every pack the message names.
2. **Grid-burner defacement.** patch4 draws ~11px default-font labels + per-cell
   grid lines at the scored config's `MULTIMODAL_UPSCALE=4` (4px cells) —
   substantial frame defacement, only ever validated at dev upscale 16, never part
   of a scored config. Now default OFF behind `TAAF_GRID_BURNER` (2026-08-03).

## Laws

1. **Never submit a version that has not positively read COMPLETE, plus a settle.**
   A submit racing the version publish scores a 3411-byte never-played parquet as
   0.00 and burns the slot.
2. **`scripts/submit_gated.py` is the only sanctioned submit path**: COMPLETE +
   10-min settle before; 90-min fingerprint watch after (alarm on total_bytes ≤
   3500 or COMPLETE in under 1 h); builder-honesty marker check on the message.
3. **A 0.00 with total_bytes 3411 is a procedure failure, not a model result.** Do
   not spend cycles on patch archaeology before checking the byte size and the
   submit-vs-push timing.
4. **The pre-push e2e gate** (`submission/_duck_patched/test_notebook_hook_e2e.py`)
   execs the shipped notebook's hook cell Kaggle-style (no `__file__`) against the
   scored bundle bytes and drives a mock game — run it before every push; it also
   catches stale notebooks and the patch6/patch9 structural classes.

## Archived artifacts (docs/test-artifacts-2026-08-02/zero_diag/)

- `NOTES.md` — the working notes this document summarizes
- `repro_v5_full.py` — full v5 hook-cell repro (exec without `__file__`, scored
  bundle, submission mode, mock brain)
- `repro_control.py` — no-patch control (identical behaviour incl. 400-on-RESET noise)
