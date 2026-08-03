# Zero-diag working notes (sub 55197631, duck-patched v5, 0.00)

## Established facts
- 0.78 (55187670) = arc-agi-3-duck-depth kernel: duck-base.ipynb + depth_pack.py, bundle ahmedmobasher86/taaf-src-hybrid.
- 0.00 (55197631) = arc-agi-3-duck-patched v5: duck-repro.ipynb + duck_patches.py apply_all, bundle jeroencottaar/taaf-kaggle-source-share.
- Bundles are the same content: taaf-src-hybrid (uploaded Jul 3) is a copy of the Jun-12 share bundle; every listed file size matches byte-for-byte (setup_commands.json 10359, tool_agent.py 89299, vision_context.py 2311, pkls 3281/823); identical git_status lines in both commit logs.
- v5 does NOT contain the depth pack. duck_patches.py has no cross_level_notes/death-safe code; build_duck_patched.py inlines only duck_patches.py. The submission description ("WMR pack on depth-pack v1") is wrong about its own content.
- patch6 (`patch_tool_agent_analyze`) FAIL (`__file__` NameError): the NameError fires while building `possible_paths`, BEFORE any mutation -> clean no-op. apply_all catches per-patch; patch7-10 still applied. Same failure would occur in ANY notebook-exec context; patch6 has never been part of a scored config. Cannot zero the run.
- Local full-stack repro (repro_v5_full.py): exact hook-cell semantics (exec without __file__), full apply_all, TAAF_RUN_AS_SUBMISSION=1, competition-mode arcade, real scored bundle (scratchpad/taaf_scored_ref), mock brain. Result: behaviorally identical to no-patch control (same action counts, same benchmark completion, scorecard closes). Sandbox alive, grid-burner renders.
- ab-wmr kernel (real GPU, real 27B, curated patch subset incl. watchdog/hud/replay): banked ~1 level/game-run in all 4 waves; 3 watchdog recoveries, 0 kills. Patch trio works live.

## The decisive evidence
- Submitted 2026-08-03 00:07:39 UTC; COMPLETE with publicScore 0.00 by <= 00:23 UTC (Ahmed had score+logs then). A real run takes ~9h (110 games, 27B, concurrency 28). The hidden games were NEVER PLAYED.
- Kaggle API `total_bytes` of the scored submission file:
  - All genuinely-played duck runs: 3642-3710 bytes.
  - 55197631 (this zero): **3411**
  - 54847434 (EWM probe 0.00, Jul 20): **3411**
  - 54344660 (best-of-3 crater 0.00, Jul 5): **3411**
  - 3411 = the canonical "empty scorecard / all games zero-play" parquet. Commit dummy parquet is 2648 bytes -> the scored file is Kaggle/gateway-produced, not our dummy.
- Submission procedure delta vs every previously scored sub: submit_wmr.py PUSHED v5 at 00:05 UTC and SUBMITTED it ~90-150 s later with NO `kaggle kernels status`==COMPLETE gate (autosubmit_sft.sh has one; depth was submitted ~2 h after its push; farm draws submitted a days-old version).

## Scorecard trap (for the record; not the cause here)
- arc_agi scorecard.py `_calculate_score`: `len(env_info.baseline_actions) < len(card.actions_by_level[idx])` -> score 0.0 "Human baseline actions size mismatch". `Card.set_levels_completed` appends on ANY levels change (including decreases). A RESET-at-WIN that does NOT open a new play would re-append level entries and zero that game. Verified new-play on prod API + local sim, never on the Kaggle gateway. Per-game risk only; needs play to have happened.

## Repro scripts here
- repro_v5_full.py (full v5 hook cell, PASS on 2/3 games; k000 zero-action is a mock artifact, identical in control)
- repro_control.py (no patches; identical behavior incl. the 400-on-RESET noise)
- depth_out/ (duck-depth commit log), ab_out/ (ab-wmr real-GPU A/B log + ab_result.json), v5log parquet vs depth parquet (both 1-row dummies, 2648 bytes)
