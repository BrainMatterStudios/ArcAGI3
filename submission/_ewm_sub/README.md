# EWM competition submission (verification-by-execution, Qwen3.6-27B)

`ewm-submission.ipynb` = the **proven duck submission harness** (dataset mounts, `setup_commands`
that serve Qwen3.6-27B-FP8 on `127.0.0.1:1234`, COMPETITION gateway game list, benchmark load,
`submission.parquet`, teardown) with **one change: `bm.solver = SolverEWM(...)`**. Every action the
EWM loop takes flows through TAAF `game.execute_action` and is scored.

## What's validated (offline, on this host)
- `SolverEWM` (taaf.solver.Solver) + `server_taaf` (wraps the TAAF game) + `ewm_agent` + `search_lib`
  ran end-to-end through a real TAAF Benchmark on a dev game with an MLX brain → TAAF produced a
  valid scorecard. Plumbing is exact.
- The embedded-scaffold layout + imports + `soft_end_time` pacing verified locally.
- Reused wiring is the same that produced the duck's 1.26.

## What's only testable at eval (the honest unknowns)
- Whether a **local Qwen-27B** (weaker than the Claude brain that scored ~15.6/100 mean on 9 hard
  dev games) solves anything at the **~290 s/game** box (`EWM_PER_GAME_SECONDS`, ~110 games in 9h).
- This is a **PROBE for signal**, not a guaranteed climb: it is not strictly floor-safe (EWM does
  not run the duck), and Qwen at 5 min/game is far tighter than Claude's ~90 min/game — it may
  score **below 1.26** while still proving the method + measuring the real hidden-game solve rate.
- Read the kernel logs (`[ewm] ...`, per-game `levels_completed`) for the real signal even if the
  score is low. Knobs: `EWM_PER_GAME_SECONDS` (fewer-deeper vs more-shallow), `EWM_N_PASSES`
  (1=probe; ≥2 needs a replay pass — TODO, see design doc).

## Submit mechanics + GPU
- The **scored rerun** runs `TRUE_SUBMISSION` mode on the **RTX Pro 6000** (FP8-capable), on infra
  **separate from the 30h interactive quota**.
- BUT pushing/committing the kernel uses interactive GPU (currently **quota-exhausted**). If a plain
  `kaggle kernels push` is quota-blocked, wait for the weekly reset, then push + Submit to Competition.
- Build: `python submission/_ewm_sub/build_ewm_submission.py` (regenerates from the validated scaffold).

## Next iteration (after the probe reads back)
- If Qwen solves games but the box is too tight → raise `EWM_PER_GAME_SECONDS`, fewer games.
- Add the n_passes≥2 explore-then-replay so a clean pass banks the efficient score (scoring is
  cumulative-within-pass; see SELECTIVE_EWM_SUBMISSION_DESIGN.md).
- If Qwen is too weak → swap the brain to gpt-oss-120b (needs the harmony-offline vocab fix).
