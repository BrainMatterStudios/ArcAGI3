# Selective-EWM submission — design (execute IF the Qwen gate is positive)

Goal: a floor-safe competition submission that runs verification-by-execution on the 110 hidden
games with a LOCAL Qwen-27B brain, banks EWM cracks, and never scores below the duck floor.

## What the gate must return first (gates this whole build)
- Qwen mean RHAE >~10/100 on the 3 dev games → viable.
- tok/s (serve throughput) and out_tok/game → sizes the per-game time-box + how many games EWM can attempt in 9h.

## Interface: drive the live gateway, not my flask server
At eval the games are served by the Kaggle gateway (`ARC_BASE_URL=http://gateway:8001/`). `arc_agi`
already speaks it: `RemoteEnvironmentWrapper` → `{base_url}/api/cmd/RESET` + action endpoints + a single
competition `card_id` (competition mode forbids multiple open scorecards). Same `reset()`/`step(action, data)`
surface as the OFFLINE wrapper used in the gate.
=> Reuse the ENTIRE ewm scaffold. Two integration options:
  (A) In-process adapter (preferred): replace the flask-server+HTTP-client with a thin `client` shim
      that calls the framework-provided remote env object directly and writes the SAME session artifacts
      (level_XX_attempt_YY/step_*.txt + metadata) the verifier reads. No HTTP, no ports, no flask.
  (B) Keep flask server but point its Arcade at competition/remote mode. Simpler code reuse, but runs an
      extra local HTTP hop and needs the gateway creds/card wired into the server. Fallback if (A) is fiddly.

## Floor-safe selective scheduler (fit 9h / 110 games / 1 GPU)
Per-game budget T_ewm (from gate tokens/game ÷ tok/s; e.g. cap 10–15 min/game). For each hidden game:
  1. DUCK FLOOR pass first (cheap, banks the known ~1.26-class score for that game) — OR run duck as the
     global pass-0 exactly as today, so the submission is duck ∪ EWM (max-over-plays can only add).
  2. If budget remains, EWM pass (time-boxed): serve Qwen, run ewm_agent on that game via the gateway env,
     execute the verified plan as ADDITIONAL plays. Fail-fast if verify doesn't converge in the box.
  3. max-over-plays banks whichever (duck vs EWM) scored higher on that game.
CRITICAL LESSON ([[arcagi3-leap-findings]] LEAP-2 = 0.83 FAIL): do NOT cap pass-0 (duck) to free EWM slack —
that shortened the duck and regressed. Instead EWM must get its budget from games the duck FAST-FAILS
(0-score games that finish quickly), OR run EWM only on a curated slot count that provably doesn't starve
the duck. The gate's tok/s decides how many games EWM can touch without stealing duck budget.
  - Safest v1: EWM runs ONLY after the duck's full 110-game pass completes, on remaining budget, on the
    games the duck scored 0 (identifiable at runtime). If no slack (duck fills 9h), EWM simply doesn't fire
    → submission == duck floor (never worse). Then iterate the box/selection with live feedback.

## Build checklist (post-gate)
- [ ] client shim: gateway-remote env → session-artifact writer (option A).
- [ ] ewm_agent unchanged (already portable; point --base-url at the in-notebook vLLM).
- [ ] scheduler: duck pass-0 (reuse TAAF harness) + post-pass EWM on 0-score games within remaining wall-clock.
- [ ] floor guard: EWM plays are additive (new plays); a failed/timed-out EWM game leaves the duck score intact.
- [ ] budget instrumentation: log per-game wall-clock + tokens so we can tune the box across submissions.
- [ ] one dev dry-run of the full notebook against the OFFLINE engine (proxy for gateway) before submitting.

## *** DE-RISKED PATH (2026-07-14): integrate EWM as a TAAF Solver — OFFLINE-VALIDATABLE ***
The TAAF framework source is vendored at `submission/_adopt/taaf-src/src/tufa-arc-agi-framework/` and IMPORTS+RUNS on this Mac (deps: arc_agi, arcengine, numpy, scipy, matplotlib — all present). TAAF has an OFFLINE mode (`ArcadeSpec(operation_mode=OFFLINE, environments_dir=...)`) → the full harness (scorecard, submission.parquet, budget) can be run LOCALLY on dev games with an MLX brain to validate the integration BEFORE submitting. This removes the "unvalidatable gateway/parquet" risk that made a blind submission a gamble.

INSERTION POINT: subclass `taaf.solver.Solver` and implement `async _run_games(games)`. Contract (taaf/solver.py): play each `game` via `game.execute_action(ActionInput)`; on `asyncio.CancelledError` call `game.finish_game()` on every still-`playing` game and re-raise; yield >=every 5s; respect `self.soft_end_time` (pacing hint; Benchmark enforces cancel). `Game`/`GameState` (taaf/game.py): `.execute_action(action)->GameState`, `.frame` (2D int8 0-15), `.available_actions`, `.levels_completed`, `.won`, `.start_game()`, `.finish_game()`.

EWM SOLVER ARCHITECTURE (`SolverEWM._run_games`): for each game (until soft_end_time), (1) start a flask server that WRAPS THIS `game` object — a variant of `server.py` whose /game/action calls `game.execute_action` and records the session artifacts the verifier reads (reuse serialize_frame; RESET -> game reset semantics); (2) create a fresh workspace (workspace_init copy + wrappers); (3) run `ewm_agent.py` (subprocess) driven by the vLLM Qwen endpoint with a per-game time-box; (4) the plan-execution actions flow through the wrapped game -> TAAF scores them. Floor: give EWM a bounded total budget (soft_end_time slice); duck solver handles the rest OR (simpler v1) EWM-only on all games with a per-game box. Because TAAF owns the scorecard/parquet, the submission is VALID by construction.

BUILD + VALIDATE ORDER: (a) confirm a trivial TAAF OFFLINE benchmark run locally (SolverRandom on 1 dev game) -> foundation. (b) build the game-wrapping server variant + SolverEWM. (c) OFFLINE-validate SolverEWM on 2-3 dev games with MLX-7B brain (checks the FULL integration incl. artifacts+verify+plan-exec+scoring; 7B won't solve but the PLUMBING must be exact). (d) build the submission notebook = duck's setup_commands (serve Qwen) + TAAF benchmark with solver swapped to SolverEWM; keep duck as fallback/floor if budget allows. (e) submit; read logs for real EWM-on-hidden solve rate even if score is a probe.
The 1.26 duck notebooks (`submission/_finetuned`, `_repro`) show the exact COMPETITION Arcade + gateway + benchmark wiring to copy.

## Open params (set from gate)
- T_ewm per-game time-box; number of EWM-eligible games; Qwen serve flags (max-model-len, fp8).
- Whether to switch brain to gpt-oss-120b (if Qwen too weak) — needs the harmony-offline vocab fix.
