# R9 — Path to 7+: first-principles gap analysis (fresh 25-game v31 run)

**Date:** 2026-08-31. **Method:** deliberately memory-blind; everything below derived from the raw v31 offline run (`benchmark.json` per-action history + 13 downloaded `*_p0_events.jsonl` traces) and the scoring formula. No prior repo conclusions consulted.

**Run under analysis:** 25 games, 1 pass, all 25 concurrent for the full 7,920s wall, local mean **4.50**, 1.12 levels/game, 7 zero-level games, 2,733 actions, 3.78M generated tokens (477 tok/s aggregate).

---

## 1. Arithmetic: what 7.0–7.5 actually requires

### 1.1 Formula, verified against the run

Reconstructed game score = `min( Σ_completed w_i · min(115, 100·(b_i/a_i)²) / Σ_all w_j , 100 · Σ_reached w / Σ_all w )` with `w_i = level_index+1`. This reproduces **all 25 reported scores exactly** (the 8 initial "mismatches" were all the completion-share cap binding — e.g. ar25 uncapped 3.19 → cap 100·1/36 = 2.78 reported).

**Consequence 1 (cap law, re-derived from raw data):** the per-level 115% bonus is almost always clipped by the completion-share cap. Beating the human baseline buys ~nothing; falling below it costs **quadratically**. Efficiency is a one-sided penalty, not a reward.

**Consequence 2 (retry trap):** retries accumulate into the same level bucket, so a level cleared on attempt 10 scores `(b/Σa)²` ≈ 0. sp80 cleared L1 after 333 actions (baseline 39): it kept 0.07 of an available 4.76 pts — a 99% loss *despite clearing the level*.

### 1.2 Current distribution (local 25)

- mean 4.50, median 1.82; top: ft09 37.4 (4 lv), lp85 16.7 (3 lv), vc33 10.7, s5i5 8.3, cd82 7.0
- 7 games at 0.00 (0 levels), 6 more below 1.5
- Actions spent: 2,733. Human-baseline cost of the levels we actually cleared: **865**. We spent 3.2× the baseline price for the depth we got.

### 1.3 Scenario table (exact, on the local 25, cap applied)

| profile | local mean |
|---|---|
| current | **4.50** |
| P1: efficiency-to-cap at current depth (no new levels) | **5.93** (+1.42) |
| P2: P1 + the 7 zero-games clear L1 | **7.04** (+2.54) |
| every game exactly 1 level @cap | 3.52 |
| every game exactly 2 levels @cap | 10.57 |
| P4: efficiency-to-cap + every game ≥2 levels | **12.24** |
| P3: efficiency-to-cap + every game +1 level | **13.28** |
| every game 3 levels @cap | 21.14 |

Efficiency loss breakdown (P1's 35.6 game-pts): ft09 10.2 (L3: 63 acts vs baseline 23), cd82 7.3 (L2: 20 vs 8), **sp80 4.7 (L1: 333 vs 39, 14 game-overs)**, dc22 2.7, ka59 2.5, re86 2.4, su15 2.1, tu93 1.5, bp35 1.5, wa30 0.7.

### 1.4 The honest LB conversion

Local 4.50 projects to LB 2.2–2.65 ⇒ transfer ratio **0.49–0.59** (hidden set is harder/larger). Therefore:

- **LB 7.0 requires local-equivalent ≈ 11.9–14.3.**
- **LB 7.51 (cstl) requires local ≈ 12.8–15.4.**

That is exactly the P3/P4 band. **The headline: 7+ on the LB ≈ "every game one level deeper than today AND zero efficiency leakage" — roughly 2.1 levels/game at ≥baseline efficiency.** The minimum-change path (P2 = 7.04 local) only lands ~3.5–4.2 LB. Neither depth alone nor efficiency alone gets there; cstl's 7.51 implies their harness clears ~2 levels/game on hidden games at near-baseline cost.

Ranked marginal yields (independent, local mean):
1. +1 level on all 25 games at cap: **+7.35**
2. efficiency-to-cap, no new levels: **+1.42**
3. 7 zeros → L1 at cap: **+1.11**
4. each fully-completed game: +4.00 each (crack lane; doesn't scale)

Action budget is NOT the constraint: the P2 profile costs 1,178 actions at baseline (43% of what we spent); the full 2-level profile costs 2,542 — less than the 2,733 we already emit. **We spend enough actions; they land in the wrong buckets.**

## 2. Trace forensics (13 games: best 2, all 6 traced zeros, 5 mids)

### 2.1 Turn economics — the real currency is LLM turns, and half are wasted

From `benchmark.json` (all 25): 703 acting turns total, **28.1 acting turns/game** in the 2h12m window, ~230s and ~5.4k generated tokens per turn (~19 tok/s/stream at 25-way concurrency). Kaggle's box gives each game a similar ~2.3h slice (9h × 28 / 110).

From the 13 event traces (683 analysis turns):

| game | score | acts | turns | idle turns | pre-1st-clear acts | GAME_OVERs | no-effect acts | mean batch | tail acts (next lvl) |
|---|---|---|---|---|---|---|---|---|---|
| ft09 | 37.4 | 115 | 41 | 20 | 6 | 0 | 2 | 13.6 | 11 |
| lp85 | 16.7 | 70 | 40 | 19 | 6 | 0 | 3 | 8.3 | 4 |
| vc33 | 10.7 | 127 | 54 | 27 | 6 | 1 | 8 | 1.1 | 109 |
| sb26 | 2.8 | 127 | 69 | 31 | 12 | 1 | 6 | 5.3 | 115 |
| dc22 | 2.1 | 167 | 60 | 19 | 90 | 0 | 35 | 7.5 | 77 |
| r11l | 4.8 | 7 | 32 | 24 | 5 | 0 | 0 | 1.0 | 2 |
| sp80 | 0.07 | 370 | 66 | 24 | 333 | **14** | 14 | 6.2 | 37 |
| g50t | 0 | 100 | 53 | **36** | 100 | 0 | **38** | 16.8 | 0 |
| m0r0 | 0 | 296 | 70 | 38 | 296 | 1 | 47 | 1.0 | 0 |
| sc25 | 0 | 114 | 59 | 23 | 114 | 2 | 13 | 1.1 | 0 |
| sk48 | 0 | 50 | 57 | 27 | 50 | 0 | 7 | 2.6 | 0 |
| ls20 | 0 | 56 | 40 | 20 | 56 | 0 | 2 | 2.5 | 0 |
| tn36 | 0 | 44 | 42 | 22 | 44 | 0 | 0 | 1.0 | 0 |

**Bucket totals (1,643 traced actions, 683 turns):**
- **Idle turns: 330/683 = 48%** — `step_executed: False`, overwhelmingly `Yielded control to solver: turn_time_budget` (the 60s yield cut the model off before it reached a tool call; the thinking is discarded), plus 13 vLLM read-timeout turns.
- Exploration before first clear: 660 actions across the 6 zero games (40% of traced actions, zero points), plus dc22's 90 and sp80's 333.
- Retry-replay after GAME_OVER: sp80 alone ~300 actions across 15 attempts.
- Tail (next level entered, never converted): **338 actions** (sb26 115 @L2 base 28, vc33 109 @L3 base 44, dc22 77 @L2 base 102, sp80 37) — worth 5.6–10.7 game-pts each if converted.
- No-effect actions: 175 (11%); concentrated in zeros (m0r0 16%, sk48 17%, g50t 38%).

### 2.2 Case studies (the mechanisms behind the numbers)

**r11l — 6,000s livelock (75% of its window).** 7 actions in the first 1,902s (cleared L1 in 5 — the game is easy), then **20+ consecutive idle turns**: each generated a 35–45k-char transcript with *no assistant output and no tool call* — near-identical lengths (44934, 44925, 44922 twice each) ⇒ deterministic regeneration of the same doomed turn until a final read-timeout. L2 (worth 9.5 pts, baseline 33) never attempted. This one bug cost more than most games score.

**sp80 — the retry-accumulation trap.** 15 attempts on L1 (visible move-budget → GAME_OVER loop). Attempt compositions show fresh exploration each time — no distilled replay plan after deaths; 35 RESETs sprinkled inside attempts. Outcome: level cleared, 99% of its value destroyed. dc22 (90 pre-clear vs 59 baseline), re86 (66 vs 26), ka59 (51 vs 28) are milder versions.

**sc25 — solvable-in-sandbox puzzle played by hand.** A 3×3 toggle panel (Lights-Out family; baseline 36 actions). The agent clicked interactively — including a long stretch re-clicking the same cell (`MOUSE(50,30)` ≥8 times in 20 steps) — for 114 actions and two timer game-overs. The toggle matrix was extractable from observed transitions and solvable exactly in the python sandbox; the harness never pushed it there.

**m0r0 — good science, no search.** 296 actions, 70 turns. The transcripts show genuinely competent exploration (segmentation, blocked-move tables from the `transitions` API, world-model text). But it exhausted the step budget mid-exploration, and post-reset it *re-probed* rather than running an offline search over the transition model it had already built.

**ft09 / lp85 — what winning looks like.** First clear in 5–6 actions (~2 turns), batch 8–14 actions/turn, single action type (clicks), 2–4% no-effect. When the model has the mechanic, ~20 acting turns buy 3–4 levels. Even here the leak shows: ft09's L3 cost 63 vs baseline 23 (-10.2 pts) — new-layout learning charged to the scoring bucket.

**Context/runtime facts visible in every transcript:** `context_budget_tokens: 31744`, `history_messages: 5–23` (aggressive rolling trim — cross-turn state survives only as a re-emitted "world model" text blob), `yield_seconds: 60`, `tool_output_tokens: 1024`, 13 request-error turns clustered at run end.

## 3. Synthesis: ranked harness changes

Enabler chain: **kill idle turns → acting turns ~2× → spend them as plan-then-batch → actions land in the right buckets → depth**. Budget check: 2 levels/game at baseline is ~83 actions (median); at batch ~8 that is 10–25 acting turns — feasible inside today's 28-turn window *only if* the idle half is recovered.

**#1 — Turn-pipeline repair: no discarded thinking, livelock breaker.** *(highest confidence, throughput ×~2)*
48% of turns produce nothing; the 60s `turn_time_budget` yield throws away the whole turn's generation, and identical context + near-deterministic decode regenerates the same failure (r11l: 20 turns/6,000s; g50t 36/53 idle; m0r0 38/70). Changes: (a) persist/resume truncated turns instead of restarting from scratch; (b) detect N consecutive no-tool-call turns → perturb (sampling bump, simplified re-prompt, or forced cheap probe action); (c) treat a repeated-transcript-hash as a hard alarm. Evidence: idle table §2.1, r11l case. Score delta: recovers turn-starved games (tn36 44 acts, sk48 50, ls20 56, r11l) and feeds every other change; direct value ≈ **+0.5–1.5 local** (r11l's L2 alone is +0.38 mean), enabling value much larger. Confidence: mechanism certain; conversion estimate medium.

**#2 — First-attempt discipline: bounded probes, distilled replay, never grind a bucket.** *(+~1.0–1.4 local, high confidence on the mechanism)*
Score ∝ 1/a² per bucket makes retries near-fatal. Harness rules: track bucket-actions vs the level's plausible baseline; separate "probe phase" (minimal actions, maximal python-side analysis of `transitions`) from "execute phase" (short batch from a solved plan); after any GAME_OVER, require an explicit distilled plan before acting (sp80's 15 unlearned attempts are the anti-pattern); stop grinding a level once the bucket exceeds ~2× baseline — the marginal score there is ~0 and the turns are worth more on the next game/level. Evidence: §1.3 efficiency table (35.6 pts lost), sp80/dc22/re86/ka59 traces. Recoverable ≈ +1.0–1.4 local (part of the 1.42 is unavoidable new-level learning cost).

**#3 — Convert the tails: model-first batching + offline solving in the sandbox.** *(the depth lever; +1.5–2.5 local plausible, medium confidence)*
338 traced actions sat on next levels that never cleared (sb26 L2, vc33 L3, dc22 L2, sp80 L2 — 5.6–10.7 pts each), and sc25-class games are exactly solvable offline. Push the loop the winners already use: build a transition model in python, search/solve there (BFS over observed dynamics, linear-algebra for toggle puzzles), then emit one verified batch. Batching without a model is the g50t/sp80 failure mode (batch 16.8/6.2, score 0/0.07) — the change is *plan-then-batch*, not batch-harder. Evidence: batch column §2.1, sc25/m0r0/sb26 transcripts. This is where the P3/P4 band (LB 7) actually lives; converting only the 4 observed near-miss tails is +1.3 local.

**#4 — Global budget allocation across the box (scheduler, not per-game loop).** *(+0.5–1.0 local, medium-low confidence)*
All 25 games got identical 7,920s regardless of trajectory. Marginal values differ 4×: a near-clear L2/L3 tail is worth 5.6–14.3 pts while a 60-turn-stuck zero's L1 is worth 2.8–4.8 with low success probability (m0r0 had 70 turns and still lacked the mechanic). At eval (28 streams / 110 games), abandoning stuck games early frees decode throughput for games with live tails — raising everyone's tok/s. Rule sketch: after K acting turns with no level and no world-model progress, park the game; revisit only if slack remains. Evidence: turn economics + marginal-value table §1.4. Cannot fully establish: whether freed capacity converts (depends on #3).

**#5 — Serving robustness.** *(small but free)*
13 read-timeout turns, clustered at run end (contention/shutdown); each costs a turn and sometimes strands a game (m0r0, sc25, r11l all end on `request_error`). Retry-with-backoff on client timeouts, and don't let one timeout end acting for a game with wall time left.

**Flagged as NOT establishable from this evidence:**
- **Thinking-token caps.** Slow-turn games (ar25 12.8k tok/turn, ls20 9.9k, tn36 8.1k) score poorly, but hard games *induce* long thinking — causation unproven here, and a cap risks the comprehension that wins ft09/lp85. Needs an A/B.
- **Zeros→L1 by throughput alone.** m0r0/sc25/g50t had 53–70 turns each; their wall is comprehension. #1/#3 give them more and better-aimed turns, but +1.11 (the full zeros conversion) should not be booked — book ~half.
- **Local→LB transfer of any specific delta.** The 0.49–0.59 ratio is assumed stable; hidden-set difficulty mix could compress gains. Every projection above is local unless divided by ~1.8.

### Bottom line

7.5 LB is not a tuning distance from here — it is **~2 levels/game at ≥baseline efficiency** (local-equivalent 12–15 vs our 4.5). The run's own data says the raw resources already suffice (actions emitted > actions needed; turns needed < turns available *if* the idle 48% is recovered); what's missing is a harness that (1) never throws a turn away, (2) never lets a scoring bucket absorb exploration, and (3) solves in the sandbox and spends actions only to execute.
