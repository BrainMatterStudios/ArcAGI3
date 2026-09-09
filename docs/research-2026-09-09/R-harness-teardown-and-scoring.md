# TEARDOWN 2026-09-09 — the scoring rule, the constraint chain, and what is still open

Brief (visual): published artifact "The Depth Ledger".
Written after A1/A2 died and A3/A4 were gated out, i.e. with every incremental lane closed.

## 1. The scoring rule, read from source (not memory)

`arc_agi/scorecard.py:146-207` (`EnvironmentScoreCalculator`), call site `:474-490`:

```
s_i    = min(115, 100 * (baseline_i / actions_i)**2)   if level i cleared else 0
score  = min( sum(s_i * i) / sum(i) ,  100 * sum_{cleared}(i) / sum(i) )
```

`for level_idx in range(len(env_info.baseline_actions))` and `level_index=level_idx+1`, so
**i runs 1..N over EVERY level of the game**, including levels never reached. The denominator is
fixed at N(N+1)/2. Public games: N = 6..10, mean 7.32.

VERIFIED: `tn36`, 2 of 7 cleared -> 100*(1+2)/28 = 10.714285…; the harness recorded
10.714285714285714. Formula reproduces exactly.

Three consequences:
1. **Weights are 1,2,3..N** -> the marginal value of the k-th level is proportional to k. The score
   is CONVEX in depth. Mean over the 25 public games: k=1 -> 3.5, k=2 -> 10.6, k=3 -> 21.1,
   k=4 -> 35.2, k=5 -> 52.9, k=6 -> 74.0.
2. **The completion-share term is a ceiling and it BINDS for us** (65% of our cleared levels come in
   at/above the human baseline, so per-level scores cap). Our score is essentially
   `100 * weighted fraction of levels cleared`. Efficiency ABOVE parity is worth exactly zero.
3. **But inefficiency is punished quadratically**: a level cleared at 3x baseline contributes 11% of
   its weight, at 10x it contributes 1%. So you cannot brute-force explore a level you intend to
   clear. This is the constraint that kills most "just try more things" designs.

Engine takes MAX over plays (`geodesic_postpass.py:252`); framework reports the mean
(`diagnostics.py:602-619`). `n_passes = 1` live.

## 2. Where we are (275 game-runs pooled, 11 waves, live geometry)

| levels | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| games | 47 | 130 | 51 | 26 | 15 | 6 |
| share | 17.1% | 47.3% | 18.5% | 9.5% | 5.5% | 2.2% |

mean 1.46 lv/game, mean score 7.10 (public-25 scale). Loss decomposition: **~93% depth, ~7%
inefficiency**. Failure is per-game and REPRODUCIBLE: 7 of 25 games never exceeded 1 level in ANY of
11 runs; ft09 averages 3.55; the top 6 games carry ~65% of the wave score.

Converting scales, the leaderboard leader at 11.04 is clearing roughly 2 levels per hidden game where
we clear 1. **The gap to the top is about one level of depth, not a different paradigm.**

## 3. The constraint chain (all measured)

| step | value |
|---|---|
| GPUs | 1 |
| vLLM sequences actually running (5 GiB KV) | 3.0 running / 20 waiting |
| server throughput | ~0.17 requests/s |
| end-to-end per call | 139.9 s — **queue 120.9 s (87%)**, inference 17.8 s, prefill 1.6 s |
| model calls per game (7,920 s) | ~56 |
| actions per call (44% of calls fire none) | 2.76 |
| actions per game | 154 |
| levels | 1.46 |

Baseline actions needed: k=2 -> 102, k=3 -> 170, **k=4 -> 248**, k=5 -> 383. At 2.76 actions/call,
level 4 needs ~86 calls and we have ~56.

**The reframe: this is a QUEUEING budget, not a compute budget.** The first call of a game, into an
empty queue, returns in 11.6 s vs 139.9 s pooled — a 12x penalty from 28 games contending for 3 slots.

## 4. Where the budget goes

- **59.1% of wall, 53.8% of calls, 62.5% of actions are spent AFTER the last level a game will ever
  clear.** Median last clear at 43% of the clock; the final 28% of the clock yields 14% of all clears.
- 17.1% of games clear zero levels and consume their full clock.
- 44% of calls fire no action (43% of model wall time). NOTE the counter-evidence: the pre-registered
  experiment that forced acting (keith_probe) read 41 vs base 39.33 = dead. Not convertible by cadence.
- **Prefix caching is OFF** and must be — MTP + APC corrupts generations on vLLM 0.19 (fix PR #47861
  is post-0.19). Measured hit rate when enabled was 41.3%. Consequence: 95.5% of the 28M prompt tokens
  per wave is text the server already saw in that run; prefill:decode volume is 13:1.
- 55% of actions were spent inside a life that ended in GAME_OVER (23 of 37 GOs on level 2).
- 271/275 runs end `gave_up` at 98% of clock. Scheduler idle is ~0 (1.8%, tail only).

**Counter-finding that frames all of it:** on the level where the run dies, the median game has spent
~1.0x the human action baseline and **48% of games have spent LESS than a human needs**. The agent is
not flailing on the wall. It never gets enough attempts at the wall.

## 5. NEW NEGATIVE RESULT — verified models do not transfer across levels

`docs/research-2026-09-09/cross_level_transfer.py` (reproducible). Four backtest-green Stage-1 models,
cross-tested against the neighbouring level of the SAME game, teacher-forced:

| model | own level | other level |
|---|---|---|
| dc22L1 draw1 | 20/20 100% | 2/20 **10%** |
| dc22L1 draw2 | 20/20 100% | 2/20 **10%** |
| dc22L2 draw1 | 20/20 100% | 0/20 **0%** |
| dc22L2 draw2 | 20/20 100% | 9/20 **45%** |

Reading the sources shows why: they encode literal row/column rectangles of that level's board, not
the game's mechanic. **"Backtest-green" is weaker evidence of understanding than it looked**, and the
cost of a verified model never amortises across a game's levels. This independently confirms the
Polyphony cost gate and closes the last escape route for the executable-model lane at our budget.

## 6. Claims I checked and REFUTED

- *"The token estimator gives away ~8k of the 32k window."* No: measured max prompt is 31,304 against
  a 31,744 budget = 98.6% fill. The estimator is accurate enough in practice.
- *"Front-of-window eviction destroys the KV prefix cache."* Moot — there is no prefix cache
  (`--no-enable-prefix-caching`, forced by MTP).
- *"There is free session time."* No: the notebook budget is pinned at `max_runtime_s = 32400` and
  110 games x 7,920 s / 28 concurrent already fills it.

## 7. What is still open, ranked

1. **DOES BUDGET CONVERT TO LEVELS? Never measured.** Every wave runs 28 games at 7,920 s and 271/275
   ended on the clock, so every game was still going when we stopped it. Experiment: same 25 games at
   **concurrency 9** instead of 28 — same server, same clock, ~3x the calls per game. ~6 h rig, ~$25,
   no submission slot. Pre-register: >= 2.5 lv/game -> budget converts, allocation is the campaign;
   ~1.5 -> comprehension is the wall and only #5 matters. **Nothing else should be built first.**
2. **Adaptive allocation.** There is none in the harness (one scalar, one clock check). Triage signal
   validated: clearing L1 in <30 actions -> 2.1 lv mean and 31-38% reach 3+; >30 actions -> 1.2 lv and
   ~0% reach 3+ (2% and 0% in the slower bins). Because the constraint is global throughput, killing a
   doomed game really does hand its share to survivors; ~30-40% recoverable. Needs out-of-sample check.
3. **Buy queue slots by shrinking the prompt.** Running concurrency = KV / sequence length. Honest
   counterweight: doubling KV directly was tried, moved 3.2 -> 6.4x, and bought **+0.28 levels only**,
   so the GPU is nearer saturation than 87%-queue suggests. Untested direction: whether REMOVING
   context costs anything (5 replications say adding knowledge buys nothing). Cheap version: the
   system prompt is 3,170 tokens re-sent 56x and the user prompt is 92% byte-identical turn to turn.
4. **Burn level 1 as a laboratory.** L1 carries weight 1 of a 21-55 denominator; sacrificing it costs
   2.8-4.8 pts, clearing L2-L4 efficiently is worth 14-43. The quadratic penalty is per level and
   never retroactive, so this is the one strategy it does not block. Caveat: needs mechanic knowledge
   to transfer, and §5 says coordinate models transfer at 0-45%. Note the harness deliberately blanks
   `world_model`/`goal_model`/`action_model`/`recent_findings`/`open_questions`/`current_plan` at every
   level transition (`tool_agent.py:1113-1126`), keeping only `cross_level_notes` — so the carrying
   version has never been observed.
5. **Change the brain.** If #1 reads flat, this is the residual. Every behavioural, memory, budget and
   tooling lever is now dead on this model — the signature of a capability ceiling, not a harness
   defect. Sub-branches: a stronger open checkpoint at this throughput, and test-time training, where
   the serving gate is passed but **no fine-tune has ever been served AND scored** — the one major axis
   opened and never closed.
