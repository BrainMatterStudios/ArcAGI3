# Score arithmetic: what 7.0 and 11.0 require, and where the lift must come from

Date 2026-09-08. Inputs: the three best 25-game Modal waves (75 game rows), the official scorer, the frozen harness scorer, and the live base-family draws. All scripts and intermediate JSON are in this directory (`analysis.py`, `part2.py`, `part3.py`, `part4.py`, `results_arith.json`, `time_split.json`). Nothing in the repo was modified.

## 0. Formula, confirmed against code

Official (`reference/arc-agi-toolkit/arc_agi/scorecard.py`):

- per level: `score = min(115, 100 * (baseline_actions / actions_taken)^2)` if completed, else 0 — `scorecard.py:170-171`
- weight = `level_index`, which the caller passes as `level_idx + 1` — `scorecard.py:486-491`; so L1 weight 1, L2 weight 2, ..., Ln weight n
- per game: `score = sum(score_l * w_l) / sum(w_l)`, then `score = min(score, max_weights/total_weights*100)` where `max_weights` sums only levels with score > 0 — `scorecard.py:196-206` (the completion-share cap: the 115 bonus can never push a game above the weight share of the levels it completed)
- per game = max over runs — `scorecard.py:239-241`; leaderboard = mean over games — `scorecard.py:613-616`

Frozen harness scorer (what `results.json`/`score.json` in the waves contain): `scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src/taaf/game.py:381-415` (`GameRun._compute_final_score`) is a line-for-line mirror of the above (docstring says so; weights `level_idx + 1`, cap 115, completion-share cap). `inference/tools/eval.py:298-320` (`_framework_score`) reads `run.final_score` and `evaluate_runs` averages it.

The runner's `live_cap_score` (`offkaggle/run_regime_wave.py:236-252`, `LIVE_CAP = 1.15`) is the same sum WITHOUT the completion-share cap. It overstates the official score whenever a game has levels at the 115 cap (8/25 games in the probe wave; 7.92 vs official 7.44 there).

Verification: I recomputed the official formula from `baselines` + `actions_per_level` + `levels_completed` for all 75 rows and matched `score.json` exactly (max |diff| = 0.00 in all three waves; `analysis.py`). The memory files (`arcagi3-eval-noops-and-scoring-cap.md` section 2, `arcagi3-scoring-and-transfer.md`) state the same formula.

Score scale: a per-game score is 0-100; the leaderboard is the mean over ~110 hidden games. "Local" below = mean over the public 25 under this formula.

## 1. The three waves

| wave | levels | lv/game | zero-level games | local mean (official) | mean per-level eff pts (of 115) | median actions/baseline on completed levels | completed levels at the 115 cap |
|---|---|---|---|---|---|---|---|
| probe 09-08 (`20260908T0752-keith_probe`) | 41 | 1.64 | 4 | 7.44 | 83.2 | 0.86 | 22/41 |
| kv10 09-03 (`20260903T0553-kv10-keith`) | 43 | 1.72 | 6 | 10.47 | 83.6 | 0.96 | 20/43 |
| yield900 09-07 (`20260906T2251-keith_yield900`) | 41 | 1.64 | 1 | 8.46 | 93.6 | 0.72 | 27/41 |
| pooled 75 rows | 125 | 1.67 | 11 | 8.79 | 86.8 | 0.86 | 69/125 |

kv10's 10.47 is one game: ft09 won 6/6 (100 pts). Same 41-43 levels otherwise.

Calibration to the live leaderboard. Live draws of this config family on the hidden 110: 3.25, 2.58 (keith byte-copy base), 4.31, 2.45 (yield900) → mean 3.15, sd 0.85. Local means of the two KV5 waves (the flown regime): 7.44 and 8.46 → 7.95.

| calibration | ratio live/local |
|---|---|
| KV5 waves (7.95) vs base-family live mean (3.15) | **0.40** |
| all three waves (8.79) vs 3.15 | 0.36 |
| keith V14 commit run 6.76 local vs its first live draw 3.25 | 0.48 |

I use 0.40 as the central ratio, range 0.36-0.48. Because the 25 public games are inside the 110, the ratio implies the 85 unseen games average about **1.7** per game (= (110 x 3.15 - 25 x 7.95)/85) if the Kaggle run scores the public 25 the way the rig does; if Kaggle also scores the public 25 lower, the unseen games are somewhat better than 1.7 and the public 25 somewhat worse. Either way the hidden set is where we lose. Draw noise is large on both sides (local draws of one regime range 6.4-10.5; live sd 0.85), so the ratio is good to roughly +-30%.

Per-game table for the probe wave (weights, efficiency, and where the time went): see `probe_table.md`; excerpt:

| game | n | L | score | share of weight completed | actions/baseline per completed level | wall-level actions / its baseline | % of budget on the never-completed level |
|---|---|---|---|---|---|---|---|
| r11l | 6 | 3 | 28.57 | 28.6 | [0.32, 0.39, 0.47] | 0/26 | 12% |
| re86 | 8 | 4 | 27.78 | 27.8 | [0.81, 0.86, 0.58, 0.57] | 250/189 | 52% |
| ft09 | 6 | 4 | 22.19 | 47.6 | [0.58, 1.0, 2.35, 2.04] | 0/65 | 5% |
| vc33 | 7 | 3 | 21.18 | 21.4 | [1.43, 0.61, 0.98] | 99/61 | 76% |
| m0r0 | 6 | 1 | 4.76 | 4.8 | [0.7] | 161/111 | 92% |
| sb26 | 8 | 1 | 2.78 | 2.8 | [0.61] | 144/28 | 95% |
| cn04 | 6 | 1 | 0.49 | 4.8 | [3.1] | 11/54 | 32% |
| tn36 / sp80 / g50t / tr87 | 6-7 | 0 | 0 | 0 | [] | 99/32, 310/39, 160/78, 155/54 | 100% |

## 2. Decomposition

Where the points come from (uncapped per-level contributions, pooled 75 rows):

| level | share of our score | points per game |
|---|---|---|
| L1 | 27.6% | 2.43 |
| L2 | 31.6% | 2.78 |
| L3+ | 47.7% | 4.19 |

(probe 30/42/34%; kv10 18/28/61% because of the ft09 full win; yield900 37/28/43%.) Half the score already comes from L3+ even though only 31% of rows reach L3. Depth weighting works as designed.

Where the points are lost (out of 100 per game, pooled):

| loss | points | share of loss |
|---|---|---|
| levels not reached (100 minus the weight share of completed levels) | **89.3** | 97.9% |
| efficiency inside completed levels (weight share minus actual score) | 1.9 | 2.1% |

Per wave: depth loss 90.0 / 87.6 / 90.3; efficiency loss 2.56 / 1.89 / 1.28. Completed levels are already at or near the cap (median 0.72-0.96 actions per baseline action, 55% of completed levels AT 115). Only 15 of 125 completed levels score under 25 points (>2x baseline), almost all L1s where the agent was still learning the mechanic (cn04 3.1x, bp35 2.4x, ls20 1.9x, ka59 1.7x, sc25-L1 4.0x).

## 3. Counterfactuals on the 25 games (pooled 75 rows; LB-eq = local x 0.40, range x0.36-0.48)

| counterfactual | local mean | delta local | LB-equivalent |
|---|---|---|---|
| base | 8.79 | - | 3.5 |
| (a) every game +1 level at its current efficiency (zero-level games at the wave-mean efficiency) | 15.68 | +6.9 | 6.3 |
| (b) current levels at perfect efficiency (actions = baseline, 100 pts/level) | 10.70 | +1.9 | 4.3 |
| (c) every zero-level game reaches L1 (wave-mean efficiency) | 9.26 | +0.5 | 3.7 |
| (d) levels doubled at current efficiency (0 stays 0) | 25.28 | +16.5 | 10.1 |
| (d') levels doubled, zero-level games to L1 | 25.76 | +17.0 | 10.3 |
| (e) every level completed at current efficiency | 77.67 | +68.9 | 31.1 |
| (f) current levels at 2x baseline actions (25 pts/level) | 2.67 | -6.1 | 1.1 |
| (g) +1 level everywhere at perfect efficiency | 17.67 | +8.9 | 7.1 |

Per wave (a)/(b): probe 14.37/10.00, kv10 16.38/12.36, yield900 16.25/9.73.

Reading:
- One more level per game is worth 3.6x the ENTIRE remaining efficiency headroom (+6.9 vs +1.9). The efficiency term is saturated: it can add at most ~2 points local (~0.8 LB) and nothing more, ever, on the levels we already complete.
- Efficiency is a multiplier to protect, not a lever to pull: dropping to 2x baseline on completed levels multiplies the score by 0.25-0.30 (8.79 -> 2.67). A depth gain bought with sloppy play is mostly cancelled.
- Fixing the zero-level games alone is nearly worthless (+0.5): their L1 weights are 1/21-1/28 of the game. Zero-level games matter only as the entry ticket to L2/L3.
- Because the weight of level k is k, the marginal level is worth more the deeper it is: for a 7-level game L1 = 3.6%, L2 = 7.1%, L3 = 10.7%, L4 = 14.3% of the game. Going from 1 to 2 levels roughly triples a game's score; 2 to 3 doubles it again.

So the one dimension that moves the score per unit of difficulty is depth on games already partially understood: +1 level ≈ +6.9 local; the same +6.9 via efficiency is impossible (cap), and via L1-on-zero-games would need ~14 zero-level games we do not have.

## 4. What a 7.0 and an 11.0 profile look like

Two ways to frame it, both on the hidden 110-game mean.

(i) Via the calibration ratio, in local-25 units: 7.0 LB ≈ 17.5 local-equivalent (14.6-19.4 across the ratio range); 11.0 ≈ 27.5 (22.9-30.6). That is 2.0x and 3.1x our current 8.8. Against the counterfactual table: +1 level per game everywhere (15.7) does NOT reach the 7.0-equivalent; 7.0 ≈ "+1.3 levels per game at current efficiency" (about 3.0 lv/game on the public 25); 11.0 ≈ "levels doubled" (25.3-25.8; about 3.3-3.5 lv/game on the public 25).

(ii) Directly, as a per-game profile on the hidden set. Assumption: the hidden games have the public-25 level-count mix (n = 6-10, mean 7.3, so mean weight share of the first k levels = 3.5 / 10.6 / 21.1 / 35.2 / 52.9 % for k = 1..5). Score = efficiency factor x sum_k p_k x share_k.

| profile | lv/game | score at <= baseline actions | at 1.25x baseline (x0.64) | at 1.5x (x0.44) | at 2x (x0.25) |
|---|---|---|---|---|---|
| A: every game L1 only | 1.00 | 3.5 | 2.3 | 1.6 | 0.9 |
| B: 50% L1, 50% L2 | 1.50 | 7.1 | 4.5 | 3.1 | 1.8 |
| C: 25% zero, 25% L1, 30% L2, 20% L3 | 1.45 | 8.3 | 5.3 | 3.7 | 2.1 |
| D: every game L2 | 2.00 | 10.6 | 6.8 | 4.7 | 2.6 |
| E: 20% zero, 20% L1, 30% L2, 20% L3, 10% L4 | 1.80 | 11.6 | 7.4 | 5.2 | 2.9 |
| G: 15% zero, 15% L1, 25% L2, 25% L3, 20% L4 | 2.20 | 15.5 | 9.9 | 6.9 | 3.9 |
| F: every game L3 | 3.00 | 21.1 | 13.5 | 9.4 | 5.3 |
| ours, local 75-row mix (15% zero, 41% L1, 21% L2, 12% L3, 8% L4, 2% L5+) | 1.67 | 10.8 | 6.9 | 4.8 | 2.7 |

Our own local mix at our observed efficiency factor 0.87 gives 9.4 (actual 8.8; the difference is the completion cap and which levels are deep). Our live 3.15 corresponds to profile A/B territory on the hidden set.

Plausible 7.0 team: ~1.5-1.8 levels per game across the hidden set, completed levels at or near baseline actions (profile B at cap efficiency, or E at 1.25x baseline). Equivalently: what we do on the public 25, done on all 110 games. Zero-level games 20-25% is compatible with 7.0 if the rest reach L2-L3.

Plausible 11.0 team: ~2.0-2.2 levels per game at near-baseline efficiency (D at cap, or G at ~0.7 efficiency = 1.2x baseline), with 15-20% of games at L4+. An 11.0 team that plays at 1.5x baseline would need 3 levels on every game (F) — implausible, so the 11.0 teams are almost certainly near-baseline-efficient AND ~0.5 lv/game deeper than our public-25 play, on the full hidden set.

Assumptions to be explicit about: hidden level-count mix = public mix; hidden difficulty mix unknown (our implied 1.7 on the unseen 85 says the unseen games are much harder for THIS agent, not necessarily for others); per-game score = single run (runs = 1 at eval); efficiency factor applied uniformly across levels.

## 5. Time budget

Per-game budget 7920 s (`run_regime_wave.py` geometry, `max_runtime_s_per_game: 7920`). CADENCE lines: probe 51.9 calls/game, 3.04 actions/call, e2e 151 s/call; yield900 51.5 calls/game, 1.98 actions/call, 153 s/call; kv10 82.2 calls/game, 2.96 actions/call, 94 s/call (kv10 is not the live regime). At 150 s/call the budget is 52.8 calls/game.

Time on levels never completed (from the transcript step-header clock times and the events' step->level map; `part4.py`, `time_split.json`; fraction = time from the first analysis step at level L+1 to the end of the run):

| wave | mean fraction of 7920 s on the never-completed level | median | games with >= 1 level only | zero-level games (100%) | calls made at the wall level |
|---|---|---|---|---|---|
| probe | 55% | 54% | 47% | 4 | 53% |
| kv10 | 64% | 71% | 52% | 6 | 62% |
| yield900 | 62% | 66% | 61% | 1 | 55% |
| pooled | **60%** | 66% | | 11/75 | |

Mean time per completed level = 1882 s (~12.5 calls). Wall-level actions are median 1.0x that level's baseline (mean 2.3x, 51% of games spend more than the baseline on the wall) — the agent has "enough actions" at the wall and still fails; it is not action-starved, it is stuck.

Calls a human-baseline solve would need at 3 actions/call (baseline totals mean 685 actions/game, median 638, range 171-1843):

| scope | baseline actions (mean over 25) | calls at 3/call | vs 52.8 available |
|---|---|---|---|
| L1 | 35 | 11.6 (max 26) | 22% of budget |
| L1-L2 | 102 | 33.9 | 64% |
| L1-L3 | 170 | 56.6 | 107% (over) |
| L1-L4 | 248 | 82.5 | 156% |
| full game | 685 | **228** (median 213, min 57, max 614) | 4.3x the budget |

Cadence ceiling (perfect comprehension, actions = baseline, no wasted calls):

| calls x actions/call | actions/game | max lv/game | full solves | local score (all at 100) | LB-eq x0.40 |
|---|---|---|---|---|---|
| 52 x 1 | 52 | 0.92 | 0 | 3.7 | 1.5 |
| 52 x 3 (current) | 156 | 2.88 | 0 | 22.8 | 9.1 |
| 80 x 3 | 240 | 4.00 | 4 | 40.0 | 16.0 |
| 52 x 5 | 260 | 4.16 | 4 | 41.8 | 16.7 |
| 52 x 10 | 520 | 5.92 | 12 | 72.3 | 28.9 |

So under the current cadence (52 calls, 3 actions/call) the ceiling with PERFECT comprehension is 2.9 levels/game, 22.8 local, ~9 LB-equivalent; our actual 1.67 lv/game / 8.8 local is 39% of that ceiling. The 11.0 target (27.5 local-equivalent) is above the current-cadence ceiling: it needs both comprehension AND more actions per call (5+/call, or shorter calls) once comprehension is there. Today the binding term is comprehension (60% of the clock burned at a wall with baseline-level action budgets), not the clock; the clock becomes binding only after the walls fall.

## 6. Bottom line

1. Formula confirmed; our scorer and the official one agree to the last digit on 75 rows.
2. 98% of our lost points are levels not reached; the efficiency term has ~2 points (~0.8 LB) of headroom left in total. Efficiency must be protected (2x baseline = x0.25), not pursued.
3. Marginal value: +1 level/game ≈ +6.9 local ≈ +2.8 LB; L1 on the zero-level games ≈ +0.5 local; perfect efficiency ≈ +1.9 local.
4. 7.0 ≈ our public-25 play (1.5-1.8 lv/game at near-baseline efficiency) generalised to all 110 games; 11.0 ≈ 2.0-2.2 lv/game at near-baseline efficiency on all 110. The lift is depth on the 85 unseen games (implied ~1.7/game today), i.e. comprehension of new mechanics, then L2->L3 depth.
5. 60% of every game's clock is spent at the wall level with roughly baseline action counts; a full baseline solve would need 228 calls vs 53 available, and the current cadence caps even perfect play at ~2.9 lv/game (~9 LB-eq). Comprehension first; cadence (actions/call) becomes the next binding constraint past ~9.
