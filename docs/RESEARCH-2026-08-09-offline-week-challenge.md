# RESEARCH - 2026-08-09 - Offline Week Challenge (Multi-Agent Verification)

Conducted by the main agent + 3 subagents (codebase audit, external intel, trace mining) on branch `winning/duck-patched`. Every claim below was re-verified live or in code by this session, not taken from memory.

## 1. Ground truth re-verified this session

### Leaderboard (Kaggle API live, 2026-08-09)
- **#1 YUTO KOJIMA 1.86** (2026-08-09), then 1.69 / 1.65 / 1.64; ~55 teams at >=1.50.
- **Our row: "Ahmed Mobasher" 1.300, 2026-08-09 — exact rank 196** (ties at 1.29-1.31 span ~rows 197-209).
- "113" has two readings: (a) headline score 1.13 — already exceeded by individual draws (1.14/1.27/1.30) but the *config mean* is still below it; (b) rank 113 — requires **score >=1.58**. Both readings point at the same strategy: raise config mean, then let max-selection work.

### Our live draws (n=9, byte-identical base v2 config)
Sequence: 0.92, 1.14, 0.82, 0.75, 0.96, 0.88, 1.27, 0.69, 1.30.
**mean = 0.970, sd = 0.220** (older docs quote "n=8 mean 0.929" — stale by one draw).
- E[max-of-2 finals] ≈ **1.09**; E[max-of-90 remaining draws] ≈ **1.51**. Pure farming cannot reach 1.58/1.86.

### Kaggle rules (rules primary text, not hearsay)
- "You may submit a maximum of 1 Submission per day" — the nightly cadence is correct.
- "You may select up to two (2) Final Submissions for judging" — best-of-2 doctrine (campaign's daily-max assumption) is valid.

### Scoring formula (verified in code, not assumed)
`scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src/taaf/game.py:355-371`:
- per-level score = min(1.15, (baseline / total_actions)²) × weight, where **weight = level_idx + 1** (L1 scores 1x, L5 scores 5x).
- `total_actions` **sums across ALL attempts on that level** (line 362) — retries pollute the efficiency of the eventual completion.
- Completed levels **never re-bank** (line 360-361); `level_reset` restores the level but does not re-score.
- `baseline[level_idx]` = human median actions per level — the hidden per-level numbers.

### Human baselines (mined from the 340 in-repo public human replays, 25 games)
Median-of-medians across games: **L1 = 30** (range 13-78), **L2 = 84** (34-228), **L3 = 152** (70-362), **L4 = 218** (94-508), **L5 = 302** (143-869). Full per-game table appended.
- Implications: our agent completes L1 in ~60 actions -> efficiency (30/60)² = 0.25; the 1.15 cap is *unreachable* at our action counts; **every saved action yields quadratic score** on that level.

## 2. Validated claims vs. reality (skepticism ledger)

| Recorded claim | Verdict (evidence) |
|---|---|
| Base mean 0.929 (n=8) | Outdated — n=9 -> mean 0.970, sd 0.220 |
| 35B gate NO_GO (protocol 143/156=91.7% <95%, 0/4 target unlocks) | Confirmed via logs; Track B (35B/AgentWorld) exhaustively dead |
| AgentWorld NO_GO (no serve path) | Confirmed |
| Patched-family draws (v7 0.80/0.93) below base | Confirmed (0.80, 0.93 vs 0.970 mean) |
| Wipe on game_over / level_transition destroys world model (tool_agent.py:1113-1126) | **Confirmed in code — the load-bearing defect** |
| Model only ever sees frame[-1]; animation unreachable (game.py:170) | Confirmed |
| Token-estimator ~1.44x overcount; trims context to ~31.7k | Confirmed |
| "Actions on never-completed levels bill nothing" | Confirmed — but their actions still pollute the level's accumulated total if the level later completes (game.py:362) |
| Agent never RESETs | Confirmed (0 RESETs / 20 transcripts; humans rollback-replay 35.8%) |
| Grinder "dormant by construction" | **Superseded** — age-triggered grinder exists now (TAAF_GRAPH_GRIND_AGE_TURNS=10) and engages |
| Watchdog default 900s | Superseded — 600s is the shipped default; 900s only in the regression arm |
| ACTION7-injection / "6 broken games" | Superseded — patch_action7 shipped in scored duck; applies only to the Franken base |
| "A-not-B brake" / stuck-RESET built and live | **False — designed+measured GO, never shipped to a live submission** |

## 3. Behavioral evidence mined from traces (Agent C)

- 38/48 real transcripts stuck on L1; 34 of those with >40% identical-consecutive actions.
- Identical-repeat fraction: 32.6% (r2), 58.8% (r3) of all events.
- 0 RESETs anywhere. Level-transition wipes occur exactly where strategies became successful.
- Struct plan-channel adoption replicated 2-3x (2.01 / 2.59 / 2.57) but converted into 0 level-delivery — *adoption is real, action-dilution suppresses yield*.

## 4. Mechanism portfolio (ranked by (impact x confidence) / build-cost)

1. **Stuck-loop brake + forced RESET** (A-not-B brake already measured safe on masked games, designed, never shipped): the single largest quantified pathology (38/48 stuck; ~40-59% repeat waste) and the only mechanism that *preserves* per-level efficiency (actions sum across attempts).
2. **Budget HUD / 5x cap visibility**: the model is blind to baselines (~30/84/152/218/302) and the 5x cap; giving it per-level budget in the observation converts the quadratic efficiency from accident to design.
3. **Persistent action-log REPL memory** (~1 file per game, append-only): defeats the verified wipe defect at engine level; model re-reads its own history on demand instead of losing it every game_over/level_transition.
4. **Diff-channel**: surface "what changed this turn" (patch-15 has the `last_animation` hook) — cheap, unmeasured baseline.
5. **Vision slot**: grid-as-ASCII costs ~8k tokens/turn vs ~1.6k as an image; Qwen VL weight offered for the 27B class — unexplored in this campaign.
6. **Struct plan-channel pushed to live** (adoption 2.5x behavior is replicated; never live): direct wall-clock depth gain per deliberation.
Rejected permanently: 35B route (protocol blocked), AgentWorld delta (no serve path), prefix re-player (no re-banking).

## 5. Recommended next moves (changing the current plan)

1. **Tonight (00:01 UTC 08-10)**: keep the byte-identical base draw ONLY if no candidate passes gate; otherwise ship the highest-EV gated arm (stuck-brake + budget-HUD) — one live discriminating slot, pre-registered fingerprint in kernel trace.
2. **Offline first**: 28-games @ 7920s eval-geometry A/B: base vs {stuck-brake + RESET} vs {+budget-HUD} vs {+persistent-log} — read "levels ex-ft09" per wave discipline.
3. **Kill**: 35B re-push staging, 08-09 dict amendment, adoption-only quick wins (struct plateau re-verified).
4. **Keep**: nightly base farming as the measurement arm; best-of-2 finals doctrine.
5. E[final] under the plan = 1.09; under "ship the top-2 brakes offline-based" = unknown but the only route with any evidence of lift.

## Appendix: human per-level medians by game (actions, session-level)

```
game    n   WR  medSessA  L1   L2   L3   L4   L5
ar25   10   90%     861   32   71  155  181  336
bp35   14   79%     599   21   77  123  175  210
cd82   11   91%     192   53   60  100  123  147
cn04   12   92%     620   29   97  155  508  733
dc22   11   91%     600   56  153  237  334  638
ft09   10   80%     122   39   56   79  133  161
g50t   12   92%     475   78  228  362  332  419
ka59   10   70%     290   28  125  176  227  259
lf52   11  100%    1213   32  116  179  248  478
lp85   54   74%     227   18   55   93  113  143
ls20   13   85%     595   22  144  236  307  409
m0r0   11  100%     995   30  156  359  408  858
r11l   10  100%     239   17   52  101  123  179
re86   11  100%    1073   26   66  152  291  486
s5i5   11  100%     610   20   98  179  296  433
sb26   12   92%     183   18   55   70   94  160
sc25   15   87%     356   36   44   76  145  302
sk48   14  100%     879   53  220  312  352  573
sp80   12   92%     272   39   84   87  218  337
su15   13  100%     218   22   53   92  240  265
tn36   14   71%     300   30  100  140  168  246
tr87   12   92%     444   54  135  167  206  283
tu93   13  100%     441   19   39   77  120  221
vc33   10   90%     432   13   34  125  157  245
wa30   14   93%    1592   71  183  342  479  869
```
Median-of-medians: L1=30, L2=84, L3=152, L4=218, L5=302.
---

# APPENDIX (2026-08-09, second session) — "Beat 1.86" deep research

Goal restated by the principal: score HIGHER than 1.86 (current #1, YUTO KOJIMA). Five-agent team
(objective-hacker, world-intel, trace-miner, test-engineer, human-strategist) + main-session calibration.

## A1. The exact objective (verified in code + empirical micro-tests)

From `reference/arc-agi-toolkit/arc_agi/scorecard.py` (146-241) and engine micro-tests on synthetic games:
- per-level: `min(115, 100·(baseline_l / actions_l)²)`; absent completion -> 0.
- per-run: weighted mean by level index, capped by completion share: `min(Σ(ls·w)/W, 100·Σ(w_completed)/W)`.
- per-game: **max over PLAYS** (fresh full-reset runs get zeroed counters; a better replay replaces the score).
- LB = mean of per-game scores over the eval games (hidden set ≈ 55 games, ~50% of test data).
- RESET mid-level = +1 action into the SAME level bucket (scorecard.py:701); full reset (at WIN or action_count 0) opens a NEW play.
- Competition mode: game restarts forbidden (they become level resets), 5x-baseline hard cutoff per run, ~9h budget.
- Baselines: per-(game,level) official list, live in `environment_files/<game>/*/metadata.json`.

## A2. Calibration experiment (this session, real corpus, official formula)

Parsed 47 real episodes across all 25 public games from `scratchpad/rl_gate/episodes/*/artifacts/viewer_data_events.jsonl`,
applied the official scorecard with official baselines, max over episodes per game:
- public mean-of-max = **11.78** (SB26 fully solved = 100.0; AR25 41.7; 6 games = 0.00).
- live LB 1.30 on hidden -> **implied hidden/public difficulty multiplier = 0.110** (hidden ~9x harder than public).
- zero-games (6/25=24%) burned 100-600 actions with zero completions (dc22 600!, m0r0 239, tn36 300, sk48 100, tr87 97).

Counterfactual "rescue the last-attempted (near-miss) level of every game":
- at capped efficiency (agent's own observed sub-baseline economy: L1 18.5 vs b~27, L2 19 vs 47, L3 39, L4 22, L5 37.5): **+1.06 -> LB 2.36 (beats 1.86)**
- at 1.5x baseline: +0.41 -> 1.71; at 2x baseline: +0.23 -> 1.53. (Efficiency on the rescued level is the swing factor.)

## A3. The crack strategy (ranked, with evidence)

1. **Finish-the-level persistence + stuck-brake + RESET reorientation** (converts CF1): 60.1% of actions sit in
   never-completed levels; 30/35 runs hit the wall on the near-miss level; humans reset 20x/game, agent 0.7%.
2. **Efficiency rails**: agent already completes at sub-baseline (capped); a brake BEFORE the bucket pollutes
   keeps CF1 in the optimistic band (50-RESET padding measured: L2 score 14.1 -> 1.1).
3. **Context retention + compaction** (external evidence, OpenAI: same model 13.3% -> 38.3% public; Tufa's own
   stated #1 gap = context compaction): lifts "attempted -> completed", i.e. CF1 fuel. Persistent-log REPL memory
   (mechanism from section 4) does this at engine level.
4. **Base model / multimodality** (Tufa: "better base models and multimodality were the main score drivers";
   GPU = RTX Pro 6000 96GB, vLLM in-kernel): vision slot = ~5x token efficiency (2.1-2.5k real tokens/frame ascii
   vs ~1.6k as image) + spatial comprehension. Untested in this campaign.
5. **Mechanical family dispatcher** (human strategist: click-first openers 46%, move-runs 2.2 mean length, A5-finish
   in 12/75 fastest, tu93 = literal 19-action script by 3 humans): saves early-level waste AND survives the
   5x-baseline run-cut; hidden games must not be overfit on public, tag-driven transfer only.
6. **Best-of-N submission selection** (rules: 2 finals; E[max] ≈ +0.1-0.3 per the draw distribution 0.69-1.30).
7. **Passes structure**: 3 passes exist in the harness; each extra full pass at ~game scale is +EV via max-over-plays
   (SB26 went 100 on one of two episodes; geodesic_postpass live-validated +38% dev).

## A4. Risks (honest)
- Kojima's technique is unknown (no writeup); the #1 moves daily.
- LB = ~50% of test data; final standings shift on the other half.
- CF1's optimistic band assumes near-miss levels complete at the agent's OWN observed economy — the bottleneck is
  completion, and every mechanism above targets exactly that.
