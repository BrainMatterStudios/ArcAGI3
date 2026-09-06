# loss-ledger-2 — working notes (2026-09-06)

Data: 5 runs x 25 public games = 125 game-runs of the keith V14 regime (Flash-Next NVFP4, stock June duck,
window 32768, MAX_OUTPUT 0, yield 60 s (M2: 180 s), temp 0.6, 2.2 h/game, conc 28).
- K  = Kaggle commit run arc3-keith-copy v4 (scriptVersionId 347562879): 42 lv (1.68/g), 10.42, 4 zero games
- M1 = Modal 20260902T2121-keith (base): 36 lv (1.44), 6.40, 2 zero
- M2 = Modal 20260903T0202-keith_yield180: 34 lv (1.36), 5.30, 4 zero
- M3 = Modal 20260903T0553-kv10-keith (kv10 serving profile, 82 calls/game): 43 lv (1.72), 10.47, 6 zero
- M4 = Modal 20260903T1014-keith_retry (retries not engaged): 40 lv (1.60), 8.62, 4 zero
Pooled: 1.56 lv/game, 8.24 score; kv5-only (K,M1,M2,M4): 1.52 lv/game, 7.69.
Scoring formula recomputed from actions_per_level on all 125 runs: 0 mismatches.
Loader: load.py -> cache.json (turns split into calls at [MODEL RESPONSE META]; calls == shim OK posts exactly:
1395/1351/2054/1293; actions attributed to the step_executed turn of the same analysis_step).

## Q1 outcome table (q1.out)
Classes over 5 draws: NEVER 0; CAPPED 5 (cd82, dc22, lf52, wa30 at 1; vc33 at 3); HIGHVAR (spread>=2) 9
(ar25, cn04, ft09, lp85, sb26, sc25, tn36, tr87, tu93); LOWVAR 11.
Levels histogram over 125 runs: 0:20, 1:56, 2:24, 3:12, 4:11, 5:1, 6:1.
Score carriers: ft09 36.9, re86 24.4, lp85 19.7, vc33 17.7, tr87 17.2, ar25 16.1 -> top-6 = 64% of score.
Consistency at observed max: 2.36 lv/game (+0.80), score 16.7; E[max of 2 draws] = 1.89 (+0.33); E[max of 3] = 2.09 (+0.53).
Same wall level in >=4/5 draws: 12/25 games (cd82 L2, dc22 L2, lf52 L2, ls20 L2, r11l L2, re86 L5, s5i5 L3,
sb26 L2, sk48 L1, su15 L2, vc33 L4, wa30 L2).
Per-level hazard: L1 105/125=0.84, L2 49/105=0.47, L3 25/49=0.51, L4 13/25=0.52, L5 2/13=0.15.

## Q2 timeline (q2.out, q7.out)
194 clears; per 15-min bucket: [41,27,29,26,27,16,8,12,8]; first hour 63%, last 30 min 10%.
Clear time: L1 median 1657 s, L2 2886 s, L3 4004 s, L4 5533 s. Median last clear 3623 s; 51% of the window
follows the last clear. Actions after last clear = 12630/20162 = 62.6%. Stuck bucket: median actions/baseline
1.13; 57/119 under-explored (actions < human baseline); novelty 0.86 (not looping).
GAME_OVERs 200 (1.6/run), RESETs 204; unchanged-board actions 6.9%. Post-GO gap to next real action: median
178 s (ordinary inter-call gap 140 s), total 16.7 h = 6.1% of game time. Stuck-tail wall time 49% in non-acting turns.
Clears per game per 10-turn bin: .38 .43 .36 .16 .15 .08 .16 0 0 -> marginal 0.123 lv/game per 10 turns after
turn 30; +0.5 lv/game by turns alone needs ~+41 turns/game (kv5 = 50 turns/game, kv10 = 80).

## Q3 loop (q3.out)
kv5: 50 turns/game, 55 calls/game, 141 actions/game, 2.56 actions/call, 159 s per turn (e2e 140-151 s, of which
queue 121-131 s server-side). Call categories: ACTION 50%, ANALYSIS-only python 46%, python error 3.3%,
no-tool-call 0.4%. Late-game (turn>=30) non-action share 48%. Follow-up "You have not acted yet" 3/5031 turns.
Reasoning chars/call mean 3262 median 2035 p90 7726 (act 2295 median; clear-producing calls 1208 median).
Carried note: mean 442 chars, changed in 40% of consecutive turns, median staleness at end 16 turns (45/100
runs >20 turns stale); 18% of assistant texts carry a labeled line. history_messages kept: median 22 (~7 turns).
Prompt composition (125 latest-call snapshots, chars): system 14.3%, tools 1.0%, current user/frame 21.6%,
prior user msgs 26.1%, prior REASONING 23.7%, prior tool calls 7.2%, tool results 4.6%, assistant text 1.4%.
=> history 63%, of which replayed reasoning is the largest single block (tool_agent.py:1892 stores `reasoning`
on the assistant history message; :151 keeps <=30 assistant turns; trimmed to context_budget 31744 by
_estimate_tokens = chars/3 of the JSON payload, tool_agent.py:462-467).
Real prompt tokens (shim): median 21.4-21.8k, p90 24-25k, max 27.8-29.2k; plateau ~22k from call 10 on.
=> the estimator over-counts by ~1.45x; ~10k tokens (30%) of the 32k window are never used.
e2e vs completion tokens (M1): <300 tok 127 s, 1-2k 147 s, 2-4k 167 s, >4k 197 s -> queue-dominated.

## Q6 thinking termination (q6.out)
7547 calls: finish tool_calls 7490, length 27 (0.36%), stop 30; no-tool-call 59 (0.78%). Reasoning median
pre-last-clear 2198 vs stuck tail 1705 vs zero-games 2030; stuck>pre in 49/103 paired runs. Calls >=10k chars:
400 (5.3%), of which 215 acted. => reasoning does terminate; truncation/no-conclusion is <1% of calls.
Zero-level vs >=2-level runs: identical calls/turn (1.09), longest no-action streak 6 vs 6, no-action share 0.50 vs 0.51.

## Q5 counterfactuals (125 runs)
observed 1.560 / 8.24 | (a) consistency at max 2.360 / 16.75 (mirage: E[max2] 1.89) | (b) +1 on started runs
2.392 (+0.83) / 16.2-16.8 | (c) L1 on zero runs 1.720 (+0.16) / 8.84 | (d) efficiency perfect 1.560 / 9.39 (+1.15 score, 0 lv)
| (b) on the 12 same-wall games only +0.44 lv; on the other 13 +0.39 | +2 on started 3.224 / 27.1.
Only (b) clears the +0.5 bar; it corresponds to the next-level wall on started games.

## Q4 zero-level failure modes (20 runs, read from zero_condensed.txt)
K bp35: analysis-paralysis + wrong model ("water/sponge" puzzle), 50 actions in 132 min, note never written.
K g50t: wrong hypothesis (red plug/ghost-plate theory), 202 actions.  M1 g50t: red wire believed static blocker, 83 actions.
M3 g50t: bulb/plug model, BFS on wrong state space, 94 actions.
K sk48 / M1 sk48 / M3 sk48 / M4 sk48: push/drag/grab rule probing forever; M3 528 actions, "LEFT noop x7"; M4 BFS
says goal unreachable -> "goal hypothesis wrong"; TypeError. (0/5 draws ever clear L1; same wall each time.)
K sp80 (16 GO) / M2 sp80 (9) / M4 sp80 (10): paddle "cannot reach the magenta", serpentine sweeps of all cells,
lives burned by a ~27-action timer; after a death kept pressing on a frozen board.
M2 ls20: goal predicate wrong (match glyph while covering plus), per-attempt countdown.
M2 sc25: keypad "diamond" hypothesis; two timer deaths.
M2 tn36 / M3 tn36 (12 GO, 1132 actions): believes the peg tray is the puzzle; brute-forces 32 patterns by
burning the timer to expiry each round.
M3 bp35 (4 GO): boat/drain model; M3 cn04: after a GAME_OVER at action 75 wrote "nothing actionable... no tool call"
(finish=stop) — 2400 s gap before the next real action; feet-on-target hypothesis.
M3 m0r0 / M4 m0r0: two-block mirrored control; goal unknown; BFS to "swap"; replay model mismatches; NameError.
M4 tr87: dial/cursor glyph cycling, Hamming-distance searches; 39 actions in 2.2 h (analysis-paralysis).
Tally: WRONG-HYPOTHESIS 13, ANALYSIS-PARALYSIS 4 (K bp35, M1 g50t, M4 m0r0, M4 tr87), BUDGET-CAP/life-burning
overlay on 6 (sp80 x3, tn36 M3, bp35 M3, sc25 M2), OTHER 1 (M3 cn04 game-over misread). The same game fails the
same way in every draw (g50t x3, sk48 x4, sp80 x3): failures are deterministic, not variance.

## Q4 stuck-tail classes, all 125 runs (classes_all.txt; 105 started runs classified by 3 memory-blind sub-agents
## from failmodes_part{1,2,3}.txt, 20 zero-level runs by me; q12.out)
WRONG-HYPOTHESIS 37, ANALYSIS-PARALYSIS 25, PROGRESSING 17, TIMER 14, BUDGET-CAP 14, OTHER 8, LOOPING 8, PY-ERRORS 1, WON 1.
Ceiling of +1 level (parity) per class: comprehension (WH+LOOP+OTHER, 53 runs) +0.42 lv/game; analysis-paralysis
(25) +0.20; time-limited (TIMER+PROGRESSING, 31) +0.25; budget-cap (14) +0.11. Under-explored (tail actions <
baseline): TIMER 14/14, AP 18/25, PROGRESSING 11/17, WH 9/37.
Lives ending in GAME_OVER: 21/25 games have them; 171/200 GOs on the never-cleared level; 50.7% of all actions
sit inside lives that ended in a GO (q10.out). Modal wall never passed in any of 5 draws: 12/25 games (q11.out).
Correct idea present in note but not executed: 11/35 + 6/35 + ~11/35 of started tails (classifier counts).
Outputs: q1.out q2.out q3.out q6.out q7.out q8.out q9.out q10.out q11.out q12.out, stuck_table.json, cache.json,
kaggle/ (commit-run outputs incl. vllm-openai-server.log, benchmark.json, transcripts, prompts, events).
