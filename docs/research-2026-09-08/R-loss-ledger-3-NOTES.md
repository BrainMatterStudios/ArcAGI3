# loss-ledger-3 — working notes (2026-09-08), yield900 regime vs the yield-60 base

Data (all 25 public games, 7,920 s/game, conc 28, Flash-Next NVFP4, stock June duck, window 32768, MAX_OUTPUT 0, temp 0.6):
NEW (LOCAL_ANALYZER_YIELD_SECONDS=900):
- Y1 = Modal 20260906T2251-keith_yield900: 41 lv, local 8.46 (211.4 sum), zero 1 (sk48)
- Y2 = Modal 20260907T0702-keith_yield900-cb2 (counterbalanced, first in session): 40 lv, 7.90, zero 1 (g50t)
- YK = Kaggle commit ahmedmobasher86/arc3-keith-yield900 (kf 347926973; fetched via REST, 78 files): 37 lv, 7.05, zero 5 (bp35 sc25 sk48 sp80 tn36)
BASE (yield 60):
- M1 = Modal 20260902T2121-keith: 36 lv, 6.40, zero 2 | M4 = Modal 20260903T1014-keith_retry (retries not engaged; stock loop): 40 lv, 8.62, zero 4
- K = Kaggle commit arc3-keith-copy v4: 42 lv, 10.42, zero 4
Pooled: NEW 118 lv / 75 runs = 1.573 lv/game, local score 7.80/game; BASE 118 lv / 75 = 1.573, 8.48/game.
Brief's own comparators only (M1 + K = 50 runs): 78 lv = 1.56/game. Same conclusion.
Scoring: local benchmark caps per-level efficiency at 1.0 (recomputed, 0 mismatches with cap 1.0); the live formula
min(115, 100*(b/a)^2) is recomputed here as "live-cap": NEW 8.42/game, BASE 9.06/game.
Loader: load.py -> cache.json (150 runs; turns split at '--- analysis_step=' headers, calls at '[MODEL RESPONSE META]';
actions = benchmark.history (cumulative wallclock) zipped 1:1 with events.jsonl action rows (level/clear/game_over/board);
0 length mismatches). Scripts: q1 q2 q3 q5 q7 h1b h2 timing tails; outputs *.out; tails/part1-5.txt (75 NEW tails).

## Q1 outcomes (q1.out, q7.out)
NEW levels histogram 0:7 1:42 2:12 3:7 4:4 5:3 | BASE 0:10 1:36 2:14 3:7 4:7 5:1.
NEW: NEVER 0; CAPPED (same level all 3 draws, < max): 9 (cn04 dc22 ka59 lf52 ls20 s5i5 sb26 tr87 wa30, all at 1); HIGHVAR (spread>=2): 5 (ar25 lp85 sc25 su15 tn36).
BASE: NEVER 1 (sk48); CAPPED 8 (cd82 dc22 lf52 ls20 sb26 su15 wa30 at 1; vc33 at 3); HIGHVAR 6.
Score carriers NEW: ft09 148.8, lp85 80.2, re86 57.9, ar25 40.5, vc33 39.0, r11l 33.3 = 68% of score (BASE top-6 = 62%).
Hazard NEW L1 .91 (68/75) L2 .38 (26/68) L3 .54 L4 .50 L5 .43 (3/7) | BASE L1 .87 L2 .45 L3 .52 L4 .53 L5 .12 (1/8).
8 base walls on NEW: cd82 L2 passed 1/3 (YK reached L3); dc22, lf52, ls20, re86 L5, s5i5 L3, vc33 L4, wa30: 0/3 each. On BASE 0/3 all.
Paired per-game diff NEW-BASE levels: mean 0.000, sd 0.56, se 0.11; 8 games better, 8 worse, 9 same. Live-cap score diff -0.63 +- 1.64 (se).
Per-run levels sd 1.19 => sd of a 25-game total ~6 levels: 41 vs 36 is < 1 sd.
Consistency-at-max NEW 2.00 lv/game (+0.43); BASE 2.08 (+0.51).
Modal wall never passed in any of the 6 draws (both regimes): 12 games (bp35 L2, dc22 L2, g50t L2, lf52 L2, lp85 L6, ls20 L2, r11l L3, sb26 L2, sp80 L2, tn36 L3, vc33 L4, wa30 L2).
Per-source: Y1 41 / Y2 40 / YK 37 vs M1 36 / M4 40 / K 42. YK zero games are the usual ones (bp35 sc25 sk48 sp80 tn36); YK serving matched the rig (158 s/call, 54 calls/game).

## Q2 timeline (q2.out)
Clears per 15-min bucket NEW [23,19,17,15,14,12,6,9,3] (118) | BASE [21,19,20,18,14,9,5,7,5] (118); first hour 63% vs 66%; last 30 min 10% both.
Clear-time medians NEW L2 1706 s, L3 3112, L4 4170, L5 6420 | BASE 1774 / 2886 / 3689 / 6015.
Median last clear (started runs) NEW 3828 s vs BASE 3439; window after last clear 51% vs 53%.
Actions: NEW 8627 (115/run) vs BASE 10758 (143/run). After last clear: 52.8% vs 61.7%. In 15-min buckets with no clear: 77.2% vs 79.6%.
GAME_OVERs NEW 65 (0.87/run; 29 runs; 52 on the never-cleared level) vs BASE 98 (1.31/run; 27 runs; 84 on the never-cleared level).
Lives NEW 255, GO-ended 65, actions inside GO-ended lives 2257 = 26.2% (BASE 3905 = 36.3%); actions/GO-life median 31 both.
GO games NEW: sp80 23, tu93 13, bp35 5, wa30 4, ka59/ls20/sc25/su15 3 ... (14 games) | BASE sp80 32, r11l 14, tu93 14, wa30 7, tn36 7 ... (15 games).
Unchanged-board actions 6.2% vs 7.3%. Post-GO gap to next real action median 320 s vs 252 s, 5.1% vs 5.2% of game time.
Clears per game per 10-turn bin NEW [.80 .57 .17 .03 0...] (26 turns/game) | BASE [.80 .60 .12 .04 .01 0...] (54 turns/game).
Wall-level actions/baseline median NEW 0.72 (48/75 under-explored) vs BASE 0.84 (42/75); turns on the wall median 10 vs 34.

## Q3 loop (q3.out, q7.out, timing.out)
NEW: 26.0 turns/game, 53.2 calls/game, 115 actions/game, 2.05 calls/turn, 2.16 actions/call, 4.42 actions/turn.
BASE: 54.1 turns, 55.2 calls, 143 actions, 1.02 calls/turn, 2.60 actions/call, 2.65 actions/turn.
Calls/turn NEW: 1: 49%, 2: 23%, 3: 11%, 4: 6%, 5+: 10% (0: 2% request errors). BASE: 1: 96%.
Turn patterns NEW (A=analysis-only, X=acts): X 947, AX 398, AAX 183, AAAX 73, AAAAX 40, AAAAAA 27, XX 22, AAAAA 16, AAAAAX 15.
Second call given >=2 calls: (A,A) 420, (A,X) 416, (X,X) 24, (E,A) 26 -> the second call is the act 47% of the time; act->act 3%.
Call types NEW: analysis-only 48.7%, acts 47.1%, py-error 2.8%, no-tool 1.4% | BASE 45.2 / 51.8 / 2.6 / 0.4.
=> the analysis-only share did NOT fall; the yield900 knob re-packaged the same call mix into fewer, longer turns.
No-action turns NEW 159 (8.2%): 84 turn_time_budget yields (5-7 calls each = 478 calls, 22.8 h = 14% of wall), 37 stop_requested (game end), 38 request errors.
BASE no-action turns 1995 (49.2%; 1922 turn_time_budget yields of 1 call each = 79.7 h = 48% of wall, but the response is kept in history, so nothing is discarded).
Yield turns NEW: 60/84 on the wall level; next turn acts 56/84; streaks mostly 1; by game r11l 9, tr87 9, cn04 7, m0r0 7, tu93 6, ls20 6.
Wall share NEW: acting calls 48%, non-acting 52% (BASE 53/47).
Reasoning chars/call NEW mean 3576 median 2298 p90 8537 (act 2640, analysis 2022) | BASE 3242 / 2045 / 7678. finish=length 1.25% (50/3992) vs 0.31%; no-tool-call 1.4% vs 0.4%.
Actions per cleared level 34.5 (baseline 36.4), ratio median 0.84 | BASE 34.9 (38.2), 0.83 -> efficiency unchanged.
Actions per executed turn median 3, mean 4.8, p90 11; 33% of executed turns run exactly 1 action (BASE 32%).
Note (world model carried in the prompt): mean 456 chars, changed in 33% of consecutive turns, staleness at end median 3 turns, >10 stale 17/75 (BASE 23%, median 7, 34/75).
history_messages median 23 both. Prompt tokens median 22.0-22.5k, p90 26k, max 31.3k (shim).
Where the ~150 s goes (shim, rig): e2e mean 146-153 s, median 152-161; first call of a game (queue empty) 10-18 s; e2e by completion tokens: <300 tok 98-104 s, 300-1k 133-140, 1-2k 152-159, 2-4k 172-183, >4k 212 => ~130 s is queue, ~20 s inference. Kaggle YK vLLM log: running 3.1 reqs, waiting 21.4 (KV-limited), 251 gen tok/s.
Marginal clear rate NEW after turn 20: 0.03 lv/game per 10 turns (BASE after turn 30: 0.04) — turns are not the binding resource in either regime.

## Q5 counterfactuals (q5.out)
cap 1.15 (live formula), NEW: observed 1.573 lv / 8.42 | (a) max-of-3-draws 2.00 (+0.43) / 13.20 | (b) +1 level on started runs 2.48 (+0.91) / 17.10 (+103%) | (c) L1 on zero runs 1.667 (+0.09) / 8.75 | (d) efficiency = baseline on all cleared levels 9.45 (+12%); efficiency = 1.15 cap on all 10.87 (+29%).
cap 1.15, BASE: 1.573 / 9.06 | (a) 2.08 (+0.51) / 13.99 | (b) 2.44 (+0.87) / 17.52 (+93%) | (c) +0.13 | (d) 9.27 (+2%) / 10.66 (+18%).
Efficiency loss vs 1.0 on cleared levels: NEW 1.92 pts/run (top: ft09 L3 19.7 pts over 3 runs, sc25 L2 9.2, ar25 L4 8.8, tu93 L2 8.6) vs BASE 1.13 pts/run. Live-cap bonus already earned (levels beating baseline): 0.90 vs 0.91 pts/run.
=> only (b) clears the bar in both regimes; (d) is +12% on NEW (below +20%) and requires baseline-parity on every cleared level.

## H1 timer/life discipline (h1b.out)
Edge-bar detector (rows 0/63, cols 0/63; a life's line count changes on >=50% of steps and is monotone): 19 games show a timer-like bar in some life (ar25 bp35 cd82 dc22 ft09 g50t ka59 lf52 lp85 r11l re86 s5i5 sc25 sp80 su15 tr87 tu93 vc33 wa30).
GAME_OVER at bar exhaustion (line at its max fill within 2 cells): NEW 5 of 65 GOs (8%), 135 actions = 1.6% of all actions (bp35 2, g50t 1, sp80 1, tu93 1); BASE 3 of 98 (3%), 54 actions = 0.5%.
The other 60 NEW GOs end mid-bar (collisions / rule deaths): sp80 23 (median 13 actions/life), tu93 13 (median 6), bp35 5, wa30 4.
Timer words in reasoning: 35% of calls (1407/3992) mention timer/countdown/step bar; every run mentions it; 895 calls quantify it next to a number. The model is not blind to the bar; it dies of collisions, not of the bar.

## H2 post-goal thrash (h2.out; regex proxy, first assertion on the wall level)
NEW: 26/75 runs assert "goal reached / should have completed / no level_completed" on the wall level; 208 turns (2.8/run) and 862 actions (11.5/run = 10.0% of actions) follow on that level.
BASE: 27/75 runs; 621 turns (8.3/run), 2164 actions (28.9/run = 20.1% of actions).
Largest NEW: Y2 lf52 (167 actions after), YK ls20 (117), YK sc25 (86), YK tn36 (68), Y1 sp80 (67), YK lf52 (62), Y1 wa30 (59), YK sk48 (57).

## Q4 failure classes (5 memory-blind readers over tails/part1-5.txt; filled below)
Readers' tallies over 75 NEW runs (PRIMARY): ANALYSIS-PARALYSIS 25, PROGRESSING 15, WRONG-HYPOTHESIS 13, LOOPING 11, TIMER 7, OTHER 3 (2 = clock expired on arrival at the wall: Y1 ft09 L5 at 7894 s, Y2 ka59 L2 at 7828 s; 1 = goal-blind enumeration), BUDGET-CAP 1, PY-ERRORS 0.
Shares NEW vs BASE (prior ledger, 125 runs): AP 33% vs 20%; WH 17% vs 30%; LOOP 15% vs 6% (WH+LOOP 32% vs 36%); PROG 20% vs 14%; TIMER 9% vs 11%; BUDGET-CAP 1% vs 11%.
IDEA-IN-NOTE (plausible key idea present, never executed) Y = 41/75 (AP 18/23, TIMER 5/7, WH 5/13); GOAL-REACHED-NO-CLEAR (manual) 16/75; BAR-AWARE plans 27 / mentions 32 / none 16.
Per class (parsed 71): wall-level actions/baseline median AP 0.40 (19/23 under 1x; 48 of the 84 yield turns), WH 1.05, LOOP 1.81 (24 GOs), PROG 0.62, TIMER 0.81.
Ceiling of +1 level (parity) per class: comprehension (AP+WH+LOOP started runs, 40) +0.53 lv/game; time-limited (PROG + clock-arrival, 16) +0.21; TIMER +0.09; BUDGET-CAP +0.01.
Zero-level runs (7): YK bp35 AP, Y2 g50t AP, Y1 sk48 AP, YK sk48 WH, YK sc25 OTHER, YK sp80 LOOP (108 poses, 16 deaths, 425 actions vs 39), YK tn36 LOOP. Same games as the base's zero set.
Recurring patterns named by >=3 readers independently: (1) the tail is spent on offline model-building / forensics of a past event, with the action count on the wall far under baseline (g50t 17 vs 175, ka59 7 vs 109, r11l 7 vs 51, tr87 8 vs 58, m0r0 10 vs 203); (2) the correct rule appears in the LAST call (cd82 Y1, dc22 Y2, g50t Y2, sp80 Y1, sb26 Y2) and is never executed; (3) L(n-1)'s goal carried into L(n) (lf52, sk48, s5i5, sc25, tn36, vc33, sp80); (4) self-inflicted python bugs void executed plans (m0r0 'U' vs 'UP', tr87 stale closure, r11l/tn36 indexing); (5) finish=length on 20-28k-token reasoning with no tool call (cd82 Y1 x2, sk48 Y1 x2) — still rare (1.25% of calls).
Turns with >=3 analysis-only calls before any action: 291 (15% of turns); 84 of them run 5-7 calls and yield with nothing executed (14% of wall time).
