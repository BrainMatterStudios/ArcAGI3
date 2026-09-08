# loss-ledger-3 — the yield900 regime (LOCAL_ANALYZER_YIELD_SECONDS 60 → 900), 3 runs × 25 games, 2026-09-08

Data: NEW = Modal 20260906T2251 (41 lv), Modal 20260907T0702-cb2 (40), Kaggle commit arc3-keith-yield900 (37, kf 347926973,
fetched via REST) = 118 lv / 75 runs. BASE = Modal 20260902T2121 (36), Modal 20260903T1014 retry-not-engaged (40), Kaggle
arc3-keith-copy v4 (42) = 118 lv / 75 runs. Scoring recomputed (local cap 1.0, 0 mismatches) and re-scored with the live
formula (cap 1.15). Scripts/outputs/notes: scratchpad/search/loss-ledger-3/ (cache.json, q*.out, h1b.out, h2.out, timing.out,
classes_all.txt, R-loss-ledger-3-NOTES.md).

**HEADLINE: on 75 vs 75 runs the knob is a NULL on levels (1.573 vs 1.573 lv/game; paired per-game diff 0.00 ± 0.11 se;
8 games better, 8 worse, 9 same) and slightly NEGATIVE on live-cap score (8.42 vs 9.06/game; −0.63 ± 1.64).** The rig pair
41/40 vs 36 is < 1 sd of a 25-game total (sd ≈ 6 levels); the Kaggle commit read 37 vs the base commit's 42. Efficiency is
unchanged (34.5 vs 34.9 actions per cleared level, ratio-to-baseline median 0.84 vs 0.83), so the 4.31 live draw's gain is not
visible in any local instrument; with base live draws at 3.25/2.58 a single 4.31 cannot be separated from draw noise.

Q1 outcomes NEW: levels 0:7 1:42 2:12 3:7 4:4 5:3. NEVER 0; CAPPED-at-1 in all 3 draws: 9 games (cn04 dc22 ka59 lf52 ls20
s5i5 sb26 tr87 wa30); HIGHVAR 5. Top-6 carriers (ft09 lp85 re86 ar25 vc33 r11l) = 68% of score. Hazard L1 .91 L2 .38 L3 .54
L4 .50 L5 .43 (BASE .87/.45/.52/.53/.12). Of the 8 base walls only cd82 L2 passed, once (YK); dc22/lf52/ls20/re86-L5/s5i5/vc33-L4/
wa30: 0/3 each. 12 games' modal wall was never passed in any of the 6 draws of either regime. Consistency-at-max +0.43 (mirage).
Q2 timeline: clears per 15 min [23,19,17,15,14,12,6,9,3] vs BASE [21,19,20,18,14,9,5,7,5] — the same curve; median last clear
3828 s vs 3439; 77% of actions fall in no-clear buckets (BASE 80%); actions after last clear 53% vs 62%. What the knob DID
change: actions/game 143 → 115; GAME_OVERs 1.31 → 0.87/run; actions inside GO-ended lives 36% → 26%; post-goal thrash actions
20% → 10%; note staleness 7 → 3 turns; wall-level exploration FELL from 0.84× to 0.72× the human baseline (48/75 under-explored).
Q3 loop: turns/game 54 → 26, calls/game 55 → 53, calls/turn 1.02 → 2.05 (1: 49%, 2: 23%, 3: 11%, 4: 6%, 5+: 10%); second
call is the act 47% / another analysis 47% / act→act 3%. **Analysis-only call share 45% → 49% — the call mix did not change;
the knob only re-packaged the same calls into fewer turns.** In the base a 60 s yield never discarded the response (it stays in
history), so "turn ended after one call" cost nothing. No-action turns 49% → 8%, but 84 turns now run 5-7 analysis calls and
yield with nothing executed (478 calls, 14% of wall); non-acting calls hold 52% of wall time (BASE 47%). Reasoning chars/call
3576/2298 (BASE 3242/2045); finish=length 1.25% (4× the base, still small). Where the ~150 s goes: first call of a game 10-18 s;
e2e by completion tokens <300: 100 s, 1-2k: 155 s, >4k: 212 s → ~130 s queue + ~20 s inference; Kaggle box: 3 running / 21
waiting. Marginal clears after turn 20: 0.03 lv/game per 10 turns — turns are not the binding resource in either regime.
Q4 failure classes (75 NEW tails, 5 memory-blind readers): ANALYSIS-PARALYSIS 25 (33%; BASE 20%), PROGRESSING 15 (20%; 14%),
WRONG-HYPOTHESIS 13 (17%; 30%), LOOPING 11 (15%; 6%), TIMER 7 (9%; 11%), OTHER 3, BUDGET-CAP 1 (1%; 11%). BUDGET-CAP collapsed
with the GO count; AP grew by the same amount: **the freed calls went into more offline model-building, not into probes** (AP
runs act at 0.40× baseline on the wall; 48 of the 84 yield turns are theirs). Plausible key idea present in the note/thinking
but never executed: 41/75 (AP 18/23); correct rule found in the LAST call: 5 runs; previous level's goal carried into the wall
level: 7 runs; goal-reached-no-clear (manual) 16/75. Zero-level runs (7) are the base's zero games (sk48×2, bp35, g50t, sp80,
sc25, tn36) failing the same way.
Q5 counterfactuals (live cap): (a) max-of-3 +0.43 lv; **(b) +1 level on started runs +0.91 lv / +103% — the only bucket over
the bar**, by class: comprehension (AP+WH+LOOP, 40 started) +0.53, time-limited +0.21, TIMER +0.09; (c) L1 on zero runs +0.09;
(d) efficiency at baseline on every cleared level +12% score (0 lv; loss 1.92 pts/run, a third of it ft09 L3) — below the +20%
bar; only beating baseline everywhere (cap 1.15) reaches +29%.
H1 TIMER discipline: 19 games show a timer-like edge bar; but only 5 of 65 GAME_OVERs (8%) end at bar exhaustion = 135 actions
= 1.6% of actions (BASE 3/98, 0.5%); the other GOs are collisions/rule deaths (sp80 23, tu93 13). The model mentions the bar in
35% of calls and plans against it in 27/75 runs; the 7 TIMER runs (ls20×2, wa30×2, su15, tn36-secondary) mostly mis-measured
the bar or drained it on purpose. A "plan against the bar" instruction has ≤ +0.09 lv/game to win. DEAD as a step.
H2 POST-GOAL THRASH: 26/75 runs (regex) / 16/75 (readers) assert the goal reached without a clear; 862 actions (10%, 11.5/run)
and 208 turns follow — half the base's 20%. Ceiling ≤ +0.21 lv/game (16 runs). Not a step by itself; folds into #1.

## Ranked candidates (bar ≥ +0.5 lv/game or ≥ +20% score)
**#1 Harness-enforced probe discipline on the wall level** (not a prompt aid — graft_hypo took 0.8% uptake): cap analysis-only
calls at 2 per turn; the 3rd call's tool result carries "[PROBE REQUIRED] execute a ≤5-action test of your leading hypothesis
now and report what changed"; if a turn still ends without acting, the next turn's prompt lists the untested hypotheses from the
note. Target: AP 25 runs (0.40× baseline, 41 idea-in-note runs). Ceiling +0.27 (AP) to +0.53 (all comprehension); realistic far
lower — the same games fail identically across 6 draws. Cheapest rig experiment: arm `keith_probe` (graft installed via
ARM_GRAFTS like graft_evidence), 25 games, 1 draw, conc 28, 7920 s (≈$9), run FIRST in a fresh session. Pre-registered rules:
ENGAGEMENT = turns with ≥3 analysis-only calls < 5% (now 15%) AND turn_time_budget yields < 20 (now 27-30/draw) AND wall-level
actions/baseline median ≥ 1.0 (now 0.72). PRIMARY = levels ≥ 52 (+0.5 over the 3-draw mean 39.3; sd of a total ≈ 6) AND ≥ 3 of
the 12 six-draw never-passed walls passed → STEP candidate; 45-51 → positive, counterbalanced redraw; < 45 or engaged-but-flat →
dead (the model probes and still fails: comprehension, not cadence).
**#2 Nothing else clears the bar.** Efficiency-only +12% (needs baseline parity on every cleared level; ft09 L3 alone is a
third of the loss — a per-game replay of a known solution would be a specialist, not a lever). Time-limited +0.21. TIMER +0.09.
Post-goal re-derivation +0.21 (rider inside #1). Throughput: 0.03 lv/game per 10 turns.
**Do not fly yield900 as a +1.0 lever.** Its expected live value is the base's (levels equal, local score −7%); it is harmless
(fewer GOs, cleaner notes) but the 4.31 draw is one draw against a base pair that itself spans 2.58-3.25.
Cannot resolve from this data: the per-game composition of the 4.31 live draw (no live per-game feed); the live draw sd (2 base
draws); whether the AP-share rise is real or reader drift (different readers than the base ledger; WH+LOOP is flat at 32-36%);
the timer detector reads only border lines (inner HUD bars are missed); H2 regex vs manual counts differ 26 vs 16.
