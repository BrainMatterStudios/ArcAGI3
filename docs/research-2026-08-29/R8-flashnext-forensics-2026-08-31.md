# Flash-Next 25-game smoke forensics (arc3-flashnext-smoke v4)

Sources: submission/_flashnext_smoke/results_v4/flashnext_smoke_results.json, 11 fetched transcripts
(results_v4/transcripts/) and 11 events.jsonl (results_v4/artifacts/): ka59, bp35, sc25, ar25, ft09,
vc33, tu93, g50t, sk48, tr87, wa30. Every count below is from events.jsonl (per-action records with
board_changed / game_over / level_completed / batch_index / batch_size / level / board_ascii) or from
quoted transcript text. Items marked INFERRED are counterfactual estimates; everything else is MEASURED.

Scoring model used for recovery math (MEASURED - reproduces final_score to 3 decimals on ft09 17.703,
ka59 0.032, ar25 4.112): final = SUM_completed min(115, 100*(b/a)^2)*L / SUM_all_levels L, L = 1-indexed
level. One anomaly: cd82/cn04-class games read as 100-capped (4.762 = 100*1/21) where the formula gives
115*1/21 = 5.48 - a +/-0.7-pt cap ambiguity on ratio>1 levels that changes no ranking below.
Smoke mean local score: 4.07/game (27B stock same geometry: 3.5-4.9).

## One-sentence verdict

Flash-Next is NOT the 27B: it acts on 99% of turns, emits zero malformed tool calls, and takes ~300
actions / ~99 turns per game. Its points die in two places: (1) per-level move-budget GAME_OVERs it
replays blind - 72.5% of all 4,261 fetched-game actions sat inside attempts that ended in a GAME_OVER -
and (2) the carried world model dies permanently at the first GAME_OVER (and is empty on most turns
anyway: the harvest reads the assistant-text channel and ~65% of responses have 0 content chars), after
which the model degenerates into byte-identical repeat loops.

## QUESTION A - efficiency blowups, bucket decomposition

Disjoint buckets per level: B2 = actions before the mechanic was first correctly stated (quoted step,
mapped to level-action count); B1 = actions in GAME_OVER-terminated attempts AFTER the mechanic was
known (death replay incl. RESETs); B3 = post-knowledge actions in the final successful segment.
B4 (noops / repeats of known-ineffective actions) and B5 (format failures) are overlays.

| game/level | total (base) | deaths (cause) | B1 | B2 | B3 | B4 noop | B5 | dominant |
|---|---|---|---|---|---|---|---|---|
| ka59 L1 | 296 (28) | 2 @100,201 (both budget expiry; budget ~100, HUD identified at cum 12) | 178 | 23 (mech @16, goal @23) | 95 (3.4x base, belief-thrash execution) | 37 | 12 tool timeouts, 1 length turn | B1 60% |
| bp35 L1 | 189 (21) | 3 @64,129,146 (2 budget - one deliberate "raise the water" suicide; 1 real drain death) | 139 | 7 (mech @7) | 43 (2x base) | 0 | 2 length, 3 timeouts, 8 truncated outputs | B1 74% |
| sc25 L1 | 83 (36) | 1 @51 (attempt-budget expiry, unseen - its own print filter dropped game_over) | 0 | 52 (mech @16, lost to col-sampling bug until @52) | 31 (built the WRONG pattern; cleared by accidental LEFT x14 knob probe) | 8 | 1 zero-action stop | B2 63% |
| ar25 L2 | 123 (50) | 1 @81 (col-63 budget expiry mid-plan; strip called "HUD timer, ignore" @67, re-promoted to gameplay 2 turns later) | 55 (wrong-goal + HUD-as-mechanic detour) | 26 (mech @26) | 42 (clear, UNDER baseline) | 18 | 0 | goal-hunt 66% |
| ft09 L3 | 77 (23) | 0 | 0 | 0 (mech carried from L2, correct from action 1) | 77 - 68 painting six whole-board pattern guesses due to a stencil pixel-sampling bug (c0+2+2j vs c0+2j); 9-click clean solve once fixed @68 | 11 (refuted "click the 4 map tiles" re-tested 3x) | 0 | B3 88% |
| vc33 L3 | 99 (44) | 1 @76 (timer expiry) | 29 | 47 (18 scattered dead probes; valve mechanic @47) | 22 (0.5x baseline - perfect once known) | 6 (dry-valve clicks, diagnosed 3x, kept clicking) | 5 consecutive length turns post-death (~53k reasoning chars, 0 actions) | B2 47% |

Aggregate over the six A-levels (866 actions): B1 = 401 (46%), B2 = 155 (18%), B3 = 310 (36%, of which
~180 mis-grounded goal-testing, ~130 clean execution). 8 deaths: 7 budget/timer expiries, 1 hazard.
DOMINANT BUCKET: GAME_OVER retry replay (B1) - expensive because the WM does not survive death (ka59
re-derived "is the purple door passable" 8x, "step size" 7x; the same mechanic was announced as a
"breakthrough" 4 separate times, 272 actions apart).

Near-universal pattern: the final segment is at or under baseline (ar25 42 vs 50, vc33 22 vs 44,
bp35 43 vs 21). Execution once the model knows is mostly fine; the cost is lives spent learning plus
full-level replays with amnesia.

## QUESTION B - depth blocks

Cadence evidence (MEASURED, deaths at level-action): sc25 L2 every 27 exactly x18; vc33 L4 every 51
exactly x3 (all three deaths are the SAME click, MOUSE(62,16)); wa30 every ~200 (200/401/602); tr87
@128 of 256; g50t @130/261; ka59 @100/201; bp35 L1 @64/129; ft09 L4 ~97 effective (noops do not consume
its timer). These are per-level move budgets, not stochastic hazards. Only tu93 (deaths every 5-9
actions - invisible kill-cell at maze (2,3)) and bp35's drill-deaths are hazards.

| game | primary | secondary | evidence |
|---|---|---|---|
| g50t (0 lvls, 315) | state-loop | mechanic-found-then-LOST | mech+goal both correct at step 3; 174 WM revisions; player identity flipped 4x; "elevator box is the only remaining goal" hallucination x28; 7 byte-identical turns at action 312; WM carry died exactly at death #1 (action 130) |
| sk48 (0 lvls, 186) | goal-not-found | time-starved-while-progressing | wire-grab mechanic stable+correct by action 44; "the goal is unknown" - 5 win-condition candidates enumerated and refuted (one by BFS); 0 deaths; WM carried 88% of turns to the end; run ended on analyzer HTTP read-timeout mid-plan; 9/99 turns lost to finish_reason length |
| tr87 (0 lvls, 256) | state-loop | analysis-paralysis | mechanic AND goal correct by action 12 ("5-slot 7-state dial, match the sky glyphs"); after death @128 the WM died and it pressed UP 44 straight turns while its own tool output printed a byte-identical 4-slot prefix all 44 times; "slot 0 receives a total of 116 UPs = 4 mod 7" contradicts its own "slots 0,1,3,4 are frozen"; 212/256 actions UP; 0 noops so no guard could fire |
| wa30 (0 lvls, 695) | GAME_OVER-spiral | state-loop | ~200 budget read as 64 (1 HUD px ~ 3 actions); all 3 deaths unacknowledged ("pink = 64/64 -> fresh run"); frozen WM paragraph x85; goal never found (3 refuted goals); wrong player sprite until action 664; 46/117 turns hit the 30s tool timeout (actions executed, observations lost) |
| ka59 stop on L2 (120, 0 deaths) | goal-found-then-lost | cut mid-novel-attempt | docking score signal found ("score 1 = 6x6 docked", cum 330) then abandoned for 4 unrelated theories in 18 turns; last 3 turns were real progress ("reached the bottom row! Exit... 4 more squares") when the run cap hit |
| bp35 stop on L2 (102, 9 deaths) | GAME_OVER-spiral | terminal state-loop | lethal rule stated 5x ("drilling the raft under/adjacent to the boat = game over... Confirmed twice") and violated twice AFTER "confirmed"; each death re-randomizes the layout so the mapping tax was paid 9x; last 4 turns emit an identical thought with no plan |
| sc25 stop on L2 (510, 18 deaths) | state-loop | goal-not-found root, GO-spiral symptom | matched the panel against the ROBOT'S FACE (rows 17-23) instead of the actual display (rows 50-59) for 500 actions; one 3-click code string emitted 45 consecutive identical turns; reasoning collapsed to a constant 231 chars for 44 turns; the exact 27-action death cadence never named; L1's own "attempt counter" diagnosis never carried forward |
| tu93 stop on L3 (321, 36 deaths) | GAME_OVER-spiral | geometry collapse at end | invisible kill-cell (2,3) "drawn the same as normal walkable cells - died there 5 times"; phase A = genuine hypothesis-per-retry (mod-4 timing, keep-moving, wait - all tested); phase B = 6 byte-identical RESET,UP,UP,LEFT death cycles (30 actions, 0 new information); 35 RIGHTs (11% of the level) deliberately drained the timer as an experiment; ended ALIVE 2 cells from the goal when the analyzer HTTP timeout killed the run |

Starvation check: NOT turn-starved. 11 games averaged 387 actions / 99 turns (27B stock ~88 actions);
all games ran to the 7,920 s wall. Only sk48 (and arguably the last turns of tu93 and ka59-L2) were
progressing at the cut; every other block is a loop or spiral. "Time-starved" is the correct primary
label in 1 of 4 zero games.

## Aggregates (i)-(iv), 11 fetched games

(i) WM carriage: present on 473/1,376 analysis turns (34%). Per game: bp35 0%, sc25 10%, vc33 15%,
wa30 19%, tu93 25%, ka59 31%, g50t 35%, tr87 38%, ar25 64%, ft09 86%, sk48 88%. The worst blowups and
all state-loop games sit at the bottom; the two cleanest runs at the top. THE CARRY DIES AT THE FIRST
GAME_OVER AND NEVER RETURNS - verified exactly in g50t (last carry action 130 = death #1), tr87 (123 vs
death 128), wa30 (193 vs death 200), ka59 (dropped one turn after death #1); sk48, the only fetched game
with zero deaths, kept its WM to the final turn. Root cause of the low baseline: the harvest reads the
assistant TEXT channel and ~65% of responses have content_chars 0. When carried, the text also freezes
(ka59 identical 46 turns; vc33 stale paragraph x53; wa30 x85).

(ii) Analysis-only vs acting: per analysis turn only 14/1,376 (1.0%) executed zero actions. Per model
invocation: 282/1,972 (14.3%) ended at the 60s yield without acting + 27 (1.4%) died at the 4,096-token
cap with no tool call => ~16% action-less invocations vs the 27B's 47%. Per python call: 61.5% acting /
38.5% inspection-only (normal).

(iii) Batching: NEVER. All 4,261 env actions were single-action action(['X']) calls - batch_size == 1 in
4,261/4,261 events. Multi-action turns exist (mean 3.9 actions per acting turn, max 33) only via Python
loops of single calls, which the 30s tool timeout guillotines: ~130 timeouts across 11 games (wa30
46/117 turns; ar25 22 of its last 32) - actions execute, observations are lost, the model dead-reckons
(ar25 L3 terminal net-zero oscillation; ft09's last 9 actions swallowed; tu93's last 3 turns on a false
premise).

(iv) Noop-guard misses: essentially absent. board_changed=False on 333/4,261 (7.8%); immediate
same-action-after-noop repeats: 3 total (all sc25). Exact (state,action) repeats 573, but 451 are
sc25(290)+tu93(161) death-replay cycles. The harness known_noop guard fired and was understood (sk48 x7,
wa30 x9). The unguarded loops run through CHANGING states (tr87's rotating slot, vc33's net-zero valve
shuttle) - only carried memory can catch those.

(v) Format (B5): small. 0 markup failures in ~1,972 responses; 27 length-truncation turns (1.4%);
~74 sandbox tracebacks; ~8-10% of turns degraded per game. vc33 and tu93 runs were terminated by
analyzer HTTP read-timeouts, sk48 mid-productive-plan.

## Interventions (ranked)

#1 Move-budget / death-cadence protocol - attacks B1 (dominant).
   What: track GAME_OVERs per level; on death #1 inject "that was a level move-budget expiry - this
   level allows ~N actions per life, you have used k; deaths replay the level; probe on life 1, bank the
   plan, execute inside one life". After 2 same-cadence deaths, state N explicitly.
   Measured target: B1 = 46% of A-level actions; 72.5% of ALL fetched actions in death-ending attempts;
   7/8 A-level deaths were budget expiries; fixed cadence (27/51/100/128/130/~200) named in 2 of 8 games,
   planned against in 0.
   Recovery (INFERRED): parity on completed blown levels = +1.43/game measured ceiling (ka59 +3.54,
   bp35 +2.19, sc25 +3.84, ar25 +4.64, ft09 +13.0, vc33 +8.6 local pts); realistic 2-life-clear share
   +0.6-1.0/game, plus unlock value where the spiral IS the depth block (bp35 L2, sc25 L2, wa30, tu93).
   Repo: NOT covered (verified: no budget/cadence logic in graft_control/economy/emission). Extend
   graft_economy's per-level action counter - small new code + prompt line.

#2 WM persistence: fly TP_KEEP_NOTES_ON_GAME_OVER=1 (graft_throughput.py:107) + TP5_WM_FROM_REASONING
   (graft_emission.py) - attacks the amnesia -> state-loop bucket.
   Measured target: WM on 34% of turns, 0% bp35; dies at first death in every dead game; every
   state-loop begins after WM death; ~65% empty assistant text starves the stock harvest; ka59 re-derived
   the same physics facts 7-8x.
   Recovery (INFERRED): tr87 alone (mechanic+goal correct at action 12, scored 0) is +1.19 to +4.76
   local; cutting re-derivation shortens every death segment. Estimate +0.5-1.0/game.
   Repo: FULLY COVERED, unflown on this model (this smoke ran stock-duck).

#3 Action-economy scoring truth (graft_economy TP6, written 2026-08-31 from this smoke) - attacks B2/B3.
   Measured target: stock prompt never says actions are scored; ft09 painted six whole-board guesses on
   a scored level; sc25 spent 500 actions on a baseline-6 level; ka59 narrated the HUD to zero across 22
   one-action turns.
   Recovery (INFERRED): shares the +1.43/game ceiling from the prompt side; prompt-only, +0.3-0.7/game.
   Repo: COVERED - submission/_throughput_v1/graft_economy.py; needs a flight.

#4 Kill the 30s python-tool guillotine - attacks B5 + the dead-reckoning tail of B3.
   What: raise LOCAL_ANALYZER_TOOL_TIMEOUT 30 -> 90 for this model class (~3.5 s/action with per-step
   segmentation), and/or make timed-out action() results durable so the next turn sees what executed.
   Measured target: ~130 timeouts / 11 games; wa30 46/117 turns; ar25 22 of last 32 -> position-tracking
   collapse; ft09's final 9 actions swallowed; vc33's L3 clear happened INSIDE a timeout.
   Recovery (INFERRED): +0.2-0.5/game direct, plus de-noising every other measurement.
   Repo: env knob (this smoke set LOCAL_ANALYZER_TOOL_TIMEOUT=30); durability = small graft_throughput seam.

Not ranked: graft_explore stall tiers (Pack 4) would mechanically break the tr87/g50t loops, but #1/#2
attack the causes those loops grow from.
