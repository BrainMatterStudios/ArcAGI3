# loss-ledger-2 — the keith V14 regime, 5 runs × 25 games (125 game-runs), 2026-09-06

Data: Kaggle commit arc3-keith-copy v4 (42 lv, 10.42) + Modal arms keith base (36), yield180 (34),
kv10 (43; different serving profile, flagged), retry-not-engaged (40). Pooled 1.56 lv/game; kv5-only
1.52. Scoring recomputed on all 125 runs, 0 mismatches. Scripts/outputs: scratchpad/search/loss-ledger-2/.

Q1 outcomes: levels 0:20 1:56 2:24 3:12 4:11 5:1 6:1. NEVER 0; CAPPED 5 (cd82, dc22, lf52, wa30 at 1;
vc33 at 3); HIGHVAR 9. Six games carry 64% of score. Consistency-at-max +0.80 is a mirage (E[max of 2]
+0.33). **12/25 games hit the same wall level in ≥4/5 draws; 12/25 modal walls were never passed in
any draw — failures are deterministic, not variance.** Hazard L1 .84, L2 .47, L3 .51, L4 .52, L5 .15.
Q2 timeline: clears per 15 min [41,27,29,26,27,16,8,12,8] (flatter than the 27B base); median last
clear 3623 s; 62.6% of actions in never-clearing buckets (novelty 0.86, not looping); 200 GAME_OVERs
(171 on the never-cleared level); 21/25 games have per-life timers/budgets; 50.7% of actions sit in
lives that ended in GAME_OVER; marginal clear rate after turn 30 = 0.123 lv/game per 10 turns.
Q3 loop: 50 turns/game, 55 calls, 141 actions, 159 s/turn of which 121–131 s is server queue; 50% of
calls act, 46% analysis-only python; late-game non-action 48%. Note stale median 16 turns; ~7 turns
raw history. **Where the 32k goes: system+tools 15%, current frame 22%, history 63% — replayed prior
reasoning alone 24% (tool_agent.py:1892). Real prompt plateaus ~22k vs the 31,744 budget because
_estimate_tokens is chars/3 of the JSON payload (tool_agent.py:462-467), ~1.45× too high: ~30% of the
window is never used.**
Q6 thinking termination: finish=length 0.36%, no-tool-call 0.78%; reasoning is SHORTER in stuck
tails; zero-level and ≥2-level runs have identical no-action shares. **Staged emission / thinking
budget is aimed at <1% of calls — DEAD. The model acts on wrong hypotheses.**
Q4 failure modes (105 started-stuck tails + 20 zero-level): WRONG-HYPOTHESIS 37, ANALYSIS-PARALYSIS
25, PROGRESSING 17, TIMER 14, BUDGET-CAP 14, LOOPING 8, OTHER 8. Zero-level games fail identically
across draws (sk48 rod rules; sp80 timer lives; g50t plug believed static; tn36 brute-forces 32 peg
patterns; m0r0 goal unknown; tr87/bp35 paralysis). Recurring: "reached my goal state, no clear, then
thrash"; correct rule written in the note then abandoned (~28/105); the knowledge exists in SOME
draw for 13/25 walls.
Q5 counterfactuals: (a) consistency +0.80 mirage; **(b) +1 level on started runs +0.83 lv / +8 pts —
the only bucket ≥ +0.5**; (c) L1 on zero runs +0.16; (d) efficiency 0 lv / +1.15 pts. (b) by class:
comprehension +0.42, analysis-paralysis +0.20, time-limited +0.25, budget-cap +0.11.

## Ranked candidates
**#1 Break the next-level wall: hypothesis-enumeration + discriminating-probe protocol** (harness-side
rule on uncleared levels: list ≥3 candidate mechanics/goal predicates; execute a ≤3-action probe that
discriminates them; on "goal state reached, no clear" force predicate re-derivation). Ceiling
+0.42..+0.62 lv/game. Risk: failures are consistent across draws, so the same reasoning may reproduce
the same wrong model. Experiment: one 25-game rig wave (~$9). RULE: ≥2.05 lv/game (kv5 run means
1.36–1.68, sd of run means ~0.14) AND ≥3 of the 12 never-passed walls passed; <1.8 → dead.
**#2 Turn throughput ×2** only in combination (0.123 lv/game per 10 turns ⇒ +0.5 needs ~90 turns/game;
time-limited class caps at +0.25). NB: on the Kaggle box the KV lever is closed (OOM at 8 GiB).
**Riders (free capacity):** fix the chars/3 estimator (+30% usable window); drop replayed reasoning from
history (24% of prompt → ~2× history depth); life-burning on timer games (+0.11); post-GAME_OVER dead
time (6%).
Rejected: consistency/replay; zero-game L1; efficiency; thinking budget; fresh-mind retry (12/25 walls
fail identically in 5 independent draws).
Not verified: failure classes are sampled-turn judgments; no mechanic checked in the engine; live
per-game breakdown unavailable.
