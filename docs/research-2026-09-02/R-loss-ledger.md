# loss-ledger — final report (2026-09-02)

## 1. Findings (M = measured, I = inference)

**Scoring formula verified (M).** arc_agi/scorecard.py:168-200: per-level min(115, 100·(baseline/actions)²), weight = level index (1..n), game score = min(weighted mean, weighted completion share). Recomputed all 200 final_score values from actions_per_level with 0 mismatches. Level k is worth k/(n(n+1)/2); L1 on a 6-level game caps at 4.76, the 3rd level at 14.3 — depth on already-started games dominates.

**Outcome table (M; 8 phases = 200 runs; 5 stock phases = 125 runs).** Stock: 1.176 lv/game, local score 5.22, per phase 1.04–1.36 lv, 6–8 zero games. Classes over 5 stock draws: NEVER = dc22 only; CAPPED = lf52, s5i5 (always 1); HIGH-VAR (spread ≥2) = ft09 [3,2,3,4,2], sb26 [1,4,6,1,1], tu93 [2,2,4,2,2], re86 [3,0,4,2,3], sc25 [1,2,0,0,3], cd82 [2,1,1,1,0]; 16 games flip 0↔1 or 1↔2. Six games (ft09, sb26, vc33, re86, lp85, sc25) carry 88 of 130 score points.

**Consistency ceiling is an order-statistic mirage (M).** Every game at observed max → 1.92 lv/game (+0.74), score 11.10. But E[max of 2 full draws] = 1.48 (+0.31); E[max of two half-window plays] = 1.23 (+0.05). No replay policy inside 2.2h harvests the +0.74.

**Timeline (M, 75 stock runs with event logs, 88 clears).** Clears per 15-min bucket: 24, 19, 9, 13, 11, 4, 3, 3, 2 → 74% of clears in the first hour, 14% in the last hour. Median last clear at 2820 s; on average 61% of the window follows the last clear. 60.5% of all actions (4901/8101) land in the bucket that never clears. 74 GAME_OVERs, 100 RESETs in 75 runs.

**The loop (M).** 55.4 model calls/game, 143 s mean / 113 s median between calls, 1.95 actions/call. 51.6% of calls take zero actions (46% analysis-only python, 4% no tool call, 2% python errors). Calls generating <2k chars still have a 72 s median gap (prefill/queue floor); decode ≈16 tok/s per stream, 321 tok/s aggregate. Per-call budget context_budget_tokens: 31744.

**Cross-turn memory is effectively dead (M).** tool_agent.py:1105-1112: the carried note updates only when the assistant emits `World model:`-style labeled lines; 11.2% of assistant texts do (371/3298). Note changed in 23.6% of consecutive calls; at game end it is a median 30 calls stale (48/75 games >20 calls stale); mean 458 chars. Raw history kept = median 17 messages (~5-6 turns). 17% of all calls (24% of calls #31+) are no-action calls that re-scan history/transitions in Python to rebuild what was forgotten.

**Stuck buckets (M).** 72 uncleared level attempts: median actions/human-baseline = 1.00; 35/72 have fewer actions than the human baseline; frame novelty 0.90 (not looping). Cleared levels: median baseline/actions ratio 1.21, 64% at/above the cap.

**Zero-level failure modes (M from transcripts + obfuscated environment source, partial):** m0r0 ×2 — believes the bar is a "decorative step counter"; source m0r0.py:723 `if self._action_count > 150: lose()`; correct model found at action 296/296. g50t ×3 — treats red U as static wall, BFS "no path"; its own dump shows "red changed at frame 21" and is not integrated; source moves a sprite every 2 actions and loses when it leaves. tn36 ×2 — clicks only T-marks (182 clicks, all 32 configs), never the real target. tr87 ×2 — assumes P-strip must equal S-strip; source is a rule-panel mapping. sk48 ×3 — segmentation returns HUD duplicates; tracking corrupts; per-level move budget hit. dc22 ×2 — exhaustive search under a wrong state abstraction; the 3rd draw cleared L1 at action 90. bp35, cd82, ls20, sc25, re86 — one wrong goal hypothesis pursued to the clock (sc25: two GAME_OVERs from a ~47-action timer).

**Counterfactual arithmetic (M, 5 stock phases):**
| | lv/game | score |
|---|---|---|
| observed | 1.176 | 5.22 |
| (a) consistency at observed max | 1.920 | 11.10 |
| (b) +1 level on every run with ≥1 (parity / ratio 1.21 / 0.5) | 1.904 | 11.65 / 12.05 / 7.30 |
| (c) L1 on every zero-level run | 1.448 | 6.27 |
| (c′) L1 on dc22 only | 1.216 | 5.41 |
| efficiency perfect, no new levels | 1.176 | 6.25 |
| +2 levels on runs with ≥1 | 2.632 | 20.55 |

(b) is the only bucket ≥ +0.5 lv/game that isn't a mirage; it corresponds to the stalled-in-the-next-level failure, not to zero games and not to efficiency.

## 2. Ranked step-change candidates

**#1 — Rebuild the cross-turn memory channel (dead note + 6-turn window).** The model forgets tested hypotheses/state models after ~6 calls, re-derives them (17-24% of calls are history re-scans emitting no action), and pursues wrong models to the clock. Ceiling bounded by (b) = +0.73 lv / +6.4 score locally; cannot show what fraction memory rot accounts for vs plain misunderstanding (inference). Contradicting evidence: the full transition list is available in Python (nothing truly lost, only expensive); clear-rate decay over time is equally consistent with "later levels are harder". Cheapest decisive experiment: harness-side forced note (extract entire assistant text + last tool result into the carried note every call, or mandatory summary call every N turns) — hours of code, then one 25-game 2.2h run (~2.5 GPU-h, 0 slots). Pre-registered rule: ≥1.70 lv/game (stock 1.04–1.36 across 5 draws) and ≥3 fewer games with stuck-level actions < baseline; below 1.5 → dead.

**#2 — None other qualifies. Rejected by arithmetic:** throughput (halve the 51% idle calls / 70 s floor): ≤ +0.16-0.27 lv/game for a 2× action budget; zero-level games → L1: +0.27 lv / +1.05 score max; efficiency: +1.0 score, 0 lv; replay/consistency harvesting: +0.05 lv. cstl's 7.51 would need ~+1.5 levels on every started game (20.5 local) — a comprehension change, not a budget change.

## 3. Not verified
- Game-side requirements for tn36, tr87, sk48, dc22, bp35 inferred from obfuscated source + transcripts; not played offline.
- Memory rot vs inherent difficulty not separated by any measurement.
- Live-vs-local transfer of (b); everything is on the 25 public games.
- Stock-only numbers use 5 phases (results dir holds 8 phases incl. ab-tp10).

Notes/scripts: scratchpad/search/loss-ledger/ (table.out, timeline.out, calls.out, tput.out, wmage.out, stuck.out, cf.out, failmodes.txt, kaggle/<kernel>/ raw events).
