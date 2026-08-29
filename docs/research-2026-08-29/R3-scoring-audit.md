# R3 — Scoring audit: code and recorded data vs the 08-29 handoff prose

Read-only audit (2026-08-29) of `reference/arc-agi-toolkit/arc_agi/scorecard.py`
(byte-identical to the installed `arc-agi 0.9.9`), `scripts/rescore_waves.py`, and every
recorded play in the tree. Analysis script: `R3-analyze.py`, raw output `R3-analyze.out`.

## Headline numbers

| quantity | value | source |
|---|---|---|
| Per-level score | `min(115, 100*(baseline/actions)^2)`, weight = level_index+1, then `min(weighted mean, 100*max_w/total_w)` | scorecard.py:168-171, 196-206 |
| Per-game score with several plays | **max over plays** | scorecard.py:239-241 |
| RESET | +1 action into the current level's bucket; a *full* reset opens a NEW play — only reachable after a WIN at eval | scorecard.py:701-704, 834-843; api.py:326-334; arcengine base_game.py:305-324 |
| Retry actions | all actions between two level-ups (retries after GAME_OVER, RESETs, the winning action) are attributed to the level eventually cleared | scorecard.py:479-484 |
| Cleared levels slower than human (Qwen3.8 shipped wave1, 21 levels) | **9/21 = 43%** ratio > 1.0; median 0.86, mean 1.13, Q1 0.61, Q3 1.56 | analyze.out |
| Cleared levels at the 115 cap | 11/21; **9/21 score < 100, 5/21 score < 50** | analyze.out |
| Score lost to efficiency (same wave) | per-play mean 2.529 vs cap 3.155 -> **0.63 pts = 19.8% of the cap** | analyze.out |
| Level distribution, Qwen3.8 wave1 (28 plays) | 0 lv 43%, 1 lv 39%, 2 lv 18%, 3+ **0%** | analyze.out |
| Level distribution, duck38-v12 smokes (39 runs, hand-picked games) | 0: 13%, 1: 51%, 2: 18%, **3+: 18% (7 runs)** | analyze.out |
| Counterfactual: every L1+ play gains +1 level (at cap) | 2.53 -> **7.53 local (x2.98)** | analyze.out |
| Counterfactual: every 0-level play clears L1 (at cap) | 2.53 -> **4.16 local (x1.65)** | analyze.out |
| Both | 10.48 local (x4.14) | analyze.out |
| Local (25 public) vs LB, same bytes (duck-38-v2) | local 2.53 per-play / 2.72 best-of-clones vs LB 1.29/1.74/1.45 (mean 1.49) -> **1.7-1.8x**, not 3-5x | analyze.out, submission-ledger.json |
| k=1 / k=2 / k=3 cap, mean over the 25 public N | 3.52 / 10.57 / 21.14 — handoff correct | analyze.out |
| "L1 on all games -> 2.82 max" | **not reproducible**; at cap it is >= 3.52 | analyze.out |
| "perfect efficiency on cleared levels -> 1.95 max" | recomputed **1.76**; number unsourced, and it contradicts "efficiency worth zero" | analyze.out |
| `rescore_waves.py --validate` | 224 rows, 0 mismatches — but vs a copy of the same code, and all 224 rows are single-play | rescore_waves.py:57-79 |
| Real-scorer cross-check | `arc_agi.scorecard._calculate_score` on 224+28+39 rows: 0 mismatches | R3-analyze.py `crosscheck` |

## Per-game efficiency loss, Qwen3.8 wave1

| game | lv/N | score | cap | lost to efficiency |
|---|---|---|---|---|
| cd82 | 2/6 | 8.586 | 14.286 | 5.700 |
| tu93 | 2/9 | 1.620 | 6.667 | 5.047 |
| s5i5 | 1/8 | 0.857 | 2.778 | 1.920 |
| ls20 | 1/7 | 1.921 | 3.571 | 1.651 |
| bp35 | 1/9 | 0.613 | 2.222 | 1.610 |
| lp85 | 1/8 | 2.007 | 2.778 | 0.771 |
| lf52 | 1/10 | 1.360 | 1.818 | 0.458 |
| su15 | 1/9 | 1.867 | 2.222 | 0.355 |
| 8 plays at cap (ar25x2, cn04, r11l, sb26, sc25, tn36, vc33) | | | | 0 |
| 12 plays at 0 levels | | 0 | 0 | 0 |
| **mean (28 plays)** | | **2.529** | **3.155** | **0.625 (19.8% of cap)** |

## Prose claims found wrong or unsupported

1. HANDOFF-2026-08-29 §1a "score == cap exactly … efficiency worth exactly ZERO" — false on the only
   Qwen3.8 wave in the tree (0.63/3.15 of cap lost; 9/21 cleared levels < 100). The handoff applied a
   *median* ratio (0.86) to a *per-level* cap.
2. HANDOFF §1b "clearing L1 on all 110 games -> 2.82 max" — not derivable; at cap it is >= 3.52.
3. HANDOFF §1b "perfect efficiency -> 1.95 max" — unsourced; data gives 1.76.
4. HANDOFF §1a "validated bit-exact against the real scorer (224 rows)" — the validation compares two
   copies of the same formula on single-play rows, never `arc_agi`.
5. HANDOFF §1d "ratio is 3-5x" — external; our identical-bytes local/LB ratio is 1.7-1.8x.
6. MEMORY / two-level-wall "252 plays, zero 3-level games; the LLM lane has NEVER cleared 3 levels" —
   seven 3+-level runs (one 4-level) exist in `scratchpad/multirole_corpus/` from 08-18..20.
7. two-level-wall.md:183 "9 of 21 cleared levels already hit the 115 cap" — 11/21.
8. The "+0.91 per full win" figure assumes a 110-game LB denominator; the repo's own contract memory
   says the LB shows the ~55-game public half (+1.82).

## Unverified (no evidence in the repo)

Other teams' 3-5x local/LB ratios; the AERA "non-intelligent strategies" claim; hidden games' level
counts/tags/difficulty; whether the LB denominator is 110 or 55; the origin of "1.95" and "2.82".
