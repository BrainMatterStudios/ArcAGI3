# Validation audit of the 09-09 → 09-11 experiments (written 2026-09-12)

Scope: every result recorded since the cadence arm (09-09) — cadence, effects graft (fx), budget 3x, ctx12, ctx16, nr16, the "buy-calls" sizing, the reactive-loop census, the triage bound, the model-axis closure, the A2/Stage-1 closure, the live ledger — plus the verdict statistics they all rest on. Method: one adversarial verifier per claim recomputing from `offkaggle/results/*` raw files (results.json, requests_shim.jsonl, metrics_*.prom, transcripts, artifacts), two cross-cutting skeptics (instrument, statistics), and top-level spot checks of the highest-impact corrections. Raw agent returns: `verification-journal-extract.md`.

## 0. Headline

**Every level count and every primary number reproduces from the raw files.** No result was fabricated and no verdict flips direction. What does NOT survive is a layer of framing on top of the numbers: a wrong dispersion estimate that inflates every "sd" headline, two undisclosed mid-wave server restarts, a mislabeled base row, a "first win" that was not first, an "elasticity decays" claim that is a base-choice artifact, a "0 forward simulations" census that is wrong in the strong form, and two lane closures (model axis, throughput family) that rest on mistaken arithmetic or were contradicted by the author's own next wave.

| claim | verdict | what changes |
|---|---|---|
| budget 3x: 56 lv, 169.8 calls, elasticity 0.21 | CONFIRMED with corrections | "first outright win" false; table mixes bases; restart at +297 min; 41/56 clears by 7,920 s (so base waves are consistent and the extra clock bought ~15) |
| ctx12: 12 lv, catastrophic | CONFIRMED with corrections | server died at +75 min (25× HTTP 500, 398 s gap; 5.9 % of clock); stated mechanism wrong (history room ≈2,950 real tokens ≈1 turn, not 7,349) |
| ctx16: 21 lv, dead | CONFIRMED with corrections | 1,908 KV preemptions (10-30× band) unreported; "smooth gradient" over-claimed from 3 points |
| nr16: 20 lv, dead | CONFIRMED with corrections | "base" row is the conc-6 cadence arm; graft restored +0.6 turns not +7; "turn depth refuted" is unsupported (never exercised) |
| pooled base 39.33 sd 2.34 | PARTLY WRONG | pool = 3 base-ish + 3 yield900 draws incl. 2 Kaggle commit runs with no per-game data; honest wave sd ≈ 3.2-3.5 |
| sizing closes the buy-calls family | PARTLY WRONG | mixed anchors; "ceiling settles it" false; corrected best stack 44.0 (or 42.7); closure hinges on elasticity 0.2 vs 0.4 which KV10 cannot resolve |
| 0 forward simulations in 2,047 calls | PARTLY WRONG | ≥30 hand-verified forward simulations; builder bug drops 1,288 + excludes 4,573 calls; strong form dead, weak form ("mostly reactive") stands |
| triage: AUC 0.94, bound +11 % | PARTLY WRONG | AUC is target leakage (incremental-target AUC ≈ 0.5); at elasticity 0.21 the bound is −6 % to +9 %; family still not a step |
| model axis blocked on scale | PARTLY WRONG | 180 B on disk not 131 B; 360 GB includes a 102 GB CPU-offloadable PLE table; attention weights already bf16 in the NVFP4 checkpoint; H100 $3.95/h; a 2-4 GPU attempt is tens of dollars |
| live ledger | CONFIRMED with corrections | base draw #4 = 2.73 unrecorded; sub 55543514 missing entirely; yield900 "dead" decided by 0.01 and every same-regime comparison points +0.1-0.2 lv/game |
| A2 closed / Stage-1 green | PARTLY WRONG | counts reproduce; "loop not brain" inference does not (directed calls were fresh 2-message contexts; live prompt ≠ offline prompt) |
| cadence + fx: "engaged-and-negative", class closed | PARTLY WRONG | cadence is −1.4 sd at the honest sd; confounded with a 4.5× clock cut and sequential batching; "throughput family closed" contradicted by budget 3x the next day |

## 1. Statistics — the correction that touches everything

* The "pooled six-draw base 39.33 (sd 2.34)" is `docs/research-2026-09-08/R-loss-ledger-3.md:3-5`: Modal 09-02 keith 36, Modal 09-03 keith_retry 40 (retry graft installed, fired 5×), Kaggle base commit 42, Modal yield900 41, Modal yield900-cb2 40, Kaggle yield900 commit 37. Two members have no per-game data on disk (`scratchpad/search/loss-ledger-3/cache.json` no longer exists). Half the pool is the yield900 knob.
* sd 2.34 is an n=6 point estimate; chi-square 95 % CI [1.46, 5.74]. Direct per-game decomposition on the flat draws on disk (within-game draw variance summed over 25 games; rms paired-difference sd 0.99/game) gives a 25-game-total sd of **≈3.2-3.5**. The rival "≈6" in `R-loss-ledger-3-NOTES.md:27` is a mis-derivation (cross-sectional per-run sd × 5).
* Consequences at sd 3.5 (×√(7/6) for arm-minus-pool): one wave resolves ±7.4 levels at 95 % (±9.8 at 80 % power); two waves ±5.6. The 48 bar is +2.3 sd; the 45-47 "redraw" band is inside noise. Across ~15 arms read against one bar with no multiplicity control, P(≥1 spurious step candidate) ≈ 21 %.
* Robust at any sd up to 6: budget 56 (+4.4 sd; paired t = +3.5), ctx12 12 (−7.2; t = −7.1), ctx16 21 (−4.9), nr16 20 (−5.1). Fragile: cadence 34 (−1.4 sd), fx 35 (−1.2), probe/carry 41 (+0.5). "Not flat, WORSE" is over-claimed for cadence and fx; "dead as a step" stands for all of them.
* "Elasticity decays 0.40 → 0.21": KV10's 0.40 is computed against its own-session base (36) while budget's 0.21 is against the pool (39.33). On a common base the two read 0.20 and 0.21 (KV10 ± 0.21 delta-method SE). No decay is demonstrated. Budget within-wave (41 by 7,920 s → 56 at 23,760 s) reads 0.17 ± 0.07.
* Live: with per-draw sd ≈ 0.8 the 3-draw mean has SE 0.46; the yield900 rule's step/dead lines (4.0/3.6) are under one SE apart; the 3.59 verdict was decided by 0.01 and has a 95 % CI [2.68, 4.50]. On identical bytes the v4 base reads CV 0.12 (sd 0.35, n=3) and yield900 CV 0.28 (sd 1.00, n=3); the quoted 0.25 is the mixed pool.

## 2. Instrument — two undisclosed restarts, one waived rule

| wave | fault | effect on verdict |
|---|---|---|
| budget 3x (20260910-192514) | container hit the rig's own 6 h lifetime cap at +297 min (it was 66 min old at wave start from the smoke); 215 s outage, 24× HTTP 500; tail 99 min ran at median e2e 104 s (below the 120 s cadence gate) → ~+156 surplus calls. `metrics_before.prom` is the smoke container, so the summary's VLLM row (661 requests, e2e 181.9 s) is a two-container subtraction and unreadable. | bias ≤ +0.5 level; verdict unchanged. Fix the runner: refuse to launch a wave whose clock cannot fit inside the remaining container lifetime, or restart the container first. |
| ctx12 (20260911-075839) | container died at +75-77 min (25 simultaneous HTTP 500 "Server has lost track of input", 398 s gap, new vLLM process at +79.7 min; not the reaper, not idle scaledown; cause not in local files). 1,669 pre-restart calls have no server telemetry; "preemptions 0" is post-restart only. ~464 s/game (5.9 %) lost. | 11 of 12 clears happened by +43 min and none in the 33 pre-restart minutes after that; verdict unchanged. |
| ctx16 / nr16 | 1,908 / 1,652 KV preemptions (0.9-1.1 per request) vs 0.05-0.24 on every 32k wave — a scheduler thrash specific to the 16k window under the kv5/c8 profile. Costs ≈17 calls/game and inflates decode ≈25 %. Not mentioned in either write-up. | levels move by ≈+3-4 if corrected; verdicts unchanged. |
| all waves | the pre-registered per-run VOID rule (request errors > 0, preemptions > 0) fires on 25/25 runs of every Modal wave because "errors" are the runner's own end-of-clock ReadTimeouts and preemptions are routine at 32k. It is inoperable as written and has been applied by discretion. | rewrite it: 5xx > 0 or a process restart = VOID; preemptions per request > 0.5 = flag. |

## 3. Per-claim corrections to write back (verbatim targets)

**offkaggle/REGIME_WAVE_STATUS.md**
1. Budget 3x RESULT: "first outright win in this campaign's rig history" → false; ft09 won 6/6 in the KV10 wave (`STATUS:369`; `offkaggle/results/20260903T0553-kv10-keith/.../results.json`: won, 108 actions, 7,686 s). In this wave ft09's 5th clear was at 8,012 s and the win after 9,367 s.
2. Budget 3x table: levels use the pooled base but calls (55.8), actions (154), score (6.40), zero-level (2), stuck-at-one (14), 5-6-level (0) use the single lowest 09-02 draw. Modal-pool means: 53.3 calls, 124.9 actions, 7.85 score, 5+-level games 0/1/1/1. Headline "BUDGET CONVERTS" uses the ≥60 band language; 56 is in the pre-registered 46-59 "partial" band.
3. Budget 3x: add the dating — 41 of 56 clears by 7,920 s (+0.71 sd vs pool; calls in the first 7,920 s = 53.0/game), 15 after. Add the 6 h reaper restart and mark the VLLM row unreadable.
4. ctx12 RESULT: add the +75 min server death; replace "history budget ~7,349" with the measured ≈2,950 real tokens (the stock trimmer budgets chars/3 of the JSON payload including the base64 image, `tool_agent.py:462-467`); the collapse is routine thrash (median 5 messages/call, 16 % of mid-game calls sent with zero history), not rare 10k completions (2 of 2,709). "base 20,061" is the mean prompt, median 21,421.
5. ctx16 RESULT: report 1,908 preemptions; "smooth gradient" → "linear 32k→16k then collapse at 12k; three single draws cannot distinguish". The "longest observed completion 10,439" premise was already false on disk (base 10,495, fx 10,673, carry-kill 13,800).
6. nr16 RESULT: replace the "base (32,768)" row with conc-28 numbers (median prompt 21,421, e2e ≈145 s, 55.8 calls, 2.76 actions/call). The graft restored 3.32 vs 2.72 retained assistant turns (not 12.5 vs 5.2); reasoning is ≈1/3 of a retained turn, not ≈60 %. Delete "third pre-registered outcome: turn depth was NOT the cause" — turns were never restored, so the arm is uninformative on that. Keep: 25/25 games show +46 % completion tokens; causal chain to −28 % calls confirmed.
7. Sizing: re-anchor at 55.8/39.33 → best live-legal stack 44.0 (KV +30 %, MTP-8) or 42.7 (KV +16 %, the KV^0.56 scaling KV10 actually showed); "7.7 s / 2.6×" → 8.1-8.7 s / 2.3-2.5×; delete "the ceiling settles it" (perfect MTP-8 projects 48.8-52); acceptance figures are from the 09-09 conc-6 wave. State that the closure holds at elasticity 0.2 and fails at 0.4, and that KV10 cannot tell them apart.
8. Cadence RESULT: 24 of 25 games averaged 51.6 calls (tr87 ran alone at conc 1 with 181 calls); the −5.33 is concentrated in batch 1 (6 games, 4 levels); "throughput/cadence family closed as a step" was contradicted by budget 3x the next day — replace with "cadence at a 4.5× shorter clock is not a step".
9. fx RESULT: "24/25 games saw a 7+-line block" → 16/25; the model quotes the effect table in 7.0 % of turns (thinking only, never in code); prompt cost +487 tokens/call vs the +250 pre-registered.

**docs/research-2026-09-10/R-what-our-agent-actually-does.md** — retract the strong form ("exactly zero", "never models live", "not in the live repertoire"). ≥30 of the 2,047 calls forward-simulate with a hand-written transition function (tu93, m0r0, lp85, lf52, dc22) and several verify the prediction. Fix `src/modelaxis/build_corpus.py:154` (`steps[idx]` aligns the idx-th call to the idx-th turn header; wrong whenever a turn has >1 call — 35 % do; 1,288 cleared-level calls dropped, 4,573 continuation calls excluded; true population 3,350). Commit the classifier. "11 waves" → 19 dirs / 9 arms, 72 % non-base.

**docs/research-2026-09-10/R-triage-family-bounded.md** — "prerequisite passes" is refuted: on the target a controller needs (≥1 more level after K) no eval-legal feature beats AUC ≈ 0.52. The keep-top-N sweep has no committed code. Units: elasticity is in calls, the simulation applies it to actions (0.34 / 0.14 in action units). At 0.21 the in-range rows read −6 % to +9 %. Family still not a step.

**memory `arcagi3-model-axis-blocked-on-scale.md`** — 180 B on disk (131 shards; 360 GB / 2 B), 120.8 B in routed experts, ≈7.3 B active; 360 GB includes the 51.2 B-param PLE n-gram table that serving already CPU-offloads; non-PLE bf16 257.5 GB (4×80 GB); the served NVFP4 checkpoint keeps every attention/linear-attention tensor in bf16 (78 GiB non-PLE, fits one 96 GB card) so QLoRA-style attention-only LoRA does NOT need an unquantised base; Modal H100 $3.95/h → 8 GPUs $31.6/h; a 2-4 GPU attempt is ≈$40-100. The pre-registered stop rule ("M2 not passed in two sessions") never fired because M1/M2 were never attempted. Blocked on unproven software (PEFT over modelopt-NVFP4 experts; runtime LoRA in the pinned vLLM fork), not on scale.

**memory `arcagi3-a2-workspace-closed.md` / `PREREG-a2-workspace-arm.md`** — counts stand (0/358 verifier calls, and 0/359 in the control; 24/24 length-capped inside `<think>`). Delete "failure is the LOOP not the brain": all 24 directed requests were fresh 2-message contexts (n_messages=2, has_tools=False), so play-history contamination is impossible; the live directed prompt differed from the offline one (≈170-word system vs ≈600, 12 vs 20 transitions, 60-cell lossy cap, one-line feedback, 3 calls vs 20). Pass-1 prompts were already inside the "Stage-1 window" (8.1-18.9k / 13.9-24.7k) and still produced no code. Pass-2 GPU time 5,857 s not 5,016. Cost row in `GATE-polyphony-clock.md` is the carry75 arm (83,094 tok/game); true stock is 73.6-74.2k → one verified model = 88-89 % of a stock game's decode. The reasoning in kill2/kill3 repeatedly says "use backtest" and then emits python — intent present, tool call never emitted; cause unresolved.

**memory `arcagi3-queueing-not-compute.md`** — "throughput/cadence family CLOSED as a step" is stale (budget 3x). `time_remaining_seconds` appears ≈29 times in 25 transcripts, not every tool result.

**docs/submission-ledger.json** — row 56133282: status complete, public_score 2.73, verdict "in band (2.0-4.7); base family n=6: 3.25/2.58/4.31/2.45/4.01/2.73 mean 3.22 sd 0.78 SE 0.32". Row 56111215 status → complete. Row 55634118 → complete/empty. Add missing sub 55543514 (08-16, duck-38 first flight, 1.29). Header generated_utc/counts are stale template values.

**memory index** — the field picture ("nothing public ≥ 4.33", 09-08) is stale: full LB 2026-09-11 = 2,976 teams; our best draw 4.31 is rank 41; 19 teams ≥ 5.0, 6 ≥ 6.0, 5 ≥ 7.5, top 11.04.

## 4. What is solid and should be built on

* Budget converts sublinearly: same games, same boot, 41 → 56 levels from 1× → 3× clock; elasticity ≈0.17-0.21 in calls (≈0.14 in actions). 9 of the 12 never-passed walls survived 3× calls and 4× actions (only dc22, sb26, vc33 fell); the extra levels came from games already progressing.
* Context capacity is load-bearing (12k and 16k catastrophic; paired t ≤ −4.6); content is not (six flat arms). Stripping retained reasoning inflates completions +46 % on 25/25 games.
* The served brain is a 180 B MoE whose attention weights are bf16 inside the NVFP4 checkpoint.
* The live base family: n=6, mean 3.22, identical-bytes sd 0.35 (v4) / 1.00 (yield900).
* Rig → live mapping across the two configs that exist: ≈2.0-2.2 LB per rig level/game.
