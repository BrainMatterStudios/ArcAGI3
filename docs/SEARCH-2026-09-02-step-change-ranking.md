# STEP-CHANGE SEARCH 2026-09-02 — consolidated ranking (deliverable)

Method: four memory-blind fresh-context researchers (field-delta, loss-ledger, protocol, model-axis;
docs/research-2026-09-02/R-*.md) working from primary data only, then a fresh-context adversarial
judge (J-judge.md) that verified their claims and resolved their conflicts. Bar: >= +0.5 lv/game
locally, >= +1.0 LB (set by the 09-02 instrument finding: identical stock bytes vary ~1.2 lv per
game, so nothing smaller is measurable with the instruments and ~60 slots that exist).

## Constraints discovered today
1. Local A/B minimum detectable effect ~+0.5 lv/game; live per-draw sd ~0.3 (docs/EXPERIMENT-2026-09-02-ab-instrument-control.md).
2. Weekly GPU quota (60 h) EXHAUSTED at 15:50 UTC 09-02; prior block cleared at Sat 00:00 UTC ⇒ no commit
   runs until 2026-09-05 00:00 UTC. Unrelated kernels on the account (jed-*, working-note-*) consumed part of
   this week's quota. Only already-committed versions can be submitted on 09-03 and 09-04.
3. Leaderboard-monitor series are running maxima, not draws (judge F1) — never read them as noise estimates.

## Ranked candidates (judge-verified)
1. **Byte-copy of keithtyser V14 `duck-qwen3-8-flash-next-nvfp4-mtp` (svid 346399440).** Stock June agent
   code + Flash-Next NVFP4 under a specific serving regime (8 vLLM seqs, KV 5 GiB, MTP-3, async+chunked,
   batched 8192, prefix caching off, pinned vLLM + PLE-FP8 patch, analyzer 32768/MAX_OUTPUT 0). Live
   evidence: three independent one-version forks drew 2.80(2.73), 3.22, 3.38 on first exposure; expected
   live of a copy ≈2.7 (90% 2.1–3.3) vs our 1.71/1.72/1.88. The mechanism is NOT throughput (his aggregate
   249 tok/s is half ours) — it is the per-call regime: 53 long, un-truncated reasoning calls/game (3.4k
   chars) vs our Flash-Next flight's 125 short ones (1.5k chars, 24576/4096 caps). Staged byte-identical
   at submission/_keith_copy/push (ATTEST.json, code-cell sha256 de24036a…); push refused on quota.
   PLAN: push at 2026-09-05 00:00 UTC, commit ≈2.5 h, attest per the judge's line list, fly the 09-05 slot
   (a submission any time on 09-05 UTC counts). Reading rule pre-registered in J-judge.md.
2. Knob isolation on top of V14 by telemetry (reasoning chars/call, requests/game), not by score — after it flies.
3. ~~protocol-lite Stage-0 kill test~~ — RUN 09-02 on Modal (27B on H100, ~2.5 GPU-h, 0 slots): **LANE DEAD** under the pre-registered rule (1 of 3 green; tn36 12/12 genuine mechanics model, sk48 0/41 and cn04 0/18 with NO candidate file ever emitted — the model spends the entire 40k-token call inside thinking or emits EOS mid-thought; replicated under a coordinate-explicit encoding). Report: docs/research-2026-09-02/S0-stage0-kill-test.md.
Dead (with the counter-evidence on record): memory-channel rebuild (same dead note exists in the 3.3 config);
reasoning_effort medium (wrong direction); model swaps (no servable model beats the class; best-of-N plays
impossible — arc_agi creates one play per game at scorecard open); TP9 lever (demoted by the crossover);
throughput/action budget (2× actions ≈ +0.15 lv; our Flash-Next smoke did 2.85× actions for a worse score).

## Stage-0 result (09-02 evening) and what it means for the regime axis
The 27B CAN certify a backtest-green executable world model on a simple game (tn36) but never
finishes deliberating on the harder ones within 40k output tokens, under either grid encoding.
The binding constraint is thinking discipline/termination, not perception and not the harness. This
is the same axis the serving-regime finding points at (long un-truncated calls help; but unbounded
thinking on hard mechanics never terminates). Original-work candidates that follow from it: forced
staged emission (hypothesis → code → test in separate short calls), and thinking-budget control per
call stage — to be tested by telemetry on the Modal Flash-Next rig, not by 25-game score.

## Slot plan
- 09-03 (tonight) and 09-04: redraws of committed versions only. Recommended: Flash-Next exact redraw
  (runner scripts/submit_flashnext_20260903.py; grows n on the field's base model under our serving,
  expected ~1.9) over the 27B stock redraw (n=2 already, ~1.7). Ahmed's call; nothing launched.
- 09-05: keith V14 byte-copy (after the 00:00 UTC quota reset + commit + attestation). Runner to be written
  from the attested svid/hash the moment the commit completes.
