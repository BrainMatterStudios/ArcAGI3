# RESULTS 2026-08-22 — testing campaign (all 08-21 hunt ideas)

Ahmed's go: "proceed with testing all ideas and hypotheses identified."
Everything below ran zero-GPU overnight (5 parallel streams); pre-registered
thresholds where applicable. Artifacts preserved in
`scratchpad/testing_20260822/` (falsifiers/, triage132/, human_fixtures/,
replay-mining JSONs — copied out of volatile tmp, which evicted two reference
trees mid-campaign; durable June-stock pin now at
`scratchpad/bundles/june_stock/`).

## Verdict table

| Idea | Test | Verdict | Next step |
|---|---|---|---|
| Expect-queue rider | Replay 652 batches vs halt rule (bar ≥5%) | **PASS 14.53%** saved (m0r0 26%, sk48 25%, dc22 22%); 72% of halted remainders still had effect | Build with **halt-and-return-control** (cheap re-plan), NOT hard-discard |
| No-op guard extension | Replay 629 no-ops vs repeat-predicate (bar ≥60% catch) | **KILLED — 0/265 caught**; 95/265 first-executions; churn vetoes rest; ceiling 16.6% | Drop. No slot, no build |
| Multi-tool-call turn erasure | Frequency count | **0 occurrences** / 3,482 responses | Drop stub patch. REAL eraser found: read-timeout reverts 11.9/1000 requests → preserve-history-on-request-error is the fix to build |
| Triage-at-55 | FN re-derivation at TRUE 132-min geometry (gate ≤5% FN) | **FAILS: 14.3% FN (3.8 wave), 26.9% pooled**; old 3% was the 60-min-box artifact | **Archetype rule replaces it**: kill zero-level at AVATAR 60m / MIXED 70m / CLICK never (frame-0 available_actions dispatch). 0 FN observed (n=103), ~49 wh reclaimed/run (20% of budget). `triage132/trigger_rule.json` |
| Yield-resume carryover | Built `submission/_yield_carryover/` | **11/11 tests green**; digest ≤1,500 tok worst case; 3-slice cap; duplicate-digest bug caught+fixed; ENVELOPE.md: cannot extend duration | Smoke as arm (digest-only first, then cap) |
| reasoning_effort=medium | Built `submission/_effort_medium/` | **13/13 green**; thread-local dead-completion retry forcing tool_choice; doesn't collide with stock markup recovery | **First smoke candidate** (cheapest read: dead-completion count collapse) |
| Truthful telemetry | Built `submission/_truthful_telemetry/` | **9/9 green**; RESET verified executing end-to-end through real step_env with zero code change; ~265 tok/request total | Pre-registered arm, read per-level actions/RESET usage not score |
| Mechanical bugfix pack | Built `submission/_bugfix_pack/` (8 patches) | **7/7 green**; runtime-state cap measured: 129MB→0.5MB and 1.24s→6ms per action at n=2000 (hunt claim reproduced exactly); animation() truth recovered from surviving .pyc: bbox/changes ARE preformatted strings | Mechanical arm (flags 2-6) can ride any smoke; SAFETY.md has grouping + the 6↔7 pairing caveat |
| Conc-28 throughput + MTP | Built `submission/_serving_lab/` kernel (committed b578f50) | Ready; NOT pushed (permission-blocked) | Ahmed: `kaggle kernels push -p submission/_serving_lab` (~2.5h GPU; self-limiting on quota) |
| Parquet telemetry channel | — | **NOT BUILT** — awaiting Ahmed's rules/honesty-gate call | Decision pending |
| 08-23 safe default | Runner built + mock rc=0 + re-attest clean | Launch permission-blocked | Ahmed: `nohup bash -c 'exec python3 scripts/submit_38_20260823.py >> logs/duck_38_20260823_runner.log 2>&1' &` |

All 40 unit tests re-run and verified green by the orchestrator session.

## Notable implementation facts

- **Sandbox `sys` NOT whitelisted** (judgment call): the sandbox JSON protocol
  runs on the child's real stdin/stdout; sys access could corrupt frames.
  difflib whitelisted and verified in the real sandbox.
- **Analyzer timeout is health-gated, not blanket-capped**: a flat 10s cap
  would kill healthy non-streaming decodes; the patch probes GET /models (2s)
  after a timeout — dead server → 10s fast-fail, alive → full budget.
- Carryover graft scrubs prior digest injections so exactly one is alive
  (compounding-cost bug caught by its own tests).
- June stock reference: tmp cleanup evicted BOTH prior copies mid-campaign;
  durable md5-verified pin now at `scratchpad/bundles/june_stock/`
  (= jeroencottaar/taaf-kaggle-source-share = what pack-v22 mounts).

## Smoke queue (single-variable, post pack-v22 read)

1. effort_medium (EFFORT_DEAD_RETRY=0 for purity)
2. yield_carryover (digest-only, then +cap)
3. mechanical bugfix arm (flags 2-6; pair 6 with 7 or drop 6 if it goes live)
4. expect-queue rider build (halt-and-return-control shape) → smoke
5. truthful_telemetry (pre-registered, per-level actions read)
6. archetype-triage graft build against trigger_rule.json
Envelope law: none of 1-5 can extend duration; 6 strictly reclaims.

## ADDENDUM 00:50Z — serving-lab v4 verdicts (kernel COMPLETE, 100.6 min, RTX Pro 6000)

Push lesson (3 failed attempts, ~0 GPU-min each thanks to the fail-fast
assert): metadata machine_shape AND --accelerator alone still bind P100 —
**the competition source is the RTX Pro 6000 gate** (attach
arc-prize-2026-arc-agi-3 to any kernel that needs the scored GPU).

1. **THROUGHPUT HYPOTHESIS DEAD**: baseline scored flags deliver
   **642.6 tok/min/session at conc 28** (vs 400 kill-bar, and ABOVE the Modal
   H100's 445) — 0 errors, realistic 17-25k prompts with images. Live serving
   is NOT the live-vs-offline discount. Remaining gap suspects: offline-draw
   optimism + set composition + behavioral pathologies (the validated grafts).
   Aggregate: conc8 217 tok/s / conc16 304 / conc28 300 (conc 28 remains the
   right worker count).
2. **MTP: MEASURED NULL — CLOSED.** nst=3: speedup 1.00x (conc8) / 1.07x
   (conc28) despite healthy acceptance (59.7-63.3%; nst=2: 71.6-71.8%); no
   crash (26k-37k-token soak SURVIVED, 36/36 ok, issue #40756 did not
   reproduce); parser round-trip 2/3-3/3. Under continuous batching at our
   prompt sizes the speculative gain cancels. Queue lever #2 (priced
   +0.2..+0.6) is dead on the real GPU at conc>=8. No smoke, no slot spent.
3. pack-v22 parity arm submitted on schedule: ref 55679452 at 00:11:07Z,
   gated path clean, rerun pending, read ~09:30Z.

Post-lab queue: the campaign's remaining levers are ALL harness-side —
effort_medium -> yield_carryover -> mechanical pack -> expect-queue rider ->
truthful_telemetry -> archetype-triage. Serving stack is certified as-is.

## ADDENDUM 2026-08-26 — drift investigation + a CORRECTION to my own claim

SERVING REFUTED as the cause (serving-lab3, commits a4f562a/c0c39f2): conc28
636.1 vs 642.6 tok/min/session (0.99x), conc8 1619 vs 1626 (1.00x), p50
latencies flat, KV figures identical, and the GREEDY DECODE FINGERPRINT
sha256 matches 08-22 exactly — the served model is behaviourally identical,
not merely the same bytes. Env captured for future diffs: driver 580.159.04,
CUDA 13.0, torch 2.10.0+cu128, no throttle flags.

**CORRECTION (I overstated the drift).** I called a "5 consecutive decline,
p~0.008" and a "3.5 sigma same-bytes spread". Both were wrong:
1. The 5-run sequence contains TWO effort_medium draws we independently
   suspect are harmful — it is not 5 draws of a stable process.
2. My sigma=0.19 came from the 3.6-era duck-base group whose MEAN was 0.93.
   Variance scales with the mean: measured CV is ~0.2 in ALL THREE eras
   (hybrid-explorer 0.16 @mean 0.31; duck-base 0.21 @0.93; 3.8-era 0.18
   @1.43). At our current mean, sigma ~= 0.28, so the pack-v22 same-bytes
   spread of 0.67 is z=1.69, p~0.09 — SUGGESTIVE, NOT SIGNIFICANT.
DRIFT VERDICT: unproven. Remaining candidates (hidden-set rotation, gateway
latency, plain variance) are untested; tonight's duck-38 v2 draw is a third
draw of the arm that scored 1.29/1.74 and discriminates cheaply.

**MEASUREMENT LAW (new, important): CV ~= 0.20 of the mean.**
=> at mean ~1.4, per-draw sigma ~0.28; a 2-draw read has SE 0.20 and can only
detect effects >= ~0.55. EVERY 2-draw ladder read in this campaign is
UNDERPOWERED for the +0.2-0.3 levers it was aimed at, including the
effort_medium REVERSION (pooled 1.12 vs comparator 1.55: diff 0.43 < 0.55
=> that verdict is NOT established either; effort is neither confirmed
harmful nor cleared). Consequences: stop running 2-draw reads for small
levers; only >=0.55-class levers are slot-testable at all; smaller levers
must be validated offline/by mechanism telemetry or bundled.
