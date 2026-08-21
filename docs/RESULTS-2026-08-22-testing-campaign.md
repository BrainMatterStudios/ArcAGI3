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
