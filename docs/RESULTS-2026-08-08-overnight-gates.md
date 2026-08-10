# Overnight gate results — 2026-08-08 (first GPU night of the reset week)

## Track B capability gate (35B) — NO_GO, final

Kernel `arc-agi-3-capability-qwen36` v3, stage=complete. Qwen3.6-35B-A3B-FP8 served
cleanly on RTX Pro 6000 (first-ever serve of this stack; `--max-num-seqs 512` in
force). Protocol 143/156 = **91.7%** (bar ≥95%): mostly-working protocol, not a
sonpham-style collapse. **Target unlocks 0/4** (dc22/m0r0/sk48/tr87 all zero; bar
≥2/4). Controls: ft09 0, su15 1. **Verdict: capability failure, measured by our own
instrument — the 35B route is closed.** duck-sparse v7 live submission permanently
dead. Next candidate per frozen design: AgentWorld. Iteration cost to a valid gate:
3 kernel versions, ~30 wasted GPU-minutes (fail-fast design worked as intended).

## Track A patch closure — coded contract NO_GO, exactly on the plan-text GO boundary

Base (byte-reconstructed v7: stall 900/animation 0/graph 0): 28 games, 13 levels
(11 excl ft09). Candidate (stall 600/animation on/level-age grinder on): **17
levels excl ft09 = +6 exactly**; zero regressions (no su15/tu93 veto); delta
purity clean.

Mechanism engagement is unambiguous: animation 2,004 payload deliveries (base 0),
grinder 70 engagements with **2 levels attributed to grinder narration**, watchdog
stale-closes halved (6→3).

**Honest discrepancy, on the record:** the campaign plan's Track A bar is
disjunctive — "≥2 candidate-only first unlocks … OR ≥+6 levels excluding ft09
with no ≥2-level regression". The implemented classifier froze only the
unlocks branch (0 new target unlocks → NO_GO). The measured result **exactly
meets the plan-text's second branch** (+6, no regression). A delta sitting
precisely on a bar at n=1 wave is the textbook under-powered case (A/A floor
amendment: ~13.4% FP at these magnitudes).

**Disposition:** the coded pre-registered contract governs → recorded NO_GO;
but the boundary result + clean mechanism attribution earns exactly one
replication wave pair before any v7 reversion: base re-run **pinned** (also
resolves the image-asymmetry caveat — base v1 ran unpinned, candidate v4
pinned) + candidate wave 2. ~6h GPU; ~7h of the 17.5h ceiling spent so far.
Wave 2 reproduces ≥+6 with attribution → GO + the plan's attribution ablation;
collapses → revert to base bytes for live duplicates (base-family mean 0.929 >
patched-family 0.788).

## Infrastructure lessons (both now fixed in-tree)

- Kernel metadata must pin `docker_image` (builder-enforced now).
- The competition data mounts at `/kaggle/input/competitions/<slug>` OR
  `/kaggle/input/<slug>` per machine — probe both (three candidate ERRORs).
- Never discard pip stderr in kernel cells.

## Variance arm (55336559)

Fired 00:11:05 UTC via the gated runner (re-attest + race guard + settle);
pending with the genuine long-run signature; 90-min watch closed clean; score
expected mid-morning; classification per the frozen 0.69–1.27 band rule.

## Wave-2 replication (2026-08-08 ~09:45 UTC) — closure verdict FINAL: NO_GO

- Base v2 (pinned image): 12 levels excl ft09. Candidate v5: 13. **Delta +1 = no replication of wave-1 +6.**
- Two-wave totals: base 23, candidate 30; 0 new target unlocks in either wave; no regressions; mechanisms fire (animation 1540 deliveries, grinder 1 level w2) but do not move totals reliably.
- Consequence per the frozen plan: candidate delta (animation+grinder+600s watchdog) is dead; live duplicate config reverts to BASE bytes (base-family mean 0.929 vs patched 0.788); patch work on this delta ends.
- Variance v1 scored 0.85 IN_BAND same morning -> tonight per the approved framework: idle-day byte-identical base draw.

## Package screen (2026-08-08 19:33 UTC) — STOP; the finding is adoption, not capability

- 10 levels excl-ft09 (base pair 11/12); 0 target unlocks -> pre-registered STOP.
- Mechanisms DELIVERED but the model did not USE them: **run_probe called 1 time in 28 games** despite advertisement; dispatch assigned modes (7 AVATAR/7 CLICK/6 MORPH/8 UNCLEAR) but 6 games escaped scaffolds; wiggle fired 28/28 (93 presses, 10 reprobes) — the only structurally-enforced mechanism, and the only one fully engaged.
- Protocol healthy (129 actions/game, 0 zero-action games) -> not prompt-overload; the 27B simply kept its trained one-action habit. Confirms the sweep law: advertised = performative; ENFORCED = adopted. Next iteration: make batching STRUCTURAL (PRO-LONG-style mandatory plan-list action channel), not optional.
- Parallel-load probe (28 vs 56 streams): aggregate 553 -> 782 tok/s (+41%), per-stream latency 8.1 -> 11.0s (+36%), 0 errors. Parallel cognition is subsidized, not free: right for rare commit-point verification (patch20), wrong for per-action ensembles.

## Struct screen wave 1 (2026-08-09 01:25 UTC) — STOP by letter, FIRST POSITIVE DIRECTIONAL RESULT

- 17 levels excl-ft09 (base 11/12, package 10); **g50t unlocked — first target unlock in any screen**; bar was 18 or 2 unlocks -> STOP.
- Adoption transferred partially: plan_actions_per_llm_turn 2.01 (base 1.0; dry-run bar 3), 34% multi-step plans, full 1-20 length range used, 193 nudges (load-bearing for 27B), 918 auto-wrapped singles, protocol healthy (107 actions/game).
- Replication wave (same bytes, v2) pushed per the closure-episode discipline: boundary results replicate before belief.

## Struct screen wave 2 / replication (2026-08-09 04:35 UTC) — STOP; adoption replicates, level spike does not

- 12 levels excl-ft09 (w1: 17); no target unlocks (w1 g50t did not recur). Pooled struct {17,12} mean 14.5 vs base pair {11,12} mean 11.5 -> +3/wave, directional, underpowered at n=2.
- **Adoption REPLICATED and improved: 2.59 plan-actions/deliberation (w1 2.01, base 1.0), multi-step 37%** — the structural channel reliably changes model behavior; the level payoff is modest/noisy, not the w1 spike.
- Both waves STOP per pre-registered bars. Remaining adoption headroom: ~2/3 of turns still single-step (225 nudges).
- Morning decision: iterate adoption vs consolidate; AgentWorld gate on upload (ETA ~13:00 UTC); GPU spent ~22.5h/30h week.

## Final overnight results (2026-08-09 10:43 UTC)

### Struct v3 / adoption iteration — PRIMARY NOT MET; coaching lever exhausted
- adoption 2.57 (bar >3.5; waves now 2.01/2.59/2.57 = stable ~2.5x plateau). Coaching levers (worked examples, yield nudges, commit floor) moved nothing vs wave 2. **The ceiling is the model's deliberation habit, not contract clarity.** Levels 12 excl-ft09, no unlocks. Pooled struct {17,12,12} mean 13.7 vs base {11,12} 11.5 -> ~+2/wave, inside noise at these n.
- Next lever, if any, is hard-structural (reject inspection-only turns / force action() in first call) with protocol-damage risk; not built without a fresh decision.

### AgentWorld capability gate — NO_GO, Track B brain route EXHAUSTED
- Full completion: served (bf16->dynamic FP8, text-only interface), protocol 77/89 = 86.5% (<95%), **0 levels on all 6 games incl. both controls** (35B at least got su15=1). Both alternative brains now measured NO_GO by the same instrument. The 27B remains the best available brain.

### Week ledger
- GPU ~27h/30h. Public LB high banked: **1.30** (new all-time). Replicated positive mechanism: structural plan channel (2.5x adoption, ~+2/wave directional). All four negative gates closed cleanly with zero slots burned on failures.

---

## 2026-08-09 (evening) — struct-v9 armed; two independent reviews; research doc adjudicated

### Submission: struct-v9 armed for the 2026-08-10 00:01 UTC slot
- duck-patched **kernel version 9**, canonical code-cell hash (ledger method, no trailing newline)
  `b48f64c9f47bc0e4d8da0934a21000ff8683914a2dcda8fc2e009b42d68c3112`, scriptVersionId **341311881**.
- `EXPERIMENT_ENV` is a LITERAL copy of the struct screen's `ARM_ENV` (17 keys, verified equal
  key-by-key) so the probe is a faithful transfer test. Beyond the four struct flags this also moves
  `TAAF_ANIMATION` 1->0 and `TAAF_WATCHDOG_STALL_S` 600->900 vs the settled patched arm v7/v8.
- Commit run COMPLETE in ~2 min; log printed the pins plus `patch21 struct: OK (TAAF_STRUCT=1 —
  plan channel armed)` and `patch22 gates: OK`. Wiring proven; serving still unproven by law.
- `docker_image` deliberately left UNPINNED to keep v9 comparable to the patched family and the base
  band (last patched run 08-06 completed unpinned).
- **Known limitation, pre-registered:** scored-run logs are not retrievable (only commit logs are —
  cf. `DIAGNOSIS-2026-08-03`), so this probe returns a single number and NO adoption telemetry.
  Base band is 0.69-1.30; only a draw outside it is individually actionable. The patched family's
  own five draws averaged 0.788, so a mid-band result is ambiguous between two references.

### Review 1 — v9 artifact (claim: "v9 executes patch21/22 with the screen's configuration")
- Mechanically **VERIFIED**: pins precede `apply_all()`, no import-time env reads (AST-scanned),
  inlined `duck_patches.py` byte-identical to the screen's copy, budget/concurrency identical
  (7920 s, 28 — the screen's "geometry" was the bundle default).
- Its CRITICAL finding — that patch6 (TransferExplorer) would hijack the LLM for the first 200
  actions and void the probe — is **REFUTED by the v9 commit log itself**:
  `patch6 tool_agent_analyze: SKIP (arcagi3.transfer_explorer not importable)`. Dataset mounts do
  not depend on `TRUE_SUBMISSION`, so the scored rerun mounts identically. Corollary: patch6 also
  never fired in v7, so it does **not** explain the patched family's 0.788 — that deficit is still
  unexplained and is worth its own investigation.
- Valid residual gaps (not fixed tonight): v9 only *prints* the patch6 SKIP where the screen
  *asserts* it, and a `patch21 FAIL` would warn rather than raise.

### Review 2 — fire-time path. Four confirmed defects, fixed and committed (`19fd560`)
- **Transport noise read as a kernel verdict.** `kernel_status()` combined stdout+stderr and
  aborted on the substring `ERROR`; the Kaggle CLI prints transport failures there and
  `NewConnectionError` contains "ERROR". Reproduced under a dead proxy. Now parses the positive
  `has status "..."` line only; unreadable => retry, never a verdict.
- **No retries on four fire-time API calls** — while the L24 launchd job fires five Kaggle
  submissions at 00:00:05Z, 55 s before this runner wakes. Now 5 attempts with backoff.
- **Race guard is TOCTOU** (reads the submissions list ~10 min before submit_gated submits) with no
  idempotency marker. Added a marker claimed before handoff, released if submit_gated fails.
- **`PACK_MARKERS` substring landmine, introduced the same day**: `'struct'` matched
  "construction", "structural", "instructions". In a campaign named the structural campaign the
  next base draw saying "structural campaign context" would have been REFUSED against a notebook
  with no `TAAF_STRUCT`. Now whole-word matched; regression + positive controls both pass.
- Also: `--not-after` bounds submit_gated's 6-hour COMPLETE wait to the caller's window; poll-loop
  instead of a 3.9 h monolithic sleep; 60 s slack on the window's lower bound.
- Not done, deliberate: no launchd job (a reboot silently loses the slot) and no alerting.

### Research doc (08-09 "offline-week-challenge" synthesis) — mechanics trustworthy, empirics not
- **Calibration REFUTED at the corpus.** `request.model` across `scratchpad/rl_gate/episodes/`:
  37 `moonshotai/kimi-k3`, 19 `duck-model` (smoke placeholder), 3 `qwen3-vl-235b`, 3 kimi-k2.x,
  and **one** `qwen/qwen3.6-27b` — `control_ft09_27b`, **0.0** on ft09 where K3 scored **47.6**.
  The "0.110 multiplier / hidden 9x harder / +1.06 -> LB 2.36" chain is duck-on-hidden divided by
  Kimi-K3-on-public: a model gap reported as a set-difficulty gap. Recomputing the counterfactual
  on its own terms gives +0.78..+0.85 (LB ~2.1), not +1.06.
- Transfer assumption is the fatal part: the cap is `sum(w_completed)/sum(w_all)`, so rescuing L1
  in a 6-level game buys 4.8% while rescuing L5 buys 23.8%. Where we are shallower (hidden), the
  marginal rescue is worth LESS — a flat multiplier assumes it is worth the same everywhere.
  Precedent: v13 transfer-dense (+41% dev) and CAI-prune (+47% dev) both Kaggle-inert (0.33/0.28).
- A survivorship objection I raised was TESTED AND FOUND WEAK: charging rescued levels their
  already-sunk actions moves 18.98 -> 18.36, because the completion-share cap binds long before
  efficiency does. Recorded against my own prior.
- **Scoring**: depth weight `(level+1)` confirmed (`scorecard.py:486-491`) — the synthesis's stated
  per-level formula omitted it. **"5x baseline run-cut" is UNFOUND** in the toolkit; treat as
  fabricated. **Geodesic postpass can never score at eval** (`api.py:424-425` blocks
  competition-mode remake); "+38% live-validated" is offline recomputation; its one live datum is
  sub 54312141 = 1.05. Cost is ~2 API round-trips/game (None-return precedes the BFS), so not worth
  a mid-campaign rebuild; `TAAF_GEODESIC_POSTPASS=0` is a free cleanup for the next build.
- **Human reset lever dissolves**: same-unit measurement gives humans 0.93% of actions vs agent
  0.75%. "20x/game vs 0.7%" was count-vs-rate.
- **"~11 waves fit this week" is a fresh-30h-quota figure** quoted verbatim from
  `submission/_rig/build_rig.py:23` into a week with ~3h left => ~1 wave.
- **Salvage**: hidden games DO expose public `tags` (`api.py:56-77` withholds only `private_tags`,
  `level_tags`, `baseline_actions`), so tag-driven archetype dispatch has a real channel on hidden
  — game-ID-keyed scripts do not. Vision/multimodality is genuinely untested; "stronger base" is
  dead (both swap gates NO_GO with controls).

---

## 2026-08-10 — ft09 ablation stage 1: AMBIGUOUS, stopped per pre-registration

**Verdict: AMBIGUOUS. Stage 2 NOT run.** The pre-registration (written before launch,
`ft09_ablation_config.py`) required gap >= 5.0 AND Mann-Whitney one-sided p < 0.01 to
proceed. Observed gap **+3.32**, **p = 0.088**. Direction as predicted, bars not met.

### Result (n=28 per arm, all 28 clones on ft09, eval geometry 7920s/28-way)

| arm | n | mean | sd | median | zeros | L1 actions (completers) |
|---|---|---|---|---|---|---|
| base | 28 | **8.6042** | 8.65 | 9.013 | 12/28 | median 24 (4-109) |
| struct | 28 | **5.2883** | 8.33 | 0.179 | 14/28 | median 39 (14-157) |

Re-scoring implementation validated against the driver's stored score: 56/56 rows, 0 mismatches.

### What this does to the 08-09 finding

**The "ft09 accounts for 102% of the base-vs-others gap" claim does not survive.**
It was computed from n=1 per game. At n=28:

| | ft09 gap | contribution to a 25-game mean |
|---|---|---|
| n=1 banked estimate | ~13.6 | 0.545 |
| **n=28 measurement** | **3.32** | **0.133** |

A 4x overestimate — textbook regression to the mean from a single-clone measurement.
ft09 is still the largest single contributor and still favours base, but it explains
roughly a quarter of what it appeared to, and the difference is not significant.

### The pre-registered confound is REFUTED (my hypothesis, killed by its own test)

I proposed that base's ft09 advantage might be an artifact of the stall watchdog
terminating its runs early (~3700-4240s) and thereby preserving an efficient score,
while other arms ran the full box. The data says the opposite:

| arm | stall-killed | ran full box |
|---|---|---|
| base | n=10, mean 7.77 | n=18, mean **9.07** |
| struct | n=4, mean 9.85 | n=24, mean 4.53 |

Base's stall-killed clones scored **lower**, not higher. Early termination is not the
mechanism. The confound is dead and the (weak) effect is not explained by it.

### What IS consistent

Base is more action-efficient on ft09 level 1 — median 24 actions vs struct's 39 among
clones that completed it. Directionally the same story as the banked rows, but the
banked extremes (base 7-14 vs struct 68-124) were tail draws, not typical.

### Why stage 2 is not worth booking

Certifying a 3.32 gap at 80% power needs ~103 clones per arm = 3.7 waves per arm,
**~16h of GPU for one public game** — and attribution across four flags would multiply
that. The prize (0.133 of a 25-game public mean, transfer to hidden unknown) does not
justify it. Stage 2 is cancelled, not deferred.

### Cost and process notes

- GPU: 4.40h (2.20h per arm) + ~0.3h lost to a first launch that died at the focus-subset
  guard (bare stems vs hashed ids; fixed, pinned by `test_focus_subset.py`).
- The focus-subset mechanism works and is the durable win here: it converts n=1 per game
  into n=28 in one wave, and it is what exposed the n=1 artifact. Every per-game verdict
  in this rig's history was single-clone.
- **The pre-registration did its job.** At p=0.088 with the sign in the predicted
  direction, the pull to run stage 2 "to see" was real; the bar written before launch is
  the only reason it was not spent.
