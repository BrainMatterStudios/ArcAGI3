# Campaign plan (dual-track amendment 2026-08-04, supersedes all prior plan docs)

Basis: `docs/PLAN-REVIEW-2026-08-04.md` plus
`docs/superpowers/specs/2026-08-04-dual-track-capability-campaign-design.md`.
The dual-track revision was approved by Ahmed 2026-08-04.

## Objective

**Raise the pinned config's MEAN score to ≥1.5** (bar derived from deflating rival
public maxima to implied means; Kojima 1.86/56 ≈ 1.43 mean). The final prize pays
the mean of 2 selected submissions (private, locked at run time, no rerun,
best-of-2 ≈ mean+0.56σ). A 1.5-mean config also yields a ≥1.8 public draw w.p.
~99% over the remaining slots. "Chase a high draw" is dead as an objective.

## Slot doctrine (~90 remaining)

- Every slot: a pre-registered discriminating hypothesis with an n=1-readable
  fingerprint in the sub message, OR an endgame duplicate. Never an LB mean-test
  at n<10 (MDE at n=3 is +0.28 — the LB is not an instrument).
- All arm selection happens OFFLINE at eval geometry (28 games @ 7920s — the
  permanent A/B standard; the old 10@3600 rig is ~1.27× token-richer than eval).
- Gate-passed discriminating configurations always take priority over duplicate
  draws. Before the local capability proxy reaches a 1.3-mean regime, an idle-day
  byte-identical best-config draw is opportunistic rank banking only: it has zero
  evidentiary weight and receives zero engineering priority.
- At most one live transfer probe per gate-passed patch stack and one per
  gate-passed model/workspace stack before scaling. Preserve ~10-15 endgame
  duplicates. No public-LB mean tests and no engineering organized around
  max-farming.

## Endgame playbook (committed now, per the 07-26 doctrine)

- **Sept 20:** Milestone-2 go/forfeit decision (default: forfeit; don't
  open-source the stack for an unreachable milestone).
- **~Oct 20:** config freeze. Last 6-10 slots = byte-identical duplicates of the
  frozen config via scripts/submit_gated.py (byte-check mandatory).
- **Selection rule (pre-registered):** select the 2 clean (non-zero,
  fingerprint-verified) duplicate draws of the frozen config; earliest-sub on
  ties. Never rely on Kaggle auto-select (winner's curse + 7% silent-zero rate).
- **September:** open-source repo prep (CC-BY/OSI) so a win is bankable.

## Active program — two bounded tracks

The exact schemas, resource ceiling and result states are frozen in
`docs/superpowers/specs/2026-08-04-dual-track-capability-campaign-design.md`.
Implementation plans:

- `docs/superpowers/plans/2026-08-04-patch-closure-gate.md`
- `docs/superpowers/plans/2026-08-04-capability-model-gate.md`

### Track A — close the existing patch loop

1. Reconstruct v7 explicitly: animation off, graph/grinder off, watchdog stall
   900s. Do not let changed defaults masquerade as the old arm.
2. Candidate delta: raw animation on, level-age grinder/narration on, watchdog
   stall 600s. Run as a two-kernel 28-clone @ 7920s A/B.
3. GO only on ≥2 candidate-only first unlocks among the frozen nine targets OR
   ≥+6 levels excluding ft09 with no ≥2-level regression on su15/tu93. Require
   mechanism engagement; a dormant lever is not a positive result.
4. A passing combined stack earns one attribution ablation before freeze. A
   failure returns the campaign to byte-reconstructed v7 and ends patch work.

### Track B — sparse-model capability gate

1. Brain-only swap, ordinary duck harness, exact panel
   dc22/m0r0/sk48/tr87 + controls ft09/su15, 283 GPU-s/game.
2. Candidates: existing unrun Qwen3.6-35B-A3B-FP8 notebook/dataset, then
   `keras/qwen-agentworld/transformers/default/1`. Separate commit kernels;
   identity, no-CPU-offload, throughput and protocol proofs are mandatory.
3. GO requires ≥95% protocol success, ≥2/4 target first unlocks, no ≥2-level
   combined control regression, and no over-budget game or server crash.
4. Only the selected passing brain earns a Tycho-inspired structured evidence
   workspace A/B. No executable world-model builder until that second gate wins.
5. Neither brain passing kills the route. Do not repeat the prior EWM architecture
   on another unproven local model.

### Deferred levers

- SFT is frozen until a capability or teacher census identifies a transferable
  behavior. Any future corpus must preserve deliberation and hold out whole
  mechanic families; NLL alone is disqualified.
- UPSCALE 4→8 remains a cheap transcription probe, but is lower priority than
  completing Tracks A and B.

## Instruments and hygiene

- Submission→bytes→patch-set LEDGER (docs/SUBMISSION-LEDGER.md): every past and
  future sub mapped to kernel version, notebook hash, patch set, pins,
  hypothesis, result. No conclusion may cite a live draw absent from the ledger.
- Eval-geometry patch closure and the two sparse-model capability kernels are
  the next GPU work. The first seven-day ceiling is 17.5h; the rest of the
  30h quota remains uncommitted until a gate is positive.
- Prompt/prefill token accounting added to the rig (gen-tokens alone is half
  the picture).

## First-seven-day execution order

1. Build and dry-run both gate families locally; all tests and identity contracts
   must pass.
2. Stop for approval before `kaggle kernels push`.
3. Run the two patch-closure kernels in randomized order.
4. Run Qwen3.6-35B-A3B and AgentWorld capability kernels. One infrastructure-only
   rerun is allowed; behavioral failures are not rerun.
5. Classify results under the frozen thresholds. Spend the reserved 5 GPU hours
   only on the winning track.
6. A gate-passed configuration receives one live transfer probe, again only after
   explicit submission approval.

## Honest odds (on the record)

The existing patch stack may add +0.05-0.3 mean, but the measured failure mode is
capability. A top-20 climb therefore requires either new target-game unlocks from
the sparse-model route or an unexpectedly strong patch-closure result. Top-5 is
not the central case until a gate-passed configuration demonstrates a credible
≥1.3 mean regime. The mean-not-max framing and endgame mechanics remain what make
any improvement bankable.
