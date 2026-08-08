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
