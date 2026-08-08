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
