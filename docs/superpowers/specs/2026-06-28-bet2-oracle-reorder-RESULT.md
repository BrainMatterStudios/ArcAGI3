# BET 2 — ORACLE-REORDER CEILING PROBE: RESULT (2026-06-28)

**Verdict: KILL the click-affordance arm. Do NOT build the within-level Conv2d head.**

Probe: `scripts/oracle_reorder_probe.py` (no training; pure replay over banked `transfer-dense` offline
trajectories). Per completed level, count no-op clicks (self-loop graph edges = masked key unchanged,
the CAIPrune / Phase-E ~51%-no-op definition), give the explorer a PERFECT change-affordance oracle
that removes every no-op click, recompute per-level RHAE = `min(1.15,(200/actions)^2)` and the
Kaggle-style per-game weighted score (weight = 1-indexed level). This is the UPPER BOUND of the whole
lever — a real Conv2d head can only do worse. Budget 12000, eval TUNE/HOLDOUT split.

## Headline
- **HOLDOUT (scored-set proxies): mean per-game weighted-RHAE gain = 0.168, but MEDIAN = 0.000.**
  The mean is a single-game artifact (su15 +0.840); 4 of 5 scoring games are inert.
- **TUNE: mean 0.011** (lp85 +0.069 the only non-trivial; everything else ~0).

| HOLDOUT game | actual L0 actions | no-op clicks | oracle | wRHAE gain | why |
|---|---|---|---|---|---|
| su15 | 359 | 228 | 131 | **+0.840** | cap-boundary: 359→131 flips 0.31→1.15 (the entire signal) |
| m0r0 | 1940 | 24 | 1916 | +0.000 | far below the 200-action cap → removal invisible |
| ls20 | 7901 | **0** | 7901 | +0.000 | level-up is MOVEMENT-gated, zero clicks → lever structurally inert |
| tn36 | 141 | 67 | 74 | +0.000 | already pinned at the 1.15 cap |
| tr87 | 211 | 0 | 211 | +0.000 | no no-op clicks |
| sk48/re86/wa30 | — | — | — | inert | 0 completed levels @ budget |

## Why KILL despite the literal threshold trip (mean 0.168 > 0.15 "PROCEED")
1. **Outlier, not a lever.** Drop su15 → mean 0.000; median 0.000; 4/5 scored games gain nothing.
2. **Cap-boundary mirage.** Gain is only visible when L0 lands just under the 200-action efficiency
   cap (su15). On the real deep wall games (ls20 7901, m0r0 1940 actions) no-op removal is a rounding
   error; on cheap games (tn36) it is already capped. su15 is the lone game in the visible band.
3. **ls20 — the canonical wall game — has ZERO no-op clicks** (movement-gated L0). The click-affordance
   lever cannot, by construction, touch the hardest scored games.
4. **Mechanism = CAIPrune, already shipped → 0.28 (REGRESSED vs 0.33).** A perfect change-affordance
   oracle that skips no-op clicks is exactly what `cai_prune_explorer.py` approximated online (prune
   proven-no-op colors). CAIPrune posted +47% dev efficiency and lost on Kaggle (lf52-style over-prune
   hit a hidden game). This probe is an UPPER BOUND on a mechanism whose realized Kaggle EV is
   already known-negative.

This is precisely the **"dev-efficiency is Kaggle-inert"** reduction the build plan pre-registered as the
kill condition (the same one that killed CAIPrune 0.28) — masked in the mean by one cap-boundary game.

## Honesty note on pre-registration
By the literal pre-registered rule (HOLDOUT mean > 0.15 → PROCEED) this reads PROCEED. Overridden to KILL
because the signal is a single-game cap artifact (median 0.000) over a mechanism class with a known
negative Kaggle result. The mean-over-n aggregate the plan named did not anticipate an outlier-dominated
distribution; median + per-game breakdown is the honest summary.

## Next
Per the build plan sequence, the click arm is settled. Move to **BET 3** (held-out-family synthetic
feasibility probe — the only arm attacking the actual wall, settles W1 weak-vs-strong), and hold dual-T4
capacity for **BET 1** (adopt-and-harden the June-30 #1 winner). Do not spend a day on the Conv2d head.
