# CONVEX-HULL SCALING PROBE — RESULT (2026-06-28)

**Verdict: convex-hull CLOSED. Scaling mechanic-family coverage does NOT make a held-out mechanic
inferable. The offline rich-prior program is capped near 0.33. Bank it.**

This was THE decisive gate from the new-angles survey: does an offline generator with MORE affordance-
diverse families let an in-context model infer a held-out (novel) mechanic — i.e., is a real ARC mechanic
in the convex hull of a rich-enough curriculum? Built by scaling the BET 3 engine 4 → 8 families (TDD;
+AVOID/TOGGLE/CHASE/KEYDOOR; 48 synthgen tests green), then leave-one-family-out at N = 4, 6, 8.

## What happened
- **N=4 (valid, in-dist 0.505):** held-out novel CLICK affordance (GATE, no sibling click family) = **0.037
  ≈ 0** — reproduces BET 3 (a never-seen affordance class is not inferred).
- **N=6 and N=8: in-distribution accuracy COLLAPSED to ~0.30** (chance = 0.20) even with a bigger model
  (width 64, head 128) and 600 examples/family × 40 epochs. The held-out numbers at N≥6 are therefore not
  directly trustworthy — so I diagnosed the collapse instead of reading through it.

## Diagnosis — the collapse is OBSERVATIONAL ALIASING (= W1), not capacity
`scripts/bet3_aliasing_diag.py`: train on all 8 families WITH vs WITHOUT the true family-id appended.
- WITHOUT family-id: in-dist **0.300** (model can't tell which family it's in from the context).
- WITH family-id: in-dist **0.415** (+0.115) — back to ~the per-family ceiling (BET 3's 4-family in-dist
  was ~0.46). So **essentially the entire 4→8 collapse is family-CONFUSION**: knowing the mechanic restores
  learnability; not knowing it (the real setting) drops to ~chance.
- **Information-theoretic clincher (no model can beat this):** several families are aliased BY CONSTRUCTION.
  REACH and AVOID, under role-randomization and pre-first-reward random exploration, have IDENTICAL
  observation distributions — avatar wandering near one salient cell; the only difference is hidden
  goal-polarity (approach vs flee), which is not in the frame and not revealed until a reward is hit. No
  encoder/context model of any size can disambiguate them from interaction history. The aliasing is
  irreducible, not a CPU/architecture artifact.

## Why this closes the convex-hull question
Two independent legs, both pointing the same way:
1. **No interpolation of novel affordances** (the valid N=4 fold): a held-out affordance class never seen
   in training is a hard ~0, exactly as BET 3 found.
2. **Scaling makes it WORSE, not better:** adding families adds observationally-aliased mechanics, so the
   in-context model cannot even identify which mechanic it faces (in-dist collapses to chance without an
   oracle family-id). More curriculum coverage increases aliasing; it does not expand an inferable hull.

So the premise of the offline rich-prior program — "enough family coverage makes a novel mechanic
in-distribution" — fails: the limiting factor isn't curriculum size or model capacity, it's that **the
mechanic is not in the observable interaction history (W1)**, and ARC's adversarial novelty + role
randomization guarantee pre-reward aliasing. This matches the external evidence (AdA generalizes only
WITHIN a generator's support; ARC is out-of-support by design) and the proven winner's mechanism (it
escapes W1 with an *online frontier-LLM goal prior* + interactive hypothesis-testing, not by inferring the
mechanic from a small offline curriculum — and that path is offline-illegal for Kaggle).

## Honest caveats
- The probe runs on CPU with a small conv over short random context; a GPU-scale model with much longer
  context would raise the per-family ceiling (0.415). But it CANNOT remove the information-theoretic
  aliasing for opposite-polarity families (REACH/AVOID and similar), so the qualitative conclusion is
  robust to scale. The "CLOSED" call rests on (1) the valid N=4 novel-affordance≈0 + (2) the aliasing
  mechanism being irreducible — not on the (invalid) collapsed N≥6 held-out numbers.
- This is analysis-only; the banked 0.33 is untouched (firewall trivially intact).

## Bottom line
The convex-hull gate is CLOSED. No offline rich-prior curriculum (Bet 1/Bet 2 from the survey) clears 0.33,
because the mechanic isn't observable pre-reward and adversarial novelty keeps hidden mechanics out of any
buildable generator's hull. The only leader-class path remains BET 1 — adopt-and-harden the external #1
winner (the online executable-world-model we are legally barred from running ourselves).
