# BET 3 Stage 0 — IN-CONTEXT INFERENCE PROBE: RESULT (2026-06-28)

**Verdict: KILL the meta-RL / in-context-inference arm for WINNING. Do NOT build Stage 1 (RL² +
leave-one-real-game-out).** Nuanced finding below; this sharpens (not just repeats) the W1-strong prior.

Built test-first: a minimal interactive synthetic engine (`synthgen/world.py`, 4 role-randomized
mechanic families REACH/COLLECT/GATE/PUSH, 34 tests) + an in-context probe (`synthgen/probe.py` data
pipeline; `scripts/bet3_incontext_probe.py` model + leave-one-mechanic-family-out). The handoff's "use
the existing synthgen generator" was wrong — synthgen had no engine (static classifier frames only); the
interactive layer is new. Sokoban PUSH uses a real BFS micro-solver for a balanced, valid label.

## Setup
- Leave-one-mechanic-family-out: train on 3 families, test on the never-seen 4th. Colours role-
  randomized (canvas.roles) so identity carries zero family signal — the family must be read from
  interaction DYNAMICS. Label = the rewarding-action affordance class (U/D/L/R/CLICK); chance = 0.20.
- Model = motion-map conv (the interaction-history signal that identifies controllable cells under role
  randomization) over [query frame, last context frame, motion map]. A NO-CONTEXT (query-only) baseline
  and an IN-DISTRIBUTION validity guard gate the reading.

## Numbers (n_per_family=600, epochs=30, T=12, 3 seeds)
| metric | value | reading |
|---|---|---|
| in-distribution val acc | **0.461** | ≫ chance 0.20 → probe is VALID (model learns the affordance) |
| no-context held-out acc | **0.259** | ≈ chance → a single frame is role-ambiguous; context is required |
| **held-out acc** | **0.439 ±0.005** | ≈ in-dist (0.461) → **strong shared-affordance transfer** |
| context lift (held-out − no-ctx) | **+0.180** | large, stable → interaction history genuinely drives inference |
| GATE held-out, **move**-labeled | 0.482 | movement affordance transfers to the held-out family |
| GATE held-out, **CLICK**-labeled | **0.000** | the genuinely-NOVEL affordance: a hard zero |

## Interpretation — why GREY mechanically but KILL for winning
1. **In-context inference DOES generalize across families that share an affordance vocabulary.**
   Held-out ≈ in-dist, large context lift, stable over seeds: the model uses interaction history to
   identify the controllable cell and move it toward the goal — and this transfers to an unseen family.
2. **It CANNOT infer a genuinely novel affordance class.** Trained only on movement-reward families,
   the model never learns that a CLICK can be goal-relevant — GATE's CLICK affordance is a hard 0.000,
   even though the switch's disappearance IS visible in the context. Nothing in training associates
   clicks with reward, so there is no basis to infer click-relevance. (This is structural, not a
   capacity artifact — see the high-capacity confirmation below.)
3. **ARC defeats exactly the part that transfers.** ARC-AGI-3's hidden games are adversarially novel
   BY DESIGN — each a mechanic outside the training distribution. The probe shows the transferable
   component is the SHARED affordance (which novel games don't share) and the novel-affordance component
   is zero. So in-context inference would carry only incidental shared structure to a real hidden game
   and fail on the mechanic that actually gates the score. This is the **letter-vs-spirit** escape: it
   beats leave-one-family-out on shared structure, but that is precisely what adversarial novelty nulls.

This confirms **W1-strong in the form that matters for winning**, with zero sim-to-real gap (the easiest
possible version of the problem). RL² (Stage 1) is strictly harder to train and adds sim-to-real on real
adversarial games; it cannot beat this synthetic ceiling on the novel-mechanic axis. The pre-registered
PROCEED bar (held-out ≥ 0.55 AND GATE,PUSH ≥ 0.40) is not met — and the decisive GATE-CLICK = 0.000 is
the reason. **Do not spend the multi-day RL² + T4 build.**

## High-capacity confirmation (rules out "model too weak")
n_per_family=1000, epochs=50: in-dist rises to **0.498** and held-out tracks it to **0.468** (lift
**+0.204**), no-context still **0.264** (≈ chance). Per fold: REACH 0.495, COLLECT 0.507, GATE 0.466,
PUSH 0.464. **GATE CLICK-labeled acc = 0.000 STILL**, with move-labeled acc up to 0.527. So adding
capacity/data lifts in-distribution learning AND shared-affordance transfer in lockstep, but the novel
CLICK affordance stays a hard zero — confirming it is **structural** (no training gradient ever rewards a
click), not a capacity limitation. The "model too weak" objection is ruled out.

## Honest caveats
- Modest capacity (in-dist ~0.46). But the held-out≈in-dist coupling already shows shared-affordance
  transfer IS captured, and GATE-CLICK = 0.000 is structural (no training gradient ever rewards a click),
  so more capacity raises in-dist/move-acc but cannot manufacture novel-affordance inference.
- The probe tested the cleanest setting (synthetic, zero sim-to-real). The plan's logic holds: failure
  here ⇒ failure on real adversarial games. It failed on the novel-affordance axis.

## Next
The two cheap research arms (BET 2 click-affordance, BET 3 in-context inference) are both settled and
graveyarded with evidence. The remaining leader-class path is **BET 1 — adopt-and-harden the June-30
open-source #1 winner** into the hardened LearnedExplorer harness (the only ceiling that is leader-class,
not "0.33 + capped margin"). Hold dual-T4 capacity for fast integration on the drop.
