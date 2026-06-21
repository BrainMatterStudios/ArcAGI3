# Research Program: General Mechanic-Inference Explorer

**Date:** 2026-06-21 · **Branch:** phase0-prune-oracle · **Status:** Increment 1 (kill-gate)

## The bet

The wall-regime games (ls20 maze+key-door+budget, re86/wa30 slide, sk48 large-state) gate
both the public 0.33 ceiling and — likely — the private prize. They need genuine
spatial/object reasoning. Every *learned* or *signal-steered* approach has been killed:
- learned frame-change / MBRL → regress or sparse-reward-starved
- single motion model → fragile (sokoban, variable slide stride)
- program/LLM synthesis → too slow / small-model insufficient offline
- reorder/abstract without a goal signal → corrupts coverage

**What's never been tried as a program:** active mechanic INFERENCE over a small FIXED
ontology of core-knowledge mechanics, using the live environment as its own simulator
(reset-rollback) — Chollet's ARC "core knowledge priors" applied to ARC-AGI-3. Not learned
(no sparse-reward problem), not frame-change-steered, not one fragile motion model (a
PORTFOLIO of hypotheses each tested empirically and used only where confirmed), not slow
synthesis. The agent figures out *what the game's rules are* by experiment, then plans.

## Core-knowledge ontology (fixed, hand-coded detectors)

- **Agency:** which object translates consistently with a specific action id (the avatar).
- **Movement type:** fixed 1-step delta vs slide-until-obstacle (variable stride to a wall).
- **Contact interactions:** on avatar↔object contact — collect (object vanishes), push
  (object translates), block (avatar stops), toggle (object changes color/state).
- **Prerequisite/causal:** acting on/at object A enables a change at object B (key→door).
- **Goal:** the state-change coinciding with level-up (we already capture reward edges).

## Increment 1 — Mechanic-inference probe (THE KILL GATE, cheap)

Build `mechanic_inference.py` + `scripts/infer_mechanics.py`: drive a game with the existing
SalienceExplorer (capture frames+actions+reward edges), then analyze the trajectory to emit a
structured **mechanic report** per game: avatar object, movement type, observed contact
interactions, and the level-up state-change. Reuse `movement.py` (avatar/translation),
`perception.py` (objects), and the trajectory-capture pattern from `prune_oracle.py`.

**Pre-registered gate (no planner built until this passes):** on a held set of games whose
mechanics I can verify by hand — local toys (`maze` = blocked move, `push` = push, `navg` =
free move) + real `ls20` (avatar+key/door) — the probe must correctly identify **avatar +
movement-type + ≥1 contact interaction on ≥3 of the 4**. If it cannot even reliably *infer*
mechanics, the planning program is moot → KILL cheaply (this is the whole point of going
gate-first: the project's documented failure mode is building the expensive thing first).

Honest risk noted up front: the prior "spatial-coverage" kill found motion models misfit the
wall games (tu93 large inconsistent deltas). Increment 1 directly RE-TESTS that as a
falsifiable measurement — if avatar/movement inference is unreliable on real games, we learn
it cheaply and stop, rather than assuming.

## Increment 2 — Mechanic-guided planner (ONLY if 1 passes)

A planner that uses *confirmed* mechanics + the live env (reset-rollback) to reach the
level-up goal-change efficiently. Gated on TUNE/HOLDOUT no-regression (0.33 firewall) +
measurable wall-game progress (actions-to-level on ls20/re86). Design after Increment 1's
report; do not pre-commit.

## Discipline

Firewall the banked agent throughout (default policy stays TransferExplorer/v6-superset).
Every increment measure-first with a pre-registered gate. No game-specific hacks (overfitting
= the StochasticGoose 12.58%→0.25% trap); detectors must be general. Document kills honestly.
