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

## Increment 2 — Mechanic-guided planner — BUILT, MEASURED, KILLED (2026-06-21)

`SlideNavExplorer`: inference-gated slide-aware spatial explorer (avatar-cell + #collectibles
state, BFS slide-nav, push-safety gate, stall-triggered engagement). RESULT:
- **Isolated ls20: L1@3426 vs salience 7961 — a real 2.3x win** (first concrete wall-game
  efficiency gain; validates the slide-aware idea).
- **Does NOT compose.** stall=0 (engage from start): gains ls20 but regresses tu93 L5->L4 +
  tr87 L1->L0 (net-negative). stall=1500 (additive): preserves all levels but engaging
  mid-game is *worse* than letting salience continue (ls20 7961->14823, tu93 eff 1.87->1.756)
  -> aggregate == salience or slightly worse. No configuration helps the aggregate.
- ROOT: avatar-cell collapse trades board/depth info for coverage-speed; restarting mid-maze
  wastes the explorer's partial work. KILL (4th spatial-nav kill in project history).

Firewall held throughout: SlideNavExplorer lives only in the eval harness, never in the
submission (default stays TransferExplorer). v13=0.33 and the 0.33 floor untouched.

## Verdict

The program produced one genuine, durable finding — **wall-game mechanics ARE cheaply
inferable, and the true mechanic is slide-to-wall (not the fixed-delta the project assumed)**
— and one real-but-uncomposable speedup (ls20 2.3x in isolation). But the planner does not
yield a shippable lever: the wall games resist a clean general solution from this angle too.
0.33 public ceiling stands. Next real shot at >0.33 remains the June-30 open-source
adopt+harden, or a genuinely different breakthrough not yet found.

## Discipline

Firewall the banked agent throughout (default policy stays TransferExplorer/v6-superset).
Every increment measure-first with a pre-registered gate. No game-specific hacks (overfitting
= the StochasticGoose 12.58%→0.25% trap); detectors must be general. Document kills honestly.
