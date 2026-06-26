# Phase B Fallback — OC-WM (Graph-Only Object-Interaction Curiosity) Design

**Date:** 2026-06-26
**Branch:** `winning/mechanic-model-search`
**Status:** design — approved to proceed to planning/implementation
**Builds on:** `docs/superpowers/specs/2026-06-25-phase-b-rvh-no-t-design.md` (RVH No-T, killed Exp 54)

## 1. Context

RVH No-T was killed at its own reward-edge gate (Exp 54). The reward-edge probe is
structurally fatal: under the banked explorer the wall games carry **zero** positive reward
edges (`ls20=0`, `re86=0`), while only already-solved games have any (`collect 2`, `tu93 4`,
`vc33 2`). A value head trained on reward-proximity therefore has nothing to learn from on
exactly the games that matter, and the real A/B gate returned `improved=0 regressed=0 → FAIL`.

The judged Phase-B panel ranked **OC-WM** second behind RVH. We take it as the next fallback,
but with a deliberate revision driven by the Exp 54 finding and the campaign's hard lessons.

Two campaign lessons constrain the build:

- **Reward starvation is the real wall on the hard games.** Any signal that *requires* reward to
  bootstrap (RVH value head, Phase A CNN) is silent on `ls20`/`re86`. The new signal must be
  **reward-free**.
- **Learned / forward-model planning regresses games we already win.** Phase A (re-ranker) and
  Phase A′ (model-as-primary) both catastrophically regressed `tu93` and `push` because imagined
  transitions mislead methodical exploration. So execution must stay on **real observed edges
  only**.

## 2. Decision

Build **OC-WM in its graph-only, object-interaction-curiosity variant**: a default-off explorer
whose signal is **count-based novelty over object-interaction signatures** (reward-free,
self-supervised from object dynamics), used **only to re-order real frontier choices** the graph
already contains. We do **not** build the learned object-transition model or any planner over
imagined states in this phase.

### Why this variant

- The intrinsic signal needs no reward, so it is non-trivial on `ls20`/`re86` where RVH was mute.
- It re-uses the existing object-centric perception (`perception`, `scene_graph`, `tracking`),
  so it adds a signal, not a new world model.
- It is the same firewall-safe shape as the (correct, just-mute) RVH wrapper: re-orders real
  frontiers, never invents an edge, byte-identical when disabled.

### Why NOT the other OC-WM variants (this phase)

- *Learned object-transition model + plan over imagined configs* — reintroduces the exact
  hallucination/regression failure mode that killed Phase A/A′. Out of scope; only revisited if
  the graph-only variant clears its kill-tests and a higher ceiling is justified.

## 3. Goals

### Primary goal

A default-off `ObjectCuriosityExplorer` that beats the banked `TransferExplorer` on the real
holdout gate (≥1 strict improvement, zero regressions) without changing the banked path when
disabled.

### Kill-test goal (the point of the phase)

Before building the full wrapper, verify cheaply that reward-free object-interaction curiosity
can do what reward-based learning could not: **produce a first reward edge on a wall game where
the baseline produces zero.** If it cannot, the bet has failed at its own thesis and is killed
early.

## 4. Non-goals

- No learned transition model `T(o, a) → o'` in this phase.
- No planner over imagined object configurations.
- No new perception. Signatures are derived from the existing object-centric scene representation.
- No changes to submission defaults until the full holdout gate passes strictly.
- No Kaggle submission from this phase until the real local/holdout gate says the branch is
  strictly better than `TransferExplorer`.

## 5. Architecture

### 5.1 Baseline to preserve

`TransferExplorer` remains the production floor. When object curiosity is disabled, the action
trace must be byte-identical to the current banked agent for the same seed and environment.

### 5.2 Core components

#### A. Interaction-signature extractor — `src/arcagi3/object_interaction.py`

A pure function over two consecutive scenes (and the action taken) that returns a hashable
signature describing the interaction:

```
signature = (
    action,                       # which action token produced the change
    relation,                     # avatar↔changed-object: contact | overlap | adjacent | none
    object_type,                  # stable type id of the changed object (color/shape class)
    outcome,                      # moved | appeared | vanished | recolored | none
)
```

Derived only from the existing object-centric representation (`perception` /
`scene_graph` / `tracking`). Deterministic and unit-testable in isolation. If no object change is
attributable, the signature is a canonical `NONE` value (counts as "no interaction").

#### B. Count-based novelty scorer

A dict `signature → count`, updated per real transition within an episode. A candidate frontier's
score is the rarity of the interaction its expansion is expected to produce (lower count = higher
priority). Cold/unknown frontiers are treated as maximally novel so the agent probes them.

#### C. Value/ordering wrapper — `src/arcagi3/object_curiosity_explorer.py`

`ObjectCuriosityExplorer(TransferExplorer)`, `enable_object_curiosity=False` by default. Overrides
`_pick_from_batch(choices, node)` — the explorer's existing hook for choosing among equal-tier
**untried real actions**, whose docstring already sanctions subclass reordering as coverage-safe
(every action in the batch is still tried over successive visits; only the order changes). Among
the batch, prefer the action whose target descriptor (clicked object-type, or simple-action id)
has the rarest object-interaction history. If the feature is disabled or cold (no interaction
history yet), fall straight back to `super()._pick_from_batch` (uniform-random, the v6 behaviour).
It re-orders existing real choices only; it never generates a novel transition. The current grid
is stashed in `decide` so the batch scorer can resolve each click's target object-type.

## 6. Data flow

1. Banked explorer runs and builds the observed graph as today.
2. Each real transition `(s, a, s')` is converted to an interaction signature (component A).
3. The novelty scorer updates `signature → count` (component B).
4. When the explorer must choose among several equal-tier untried real actions at the current
   node, the wrapper prefers the action whose target descriptor has the rarest interaction
   history (component C).
5. Execution and reachability remain entirely on real graph edges.

No part of this loop generates novel transitions or plans over imagined states.

## 7. Acceptance gates (kill-early ordering)

### Gate 0 — interaction-coverage viability (cheapest, first)

On `ls20` and `re86`, curiosity-guided exploration must reach **more distinct interaction
signatures** than the baseline within the same budget. If the baseline already saturates the
interaction space (curiosity cannot cover more), the signal is inert → kill before building the
A/B harness.

### Gate 1 — the decisive starvation kill-test

Run the reward-edge probe **under curiosity** on `ls20` and `re86`. Curiosity must produce **at
least one** positive reward edge on a wall game where the baseline produces **zero**. If both
remain zero, the bet has failed at its own thesis → **kill immediately**; do not escalate to the
forward-model substrate.

### Gate 2 — firewall

With the feature disabled, action traces must remain byte-identical to `TransferExplorer`.

### Gate 3 — local no-regression

The local floor must remain intact when disabled and when enabled on the local suite: existing
tests green; `push` / `navg` / `btnc` not regressed; submission default path unchanged.

### Gate 4 — real holdout improvement

With the feature enabled, the real measurement harness must show ≥1 strict holdout improvement and
zero regressions versus the banked explorer. If not, kill it.

## 8. Risks

- **Curiosity wanders without converging on the terminal.** Object-interaction novelty is a proxy;
  covering interactions does not guarantee hitting the bespoke (and on `ls20` possibly invisible)
  win-condition. This is exactly what Gate 1 measures cheaply. Mitigation: hard kill at Gate 1.
- **Signature mis-attribution.** If the extractor credits incidental/agent-caused changes as
  interactions (the recurring "frame-change ≠ progress" trap), novelty becomes noise. Mitigation:
  signatures are tied to avatar↔object spatial relation and object-type stability, and the
  extractor is unit-tested on hand-built scenes before any live run.
- **Overstepping the graph.** If ordering ever drifts into inventing edges, it regresses toward
  the failed planner line. Forbidden by design: graph-only, re-order existing real paths only.

## 9. File impact

Expected new / changed units:

- `src/arcagi3/object_interaction.py` — interaction-signature extractor (pure).
- `src/arcagi3/object_curiosity_explorer.py` — default-off graph-only ordering wrapper.
- `scripts/interaction_coverage_probe.py` — Gate 0 measurement.
- `scripts/curiosity_reward_edge_probe.py` — Gate 1 measurement (reuses `reward_edge_probe`).
- `scripts/curiosity_ab.py` — Gate 4 real A/B harness.
- `src/arcagi3/runner.py` — `--agent curiosity` entry (honors the default-off flag).
- `tests/test_object_interaction.py` — deterministic signature unit tests.
- `tests/test_curiosity_firewall.py` — byte-identical disabled-path test.
- `tests/test_curiosity_ab_smoke.py` — A/B summary unit test.
- `docs/experiment-overview.html` — experiment record after measurement.

## 10. Decision rule

If Gate 0 or Gate 1 fails, do **not** iterate this architecture by guesswork and do **not**
escalate to the learned-transition / forward-model substrate. Kill it early, record the negative
in the experiment log, and return the decision to the user (consolidate-and-ship remains the
standing recommendation given the campaign evidence).

If Gates 0–1 pass, build the smallest default-off wrapper and measure Gates 2–4 immediately.
