# Phase 1 — Model-Discovery Bake-off on ls20 (black-box)

**Date:** 2026-06-22
**Branch:** phase0-prune-oracle
**Status:** design — awaiting user review

## 1. Context & motivation

Experiment 42 (`scripts/truemodel_planner.py`) proved that **given the correct transition
model, ls20 is trivial to solve**: BFS over the true model clears level 1 in **13 actions /
89 unique states / 2.6s**, versus the 7961 blind actions `SalienceExplorer` needed. The
bottleneck is therefore **online model discovery** — recovering an unseen game's transition
function and completion predicate from interaction — *not* exploration efficiency or planning
horsepower (both cheap once the model is right).

An adversarial review of an external strategy report (the "Gemini report") independently
confirmed this by contrast: its neural pillars (online action-effect CNN, visual goal
inference) optimize the cheap part, and close analogues are already killed in our campaign
(`SpatialValueExplorer` net-negative; cross-game goal-CNN LOGO AUC 0.457 < chance;
`reach_target_test.py` navigation paradigm fails 5/5). Its heaviest prescription (4B LLM REPL
+ test-time training) is infeasible offline on the verified hardware (**T4×2 16GB / 12h cap;
P100 default with broken CUDA** — not the report's "RTX 6000 / 9h"). The two genuinely reusable
ideas are the deterministic hash-graph **repurposed as a transition-sampling scaffold** and
**episodic memory segmentation**.

## 2. Goal

Run a controlled, empirical **3-way bake-off** on ls20 to determine which discovery paradigm
best recovers the game's model and converts it to score, measured the way the competition
scores. The ls20 source is used **only as an automated grader** — no engine may read it.

### Success bar (user-set)
The symbolic engine (#1) justifies advancing to Phase 2 (cross-game generalization) iff it:
1. recovers the model from **black-box** probing (no source access),
2. clears ls20 **L1 efficiently** (≪ 7961 actions),
3. **transfers** the model to clear **L2–L4** with per-level action cost *dropping* with depth,
4. at a **meaningful `(A_h / A_m)²`** efficiency (capped 1.15).

## 3. Scope & non-goals

**In scope:** ls20 only; one shared evaluation harness; build engine #1; wrap #2/#3 as
baselines; a transfer test across levels.

**Non-goals (YAGNI):**
- No other games (that is Phase 2).
- No GPU/neural work beyond reusing the existing `SpatialValueExplorer` as-is.
- No LLM in the loop (on-device LLM priors already failed the easiest known game).
- **No ls20-specific constants** anywhere in engine #1 (strict-generality guardrail).
- Minimal primitive DSL (§6) — broad primitives deferred to Phase 2.

## 4. Shared harness (the controlled experiment)

A single driver wraps the offline `Arcade` env and runs each contestant identically.

- **Interface:** black-box `reset()` / `step(action)` / observed 64×64 frame + `levels_completed`.
- **Identical conditions:** same action budget per run, same start, same RNG seed.
- **Per-level metrics, logged:** levels cleared; total environment-altering actions; efficiency
  `min(1.15, (A_h / A_m)²)`.
  - `A_h` proxy = Exp-42 BFS-optimal length for that level (13 for L1) until a real human
    baseline exists. Documented as a **proxy, not a fact** (the report's "second-best human"
    detail is unsourced).
- **Transfer test:** after L1, continue into L2–L4 **without resetting the engine's learned
  model**; report whether per-level action cost drops with depth (the within-game lever).
- **Output:** a single comparison table (engine × level → cleared?, actions, efficiency) plus
  the transfer slope per engine.

Harness location: `scripts/discovery_bakeoff.py` (driver) + `src/arcagi3/bakeoff/` (engines &
shared metrics) if shared code grows; otherwise keep engines beside existing explorers.

## 5. The three contestants

### #1 — Factored symbolic model induction *(net-new build)*
Active system-identification loop:
1. **State factorization.** Parse each frame (reuse `perception.to_grid`, object segmentation)
   into a background + a set of objects, each with a feature vector: color, cell-set / bbox,
   centroid position, shape-hash, and orientation if detectable. No ls20-specific encoding.
2. **Agent identification (probe).** From reset, probe each action; the agent is the object(s)
   whose feature vector changes consistently under actions (reuse the interaction-probe logic
   already in `reach_target_test.py` / `perception.py`).
3. **Movement induction.** Fit `action → displacement` for the agent; infer impassable cells
   (an action that yields no displacement against a neighbor → blocked/wall).
4. **Transform induction.** Systematically visit reachable special cells (graph-scaffolded
   coverage); when entering a cell changes an agent attribute (color / shape / orientation),
   induce an **on-enter-attribute-cycle** rule and learn its cycle order.
5. **Goal induction.** The transition model is built *without* knowing the goal. Candidate goal
   states = (slot-like positions × attainable attribute combos); since the factored state space
   is tiny, brute-force planning to candidates is cheap. The env's level-up signal confirms the
   true predicate; on the first level-up, run `levelup_contrastive` over (pre, post) to localize
   the predicate form (agent position + attribute match).
6. **Predict→fail→refine.** Maintain an explicit forward model; every executed action predicts
   the next factored state; on misprediction, refine (new cell type, corrected displacement,
   etc.). This is the loop the campaign never closed against interventional data.
7. **Planner.** BFS/A* over the factored state (agent pos × attributes × slot-completion) using
   the **induced** model as simulator — the Exp-42 pattern, but over the induced (not true) model.
8. **Transfer.** Carry the induced grammar (movement model, transform-cell semantics, goal-
   predicate form) into the next level; re-probe only layout-specific positions; reuse the rules.
9. **Graph use.** The deterministic hash-graph is used **only** as a transition-sampling
   scaffold (avoid re-walking cycles while probing) — **never** as a value reranker.

### #2 — Frame-hash graph explorer *(reuse `salience_explorer`)*
The honest baseline (≈7961 on L1). Wrapped unchanged into the harness. Expected to **not
transfer** (each level's frames are new hashes → discovery restarts), establishing the bar
engine #1 must beat.

### #3 — Neural action-effect + value *(reuse `SpatialValueExplorer`)*
The Gemini report's thrust; a faithful build already exists. Wrapped in as-is. **Guardrail
from the review:** it must not rerank the load-bearing frontier (tu93 finding — frontier order
is load-bearing; reranking corrupts coverage). Run in its existing non-reranking / confidence-
gated configuration. Must fail-safe to numpy if CUDA is unusable (P100 risk).

## 6. The minimal primitive DSL (engine #1)

Game-agnostic, deliberately minimal — just enough to express ls20's family. Each is a parametric
rule fit from `(s, a, s')` triples; **none reference ls20-specific colors/positions/shapes.**

| Primitive | Meaning | Fit from |
|---|---|---|
| `translate(agent, action) → (dx,dy)` | an action moves the agent by a fixed displacement | action probes |
| `impassable(cell_predicate)` | cells matching a learned predicate block movement | failed-move observations |
| `on_enter_cycle(cell_predicate, attribute)` | entering a matching cell advances one agent attribute through a cycle | transform probes |
| `terminal(predicate)` | level completes when the (induced) predicate holds — e.g. agent at slot ∧ attribute-vector matches slot spec | level-up contrast |

**Extension point:** primitives live in one registry module; adding `paint`, `collect`,
`region_transform`, `gravity`, `reflect`, etc. for Phase 2 is a single-location change to the
registry + fit dispatch. No other code changes.

## 7. Decision rule

Rank engines by: **levels cleared → total actions → transfer slope** (does per-level cost drop
with depth?). Record the full table. Engine #1 "wins Phase 1" iff it meets the §2 success bar.
Negative results are still results: if #1 fails to induce the grammar black-box, that localizes
the wall to *factorization* or *goal induction* specifically.

## 8. Risks & mitigations

- **Overfitting to ls20** (the campaign's defining failure). Mitigation: strict general DSL,
  zero ls20 constants, source as grader only; real generalization tested in Phase 2.
- **Goal-induction chicken-and-egg.** Mitigation: build the transition model goal-agnostically;
  brute-force candidate goals over the tiny factored space; confirm via the env's level-up signal.
- **Frame→object factorization brittleness.** Mitigation: reuse the existing `perception` /
  `scene_graph` segmentation; treat factorization failures as explicit, logged refinement events.
- **#3 GPU fragility (P100 broken CUDA).** Mitigation: reuse existing numpy fail-safe; #1 and the
  harness are pure-CPU.
- **Transfer corruption across levels** (tu93-style). Mitigation: episodic segmentation — flush
  layout-specific state per level, persist only the abstract induced grammar.

## 9. Testing

- **Harness unit tests:** metric computation (efficiency cap, action counting), transfer-slope
  calc, deterministic seeding.
- **Engine #1 component tests:** movement induction recovers the 4-way model; transform induction
  recovers an on-enter-cycle rule from synthetic triples; planner over a known induced model
  matches Exp-42 (13-action L1).
- **End-to-end:** the bake-off run itself is the integration test; ls20 source-derived facts
  (from Exp 42) are the oracle to validate engine #1's induced model fidelity.

## 10. Deliverable

`scripts/discovery_bakeoff.py` producing the comparison table + transfer slopes, plus a short
results write-up appended to `docs/experiment-overview.html` (Experiment 43). Banked behind the
v13 = 0.33 firewall; no Kaggle submission from Phase 1.
