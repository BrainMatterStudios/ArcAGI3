# Phase 2 — World-Transform Primitives + Path-Opening Planner Bake-off

**Date:** 2026-06-22
**Branch:** phase2-world-paint-primitive
**Status:** design — awaiting user review
**Builds on:** Phase 1 (`docs/superpowers/specs/2026-06-22-phase1-discovery-bakeoff-design.md`)

## 1. Context & motivation

Experiment 43 (Phase 1) showed the discover→plan architecture is sound — perception, movement-grammar
induction, scene-graph slot extraction, and the BFS planner all worked — but engine #1 cleared **0
levels on ls20** because its primitive DSL only modelled the agent's *own* attribute cycling.
The live ls20 mechanic is a **world transform**: the agent **paints a trail (color 11→3) as it moves**,
and painted cells **open paths**. The missing capability is inducing *world/environment* transforms and
planning when **reachability itself changes as you act**.

Phase 2 adds that capability and tests **cross-mechanic generality**: one discovery loop spanning two
distinct mechanic classes — **paint/path-opening (ls20)** and **collect (sk48)** — since generalizing to
structurally different games is the real competition bar.

## 2. Goal & success bar

Extend engine #1 so it can **discover and exploit world-transform mechanics black-box**, validated on
**both ls20 and sk48**, carrying forward Phase-1 disciplines (strict general primitives, source used only
as an automated `A_h` grader, discover→clear→transfer→efficiency).

**Success bar** (per game, black-box; source = grader only):
1. recover the model from black-box probing,
2. clear **L1** efficiently (≪ the salience baseline),
3. **transfer** the grammar to clear **L2–L4** with per-level action cost dropping with depth,
4. at a **meaningful `(A_h / A_m)²`** (capped 1.15).

**Cross-mechanic bar:** the *same* DSL + discovery loop clears L1 on **both** ls20 (paint) and sk48
(collect) — proving the approach spans ≥2 mechanic classes, not one hand-fit game.

**Planner-bake-off bar (ls20):** of the two path-opening planner backends (A1, A2), at least one clears
ls20 L1; the bake-off records which, at what cost, and whether the monotone-painting assumption held.

## 3. Scope & non-goals

**In scope:** two new general primitives (`recolor_on_move` for paint, `collect_on_contact` for collect);
world-delta observation in the discovery probe; two planner backends (A1 paint-as-traversal, A2 exact
painted-set) baked off on ls20; an sk48 `A_h` grader; bake-off runs on both games.

**Non-goals (YAGNI):**
- No third game / third mechanic (m0r0 symmetry, push, etc. — deferred).
- No GPU/neural work; no LLM.
- **Zero game-specific constants** in any engine primitive (strict-generality guardrail).
- A2 is bounded by a node cap — we do NOT attempt to make exhaustive painted-set search scale beyond
  the cap; exhaustion is a recorded result.
- Collect uses a single planner (the existing factored planner); the A1/A2 bake-off is **only** for the
  path-opening (paint) case.

## 4. Architecture

Builds on the flat `src/arcagi3/` layout and the Phase-1 modules. Changed/new units, each with one
responsibility:
- **World-delta observation** — `discovery_explorer.py` PROBE_TRANSFORMS records cells (not the agent)
  that change on a move.
- **World-transform induction** — `transform_induction.py` gains `induce_recolor_on_move` and
  `induce_collect_on_contact`.
- **Planner backends** — `factored_model.py` gains the A1 (traversal) and A2 (painted-set) reachability
  models, selected by a backend flag.
- **Discovery orchestration** — `discovery_explorer.py` wires world-transform induction into INDUCE and
  exposes a `planner_backend` constructor flag (`"traversal"` | `"painted_set"`).
- **sk48 grader** — `scripts/truemodel_sk48.py` (or a generalized `truemodel_planner`) for sk48 `A_h`.
- **Harness** — `discovery_bakeoff.py` adds the sk48 run and registers the engine variants.

## 5. World-delta observation (discovery_explorer, PROBE_TRANSFORMS)

Each step, after the agent moves, compute the set of **non-agent cells whose color changed** between
`prev_grid` and `grid` (excluding the agent's own footprint). Record a **world-delta observation**:
`{"from_color": X, "to_color": Y, "near_agent": bool, "vanished": bool}` where `vanished` means the cell
became background. These observations feed world-transform induction. Bound coverage with the existing
`_MAX_TRANSFORM_STEPS` cap. (Agent-attribute triples are still recorded too — the two are complementary.)

## 6. World-transform primitive induction (transform_induction.py)

Both general; fit from world-delta observations; no game constants.

- `induce_recolor_on_move(observations) -> list[RecolorOnMove]` where
  `RecolorOnMove(from_color, to_color)` is induced when moves consistently recolor cells `from_color→to_color`
  along/at the agent's path (min-support filtered). For ls20 this yields `RecolorOnMove(11, 3)`.
- `induce_collect_on_contact(observations) -> list[CollectOnContact]` where `CollectOnContact(color)` is
  induced when cells of a color consistently **vanish** (→ background) on agent contact. For sk48 this
  yields the collectible color(s).

These are added to the primitive DSL registry alongside the Phase-1 primitives (the single extension
point promised in Phase 1).

## 7. The two planner backends (factored_model.py)

Shared: induced model, goal = attr-match-at-slot (paint) or all-collected/slot (collect), BFS.
They differ only in the reachability/state representation for **paint**:

### A1 — paint-as-traversal
- `InducedModel` (traversal mode) treats any cell matching a `RecolorOnMove.from_color` as **traversable**;
  walls = cells neither originally-passable nor recolor-able.
- `step` moves into passable-or-paintable cells; **state stays `(pos, attrs, completed)`** (Phase-1 size).
- Assumes painting is unlimited / permanent / order-independent. **Logged as an explicit assumption.**

### A2 — exact painted-set
- `FactoredState` gains `painted: frozenset`. `step`: entering a paintable cell adds it to `painted`
  (and recolors); a cell is passable iff originally-open OR in `painted`.
- Exact under budgets/ordering/non-monotone paint. **Bounded by `max_nodes`**; exhaustion → return a
  sentinel "intractable" outcome (not a crash). Dedup key includes `painted`.

Backend selected by `plan(..., backend="traversal"|"painted_set")` (or an `InducedModel` mode flag).
The collect case uses the existing planner unchanged (collect removes cells; goal = all collectibles gone
or slot reached — no reachability-changes-as-you-act problem).

## 8. sk48 `A_h` grader (scripts/truemodel_sk48.py)

Mirror `truemodel_planner.py`: load `environment_files/sk48/.../sk48.py`, BFS over its true model to the
L1 win, return the optimal action count as `A_h`. Read `sk48.py` first to confirm its win predicate and
action semantics (do NOT assume they match ls20). Grader-only; never imported by an engine.
`bakeoff_metrics.human_baseline_actions` gains a `game` parameter to dispatch ls20 vs sk48 graders.

## 9. Bake-off setup & decision rule (discovery_bakeoff.py)

- Generalize `make_ls20()` → `make_game(prefix)` (the `get_environments().startswith(prefix)` pattern).
- Register engine variants:
  - on **ls20**: `discovery-A1` (`DiscoveryExplorer(planner_backend="traversal")`),
    `discovery-A2` (`planner_backend="painted_set"`), and `salience` (baseline).
  - on **sk48**: `discovery-collect` and `salience` (baseline).
- Per (engine, game): levels cleared, L1 actions, L1 efficiency `(A_h/A_m)²`, transfer slope, and for A2
  the nodes-expanded / intractable flag.
- **Decision rule:** the winning ls20 paint backend = the one that clears L1 (then L2–L4) at acceptable
  cost; it becomes engine #1's default. Record whether A1's assumption held (A1 solves AND A2 confirms the
  same reachability) or was wrong (A1 path fails live but A2 solves).

## 10. Success criteria (what Phase 2 must produce)

A bake-off table + Experiment-44 write-up establishing, honestly:
1. whether engine #1 clears **ls20 L1** via A1 and/or A2 (and the assumption verdict),
2. whether it clears **sk48 L1** via collect,
3. transfer slopes per game,
4. efficiency `(A_h/A_m)²` per game,
and therefore whether **one discovery loop spans paint + collect** black-box. Negative results are valid
and must be localized (which phase/primitive/backend failed).

## 11. Risks & mitigations

- **A2 state explosion** → node cap + painted-set dedup; exhaustion recorded, not crashed.
- **A1 over-approximation** (assumes nice painting) → the bake-off itself detects this (A1 plans but live
  execution fails / no level-up) and falls to A2; assumption logged.
- **World-delta attribution** (distinguishing the agent's own footprint from genuine world recolors) →
  exclude the agent's current+previous cells from the delta set; require min-support before inducing a rule.
- **sk48 mechanic differs from assumption** → read `sk48.py` before building the grader and the collect
  rule; do not assume sk48 is "just collect" — confirm from source what L1 actually requires.
- **Overfitting to ls20/sk48** → strict general primitives; the cross-mechanic bar (one DSL, two games)
  is the guard; real generalization still tested later on unseen games.

## 12. File structure

- Modify `src/arcagi3/transform_induction.py` — `RecolorOnMove`, `CollectOnContact`, their inducers.
- Modify `src/arcagi3/factored_model.py` — A1 traversal mode + A2 painted-set `FactoredState`/`step`/`plan` backend.
- Modify `src/arcagi3/discovery_explorer.py` — world-delta observation; world-transform induction in INDUCE; `planner_backend` flag; collect handling.
- Modify `src/arcagi3/bakeoff_metrics.py` — `human_baseline_actions(game=...)` dispatch.
- Create `scripts/truemodel_sk48.py` — sk48 `A_h` grader.
- Modify `scripts/discovery_bakeoff.py` — `make_game`, sk48 run, register A1/A2/collect variants.
- Modify `docs/experiment-overview.html` — Experiment 44 write-up.
- Tests: extend `tests/test_transform_induction.py`, `tests/test_factored_model.py`,
  `tests/test_discovery_explorer.py`; gated live guards for ls20-A1/A2 and sk48.

## 13. Testing

- **Unit (synthetic, deterministic):** `induce_recolor_on_move` recovers a paint rule from synthetic
  world-deltas; `induce_collect_on_contact` recovers a collect rule from vanish-deltas; A1 planner solves
  a toy corridor where a paint-able cell gates the path; A2 planner solves the same toy and correctly
  tracks `painted`; A2 returns the intractable sentinel when the node cap is hit.
- **Component:** world-delta observation excludes the agent footprint and records correct from/to colors.
- **Integration (gated, live, `RUN_BAKEOFF=1`):** ls20 via A1 and A2; sk48 via collect — each asserting
  L1 cleared well under the salience baseline (thresholds set from observed results; not weakened to pass
  a failing engine).
- The bake-off run is the top-level integration test; ls20/sk48 source-derived `A_h` are the oracles.
