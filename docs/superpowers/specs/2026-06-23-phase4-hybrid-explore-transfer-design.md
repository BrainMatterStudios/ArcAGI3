# Phase 4 — Hybrid Explore-to-Clear, Model-Guide-to-Deepen

**Date:** 2026-06-23
**Branch:** phase4-hybrid-explore-transfer
**Status:** design — awaiting user review
**Builds on:** Phases 1–3 (discovery engine + planner + motion-lattice) and the existing `TransferExplorer`.

## 1. Context & motivation

Three phases of "understanding-first" discover→plan cleared **0 levels** — the long pixels→model→plan
chain is brittle, and the dumb salience explorer (which clears levels by search) outperforms it on the
only metric that counts. Strategic reassessment (chosen Option C): **invert the bet.** Keep the explorer
as the *floor* that actually clears levels, and use the Phase-1–3 discovery machinery as an *efficiency
layer* — guidance that biases the search toward the goal and transfers across a game's levels. This plays
to both strengths and repurposes the discovery code instead of discarding it.

The seam already exists: `TransferExplorer` (a `SalienceExplorer` subclass) learns each level's
rewarding-action *signature* and promotes matching candidates on later levels (+41% on lp85). Phase 4
enriches that with the induced *transition model* as a second, richer guidance signal — while the explorer
remains the floor (never a brittle chain).

## 2. Goal & success bar

Build `HybridTransferExplorer` = `TransferExplorer` + model-based action promotion, and run a three-arm
lp85 bake-off to answer: **does discovery-model guidance beat the shallow signature-transfer heuristic?**

**Success bar:** on lp85's multi-level run, `HybridTransferExplorer` clears the levels `TransferExplorer`
clears in **fewer total actions** than `TransferExplorer` (which already beats salience by ~41%), ideally
with the per-level gain growing with depth (model-transfer compounding). A clean negative (model guidance
does NOT beat signature-transfer) is a valid, decisive result and must be reported honestly.

**Firewall (non-negotiable):** `enable_model_guidance=False` → byte-identical action trace to
`TransferExplorer`; with `enable_transfer=False` too → identical to plain `SalienceExplorer`. This keeps
the banked baselines intact and makes the three-arm comparison exact.

## 3. Scope & non-goals

**In scope:** a `ModelGuide` guidance component (reusing Phase-1–3 induction + planner + lattice-snap);
`HybridTransferExplorer`; the three-arm lp85 bake-off; Experiment-46 write-up.

**Non-goals (YAGNI):**
- NOT trying to make the planner clear a level on its own (Phase-3's wall) — guidance is best-effort; the
  explorer is the gate. Phase-3's two open defects (conjunctive slots, off-footprint snap) are tolerated:
  imperfect guidance still helps; salience still clears.
- No new mechanic primitives; no GPU/LLM.
- No change to `SalienceExplorer` or `TransferExplorer` behavior (the hybrid is a subclass; firewall keeps
  the baselines byte-identical).
- Single game (lp85). Multi-game generalization deferred.

## 4. Architecture / components

### Component 1 — `ModelGuide` (`src/arcagi3/model_guide.py`, new)
Encapsulates "maintain an induced model from the observation stream, and suggest a goal-directed action."
- It reuses the Phase-1–3 induction + planning by driving a `DiscoveryExplorer`'s internals in **advisory
  mode**: feed each transition via the existing ingest methods (`_ingest_movement`, `_ingest_transform`,
  `_ingest_world_delta`) so it accumulates `_deltas`/`_world_obs`/`_triples`; then call `_build_model(grid)`
  + `_build_plan(grid)` to populate `_plan`. This reuses ALL Phase-1–3 code (induction, planner, Phase-3
  lattice snap, broadened targets) with zero duplication.
- API:
  - `observe(prev_grid, prev_action, grid, level)`: update the internal model from one transition; on a
    `level` increment, call the discovery engine's `on_level_change` (grammar transfers, layout flushes).
  - `suggest(grid, available) -> int | None`: ensure the model is (re)built if stale, then return
    `_plan[0]` as a simple action id if a plan exists and its first step is a simple action; else `None`.
- Caching: only re-`_build_model`/`_build_plan` when new evidence arrived (track observation counts), so
  per-step cost stays bounded.

### Component 2 — `HybridTransferExplorer(TransferExplorer)` (`src/arcagi3/hybrid_transfer_explorer.py`, new)
- `__init__(..., enable_model_guidance: bool = True)`; holds a `ModelGuide`.
- `decide(grid, gstate_terminal, gstate_notplayed, levels, available)`:
  1. feed the just-completed transition to `self._guide.observe(prev_grid, prev_token, grid, levels)`
     (using the stored previous grid/token, same bookkeeping `TransferExplorer` already does);
  2. `token = super().decide(...)` — salience floor + signature-transfer ordering;
  3. if `enable_model_guidance` and `self._guide.suggest(grid, available)` returns an action `a`, and the
     base token is a simple action, **return `("S", a)`** (promote the model's suggestion) — else return the
     base `token` unchanged.
- Firewall: `enable_model_guidance=False` → never consults the guide → identical to `TransferExplorer`.
- Store `prev_grid`/`prev_token` for the next step's `observe` (mirror `TransferExplorer`'s existing fields).

### Component 3 — Three-arm lp85 bake-off (`scripts/hybrid_bakeoff.py`, new, or extend an existing driver)
Run on lp85 (NORMAL-mode env, the `get_environments().startswith("lp85")` pattern):
- arm 1: `SalienceExplorer(seed=0)`
- arm 2: `TransferExplorer(seed=0)` (signature transfer)
- arm 3: `HybridTransferExplorer(seed=0)` (signature + model)
Per arm: levels cleared, actions-to-clear PER LEVEL, total actions, depth reached. Report the arm-2→arm-3
delta (the model's value-add) and per-level trend (does the gain grow with depth?).

### Component 4 — Experiment 46 write-up (`docs/experiment-overview.html`)
Honest three-arm table + verdict: did model guidance beat signature-transfer? By how much, and where
(shallow vs deep levels)?

## 5. Data flow

Per step: `HybridTransferExplorer.decide` → `ModelGuide.observe(prev transition)` updates the induced model
→ `super().decide` produces the salience+signature ordering → `ModelGuide.suggest` proposes a goal-directed
action → if present, it's promoted; else the base token stands. Guidance is best-effort; salience is the
floor. The model and its grammar transfer across levels (via the discovery engine's `on_level_change`).

## 6. Risks & mitigations

- **Guidance doesn't beat signature-transfer** → that's the experiment; report honestly (valid negative).
- **Per-step induction cost** → cache; only rebuild the model/plan when new evidence arrived; cap planner
  nodes (already bounded). If still too slow, rebuild every K steps.
- **lp85's mechanic may not suit model guidance** (e.g. not reach-target) → the bake-off reveals it; the
  explorer still clears via salience regardless, so a poor fit costs efficiency, not correctness.
- **Firewall regressions** → the `enable_model_guidance=False` identical-trace test is the guard; run it.
- **Advisory-mode coupling to `DiscoveryExplorer` internals** → `ModelGuide` depends on private methods of
  `DiscoveryExplorer`; if that's too brittle, the fallback is to call the underlying free functions
  (`movement.infer_all_translations`, `transform_induction.induce_*`, `factored_model.plan`, `motion_lattice`)
  directly. Prefer reuse; note the coupling.

## 7. File structure

- Create `src/arcagi3/model_guide.py` — `ModelGuide` (advisory wrapper over discovery induction/planning).
- Create `src/arcagi3/hybrid_transfer_explorer.py` — `HybridTransferExplorer(TransferExplorer)`.
- Create `scripts/hybrid_bakeoff.py` — three-arm lp85 driver.
- Create `tests/test_model_guide.py` — observe→suggest units.
- Create `tests/test_hybrid_transfer_explorer.py` — firewall + promotion units.
- Modify `docs/experiment-overview.html` — Experiment 46.

## 8. Testing

- **`ModelGuide` units:** on a synthetic nav scenario (agent + a reachable goal cell, fed a few transitions),
  `suggest` returns a sensible goal-directed action; with no model yet (no observations), `suggest` returns
  `None`; a `level` increment routes through `on_level_change` without crashing.
- **`HybridTransferExplorer` firewall test:** with `enable_model_guidance=False`, the `decide` token sequence
  is identical to `TransferExplorer` on a fixed scripted grid/observation sequence (byte-identical trace).
- **Promotion test:** with a stubbed `ModelGuide.suggest` returning a fixed action, `decide` returns that
  action (promoted) when the base token is a simple action.
- **Integration (gated, live, `RUN_BAKEOFF=1`):** the three-arm lp85 run; assert arm-3 clears ≥ the levels
  arm-2 clears in ≤ arm-2's total actions (the success bar; report honestly if not met — do not weaken).
