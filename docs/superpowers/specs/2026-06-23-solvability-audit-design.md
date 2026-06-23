# Solvability Audit — which wall are we actually against?

**Date:** 2026-06-23
**Branch:** factory/experiment
**Status:** design — awaiting user review
**Builds on:** Exp 42 (true-model BFS, `truemodel_planner`), Exp 43–49 (discover→plan loop), `factored_model.plan`, `discovery_bakeoff`.

## 1. Context & motivation

Two turns of adversarial review converged on a single unresolved question: when the discover→plan
loop clears 0 levels on a game, **which wall is binding?** Three candidates have been conflated all
campaign:

- **Wall A — model correctness.** Does induction recover the game's *true* transition laws
  (the right primitives, the right movement model, the right win-condition)?
- **Wall B — search tractability.** Given the *correct* model, can a bounded planner find a
  human-comparable plan within the offline action budget?
- **Wall C — planner implementation.** Bugs in `_build_plan` / `plan()` that fail even when the
  model is correct and the search is tractable (e.g. Exp-45 conjunctive-slot poison, snap-off-footprint).

The ledger never separated these. "Coverage %" (the Mechanic Coverage Benchmark) measures progress
toward Wall A only and is **blind to B and C** — it would print `ls20: 80%` for a game that clears 0.
Worse, a coverage chart is a new TUNE metric with no HOLDOUT, exactly the class of dev signal that has
only ever misled optimistically (lesson #1).

**A correcting fact found in the code (2026-06-23):** `truemodel_planner.bfs_solve` clears ls20 L0 in
13 and L1 in 45 over the TRUE model. So for ls20, **search given the correct model is tractable** —
Wall B is *not* binding there. That means ls20's 0-clear is Wall A and/or Wall C, not search. This is
the kind of thing the audit must establish per game *before* anyone spends weeks on a coverage benchmark
aimed at the wrong wall.

## 2. Goal & success bar

Build a **four-rung ladder** that, for each game with an executable true model, localizes the binding
wall by measuring where in the chain the level-up is lost. The audit is a referee between two hypotheses:

- **H_induction (user):** the residual is model correctness (Wall A) — induction doesn't recover the
  true laws; once it does, planning is cheap. Implication: invest in induction generality.
- **H_search (Claude):** the residual is search/planning given a correct model (Wall B/C). Implication:
  invest in general multi-step planning, not more primitives.

**Success bar:** for each audited game, a filled row of the ladder table (§4) that names exactly which
rung first fails. The deliverable is the *diagnosis*, not a new score. A clean result either way is
decisive and must be reported honestly — including if it contradicts the framing in this very spec.

**Non-goal:** no Kaggle submission, no new explorer, no new primitive. The audit only *measures* the
existing discover→plan stack against the existing true models. The banked v13 = 0.33 is untouched.

## 3. The four rungs (per game, per level)

Each rung swaps in a *more realistic* component and measures whether L1 still clears and at what `A_m`
(agent actions) vs `A_h` (the Exp-42 BFS-optimal baseline). The **gap between adjacent rungs localizes
the wall.**

| Rung | Model | Planner | Budget | Isolates | Pass = |
|------|-------|---------|--------|----------|--------|
| R0 | **true** | `truemodel.bfs_solve` | generous (200k nodes) | "solvable in principle" + sets `A_h` | clears; gives optimal `A_h` |
| R1 | **true** | `truemodel` BFS | **realistic** node cap (live latency budget) | **Wall B** (search tractability) | clears with `A_m ≈ A_h` |
| R2 | **true** (injected) | `factored_model.plan` / `_build_plan` (production planner) | realistic | **Wall C** (planner bugs vs a correct model) | clears; `A_m` finite |
| R3 | **induced** (discovery end-to-end) | production planner | realistic | **Wall A** (induction correctness) | clears |

Reading the ladder:

- **R0 fails** → game is not solvable by BFS over its own true model at all (model encoding or
  branching problem) — out of scope, flag and skip.
- **R0 passes, R1 fails** → **Wall B binds**: search over the correct model is intractable under a
  realistic budget. Supports H_search. (Not expected for ls20; open for sk48.)
- **R1 passes, R2 fails** → **Wall C binds**: the production planner has a bug the true model exposes
  (conjunctive-slot poison, lattice snap, footprint). These are *fixable engineering*, not a research wall.
- **R2 passes, R3 fails** → **Wall A binds**: the planner+search are fine on a correct model; induction
  failed to produce one. Supports H_induction — and tells you *exactly* what the induced model got wrong
  (diff induced-vs-true: missing primitive? wrong movement? wrong terminal?).
- **R3 passes** → the loop already clears this game; record `A_m`/efficiency.

The "diff induced-vs-true" at the R2→R3 boundary is the real prize: it turns "coverage 80%" into a named
defect ("induced terminal = reach_target, true terminal = reach color-9 *after* path-opening; moving-target
law absent").

## 4. Deliverable: the audit table

```
game    R0(true/gen)  R1(true/real)  R2(true/prod-planner)  R3(induced/e2e)  binding wall   A_h  A_m@first-fail
ls20    PASS (Ah=45)  ?              ?                      0 (known)        ?              45   —
collect PASS          ?              ?                      PASS (14x)       none (clears)  39   ~53
sk48    ?             ?              ?                      0 (known)        ?              ?    —
```

Plus, for every Wall-A row, an **induced-vs-true model diff** (the named missing law).

## 5. Scope

**In scope (executable true models exist):** `ls20`, `collect`, `sk48` — via `scripts/truemodel_planner.py`
(`load_ls20_class`), `truemodel_collect.py` (`load_collect_class`), `truemodel_sk48.py`. These three are the
audit's spine and cover three distinct families (paint/path-open, directional-collect, snake/block-push).

**Stretch (source read in Exp 48, no executable true model yet):** `m0r0`, `re86`, `wa30`, `tr87`, `tn36`.
Each needs a `truemodel_*.py` built first; defer unless the spine result motivates it.

**Out of scope:** any change to shipped `my_agent.py`; any new primitive class; any submission.

## 6. Harness design

Reuse, don't rebuild. The pieces already exist:

- **True models + R0/A_h:** `bakeoff_metrics.human_baseline_actions` and
  `truemodel_planner.bfs_solve_current(game, max_nodes, key_fn, moves)` already do BFS over a true model
  per level. R0 = call it with a generous cap; R1 = same call with the realistic cap.
- **R2 (true model + production planner):** inject a true `InducedModel`-shaped object into
  `factored_model.plan(model, start, slots, max_nodes)`. Needs a thin adapter that wraps the truemodel
  class in the `InducedModel.step` / `FactoredState` contract (the true model already exposes
  `step`/`state_key`/`MOVES`). The slots come from the true win-condition, not the scene graph — this is
  what isolates planner-vs-model.
- **R3 (induced, end-to-end):** the existing `discovery_bakeoff.run_engine(name, DiscoveryExplorer(...), budget, game)`.
- **New code is small:** `scripts/solvability_audit.py` — a driver that runs R0–R3 for each game in a list,
  fills the §4 table, and emits the induced-vs-true diff for Wall-A rows. The truemodel→InducedModel adapter
  is the only non-trivial new component.

**Realistic budget (R1/R2) — define precisely up front.** The binding offline resource is **agent actions**,
not planner nodes (Exp 29: decide() is 0.5ms, the 8-hr budget allows ~hundreds of thousands of actions). So:
- node cap for R1/R2 BFS = the production `plan()` default (200k) — the planner the loop actually ships with;
- "pass" requires not just *a* plan but `A_m` within a stated multiple of `A_h` (propose: efficiency ≥ 0.25,
  i.e. `A_m ≤ 2·A_h`), because a 10×-suboptimal plan scores ≈0 and is a *practical* Wall-B failure even if BFS
  technically returns something.

## 7. What each outcome means for allocation

This is the point of the audit — it redirects the 70/20/10 debate with evidence:

- **Mostly Wall A** (R2 passes, R3 fails across games) → H_induction wins. The mechanic-coverage benchmark
  is justified — but build it in **frozen-vocabulary + solvability-gated** form (design-set vs held-out-set;
  success = R3 clears, not coverage %). Coverage alone remains banned as a standalone metric.
- **Mostly Wall C** (R1 passes, R2 fails) → neither hypothesis; it's *bugs*. Cheapest possible win: fix the
  named `_build_plan` defects, re-run R3. Could move the needle with days of work, no research.
- **Mostly Wall B** (R0 passes, R1 fails) → H_search wins. The lever is general bounded multi-step planning
  over a correct model — harder, more AGI-shaped; reconsider whether it's offline-feasible at all before
  investing.

Any mix is itself the answer: it tells you the *per-family* wall, which is exactly the breadth question the
whole campaign has been circling.

## 8. Risks & honesty guards

- **The adapter could leak the true win-condition into R3.** Keep R3 strictly black-box (induced model only);
  the true model touches R0–R2 only, never the discovery engine. Same firewall discipline as `discovery_bakeoff`
  (source is grader-only).
- **`A_h` is itself a BFS-optimal proxy, not a human trace.** Stated as such; it's the same baseline the whole
  ledger uses, so it's consistent, not novel error.
- **Three games is a small n.** The audit *localizes* walls; it does not prove a distribution over the hidden
  set. Claims stay scoped to "on the games with executable true models, the binding wall is X." No extrapolation
  to Kaggle score without the frozen-vocab holdout step.
- **Confirmation risk.** This spec states a hypothesis (H_search) the author favored last turn and a fact
  (true-model BFS solves ls20) that already undercuts it for ls20. Report rows that contradict the spec
  prominently; a result that vindicates H_induction is the expected and welcome outcome for ls20.

## 9. Definition of done

`scripts/solvability_audit.py` runs R0–R3 on `{ls20, collect, sk48}`, prints the §4 table with the binding
wall per game and the induced-vs-true diff for every Wall-A row, and a short write-up (Exp 50) states which
wall binds per game and what that implies for the 70/20/10 allocation. No submission; v13 = 0.33 untouched.
```
