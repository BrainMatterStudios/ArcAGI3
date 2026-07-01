# ARC-AGI-3 — Research Continuation Handoff (2026-07-01)

Written at the end of a long session that mapped the multi-level / goal-inference frontier. Purpose: let a
fresh session explore **new angles and paradigms** without re-walking the dead-ends. Read the auto-memory index
first (`MEMORY.md`); this doc is the forward-looking synthesis.

---

## 1. Deployed state (the submission)

- **Agent:** `PortfolioPolicy` (13 strategies, MAX-over-plays, strict-superset / floor-safe). Anchor = banked
  `TransferExplorer`; last play = `general_agent` (search-replay coroutine).
- **Capability:** solves **13/25 dev games at L0** reactively, zero per-game code; **palette-invariant**
  (generalizes to re-skinned hidden games — verified by leave-color-out). Goal-agnostic novelty search reaches
  ar25/m0r0.
- **Paradigm:** SEARCH-REPLAY — the deterministic+resettable env IS the world-model; search a dirty run (cost
  discarded because scoring is MAX-over-runs), double-reset + replay a clean run for the score. Sidesteps the
  "W1" wall (tiny CNN world-models can't infer a novel affordance sample-efficiently).
- **Submission:** `submission/notebook.ipynb` (self-contained, 25 embedded modules, rebuild with
  `python submission/build_notebook.py`). `submission/my_agent.py` = `MyAgent(Agent)`, is_done at WIN or 8h.
- **Branch:** `winning/mechanic-model-search`. Tests: `pytest tests/test_coroutine_strategy.py
  tests/test_portfolio_policy.py tests/test_palette_generalization.py tests/test_submission.py` (13 pass).
- **DO NOT** auto-submit to Kaggle (1/day, user-gated). The notebook is ready; the user pushes the daily slot.

## 2. Scoring & constraints — what to actually optimize

- **RHAE per level = min(1.15, (human_actions / agent_actions)^2)**, LEVEL-WEIGHTED, aggregated to a % (whole
  public field < 1.2%; Gemini 3.1 Pro ~0.37%, Tufa 1.21% outlier). Scored MAX-over-plays; the graph persists.
- **Two levers:** (a) solve MORE levels (level-weighted) — multi-level; (b) solve each level in FEWER actions
  (squared efficiency) — RHAE. Efficiency is the dominant per-level lever (geodesic replay gives 7–109x).
- **Binding constraint = WALL-TIME.** Eval is offline, Kaggle T4x2, **<12h TOTAL**, actions over a **localhost
  REST gateway**. Search cost is free in SCORE (clean replay) but NOT in wall-time. This kills deep/naive search.
- **Floor-safety doctrine:** every new capability is an ADDITIVE portfolio play that ABSTAINS on non-match, so
  max-over-plays can only add. Never regress the 13/25.
- **Distribution:** CLICK-dominated — 19/25 dev games need ACTION6 clicks, only 3 move-only. Movement-only
  methods have tiny reach.

## 3. What was tried this session, and the walls (compressed — details in memory)

- **Multi-level via goal-agnostic search-replay** → WALL-TIME-PROHIBITIVE (every deeper-level candidate must
  reset+replay the accumulated prefix; ar25 63s, dc22 76s for a FAILED L1). See
  `arcagi3-cgpl-multilevel-prohibitive`.
- **Architecture question** (dedicated workflow) → search-replay is NOT wrong, it's the correct W1-robust FLOOR;
  the missing layer is AMORTIZATION. See `arcagi3-architecture-verdict`.
- **Factored-model amortization** (learn dynamics on L0, plan L1 in-model) → the model transfers per-action
  DELTAS (move_acc 0.5–0.97) but a reach-a-color predicate is NOT the win. The "kill experiment PASSED" was a
  MEASUREMENT BUG (checked levels>0 when already 1 from L0). RETRACTED. See `arcagi3-architecture-verdict`.
- **Goal inference** (contrast + causal filter + submit-aware + translation-search template + cover) → INDUCTION
  identifies the STRUCTURE correctly for the two source-verified games (cd82 TEMPLATE, tu93 COVER) and dominates
  reach/collect decoys. BUT: (1) the VERIFY gate (re-derive the L0 win) certifies L0-VALIDITY, **not**
  L1-CORRECTNESS — a loose CORRELATE passes it (lp85 "cover" was really a BUTTON-CONFIG puzzle; beam reached
  32/32 covered with NO win); (2) planning-to-satisfy needs each game's SPECIFIC mechanic; (3) mechanic/effect
  probing LEARNS the mechanic but hits game-specificity. **NO closed multi-level loop achieved.** See
  `arcagi3-goal-inference-stage1`.

**One-line meta-lesson:** general methods reliably identify *structure* but keep hitting deep *per-game
specificity* (button configs, bridges, rotation phases, complex fills) AND the correlate-vs-cause trap. Every
"impossible" verdict earlier in the project fell to deeper decode — but multi-level is a genuine wall for the
current toolset, mapped at every layer.

## 4. CRITICAL lessons for the next session (do not relearn the hard way)

1. **DO NOT TRUST GREEN RESULTS.** Two separate "successes" this session were measurement bugs (factored-model
   kill2; lp85 32/32). Always: (a) count a level win ONLY as `levels_completed` going k -> k+1 DURING the plan
   (never `levels>0`), (b) independently re-check on a disjoint code path, (c) cross-check a "learned goal"
   against the game SOURCE for dev games before believing it.
2. **The VERIFY-on-L0 gate is INSUFFICIENT.** A predicate that re-derives L0 can still be an L0-only correlate.
   Real disambiguation needs MULTI-WIN intersection (several independent wins) — which is chicken-and-egg for L1.
3. **Wall-time is the real budget, not score.** Any method that searches live at eval is bounded by 12h over a
   REST gateway. Prefer amortized/in-simulation planning; measure per-game AND full-portfolio wall-time.
4. **Floor-safety is non-negotiable.** Additive plays, abstain-on-non-match, verified bit-identical coverage on
   the current 13/25 before shipping anything.

## 5. Assets built (reusable; in the session scratchpad + repo)

- `goal_harness.py` — FALSE-POSITIVE-PROOF measurement harness: `capture()` records the L0 solution + win
  transition + ordinary baseline + fitted delta model; `evaluate()` counts a win only on an in-window k->k+1;
  validated with a guaranteed positive + the reach-goal negative. **Use this for any multi-level experiment.**
- `stage1_v4.py` — structural goal INDUCER: translation-search TemplateMatch, CoverAll (occlusion-safe,
  agent-excluded), causal filter (false@P0/true@W), submit-aware win frame, baseline-abstain. Source-verified on
  cd82/tu93.
- `effect_probe.py` / `greedy_cover.py` / `beam_cover.py` — mechanic learning (probe action effects on the
  predicate; greedy/beam over probed effects).
- Repo: `src/arcagi3/mechanics/` has `fit_world_model` (factored dynamics), `world_model.plan_to`,
  `reward_goal.RewardGoalLearner`, `primitives.py` (Goal DSL). `winning_explorer.py` = a pure-model agent
  (weak; do not treat as ready).
- Scratchpad path (this session): `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/12301981-8553-4e64-be28-1d091dc4acd3/scratchpad`

## 6. UNEXPLORED angles / paradigms — the forward menu (ranked by novelty x plausibility)

These were NOT attempted (or only named) this session. A fresh session should pick 1–2 and de-risk with a cheap
kill experiment BEFORE building — using the false-positive-proof harness.

1. **Local (V)LM reasoner + deterministic verifier (HIGH ceiling, HIGH risk).** A small local model (fits T4,
   offline) reads the object-list + probe outcomes and PROPOSES the mechanic/goal/plan in structured form; the
   env VERIFIES by replay before committing (never ships a hallucinated plan). This is the ONLY route to a
   genuinely NOVEL affordance (the W1 gap) and directly attacks the correlate-vs-cause problem (a reasoner can
   read "click buttons to arrange", which pixel-contrast cannot). Kill experiment: on 3 dev games, does a local
   model, given the object list + a few probe transitions, name the win condition matching source? Risks: which
   model fits T4 offline; latency vs 12h; grounding. (Workflow rank-6; never built.)

2. **Multi-win predicate intersection (attacks the correlate trap directly, CHEAP).** Solve L0 several times via
   DIFFERENT strategies (different search seeds / paths) -> several win transitions -> INTERSECT the surviving
   predicates. A correlate that held for one L0 path likely fails another; the causal predicate survives all.
   This tightens induction BEFORE touching L1 and needs no new eval. Kill experiment: does intersecting 3 L0-win
   predicates on lp85 REMOVE the spurious "cover" (leaving nothing / the true button structure)?

3. **Faithful StochasticGoose / learned effect-model + RL, retrain per level (the KNOWN 0.25 approach).** The
   preview winner (supervised CNN effect model + planning, resets per level) scored 0.25 offline — ABOVE this
   repo's ceiling — and was never built faithfully here. Public code: github.com/DriesSmit/ARC3-solution. See
   `arcagi3-033-ceiling-is-codebase-not-paradigm` + `arcagi3-stochasticgoose-facts`. Highest-evidence forward
   bet. Kill experiment: reproduce their L0 solve rate on 5 dev games with their offline-adapted loop.

4. **Within-level RHAE efficiency instead of multi-level (SAFE, additive).** RHAE is SQUARED efficiency; halving
   agent_actions on L0 quadruples that level's score. The geodesic-replay play already gives 7–109x per play but
   may not be maximized across the portfolio. Angle: a general shortest-path/macro-compression replay for EVERY
   solved level. Lower ceiling, but floor-safe and does not need the hard multi-level machinery.

5. **Behavioral-signature mechanic LIBRARY + matcher (extensible).** Formalize each solver as
   (behavioral-signature, parameterized-solver); a matcher classifies a hidden game to the nearest class by
   SIGNATURE (not colors) and handles COMPOSITIONS of known classes. Turns "add a mechanic" into data. The
   goal-classes (peg/template/centroid/pull) are a partial start.

6. **Learned search heuristic / value function (attacks wall-time).** Train offline on dev solutions a
   progress/value net to GUIDE the frontier so far fewer candidates expand -> deep search becomes wall-time
   affordable. Pairs with any predicate as the goal test. Watch: heuristic transfer to novel mechanics.

7. **Program synthesis of the FULL solver (not just the goal).** Induce the mechanic AS a small executable
   program over a typed primitive DSL (move/paint/collect/push/toggle/rotate/gravity/launch/clone...), MDL-scored
   from interaction; the program IS the amortizing model -> plan any level. Bounded by the DSL; out-of-DSL
   mechanics abstain. (Workflow rank-3/SPINE-like.)

## 7. Concrete first move for the next session

Recommended order: **(2) multi-win intersection first** — it is cheap, needs no eval, directly attacks the
correlate-vs-cause failure that broke this session, and REUSES `goal_harness.py` + `stage1_v4.py`. If it tightens
induction, the goal-inference line revives. In parallel/next, **(1) local reasoner** kill experiment — the one
paradigm that could break W1 and read semantic mechanics (buttons, bridges) that pixel-contrast cannot.

Always start any experiment by wiring it through the false-positive-proof harness and cross-checking against dev
source. Bank the 13/25 floor untouched.
