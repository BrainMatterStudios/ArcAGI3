# ARC-AGI-3 — Handoff 2026-07-02: the walls, mapped end-to-end

## *** ADDENDUM (later same day): adversarial audit OVERTURNED several "walls" — read this first ***
A parallel 4-agent adversarial audit + re-tests corrected multiple over-confident conclusions below. NET: the
situation is substantially BETTER than the body of this doc says. Corrections:
1. "Deep multi-level is wall-time-PROHIBITIVE at eval" = WRONG. Eval = 8h/game WALL-CLOCK, PARALLEL (1 thread/game),
   ~6 games (not 25), NO action cap (my_agent MAX_ACTIONS=inf), measured ~510 act/s. A 400k-step search ≈ 13 min/game.
   Reset+replay deep multi-level IS eval-affordable; snapshot was only an offline speedup. (verified: my_agent.py:76,82,108;
   ARC-AGI-3-Agents/agents/{agent.py,swarm.py}; phase0 spec throughput note.)
2. "Coverage = 1/12 (click-affordance wall)" = WRONG, weak-probe artifact. With object-CENTROID clicks + object_state_key
   (collapses animation jitter) + fine-grid AFFORDANCE probe + realistic budget: 5/12 NO_L0 games crack at L0
   fresh-verified (ft09, ka59, ls20, sk48, wa30). ka59 solved in 11 acts (~ hand-solve) where the old probe exhausted 814.
   Coverage is real, material (zero->win ~4-5%/game under depth-weighting, ~+1% leaderboard vs SOTA 1.17%), EVAL-DEPLOYABLE
   (L0 has no prefix), floor-safe additive. Assets: scratchpad coverage_v2.py, affordance_search.py.
3. "Forward model too weak" / "goal heuristic inert" = SHAKY (tested only movement-models / gradient-free tu93). The
   CLICK-dynamics model was NEVER built (19/25 games). Probe shows click effects are DETERMINISTIC (10/10) + SPARSE (1-2
   cells) but NON-LOCAL (indirection: click->distant cell); a naive local model gets 0% — the RIGHT model (click-target ->
   affected-target mapping) is untested. NOT killed.
4. SHARED BOTTLENECK for both coverage AND deep-multi-level = WITHIN-LEVEL SEARCH QUALITY (object-centric clicks +
   object_state_key + settle-for-physics + affordance probe + distance heuristic). This is ENGINEERING, not a paradigm wall.
   Deep-multi-level already goes deep where the search reaches (tu93 L1->L9 full, m0r0 L2, vc33 L1); it stalls on click
   games only because snapshot_chain used the OLD weak probe.
STILL SOLID (below): snapshot not eval-deployable; squared depth-weighted scorer. STILL OPEN/HARD: generalization to
HIDDEN games (W1) — dev coverage != hidden coverage, though the object-centric methods are general (no per-game code).
Remaining un-cracked NO_L0 (7): cn04, g50t(confined), sc25(inert-start), re86, tn36, bp35, tr87 — need settle/combined probes.
See [[arcagi3-walltime-snapshot]] (corrected).

---


Continues HANDOFF-2026-07-01. A long research push that ran the forward menu to ground, made two real
breakthroughs, corrected the scoring model, and precisely characterized why the eval-time wall does not crack.
Floor (13/25 @ L0, submission) is UNTOUCHED — all work is research (scripts/research_2026_07_01/ + scratchpad).

## Breakthroughs (real, kept)
1. **Role-typed perception** ([[arcagi3-role-typed-perception]]): behavioral typing (agent/collectible/editable/
   goal/wall/hud). Flipped the local reasoner (Exp B) from 0/9 to right-win-frame 6/6 (cd82/tu93/sb26). The
   reasoner's proven value = NAMING the win frame for a deterministic verifier; NOT search-guiding.
   Prototype: scratchpad/role_typing.py.
2. **Double-reset search-replay solves multi-level** ([[arcagi3-closed-loop-multilevel]]): single reset() does
   not wipe -> phantom wins (caught by fresh-env verify; even goal_harness's own positive control phantomed).
   With double-reset, tu93 L1->L5 fresh-verified.
3. **Snapshot oracle cracks multi-level OFFLINE** ([[arcagi3-walltime-snapshot]]): copy.deepcopy(env) snapshot/
   restore eliminates prefix-replay -> tu93 FULL game L1->L9 in 57s, fresh-verified, ~100/100 depth-weighted.
   Proves the ceiling is algorithmically reachable. NOT eval-deployable (eval agent talks REST, no game object).

## The scoring correction (pivotal, changes strategy)
[[arcagi3-efficiency-lever]]: the REAL RHAE (reference/arc-agi-toolkit, = leaderboard) is DEPTH-WEIGHTED:
per-run min( Σ score_i·(i+1) / Σ_ALL_levels(i+1), max_weights/total_weights·100 ), score_i=min(115,(base/act)²·100),
UNREACHED levels score 0 but their weight counts; game score = MAX over runs. => efficient SHALLOW completion is
~worthless (perfect L0 on a 10-level game ≈ 2 pts; verified). The ONLY thing that scores is EFFICIENT DEEP
completion. The offline gate (ab_salience.py) is miscalibrated (linear cap-100, unweighted). Correct offline
scorer built: scripts/research_2026_07_01/ab_efficiency.py (deployed portfolio scores MEAN 0.88/100 on 3 games,
matching leaderboard <1.2%).

## Why the EVAL wall does not crack (all attempted + measured)
Efficient deep multi-level at eval requires either cheap search or model-planning. Every path fails:
- reset+replay search over REST: ~400-500k eval-steps/game (tu93) = over 12h budget at 25 games.
- gradient distance heuristic: only 1.2x node reduction (sokoban-hard levels dominate). Insufficient.
- forward-model internal planning (would be eval-affordable): movement accuracy is game-specific (dc22 100%,
  tu93 83%, m0r0 40%); and even with perfect dynamics it fails on GOAL INFERENCE (dc22: reach role-typed goal
  color != win, which is exact 2x2 coincidence) AND CONFINEMENT (dc22 agent reaches only 36 states, goal
  unreachable by movement, like g50t). tu93 has a consumption process the model misses.
The wall DECOMPOSES into a compounding stack — goal-inference + mechanic-specificity/confinement + model-accuracy
+ search-cost — each individually hard. This is why the whole field is <1.2%. HONEST NEGATIVE; no eval crack found.

## Dead ends confirmed (don't rechase)
- Near-miss refutation ([[arcagi3-nearmiss-refutation]]): works but exploration-bound; source audit = 1/10 clean.
- Multi-win-STATE intersection: win states are unique -> no disambiguation.
- Reasoner-goal as a SEARCH GUIDE: inert (goal==blind, byte-identical) — boolean goals give no gradient.
- L0/first-reward efficiency play: ~2%/game under depth-weighting. Not worth building.

## Assets (scripts/research_2026_07_01/ unless noted)
role_typing.py (scratchpad), goal_inference/ (exp A/B, audit, role_filter), ab_efficiency.py (correct scorer),
efficiency_audit.py, l0_shortest.py, closed_loop*/verify (scratchpad), snapshot_chain.py/reset_semantics.py/
snapshot_test.py (scratchpad), model_nav_chain.py (scratchpad), eval_cost.py (scratchpad).

## Forward-menu items ATTEMPTED this session (both negative — walls confirmed)
- **(a) Oracle-distilled policy** (scratchpad/leave_one_out.py): imitation policy over transferable relational
  features (agent->goal quadrant + wall occupancy), trained on N-1 synthetic games' snapshot-oracle trajectories,
  leave-one-game-out. RESULT: PARTIAL + BRITTLE transfer — maze solved (27 vs random 94), navg optimal w/o wall
  feats, but wall feats helped maze while breaking navg/push; navgc/collect/switchdoor fail. No robust general
  policy. Confirms the generalization ceiling (bet-3 redux). NOT a deployable crack.
- **(b) Coverage via L0 first-reward search** (scratchpad/coverage_l0.py): KEY — L0 has no prefix so reset+replay
  L0 search IS eval-affordable + floor-safe. Ran role-typed-goal-guided L0 search on the 12 NO_L0 games. RESULT:
  1/12 (sk48 only, fresh-verified). Most games: role-typing found NO goal (goal=[]) -> blind search; several
  exhausted the frontier in <130 nodes -> the winning interaction is OUTSIDE the generic action set (specific
  clicks, not _salient) = the click-affordance wall ([[arcagi3-bet2-click-affordance-killed]]). Generic L0 search
  can't crack game-specific first-reward mechanics.

## Forward menu (what's actually left, ranked by honesty)
1. **Amortized POLICY distilled from the snapshot oracle.** The oracle solves dev games fully; imitation-learn a
   reactive policy (state->action) from its optimal plans and deploy it (fast, no search at eval). The ONE
   untried eval-compatible route to efficient deep play. RISK: generalization to hidden games (the W1 wall;
   convex-hull-closed says novel mechanics ≈0). Kill experiment: leave-one-game-out — does the policy transfer?
2. **Reasoner-as-namer for FIRST-REWARD on zero/NO_L0 games** (role-typing + reasoner naming + env verify) to
   convert 0-score games to >0 (coverage). Depth-weighting still rewards this modestly.
3. **Accept the wall; ship the floor.** The deployed 13/25 portfolio ≈ field median; further gains need a
   generalization breakthrough not in the current toolset.
