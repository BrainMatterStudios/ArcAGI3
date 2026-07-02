# Closed-loop kill experiment: reasoner-goal + search-replay for multi-level (2026-07-02)

Hypothesis: reasoner names goal -> re-instantiate per level -> directed search finds plan -> env verifies.
Test game: tu93 (move-only cover goal, reasoner-named "movers onto goal cells", key color 14).

## Findings

### 1. ⚠️ Near-false-green CAUGHT (handoff lesson #1 validated live)
Single-reset search reported "L1->L2 solved in 8 actions". THREE independent checks refuted it:
- independent fresh-env replay of sol+plan -> level 1 (not 2)
- ablation goal vs blind -> byte-identical (heuristic doing nothing)
- chaining L2->L3 -> "reach base=1" (prefix doesn't actually reach L2)
Root cause: env.reset() does NOT wipe level state (the "resets don't wipe" mechanic); a reused env's search
env desyncs from a fresh env, producing PHANTOM wins. Even goal_harness's OWN positive control phantomed
(reported an 8-move win that evaluate() re-verified as level 1). evaluate() (fresh env) is the trustworthy layer.

### 2. ✅ Double-reset fixes it -> REAL multi-level (first in project)
reach() with env.reset(); env.reset() (double) makes re-reach deterministic (matches portfolio fix df9b96b).
tu93 then chains, ALL fresh-verified: L1->L2->L3->L4->L5. (L5->L6 exhausts the 120s budget.)

### 3. ❌ The reasoner-named GOAL is INERT (the crux)
Goal-guided vs blind (null-heuristic) chains are BYTE-IDENTICAL at every level: nodes 309/1306/885/3699,
both reach L5 in ~118s. The multi-level success is 100% DOUBLE-RESET SEARCH-REPLAY; the goal predicate adds
nothing. WHY: the cover predicate gives NO search gradient for sokoban — boxes cover exits only at completion,
so "exits covered" is flat until the win and best-first degenerates to BFS. A useful heuristic would need
distance-to-exit (requires role-typed agent/box/exit positions + a metric), not the boolean goal.

### 4. ⏱️ Wall-time-prohibitive at depth (confirms [[arcagi3-cgpl-multilevel-prohibitive]])
Per-level: 1.6s, 11.7s, 12.1s, 92.7s (then L6 exhausts 120s). Cost ~ prefix_len x nodes (double-reset replays
the whole prefix per node). Quadratic-ish blowup. Affords a FEW levels on a FEW games in a 12h portfolio eval;
NOT a general/deep multi-level solver.

## Verdict
- POSITIVE: double-reset search-replay genuinely cracks multi-level (verified tu93 L1-L5). Floor-safe as a
  bounded additive play (abstain past a wall-time budget).
- NEGATIVE for the reasoner-goal-as-search-guide hypothesis (inert on tu93; boolean goals give no gradient).
- The reasoner's proven value stays in NAMING for a VERIFIER (Exp B 0/9->6/6), not in guiding search.
- Open (high bar): does a goal help on a HIGH-branching (click) game where guidance could prune the frontier,
  AND with a gradient-bearing heuristic (distance-to-goal via role-typed positions)? Untested.

Assets: scratchpad closed_loop.py, verify_closed_loop.py (double-reset), batch_gate.py, role_typing.py.
