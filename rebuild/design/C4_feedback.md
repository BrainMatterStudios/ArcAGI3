# C4 — W2 judge feedback (FAILED)

issues:
- CRITICAL (95): Scene.key() (forward_model.py:131-139) builds tuples per-entity, but P.object_state_key merges 4-adjacent same-color cells into one connected component. After a push merges/splits same-color regions, predicted key != object_state_key(real grid) on a CORRECT, confident prediction. Breaks the linchpin: verify_step returns False on correct predictions (reproduced), C6 aborts valid plans, keys fail to unify with WorldModel.nodes. Fix: re-derive connectivity in Scene.key() (rasterize cells + connected_components, or merge same-color adjacency before _object_tuples).
- TEST GAP (90): test_scene_key_equals_object_state_key only checks freshly-LIFTED scenes (entities==components trivially). The linchpin is never asserted on a PREDICTED scene whose entities merged/split; push-fidelity test uses single-block push so the merge case is never exercised. Add regression tests for push-causes-merge and split.
- GENERALITY (80): push_goal reward branch (forward_model.py:330-339) relies on the target color being labeled GOAL, but real C3 (affordance.py _observe_step) attributes the level-up reward to the BLOCK color (6) on contact, never to the target (4). With _C3Adapter the push-success/goal modelling is unreachable; tests pass only via hand-stubbed GOAL for color 4. Add a test using real _C3Adapter verdicts.
- MINOR (80): avatar walking ONTO a GOAL sets reward=1.0 (forward_model.py:318-320) even with no push; in push.py agent-on-target does not win. Latent reward hallucination for delivery-style goals; document avatar-goal vs delivery-goal distinction.
- PROCESS (low): `git diff` shows only perception.py; 3/4 of C4 (forward_model.py, test_forward_model.py, fm_consistency.py) are untracked and hidden from a plain diff review. Use git status / git add -N when reviewing.

feedback:
REVIEW SCOPE: C4 forward world model. Tracked diff = perception.py only (extract _object_tuples). Untracked (the bulk of C4, also reviewed): src/arcagi3/forward_model.py, tests/test_forward_model.py, scripts/fm_consistency.py. NOTE: `git diff` alone hides 3/4 of the change because the new files are untracked — review them with `git status`/`git add -N`.

VERDICT: NO REGRESSION RISK to push or the local suite — that part is solid and well-defended. But there is ONE real, high-confidence correctness bug that defeats the design's central safety claim (the "linchpin"). It is latent today only because nothing imports C4 yet and the tests never exercise the failing case; it will bite C6/C7 the moment they consume C4.

=== NO-REGRESSION: PASS (verified, not asserted) ===
- Grep gate empty: `grep -rn forward_model src/arcagi3/{policy,agent,world_model}.py` returns nothing (exit 1). C4 is not in the decision path.
- Full suite: 111 passed (10.3s). forward_model+perception: 33 passed. Push semantics untouched (games/push/push.py unchanged; HybridPolicy unchanged).
- perception.py refactor is BYTE-IDENTICAL to the old inline body: verified programmatically across 50 random grids x 4 ignore-variants (None/empty/{1}/{1,2,3}) — all repr() bytes matched. test_perception stays green. `ignore = ignore_colors or set()` correctly preserves the old None-handling.
- scripts/fm_consistency.py runs read-only and reports sane numbers (navg 99% confident / 97% known-acc; collect 53% confident / 97% known-acc) with zero behaviour change. (Must be run with PYTHONPATH=src — `uv run` does not pick up pyproject's pytest-only pythonpath; the script's own docstring says PYTHONPATH=src, fine.)

=== CRITICAL BUG (confidence 95): Scene.key() breaks the linchpin on same-color merge/split ===
Scene.key() (forward_model.py:131-139) builds object tuples PER-ENTITY (one (color,bbox,size) per Entity). But P.object_state_key derives tuples from P.connected_components, which MERGES any 4-adjacent same-color cells into ONE component. These are only equal when every entity is its own isolated component — i.e. for a freshly-lifted scene. After predict() moves an object so two same-color regions become adjacent (or splits one apart), the predicted key OVER-SEGMENTS relative to the real grid's key and they DIVERGE.

Reproduced (avatar pushes block@(5,4)->(5,5); a second same-color block sits at (5,6); push is legal, PUSH conf 1.0, known=True):
  predicted key tuples: [(6,5,5,5,5,1),(6,5,6,5,6,1),(14,5,4,5,4,1)]
  real grid tuples:     [(6,5,5,5,6,2),(14,5,4,5,4,1)]   # the two 6-blocks are now ONE size-2 component
  => p.scene.key() != object_state_key(real_next)  on a CORRECT, CONFIDENT prediction.

Consequences for the stated design contracts (all in the docstring/design as the reason C4 is safe):
1. verify_step() returns FALSE on a correct confident push (demonstrated). The design says C7 uses this to "disable planning per-game on confident-but-wrong" — so the self-falsifying safety net misfires AGAINST correct predictions, the opposite of intended.
2. C6's "abort plan on first key mismatch" aborts valid plans, wasting actions.
3. Predicted keys fail to unify with WorldModel.nodes for any merged-component state, breaking predict-then-verify graph unification.
This is general: ANY two same-color entities the avatar's motion brings into adjacency (multi-block sokoban, any game with multiple same-color movable pieces) trips it. The single-block Push game never does, which is exactly why test_push_fidelity_against_real_env passes and the bug is invisible.

WHY THE TESTS MISS IT: test_scene_key_equals_object_state_key (test_forward_model.py:60-71) only checks freshly-LIFTED scenes (entities==components by construction => trivially equal). The linchpin is never asserted on a PREDICTED scene whose entities merged/split. The only predicted-vs-real key check (push fidelity) uses single-block push.

FIX: Scene.key() must re-derive connectivity, not trust per-entity boundaries. Either (a) union all entity cells into a temp grid (bg elsewhere) and call P.connected_components, or (b) merge same-color 4-adjacent cells across entities before _object_tuples. Then add a regression test: predict a push that brings two same-color blocks adjacent and assert predicted key == object_state_key(real grid); also a split case.

=== GENERALITY GAP (confidence 80, not a regression): push_goal reward is unreachable with the real C3 adapter ===
The push-onto-goal reward branch (forward_model.py:330-339) and test_push_onto_goal_wins / the fidelity test all MANUALLY stub GOAL:(GOAL,1.0) for the target color (4). But real C3 (affordance.py _observe_step) only labels a color GOAL when AVATAR CONTACT yields reward. In push.py the level-up fires when the BLOCK lands on target (next_level), on a step where the avatar contacts the BLOCK (color 6) with reward>0 — so C3 records GOAL for color 6 (the block), not color 4 (the target). The target color is only ever walked over with reward 0 => PASS. So with _C3Adapter, color 4 is never GOAL and the push_goal branch never fires; worse, color 6 gets conflicting GOAL/PUSH evidence. C4 is correct given its inputs, but its push-success modelling depends on an affordance the real C3 will not supply. Flag for C6/C7 wiring; document or add a test using the real _C3Adapter verdicts rather than hand-stubbed GOAL.

=== MINOR (confidence 80, by-design but undocumented edge): avatar-onto-GOAL over-rewards ===
forward_model.py:318-320: any GOAL contact in the contacts loop sets reward=1.0 even when the avatar merely walks ONTO the goal (no push). In push.py walking the agent onto the target does NOT win — only block-on-target wins. For games where GOAL means "avatar reaches it" this is right; for push-style "deliver object to goal" it would over-predict reward on a plain walk-over. Given the C3 mislabel above this is currently masked, but it is a latent reward hallucination for sokoban-style goals. Worth a comment distinguishing avatar-goal from delivery-goal.

OVERALL: The refactor and the no-regression posture are excellent and verified. Land the perception refactor as-is. Do NOT rely on Scene.key()/verify_step downstream until the connectivity bug is fixed — it silently inverts the very safety mechanism the design leans on.
