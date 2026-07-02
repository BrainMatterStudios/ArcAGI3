# Role-filtered near-miss gate — precision re-measure (2026-07-02)

Task #4: apply behavioral role-typing as a precision FILTER on top of the completion-aware near-miss gate,
then re-measure against the source audit (10 GOAL-verdict games).

Filter rule (principled, not per-game tuned):
- CoverAll(C): keep iff role(C) in {goal?, collectible}
- Collect(C):  keep iff role(C) == collectible
- Reach(C):    keep iff role(C) == goal?
- Template:    keep iff the matched region overlaps the spatial EDITABLE mask (cells changed by clicks/A5/A7)

## Result (color-predicate filter = strong; template filter = exploration-bound)

Near-miss ALONE: 10 GOAL verdicts, source audit = 1/10 clean true goal (cd82), 3 pure-correlate, 6 mixed.

Near-miss + role-filter:
| outcome | games | note |
|---|---|---|
| correct ABSTAIN | ar25, su15, m0r0, lf52, lp85, s5i5, vc33 (7) | correlate/true-goal-not-capturable; filter drops palette/frame/glyph/static covers cleanly |
| kept GOAL — TRUE/PARTIAL | tu93 (cover on color14=exit/goal marker ≈ boxes-on-exits) | correct keep |
| kept GOAL — FALSE POSITIVE | dc22 (cover on color11=goal marker; true goal is REACH, cover is grid-swap artifact) | 1 residual false goal |
| wrong ABSTAIN — FALSE NEGATIVE | cd82 (true template) | lost: copy region never exercised by random probe |

Net: false goals cut from ~9 → 1 (dc22), at the cost of 1 false negative (cd82). For a FLOOR-SAFE additive
gate this is a good trade — abstaining is safe; shipping a false goal wastes eval budget on a wrong target.

## The unifying meta-conclusion
cd82's false-negative is NOT a filter-logic bug — it is EXPLORATION-BOUND. The editable-region mask is built by
random probing; cd82's built-copy region requires purposeful play (move-rotate + A5-paint at a cursor), which
random probing doesn't exercise, so the mask misses it. This is the SAME wall as near-miss refutation
([[arcagi3-nearmiss-refutation]]) and the SAME wall as W1: everything reduces to EXPLORATION that exercises
the game's mechanics enough to reveal structure. Representation (role-typing) and reasoning (the local model,
which flipped 0/9→6/6) are NOT the ceiling — mechanic-exercising exploration is. The reasoner partially escapes
because it needs only coarse typing to name the win FRAME (then the env verifies precision).

## Deployable takeaway
A role-filtered near-miss gate is floor-safe (abstains rather than ships false goals) and could be added as an
ADDITIVE play that fires only on high-confidence, well-explored cases (cd82/tu93-like). The gating lever for
raising recall is a MODEST purposeful-exploration budget (exercise A5/rotate/drag mechanics), not more filter
heuristics — those would overfit the 10 dev games (generalization-skeptic veto).

Assets: scratchpad role_typing.py (probe + editable mask), role_gate.py (filter + re-measure), batch_gate.py.
