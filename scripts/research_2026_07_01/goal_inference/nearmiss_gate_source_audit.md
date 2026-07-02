# Near-miss gate — source-grounded precision audit (2026-07-01)

Generalization-skeptic workflow (10 parallel agents, each reads game SOURCE win-check) judging whether the
predicates that SURVIVED the completion-aware near-miss gate are the TRUE goal or a slipped-through correlate.

## Verdicts (10 GOAL-verdict games)
- TRUE_GOAL_FOUND (1): **cd82** — Template = editable-region == reference pattern (except 2 diagonals). Clean.
- ONLY_CORRELATES (3): **ar25, dc22, su15** — every surviving predicate is a false goal.
- MIXED (6): lf52, lp85, m0r0, s5i5, tu93, vc33 — some partial/true, some correlate.

## Why correlates slipped (the key pattern) — almost all on NON-ENTITY colors
- ar25: CoverAll(color10) — color10 is the MIRROR-AXIS PALETTE color; true win = target cells covered by block+mirror reflections.
- dc22: CoverAll(color2) — color2 vanishing is an artifact of next_level() swapping the whole grid; true win = avatar reaches goal (reach/nav).
- su15: Collect(color3) — color3 vanish is a win-FLASH recoloring byproduct; true win = exact per-category tile/piece count in goal region.
- m0r0: Reach(color6) — color6 is a level PALETTE/FRAME color, not an entity; true win = merge mirror-linked pieces to zero.
- s5i5: CoverAll(color2) — incidental bulk vanish; true win = each arm TIP coincident with its target (geometric placement).
- lp85: CoverAll(color8/11) — color8 is the BUTTON/GLYPH color; true win = each 2x2 block centered in its color-matched frame.
- lf52: CoverAll(color9) PARTIAL — peg-solitaire; pieces vacate as side-effect but real rule = reduce to exactly 1 peg.
- tu93: CoverAll(color14) — color14 = exit tiles; covering them ≈ boxes-on-exits (closest of the covers; rated strict).

## Reading
- The completion-aware near-miss gate is NOT precise enough to plan toward alone: 1/10 clean true goal.
- The slipped correlates are dominated by PALETTE / FRAME / GLYPH / bulk colors — exactly what behavioral
  ROLE-TYPING demotes (not agent/goal/editable). => role-typing is the natural precision filter for the gate.
- Confirms [[arcagi3-nearmiss-refutation]] exploration-bound limit against ground truth, and motivates the
  role-typed-perception lever ([[arcagi3-local-reasoner-killed]] re-test flips 0/9 -> right frame 6/6).

Full structured verdicts: workflow run wf_3dc81488-f4c return value.
