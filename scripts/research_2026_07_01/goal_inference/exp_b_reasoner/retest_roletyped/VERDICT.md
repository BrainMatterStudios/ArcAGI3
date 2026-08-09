# Exp B RE-TEST with role-typed scene (2026-07-01)

Follow-up to Exp B (naive flat object-list → 0/9 MATCH, models locked onto status bars).
Change: feed a ROLE-TYPED scene (behavioral typing: agent/collectible/editable/goal?/wall/static;
HUD masked) instead of a flat object-list. Same 3 games, same 2 top models, temp 0, ollama.

## Result: role-typing UNBLOCKS win-condition naming.
0/9 MATCH (flat)  →  right FRAME on 6/6 (game × model) queries (role-typed). No status-bar lock-on.

| game | true win (from source) | qwen2.5:14b | qwen2.5-coder:14b |
|------|------------------------|-------------|-------------------|
| cd82 | editable 10x10 == target pattern (except 2 diagonals) | "editable must match a goal pattern" (+spurious 'collectibles removed') = MATCH-frame | "editable arranged/filled to a specific pattern" = MATCH-frame |
| tu93 | every movable box on an exit tile | "9 goal cells occupied by movers/agents" = MATCH | "agents reach the goal cell (color14)" = MATCH |
| sb26 | fill each slot to reproduce target color pattern | "editable arranged to match static reference pattern" = MATCH | "fill editable to match static reference patterns" = MATCH |

## Reading
- The blocker for the local-reasoner paradigm was PERCEPTION (flat object-list), not model capacity or latency.
- Role-typing (which object is agent / editable / goal / HUD) is the missing primitive shared with Exp A.
- Reasoner output is a FRAME, not an exact predicate — precision (e.g. cd82's diagonal exception) is left to
  the deterministic env-replay verifier. That is the intended division of labor.
- Caveats (honest): typing is imperfect (tu93 over-labels 3 movers as agent; cd82 color4 'collectible' likely
  the rotating pattern not a true pickup). Improving typing precision (esp. exploration-bound collectibles)
  is the next lever, but even imperfect typing already flips the outcome.

Raw scenes/prompts/outputs in this directory. Ground truth in ../ground_truth.md (never shown to models).
