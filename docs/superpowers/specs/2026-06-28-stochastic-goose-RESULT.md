# StochasticGoose standalone build — RESULT (2026-06-28)

## What was built
`src/arcagi3/stochastic_goose_explorer.py` (TDD, 10 tests green): a FAITHFUL STANDALONE StochasticGoose-
class policy — effect-CNN action selection + masked-state-hash dedup + online TTT carried across levels,
NO graph navigation backbone. Wired as `--agent goose`. Banked TransferExplorer (0.33) untouched.

## Decisive measurements (offline dev games, budget 3000, CPU)
- **tu93 (movement, avail=[1,2,3,4]):** transfer L4 (560 states, levelup@431) vs goose L0 (99 states).
  goose-model and goose-NOmodel are IDENTICAL (~100 states, L0) → the CNN is NOT the cause; the cause is
  **no navigation-to-frontier**: once local moves are exhausted goose random-walks instead of pathing back
  to an unexplored frontier, capping coverage at ~100 vs the graph's 560.
- **Click games (lp85/vc33/cd82/tn36/sc25):** transfer L9 vs standalone goose L2. Under-covers everywhere.
- **Coverage-safe composition spike** (`scratchpad/goose_ranked.py` = TransferExplorer + CNN ordering ONLY
  within the equal-tier untried batch, the `_pick_from_batch` hook's documented coverage-safe contract):
  transfer L13 vs ranked L7. **tu93 4→0 even though tu93 has no clicks** — within-tier reorder of movement
  directions alone tanks it. Only sc25 0→2 unlocks (real but narrow), lp85/cd82 regress.

## Conclusion
The faithful-standalone hypothesis is FALSIFIED as a path to beat 0.33: navigation-to-frontier (the graph's
BFS coverage) is essential on BOTH movement and click games, and the CNN effect-model does not compensate
for losing it. Adding the CNN back onto the graph — even via the most conservative coverage-safe within-tier
ordering — reproduces the campaign's documented W3 wall (tu93 4→0; only sc25 narrow unlock). This is the
THIRD independent confirmation this session that learned-CNN-on-our-explorer regresses; it is NOT a bug.

## What this does and does NOT change
- DOES NOT restore the "offline paradigm caps at 0.33" framing — the live leaderboard still shows offline
  methods at 0.5–1.21 (see `arcagi3-033-ceiling-is-codebase-not-paradigm`). Others escape offline.
- DOES establish that the escape is NOT "our salience-graph explorer + a CNN tweak." The leaders use a
  DIFFERENT base (likely: a proper RL agent with the level-completion REWARD as signal, trained at T4 scale
  with off-policy replay + between-level retrain — NOT the self-supervised frame-change effect signal my
  goose used, and NOT a reranker on a salience graph). My goose used the frame-CHANGE label, not the
  level-completion REWARD — the directedness StochasticGoose gets from RL-on-reward was never in it.

## Status
goose kept as a clean tested standalone reference + characterized negative (firewalled: separate `--agent
goose`, banked submission untouched). Scratchpad probes: `ab_goose.py`, `diag.py`, `goose_ranked.py`,
`ab_ranked.py`. Recommended next: the June-30 (2 days) leader-class open-source drop is the evidence-backed
path; the in-session salience-graph+CNN line is exhausted (triple-confirmed).
