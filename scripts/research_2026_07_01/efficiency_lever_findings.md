# Within-level RHAE efficiency lever — findings (2026-07-02)

Forward-menu #4. Question: is there recoverable per-level efficiency headroom beyond the deployed
GeodesicReplayExplorer, and is it worth building?

## 1. Scorecard MISMATCH (measurement issue, affects all efficiency work)
- Installed arc_agi 0.9.1 (used by the PRIMARY offline A/B sweep scripts/ab_salience.py via client.open_scorecard):
  per-level score = min((baseline/actions)*100, 100)  -> LINEAR, cap 100.
- Reference toolkit + real leaderboard + src/arcagi3/bakeoff_metrics.py: min((baseline/actions)^2*100, 115)
  -> SQUARED, cap 115.
- Consequence: the main offline gate UNDER-rewards efficiency. Below baseline it caps at 100 either way, but
  the squared eval (a) caps ~7% under baseline and (b) QUADRATICALLY penalizes levels ABOVE baseline. Efficiency
  gains that push a sub-baseline level are invisible in the linear gate but valuable in the real eval.
  FIX: score the efficiency A/B under bakeoff_metrics.rhae (squared), not the Arcade linear scorecard.

## 2. Residual headroom is concentrated at FIRST REWARD (L0)
efficiency_audit.py (SalienceExplorer, budget 8000), per level: human baseline vs actual vs explored-graph
geodesic. L1+ geodesics already hit the cap (rhae_geo=1.15) -> deployed GeodesicReplayExplorer captures them.
But L0/first-reward geodesics stay WELL ABOVE baseline (sub-cap): tu93 L0 geo=67 vs human=19 (rhae 0.08),
ar25 L0 geo=101 vs 32 (0.10), lf52 L0 geo=71 vs 32 (0.20), su15 L0 geo=36 vs 22 (0.37), cd82 L0 unreachable.

## 3. A dedicated BFS-to-first-reward RECOVERS it (big, cheap, floor-safe)
True BFS shortest path to first level-up (exact-frame keys, DOUBLE-RESET re-reach, fresh-verified):
| game (L0) | deployed geodesic | BFS true-shortest | human baseline | RHAE(sq): geo -> BFS |
|-----------|-------------------|-------------------|----------------|----------------------|
| tu93 (moves)  | 67 | 18 (2114 nodes, 6s) | 19 | 0.08 -> 1.11 (~14x) |
| lf52 (clicks) | 71 |  8 (1001 nodes, 4s) | 32 | 0.20 -> 1.15 cap    |
| su15 (clicks) | 36 | NOT FOUND (frontier exhausted at 284 states, 1.7s) | 22 | abstains |
BFS beats even the human baseline where the action set covers the solution (tu93, lf52). Fails cleanly where
the winning click is outside the _salient target set (su15) -> ABSTAIN (floor-safe). Deployed geodesic is far
from optimal at L0 because it only searches its incidental exploration graph, not a true BFS.

## 4. Why the deployed geodesic misses it
GeodesicReplayExplorer replays shortest-path IN THE EXPLORED GRAPH (SalienceExplorer's wandering edges), which
lacks the optimal edges. A dedicated BFS to first reward finds the true shortest path.

## Recommended build (floor-safe, additive)
A "shortest-first-reward" portfolio play: BFS the exact-frame state graph from the level start to the first
level-up (double-reset per expansion, capped nodes/seconds), replay it as its own play (double-reset -> new
play, max-over-plays), ABSTAIN if not found in budget. Targets the validated L0 residual headroom; invisible in
the linear offline gate so MUST be A/B'd under squared RHAE (bakeoff_metrics). Wall-time ~2-6s/game where it
works; only helps first-reward (but that is 1/2-1/3 of a shallow game's averaged score).

Assets: scratchpad efficiency_audit.py, l0_shortest.py. See [[arcagi3-rhae-headroom]], [[arcagi3-closed-loop-multilevel]] (double-reset).

## *** CORRECTION (2026-07-02): the L0 finding is LOW-VALUE under the REAL formula ***
The authoritative RHAE (reference/arc-agi-toolkit scorecard, = leaderboard) is DEPTH-WEIGHTED:
per-run score = min( sum_i score_i*(i+1) / sum_ALL_levels (i+1),  max_weights/total_weights*100 ),
score_i = min(115,(base_i/act_i)^2*100); UNREACHED levels score 0 but their weight STILL counts;
game score = MAX over runs. => a perfect L0 on a 10-level game is worth ~2 points (weight 1/55).
Verified with ab_efficiency.py (official formula, reads engine scorecard runs) on the deployed portfolio:
  tu93 0.31 | lf52 2.32 (best run: perfect L0=8 actions, nothing deep) | m0r0 0.00 | MEAN 0.88 / 100.
This matches the leaderboard reality (whole field <1.2%).
CONCLUSION: within-level efficiency COLLAPSES INTO multi-level — the score is dominated by efficiently
completing DEEP levels, which needs multi-level solving (wall-time-prohibitive, see closed_loop_results.md).
DO NOT build the shortest-first-reward L0 play (~2%/game). Use ab_efficiency.py (this dir) as the correct
offline scorer; ab_salience.py (linear cap-100, unweighted) and installed arc_agi 0.9.1 (linear) are wrong.
