# ENGINE / SCORING MECHANICS — VERIFIED FINDINGS (2026-06-28)

Evidence-grounded investigation of long-open engine/scoring questions (user: "continue to research and
explore new ideas"). All read from the official `arcengine` / `arc_agi` packages + confirmed by offline runs.

## 1. Reset cost — CLOSED (negative). Resets are NOT free and do NOT wipe scored cost.
My scoring note had flagged "reset costs unclear — could reshape everything." Resolved:
- The ENGINE's internal `_action_count` (`base_game.py:278`) excludes RESET and zeroes on level-reset
  (`set_level:160`). This is NOT the scored metric — it only routes full-vs-level reset (`handle_reset:313`).
- The SCORED metric is the SCORECARD (`arc_agi/scorecard.py`): `inc_reset_count` does `self.actions += 1`
  (line 592) — **a RESET counts as a scored action** — and `self.actions` is a monotonic cumulative
  counter; per-level A_m = the DELTA of cumulative actions between level-ups (line 300/417).
- Empirically (offline vc33): 5 actions → `actions=[_,5]`; a mid-level RESET → `actions=[_,6]`, `resets=[_,1]`
  (level-reset costs +1, does NOT wipe); further RESETs (engine count 0) become FULL-resets that start a
  new play slot at ~0 and DISCARD all level progress (score→0).
- Consequence: explore a level for 500 actions, reset, replay the 13-step solution → scored cost
  500+1+13, WORSE than not replaying. **Geodesic-replay-within-a-play cannot recover the 35× headroom.**

## 2. Action space — CLOSED (negative). No missing interaction modality.
Actions: RESET, ACTION1-5 (simple), ACTION6 (click/complex with x,y), ACTION7. The "placeable-sprite"
mechanic reuses ACTION6 at coordinates over `placeable_areas` (`base_game.py:517`) — our dense click
lattice already covers it. ACTION7 is the known-inert button (excluded by SalienceExplorer). We exercise
the full action space; there is no untried modality behind the games we score 0 on.

## 3. Per-game score = MAX over plays — VERIFIED (new, positive in principle).
`EnvironmentScoreList.score = max(run.score for run in self.runs)` (`scorecard.py:181`); `from_scorecard`
builds one run PER full-reset PLAY (line 480-489); final leaderboard = mean over games of each game's
**best play** (line 503). And the AGENT's learned graph PERSISTS across a full-reset (separate Python
object). So in principle: Play 1 explores (low score, not the max), full-reset, Play 2 replays the learned
solution efficiently, env score = max = Play 2. This is the mechanism by which the 35× headroom COULD be
captured — and it was never exploited (our eval + submission drive single plays).

## 4. BUT geodesic replay does NOT physically work — DEMONSTRATED (the blocker).
End-to-end offline test (`scripts/multiplay_geodesic_probe.py`): Play 1 on vc33 completes 2 levels; build
the BFS-shortest action path root→reward-trigger over the learned graph (68 actions); full-reset (root
frame byte-identical); replay → **0 level-ups**, divergence at **step 0**. Cause: the SalienceExplorer's
masked state-key is NON-STATIONARY (volatility/border mask evolves over the run) and ALIASES distinct
physical states, so graph edges don't compose into an executable trajectory. The 35× "headroom" is a
graph-DISTANCE fiction, not a physically realizable path. **This is the concrete reason it's uncapturable**
(deeper than "greenlit but uncaptured" — the greenlight is hereby REVOKED).

## Net
Two old questions closed negative (reset-cost, action-space). One genuinely-new scoring fact verified
(max-over-plays + graph persistence) — the only fresh positive surface in the campaign — but the obvious
exploit (geodesic replay) is blocked by state-aliasing desync.

## The one remaining untested variant (decision needed, NOT a cheap probe)
Capturing the max-over-plays lever needs a PHYSICALLY-FAITHFUL short replay, i.e. a Go-Explore-style
explorer that stores the literal shortest ACTION TRAJECTORY (from reset) to each cell with a STATIONARY
key, and iteratively shortens it (replay-to-cell then explore). Risks: (a) basic Go-Explore was already
killed on single-play eval (but never under the max-over-plays insight); (b) stationary/exact keys explode
the graph on wall games (the reason masking exists) — though the 0.33-regime games we already complete do
NOT explode, so a faithful replay there could lift their per-level efficiency toward the 1.15 cap;
(c) the offline client's max-over-plays must match the real Kaggle grader (likely, but only a submission
confirms). This is a multi-day build with real uncertainty — present for a go/no-go, do not auto-build.
