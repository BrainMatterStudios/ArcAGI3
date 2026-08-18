# Explorer-floor fixtures — July mechanical wins re-derived offline (2026-08-18)

Replayable, engine-verified action sequences for the six "explorer-winnable" games,
re-derived against the LOCAL offline engine (`arc_agi` OFFLINE mode,
`environment_files/`, `ONLY_RESET_LEVELS=true` for competition parity). Pure CPU,
frame-only perception — no engine internals, no network.

Derivation: `submission/_explorer_floor/solve_floor.py`
(reset-replay BFS with HUD-masked frame dedup per `scratchpad/ideas/hud_mask.py` —
the dc22 technique of `scripts/research_2026_07_01/general_search.py` generalized —
then delete-and-replay minimization per `minimize_solution.py`).
Each fixture was verified by replaying on TWO fresh envs: `levels_completed`
advances to the recorded level and every per-step frame is identical across runs.

| game | solved level | actions | search time | method |
|------|--------------|---------|-------------|--------|
| dc22 | 1 | 20 | 2.5 s | BFS, moves + panel-button clicks (bridge toggles) |
| ka59 | 1 | 11 | 26.4 s | BFS, moves + dynamic click-select of units at CURRENT position |
| m0r0 | 1 | 15 | 5.9 s | BFS, moves + A5 + click-select of pieces |
| sk48 | 1 | 14 | 5.4 s | BFS, moves (chain grow/turn) |
| wa30 | 1 | 26 | 0.5 s | frame perception + A* over grab-drag forward model (`wa30_planner.py`) |
| tu93 | **9 (ALL)** | 185 | 85 s total | per-level BFS, moves only (coupled-unit movement) |

All 6/6 targets solved; no failures. tu93 cleared the FULL game (9/9 levels; the
July campaign reached 8/9). Solution lengths are at or below the July human-proxy
baselines (dc22 20 vs 20, ka59 11 vs 28, m0r0 15 vs 30, sk48 14 vs 61, wa30 26,
tu93 L1 18 vs 19) — BFS segments are optimal-length, so delete-and-replay
minimization removed nothing further.

Notes captured during re-derivation:

- **ka59**: clicks on block/unit objects are visual no-ops in the frame; selection
  is a 1-px center-marker on the two c14 units (c0 vs c5). Click targets must be
  recomputed per node at the units' CURRENT positions, and the salient-target
  filter's "centroid pixel == component color" rule excludes exactly these units —
  the July `ht_ka59_solve.py` dynamic-click insight was required.
- **wa30**: blind BFS is intractable at L1 depth (~26+ actions, 113k masked states
  explored without a win in 10 min); the July frame-only planner cracks it in 0.5 s.
- **sk48/m0r0**: crack easily under plain masked-frame BFS (5-6 s each), consistent
  with their membership in July's `minimize_solution.py` solved list.

Fixture schema: `{game, versioned_id, level, actions: [{name, x?, y?}...], length,
verified, frames_checked, search_stats, derived}`. Replay with
`GameAction.<name>` (+ `data={x,y}` for ACTION6) from `env.reset()` on a fresh env.
