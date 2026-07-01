# Research 2026-07-01 — strategy reframe + Exp 1/Exp 2

Evidence scripts behind `docs/2026-07-01-strategy-reframe.html` and `docs/2026-07-01-experiments-results.html`.
Run from repo root with `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/research_2026_07_01/<script>`.

- **efficiency_audit.py** — Exp 1. Drives SalienceExplorer, computes ACTUAL offline per-level RHAE using the
  REAL human `baseline_actions` (present in all 25 local envs) vs the geodesic-optimal ceiling.
  Result: mean per-level RHAE 0.070 → 0.693 = **~10× recoverable**; headroom lives on deep levels L1–L4.
- **dc22_verify.py** — in-engine verification of the decoded dc22 win-condition (avatar c14 → goal c11).
  Confirms move=±2/action, L0 step-limit=128, and that pure nav is wall-blocked (gate mechanic required).
- **goal_solver.py** — Lever-B seed: auto-detect avatar + bind visible goal + greedy nav + stall-interaction.
  `<game> <goal_color> [budget]`. Result: binds the goal on all 4 tractable games but **0 wins** — goal-binding
  is easy, per-mechanic planning is the wall.
- **inspect_offline.py** — offline initial-frame dumper (coarse grid + object/color histogram) for entity ID.

Key finding: the 8 "W1 zero-games" all have VISIBLE goals; the barrier is hidden-mechanic discovery + planning
depth, not goal unobservability. See memory `arcagi3-leaderboard-is-percentages` and `arcagi3-cgpp-goal-prior-bet`.
