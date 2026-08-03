# A/B round 2 — patch11 (graph) and patch12 (compact) arms (2026-08-03)

Kernel ab-wmr v2, COMPLETE in 6.18h, 6 waves B,C,D,D,C,B, 10-game panel,
serving assert passed, per-wave 6-switch toggle proof correct for every wave.
Token accounting fixed (sums run.history records): ~50-62k gen tokens/game, all rows.

## Headline

| arm | config | levels (2 waves) |
|---|---|---|
| B | WMR trio (= submitted v6) | **20** |
| C | B + frontier graph | 16 |
| D | B + compaction/plan-queue | 12 |

B repeats its round-1 total exactly (20), now stable across two independent
kernels (4 waves). Pooled vs round-1 stock A (16): WMR +25% directional holds.

## Verdict: patch11 (C) — mechanisms NEVER FIRED; no ship, no harm proven

Graph built (60-455 nodes/game) but across all 20 C game-runs: vetoes_issued=0,
grinder_engagements=0, levels_unlocked_by_grinder=0. The intervention conditions
never occur in real play: the 27B rarely repeats an exact no-op at the same
masked node (veto needs >=2), and the watchdog stall signal (900s no-progress)
almost never fires while an LLM keeps emitting actions (1 stall in 60 game-runs
this round). With zero interventions C is mechanically B; C-B = -4 levels is
within the A/A floor (sd of a 20-run sum diff ~4.4). Ship verdict: NO — dormant
machinery. Revisit only with retuned triggers (frontier-exhaustion grinding,
shorter stall window, veto>=1 with stricter guards) — and re-A/B.

## Verdict: patch12 (D) — pre-registered KILL, mechanism broken in production

- queue_plans = 0 in all 20 D game-runs: the 27B never emitted a plan_queue
  despite the system-prompt guidance. llm_calls_saved = 0.
- compactions: 3071 evictions seen, 303 compaction attempts, **0 succeeded,
  303 failed** (~7.5 failures/game — each a wasted bounded LLM call inside the
  game loop; fallback to silent drop worked as designed).
- D = 12 levels vs B = 20 (-8, ~1.8 sigma — the largest arm gap either round
  has produced, in the harmful direction, with broken mechanics as the
  plausible cause).
Pre-registered criterion was "kill if delta <= run noise"; actual is delta
negative with zero functional benefit. KILLED. If ever revisited: first fix the
compaction-call failure mode offline (inspect response/parse/timeout), and
accept that plan-queue JSON emission likely needs SFT, not prompting, at 27B.

## Consequences

1. Pinned config stays WMR trio (v6) — tonight's live draw measures it.
2. Patches 11/12 remain in the bundle but pinned OFF (v6 pins already do this).
3. The unlock gap persists: nothing in this round moved g50t (0 in all 12
   runs across both rounds) or cracked deeper levels. Remaining unlock levers:
   SFT track (double-gated) and capability-side ideas — the harness-side menu
   from the research report is now largely explored.
