# C6 — W2 judge feedback (FAILED)

issues:
- CRITICAL (policy.py _planner_step): _planned_this_level=True is latched before the occ.usable gate; at first planner reach mm.deltas often covers only one axis so occ is unusable, the one-shot is wasted, and the planner never plans again that level. Measured: 0 planning attempts on maze/switchdoor across all levels. Move the latch to after a real planning attempt.
- CRITICAL: C6.md acceptance gate (lines 135,159) requires push actions-to-solve to DROP; measured push=3051 OFF vs 3052 ON (one action worse). Planner returns an executable action 5x total on push and 0x on collect/switchdoor/maze/navg. Feature delivers no gain; do not flip enable_planner default to True.
- IMPORTANT (policy.py _map_c5_goal): TOGGLE_THEN_REACH mapping never sets push_color, but planner._plan_toggle_then_reach uses goal.push_color as the door-unblock color, so C5-sourced toggle goals can never open doors and will decline whenever the goal is sealed behind one. Latent-dead today (masked by the latch bug).
- IMPORTANT (policy.py _planner_step/_ground_plan_inputs): bare `except Exception` plus the silent decline paths make a non-functional planner indistinguishable from a working one; no decline-reason counter exists so the design's measurement A/B gate cannot actually be evaluated.

feedback:
C6 passes the no-regression bar (flag default-OFF, empirically byte-identical OFF-vs-ON across all 8 local games; full suite 134 pass; push not regressed below baseline; spatial.py untouched). However it FAILS its own acceptance gate and contains a defect that makes the planner nearly inert when enabled.

CRITICAL — One-shot latch consumed before usability gate (policy.py `_planner_step`): `self._planned_this_level = True` is set BEFORE `if not self.occ.usable: return None`. At first planner reach, mm.deltas usually has only one axis observed (measured {4:(0,10)} on maze, {4:(0,12)} on switchdoor), so OccupancyMap(mm.deltas).usable is False, the planner returns None, but the one-shot is already burned and never re-plans that level. Measured effect: planner reaches grounding 0 times on maze and switchdoor (all 3 levels each); 0 executable plan actions on collect/switchdoor/maze/navg. The planner is dead code on exactly the pathfinding games it targets. Fix: latch _planned_this_level only after a real planning attempt (after occ.usable/occ-build succeeds).

CRITICAL — Acceptance gate unmet (C6.md lines 135/159 require push actions-to-solve to DROP): measured push = 3051 actions flag-OFF vs 3052 flag-ON (one WORSE). Planner fires exactly once on push then the latch kills it; other levels solved by graph fallback. No measurable gain anywhere. Do NOT promote enable_planner default to True.

IMPORTANT — C5 TOGGLE mapping drops door_color: `_map_c5_goal` TOGGLE_THEN_REACH branch never sets push_color, but planner._plan_toggle_then_reach uses goal.push_color as the door-unblock color, so any door-sealed goal declines. Latent-dead today (masked by the latch bug), real once that's fixed.

IMPORTANT — Bare `except Exception` in _planner_step/_ground_plan_inputs silently degrades to no-plan with no decline-reason counter, so a dead planner is indistinguishable from a working one and the measurement gate can't actually be evaluated. Add a debug counter.

GOOD: flag-OFF guarding verified correct; sokoban corner-deadlock pruning correct and isolation-tested; coordinate discipline consistent; occ.blocked restored in finally; per-level occ reset correct.
