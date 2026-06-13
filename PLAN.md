# ARC-AGI-3 — Plan to Win (ARC Prize 2026)

> Goal: win the ARC Prize 2026 ARC-AGI-3 Kaggle competition.
> Scoring: levels completed (primary) → fewest total actions (tiebreaker). 6 games
> (3 public, 3 private), 8–10 levels each, escalating mechanics. Offline eval, ≤12h,
> no internet (no hosted LLMs). Must open-source (MIT-0/CC0).

## North star

A **general, training-free interactive agent** that, dropped into an unseen game with
no instructions, explores efficiently, infers the goal from the `levels_completed`
reward, builds a world model, and plans to clear levels with minimal actions. It must
**generalize to unseen private games** — no per-game hardcoding.

Beat the documented SOTA (graph-based exploration, 3rd place, ~16–19 levels) and push
toward 100%.

## Architecture

```
frame (N,64,64) ─▶ Perception ─▶ object-centric state + state-hash
                                      │
                          World Model / State Graph (transitions, dynamics)
                                      │
                   Explorer + Planner (frontier exploration → goal-directed planning)
                                      │
                                  GameAction
```

- **Perception** (`perception.py`): connected-component objects, background detection,
  dynamic/status-bar cell masking, salient click-target proposal, state hashing.
- **World model / graph** (`world_model.py`): directed graph of states & action
  transitions; learned dynamics; reward (Δlevels_completed) tracking.
- **Explorer/planner** (`agent.py`): hierarchical action selection — exploit known
  reward-increasing paths; otherwise explore the frontier (states with untried
  actions), prioritized by salience; replan via shortest path; use ACTION7 (undo) for
  safe probing.
- **Runner/eval** (`runner.py`, `eval.py`): drive Arcade offline, budget control,
  per-game levels+actions reporting, regression suite.
- **Local game suite** (`games/`): diverse home-made ARCBaseGame environments
  (navigation, click-target, push/sokoban, toggle/switch, collect, sequence) to develop
  & regression-test generalization. Validate on real public games when an API key
  is available.

## Phases

1. **Foundation** (in progress): offline dev loop ✅, perception, runner, eval harness,
   2–3 local games, a graph-exploration baseline. Measure levels/actions.
2. **Match SOTA**: status-bar masking, salience-tiered click proposals, frontier
   exploration with shortest-path replanning. Target ~16 levels on the local suite.
3. **Beat SOTA**: learned transition dynamics, object affordances, goal inference,
   planning, cross-level transfer. Larger local suite incl. big state spaces.
4. **Package & submit**: Kaggle notebook, COMPETITION mode, ≤12h budget, MIT-0 license.
   Hit open-source milestone (#1 Jun 30, #2 Sep 30) then final (Nov 2).

## Open external dependency

ARC_API_KEY (register at https://three.arcprize.org) to validate against the 3 *real*
public games (ls20, ft09, vc33, as66…). Not required to build — local games suffice for
iteration — but important to avoid self-deception about difficulty.
