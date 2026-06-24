"""Planning over the model beam (winning/mechanic-model-search, Phase 5).

For the highest-confidence model, enumerate plausible goal hypotheses, plan to each through the
executable model, and return the shortest plan that passes the acceptance gate. The WinningExplorer
executes only a short prefix, verifies predictions, and replans — so a MOVING goal is chased by
re-planning to its freshly-observed position, never by trusting a long brittle plan.

Acceptance gate (mission Phase 5):
    model.transition_accuracy >= MIN_CONF (0.85)
    plan_length <= MAX_LEN
    goal in the enumerated plausible goals (guaranteed by construction)
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_CONF = 0.85
MAX_LEN = 80


@dataclass
class BeamPlan:
    goal: object
    plan: list
    model: object
    confidence: float


def best_plan(search, grid, min_conf=MIN_CONF, max_len=MAX_LEN) -> BeamPlan | None:
    """Return the shortest accepted plan to a plausible goal over the best model, or None."""
    model = search.best()
    if model is None or model.transition_accuracy < min_conf:
        return None
    goals = search.goals(grid)
    if not goals:
        return None
    best = None
    for goal in goals:
        # a goal already satisfied is not a target (no-op); skip
        ap = model.agent_pos(grid)
        if goal.satisfied(grid, ap, model.agent_colors):
            continue
        plan = model.plan_to(grid, goal)
        if not plan or len(plan) > max_len:
            continue
        if best is None or len(plan) < len(best.plan):
            best = BeamPlan(goal=goal, plan=plan, model=model, confidence=model.transition_accuracy)
    return best
