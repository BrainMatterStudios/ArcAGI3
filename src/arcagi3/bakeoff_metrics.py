"""Bake-off scoring: the competition's per-level efficiency + a transfer-slope summary.

efficiency mirrors the verified scoring formula: per level min(1.15, (A_h / A_m)^2),
uncompleted level = 0. A_h is the human-baseline proxy (Exp-42 BFS-optimal length); A_m is
the agent's environment-altering action count for that level.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_TRUEMODEL = Path(__file__).resolve().parents[2] / "scripts" / "truemodel_planner.py"


def efficiency(a_h: int, a_m: int | None) -> float:
    if a_m is None or a_m <= 0:
        return 0.0
    return min(1.15, (a_h / a_m) ** 2)


def transfer_slope(per_level_actions: list[int]) -> float:
    """Least-squares slope of action-cost vs level index. Negative = costs drop with depth."""
    n = len(per_level_actions)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(per_level_actions) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, per_level_actions))
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else num / den


def human_baseline_actions(level: int = 0, max_nodes: int = 500_000) -> int | None:
    """A_h proxy = BFS-optimal action count over ls20's TRUE model (grader-only source access).

    # NOTE: level>0 requires driving the env to that level first; not yet supported.
    """
    spec = importlib.util.spec_from_file_location("truemodel_planner", _TRUEMODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sol, *_ = mod.bfs_solve(mod.load_ls20_class(), start_level=level, max_nodes=max_nodes)
    return None if sol is None else len(sol)
