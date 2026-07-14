from __future__ import annotations

# Working planner template. You usually only need to edit `candidate_actions` to add the
# meaningful ACTION6 click targets for this game; the search itself comes from search_lib.
from search_lib import bfs_plan, greedy_plan, MOVES  # noqa: F401
from world_model_engine import world_model_engine


def candidate_actions(state: dict) -> list[dict]:
    """Return the SENSIBLE actions to try from `state`.

    Default = the simple moves (ACTION1-5,7). If this game uses ACTION6 clicks, ADD only the
    handful of meaningful click targets (e.g. object centroids, palette swatches, buttons) as
    {"name": "ACTION6", "x": X, "y": Y} — never all 64x64 cells.
    """
    return list(MOVES)


def planner(state: dict) -> list[dict] | None:
    """Plan to LEVEL_COMPLETED through the world model. Returns action dicts or None."""
    return bfs_plan(state, world_model_engine, candidate_actions, max_nodes=200_000)
