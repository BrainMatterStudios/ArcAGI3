"""Acceptance tests for generated games.

A generated game is VALID iff:
  1. it loads and instantiates through the exact same arc_agi local_wrapper
     exec path the real environment_files games use (Arcade OFFLINE scan ->
     make() -> LocalEnvironmentWrapper),
  2. replaying the reference trace completes every level (final state WIN,
     levels_completed == win_levels), with per-level completion happening at
     the step the solver predicted,
  3. the trace replays identically twice (fresh wrapper instances): every
     rendered frame, state, and score matches step for step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

logging.getLogger("arc_agi").setLevel(logging.ERROR)


def _step_env(env: Any, action: dict[str, Any]) -> Any:
    game_action = GameAction.from_name(action["id"])
    data = {"x": action["x"], "y": action["y"]} if action["id"] == "ACTION6" else None
    return env.step(game_action, data=data)


def replay(
    env_root: Path, game_id: str, trace: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fresh env; RESET then the trace. Returns one record per action with the
    last rendered frame, state name, and score."""
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_root))
    env = arcade.make(game_id, save_recording=False)
    if env is None:
        raise RuntimeError(f"local_wrapper could not load {game_id} from {env_root}")
    records = []
    frame = env.reset()
    if frame is None:
        raise RuntimeError(f"reset() returned None for {game_id}")
    records.append(
        {
            "action": {"id": "RESET"},
            "frame": np.asarray(frame.frame[-1]).tolist(),
            "state": frame.state.name,
            "score": frame.levels_completed,
            "win_levels": frame.win_levels,
        }
    )
    for action in trace:
        frame = _step_env(env, action)
        if frame is None:
            raise RuntimeError(f"step() returned None for {game_id} on {action}")
        records.append(
            {
                "action": action,
                "frame": np.asarray(frame.frame[-1]).tolist(),
                "state": frame.state.name,
                "score": frame.levels_completed,
                "win_levels": frame.win_levels,
            }
        )
    return records


def validate_game(env_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Run the full acceptance test for one generated game."""
    game_id = spec["game_id"]
    per_level = spec["solution_levels"]
    trace = [a for lvl in per_level for a in lvl]
    report: dict[str, Any] = {"game_id": game_id, "family": spec["family"], "ok": False}

    run1 = replay(env_root, game_id, trace)
    run2 = replay(env_root, game_id, trace)

    # 1. loadability is implied by replay() not raising
    report["loaded"] = True

    # 2. solvability: final WIN with all levels completed
    final = run1[-1]
    report["final_state"] = final["state"]
    report["levels_completed"] = final["score"]
    report["win_levels"] = final["win_levels"]
    solved = final["state"] == "WIN" and final["score"] == final["win_levels"] == len(per_level)
    report["solved"] = solved

    # per-level completion at the predicted step
    boundaries = []
    idx = 0
    for lvl in per_level:
        idx += len(lvl)
        boundaries.append(idx)  # trace index whose action completes the level
    level_ok = all(
        run1[b]["score"] == want + 1 for want, b in enumerate(boundaries)
    )
    report["level_boundaries_ok"] = level_ok

    # no accidental deaths mid-trace
    report["no_game_over"] = all(r["state"] != "GAME_OVER" for r in run1)

    # 3. determinism
    deterministic = len(run1) == len(run2) and all(
        r1["state"] == r2["state"]
        and r1["score"] == r2["score"]
        and r1["frame"] == r2["frame"]
        for r1, r2 in zip(run1, run2)
    )
    report["deterministic"] = deterministic

    report["trace_len"] = len(trace)
    report["ok"] = bool(solved and level_ok and deterministic and report["no_game_over"])
    return report


def validate_from_dir(game_dir: Path) -> dict[str, Any]:
    """Validate a previously written game dir (uses its spec.json+solution.json)."""
    spec = json.loads((game_dir / "spec.json").read_text())
    solution = json.loads((game_dir / "solution.json").read_text())
    spec["solution_levels"] = solution["per_level"]
    env_root = game_dir.parent.parent
    return validate_game(env_root, spec)
