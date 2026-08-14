"""Offline engine access for the engineered agent.

Everything is anchored to the repo root (two levels above this file), so the
modules work regardless of caller cwd — wrong-cwd is a known recurring failure
mode in this repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """/…/ArcAGI3 — the directory containing environment_files/."""
    return Path(__file__).resolve().parents[2]


def game_stem(game_id: str) -> str:
    """'tu93-0768757b' -> 'tu93'. A bare stem passes through unchanged."""
    return game_id.split("-")[0][:4]


def open_arcade() -> Any:
    """Offline Arcade over the 25 public games. Import deferred so the pure
    perception/battery code stays importable without the engine installed."""
    from arc_agi import Arcade, OperationMode

    return Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root() / "environment_files"),
    )


def resolve_game_ids(arcade: Any) -> dict[str, str]:
    """stem -> full game_id for every offline environment."""
    return {game_stem(e.game_id): e.game_id for e in arcade.get_environments()}


if __name__ == "__main__":  # standalone smoke: list the offline games
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    arcade = open_arcade()
    ids = resolve_game_ids(arcade)
    print(f"{len(ids)} offline games: {' '.join(sorted(ids))}")
