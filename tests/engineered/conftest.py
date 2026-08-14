"""Shared fixtures for the engineered-agent perception tests.

The offline engine tests use the 25 public games under environment_files/;
they are sub-second each (Stage-0 measured 25-game load+step at 0.6 s).
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

logging.disable(logging.CRITICAL)


@pytest.fixture(scope="session")
def arcade() -> Any:
    pytest.importorskip("arc_agi")
    from engineered.envs import open_arcade

    return open_arcade()


@pytest.fixture(scope="session")
def gid_of(arcade: Any) -> dict[str, str]:
    from engineered.envs import resolve_game_ids

    return resolve_game_ids(arcade)
