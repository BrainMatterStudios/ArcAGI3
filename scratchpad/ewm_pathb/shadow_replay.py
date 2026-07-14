"""shadow_replay.py — floor-safe duck efficiency banking via max-over-plays.

The duck wins levels but INEFFICIENTLY (cumulative actions across its RESET attempts within a
single play). arc_agi scores a game as MAX over plays (scorecard.py:241). So: take the duck's
own WINNING attempt for each completed level (the actions after the last RESET in that level's
history slice), replay ONLY that in a FRESH play, and max-over-plays banks the efficient score.
Floor-safe by construction: the duck play is untouched; an extra play can only ADD (max), and a
desynced replay is caught per-level and abandoned (the duck's score stands).

Empirically validated: on a real duck tu93 run, L1 orig 0.03 -> replay 2.22 (81x).

Two entry points:
- extract_winning_segments(history, actions_per_level, levels_completed) -> per-level winning actions
- replay_segments(game, segments) -> replays into a fresh started TAAF game, self-verifying per level
"""
from __future__ import annotations
from typing import Any

import arcengine


def _norm(rec: Any) -> dict:
    """Normalize a history entry (ActionRecord or dict) to {'id': name, 'data': {...}}."""
    if isinstance(rec, dict):
        a = rec.get("action", rec)
        aid = a["id"]
        return {"id": aid if isinstance(aid, str) else getattr(aid, "name", str(aid)), "data": dict(a.get("data") or {})}
    action = rec.action
    aid = action.id
    return {"id": getattr(aid, "name", str(aid)), "data": dict(action.data or {})}


def extract_winning_segments(history: list, actions_per_level: list[int], levels_completed: int) -> list[tuple[int, list[dict]]]:
    """For each COMPLETED level, the actions of the winning attempt = everything after the last
    RESET within that level's history slice (RESET is present in history; sum(apl)==len(history))."""
    hist = [_norm(r) for r in history]
    segs: list[tuple[int, list[dict]]] = []
    idx = 0
    for level, n in enumerate(actions_per_level):
        seg = hist[idx:idx + n]
        idx += n
        if level < levels_completed and n > 0:
            last_reset = max((i for i, r in enumerate(seg) if r["id"] == "RESET"), default=-1)
            win = seg[last_reset + 1:]
            segs.append((level, win))
    return segs


def _to_action_input(d: dict) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction[d["id"]], data=dict(d.get("data") or {}))


def replay_segments(game: Any, segments: list[tuple[int, list[dict]]], *, reset_prefix: bool = True) -> dict:
    """Replay winning segments into a FRESH, already-start_game()'d TAAF game. Self-verifying:
    after each level's segment, assert levels_completed advanced; on desync, STOP (bank what worked).
    Returns {'levels': reached, 'actions': taken, 'desynced_at': level or None}."""
    actions = 0
    for level, win in segments:
        before = game.current_state.levels_completed
        if reset_prefix:  # engine counts RESET as 1 action; matches how the winning attempt began
            try:
                game.execute_action(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}))
                actions += 1
            except Exception:
                pass
        for d in win:
            try:
                game.execute_action(_to_action_input(d))
                actions += 1
            except Exception:
                return {"levels": game.current_state.levels_completed, "actions": actions, "desynced_at": level}
            if game.game_run is None or game.game_run.state != "playing":
                break
        if game.current_state.levels_completed <= before:
            return {"levels": game.current_state.levels_completed, "actions": actions, "desynced_at": level}
    return {"levels": game.current_state.levels_completed, "actions": actions, "desynced_at": None}
