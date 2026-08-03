"""Tests for patch 10 (replay-at-WIN harvest).

The end-to-end test reproduces docs/test-artifacts-2026-08-02/test1_local_rest.py
through the PATCHED AGENT PATH: the shipped arc_agi Flask app in competition mode
(ONLY_RESET_LEVELS=true, the eval setting), a real taaf GameAPI in COMPETITION
mode over HTTP, the real _HarnessGameSession.play loop with a scripted analyzer
that wins sb26 sloppily (partial level 8 + waste + RESET + clean finish), and
asserts the patch then: sends one RESET at WIN, mechanically replays the clean
trace with zero LLM calls, producing plays [sloppy, clean] with closed score ==
max over plays.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_win_replay.py -v
"""
from __future__ import annotations

import json
import logging
import os
import socketserver
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import pytest

# Must be set before the server app is built; arcengine reads it at RESET time.
os.environ["ONLY_RESET_LEVELS"] = "true"
os.environ.setdefault("ARC_API_KEY", "win-replay-test-key")

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
TRACE_PATH = REPO / "docs/test-artifacts-2026-08-02/sb26_win_trace.json"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import build_win_replay_plan  # noqa: E402

GAME = "sb26-7fbdac44"
L8_START = 125  # per-level actions [9,33,15,15,17,19,17,17] -> level 8 at index 125


# --- plan compression (pure unit tests) -----------------------------------------


def _rec(name: str, before: int, after: int, state: str = "NOT_FINISHED",
         changed: bool = True, executed: bool = True, **data) -> dict:
    return {
        "name": name,
        "data": data,
        "levels_before": before,
        "levels_after": after,
        "state": state,
        "changed": changed,
        "executed": executed,
    }


def test_plan_keeps_straight_win():
    trace = [
        _rec("ACTION1", 0, 0),
        _rec("ACTION2", 0, 1),
        _rec("ACTION3", 1, 1),
        _rec("ACTION4", 1, 2, state="WIN"),
    ]
    clean, boundaries = build_win_replay_plan(trace)
    assert [r["name"] for r in clean] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    assert boundaries == [(2, 1), (4, 2)]


def test_plan_segment_drop_removes_pre_reset_attempts():
    """Everything buffered in the current level before a RESET was undone."""
    trace = [
        _rec("ACTION1", 0, 1),
        # failed attempt at level 2, then death, then auto-RESET
        _rec("ACTION2", 1, 1),
        _rec("ACTION3", 1, 1, state="GAME_OVER"),
        _rec("RESET", 1, 1),
        # second failed attempt (gave up), manual RESET
        _rec("ACTION2", 1, 1),
        _rec("RESET", 1, 1),
        # the successful attempt
        _rec("ACTION4", 1, 2, state="WIN"),
    ]
    clean, boundaries = build_win_replay_plan(trace)
    assert [r["name"] for r in clean] == ["ACTION1", "ACTION4"]
    assert boundaries == [(1, 1), (2, 2)]


def test_plan_drops_provable_noops_but_keeps_unclear_actions():
    trace = [
        _rec("ACTION1", 0, 0, changed=False),          # provable no-op: dropped
        _rec("ACTION2", 0, 0, changed=True),           # changed board: kept
        _rec("ACTION6", 0, 1, changed=False, x=3, y=4),  # no pixel change but level up: kept
    ]
    clean, boundaries = build_win_replay_plan(trace)
    assert [r["name"] for r in clean] == ["ACTION2", "ACTION6"]
    assert boundaries == [(2, 1)]

    raw, _ = build_win_replay_plan(trace, drop_noops=False)
    assert [r["name"] for r in raw] == ["ACTION1", "ACTION2", "ACTION6"]


def test_plan_ignores_unexecuted_and_post_win_noise():
    trace = [
        _rec("ACTION1", 0, 1),
        _rec("ACTION9", 1, 1, executed=False),
        _rec("ACTION2", 1, 2, state="WIN"),
        _rec("ACTION3", 2, 2, state="WIN"),  # trailing noise after the win
    ]
    clean, boundaries = build_win_replay_plan(trace)
    assert [r["name"] for r in clean] == ["ACTION1", "ACTION2"]
    assert boundaries == [(1, 1), (2, 2)]


# --- end-to-end against the local competition-mode REST server ------------------


class _ThreadingWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):  # noqa: D102
        return


class _ScriptedAnalyzer:
    """Replays a fixed action script through step_env — zero LLM involvement."""

    generated_tokens = 0
    _timeout = None

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self._index = 0

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                **kwargs):
        if self._index >= len(self._script):
            return SimpleNamespace(step_executed=False, retryable_failure=False,
                                   yielded_control=False, reasoning="script done")
        args = self._script[self._index]
        self._index += 1
        step_env(args)
        return SimpleNamespace(step_executed=True, retryable_failure=False,
                               yielded_control=False, reasoning="")


def _to_step_args(entry: dict) -> dict:
    if entry["name"] == "ACTION6":
        return {"action": "ACTION6", "row": entry["y"], "col": entry["x"]}
    return {"action": entry["name"]}


@pytest.fixture(scope="module")
def rest_server():
    logging.disable(logging.CRITICAL)
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    arcade.available_environments = [
        e for e in arcade.available_environments if e.game_id == GAME
    ]
    assert len(arcade.available_environments) == 1
    app, api = create_app(arcade, competition_mode=True)
    server = make_server(
        "127.0.0.1", 0, app,
        server_class=_ThreadingWSGIServer, handler_class=_QuietHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", api
    server.shutdown()
    logging.disable(logging.NOTSET)


def test_win_replay_end_to_end_sloppy_then_clean(rest_server, tmp_path, monkeypatch):
    base_url, api = rest_server
    monkeypatch.setenv("TAAF_WIN_REPLAY", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "900")  # keep the watchdog quiet
    monkeypatch.setenv("TAAF_GRAPH", "0")  # isolate patch 10: the wasted-click
    # script deliberately repeats a no-op, which patch 11 would veto

    import arc_agi
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from inference.framework import solver as duck_solver

    # The full patched agent path: HUD mask (for no-op classification) + replay.
    for status in (
        duck_patches.patch_hud_board_identity(),
        duck_patches.patch_win_replay(),
        duck_patches.patch_watchdog(),
    ):
        assert "OK" in status or "SKIP" in status, status

    trace = json.load(TRACE_PATH.open())
    assert len(trace) == 142

    # Sloppy script: levels 1-7 + 5 actions into level 8, 30 wasted corner
    # clicks, a RESET (restarts level 8), then the full clean level 8.
    script = (
        [_to_step_args(a) for a in trace[:L8_START + 5]]
        + [{"action": "ACTION6", "row": 63, "col": 63}] * 30
        + [{"action": "RESET"}]
        + [_to_step_args(a) for a in trace[L8_START:]]
    )

    run_session = RunSession(record_intermediate_states=False)
    game = GameAPI(
        env_name=GAME,
        arcade_spec=ArcadeSpec(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=base_url,
            environments_dir="",
        ),
    )
    game.start_game(run_session)
    comp = game._competition_scorecard
    assert comp is not None
    comp.open_run()  # test's own hold so the card survives session.play()

    solver = duck_solver.HarnessSolver(
        label="win-replay-test", model="scripted", analyzer_timeout=30.0, concurrency=1,
    )
    solver.job_dir = tmp_path
    session = duck_solver._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=_ScriptedAnalyzer(script),
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "artifacts" / "runtime_state.json",
        transcript_path=tmp_path / "transcripts" / f"{GAME}.txt",
        analysis_html_relpath=f"solver_analysis/{GAME}.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "artifacts" / "viewer_data.json",
    )

    session.play()

    # 1. The duck's own play is banked as a full win.
    assert game.game_run.state == "won"
    assert game.game_run.levels_completed == 8

    # 2. The patch replayed once, cleanly, with zero LLM calls.
    result = getattr(session, "_win_replay_result", None)
    assert result is not None, "patched play() must record a replay result"
    assert result["status"] == "replayed", result
    assert result["replay_won"] is True and result["desynced"] is False, result
    assert result["replay_levels"] == 8, result
    assert result["replay_actions"] <= 142, result
    assert result["replay_actions"] < result["original_actions"], result
    assert getattr(game, "_win_replay_done", False) is True

    # 3. Server-side scorecard: two plays [sloppy, clean-replay], both WIN,
    #    and the replay play is strictly shorter.
    card = api.arcade.scorecard_manager.get_scorecard(
        game._scorecard_id, os.environ["ARC_API_KEY"]
    ).cards.get(GAME)
    assert card is not None
    assert card.total_plays == 2, vars(card)
    assert [s.name for s in card.states] == ["WIN", "WIN"], card.states
    assert card.actions[1] == result["replay_actions"], card.actions
    assert card.actions[0] > card.actions[1], card.actions

    # 4. Close the card: game score == max over plays, clean beats sloppy.
    closed = comp.finish_run()
    assert closed is not None, "test hold should be the last one; close must return"
    entry = closed.find_environment(GAME)
    run_scores = [r.score for r in entry.runs]
    assert len(run_scores) == 2, run_scores
    assert run_scores[1] > run_scores[0], (
        f"clean replay must strictly beat the sloppy play: {run_scores}"
    )
    assert abs(entry.score - max(run_scores)) < 1e-9, (entry.score, run_scores)


def test_win_replay_never_fires_twice(rest_server, tmp_path):
    """The once-per-game guard: a second call must refuse."""
    game = SimpleNamespace(
        env=object(), game_run=SimpleNamespace(levels_completed=8, game_id=GAME),
        number_of_levels=8, _win_replay_done=True,
    )
    session = SimpleNamespace(
        game=game, stop_event=threading.Event(),
        solver=SimpleNamespace(soft_time_remaining_seconds=lambda: None),
        _replay_trace=[_rec("ACTION1", 0, 8, state="WIN")],
    )
    result = duck_patches._maybe_replay_at_win(session)
    assert result == {"status": "skipped", "reason": "already replayed"}


def test_win_replay_refuses_partial_wins():
    game = SimpleNamespace(
        env=object(), game_run=SimpleNamespace(levels_completed=5, game_id=GAME),
        number_of_levels=8,
    )
    # _is_run_complete reads current_state; give it a non-WIN state.
    game.current_state = SimpleNamespace(
        raw=SimpleNamespace(state=SimpleNamespace(name="NOT_FINISHED"))
    )
    session = SimpleNamespace(
        game=game, stop_event=threading.Event(),
        solver=SimpleNamespace(soft_time_remaining_seconds=lambda: None),
        _replay_trace=[_rec("ACTION1", 0, 5)],
    )
    result = duck_patches._maybe_replay_at_win(session)
    assert result["status"] == "skipped"
    assert result["reason"] in ("not WON", "not fully won")


def test_win_replay_disabled_env():
    session = SimpleNamespace(game=None)
    os.environ["TAAF_WIN_REPLAY"] = "0"
    try:
        assert duck_patches._maybe_replay_at_win(session) == {"status": "disabled"}
    finally:
        os.environ["TAAF_WIN_REPLAY"] = "1"
