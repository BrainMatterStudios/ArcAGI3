"""Tests for patch 7 (run watchdog) against a REAL offline harness loop.

There is no runnable local multi-game LLM rig (the g0/bsm rig dumps came from
Kaggle kernels), so this file is the minimal offline harness driver the watchdog
protocol calls for: a real ``taaf.GameAPI`` over ``environment_files/`` driven by
the real ``_HarnessGameSession.play`` loop, with scripted analyzers that inject
the failure modes the watchdog exists for:

  1. an analyzer that HANGS inside ``analyze`` (polling ``should_stop`` like the
     streaming client does) — the watchdog must detect the stall, attempt one
     recovery RESET, then stop the game and let the run complete;
  2. an analyzer wedged in an infinite retryable-failure loop — same recovery
     chain, exercised from the play loop rather than from inside ``analyze``;
  3. a healthy but endless analyzer — the per-game wall cap must stop it.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_watchdog.py -v
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
ENV_DIR = REPO / "environment_files"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402

GAME = "ft09"


def _result(step_executed: bool, retryable: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        step_executed=step_executed,
        retryable_failure=retryable,
        yielded_control=False,
        reasoning="",
    )


class _HangingAnalyzer:
    """Simulates a wedged in-flight request: blocks until should_stop fires."""

    generated_tokens = 0
    _timeout = None

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                should_stop=None, **kwargs):
        self.calls += 1
        deadline = time.monotonic() + 60.0  # hard test bound, never reached
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                break
            time.sleep(0.05)
        return _result(step_executed=False)


class _RetryLoopAnalyzer:
    """Simulates an endpoint that fails retryably forever."""

    generated_tokens = 0
    _timeout = None

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, *args, **kwargs):
        self.calls += 1
        return _result(step_executed=False, retryable=True)


class _BusyAnalyzer:
    """Healthy progress forever: executes one valid action per turn."""

    generated_tokens = 0
    _timeout = None

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                **kwargs):
        name = (valid_actions or ["ACTION1"])[0]
        args = {"action": name}
        if name == "ACTION6":
            args.update({"row": 0, "col": 0})
        step_env(args)
        time.sleep(0.02)  # keep the loop from spinning thousands of actions
        return _result(step_executed=True)


def _make_session(tmp_path: Path, analyzer):
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from inference.framework import solver as duck_solver

    assert "OK" in duck_patches.patch_watchdog() or "SKIP" in duck_patches.patch_watchdog()

    run_session = RunSession(record_intermediate_states=False)
    game = GameAPI(env_name=GAME, arcade_spec=ArcadeSpec(environments_dir=str(ENV_DIR)))
    game.start_game(run_session)

    solver = duck_solver.HarnessSolver(
        label="watchdog-test",
        model="stub",
        analyzer_timeout=5.0,
        concurrency=1,
    )
    solver.job_dir = tmp_path
    session = duck_solver._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "artifacts" / "runtime_state.json",
        transcript_path=tmp_path / "transcripts" / f"{GAME}.txt",
        analysis_html_relpath=f"solver_analysis/{GAME}.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "artifacts" / "viewer_data.json",
    )
    return game, session


def _reset_count(game) -> int:
    return sum(1 for rec in game.game_run.history if rec.action.id.name == "RESET")


def test_watchdog_detects_in_call_hang_and_completes_run(tmp_path, monkeypatch):
    """Injected hang inside analyze: detect -> recovery RESET -> kill -> run banked."""
    monkeypatch.setenv("TAAF_GRAPH", "0")  # isolate patch 7 from the patch-11 grinder
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "1.5")
    monkeypatch.setenv("TAAF_WATCHDOG_MAX_RESETS", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_WALL_CAP_S", "0")

    analyzer = _HangingAnalyzer()
    game, session = _make_session(tmp_path, analyzer)

    t0 = time.monotonic()
    session.play()
    elapsed = time.monotonic() - t0

    wd = session._watchdog_state
    assert wd["killed"] == "stall", wd
    assert wd["resets_done"] == 1, wd
    assert _reset_count(game) == 1, "exactly one recovery RESET must be in the history"
    assert game.game_run.state == "gave_up", game.game_run.state
    assert game.game_run.final_score is not None, "run must be banked (finish_game ran)"
    assert elapsed < 30.0, f"run did not complete promptly ({elapsed:.1f}s)"


def test_watchdog_detects_retry_loop_stall(tmp_path, monkeypatch):
    """Endless retryable failures: recovery RESET, then a clean stop."""
    monkeypatch.setenv("TAAF_GRAPH", "0")  # isolate patch 7 from the patch-11 grinder
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "1.5")
    monkeypatch.setenv("TAAF_WATCHDOG_MAX_RESETS", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_WALL_CAP_S", "0")

    analyzer = _RetryLoopAnalyzer()
    game, session = _make_session(tmp_path, analyzer)

    t0 = time.monotonic()
    session.play()
    elapsed = time.monotonic() - t0

    wd = session._watchdog_state
    assert wd["killed"] == "stall", wd
    assert wd["resets_done"] == 1
    assert _reset_count(game) == 1
    assert game.game_run.state == "gave_up"
    assert analyzer.calls >= 1
    assert elapsed < 30.0, f"run did not complete promptly ({elapsed:.1f}s)"


def test_watchdog_wall_cap_stops_endless_progress(tmp_path, monkeypatch):
    """A game that always progresses must still hit the per-game wall cap."""
    monkeypatch.setenv("TAAF_GRAPH", "0")  # isolate patch 7 from the patch-11 grinder
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "300")
    monkeypatch.setenv("TAAF_WATCHDOG_WALL_CAP_S", "2")

    analyzer = _BusyAnalyzer()
    game, session = _make_session(tmp_path, analyzer)

    t0 = time.monotonic()
    session.play()
    elapsed = time.monotonic() - t0

    wd = session._watchdog_state
    assert wd["killed"] == "wall_cap", wd
    assert game.game_run.state in ("gave_up", "won")
    assert game.game_run.final_score is not None
    assert elapsed < 30.0, f"wall cap did not bound the run ({elapsed:.1f}s)"


def test_watchdog_disabled_is_inert(tmp_path, monkeypatch):
    """TAAF_WATCHDOG=0: the session runs and stops on its own limits only."""
    monkeypatch.setenv("TAAF_GRAPH", "0")  # isolate patch 7 from the patch-11 grinder
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "1")

    analyzer = _BusyAnalyzer()
    game, session = _make_session(tmp_path, analyzer)
    session.solver.max_actions_per_game = 5

    session.play()

    wd = getattr(session, "_watchdog_state", None)
    assert wd is None or wd["killed"] is None
    assert _reset_count(game) == 0
    assert game.game_run.state in ("gave_up", "won")


def test_watchdog_respects_existing_solver_runtime_cap(tmp_path, monkeypatch):
    """When max_runtime_s_per_game is set, the watchdog wall cap must defer to it."""
    monkeypatch.setenv("TAAF_GRAPH", "0")  # isolate patch 7 from the patch-11 grinder
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "300")
    monkeypatch.setenv("TAAF_WATCHDOG_WALL_CAP_S", "1")

    analyzer = _BusyAnalyzer()
    game, session = _make_session(tmp_path, analyzer)
    session.solver.max_runtime_s_per_game = 2.0

    t0 = time.monotonic()
    session.play()
    elapsed = time.monotonic() - t0

    wd = session._watchdog_state
    assert wd["killed"] is None, "solver's own cap fired; watchdog must not claim the stop"
    assert elapsed >= 1.9, "session must run to the solver's cap, not the watchdog's"
    assert game.game_run.state in ("gave_up", "won")
