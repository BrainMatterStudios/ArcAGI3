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


def test_watchdog_default_is_heartbeat_under_scorecard_stale_close():
    """The default stall MUST undercut the gateway's 15-min stale-close.

    arc_agi auto-closes a scorecard idle >= DEFAULT_STALE_MINUTES (cleanup
    thread polls every 60s, arc_agi/api.py scorecard_cleanup_loop). The old
    900s default EQUALED that threshold — zero margin. The heartbeat default
    must leave a margin larger than the 60s close granularity plus generous
    should_stop polling slack. LIVE gateway threshold UNVERIFIED (the
    scorecard_timeout param path is unclamped); this pins the documented
    default relationship only.
    """
    from arc_agi.scorecard import DEFAULT_STALE_MINUTES

    wd = duck_patches._watchdog_state(SimpleNamespace())
    assert wd["stall_s"] == 600.0, wd["stall_s"]
    stale_s = DEFAULT_STALE_MINUTES * 60
    assert wd["stall_s"] + 60 + 120 <= stale_s, (
        f"default stall {wd['stall_s']}s must beat the {stale_s}s stale-close "
        "with >60s close-granularity + polling margin"
    )


class _WedgedSlowPollAnalyzer:
    """A wedged in-flight request that surfaces ``should_stop`` only rarely.

    This is the realistic losing shape of the stale-close race: the streaming
    client is blocked and only checks ``should_stop`` sporadically (here every
    ``poll_s``), so the watchdog cannot act at exactly its threshold — it acts
    at the first poll AFTER it. With zero margin (old 900s == 15-min close)
    the server close wins; with the 600s heartbeat the RESET lands well before
    the close threshold.
    """

    generated_tokens = 0
    _timeout = None

    def __init__(self, poll_s: float) -> None:
        self.poll_s = poll_s

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                should_stop=None, **kwargs):
        deadline = time.monotonic() + 60.0  # hard test bound
        while time.monotonic() < deadline:
            time.sleep(self.poll_s)  # wedged: no fine-grained polling
            if should_stop is not None and should_stop():
                break
        return _result(step_executed=False)


class _CleanupLoop:
    """Faithful port of arc_agi/api.py ``scorecard_cleanup_loop`` (the local
    rig never starts that thread — this test does), with the 60s wake-up
    scaled down alongside every other duration. Uses the REAL
    ``get_stale_cards`` / ``should_auto_close_scorecard`` /
    ``close_scorecard`` sequence on the game's own offline Arcade."""

    def __init__(self, arcade, poll_s: float, on_close) -> None:
        self._arcade = arcade
        self._poll_s = poll_s
        self._on_close = on_close
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10.0)

    def _loop(self):
        mgr = self._arcade.scorecard_manager
        while not self._stop.wait(self._poll_s):
            for cid in mgr.get_stale_cards():
                if not mgr.should_auto_close_scorecard(cid):
                    continue
                scorecard, guids, _ = mgr.close_scorecard(cid, None)
                if scorecard is not None:
                    self._on_close(cid, scorecard)


# Scaled stale-close race geometry (documented real-world equivalents).
# Real: stale close 900s @ 60s poll granularity; old watchdog 900s; new 600s;
# a wedged client surfacing should_stop every ~few minutes; "16-minute stall".
_SC = {
    "stale_s": 6.0,        # = 900s  (DEFAULT_STALE_MINUTES)
    "cleanup_poll_s": 0.4,  # = 60s  (cleanup thread wake-up)
    "old_stall_s": 6.0,     # = 900s (old default == stale threshold: no margin)
    "new_stall_s": 4.0,     # = 600s (heartbeat default: 300s margin)
    "wedge_poll_s": 2.5,    # wedged client's sporadic should_stop surfacing
    "sixteen_min_s": 6.4,   # = 960s (the simulated 16-minute stall mark)
}


def _run_stale_close_arm(tmp_path, monkeypatch, stall_s: float):
    """Drive a real session into a wedge with the REAL stale-close machinery
    running against its own offline Arcade. Returns race evidence."""
    from datetime import timedelta

    monkeypatch.setenv("TAAF_GRAPH", "0")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", str(stall_s))
    monkeypatch.setenv("TAAF_WATCHDOG_MAX_RESETS", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_WALL_CAP_S", "0")

    analyzer = _WedgedSlowPollAnalyzer(poll_s=_SC["wedge_poll_s"])
    game, session = _make_session(tmp_path, analyzer)

    arcade = game._arcade
    assert arcade is not None, "offline GameAPI must own an Arcade"
    mgr = arcade.scorecard_manager
    assert mgr.scorecards, "start_game must have opened a local scorecard"
    card_id = next(iter(mgr.scorecards))
    # Scale the REAL manager's threshold (set_idle_for only takes minutes).
    mgr.idle_for = timedelta(seconds=_SC["stale_s"])

    closed = {}

    def on_close(cid, scorecard):
        wd = getattr(session, "_watchdog_state", None) or {}
        closed.update(
            cid=cid,
            t=time.monotonic(),
            watchdog_resets_at_close=wd.get("resets_done"),
            watchdog_killed_at_close=wd.get("killed"),
        )
        session.stop_event.set()  # arm A: lost — end the run promptly

    cleanup = _CleanupLoop(arcade, _SC["cleanup_poll_s"], on_close).start()
    t0 = time.monotonic()
    try:
        worker = threading.Thread(target=session.play, daemon=True)
        worker.start()
        # Hold the wedge past the simulated 16-minute mark, then release.
        while time.monotonic() - t0 < _SC["sixteen_min_s"]:
            if closed:
                break
            time.sleep(0.05)
        survived_16min = card_id in mgr.scorecards
        resets_at_16min = getattr(session, "_watchdog_state", {}).get("resets_done", 0)
        session.stop_event.set()
        worker.join(timeout=30.0)
        assert not worker.is_alive(), "session did not stop"
    finally:
        cleanup.stop()
    return {
        "card_id": card_id,
        "mgr": mgr,
        "closed": closed,
        "survived_16min": survived_16min,
        "resets_at_16min": resets_at_16min,
        "wd": session._watchdog_state,
        "game": game,
    }


def test_stale_close_race_lost_without_heartbeat(tmp_path, monkeypatch):
    """OLD default geometry (stall == stale threshold): a 16-minute wedge
    loses the scorecard to the server's stale-close BEFORE the watchdog acts.
    The card is closed at partial score; the RESET can no longer revive it."""
    r = _run_stale_close_arm(tmp_path, monkeypatch, stall_s=_SC["old_stall_s"])
    assert r["closed"], "stale-close never fired — cleanup machinery inactive?"
    assert r["closed"]["cid"] == r["card_id"]
    assert not r["survived_16min"], "card must be gone by the 16-minute mark"
    assert r["card_id"] not in r["mgr"].scorecards
    # The race was lost: at close time the watchdog had done NOTHING yet.
    assert r["closed"]["watchdog_resets_at_close"] in (0, None)
    assert r["closed"]["watchdog_killed_at_close"] is None


def test_stale_close_survived_with_600s_heartbeat(tmp_path, monkeypatch):
    """NEW default geometry (600 vs 900: margin > close granularity + polling
    slack): the watchdog's recovery RESET lands before the stale threshold,
    bumps the card's last_update (real Scorecard.update_scorecard path), and
    the game is still alive at the 16-minute mark."""
    r = _run_stale_close_arm(tmp_path, monkeypatch, stall_s=_SC["new_stall_s"])
    assert not r["closed"], f"stale-close fired despite heartbeat: {r['closed']}"
    assert r["survived_16min"], "card must still be open at the 16-minute mark"
    # (After the test releases the run, the session's own finish path closes
    # the card cleanly — only the STALE close callback above counts as a loss.)
    assert r["resets_at_16min"] == 1, "exactly one heartbeat RESET expected"
    assert _reset_count(r["game"]) >= 1


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
