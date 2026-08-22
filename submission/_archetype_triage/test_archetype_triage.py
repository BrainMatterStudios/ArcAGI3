"""Offline validation for the archetype_triage graft (no GPU/network/Kaggle).

Covers: install/idempotency; archetype dispatch on the full edge-menu table
(RESET stripping, ACTION7/{5}-style unknowns => CLICK never-kill, empty menu
defers); kill fires only past the archetype threshold AND at zero levels;
CLICK never killed; completed-level sessions never killed; archetype frozen
at frame-0; stock should_stop verdicts untouched (including through the
REAL play loop's finally banking a "gave_up" run); threshold env overrides;
flags-off pass-through; fail-open.

Sessions are the REAL ``_HarnessGameSession`` over a duck-typed engine game
(same fixture family as submission/_truthful_telemetry test 08).

Run:  .venv/bin/python submission/_archetype_triage/test_archetype_triage.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(
    os.environ.get(
        "GRAFT_TEST_BUNDLE",
        str(_HERE.parents[1] / "scratchpad/bundles/june_stock/src/ARC3-Inference"),
    )
)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_triage
from graft_triage import (
    ARCHETYPE_AVATAR,
    ARCHETYPE_CLICK,
    ARCHETYPE_MIXED,
    classify_menu,
    install,
    kill_minutes,
)

_FLAGS = ("TRIAGE_KILL", "TRIAGE_AVATAR_MIN", "TRIAGE_MIXED_MIN", "TRIAGE_CLICK_MIN")


def _clear_flags() -> None:
    for name in _FLAGS:
        os.environ.pop(name, None)


class _FakeEngineState:
    def __init__(self, arcengine, available_actions, levels_completed: int = 0) -> None:
        self.raw = SimpleNamespace(state=arcengine.GameState.NOT_FINISHED)
        self.available_actions = list(available_actions)
        self.levels_completed = levels_completed
        self.won = False
        self.just_won_level = False
        self.frame = SimpleNamespace(data=[[0, 0], [0, 0]])


class _FakeEngineGame:
    def __init__(self, arcengine, available_actions, levels_completed: int = 0) -> None:
        self._state = _FakeEngineState(arcengine, available_actions, levels_completed)
        self.game_run = SimpleNamespace(
            state="playing",
            history=[],
            solver_analysis_html=None,
            game_id="fake",
            final_score=None,
            solver_note=None,
        )
        self.number_of_levels = 5
        self.finished = False

    @property
    def current_state(self):
        return self._state

    def execute_action(self, action, *, generated_tokens=0, uncached_input_tokens=0):
        self.game_run.history.append(action)
        return self._state

    def finish_game(self) -> None:
        self.finished = True
        if self.game_run.state == "playing":
            self.game_run.state = "gave_up"
        self.game_run.final_score = float(self._state.levels_completed)


class ArchetypeTriageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        import arcengine  # noqa: PLC0415

        from inference.framework import solver as solver_mod  # noqa: PLC0415

        cls.arcengine = arcengine
        cls.solver_mod = solver_mod
        cls.install_status = install()

    def setUp(self) -> None:
        _clear_flags()

    # --- fixtures -----------------------------------------------------------

    def _session(self, game, *, elapsed_minutes: float = 0.0):
        tmp = Path(tempfile.mkdtemp(prefix="tri_test_"))
        session = self.solver_mod._HarnessGameSession(
            solver=SimpleNamespace(
                max_runtime_s_per_game=None,
                max_actions_per_game=None,
                soft_time_remaining_seconds=lambda: None,
                job_dir=None,
                label="test",
            ),
            game=game,
            analyzer=SimpleNamespace(generated_tokens=0, total_tokens=0),
            game_index=0,
            pass_index=0,
            state_path=tmp / "tool_runtime_state.json",
            transcript_path=tmp / "transcript.txt",
            analysis_html_relpath="x.html",
            stop_event=threading.Event(),
            viewer_data_path=tmp / "viewer.json",
        )
        session.started_at = time.monotonic() - elapsed_minutes * 60.0
        return session

    def _game(self, menu, levels_completed: int = 0):
        return _FakeEngineGame(self.arcengine, menu, levels_completed)

    # --- installation -------------------------------------------------------

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "archetype_triage: OK")
        self.assertEqual(install(), "archetype_triage: SKIP (already applied)")

    # --- dispatch table -----------------------------------------------------

    def test_02_dispatch_table_including_edge_menus(self) -> None:
        cases = [
            ([6], ARCHETYPE_CLICK),  # pure click hammer
            ([0, 6], ARCHETYPE_CLICK),  # RESET stripped first
            ([1, 2, 3, 4], ARCHETYPE_AVATAR),
            ([0, 1, 2, 3, 4], ARCHETYPE_AVATAR),
            ([1, 2, 3, 4, 5], ARCHETYPE_AVATAR),  # +5 still avatar
            ([1, 2], ARCHETYPE_AVATAR),  # partial movement menu
            ([0, 4, 5], ARCHETYPE_AVATAR),
            ([1, 2, 3, 4, 6], ARCHETYPE_MIXED),  # both families
            ([0, 1, 6], ARCHETYPE_MIXED),
            ([2, 5, 6], ARCHETYPE_MIXED),
            ([5, 6], ARCHETYPE_CLICK),  # click-family without movement
            ([6, 7], ARCHETYPE_CLICK),
            ([5], ARCHETYPE_CLICK),  # no movement, no click => safe default
            ([7], ARCHETYPE_CLICK),
            ([1, 2, 7], ARCHETYPE_CLICK),  # ACTION7 breaks avatar confidence
            ("garbage-not-a-menu", ARCHETYPE_CLICK),  # unreadable => safe default
        ]
        for menu, want in cases:
            self.assertEqual(classify_menu(menu), want, msg=f"menu={menu!r}")
        # empty / RESET-only menus defer (retry next call), not classify
        self.assertIsNone(classify_menu([]))
        self.assertIsNone(classify_menu([0]))

    def test_03_deferred_menu_classifies_on_next_call(self) -> None:
        game = self._game([0])  # frame-0 menu not readable yet
        session = self._session(game, elapsed_minutes=61)
        self.assertFalse(session.should_stop())  # defers, no kill, no freeze
        self.assertIsNone(getattr(session, "_triage_archetype", None))
        game._state.available_actions = [0, 1, 2, 3, 4]
        self.assertTrue(session.should_stop())  # now AVATAR, 61m, zero levels
        self.assertEqual(session._triage_archetype, ARCHETYPE_AVATAR)

    # --- kill thresholds ----------------------------------------------------

    def test_04_avatar_kills_at_60_not_59(self) -> None:
        game = self._game([0, 1, 2, 3, 4])
        self.assertFalse(self._session(game, elapsed_minutes=59).should_stop())
        killed = self._session(game, elapsed_minutes=61)
        self.assertTrue(killed.should_stop())
        self.assertIn("triage_kill: AVATAR", game.game_run.solver_note)

    def test_05_mixed_kills_at_70_not_69(self) -> None:
        game = self._game([0, 1, 2, 3, 4, 6])
        self.assertFalse(self._session(game, elapsed_minutes=69).should_stop())
        self.assertTrue(self._session(game, elapsed_minutes=71).should_stop())

    def test_06_click_never_killed(self) -> None:
        for menu in ([0, 6], [0, 5], [0, 6, 7]):
            game = self._game(menu)
            session = self._session(game, elapsed_minutes=500)
            self.assertFalse(session.should_stop(), msg=f"menu={menu}")

    def test_07_completed_level_session_never_killed(self) -> None:
        game = self._game([0, 1, 2, 3, 4], levels_completed=1)
        session = self._session(game, elapsed_minutes=131)
        self.assertFalse(session.should_stop())
        # and a MIXED session that banked late is also safe
        game2 = self._game([0, 1, 6], levels_completed=3)
        self.assertFalse(self._session(game2, elapsed_minutes=131).should_stop())

    def test_08_archetype_frozen_at_frame0(self) -> None:
        game = self._game([0, 1, 2, 3, 4])  # frame-0: AVATAR
        session = self._session(game, elapsed_minutes=10)
        self.assertFalse(session.should_stop())  # classifies + freezes
        self.assertEqual(session._triage_archetype, ARCHETYPE_AVATAR)
        game._state.available_actions = [0, 6]  # menu later turns click-only
        session.started_at = time.monotonic() - 61 * 60.0
        self.assertTrue(session.should_stop())  # still AVATAR: 60m rule holds

    # --- stock verdicts untouched, banking verified through real play() ----

    def test_09_stock_should_stop_verdicts_pass_through(self) -> None:
        game = self._game([0, 6])  # CLICK, would never be triage-killed
        session = self._session(game, elapsed_minutes=1)
        self.assertFalse(session.should_stop())
        session.stop_event.set()
        self.assertTrue(session.should_stop())  # stock stop_event path
        session.stop_event.clear()
        game.game_run.state = "gave_up"
        self.assertTrue(session.should_stop())  # stock run-state path

    def test_10_kill_exits_real_play_loop_and_banks_gave_up(self) -> None:
        game = self._game([0, 1, 2, 3, 4])  # AVATAR
        session = self._session(game, elapsed_minutes=61)

        class _NeverCalledAnalyzer:
            generated_tokens = 0
            total_tokens = 0
            _timeout = None

            def analyze(self, *args, **kwargs):
                raise AssertionError("kill must fire before any analyzer turn")

        session.analyzer = _NeverCalledAnalyzer()
        session.play()  # returns instead of looping => worker slot released
        self.assertTrue(game.finished)
        self.assertEqual(game.game_run.state, "gave_up")
        self.assertEqual(game.game_run.final_score, 0.0)
        self.assertIn("triage_kill: AVATAR", game.game_run.solver_note)

    # --- overrides, flags, fail-open ----------------------------------------

    def test_11_threshold_env_overrides(self) -> None:
        os.environ["TRIAGE_AVATAR_MIN"] = "30"
        game = self._game([0, 1, 2, 3, 4])
        self.assertTrue(self._session(game, elapsed_minutes=31).should_stop())
        self.assertFalse(self._session(game, elapsed_minutes=29).should_stop())
        os.environ["TRIAGE_AVATAR_MIN"] = "never"
        self.assertFalse(self._session(game, elapsed_minutes=131).should_stop())
        os.environ["TRIAGE_CLICK_MIN"] = "50"
        game2 = self._game([0, 6])
        self.assertTrue(self._session(game2, elapsed_minutes=51).should_stop())
        os.environ["TRIAGE_MIXED_MIN"] = "not-a-number"  # falls back to default 70
        self.assertEqual(kill_minutes(ARCHETYPE_MIXED), 70.0)

    def test_12_flag_off_pure_passthrough(self) -> None:
        os.environ["TRIAGE_KILL"] = "0"
        game = self._game([0, 1, 2, 3, 4])
        session = self._session(game, elapsed_minutes=131)
        self.assertFalse(session.should_stop())
        self.assertIsNone(getattr(session, "_triage_archetype", None))
        self.assertIsNone(game.game_run.solver_note)

    def test_13_fail_open_broken_menu_returns_stock_verdict(self) -> None:
        # Break ONLY the graft-read surface (available_actions); the stock
        # reads (raw.state, game_run, history) stay healthy, so the stock
        # verdict (False) must come back instead of a crash or a kill.
        game = self._game([0, 1, 2, 3, 4])
        session = self._session(game, elapsed_minutes=131)

        class _BoomState:
            raw = game._state.raw
            levels_completed = 0

            @property
            def available_actions(self):
                raise RuntimeError("boom")

        game._state = _BoomState()
        self.assertFalse(session.should_stop())

    def test_14_kill_condition_is_reevaluated_not_latched_wrongly(self) -> None:
        # A session that completes a level AFTER crossing the threshold but
        # BEFORE any kill-check ran must not be killed.
        game = self._game([0, 1, 2, 3, 4])
        session = self._session(game, elapsed_minutes=10)
        self.assertFalse(session.should_stop())  # freeze AVATAR early
        game._state.levels_completed = 1  # banks a level at minute ~50
        session.started_at = time.monotonic() - 90 * 60.0
        self.assertFalse(session.should_stop())


if __name__ == "__main__":
    unittest.main(verbosity=2)
