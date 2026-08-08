"""Tests for patch 17 (LLM-free opening wiggle battery -> controllability masks).

Rank 2 of the 2026-08-07 human-play idea sweep: before the model's first
deliberation the session presses the available arrows (interleaved 2x per
direction, <= 12 scored actions, abort after 4 straight masked no-ops), ports
the validated contingency classifier (IoU >= 0.70 shape-verified translation,
>= 2 direction-matching presses, consistency >= 0.5, <= 600 cells) and keeps
SELF / REACTIVE / DEAD masks + a GAME MODE verdict fresh via a standing
observer and a 2-press mini-probe after every level transition.

Layers: pure classifier units (translation lock, cd82 rotation trap, HUD-tick
tolerance, click reactivity incl. remote effects), battery control flow on
scripted synthetic sessions (lock-in-4, click costs 0, early abort, cap,
confirm phase, RESET never emitted), per-level re-probe, the level-completed
trigger through the REAL patched session stack, legend injection position and
coexistence with patch 16, sandbox globals in a REAL sandbox subprocess,
OFF-by-default no-op, and the scored-bundle staleness gate.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_wiggle.py -v
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
ENV_DIR = REPO / "environment_files"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import (  # noqa: E402
    _WIGGLE_MAX_PRESSES,
    _WIGGLE_TLS,
    WiggleState,
    _wiggle_lock_evidence,
    _wiggle_prompt_block,
    _wiggle_reprobe,
    _wiggle_run_battery,
    wiggle_observe_transition,
    wiggle_verdict,
)

LEGEND_MARKER = "CONTROLLABILITY LEGEND"
DIFF_MARKER = "STRUCTURED CHANGE REPORT"


def _apply() -> str:
    result = duck_patches.patch_wiggle()
    assert "OK" in result or "SKIP" in result, result
    return result


# ---------------------------------------------------------------------------------
# board helpers
# ---------------------------------------------------------------------------------


def _board(fill: int = 0, size: int = 16) -> list[list[int]]:
    return [[fill] * size for _ in range(size)]


def _paint(board, cells, color):
    out = [list(row) for row in board]
    for y, x in cells:
        out[y][x] = color
    return out


def _block(y0, x0, h, w):
    return [(y, x) for y in range(y0, y0 + h) for x in range(x0, x0 + w)]


def _shift_rule(color: int, dy: int, dx: int):
    """Board transform: translate every `color` cell by (dy, dx)."""

    def rule(board):
        h, w = len(board), len(board[0])
        cells = [(y, x) for y in range(h) for x in range(w) if board[y][x] == color]
        new = [row[:] for row in board]
        for y, x in cells:
            new[y][x] = 0
        for y, x in cells:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                new[ny][nx] = color
        return new

    return rule


def _toggle_rule(cells, color_a: int, color_b: int):
    """Board transform: recolor a region back and forth (morph, not motion)."""

    def rule(board):
        new = [row[:] for row in board]
        for y, x in cells:
            new[y][x] = color_b if new[y][x] == color_a else color_a
        return new

    return rule


# ---------------------------------------------------------------------------------
# pure classifier units
# ---------------------------------------------------------------------------------


def test_translation_press_records_move_evidence():
    state = WiggleState()
    pre = _paint(_board(), _block(5, 3, 2, 2), 9)
    post = _paint(_board(), _block(6, 3, 2, 2), 9)
    out = wiggle_observe_transition(state, "ACTION2", {}, pre, post)
    assert out["kind"] == "move"
    assert out["sign"] == (1, 0)
    moves = state.press_stats["ACTION2"]["moves"]
    assert len(moves) == 1 and len(moves[0]["cells"]) == 4
    assert moves[0]["colors"] == [9]


def test_avatar_lock_two_presses_same_action():
    state = WiggleState()
    state.directionals_at_start = ["ACTION2"]
    board = _paint(_board(), _block(5, 3, 2, 2), 9)
    for _ in range(2):
        post = _shift_rule(9, 1, 0)(board)
        wiggle_observe_transition(state, "ACTION2", {}, board, post)
        board = post
    lock = _wiggle_lock_evidence(state)
    assert lock is not None and lock["matches"] == 2
    assert wiggle_verdict(state)[0] == "AVATAR"


def test_avatar_lock_two_actions_different_directions():
    state = WiggleState()
    state.directionals_at_start = ["ACTION2", "ACTION4"]
    board = _paint(_board(), _block(5, 3, 2, 2), 9)
    post = _shift_rule(9, 1, 0)(board)
    wiggle_observe_transition(state, "ACTION2", {}, board, post)
    post2 = _shift_rule(9, 0, 1)(post)
    wiggle_observe_transition(state, "ACTION4", {}, post, post2)
    assert _wiggle_lock_evidence(state) is not None
    assert wiggle_verdict(state)[0] == "AVATAR"


def test_no_lock_on_ambient_drift_same_direction():
    """Two different keys 'moving' the object the SAME way is drift, not control."""
    state = WiggleState()
    state.directionals_at_start = ["ACTION1", "ACTION2"]
    board = _paint(_board(), _block(5, 3, 2, 2), 9)
    post = _shift_rule(9, 1, 0)(board)
    wiggle_observe_transition(state, "ACTION1", {}, board, post)
    post2 = _shift_rule(9, 1, 0)(post)
    wiggle_observe_transition(state, "ACTION2", {}, post, post2)
    assert _wiggle_lock_evidence(state) is None
    assert wiggle_verdict(state)[0] != "AVATAR"


def test_oversize_translation_is_not_an_avatar():
    state = WiggleState()
    board = _paint(_board(0, 40), _block(0, 0, 32, 32), 9)  # 1024 cells > 600
    post = _shift_rule(9, 1, 0)(board)
    out = wiggle_observe_transition(state, "ACTION2", {}, board, post)
    assert out["kind"] == "morph"
    assert not state.press_stats["ACTION2"]["moves"]


def test_rotation_reads_as_morph_not_avatar():
    """cd82 trap: arrows transform an object in place — never a translation."""
    state = WiggleState()
    state.directionals_at_start = ["ACTION1"]
    base = _board()
    pre = _paint(base, [(5, 5), (6, 5), (7, 5), (7, 6)], 9)  # L
    post = _paint(base, [(5, 5), (5, 6), (5, 7), (6, 5)], 9)  # rotated L
    for _ in range(2):
        out = wiggle_observe_transition(state, "ACTION1", {}, pre, post)
        pre, post = post, pre
    assert out["kind"] == "morph"
    assert wiggle_verdict(state)[0] == "ARROW-MORPH"


def test_slot_cursor_jump_beyond_search_radius_is_a_move():
    """tr87 class: a conserved marker jumping ~28 cols must still read as motion."""
    state = WiggleState()
    state.directionals_at_start = ["ACTION4"]
    marker = [(48, x) for x in range(15, 20)] + [(49, x) for x in range(15, 20)]
    pre = _paint(_board(0, 64), marker, 3)
    post = _paint(_board(0, 64), [(y, x + 28) for y, x in marker], 3)
    out = wiggle_observe_transition(state, "ACTION4", {}, pre, post)
    assert out["kind"] == "move" and out["sign"] == (0, 1)
    # a second identical press locks the translation — and the 2026-08-09
    # autopsy repair demotes it to CURSOR (displacement 28 >> body extent 5;
    # tr87 WAS this misfire: a selector locked as AVATAR, avatar playbook
    # misled the model). The motion itself must still be read as a move.
    pre2 = post
    post2 = _paint(_board(0, 64), [(y, x + 42) for y, x in marker], 3)  # jump again
    out2 = wiggle_observe_transition(state, "ACTION4", {}, pre2, post2)
    assert out2["kind"] == "move" and out2["sign"] == (0, 1)
    mode, reason = wiggle_verdict(state)
    assert mode == "CURSOR" and "displacement" in reason


def test_jump_rejects_shape_change_and_unconserved_colors():
    state = WiggleState()
    pre = _paint(_board(0, 64), _block(10, 10, 2, 3), 3)
    post = _paint(_board(0, 64), _block(40, 40, 3, 2), 3)  # transposed shape
    out = wiggle_observe_transition(state, "ACTION1", {}, pre, post)
    assert out["kind"] == "morph"


def test_hud_tick_tolerance_two_cells_counts_as_noop():
    state = WiggleState()
    pre = _board()
    post = _paint(pre, [(0, 3), (0, 4)], 5)  # unconfirmed HUD bar ticking
    out = wiggle_observe_transition(state, "ACTION1", {}, pre, post)
    assert out == {"kind": "noop", "action": "ACTION1", "hud_suspect": True}
    assert state.consecutive_noops == 1
    assert state.press_stats["ACTION1"]["tick_noops"] == 1


def test_hud_masked_tick_is_a_plain_noop():
    state = WiggleState()
    pre = _board()
    post = _paint(pre, [(0, 3)], 5)
    out = wiggle_observe_transition(state, "ACTION1", {}, pre, post, [[0, 3]])
    assert out == {"kind": "noop", "action": "ACTION1"}
    assert not state.changed_ever  # masked cells never enter the changed set


def test_click_masks_near_and_remote():
    state = WiggleState()
    pre = _board(0, 32)
    near_cells = [(10, 10), (10, 11)]
    far_cells = [(25, 25), (25, 26), (26, 25)]  # 3 cells: a real remote effect
    post = _paint(_paint(pre, near_cells, 7), far_cells, 8)
    out = wiggle_observe_transition(
        state, "ACTION6", {"y": 10, "x": 10}, pre, post
    )
    assert out == {"kind": "click", "changed": True}
    assert state.reactive == set(near_cells)
    assert state.reactive_remote == set(far_cells)
    assert state.remote_click_events == 1


def test_click_far_tick_is_not_a_remote_effect():
    state = WiggleState()
    pre = _board(0, 32)
    post = _paint(_paint(pre, [(10, 10)], 7), [(31, 5)], 8)  # 1 far cell = tick
    wiggle_observe_transition(state, "ACTION6", {"y": 10, "x": 10}, pre, post)
    assert state.reactive == {(10, 10)}
    assert not state.reactive_remote and state.remote_click_events == 0


def test_verdicts_click_paths():
    state = WiggleState()
    state.directionals_at_start = []
    assert wiggle_verdict(state) == (
        "CLICK",
        "no directional actions offered (0 presses spent)",
    )
    state2 = WiggleState()
    state2.directionals_at_start = ["ACTION1"]
    pre = _board()
    for _ in range(4):
        wiggle_observe_transition(state2, "ACTION1", {}, pre, pre)
    mode, reason = wiggle_verdict(state2)
    assert mode == "CLICK" and "no-ops" in reason


# ---------------------------------------------------------------------------------
# battery control flow on scripted synthetic sessions
# ---------------------------------------------------------------------------------


def _fake_state(board, avail, levels=0, won=False, name="NOT_FINISHED"):
    return SimpleNamespace(
        frame=SimpleNamespace(data=[list(r) for r in board]),
        available_actions=list(avail),
        levels_completed=levels,
        won=won,
        raw=SimpleNamespace(state=SimpleNamespace(name=name)),
    )


class ScriptedSession:
    """Synthetic battery target: `rules` maps an action name to a board rule."""

    def __init__(self, board, avail, rules, levels_completed=0):
        self.board = [list(r) for r in board]
        self.avail = list(avail)
        self.rules = rules
        self.pressed: list[str] = []
        self.game = SimpleNamespace(
            number_of_levels=3,
            game_run=SimpleNamespace(history=[]),
            current_state=_fake_state(self.board, self.avail, levels_completed),
        )
        self._levels = levels_completed

    def should_stop(self):
        return False

    def _execute_action(self, action, *, batch_index=1, batch_size=1,
                        generated_tokens=None, **kwargs):
        import arcengine

        name = action.id.name
        assert name != "RESET", "the battery must never emit RESET"
        assert name in duck_patches._WIGGLE_DIRECTIONALS
        assert int(arcengine.GameAction.from_name(name).value) in self.avail
        self.pressed.append(name)
        rule = self.rules.get(name)
        if rule:
            self.board = rule(self.board)
        self.game.current_state = _fake_state(self.board, self.avail, self._levels)
        return {
            "executed": True,
            "action_num": len(self.pressed),
            "level": 1,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
        }


def test_battery_avatar_locks_in_four_presses():
    board = _paint(_board(), _block(5, 3, 2, 2), 9)
    session = ScriptedSession(
        board,
        avail=[1, 2, 3, 4],
        rules={"ACTION2": _shift_rule(9, 1, 0), "ACTION4": _shift_rule(9, 0, 1)},
    )
    state = WiggleState()
    _wiggle_run_battery(session, state)
    assert session.pressed == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    assert state.locked_in == 4 and state.battery_presses == 4
    assert state.mode == "AVATAR"
    assert state.avatar_color == "b"  # ARC_COLOR_CHARS[9]
    assert len(state.avatar_cells) == 4
    assert state.battery_done and state.mode_history == [(1, "AVATAR")]


def test_battery_click_game_costs_zero_directional_actions():
    session = ScriptedSession(_board(), avail=[6], rules={})
    state = WiggleState()
    _wiggle_run_battery(session, state)
    assert session.pressed == []
    assert state.battery_presses == 0
    assert state.mode == "CLICK"
    assert "no directional actions offered" in state.mode_reason


def test_battery_early_abort_after_four_noop_presses():
    session = ScriptedSession(_board(), avail=[1, 2, 3, 4], rules={})
    state = WiggleState()
    _wiggle_run_battery(session, state)
    assert state.battery_presses == 4  # one interleaved round, then the fast negative
    assert state.mode == "CLICK"
    assert "no-ops" in state.mode_reason


def test_battery_cap_and_arrow_morph_verdict():
    toggle = _toggle_rule(_block(8, 8, 4, 4), 3, 4)
    session = ScriptedSession(
        _board(), avail=[1, 2, 3, 4],
        rules={name: toggle for name in duck_patches._WIGGLE_DIRECTIONALS},
    )
    state = WiggleState()
    _wiggle_run_battery(session, state)
    # 2 interleaved rounds, no translations -> no confirm phase, no lock
    assert state.battery_presses == 8
    assert state.mode == "ARROW-MORPH"
    assert state.battery_presses <= _WIGGLE_MAX_PRESSES <= 16


def test_battery_confirm_phase_locks_single_direction():
    presses = {"n": 0}

    def sometimes_shift(board):
        presses["n"] += 1
        return board if presses["n"] == 1 else _shift_rule(9, 0, 1)(board)

    board = _paint(_board(), _block(5, 3, 2, 2), 9)
    session = ScriptedSession(board, avail=[4], rules={"ACTION4": sometimes_shift})
    state = WiggleState()
    _wiggle_run_battery(session, state)
    assert session.pressed == ["ACTION4"] * 3  # 2 base + 1 confirm press
    assert state.locked_in == 3
    assert state.mode == "AVATAR"


def test_battery_skips_unavailable_directions():
    board = _paint(_board(), _block(5, 3, 2, 2), 9)
    session = ScriptedSession(
        board, avail=[2, 4],
        rules={"ACTION2": _shift_rule(9, 1, 0), "ACTION4": _shift_rule(9, 0, 1)},
    )
    state = WiggleState()
    _wiggle_run_battery(session, state)
    assert set(session.pressed) <= {"ACTION2", "ACTION4"}
    assert state.directionals_at_start == ["ACTION2", "ACTION4"]
    assert state.mode == "AVATAR"


# ---------------------------------------------------------------------------------
# per-level mini-probe
# ---------------------------------------------------------------------------------


def _locked_state() -> WiggleState:
    state = WiggleState()
    state.battery_done = True
    state.mode = "AVATAR"
    state.avatar_action = "ACTION4"
    state.level = 1
    state.changed_ever = {(1, 1)}  # stale level-1 evidence
    state.mode_history = [(1, "AVATAR")]
    return state


def test_reprobe_reconfirms_avatar_and_resets_masks():
    board = _paint(_board(), _block(9, 9, 2, 2), 9)
    session = ScriptedSession(
        board, avail=[1, 2, 3, 4], rules={"ACTION4": _shift_rule(9, 0, 1)},
        levels_completed=1,
    )
    state = _locked_state()
    _wiggle_reprobe(session, state)
    assert session.pressed == ["ACTION4", "ACTION4"]  # repeat the locked direction
    assert state.reprobe_presses == 2
    assert state.mode == "AVATAR"
    assert state.mode_reason.startswith("L2 reprobe:")
    assert (1, 1) not in state.changed_ever  # per-level masks were reset
    assert state.mode_history[-1] == (2, "AVATAR")


def test_reprobe_detects_controllability_loss():
    session = ScriptedSession(
        _board(), avail=[1, 2, 3, 4], rules={}, levels_completed=1
    )
    state = _locked_state()
    _wiggle_reprobe(session, state)
    assert state.mode != "AVATAR"  # lf52 class: the body did not come along
    assert state.mode == "CLICK" and "no-ops" in state.mode_reason


def test_reprobe_without_directionals_costs_zero():
    session = ScriptedSession(_board(), avail=[6], rules={}, levels_completed=1)
    state = _locked_state()
    _wiggle_reprobe(session, state)
    assert session.pressed == []
    assert state.reprobe_presses == 0
    assert state.mode == "CLICK"


# ---------------------------------------------------------------------------------
# real patched stack: e2e battery + level-completed trigger + OFF default
# ---------------------------------------------------------------------------------


def _result(step_executed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        step_executed=step_executed,
        retryable_failure=False,
        yielded_control=False,
        reasoning="",
    )


class _StopAnalyzer:
    """Records the scored action count at its first turn, then stops the game."""

    generated_tokens = 0
    _timeout = None

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.first_action_count: int | None = None
        self.turns = 0

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                **kwargs):
        self.turns += 1
        if self.first_action_count is None:
            self.first_action_count = action_num
        self.stop_event.set()
        return _result(step_executed=False)


def _real_session(tmp_path, game_name: str):
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from inference.framework import solver as duck_solver

    run_session = RunSession(record_intermediate_states=False)
    game = GameAPI(env_name=game_name,
                   arcade_spec=ArcadeSpec(environments_dir=str(ENV_DIR)))
    game.start_game(run_session)
    solver = duck_solver.HarnessSolver(
        label="wiggle-test", model="stub", analyzer_timeout=5.0, concurrency=1
    )
    solver.job_dir = tmp_path
    stop_event = threading.Event()
    analyzer = _StopAnalyzer(stop_event)
    session = duck_solver._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "artifacts" / "runtime_state.json",
        transcript_path=tmp_path / "transcripts" / f"{game_name}.txt",
        analysis_html_relpath=f"solver_analysis/{game_name}.html",
        stop_event=stop_event,
        viewer_data_path=tmp_path / "artifacts" / "viewer_data.json",
    )
    return game, session, analyzer


@pytest.mark.skipif(not ENV_DIR.is_dir(), reason="environment files absent")
def test_e2e_ls20_battery_runs_before_first_deliberation(tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    _apply()
    game, session, analyzer = _real_session(tmp_path, "ls20")  # arrows-only game
    session.play()
    state = session._wiggle_state
    assert state.battery_done
    assert 1 <= state.battery_presses <= _WIGGLE_MAX_PRESSES
    # the battery's scored actions all landed BEFORE the first deliberation
    assert analyzer.first_action_count == state.battery_presses
    assert state.mode in ("AVATAR", "CLICK", "ARROW-MORPH", "UNCLEAR")
    history = [str(a) for a in (game.game_run.history or [])]
    assert not any("RESET" in h.upper() for h in history)


@pytest.mark.skipif(not ENV_DIR.is_dir(), reason="environment files absent")
def test_e2e_ft09_click_game_spends_zero_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    _apply()
    game, session, analyzer = _real_session(tmp_path, "ft09")  # MOUSE-only game
    session.play()
    state = session._wiggle_state
    assert state.battery_done and state.battery_presses == 0
    assert analyzer.first_action_count == 0
    assert state.mode == "CLICK"


@pytest.mark.skipif(not ENV_DIR.is_dir(), reason="environment files absent")
def test_e2e_off_by_default_spends_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TAAF_WIGGLE", raising=False)
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    _apply()
    game, session, analyzer = _real_session(tmp_path, "ls20")
    session.play()
    assert analyzer.first_action_count == 0  # zero probe actions when dormant
    assert getattr(session, "_wiggle_state", None) is None
    assert getattr(_WIGGLE_TLS, "state", None) is None


class _LevelUpGame:
    """Fake engine game whose next action completes level 1."""

    def __init__(self, board):
        import arcengine

        self._arcengine = arcengine
        self.number_of_levels = 3
        self.game_run = SimpleNamespace(history=[], state="playing")
        self._board = board
        self.current_state = self._state(levels=0, just_won=False)

    def _state(self, levels: int, just_won: bool):
        return SimpleNamespace(
            frame=SimpleNamespace(data=[list(r) for r in self._board]),
            available_actions=[1, 2, 3, 4, 6],
            levels_completed=levels,
            won=False,
            just_won_level=just_won,
            raw=SimpleNamespace(state=self._arcengine.GameState.NOT_FINISHED),
            animation_frames=[],
        )

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        self.game_run.history.append(action.id.name)
        self.current_state = self._state(levels=1, just_won=True)
        return self.current_state


def test_level_completed_triggers_reprobe_through_patched_stack(tmp_path, monkeypatch):
    import arcengine
    from inference.framework import solver as duck_solver

    monkeypatch.setenv("TAAF_WIGGLE", "1")
    _apply()
    calls = []
    monkeypatch.setattr(duck_patches, "_wiggle_reprobe", lambda s, st: calls.append(st))

    game = _LevelUpGame(_board())
    solver = duck_solver.HarnessSolver(label="t", model="stub", concurrency=1)
    solver.job_dir = None  # viewer writes no-op
    session = duck_solver._HarnessGameSession(
        solver=solver, game=game, analyzer=SimpleNamespace(),
        game_index=0, pass_index=0,
        state_path=tmp_path / "runtime_state.json",
        transcript_path=tmp_path / "t.txt",
        analysis_html_relpath="t.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "viewer_data.json",
    )
    state = WiggleState()
    state.battery_done = True
    session._wiggle_state = state
    action = arcengine.ActionInput(id=arcengine.GameAction.ACTION6,
                                   data={"x": 3, "y": 3})
    payload = session._execute_action(action, batch_index=1, batch_size=1,
                                      generated_tokens=0)
    assert payload["level_completed"]
    assert len(calls) == 1 and calls[0] is state


# ---------------------------------------------------------------------------------
# legend injection + coexistence with patch 16
# ---------------------------------------------------------------------------------


def _agent():
    from inference.agent.tool_agent import ToolAgent

    _apply()
    duck_patches.patch_diff_lines()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/wiggle-test")
    return agent


def _build(agent):
    from inference.agent.runtime_state import Frame

    return agent._build_user_prompt(
        1, valid_actions=["ACTION1"],
        current_frame=Frame(grid=((1, 2), (3, 4)), step=1, level=1),
    )


def _legend_state(mode="AVATAR") -> WiggleState:
    state = WiggleState()
    state.battery_done = True
    state.battery_presses = 4
    state.mode = mode
    state.mode_reason = "shape-verified translation lock (4 presses)"
    state.avatar_cells = [(5, 3), (5, 4), (6, 3), (6, 4)]
    state.avatar_color = "b"
    state.avatar_action = "ACTION2"
    state.avatar_shift = (1, 0)
    state.shape = (16, 16)
    state.changed_ever = {(5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (7, 4)}
    return state


def test_legend_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("TAAF_WIGGLE", raising=False)
    agent = _agent()
    _WIGGLE_TLS.state = _legend_state()
    try:
        assert LEGEND_MARKER not in _build(agent)
    finally:
        _WIGGLE_TLS.state = None


def test_legend_absent_before_battery(monkeypatch):
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    agent = _agent()
    _WIGGLE_TLS.state = WiggleState()  # battery not yet run
    try:
        assert LEGEND_MARKER not in _build(agent)
    finally:
        _WIGGLE_TLS.state = None


def test_legend_below_diff_report_above_grids(monkeypatch):
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    monkeypatch.setenv("TAAF_DIFF_LINES", "1")
    agent = _agent()
    _WIGGLE_TLS.state = _legend_state()
    duck_patches._DIFF_TLS.entries = [
        {"action": "ACTION2", "action_num": 5, "lines": ["CHANGE #1: test-line"]}
    ]
    try:
        prompt = _build(agent)
        assert prompt.startswith(DIFF_MARKER)
        assert LEGEND_MARKER in prompt
        assert prompt.index(DIFF_MARKER) < prompt.index(LEGEND_MARKER)
        assert prompt.index(LEGEND_MARKER) < prompt.index("Current state:")
        assert "GAME MODE: AVATAR" in prompt
        assert "4 'b' cell(s)" in prompt
        # persistent: unlike the diff report, the legend repeats next turn
        assert LEGEND_MARKER in _build(agent)
    finally:
        _WIGGLE_TLS.state = None
        duck_patches._DIFF_TLS.entries = []


def test_legend_wording_click_and_morph_and_dead(monkeypatch):
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    state = _legend_state(mode="CLICK")
    state.mode_reason = "all 4 directional presses were masked no-ops"
    state.clicks = 3
    state.reactive = {(1, 1), (1, 2)}
    state.reactive_remote = {(9, 9), (9, 10), (10, 9)}
    state.remote_click_events = 1
    _WIGGLE_TLS.state = state
    try:
        block = _wiggle_prompt_block()
        assert block.startswith(LEGEND_MARKER)
        assert "GAME MODE: CLICK" in block
        assert "Do not spend more actions" in block
        assert "REACTIVE: 2 cell(s)" in block and "remote-effect click(s)" in block
        assert f"DEAD: {16 * 16 - 6}/{16 * 16}" in block
        state.mode = "ARROW-MORPH"
        assert "ARROW-MORPH" in _wiggle_prompt_block()
    finally:
        _WIGGLE_TLS.state = None


# ---------------------------------------------------------------------------------
# sandbox globals
# ---------------------------------------------------------------------------------


def _minimal_state(extra=None):
    state = {
        "current_frame": {"ascii": "", "step": 1, "level": 1,
                          "shape": [2, 2], "grid": [[0, 0], [0, 0]]},
        "history": [],
        "valid_actions": ["ACTION1"],
        "last_action_result": {},
    }
    if extra:
        state.update(extra)
    return state


def test_sandbox_globals_present_in_real_sandbox(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.setenv("TAAF_WIGGLE", "1")
    _apply()
    assert "WIGGLE_MASKS" in sandbox_mod._SANDBOX_BOOTSTRAP
    _WIGGLE_TLS.state = _legend_state()
    try:
        out = sandbox_mod.run_sandboxed_python(
            code=(
                "result = {'mode': GAME_MODE, 'self': WIGGLE_MASKS['self'],\n"
                "          'dead': WIGGLE_MASKS['dead_count'],\n"
                "          'dead_row5': WIGGLE_MASKS['dead_rows'][5][:6]}\n"
            ),
            timeout_seconds=30,
            initial_state=_minimal_state(),
            action_handler=lambda actions: {},
        )
        assert not out.get("error"), out
        assert out["result"] == {
            "mode": "AVATAR",
            "self": [[5, 3], [5, 4], [6, 3], [6, 4]],
            "dead": 16 * 16 - 6,
            "dead_row5": "111001",  # cols 3-4 changed on row 5
        }, out["result"]
    finally:
        _WIGGLE_TLS.state = None


def test_sandbox_payload_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("TAAF_WIGGLE", raising=False)
    _WIGGLE_TLS.state = _legend_state()
    try:
        assert duck_patches._wiggle_current_payload() == {}
    finally:
        _WIGGLE_TLS.state = None


def test_bootstrap_injection_is_idempotent():
    from inference.agent import python_tool_sandbox as sandbox_mod

    _apply()
    _apply()
    assert sandbox_mod._SANDBOX_BOOTSTRAP.count('runtime_globals["WIGGLE_MASKS"]') == 1


def test_patch_applies_and_is_idempotent():
    first = _apply()
    assert first.startswith("patch17 wiggle: OK") or "SKIP" in first
    assert duck_patches.patch_wiggle() == "patch17 wiggle: SKIP (already applied)"


def test_patch_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.framework import solver

    monkeypatch.setattr(solver, "_grid_from_state", None)
    msg = duck_patches.patch_wiggle()
    assert msg.startswith("patch17 wiggle: FAIL"), msg


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE staleness gate (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------

SCORED_INFERENCE = SCORED_REF / "src/ARC3-Inference"


def test_scored_bundle_has_every_symbol_patch17_touches():
    solver_src = (SCORED_INFERENCE / "inference/framework/solver.py").read_text()
    for symbol in (
        "import arcengine",
        "def _grid_from_state",
        "class _HarnessGameSession",
        "def _execute_action",
        "def play",
        "def seed_initial_history",
    ):
        assert symbol in solver_src, f"scored bundle lost {symbol!r} — re-validate patch17"
    agent_src = (SCORED_INFERENCE / "inference/agent/tool_agent.py").read_text()
    assert "def _build_user_prompt" in agent_src
    sandbox_src = (SCORED_INFERENCE / "inference/agent/python_tool_sandbox.py").read_text()
    assert duck_patches._HUD_REFRESH_ANCHOR in sandbox_src, (
        "scored bundle lost the refresh anchor — sandbox globals would not ship"
    )


@pytest.mark.skipif(not SCORED_INFERENCE.is_dir(), reason="scored bundle bytes absent")
def test_patch17_applies_on_scored_bundle_bytes():
    """Apply patch 17 on the scored bytes in a clean subprocess: enabled, the
    legend must land above the grids; disabled, the prompt must be untouched."""
    script = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(SCORED_INFERENCE)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

r = duck_patches.patch_wiggle()
assert r.startswith("patch17 wiggle: OK"), r
from inference.agent import python_tool_sandbox as sandbox_mod
assert "WIGGLE_MASKS" in sandbox_mod._SANDBOX_BOOTSTRAP, "sandbox globals missing"

from inference.agent.tool_agent import ToolAgent
from inference.agent.runtime_state import Frame
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/wiggle-scored")
frame = Frame(grid=((1, 2), (3, 4)), step=1, level=1)

def build():
    return agent._build_user_prompt(1, valid_actions=["ACTION1"], current_frame=frame)

state = duck_patches.WiggleState()
state.battery_done = True
state.battery_presses = 4
state.mode = "CLICK"
state.mode_reason = "all 4 directional presses were masked no-ops"
state.shape = (8, 8)
duck_patches._WIGGLE_TLS.state = state

# disabled (default): the legend must NOT leak into the scored-bundle prompt
assert "CONTROLLABILITY LEGEND" not in build(), "dormant patch leaked into prompt"

os.environ["TAAF_WIGGLE"] = "1"
prompt = build()
assert "CONTROLLABILITY LEGEND" in prompt, prompt[:200]
assert "GAME MODE: CLICK" in prompt
assert prompt.index("CONTROLLABILITY LEGEND") < prompt.index("Current state:")
print("SCORED-WIGGLE-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-WIGGLE-OK" in proc.stdout
