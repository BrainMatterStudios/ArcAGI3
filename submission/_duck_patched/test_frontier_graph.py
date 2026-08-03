"""Tests for patch 11 (frontier-graph substrate + no-op edge veto + stall grinder).

Three layers, none of which needs an LLM:

  1. pure unit tests for the FrontierGraph data structure and the poby-mined
     salience click ordering;
  2. scripted-engine tests that drive the REAL patched ``_HarnessGameSession``
     class over a deterministic in-memory mini-game, covering the veto guards
     (threshold, first-N-of-level, per-level cap, RESET exemption, kill switch)
     and every grinder stop condition (unlock, GAME_OVER, budget, frontier
     exhaustion, disengage-on-completed-level);
  3. end-to-end tests over the real engine (``taaf.GameAPI`` on
     environment_files/, the stub-brain harness driver pattern from
     test_watchdog.py): a scripted stall must trigger the grinder through the
     watchdog's own stall machinery, and a stub brain repeating a known no-op
     click must get zero-cost synthetic vetoes;
  plus scored-bundle validation: the patch must decline cleanly when the bundle
  lacks a wrapped symbol, and a subprocess proves it applies against the actual
  scored bundle bytes in scratchpad/taaf_scored_ref.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_frontier_graph.py -v
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref"
ENV_DIR = REPO / "environment_files"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import FrontierGraph, graph_click_candidates  # noqa: E402


# --- unit: salience click candidates --------------------------------------------


def test_click_candidates_salience_order_and_sweep():
    # 10x10 zeros background; a 1-cell blob (very button-like) and a sprawling
    # 1x6 bar (larger, same fill) — the small compact blob must rank first.
    rows = [[0] * 10 for _ in range(10)]
    rows[7][7] = 5          # small blob, likeness 1/(1+1) = 0.5
    for c in range(1, 7):
        rows[2][c] = 3      # 1x6 bar, likeness 1/(1+6) ~= 0.143
    out = graph_click_candidates(rows, limit=64, step=8)
    assert out[0] == (7, 7), out[:3]
    assert out[1] == (3, 2), out[:3]  # bar centroid (x=3..4, y=2)
    # coarse sweep fills coverage without duplicating blob centroids
    assert (4, 4) in out
    assert len(out) == len(set(out))


def test_click_candidates_respect_limit_and_empty():
    assert graph_click_candidates([], limit=8, step=4) == []
    rows = [[0] * 20 for _ in range(20)]
    out = graph_click_candidates(rows, limit=5, step=4)
    assert len(out) == 5  # capped


def test_click_candidates_exclude_background():
    rows = [[7] * 6 for _ in range(6)]  # uniform: everything is background
    out = graph_click_candidates(rows, limit=64, step=4)
    assert all(isinstance(p, tuple) for p in out)
    # only the sweep points remain, no "component" of the background color first
    assert out[0] == (2, 2)


# --- unit: graph bookkeeping ------------------------------------------------------


def _mk_graph() -> FrontierGraph:
    return FrontierGraph(click_limit=8, click_step=8, max_nodes=100)


A = (1, 8, 8, 111)
B = (1, 8, 8, 222)
C = (1, 8, 8, 333)
OTHER_LEVEL = (2, 8, 8, 444)


def test_noop_edge_needs_two_observations_and_purity():
    g = _mk_graph()
    plan = ("ACTION1",)
    g.record(A, plan, A, changed=False, level_up=False, game_over=False)
    assert not g.is_known_noop(A, plan), "one observation must not be enough"
    g.record(A, plan, A, changed=False, level_up=False, game_over=False)
    assert g.is_known_noop(A, plan)

    # An edge that ever changed the board is never a known no-op.
    g2 = _mk_graph()
    g2.record(A, plan, A, changed=False, level_up=False, game_over=False)
    g2.record(A, plan, B, changed=True, level_up=False, game_over=False)
    g2.record(A, plan, A, changed=False, level_up=False, game_over=False)
    assert not g2.is_known_noop(A, plan)

    # Level-up and game-over edges are never no-ops, whatever the pixels said.
    g3 = _mk_graph()
    g3.record(A, plan, A, changed=False, level_up=True, game_over=False)
    g3.record(A, plan, A, changed=False, level_up=False, game_over=False)
    assert not g3.is_known_noop(A, plan)
    g4 = _mk_graph()
    g4.record(A, plan, A, changed=False, level_up=False, game_over=True)
    g4.record(A, plan, A, changed=False, level_up=False, game_over=False)
    assert not g4.is_known_noop(A, plan)


def test_untested_plans_and_mark_dead():
    g = _mk_graph()
    g.ensure_node(A, ["ACTION1", "ACTION2", "RESET"], [[0] * 8 for _ in range(8)])
    assert g.untested_plans(A) == [("ACTION1",), ("ACTION2",)]  # RESET never planned
    g.record(A, ("ACTION1",), B, changed=True, level_up=False, game_over=False)
    assert g.untested_plans(A) == [("ACTION2",)]
    g.mark_dead(A, ("ACTION2",))
    assert g.untested_plans(A) == []
    assert g.pop_untested(A) is None


def test_ensure_node_includes_salience_clicks_when_action6_available():
    g = _mk_graph()
    rows = [[0] * 8 for _ in range(8)]
    rows[2][2] = 5
    g.ensure_node(A, ["ACTION1", "ACTION6"], rows)
    plans = g.nodes[A]["plans"]
    assert ("ACTION1",) in plans
    assert ("ACTION6", 2, 2) in plans, plans


def test_bfs_finds_frontier_and_respects_edge_filters():
    g = _mk_graph()
    empty = [[0] * 8 for _ in range(8)]
    for key in (A, B, C):
        g.ensure_node(key, ["ACTION1", "ACTION2"], empty)
    # A --ACTION1--> B --ACTION1--> C ; C has untested plans, A/B fully tested.
    g.record(A, ("ACTION1",), B, changed=True, level_up=False, game_over=False)
    g.record(A, ("ACTION2",), A, changed=False, level_up=False, game_over=False)
    g.record(B, ("ACTION1",), C, changed=True, level_up=False, game_over=False)
    g.record(B, ("ACTION2",), B, changed=False, level_up=False, game_over=False)
    assert g.bfs_to_frontier(A) == [("ACTION1",), ("ACTION1",)]

    # Danger edges are never traversed.
    g.edges[B][("ACTION1",)]["danger"] = True
    assert g.bfs_to_frontier(A) is None
    g.edges[B][("ACTION1",)]["danger"] = False

    # Cross-level and level-up edges are never traversed.
    g.edges[B][("ACTION1",)]["level_up"] = True
    assert g.bfs_to_frontier(A) is None
    g.edges[B][("ACTION1",)]["level_up"] = False

    # Inconsistent (non-deterministic) edges are never traversed.
    g.record(B, ("ACTION1",), OTHER_LEVEL, changed=True, level_up=False, game_over=False)
    assert g.edges[B][("ACTION1",)]["consistent"] is False
    assert g.bfs_to_frontier(A) is None


def test_node_key_ignores_masked_cells():
    g = _mk_graph()
    grid_a = [[1] * 8 for _ in range(8)]
    grid_b = [list(row) for row in grid_a]
    grid_b[0][3] = 9  # differs only at the masked cell
    mask = [[0, 3]]
    assert g.node_key(1, grid_a, mask) == g.node_key(1, grid_b, mask)
    assert g.node_key(1, grid_a, []) != g.node_key(1, grid_b, [])
    assert g.node_key(1, grid_a, mask) != g.node_key(2, grid_a, mask)


# --- scripted engine: the real session class over a deterministic mini-game ------


import arcengine  # noqa: E402


class _ScriptedState:
    def __init__(self, grid, levels: int, state: "arcengine.GameState", avail):
        self.frame = SimpleNamespace(data=[list(row) for row in grid])
        self.levels_completed = levels
        self.available_actions = list(avail)
        self.raw = SimpleNamespace(state=state)
        self.won = state == arcengine.GameState.WIN
        self.just_won_level = False


class ScriptedGame:
    """Deterministic 8x8 mini-game driven through the REAL session class.

    Mechanics (all coordinates are (x=col, y=row)):
      * click on ``unlock_cell`` advances the level (WIN after ``number_of_levels``);
      * click on ``danger_cell`` -> GAME_OVER;
      * ACTION1 is always a no-op;
      * ACTION2 toggles the cell at (row 0, col 7) between 3 and 4;
      * RESET repaints the current level (recovers from GAME_OVER).
    """

    def __init__(self, unlock_cell=(2, 2), danger_cell=(5, 5), levels=3):
        self.unlock_cell = unlock_cell
        self.danger_cell = danger_cell
        self.number_of_levels = levels
        self.levels = 0
        self.toggled = False
        self.state_name = arcengine.GameState.NOT_FINISHED
        self.game_run = SimpleNamespace(
            state="playing", game_id="scripted", history=[], levels_completed=0,
            final_score=None, number_of_levels=levels, actions_per_level=[],
        )
        self._refresh()

    def _grid(self):
        rows = [[0] * 8 for _ in range(8)]
        rows[7][0] = self.levels + 1  # level indicator: every level looks different
        rows[0][7] = 4 if self.toggled else 3
        if self.unlock_cell is not None:
            x, y = self.unlock_cell
            rows[y][x] = 5
        if self.danger_cell is not None:
            x, y = self.danger_cell
            rows[y][x] = 6
        return rows

    def _refresh(self):
        avail = [
            arcengine.GameAction.RESET.value,
            arcengine.GameAction.ACTION1.value,
            arcengine.GameAction.ACTION2.value,
            arcengine.GameAction.ACTION6.value,
        ]
        self.current_state = _ScriptedState(
            self._grid(), self.levels, self.state_name, avail
        )

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        name = action.id.name
        self.game_run.history.append(
            SimpleNamespace(action=SimpleNamespace(id=SimpleNamespace(name=name)))
        )
        if name == "RESET":
            self.toggled = False
            self.state_name = arcengine.GameState.NOT_FINISHED
        elif self.state_name == arcengine.GameState.NOT_FINISHED:
            if name == "ACTION2":
                self.toggled = not self.toggled
            elif name == "ACTION6":
                point = (int(action.data.get("x", -1)), int(action.data.get("y", -1)))
                if self.unlock_cell is not None and point == self.unlock_cell:
                    self.levels += 1
                    self.toggled = False
                    self.game_run.levels_completed = self.levels
                    if self.levels >= self.number_of_levels:
                        self.state_name = arcengine.GameState.WIN
                elif self.danger_cell is not None and point == self.danger_cell:
                    self.state_name = arcengine.GameState.GAME_OVER
        self._refresh()
        return self.current_state

    def finish_game(self):
        self.game_run.final_score = float(self.levels)


class _IdleAnalyzer:
    generated_tokens = 0
    _timeout = None


@pytest.fixture(scope="module")
def patched_solver():
    """The full patch chain (same order as apply_all), applied once per process."""
    for status in (
        duck_patches.patch_watchdog(),
        duck_patches.patch_hud_board_identity(),
        duck_patches.patch_win_replay(),
        duck_patches.patch_frontier_graph(),
    ):
        assert "OK" in status or "SKIP" in status, status
    from inference.framework import solver

    return solver


def _scripted_session(solver_mod, tmp_path, game=None):
    game = game or ScriptedGame()
    solver = solver_mod.HarnessSolver(
        label="graph-test", model="stub", analyzer_timeout=5.0, concurrency=1
    )
    solver.job_dir = tmp_path
    session = solver_mod._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=_IdleAnalyzer(),
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "artifacts" / "runtime_state.json",
        transcript_path=tmp_path / "transcripts" / "scripted.txt",
        analysis_html_relpath="solver_analysis/scripted.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "artifacts" / "viewer_data.json",
    )
    return game, session


def _stalled_watchdog(session):
    """Fake patch-7 state showing a stall on the session's own thread."""
    session._watchdog_state = {
        "stall_s": 1.0,
        "wall_cap_s": 0.0,
        "max_resets": 1,
        "progress": None,
        "t_progress": time.monotonic() - 60.0,
        "resets_done": 0,
        "in_reset": False,
        "killed": None,
        "thread_id": threading.get_ident(),
    }
    return session._watchdog_state


def test_veto_engages_after_threshold_and_counts(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_GRAPH_VETO_MIN_LEVEL_ACTIONS", "2")
    monkeypatch.setenv("TAAF_GRAPH_VETO_CAP", "2")
    game, session = _scripted_session(patched_solver, tmp_path)

    p1 = session.step_env({"action": "ACTION1"})
    p2 = session.step_env({"action": "ACTION1"})
    assert p1["executed"] and "vetoed_noop" not in p1
    assert p2["executed"] and "vetoed_noop" not in p2
    assert session.action_count == 2

    p3 = session.step_env({"action": "ACTION1"})
    assert p3.get("vetoed_noop") is True, p3
    assert p3["executed"] is True and p3["board_changed"] is False
    assert session.action_count == 2, "a vetoed action must not reach the engine"

    p4 = session.step_env({"action": "ACTION1"})
    assert p4.get("vetoed_noop") is True

    # Per-level cap: the third would-be veto executes for real instead.
    p5 = session.step_env({"action": "ACTION1"})
    assert "vetoed_noop" not in p5 and session.action_count == 3

    diag = duck_patches.graph_diagnostics(session)
    assert diag["vetoes_issued"] == 2
    assert diag["vetoes_capped"] >= 1
    assert diag["nodes"] >= 1 and diag["edges"] >= 1


def test_veto_never_fires_for_reset_or_changing_actions(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_GRAPH_VETO_MIN_LEVEL_ACTIONS", "0")
    game, session = _scripted_session(patched_solver, tmp_path)

    # ACTION2 toggles a cell every time: never a no-op, never vetoed.
    for _ in range(4):
        payload = session.step_env({"action": "ACTION2"})
        assert payload["executed"] and "vetoed_noop" not in payload
        assert payload["board_changed"] is True
    # RESET is exempt even when its edge looks like a self-loop.
    for _ in range(3):
        payload = session.step_env({"action": "RESET"})
        assert "vetoed_noop" not in payload
    assert duck_patches.graph_diagnostics(session)["vetoes_issued"] == 0


def test_veto_respects_first_n_actions_of_level_guard(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_GRAPH_VETO_MIN_LEVEL_ACTIONS", "10")
    game, session = _scripted_session(patched_solver, tmp_path)
    for i in range(6):
        payload = session.step_env({"action": "ACTION1"})
        assert "vetoed_noop" not in payload, f"action {i} vetoed inside the guard window"
    assert session.action_count == 6


def test_veto_kill_switch_is_inert(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "0")
    game, session = _scripted_session(patched_solver, tmp_path)
    for _ in range(4):
        payload = session.step_env({"action": "ACTION1"})
        assert "vetoed_noop" not in payload
    assert session.action_count == 4
    assert getattr(session, "_graph_state", None) is None, (
        "TAAF_GRAPH=0 must keep the substrate fully inert"
    )


def test_grinder_unlocks_level_and_refreshes_watchdog(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_BUDGET", "50")
    game, session = _scripted_session(patched_solver, tmp_path)
    wd = _stalled_watchdog(session)

    stopped = session.should_stop()
    assert stopped is False

    diag = duck_patches.graph_diagnostics(session)
    assert diag["grinder_engagements"] == 1
    assert diag["levels_unlocked_by_grinder"] == 1, diag
    assert game.levels == 1, "the frontier walk must have clicked the unlock cell"
    assert 1 <= diag["grinder_actions"] <= 8, diag
    assert session._graph_state["grinding"] is False
    # progress was made -> the watchdog timer must have been re-armed, so the
    # recovery RESET did not fire on the same poll.
    assert wd["resets_done"] == 0
    assert (time.monotonic() - wd["t_progress"]) < 5.0


def test_grinder_stops_on_game_over(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_BUDGET", "50")
    # No unlock cell; the danger cell is the most button-like blob -> reached fast.
    game = ScriptedGame(unlock_cell=None, danger_cell=(1, 1))
    game, session = _scripted_session(patched_solver, tmp_path, game=game)
    _stalled_watchdog(session)

    session.should_stop()

    diag = duck_patches.graph_diagnostics(session)
    assert diag["grinder_engagements"] == 1
    assert diag["levels_unlocked_by_grinder"] == 0
    assert game.state_name == arcengine.GameState.GAME_OVER
    assert diag["grinder_actions"] < 50, "grind must stop AT the game over"


def test_grinder_respects_budget(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_BUDGET", "3")
    game = ScriptedGame(unlock_cell=None, danger_cell=None)
    game, session = _scripted_session(patched_solver, tmp_path, game=game)
    _stalled_watchdog(session)

    session.should_stop()

    diag = duck_patches.graph_diagnostics(session)
    assert diag["grinder_actions"] == 3, diag
    assert game.levels == 0


def test_grinder_exhausts_frontier_and_never_respins(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_BUDGET", "500")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_MAX_PER_LEVEL", "5")
    game = ScriptedGame(unlock_cell=None, danger_cell=None)
    game, session = _scripted_session(patched_solver, tmp_path, game=game)
    wd = _stalled_watchdog(session)

    session.should_stop()
    diag = duck_patches.graph_diagnostics(session)
    assert diag["grinder_engagements"] == 1
    assert 0 < diag["grinder_actions"] < 500, "tiny game must exhaust before budget"
    level = patched_solver._level_number(game)
    assert level in session._graph_state["grind_exhausted"]

    # Still stalled: the next poll must NOT re-engage (exhausted level), so the
    # watchdog's own recovery path stays reachable.
    wd["t_progress"] = time.monotonic() - 60.0
    session.should_stop()
    assert duck_patches.graph_diagnostics(session)["grinder_engagements"] == 1


def test_grinder_disengages_permanently_on_completed_level(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    game, session = _scripted_session(patched_solver, tmp_path)

    # Complete level 1 for real (scored actions accumulated on its counter).
    payload = session.step_env({"action": "MOUSE", "row": 2, "col": 2})
    assert payload["executed"] and game.levels == 1
    assert 1 in session._graph_state["completed_levels"]

    # Simulate a fresh play visiting the completed level again (full reset).
    game.levels = 0
    game.game_run.levels_completed = 0
    game.state_name = arcengine.GameState.NOT_FINISHED
    game._refresh()
    _stalled_watchdog(session)

    session.should_stop()
    diag = duck_patches.graph_diagnostics(session)
    assert diag["grinder_engagements"] == 0, (
        "grinding a completed level's counter destroys its score — must disengage"
    )

    # But a never-completed level still engages.
    game.levels = 1
    game.game_run.levels_completed = 1
    game._refresh()
    _stalled_watchdog(session)
    session.should_stop()
    assert duck_patches.graph_diagnostics(session)["grinder_engagements"] == 1


def test_grinder_dormant_without_watchdog_state(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    game, session = _scripted_session(patched_solver, tmp_path)
    # No _watchdog_state (watchdog never polled / disabled): nothing may engage.
    session.should_stop()
    assert duck_patches.graph_diagnostics(session).get("grinder_engagements", 0) == 0


def test_diagnostics_dict_shape(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "1")
    game, session = _scripted_session(patched_solver, tmp_path)
    session.step_env({"action": "ACTION1"})
    diag = duck_patches.graph_diagnostics(session)
    assert set(diag) == {
        "nodes", "edges", "vetoes_issued", "vetoes_capped",
        "grinder_engagements", "grinder_actions", "levels_unlocked_by_grinder",
    }
    assert all(isinstance(value, int) for value in diag.values())


# --- end-to-end over the real engine (stub-brain harness driver) -----------------


GAME = "ft09"


def _result(step_executed: bool, retryable: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        step_executed=step_executed,
        retryable_failure=retryable,
        yielded_control=False,
        reasoning="",
    )


class _StallForeverAnalyzer:
    """Wedged endpoint: every turn is a retryable failure (test_watchdog pattern)."""

    generated_tokens = 0
    _timeout = None

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, *args, **kwargs):
        self.calls += 1
        return _result(step_executed=False, retryable=True)


class _NoopClickAnalyzer:
    """Repeats a known no-op click (ft09: CLICK(31,31) provably changes nothing),
    collecting the payloads, then asks the session to stop."""

    generated_tokens = 0
    _timeout = None

    def __init__(self, stop_event: threading.Event, turns: int = 6) -> None:
        self.turns = turns
        self.payloads: list[dict] = []
        self._stop_event = stop_event

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                **kwargs):
        if len(self.payloads) >= self.turns:
            self._stop_event.set()
            return _result(step_executed=False)
        self.payloads.append(step_env({"action": "MOUSE", "row": 31, "col": 31}))
        return _result(step_executed=True)


def _real_session(tmp_path: Path, analyzer, patched_solver):
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI

    run_session = RunSession(record_intermediate_states=False)
    game = GameAPI(env_name=GAME, arcade_spec=ArcadeSpec(environments_dir=str(ENV_DIR)))
    game.start_game(run_session)

    solver = patched_solver.HarnessSolver(
        label="graph-e2e", model="stub", analyzer_timeout=5.0, concurrency=1
    )
    solver.job_dir = tmp_path
    session = patched_solver._HarnessGameSession(
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


def test_e2e_scripted_stall_triggers_grinder(patched_solver, tmp_path, monkeypatch):
    """Stub-brain stall on the real engine: the watchdog's stall signal must hand
    control to the grinder, which executes real engine-speed actions, and the
    watchdog RESET/kill chain must still complete the run behind it."""
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "1.5")
    monkeypatch.setenv("TAAF_WATCHDOG_MAX_RESETS", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_WALL_CAP_S", "0")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_BUDGET", "40")
    monkeypatch.setenv("TAAF_GRAPH_GRIND_MAX_PER_LEVEL", "1")

    analyzer = _StallForeverAnalyzer()
    game, session = _real_session(tmp_path, analyzer, patched_solver)

    t0 = time.monotonic()
    session.play()
    elapsed = time.monotonic() - t0

    diag = duck_patches.graph_diagnostics(session)
    assert diag["grinder_engagements"] >= 1, diag
    assert 1 <= diag["grinder_actions"] <= 40, diag
    assert diag["nodes"] >= 1 and diag["edges"] >= 1, diag
    # The grinder's engine actions are real scored engine actions on the run.
    assert session.action_count >= diag["grinder_actions"]
    # Behind the (capped) grinder the watchdog chain still ends the game.
    wd = session._watchdog_state
    assert wd["killed"] == "stall", wd
    assert game.game_run.state == "gave_up"
    assert game.game_run.final_score is not None, "run must be banked"
    # Grind bulk records are trimmed: the model-facing history must not have
    # accumulated one entry per grind action (milestone/reset entries remain).
    assert len(session.history_entries) <= max(10, diag["grinder_actions"] // 2), (
        len(session.history_entries), diag)
    assert elapsed < 60.0, f"run did not complete promptly ({elapsed:.1f}s)"


def test_e2e_repeated_noop_click_is_vetoed(patched_solver, tmp_path, monkeypatch):
    """ft09 CLICK(31,31) is a measured no-op: after two real observations the
    stub brain's repeats must be answered synthetically at zero action cost."""
    monkeypatch.setenv("TAAF_GRAPH", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "900")
    monkeypatch.setenv("TAAF_GRAPH_VETO_MIN_LEVEL_ACTIONS", "2")

    stop_event = threading.Event()
    analyzer = _NoopClickAnalyzer(stop_event, turns=6)
    game, session = _real_session(tmp_path, analyzer, patched_solver)
    session.stop_event = stop_event

    session.play()

    payloads = analyzer.payloads
    assert len(payloads) == 6
    assert all(p.get("executed") for p in payloads), payloads
    assert "vetoed_noop" not in payloads[0] and "vetoed_noop" not in payloads[1]
    for late in payloads[2:]:
        assert late.get("vetoed_noop") is True, late
        assert late["board_changed"] is False
    # Only the two real observations reached the engine (plus zero resets).
    assert session.action_count == 2, session.action_count
    diag = duck_patches.graph_diagnostics(session)
    assert diag["vetoes_issued"] == 4, diag
    assert game.game_run.final_score is not None


def test_e2e_kill_switch_leaves_run_untouched(patched_solver, tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "0")
    monkeypatch.setenv("TAAF_WATCHDOG", "1")
    monkeypatch.setenv("TAAF_WATCHDOG_STALL_S", "900")

    stop_event = threading.Event()
    analyzer = _NoopClickAnalyzer(stop_event, turns=4)
    game, session = _real_session(tmp_path, analyzer, patched_solver)
    session.stop_event = stop_event

    session.play()

    assert all("vetoed_noop" not in p for p in analyzer.payloads)
    assert session.action_count == 4
    assert getattr(session, "_graph_state", None) is None


# --- scored-bundle validation ----------------------------------------------------


def test_patch_declines_when_bundle_lacks_symbol(monkeypatch):
    """Presence-gate: a bundle without a wrapped symbol must get a clean SKIP,
    with the session class left completely untouched (the 7edec38 law)."""
    from inference.framework import solver

    before = (
        solver._HarnessGameSession._execute_action,
        solver._HarnessGameSession.step_env,
        solver._HarnessGameSession.should_stop,
    )
    monkeypatch.delattr(solver, "_level_number")
    status = duck_patches.patch_frontier_graph()
    assert "SKIP (bundle solver lacks" in status and "_level_number" in status, status
    after = (
        solver._HarnessGameSession._execute_action,
        solver._HarnessGameSession.step_env,
        solver._HarnessGameSession.should_stop,
    )
    assert before == after, "a declined patch must not wrap anything"


def test_patch_is_idempotent(patched_solver):
    assert duck_patches.patch_frontier_graph() == "patch11 graph: SKIP (already applied)"


def test_patch_preserves_underlying_patch_markers(patched_solver):
    """The A/B kernel's patch proof reads _watchdog_patched/_hud_patched off the
    class methods; wrapping must carry them forward."""
    cls = patched_solver._HarnessGameSession
    assert getattr(cls.should_stop, "_graph_patched", False)
    assert getattr(cls.should_stop, "_watchdog_patched", False)
    assert getattr(cls._execute_action, "_graph_patched", False)
    assert getattr(cls._execute_action, "_hud_patched", False)
    assert getattr(cls._execute_action, "_win_replay_patched", False)


_SCORED_BUNDLE_PROBE = r"""
import sys

scored = {scored!r}
sys.path.insert(0, scored + "/src/tufa-arc-agi-framework/src")
sys.path.insert(0, scored + "/src/ARC3-Inference")
sys.path.insert(0, {patch_dir!r})

from inference.framework import solver
assert solver.__file__.startswith(scored), solver.__file__

import duck_patches

results = duck_patches.apply_all(verbose=False)
for line in results:
    print(line)
line = next(l for l in results if l.startswith("patch11"))
assert line == "patch11 graph: OK", line
assert getattr(solver._HarnessGameSession._execute_action, "_graph_patched", False)
assert getattr(solver._HarnessGameSession.step_env, "_graph_patched", False)
assert getattr(solver._HarnessGameSession.should_stop, "_graph_patched", False)
assert getattr(solver._HarnessGameSession.should_stop, "_watchdog_patched", False)
print("SCORED-BUNDLE-APPLY-OK")
"""


def test_patch_applies_on_scored_bundle_bytes(tmp_path):
    """Run apply_all in a subprocess whose sys.path serves the ACTUAL scored
    bundle bytes (scratchpad/taaf_scored_ref): patch11 must report OK there —
    the drifted _adopt tree is NOT what runs at eval."""
    if not SCORED_REF.is_dir():
        pytest.skip("scored bundle reference not present")
    script = _SCORED_BUNDLE_PROBE.format(
        scored=str(SCORED_REF), patch_dir=str(Path(__file__).parent)
    )
    probe = tmp_path / "scored_probe.py"
    probe.write_text(script)
    proc = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-BUNDLE-APPLY-OK" in proc.stdout
    assert "patch11 graph: OK" in proc.stdout
