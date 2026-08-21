"""Offline validation for the truthful_telemetry graft (no GPU/network/Kaggle).

Covers: per-flag presence/absence of each injection, no duplicates across
turns, flags-off byte-identity, the level-count transport through
``HarnessSolver._make_analyzer`` (including wrapped analyzers), fail-open —
and test 08 replays the graft's RESET claim end-to-end through the REAL
``_HarnessGameSession.step_env`` with a duck-typed engine game:
``action(['RESET'])``-shaped input executes with zero code change even
though RESET is stripped from the model-visible valid_actions.

Run:  .venv/bin/python submission/_truthful_telemetry/test_truthful_telemetry.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
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

from graft_telemetry import PAYLOAD_LINE, RESET_LINE, TIME_LINE, install


def _clear_flags() -> None:
    for name in ("TT_RESET", "TT_TIME", "TT_LEVELS", "TT_PAYLOAD"):
        os.environ.pop(name, None)


class _FakeEngineState:
    """Duck-typed taaf GameState: RESET (id 0) present in available_actions,
    exactly the shape solver.py:109-120 filters and solver.py:608 validates."""

    def __init__(self, arcengine) -> None:
        self.raw = SimpleNamespace(state=arcengine.GameState.NOT_FINISHED)
        self.available_actions = [
            arcengine.GameAction.RESET.value,
            arcengine.GameAction.ACTION1.value,
        ]
        self.levels_completed = 0
        self.won = False
        self.just_won_level = False
        self.frame = SimpleNamespace(data=[[0, 0], [0, 0]])


class _FakeEngineGame:
    def __init__(self, arcengine) -> None:
        self._state = _FakeEngineState(arcengine)
        self.game_run = SimpleNamespace(
            state="playing", history=[], solver_analysis_html=None, game_id="fake"
        )
        self.number_of_levels = 5
        self.executed: list = []

    @property
    def current_state(self):
        return self._state

    def execute_action(self, action, *, generated_tokens=0, uncached_input_tokens=0):
        self.executed.append(action)
        self.game_run.history.append(action)
        return self._state


class TruthfulTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        import arcengine  # noqa: PLC0415

        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415

        cls.arcengine = arcengine
        cls.agent_mod = agent_mod
        cls.solver_mod = solver_mod
        cls.install_status = install()

    def setUp(self) -> None:
        _clear_flags()

    def _agent(self):
        return self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def _frame(self, level: int = 2):
        from inference.agent.runtime_state import Frame  # noqa: PLC0415

        return Frame(grid=((0, 1), (1, 0)), step=3, level=level)

    # --- installation -------------------------------------------------------

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "truthful_telemetry: OK")
        self.assertEqual(install(), "truthful_telemetry: SKIP (already applied)")

    # --- system-prompt lines, per flag -------------------------------------

    def test_02_all_lines_present_exactly_once_by_default(self) -> None:
        prompt = self._agent()._system_prompt
        for line in (RESET_LINE, TIME_LINE, PAYLOAD_LINE):
            self.assertEqual(prompt.count(line), 1, msg=line[:40])

    def test_03_each_flag_independently_removes_its_line(self) -> None:
        for flag, line, others in (
            ("TT_RESET", RESET_LINE, (TIME_LINE, PAYLOAD_LINE)),
            ("TT_TIME", TIME_LINE, (RESET_LINE, PAYLOAD_LINE)),
            ("TT_PAYLOAD", PAYLOAD_LINE, (RESET_LINE, TIME_LINE)),
        ):
            _clear_flags()
            os.environ[flag] = "0"
            prompt = self._agent()._system_prompt
            self.assertNotIn(line, prompt, msg=flag)
            for other in others:
                self.assertEqual(prompt.count(other), 1, msg=f"{flag} removed {other[:30]}")

    def test_04_all_flags_off_byte_identical_to_stock(self) -> None:
        for name in ("TT_RESET", "TT_TIME", "TT_LEVELS", "TT_PAYLOAD"):
            os.environ[name] = "0"
        stock_system = install.originals["_build_system_prompt"](tool_output_tokens=1024)
        patched_system = self.agent_mod._build_system_prompt(tool_output_tokens=1024)
        self.assertEqual(patched_system, stock_system)
        agent = self._agent()
        agent._tt_total_levels = 6
        frame = self._frame()
        stock_user = install.originals["_build_user_prompt"](agent, 3, valid_actions=["ACTION1"], current_frame=frame)
        patched_user = agent._build_user_prompt(3, valid_actions=["ACTION1"], current_frame=frame)
        self.assertEqual(patched_user, stock_user)

    # --- level X of N -------------------------------------------------------

    def test_05_level_line_present_correct_and_not_duplicated_across_turns(self) -> None:
        agent = self._agent()
        agent._tt_total_levels = 6
        first = agent._build_user_prompt(3, valid_actions=["ACTION1"], current_frame=self._frame(level=2))
        self.assertEqual(first.count("level 2 of 6"), 1)
        self.assertIn("later levels are worth more", first)
        # next turn, deeper level: the line is rebuilt, not accumulated
        second = agent._build_user_prompt(9, valid_actions=["ACTION1"], current_frame=self._frame(level=3))
        self.assertEqual(second.count("Level progress:"), 1)
        self.assertIn("level 3 of 6", second)
        self.assertNotIn("level 2 of 6", second)

    def test_06_level_line_absent_without_total_or_with_flag_off(self) -> None:
        agent = self._agent()  # no _tt_total_levels stamped
        prompt = agent._build_user_prompt(3, valid_actions=["ACTION1"], current_frame=self._frame())
        self.assertNotIn("Level progress:", prompt)
        os.environ["TT_LEVELS"] = "0"
        agent._tt_total_levels = 6
        prompt = agent._build_user_prompt(3, valid_actions=["ACTION1"], current_frame=self._frame())
        self.assertNotIn("Level progress:", prompt)

    def test_07_make_analyzer_stamps_total_levels_including_wrapped(self) -> None:
        game = _FakeEngineGame(self.arcengine)
        game.number_of_levels = 8
        stub_cls_holder = self._agent()  # real ToolAgent has _build_user_prompt
        solver = self.solver_mod.HarnessSolver(analyzer_factory=lambda g, i: stub_cls_holder)
        analyzer = solver._make_analyzer(game, 0, None)
        self.assertEqual(analyzer._tt_total_levels, 8)

        # wrapped analyzer: stamp lands on the innermost via _inner walking
        inner = self._agent()

        class Wrapper:
            def __init__(self, inner):
                self._inner = inner

        solver2 = self.solver_mod.HarnessSolver(analyzer_factory=lambda g, i: Wrapper(inner))
        solver2._make_analyzer(game, 0, None)
        self.assertEqual(inner._tt_total_levels, 8)

        # fail-open: game without number_of_levels
        stub2 = self._agent()
        solver3 = self.solver_mod.HarnessSolver(analyzer_factory=lambda g, i: stub2)
        out = solver3._make_analyzer(SimpleNamespace(), 0, None)
        self.assertIs(out, stub2)
        self.assertFalse(hasattr(stub2, "_tt_total_levels"))

    # --- the RESET claim, replayed through the real solver seams ------------

    def test_08_reset_executes_through_real_step_env_despite_stripped_list(self) -> None:
        arcengine = self.arcengine
        game = _FakeEngineGame(arcengine)
        # 1) the model-visible list REALLY strips RESET (solver.py:116-117)
        visible = self.solver_mod._engine_action_names(game)
        self.assertEqual(visible, ["ACTION1"])
        self.assertIn(arcengine.GameAction.RESET.value, game.current_state.available_actions)

        # 2) a real session's step_env executes RESET anyway (solver.py:608
        #    validates against the RAW engine list, not the stripped one)
        tmp = Path(tempfile.mkdtemp(prefix="tt_test_"))
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
        payload = session.step_env({"actions": [{"action": "RESET"}]})
        self.assertTrue(payload.get("executed"), msg=f"payload={payload}")
        self.assertEqual(payload.get("action_name"), "RESET")
        self.assertEqual(len(game.executed), 1)
        self.assertEqual(game.executed[0].id, arcengine.GameAction.RESET)
        # the claimed telemetry keys really ride the payload (solver.py:729)
        self.assertIn("run_elapsed_seconds", payload)
        self.assertIn("time_remaining_seconds", payload)
        self.assertEqual(payload.get("score"), 0)  # score == levels_completed
        self.assertEqual(payload.get("reward"), 0.0)  # completion-fraction delta

    def test_09_fail_open_prompt_builder_crash_returns_stock(self) -> None:
        agent = self._agent()
        agent._tt_total_levels = "not-an-int-at-all"  # int() raises -> fail open
        frame = self._frame()
        prompt = agent._build_user_prompt(3, valid_actions=["ACTION1"], current_frame=frame)
        stock = install.originals["_build_user_prompt"](agent, 3, valid_actions=["ACTION1"], current_frame=frame)
        self.assertEqual(prompt, stock)


if __name__ == "__main__":
    unittest.main(verbosity=2)
