"""Offline validation for the expect_queue graft (no GPU/network/Kaggle).

Covers: install/idempotency; per-step expect verification (pass and
mismatch); halt-and-return-control truncation with the compact diff;
strict no-op halt on expect-less batches; level-completion steps never
halting; final-step mismatch reporting without truncation; stock-parity of
the ported loop on invalid actions; transport of `expect` through the REAL
sandbox `action(actions)` path; prompt addendum; flags-off = stock; and
fail-open on graft-internal crashes.

Batches are driven through the REAL ``_HarnessGameSession.step_env`` seam
with a duck-typed scripted engine game (same fixture family as
submission/_truthful_telemetry test 08).

Run:  .venv/bin/python submission/_expect_queue/test_expect_queue.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(
    os.environ.get(
        "GRAFT_TEST_BUNDLE",
        str(_HERE.parents[1] / "scratchpad/bundles/june_stock/src/ARC3-Inference"),
    )
)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_expect
from graft_expect import (
    EXPECT_ADDENDUM_MARKER,
    STOP_REASON_EXPECT,
    STOP_REASON_NOOP,
    install,
)


def _clear_flags() -> None:
    os.environ.pop("EXPECT_QUEUE", None)


def _grid(fill: int, size: int = 4, marks: dict[tuple[int, int], int] | None = None):
    rows = [[fill] * size for _ in range(size)]
    for (row, col), color in (marks or {}).items():
        rows[row][col] = color
    return rows


class _ScriptedState:
    def __init__(
        self,
        arcengine,
        grid,
        *,
        levels_completed: int = 0,
        just_won_level: bool = False,
        state=None,
    ) -> None:
        self.raw = SimpleNamespace(state=state or arcengine.GameState.NOT_FINISHED)
        self.available_actions = [0, 1, 2, 3, 4, 5, 6]
        self.levels_completed = levels_completed
        self.just_won_level = just_won_level
        self.won = False
        self.frame = SimpleNamespace(data=grid)


class _ScriptedGame:
    """Duck-typed taaf Game: each execute_action pops the next scripted state
    (or keeps the current one when the queue is exhausted = strict no-op)."""

    def __init__(self, arcengine, initial: _ScriptedState, queue: list[_ScriptedState]):
        self._state = initial
        self._queue = list(queue)
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
        if self._queue:
            self._state = self._queue.pop(0)
        return self._state


class ExpectQueueTests(unittest.TestCase):
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

    # --- fixtures -----------------------------------------------------------

    def _session(self, game):
        tmp = Path(tempfile.mkdtemp(prefix="eq_test_"))
        return self.solver_mod._HarnessGameSession(
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

    def _game(self, queue_grids, initial_grid=None, **state_kwargs_by_index):
        """Game whose k-th executed action lands on queue_grids[k]."""
        arc = self.arcengine
        initial = _ScriptedState(arc, initial_grid or _grid(0))
        queue = []
        for index, grid in enumerate(queue_grids):
            kwargs = state_kwargs_by_index.get(f"s{index}", {})
            queue.append(_ScriptedState(arc, grid, **kwargs))
        return _ScriptedGame(arc, initial, queue)

    # --- installation -------------------------------------------------------

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(self.install_status, "expect_queue: OK")
        self.assertEqual(install(), "expect_queue: SKIP (already applied)")

    # --- expects verified per step ------------------------------------------

    def test_02_matching_expects_execute_full_batch(self) -> None:
        game = self._game([
            _grid(0, marks={(1, 1): 3}),
            _grid(0, marks={(2, 2): 5}),
            _grid(0, marks={(3, 3): 7}),
        ])
        session = self._session(game)
        payload = session.step_env(
            {
                "actions": [
                    {"action": "ACTION1", "expect": [[1, 1, 3]]},
                    {"action": "ACTION2", "expect": [[2, 2, 5], [0, 0, 0]]},
                    {"action": "ACTION3"},
                ]
            }
        )
        self.assertTrue(payload["executed"], msg=f"payload={payload}")
        self.assertEqual(payload["executed_count"], 3)
        self.assertEqual(payload["requested_count"], 3)
        self.assertFalse(payload["stopped_early"])
        self.assertNotIn("stop_reason", payload)
        self.assertNotIn("expect_mismatch", payload)
        self.assertEqual(len(game.executed), 3)

    def test_03_mismatch_truncates_with_compact_diff(self) -> None:
        game = self._game([
            _grid(0, marks={(1, 1): 3}),
            _grid(0, marks={(1, 1): 3, (0, 3): 9}),  # step 2: (2,2) stays 0, not 5
            _grid(1),
            _grid(2),
        ])
        session = self._session(game)
        payload = session.step_env(
            {
                "actions": [
                    {"action": "ACTION1", "expect": [[1, 1, 3]]},
                    {"action": "ACTION2", "expect": [[2, 2, 5]]},
                    {"action": "ACTION3"},
                    {"action": "ACTION4"},
                ]
            }
        )
        self.assertEqual(payload["executed_count"], 2)
        self.assertEqual(payload["requested_count"], 4)
        self.assertTrue(payload["stopped_early"])
        self.assertEqual(payload["stop_reason"], STOP_REASON_EXPECT)
        self.assertIn("step 2 halted the batch", payload["stop_detail"])
        self.assertIn("remaining 2 actions not executed", payload["stop_detail"])
        self.assertIn("[2,2] got 0 want 5", payload["stop_detail"])
        self.assertEqual(
            payload["expect_mismatch"], [{"row": 2, "col": 2, "want": 5, "got": 0}]
        )
        self.assertEqual(len(game.executed), 2)
        # the diff survives compaction into the model-visible tool result
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        compact = agent._compact_action_result(payload)
        self.assertEqual(compact["stop_reason"], STOP_REASON_EXPECT)
        self.assertIn("step 2 halted the batch", compact["stop_detail"])
        self.assertEqual(compact["executed_count"], 2)

    def test_04_no_expect_batch_halts_on_strict_noop(self) -> None:
        noop_grid = _grid(0, marks={(1, 1): 3})
        game = self._game([
            noop_grid,
            [list(row) for row in noop_grid],  # step 2: equal-valued grid = no-op
            _grid(9),
        ])
        session = self._session(game)
        payload = session.step_env(
            {"actions": [{"action": "ACTION1"}, {"action": "ACTION2"}, {"action": "ACTION3"}]}
        )
        self.assertEqual(payload["executed_count"], 2)
        self.assertTrue(payload["stopped_early"])
        self.assertEqual(payload["stop_reason"], STOP_REASON_NOOP)
        self.assertIn("step 2 halted the batch", payload["stop_detail"])
        self.assertIn("remaining 1 actions not executed", payload["stop_detail"])
        self.assertEqual(len(game.executed), 2)

    def test_05_level_completion_never_halts_via_graft(self) -> None:
        # Step 2 completes a level AND its expect mismatches AND the frame is
        # identical to step 1's — the stock level_completed break must win.
        same = _grid(4)
        game = self._game(
            [same, [list(row) for row in same], _grid(5)],
            initial_grid=_grid(0),  # step 1 really changes the board
            s1={"levels_completed": 1, "just_won_level": True},
        )
        session = self._session(game)
        payload = session.step_env(
            {
                "actions": [
                    {"action": "ACTION1"},
                    {"action": "ACTION2", "expect": [[0, 0, 9]]},
                    {"action": "ACTION3"},
                ]
            }
        )
        self.assertEqual(payload["stop_reason"], "level_completed")
        self.assertEqual(payload["executed_count"], 2)
        self.assertNotIn("expect_mismatch", payload)
        self.assertNotIn("stop_detail", payload)

    def test_06_final_step_mismatch_reports_without_truncation(self) -> None:
        game = self._game([
            _grid(0, marks={(1, 1): 3}),
            _grid(0, marks={(1, 1): 3, (2, 2): 4}),  # want 5, got 4
        ])
        session = self._session(game)
        payload = session.step_env(
            {
                "actions": [
                    {"action": "ACTION1", "expect": [[1, 1, 3]]},
                    {"action": "ACTION2", "expect": [[2, 2, 5]]},
                ]
            }
        )
        self.assertEqual(payload["executed_count"], 2)
        self.assertFalse(payload["stopped_early"])
        self.assertEqual(payload["stop_reason"], STOP_REASON_EXPECT)
        self.assertIn("batch already complete (0 remaining)", payload["stop_detail"])
        self.assertEqual(
            payload["expect_mismatch"], [{"row": 2, "col": 2, "want": 5, "got": 4}]
        )

    # --- stock parity inside the ported loop --------------------------------

    def test_07_invalid_actions_keep_stock_semantics(self) -> None:
        game = self._game([_grid(1), _grid(2)])
        game.current_state.available_actions = [0, 1]  # ACTION2 invalid at start
        session = self._session(game)
        # invalid first action of a batch -> error payload, nothing executed
        payload = session.step_env(
            {"actions": [{"action": "ACTION2"}, {"action": "ACTION1"}]}
        )
        self.assertFalse(payload["executed"])
        self.assertIn("not valid right now", payload["error"])
        self.assertEqual(len(game.executed), 0)
        # invalid LATER action -> executed prefix + stop_reason invalid_action
        game2 = self._game([_grid(1), _grid(2)])
        for state in [game2._state, *game2._queue]:
            state.available_actions = [0, 1]
        session2 = self._session(game2)
        payload2 = session2.step_env(
            {"actions": [{"action": "ACTION1"}, {"action": "ACTION2"}]}
        )
        self.assertEqual(payload2["executed_count"], 1)
        self.assertEqual(payload2["stop_reason"], "invalid_action")
        self.assertTrue(payload2["stopped_early"])
        self.assertEqual(len(game2.executed), 1)

    def test_08_single_expectless_action_delegates_to_stock(self) -> None:
        game = self._game([_grid(1)])
        session = self._session(game)
        with mock.patch.object(
            graft_expect, "_run_batch_with_expects", wraps=graft_expect._run_batch_with_expects
        ) as ported:
            payload = session.step_env({"actions": [{"action": "ACTION1"}]})
            self.assertTrue(payload["executed"])
            ported.assert_not_called()
            payload2 = session.step_env(
                {"actions": [{"action": "ACTION1"}, {"action": "ACTION2"}]}
            )
            self.assertTrue(payload2["executed"])
            ported.assert_called_once()

    # --- transport: expect survives the real sandbox action() path ----------

    def test_09_normalize_transport_and_validation(self) -> None:
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        normalized = agent._normalize_python_actions(
            [{"action": "ACTION1", "expect": [[1, 2, 3]]}, "ACTION2"]
        )
        self.assertEqual(normalized[0]["expect"], [[1, 2, 3]])
        self.assertNotIn("expect", normalized[1])
        for bad in (
            {"action": "ACTION1", "expect": "nope"},
            {"action": "ACTION1", "expect": []},
            {"action": "ACTION1", "expect": [[1, 2]]},
            {"action": "ACTION1", "expect": [[99, 0, 1]]},
            {"action": "ACTION1", "expect": [[0, 0, 1]] * 17},
            {"action": "ACTION1", "expect": [["a", 0, 1]]},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                agent._normalize_python_actions([bad])
        # flag off: expect silently dropped, stock output
        os.environ["EXPECT_QUEUE"] = "0"
        normalized_off = agent._normalize_python_actions(
            [{"action": "ACTION1", "expect": [[1, 2, 3]]}]
        )
        self.assertNotIn("expect", normalized_off[0])

    def test_10_expect_reaches_step_env_through_real_sandbox(self) -> None:
        from inference.agent.runtime_state import (  # noqa: PLC0415
            Frame,
            HistoryEntry,
            write_runtime_state,
        )

        tmp = Path(tempfile.mkdtemp(prefix="eq_sandbox_"))
        state_path = tmp / "tool_runtime_state.json"
        frame = Frame(grid=((0, 1), (1, 0)), step=0, level=1)
        write_runtime_state(
            state_path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)]
        )
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        agent._current_valid_actions = ["ACTION1", "ACTION2"]
        seen: list[dict] = []

        def _capture(arguments):
            seen.append(arguments)
            return {
                "executed": True,
                "action_num": 1,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["ACTION1", "ACTION2"],
                "board_changed": True,
            }

        agent._step_env_callback = _capture
        result = agent._run_python_tool(
            state_path,
            {
                "code": (
                    "action([{'action': 'ACTION1', 'expect': [[0, 0, 3], [1, 1, 0]]}, "
                    "'ACTION2'])"
                )
            },
        )
        self.assertTrue(result.step_executed, msg=result.content)
        self.assertEqual(len(seen), 1)
        actions = seen[0]["actions"]
        self.assertEqual(actions[0]["action"], "ACTION1")
        self.assertEqual(actions[0]["expect"], [[0, 0, 3], [1, 1, 0]])
        self.assertNotIn("expect", actions[1])

    def test_10b_malformed_expect_error_is_model_visible_in_sandbox(self) -> None:
        from inference.agent.runtime_state import (  # noqa: PLC0415
            Frame,
            HistoryEntry,
            write_runtime_state,
        )

        tmp = Path(tempfile.mkdtemp(prefix="eq_sandbox_bad_"))
        state_path = tmp / "tool_runtime_state.json"
        frame = Frame(grid=((0, 1), (1, 0)), step=0, level=1)
        write_runtime_state(
            state_path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)]
        )
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        agent._current_valid_actions = ["ACTION1"]
        agent._step_env_callback = lambda arguments: self.fail("must not reach step_env")
        result = agent._run_python_tool(
            state_path,
            {"code": "action([{'action': 'ACTION1', 'expect': [[1, 2]]}])"},
        )
        self.assertFalse(result.step_executed)
        self.assertIn("expect", result.content)
        self.assertIn("[row, col, color]", result.content)

    # --- prompt addendum ----------------------------------------------------

    def test_11_prompt_addendum_once_on_and_absent_off(self) -> None:
        agent = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        self.assertEqual(agent._system_prompt.count(EXPECT_ADDENDUM_MARKER), 1)
        os.environ["EXPECT_QUEUE"] = "0"
        agent_off = self.agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )
        self.assertNotIn(EXPECT_ADDENDUM_MARKER, agent_off._system_prompt)
        stock = install.originals["_build_system_prompt"](tool_output_tokens=1024)
        self.assertEqual(agent_off._system_prompt, stock)

    # --- flags off = stock --------------------------------------------------

    def test_12_flag_off_batches_never_halt(self) -> None:
        os.environ["EXPECT_QUEUE"] = "0"
        noop_grid = _grid(3)
        game = self._game([
            noop_grid,
            [list(row) for row in noop_grid],  # would be a no-op halt when on
            _grid(9),
        ])
        session = self._session(game)
        payload = session.step_env(
            {"actions": [{"action": "ACTION1"}, {"action": "ACTION2"}, {"action": "ACTION3"}]}
        )
        self.assertEqual(payload["executed_count"], 3)
        self.assertFalse(payload["stopped_early"])
        self.assertNotIn("stop_reason", payload)
        self.assertNotIn("stop_detail", payload)
        self.assertNotIn("expect_mismatch", payload)
        self.assertEqual(len(game.executed), 3)

    # --- fail-open ----------------------------------------------------------

    def test_13_extraction_crash_delegates_to_stock(self) -> None:
        class EvilDict(dict):
            def get(self, key, default=None):
                if key == "expect":
                    raise RuntimeError("boom")
                return super().get(key, default)

        game = self._game([_grid(1), _grid(2)])
        session = self._session(game)
        payload = session.step_env(
            {"actions": [EvilDict(action="ACTION1"), EvilDict(action="ACTION2")]}
        )
        self.assertTrue(payload["executed"])
        self.assertEqual(payload["executed_count"], 2)
        self.assertNotIn("stop_detail", payload)
        self.assertEqual(len(game.executed), 2)

    def test_14_halt_verdict_crash_never_breaks_the_batch(self) -> None:
        game = self._game([_grid(1), _grid(2), _grid(3)])
        session = self._session(game)
        with mock.patch.object(
            graft_expect, "_halt_verdict", side_effect=RuntimeError("boom")
        ):
            payload = session.step_env(
                {
                    "actions": [
                        {"action": "ACTION1", "expect": [[0, 0, 9]]},  # would mismatch
                        {"action": "ACTION2"},
                        {"action": "ACTION3"},
                    ]
                }
            )
        self.assertEqual(payload["executed_count"], 3)
        self.assertFalse(payload["stopped_early"])
        self.assertNotIn("stop_reason", payload)
        self.assertEqual(len(game.executed), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
