"""Offline validation for the explorer fallback graft (Pack 4).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_explore.py
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(os.environ.get(
    "GRAFT_TEST_BUNDLE",
    str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference"),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_control as tc  # noqa: E402
import graft_explore as te  # noqa: E402
import graft_throughput as tp  # noqa: E402


def _clear_flags() -> None:
    for name in list(os.environ):
        if name.startswith(("TP_", "TP2_", "TP4_")):
            os.environ.pop(name, None)


def grid_with(cells, h=16, w=16):
    g = [[0] * w for _ in range(h)]
    for (r, c), v in cells.items():
        g[r][c] = v
    return tuple(tuple(row) for row in g)


class _FakeState:
    def __init__(self, grid, available=(1, 2, 3, 4), levels_completed=0):
        self.frame = types.SimpleNamespace(data=[list(r) for r in grid])
        self.available_actions = list(available)
        self.levels_completed = levels_completed
        self.raw = types.SimpleNamespace(state="NOT_FINISHED")
        self.just_won_level = False
        self.won = False


class _FakeSession:
    """Moves a 2x2 block right on ACTION4, otherwise no effect; completing
    the level when the block reaches column 12."""

    def __init__(self):
        self.col = 2
        self.game = types.SimpleNamespace(current_state=_FakeState(self._grid()),
                                          game_run=types.SimpleNamespace(game_id="fake"))
        self.calls = []
        self.action_count = 0
        self.level = 1
        self.analyzer = types.SimpleNamespace()

    def _grid(self):
        return grid_with({(5, self.col): 9, (5, self.col + 1): 9, (6, self.col): 9, (6, self.col + 1): 9})

    def should_stop(self):
        return False

    def timing_payload(self):
        return {"run_elapsed_seconds": 100.0, "time_remaining_seconds": 5000.0}

    def write_runtime_state(self):
        pass

    def _execute_action(self, action, *, batch_index, batch_size, generated_tokens=None, **kw):
        name = action.id.name
        self.calls.append(name)
        self.action_count += 1
        level_completed = False
        if name == "ACTION4" and self.col < 12:
            self.col += 1
            if self.col == 12:
                level_completed = True
                self.level = 2
        self.game.current_state = _FakeState(self._grid(), levels_completed=self.level - 1)
        return {"executed": True, "level": self.level, "board_changed": name == "ACTION4", "frame_count": 1,
                "action_num": self.action_count, "level_completed": level_completed, "game_over": False,
                "run_complete": False}


class ExploreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415
        import arcengine  # noqa: PLC0415
        import frontier_explorer as fe  # noqa: PLC0415
        cls.agent_mod, cls.solver_mod, cls.arcengine, cls.fe = agent_mod, solver_mod, arcengine, fe
        tp.install()
        tc.install()
        cls.status = te.install()

    def setUp(self) -> None:
        _clear_flags()

    def test_01_install(self) -> None:
        self.assertIn(self.status, {"explore: OK", "explore: SKIP (already applied)"})
        self.assertTrue(hasattr(self.agent_mod.ToolAgent.analyze, "_tp4_stock"))
        self.assertTrue(hasattr(tc._after_action, "_tp4_stock"))

    def test_02_run_explorer_completes_the_fake_level(self) -> None:
        sess = _FakeSession()
        with mock.patch.object(self.solver_mod, "_level_number", lambda g: 1 + int(g.current_state.levels_completed)), \
             mock.patch.object(self.solver_mod, "_is_engine_game_over", lambda g: False):
            rec = te.run_explorer(sess, self.solver_mod, self.arcengine, self.fe, max_actions=400, why="test")
        self.assertIn("COMPLETED level 1", rec["outcome"])
        self.assertLessEqual(rec["actions"], 400)
        self.assertGreaterEqual(sess.calls.count("ACTION4"), 10)

    def test_03_budget_respected_when_stuck(self) -> None:
        sess = _FakeSession()
        sess.col = 12   # already at the wall: nothing progresses
        with mock.patch.object(self.solver_mod, "_level_number", lambda g: 1), \
             mock.patch.object(self.solver_mod, "_is_engine_game_over", lambda g: False):
            rec = te.run_explorer(sess, self.solver_mod, self.arcengine, self.fe, max_actions=25, why="test")
        self.assertEqual(rec["actions"], 25)
        self.assertIn("no level progress", rec["outcome"])

    def test_04_analyze_triggers_at_t3_and_leaves_a_note(self) -> None:
        os.environ["TP4_STALL_T3"] = "5"
        agent = self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
        sess = _FakeSession()
        st = tc._state(sess)
        st.since_new = 7
        st.level = 1
        st.resets_this_level = 1
        captured = {}

        def fake_stock(self_, state_path, action_num, valid_actions=None, step_env=None, **kw):
            captured["action_num"] = action_num
            return None

        fake_rec = {"actions": 3, "outcome": "made no level progress", "tested": 3, "nodes": 2, "why": "w", "wall_s": 0.0}
        with mock.patch.object(te, "run_explorer", return_value=fake_rec) as run, \
             mock.patch.dict(te._ANALYZE_STOCK, {"fn": fake_stock}), \
             mock.patch.object(self.solver_mod, "_engine_action_names", lambda g: ["ACTION1"]):
            bound = types.MethodType(lambda self_, a: None, sess)
            self.agent_mod.ToolAgent.analyze(agent, Path("/nonexistent"), 7, valid_actions=["UP"], step_env=bound)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(captured["action_num"], sess.action_count)
        self.assertEqual(st.since_new, 0)
        self.assertIn("model-free explorer took over", st.last_reset_note)
        self.assertEqual(st.explorer_runs_this_level, 1)

    def test_05_disabled_is_passthrough(self) -> None:
        os.environ["TP4_ENABLE"] = "0"
        os.environ["TP4_STALL_T3"] = "1"
        agent = self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
        sess = _FakeSession()
        st = tc._state(sess)
        st.since_new = 50
        st.resets_this_level = 1
        with mock.patch.object(te, "run_explorer") as run, \
             mock.patch.dict(te._ANALYZE_STOCK, {"fn": lambda *a, **k: None}), \
             mock.patch.dict(tc._ANALYZE_STOCK, {"fn": lambda *a, **k: None}):
            bound = types.MethodType(lambda self_, a: None, sess)
            self.agent_mod.ToolAgent.analyze(agent, Path("/nonexistent"), 7, valid_actions=["UP"], step_env=bound)
        self.assertEqual(run.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
