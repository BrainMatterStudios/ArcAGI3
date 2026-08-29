"""Offline validation for the control graft (Pack 2). No GPU/network/Kaggle.

Run:  .venv/bin/python submission/_throughput_v1/test_graft_control.py
"""
from __future__ import annotations

import json
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
import graft_throughput as tp  # noqa: E402

FLAGS = [k for k in os.environ if k.startswith("TP2_") or k.startswith("TP_")]


def _clear_flags() -> None:
    for name in list(os.environ):
        if name.startswith("TP2_") or name.startswith("TP_"):
            os.environ.pop(name, None)


def G(rows):
    return tuple(tuple(r) for r in rows)


BASE = [[0] * 8 for _ in range(8)]


def grid_with(cells: dict[tuple[int, int], int]):
    g = [row[:] for row in BASE]
    for (r, c), v in cells.items():
        g[r][c] = v
    return G(g)


class PrimitiveTests(unittest.TestCase):
    def test_01_hash_and_mask(self) -> None:
        a = grid_with({(1, 1): 3})
        b = grid_with({(1, 1): 3, (7, 7): 9})
        self.assertNotEqual(tc.grid_hash(a), tc.grid_hash(b))
        self.assertEqual(tc.grid_hash(a, {(7, 7)}), tc.grid_hash(b, {(7, 7)}))

    def test_02_hud_mask_learns_timer_cell(self) -> None:
        hud = tc.HudMask()
        prev = grid_with({(0, 0): 1})
        for i in range(2, 12):
            nxt = grid_with({(0, 0): i})          # (0,0) ticks every transition
            if i % 3 == 0:
                nxt = tuple(tuple(v if (r, c) != (4, 4) else 5 for c, v in enumerate(row)) for r, row in enumerate(nxt))
            hud.observe(prev, nxt)
            prev = nxt
        m = hud.mask()
        self.assertIn((0, 0), m)
        self.assertNotIn((4, 4), m)

    def test_02b_edge_band_extends_to_whole_row(self) -> None:
        hud = tc.HudMask()
        prev = tuple(tuple(0 for _ in range(64)) for _ in range(64))
        for i in range(1, 12):
            nxt = [list(r) for r in prev]
            nxt[0][63] = i % 7                 # units digit ticks every transition
            if i % 5 == 0:
                nxt[0][62] = i // 5            # tens digit ticks rarely
            nxt = tuple(tuple(r) for r in nxt)
            hud.observe(prev, nxt)
            prev = nxt
        m = hud.mask()
        self.assertIn((0, 62), m)              # slow digit masked via the row extension
        self.assertIn((0, 5), m)
        self.assertNotIn((30, 30), m)

    def test_02c_edge_only(self) -> None:
        self.assertTrue(tc._edge_only({"changed_ex_hud": 2, "bbox": [0, 60, 0, 63]}))
        self.assertTrue(tc._edge_only({"changed_ex_hud": 2, "bbox": [10, 62, 40, 63]}))
        self.assertFalse(tc._edge_only({"changed_ex_hud": 2, "bbox": [10, 10, 12, 12]}))
        self.assertFalse(tc._edge_only({"changed_ex_hud": 0, "bbox": [0, 0, 0, 0]}))

    def test_03_diff_summary(self) -> None:
        a = grid_with({(1, 1): 3, (2, 2): 4})
        b = grid_with({(1, 2): 3, (2, 2): 4, (0, 0): 7})
        d = tc.diff_summary(a, b, mask={(0, 0)})
        self.assertEqual(d["changed"], 3)
        self.assertEqual(d["changed_ex_hud"], 2)
        self.assertEqual(d["bbox"], [1, 1, 1, 2])
        self.assertIn(3, d["colors_added"])
        self.assertIn(3, d["colors_removed"])

    def test_04_components_and_salient_clicks(self) -> None:
        g = grid_with({(1, 1): 2, (1, 2): 2, (2, 1): 2, (2, 2): 2, (5, 5): 6, (5, 6): 6, (7, 0): 2})
        comps = tc.components(g)
        self.assertEqual(sorted(c["area"] for c in comps), [1, 2, 4])
        picks = tc.salient_clicks(g, 2)
        self.assertEqual(len(picks), 2)
        r, c, info = picks[0]
        self.assertEqual(info["color"], 6)            # rarest colour with area >= 2 first
        self.assertEqual(g[r][c], 6)


class _FakeState:
    def __init__(self, grid, available=(1, 2, 3, 4, 6), levels_completed=0):
        self.frame = types.SimpleNamespace(data=[list(r) for r in grid])
        self.available_actions = list(available)
        self.levels_completed = levels_completed
        self.raw = types.SimpleNamespace(state="NOT_FINISHED")
        self.just_won_level = False
        self.won = False


class _FakeSession:
    """Minimal stand-in for _HarnessGameSession used by the seam tests."""

    def __init__(self, grid):
        self.game = types.SimpleNamespace(current_state=_FakeState(grid), game_run=types.SimpleNamespace(game_id="t"))
        self.calls = []
        self.action_count = 0
        self.history_entries = []
        self.analyzer = types.SimpleNamespace()
        self._next = {}

    def seed_initial_history(self):
        self.history_entries.append("init")

    def write_runtime_state(self):
        pass

    def _error_payload(self, message):
        return {"executed": False, "error": message}

    def _execute_action(self, action, *, batch_index, batch_size, generated_tokens=None, **kw):
        name = action.id.name
        self.calls.append((name, dict(action.data)))
        self.action_count += 1
        nxt = self._next.get(name)
        if nxt is not None:
            self.game.current_state = _FakeState(nxt)
        payload = {"executed": True, "level": 1, "board_changed": nxt is not None, "frame_count": 1,
                   "action_num": self.action_count, "level_completed": False, "game_over": False,
                   "run_complete": False}
        return payload


class SeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear_flags()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415
        import arcengine  # noqa: PLC0415
        cls.agent_mod, cls.solver_mod, cls.arcengine = agent_mod, solver_mod, arcengine
        tp.install()
        cls.status = tc.install()

    def setUp(self) -> None:
        _clear_flags()

    def _agent(self):
        return self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def test_10_install_and_seams(self) -> None:
        self.assertIn(self.status, {"control: OK", "control: SKIP (already applied)"})
        self.assertTrue(tc._STATE["installed"])
        self.assertIs(tp.ON_CUT.__name__, "on_cut") if hasattr(tp.ON_CUT, "__name__") else None
        for name in ("_build_user_prompt", "analyze", "_compact_action_result", "_run_python_tool",
                     "_update_summarized_knowledge_from_step_summary"):
            self.assertTrue(hasattr(getattr(self.agent_mod.ToolAgent, name), "_tp2_stock"), name)
        for name in ("play", "step_env", "_execute_action"):
            self.assertTrue(hasattr(getattr(self.solver_mod._HarnessGameSession, name), "_tp2_stock"), name)

    def test_11_after_action_updates_diff_stall_streak(self) -> None:
        st = tc.SessionState()
        a = grid_with({(1, 1): 3})
        b = grid_with({(1, 2): 3})
        p = {"executed": True, "level": 1, "board_changed": True, "frame_count": 1}
        tc._after_action(st, a, b, p)
        self.assertEqual(p["diff"]["changed"], 2)
        self.assertEqual(st.since_new, 0)
        self.assertEqual(st.streak, 0)
        # same state again -> since_new climbs; no change -> streak climbs
        p2 = {"executed": True, "level": 1, "board_changed": False, "frame_count": 1}
        tc._after_action(st, b, b, p2)
        tc._after_action(st, b, b, dict(p2))
        self.assertEqual(st.since_new, 2)
        self.assertEqual(st.streak, 2)
        # level change resets stall counters
        tc._after_action(st, b, a, {"executed": True, "level": 2, "board_changed": True, "frame_count": 1})
        self.assertEqual(st.level, 2)
        self.assertEqual(st.since_new, 0)

    def test_12_streak_halts_step_env_and_tool_call_resets(self) -> None:
        sess = _FakeSession(grid_with({(1, 1): 3}))
        st = tc._state(sess)
        st.streak = 3
        out = self.solver_mod._HarnessGameSession.step_env(sess, {"actions": [{"action": "UP"}]})
        self.assertFalse(out["executed"])
        self.assertIn("no_effect_streak", out["error"])
        # animation queries pass through (would reach stock, which needs a real session) -> guarded by flag only
        agent = self._agent()
        agent._step_env_callback = sess.step_env if hasattr(sess, "step_env") else None
        agent._step_env_callback = types.MethodType(lambda self_, a: None, sess)
        agent._run_python_tool(Path("/nonexistent"), {"code": ""})
        self.assertEqual(st.streak, 0)

    def test_13_prompt_gets_probe_and_stagnation(self) -> None:
        agent = self._agent()
        sess = _FakeSession(grid_with({(1, 1): 3}))
        agent._step_env_callback = types.MethodType(lambda self_, a: None, sess)
        agent._tp2_probe = {"level": 1, "text": "Harness probe: UP: no effect", "actions": 4}
        st = tc._state(sess)
        st.since_new = 12
        frame = self.agent_mod.Frame(grid=grid_with({(1, 1): 3}), step=5, level=1)
        text = agent._build_user_prompt(5, valid_actions=["UP", "DOWN"], current_frame=frame)
        self.assertIn("Harness probe", text)
        self.assertIn("STAGNATION WARNING", text)
        frame2 = self.agent_mod.Frame(grid=grid_with({(1, 1): 3}), step=9, level=2)
        st.since_new = 0
        text2 = agent._build_user_prompt(9, valid_actions=["UP"], current_frame=frame2)
        self.assertNotIn("Harness probe", text2)
        self.assertNotIn("STAGNATION WARNING", text2)

    def test_14_analyze_resets_level_at_t2(self) -> None:
        agent = self._agent()
        sess = _FakeSession(grid_with({(1, 1): 3}))
        sess.game.current_state.available_actions = [0, 1, 2, 3, 4]
        st = tc._state(sess)
        st.since_new = 45
        captured = {}

        def fake_stock(self_, state_path, action_num, valid_actions=None, step_env=None, **kw):
            captured["action_num"] = action_num
            return None

        with mock.patch.dict(tc._ANALYZE_STOCK, {"fn": fake_stock}), \
             mock.patch.object(self.solver_mod, "_engine_action_names", lambda g: ["ACTION1", "RESET"]):
            bound = types.MethodType(lambda self_, a: None, sess)
            # call the Pack-2 wrapper directly (later grafts may wrap it again)
            wrapper = getattr(self.agent_mod.ToolAgent.analyze, "_tp4_stock", self.agent_mod.ToolAgent.analyze)
            wrapper(agent, Path("/nonexistent"), 45, valid_actions=["UP", "RESET"], step_env=bound)
        self.assertEqual(captured["action_num"], sess.action_count)
        self.assertEqual([c[0] for c in sess.calls], ["RESET"])
        self.assertEqual(st.resets_this_level, 1)
        self.assertEqual(st.since_new, 0)
        self.assertIn("RESET by the harness", st.last_reset_note)

    def test_15_probe_runs_keyboard_then_clicks(self) -> None:
        g = grid_with({(1, 1): 2, (1, 2): 2, (5, 5): 6, (5, 6): 6})
        sess = _FakeSession(g)
        sess._next["ACTION1"] = grid_with({(0, 1): 2, (0, 2): 2, (5, 5): 6, (5, 6): 6})
        rec = tc.run_probe(sess, self.solver_mod, self.arcengine)
        names = [c[0] for c in sess.calls]
        self.assertEqual(names[:4], ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        self.assertEqual(names.count("ACTION6"), 2)   # two components only
        self.assertIn("UP: changed", rec["text"])
        self.assertIn("DOWN: no effect", rec["text"])
        self.assertEqual(rec["level"], 1)
        self.assertEqual(rec["actions"], 6)

    def test_16_summary_merges_notes_and_fails_open(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "old"
        dropped = [
            {"role": "user", "content": "The code executed 1 action.\nstate"},
            {"role": "assistant", "content": "World model: walls", "tool_calls": [
                {"id": "c1", "function": {"name": "python", "arguments": json.dumps({"code": "action('UP')"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"executed\": true}"},
        ]
        reply = {"choices": [{"message": {"content":
                 "World model: player at (3,4), walls block\nGoal model: reach the door (guess)\n"
                 "Action model: UP moves player up 1\nRecent findings: door at (0,9)\n"
                 "Open questions: what the key does\nPlan: go up then right\nCross-level notes: keys open doors"}}]}
        resp = mock.Mock(); resp.raise_for_status = lambda: None; resp.json = lambda: reply
        with mock.patch("requests.post", return_value=resp) as post:
            self.assertTrue(tc._summarize_into_notes(agent, dropped, self.agent_mod, force=True))
            payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 300)
        self.assertIn("Previous notes:", payload["messages"][1]["content"])
        self.assertEqual(agent._summarized_knowledge["world_model"], "player at (3,4), walls block")
        self.assertEqual(agent._summarized_knowledge["cross_level_notes"], "keys open doors")
        # failure leaves notes untouched (ON_CUT swallows; here the helper raises)
        with mock.patch("requests.post", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                tc._summarize_into_notes(agent, dropped, self.agent_mod, force=True)
        self.assertEqual(agent._summarized_knowledge["world_model"], "player at (3,4), walls block")

    def test_17_level_boundary_keeps_action_model_and_cross_level(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge.update({"world_model": "w", "goal_model": "g", "action_model": "a-old",
                                            "current_plan": "p", "cross_level_notes": "old"})
        agent._history_messages = [{"role": "assistant", "content": "Recent findings: x"}]
        agent._last_step_summary = {"level_transition": True}
        reply = {"choices": [{"message": {"content": "Action model: UP moves 1\nCross-level notes: doors need keys"}}]}
        resp = mock.Mock(); resp.raise_for_status = lambda: None; resp.json = lambda: reply
        with mock.patch("requests.post", return_value=resp):
            agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")
        self.assertEqual(agent._summarized_knowledge["current_plan"], "")
        self.assertEqual(agent._summarized_knowledge["action_model"], "UP moves 1")
        self.assertEqual(agent._summarized_knowledge["cross_level_notes"], "doors need keys")

    def test_19_summary_rate_limited(self) -> None:
        agent = self._agent()
        dropped = [{"role": "tool", "tool_call_id": "c", "content": "x" * 700} for _ in range(5)]
        reply = {"choices": [{"message": {"content": "World model: w"}}]}
        resp = mock.Mock(); resp.raise_for_status = lambda: None; resp.json = lambda: reply
        with mock.patch("requests.post", return_value=resp) as post:
            self.assertTrue(tc._summarize_into_notes(agent, dropped, self.agent_mod))
            self.assertFalse(tc._summarize_into_notes(agent, dropped, self.agent_mod))   # too soon
            self.assertEqual(post.call_count, 1)
            os.environ["TP2_SUMMARY_MIN_INTERVAL_S"] = "0"
            self.assertFalse(tc._summarize_into_notes(agent, [{"role": "tool", "content": "short"}], self.agent_mod))
            self.assertTrue(tc._summarize_into_notes(agent, dropped, self.agent_mod))
            self.assertEqual(post.call_count, 2)

    def test_20_reset_available_reads_engine_ids(self) -> None:
        sess = _FakeSession(grid_with({(1, 1): 3}))
        sess.game.current_state.available_actions = [0, 1, 2]
        self.assertTrue(tc._reset_available(sess, self.arcengine))
        sess.game.current_state.available_actions = [1, 2]
        self.assertFalse(tc._reset_available(sess, self.arcengine))

    def test_21_batch_aggregate_keeps_diff(self) -> None:
        r1 = {"executed": True, "action_num": 1, "level": 1, "reward": 0.0, "state": "NOT_FINISHED",
              "board_changed": True, "frame_count": 1, "diff": {"changed": 3, "changed_ex_hud": 3, "bbox": [1, 1, 2, 2],
              "colors_added": [3], "colors_removed": [0]}}
        r2 = dict(r1, action_num=2, diff={"changed": 2, "changed_ex_hud": 1, "bbox": [5, 5, 5, 6], "colors_added": [], "colors_removed": []})
        out = self.agent_mod._aggregate_action_batch_result(
            requested_count=2, executed_results=[r1, r2], blocked_actions=[], last_failed=None,
            valid_actions=["UP"], fallback={})
        self.assertEqual(out["diff"]["changed_ex_hud"], 1)
        self.assertEqual(out["diff"]["batch_changed_ex_hud_total"], 4)

    def test_22_async_summary_merges_in_background(self) -> None:
        import threading as _t
        agent = self._agent()
        dropped = [{"role": "tool", "tool_call_id": "c", "content": "x" * 700} for _ in range(5)]
        reply = {"choices": [{"message": {"content": "World model: async-merged"}}]}
        resp = mock.Mock(); resp.raise_for_status = lambda: None; resp.json = lambda: reply
        done = _t.Event()
        real = tc._summarize_into_notes

        def wrapped(*a, **k):
            try:
                return real(*a, **k)
            finally:
                done.set()

        with mock.patch("requests.post", return_value=resp), mock.patch.object(tc, "_summarize_into_notes", wrapped):
            tp.ON_CUT(agent, dropped)
            self.assertTrue(done.wait(5.0))
        self.assertEqual(agent._summarized_knowledge["world_model"], "async-merged")
        self.assertFalse(getattr(agent, "_tp2_summary_inflight", False))

    def test_18_disabled_is_passthrough(self) -> None:
        os.environ["TP2_ENABLE"] = "0"
        agent = self._agent()
        sess = _FakeSession(grid_with({(1, 1): 3}))
        agent._step_env_callback = types.MethodType(lambda self_, a: None, sess)
        agent._tp2_probe = {"level": 1, "text": "Harness probe", "actions": 1}
        tc._state(sess).since_new = 99
        frame = self.agent_mod.Frame(grid=grid_with({(1, 1): 3}), step=5, level=1)
        text = agent._build_user_prompt(5, valid_actions=["UP"], current_frame=frame)
        self.assertNotIn("Harness probe", text)
        self.assertNotIn("STAGNATION", text)
        self.assertFalse(tc.summary_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
