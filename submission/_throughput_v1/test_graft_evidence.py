"""Offline validation for the evidence-integrity aid (EVID).

Run:  GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference \
          .venv/bin/python submission/_throughput_v1/test_graft_evidence.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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

import graft_evidence as ev  # noqa: E402

_FLAGS = ("EVID_ENABLE", "EVID_MAX_ENTRIES", "EVID_MAX_CHARS", "EVID_TRACE", "EVID_TRACE_MAX",
          "EVID_CONNECTIVITY", "EVID_BG_FRACTION")


def _clear() -> None:
    for name in _FLAGS:
        os.environ.pop(name, None)


def _reset_counters() -> None:
    with ev._LOCK:
        for k in ("diffs_emitted", "level_flags", "chars_added", "traces_emitted", "errors"):
            ev._STATE[k] = 0
        ev._STATE["skips"] = {}
        ev._STATE["per_game"] = {}


def blank(h=12, w=12, fill=0):
    return [[fill] * w for _ in range(h)]


def put(g, r, c, color, h=1, w=1):
    for rr in range(r, r + h):
        for cc in range(c, c + w):
            g[rr][cc] = color
    return g


def tup(g):
    return tuple(tuple(int(v) for v in row) for row in g)


class _FakeSession:
    """The bits of _HarnessGameSession the graft reads: history_entries (live
    list of HistoryEntry), game.game_run.game_id, and a scripted step_env that
    appends one entry per executed action with the next scripted (grid, level)."""

    def __init__(self, runtime_mod, state_path: Path, script: list[tuple[list, int]], *, first: list, game_id="cd82-test"):
        self.runtime_mod = runtime_mod
        self.state_path = state_path
        self.script = list(script)              # (grid after action k, level after action k)
        self.history_entries = [runtime_mod.HistoryEntry(action="", frame=runtime_mod.Frame(grid=tup(first), step=0, level=1))]
        self.game = types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=game_id, state="playing"),
                                          current_state=types.SimpleNamespace(won=False))
        self.write_runtime_state()

    def write_runtime_state(self):
        self.runtime_mod.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame,
                                             history=self.history_entries)

    def step_env(self, arguments):
        acts = arguments.get("actions") or []
        payloads = []
        for i, raw in enumerate(acts, start=1):
            if not self.script:
                break
            grid, level = self.script.pop(0)
            name = str(raw.get("action", "UP")).upper()
            if name == "MOUSE":
                name = f"MOUSE(row={raw.get('row')}, col={raw.get('col')})"
            prev_level = self.history_entries[-1].frame.level
            step = len(self.history_entries)
            self.history_entries.append(self.runtime_mod.HistoryEntry(
                action=name, frame=self.runtime_mod.Frame(grid=tup(grid), step=step, level=level)))
            self.write_runtime_state()
            payloads.append({"executed": True, "action_num": step, "level": level, "score": level - 1, "reward": 0.0,
                             "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"],
                             "board_changed": True, "done": False, "level_completed": level > prev_level,
                             "game_over": False, "run_complete": False, "action_display": name,
                             "batch_index": i, "batch_size": len(acts)})
            if level > prev_level:
                break
        final = dict(payloads[-1])
        final.update({"batched": len(acts) > 1, "requested_count": len(acts), "executed_count": len(payloads),
                      "executed_actions": [p["action_display"] for p in payloads],
                      "stopped_early": len(payloads) < len(acts)})
        if final["level_completed"]:
            final["stop_reason"] = "level_completed"
        return final


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _clear()
        from inference.agent import runtime_state as runtime_mod
        from inference.agent import tool_agent as agent_mod
        cls.runtime_mod = runtime_mod
        cls.agent_mod = agent_mod
        cls.stock = {"run": agent_mod.ToolAgent._run_python_tool}
        os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "test-model")
        cls.install_status = ev.install()

    def setUp(self):
        _clear()
        _reset_counters()
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / "artifacts" / "cd82-test_p0_tool_runtime_state.json"
        self.state_path.parent.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()
        _clear()

    # -- helpers -----------------------------------------------------------
    def _agent(self, sess):
        agent = self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
        agent._ensure_session(self.state_path)
        agent._step_env_callback = sess.step_env
        agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]
        return agent

    def _run(self, agent, code):
        return agent._run_python_tool(self.state_path, {"code": code})

    # 01 ---------------------------------------------------------------------
    def test_01_install(self):
        self.assertEqual(self.install_status, "evidence: OK")
        self.assertTrue(ev._STATE["installed"])
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._run_python_tool, "_evid_stock"))
        self.assertIs(self.agent_mod.ToolAgent._run_python_tool._evid_stock, self.stock["run"])
        self.assertEqual(ev.install(), "evidence: SKIP (already applied)")

    # 02 flag off == stock bytes -------------------------------------------
    def test_02_flag_off_byte_identical(self):
        first = put(blank(), 2, 2, 9, 2, 2)
        after = put(blank(), 2, 5, 9, 2, 2)
        os.environ["EVID_ENABLE"] = "0"
        sess = _FakeSession(self.runtime_mod, self.state_path, [(after, 1)], first=first)
        agent = self._agent(sess)
        wrapped = self._run(agent, "r = action(['RIGHT'])\nprint('moved', r['board_changed'])")
        # stock on an identical fresh session
        sess2 = _FakeSession(self.runtime_mod, self.state_path, [(after, 1)], first=first)
        agent2 = self._agent(sess2)
        stock = self.stock["run"](agent2, self.state_path, {"code": "r = action(['RIGHT'])\nprint('moved', r['board_changed'])"})
        self.assertEqual(wrapped.content, stock.content)
        self.assertTrue(wrapped.step_executed and stock.step_executed)
        self.assertNotIn(ev.MARKER, wrapped.content)
        self.assertEqual(ev.status()["diffs_emitted"], 0)

    # 03 diff kinds on synthetic grids ---------------------------------------
    def test_03_diff_kinds(self):
        before = blank()
        put(before, 2, 2, 9, 2, 2)      # blue 2x2 -> moves right by 3
        put(before, 8, 1, 11, 1, 2)     # yellow 1x2 -> vanishes
        put(before, 6, 6, 2, 3, 3)      # gray 3x3 -> recolored to yellow
        after = blank()
        put(after, 2, 5, 9, 2, 2)
        put(after, 6, 6, 11, 3, 3)
        put(after, 0, 11, 8)            # red single cell appears at the corner (edge)
        d = ev.diff_grids(before, after, conn=4)
        kinds = {e["kind"]: e for e in d["entries"]}
        self.assertEqual(set(kinds), {"MOVED", "APPEARED", "VANISHED", "RECOLORED"})
        self.assertEqual(kinds["MOVED"]["bbox"], (2, 2, 3, 3))
        self.assertEqual(kinds["MOVED"]["bbox2"], (2, 5, 3, 6))
        self.assertEqual(kinds["MOVED"]["color"], 9)
        self.assertEqual(kinds["VANISHED"]["bbox"], (8, 1, 8, 2))
        self.assertEqual(kinds["RECOLORED"]["bbox"], (6, 6, 8, 8))
        self.assertEqual((kinds["RECOLORED"]["color"], kinds["RECOLORED"]["color2"]), (2, 11))
        self.assertEqual(kinds["APPEARED"]["bbox"], (0, 11, 0, 11))
        self.assertTrue(kinds["APPEARED"]["edge"])
        self.assertEqual(d["changed_cells"], 4 + 4 + 2 + 9 + 1)
        text = "\n".join(ev.render_entry(e) for e in d["entries"])
        self.assertIn("MOVED b/blue size 4: (2,2)-(3,3) -> (2,5)-(3,6) (d row +0, col +3)", text)
        self.assertIn("APPEARED R/red size 1 at (0,11) [edge]", text)
        self.assertIn("VANISHED Y/yellow size 2 at (8,1)-(8,2)", text)
        self.assertIn("RECOLORED size 9 at (6,6)-(8,8): g/gray -> Y/yellow", text)
        # the background (white, > 25 % of the grid) is never listed even though its cells changed
        self.assertFalse(any(e["color"] == 0 for e in d["entries"]))
        # no change
        self.assertEqual(ev.diff_grids(before, before)["changed_cells"], 0)
        # reshaped HUD bar at the edge (same color, overlapping, shorter)
        b2 = put(blank(), 0, 0, 8, 1, 10)
        a2 = put(blank(), 0, 0, 8, 1, 8)
        d2 = ev.diff_grids(b2, a2)
        self.assertEqual([e["kind"] for e in d2["entries"]], ["RESHAPED"])
        self.assertIn("RESHAPED R/red size 10->8 at (0,0)-(0,9) -> (0,0)-(0,7) [edge]", ev.render_entry(d2["entries"][0]))
        # 8-connectivity joins a diagonal pair into one object; 4 does not
        b3 = blank()
        b3[3][3] = 9
        b3[4][4] = 9
        self.assertEqual(sum(1 for c in ev.components(b3, 8) if c.color == 9), 1)
        self.assertEqual(sum(1 for c in ev.components(b3, 4) if c.color == 9), 2)

    # 04 coordinate convention == the ASCII view the model reads ---------------
    def test_04_coordinate_convention(self):
        from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii
        r, c = 2, 7
        before = blank(8, 10)
        after = put(blank(8, 10), r, c, 9)
        frame = self.runtime_mod.Frame(grid=tup(after), step=1, level=1)
        lines = frame.ascii.splitlines()
        self.assertEqual(format_grid_ascii(tup(after)), frame.ascii)
        self.assertEqual(lines[r][c], ARC_COLOR_CHARS[9])           # row = line index, col = char index
        self.assertEqual(sum(ch == ARC_COLOR_CHARS[9] for ln in lines for ch in ln), 1)
        d = ev.diff_grids(before, after)
        self.assertEqual(ev.render_entry(d["entries"][0]), "APPEARED b/blue size 1 at (2,7)")
        # the solver maps MOUSE row -> engine y, col -> engine x (solver.py _model_mouse_action_data)
        from inference.framework import solver as solver_mod
        self.assertEqual(solver_mod._model_mouse_action_data({"x": c, "y": r}), {"row": r, "col": c})
        self.assertEqual(ev.ARC_COLOR_CHARS, ARC_COLOR_CHARS)
        text, _ = ev.build_evidence([before, after], ["MOUSE(row=2, col=7)"], [1, 1], steps=[1])
        self.assertIn("coords are (row,col), 0-based: row = line index of `.ascii` from the top", text)

    # 05 batch trace ---------------------------------------------------------
    def test_05_batch_trace(self):
        f0 = put(blank(), 5, 1, 9, 2, 2)
        f1 = put(blank(), 5, 3, 9, 2, 2)
        f2 = put(blank(), 5, 5, 9, 2, 2)
        f3 = put(blank(), 5, 5, 9, 2, 2)              # no change
        f4 = put(put(blank(), 5, 7, 9, 2, 2), 0, 0, 8, 1, 3)   # mover + a bar appears: still one mover
        text, info = ev.build_evidence([f0, f1, f2, f3, f4], ["RIGHT", "RIGHT", "UP", "RIGHT"], [1] * 5, steps=[1, 2, 3, 4])
        self.assertTrue(info["trace"])
        self.assertIn("TRACE per action: 1 RIGHT: mover b/blue size 4 -> (5,3)-(6,4) | 2 RIGHT: mover b/blue size 4 -> "
                      "(5,5)-(6,6) | 3 UP: no change | 4 RIGHT: mover b/blue size 4 -> (5,7)-(6,8)", text)
        self.assertIn("diff, before action 1 -> after action 4:", text)
        self.assertIn("MOVED b/blue size 4: (5,1)-(6,2) -> (5,7)-(6,8) (d row +0, col +6)", text)
        # no single mover -> cells changed
        g0 = put(put(blank(), 2, 2, 9), 8, 8, 9)
        g1 = put(put(blank(), 2, 3, 9), 8, 9, 9)
        text2, _ = ev.build_evidence([g0, g1, g1], ["RIGHT", "UP"], [1, 1, 1])
        self.assertIn("1 RIGHT: 4 cells changed | 2 UP: no change", text2)
        # single action: no trace
        text3, info3 = ev.build_evidence([f0, f1], ["RIGHT"], [1, 1], steps=[7])
        self.assertNotIn("TRACE", text3)
        self.assertNotIn("trace", info3)
        self.assertIn("for action 7 (1 executed in this call)", text3)
        # EVID_TRACE=0 disables it; EVID_TRACE_MAX caps it
        os.environ["EVID_TRACE"] = "0"
        self.assertNotIn("TRACE", ev.build_evidence([f0, f1, f2], ["RIGHT", "RIGHT"], [1, 1, 1])[0])
        os.environ["EVID_TRACE"] = "1"
        os.environ["EVID_TRACE_MAX"] = "2"
        cut, _ = ev.build_evidence([f0, f1, f2, f3, f4], ["A", "B", "C", "D"], [1] * 5)
        self.assertIn("| (+1 more actions not traced) | 4 D: mover b/blue size 4 -> (5,7)-(6,8)", cut)   # the FINAL action is always traced
        self.assertNotIn("3 C:", cut)

    # 06 level-transition flag + split ----------------------------------------
    def test_06_level_transition_flag(self):
        l1a = put(blank(), 5, 1, 9, 2, 2)
        l1b = put(blank(), 5, 3, 9, 2, 2)             # action 1 moves the blue block
        l2a = put(put(blank(), 0, 0, 3, 12, 1), 8, 8, 14, 2, 2)   # action 2 clears: L2 board (dark-gray column + green 2x2)
        l2b = put(put(blank(), 0, 0, 3, 12, 1), 8, 6, 14, 2, 2)   # action 3 moves the green block on L2
        text, info = ev.build_evidence([l1a, l1b, l2a, l2b], ["RIGHT", "SPACE", "LEFT"], [1, 1, 2, 2], steps=[10, 11, 12])
        self.assertEqual(info["level_flags"], 1)
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("[EVID] harness object diff for actions 10-12"))
        self.assertEqual(lines[1], "LEVEL CLEARED after action 2 (SPACE) — the frames after it belong to the NEXT level "
                                   "(level 2); do not diff them against this level. The completed board of the old level is "
                                   "never returned — the last frame you saw before the clearing action is NOT the completion "
                                   "state; do not read it as one.")
        self.assertIn("level 1 diff, before action 1 -> after action 1: 8 cells changed, 1 object changes", text)
        self.assertIn("MOVED b/blue size 4: (5,1)-(6,2) -> (5,3)-(6,4)", text)
        self.assertIn("level 2 start frame (after action 2): 2 non-background objects", text)
        self.assertIn("level 2 diff, before action 3 -> after action 3: 8 cells changed, 1 object changes", text)
        self.assertIn("MOVED N/light green size 4: (8,8)-(9,9) -> (8,6)-(9,7)", text)
        # the cross-level pair (L1 frame vs L2 frame) is never diffed
        self.assertNotIn("VANISHED b/blue", text)
        self.assertNotIn("APPEARED G/dark gray", text)
        self.assertIn("TRACE per action: 1 RIGHT: mover b/blue size 4 -> (5,3)-(6,4) | 2 SPACE: LEVEL CLEARED "
                      "(frame now level 2) | 3 LEFT: mover N/light green size 4 -> (8,6)-(9,7)", text)
        # clear on the first action of the call
        text2, info2 = ev.build_evidence([l1b, l2a, l2b], ["SPACE", "LEFT"], [1, 2, 2])
        self.assertEqual(info2["level_flags"], 1)
        self.assertIn("LEVEL CLEARED after action 1 (SPACE)", text2)
        self.assertIn("level 1: the clearing action was the first of this call; nothing to diff on this level", text2)
        # clear on the last action: nothing after it
        text3, info3 = ev.build_evidence([l1a, l1b, l2a], ["RIGHT", "SPACE"], [1, 1, 2])
        self.assertEqual(info3["level_flags"], 1)
        self.assertIn("level 2 start frame (after action 2): 2 non-background objects", text3)
        self.assertNotIn("level 2 diff", text3)
        # no clear: no flag
        self.assertEqual(ev.build_evidence([l1a, l1b], ["RIGHT"], [1, 1])[1]["level_flags"], 0)

    # 07 caps ---------------------------------------------------------------
    def test_07_caps(self):
        before = blank(30, 30)
        after = blank(30, 30)
        for i in range(30):                           # 30 single cells appear (distinct, non-adjacent)
            after[i][(i * 7) % 30] = 8 + (i % 3)
        text, info = ev.build_evidence([before, after], ["UP"], [1, 1], entry_cap=5)
        self.assertEqual(info["entries_total"], 30)
        self.assertEqual(info["entries_shown"], 5)
        self.assertIn("(+25 more not shown)", text)
        os.environ["EVID_MAX_ENTRIES"] = "7"
        self.assertIn("(+23 more not shown)", ev.build_evidence([before, after], ["UP"], [1, 1])[0])
        os.environ["EVID_MAX_ENTRIES"] = "40"
        os.environ["EVID_MAX_CHARS"] = "400"
        text2, info2 = ev.build_evidence([before, after], ["UP"], [1, 1])
        self.assertLessEqual(len(text2), 400)
        self.assertTrue(info2["truncated"])
        self.assertTrue(text2.startswith(ev.MARKER))
        self.assertIn("lines cut by EVID_MAX_CHARS", text2)
        # the level flag survives the char cap
        os.environ["EVID_MAX_CHARS"] = "600"
        text3, _ = ev.build_evidence([before, after, after], ["UP", "SPACE"], [1, 1, 2])
        self.assertLessEqual(len(text3), 600)
        self.assertIn("LEVEL CLEARED after action 2 (SPACE)", text3)
        # defaults
        _clear()
        self.assertEqual((ev.max_entries(), ev.max_chars(), ev.trace_enabled(), ev.trace_max()), (40, 1500, True, 24))
        big, _ = ev.build_evidence([before, after], ["UP"], [1, 1])
        self.assertLessEqual(len(big), 1500)

    # 08 the wrapper on the harness path: JSON stdout carries the block ----------
    def test_08_wrapper_injects_into_stdout(self):
        first = put(blank(), 2, 2, 9, 2, 2)
        a1 = put(blank(), 2, 3, 9, 2, 2)
        a2 = put(blank(), 2, 4, 9, 2, 2)
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1), (a2, 1)], first=first)
        agent = self._agent(sess)
        res = self._run(agent, "r = action(['RIGHT', 'RIGHT'])\nprint('n', r['executed_count'])")
        self.assertTrue(res.step_executed)
        payload = json.loads(res.content)                      # still the stock JSON shape
        self.assertEqual(payload["tool"], "python")
        self.assertTrue(payload["stdout"].startswith("n 2\n\n[EVID] harness object diff for actions 1-2 (2 executed"))
        self.assertIn("MOVED b/blue size 4: (2,2)-(3,3) -> (2,4)-(3,5) (d row +0, col +2)", payload["stdout"])
        self.assertIn("TRACE per action: 1 RIGHT: mover b/blue size 4 -> (2,3)-(3,4) | 2 RIGHT: mover b/blue size 4 -> (2,4)-(3,5)",
                      payload["stdout"])
        # the transcript renderer shows stdout -> the marker lands in [TOOL RESULT: python]
        self.assertIn("[EVID]", self.agent_mod._render_tool_result_display(res.content))
        st = ev.status()
        self.assertEqual((st["diffs_emitted"], st["level_flags"], st["traces_emitted"]), (1, 0, 1))
        self.assertEqual(st["chars_added"], len(res.content) - len(json.dumps({"tool": "python", "returncode": 0, "stdout": "n 2\n"}, indent=2)))
        self.assertEqual(st["per_game"]["cd82-test"]["diffs"], 1)
        # snippet with no print: stdout is created; result key kept
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1)], first=first)
        agent = self._agent(sess)
        res2 = self._run(agent, "action(['RIGHT'])")
        p2 = json.loads(res2.content)
        self.assertTrue(p2["stdout"].startswith("[EVID]"))
        self.assertIn("result", p2)
        # inspection-only call (no action): untouched
        res3 = self._run(agent, "print(len(history))")
        self.assertFalse(res3.step_executed)
        self.assertNotIn("[EVID]", res3.content)
        self.assertEqual(ev.status()["diffs_emitted"], 2)

    # 09 level flag through the wrapper + GAME_OVER note ---------------------------
    def test_09_wrapper_level_flag(self):
        first = put(blank(), 2, 2, 9, 2, 2)
        a1 = put(blank(), 2, 3, 9, 2, 2)
        l2 = put(blank(), 9, 9, 14, 2, 2)
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1), (l2, 2), (l2, 2)], first=first)
        agent = self._agent(sess)
        res = self._run(agent, "action(['RIGHT', 'SPACE', 'LEFT'])")
        out = json.loads(res.content)["stdout"]
        self.assertIn("LEVEL CLEARED after action 2 (SPACE) — the frames after it belong to the NEXT level (level 2)", out)
        self.assertIn("level 2 start frame (after action 2): 1 non-background objects", out)
        self.assertEqual(ev.status()["level_flags"], 1)
        self.assertEqual(ev.status()["per_game"]["cd82-test"]["level_flags"], 1)
        # the stock summary still says level transition (we did not touch it)
        self.assertTrue(agent._last_step_summary["level_transition"])
        # GAME_OVER note from the stock last_action_result
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1)], first=first)
        agent = self._agent(sess)
        inner = sess.step_env

        def go(args):
            p = inner(args)
            p["game_over"] = True
            p["state"] = "GAME_OVER"
            return p
        agent._step_env_callback = go
        agent._step_env_callback = types.MethodType(lambda self_, args: go(args), sess)
        res = self._run(agent, "action(['RIGHT'])")
        self.assertIn("GAME_OVER on the last action — the harness auto-resets this level", json.loads(res.content)["stdout"])

    # 10 exception fallback --------------------------------------------------
    def test_10_exception_fallback(self):
        first = put(blank(), 2, 2, 9, 2, 2)
        a1 = put(blank(), 2, 3, 9, 2, 2)
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1)], first=first)
        agent = self._agent(sess)
        with mock.patch.object(ev, "build_evidence", side_effect=RuntimeError("boom")):
            res = self._run(agent, "action(['RIGHT'])\nprint('ok')")
        self.assertTrue(res.step_executed)
        self.assertNotIn("[EVID]", res.content)
        self.assertEqual(json.loads(res.content)["stdout"], "ok\n")
        self.assertEqual(ev.status()["errors"], 1)
        # a broken history reader before the call also falls back to stock
        with mock.patch.object(ev, "_history", side_effect=RuntimeError("boom")):
            sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1)], first=first)
            res = self._run(self._agent(sess), "action(['RIGHT'])\nprint('ok2')")
        self.assertEqual(json.loads(res.content)["stdout"], "ok2\n")

    # 11 runtime-state fallback when no session is reachable --------------------
    def test_11_runtime_state_fallback(self):
        first = put(blank(), 2, 2, 9, 2, 2)
        a1 = put(blank(), 2, 3, 9, 2, 2)
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1)], first=first)
        agent = self._agent(sess)
        agent._step_env_callback = lambda args: sess.step_env(args)     # plain function: no __self__
        res = self._run(agent, "action(['RIGHT'])")
        self.assertIn("MOVED b/blue size 4: (2,2)-(3,3) -> (2,3)-(3,4)", json.loads(res.content)["stdout"])

    # 12 inject() shapes ------------------------------------------------------
    def test_12_inject(self):
        self.assertEqual(json.loads(ev.inject(json.dumps({"tool": "python", "stdout": "x\n"}), "[EVID] y"))["stdout"], "x\n\n[EVID] y")
        self.assertEqual(json.loads(ev.inject(json.dumps({"tool": "python", "result": 1}), "[EVID] y"))["stdout"], "[EVID] y")
        self.assertEqual(ev.inject("not json", "[EVID] y"), "not json\n\n[EVID] y")
        self.assertEqual(ev.inject(json.dumps([1, 2]), "[EVID] y"), "[1, 2]\n\n[EVID] y")

    # 13 status shape ---------------------------------------------------------
    def test_13_status(self):
        st = ev.status()
        for k in ("installed", "enabled", "max_entries", "max_chars", "trace", "connectivity", "bg_fraction",
                  "diffs_emitted", "level_flags", "chars_added", "traces_emitted", "errors", "skips", "per_game"):
            self.assertIn(k, st)
        os.environ["EVID_CONNECTIVITY"] = "8"
        os.environ["EVID_BG_FRACTION"] = "0.5"
        self.assertEqual((ev.connectivity(), ev.bg_fraction()), (8, 0.5))

    # 14 end to end through analyze(): the block reaches the transcript and the kept history ----
    def test_14_analyze_transcript_and_history(self):
        first = put(blank(), 2, 2, 9, 2, 2)
        a1 = put(blank(), 2, 3, 9, 2, 2)
        sess = _FakeSession(self.runtime_mod, self.state_path, [(a1, 1)], first=first)
        agent = self.agent_mod.ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
        replies = iter([
            self.agent_mod._ChatCompletionResult(message={"role": "assistant", "content": "Plan: probe.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "action(['RIGHT'])\nprint('p')"})}}]},
                finish_reason="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ])
        transcript = Path(self.tmp.name) / "transcripts" / "cd82-test_p0.txt"
        transcript.parent.mkdir(parents=True)
        with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", side_effect=lambda *a, **k: next(replies)):
            result = agent.analyze(self.state_path, 0, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"], step_env=sess.step_env,
                                   transcript_path=transcript, analysis_step=1)
        self.assertTrue(result.step_executed)
        text = transcript.read_text()
        self.assertEqual(text.count("[EVID] harness object diff"), 1)
        self.assertIn("[TOOL RESULT: python]\np\n\n[EVID] harness object diff for action 1 (1 executed in this call)", text)
        tool_msgs = [m for m in agent._history_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("[EVID]", tool_msgs[0]["content"])           # persisted for later turns

    # 15 judge fix: in-place overlap BEFORE the shape matcher ---------------------------
    def test_15_in_place_before_moved(self):
        b = put(put(blank(), 2, 2, 9, 2, 2), 8, 8, 9, 2, 2)              # T1 at (2,2), T2 at (8,8), same shape+color
        a = put(put(put(blank(), 2, 2, 9, 2, 2), 8, 1, 9, 2, 2), 3, 3, 8)  # T1 stays but one cell is occluded; T2 moved
        d = ev.diff_grids(b, a)
        moved = [e for e in d["entries"] if e["kind"] == "MOVED"]
        self.assertEqual(len(moved), 1)
        self.assertEqual((moved[0]["bbox"], moved[0]["bbox2"]), ((8, 8, 9, 9), (8, 1, 9, 2)))   # T2, not T1
        self.assertFalse(any(e["kind"] == "MOVED" and e["bbox"] == (2, 2, 3, 3) for e in d["entries"]))
        self.assertIn("APPEARED R/red size 1 at (3,3)", "\n".join(ev.render_entry(e) for e in d["entries"]))
        # a big tile stepping one cell still overlaps itself (Jaccard 0.6): in-place pairing labels it MOVED
        b2 = put(blank(), 4, 4, 9, 4, 4)
        a2 = put(blank(), 4, 5, 9, 4, 4)
        d2 = ev.diff_grids(b2, a2)
        self.assertEqual([(e["kind"], e["bbox2"]) for e in d2["entries"]], [("MOVED", (4, 5, 7, 8))])
        # the uncovered tile: an identical tile appears elsewhere while this one is partly uncovered -> no swap
        b3 = put(put(blank(), 2, 2, 9, 2, 2), 3, 3, 8)                    # T1 at (2,2) with its corner covered by red
        a3 = put(put(blank(), 2, 2, 9, 2, 2), 8, 8, 9, 2, 2)              # red gone (T1 whole), new identical tile at (8,8)
        d3 = ev.diff_grids(b3, a3)
        self.assertEqual([e["kind"] for e in d3["entries"]], ["APPEARED", "VANISHED"])
        self.assertEqual([e["bbox"] for e in d3["entries"]], [(8, 8, 9, 9), (3, 3, 3, 3)])

    # 16 judge fix verified on the recorded frames (keith run, dc22 action 35, lf52 action 22) ----------
    def test_16_recorded_frames(self):
        fix = json.loads((_HERE / "fixtures_evid_frames.json").read_text())
        d = ev.diff_grids(fix["dc22-fdcac232"]["before"], fix["dc22-fdcac232"]["after"])
        lines, hidden = ev.render_entries(d["entries"], 40)
        self.assertEqual(lines, ["MOVED N/light green size 4: (38,8)-(39,9) -> (40,8)-(41,9) (d row +2, col +0)"])
        self.assertEqual(hidden, 0)                                    # the 32->32 same-bbox frame reshape is noise, dropped
        d = ev.diff_grids(fix["lf52-271a04aa"]["before"], fix["lf52-271a04aa"]["after"])
        lines, _ = ev.render_entries(d["entries"], 40)
        self.assertFalse(any(ln.startswith("MOVED") for ln in lines))    # nothing moved: recolors + appear/vanish only
        self.assertIn("APPEARED N/light green size 12 at (15,13)-(18,16)", lines)
        self.assertIn("VANISHED w/light gray size 16 at (15,13)-(18,16)", lines)
        self.assertIn("RECOLORED size 12 at (15,31)-(18,34): g/gray -> N/light green", lines)
        self.assertEqual([ln for ln in lines if ln.startswith("HUD/edge bars reshaped: 2 (W/white, w/light gray; sizes 56->55, 8->9)")].__len__(), 1)
        self.assertFalse(any("[edge]" in ln and ln.startswith("RESHAPED") for ln in lines))

    # 17 noise rules ------------------------------------------------------------------
    def test_17_noise_rules(self):
        # same bbox, |delta| <= 2 -> dropped; |delta| > 2 or bbox change -> kept
        b = put(blank(), 4, 4, 2, 4, 4)
        a = put(put(blank(), 4, 4, 2, 4, 4), 5, 5, 9, 2, 1)             # two interior cells recolored: frame 16->14, same bbox
        kinds = [e["kind"] for e in ev.diff_grids(b, a)["entries"]]
        self.assertEqual(kinds, ["APPEARED"])
        a2 = put(put(blank(), 4, 4, 2, 4, 4), 5, 5, 9, 2, 2)            # 16->12: kept
        self.assertIn("RESHAPED", [e["kind"] for e in ev.diff_grids(b, a2)["entries"]])
        # edge bars collapse to one line and never count against the entry cap
        b3 = put(put(blank(20, 20), 0, 0, 8, 1, 15), 19, 0, 11, 1, 15)
        a3 = put(put(blank(20, 20), 0, 0, 8, 1, 12), 19, 0, 11, 1, 10)
        lines, hidden = ev.render_entries(ev.diff_grids(b3, a3)["entries"], 1)
        self.assertEqual(hidden, 0)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("HUD/edge bars reshaped: 2 (R/red, Y/yellow; sizes 15->12, 15->10)"))
        text, _ = ev.build_evidence([b3, a3], ["UP"], [1, 1])
        self.assertIn("HUD/edge bars reshaped: 2", text)
        self.assertNotIn("[edge]", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
