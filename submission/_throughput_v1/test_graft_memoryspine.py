"""Offline validation for the memory-spine graft (TP10).

Installs graft_deaths FIRST, then graft_memoryspine, so every test runs with
TP10 stacked on top of an already-wrapped _build_user_prompt.

Run:  .venv/bin/python submission/_throughput_v1/test_graft_memoryspine.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(os.environ.get(
    "GRAFT_TEST_BUNDLE",
    str(_HERE.parents[1] / "submission/_inspect_replay/assets_build/ARC3-Inference"),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_deaths as t7  # noqa: E402
import graft_memoryspine as t10  # noqa: E402

_FLAGS = ("TP10_ENABLE", "TP10_NOTES", "TP10_ECHO", "TP10_ACCOUNTING", "TP10_NOTES_CAP", "TP7_ENABLE")


def _clear() -> None:
    for name in _FLAGS:
        os.environ.pop(name, None)


class MemorySpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.agent_mod = agent_mod
        cls.deaths_status = t7.install()      # wrap _build_user_prompt first
        cls.status = t10.install()            # TP10 stacks on top

    def setUp(self) -> None:
        _clear()
        t10._NOTES.clear()

    def _agent(self, state_path: str | None = None):
        agent = self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")
        if state_path is not None:
            agent._ensure_session(Path(state_path))
        return agent

    # ------------------------------------------------------------------ 01
    def test_01_install_and_stacking_chain(self) -> None:
        self.assertIn(self.deaths_status, {"deaths: OK", "deaths: SKIP (already applied)"})
        self.assertIn(self.status, {"memoryspine: OK", "memoryspine: SKIP (already applied)"})
        top = self.agent_mod.ToolAgent._build_user_prompt
        self.assertTrue(hasattr(top, "_tp10_stock"))
        # the stock TP10 captured must be the deaths wrapper -> stacking chain intact
        self.assertTrue(hasattr(t10._STOCK["build_user_prompt"], "_tp7_stock"))
        for name in ("update_from_assistant", "update_from_step_summary",
                     "ensure_session", "summarize_step_sequence"):
            self.assertIn(name, t10._STOCK)

    # ------------------------------------------------------------------ 02
    def test_02_summary_carries_committed_and_state(self) -> None:
        agent = self._agent()
        items = [
            {"executed": True, "action_num": 8, "level": 2, "state": "NOT_FINISHED",
             "requested_count": 5, "executed_count": 3, "stopped_early": True,
             "stop_reason": "known_noop",
             "executed_actions": ["UP", "UP", "DOWN"]},
        ]
        summary = agent._summarize_step_sequence(items)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["executed_count"], 3)
        self.assertEqual(summary["tp10_committed"], 5)
        self.assertEqual(summary["tp10_state"], "NOT_FINISHED")

    # ------------------------------------------------------------------ 03
    def test_03_accounting_line_in_prompt(self) -> None:
        agent = self._agent("/tmp/tp10run/g03aaaa_p0_tool_runtime_state.json")
        items = [
            {"executed": True, "action_num": 8, "level": 2, "state": "NOT_FINISHED",
             "requested_count": 5, "executed_count": 3, "stopped_early": True,
             "stop_reason": "known_noop",
             "executed_actions": ["UP", "UP", "DOWN"]},
        ]
        agent._last_step_summary = agent._summarize_step_sequence(items)
        text = agent._build_user_prompt(8, valid_actions=["UP"])
        self.assertIn(
            "LAST TURN: committed 5 action(s), 3 executed, ended level=2, state=NOT_FINISHED.",
            text,
        )
        self.assertIn("2 committed action(s) were dropped before execution (stop_reason=known_noop).", text)
        # the accounting line leads the prompt
        self.assertTrue(text.startswith("LAST TURN:"))

    # ------------------------------------------------------------------ 04
    def test_04_accounting_no_drop_no_drop_sentence(self) -> None:
        agent = self._agent("/tmp/tp10run/g04aaaa_p0_tool_runtime_state.json")
        items = [{"executed": True, "action_num": 3, "level": 1, "state": "NOT_FINISHED",
                  "executed_actions": ["UP"]}]
        agent._last_step_summary = agent._summarize_step_sequence(items)
        text = agent._build_user_prompt(3, valid_actions=["UP"])
        self.assertIn("LAST TURN: committed 1 action(s), 1 executed, ended level=1, state=NOT_FINISHED.", text)
        self.assertNotIn("were dropped", text)

    # ------------------------------------------------------------------ 05
    def test_05_suggestion_harvest_and_one_shot_echo(self) -> None:
        agent = self._agent("/tmp/tp10run/g05aaaa_p0_tool_runtime_state.json")
        agent._update_summarized_knowledge_from_assistant(
            "World model: keys open doors\n- Next: try the blue door with UP, then MOUSE(3,4)"
        )
        text = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertIn("YOUR PRIOR INTENT", text)
        self.assertIn("try the blue door with UP, then MOUSE(3,4)", text)
        self.assertIn("echoed back to you verbatim", text)  # the how-to line
        self.assertEqual(text.count("try the blue door with UP, then MOUSE(3,4)"), 1)
        # consumed after one echo — J10-F1: the intent text itself must be
        # GONE on turn 2, not just the header (no gluing into stock fields)
        text2 = agent._build_user_prompt(2, valid_actions=["UP"])
        self.assertNotIn("YOUR PRIOR INTENT", text2)
        self.assertEqual(text2.count("try the blue door with UP, then MOUSE(3,4)"), 0)

    def test_05b_harvest_intent_parsing(self) -> None:
        self.assertEqual(t10.harvest_intent("Suggestion: probe the lever"), "probe the lever")
        self.assertEqual(t10.harvest_intent("Next: A\nblah\nNext: B"), "B")  # last one wins
        self.assertIsNone(t10.harvest_intent("Next test: not a note-to-self"))
        self.assertIsNone(t10.harvest_intent("no labels here"))
        self.assertIsNone(t10.harvest_intent(""))
        # split removes every intent line, keeps the rest verbatim
        stripped, intent = t10.split_intent("Plan: go up\nNext: A\nmore plan text\nSuggestion: B")
        self.assertEqual(intent, "B")
        self.assertEqual(stripped, "Plan: go up\nmore plan text")
        stripped, intent = t10.split_intent("no note here")
        self.assertIsNone(intent)
        self.assertEqual(stripped, "no note here")

    def test_05c_intent_never_glued_into_stock_fields(self) -> None:
        """J10-F1 regression: `Next:` inside a labeled-block context must not
        be absorbed into current_plan by _extract_labeled_blocks."""
        agent = self._agent("/tmp/tp10run/g05caaa_p0_tool_runtime_state.json")
        agent._update_summarized_knowledge_from_assistant(
            "World model: keys open doors\nPlan: hug the left wall\nNext: try the blue door with UP"
        )
        self.assertEqual(agent._summarized_knowledge["current_plan"], "hug the left wall")
        for field, value in agent._summarized_knowledge.items():
            self.assertNotIn("try the blue door", value, f"intent leaked into {field}")
        turn1 = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertEqual(turn1.count("try the blue door"), 1)   # echo only, once
        turn2 = agent._build_user_prompt(2, valid_actions=["UP"])
        self.assertEqual(turn2.count("try the blue door"), 0)   # gone for good

    # ------------------------------------------------------------------ 06
    def test_06_notes_persist_across_game_over_wipe(self) -> None:
        agent = self._agent("/tmp/tp10run/g06aaaa_p0_tool_runtime_state.json")
        agent._update_summarized_knowledge_from_assistant(
            "World model: spikes kill on touch\nPlan: hug the left wall"
        )
        # while the live block still carries the notes, TP10 adds no duplicate block
        text_live = agent._build_user_prompt(3, valid_actions=["UP"])
        self.assertNotIn("YOUR NOTES (persistent):", text_live)
        # simulated GAME_OVER turn -> stock wipes the live knowledge
        items = [{"executed": True, "action_num": 4, "level": 1, "state": "GAME_OVER",
                  "game_over": True, "executed_actions": ["UP"]}]
        agent._last_step_summary = agent._summarize_step_sequence(items)
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")  # stock wipe fired
        text = agent._build_user_prompt(5, valid_actions=["UP"])
        self.assertIn("YOUR NOTES (persistent):", text)
        self.assertIn("spikes kill on touch", text)
        self.assertIn("hug the left wall", text)

    # ------------------------------------------------------------------ 07
    def test_07_notes_persist_across_agent_instances_same_game(self) -> None:
        path_p0 = "/tmp/tp10run/g07aaaa_p0_tool_runtime_state.json"
        path_p1 = "/tmp/tp10run/g07aaaa_p1_tool_runtime_state.json"
        agent = self._agent(path_p0)
        agent._update_summarized_knowledge_from_assistant("Cross-level notes: level 1 exit is top-right")
        agent._update_summarized_knowledge_from_assistant("World model: two-phase toggle panel")
        # a NEW agent (new pass of the same game) starts with empty live knowledge
        fresh = self._agent(path_p1)
        self.assertEqual(t10.game_key_of(agent), t10.game_key_of(fresh))
        text = fresh._build_user_prompt(0, valid_actions=["UP"])
        self.assertIn("YOUR NOTES (persistent):", text)
        self.assertIn("two-phase toggle panel", text)
        self.assertIn("level 1 exit is top-right", text)
        # a different game shares nothing
        other = self._agent("/tmp/tp10run/g07bbbb_p0_tool_runtime_state.json")
        text_other = other._build_user_prompt(0, valid_actions=["UP"])
        self.assertNotIn("two-phase toggle panel", text_other)

    # ------------------------------------------------------------------ 08
    def test_08_level_transition_wipe_snapshots_previous_level(self) -> None:
        agent = self._agent("/tmp/tp10run/g08aaaa_p0_tool_runtime_state.json")
        agent._last_step_summary = {"level": 1}
        agent._update_summarized_knowledge_from_assistant("World model: L1 is a maze; walls are B")
        items = [{"executed": True, "action_num": 9, "level": 2, "state": "NOT_FINISHED",
                  "level_completed": True, "executed_actions": ["UP"]}]
        agent._last_step_summary = agent._summarize_step_sequence(items)
        self.assertTrue(agent._last_step_summary["level_transition"])
        agent._update_summarized_knowledge_from_step_summary()   # stock wipes on level up
        self.assertEqual(agent._summarized_knowledge["world_model"], "")
        text = agent._build_user_prompt(10, valid_actions=["UP"])
        self.assertIn("[L1] World model: L1 is a maze; walls are B", text)

    # ------------------------------------------------------------------ 09
    def test_09_reinjection_cap_keeps_newest_tail(self) -> None:
        os.environ["TP10_NOTES_CAP"] = "200"
        agent = self._agent("/tmp/tp10run/g09aaaa_p0_tool_runtime_state.json")
        journal = t10._NOTES.setdefault(t10.game_key_of(agent), [])
        for level in (1, 2, 3):
            journal.append({
                "key": "world_model", "label": "World model", "level": level,
                "text": f"level {level} mechanics " + ("x" * 90),
            })
        text = agent._build_user_prompt(5, valid_actions=["UP"])
        self.assertIn("YOUR NOTES (persistent):", text)
        self.assertIn("(older notes trimmed)", text)
        self.assertIn("level 3 mechanics", text)      # newest kept
        self.assertNotIn("level 1 mechanics", text)   # oldest trimmed
        block = text.split("YOUR NOTES (persistent):", 1)[1]
        note_lines = []
        for line in block.splitlines()[1:]:
            if not line.startswith("- "):
                break
            note_lines.append(line)
        body = "\n".join(note_lines)
        self.assertLessEqual(len(body), 200)          # TP10_NOTES_CAP honored

    # ------------------------------------------------------------------ 10
    def test_10_disabled_master_is_passthrough(self) -> None:
        os.environ["TP10_ENABLE"] = "0"
        agent = self._agent("/tmp/tp10run/g10aaaa_p0_tool_runtime_state.json")
        agent._tp10_intent = "stale"
        t10._NOTES[t10.game_key_of(agent)] = [
            {"key": "world_model", "label": "World model", "level": 1, "text": "hidden"},
        ]
        agent._last_step_summary = {"executed_count": 2, "level": 1, "tp10_committed": 4}
        text = agent._build_user_prompt(2, valid_actions=["UP"])
        self.assertNotIn("LAST TURN:", text)
        self.assertNotIn("YOUR PRIOR INTENT", text)
        self.assertNotIn("YOUR NOTES (persistent):", text)
        self.assertNotIn("note-to-self", text)

    # ------------------------------------------------------------------ 11
    def test_11_individual_toggles(self) -> None:
        base = "/tmp/tp10run/g11aaaa_p0_tool_runtime_state.json"

        os.environ["TP10_NOTES"] = "0"
        agent = self._agent(base)
        agent._update_summarized_knowledge_from_assistant("World model: alpha\nNext: go left")
        agent._summarized_knowledge["world_model"] = ""   # would render if notes were on
        agent._last_step_summary = {"executed_count": 1, "level": 1}
        text = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertNotIn("YOUR NOTES (persistent):", text)
        self.assertIn("YOUR PRIOR INTENT", text)
        self.assertIn("LAST TURN:", text)
        _clear()

        os.environ["TP10_ACCOUNTING"] = "0"
        agent = self._agent("/tmp/tp10run/g11bbbb_p0_tool_runtime_state.json")
        agent._last_step_summary = {"executed_count": 1, "level": 1}
        text = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertNotIn("LAST TURN:", text)
        self.assertIn("note-to-self", text)   # echo how-to still present
        _clear()

        os.environ["TP10_ECHO"] = "0"
        agent = self._agent("/tmp/tp10run/g11cccc_p0_tool_runtime_state.json")
        agent._tp10_intent = "held intent"
        agent._last_step_summary = {"executed_count": 1, "level": 1}
        text = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertNotIn("YOUR PRIOR INTENT", text)
        self.assertNotIn("note-to-self", text)
        self.assertIn("LAST TURN:", text)

    # ------------------------------------------------------------------ 12
    def test_12_stacked_deaths_block_still_renders(self) -> None:
        import types  # noqa: PLC0415
        agent = self._agent("/tmp/tp10run/g12aaaa_p0_tool_runtime_state.json")
        sess = types.SimpleNamespace()
        d = t7._dstate(sess)
        d.level = 1
        d.level_actions = 40
        d.death_at = [27]
        agent._step_env_callback = types.MethodType(lambda self_, a: None, sess)
        agent._last_step_summary = {"executed_count": 2, "level": 1, "tp10_committed": 2}
        text = agent._build_user_prompt(40, valid_actions=["UP"])
        self.assertIn("DEATH PROTOCOL", text)          # inner (deaths) wrapper ran
        self.assertIn("LAST TURN:", text)              # outer (TP10) wrapper ran
        self.assertLess(text.index("LAST TURN:"), text.index("DEATH PROTOCOL"))

    # ------------------------------------------------------------------ F2
    def _run_cap_order_check(self, order: str) -> dict:
        """Fresh process: install TP + TP10 in the given order, cap=3, batch=10."""
        import json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        script = (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "here, bundle, order = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "sys.path.insert(0, here); sys.path.insert(0, bundle)\n"
            "os.environ['TP_BATCH_CAP'] = '3'\n"
            "import graft_throughput as tp\n"
            "import graft_memoryspine as t10\n"
            "mods = [tp, t10] if order == 'tp_first' else [t10, tp]\n"
            "for mod in mods:\n"
            "    assert mod.install().endswith('OK')\n"
            "from inference.agent import tool_agent as am\n"
            "agent = am.ToolAgent(model='m', base_url='http://127.0.0.1:9/v1', provider='vllm')\n"
            "agent._tp10_raw_committed = []\n"
            "tp.begin_tool_call()\n"
            "normalized = agent._normalize_python_actions([{'action': 'UP'}] * 10)\n"
            "items = [{'executed': True, 'action_num': 3, 'level': 1, 'state': 'NOT_FINISHED',\n"
            "          'requested_count': len(normalized), 'executed_count': len(normalized),\n"
            "          'executed_actions': ['UP'] * len(normalized)}]\n"
            "summary = agent._summarize_step_sequence(items)\n"
            "print(json.dumps({'normalized': len(normalized), 'raw': agent._tp10_raw_committed,\n"
            "                  'committed': summary.get('tp10_committed'),\n"
            "                  'cap_dropped': summary.get('tp10_cap_dropped'),\n"
            "                  'line': t10.accounting_line(summary)}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script, str(_HERE), str(_BUNDLE), order],
            capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(proc.returncode, 0, f"[{order}] stderr: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_14_committed_is_preclamp_both_install_orders(self) -> None:
        """J10-F2 regression: with graft_throughput's batch cap active, the
        accounting line must report the model's TRUE batch size, not the
        post-truncation one — regardless of install order."""
        for order in ("tp_first", "tp10_first"):
            out = self._run_cap_order_check(order)
            self.assertEqual(out["normalized"], 3, order)     # cap really fired
            self.assertEqual(out["raw"], [10], order)         # raw batch captured
            self.assertEqual(out["committed"], 10, order)
            self.assertEqual(out["cap_dropped"], 7, order)
            self.assertIn("committed 10 action(s), 3 executed", out["line"], order)
            self.assertIn("7 truncated by the harness batch cap", out["line"], order)

    def test_15_no_cap_no_phantom_clamp(self) -> None:
        # without the raw capture list, committed falls back to requested_count
        agent = self._agent()
        items = [{"executed": True, "action_num": 5, "level": 1, "state": "NOT_FINISHED",
                  "requested_count": 4, "executed_count": 4, "executed_actions": ["UP"] * 4}]
        summary = agent._summarize_step_sequence(items)
        self.assertEqual(summary["tp10_committed"], 4)
        self.assertNotIn("tp10_cap_dropped", summary)

    # ------------------------------------------------------------------ F3
    def test_16_level_none_note_superseded_by_levelled_entry(self) -> None:
        """J10-F3 regression: a refuted pre-first-summary guess must not be
        re-served beside its own correction."""
        agent = self._agent("/tmp/tp10run/g16aaaa_p0_tool_runtime_state.json")
        t10._NOTES[t10.game_key_of(agent)] = [
            {"key": "world_model", "label": "World model", "level": None,
             "text": "WRONG early guess: the goal is the red square"},
            {"key": "world_model", "label": "World model", "level": 1,
             "text": "CORRECTED: the goal is the exit door"},
        ]
        text = agent._build_user_prompt(4, valid_actions=["UP"])
        self.assertIn("CORRECTED: the goal is the exit door", text)
        self.assertNotIn("WRONG early guess", text)
        # but a level-None note that is the NEWEST for its key still renders
        t10._NOTES[t10.game_key_of(agent)].append(
            {"key": "world_model", "label": "World model", "level": None,
             "text": "NEWEST: revised after reset"})
        text2 = agent._build_user_prompt(5, valid_actions=["UP"])
        self.assertIn("NEWEST: revised after reset", text2)

    # ------------------------------------------------------------------ F4
    def test_17_defaults_per_j10_f4(self) -> None:
        self.assertEqual(t10.notes_cap(), 2000)
        self.assertEqual(t10.howto_every(), 8)

    def test_18_howto_cadence_first_nth_and_after_wipe(self) -> None:
        agent = self._agent("/tmp/tp10run/g18aaaa_p0_tool_runtime_state.json")
        fires = ["note-to-self" in agent._build_user_prompt(i, valid_actions=["UP"])
                 for i in range(9)]
        self.assertTrue(fires[0])                 # prompt 1
        self.assertFalse(any(fires[1:8]))         # prompts 2-8 stay clean
        self.assertTrue(fires[8])                 # prompt 9 (every 8th)
        # a knowledge wipe re-arms the how-to on the next prompt
        items = [{"executed": True, "action_num": 20, "level": 1, "state": "GAME_OVER",
                  "game_over": True, "executed_actions": ["UP"]}]
        agent._last_step_summary = agent._summarize_step_sequence(items)
        agent._update_summarized_knowledge_from_step_summary()
        text10 = agent._build_user_prompt(10, valid_actions=["UP"])  # prompt 10: not periodic
        self.assertIn("note-to-self", text10)
        text11 = agent._build_user_prompt(11, valid_actions=["UP"])
        self.assertNotIn("note-to-self", text11)  # wipe flag consumed

    # ------------------------------------------------------------------ 13
    def test_13_fail_open_on_bad_summary(self) -> None:
        agent = self._agent("/tmp/tp10run/g13aaaa_p0_tool_runtime_state.json")
        agent._last_step_summary = {"executed_count": object(), "level": 1}
        text = agent._build_user_prompt(2, valid_actions=["UP"])
        self.assertIn("Current state:", text)          # stock prompt intact
        self.assertNotIn("LAST TURN:", text)           # bad record -> no invented line


if __name__ == "__main__":
    unittest.main(verbosity=2)
