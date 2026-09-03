"""Offline validation for the fresh-mind level retry graft (RETRY).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_retry.py
      GRAFT_TEST_BUNDLE=scratchpad/bundles/june_stock/src/ARC3-Inference \
          .venv/bin/python submission/_throughput_v1/test_graft_retry.py
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

import graft_retry as tr  # noqa: E402

_FLAGS = ("RETRY_ENABLE", "RETRY_K", "RETRY_ABS", "RETRY_COOLDOWN", "RETRY_MAX", "RETRY_CLEAR_HISTORY")
_GRID = tuple(tuple((r * 7 + c) % 5 for c in range(8)) for r in range(8))


def _clear() -> None:
    for name in _FLAGS:
        os.environ.pop(name, None)


def _reset_counters() -> None:
    with tr._LOCK:
        tr._STATE["retries_fired"] = 0
        tr._STATE["levels_cleared_after_retry"] = 0
        tr._STATE["retry_log"] = []
        tr._STATE["clear_log"] = []
        tr._STATE["skips"] = {}


class _FakeState:
    def __init__(self, arcengine, levels_completed=0, available=(0, 1, 2, 3, 4), state=None):
        self.frame = types.SimpleNamespace(data=[list(r) for r in _GRID])
        self.available_actions = list(available)
        self.levels_completed = levels_completed
        self.raw = types.SimpleNamespace(state=state if state is not None else arcengine.GameState.NOT_FINISHED)
        self.just_won_level = False
        self.won = False


class _FakeSession:
    """Stand-in for _HarnessGameSession: the fields graft_retry reads plus the
    two action paths (step_env for the model, _execute_action for the harness)."""

    def __init__(self, arcengine, runtime_mod, state_path: Path, *, game_id="tu93-test", levels=3,
                 baselines=(20, 30, 40)):
        self.arcengine = arcengine
        self.runtime_mod = runtime_mod
        self.state_path = state_path
        run = types.SimpleNamespace(game_id=game_id, number_of_levels=levels,
                                    actions_per_level=[0] * levels,
                                    base_actions_per_level=list(baselines) if baselines is not None else None,
                                    levels_completed=0, history=[], state="playing")
        self.game = types.SimpleNamespace(current_state=_FakeState(arcengine), game_run=run,
                                          number_of_levels=levels,
                                          base_actions_per_level=run.base_actions_per_level)
        self.last_engine_action = None
        self.history_entries = []
        self.executed = []      # (name, batch_index, batch_size, generated_tokens)
        self.seed_initial_history()
        self.write_runtime_state()

    # -- bookkeeping ---------------------------------------------------------
    @property
    def action_count(self) -> int:
        return len(self.game.game_run.history)

    def _frame(self):
        return self.runtime_mod.Frame(grid=_GRID, step=self.action_count,
                                      level=self.game.current_state.levels_completed + 1)

    def seed_initial_history(self):
        if not self.history_entries:
            self.history_entries.append(self.runtime_mod.HistoryEntry(action="", frame=self._frame()))

    def write_runtime_state(self):
        self.runtime_mod.write_runtime_state(self.state_path, current_frame=self._frame(),
                                             history=self.history_entries)

    def set_level_actions(self, idx: int, n: int) -> None:
        """Pretend the level bucket already holds n actions (history length follows)."""
        run = self.game.game_run
        run.actions_per_level[idx] = n
        run.history = ["x"] * sum(run.actions_per_level)

    def advance_level(self) -> None:
        run = self.game.game_run
        run.levels_completed += 1
        self.game.current_state = _FakeState(self.arcengine, levels_completed=run.levels_completed)

    # -- the two action paths -----------------------------------------------
    def _execute_action(self, action, *, batch_index, batch_size, generated_tokens=None, flush_viewer_payload=True):
        name = action.id.name
        run = self.game.game_run
        idx = self.game.current_state.levels_completed
        run.history.append(name)
        run.actions_per_level[idx] += 1
        self.last_engine_action = name
        self.executed.append((name, batch_index, batch_size, generated_tokens))
        self.history_entries.append(self.runtime_mod.HistoryEntry(action=name, frame=self._frame()))
        self.write_runtime_state()
        return {"executed": True, "action_num": self.action_count, "level": idx + 1, "score": idx,
                "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
                "board_changed": True, "done": False, "level_completed": False, "game_over": False,
                "run_complete": False, "action_name": name, "action_data": dict(action.data),
                "action_display": name, "batch_index": batch_index, "batch_size": batch_size}

    def step_env(self, arguments):
        acts = arguments.get("actions") or []
        payloads = []
        for i, raw in enumerate(acts, start=1):
            name = str(raw.get("action", "UP")).upper()
            aid = {"UP": self.arcengine.GameAction.ACTION1, "DOWN": self.arcengine.GameAction.ACTION2,
                   "LEFT": self.arcengine.GameAction.ACTION3, "RIGHT": self.arcengine.GameAction.ACTION4}[name]
            payloads.append(self._execute_action(self.arcengine.ActionInput(id=aid, data={}),
                                                 batch_index=i, batch_size=len(acts), flush_viewer_payload=False))
        final = dict(payloads[-1])
        final.update({"batched": len(acts) > 1, "requested_count": len(acts), "executed_count": len(payloads),
                      "requested_actions": [p["action_display"] for p in payloads],
                      "executed_actions": [p["action_display"] for p in payloads], "stopped_early": False})
        return final


class RetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        import arcengine  # noqa: PLC0415
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.agent import runtime_state as runtime_mod  # noqa: PLC0415
        cls.arcengine, cls.agent_mod, cls.runtime_mod = arcengine, agent_mod, runtime_mod
        cls.status = tr.install()

    def setUp(self) -> None:
        _clear()
        _reset_counters()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "gameA" / "state.json"
        self.transcript = self.root / "gameA" / "transcript.txt"
        self.state_path.parent.mkdir(parents=True)
        self.transcript.touch()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- helpers ----------------------------------------------------------
    def _agent(self):
        return self.agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")

    def _session(self, **kw):
        return _FakeSession(self.arcengine, self.runtime_mod, self.state_path, **kw)

    def _turn(self, agent, sess, *, stock=None, should_stop=None):
        """One analyze() through the wrapper with the stock analyze mocked out."""
        if stock is None:
            def stock(s, sp, an, **k):  # noqa: ANN001
                # the stock turn builds its prompt exactly once
                s._build_user_prompt(an, valid_actions=k.get("valid_actions"))
                return self.agent_mod.AnalyzerTurnResult(step_executed=True, reasoning="ok")
        with mock.patch.dict(tr._STOCK, {"analyze": stock}):
            return agent.analyze(self.state_path, sess.action_count, valid_actions=["UP", "DOWN"],
                                 step_env=sess.step_env, transcript_path=self.transcript,
                                 analysis_step=1, should_stop=should_stop)

    def _resets(self, sess):
        return [e for e in sess.executed if e[0] == "RESET"]

    # ------------------------------------------------------------ 01 install
    def test_01_install(self) -> None:
        self.assertIn(self.status, {"retry: OK", "retry: SKIP (already applied)"})
        self.assertTrue(tr._STATE["installed"])
        self.assertTrue(hasattr(self.agent_mod.ToolAgent.analyze, "_retry_stock"))
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._build_user_prompt, "_retry_stock"))
        self.assertEqual(set(tr._STOCK), {"analyze", "build_user_prompt"})

    # ------------------------------------------------- 02 flag-off passthrough
    def test_02_flag_off_is_byte_identical_to_stock(self) -> None:
        """Stock analyze vs the wrapper with RETRY_ENABLE=0, on a mock game whose
        level bucket is far past the trigger: same transcript bytes, same
        prompts, no RESET, no state object."""
        os.environ["RETRY_ENABLE"] = "0"
        reply = self.agent_mod._ChatCompletionResult(
            message={"role": "assistant", "content": "World model: the wrong idea.", "reasoning": "hmm",
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "python",
                                                  "arguments": json.dumps({"code": "action(['UP'])"})}}]},
            finish_reason="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        transcripts = []
        prompts = []
        for variant in ("stock", "wrapped"):
            root = self.root / variant
            state_path = root / "state.json"
            transcript = root / "transcript.txt"
            root.mkdir()
            transcript.touch()
            sess = _FakeSession(self.arcengine, self.runtime_mod, state_path)
            sess.set_level_actions(0, 999)      # would trigger at 60 if the flag were on
            agent = self._agent()
            with mock.patch.object(self.agent_mod.ToolAgent, "_chat_completion", return_value=reply), \
                    mock.patch.object(self.agent_mod.time, "strftime", return_value="00:00:00"):
                stock_fn = tr._STOCK["analyze"]
                fn = (lambda *a, **k: stock_fn(agent, *a, **k)) if variant == "stock" else agent.analyze
                seen = []
                inner = tr._STOCK["build_user_prompt"]

                def spy(s, an, *a, **k):  # noqa: ANN001
                    text = inner(s, an, *a, **k)
                    seen.append(text)
                    return text
                with mock.patch.dict(tr._STOCK, {"build_user_prompt": spy}):
                    for _ in range(3):
                        res = fn(state_path, sess.action_count, valid_actions=["UP", "DOWN"],
                                 step_env=sess.step_env, transcript_path=transcript, analysis_step=1)
                        self.assertTrue(res.step_executed)
            self.assertEqual(self._resets(sess), [])
            self.assertFalse(hasattr(agent, "_retry"))
            transcripts.append(transcript.read_bytes())
            prompts.append(seen)
        self.assertEqual(transcripts[0], transcripts[1])
        self.assertEqual(prompts[0], prompts[1])
        self.assertNotIn(b"[RETRY]", transcripts[1])
        self.assertNotIn(b"FRESH MIND", transcripts[1])
        self.assertEqual(tr._STATE["retries_fired"], 0)

    # ------------------------------------------------------- 03 trigger at K*b
    def test_03_trigger_fires_at_exactly_k_times_baseline(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 59)
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        sess.set_level_actions(0, 60)          # K=3 x 20
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 1)
        self.assertEqual(sess.game.game_run.actions_per_level[0], 61)   # RESET landed in the bucket
        self.assertEqual(tr._STATE["retries_fired"], 1)
        text = self.transcript.read_text()
        self.assertIn("[HARNESS RETRY]\n[RETRY] game=tu93-test level=1 actions=60 baseline=20 threshold=60 "
                      "retry=1/2 action_num=61", text)

    def test_04_absolute_fallback_when_no_baseline(self) -> None:
        os.environ["RETRY_ABS"] = "30"
        agent = self._agent()
        sess = self._session(baselines=None)
        sess.set_level_actions(0, 29)
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        sess.set_level_actions(0, 30)
        seen = []

        def stock(s, sp, an, **k):  # noqa: ANN001
            seen.append(s._build_user_prompt(an, valid_actions=k.get("valid_actions")))
            return self.agent_mod.AnalyzerTurnResult(step_executed=True)

        self._turn(agent, sess, stock=stock)
        self.assertEqual(len(self._resets(sess)), 1)
        self.assertIn("[RETRY] game=tu93-test level=1 actions=30 baseline=- threshold=30", self.transcript.read_text())
        self.assertIn("not cleared after 30 actions (limit 30 actions)", seen[0])

    def test_05_fractional_k_and_thresholds(self) -> None:
        os.environ["RETRY_K"] = "2.5"
        self.assertEqual(tr.threshold_for(20), 50)
        self.assertEqual(tr.threshold_for(21), 53)       # ceil(52.5)
        self.assertEqual(tr.threshold_for(None), 200)    # RETRY_ABS default
        os.environ["RETRY_K"] = "garbage"
        self.assertEqual(tr.retry_k(), tr.DEFAULT_K)

    # ------------------------------------------------- 06 cooldown / max retries
    def test_06_cooldown_and_max_retries_honoured(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)                              # retry 1 at 60 -> bucket 61
        self.assertEqual(len(self._resets(sess)), 1)
        sess.last_engine_action = "UP"
        sess.set_level_actions(0, 61 + 149)                  # 149 < cooldown 150
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 1)
        self.assertEqual(tr._STATE["skips"].get("cooldown"), 1)
        sess.last_engine_action = "UP"
        sess.set_level_actions(0, 61 + 150)                  # cooldown satisfied
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 2)         # retry 2 (max)
        sess.last_engine_action = "UP"
        sess.set_level_actions(0, 1000)
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 2)         # max 2 per level
        self.assertEqual(tr._STATE["skips"].get("max_retries"), 1)
        log = tr._STATE["retry_log"]
        self.assertEqual([r["retry"] for r in log], [1, 2])
        self.assertEqual([r["actions"] for r in log], [60, 211])

    def test_07_max_per_level_not_per_game(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)
        sess.last_engine_action = "UP"
        sess.set_level_actions(0, 300)
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 2)
        sess.advance_level()                                 # level 2, baseline 30 -> threshold 90
        sess.last_engine_action = "UP"
        sess.set_level_actions(1, 89)
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 2)
        sess.set_level_actions(1, 90)
        self._turn(agent, sess)
        self.assertEqual(len(self._resets(sess)), 3)         # fresh budget on the new level
        self.assertEqual(tr._STATE["retry_log"][-1]["level"], 2)

    # ---------------------------------------------------- 08 fresh-mind block
    def test_08_fresh_mind_block_once_on_next_prompt(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "the red bar is a decorative step counter"
        agent._summarized_knowledge["goal_model"] = "reach the exit"
        agent._summarized_knowledge["cross_level_notes"] = "UP moves the player up"
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        seen = []

        def stock(s, sp, an, **k):  # noqa: ANN001
            seen.append(s._build_user_prompt(an, valid_actions=k.get("valid_actions")))
            return self.agent_mod.AnalyzerTurnResult(step_executed=True)

        self._turn(agent, sess, stock=stock)
        self.assertEqual(len(seen), 1)
        text = seen[0]
        self.assertEqual(text.count("FRESH MIND"), 1)
        self.assertIn("harness level retry 1/2", text)
        self.assertIn("RESET this level at action 61", text)
        self.assertIn("not cleared after 60 actions (human baseline ~20, limit 3x)", text)
        self.assertIn("World model: the red bar is a decorative step counter | Goal model: reach the exit", text)
        self.assertIn("TWO alternative hypotheses", text)
        self.assertIn("at most 5 actions", text)
        # the stale note is gone from the stock world-model block, cross-level notes survive
        self.assertNotIn("Working world model carried from earlier turns:\n- World model: the red bar", text)
        self.assertIn("- Cross-level notes: UP moves the player up", text)
        # absent on the following prompt
        self.assertNotIn("FRESH MIND", agent._build_user_prompt(62, valid_actions=["UP"]))
        self.assertNotIn("FRESH MIND", agent._build_user_prompt(63, valid_actions=["UP"]))

    def test_09_block_budget_and_quote_cap(self) -> None:
        long_note = {"world_model": "w" * 900, "goal_model": "g" * 400, "action_model": "a" * 300,
                     "current_plan": "p" * 300}
        quote = tr.quote_note(long_note)
        self.assertLessEqual(len(quote), tr.QUOTE_CHARS)
        block = tr.render_block({"retry": 2, "max": 2, "action_num": 12345, "actions": 9999, "baseline": 123,
                                 "threshold": 369, "level": 9, "quote": quote, "k": 3.0})
        self.assertLessEqual(len(block), tr.BLOCK_CHARS)
        self.assertIn("<<<\n" + "World model: " + "w" * 500, block)
        self.assertEqual(tr.quote_note({}), "(no world-model note was recorded)")
        self.assertEqual(tr.quote_note({"cross_level_notes": "kept but not quoted"}),
                         "(no world-model note was recorded)")

    def test_10_note_suppression_only_for_that_level(self) -> None:
        agent = self._agent()
        k = agent._summarized_knowledge
        for key in tr._WIPED_FIELDS:
            k[key] = f"stale {key}"
        k["cross_level_notes"] = "keep me"
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)
        for key in tr._WIPED_FIELDS:
            self.assertEqual(k[key], "", key)
        self.assertEqual(k["cross_level_notes"], "keep me")
        self.assertEqual(agent._history_messages, [])      # default: history untouched (was empty)
        # a later level's notes are never touched by the graft
        sess.advance_level()
        sess.last_engine_action = "UP"
        k["world_model"] = "level 2 model"
        sess.set_level_actions(1, 5)
        self._turn(agent, sess)
        self.assertEqual(k["world_model"], "level 2 model")

    def test_11_optional_history_clear(self) -> None:
        os.environ["RETRY_CLEAR_HISTORY"] = "1"
        agent = self._agent()
        agent._history_messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old"}]
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)
        self.assertEqual(agent._history_messages, [])
        _clear()
        agent2 = self._agent()
        agent2._history_messages = [{"role": "user", "content": "old"}]
        sess2 = self._session(baselines=(20, 30, 40))
        sess2.set_level_actions(0, 60)
        self._turn(agent2, sess2)
        self.assertEqual(len(agent2._history_messages), 1)   # default 0: kept

    # --------------------------------------------- 12 RESET via the normal path
    def test_12_reset_goes_through_execute_action_and_refreshes_turn_args(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        captured = {}

        def stock(s, sp, an, **k):  # noqa: ANN001
            captured.update({"action_num": an, "valid_actions": k.get("valid_actions"),
                             "step_env": k.get("step_env"), "transcript_path": k.get("transcript_path")})
            s._build_user_prompt(an, valid_actions=k.get("valid_actions"))
            return self.agent_mod.AnalyzerTurnResult(step_executed=True)

        self._turn(agent, sess, stock=stock)
        self.assertEqual(sess.executed[-1], ("RESET", 1, 1, 0))          # batch 1/1, generated_tokens=0
        self.assertEqual(sess.game.game_run.history[-1], "RESET")        # recorded like any action
        self.assertEqual(sess.last_engine_action, "RESET")
        self.assertEqual(captured["action_num"], 61)                     # stock sees the post-RESET count
        self.assertEqual(captured["valid_actions"], ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        self.assertIs(captured["step_env"].__self__, sess)
        self.assertEqual(captured["transcript_path"], self.transcript)
        frame, hist = self.runtime_mod.load_runtime_state(self.state_path)
        self.assertEqual(hist[-1].action, "RESET")                       # runtime state rewritten before the turn

    # --------------------------------------------------------------- 13 guards
    def test_13_guards_never_fire(self) -> None:
        agent = self._agent()
        # (a) GAME_OVER: the harness auto-reset is pending
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        sess.game.current_state.raw.state = self.arcengine.GameState.GAME_OVER
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        self.assertEqual(tr._STATE["skips"].get("terminal_state"), 1)
        # (b) the last engine action was already a RESET (auto-reset just happened)
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        sess.last_engine_action = "RESET"
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        # (c) RESET not listed by the engine
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        sess.game.current_state.available_actions = [1, 2, 3, 4]
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        self.assertEqual(tr._STATE["skips"].get("reset_unavailable"), 1)
        # (d) should_stop() true
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess, should_stop=lambda: True)
        self.assertEqual(self._resets(sess), [])
        self.assertEqual(tr._STATE["skips"].get("should_stop"), 1)
        # (e) run no longer playing
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        sess.game.game_run.state = "won"
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        # (f) RETRY_MAX=0
        os.environ["RETRY_MAX"] = "0"
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)
        self.assertEqual(self._resets(sess), [])
        # (g) no step_env at all (analyze called without a session) -> stock only
        _clear()
        with mock.patch.dict(tr._STOCK, {"analyze": lambda s, sp, an, **k: self.agent_mod.AnalyzerTurnResult(step_executed=True)}):
            out = agent.analyze(self.state_path, 0, valid_actions=["UP"], step_env=None, transcript_path=self.transcript)
        self.assertTrue(out.step_executed)
        self.assertEqual(tr._STATE["retries_fired"], 0)

    # ---------------------------------------------------------- 14 clear log
    def test_14_clear_after_retry_is_logged_and_counted(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)
        self.assertEqual(tr._STATE["levels_cleared_after_retry"], 0)
        # the level clears during the next turn (levels_completed moves on)
        sess.last_engine_action = "UP"

        def stock_clears(s, sp, an, **k):  # noqa: ANN001
            sess.set_level_actions(0, 70)
            sess.advance_level()
            return self.agent_mod.AnalyzerTurnResult(step_executed=True)

        self._turn(agent, sess, stock=stock_clears)
        self.assertEqual(tr._STATE["levels_cleared_after_retry"], 1)
        self.assertEqual(tr._STATE["clear_log"], [{"game": "tu93-test", "level": 1, "actions": 70, "retries": 1}])
        text = self.transcript.read_text()
        self.assertIn("[RETRY-CLEAR] game=tu93-test level=1 actions=70 retries=1", text)
        self.assertEqual(text.count("[RETRY-CLEAR]"), 1)
        self._turn(agent, sess)                                     # idempotent
        self.assertEqual(self.transcript.read_text().count("[RETRY-CLEAR]"), 1)
        # a level that clears WITHOUT a retry is not counted
        sess.advance_level()
        self._turn(agent, sess)
        self.assertEqual(tr._STATE["levels_cleared_after_retry"], 1)

    # ---------------------------------------------- 15 dead request re-arms it
    def test_15_block_rearmed_when_its_request_dies(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        seen = []

        def dead(s, sp, an, **k):  # noqa: ANN001
            seen.append(s._build_user_prompt(an, valid_actions=k.get("valid_actions")))
            return self.agent_mod.AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning="")

        self._turn(agent, sess, stock=dead)
        self.assertIn("FRESH MIND", seen[0])
        sess.last_engine_action = "UP"
        self._turn(agent, sess, stock=dead)                          # solver retries the step
        self.assertIn("FRESH MIND", seen[1])                         # restored, not lost
        self.assertEqual(len(self._resets(sess)), 1)                 # no second RESET (cooldown)
        self._turn(agent, sess)                                      # a live turn consumes it
        self.assertNotIn("FRESH MIND", agent._build_user_prompt(70, valid_actions=["UP"]))

    # ------------------------------------------------------ 16 game change
    def test_16_game_change_resets_state(self) -> None:
        agent = self._agent()
        sess = self._session(baselines=(20, 30, 40))
        sess.set_level_actions(0, 60)
        self._turn(agent, sess)
        self.assertTrue(agent._retry.fires)
        other = self.root / "gameB" / "state.json"
        other.parent.mkdir()
        sess_b = _FakeSession(self.arcengine, self.runtime_mod, other, game_id="gameB")
        sess_b.set_level_actions(0, 60)
        with mock.patch.dict(tr._STOCK, {"analyze": lambda s, sp, an, **k: self.agent_mod.AnalyzerTurnResult(step_executed=True)}):
            agent.analyze(other, sess_b.action_count, valid_actions=["UP"], step_env=sess_b.step_env,
                          transcript_path=self.root / "gameB" / "t.txt")
        self.assertEqual(len(self._resets(sess_b)), 1)               # fresh per-level budget
        self.assertEqual(agent._retry.runtime_dir, other.parent)

    # ---------------------------------------------------------------- 17 status
    def test_17_status_shape(self) -> None:
        os.environ["RETRY_K"] = "2"
        os.environ["RETRY_COOLDOWN"] = "10"
        st = tr.status()
        self.assertTrue(st["installed"] and st["enabled"])
        self.assertEqual((st["k"], st["abs"], st["cooldown"], st["max"], st["clear_history"]), (2.0, 200, 10, 2, False))
        for key in ("retries_fired", "levels_cleared_after_retry", "retry_log", "clear_log", "skips"):
            self.assertIn(key, st)
        os.environ["RETRY_ENABLE"] = "off"
        self.assertFalse(tr.status()["enabled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
