"""Offline validation for the turn-pipeline repair graft (TP9).

Run:  .venv/bin/python submission/_throughput_v1/test_graft_pipeline.py
"""
from __future__ import annotations

import os
import sys
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

import graft_pipeline as t9  # noqa: E402


def _clear() -> None:
    for name in ("TP9_ENABLE", "TP9_RESUME", "TP9_LIVELOCK", "TP9_LIVELOCK_K",
                 "TP9_LIVELOCK_COOLDOWN", "TP9_RETRY", "TP9_RETRY_BACKOFF",
                 "TP9_RESUME_CHARS", "TP9_RETRY_MIN_BUDGET"):
        os.environ.pop(name, None)


def _agent(agent_mod):
    return agent_mod.ToolAgent(model="m", base_url="http://127.0.0.1:9/v1", provider="vllm")


def _result(agent_mod, *, step_executed=False, retryable=False, reasoning="", yielded=False):
    return agent_mod.AnalyzerTurnResult(
        step_executed=step_executed,
        retryable_failure=retryable,
        reasoning=reasoning,
        yielded_control=yielded,
    )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _clear()
        import requests  # noqa: PLC0415
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        cls.requests = requests
        cls.agent_mod = agent_mod
        cls.status = t9.install()

    def setUp(self) -> None:
        _clear()
        self.state_path = Path("/tmp-tp9/gameA/state.json")

    def test_01_install(self) -> None:
        self.assertIn(self.status, {"pipeline: OK", "pipeline: SKIP (already applied)"})
        self.assertTrue(hasattr(self.agent_mod.ToolAgent.analyze, "_tp9_stock"))
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._build_user_prompt, "_tp9_stock"))
        self.assertTrue(hasattr(self.agent_mod.ToolAgent._chat_completion, "_tp9_stock"))
        self.assertEqual(set(t9._STOCK), {"analyze", "build_user_prompt", "chat_completion"})

    # ------------------------------------------------- (a) persist/resume ---
    def test_02_yielded_reasoning_is_injected_once(self) -> None:
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="The key insight: blue tiles toggle.", yielded=True)
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: idle}):
            out = agent.analyze(self.state_path, 5)
        self.assertFalse(out.step_executed)
        text = agent._build_user_prompt(6, valid_actions=["UP"])
        self.assertIn("RESUME:", text)
        self.assertIn("turn time budget expired", text)
        self.assertIn("blue tiles toggle", text)
        # consumed once: the next prompt is clean
        text2 = agent._build_user_prompt(7, valid_actions=["UP"])
        self.assertNotIn("RESUME:", text2)

    def test_03_retryable_failure_reasoning_survives_and_tail_is_capped(self) -> None:
        os.environ["TP9_RESUME_CHARS"] = "300"
        agent = _agent(self.agent_mod)
        reasoning = ("x" * 5000) + "FINAL-CONCLUSION"
        failed = _result(self.agent_mod, retryable=True, reasoning=reasoning)
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: failed}):
            agent.analyze(self.state_path, 5)
        st = agent._tp9
        self.assertLessEqual(len(st.resume_tail), 300)
        self.assertTrue(st.resume_tail.endswith("FINAL-CONCLUSION"))  # tail, not head
        text = agent._build_user_prompt(6, valid_actions=["UP"])
        self.assertIn("request failed", text)
        self.assertIn("FINAL-CONCLUSION", text)

    def test_04_executed_step_clears_pending_resume(self) -> None:
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="half a thought", yielded=True)
        acted = _result(self.agent_mod, step_executed=True, reasoning="acted")
        results = iter([idle, acted])
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: next(results)}):
            agent.analyze(self.state_path, 5)
            agent.analyze(self.state_path, 5)
        text = agent._build_user_prompt(6, valid_actions=["UP"])
        self.assertNotIn("RESUME:", text)

    def test_05_game_change_resets_state(self) -> None:
        agent = _agent(self.agent_mod)
        idle_a = _result(self.agent_mod, reasoning="game A secret plan", yielded=True)
        idle_b = _result(self.agent_mod, reasoning="", yielded=True)
        seq = iter([idle_a, idle_b])
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: next(seq)}):
            agent.analyze(Path("/tmp-tp9/gameA/state.json"), 5)
            # new game: the pre-turn reset must drop game A's stashed tail
            agent.analyze(Path("/tmp-tp9/gameB/state.json"), 0)
        text = agent._build_user_prompt(1, valid_actions=["UP"])
        self.assertNotIn("game A secret plan", text)

    # ---------------------------------------------- (b) livelock detector ---
    def test_06_livelock_perturbation_after_k_identical_idle_turns(self) -> None:
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="", yielded=True)  # r11l shape: empty output
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: idle}):
            before = t9._STATE["perturbations_injected"]
            agent.analyze(self.state_path, 5)
            agent.analyze(self.state_path, 5)
            self.assertFalse(agent._tp9.perturb_pending)  # streak 2 < K=3
            agent.analyze(self.state_path, 5)
            self.assertTrue(agent._tp9.perturb_pending)
        text = agent._build_user_prompt(6, valid_actions=["UP"])
        self.assertIn("LOOP BREAKER", text)
        self.assertIn("identical output 3 times", text)
        self.assertIn("ONE new hypothesis", text)
        self.assertEqual(t9._STATE["perturbations_injected"], before + 1)
        # consumed once
        self.assertNotIn("LOOP BREAKER", agent._build_user_prompt(7, valid_actions=["UP"]))

    def test_07_differing_output_and_executed_steps_reset_streak(self) -> None:
        os.environ["TP9_LIVELOCK_K"] = "2"
        agent = _agent(self.agent_mod)
        a = _result(self.agent_mod, reasoning="thought A", yielded=True)
        b = _result(self.agent_mod, reasoning="thought B", yielded=True)
        acted = _result(self.agent_mod, step_executed=True)
        seq = iter([a, a])
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: next(seq)}):
            agent.analyze(self.state_path, 1)
            st = agent._tp9
            self.assertEqual(st.idle_streak, 1)
            agent.analyze(self.state_path, 1)
            self.assertTrue(st.perturb_pending)  # K=2 reached
            st.perturb_pending = False
        seq = iter([b, acted, a])
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: next(seq)}):
            agent.analyze(self.state_path, 1)
            self.assertEqual(agent._tp9.idle_streak, 1)  # differing hash resets to 1
            agent.analyze(self.state_path, 1)
            self.assertEqual(agent._tp9.idle_streak, 0)  # executed step resets to 0
            agent.analyze(self.state_path, 1)
            self.assertEqual(agent._tp9.idle_streak, 1)

    # --------------------------------------------- (c) client-side retry ---
    def test_08_read_timeout_retried_once_then_succeeds(self) -> None:
        os.environ["TP9_RETRY_BACKOFF"] = "0"
        agent = _agent(self.agent_mod)
        sentinel = object()
        calls = []

        def flaky(s, messages, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise self.requests.exceptions.ReadTimeout("read timed out")
            return sentinel

        before = t9._STATE["retries_used"]
        with mock.patch.dict(t9._STOCK, {"chat_completion": flaky}):
            out = agent._chat_completion([{"role": "user", "content": "x"}], tools=None)
        self.assertIs(out, sentinel)
        self.assertEqual(len(calls), 2)
        self.assertEqual(t9._STATE["retries_used"], before + 1)

    def test_09_timeout_on_every_attempt_finally_raises(self) -> None:
        os.environ["TP9_RETRY_BACKOFF"] = "0"
        agent = _agent(self.agent_mod)
        calls = []

        def always_timeout(s, messages, **kwargs):
            calls.append(1)
            raise self.requests.exceptions.ReadTimeout("read timed out")

        with mock.patch.dict(t9._STOCK, {"chat_completion": always_timeout}):
            with self.assertRaises(self.requests.Timeout):
                agent._chat_completion([{"role": "user", "content": "x"}], tools=None)
        self.assertEqual(len(calls), 2)  # 1 try + TP9_RETRY(=1) retry

    def test_10_non_timeout_request_errors_are_not_retried(self) -> None:
        agent = _agent(self.agent_mod)
        calls = []

        def http_error(s, messages, **kwargs):
            calls.append(1)
            raise self.requests.RequestException("400 | response: context length exceeded")

        with mock.patch.dict(t9._STOCK, {"chat_completion": http_error}):
            with self.assertRaises(self.requests.RequestException):
                agent._chat_completion([{"role": "user", "content": "x"}], tools=None)
        self.assertEqual(len(calls), 1)  # the context-overflow recovery seam must see it untouched

    # -------------------------------------------------------- passthrough ---
    def test_11_disabled_is_passthrough(self) -> None:
        os.environ["TP9_ENABLE"] = "0"
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="should not be stashed", yielded=True)
        calls = []

        def always_timeout(s, messages, **kwargs):
            calls.append(1)
            raise self.requests.exceptions.ReadTimeout("read timed out")

        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: idle,
                                         "chat_completion": always_timeout}):
            agent.analyze(self.state_path, 5)
            with self.assertRaises(self.requests.Timeout):
                agent._chat_completion([{"role": "user", "content": "x"}], tools=None)
        self.assertEqual(len(calls), 1)  # no retry
        text = agent._build_user_prompt(6, valid_actions=["UP"])
        self.assertNotIn("RESUME:", text)
        self.assertNotIn("LOOP BREAKER", text)

    def test_12_per_behavior_flags(self) -> None:
        os.environ["TP9_RESUME"] = "0"
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="same words", yielded=True)
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: idle}):
            for _ in range(3):
                agent.analyze(self.state_path, 5)
        text = agent._build_user_prompt(6, valid_actions=["UP"])
        self.assertNotIn("RESUME:", text)          # resume off
        self.assertIn("LOOP BREAKER", text)        # livelock still on
        os.environ.pop("TP9_RESUME", None)
        os.environ["TP9_LIVELOCK"] = "0"
        agent2 = _agent(self.agent_mod)
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: idle}):
            for _ in range(3):
                agent2.analyze(self.state_path, 5)
        text2 = agent2._build_user_prompt(6, valid_actions=["UP"])
        self.assertIn("RESUME:", text2)            # resume on
        self.assertNotIn("LOOP BREAKER", text2)    # livelock off

    def test_13_status_shape(self) -> None:
        st = t9.status()
        self.assertTrue(st["installed"])
        self.assertTrue(st["enabled"])
        self.assertEqual(st["livelock_k"], 3)
        self.assertEqual(st["livelock_cooldown"], 5)
        self.assertEqual(st["retry"], 1)
        self.assertEqual(st["retry_min_budget"], 5.0)

    # ------------------------------------------- J9 fixes: F3a/F3b/F4/F6 ---
    def test_14_server_outage_never_arms_breaker(self) -> None:
        # F3a: 3 consecutive retryable_failure turns (3 s vLLM hiccup at the
        # solver's 1 s retry cadence) carry no model output and must not arm.
        agent = _agent(self.agent_mod)
        failed = _result(self.agent_mod, retryable=True, reasoning="")
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: failed}):
            for _ in range(5):
                agent.analyze(self.state_path, 1)
        st = agent._tp9
        self.assertEqual(st.idle_streak, 0)
        self.assertFalse(st.perturb_pending)
        self.assertNotIn("LOOP BREAKER", agent._build_user_prompt(2, valid_actions=["UP"]))

    def test_15_failure_turns_do_not_reset_a_genuine_streak(self) -> None:
        # F3a: failures neither build nor reset — a real identical-output
        # streak interleaved with one failed request still arms at K.
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="same doomed thought", yielded=True)
        failed = _result(self.agent_mod, retryable=True, reasoning="")
        seq = iter([idle, idle, failed, idle])
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: next(seq)}):
            for _ in range(4):
                agent.analyze(self.state_path, 1)
        self.assertEqual(agent._tp9.idle_streak, 3)
        self.assertTrue(agent._tp9.perturb_pending)

    def test_16_injection_rate_bounded_in_persistent_livelock(self) -> None:
        # F3b: 100 turns of a persistent identical livelock must see a
        # bounded number of LOOP BREAKER injections (once per K+cooldown
        # cycle = 8 turns at defaults), not one per prompt.
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="", yielded=True)  # r11l shape
        injections = 0
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: idle}):
            for _ in range(100):
                text = agent._build_user_prompt(1, valid_actions=["UP"])
                injections += "LOOP BREAKER" in text
                agent.analyze(self.state_path, 1)
        self.assertGreaterEqual(injections, 10)   # still fires repeatedly
        self.assertLessEqual(injections, 15)      # but bounded: ~13, not ~100
        self._livelock_sim_injections = injections
        print(f"[tp9-sim] persistent-livelock 100 turns -> {injections} LOOP BREAKER injections")

    def test_17_consumed_resume_tail_survives_a_dead_request(self) -> None:
        # F6: tail consumed at prompt build; the injected request then dies
        # with no new reasoning -> tail restored; executed step clears it.
        agent = _agent(self.agent_mod)
        idle = _result(self.agent_mod, reasoning="precious partial plan", yielded=True)
        dead = _result(self.agent_mod, retryable=True, reasoning="")
        acted = _result(self.agent_mod, step_executed=True)
        seq = iter([idle, dead, acted])
        with mock.patch.dict(t9._STOCK, {"analyze": lambda s, sp, an, **k: next(seq)}):
            agent.analyze(self.state_path, 1)
            text1 = agent._build_user_prompt(2, valid_actions=["UP"])  # consume
            self.assertIn("precious partial plan", text1)
            agent.analyze(self.state_path, 1)  # the injected request dies
            text2 = agent._build_user_prompt(2, valid_actions=["UP"])  # restored
            self.assertIn("precious partial plan", text2)
            agent.analyze(self.state_path, 1)  # executed step
            text3 = agent._build_user_prompt(3, valid_actions=["UP"])
            self.assertNotIn("precious partial plan", text3)

    def test_18_timeout_retry_skipped_when_budget_exhausted(self) -> None:
        # F4: a 2.0 s request budget must not be doubled — the measured 5.0 s
        # burn. Leftover (2.0 - elapsed - backoff) < TP9_RETRY_MIN_BUDGET(5.0)
        # -> exception propagates after ONE call.
        os.environ["TP9_RETRY_BACKOFF"] = "0"
        agent = _agent(self.agent_mod)
        calls = []

        def always_timeout(s, messages, **kwargs):
            calls.append(kwargs.get("request_timeout_seconds"))
            raise self.requests.exceptions.ReadTimeout("read timed out")

        with mock.patch.dict(t9._STOCK, {"chat_completion": always_timeout}):
            with self.assertRaises(self.requests.Timeout):
                agent._chat_completion([{"role": "user", "content": "x"}],
                                       tools=None, request_timeout_seconds=2.0)
        self.assertEqual(len(calls), 1)

    def test_19_retry_gets_only_the_leftover_budget(self) -> None:
        # F4: with wall room, a fast ConnectionError is retried but the retry
        # inherits only budget - elapsed - backoff, never the full budget.
        os.environ["TP9_RETRY_BACKOFF"] = "0"
        agent = _agent(self.agent_mod)
        sentinel = object()
        calls = []

        def flaky(s, messages, **kwargs):
            calls.append(kwargs.get("request_timeout_seconds"))
            if len(calls) == 1:
                raise self.requests.ConnectionError("refused")
            return sentinel

        with mock.patch.dict(t9._STOCK, {"chat_completion": flaky}):
            out = agent._chat_completion([{"role": "user", "content": "x"}],
                                         tools=None, request_timeout_seconds=60.0)
        self.assertIs(out, sentinel)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], 60.0)
        self.assertLess(calls[1], 60.0)          # shrunk, not reused
        self.assertGreater(calls[1], 50.0)       # near-free failure keeps most of it


if __name__ == "__main__":
    unittest.main(verbosity=2)
