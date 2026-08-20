"""Offline validation for the retry_guard graft (no GPU/network/Kaggle).

Part A — unit tests of backoff/probe logic against a mock inner analyzer
(fake clock, loopback-only sockets).
Part B — integration against the Aug-07 anim bundle: install() onto the real
``HarnessSolver._make_analyzer`` seam, healthy-path transparency, and a
dead-server simulation at the fidelity of tool_agent.py:2363-2380 (requests
exception -> retryable AnalyzerTurnResult) with a refused /models probe.

Run:  .venv/bin/python submission/_retry_guard/test_retry_guard.py
"""

from __future__ import annotations

import http.server
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_BUNDLE = Path(
    "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
    "f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad/bundles/anim/src/ARC3-Inference"
)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BUNDLE))

import graft_retry
from graft_retry import RetryGuard, install


class _Result:
    def __init__(self, retryable: bool) -> None:
        self.retryable_failure = retryable
        self.step_executed = not retryable


class _MockInner:
    """Mock analyzer: scripted results + duck attrs the session reads."""

    def __init__(self, results) -> None:
        self._results = list(results)
        self.calls = 0
        self._timeout = 120.0
        self.generated_tokens = 42
        self.total_tokens = 100
        self.animation_counters = {"anim": 1}

    def analyze(self, state_path, action_num, **kwargs):
        self.calls += 1
        return self._results.pop(0)


class _FakeClock:
    """Deterministic monotonic clock; sleep() advances it and records calls."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dead_base_url() -> str:
    return f"http://127.0.0.1:{_free_port()}/v1"


class UnitBackoffProbe(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("RETRY_GUARD", None)
        os.environ["LOCAL_ANALYZER_BASE_URL"] = _dead_base_url()

    def test_healthy_turn_is_identity_passthrough(self) -> None:
        inner = _MockInner([_Result(False)])
        guard = RetryGuard(inner)
        r = _Result(False)
        inner._results = [r]
        started = time.monotonic()
        out = guard.analyze("state", 1, request_timeout_seconds=120.0, should_stop=lambda: False)
        elapsed = time.monotonic() - started
        self.assertIs(out, r)  # exact object, untouched
        self.assertEqual(guard.consecutive_failures, 0)
        self.assertEqual(guard.events, [])
        self.assertLess(elapsed, 0.05)  # no added latency path

    def test_below_threshold_no_probe_no_backoff(self) -> None:
        inner = _MockInner([_Result(True)] * 29)
        guard = RetryGuard(inner)
        for _ in range(29):
            guard.analyze("s", 1, request_timeout_seconds=120.0)
        self.assertEqual(guard.consecutive_failures, 29)
        self.assertEqual(guard.events, [])
        self.assertEqual(guard.backoff_seconds, 0.0)

    def test_threshold_plus_dead_probe_engages_exponential_backoff(self) -> None:
        clock = _FakeClock()
        inner = _MockInner([_Result(True)] * 40)
        guard = RetryGuard(inner, probe_timeout_s=0.2)
        with mock.patch.object(graft_retry.time, "monotonic", clock.monotonic), \
             mock.patch.object(graft_retry.time, "sleep", clock.sleep):
            for _ in range(40):
                guard.analyze("s", 1, request_timeout_seconds=600.0, should_stop=lambda: False)
        backoffs = [e[1] for e in guard.events if e[0] == "backoff"]
        # turns 30..40 -> 11 backoffs: 1,2,4,...,60-capped
        self.assertEqual(backoffs, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        probes = [e for e in guard.events if e[0] == "probe"]
        self.assertTrue(probes and all(e[1] is False for e in probes))
        self.assertTrue(all(s <= 5.0 for s in clock.sleeps))  # <=5s slices

    def test_backoff_clamped_to_remaining_budget(self) -> None:
        clock = _FakeClock()
        inner = _MockInner([_Result(True)] * 31)
        guard = RetryGuard(inner, probe_timeout_s=0.1)
        with mock.patch.object(graft_retry.time, "monotonic", clock.monotonic), \
             mock.patch.object(graft_retry.time, "sleep", clock.sleep):
            for _ in range(31):
                guard.analyze("s", 1, request_timeout_seconds=0.5, should_stop=None)
        for _, _nominal, target in (e for e in guard.events if e[0] == "backoff"):
            self.assertLessEqual(target, 0.5 - 0.25 + 1e-9)  # budget minus margin
        self.assertLessEqual(sum(clock.sleeps), 0.5)

    def test_healthy_probe_means_no_backoff(self) -> None:
        port = _free_port()

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):  # silence
                pass

        srv = http.server.HTTPServer(("127.0.0.1", port), H)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            os.environ["LOCAL_ANALYZER_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
            inner = _MockInner([_Result(True)] * 31)
            guard = RetryGuard(inner)
            for _ in range(31):
                guard.analyze("s", 1, request_timeout_seconds=30.0)
            probes = [e for e in guard.events if e[0] == "probe"]
            self.assertTrue(probes and all(e[1] is True for e in probes))
            self.assertEqual([e for e in guard.events if e[0] == "backoff"], [])
            self.assertEqual(guard.backoff_seconds, 0.0)
        finally:
            srv.shutdown()

    def test_probe_sends_bearer_header_when_key_set(self) -> None:
        port = _free_port()
        seen_auth: list[str | None] = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                seen_auth.append(self.headers.get("Authorization"))
                if self.headers.get("Authorization") != "Bearer sekrit":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            os.environ["LOCAL_ANALYZER_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
            os.environ["LOCAL_ANALYZER_API_KEY"] = "sekrit"
            inner = _MockInner([_Result(True)] * 30)
            guard = RetryGuard(inner)
            for _ in range(30):
                guard.analyze("s", 1, request_timeout_seconds=30.0)
            self.assertEqual(seen_auth, ["Bearer sekrit"])
            probes = [e for e in guard.events if e[0] == "probe"]
            self.assertEqual([p[1] for p in probes], [True])  # authed => alive
            self.assertEqual([e for e in guard.events if e[0] == "backoff"], [])
        finally:
            os.environ.pop("LOCAL_ANALYZER_API_KEY", None)
            srv.shutdown()

    def test_should_stop_aborts_backoff_within_one_slice(self) -> None:
        clock = _FakeClock()
        inner = _MockInner([_Result(True)] * 30)
        guard = RetryGuard(inner, probe_timeout_s=0.1, backoff_cap_s=60.0)
        guard._backoff_s = 30.0  # next backoff -> 60s nominal
        stop_calls = []

        def should_stop() -> bool:
            stop_calls.append(clock.now)
            return len(stop_calls) >= 3  # allow two slices then stop

        with mock.patch.object(graft_retry.time, "monotonic", clock.monotonic), \
             mock.patch.object(graft_retry.time, "sleep", clock.sleep):
            for _ in range(30):
                guard.analyze("s", 1, request_timeout_seconds=600.0, should_stop=should_stop)
        self.assertEqual(len(clock.sleeps), 2)  # aborted after 2 slices, not 12
        self.assertTrue(all(s <= 5.0 for s in clock.sleeps))
        self.assertGreaterEqual(len(stop_calls), 3)

    def test_streak_resets_on_healthy_turn(self) -> None:
        inner = _MockInner([_Result(True)] * 10 + [_Result(False)] + [_Result(True)])
        guard = RetryGuard(inner)
        for _ in range(12):
            guard.analyze("s", 1, request_timeout_seconds=60.0)
        self.assertEqual(guard.consecutive_failures, 1)

    def test_kill_switch_disables_governing(self) -> None:
        os.environ["RETRY_GUARD"] = "0"
        try:
            inner = _MockInner([_Result(True)] * 35)
            guard = RetryGuard(inner)
            started = time.monotonic()
            for _ in range(35):
                guard.analyze("s", 1, request_timeout_seconds=60.0)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(guard.events, [])
            self.assertEqual(guard.consecutive_failures, 0)
        finally:
            os.environ.pop("RETRY_GUARD", None)

    def test_guard_logic_error_fails_open(self) -> None:
        class Poison:
            @property
            def retryable_failure(self):
                raise RuntimeError("poisoned result")

        inner = _MockInner([Poison()])
        guard = RetryGuard(inner)
        out = guard.analyze("s", 1, request_timeout_seconds=60.0)
        self.assertIsInstance(out, Poison)  # inner result still returned

    def test_inner_crash_propagates_as_stock(self) -> None:
        class Crasher:
            def analyze(self, *a, **k):
                raise ValueError("inner crash")

        guard = RetryGuard(Crasher())
        with self.assertRaises(ValueError):
            guard.analyze("s", 1)

    def test_duck_attr_proxy(self) -> None:
        inner = _MockInner([])
        guard = RetryGuard(inner)
        self.assertEqual(guard._timeout, 120.0)
        self.assertEqual(guard.generated_tokens, 42)
        self.assertEqual(guard.total_tokens, 100)
        self.assertEqual(guard.animation_counters, {"anim": 1})
        self.assertTrue(hasattr(guard, "generated_tokens"))  # solver.py:86-90 path
        self.assertFalse(hasattr(guard, "no_such_attr"))


class IntegrationAnimBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("RETRY_GUARD", None)
        from inference.framework import solver as solver_mod  # noqa: PLC0415
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

        cls.solver_mod = solver_mod
        cls.agent_mod = agent_mod

    def test_01_install_ok_then_idempotent(self) -> None:
        self.assertEqual(install(), "retry_guard: OK")
        self.assertEqual(install(), "retry_guard: SKIP (already applied)")

    def test_02_make_analyzer_returns_wrapped_analyzer(self) -> None:
        install()
        stub = _MockInner([_Result(False)])
        solver = self.solver_mod.HarnessSolver(analyzer_factory=lambda game, index: stub)
        analyzer = solver._make_analyzer(None, 0, None)
        self.assertIsInstance(analyzer, RetryGuard)
        self.assertIs(analyzer._inner, stub)
        # duck-reads the session performs (solver.py:86-90, :269, :470)
        self.assertEqual(self.solver_mod._analyzer_reported_tokens(analyzer), 42)
        self.assertEqual(analyzer._timeout, 120.0)

    def test_03_healthy_path_transparent_through_real_seam(self) -> None:
        install()
        real_result = self.agent_mod.AnalyzerTurnResult(step_executed=True)
        stub = _MockInner([real_result])
        solver = self.solver_mod.HarnessSolver(analyzer_factory=lambda game, index: stub)
        analyzer = solver._make_analyzer(None, 0, None)
        started = time.monotonic()
        out = analyzer.analyze(
            "state_path",
            5,
            valid_actions=["ACTION1"],
            step_env=None,
            transcript_path=None,
            analysis_step=1,
            request_timeout_seconds=120.0,
            should_stop=lambda: False,
        )
        elapsed = time.monotonic() - started
        self.assertIs(out, real_result)
        self.assertFalse(out.retryable_failure)
        self.assertLess(elapsed, 0.05)
        self.assertEqual(analyzer.events, [])

    def test_04_dead_server_simulation_engages_backoff(self) -> None:
        """Fidelity of tool_agent.py:2363-2380: a requests exception inside
        analyze becomes AnalyzerTurnResult(retryable_failure=True); the play
        loop would re-call analyze forever. With the guard, turn 30+ probes
        the dead /models endpoint (connection refused) and backs off."""
        import requests  # noqa: PLC0415

        install()
        dead = _dead_base_url()
        os.environ["LOCAL_ANALYZER_BASE_URL"] = dead
        agent_mod = self.agent_mod

        class DeadServerAgent:
            _timeout = 120.0
            generated_tokens = 0
            total_tokens = 0

            def analyze(self, state_path, action_num, **kwargs):
                try:
                    requests.post(f"{dead}/chat/completions", json={}, timeout=0.2)
                except requests.RequestException:
                    return agent_mod.AnalyzerTurnResult(
                        step_executed=False, retryable_failure=True
                    )
                raise AssertionError("dead port unexpectedly accepted")

        solver = self.solver_mod.HarnessSolver(
            analyzer_factory=lambda game, index: DeadServerAgent()
        )
        analyzer = solver._make_analyzer(None, 0, None)
        stop_polls = []

        def should_stop() -> bool:
            stop_polls.append(time.monotonic())
            return False

        started = time.monotonic()
        for _ in range(32):
            out = analyzer.analyze(
                "s", 1, request_timeout_seconds=0.6, should_stop=should_stop
            )
            self.assertTrue(out.retryable_failure)  # result still stock
        elapsed = time.monotonic() - started
        probes = [e for e in analyzer.events if e[0] == "probe"]
        backoffs = [e for e in analyzer.events if e[0] == "backoff"]
        self.assertEqual(len(probes), 3)  # turns 30, 31, 32
        self.assertTrue(all(alive is False for _, alive, _ in probes))
        self.assertEqual(len(backoffs), 3)
        self.assertEqual([b[1] for b in backoffs], [1.0, 2.0, 4.0])
        # each turn's sleep clamped to the 0.6s budget minus margin
        self.assertTrue(all(t <= 0.6 for _, _, t in backoffs))
        self.assertGreater(len(stop_polls), 0)  # should_stop was polled
        self.assertLess(elapsed, 10.0)  # bounded despite 32 dead turns


if __name__ == "__main__":
    unittest.main(verbosity=2)
