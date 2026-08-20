"""Retry-guard graft — bounded retry + vLLM health-probe on the analyzer.

Motivation (observed on our own v12smoke run: sb26 died in a 'Read timed
out' loop): the stock play loop retries a ``retryable_failure`` turn every
1s FOREVER (``ANALYZER_RETRY_BACKOFF_SECONDS = 1.0``, solver.py:64; sleep at
solver.py:356), so a dead local vLLM server burns the game's full
``max_runtime_s_per_game`` (7920s) at one useless request per second.

Port of the audited community fix (banking bundle,
taaf-grafts/taaf_grafts/retry_guard.py — written against the June bundle):
after ``failure_threshold`` (30) consecutive retryable turns AND a health
probe of ``{LOCAL_ANALYZER_BASE_URL}/models`` confirming the server is dead,
the guard absorbs an exponential backoff (1s doubling to a 60s cap) *inside*
``analyze`` before returning the same retryable result, clamped to the
remaining per-request budget, polling ``should_stop`` in <=5s slices.
Transparent pass-through on every healthy turn: the inner
``AnalyzerTurnResult`` is returned untouched, always.

Seams (verified against the Aug-07 anim bundle, ARC3-Inference):
- solver.py:352-357 — the unbounded retry loop this closes
  (``if result.retryable_failure: ... time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)``).
- solver.py:333-342 — the ``analyzer.analyze(state_path, action_count,
  ...)`` call site passes ``request_timeout_seconds=`` and ``should_stop=``
  as keywords; the guard's signature matches (2 positional + **kwargs).
- tool_agent.py:2090-2101 — ``ToolAgent.analyze`` signature;
  tool_agent.py:2363-2380 — the retryable path (requests.RequestException ->
  ``AnalyzerTurnResult(step_executed=False, retryable_failure=True, ...)``).
- solver.py:86-90 (generated_tokens/total_tokens via hasattr), :269
  (``_timeout``), :470 (``animation_counters``) — duck-reads off the
  analyzer; RetryGuard.__getattr__ proxies all of them to the inner agent.
- Install point: ``HarnessSolver._make_analyzer`` (solver.py:1339-1366),
  invoked ONLY as ``self._make_analyzer(...)`` at solver.py:1383 — a
  class-attribute patch reaches every call, so NO dual-namespace rebinding
  is needed (checked: grep shows no by-name import of ``_make_analyzer``
  anywhere; ``ToolAgent`` IS imported by name at solver.py:38 and
  agent/__init__.py:3, but we wrap the *instance* at the factory seam, not
  the class, so those bindings are untouched). Wrapping the factory also
  covers ``analyzer_factory``-produced analyzers (solver.py:1345-1346).

Fail-open invariants:
- The inner ``analyze`` call is never wrapped in try/except — an inner
  crash propagates exactly as stock.
- All guard logic (streak/probe/backoff) sits inside a blanket try/except
  that returns the already-computed inner result on any error.
- Probe socket timeout and backoff sleep are both clamped to the remaining
  ``request_timeout_seconds`` budget (minus a 0.25s margin), so analyze
  wall time never exceeds the solver's per-request budget.
- RETRY_GUARD=0 disables: at install time (no wrap) and at call time
  (installed guard becomes a pure pass-through).
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Any

DEFAULT_FAILURE_THRESHOLD = 30
DEFAULT_PROBE_TIMEOUT_S = 5.0
DEFAULT_BACKOFF_CAP_S = 60.0

_BACKOFF_BASE_S = 1.0
_SHOULD_STOP_SLICE_S = 5.0
_TIMEOUT_MARGIN_S = 0.25
_MAX_EVENTS = 4096


def _enabled() -> bool:
    return os.environ.get("RETRY_GUARD", "1").strip() not in {"0", "false", "False"}


class RetryGuard:
    """Analyzer wrapper: bounded retry + health-gated exponential backoff.

    Drop-in for any object with the stock ``analyze(...)`` signature. Every
    unknown attribute proxies to ``inner`` so the session's duck-typed reads
    (token counters, ``_timeout``, ``animation_counters``) pass through.
    """

    def __init__(
        self,
        inner: Any,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        backoff_cap_s: float = DEFAULT_BACKOFF_CAP_S,
    ) -> None:
        self._inner = inner
        self._failure_threshold = max(1, int(failure_threshold))
        self._probe_timeout_s = max(0.1, float(probe_timeout_s))
        self._backoff_cap_s = max(_BACKOFF_BASE_S, float(backoff_cap_s))
        self._streak = 0
        self._backoff_s = 0.0
        self.events: list[tuple[Any, ...]] = []

    # -- observability (real attrs so they win over the inner proxy) ---------

    @property
    def consecutive_failures(self) -> int:
        return self._streak

    @property
    def backoff_seconds(self) -> float:
        return self._backoff_s

    # -- analyzer protocol ---------------------------------------------------

    def analyze(
        self,
        state_path: Any,
        action_num: int,
        *args: Any,
        request_timeout_seconds: float | None = None,
        should_stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        started = time.monotonic()
        # Never guard the inner call: an inner crash must propagate as stock.
        result = self._inner.analyze(
            state_path,
            action_num,
            *args,
            request_timeout_seconds=request_timeout_seconds,
            should_stop=should_stop,
            **kwargs,
        )
        try:
            if _enabled():
                self._govern(result, started, request_timeout_seconds, should_stop)
        except Exception:  # noqa: BLE001 — the guard must never break the turn
            pass
        return result

    # -- guard logic ---------------------------------------------------------

    def _govern(
        self,
        result: Any,
        started: float,
        request_timeout_seconds: float | None,
        should_stop: Any,
    ) -> None:
        if result is None or not getattr(result, "retryable_failure", False):
            self._streak = 0
            self._backoff_s = 0.0
            return
        self._streak += 1
        if self._streak < self._failure_threshold:
            return
        # The probe's socket timeout is spent inside analyze(), so it must
        # fit the same per-request budget as the backoff. With no budget
        # left, skip the probe (assume dead); the self-clamping backoff then
        # records the growing cadence at ~0 actual sleep.
        budget = self._remaining_budget(started, request_timeout_seconds)
        if budget is None or budget > 0.0:
            alive = self._probe(budget)
            self._record(("probe", alive, self._streak))
            if alive:
                # Server up: the stock 1s retry cadence is the right response
                # to a transient hiccup; pass the result through cleanly.
                return
        self._backoff(started, request_timeout_seconds, should_stop)

    @staticmethod
    def _remaining_budget(
        started: float, request_timeout_seconds: float | None
    ) -> float | None:
        """Seconds left in the per-request budget (minus the safety margin),
        or None when the session imposes no request timeout."""
        if request_timeout_seconds is None:
            return None
        try:
            return (
                float(request_timeout_seconds)
                - _TIMEOUT_MARGIN_S
                - (time.monotonic() - started)
            )
        except (TypeError, ValueError):
            return None

    def _probe(self, budget: float | None) -> bool:
        base = os.environ.get("LOCAL_ANALYZER_BASE_URL", "").strip().rstrip("/")
        if not base:
            return False
        url = f"{base}/models"
        timeout = self._probe_timeout_s
        if budget is not None:
            timeout = min(timeout, max(0.1, budget))
        # Anim-bundle adaptation: the local vLLM is served WITH an api key
        # (solver.py:1136 sets LOCAL_ANALYZER_API_KEY), and an authed vLLM
        # 401s an unauthenticated /v1/models — which urlopen raises on and
        # the except below would misread as "dead". Send the Bearer header
        # when the key exists so a healthy authed server probes alive.
        headers = {}
        api_key = os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", None) or resp.getcode())
                return 200 <= status < 300
        except Exception:  # noqa: BLE001 — any probe failure means "dead"
            return False

    def _backoff(
        self,
        started: float,
        request_timeout_seconds: float | None,
        should_stop: Any,
    ) -> None:
        self._backoff_s = (
            _BACKOFF_BASE_S
            if self._backoff_s <= 0.0
            else min(self._backoff_cap_s, self._backoff_s * 2.0)
        )
        target = self._backoff_s
        if request_timeout_seconds is not None:
            try:
                budget = (
                    float(request_timeout_seconds)
                    - _TIMEOUT_MARGIN_S
                    - (time.monotonic() - started)
                )
            except (TypeError, ValueError):
                budget = target
            target = min(target, max(0.0, budget))
        self._record(("backoff", self._backoff_s, target))

        end = time.monotonic() + target
        while True:
            if should_stop is not None:
                try:
                    if should_stop():
                        return
                except Exception:  # noqa: BLE001 — broken predicate == stop
                    return
            remaining = end - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(_SHOULD_STOP_SLICE_S, remaining))

    def _record(self, event: tuple[Any, ...]) -> None:
        self.events.append(event)
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]

    # -- transparent proxy to the inner agent --------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup misses (i.e. not a RetryGuard attr).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            inner = self.__dict__["_inner"]
        except KeyError as exc:  # during __init__, before _inner is bound
            raise AttributeError(name) from exc
        return getattr(inner, name)


def install() -> str:
    if not _enabled():
        return "retry_guard: SKIP (RETRY_GUARD=0)"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"retry_guard: SKIP (solver module missing: {exc!r})"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"retry_guard: SKIP (tool_agent module missing: {exc!r})"

    # Presence gates — every seam symbol, fail toward stock on any mismatch.
    harness = getattr(solver_mod, "HarnessSolver", None)
    if harness is None:
        return "retry_guard: SKIP (missing HarnessSolver)"
    original_make = getattr(harness, "_make_analyzer", None)
    if original_make is None:
        return "retry_guard: SKIP (missing HarnessSolver._make_analyzer)"
    if not hasattr(solver_mod, "ANALYZER_RETRY_BACKOFF_SECONDS"):
        return "retry_guard: SKIP (retry loop constant missing — seam moved)"
    turn_result = getattr(agent_mod, "AnalyzerTurnResult", None)
    if turn_result is None or not hasattr(turn_result, "retryable_failure"):
        # dataclass field check: class attr exists because it has a default
        return "retry_guard: SKIP (AnalyzerTurnResult.retryable_failure missing)"
    if not hasattr(getattr(agent_mod, "ToolAgent", None), "analyze"):
        return "retry_guard: SKIP (ToolAgent.analyze missing)"
    if getattr(original_make, "_retry_guard_patched", False):
        return "retry_guard: SKIP (already applied)"

    def make_analyzer_with_guard(self: Any, *args: Any, **kwargs: Any) -> Any:
        analyzer = original_make(self, *args, **kwargs)
        try:
            if not _enabled() or analyzer is None or isinstance(analyzer, RetryGuard):
                return analyzer
            return RetryGuard(analyzer)
        except Exception:  # noqa: BLE001 — fail open to the stock analyzer
            return analyzer

    make_analyzer_with_guard._retry_guard_patched = True  # type: ignore[attr-defined]
    # Single-namespace by design: _make_analyzer is only ever resolved via
    # ``self._make_analyzer`` (solver.py:1383), never imported by name.
    harness._make_analyzer = make_analyzer_with_guard
    return "retry_guard: OK"
