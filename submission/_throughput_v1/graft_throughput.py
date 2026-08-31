"""Throughput graft (Pack 1, 2026-08-29) — plumbing-only changes that raise
actions-per-game of the anim-bundle ToolAgent. No prompt text changes.

Evidence: docs/research-2026-08-29/R4-harness-throughput-audit.md — every
play dies on the 7,920 s clock at ~88 actions/game; each call re-prefills a
16-20k-token prompt (prefix-cache hit 0-20%) because the trimmer drops one
block per turn; the 60 s yield is shorter than one call (26% of wall in
action-less slices); carried notes are wiped on every GAME_OVER; blind
20-140-action batches burn efficiency and trigger GAME_OVERs.

Seams (anim bundle inference/agent/tool_agent.py, the code that plays at eval):
- _trim_messages_for_context (:2056) — hysteresis: when over budget, cut to
  TP_TRIM_LOW_WATER x budget in one go so the cached prefix survives turns.
- __init__ — _context_budget_tokens / _yield_seconds / _tool_steps are set
  from module constants read at import; override per instance from
  TP_CONTEXT_WINDOW / TP_YIELD_SECONDS / TP_TOOL_STEPS.
- _update_summarized_knowledge_from_step_summary (:1343) — wipes the carried
  notes on game_over; keep them (TP_KEEP_NOTES_ON_GAME_OVER). Level-up and
  run-complete wipes are untouched.
- _run_python_tool (:1693) + _normalize_python_actions (:1614) — per tool
  call action budget (TP_BATCH_CAP): requests beyond the cap are truncated,
  and a further action() call raises ValueError inside the sandbox so the
  model sees a readable error instead of a blind 100-action walk.

Flags (read at call time; TP_ENABLE=0 turns every seam into a pass-through):
  TP_ENABLE=1  TP_TRIM_LOW_WATER=0.5  TP_CONTEXT_WINDOW=24576
  TP_YIELD_SECONDS=900  TP_TOOL_STEPS=8  TP_KEEP_NOTES_ON_GAME_OVER=1
  TP_BATCH_CAP=10

Fail-open: graft logic errors fall through to the stock method; the stock
method is never wrapped in try/except, so its own errors propagate as stock.
"""
from __future__ import annotations

import os
import threading
from typing import Any

_tls = threading.local()
_STATE = {"installed": False}
# Pack 2 hook: called as ON_CUT(agent, dropped_messages) after a hysteresis cut.
ON_CUT = None

DEFAULT_TRIM_LOW_WATER = 0.5
DEFAULT_CONTEXT_WINDOW = 24576
DEFAULT_YIELD_SECONDS = 900.0
DEFAULT_TOOL_STEPS = 8
DEFAULT_BATCH_CAP = 10
BATCH_CAP_MESSAGE = (
    "action batch cap reached: at most {cap} actions per python tool call. "
    "Observe the results so far, then call the tool again for more actions."
)
_OFF = {"0", "false", "no", "off"}


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("TP_ENABLE", "1").lower() not in _OFF


def trim_low_water() -> float:
    """Fraction of the budget to cut down to when over budget; 1.0 = stock."""
    if not enabled():
        return 1.0
    try:
        value = float(_env("TP_TRIM_LOW_WATER", str(DEFAULT_TRIM_LOW_WATER)))
    except ValueError:
        return DEFAULT_TRIM_LOW_WATER
    return min(1.0, max(0.05, value))


def context_window() -> int:
    """Context window for the request budget; 0 = leave stock."""
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))))
    except ValueError:
        return DEFAULT_CONTEXT_WINDOW


def yield_seconds() -> float:
    """Turn yield in seconds; -1 = leave stock; 0 = disable the yield."""
    if not enabled():
        return -1.0
    try:
        return float(_env("TP_YIELD_SECONDS", str(DEFAULT_YIELD_SECONDS)))
    except ValueError:
        return DEFAULT_YIELD_SECONDS


def tool_steps() -> int:
    """Calls per turn; -1 = leave stock; 0 = unlimited."""
    if not enabled():
        return -1
    try:
        return int(_env("TP_TOOL_STEPS", str(DEFAULT_TOOL_STEPS)))
    except ValueError:
        return DEFAULT_TOOL_STEPS


def tool_timeout() -> int:
    """Python sandbox timeout override; 0 = leave stock (clamped 30 s)."""
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP_TOOL_TIMEOUT", "0")))
    except ValueError:
        return 0


def keep_notes_on_game_over() -> bool:
    if not enabled():
        return False
    return _env("TP_KEEP_NOTES_ON_GAME_OVER", "1").lower() not in _OFF


def batch_cap() -> int:
    """Requested actions per python tool call; 0 = unlimited."""
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP_BATCH_CAP", str(DEFAULT_BATCH_CAP))))
    except ValueError:
        return DEFAULT_BATCH_CAP


# ------------------------------------------------------------- seam: trim ---
def _patch_trim(cls: Any) -> None:
    stock_trim = cls._trim_messages_for_context

    def trim(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        low = trim_low_water()
        if low >= 1.0 or not messages:
            return stock_trim(self, messages, tools=tools, preserve_recent=preserve_recent,
                              extra_safety_tokens=extra_safety_tokens)
        try:
            system_message = messages[0]
            history = list(messages[1:])
            preserve_recent = max(0, preserve_recent)
            budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
            estimate = self._estimate_request_input_tokens([system_message, *history], tools=tools)
            if estimate <= budget:
                return [system_message, *self._drop_until_first_user_message(history)]
            target = max(1, int(budget * low))
            original = list(history)

            def user_count(items):
                return sum(1 for m in items if str(m.get("role", "")).strip() == "user")

            while history and estimate > target:
                # Never cut away the most recent user message: a request whose
                # history is only assistant/tool turns is rejected by the
                # server ("No user query found in messages", measured live).
                if user_count(history) <= 1:
                    break
                if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                    break
                estimate = self._estimate_request_input_tokens([system_message, *history], tools=tools)
            history = self._drop_until_first_user_message(history)
            if not history and original:
                # fall back to the stock trim rather than send an empty history
                return stock_trim(self, messages, tools=tools, preserve_recent=preserve_recent,
                                  extra_safety_tokens=extra_safety_tokens)
            dropped = original[: max(0, len(original) - len(history))]
            hook = ON_CUT
            if hook is not None and dropped:
                try:
                    hook(self, dropped)
                except Exception:  # noqa: BLE001 — a summary failure never blocks the turn
                    pass
            return [system_message, *history]
        except Exception:  # noqa: BLE001 — fail open to stock
            return stock_trim(self, messages, tools=tools, preserve_recent=preserve_recent,
                              extra_safety_tokens=extra_safety_tokens)

    trim._tp_stock = stock_trim
    cls._trim_messages_for_context = trim


# ------------------------------------------------------------- seam: init ---
def _patch_init(cls: Any) -> None:
    stock_init = cls.__init__

    def init(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        try:
            window = context_window()
            if window > 0:
                self._context_budget_tokens = max(
                    1024, window - self._reply_reserve_tokens - self._request_safety_margin_tokens)
            ys = yield_seconds()
            if ys >= 0:
                self._yield_seconds = None if ys == 0 else float(ys)
            ts = tool_steps()
            if ts >= 0:
                self._tool_steps = None if ts == 0 else max(1, ts)
            tt = tool_timeout()
            if tt > 0:
                # stock clamps LOCAL_ANALYZER_TOOL_TIMEOUT at 30 s; the measured
                # cost on Flash-Next is ~130 guillotined turns per 11 games
                # (R8 forensics) — actions execute but observations are lost.
                self._python_timeout = max(1, tt)
        except Exception:  # noqa: BLE001
            pass

    init._tp_stock = stock_init
    cls.__init__ = init


# ------------------------------------------------------------ seam: notes ---
def _patch_notes(cls: Any) -> None:
    stock = cls._update_summarized_knowledge_from_step_summary

    def update(self):
        if not keep_notes_on_game_over():
            return stock(self)
        try:
            summary = self._last_step_summary
            if not summary:
                return None
            if summary.get("level_transition") or summary.get("run_complete"):
                return stock(self)
            return None  # game_over or nothing: keep every carried note
        except Exception:  # noqa: BLE001
            return stock(self)

    update._tp_stock = stock
    cls._update_summarized_knowledge_from_step_summary = update


# -------------------------------------------------------- seam: batch cap ---
def begin_tool_call() -> None:
    """Reset the per-tool-call action budget (called at every _run_python_tool)."""
    _tls.remaining = batch_cap()


def _remaining() -> int | None:
    cap = batch_cap()
    if cap <= 0:
        return None
    remaining = getattr(_tls, "remaining", None)
    if remaining is None:
        remaining = cap
        _tls.remaining = remaining
    return remaining


def _patch_batch_cap(cls: Any) -> None:
    stock_normalize = cls._normalize_python_actions
    stock_run = cls._run_python_tool

    def normalize(self, value):
        normalized = stock_normalize(self, value)
        try:
            remaining = _remaining()
        except Exception:  # noqa: BLE001
            return normalized
        if remaining is None:
            return normalized
        if remaining <= 0:
            raise ValueError(BATCH_CAP_MESSAGE.format(cap=batch_cap()))
        if len(normalized) > remaining:
            normalized = normalized[:remaining]
        _tls.remaining = remaining - len(normalized)
        return normalized

    def run(self, state_path, arguments):
        begin_tool_call()
        return stock_run(self, state_path, arguments)

    normalize._tp_stock = stock_normalize
    run._tp_stock = stock_run
    cls._normalize_python_actions = normalize
    cls._run_python_tool = run


# ------------------------------------------------------------ time guard ---
def time_guard_per_game_s(stock_per_game_s: float, *, setup_elapsed_s: float, games: int,
                          concurrency: int, total_budget_s: float = 32400.0,
                          margin_s: float = 240.0) -> float:
    """Shrink the per-game box only if setup + waves would overrun the 9 h box.

    Never grows the box; never returns less than 600 s.
    """
    try:
        waves = max(1, -(-int(games) // max(1, int(concurrency))))
        fits = (float(total_budget_s) - float(setup_elapsed_s) - float(margin_s)) / waves
        return float(min(float(stock_per_game_s), max(600.0, fits)))
    except Exception:  # noqa: BLE001
        return float(stock_per_game_s)


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "throughput: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"throughput: SKIP (tool_agent module missing: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "throughput: SKIP (missing ToolAgent)"
    for name in ("_trim_messages_for_context", "_update_summarized_knowledge_from_step_summary",
                 "_normalize_python_actions", "_run_python_tool", "_estimate_request_input_tokens",
                 "_drop_oldest_history_block", "_drop_until_first_user_message"):
        if getattr(cls, name, None) is None:
            return f"throughput: SKIP (ToolAgent.{name} missing)"
    _patch_trim(cls)
    _patch_init(cls)
    _patch_notes(cls)
    _patch_batch_cap(cls)
    _STATE["installed"] = True
    return "throughput: OK"


def status() -> dict[str, Any]:
    return {
        "installed": _STATE["installed"],
        "enabled": enabled(),
        "trim_low_water": trim_low_water(),
        "context_window": context_window(),
        "yield_seconds": yield_seconds(),
        "tool_steps": tool_steps(),
        "keep_notes_on_game_over": keep_notes_on_game_over(),
        "tool_timeout": tool_timeout(),
        "batch_cap": batch_cap(),
    }
