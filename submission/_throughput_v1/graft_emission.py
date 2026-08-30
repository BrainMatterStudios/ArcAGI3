"""Emission graft (Pack 5, 2026-08-30) — attack the two measured modal
failures of the stock loop (docs/research-2026-08-29/R7-stock-failure-
forensics-2026-08-30.md):

  G  analysis-paralysis: 63% of wall time sits in model calls that execute
     ZERO env actions; 47% of calls end at the 60 s yield without acting
     (tn36: 50 identical calls, 0 actions in 2.2 h).
  amnesia-by-channel: in 4/16 games the assistant text is empty all run, so
     the note harvest gets nothing and the model restarts from scratch every
     call (re86: 54 responses with 0 content chars). The world model lives in
     the hidden REASONING channel (Feng's 66.8% field finding).

Two independent, flag-gated behaviours (installed after graft_throughput;
TP5_ENABLE=0 = pass-through):

  (A) TP5_WM_FROM_REASONING (default 1)
      When a model response carries NO parsable note in its assistant text,
      harvest `World model:`-style labelled blocks from the reasoning text
      instead. Purely additive: the assistant channel wins when non-empty.

  (B) TP5_ACT_FLOOR (default 3)
      Track consecutive model calls WITHOUT an executed env action, across
      turns, per agent (the session's own counter — reset whenever an action
      executes). When the streak reaches the floor, the NEXT request forces
      `tool_choice` to the python function and appends one user line telling
      the model to act on its best current hypothesis. Mechanism reused from
      submission/_effort_medium/graft_effort.py (the dead-retry seam), which
      passed its offline suite; the trigger here is broader (no action
      executed, not just empty completions) and the injected line names the
      analysis streak. TP5_ACT_FLOOR=0 disables.

Fail-open: errors fall through to stock behaviour.
"""
from __future__ import annotations

import os
import threading
from typing import Any

_STATE = {"installed": False}
_RUN_STOCK: dict = {}
_OFF = {"0", "false", "no", "off"}
_tls = threading.local()

ACT_LINE = (
    "You have made {n} analyses in a row without executing any action. Analysis is no longer "
    "buying information the board can't give you faster. Call the `python` tool NOW and make it "
    "end with `action(...)` executing the best action or short batch under your current best "
    "hypothesis — acting and observing the result IS the experiment."
)
FORCED_TOOL_CHOICE = {"type": "function", "function": {"name": "python"}}


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("TP5_ENABLE", "1").lower() not in _OFF


def wm_from_reasoning() -> bool:
    return enabled() and _env("TP5_WM_FROM_REASONING", "1").lower() not in _OFF


def act_floor() -> int:
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP5_ACT_FLOOR", "3")))
    except ValueError:
        return 3


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "enabled": enabled(),
            "wm_from_reasoning": wm_from_reasoning(), "act_floor": act_floor()}


def _calls_without_action(agent: Any) -> int:
    return int(getattr(agent, "_tp5_calls_without_action", 0) or 0)


def install() -> str:
    if _STATE["installed"]:
        return "emission: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
        from inference.utils import openai_compat as compat_mod
    except Exception as exc:  # noqa: BLE001
        return f"emission: SKIP (import failed: {exc!r})"
    agent_cls = getattr(agent_mod, "ToolAgent", None)
    if agent_cls is None:
        return "emission: SKIP (missing ToolAgent)"
    for name in ("_update_summarized_knowledge_from_assistant", "_chat_completion",
                 "_run_python_tool", "_extract_scientist_note"):
        if getattr(agent_cls, name, getattr(agent_mod, name, None)) is None:
            return f"emission: SKIP ({name} missing)"
    if getattr(agent_mod, "build_chat_payload", None) is None:
        return "emission: SKIP (build_chat_payload rebind seam missing)"

    # (A) note harvest falls back to the reasoning channel ------------------
    stock_chat = agent_cls._chat_completion

    def chat(self, messages, **kwargs):
        # act-floor: arm the forced tool choice for THIS request when the
        # analysis streak has reached the floor
        floor = act_floor()
        force = floor > 0 and _calls_without_action(self) >= floor
        _tls.force_now = force
        _tls.act_line_n = _calls_without_action(self)
        try:
            if force:
                messages = list(messages) + [{"role": "user", "content": ACT_LINE.format(n=_tls.act_line_n)}]
            result = stock_chat(self, messages, **kwargs)
        finally:
            _tls.force_now = False
        try:
            self._tp5_calls_without_action = _calls_without_action(self) + 1
            if wm_from_reasoning():
                message = getattr(result, "message", None)
                if isinstance(message, dict):
                    content = agent_mod._normalize_message_content(message.get("content", ""))
                    if not agent_mod._extract_scientist_note(str(content or "")):
                        reasoning = agent_mod._extract_reasoning_text(message)
                        note = agent_mod._extract_scientist_note(str(reasoning or ""))
                        if note:
                            for key, value in note.items():
                                if value:
                                    self._summarized_knowledge[key] = value
        except Exception:  # noqa: BLE001
            pass
        return result

    chat._tp5_stock = stock_chat
    agent_cls._chat_completion = chat

    # forced tool choice rides build_chat_payload only while armed ----------
    stock_build = compat_mod.build_chat_payload

    def build(*args, **kwargs):
        try:
            if getattr(_tls, "force_now", False) and kwargs.get("tools"):
                kwargs = dict(kwargs)
                kwargs["tool_choice"] = FORCED_TOOL_CHOICE
        except Exception:  # noqa: BLE001
            pass
        return stock_build(*args, **kwargs)

    build._tp5_stock = stock_build
    compat_mod.build_chat_payload = build
    agent_mod.build_chat_payload = build  # dual-namespace rebind (imported by name)

    # streak reset: an executed action clears the counter -------------------
    _RUN_STOCK["fn"] = agent_cls._run_python_tool

    def run_tool(self, state_path, arguments):
        result = _RUN_STOCK["fn"](self, state_path, arguments)
        try:
            if getattr(result, "step_executed", False):
                self._tp5_calls_without_action = 0
        except Exception:  # noqa: BLE001
            pass
        return result

    run_tool._tp5_stock = _RUN_STOCK["fn"]
    agent_cls._run_python_tool = run_tool

    _STATE["installed"] = True
    return "emission: OK"
