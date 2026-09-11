"""graft_noreason.py — keep the TURNS, drop the reasoning text, buy the slots.

WHAT THE PREVIOUS TWO ARMS ESTABLISHED
  Shrinking the context window buys calls (55.8 -> 81.8 at 16k, -> 108.4 at 12k) and
  destroys the agent (actions/call 2.76 -> 1.35 -> 0.27; levels 39.33 -> 21 -> 12).
  The measured cause is TURN DEPTH: base retains ~12.6 turns, ctx16 only ~5.2, because a
  retained turn costs ~2,217 tokens and ~1,300 of those are the assistant's REASONING.

THE IDEA
  Pay for turns in a different currency. Strip `reasoning` from messages as they enter
  RETAINED history and a turn costs ~917 tokens instead of ~2,217 -- 2.4x cheaper. At a
  16,384 window that restores ~12.5 retained turns, the SAME history depth as the 32,768
  base, at HALF the sequence length. Same turns, twice the running slots, twice the calls.

WHAT IS AND IS NOT LOST
  The CURRENT turn keeps its reasoning: the model's own chain of thought is in the
  response it just produced and in the in-flight message list. Only PAST turns lose the
  reasoning text; their content, tool calls and tool results all survive.

  Alone this graft buys NOTHING -- the trimmer refills the budget, so you simply retain
  more turns at the same sequence length. It is a lever only when paired with a smaller
  window, which is why the arm sets both.

THE HONEST COUNTER-EVIDENCE, recorded before the run
  OpenAI's public ARC-AGI-3 result attributes 13.3% -> 38.3% to RETAINED REASONING plus
  compaction. If that transfers, this graft removes the single most valuable thing in the
  context and the arm should read badly. Our own A1 probe separately established that the
  stock already carries reasoning on every retained turn (1,268 request pairs, slope 1.0)
  -- so this is a real subtraction, not a no-op. That tension is exactly the point: nobody
  has measured what retained reasoning is worth when its cost is paid in CALLS.
"""
from __future__ import annotations

import threading
from typing import Any

_STATE: dict[str, Any] = {"installed": False, "stripped": 0, "messages_seen": 0}
_LOCK = threading.Lock()


def status() -> dict[str, Any]:
    return dict(_STATE)


def _strip(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    n = 0
    for m in messages:
        if str(m.get("role", "")).strip() == "assistant" and m.get("reasoning"):
            m = {k: v for k, v in m.items() if k != "reasoning"}
            # an assistant message with neither content nor tool_calls would be malformed
            # once reasoning is gone; give it an empty string rather than a null.
            if m.get("content") is None and not m.get("tool_calls"):
                m["content"] = ""
            n += 1
        out.append(m)
    if n:
        with _LOCK:
            _STATE["stripped"] += n
    with _LOCK:
        _STATE["messages_seen"] += len(messages)
    return out


def install() -> str:
    if _STATE["installed"]:
        return "noreason: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"noreason: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None or getattr(cls, "_persistent_history_messages", None) is None:
        return "noreason: SKIP (_persistent_history_messages missing)"

    stock = cls._persistent_history_messages

    def _persistent_history_messages(self, messages, *, tools=None):
        # strip on the way OUT, so the trimmer still sees true token costs on the way in
        # and the current turn's in-flight messages are untouched
        return _strip(stock(self, messages, tools=tools))

    cls._persistent_history_messages = _persistent_history_messages
    _STATE["installed"] = True
    return "noreason: _persistent_history_messages: OK"
