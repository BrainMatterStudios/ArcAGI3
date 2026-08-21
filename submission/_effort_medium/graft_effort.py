"""Effort-medium graft — reasoning_effort=medium on every chat request +
dead-completion retry hardening.

Motivation (docs/RESEARCH-2026-08-21-bug-lever-hunt.md Tier-1 #2, as amended
by the wave-2 CORRECTIONS: shipping arms already run temp 0.6/top_p 0.95/
top_k 20, so sampling is NOT touched here — the surviving lever is
reasoning_effort only):
- The official chat template defaults ``reasoning_effort`` to **xhigh**
  (verified by rendering our snapshot's template); the harness sends only
  ``enable_thinking``, so every scored call carries xhigh with
  max_tokens=None.
- Measured live symptom: **122 thinking-only dead completions**
  (finish_reason=stop, zero tool calls, 111 with zero content) = 2.62M
  reasoning chars ~= 650-750k tokens ~= ~6h of decode producing nothing.
- Paired hardening (same cluster): when a completion returns the dead
  signature, the NEXT request in the slice forces ``tool_choice`` to the
  ``python`` function and appends one user line:
  "Your previous reasoning produced no action — act now."

Seams (verified against the June stock tree,
scratchpad/bundles/june_stock/src/ARC3-Inference; tool_agent.py md5
7fea036d7d366b8a07dafa6d8e39a821):
- openai_compat.py:64-68 — the vllm branch sets
  ``payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}``;
  the wrapper adds ``reasoning_effort`` (setdefault: an explicit future
  value always wins) into that same dict, so the key rides only on requests
  that already carry chat_template_kwargs (vllm requests — the scored path).
- tool_agent.py:34 — ``build_chat_payload`` is imported BY NAME into
  tool_agent (and tools/chat.py:10, a dev CLI) => dual-namespace rebinding:
  the wrapper is bound into openai_compat AND tool_agent (and any already-
  imported ``inference.tools.chat``).
- tool_agent.py:1282-1301 — ``ToolAgent._chat_completion`` builds the
  payload; ``tool_choice`` comes from ``_request_tool_choice(tools)``
  (tool_agent.py:311-312, always "auto"), with no tool_choice parameter on
  ``_chat_completion`` itself — hence the force mechanism: the patched
  ``_chat_completion`` sets a thread-local flag around the inner call and
  the patched ``_request_tool_choice`` returns the named-function choice
  while it is set (games run concurrently in solver threads; thread-local
  keeps arms independent). The only ``_request_tool_choice`` call inside
  that window is the payload build at tool_agent.py:1299 (the other call
  sites, :1605 and :1789, run outside the window).
- tool_agent.py:1854-1856 + :1894-1927 — the dead-completion signature and
  the stock no-tool-call retry loop this hardens: on
  finish_reason=="stop" with zero tool calls and empty content (and no
  ``<tool_call>`` markup, which stock recovery at :1859-1861 handles
  itself), the next request is forced. The injected user line exists only
  in the wire request (the loop's own ``messages`` list is not mutated),
  and pending state is cleared at every ``analyze`` entry so the force
  never leaks across slices.

Fail-open invariants:
- Inner calls are never wrapped in try/except — crashes propagate as stock.
- All graft logic sits in blanket try/except; any error => stock behavior.
- EFFORT_MEDIUM=0 disables the kwargs injection; EFFORT_DEAD_RETRY=0
  disables the retry hardening — each checked at call time; both off at
  install time => SKIP (no patch applied).
- EFFORT_LEVEL overrides the injected level (default "medium").
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

ACT_NOW_LINE = "Your previous reasoning produced no action — act now."
FORCED_TOOL_CHOICE = {"type": "function", "function": {"name": "python"}}

_tls = threading.local()


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip() not in {"0", "false", "False"}


def _effort_enabled() -> bool:
    return _flag_enabled("EFFORT_MEDIUM")


def _dead_retry_enabled() -> bool:
    return _flag_enabled("EFFORT_DEAD_RETRY")


def _effort_level() -> str:
    return os.environ.get("EFFORT_LEVEL", "medium").strip() or "medium"


def _is_dead_completion(result: Any, agent_mod: Any) -> bool:
    """The measured signature: finish_reason=stop, zero tool calls, empty
    content. Completions carrying <tool_call> markup are NOT dead — stock
    markup recovery (tool_agent.py:1859-1861) owns those."""
    try:
        finish_reason = str(getattr(result, "finish_reason", "") or "")
        if finish_reason != "stop":
            return False
        message = getattr(result, "message", None)
        if not isinstance(message, dict):
            return False
        if message.get("tool_calls"):
            return False
        normalize = getattr(agent_mod, "_normalize_message_content", None)
        raw_content = message.get("content", "")
        content = normalize(raw_content) if callable(normalize) else str(raw_content or "")
        if str(content or "").strip():
            return False
        extract = getattr(agent_mod, "_extract_reasoning_text", None)
        reasoning = extract(message) if callable(extract) else ""
        has_markup = getattr(agent_mod, "_contains_tool_call_markup", None)
        if callable(has_markup) and has_markup(str(reasoning or ""), str(content or "")):
            return False
        return True
    except Exception:  # noqa: BLE001 — unparseable result => not dead
        return False


def install() -> str:
    if not _effort_enabled() and not _dead_retry_enabled():
        return "effort_medium: SKIP (EFFORT_MEDIUM=0 and EFFORT_DEAD_RETRY=0)"
    try:
        from inference.utils import openai_compat as compat_mod
    except Exception as exc:  # noqa: BLE001
        return f"effort_medium: SKIP (openai_compat module missing: {exc!r})"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"effort_medium: SKIP (tool_agent module missing: {exc!r})"

    original_build = getattr(compat_mod, "build_chat_payload", None)
    if original_build is None:
        return "effort_medium: SKIP (build_chat_payload missing)"
    if getattr(agent_mod, "build_chat_payload", None) is None:
        return "effort_medium: SKIP (tool_agent.build_chat_payload rebind seam missing)"
    tool_agent_cls = getattr(agent_mod, "ToolAgent", None)
    if tool_agent_cls is None:
        return "effort_medium: SKIP (missing ToolAgent)"
    original_chat = getattr(tool_agent_cls, "_chat_completion", None)
    if original_chat is None:
        return "effort_medium: SKIP (ToolAgent._chat_completion missing)"
    original_tool_choice = getattr(agent_mod, "_request_tool_choice", None)
    if original_tool_choice is None:
        return "effort_medium: SKIP (_request_tool_choice missing — force seam moved)"
    original_analyze = getattr(tool_agent_cls, "analyze", None)
    if original_analyze is None:
        return "effort_medium: SKIP (ToolAgent.analyze missing)"
    if getattr(compat_mod.build_chat_payload, "_effort_medium_patched", False):
        return "effort_medium: SKIP (already applied)"

    # --- 1. reasoning_effort alongside enable_thinking (openai_compat.py:68) ---

    def build_payload_with_effort(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_build(*args, **kwargs)
        try:
            if _effort_enabled():
                template_kwargs = payload.get("chat_template_kwargs")
                if isinstance(template_kwargs, dict) and "enable_thinking" in template_kwargs:
                    template_kwargs.setdefault("reasoning_effort", _effort_level())
        except Exception:  # noqa: BLE001 — payload shape drift => stock payload
            pass
        return payload

    # --- 2. forced tool_choice while a dead-retry request is in flight ---

    def request_tool_choice_with_force(tools: Any) -> Any:
        try:
            if tools and getattr(_tls, "force_python", False):
                return dict(FORCED_TOOL_CHOICE)
        except Exception:  # noqa: BLE001
            pass
        return original_tool_choice(tools)

    # --- 3. dead-completion detection + hardened retry request ---

    def chat_completion_with_dead_retry(self: Any, messages: Any, *args: Any, **kwargs: Any) -> Any:
        forced = False
        try:
            if _dead_retry_enabled() and getattr(self, "_eff_dead_pending", False):
                self._eff_dead_pending = False
                messages = list(messages) + [{"role": "user", "content": ACT_NOW_LINE}]
                _tls.force_python = True
                forced = True
        except Exception:  # noqa: BLE001
            forced = False
        try:
            # Never guard the inner call: crashes/RequestExceptions are stock.
            result = original_chat(self, messages, *args, **kwargs)
        finally:
            if forced:
                try:
                    _tls.force_python = False
                except Exception:  # noqa: BLE001
                    pass
        try:
            if _dead_retry_enabled():
                self._eff_dead_pending = _is_dead_completion(result, agent_mod)
        except Exception:  # noqa: BLE001
            pass
        return result

    # --- 4. pending state never leaks across slices ---

    def analyze_with_clear(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            self._eff_dead_pending = False
        except Exception:  # noqa: BLE001
            pass
        return original_analyze(self, *args, **kwargs)

    build_payload_with_effort._effort_medium_patched = True  # type: ignore[attr-defined]
    compat_mod.build_chat_payload = build_payload_with_effort
    agent_mod.build_chat_payload = build_payload_with_effort  # by-name import, tool_agent.py:34
    chat_cli = sys.modules.get("inference.tools.chat")
    if chat_cli is not None and getattr(chat_cli, "build_chat_payload", None) is not None:
        chat_cli.build_chat_payload = build_payload_with_effort  # tools/chat.py:10 (dev CLI)
    agent_mod._request_tool_choice = request_tool_choice_with_force
    tool_agent_cls._chat_completion = chat_completion_with_dead_retry
    tool_agent_cls.analyze = analyze_with_clear
    install.originals = {  # type: ignore[attr-defined]
        "build_chat_payload": original_build,
        "_request_tool_choice": original_tool_choice,
        "_chat_completion": original_chat,
        "analyze": original_analyze,
    }
    return "effort_medium: OK"
