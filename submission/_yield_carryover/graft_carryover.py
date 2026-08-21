"""Yield-carryover graft — slice-digest carry-forward + slice cap on yield-resume.

Motivation (live-measured, docs/RESEARCH-2026-08-21-bug-lever-hunt.md Tier-1 #1):
2,234 turn-slices audited; 43.3% of slices ended with NO env action
("Yielded control to solver: turn_time_budget") and 51.5% of ALL wall
(130,631s/253,830s) elapsed in slices that ended without an action. On
yield-resume the turn conversation is rebuilt with only ~2-4 messages
(history_messages 14->11->4->2 inside one turn) — every tool result and all
reasoning from the previous slice is discarded, the model re-grounds from
zero, and 71 byte-identical python snippets were re-issued within single
turns (worst turn: 35 slices; 18 slices/3,584s/118k tokens for ONE action).

Two mechanisms, separately flagged:

1. YIELD_CARRYOVER (default on): when a slice ends yielded-without-action,
   build a bounded structured digest of that slice — every tool call's code
   (first ~200 chars) + its result (first ~400 chars) + the last assistant
   reasoning tail (~500 chars) — and inject it exactly once as a block inside
   the resumed slice's initial user prompt ("Previous slice this turn already
   ran ..."). Empty yielded slices (yield fired before any request) keep the
   previous slice's digest instead of erasing it.
2. YIELD_SLICE_CAP (default "3", 0 disables): counts slices per turn
   (turn = same ``analysis_step`` re-entered after yielded_control, the
   solver's resume contract). From slice 3 onward the resumed slice's user
   prompt is prefixed with a FINAL-SLICE instruction: no further exploratory
   analysis, the response must end in an ``action(...)`` call. The cap is
   behavioral (an instruction, not a synthetic action) — the harness never
   fabricates actions on the model's behalf.

Seams (verified against the June stock tree,
scratchpad/bundles/june_stock/src/ARC3-Inference; tool_agent.py md5
7fea036d7d366b8a07dafa6d8e39a821, solver.py md5
7c2840743245402a55f8f62485752b5e — byte-identical to the public
jeroencottaar June source share and the banking_v22 bundle):

- solver.py:311-322 — the resume contract this graft keys on:
  ``if getattr(result, "yielded_control", False): retry_analysis_step =
  analysis_step; continue`` — the re-entry calls ``analyze`` with the SAME
  ``analysis_step`` (chosen at solver.py:284-288, passed at solver.py:299).
- tool_agent.py:1777-1778 — the yield trigger ("turn_time_budget" once
  ``self._yield_seconds`` elapses in a slice).
- tool_agent.py:2015-2019 — the context wipe: ``finally:`` rebinds
  ``self._history_messages = self._persistent_history_messages(messages,...)``
  where ``messages`` is the FULL slice conversation. That call is our digest
  source: the patched ``_persistent_history_messages`` stashes ``messages``
  before delegating, so the digest sees everything the trim throws away
  (tool_agent.py:1653-1670 keeps <=30 assistant turns, then the token trim at
  tool_agent.py:1672-1690 under LOCAL_ANALYZER_CONTEXT_WINDOW=32768 produces
  the observed 2-4 survivors).
- tool_agent.py:1727-1733 — the slice's initial user prompt is built once per
  ``analyze`` call by ``_build_user_prompt`` (def at tool_agent.py:1161);
  the patched ``_build_user_prompt`` records its first 80 chars as the
  slice-boundary marker (the slice's own messages are everything after the
  last user message starting with it) and applies any pending injection —
  hence "injected exactly once".
- tool_agent.py:1706-1717 — ``ToolAgent.analyze`` signature
  (``analysis_step`` keyword at :1713); the wrapper's turn key is
  ``(str(state_path), analysis_step)``.
- tool_agent.py:2059-2062 / dataclass at :369-374 — ``AnalyzerTurnResult``
  carries ``yielded_control`` + ``step_executed``, the post-call signal for
  "this slice yielded without acting".

Envelope note: see ENVELOPE.md — neither mechanism can extend run duration
(the digest only adds bounded prompt tokens; the cap only shortens turns).

Fail-open invariants:
- The inner ``analyze`` / ``_persistent_history_messages`` /
  ``_build_user_prompt`` calls are never wrapped in try/except — an inner
  crash propagates exactly as stock.
- All graft logic (turn tracking, digest build, injection) sits inside
  blanket try/except; any error means "no injection, stock behavior".
- Both flags checked at call time: flags off => the wrappers are pure
  pass-throughs (byte-identical prompts and results).
- install() presence-gates every seam symbol and fails toward stock.
"""

from __future__ import annotations

import json
import os
from typing import Any

DIGEST_CODE_CHARS = 200
DIGEST_RESULT_CHARS = 400
DIGEST_REASONING_CHARS = 500
DIGEST_MAX_ENTRIES = 10
DIGEST_MAX_CHARS = 6000
_HEAD_MARKER_CHARS = 80

DIGEST_HEADER = "Previous slice this turn already ran (carried over so you do not repeat work):"
DIGEST_FOOTER = (
    "Do not re-run identical inspections; build on these carried-over results and move toward an action."
)
FINAL_SLICE_INSTRUCTION = (
    "FINAL SLICE FOR THIS TURN: you have used the allowed analysis slices for this turn without acting. "
    "Do not run further exploratory analysis. Respond with a single `python` tool call whose code ends by "
    "calling `action(actions)` with the best valid action or ordered batch you currently have."
)


def _flag_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in {"0", "false", "False"}


def _carryover_enabled() -> bool:
    return _flag_enabled("YIELD_CARRYOVER")


def _slice_cap() -> int:
    """Configured cap (slices per turn); 0 disables the cap mechanism."""
    raw = os.environ.get("YIELD_SLICE_CAP", "3").strip()
    if raw in {"", "false", "False"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} ...[+{len(text) - limit} chars]"


def _message_text(message: dict[str, Any]) -> str:
    """Text of a message whose content is either a string or a parts list."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return ""


def _tool_call_code(tool_call: Any) -> str:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    raw_arguments = function.get("arguments", "")
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except (ValueError, TypeError):
            return raw_arguments
    elif isinstance(raw_arguments, dict):
        parsed = raw_arguments
    else:
        return str(raw_arguments)
    if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
        return parsed["code"]
    return json.dumps(parsed, ensure_ascii=True)


def build_slice_digest(
    messages: list[dict[str, Any]] | None,
    head_marker: str | None,
    *,
    code_chars: int = DIGEST_CODE_CHARS,
    result_chars: int = DIGEST_RESULT_CHARS,
    reasoning_chars: int = DIGEST_REASONING_CHARS,
    max_entries: int = DIGEST_MAX_ENTRIES,
    max_chars: int = DIGEST_MAX_CHARS,
) -> str:
    """Bounded digest of the yielded slice's own messages.

    The slice boundary is the LAST user message whose text starts with
    ``head_marker`` (the slice's initial user prompt, recorded by the patched
    ``_build_user_prompt``); everything after it is this slice's work. Returns
    "" when there is nothing to carry (the empty-yield case).
    """
    if not messages or not head_marker:
        return ""
    boundary = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip() != "user":
            continue
        if _message_text(message).startswith(head_marker):
            boundary = index
            break
    if boundary is None:
        return ""

    entries: list[dict[str, Any]] = []
    last_reasoning = ""
    for message in messages[boundary + 1 :]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        if role == "assistant":
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                last_reasoning = reasoning
            for tool_call in message.get("tool_calls") or []:
                entries.append(
                    {
                        "id": tool_call.get("id") if isinstance(tool_call, dict) else None,
                        "code": _truncate(_tool_call_code(tool_call), code_chars),
                        "result": None,
                    }
                )
        elif role == "tool":
            result_text = _truncate(_message_text(message) or str(message.get("content", "")), result_chars)
            call_id = message.get("tool_call_id")
            target = None
            for entry in reversed(entries):
                if entry["result"] is None and (entry["id"] == call_id or call_id in (None, "")):
                    target = entry
                    break
            if target is None:
                for entry in reversed(entries):
                    if entry["result"] is None:
                        target = entry
                        break
            if target is not None:
                target["result"] = result_text
            else:
                entries.append({"id": call_id, "code": None, "result": result_text})

    entries = entries[-max_entries:]
    if not entries and not last_reasoning:
        return ""

    lines = [DIGEST_HEADER]
    for number, entry in enumerate(entries, start=1):
        if entry["code"] is not None:
            lines.append(f"[{number}] python: {entry['code']}")
        else:
            lines.append(f"[{number}] python: (code unavailable)")
        if entry["result"] is not None:
            lines.append(f"    result: {entry['result']}")
        else:
            lines.append("    result: (no result captured before yield)")
    if last_reasoning:
        tail = last_reasoning.strip()[-reasoning_chars:]
        lines.append(f"Last reasoning tail: {tail}")
    lines.append(DIGEST_FOOTER)

    digest = "\n".join(lines)
    while len(digest) > max_chars and len(lines) > 3:
        # Drop the oldest entry line pair (after the header) until bounded.
        del lines[1 : 3 if lines[2].startswith("    result:") else 2]
        digest = "\n".join(lines)
    return digest


def install() -> str:
    if not _carryover_enabled() and _slice_cap() <= 0:
        return "yield_carryover: SKIP (YIELD_CARRYOVER=0 and YIELD_SLICE_CAP=0)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"yield_carryover: SKIP (tool_agent module missing: {exc!r})"

    tool_agent_cls = getattr(agent_mod, "ToolAgent", None)
    if tool_agent_cls is None:
        return "yield_carryover: SKIP (missing ToolAgent)"
    original_analyze = getattr(tool_agent_cls, "analyze", None)
    original_persist = getattr(tool_agent_cls, "_persistent_history_messages", None)
    original_build_prompt = getattr(tool_agent_cls, "_build_user_prompt", None)
    if original_analyze is None:
        return "yield_carryover: SKIP (ToolAgent.analyze missing)"
    if original_persist is None:
        return "yield_carryover: SKIP (ToolAgent._persistent_history_messages missing — wipe seam moved)"
    if original_build_prompt is None:
        return "yield_carryover: SKIP (ToolAgent._build_user_prompt missing — inject seam moved)"
    turn_result = getattr(agent_mod, "AnalyzerTurnResult", None)
    if turn_result is None or not hasattr(turn_result, "yielded_control"):
        return "yield_carryover: SKIP (AnalyzerTurnResult.yielded_control missing)"
    if getattr(original_analyze, "_yield_carryover_patched", False):
        return "yield_carryover: SKIP (already applied)"

    # --- digest source: stash the FULL slice conversation at the wipe seam ---

    def persist_with_stash(self: Any, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        try:
            if _carryover_enabled() or _slice_cap() > 0:
                self._yc_full_slice_messages = list(messages)
        except Exception:  # noqa: BLE001 — stash must never break the turn
            pass
        return original_persist(self, messages, *args, **kwargs)

    # --- injection point: the slice's initial user prompt ---

    def build_prompt_with_injection(self: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original_build_prompt(self, *args, **kwargs)
        try:
            if not (_carryover_enabled() or _slice_cap() > 0):
                return prompt
            self._yc_slice_head = prompt[:_HEAD_MARKER_CHARS]
            pending = getattr(self, "_yc_pending_inject", None)
            if not isinstance(pending, dict):
                return prompt
            self._yc_pending_inject = None  # consume: injected exactly once
            digest = pending.get("digest")
            if pending.get("final") and FINAL_SLICE_INSTRUCTION not in prompt:
                prompt = f"{FINAL_SLICE_INSTRUCTION}\n\n{prompt}"
            if digest and DIGEST_HEADER not in prompt:
                prompt = f"{prompt}\n\n{digest}"
        except Exception:  # noqa: BLE001 — injection failure => stock prompt
            pass
        return prompt

    # --- keep exactly ONE injection alive: scrub old blocks from history ---

    def _strip_injections(text: str) -> str:
        index = text.find(DIGEST_HEADER)
        if index != -1:
            start = text.rfind("\n\n", 0, index)
            start = start if start != -1 else index
            end = text.find(DIGEST_FOOTER, index)
            end = end + len(DIGEST_FOOTER) if end != -1 else len(text)
            text = text[:start] + text[end:]
        if text.startswith(FINAL_SLICE_INSTRUCTION):
            text = text[len(FINAL_SLICE_INSTRUCTION) :].lstrip("\n")
        return text

    def _scrub_history(self: Any) -> None:
        """Remove digest/final blocks from carried user messages so a resumed
        slice never sees two copies (and the token cost never compounds)."""
        history = getattr(self, "_history_messages", None)
        if not isinstance(history, list):
            return
        for message in history:
            if not isinstance(message, dict) or str(message.get("role", "")).strip() != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and (DIGEST_HEADER in content or content.startswith(FINAL_SLICE_INSTRUCTION)):
                message["content"] = _strip_injections(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = str(part.get("text", ""))
                        if DIGEST_HEADER in text or text.startswith(FINAL_SLICE_INSTRUCTION):
                            part["text"] = _strip_injections(text)

    # --- turn/slice tracking around analyze ---

    def _pre_analyze(self: Any, state_path: Any, kwargs: dict[str, Any]) -> None:
        _scrub_history(self)
        analysis_step = kwargs.get("analysis_step")
        key = (str(state_path), analysis_step)
        previous_key = getattr(self, "_yc_key", None)
        slices_done = int(getattr(self, "_yc_slices", 0) or 0)
        is_resume = analysis_step is not None and key == previous_key and slices_done >= 1
        if not is_resume:
            self._yc_key = key
            self._yc_slices = 0
            self._yc_digest = None
            self._yc_pending_inject = None
            return
        slice_number = slices_done + 1
        inject: dict[str, Any] = {}
        if _carryover_enabled() and getattr(self, "_yc_digest", None):
            inject["digest"] = self._yc_digest
        cap = _slice_cap()
        if cap > 0 and slice_number >= cap:
            inject["final"] = True
        self._yc_pending_inject = inject or None

    def _post_analyze(self: Any, result: Any) -> None:
        self._yc_pending_inject = None  # never let an unconsumed block leak forward
        yielded = result is not None and bool(getattr(result, "yielded_control", False))
        acted = result is not None and bool(getattr(result, "step_executed", False))
        if yielded and not acted:
            self._yc_slices = int(getattr(self, "_yc_slices", 0) or 0) + 1
            if _carryover_enabled():
                digest = build_slice_digest(
                    getattr(self, "_yc_full_slice_messages", None),
                    getattr(self, "_yc_slice_head", None),
                )
                if digest:
                    # Empty yielded slices keep the previous digest alive.
                    self._yc_digest = digest
            return
        self._yc_key = None
        self._yc_slices = 0
        self._yc_digest = None

    def analyze_with_carryover(self: Any, state_path: Any, action_num: int, *args: Any, **kwargs: Any) -> Any:
        enabled = _carryover_enabled() or _slice_cap() > 0
        if enabled:
            try:
                _pre_analyze(self, state_path, kwargs)
            except Exception:  # noqa: BLE001
                pass
        # Never guard the inner call: an inner crash must propagate as stock.
        result = original_analyze(self, state_path, action_num, *args, **kwargs)
        if enabled:
            try:
                _post_analyze(self, result)
            except Exception:  # noqa: BLE001
                pass
        return result

    analyze_with_carryover._yield_carryover_patched = True  # type: ignore[attr-defined]
    tool_agent_cls._persistent_history_messages = persist_with_stash
    tool_agent_cls._build_user_prompt = build_prompt_with_injection
    tool_agent_cls.analyze = analyze_with_carryover
    # Expose originals for tests / debugging (single-namespace: methods are
    # only ever resolved via ``self.`` — no by-name imports of these symbols).
    install.originals = {  # type: ignore[attr-defined]
        "analyze": original_analyze,
        "_persistent_history_messages": original_persist,
        "_build_user_prompt": original_build_prompt,
    }
    return "yield_carryover: OK"
