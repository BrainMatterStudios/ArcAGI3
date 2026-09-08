"""Compaction instead of eviction (CARRY, Track A1 of the 2026-09-08 plan).

WHAT WAS VERIFIED FIRST (docs/research-2026-09-08/R-reasoning-carriage-0908.md):
the stock duck ALREADY carries the model's prior-turn reasoning. Every assistant
message the harness keeps in `_history_messages` carries a "reasoning" field
(tool_agent.py `assistant_message["reasoning"] = reasoning`); the pinned vLLM
build copies a request-side `reasoning` onto `reasoning_content`, and the
Qwen3.8-Flash-Next chat template renders `<think>reasoning</think>` for EVERY
assistant message (`preserve_thinking` is undefined -> true), not only after the
last user message. On two rig waves (947 within-turn + 321 cross-turn request
pairs) the server's prompt_tokens delta equals tokenized reasoning + content +
tool result to within the fixed template overhead (slope 0.999-1.000). So the
"carry" half of A1 needs no re-injection; it is MEASURED here per call
(reasoning messages / chars in each request).

WHAT IS LOST is everything the trimmer EVICTS: `_trim_messages_for_context`
drops the oldest history block whenever the estimated request exceeds
`_context_budget_tokens` (31,744); on the rig that fired on 27 % of consecutive
requests, the window fills after ~10 turns, and everything older survives only
as the one-line "World model:" note. This graft replaces eviction with a
model-written COMPACTION: when the trimmer must drop, it drops a CHUNK (down to
CARRY_TARGET_FRACTION of the budget; 0.5 => ~7 compactions per 52-call game, each
a queued call, vs ~11 at 0.65 — fewer, bigger compactions are cheaper in a
queue-bound regime), asks the model (one extra call, no tools,
thinking off by default, max_tokens capped) to fold the dropped turns into a
persistent "compacted knowledge" block, and keeps that block appended to the
SYSTEM message of every later request (single copy; the system message is
rebuilt per request and never persisted). The hard budget is still enforced by
the stock trimmer afterwards.

WHERE IT HOOKS (verified identical in both bundles):
  * ToolAgent._trim_messages_for_context(messages, *, tools, preserve_recent,
    extra_safety_tokens) — called at turn start, before every request, on the
    context-overflow retry (extra_safety_tokens > 0: stock eviction only) and by
    _persistent_history_messages at turn end. The wrapper (a) rebuilds
    messages[0] = system prompt + current compacted block, (b) if over budget,
    evicts to the target with the STOCK _drop_oldest_history_block, (c) compacts
    the dropped messages, (d) returns the stock trim of the result.
  * ToolAgent._chat_completion(messages, **kw) — after the stock call: counts
    the request's assistant messages that carry reasoning, the reasoning chars,
    whether the compacted block is present, and usage.prompt_tokens (the
    window read); writes one [CARRY-CALL] marker per call.
  * ToolAgent.analyze — per-turn state (game key, transcript path, level,
    request timeout).

FLAGS (read at call time; CARRY_ENABLE=0 or the graft not installed = stock
bytes' behaviour, byte-identical transcript / prompts / history):
  CARRY_ENABLE [1 once installed], CARRY_TARGET_FRACTION [0.5],
  CARRY_SUMMARY_CHARS [4800], CARRY_INPUT_CHARS [48000],
  CARRY_COMPACT_MAX_TOKENS [1500], CARRY_COMPACT_THINKING [0],
  CARRY_MIN_DROP_MSGS [2], CARRY_TIMEOUT_S [600], CARRY_WINDOW_TOKENS [32768].

TELEMETRY: transcript sections "[HARNESS CARRY]" holding
  "[CARRY-CALL] game=<stem> turn=<n> req=<i> msgs=<N> reasoning_msgs=<R>
   reasoning_chars=<C> summary_chars=<S> prompt_tokens=<P> completion_tokens=<Q>"
  "[CARRY-COMPACT] game=<stem> turn=<n> dropped_msgs=<k> input_chars=<c>
   summary_chars=<s> prompt_tokens=<p> completion_tokens=<q> e2e_s=<t> ok=1|0 [err=<why>]"
   followed by "SUMMARY:" and the block text when ok=1.
status(): calls_total, calls_with_summary, reasoning_msgs_total,
reasoning_chars_total, prompt_tokens_total, prompt_tokens_max,
prompt_over_window, compactions, compaction_failures, dropped_msgs_total,
dropped_chars_total, summary_chars_total, compaction_prompt_tokens,
compaction_completion_tokens, compaction_e2e_s, turns_total, per_game,
errors, skips (small_drop, overflow_path, no_time, keep_last_user,
no_user_after_trim).

Conventions (graft_probe): module-level _STATE/_STOCK, install() rebinds
ToolAgent methods only, fail-open try/except around every graft branch (an
exception returns the stock result), no threads, no file writes outside the
transcript. install() -> "carry: OK" / "carry: SKIP (...)".
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

_PER_GAME_KEYS = ("calls_total", "calls_with_summary", "reasoning_msgs_total", "reasoning_chars_total",
                  "prompt_tokens_total", "prompt_tokens_max", "prompt_over_window", "compactions",
                  "compaction_failures", "dropped_msgs_total", "dropped_chars_total", "summary_chars_total",
                  "compaction_prompt_tokens", "compaction_completion_tokens", "compaction_e2e_s", "turns_total")
_MAX_KEYS = {"prompt_tokens_max"}
_STATE: dict[str, Any] = {"installed": False, "errors": 0, "skips": {}, "per_game": {}}
for _k in _PER_GAME_KEYS:
    _STATE[_k] = 0
_STOCK: dict[str, Any] = {}
_LOCK = threading.Lock()
_OFF = {"0", "false", "no", "off"}

DEFAULT_TARGET_FRACTION = 0.5
DEFAULT_SUMMARY_CHARS = 4800
DEFAULT_INPUT_CHARS = 48000
DEFAULT_COMPACT_MAX_TOKENS = 1500
DEFAULT_COMPACT_THINKING = 0
DEFAULT_MIN_DROP_MSGS = 2
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_WINDOW_TOKENS = 32768
MIN_TIME_FOR_COMPACTION_S = 30.0

TRANSCRIPT_LABEL = "HARNESS CARRY"
CALL_MARK = "[CARRY-CALL]"
COMPACT_MARK = "[CARRY-COMPACT]"
SUMMARY_HEADER = "# Compacted knowledge from your earlier turns"
SUMMARY_INTRO = (
    f"{SUMMARY_HEADER} (the harness removed those turns from the context and preserved this block; "
    "it is your own summary of what you verified, refuted, and planned - trust it unless the current frame contradicts it):"
)
COMPACT_SYSTEM_HEAD = "You are the same agent that played the turns below"
COMPACT_SYSTEM = (
    COMPACT_SYSTEM_HEAD + " in an ARC-AGI-3 grid game. The harness is about to remove them from your context "
    "window. Compress everything you learned into a compact knowledge block you will rely on in later turns.\n"
    "Use exactly these headings, each followed by terse bullet points (omit a heading only if it has nothing):\n"
    "MECHANICS VERIFIED (action -> observed effect, with the evidence)\n"
    "HYPOTHESES REFUTED (what was tried, what happened - do not retry)\n"
    "GOAL MODEL (what seems to complete the level, and why)\n"
    "KEY OBJECTS AND COORDINATES (colors, shapes, positions, counts that matter)\n"
    "TRIED ON THIS LEVEL (action sequences and their results)\n"
    "OPEN QUESTIONS\n"
    "CURRENT PLAN (the next concrete test or move sequence)\n"
    "EARLIER LEVELS (at most 3 lines)\n"
    "Keep every item from the previous compacted knowledge that is still true; merge new evidence into it instead of "
    "repeating it. Be concrete: action names, colors, row/col coordinates, counts. Plain text only, no code fences, "
    "no preamble, at most {words} words."
)
# lines of the stock user prompt worth keeping when a dropped turn is rendered for the compaction call
_KEEP_USER_PREFIXES = ("The code executed", "Executed actions", "You have progressed", "You have completed",
                       "You are still on", "The game is over", "No previous", "Current state:", "Valid actions")
_NOTE_START = "Working world model carried from earlier turns:"
_NOTE_END = "end of world model"
_FOLLOWUP_HEAD = "You have not acted yet"
_NOTOOL_HEAD = "You did not call a tool"
REASONING_HEAD_CHARS, REASONING_TAIL_CHARS = 3500, 1500
CODE_CHARS, TOOL_RESULT_CHARS, USER_CHARS = 1500, 1000, 900


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("CARRY_ENABLE", "1").lower() not in _OFF


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(_env(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(_env(name, str(default)))))
    except ValueError:
        return default


def target_fraction() -> float:
    return _env_float("CARRY_TARGET_FRACTION", DEFAULT_TARGET_FRACTION, 0.2, 0.95)


def summary_chars() -> int:
    return _env_int("CARRY_SUMMARY_CHARS", DEFAULT_SUMMARY_CHARS, 200)


def input_chars() -> int:
    return _env_int("CARRY_INPUT_CHARS", DEFAULT_INPUT_CHARS, 2000)


def compact_max_tokens() -> int:
    return _env_int("CARRY_COMPACT_MAX_TOKENS", DEFAULT_COMPACT_MAX_TOKENS, 64)


def compact_thinking() -> bool:
    return _env("CARRY_COMPACT_THINKING", str(DEFAULT_COMPACT_THINKING)).lower() not in _OFF


def min_drop_msgs() -> int:
    return _env_int("CARRY_MIN_DROP_MSGS", DEFAULT_MIN_DROP_MSGS, 1)


def timeout_s() -> float:
    return _env_float("CARRY_TIMEOUT_S", DEFAULT_TIMEOUT_S, 5.0, 3600.0)


def window_tokens() -> int:
    return _env_int("CARRY_WINDOW_TOKENS", DEFAULT_WINDOW_TOKENS, 1024)


def status() -> dict[str, Any]:
    with _LOCK:
        out: dict[str, Any] = {
            "installed": _STATE["installed"], "enabled": enabled(),
            "target_fraction": target_fraction(), "summary_chars_cap": summary_chars(), "input_chars_cap": input_chars(),
            "compact_max_tokens": compact_max_tokens(), "compact_thinking": compact_thinking(),
            "min_drop_msgs": min_drop_msgs(), "window_tokens": window_tokens(),
        }
        for key in _PER_GAME_KEYS:
            out[key] = _STATE[key]
        out["reasoning_msgs_per_call"] = _share(_STATE["reasoning_msgs_total"], _STATE["calls_total"])
        out["reasoning_chars_per_call"] = _share(_STATE["reasoning_chars_total"], _STATE["calls_total"])
        out["prompt_tokens_per_call"] = _share(_STATE["prompt_tokens_total"], _STATE["calls_total"])
        out["calls_with_summary_share"] = _share(_STATE["calls_with_summary"], _STATE["calls_total"])
        out["summary_chars_mean"] = _share(_STATE["summary_chars_total"], _STATE["compactions"])
        out["compaction_e2e_s_mean"] = _share(_STATE["compaction_e2e_s"], _STATE["compactions"] + _STATE["compaction_failures"])
        out["errors"] = _STATE["errors"]
        out["skips"] = dict(_STATE["skips"])
        out["per_game"] = {k: dict(v) for k, v in _STATE["per_game"].items()}
        return out


def _share(a: float, b: float) -> float | None:
    return None if not b else a / b


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] = _STATE["skips"].get(reason, 0) + 1


def _error() -> None:
    with _LOCK:
        _STATE["errors"] += 1


def _bump(game: str, key: str, n: float = 1) -> None:
    with _LOCK:
        pg = _STATE["per_game"].setdefault(game, {k: 0 for k in _PER_GAME_KEYS})
        if key in _MAX_KEYS:
            _STATE[key] = max(_STATE[key], n)
            pg[key] = max(pg.get(key, 0), n)
        else:
            _STATE[key] += n
            pg[key] = pg.get(key, 0) + n


# ------------------------------------------------------------ per-game state ---
class CarryState:
    """One per agent (= per game run); reset when the runtime dir (game) changes."""

    def __init__(self) -> None:
        self.runtime_dir: Any = None
        self.game: str = "?"
        self.transcript_path: Path | None = None
        self.state_path: Any = None
        self.agent_mod: Any = None
        self.turn: int = 0
        self.turns_seen: int = 0
        self.in_turn: bool = False
        self.req_in_turn: int = 0
        self.level: Any = None
        self.request_timeout: float | None = None
        self.summary: str = ""              # the compacted knowledge block (model-written)
        self.compactions: int = 0


def _cstate(agent: Any, state_path: Any = None) -> CarryState:
    st = getattr(agent, "_carry", None)
    if st is None:
        st = CarryState()
        try:
            agent._carry = st
        except Exception:  # noqa: BLE001
            pass
    if state_path is not None:
        try:
            runtime_dir = Path(state_path).parent
            if st.runtime_dir is not None and st.runtime_dir != runtime_dir:
                fresh = CarryState()
                fresh.runtime_dir = runtime_dir
                try:
                    agent._carry = fresh
                except Exception:  # noqa: BLE001
                    pass
                return fresh
            st.runtime_dir = runtime_dir
        except Exception:  # noqa: BLE001
            pass
    return st


def _game_key(step_env: Any, transcript_path: Path | None, state_path: Any) -> str:
    if transcript_path is not None:
        try:
            stem = Path(transcript_path).stem
            if stem:
                return stem
        except Exception:  # noqa: BLE001
            pass
    sess = getattr(step_env, "__self__", None)
    try:
        gid = str(sess.game.game_run.game_id or "")
        if gid:
            return gid
    except Exception:  # noqa: BLE001
        pass
    try:
        return Path(state_path).parent.name or "?"
    except Exception:  # noqa: BLE001
        return "?"


def _level_of(step_env: Any, state_path: Any, agent_mod: Any) -> Any:
    """Display level (1-based): the live session's levels_completed + 1, else the runtime frame's level."""
    sess = getattr(step_env, "__self__", None)
    try:
        return int(sess.game.current_state.levels_completed) + 1
    except Exception:  # noqa: BLE001
        pass
    try:
        frame, _ = agent_mod.load_runtime_state(Path(state_path))
        if frame is not None:
            return int(frame.level)
    except Exception:  # noqa: BLE001
        pass
    return None


def _transcript_path(state_path: Any, kwargs: dict[str, Any]) -> Path | None:
    path = kwargs.get("transcript_path")
    if path is not None:
        return Path(path)
    try:
        sp = Path(state_path)
        return sp.parent / f"{sp.stem}_analyzer.txt"
    except Exception:  # noqa: BLE001
        return None


def _write_marker(agent_mod: Any, transcript_path: Path | None, text: str) -> None:
    if transcript_path is None or agent_mod is None:
        return
    try:
        agent_mod._append_transcript_section(transcript_path, TRANSCRIPT_LABEL, text)
    except Exception:  # noqa: BLE001
        _error()


# ------------------------------------------------------------- rendering ---
def _text_of(agent_mod: Any, content: Any) -> str:
    try:
        return str(agent_mod._normalize_message_content(content) or "")
    except Exception:  # noqa: BLE001
        return content if isinstance(content, str) else ""


def _cut(text: str, head: int, tail: int = 0) -> str:
    text = text or ""
    if len(text) <= head + tail:
        return text
    if tail <= 0:
        return text[:head].rstrip() + f" [... {len(text) - head} chars cut]"
    return text[:head].rstrip() + f"\n[... {len(text) - head - tail} chars cut ...]\n" + text[-tail:].lstrip()


def slim_user_prompt(text: str) -> str:
    """The state lines + the model's own carried note of a stock user prompt; the boilerplate
    instructions (identical every turn) are dropped. Harness follow-ups become one line."""
    text = text or ""
    if text.startswith(_FOLLOWUP_HEAD) or text.startswith(_NOTOOL_HEAD):
        return "[harness: no tool call in the previous reply; asked to act]"
    kept: list[str] = []
    in_note = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(_NOTE_START):
            in_note = True
            kept.append("Note carried at that time:")
            continue
        if in_note:
            if s.startswith(_NOTE_END):
                in_note = False
            elif s.startswith("- ") and not s.startswith("- Revise any item"):
                kept.append(s)
            continue
        if s.startswith(_KEEP_USER_PREFIXES):
            kept.append(s)
    out = "\n".join(kept) if kept else _cut(text, 300)
    return _cut(out, USER_CHARS)


def _tool_call_code(agent_mod: Any, tool_call: Any) -> str:
    try:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        args = function.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return _cut(args, CODE_CHARS)
        if isinstance(args, dict) and "code" in args:
            return _cut(str(args.get("code") or ""), CODE_CHARS)
        return _cut(json.dumps(args, ensure_ascii=False), CODE_CHARS)
    except Exception:  # noqa: BLE001
        return ""


def render_dropped(agent_mod: Any, messages: list[dict[str, Any]], cap: int | None = None) -> str:
    """The dropped turns as text for the compaction call: state lines + the model's own reasoning,
    code and tool results (per-message caps, then a whole-text cap that cuts the middle)."""
    if cap is None:
        cap = input_chars()
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if role == "user":
            blocks.append("[TURN STATE]\n" + slim_user_prompt(_text_of(agent_mod, msg.get("content"))))
        elif role == "assistant":
            parts: list[str] = []
            reasoning = ""
            try:
                reasoning = str(agent_mod._extract_reasoning_text(msg) or "")
            except Exception:  # noqa: BLE001
                reasoning = str(msg.get("reasoning") or "")
            if reasoning.strip():
                parts.append("[YOUR REASONING]\n" + _cut(reasoning.strip(), REASONING_HEAD_CHARS, REASONING_TAIL_CHARS))
            content = _text_of(agent_mod, msg.get("content")).strip()
            if content:
                parts.append("[YOUR NOTE]\n" + _cut(content, USER_CHARS))
            for tc in msg.get("tool_calls") or []:
                code = _tool_call_code(agent_mod, tc)
                if code:
                    parts.append("[YOUR CODE]\n" + code)
            if parts:
                blocks.append("\n".join(parts))
        elif role == "tool":
            try:
                shown = str(agent_mod._render_tool_result_display(msg.get("content")) or "")
            except Exception:  # noqa: BLE001
                shown = _text_of(agent_mod, msg.get("content"))
            blocks.append("[TOOL RESULT]\n" + _cut(shown.strip(), TOOL_RESULT_CHARS))
    text = "\n\n".join(b for b in blocks if b.strip())
    if len(text) > cap:
        head = int(cap * 0.6)
        tail = cap - head
        text = _cut(text, head, tail)
    return text


def compaction_messages(level: Any, previous_summary: str, rendered: str, n_msgs: int) -> list[dict[str, Any]]:
    words = max(60, summary_chars() // 6)
    user = (
        f"Game level in play: {level if level is not None else 'unknown'}.\n\n"
        f"PREVIOUS COMPACTED KNOWLEDGE:\n{previous_summary.strip() if previous_summary.strip() else '(none yet)'}\n\n"
        f"TURNS ABOUT TO BE REMOVED FROM CONTEXT ({n_msgs} messages, oldest first):\n{rendered}\n\n"
        "Write the updated compacted knowledge now."
    )
    return [{"role": "system", "content": COMPACT_SYSTEM.format(words=words)}, {"role": "user", "content": user}]


def is_compaction_payload(payload: dict[str, Any]) -> bool:
    """For mocks: a compaction request carries no tools and the compaction system prompt."""
    try:
        if payload.get("tools"):
            return False
        first = (payload.get("messages") or [{}])[0]
        return first.get("role") == "system" and str(first.get("content", "")).startswith(COMPACT_SYSTEM_HEAD)
    except Exception:  # noqa: BLE001
        return False


def system_with_summary(base: str, summary: str) -> str:
    summary = (summary or "").strip()
    if not summary:
        return base
    return f"{base}\n\n{SUMMARY_INTRO}\n{summary}"


def has_user_message(messages: list[dict[str, Any]] | None) -> bool:
    """The chat template raises `No user query found in messages.` unless at least one message has
    role == "user" (tool results ride as role "tool" and are rendered as <tool_response>, which the
    template explicitly does not count). Every request we hand back must satisfy this."""
    return any(str(m.get("role", "")).strip() == "user" for m in messages or [])


def summary_in_messages(messages: list[dict[str, Any]]) -> int:
    """Chars of the compacted block carried by the request's system message (0 when absent)."""
    try:
        first = messages[0] if messages else {}
        if str(first.get("role", "")) != "system":
            return 0
        content = first.get("content")
        text = content if isinstance(content, str) else ""
        i = text.find(SUMMARY_INTRO)
        if i < 0:
            return 0
        return len(text) - i - len(SUMMARY_INTRO) - 1
    except Exception:  # noqa: BLE001
        return 0


def reasoning_stats(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """(assistant messages carrying a non-empty reasoning field, their total chars)."""
    n = chars = 0
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        r = m.get("reasoning")
        if r in (None, ""):
            r = m.get("reasoning_content")
        if isinstance(r, str) and r.strip():
            n += 1
            chars += len(r)
    return n, chars


# ------------------------------------------------------------- the compaction call ---
def _compact(agent: Any, agent_mod: Any, st: CarryState, dropped: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    """One model call (no tools, thinking per flag, capped max_tokens) that folds the dropped turns into
    the compacted block. Returns (new_summary | None, meta). Never raises."""
    meta: dict[str, Any] = {"dropped_msgs": len(dropped), "input_chars": 0, "prompt_tokens": None,
                            "completion_tokens": None, "e2e_s": None, "ok": False, "err": ""}
    t0 = time.monotonic()
    try:
        rendered = render_dropped(agent_mod, dropped)
        meta["input_chars"] = len(rendered)
        msgs = compaction_messages(st.level, st.summary, rendered, len(dropped))
        payload = agent_mod.build_chat_payload(
            provider=agent._model.provider, model=agent._model.model_id, messages=msgs,
            max_tokens=compact_max_tokens(), temperature=agent_mod._LOCAL_ANALYZER_TEMPERATURE,
            top_p=agent_mod._LOCAL_ANALYZER_TOP_P, top_k=agent_mod._LOCAL_ANALYZER_TOP_K,
            thinking=compact_thinking(), tools=None, tool_choice=None, seed=agent_mod._LOCAL_ANALYZER_SEED)
        timeout = timeout_s()
        if st.request_timeout is not None:
            timeout = min(timeout, float(st.request_timeout))
        response = agent_mod.requests.post(f"{agent._model.base_url.rstrip('/')}/chat/completions",
                                           headers=agent._headers(), json=payload, timeout=timeout)
        meta["e2e_s"] = round(time.monotonic() - t0, 3)
        status = getattr(response, "status_code", 200)
        if status >= 400:
            meta["err"] = f"http_{status}"
            return None, meta
        body = response.json()
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            meta["prompt_tokens"] = usage.get("prompt_tokens")
            meta["completion_tokens"] = usage.get("completion_tokens")
            try:
                agent._accumulate_usage_tokens(usage)
            except Exception:  # noqa: BLE001
                pass
        choices = body.get("choices") or []
        if not choices:
            meta["err"] = "no_choices"
            return None, meta
        message = choices[0].get("message") or {}
        text = _text_of(agent_mod, message.get("content")).strip()
        if not text:
            meta["err"] = "empty" if str(choices[0].get("finish_reason", "")) != "length" else "length_no_content"
            return None, meta
        cap = summary_chars()
        if len(text) > cap:
            text = text[: cap - 1].rstrip() + "…"
        meta["ok"] = True
        return text, meta
    except Exception as exc:  # noqa: BLE001
        if meta["e2e_s"] is None:
            meta["e2e_s"] = round(time.monotonic() - t0, 3)
        meta["err"] = f"{type(exc).__name__}"
        return None, meta


def _record_compaction(st: CarryState, agent_mod: Any, summary: str | None, meta: dict[str, Any]) -> None:
    game = st.game
    if summary is not None:
        _bump(game, "compactions")
        _bump(game, "summary_chars_total", len(summary))
    else:
        _bump(game, "compaction_failures")
    _bump(game, "dropped_msgs_total", int(meta.get("dropped_msgs") or 0))
    _bump(game, "dropped_chars_total", int(meta.get("input_chars") or 0))
    if meta.get("prompt_tokens") is not None:
        _bump(game, "compaction_prompt_tokens", int(meta["prompt_tokens"]))
    if meta.get("completion_tokens") is not None:
        _bump(game, "compaction_completion_tokens", int(meta["completion_tokens"]))
    if meta.get("e2e_s") is not None:
        _bump(game, "compaction_e2e_s", float(meta["e2e_s"]))
    line = (f"{COMPACT_MARK} game={game} turn={st.turn} dropped_msgs={meta.get('dropped_msgs')} "
            f"input_chars={meta.get('input_chars')} summary_chars={len(summary) if summary is not None else 0} "
            f"prompt_tokens={meta.get('prompt_tokens')} completion_tokens={meta.get('completion_tokens')} "
            f"e2e_s={meta.get('e2e_s')} ok={1 if summary is not None else 0}")
    if summary is None:
        line += f" err={str(meta.get('err') or '?').replace(' ', '_')[:60]}"
    else:
        line += "\nSUMMARY:\n" + summary
    _write_marker(agent_mod, st.transcript_path, line)


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "carry: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"carry: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "carry: SKIP (missing ToolAgent)"
    for name in ("analyze", "_trim_messages_for_context", "_chat_completion", "_drop_oldest_history_block",
                 "_estimate_request_input_tokens", "_headers", "_accumulate_usage_tokens"):
        if getattr(cls, name, None) is None:
            return f"carry: SKIP (ToolAgent.{name} missing)"
    for name in ("_append_transcript_section", "build_chat_payload", "requests", "_normalize_message_content",
                 "_extract_reasoning_text", "_render_tool_result_display", "_LOCAL_ANALYZER_TEMPERATURE",
                 "_LOCAL_ANALYZER_TOP_P", "_LOCAL_ANALYZER_TOP_K", "_LOCAL_ANALYZER_SEED"):
        if getattr(agent_mod, name, None) is None:
            return f"carry: SKIP ({name} missing)"

    _STOCK["analyze"] = cls.analyze
    _STOCK["trim"] = cls._trim_messages_for_context
    _STOCK["chat_completion"] = cls._chat_completion

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        st = None
        if enabled():
            try:
                st = _cstate(self, state_path)
                st.agent_mod = agent_mod
                st.state_path = state_path
                st.transcript_path = _transcript_path(state_path, kwargs)
                st.game = _game_key(step_env, kwargs.get("transcript_path"), state_path)
                st.level = _level_of(step_env, state_path, agent_mod)
                st.request_timeout = kwargs.get("request_timeout_seconds")
                st.turns_seen += 1
                step = kwargs.get("analysis_step")
                st.turn = int(step) if step is not None else st.turns_seen
                st.req_in_turn = 0
                st.in_turn = True
                _bump(st.game, "turns_total")
            except Exception:  # noqa: BLE001
                _error()
                st = None
        try:
            return _STOCK["analyze"](self, state_path, action_num, valid_actions=valid_actions, step_env=step_env, **kwargs)
        finally:
            if st is not None:
                st.in_turn = False

    def _trim_messages_for_context(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        if not enabled():
            return _STOCK["trim"](self, messages, tools=tools, preserve_recent=preserve_recent,
                                  extra_safety_tokens=extra_safety_tokens)

        def stock(msgs):
            return _STOCK["trim"](self, msgs, tools=tools, preserve_recent=preserve_recent,
                                  extra_safety_tokens=extra_safety_tokens)

        def guarded(msgs):
            """Never hand back a request the template will reject with `No user query found in
            messages.` (400). That state is reachable in the UNMODIFIED stock too — when one turn's
            own assistant+tool pairs exceed the budget, the stock drop loop plus
            _drop_until_first_user_message leaves the system message alone (reproduced on the stock
            bundle at 10 calls x 8k reasoning) — but our system message carries the compacted block,
            so we reach it sooner and must not convert a playable turn into a failed call.
            Recovery, in order: (1) re-trim with the stock-sized system message (more headroom);
            (2) keep the turn's own user prompt and drop the assistant/tool pairs."""
            out = stock(msgs)
            if has_user_message(out) or not has_user_message(messages):
                return out
            _skip("no_user_after_trim")
            retry = stock(messages)
            if has_user_message(retry):
                return retry
            last_user = next((m for m in reversed(messages) if str(m.get("role", "")).strip() == "user"), None)
            if last_user is None:
                return out
            _skip("rebuilt_from_last_user")
            head = out[0] if out else msgs[0]
            return stock([head, last_user])

        st = getattr(self, "_carry", None)
        try:
            if st is None or not messages or str(messages[0].get("role", "")) != "system":
                return _STOCK["trim"](self, messages, tools=tools, preserve_recent=preserve_recent,
                                      extra_safety_tokens=extra_safety_tokens)
            base = getattr(self, "_system_prompt", None)
            if not isinstance(base, str):
                base = messages[0].get("content") if isinstance(messages[0].get("content"), str) else ""
            system = {**messages[0], "content": system_with_summary(base, st.summary)}
            history = list(messages[1:])
            work = [system, *history]
            if extra_safety_tokens and extra_safety_tokens > 0:
                _skip("overflow_path")       # the server rejected the request as too long: evict fast, no extra call
                return guarded(work)
            budget = max(1, int(self._context_budget_tokens))
            est = self._estimate_request_input_tokens(work, tools=tools)
            if est <= budget:
                return guarded(work)
            target = int(budget * target_fraction())
            dropped: list[dict[str, Any]] = []
            keep = max(0, int(preserve_recent))
            while history and est > target:
                before = list(history)
                if not self._drop_oldest_history_block(history, preserve_recent=keep):
                    break
                if not has_user_message(history):
                    history[:] = before      # dropping to the TARGET must not pass the last user message
                    _skip("keep_last_user")  # (the stock only drops to the budget and never gets here)
                    break
                dropped.extend(before[: len(before) - len(history)])
                est = self._estimate_request_input_tokens([system, *history], tools=tools)
            if len(dropped) < min_drop_msgs():
                _skip("small_drop")
                return guarded(work)
            if st.request_timeout is not None and float(st.request_timeout) < MIN_TIME_FOR_COMPACTION_S:
                _skip("no_time")             # the game is ending: eviction only
                return guarded([system, *history])
            new_summary, meta = _compact(self, agent_mod, st, dropped)
            _record_compaction(st, agent_mod, new_summary, meta)
            if new_summary is not None:
                st.summary = new_summary
                st.compactions += 1
                system = {**messages[0], "content": system_with_summary(base, st.summary)}
            return guarded([system, *history])
        except Exception:  # noqa: BLE001
            _error()
            return _STOCK["trim"](self, messages, tools=tools, preserve_recent=preserve_recent,
                                  extra_safety_tokens=extra_safety_tokens)

    def _chat_completion(self, messages, **kwargs):
        result = _STOCK["chat_completion"](self, messages, **kwargs)
        if not enabled():
            return result
        try:
            st = getattr(self, "_carry", None)
            if st is None or not st.in_turn:
                return result
            st.req_in_turn += 1
            n_reason, chars = reasoning_stats(messages)
            s_chars = summary_in_messages(messages)
            usage = getattr(result, "usage", None) or {}
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            game = st.game
            _bump(game, "calls_total")
            _bump(game, "reasoning_msgs_total", n_reason)
            _bump(game, "reasoning_chars_total", chars)
            if s_chars > 0:
                _bump(game, "calls_with_summary")
            if isinstance(prompt_tokens, (int, float)):
                _bump(game, "prompt_tokens_total", int(prompt_tokens))
                _bump(game, "prompt_tokens_max", int(prompt_tokens))
                if int(prompt_tokens) > window_tokens():
                    _bump(game, "prompt_over_window")
            _write_marker(agent_mod, st.transcript_path,
                          f"{CALL_MARK} game={game} turn={st.turn} req={st.req_in_turn} msgs={len(messages)} "
                          f"reasoning_msgs={n_reason} reasoning_chars={chars} summary_chars={s_chars} "
                          f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}")
        except Exception:  # noqa: BLE001
            _error()
        return result

    analyze._carry_stock = _STOCK["analyze"]
    _trim_messages_for_context._carry_stock = _STOCK["trim"]
    _chat_completion._carry_stock = _STOCK["chat_completion"]
    cls.analyze = analyze
    cls._trim_messages_for_context = _trim_messages_for_context
    cls._chat_completion = _chat_completion
    _STATE["installed"] = True
    return "carry: OK"
