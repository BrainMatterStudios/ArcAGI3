"""Hypothesis-enumeration + probe rule (HYPO, 2026-09-06).

Evidence (docs/research-2026-09-06/R-loss-ledger-2.md Q4/Q6): 37/105 stuck
tails are WRONG-HYPOTHESIS and 25 ANALYSIS-PARALYSIS; the model acts on one
wrong model to the clock, "reaches its goal state, no clear, then thrashes",
and never explicitly falsifies a predicate. The judge (J-judge-0906.md §B)
gives a prompt rule <15% prior of being a step, so this arm is a kill test:
the rule rides every analyzer turn on an uncleared level and the wave reads
whether the cd82/dc22/lf52 L2 walls fall.

MECHANISM (all flags read at call time; HYPO_ENABLE=0 is byte-identical stock):
  ToolAgent._build_user_prompt is wrapped; when the run is still playing
  (no run_complete in the previous step summary, the live session's game_run
  state is "playing" and the engine has not WON) the fixed block below
  (<= 900 chars, marker "[HYPO]") is appended to the user prompt exactly
  once per analyze() call. The follow-up prompts inside a turn ("You have
  not acted yet ...") are built inline by analyze(), not by
  _build_user_prompt, so the block never repeats inside a turn. It is
  absent after the run completes (WIN / not playing), which is the only
  "cleared" state an analyzer call can be made in: the level in play is by
  definition uncleared. The turn right after a level clear keeps the block
  (the new level is the next wall; the engagement gate counts wall turns).

The session is reached through self._step_env_callback (analyze() sets it
before building the prompt; solver passes step_env=session.step_env). No
session (unit tests, other harnesses) -> the summary alone decides.

HISTORY STRIP (judge fix 09-06): the block must ride ONLY the current turn.
analyze() persists the turn's messages through
self._persistent_history_messages(messages, tools=...) (its `finally`), and
the next turn rebuilds the request as [system, *persisted history, new user
message]. The graft wraps _persistent_history_messages and returns a copy in
which every user message (str content or the multimodal [text, image] list)
has the block removed, so older prompts carry no copy and exactly one block
is in flight per request. Counted in status()["blocks_stripped"].

Telemetry: status()["blocks_injected"], per-game counts, skips; the runner
counts "[HYPO]" lines in [USER PROMPT] sections.
Conventions: module _STATE/_STOCK, install() -> "hypo: OK" / "hypo: SKIP (...)".
"""
from __future__ import annotations

import os
import threading
from typing import Any

_STATE: dict[str, Any] = {
    "installed": False,
    "blocks_injected": 0,
    "blocks_stripped": 0,
    "errors": 0,
    "skips": {},
    "per_game": {},
}
_STOCK: dict[str, Any] = {}
_LOCK = threading.Lock()
_OFF = {"0", "false", "no", "off"}

MARKER = "[HYPO]"
BLOCK_CHARS = 900

HYPO_BLOCK = (
    "[HYPO] Hypothesis discipline for this uncleared level (harness rule, every turn):\n"
    "1. List >=3 candidate mechanics / goal predicates (what would make this level count as cleared), each "
    "consistent with EVERY transition observed on this level so far; with no transitions yet, derive them from "
    "the layout. For each, name one observation that would falsify it.\n"
    "2. Pick the cheapest probe (<=3 actions) whose outcome differs between at least two candidates. Write the "
    "predicted outcome per candidate, then run that probe FIRST as one action([...]) batch, before any longer plan.\n"
    "3. If the previous batch was expected to clear the level and did not, state which predicate that falsified, "
    "drop it, and re-derive the goal from the transitions; do not repeat the same plan with small variations.\n"
    "4. Judge what changed from a computed before/after diff of the frames, never from memory."
)


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("HYPO_ENABLE", "1").lower() not in _OFF


def block_text() -> str:
    text = HYPO_BLOCK
    if len(text) > BLOCK_CHARS:
        text = text[: BLOCK_CHARS - 1].rstrip() + "…"
    return text


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "installed": _STATE["installed"],
            "enabled": enabled(),
            "block_chars": len(block_text()),
            "blocks_injected": _STATE["blocks_injected"],
            "blocks_stripped": _STATE["blocks_stripped"],
            "errors": _STATE["errors"],
            "skips": dict(_STATE["skips"]),
            "per_game": dict(_STATE["per_game"]),
        }


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] = _STATE["skips"].get(reason, 0) + 1


# ------------------------------------------------------------ decision ---
def _session_of(agent: Any) -> Any:
    cb = getattr(agent, "_step_env_callback", None)
    return getattr(cb, "__self__", None)


def _game_id(sess: Any) -> str:
    try:
        return str(sess.game.game_run.game_id or "?")
    except Exception:  # noqa: BLE001
        return "?"


def should_inject(agent: Any, previous_step_summary: Any) -> tuple[bool, str]:
    """(inject?, reason). False only once the run is over: the level in play is uncleared by definition."""
    if isinstance(previous_step_summary, dict) and previous_step_summary.get("run_complete"):
        return False, "run_complete"
    sess = _session_of(agent)
    if sess is not None:
        try:
            run = sess.game.game_run
            if run is not None and str(getattr(run, "state", "playing")) != "playing":
                return False, "not_playing"
        except Exception:  # noqa: BLE001
            pass
        try:
            if bool(getattr(sess.game.current_state, "won", False)):
                return False, "won"
        except Exception:  # noqa: BLE001
            pass
    return True, "uncleared"


def strip_block(text: str) -> tuple[str, int]:
    """Remove every copy of the block (with the newline the graft added) from a prompt string."""
    block = block_text()
    n = text.count(block)
    if not n:
        return text, 0
    return text.replace("\n" + block, "").replace(block, ""), n


def strip_history(messages: list[Any]) -> tuple[list[Any], int]:
    """Copy of `messages` with the block removed from every user message (str or multimodal list)."""
    out: list[Any] = []
    stripped = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, str):
            new, n = strip_block(content)
            if n:
                m = {**m, "content": new}
                stripped += n
        elif isinstance(content, list):
            parts = []
            changed = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    new, n = strip_block(part["text"])
                    if n:
                        part = {**part, "text": new}
                        changed = True
                        stripped += n
                parts.append(part)
            if changed:
                m = {**m, "content": parts}
        out.append(m)
    return out, stripped


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "hypo: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"hypo: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "hypo: SKIP (missing ToolAgent)"
    if getattr(cls, "_build_user_prompt", None) is None:
        return "hypo: SKIP (ToolAgent._build_user_prompt missing)"
    if getattr(cls, "_persistent_history_messages", None) is None:
        return "hypo: SKIP (ToolAgent._persistent_history_messages missing)"

    _STOCK["build_user_prompt"] = cls._build_user_prompt
    _STOCK["persistent_history_messages"] = cls._persistent_history_messages

    def _build_user_prompt(self, action_num, *args, **kwargs):
        text = _STOCK["build_user_prompt"](self, action_num, *args, **kwargs)
        if not enabled():
            return text
        try:
            ok, reason = should_inject(self, kwargs.get("previous_step_summary"))
            if not ok:
                _skip(reason)
                return text
            game = _game_id(_session_of(self))
            with _LOCK:
                _STATE["blocks_injected"] += 1
                _STATE["per_game"][game] = _STATE["per_game"].get(game, 0) + 1
            return text + "\n" + block_text()
        except Exception:  # noqa: BLE001
            with _LOCK:
                _STATE["errors"] += 1
            return text

    def _persistent_history_messages(self, messages, *args, **kwargs):
        kept = _STOCK["persistent_history_messages"](self, messages, *args, **kwargs)
        if not enabled():
            return kept
        try:
            stripped, n = strip_history(list(kept))
            if n:
                with _LOCK:
                    _STATE["blocks_stripped"] += n
            return stripped
        except Exception:  # noqa: BLE001
            with _LOCK:
                _STATE["errors"] += 1
            return kept

    _build_user_prompt._hypo_stock = _STOCK["build_user_prompt"]
    _persistent_history_messages._hypo_stock = _STOCK["persistent_history_messages"]
    cls._build_user_prompt = _build_user_prompt
    cls._persistent_history_messages = _persistent_history_messages
    _STATE["installed"] = True
    return "hypo: OK"
