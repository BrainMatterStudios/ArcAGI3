"""Truthful-telemetry graft — the harness↔engine mismatch pack, prompt/payload
definitions only (zero mechanics changed).

Motivation (docs/RESEARCH-2026-08-21-bug-lever-hunt.md Tier-2 #6): the model
plays with four pieces of live telemetry it was never told about:

1. RESET is hidden — solver.py:109-120 ``_engine_action_names`` strips the
   name (the ``if name == "RESET": continue`` at solver.py:116-117) from the
   valid-action list shown to the model, and prompts.py contains zero RESET
   mentions — while the engine keeps RESET always-legal, so soft-locked
   levels have no exit today.
   VERIFIED zero-code-change executability of ``action(['RESET'])``:
   the sandbox's ``action`` handler (tool_agent.py:1495-1529 ``_handle_action``)
   does NO valid-action filtering — it normalizes
   (tool_agent.py:1378-1411, the string 'RESET' -> {"action": "RESET"}) and
   calls ``step_env`` directly; ``step_env`` normalizes via
   ``to_engine_action`` (action_names.py:25-31 maps "RESET" -> "RESET") and
   validates each action id against the RAW engine list
   ``self.game.current_state.available_actions`` (solver.py:608) — NOT the
   RESET-stripped display list — and the strip at solver.py:116-117 exists
   precisely because the engine's available_actions contains RESET. The
   auto-reset path (solver.py:663-665) executes RESET through the very same
   ``_execute_action``. Test 08 in test_truthful_telemetry.py replays this
   end-to-end through the real ``_HarnessGameSession.step_env``.
2. ``run_elapsed_seconds`` / ``time_remaining_seconds`` ride every action
   result (solver.py:219-225 ``timing_payload``, attached at solver.py:549,
   :585 and :729) but are never defined for the model.
3. Total level count is held by the harness (``game.number_of_levels``,
   solver.py:101-106, the engine's initial win_levels) but never surfaced —
   the model cannot value depth (score weight = level+1).
4. Payload traps: ``"score"`` = levels_completed (solver.py:570, :711) and
   ``reward`` = completion-fraction delta (solver.py:698-701) — neither
   defined anywhere. No fields are renamed; we only define them.

Mechanism (each behind its OWN flag, all prompt/payload-only):
- TT_RESET   — one RESET definition line appended to the system prompt.
- TT_TIME    — one line defining the two time keys + that expiry banks
               completed levels.
- TT_LEVELS  — ``HarnessSolver._make_analyzer`` (solver.py:1181-1206, only
               ever invoked as ``self._make_analyzer(...)`` at solver.py:1223)
               is wrapped to stamp ``game.number_of_levels`` onto the
               analyzer (walking ``_inner`` so wrapper analyzers pass it
               through); ``ToolAgent._build_user_prompt``
               (tool_agent.py:1161-1256) then appends "level X of N" per
               turn, X parsed from the stock prompt's own state line
               (tool_agent.py:1215 "Current state: step S, level L").
- TT_PAYLOAD — one line defining ``score`` and ``reward``.

System-prompt lines are injected by wrapping module-level
``_build_system_prompt`` (tool_agent.py:350-359; called only from
``ToolAgent.__init__`` at tool_agent.py:941 — single namespace). Every
injection is marker-guarded so it can never appear twice in one prompt, and
the per-turn line is replaced (not accumulated) on every build.

Fail-open invariants:
- Inner calls are never wrapped in try/except — crashes propagate as stock.
- All graft logic sits in blanket try/except; any error => stock prompt.
- Each flag checked at call time; a disabled flag's line vanishes from the
  next constructed prompt. All four off at install time => SKIP.
"""

from __future__ import annotations

import os
import re
from typing import Any

RESET_LINE = (
    "- `RESET` is a real action you may take at any time with `action(['RESET'])`, even though RESET is "
    "never listed in `valid_actions`. RESET restarts the current level, costs 1 action, and does NOT clear "
    "the level's accumulated action count; use it only when the level is unwinnable from the current state "
    "(soft-locked, or a mistake made the goal unreachable)."
)
TIME_LINE = (
    "- Every action result includes `run_elapsed_seconds` (wall-clock seconds this game has been running) "
    "and `time_remaining_seconds` (seconds left before this game is stopped). When time expires, levels you "
    "have already completed stay banked and still score; only progress on the unfinished level is lost."
)
PAYLOAD_LINE = (
    "- In action results, `score` is the number of levels completed so far in this game, and `reward` is the "
    "completion-fraction delta of the last call: levels newly completed divided by the game's total level "
    "count (so 0.0 means no level was completed by that call)."
)
_LEVELS_PREFIX = "Level progress: level "
_STATE_LINE_RE = re.compile(r"Current state: step \d+, level (\d+)")


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip() not in {"0", "false", "False"}


def _levels_line(current_level: int, total_levels: int) -> str:
    return (
        f"{_LEVELS_PREFIX}{current_level} of {total_levels} in this game; later levels are worth more "
        "(score weight grows with depth), so completing deeper levels matters most."
    )


def _any_enabled() -> bool:
    return any(_flag_enabled(name) for name in ("TT_RESET", "TT_TIME", "TT_LEVELS", "TT_PAYLOAD"))


def install() -> str:
    if not _any_enabled():
        return "truthful_telemetry: SKIP (all TT_* flags off)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"truthful_telemetry: SKIP (tool_agent module missing: {exc!r})"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"truthful_telemetry: SKIP (solver module missing: {exc!r})"

    original_system_prompt = getattr(agent_mod, "_build_system_prompt", None)
    if original_system_prompt is None:
        return "truthful_telemetry: SKIP (_build_system_prompt missing)"
    tool_agent_cls = getattr(agent_mod, "ToolAgent", None)
    if tool_agent_cls is None:
        return "truthful_telemetry: SKIP (missing ToolAgent)"
    original_build_prompt = getattr(tool_agent_cls, "_build_user_prompt", None)
    if original_build_prompt is None:
        return "truthful_telemetry: SKIP (ToolAgent._build_user_prompt missing)"
    harness = getattr(solver_mod, "HarnessSolver", None)
    if harness is None:
        return "truthful_telemetry: SKIP (missing HarnessSolver)"
    original_make = getattr(harness, "_make_analyzer", None)
    if original_make is None:
        return "truthful_telemetry: SKIP (HarnessSolver._make_analyzer missing)"
    if getattr(agent_mod._build_system_prompt, "_truthful_telemetry_patched", False):
        return "truthful_telemetry: SKIP (already applied)"

    # --- system prompt: static definition lines (marker-guarded) ------------

    def system_prompt_with_telemetry(*args: Any, **kwargs: Any) -> str:
        prompt = original_system_prompt(*args, **kwargs)
        try:
            additions = []
            if _flag_enabled("TT_RESET") and RESET_LINE not in prompt:
                additions.append(RESET_LINE)
            if _flag_enabled("TT_TIME") and TIME_LINE not in prompt:
                additions.append(TIME_LINE)
            if _flag_enabled("TT_PAYLOAD") and PAYLOAD_LINE not in prompt:
                additions.append(PAYLOAD_LINE)
            if additions:
                prompt = prompt + "\n" + "\n".join(additions) + "\n"
        except Exception:  # noqa: BLE001 — injection failure => stock prompt
            pass
        return prompt

    # --- per-turn user prompt: level X of N (replaced, never accumulated) ---

    def build_prompt_with_levels(self: Any, *args: Any, **kwargs: Any) -> str:
        prompt = original_build_prompt(self, *args, **kwargs)
        try:
            if not _flag_enabled("TT_LEVELS"):
                return prompt
            total_levels = getattr(self, "_tt_total_levels", None)
            if not total_levels:
                return prompt
            total_levels = int(total_levels)
            if total_levels <= 0 or _LEVELS_PREFIX in prompt:
                return prompt
            match = _STATE_LINE_RE.search(prompt)
            if match is None:
                return prompt
            prompt = f"{prompt}\n{_levels_line(int(match.group(1)), total_levels)}"
        except Exception:  # noqa: BLE001
            pass
        return prompt

    # --- transport: stamp number_of_levels onto the analyzer ----------------

    def make_analyzer_with_levels(self: Any, game: Any, index: Any, *args: Any, **kwargs: Any) -> Any:
        analyzer = original_make(self, game, index, *args, **kwargs)
        try:
            if not _flag_enabled("TT_LEVELS") or analyzer is None:
                return analyzer
            total_levels = int(getattr(game, "number_of_levels", 0) or 0)
            if total_levels <= 0:
                return analyzer
            target = analyzer
            for _ in range(4):  # unwrap analyzer wrappers (e.g. RetryGuard._inner)
                if hasattr(type(target), "_build_user_prompt"):
                    break
                inner = getattr(target, "_inner", None)
                if inner is None:
                    break
                target = inner
            target._tt_total_levels = total_levels
        except Exception:  # noqa: BLE001 — stamp failure => no level line
            pass
        return analyzer

    system_prompt_with_telemetry._truthful_telemetry_patched = True  # type: ignore[attr-defined]
    agent_mod._build_system_prompt = system_prompt_with_telemetry
    tool_agent_cls._build_user_prompt = build_prompt_with_levels
    harness._make_analyzer = make_analyzer_with_levels
    install.originals = {  # type: ignore[attr-defined]
        "_build_system_prompt": original_system_prompt,
        "_build_user_prompt": original_build_prompt,
        "_make_analyzer": original_make,
    }
    return "truthful_telemetry: OK"
