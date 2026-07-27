"""Grounded effect memory — harness-measured action/click effect statistics.

The Sensi post-mortem (arXiv 2603.17683) shows LLM-judged knowledge collapses
into self-consistent hallucination cascades without ground-truth anchoring;
StochasticGoose proves binary frame-change is a cheap dense signal. This pack
ports both: the HARNESS (not the model) records, per level, which actions
changed the board and which click cells were productive, and surfaces a
compact measured summary in every user prompt. The model gets one knowledge
source it can trust unconditionally because no LLM judgment touches it.

Wraps two ToolAgent methods:
    _summarize_step_sequence  — records effects from executed action results
                                (delegates first; recording is read-only on
                                the results and never alters the summary).
    _build_user_prompt        — appends the measured block when non-empty.
Plus the module-level _build_system_prompt for a one-line explanation.

Toggle (env, read at CALL time): EFFECT_MEMORY=1. Applied-but-disabled
delegates verbatim on every wrapped path — Stage-1 ReplayMockLLM must prove
applied-off byte-identical, same as the doctrine/ledger packs.

Attribution is conservative (measured means measured):
  * single-action results attribute board_changed to that action exactly;
  * multi-action batches with board_changed=False mark EVERY executed action
    unchanged (sound: any(changed)=False);
  * multi-action batches with board_changed=True are skipped entirely
    (ambiguous which action changed the board; ambiguity must never
    manufacture evidence).
Stats reset on level change (layouts change; rules persisting is what the
cross-level notes are for). Deaths do NOT reset stats (same level, same
physics). game_over/level-transition FRAMES can differ trivially; the last
result of a terminal batch is still attributed — acceptable noise, the
terminal action by definition changed the game state.
"""
from __future__ import annotations

import os
import re
from typing import Any

_MOUSE_RE = re.compile(r"^MOUSE\(row=(\d+), col=(\d+)\)$")

# reporting thresholds — a single no-effect observation is not "never works"
NOOP_MIN_TRIED = 2
MAX_REPORT_ITEMS = 10


def _enabled() -> bool:
    return os.environ.get("EFFECT_MEMORY", "").strip().lower() in ("1", "true", "yes")


def fresh_memory() -> dict[str, Any]:
    return {"level": None, "actions": {}, "clicks": {}}


def _record_one(memory: dict[str, Any], name: str, changed: bool) -> None:
    mouse = _MOUSE_RE.match(name)
    if mouse:
        key = (int(mouse.group(1)), int(mouse.group(2)))
        tried, hit = memory["clicks"].get(key, (0, 0))
        memory["clicks"][key] = (tried + 1, hit + (1 if changed else 0))
    else:
        tried, hit = memory["actions"].get(name, (0, 0))
        memory["actions"][name] = (tried + 1, hit + (1 if changed else 0))


def record_results(memory: dict[str, Any], action_results: list[dict[str, Any]]) -> None:
    """Fold executed action results into the memory (pure, in place)."""
    for item in action_results:
        if not item.get("executed"):
            continue
        level = item.get("level")
        if level is not None and level != memory["level"]:
            memory.clear()
            memory.update(fresh_memory())
            memory["level"] = level
        names = [str(n).strip() for n in (item.get("executed_actions") or []) if str(n).strip()]
        if not names:
            fallback = str(item.get("action_display") or "").strip()
            names = [fallback] if fallback else []
        if not names:
            continue
        changed = bool(item.get("board_changed"))
        if len(names) == 1:
            _record_one(memory, names[0], changed)
        elif not changed:
            for name in names:
                _record_one(memory, name, False)
        # multi-action batch that changed the board: ambiguous, not attributed


def build_effect_block(memory: dict[str, Any]) -> str:
    """The measured-summary prompt block; empty string when nothing to say."""
    noop_actions = sorted(
        (name, tried) for name, (tried, hit) in memory["actions"].items()
        if hit == 0 and tried >= NOOP_MIN_TRIED
    )
    productive_clicks = sorted(
        ((rc, tried, hit) for rc, (tried, hit) in memory["clicks"].items() if hit > 0),
        key=lambda entry: (-entry[2], entry[0]),
    )[:MAX_REPORT_ITEMS]
    noop_clicks = sorted(
        (rc, tried) for rc, (tried, hit) in memory["clicks"].items()
        if hit == 0 and tried >= NOOP_MIN_TRIED
    )[:MAX_REPORT_ITEMS]
    if not (noop_actions or productive_clicks or noop_clicks):
        return ""

    lines = [
        "",
        "Grounded effect memory (measured by the harness from actual frames on"
        " this level - trust it over recollection):",
    ]
    if noop_actions:
        rendered = ", ".join(f"{name} (0/{tried})" for name, tried in noop_actions)
        lines.append(f"- Actions that have NEVER changed the board here: {rendered}.")
    if productive_clicks:
        rendered = ", ".join(
            f"({row},{col}): {hit}/{tried}" for (row, col), tried, hit in productive_clicks
        )
        lines.append(f"- Click cells that changed the board (changed/tried): {rendered}.")
    if noop_clicks:
        rendered = ", ".join(f"({row},{col}) 0/{tried}" for (row, col), tried in noop_clicks)
        lines.append(f"- Click cells with no effect so far: {rendered}.")
    return "\n".join(lines)


SYSTEM_LINE = (
    "\n\nA 'Grounded effect memory' section may appear in the state: it is computed"
    " programmatically from real frame diffs, never from guesses. Prefer it over your"
    " own notes when they conflict, do not repeat actions it marks as never-changing"
    " on this level, and favor its productive click cells when exploring."
    " Mark a note line CONFIRMED only with the observed transition that verified it"
    " (action -> observed change); never mark CONFIRMED from inference alone.\n"
)


def _memory_of(agent: Any) -> dict[str, Any]:
    memory = getattr(agent, "_effect_memory", None)
    if memory is None:
        memory = fresh_memory()
        agent._effect_memory = memory
    return memory


def patch_effect_memory() -> str:
    """Wrap recording + surfacing; delegate verbatim when env-disabled."""
    from inference.agent import tool_agent as ta

    cls = ta.ToolAgent
    prior_summarize = cls._summarize_step_sequence
    if getattr(prior_summarize, "_effect_memory", False):
        return "effects: SKIP (already applied)"

    def _summarize(self: Any, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        summary = prior_summarize(self, action_results)
        if _enabled():
            record_results(_memory_of(self), action_results or [])
        return summary

    _summarize._effect_memory = True  # type: ignore[attr-defined]
    cls._summarize_step_sequence = _summarize

    prior_user_prompt = cls._build_user_prompt

    def _user_prompt(self: Any, *args: Any, **kwargs: Any) -> str:
        out = prior_user_prompt(self, *args, **kwargs)
        if not _enabled():
            return out
        block = build_effect_block(_memory_of(self))
        return out + block if block else out

    _user_prompt._effect_memory = True  # type: ignore[attr-defined]
    cls._build_user_prompt = _user_prompt

    build = getattr(ta, "_build_system_prompt", None)
    if build is not None and not getattr(build, "_effect_prompt", False):
        def _build(*args: Any, **kwargs: Any) -> str:
            out = build(*args, **kwargs)
            return out + SYSTEM_LINE if _enabled() else out
        _build._effect_prompt = True  # type: ignore[attr-defined]
        ta._build_system_prompt = _build

    return "effects: OK (measured effect memory + prompt surfacing; env-gated EFFECT_MEMORY)"


def apply_all(verbose: bool = True) -> list[str]:
    results = [patch_effect_memory()]
    if verbose:
        for line in results:
            print(f"[effects] {line}", flush=True)
    return results
