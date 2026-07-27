"""Two-tier knowledge ledger — the audit-amended successor to FIX A's wipe fix.

FIX A (duck_fixes.patch_gameover_wipe) stopped game_over from erasing the
world model, but it keeps EVERYTHING — including the exact hypothesis that
just got the agent killed, which survives verbatim and can be re-trusted.
The two-tier ledger splits knowledge by evidence status:

    CONFIRMED lines  — statements the model marked as observation-verified
                       ("CONFIRMED: ..." prefix) — SURVIVE a death untouched.
    unmarked lines   — hypotheses — are DEMOTED on death into Open questions
                       as re-test items; the death that falsified them is the
                       demotion trigger.

Design guarantees:
  * Env-gated at CALL time (LEDGER_TWO_TIER=1): the patch applied-but-disabled
    delegates to whatever method was live before it (upstream or FIX A) — the
    Stage-1 ReplayMockLLM gate proves applied-off is behavior-identical.
  * Graceful degradation: if the model never marks anything CONFIRMED, the
    transform falls back to FIX A behavior (keep all + death note) — the
    ledger can never do worse than FIX A on a convention-ignoring model.
  * level_transition / run_complete keep the upstream full clear (deliberate
    design; cross_level_notes carries over).
  * A companion system-prompt line (same env toggle) teaches the convention;
    with doctrine D3, a "CONFIRMED: ACTION7_SEMANTICS ..." line now survives
    deaths — the two features compose.

Post-WIN replay note (audit 2026-07-26): if the post-WIN replay lever ships,
run_complete must ALSO become ledger-preserving — the wipe would erase the
solution exactly when the replay starts. Left untouched here (replay is not
shipped); revisit with that bundle.
"""
from __future__ import annotations

import os
from typing import Any

CONFIRMED_PREFIXES = ("confirmed:", "- confirmed:", "* confirmed:", "[confirmed]")
DEMOTED_CAP_CHARS = 600

LEDGER_PROMPT = (
    "\n\nKnowledge ledger protocol: in your World model / Goal model / Action model"
    " notes, prefix a line with 'CONFIRMED:' ONLY after you have verified the"
    " statement against an observed frame transition. Leave hypotheses unmarked."
    " When you die, CONFIRMED lines are kept; unmarked lines are moved to Open"
    " questions for re-testing - so confirming what you verify is how you keep"
    " knowledge across deaths.\n"
)


def _enabled() -> bool:
    return os.environ.get("LEDGER_TWO_TIER", "").strip().lower() in ("1", "true", "yes")


def _is_confirmed(line: str) -> bool:
    s = line.strip().lower()
    return any(s.startswith(p) for p in CONFIRMED_PREFIXES)


def split_ledger(text: str) -> tuple[str, list[str]]:
    """(confirmed_text, demoted_lines) — preserves confirmed lines verbatim."""
    if not text:
        return "", []
    kept, demoted = [], []
    for line in text.splitlines():
        if _is_confirmed(line):
            kept.append(line)
        elif line.strip():
            demoted.append(line.strip())
    return "\n".join(kept), demoted


def apply_death_transform(knowledge: dict) -> dict:
    """Pure transform of the _summarized_knowledge dict on game_over."""
    tiered = {}
    all_demoted: list[str] = []
    any_confirmed = False
    for key in ("world_model", "goal_model", "action_model"):
        confirmed, demoted = split_ledger(knowledge.get(key, ""))
        tiered[key] = confirmed
        all_demoted.extend(demoted)
        any_confirmed = any_confirmed or bool(confirmed)

    if not any_confirmed:
        # convention not in use — FIX A fallback: keep everything
        return knowledge

    knowledge.update(tiered)
    if all_demoted:
        demoted_note = "Unverified at death, re-test: " + " | ".join(all_demoted)
        if len(demoted_note) > DEMOTED_CAP_CHARS:
            demoted_note = demoted_note[: DEMOTED_CAP_CHARS - 3] + "..."
        oq = knowledge.get("open_questions", "")
        knowledge["open_questions"] = f"{oq}\n{demoted_note}".strip()
    return knowledge


def patch_two_tier_ledger() -> str:
    """Wrap the knowledge-update method; delegate verbatim when env-disabled."""
    from inference.agent import tool_agent as ta

    cls = ta.ToolAgent
    prior = cls._update_summarized_knowledge_from_step_summary
    if getattr(prior, "_two_tier_ledger", False):
        return "ledger: SKIP (already applied)"

    def _update(self: Any) -> None:
        if not _enabled():
            return prior(self)
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("level_transition") or summary.get("run_complete"):
            return prior(self)  # upstream/FIX-A handling, unchanged
        if summary.get("game_over"):
            self._summarized_knowledge["current_plan"] = ""
            apply_death_transform(self._summarized_knowledge)
            findings = self._summarized_knowledge.get("recent_findings", "")
            death_note = ("GAME_OVER just occurred; the last action(s) were fatal — "
                          "avoid repeating them.")
            if death_note not in findings:
                self._summarized_knowledge["recent_findings"] = (
                    f"{findings} {death_note}".strip())

    _update._two_tier_ledger = True  # type: ignore[attr-defined]
    cls._update_summarized_knowledge_from_step_summary = _update

    # companion prompt line, same toggle, checked at build time
    build = getattr(ta, "_build_system_prompt", None)
    if build is not None and not getattr(build, "_ledger_prompt", False):
        def _build(*args: Any, **kwargs: Any) -> str:
            out = build(*args, **kwargs)
            return out + LEDGER_PROMPT if _enabled() else out
        _build._ledger_prompt = True  # type: ignore[attr-defined]
        ta._build_system_prompt = _build

    return "ledger: OK (two-tier death transform + prompt line; env-gated LEDGER_TWO_TIER)"


def apply_all(verbose: bool = True) -> list[str]:
    results = [patch_two_tier_ledger()]
    if verbose:
        for line in results:
            print(f"[ledger] {line}", flush=True)
    return results
