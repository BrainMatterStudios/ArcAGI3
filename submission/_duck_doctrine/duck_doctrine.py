"""Prompt doctrine pack — three independently-toggled system-prompt addenda.

Wraps `inference.agent.tool_agent._build_system_prompt` and appends up to three
blocks. With every toggle off (the default) the wrapper returns the original
prompt byte-identically, so shipping this module inert is a no-op.

Toggles (env vars, read at APPLY time, not import time):
    DOCTRINE_FIELDGUIDE=1  D1: mechanic field guide (engine priors, ~340 tok)
    DOCTRINE_PLAYBOOK=1    D2: K3 win-behavior playbook + anti-HUD (~230 tok)
    DOCTRINE_ACTION7=1     D3: guided probe-once ACTION7 protocol (~70 tok)

D1 is the compendium's prompt-injectable field guide (mechanics_compendium.md
section 4) — engine-level priors only, no dev-game names. D2 encodes the nine
verified K3 win-behaviors from trace_forensics as imperatives; its anti-HUD
line targets the 27B's #1 verified killer (HUD-as-reward anchoring). D3 is
the guidance half of the ACTION7 lesson: the mapping fix WITHOUT guidance
scored 0.81/0.85 (probe spam against squared efficiency), and boristown
independently rolled their guidance back for the same reason — so D3 must
only ever ship TOGETHER WITH duck_patches.patch_action7(), and the probe-once
budget is the load-bearing sentence.

Offline A/B contract (pre-registered, run via rl_gate/run_rollout.py):
    GO iff first_board_changing_action median <= 2, HUD-chasing game_overs
    strictly down, levels won >= base, actions/level inflation <= 10%.
"""
from __future__ import annotations

import os
from typing import Any

FIELD_GUIDE = (
    "\n\nMechanic field guide (families seen across many games in this engine):\n"
    "- REACH/NAVIGATE: an avatar moves with the arrow actions; walls block. Win: reach a"
    " marked cell. Watch for doors that need keys or buttons.\n"
    "- COVER-ALL: several movable pieces and equally many target cells. Win: EVERY piece"
    " on a target - one is never enough. Movement may be coupled (all pieces move"
    " together) or via push.\n"
    "- MATCH/COPY: two panels, one editable and one reference. Win: make them equal"
    " (sometimes ignoring marked cells). Clicking often cycles colors; a submit button"
    " may finalize.\n"
    "- CONSTRAINT-COLOR: cells with glyph corners/edges encode neighbor constraints"
    " (must equal / must differ). Satisfy all clues.\n"
    "- MODAL CONTROLS: clicking an object can SELECT it (highlight); arrows then move"
    " the selection. Extra actions often mean submit, undo, phase-switch, or"
    " record/replay. If arrows seem dead, try click-then-arrow.\n"
    "- BUILD-THEN-RUN: place tiles or program steps, then a run action executes them;"
    " no feedback until the run.\n"
    "- TIMING: objects open and close on their own every few actions; act while open;"
    " misses cost strikes.\n"
    "- PHYSICS: things fall (gravity may be INVERTED - check which way loose objects"
    " drift); water and projectiles propagate.\n"
    "Universal: budgets are tight (a step counter usually shows remaining actions;"
    " running out loses the level; the level restarts free). Wins are almost always"
    " ALL-quantified - complete every subgoal, not one. The avatar may be an unusual"
    " color or shape, may grow like a snake, or may not exist (click-only game).\n"
)

PLAYBOOK = (
    "\n\nPlaybook (behaviors of the strongest players of these games - follow them):\n"
    "1. First strike: make a board-changing action within your first 2 turns. Never"
    " spend more than 2 turns only inspecting.\n"
    "2. Probe then commit: test a hypothesis with ONE cheap action; once confirmed,"
    " commit the full sequence in a burst.\n"
    "3. Check every result: after each action, diff the new frame against your"
    " prediction and state what changed. A mismatch is information.\n"
    "4. Keep a named hypothesis ledger (H1, H2, ...) in your notes with status"
    " confirmed/open/refuted; revise it explicitly when contradicted.\n"
    "5. Once a mechanic is understood, write small throwaway solver code to compute the"
    " whole action sequence instead of choosing moves one by one.\n"
    "6. HUD elements (bars, counters, small indicators at the edges) display STATE -"
    " they are never the goal. Never take actions to make a bar bigger; win by"
    " completing the level. Read the HUD as information only.\n"
    "7. When stuck, re-read your own action history and hunt for one observation that"
    " contradicts your current hypothesis.\n"
    "8. If progress appears undone, identify which action undid it and stop using it in"
    " that state.\n"
    "9. Carry confirmed mechanics forward across levels: expect the layout to change"
    " and the rules to persist.\n"
)

ACTION7_GUIDE = (
    "\n\nACTION7 protocol: ACTION7's meaning varies by game (often undo, phase-switch,"
    " or a special ability). If it is offered, test it AT MOST ONCE per game, early and"
    " in a safe state, record its observed effect in your notes as ACTION7_SEMANTICS,"
    " and afterwards use it only when that recorded effect is exactly what you want."
    " Never probe it repeatedly - every probe is a scored action.\n"
)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "yes")


def doctrine_suffix() -> str:
    """The blocks the current env toggles select, in fixed order."""
    parts = []
    if _enabled("DOCTRINE_FIELDGUIDE"):
        parts.append(FIELD_GUIDE)
    if _enabled("DOCTRINE_PLAYBOOK"):
        parts.append(PLAYBOOK)
    if _enabled("DOCTRINE_ACTION7"):
        parts.append(ACTION7_GUIDE)
    return "".join(parts)


def patch_doctrine() -> str:
    """Wrap tool_agent._build_system_prompt to append the enabled blocks."""
    from inference.agent import tool_agent as ta

    original = getattr(ta, "_build_system_prompt", None)
    if original is None:
        return "doctrine: FAIL (_build_system_prompt not found)"
    if getattr(original, "_doctrine_patched", False):
        return "doctrine: SKIP (already applied)"

    if _enabled("DOCTRINE_ACTION7"):
        # D3 without the mapping fix would tell the model to use an action the
        # engine rejects; refuse rather than ship an incoherent arm.
        from inference.agent import action_names as an
        if an.to_engine_action("ACTION7") != "ACTION7":
            return ("doctrine: FAIL (DOCTRINE_ACTION7 set but ACTION7 mapping fix "
                    "not applied - run duck_patches.patch_action7() first)")

    def _build_system_prompt_doctrine(*args: Any, **kwargs: Any) -> str:
        return original(*args, **kwargs) + doctrine_suffix()

    _build_system_prompt_doctrine._doctrine_patched = True  # type: ignore[attr-defined]
    ta._build_system_prompt = _build_system_prompt_doctrine
    enabled = [n for n in ("DOCTRINE_FIELDGUIDE", "DOCTRINE_PLAYBOOK",
                           "DOCTRINE_ACTION7") if _enabled(n)]
    return f"doctrine: OK (enabled: {', '.join(enabled) or 'none - inert'})"


def apply_all(verbose: bool = True) -> list[str]:
    results = [patch_doctrine()]
    if verbose:
        for line in results:
            print(f"[doctrine] {line}", flush=True)
    return results
