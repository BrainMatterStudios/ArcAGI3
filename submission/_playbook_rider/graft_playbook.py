"""Playbook-rider graft — static cross-game human-playbook prior in the first
user prompt.

Motivation (docs/RESEARCH-2026-08-22-slotmath-and-top3.md §E): in-run
cross-game transfer is topology-capped (28x4 generations; 25% of plays
unreachable; the first 28 plays get zero benefit by construction), so the
research verdict is: ship the STATIC playbook prompt rider — per-archetype
priors mined from 340 human replays — and skip the dynamic-store plumbing.

Data source: scratchpad/testing_20260822/playbook.json (+ dispatch.json for
the frame-0 menu evidence), mined 2026-08-21 from the 340-trace human replay
corpus. The per-archetype numbers are EMBEDDED below as build-time constants
(read from playbook.json at build time), so the graft has no runtime file
dependency and is malformed-data-proof by construction.

Mechanism:
- At game start, classify the archetype from the frame-0 available_actions
  menu — the same family table submission/_archetype_triage/graft_triage.py
  uses (logic copied, modules deliberately NOT coupled):
      movement(1-4) AND click(6)                => MIXED
      movement(1-4), menu within {1,2,3,4,5}    => AVATAR
      click(6), no movement                     => CLICK
  ONE divergence from triage: triage's fallback default is CLICK because
  CLICK is its never-kill SAFE default; a prompt prior injected on an
  unconfident menu would be wrong information, so here anything else
  (empty, {5}, {7}, unreadable) => None => NO injection.
- Inject ONE short block (<=120 tokens, measured in the tests) into the
  FIRST user prompt of the game only, phrased as observed human statistics —
  never as instructions. Never injected again on later turns or yield-resume
  slices: later prompts stay byte-identical to stock, keeping the shared
  vLLM prefix-cache stable.

Seams (verified against the June stock tree,
scratchpad/bundles/june_stock/src/ARC3-Inference):
- tool_agent.py:1161-1256 — ``ToolAgent._build_user_prompt`` (the injection
  seam; precedent submission/_yield_carryover/graft_carryover.py:291-308).
  It receives ``valid_actions`` as a keyword (engine names, RESET already
  stripped by solver.py:109-121 ``_engine_action_names``) and is called once
  per analyze() at tool_agent.py:1727-1733 with the frame-0 menu on the
  game's first turn (solver.py:296 passes the live engine menu).
- solver.py:1188-1206 + :1224 — ONE ToolAgent per game (``_make_analyzer``
  inside ``_play_one``), so "first _build_user_prompt call on the instance"
  == "first user prompt of the game".
- USER prompt injection ONLY — a per-game system-prompt variant would split
  the shared vLLM prefix-cache KV and the system message is never trimmed
  (docs/RESEARCH-2026-08-22-slotmath-and-top3.md appendix). The block is
  APPENDED to the first user prompt, so the carryover graft's 80-char head
  marker (graft_carryover.py:296) is unaffected whichever installs first.

Fail-open invariants:
- The inner ``_build_user_prompt`` call is never wrapped in a swallowing
  except: an inner crash propagates exactly as stock.
- All rider logic (classification, block lookup, append) sits inside a
  blanket try/except; any error => stock prompt, byte-identical.
- PLAYBOOK_PRIOR=0 disables at install time (no patch) and at call time
  (installed wrapper is a pure pass-through).
- Malformed playbook constants (a non-string / empty block) => no injection,
  never a crash.
- install() presence-gates every seam symbol and fails toward stock.
"""

from __future__ import annotations

import os
from typing import Any

ARCHETYPE_CLICK = "CLICK"
ARCHETYPE_AVATAR = "AVATAR"
ARCHETYPE_MIXED = "MIXED"

# ---------------------------------------------------------------------------
# Playbook blocks — constants baked from scratchpad/testing_20260822/
# playbook.json (mined 2026-08-21, 340 human replay traces):
#   CLICK : probe_before_L1 p50 18.5 / p90 37; actions_per_level p50 39.5 /
#           p90 131; resets p50 3; same-cell click repeat frac 0.317
#   AVATAR: probe_before_L1 p50 33 / p90 70.5; actions_per_level p50 78 /
#           p90 237; resets p50 4
#   MIXED : probe_before_L1 p50 22.5; actions_per_level p50 48 / p90 102;
#           resets p50 11
# Phrased as observed statistics, not instructions.
# ---------------------------------------------------------------------------

PLAYBOOK_HEADER = "Cross-game prior (statistics observed in 340 human replays, not instructions):"

PLAYBOOK: dict[str, str] = {
    ARCHETYPE_CLICK: (
        f"{PLAYBOOK_HEADER} "
        "Human winners on click games typically probed ~18-37 actions before "
        "completing level 1 and finished each level in under ~131 actions "
        "(median ~40). Repeated clicking on the same cell was common and "
        "legitimate (~32% of their clicks); they reset ~3 times per game."
    ),
    ARCHETYPE_AVATAR: (
        f"{PLAYBOOK_HEADER} "
        "Human winners on movement games typically probed ~33-70 actions "
        "before completing level 1 and finished each level in under ~237 "
        "actions (median ~78). They reset ~4 times per game, so an "
        "occasional RESET-and-retry was normal."
    ),
    ARCHETYPE_MIXED: (
        f"{PLAYBOOK_HEADER} "
        "Human winners on mixed movement+click games typically probed ~22 "
        "actions before completing level 1 and finished each level in under "
        "~102 actions (median ~48). They reset ~11 times per game, so "
        "frequent RESET-and-retry was normal."
    ),
}

_MOVEMENT_IDS = {1, 2, 3, 4}
_AVATAR_FAMILY = {1, 2, 3, 4, 5}
_CLICK_ID = 6

# Both engine names (what _build_user_prompt receives on June stock) and
# model labels (post-normalization lineages) resolve to engine action ids.
_NAME_TO_ID = {
    "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
    "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
    "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "MOUSE": 6,
    "RESET": 0,
}


def _enabled() -> bool:
    return os.environ.get("PLAYBOOK_PRIOR", "1").strip() not in {"0", "false", "False"}


def _menu_ids(valid_actions: Any) -> set[int] | None:
    """Engine action-id set for a menu, or None when any token is unreadable
    (an unconfident menu must never pick up a prior)."""
    try:
        items = list(valid_actions)
    except TypeError:
        return None
    ids: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            ids.add(item)
            continue
        raw = str(item).strip().upper()
        if raw in _NAME_TO_ID:
            ids.add(_NAME_TO_ID[raw])
        elif raw.startswith("ACTION") and raw[6:].isdigit():
            ids.add(int(raw[6:]))
        else:
            return None
    ids.discard(0)  # RESET rides along in raw engine menus; strip it
    return ids


def classify_menu(valid_actions: Any) -> str | None:
    """Frame-0 archetype from an available-actions menu, or None.

    Same family table as graft_triage.classify_menu; the fallback differs by
    design — triage defaults unknowns to CLICK (its never-kill safety), a
    prompt prior defaults unknowns to None (no injection).
    """
    ids = _menu_ids(valid_actions)
    if not ids:
        return None
    has_movement = bool(ids & _MOVEMENT_IDS)
    has_click = _CLICK_ID in ids
    if has_movement and has_click:
        return ARCHETYPE_MIXED
    if has_movement and ids <= _AVATAR_FAMILY:
        return ARCHETYPE_AVATAR
    if has_click:
        return ARCHETYPE_CLICK
    return None  # {5}, {7}, movement+7 without click, ... => no injection


def playbook_block(archetype: str | None) -> str:
    """The prior block for an archetype; "" when there is none (including a
    malformed PLAYBOOK entry — fail toward no injection, never a crash)."""
    if archetype is None:
        return ""
    try:
        block = PLAYBOOK.get(archetype)
    except Exception:  # noqa: BLE001 — a gutted PLAYBOOK means no injection
        return ""
    if not isinstance(block, str) or not block.strip():
        return ""
    return block


def install() -> str:
    if not _enabled():
        return "playbook_rider: SKIP (PLAYBOOK_PRIOR=0)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"playbook_rider: SKIP (tool_agent module missing: {exc!r})"
    tool_agent_cls = getattr(agent_mod, "ToolAgent", None)
    if tool_agent_cls is None:
        return "playbook_rider: SKIP (missing ToolAgent)"
    original_build_prompt = getattr(tool_agent_cls, "_build_user_prompt", None)
    if original_build_prompt is None:
        return "playbook_rider: SKIP (ToolAgent._build_user_prompt missing — inject seam moved)"
    if getattr(original_build_prompt, "_playbook_rider_patched", False):
        return "playbook_rider: SKIP (already applied)"

    def build_prompt_with_playbook(self: Any, *args: Any, **kwargs: Any) -> str:
        # Never guard the inner call: an inner crash propagates as stock.
        prompt = original_build_prompt(self, *args, **kwargs)
        try:
            if not _enabled():
                return prompt
            if getattr(self, "_playbook_prior_done", False):
                return prompt
            # First user prompt of the game (one ToolAgent per game) — the
            # one and only injection attempt, whatever the menu says. Later
            # prompts stay byte-identical to stock (prefix-cache stability).
            self._playbook_prior_done = True
            archetype = classify_menu(kwargs.get("valid_actions"))
            block = playbook_block(archetype)
            if not block or block in prompt:
                return prompt
            prompt = f"{prompt}\n\n{block}"
        except Exception:  # noqa: BLE001 — any rider error => stock prompt
            pass
        return prompt

    build_prompt_with_playbook._playbook_rider_patched = True  # type: ignore[attr-defined]
    # Single-namespace: _build_user_prompt is only ever resolved via ``self.``
    # (tool_agent.py:1727) — never imported by name.
    tool_agent_cls._build_user_prompt = build_prompt_with_playbook
    install.originals = {  # type: ignore[attr-defined]
        "_build_user_prompt": original_build_prompt,
    }
    return "playbook_rider: OK"
