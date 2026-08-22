"""Archetype-triage graft — kill zero-level sessions at archetype-specific
wall-time thresholds, releasing the concurrency slot for queued games.

Motivation (measured, scratchpad/testing_20260822/triage132/trigger_rule.json,
re-derived at the TRUE 132-min competition geometry): a flat kill-at-55 is
DEAD (14.3% FN on the shipped 3.8 wave, 26.9% pooled vs the <=5% gate). The
archetype rule is the survivor: dispatch on frame-0 available_actions, then
kill a session that has completed ZERO levels at

    AVATAR : 60 min  (0 FN in 13 pooled zero@60 sessions; latest observed
             AVATAR first completion 54.5m — ls20, bracket 53.4-55.6)
    MIXED  : 70 min  (n=2 evidence, bp35; one completion at 56.9m; 70
             leaves ~12m margin)
    CLICK  : never   (heavy late tail: lp85 first completion 88.0m at score
             2.01; ft09 3.6-corpus completions at 104.7-122.6m; even @90
             pooled CLICK FN would be 5/33 = 15.2%)

Observed on the pooled corpus: 0 FN in all 103 sessions, ~41 kills and
~48.9 reclaimed worker-hours per 110-game run (~20% of budget).

Dispatch (playbook.json dispatch_rule + trigger_rule.json, reconciled):
frame-0 menu with RESET(0) stripped —
    exactly click-family (has 6, no movement 1-4)          => CLICK
    movement-family within {1,2,3,4,5} (>=1 of 1-4, no 6)  => AVATAR
    both families (>=1 of 1-4 AND 6)                       => MIXED
    anything else (empty, {5}, ACTION7 menus, unreadable)  => CLICK
CLICK doubles as the never-kill safe default, so an unconfident dispatch can
never cost a session. The archetype is classified once from the first
readable menu and frozen (menus can change mid-game; the rule is a frame-0
rule). By design this graft is a PURE kill-rule: no TimeBank, no grant
machinery (that was the v7 explorer's job) — single-variable.

Envelope note: see ENVELOPE.md — this graft strictly RECLAIMS wall time. It
can only make ``should_stop`` return True EARLIER; it adds no sleeps,
retries, or extensions, and a fail-open error always falls back to the
stock verdict.

Seams (verified against the June stock tree,
scratchpad/bundles/june_stock/src/ARC3-Inference, plus the vendored taaf at
submission/_adopt/taaf-src/src/tufa-arc-agi-framework/src/taaf):

- solver.py:246-261 — ``_HarnessGameSession.should_stop``: the patch point.
  Consulted by the play loop (:276), by the per-action batch check
  (:605-607), and passed per analyze call as ``should_stop=self.should_stop``
  (:301) — all attribute lookups, so a class-attribute patch reaches every
  session and every already-bound callback; single-namespace (grep: no
  by-name import of ``should_stop``).
- Banking on kill — VERIFIED chain: ``play``'s finally (solver.py:331-338)
  calls ``_finish_if_needed`` (:340-345), which — stop_event unset, run
  still "playing" — calls ``game.finish_game()``; taaf/game.py:596-632
  transitions a sync non-cancelled playing run to state="gave_up" (:629-631)
  and banks ``final_score = run._compute_final_score()`` (:632).
  "gave_up" is a completed, fully-scored state (inference/tools/eval.py:33-34
  counts it in COMPLETED_GAME_RUN_STATES/FINALIZED_SCORING_STATES), so any
  levels completed before the kill keep their score. (Zero-level kills bank
  a zero — which is why the rule requires levels_completed == 0.)
- Slot release — VERIFIED chain: ``_run_games`` (solver.py:896-917) runs
  each game inside ``async with semaphore`` (:898, :904) on the
  ThreadPoolExecutor worker pool (:884-887). When ``should_stop`` returns
  True the play loop exits, ``_play_one`` (solver.py:1208-1236) returns,
  the executor thread is freed and the semaphore released — the next queued
  game starts immediately.
- Dispatch input: ``self.game.current_state.available_actions`` — the same
  list step_env validates against (solver.py:608). taaf/game.py:183-189
  ALWAYS prepends RESET(0), so the classifier strips 0 first. Frame-0 is
  real: taaf's ``GameAPI._start_game`` captures the post-make-RESET
  observation (game_api.py:204-245), so the first ``should_stop`` call
  (play loop :276, before any action) sees the frame-0 menu.
- Clock: ``started_at`` (solver.py:179, ``time.monotonic`` at session
  construction) — the same zero the stock ``runtime_limit_reached``
  (solver.py:212-217) measures against.
- Zero-level check: ``int(self.game.current_state.levels_completed)``
  (taaf/game.py:190-192), the same counter the scorer uses.

Fail-open invariants:
- The stock ``should_stop`` runs FIRST, unguarded — its verdict (including
  exceptions) propagates exactly as stock; a stock True is returned before
  any graft logic runs.
- All triage logic (menu read, classification, elapsed math, note stamp)
  sits inside a blanket try/except that returns the stock verdict (False)
  on any error.
- TRIAGE_KILL=0 disables at install time (no patch) and at call time
  (installed wrapper is a pure pass-through).
- Threshold overrides: TRIAGE_AVATAR_MIN / TRIAGE_MIXED_MIN /
  TRIAGE_CLICK_MIN (minutes; empty / "never" / non-numeric / <=0 => never
  kill that archetype). Defaults 60 / 70 / never per trigger_rule.json.
"""

from __future__ import annotations

import os
import time
from typing import Any

ARCHETYPE_CLICK = "CLICK"
ARCHETYPE_AVATAR = "AVATAR"
ARCHETYPE_MIXED = "MIXED"

DEFAULT_KILL_MINUTES: dict[str, float | None] = {
    ARCHETYPE_AVATAR: 60.0,
    ARCHETYPE_MIXED: 70.0,
    ARCHETYPE_CLICK: None,  # never
}

_MOVEMENT_IDS = {1, 2, 3, 4}
_AVATAR_FAMILY = {1, 2, 3, 4, 5}
_CLICK_ID = 6

_FLAG_ENV = {
    ARCHETYPE_AVATAR: "TRIAGE_AVATAR_MIN",
    ARCHETYPE_MIXED: "TRIAGE_MIXED_MIN",
    ARCHETYPE_CLICK: "TRIAGE_CLICK_MIN",
}


def _enabled() -> bool:
    return os.environ.get("TRIAGE_KILL", "1").strip() not in {"0", "false", "False"}


def kill_minutes(archetype: str) -> float | None:
    """Kill threshold in minutes for an archetype, or None for never."""
    raw = os.environ.get(_FLAG_ENV.get(archetype, ""), "").strip()
    if raw:
        if raw.lower() in {"never", "none", "off"}:
            return None
        try:
            minutes = float(raw)
        except ValueError:
            return DEFAULT_KILL_MINUTES.get(archetype)
        return minutes if minutes > 0 else None
    return DEFAULT_KILL_MINUTES.get(archetype)


def classify_menu(available_actions: Any) -> str | None:
    """Frame-0 archetype from an available_actions menu, or None if unreadable.

    RESET(0) is stripped (taaf always includes it). CLICK is the never-kill
    safe default: any menu that is not confidently movement-family (AVATAR)
    or both-families (MIXED) classifies as CLICK. An empty-after-strip menu
    returns None so classification can retry on the next call.
    """
    try:
        ids = {int(item) for item in available_actions}
    except (TypeError, ValueError):
        return ARCHETYPE_CLICK
    ids.discard(0)
    if not ids:
        return None  # not yet readable — retry next call
    has_movement = bool(ids & _MOVEMENT_IDS)
    has_click = _CLICK_ID in ids
    if has_movement and has_click:
        return ARCHETYPE_MIXED
    if has_movement and ids <= _AVATAR_FAMILY:
        return ARCHETYPE_AVATAR
    # {6}, {5,6}, {6,7}, {5}, {7}, movement+ACTION7, ... => CLICK (never kill)
    return ARCHETYPE_CLICK


def _triage_should_kill(session: Any) -> bool:
    """True iff the frozen archetype's threshold has passed with zero levels."""
    archetype = getattr(session, "_triage_archetype", None)
    if archetype is None:
        archetype = classify_menu(session.game.current_state.available_actions)
        if archetype is None:
            return False
        session._triage_archetype = archetype  # freeze at first readable menu
    minutes = kill_minutes(archetype)
    if minutes is None:
        return False
    elapsed_s = time.monotonic() - session.started_at
    if elapsed_s < minutes * 60.0:
        return False
    if int(session.game.current_state.levels_completed) != 0:
        return False
    if not getattr(session, "_triage_killed", False):
        session._triage_killed = True
        try:
            run = session.game.game_run
            if run is not None and getattr(run, "solver_note", None) is None:
                run.solver_note = (
                    f"triage_kill: {archetype} zero-level at {elapsed_s / 60.0:.1f}m "
                    f"(threshold {minutes:.0f}m)"
                )
        except Exception:  # noqa: BLE001 — the note is observability only
            pass
    return True


def install() -> str:
    if not _enabled():
        return "archetype_triage: SKIP (TRIAGE_KILL=0)"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"archetype_triage: SKIP (solver module missing: {exc!r})"

    # Presence gates — every seam symbol, fail toward stock on any mismatch.
    session_cls = getattr(solver_mod, "_HarnessGameSession", None)
    if session_cls is None:
        return "archetype_triage: SKIP (missing _HarnessGameSession)"
    original_should_stop = getattr(session_cls, "should_stop", None)
    if original_should_stop is None:
        return "archetype_triage: SKIP (missing _HarnessGameSession.should_stop)"
    for name in ("play", "step_env", "_finish_if_needed"):
        if getattr(session_cls, name, None) is None:
            return f"archetype_triage: SKIP (missing _HarnessGameSession.{name} — banking seam moved)"
    if "started_at" not in getattr(session_cls, "__dataclass_fields__", {}):
        return "archetype_triage: SKIP (started_at field missing — clock seam moved)"
    if getattr(original_should_stop, "_archetype_triage_patched", False):
        return "archetype_triage: SKIP (already applied)"

    def should_stop_with_triage(self: Any) -> bool:
        # Stock verdict first, unguarded: stock True (or a stock exception)
        # propagates exactly as before the graft.
        if original_should_stop(self):
            return True
        if not _enabled():
            return False
        try:
            return _triage_should_kill(self)
        except Exception:  # noqa: BLE001 — any triage error means "don't kill"
            return False

    should_stop_with_triage._archetype_triage_patched = True  # type: ignore[attr-defined]
    # Single-namespace by design: should_stop is only ever resolved via
    # ``self.should_stop`` (solver.py:276, :301, :592, :605) — never imported
    # by name.
    session_cls.should_stop = should_stop_with_triage
    install.originals = {  # type: ignore[attr-defined]
        "should_stop": original_should_stop,
    }
    return "archetype_triage: OK"
