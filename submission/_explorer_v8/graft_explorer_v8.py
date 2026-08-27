"""Explorer v8 — the live SearchCore graft (build stage 7).

WHAT THIS IS
------------
v7 (submission/_explorer_floor/graft_explorer.py) flew live at 1.51 with a
BLIND reset-replay BFS (``_grind`` -> ``bfs_one_level``): one RESET per action
tested, component-centroid clicks only, no novelty ordering, no macros, no
archive, no specialists.  It never cracked a game, so its banking path never
fired.

v8 replaces ONLY the grind ALGORITHM with the measured SearchCore portfolio
(submission/_search_core/{search_core,specialists}.py, docs/RESEARCH-2026-08-23
Addenda 3+4) on the SAME reset-replay backend, keeps v7's trigger, self-harm
gate, semaphore and envelope guards VERBATIM, enforces those guards on EVERY
engine call instead of only at engagement entry, and hands a full-game crack
to the validated banking invariants.

Measured 2026-08-26 by driving THIS graft end-to-end on the offline engine
with backend="reset_replay", counting every engine call and pricing it at the
live gateway rate (test_H; the standalone falsifier run at
submission/_search_core/results/falsifier_livesim_v8_20260825_234923.json
agrees to within the graft's own 2 extra bookkeeping resets):

    ft09   6 levels, FULL CRACK,  1_233 engine actions  ->   9.5 s @130 act/s
    tu93   9 levels, FULL CRACK, 84_687 engine actions  -> 651.4 s @130 act/s

Both were BANKED end-to-end: 75- and 187-action minimal replays on the fresh
play post-WIN RESET opens.  For contrast, v7's blind BFS never cracked a game
in any measurement (hence its banking path never fired), v7's own note records
tu93's full win at ~3300 s of grind, and on ft09 the closest generic baseline
is 1 level at 120 s/game (run_falsifier.STAGE2_120).

CHANGES vs v7 (all four are the graft's reason to exist)
-------------------------------------------------------
1. ALGORITHM.  ``bfs_one_level`` -> SearchCore lane rotation over
   {nbfs, nbfs+macros, go-explore} on the frame-0 archetype dispatch order,
   with the 4-tier click generator (dead-click pruning scoped to bulk tiers
   3/4 only — the sb26 lesson), run-length macros that emit intermediate
   states, composite ignition probes at inert roots, and Go-Explore archive
   scheduling whose rollouts spend ONE reset per rollout with 0.92
   action-repeat momentum (not one reset per action).
2. SPECIALIST TIER FIRST.  ``specialists.detect`` runs once after warmup;
   on a match the specialist opens the lane ranking.  Fail-open: a failed
   detection or a failed specialist attempt closes that lane and the generic
   lanes proceed unchanged.
3. BANKING.  On a full-game crack the concatenated per-level minimal token
   plan is replayed on the fresh play that post-WIN RESET opens, under the
   validated ``graft_bank`` invariants (fresh-play check + the RUN-WIDE kill
   switch, shared with the banking graft so one trip disables both).
4. ENVELOPE.  Every v7 guard is kept at its v7 value and is now checked on
   every engine call through ``GuardedEnv`` (v7 checked the run-level guards
   only at engagement entry, so a single 1500 s engagement could overrun the
   cumulative budget).  Two new hard caps: an engine-action ceiling derived
   from the wall cap, and an explicit bank-replay action cap.
   Safety case with worst-case arithmetic: docs/ENVELOPE-2026-08-26-v8.md.

FLAGS OFF = v7.  ``EXPLORER_V8`` defaults to "0"; with it off ``install()``
installs v7 unchanged and touches nothing else, so the flown-at-1.51 lane is
byte-identical.  ``EXPLORER=0`` still disables the whole floor.

Fail-open at every seam: an unimportable SearchCore declines the swap (v7
stays), a failing warmup/detect/lane/bank is caught and the engagement ends
with the env RESET to the current level's start for the LLM.
"""

from __future__ import annotations

import os
import sys
import threading as _threading
import time as _time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBMISSION = os.path.dirname(_HERE)


def _candidate_dirs() -> list[str]:
    """Where search_core/specialists/graft_explorer/graft_bank may live.

    Dev tree: sibling ``_search_core`` / ``_explorer_floor`` / ``_duck38_v12_bank``
    directories.  Kaggle bundle: everything flat next to this file.  Override
    with EXPLORER_V8_CORE_DIR (os.pathsep-separated)."""
    out = [d for d in os.environ.get("EXPLORER_V8_CORE_DIR", "").split(os.pathsep) if d]
    out += [
        _HERE,
        os.path.join(_SUBMISSION, "_search_core"),
        os.path.join(_SUBMISSION, "_explorer_floor"),
        os.path.join(_SUBMISSION, "_duck38_v12_bank"),
    ]
    return out


for _d in _candidate_dirs():
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

import graft_explorer as v7  # noqa: E402  (v7 IS the substrate: trigger, gate, guards)


# --------------------------------------------------------------------------
# env knobs
# --------------------------------------------------------------------------

def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() not in {"0", "false", "False", ""}


def _env_int(name: str, default: int) -> int:
    return v7._env_int(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def v8_enabled() -> bool:
    """v8 is opt-in: with it off this module contributes nothing (v7 lane)."""
    return v7._enabled() and _flag("EXPLORER_V8", "0")


# Live gateway throughput measured on the scored run (v7 postmortem): ~130
# engine actions/second against the one shared server at concurrency 28.
GATEWAY_ACT_PER_S = 130.0


def _hard_cutoff_s() -> float:
    """Absolute run-clock ceiling for grinding, measured from _RUN_T0.

    v7's START gate (EXPLORER_RUN_CUTOFF_S, 18000 s) is kept exactly, but v7
    had no mid-flight check at all, so a grind admitted at 17999 s could run
    its full owned cap to ~19500 s — an IMPLICIT bound nothing enforced. v8
    makes that same 19500 s bound explicit and hard: an admitted engagement is
    still allowed to finish (no work is thrown away that v7 would have kept),
    and nothing can run past it."""
    return float(_env_int("EXPLORER_RUN_HARD_CUTOFF_S",
                          _env_int("EXPLORER_RUN_CUTOFF_S", 18000)
                          + _env_int("EXPLORER_OWNED_TIME_S", 1500)))


class _GrindAbort(Exception):
    """A guard tripped mid-engagement. Carries the stop reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# v8.1 EARLY SPECIALIST PROBE — its own bounded gate, separate from the
# run-wide grind semaphore. A probe is ~123 engine actions (measured mean over
# 25 fixtures, worst 206), i.e. ~1 s at the live gateway rate, so a couple may
# safely overlap where a 600-1500 s grind may not. Non-blocking: a session
# that cannot acquire retries on a later poll, inside its action window.
_PROBE_GATE = _threading.BoundedSemaphore(
    max(1, _env_int("EXPLORER_V8_EARLY_CONCURRENCY", 2)))


# --------------------------------------------------------------------------
# GuardedEnv — every envelope guard, enforced on EVERY engine call
# --------------------------------------------------------------------------

class GuardedEnv:
    """``arc_agi.EnvironmentWrapper``-compatible facade that enforces the run
    envelope on every ``reset``/``step`` and counts engine actions.

    v7 polled the run-level guards (cumulative grind budget, run cutoff) ONLY
    at engagement entry: a single engagement admitted just under the budget
    could then run its full 1500 s owned cap on top, overshooting by 55%.  v8
    re-checks them inside the loop, so the cumulative cap is a real cap.

    Guards, cheapest first (the expensive session probes are sampled every
    ``PROBE_EVERY`` calls, so a trip is detected within 32 actions ~= 0.25 s):

      budget        engine actions >= action_cap
      time_cap      engagement wall >= 600 s (1500 s once grind-owned)
      run_grind     cumulative run grind wall >= EXPLORER_RUN_GRIND_BUDGET_S
      run_cutoff    run wall >= EXPLORER_RUN_CUTOFF_S
      cancelled     session.stop_event set
      runtime_cap   session.runtime_limit_reached()
      soft_time     solver.soft_time_remaining_seconds() < 60 s
      action_cap    session.action_count + spent >= solver.max_actions_per_game
    """

    PROBE_EVERY = 32

    def __init__(self, env: Any, session: Any, xs: dict[str, Any], *,
                 t0: float, action_cap: int, time_cap_s: float,
                 owned_time_cap_s: float, soft_floor_s: float = 60.0,
                 label: str = "grind"):
        self._env = env
        self._session = session
        self._xs = xs
        self._t0 = float(t0)
        self._action_cap = int(action_cap)
        self._time_cap_s = float(time_cap_s)
        self._owned_time_cap_s = float(owned_time_cap_s)
        self._soft_floor_s = float(soft_floor_s)
        self._label = label
        self.actions = 0
        self.resets = 0
        self.aborted: str | None = None

    # ---- guard ------------------------------------------------------------

    def wall(self) -> float:
        return _time.monotonic() - self._t0

    def wall_cap(self) -> float:
        """Owned games (the grinder unlocked at least one level) get the long
        cap: the alternative use of their box is worth zero (v7 rule)."""
        return self._owned_time_cap_s if self._xs.get("grind_unlocked_levels") \
            else self._time_cap_s

    def remaining_s(self) -> float:
        """Seconds this engagement may still spend under EVERY wall guard."""
        left = [self.wall_cap() - self.wall()]
        with v7._RUN_LOCK:
            spent = v7._GRIND_WALL_SPENT[0]
            run_t0 = v7._RUN_T0
        left.append(_env_int("EXPLORER_RUN_GRIND_BUDGET_S", 2700) - (spent + self.wall()))
        if run_t0 is not None:
            left.append(_hard_cutoff_s() - (_time.monotonic() - run_t0))
        try:
            soft = self._session.solver.soft_time_remaining_seconds()
            if soft is not None:
                left.append(float(soft) - self._soft_floor_s)
        except Exception:  # noqa: BLE001
            pass
        return min(left)

    def remaining_actions(self) -> int:
        left = self._action_cap - self.actions
        try:
            cap = self._session.solver.max_actions_per_game
            if cap is not None:
                left = min(left, int(cap) - int(self._session.action_count)
                           - self.actions)
        except Exception:  # noqa: BLE001
            pass
        return max(0, left)

    def check(self) -> None:
        if self.actions >= self._action_cap:
            raise _GrindAbort("budget")
        if self.wall() >= self.wall_cap():
            raise _GrindAbort("time_cap")
        if self.actions % self.PROBE_EVERY == 0:
            with v7._RUN_LOCK:
                spent = v7._GRIND_WALL_SPENT[0]
                run_t0 = v7._RUN_T0
            if spent + self.wall() >= _env_int("EXPLORER_RUN_GRIND_BUDGET_S", 2700):
                raise _GrindAbort("run_grind_budget")
            if run_t0 is not None and (_time.monotonic() - run_t0) >= _hard_cutoff_s():
                raise _GrindAbort("run_cutoff")
            try:
                if self._session.stop_event.is_set():
                    raise _GrindAbort("cancelled")
                if self._session.runtime_limit_reached():
                    raise _GrindAbort("runtime_cap")
                soft = self._session.solver.soft_time_remaining_seconds()
                if soft is not None and float(soft) < self._soft_floor_s:
                    raise _GrindAbort("soft_time")
                cap = self._session.solver.max_actions_per_game
                if cap is not None and (int(self._session.action_count)
                                        + self.actions) >= int(cap):
                    raise _GrindAbort("action_cap")
            except _GrindAbort:
                raise
            except Exception:  # noqa: BLE001 — a broken probe never blocks
                pass

    # ---- engine facade ----------------------------------------------------

    def _count(self) -> None:
        self.actions += 1
        try:
            self._xs["diag"]["grinder_actions"] += 1
        except Exception:  # noqa: BLE001
            pass

    def reset(self) -> Any:
        self.check()
        import arcengine

        self._count()
        self.resets += 1
        return self._env.step(arcengine.GameAction.RESET, data={})

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        self.check()
        self._count()
        return self._env.step(action, data=dict(data or {}))

    # arc_agi.EnvironmentWrapper passthroughs some specialists may touch
    @property
    def observation_space(self) -> Any:
        return getattr(self._env, "observation_space", None)

    @property
    def action_space(self) -> Any:
        return getattr(self._env, "action_space", None)

    @property
    def info(self) -> Any:
        return getattr(self._env, "info", None)

    @property
    def environment_info(self) -> Any:
        return getattr(self._env, "environment_info", None)


# --------------------------------------------------------------------------
# banking handoff
# --------------------------------------------------------------------------

def _bank_modules() -> Any:
    """graft_bank if importable (shares the RUN-WIDE kill switch), else None."""
    try:
        import graft_bank

        return graft_bank
    except Exception:  # noqa: BLE001 — banking is optional, never fatal
        return None


def bank_plan_actions(level_seqs: dict[int, list[tuple]]) -> list[tuple]:
    """Concatenate per-level minimal token sequences into one replay plan.

    Under ONLY_RESET_LEVELS=true a search reset restarts the CURRENT level, so
    every recorded path is level-relative and completing level N leaves the
    engine at level N+1's start — the concatenation in level order is exactly
    the plan for a fresh play (post-WIN RESET is a FULL reset,
    arcengine/base_game.py:311-314)."""
    return [tok for lvl in sorted(level_seqs) for tok in level_seqs[lvl]]


def bank_crack(env: Any, level_seqs: dict[int, list[tuple]], *,
               guard: GuardedEnv | None = None,
               log: Any = None) -> tuple[bool, str]:
    """Replay the minimal plan on the fresh play post-WIN RESET opens.

    Runs the RAW env (not the guarded one): the plan is hard-capped at
    EXPLORER_V8_BANK_MAX_ACTIONS and pre-checked against the remaining
    envelope, so it cannot extend the run — see the envelope doc.  Honours and
    trips ``graft_bank._BankingKillSwitch`` so a server-side patch of the
    replay path disables banking run-wide for BOTH grafts.

    Returns (banked, reason)."""
    import arcengine

    if not _flag("EXPLORER_V8_BANK", "1"):
        return False, "disabled"
    # The replay starts on a FRESH play, so the plan must cover level 1 onward
    # with no gaps. It will not when the grinder took over a game on which the
    # LLM had already completed a level (v7's bounded-takeover path): those
    # early levels have no recorded minimal sequence, and replaying from
    # level k+1 on a fresh play cannot reproduce the win.
    lvls = sorted(level_seqs)
    if lvls and lvls != list(range(1, len(lvls) + 1)):
        return False, f"plan does not start at level 1 (covers {lvls})"
    empty = [n for n in lvls if not level_seqs[n]]
    if empty:
        return False, f"no recorded sequence for level(s) {empty}"
    plan = bank_plan_actions(level_seqs)
    if not plan:
        return False, "empty_plan"
    cap = _env_int("EXPLORER_V8_BANK_MAX_ACTIONS", 2000)
    if len(plan) > cap:
        return False, f"plan {len(plan)} > cap {cap}"

    gb = _bank_modules()
    if gb is not None and getattr(gb._BankingKillSwitch, "tripped", False):
        return False, f"kill switch tripped ({gb._BankingKillSwitch.reason})"

    # +1 for the RESET; a deliberately pessimistic per-action cost (4x the
    # measured 1/130 s) plus a finish margin must fit what is left.
    need = (len(plan) + 1) * _env_float("EXPLORER_V8_BANK_SEC_PER_ACTION", 0.03) \
        + _env_float("EXPLORER_V8_BANK_MARGIN_S", 10.0)
    if guard is not None:
        if guard.remaining_s() < need:
            return False, f"budget {guard.remaining_s():.0f}s < needed {need:.0f}s"
        if guard.remaining_actions() < len(plan) + 1:
            return False, "action headroom exhausted"

    def _apply(tok: tuple) -> Any:
        if tok[0] == "C":
            return env.step(arcengine.GameAction.ACTION6,
                            data={"x": int(tok[1]), "y": int(tok[2])})
        return env.step(arcengine.GameAction.from_id(int(tok[1])), data={})

    try:
        resp = env.step(arcengine.GameAction.RESET, data={})
        if resp is None or not resp.frame:
            if gb is not None:
                gb._BankingKillSwitch.trip("RESET rejected")
            return False, "abort+KILL: RESET rejected"
        if int(resp.levels_completed) != 0 or resp.state == arcengine.GameState.WIN:
            # Server-patch signature (graft_bank.py:242-247): post-WIN RESET no
            # longer opens a fresh play. Disable banking for the whole run.
            if gb is not None:
                gb._BankingKillSwitch.trip("post-WIN RESET did not open a fresh play")
            return False, "abort+KILL: RESET did not open a fresh play"
        levels = 0
        for i, tok in enumerate(plan, start=1):
            resp = _apply(tok)
            if resp is None or not resp.frame:
                return False, f"abort: engine refused step {i}/{len(plan)}"
            if resp.state == arcengine.GameState.GAME_OVER:
                return False, f"abort: replay died at {i}/{len(plan)}"
            if int(resp.levels_completed) < levels:
                return False, f"abort: level regression at {i}/{len(plan)}"
            levels = int(resp.levels_completed)
        if resp.state != arcengine.GameState.WIN:
            return False, f"abort: replay ended in {resp.state.name}, not WIN"
        return True, f"banked: replayed win in {len(plan)} actions"
    except Exception as exc:  # noqa: BLE001 — a broken replay must not touch the win
        return False, f"abort: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# the v8 grind
# --------------------------------------------------------------------------

def _import_core() -> tuple[Any, Any]:
    import search_core
    import specialists

    return search_core, specialists


def _grind_v8(session: Any, xs: dict[str, Any], level: int) -> None:
    """SearchCore portfolio takeover, drop-in for v7's ``_grind``.

    Called by v7's ``_maybe_grind`` (unchanged): the level-age trigger, the
    grind-owned self-harm gate, the bounded takeover, the per-level grind
    count, the concurrency semaphore and the cumulative-wall accounting all
    still belong to v7.  This function owns only what happens INSIDE one
    engagement."""
    game = session.game
    game_id = getattr(getattr(game, "game_run", None), "game_id", "?")
    raw_env = getattr(game, "env", None)
    if raw_env is None:
        xs["grind_exhausted"].add(level)
        print(f"[explorer-v8] {game_id}: no engine wrapper — grind unavailable",
              flush=True)
        return

    try:
        search_core, specialists = _import_core()
    except Exception as exc:  # noqa: BLE001 — fail-open to v7's blind BFS
        print(f"[explorer-v8] {game_id}: SearchCore unavailable ({exc!r}) "
              "— falling back to v7 BFS", flush=True)
        v7._grind_v7(session, xs, level)
        return

    genv = GuardedEnv(raw_env, session, xs, t0=_time.monotonic(),
                      **_engagement_caps())
    _engage(session, xs, level, genv, raw_env, game_id, search_core, specialists)


def _engagement_caps() -> dict[str, Any]:
    """The v7 engagement caps, unchanged, in GuardedEnv's keyword form."""
    time_cap_s = max(30, _env_int("EXPLORER_GRIND_TIME_S", 600))
    owned_time_cap_s = max(time_cap_s, _env_int("EXPLORER_OWNED_TIME_S", 1500))
    # engine-action ceiling: the v7 knob, additionally clamped to what the
    # wall cap can physically produce at the measured gateway rate (v7 shipped
    # 500000, which no 1500 s cap can ever reach — an inert ceiling).
    action_cap = min(max(1, _env_int("EXPLORER_GRIND_BUDGET", 500000)),
                     int(owned_time_cap_s * GATEWAY_ACT_PER_S * 1.5))
    return {"action_cap": action_cap, "time_cap_s": time_cap_s,
            "owned_time_cap_s": owned_time_cap_s}


def _engage(session: Any, xs: dict[str, Any], level: int, genv: "GuardedEnv",
            raw_env: Any, game_id: str, search_core: Any, specialists: Any,
            prebuilt: tuple | None = None) -> str:
    """One engagement: lane rotation, banking, narration, teardown.

    Shared by the stall-triggered grind and the early specialist probe; the
    probe hands over its already-warmed core through ``prebuilt`` so nothing
    is warmed or detected twice. Returns the stop reason."""
    print(f"[explorer-v8] {game_id}: engaging on level {level} — SearchCore "
          f"portfolio, caps {genv._action_cap} acts / {genv._time_cap_s}s "
          f"({genv._owned_time_cap_s}s owned)"
          + (" [early specialist]" if prebuilt else ""), flush=True)
    xs["grinding"] = True
    xs["diag"].setdefault("v8_engagements", 0)
    xs["diag"]["v8_engagements"] += 1
    stop_reason = "budget"
    level_seqs: dict[int, list[tuple]] = {}
    banked = False
    won = False

    def narrate(text: str) -> None:
        try:
            v7._TLS.narration = {"text": text, "diag": xs["diag"]}
        except Exception:  # noqa: BLE001
            pass

    def plan_text(tok: tuple) -> str:
        if tok[0] == "C":
            return f"CLICK(x={tok[1]},y={tok[2]})"
        return f"ACTION{tok[1]}"

    try:
        stop_reason = _run_search(session, xs, genv, game_id, search_core,
                                  specialists, level_seqs, narrate, plan_text,
                                  prebuilt=prebuilt)
        won = stop_reason == "game_won"
    except _GrindAbort as abort:
        stop_reason = abort.reason
    except Exception as exc:  # noqa: BLE001 — fail-open: end the engagement
        stop_reason = f"error:{type(exc).__name__}"
        print(f"[explorer-v8] {game_id}: search error {exc!r}", flush=True)

    if stop_reason == "frontier_exhausted" and not level_seqs:
        # every lane proved the level unreachable: never engage it again
        # (v7 rule, graft_explorer.py:777-778)
        xs["grind_exhausted"].add(level)

    if won:
        xs["diag"]["games_won_by_grinder"] = xs["diag"].get("games_won_by_grinder", 0) + 1
        banked, why = bank_crack(raw_env, level_seqs, guard=genv)
        xs["diag"]["v8_bank_result"] = why
        print(f"[explorer-v8] {game_id}: {why}", flush=True)
        if banked:
            xs["diag"]["games_banked_by_grinder"] = \
                xs["diag"].get("games_banked_by_grinder", 0) + 1
        narrate(
            "[EXPLORER] This game was fully SOLVED by automated search"
            + (" and re-played optimally on a fresh attempt (score banked)"
               if banked else "")
            + ". No further actions are needed on this game; if prompted, do "
            "not reset or replay — the result is already recorded.")
        # A cracked game has nothing left to win: every further LLM action
        # lands on the post-bank play and can only add actions to a play whose
        # levels are already recorded, while holding a worker the next wave
        # needs. Ask the harness to finish this game (v8.1; flag-gated).
        xs["v8_stop_game"] = True

    try:
        if not won and stop_reason not in ("cancelled", "reset_failed"):
            # leave the LLM at the current level's start (v7 contract)
            import arcengine

            raw_env.step(arcengine.GameAction.RESET, data={})
    except Exception:  # noqa: BLE001
        pass
    xs["grinding"] = False
    try:
        session.write_runtime_state()
    except Exception:  # noqa: BLE001
        pass
    print(f"[explorer-v8] {game_id}: grind ended ({stop_reason}) after "
          f"{genv.actions} engine actions / {genv.wall():.0f}s, "
          f"{len(xs['grind_unlocked_levels'])} grinder unlocks total"
          + (" [BANKED]" if banked else ""), flush=True)
    return stop_reason


# --------------------------------------------------------------------------
# v8.1 EARLY SPECIALIST PROBE — detection is no longer gated behind the stall
# --------------------------------------------------------------------------

def early_enabled() -> bool:
    return v8_enabled() and _flag("EXPLORER_V8_EARLY", "1")


# Specialist classes MEASURED to crack their game. Detection alone is not a
# crack predictor (see the comment at the whitelist check), and a detection
# that engages and fails is charged into the LLM's play, so the default
# engages only the class with a measured crack. Set to "*" to engage every
# detection (the v8.1 behaviour), or to a comma-separated class list.
CRACKING_SPECIALISTS = "ft09_gf2"


def _engage_whitelisted(name: str) -> bool:
    raw = os.environ.get("EXPLORER_V8_ENGAGE_SPECIALISTS", CRACKING_SPECIALISTS)
    allowed = {p.strip() for p in raw.split(",") if p.strip()}
    return "*" in allowed or name in allowed


def _early_probe_caps() -> dict[str, Any]:
    """Caps for the PROBE phase (warmup + detect) — far tighter than an
    engagement's. Measured cost across all 25 fixtures: 40-111 warmup + 4-114
    detect actions, worst case 206, mean 123."""
    wall = float(_env_int("EXPLORER_V8_EARLY_PROBE_S", 120))
    return {"action_cap": max(1, _env_int("EXPLORER_V8_EARLY_PROBE_ACTIONS", 800)),
            "time_cap_s": wall, "owned_time_cap_s": wall}


def early_specialist_probe(session: Any) -> str:
    """Run ``specialists.detect`` ONCE per game, right after warmup, with no
    stall precondition; engage immediately on a hit.

    WHY (measured, v8 smoke #1, 2026-08-26 — kernel arc3-v8-smoke v1): v7's
    trigger needs 120 scored actions or 10 LLM turns on ONE never-completed
    level.  ft09 spent its whole 3600 s box on **17 LLM actions** and completed
    two levels, so the trigger never armed: engagements=0, engine actions=0,
    and the specialist tier — 1.6 s to detect, 9.5 s to crack the game — was
    never consulted.  ft09's LLM banked 14.29 (the completion-share cap for
    2/6 levels) where a banked crack projects ~100.  Gating DETECTION behind a
    soak wastes it; only GENERIC search still waits for the stall.

    Cost and blast radius, stated honestly: the probe's engine actions are
    billed to the CURRENT level, so on a game with no specialist they are a
    tax on level 1's ``(b/a)^2``.  The probe therefore runs only while the
    level is still untouched (``EXPLORER_V8_EARLY_MAX_ACTIONS``, default 24
    scored actions — wide enough to survive gate contention at t=0) and only
    while the LLM has completed NO level (past that, a crack could not be
    banked from level 1 anyway, so the upside is gone while the tax remains).

    Returns a short outcome string, also recorded in ``diag['v8_early']``."""
    if not early_enabled():
        return "disabled"
    xs = v7._session_state(session)
    if xs.get("v8_early_done") or xs.get("grinding"):
        return xs.get("v8_early_done") or "grinding"
    game = getattr(session, "game", None)
    run = getattr(game, "game_run", None)
    if game is None or run is None or getattr(run, "state", None) != "playing":
        return "not_playing"
    game_id = getattr(run, "game_id", "?")
    raw_env = getattr(game, "env", None)
    if raw_env is None:
        return _early_done(xs, "no_engine")

    # engine actions only ever from the session's own worker thread (v7 rule;
    # the probe may run before the first executed action, which is where v7's
    # own marker gets set, so claim it here when it is still unset)
    if xs["worker_thread"] is None:
        xs["worker_thread"] = _threading.get_ident()
    elif xs["worker_thread"] != _threading.get_ident():
        return "other_thread"

    if session.stop_event.is_set():
        return "cancelled"
    try:
        from inference.framework import solver

        if solver._is_run_complete(game) or solver._is_engine_game_over(game):
            return _early_done(xs, "game_over")
        level = solver._level_number(game)
    except Exception:  # noqa: BLE001 — dev/test harnesses without the bundle
        level = 1

    # --- self-harm window: untouched level, no LLM-completed level ---------
    if xs["completed_levels"]:
        return _early_done(xs, "levels_completed")
    spent = int(getattr(session, "action_count", 0) or 0)
    if spent > _env_int("EXPLORER_V8_EARLY_MAX_ACTIONS", 24):
        return _early_done(xs, f"too_late({spent} actions)")

    # --- the same run-level guards a stall-triggered grind must pass -------
    try:
        if session.runtime_limit_reached():
            return "runtime_cap"
        cap = session.solver.max_actions_per_game
        if cap is not None and session.action_count >= int(cap):
            return _early_done(xs, "action_cap")
        soft = session.solver.soft_time_remaining_seconds()
        if soft is not None and float(soft) < 120.0:
            return _early_done(xs, "soft_time")
    except Exception:  # noqa: BLE001 — a broken probe never blocks
        pass
    with v7._RUN_LOCK:
        if v7._RUN_T0 is None:
            v7._RUN_T0 = _time.monotonic()
        run_elapsed = _time.monotonic() - v7._RUN_T0
        spent_wall = v7._GRIND_WALL_SPENT[0]
    if run_elapsed >= _env_int("EXPLORER_RUN_CUTOFF_S", 18000):
        return _early_done(xs, "run_cutoff")
    if spent_wall >= _env_int("EXPLORER_RUN_GRIND_BUDGET_S", 2700):
        return _early_done(xs, "run_grind_budget")

    try:
        search_core, specialists = _import_core()
    except Exception as exc:  # noqa: BLE001 — no SearchCore, no probe
        return _early_done(xs, f"core_unavailable:{type(exc).__name__}")

    if not _PROBE_GATE.acquire(blocking=False):
        return "probe_gate_busy"      # retry on a later poll, inside the window
    t0 = _time.monotonic()
    genv = GuardedEnv(raw_env, session, xs, t0=t0, label="probe",
                      **_early_probe_caps())
    outcome = "probe_error"
    engaged = False
    retry = False
    try:
        prebuilt = _warm_and_detect(xs, genv, game_id, search_core, specialists)
        specialist = prebuilt[3]
        xs["diag"]["v8_early_probe_actions"] = genv.actions
        xs["diag"]["v8_early_detect"] = specialist
        print(f"[explorer-v8] {game_id}: EARLY probe -> specialist="
              f"{specialist!r} in {genv.actions} engine actions "
              f"({genv.wall():.1f}s, level {level}, {spent} LLM actions spent)",
              flush=True)
        if specialist is None:
            outcome = "no_specialist"
        elif not hasattr(specialists, "SOLVERS") or specialist not in specialists.SOLVERS:
            outcome = f"no_solver_for({specialist})"
        elif not _flag("EXPLORER_V8_EARLY_ENGAGE", "1"):
            outcome = f"detected({specialist})_engage_off"
        elif not _engage_whitelisted(specialist):
            # DETECTION IS NOT A CRACK PREDICTOR (measured 2026-08-25,
            # results/falsifier_specialist_20260825_220501.json): of the four
            # detectors only ft09_gf2 cracks its game (373 snapshot / 1233
            # reset-replay actions). tn36_program, sc25_glyph and wa30_grabdrag
            # each detect, unlock ~2 levels, then spend 344 848 / 878 559 /
            # 765 657 actions and still fail. A failed engagement cannot open a
            # fresh play, so those actions are billed into the play the LLM
            # keeps using — measured at -1.19 points when they land on level 1
            # and -12.14 when the search has climbed to level 3. Engage only
            # classes measured to CRACK; record the rest.
            outcome = f"detected({specialist})_not_whitelisted"
        elif not v7._GRIND_GATE.acquire(blocking=False):
            # another game is grinding; keep the detection but do NOT close
            # the probe — retry the ENGAGEMENT on a later poll, while the
            # session is still inside its action window
            outcome = "engage_gate_busy"
            retry = True
        else:
            engaged = True
            try:
                xs["grinds_per_level"][level] = xs["grinds_per_level"].get(level, 0) + 1
                xs["diag"]["grinder_engagements"] += 1
                xs["diag"]["v8_early_engagements"] = \
                    xs["diag"].get("v8_early_engagements", 0) + 1
                # widen the caps from probe-sized to engagement-sized; t0 stays
                # at the probe's start, so the probe's wall counts against the
                # engagement (conservative, never the other way round)
                caps = _engagement_caps()
                # TIGHT SPECIALIST ABORT. The generic ceiling (292 500) is far
                # too loose for a specialist lane: ft09's crack needs 1233
                # engine actions END TO END, while a specialist that is going
                # to fail spends hundreds of thousands. The cap cannot undo the
                # damage of a failure (the score cost saturates by ~2000
                # actions — measured identical at 2k and 292k), but it bounds
                # how far up the LEVEL LADDER a doomed search climbs, and that
                # is the 10x term: failing on level 1 costs -1.19, failing on
                # level 3 costs -12.14.
                spec_cap = _env_int("EXPLORER_V8_SPECIALIST_MAX_ACTIONS", 4000)
                genv._action_cap = min(caps["action_cap"],
                                       genv.actions + max(1, spec_cap))
                genv._time_cap_s = caps["time_cap_s"]
                genv._owned_time_cap_s = caps["owned_time_cap_s"]
                stop_reason = _engage(session, xs, level, genv, raw_env, game_id,
                                      search_core, specialists, prebuilt=prebuilt)
                outcome = f"engaged({specialist}):{stop_reason}"
            finally:
                v7._GRIND_GATE.release()
    except _GrindAbort as abort:
        outcome = f"abort:{abort.reason}"
    except Exception as exc:  # noqa: BLE001 — fail-open, always
        outcome = f"error:{type(exc).__name__}"
        print(f"[explorer-v8] {game_id}: early probe error {exc!r}", flush=True)
    finally:
        if not engaged:
            _PROBE_GATE.release()
            try:    # leave the LLM at the current level's start (v7 contract)
                import arcengine

                raw_env.step(arcengine.GameAction.RESET, data={})
            except Exception:  # noqa: BLE001
                pass
        with v7._RUN_LOCK:
            v7._GRIND_WALL_SPENT[0] += _time.monotonic() - t0
    print(f"[explorer-v8] {game_id}: EARLY probe done ({outcome}) after "
          f"{genv.actions} engine actions / {genv.wall():.1f}s", flush=True)
    if retry:
        return outcome          # deliberately NOT closed: a later poll retries
    return _early_done(xs, outcome)


def _early_done(xs: dict[str, Any], outcome: str) -> str:
    """Record the probe's verdict and close it for this game (one per game)."""
    xs["v8_early_done"] = outcome
    xs["diag"]["v8_early"] = outcome
    return outcome


def _maybe_grind_v8(session: Any) -> None:
    """v7's poll, with the early specialist probe in front of it.

    The probe is INDEPENDENT of the stall trigger; v7's trigger, self-harm
    gate and bounded takeover then run exactly as before for GENERIC search."""
    try:
        early_specialist_probe(session)
    except Exception:  # noqa: BLE001 — never break the poll
        pass
    if not _flag("EXPLORER_V8_STALL_GRIND", "1"):
        # CRACK-OR-NOTHING. The generic stall grind cracked nothing in smoke #2
        # (dc22 and sk48 spent 292 687 / 292 635 engine actions for 2 and 0
        # unlocks), and a non-cracking engagement cannot open a fresh play, so
        # every one of those actions is billed into the play the LLM keeps
        # using. With this off, only a specialist class measured to CRACK may
        # spend engine actions beyond the probe.
        return
    v7._maybe_grind_v7(session)


def _warm_and_detect(xs: dict[str, Any], genv: GuardedEnv, game_id: str,
                     search_core: Any, specialists: Any) -> tuple:
    """RESET -> archetype -> SearchCore -> warmup -> specialists.detect.

    The shared head of BOTH entry points: a stall-triggered engagement
    (``_run_search``) and the early probe (``early_specialist_probe``).
    Returns ``(archetype, core, go, specialist, obs0)``; raises ``_GrindAbort``
    if a guard trips. ``specialist`` is None when detection declines or is
    off."""
    obs0 = genv.reset()
    if obs0 is None:
        return None, None, None, None, None
    archetype = search_core.archetype_frame0(list(obs0.available_actions or []))

    core = search_core.SearchCore(
        genv, backend="reset_replay",
        warmup_rounds=_env_int("EXPLORER_V8_WARMUP_ROUNDS", 6),
        max_states=_env_int("EXPLORER_V8_MAX_STATES", 20000),
        dead_click_k=_env_int("EXPLORER_V8_DEAD_CLICK_K", 3),
    )
    go = search_core.GoExplorer(
        core, tier=_env_int("EXPLORER_V8_GOEXPLORE_TIER", 3),
        k_rollout=_env_int("EXPLORER_V8_ROLLOUT_K", 30),
        momentum=_env_int("EXPLORER_V8_MOMENTUM_PCT", 92) / 100.0)

    warm_cap = _env_int("EXPLORER_V8_WARMUP_ACTIONS", 600)
    a0 = genv.actions
    try:
        core.warmup_and_freeze()
    except _GrindAbort:
        raise
    except Exception as exc:  # noqa: BLE001 — an unusable mask is not fatal
        print(f"[explorer-v8] {game_id}: warmup failed ({exc!r}) — raw mask",
              flush=True)
    warm_spent = genv.actions - a0
    xs["diag"]["v8_warmup_actions"] = warm_spent
    if warm_spent > warm_cap:
        print(f"[explorer-v8] {game_id}: warmup overran ({warm_spent} > "
              f"{warm_cap} actions)", flush=True)
    print(f"[explorer-v8] {game_id}: arch={archetype} warmup {warm_spent} acts, "
          f"mask {len(core.active_mask_cells or [])} cells", flush=True)

    specialist = None
    if _flag("EXPLORER_V8_SPECIALIST", "1"):
        d0 = genv.actions
        try:
            specialist = specialists.detect(core)
        except _GrindAbort:
            raise
        except Exception:  # noqa: BLE001 — fail-open by contract
            specialist = None
        genv.check()      # detect() swallows _GrindAbort per-detector
        xs["diag"]["v8_detect_actions"] = genv.actions - d0
        print(f"[explorer-v8] {game_id}: specialist={specialist} "
              f"({genv.actions - d0} probe actions)", flush=True)
    xs["diag"]["v8_specialist"] = specialist
    return archetype, core, go, specialist, obs0


def _run_search(session: Any, xs: dict[str, Any], genv: GuardedEnv, game_id: str,
                search_core: Any, specialists: Any,
                level_seqs: dict[int, list[tuple]],
                narrate: Any, plan_text: Any, prebuilt: tuple | None = None) -> str:
    """Warmup -> specialist detection -> per-level lane rotation. Returns the
    stop reason; "game_won" means the engine reported WIN.

    ``prebuilt`` hands over an already warmed-and-detected
    ``(archetype, core, go, specialist, obs0)`` from the early probe, so a
    probe that hits does NOT pay for a second warmup or a second detection."""
    import arcengine

    if prebuilt is None:
        prebuilt = _warm_and_detect(xs, genv, game_id, search_core, specialists)
    archetype, core, go, specialist, obs0 = prebuilt
    if core is None or obs0 is None:
        return "reset_failed"

    order = list(search_core.DISPATCH_ORDER[archetype])
    ranking = (["specialist"] + order) if specialist else order
    max_tier = _env_int("EXPLORER_V8_MAX_TIER", 4)
    slice0 = max(30.0, float(_env_int("EXPLORER_V8_SLICE_S", 120)))

    base_levels = int(obs0.levels_completed or 0)
    while True:
        genv.check()
        target = base_levels + 1
        solved_res = None
        closed: dict[str, float] = {}

        def _ever() -> int:
            ec = core.change.ever_changed
            return int(ec.sum()) if ec is not None else 0

        slice_s = slice0
        while solved_res is None:
            genv.check()
            open_lanes = [n for n in ranking
                          if n not in closed or _ever() > closed[n]]
            if not open_lanes:
                return "frontier_exhausted"
            for name in open_lanes:
                remain = genv.remaining_s()
                if remain <= 2:
                    raise _GrindAbort("time_cap")
                # front (last-successful) lane gets a double slice: a
                # deterministic lane redoes all prior work after a timeout
                lane_slice = slice_s * 2 if name == ranking[0] else slice_s
                budget = min(lane_slice, remain)
                if name == "specialist":
                    res = specialists.solve_level(core, specialist, target, budget)
                else:
                    res = search_core.solve_with(core, name, target, budget,
                                                 max_tier, go)
                # specialists.solve_level and specialists.detect wrap their
                # bodies in `except Exception`, which SWALLOWS a _GrindAbort
                # raised by the guard mid-lane. Re-assert the envelope after
                # every lane so a swallowed trip cannot buy an extra lane.
                genv.check()
                if res.get("solved"):
                    solved_res = res
                    ranking.remove(name)
                    ranking.insert(0, name)     # move-to-front
                    break
                if name == "specialist":
                    closed[name] = float("inf")   # one-shot, deterministic
                elif name in search_core.DETERMINISTIC_LANES \
                        and res.get("reason") == "exhausted":
                    closed[name] = _ever()
                elif name in closed:
                    del closed[name]
            slice_s *= 2

        handle = solved_res.get("handle")
        path = list(getattr(handle, "path", None) or [])
        base_levels = target
        level_seqs[target] = path
        xs["diag"]["levels_unlocked_by_grinder"] += 1
        if not xs["grind_unlocked_levels"]:
            # cost of reaching the FIRST unlock — the only stretch governed by
            # the short (600 s) generic cap; afterwards the game is grind-owned
            # and the 1500 s owned cap applies (v7 rule, kept)
            xs["diag"]["v8_first_unlock_actions"] = genv.actions
        xs["grind_unlocked_levels"].add(target)
        xs["completed_levels"].add(target)
        core.backend.adopt(handle)
        print(f"[explorer-v8] {game_id}: level {target} UNLOCKED by "
              f"{solved_res.get('algo', '?')} ({len(path)} actions minimal, "
              f"{genv.actions} spent)", flush=True)

        obs = getattr(handle, "obs", None)
        if obs is not None and obs.state == arcengine.GameState.WIN:
            return "game_won"
        narrate(
            f"[EXPLORER UNLOCK] Level {target} was just unlocked by an "
            "automated exhaustive search, NOT by your plan. The minimal "
            "winning sequence from the level start was: "
            + ", ".join(plan_text(t) for t in path)
            + ". The LAST action crossed the boundary. Infer this game's "
            "mechanic from that sequence and apply it deliberately on the "
            "current level.")


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------

def install() -> str:
    """Install v7, then (only with EXPLORER_V8 on) swap in the v8 grind.

    With the flag off this returns v7's own verdict and mutates nothing —
    the lane is byte-identical to the arm that flew 1.51."""
    note = v7.install()
    if not v8_enabled():
        return note + " | v8: OFF (v7 grind)"
    if getattr(v7._grind, "_v8", False):
        return note + " | v8: SKIP (already applied)"
    try:
        core_mod, spec_mod = _import_core()
    except Exception as exc:  # noqa: BLE001 — decline cleanly, v7 stays
        return note + f" | v8: SKIP (SearchCore unimportable: {exc!r})"
    for name in ("archetype_frame0", "SearchCore", "GoExplorer", "solve_with",
                 "DISPATCH_ORDER", "DETERMINISTIC_LANES"):
        if not hasattr(core_mod, name):
            return note + f" | v8: SKIP (search_core lacks {name})"
    for name in ("detect", "solve_level"):
        if not hasattr(spec_mod, name):
            return note + f" | v8: SKIP (specialists lacks {name})"
    if not hasattr(v7, "_grind_v7"):
        v7._grind_v7 = v7._grind        # keep the blind BFS as a fallback lane
    _grind_v8._v8 = True                # type: ignore[attr-defined]
    v7._grind = _grind_v8
    early = _install_early(note)
    return note + " | v8: OK (SearchCore portfolio grind)" + early


def _install_early(note: str) -> str:
    """v8.1 seams: the stall-independent probe in front of v7's poll, and the
    finish-the-game-after-a-crack hook. Both are additive and flag-gated."""
    if not _flag("EXPLORER_V8_EARLY", "1"):
        return " | early: OFF"
    if not hasattr(v7, "_maybe_grind_v7"):
        v7._maybe_grind_v7 = v7._maybe_grind
    if not getattr(v7._maybe_grind, "_v8_early", False):
        _maybe_grind_v8._v8_early = True    # type: ignore[attr-defined]
        v7._maybe_grind = _maybe_grind_v8
    out = " | early: OK (stall-independent specialist probe)"
    try:
        from inference.framework import solver

        cls = solver._HarnessGameSession
        inner = cls.should_stop
        if not getattr(inner, "_v8_stop_patched", False):
            def should_stop(self: Any) -> bool:
                try:
                    xs = getattr(self, "_xpl_state", None)
                    if (xs is not None and xs.get("v8_stop_game")
                            and _flag("EXPLORER_V8_STOP_AFTER_CRACK", "1")):
                        return True     # cracked+banked: free the worker
                except Exception:  # noqa: BLE001
                    pass
                return inner(self)

            should_stop._v8_stop_patched = True   # type: ignore[attr-defined]
            should_stop._xpl_patched = True       # keep v7's idempotence marker
            cls.should_stop = should_stop
            out += " + stop-after-crack"
    except Exception as exc:  # noqa: BLE001 — optional seam, never fatal
        out += f" (stop-after-crack unavailable: {type(exc).__name__})"
    return out


def uninstall() -> None:
    """Restore v7's grind + poll (tests; never called in the scored lane)."""
    if hasattr(v7, "_grind_v7"):
        v7._grind = v7._grind_v7
    if hasattr(v7, "_maybe_grind_v7"):
        v7._maybe_grind = v7._maybe_grind_v7


def explorer_diagnostics(session: Any) -> dict[str, Any]:
    return v7.explorer_diagnostics(session)


# re-exported so a bundle can import one module
FrontierGraph = v7.FrontierGraph
VolatilityMask = v7.VolatilityMask
_GRIND_GATE = v7._GRIND_GATE
_TLS = v7._TLS
_threading = _threading
