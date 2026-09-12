"""Fresh-mind level retry graft (RETRY, 2026-09-03).

Evidence (docs/research-2026-09-02/R-loss-ledger.md): over 125 stock game-runs,
74% of level clears happen in the first hour; 60% of all actions land in a
level bucket that never clears; a stuck level has median actions ~1.0x the
human baseline with frame novelty 0.90 — the model is not looping, it is
pursuing ONE wrong hypothesis to the clock. The carried world-model note is
updated on ~11% of turns, so a failed hypothesis is never explicitly
abandoned. The only counterfactual worth >= +0.5 levels/game is "+1 level on
games already started"; clearing a stuck level, even inefficiently, unlocks
the next level whose action bucket starts fresh.

TRIGGER SOURCE (verified on the june_stock bundle + the pinned taaf
framework): the ToolAgent receives `step_env=session.step_env` on every
analyze() call (solver.py:294-303), and `step_env.__self__` is the live
`_HarnessGameSession`. From it:
  session.game.current_state.levels_completed   -> index of the level in play
  session.game.game_run.actions_per_level[idx]  -> actions accumulated in
                                                  that level's bucket (RESETs
                                                  and retries included;
                                                  taaf game.py:566)
  session.game.game_run.base_actions_per_level  -> per-level HUMAN BASELINES
                                                  (list, from arcengine
                                                  environment_info.baseline_actions
                                                  — populated OFFLINE, e.g.
                                                  tu93 [19,16,34,42,123,80,14,23,111];
                                                  None in submission mode ->
                                                  RETRY_ABS fallback)
  session.game.current_state.available_actions  -> RESET (id 0) validity
  session._execute_action(RESET, ...)           -> the normal action path
                                                  (history, actions_per_level,
                                                  runtime state, viewer event;
                                                  the same call the harness's
                                                  own GAME_OVER auto-reset uses,
                                                  solver.py:663-665)

MECHANISM (all flags read at call time; RETRY_ENABLE=0 is pure pass-through):
 1. TRIGGER, checked at the top of every analyze() turn: the current level's
    accumulated actions >= ceil(RETRY_K x baseline) [K=3] (or >= RETRY_ABS
    [200] when no baseline), the level has not cleared, no retry fired on
    this level within the last RETRY_COOLDOWN [150] actions of that level's
    bucket, and fewer than RETRY_MAX [2] retries fired on it. Guards: the run
    is playing; the engine is not GAME_OVER (the harness's own auto-reset is
    pending there and is untouched); the last engine action was not already
    a RESET; RESET is listed in available_actions; should_stop() is False.
 2. ACTION: one level RESET through session._execute_action (counted and
    logged like any action, generated_tokens=0 like the auto-reset). Then
    for the NEXT analyzer call: (a) the carried world_model / goal_model /
    action_model / recent_findings / open_questions / current_plan fields
    are blanked (cross_level_notes kept — the same field set stock wipes on
    a level transition, tool_agent.py:1113-1126); (b) a FRESH MIND block
    (<= 1,200 chars, quoting the abandoned note in <= 600 chars) is appended
    to the user prompt exactly once. Optional RETRY_CLEAR_HISTORY=1 [0] also
    drops the carried chat history so the next turn starts from the frame.
    2026-09-12 (validated plan step 4): RETRY_MODE=wipe [reset] is the
    no-RESET dose — no engine action at all; the board stays where it is,
    the chat history AND the level note are wiped, and the FRESH MIND block
    says so. RETRY_TURNS=N [0 = off] replaces the action-multiple trigger
    with "N analyze() turns on the same level since the last fire" (the
    action trigger fired only 5 times in a 25-game wave).
 3. TELEMETRY: status() exposes retries_fired, levels_cleared_after_retry,
    the per-(game, level) retry log and the clear log; the transcript gets
    "[RETRY] game=.. level=.. actions=.." and "[RETRY-CLEAR] game=.. level=..
    actions=.." marker lines written through tool_agent._append_transcript_section
    (the harness's own transcript writer) under a "[HARNESS RETRY]" section.

Conventions: module-level _STATE/_STOCK, install() rebinds ToolAgent methods
only, fail-open try/except around all graft logic, no threads, no file writes
outside the transcript. install() -> "retry: OK" / "retry: SKIP (...)".
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

_STATE: dict[str, Any] = {
    "installed": False,
    "retries_fired": 0,
    "levels_cleared_after_retry": 0,
    "retry_log": [],
    "clear_log": [],
    "skips": {},
}
_STOCK: dict[str, Any] = {}
_LOCK = threading.Lock()
_OFF = {"0", "false", "no", "off"}

DEFAULT_K = 3.0
DEFAULT_ABS = 200
DEFAULT_COOLDOWN = 150
DEFAULT_MAX = 2
QUOTE_CHARS = 600
BLOCK_CHARS = 1200
TRANSCRIPT_LABEL = "HARNESS RETRY"

_WIPED_FIELDS = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "current_plan",
)
_QUOTE_FIELDS = (
    ("world_model", "World model"),
    ("goal_model", "Goal model"),
    ("action_model", "Action model"),
    ("current_plan", "Plan"),
)

FRESH_MIND_BLOCK = (
    "FRESH MIND (harness level retry {n}/{max}): the harness RESET this level at action {at} "
    "because it was not cleared after {spent} actions ({budget}). The board is back at the start "
    "of level {level}; the world model carried for this level was cleared on purpose. "
    "The hypothesis you were pursuing is now treated as FAILED:\n<<<\n{quote}\n>>>\n"
    "Do not resume it. State TWO alternative hypotheses that explain the transitions in "
    "`history` differently (another goal, another meaning of the objects, another effect of "
    "the actions), pick the cheapest to test, and test it with at most 5 actions in one "
    "`action([...])` batch before committing to anything longer."
)


FRESH_MIND_WIPE_BLOCK = (
    "FRESH MIND (harness history wipe {n}/{max}): the harness cleared your carried chat history "
    "and the world model for this level at action {at} because level {level} was not cleared "
    "after {spent} actions ({budget}). The board is UNCHANGED — nothing was reset; `history` and "
    "`transitions` still hold what happened. The hypothesis you were pursuing is now treated as "
    "FAILED:\n<<<\n{quote}\n>>>\n"
    "Do not resume it. State TWO alternative hypotheses that explain the transitions in "
    "`history` differently (another goal, another meaning of the objects, another effect of "
    "the actions), pick the cheapest to test, and test it with at most 5 actions in one "
    "`action([...])` batch before committing to anything longer."
)


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("RETRY_ENABLE", "1").lower() not in _OFF


def retry_k() -> float:
    try:
        return max(0.1, float(_env("RETRY_K", str(DEFAULT_K))))
    except ValueError:
        return DEFAULT_K


def retry_abs() -> int:
    try:
        return max(1, int(_env("RETRY_ABS", str(DEFAULT_ABS))))
    except ValueError:
        return DEFAULT_ABS


def retry_cooldown() -> int:
    try:
        return max(0, int(_env("RETRY_COOLDOWN", str(DEFAULT_COOLDOWN))))
    except ValueError:
        return DEFAULT_COOLDOWN


def retry_max() -> int:
    try:
        return max(0, int(_env("RETRY_MAX", str(DEFAULT_MAX))))
    except ValueError:
        return DEFAULT_MAX


def clear_history() -> bool:
    return _env("RETRY_CLEAR_HISTORY", "0").lower() not in _OFF


def retry_mode() -> str:
    """'reset' (engine level RESET, the 09-03 design) or 'wipe' (no engine action; 2026-09-12)."""
    return "wipe" if _env("RETRY_MODE", "reset").lower() == "wipe" else "reset"


def retry_turns() -> int:
    """Turns-on-level trigger; 0 keeps the action-multiple trigger."""
    try:
        return max(0, int(_env("RETRY_TURNS", "0")))
    except ValueError:
        return 0


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "installed": _STATE["installed"],
            "enabled": enabled(),
            "k": retry_k(),
            "abs": retry_abs(),
            "cooldown": retry_cooldown(),
            "max": retry_max(),
            "clear_history": clear_history(),
            "mode": retry_mode(),
            "turns": retry_turns(),
            "retries_fired": _STATE["retries_fired"],
            "levels_cleared_after_retry": _STATE["levels_cleared_after_retry"],
            "retry_log": [dict(r) for r in _STATE["retry_log"]],
            "clear_log": [dict(r) for r in _STATE["clear_log"]],
            "skips": dict(_STATE["skips"]),
        }


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] = _STATE["skips"].get(reason, 0) + 1


# ------------------------------------------------------------ per-game state ---
class RetryState:
    def __init__(self) -> None:
        self.runtime_dir: Any = None
        self.fires: dict[int, int] = {}          # level idx -> retries fired
        self.last_fire_actions: dict[int, int] = {}  # level idx -> bucket count after the RESET
        self.turns_on_level: dict[int, int] = {}     # level idx -> analyze() turns seen (RETRY_TURNS trigger)
        self.last_fire_turn: dict[int, int] = {}     # level idx -> turns_on_level at the last fire
        self.cleared: set[int] = set()           # level idx with a [RETRY-CLEAR] already logged
        self.fresh_pending: dict[str, Any] | None = None
        self.injected: dict[str, Any] | None = None  # parked copy of the block injected this turn


def _rstate(agent: Any, state_path: Any = None) -> RetryState:
    """Get (or create) the agent's retry state; reset when the game changes.

    Mirrors ToolAgent._ensure_session, which keys a session by
    state_path.parent (the per-game runtime dir).
    """
    st = getattr(agent, "_retry", None)
    if st is None:
        st = RetryState()
        try:
            agent._retry = st
        except Exception:  # noqa: BLE001
            pass
    if state_path is not None:
        try:
            runtime_dir = Path(state_path).parent
            if st.runtime_dir is not None and st.runtime_dir != runtime_dir:
                fresh = RetryState()
                fresh.runtime_dir = runtime_dir
                try:
                    agent._retry = fresh
                except Exception:  # noqa: BLE001
                    pass
                return fresh
            st.runtime_dir = runtime_dir
        except Exception:  # noqa: BLE001
            pass
    return st


# -------------------------------------------------------------- session view ---
def _session_of(step_env: Any) -> Any:
    return getattr(step_env, "__self__", None)


def _game_id(sess: Any) -> str:
    try:
        return str(getattr(sess.game.game_run, "game_id", "") or "?")
    except Exception:  # noqa: BLE001
        return "?"


def _level_index(sess: Any) -> int | None:
    """0-based index of the level in play, or None when the run is over."""
    try:
        idx = int(sess.game.current_state.levels_completed)
        n = int(sess.game.number_of_levels)
    except Exception:  # noqa: BLE001
        return None
    if idx < 0 or idx >= n:
        return None
    return idx


def _actions_on_level(sess: Any, idx: int) -> int | None:
    try:
        per_level = sess.game.game_run.actions_per_level
        return int(per_level[idx])
    except Exception:  # noqa: BLE001
        return None


def _baseline(sess: Any, idx: int) -> int | None:
    """Per-level human baseline when the engine exposes it (offline), else None."""
    try:
        baselines = sess.game.game_run.base_actions_per_level
        if baselines is None:
            baselines = getattr(sess.game, "base_actions_per_level", None)
        if baselines is None:
            return None
        value = int(baselines[idx])
        return value if value > 0 else None
    except Exception:  # noqa: BLE001
        return None


def threshold_for(baseline: int | None) -> int:
    if baseline is None:
        return retry_abs()
    return max(1, int(math.ceil(retry_k() * baseline - 1e-9)))


def _reset_available(sess: Any, arcengine: Any) -> bool:
    """RESET is filtered out of the model-facing valid_actions by the solver;
    ask the engine state directly (absent list => not available)."""
    try:
        available = sess.game.current_state.available_actions
        ids = set(int(a) for a in (available or []))
    except Exception:  # noqa: BLE001
        return False
    return int(arcengine.GameAction.RESET.value) in ids


def _engine_action_names(game: Any, arcengine: Any) -> list[str]:
    """solver._engine_action_names, reproduced (RESET filtered, order kept)."""
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:  # noqa: BLE001
            continue
        if name == "RESET" or name in names:
            continue
        names.append(name)
    return names


def _transcript_path(state_path: Any, kwargs: dict[str, Any]) -> Path | None:
    path = kwargs.get("transcript_path")
    if path is not None:
        return Path(path)
    try:
        sp = Path(state_path)
        return sp.parent / f"{sp.stem}_analyzer.txt"  # stock's analyzer_log fallback
    except Exception:  # noqa: BLE001
        return None


def _write_marker(agent_mod: Any, transcript_path: Path | None, line: str) -> None:
    if transcript_path is None:
        return
    try:
        agent_mod._append_transcript_section(transcript_path, TRANSCRIPT_LABEL, line)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------- trigger logic ---
def should_fire(st: RetryState, sess: Any, arcengine: Any, should_stop: Any = None) -> dict[str, Any] | None:
    """Return the trigger record when a retry must fire now, else None."""
    if retry_max() <= 0:
        return None
    try:
        run = sess.game.game_run
        if run is None or str(getattr(run, "state", "playing")) != "playing":
            return None
        raw_state = sess.game.current_state.raw.state
        if raw_state in (arcengine.GameState.GAME_OVER, arcengine.GameState.WIN):
            _skip("terminal_state")   # GAME_OVER: the harness auto-reset is pending — never touch it
            return None
    except Exception:  # noqa: BLE001
        return None
    mode = retry_mode()
    if mode == "reset" and getattr(sess, "last_engine_action", None) == "RESET":
        return None                   # board already fresh (auto-reset or our own retry)
    idx = _level_index(sess)
    if idx is None:
        return None
    st.turns_on_level[idx] = st.turns_on_level.get(idx, 0) + 1
    actions = _actions_on_level(sess, idx)
    if actions is None:
        return None
    baseline = _baseline(sess, idx)
    turns_trigger = retry_turns()
    if turns_trigger > 0:
        # turns-on-level trigger: N analyze() turns on this level since the last fire
        limit = turns_trigger
        if st.turns_on_level[idx] - st.last_fire_turn.get(idx, 0) < turns_trigger:
            return None
    else:
        limit = threshold_for(baseline)
        if actions < limit:
            return None
    if st.fires.get(idx, 0) >= retry_max():
        _skip("max_retries")
        return None
    last = st.last_fire_actions.get(idx)
    if turns_trigger <= 0 and last is not None and actions - last < retry_cooldown():
        _skip("cooldown")
        return None
    if should_stop is not None:
        try:
            if should_stop():
                _skip("should_stop")
                return None
        except Exception:  # noqa: BLE001
            pass
    if mode == "reset" and not _reset_available(sess, arcengine):
        _skip("reset_unavailable")
        return None
    return {
        "mode": mode,
        "game": _game_id(sess),
        "level": idx + 1,
        "level_index": idx,
        "actions": actions,
        "baseline": baseline,
        "threshold": limit,
        "retry": st.fires.get(idx, 0) + 1,
    }


def quote_note(knowledge: Any) -> str:
    if not isinstance(knowledge, dict):
        return "(no world-model note was recorded)"
    parts = []
    for key, label in _QUOTE_FIELDS:
        text = " ".join(str(knowledge.get(key) or "").split())
        if text:
            parts.append(f"{label}: {text}")
    quote = " | ".join(parts)
    if not quote:
        return "(no world-model note was recorded)"
    if len(quote) > QUOTE_CHARS:
        quote = quote[: QUOTE_CHARS - 1].rstrip() + "…"
    return quote


def render_block(info: dict[str, Any]) -> str:
    baseline = info.get("baseline")
    if baseline is not None:
        budget = f"human baseline ~{baseline}, limit {info.get('k', retry_k()):g}x"
    else:
        budget = f"limit {info.get('threshold')} actions"
    template = FRESH_MIND_WIPE_BLOCK if info.get("mode") == "wipe" else FRESH_MIND_BLOCK
    block = template.format(
        n=info.get("retry"), max=info.get("max", retry_max()), at=info.get("action_num"),
        spent=info.get("actions"), budget=budget, level=info.get("level"), quote=info.get("quote"),
    )
    if len(block) > BLOCK_CHARS:
        block = block[: BLOCK_CHARS - 1].rstrip() + "…"
    return block


def fire(agent: Any, st: RetryState, sess: Any, info: dict[str, Any], *, agent_mod: Any, arcengine: Any,
         transcript_path: Path | None) -> bool:
    """Arm the fresh-mind turn, then issue the RESET through the normal action path.

    ORDER MATTERS (judge finding): every piece of bookkeeping that stops a
    re-fire — st.fires, the cooldown anchor, the [RETRY] marker, the module
    counters, the pending block — is committed BEFORE session._execute_action.
    solver._execute_action commits the engine move first and only then writes
    the runtime state / viewer payload (solver.py:694-731); if one of those
    writes raises, the wrapper swallows it, and a post-hoc bookkeeping order
    would leave the level armed to RESET again next turn. Returns True when the
    engine call returned normally.
    """
    knowledge = getattr(agent, "_summarized_knowledge", None)
    idx = int(info["level_index"])
    actions = int(info["actions"])
    wipe = info.get("mode") == "wipe"
    st.fires[idx] = st.fires.get(idx, 0) + 1
    # reset mode: the RESET lands in this level's bucket; wipe mode: nothing is added
    st.last_fire_actions[idx] = actions if wipe else actions + 1
    st.last_fire_turn[idx] = st.turns_on_level.get(idx, 0)
    info["action_num"] = None
    try:
        info["action_num"] = int(getattr(sess, "action_count", 0) or 0) + 1
    except Exception:  # noqa: BLE001
        pass
    info["quote"] = quote_note(knowledge)
    info["k"] = retry_k()
    info["max"] = retry_max()
    # (a) suppress the carried level-scoped note; cross_level_notes survives
    if isinstance(knowledge, dict):
        for key in _WIPED_FIELDS:
            knowledge[key] = ""
    if wipe or clear_history():
        try:
            agent._history_messages = []
        except Exception:  # noqa: BLE001
            pass
    st.fresh_pending = dict(info)
    st.injected = None
    marker = (f"[RETRY] game={info['game']} level={info['level']} actions={info['actions']} "
              f"baseline={info['baseline'] if info['baseline'] is not None else '-'} "
              f"threshold={info['threshold']} retry={info['retry']}/{info['max']} action_num={info['action_num']} "
              f"mode={info.get('mode', 'reset')}")
    _write_marker(agent_mod, transcript_path, marker)
    with _LOCK:
        _STATE["retries_fired"] += 1
        _STATE["retry_log"].append({k: info.get(k) for k in ("game", "level", "actions", "baseline", "threshold",
                                                                "retry", "action_num", "mode")})
    if wipe:
        return True                                  # no engine action: the board stays as it is
    action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
    try:
        sess._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
    except Exception as exc:  # noqa: BLE001
        # The engine may already have committed the RESET (history + bucket)
        # before a runtime-state/viewer write failed; the bookkeeping above
        # already stands, so this level cannot re-fire on the next turn.
        _skip("reset_error")
        _write_marker(agent_mod, transcript_path,
                      f"[RETRY-ERROR] game={info['game']} level={info['level']} {type(exc).__name__}: {str(exc)[:160]}")
        return False
    return True


def check_clears(st: RetryState, sess: Any, *, agent_mod: Any, transcript_path: Path | None) -> int:
    """Log [RETRY-CLEAR] once for every retried level the run has now passed."""
    try:
        completed = int(sess.game.game_run.levels_completed)
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for idx in sorted(st.fires):
        if idx in st.cleared or completed <= idx:
            continue
        st.cleared.add(idx)
        actions = _actions_on_level(sess, idx)
        rec = {"game": _game_id(sess), "level": idx + 1, "actions": actions, "retries": st.fires[idx]}
        _write_marker(agent_mod, transcript_path,
                      f"[RETRY-CLEAR] game={rec['game']} level={rec['level']} actions={rec['actions']} "
                      f"retries={rec['retries']}")
        with _LOCK:
            _STATE["levels_cleared_after_retry"] += 1
            _STATE["clear_log"].append(rec)
        n += 1
    return n


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "retry: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"retry: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "retry: SKIP (missing ToolAgent)"
    for name in ("analyze", "_build_user_prompt"):
        if getattr(cls, name, None) is None:
            return f"retry: SKIP (ToolAgent.{name} missing)"
    if getattr(agent_mod, "_append_transcript_section", None) is None:
        return "retry: SKIP (_append_transcript_section missing)"

    _STOCK["analyze"] = cls.analyze
    _STOCK["build_user_prompt"] = cls._build_user_prompt

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        sess = None
        st = None
        transcript_path = None
        if enabled():
            try:
                import arcengine  # noqa: PLC0415
                st = _rstate(self, state_path)
                sess = _session_of(step_env)
                transcript_path = _transcript_path(state_path, kwargs)
                if sess is not None:
                    check_clears(st, sess, agent_mod=agent_mod, transcript_path=transcript_path)
                    info = should_fire(st, sess, arcengine, kwargs.get("should_stop"))
                    if info is not None:
                        fire(self, st, sess, info, agent_mod=agent_mod, arcengine=arcengine,
                             transcript_path=transcript_path)
                        # read the LIVE session either way: a swallowed write error after
                        # the engine committed the RESET still moved action_count
                        action_num = int(getattr(sess, "action_count", action_num) or action_num)
                        valid_actions = _engine_action_names(sess.game, arcengine)
            except Exception:  # noqa: BLE001
                sess = None
        result = _STOCK["analyze"](self, state_path, action_num, valid_actions=valid_actions,
                                   step_env=step_env, **kwargs)
        if enabled() and st is not None:
            try:
                if sess is not None:
                    check_clears(st, sess, agent_mod=agent_mod, transcript_path=transcript_path)
                if result is not None and st.injected is not None:
                    if getattr(result, "retryable_failure", False) and not str(
                            getattr(result, "reasoning", "") or "").strip():
                        # the request carrying the block died before any reply: re-arm it
                        st.fresh_pending = st.injected
                    st.injected = None
            except Exception:  # noqa: BLE001
                pass
        return result

    def build_user_prompt(self, action_num, *args, **kwargs):
        text = _STOCK["build_user_prompt"](self, action_num, *args, **kwargs)
        if not enabled():
            return text
        try:
            st = getattr(self, "_retry", None)
            if st is None or not st.fresh_pending:
                return text
            block = render_block(st.fresh_pending)
            st.injected = st.fresh_pending
            st.fresh_pending = None
            return text + "\n" + block
        except Exception:  # noqa: BLE001
            return text

    analyze._retry_stock = _STOCK["analyze"]
    build_user_prompt._retry_stock = _STOCK["build_user_prompt"]
    cls.analyze = analyze
    cls._build_user_prompt = build_user_prompt
    _STATE["installed"] = True
    return "retry: OK"
