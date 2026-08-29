"""Explorer fallback graft (Pack 4, 2026-08-29) — a model-free frontier walk
takes over a stuck level, or the last minutes of a game, and hands back.

Installed AFTER graft_control. Two triggers, both checked at the top of every
LLM turn (ToolAgent.analyze) from the session state Pack 2 maintains:

  STALL TIER 3   after the harness RESET tier has fired at least once on this
                 level and since_new >= TP4_STALL_T3 (default 30) again, and
                 the explorer has not yet spent its per-level budget:
                 run frontier_explorer for up to TP4_BUDGET actions (default
                 800) or until the level changes / game ends. Engine actions
                 via the gateway cost ~8 ms each, so 800 actions is seconds.
  ENDGAME        time_remaining <= TP4_ENDGAME_S (default 300) and no level
                 progress in the last TP4_ENDGAME_QUIET_S (default 900):
                 run the explorer until TP4_ENDGAME_STOP_S (30) remain.

After a run the next user prompt carries a HARNESS NOTE (what was tried, how
many actions, whether a level was completed) and Pack 2's stall counters are
reset. Measured offline (bench_explorer.py): alone, the explorer reaches L1 on
10/25 public games within 800 actions and 15/25 within 3000; on the LLM's
zero-level games it adds 2 (800) to 4 (3000). Its job is unlocking depth for
the LLM, not scoring the level itself (an 800-action level scores ~0).
"""
from __future__ import annotations

import os
import time
from typing import Any

_STATE = {"installed": False}
_OFF = {"0", "false", "no", "off"}
_ANALYZE_STOCK: dict[str, Any] = {}

EXPLORER_NOTE = (
    "HARNESS NOTE: a model-free explorer took over for {n} actions ({why}); it {outcome}. "
    "It tried {tested} distinct (state, action) pairs across {nodes} board states. "
    "All of its actions are in `history`. {tail}"
)


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("TP4_ENABLE", "1").lower() not in _OFF


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def stall_t3() -> int:
    return max(1, _int("TP4_STALL_T3", 30))


def budget() -> int:
    return max(1, _int("TP4_BUDGET", 800))


def runs_per_level() -> int:
    return max(0, _int("TP4_RUNS_PER_LEVEL", 1))


def endgame_s() -> float:
    return float(_int("TP4_ENDGAME_S", 300))


def endgame_quiet_s() -> float:
    return float(_int("TP4_ENDGAME_QUIET_S", 900))


def endgame_stop_s() -> float:
    return float(_int("TP4_ENDGAME_STOP_S", 30))


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "enabled": enabled(), "stall_t3": stall_t3(),
            "budget": budget(), "runs_per_level": runs_per_level(), "endgame_s": endgame_s()}


# ------------------------------------------------------------- the run ---
def run_explorer(session: Any, solver_mod: Any, arcengine: Any, fe: Any, *, max_actions: int,
                 deadline_s: float | None = None, why: str = "stall") -> dict[str, Any]:
    """Drive the session's engine with the frontier explorer until the level
    changes, the game ends, the budget is spent, or the deadline passes."""
    game = session.game
    level0 = int(solver_mod._level_number(game))
    ex = fe.FrontierExplorer(seed=int(time.time()) & 0xFFFF)
    grid = solver_mod._grid_from_state(game.current_state)
    key = ex.observe(grid, list(game.current_state.available_actions))
    n = 0
    outcome = "made no level progress"
    t0 = time.monotonic()
    while n < max_actions:
        if deadline_s is not None and (time.monotonic() - t0) >= deadline_s:
            outcome = "stopped at the time limit"
            break
        try:
            if session.should_stop():
                outcome = "stopped (session ending)"
                break
        except Exception:  # noqa: BLE001
            pass
        if solver_mod._is_engine_game_over(game):
            action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
            session._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
            n += 1
            grid = solver_mod._grid_from_state(game.current_state)
            key = ex.observe(grid, list(game.current_state.available_actions))
            continue
        idx, _why = ex.choose(key)
        cand = ex.nodes[key].candidates[idx]
        action = arcengine.ActionInput(id=arcengine.GameAction.from_name(cand.action), data=dict(cand.data))
        payload = session._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
        n += 1
        if payload.get("run_complete"):
            outcome = "COMPLETED THE GAME"
            break
        if payload.get("level_completed") or int(solver_mod._level_number(game)) != level0:
            outcome = f"COMPLETED level {level0}; you are now on level {solver_mod._level_number(game)}"
            break
        grid = solver_mod._grid_from_state(game.current_state)
        new_key = ex.observe(grid, list(game.current_state.available_actions))
        ex.record(key, idx, new_key)
        key = new_key
    tested = ex.stats["edges"] + ex.stats["noops"]
    return {"actions": n, "outcome": outcome, "tested": tested, "nodes": ex.stats["nodes"],
            "why": why, "wall_s": round(time.monotonic() - t0, 1)}


def _note(rec: dict[str, Any]) -> str:
    tail = ("Build on the new level from a fresh look at `current_frame`."
            if "COMPLETED" in rec["outcome"] else
            "The explored actions did not progress the level: prefer hypotheses that explain "
            "why, and try targets or sequences the explorer could not (it never chains actions "
            "with intent).")
    return EXPLORER_NOTE.format(n=rec["actions"], why=rec["why"], outcome=rec["outcome"],
                                tested=rec["tested"], nodes=rec["nodes"], tail=tail)


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "explore: SKIP (already applied)"
    try:
        import graft_control as tc  # noqa: PLC0415
        import frontier_explorer as fe  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"explore: SKIP (module missing: {exc!r})"
    if not tc._STATE.get("installed"):
        return "explore: SKIP (graft_control not installed)"
    try:
        import arcengine  # noqa: PLC0415
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"explore: SKIP (import failed: {exc!r})"
    agent_cls = agent_mod.ToolAgent
    stock_analyze = agent_cls.analyze

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        try:
            sess = getattr(step_env, "__self__", None)
            if sess is not None and enabled():
                st = tc._state(sess)
                runs = getattr(st, "explorer_runs_this_level", 0)
                level = st.level
                if getattr(st, "explorer_level", None) != level:
                    st.explorer_level = level
                    st.explorer_runs_this_level = 0
                    runs = 0
                rec = None
                timing = {}
                try:
                    timing = sess.timing_payload()
                except Exception:  # noqa: BLE001
                    pass
                remaining = timing.get("time_remaining_seconds")
                quiet = time.monotonic() - getattr(st, "last_progress_at", time.monotonic())
                if (st.since_new >= stall_t3() and getattr(st, "resets_this_level", 0) >= 1
                        and runs < runs_per_level()):
                    rec = run_explorer(sess, solver_mod, arcengine, fe, max_actions=budget(),
                                       deadline_s=None if remaining is None else max(5.0, remaining - endgame_stop_s()),
                                       why=f"{st.since_new} actions without a new board state")
                    st.explorer_runs_this_level = runs + 1
                elif (remaining is not None and remaining <= endgame_s() and quiet >= endgame_quiet_s()
                      and not getattr(st, "endgame_done", False)):
                    st.endgame_done = True
                    rec = run_explorer(sess, solver_mod, arcengine, fe, max_actions=10 ** 6,
                                       deadline_s=max(1.0, remaining - endgame_stop_s()),
                                       why="endgame: no level progress recently and the clock is almost out")
                if rec is not None:
                    st.since_new = 0
                    st.streak = 0
                    st.last_reset_note = _note(rec)
                    st.explorer_last = rec
                    try:
                        sess.write_runtime_state()
                    except Exception:  # noqa: BLE001
                        pass
                    action_num = sess.action_count
                    valid_actions = solver_mod._engine_action_names(sess.game)
        except Exception:  # noqa: BLE001
            pass
        return _ANALYZE_STOCK["fn"](self, state_path, action_num, valid_actions=valid_actions,
                                    step_env=step_env, **kwargs)

    _ANALYZE_STOCK["fn"] = stock_analyze
    analyze._tp4_stock = stock_analyze
    agent_cls.analyze = analyze

    # progress clock for the endgame trigger: any level change stamps it
    stock_after = tc._after_action

    def after_action(st, before, after, payload):
        prev_level = st.level
        stock_after(st, before, after, payload)
        if st.level != prev_level or not hasattr(st, "last_progress_at"):
            st.last_progress_at = time.monotonic()

    after_action._tp4_stock = stock_after
    tc._after_action = after_action

    _STATE["installed"] = True
    return "explore: OK"
