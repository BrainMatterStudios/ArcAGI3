"""graft_sweep.py — the harness spends the opening actions as EXPERIMENTS, not as moves.

THE FINDING THIS COMES FROM (AutumnBench, arXiv:2510.19788, 43 interactive grid worlds,
517 humans vs frontier models): reasoning models "use less than 7 % of their actions for
resets and no-ops combined" against humans at ~12.5 % EACH, and "do not treat resets as
special actions, unlike humans" -- they fail to use them as experimental instruments. The
09-10 sweep called this the largest idea in the literature that nobody has yet tested.

Our own corpus says the same thing from the other side: across 2,047 tool calls made on
levels the run went on to CLEAR, the agent forward-simulated a next state ZERO times and
searched in 2.2 %. It does not experiment. Six results say we cannot talk it into it.

So the harness runs the experiment itself. Before the model's first decision, it fires
each valid action once and then repeats one of them, spending ~|A|+1 actions to buy a
complete one-observation-per-action map of the game's dynamics plus a determinism check.

WHY THIS IS CHEAP IN SCORE, WHICH IS THE WHOLE POINT
  Score is min(weighted mean, completion-share cap) with level weights 1..N, so level 1
  carries weight 1 of a 21-55 denominator. Spending 6 actions there costs a fraction of
  2.8-4.8 points even if it degrades level 1's own efficiency term -- and on a level the
  run never clears, actions cost EXACTLY ZERO. This is the one place the scoring rule
  actively rewards experimentation.

COMPOSES WITH graft_effects: the sweep populates the effect table from action 1 instead
of letting it fill in haphazardly over 50 turns.

SAFETY, all fatal-to-the-sweep and never to the game:
  * stops immediately on level_completed / game_over / run_complete / any non-executed action
  * never fires RESET (the stock harness filters it from valid_actions and auto-fires it
    on death; competition mode swallows it anyway -- api.py:316-334)
  * caps at SWEEP_MAX_ACTIONS regardless of how many actions are valid
  * runs once per run, at the first analyze() only
  * any exception disables it for the whole run and the stock path is untouched
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Any

MARK = "[SWEEP]"
_STATE: dict[str, Any] = {"installed": False, "disabled": False, "skips": Counter(), "done": set()}
_LOCK = threading.Lock()
_RESULTS: dict[str, list[dict]] = {}


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _cfg() -> dict[str, int]:
    return {
        "enable": _env_int("SWEEP_ENABLE", 0, 0),
        "max_actions": _env_int("SWEEP_MAX_ACTIONS", 8, 1),
        "repeat": _env_int("SWEEP_REPEAT", 1, 0),      # repeat the first action to test determinism
    }


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "disabled": _STATE["disabled"],
            "runs_swept": len(_STATE["done"]), "skips": dict(_STATE["skips"]), "cfg": _cfg()}


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] += 1


_TERMINAL = ("level_completed", "game_over", "run_complete", "done")


def _terminal(result: dict) -> bool:
    return any(bool(result.get(k)) for k in _TERMINAL)


def run_sweep(step_env, valid_actions: list[str], key: str) -> list[dict]:
    """Fire each valid action once, then repeat the first, stopping at anything terminal.

    Returns one record per fired action. MOUSE is skipped: it needs coordinates, and a
    blind click is not an experiment -- it is a guess with an unbounded target space.
    """
    cfg = _cfg()
    plan = [a for a in (valid_actions or [])
            if a and a.upper() != "RESET" and not a.upper().startswith("MOUSE")]
    plan = plan[: cfg["max_actions"]]
    if cfg["repeat"] and plan:
        plan = plan + [plan[0]]
    out: list[dict] = []
    for act in plan:
        try:
            payload = step_env({"actions": [{"action": act}]})
        except Exception:  # noqa: BLE001 — a sweep failure must never cost the game
            _skip("step_env_failed")
            break
        if not isinstance(payload, dict):
            _skip("bad_payload")
            break
        # solver.step_env returns a FLAT payload: success carries executed_count>=1,
        # while _error_payload / _terminal_payload carry executed=False. Check both.
        executed = bool(payload.get("executed_count", 0)) or bool(payload.get("executed", False))
        rec = {"action": act,
               "executed": executed,
               "board_changed": bool(payload.get("board_changed")),
               "level": payload.get("level"),
               "terminal": _terminal(payload)}
        out.append(rec)
        if not rec["executed"]:
            _skip("not_executed")
            break
        if rec["terminal"]:
            break
    _RESULTS[key] = out
    return out


def block(key: str) -> str:
    recs = _RESULTS.get(key) or []
    if not recs:
        return ""
    seen: dict[str, list[bool]] = {}
    for r in recs:
        seen.setdefault(r["action"], []).append(r["board_changed"])
    lines = []
    for act, changes in seen.items():
        if len(changes) > 1:
            det = "same result both times" if len(set(changes)) == 1 else "DIFFERENT results on repeat"
            lines.append(f"- {act}: changed board {sum(changes)}/{len(changes)} ({det})")
        else:
            lines.append(f"- {act}: {'changed the board' if changes[0] else 'did NOT change the board'}")
    return ("The harness spent the opening actions probing each action once (a controlled "
            "sweep, not moves):\n" + "\n".join(lines) + "\n")


def install() -> str:
    if _STATE["installed"]:
        return "sweep: SKIP (already applied)"
    if not _cfg()["enable"]:
        return "sweep: SKIP (SWEEP_ENABLE unset)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"sweep: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "sweep: SKIP (missing ToolAgent)"
    for name in ("analyze", "_build_user_prompt"):
        if getattr(cls, name, None) is None:
            return f"sweep: SKIP (ToolAgent.{name} missing)"

    stock_analyze = cls.analyze
    stock_prompt = cls._build_user_prompt

    def analyze(self, *args, **kwargs):
        key = str(kwargs.get("transcript_path") or (args[4] if len(args) > 4 else "") or "")
        self._sweep_key = key
        step_env = kwargs.get("step_env")
        # action_count is the second positional arg; only sweep at the very start of a run
        action_count = args[1] if len(args) > 1 else kwargs.get("action_num", 0)
        if (not _STATE["disabled"] and key and step_env is not None
                and key not in _STATE["done"] and (action_count or 0) == 0):
            with _LOCK:
                _STATE["done"].add(key)
            try:
                run_sweep(step_env, kwargs.get("valid_actions") or [], key)
            except Exception:  # noqa: BLE001
                _skip("sweep_failed")
        return stock_analyze(self, *args, **kwargs)

    def _build_user_prompt(self, *args, **kwargs):
        text = stock_prompt(self, *args, **kwargs)
        if _STATE["disabled"]:
            return text
        try:
            b = block(getattr(self, "_sweep_key", ""))
        except Exception:  # noqa: BLE001
            _skip("block_failed")
            return text
        return (text + "\n" + b) if b else text

    cls.analyze = analyze
    cls._build_user_prompt = _build_user_prompt
    _STATE["installed"] = True
    return "sweep: analyze+_build_user_prompt: OK"
