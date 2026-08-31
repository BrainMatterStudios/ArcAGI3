"""Durable-timeout graft (TP8, 2026-08-31) — a timed-out python tool call
must not discard what its actions observed.

Measured (R8 forensics): ~130 tool timeouts across 11 Flash-Next games
(wa30 46/117 turns, ar25 22 of its last 32). The sandbox returns the executed
actions' payloads even on timeout (python_tool_sandbox.run_sandboxed_python
timeout paths carry host_action_results), but ToolAgent._run_python_tool's
error branch shows the model ONLY "Tool timed out after 30s" — the outcomes
(board_changed, level, state) are dropped, and the model dead-reckons
(ar25's terminal net-zero oscillation; ft09's last 9 actions swallowed).

Fix, one seam: rebind run_sandboxed_python inside tool_agent; when the result
is a timeout WITH executed actions, rewrite the error to a recap naming the
executed count and the last action's outcome, so the stock error branch
renders it to the model. TP8_ENABLE=0 = pass-through.
"""
from __future__ import annotations

import os
from typing import Any

_STATE = {"installed": False}
_STOCK: dict = {}
_OFF = {"0", "false", "no", "off"}

RECAP = (
    "{error} — HOWEVER {n} action(s) DID execute before the timeout and their effects are real. "
    "Last executed action result: level={level}, state={state}, board_changed={bc}, score={score}, "
    "level_completed={lc}, game_over={go}. `current_frame` already reflects these actions — "
    "re-observe it instead of re-running the actions."
)


def enabled() -> bool:
    raw = os.environ.get("TP8_ENABLE")
    value = "1" if raw is None or not raw.strip() else raw.strip()
    return value.lower() not in _OFF


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "enabled": enabled()}


def _recap(result: dict[str, Any]) -> dict[str, Any]:
    try:
        error = str(result.get("error") or "")
        if "timed out" not in error.lower():
            return result
        executed = [r for r in (result.get("action_results") or [])
                    if isinstance(r, dict) and r.get("executed")]
        if not executed:
            return result
        last = executed[-1]
        out = dict(result)
        out["error"] = RECAP.format(
            error=error.rstrip("."), n=len(executed),
            level=last.get("level"), state=last.get("state"), bc=last.get("board_changed"),
            score=last.get("score"), lc=last.get("level_completed"), go=last.get("game_over"),
        )
        return out
    except Exception:  # noqa: BLE001
        return result


def install() -> str:
    if _STATE["installed"]:
        return "durable: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"durable: SKIP (import failed: {exc!r})"
    stock = getattr(agent_mod, "run_sandboxed_python", None)
    if stock is None:
        return "durable: SKIP (run_sandboxed_python not bound in tool_agent)"
    _STOCK["fn"] = stock

    def run(*args, **kwargs):
        result = _STOCK["fn"](*args, **kwargs)
        if not enabled():
            return result
        if isinstance(result, dict):
            return _recap(result)
        return result

    run._tp8_stock = stock
    agent_mod.run_sandboxed_python = run
    _STATE["installed"] = True
    return "durable: OK"
