"""Death-protocol graft (TP7, 2026-08-31) — plan around per-level move budgets.

Measured (docs/research-2026-08-29/R8-flashnext-forensics-2026-08-31.md): on
the Flash-Next smoke, 72.5% of ALL actions sat inside attempts that ended in a
GAME_OVER; the deaths come at FIXED per-level cadences (sc25 every 27 actions,
vc33 every 51, ka59 ~100, wa30 ~200 — move budgets, not hazards). The model
named the cadence in 2 of 8 games and planned against it in 0; retries replay
the same ground from scratch (retry actions ACCUMULATE into the level's scored
action count).

One flag-gated behaviour (TP7_ENABLE, default 1): track GAME_OVERs per level
on the session (actions at each death; gaps between deaths). After the first
death on a level, append a DEATH PROTOCOL block to the user prompt:
  - name the event: likely a per-level MOVE BUDGET or timer, not (only) a
    hazard; retried actions still count against the level's score;
  - after >=2 deaths with a stable gap, state the estimated budget N and
    actions already spent this life;
  - the rule: probe on an early life, BANK a complete plan in your notes,
    then execute it within ONE life as one batch.

Session state rides the graft_control SessionState when present, else its own
attribute. Prompt seam: ToolAgent._build_user_prompt (append-only, fail-open);
death detection: _HarnessGameSession._execute_action wrapper.
"""
from __future__ import annotations

import os
from typing import Any

_STATE = {"installed": False}
_OFF = {"0", "false", "no", "off"}

PROTOCOL_FIRST = (
    "DEATH PROTOCOL: this level has had {k} GAME_OVER(s) (at level-action counts {at}). "
    "On these games a GAME_OVER is usually a per-level MOVE BUDGET or timer expiring — not "
    "necessarily a mistake you made. Retried actions STILL COUNT against this level's score. "
    "Treat lives as experiment slots: probe on an early life, bank a COMPLETE plan in your notes, "
    "then execute it within one life as a single batch."
)
PROTOCOL_BUDGET = (
    " Estimated budget: ~{budget} actions per life (stable death cadence); you have spent "
    "{spent} actions this life — keep the executing plan within the remaining ~{left}."
)


def enabled() -> bool:
    raw = os.environ.get("TP7_ENABLE")
    value = "1" if raw is None or not raw.strip() else raw.strip()
    return value.lower() not in _OFF


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "enabled": enabled()}


class DeathState:
    def __init__(self) -> None:
        self.level: int | None = None
        self.level_actions = 0          # actions on the current level (all lives)
        self.death_at: list[int] = []   # level_actions at each GAME_OVER


def _dstate(session: Any) -> DeathState:
    st = getattr(session, "_tp7", None)
    if st is None:
        st = DeathState()
        try:
            session._tp7 = st
        except Exception:  # noqa: BLE001
            pass
    return st


def _session_of(agent: Any) -> Any:
    cb = getattr(agent, "_step_env_callback", None)
    return getattr(cb, "__self__", None)


def estimate_budget(death_at: list[int]) -> int | None:
    """Stable cadence: gaps between consecutive deaths within 20% of each other."""
    if len(death_at) < 2:
        return None
    gaps = [b - a for a, b in zip(death_at, death_at[1:])]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None
    mean = sum(gaps) / len(gaps)
    if all(abs(g - mean) <= 0.2 * mean + 2 for g in gaps):
        return round(mean)
    # fall back to the first death position if gaps are noisy but positive
    return None


def protocol_text(st: DeathState) -> str | None:
    if not st.death_at:
        return None
    text = PROTOCOL_FIRST.format(k=len(st.death_at), at=st.death_at[-4:])
    budget = estimate_budget(st.death_at) or (st.death_at[0] if len(st.death_at) == 1 else None)
    if budget:
        spent = max(0, st.level_actions - st.death_at[-1])
        text += PROTOCOL_BUDGET.format(budget=budget, spent=spent, left=max(0, budget - spent))
    return text


def install() -> str:
    if _STATE["installed"]:
        return "deaths: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"deaths: SKIP (import failed: {exc!r})"
    agent_cls = agent_mod.ToolAgent
    session_cls = solver_mod._HarnessGameSession
    if getattr(agent_cls, "_build_user_prompt", None) is None or getattr(session_cls, "_execute_action", None) is None:
        return "deaths: SKIP (seam missing)"

    stock_execute = session_cls._execute_action

    def execute(self, action, *args, **kwargs):
        payload = stock_execute(self, action, *args, **kwargs)
        if not enabled():
            return payload
        try:
            st = _dstate(self)
            level = payload.get("level")
            if level is not None and level != st.level:
                st.level = level
                st.level_actions = 0
                st.death_at = []
            if payload.get("executed"):
                st.level_actions += 1
            if payload.get("game_over"):
                st.death_at.append(st.level_actions)
        except Exception:  # noqa: BLE001
            pass
        return payload

    execute._tp7_stock = stock_execute
    session_cls._execute_action = execute

    stock_prompt = agent_cls._build_user_prompt

    def build(self, action_num, **kwargs):
        text = stock_prompt(self, action_num, **kwargs)
        if not enabled():
            return text
        try:
            sess = _session_of(self)
            if sess is not None:
                block = protocol_text(_dstate(sess))
                if block:
                    text = text + "\n" + block
        except Exception:  # noqa: BLE001
            pass
        return text

    build._tp7_stock = stock_prompt
    agent_cls._build_user_prompt = build

    _STATE["installed"] = True
    return "deaths: OK"
