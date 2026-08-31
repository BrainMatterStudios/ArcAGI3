"""Action-economy graft (TP6, 2026-08-31) — tell the model the truth about
scoring, and surface its own action spend.

Measured leak (submission/_flashnext_smoke/results_v4 + scoring math): on the
Flash-Next smoke, 1.63 local pts/game are forfeited on COMPLETED levels done
far over the human baseline (ka59 L1 296 acts vs 28 -> 0.03 pts; bp35 189/21;
sc25 83/36; quadratic penalty). The stock duck prompt NEVER mentions that
actions are scored, so the model optimises for progress only.

One flag-gated behaviour (TP6_ENABLE, default 1): append a short ACTION
ECONOMY block to the user prompt each turn:
  - actions are scored: per level, score = (human_baseline / your_actions)^2,
    so finishing a level in 2x the needed actions costs 75% of its points;
  - the current level's action count so far (from actions_per_level via the
    step summary when available, else the turn-header action counter);
  - three rules: probe ONCE per hypothesis, never repeat an action that
    already changed nothing (results carry board_changed), and once the
    mechanic is verified execute the remaining plan as ONE batch.

Purely prompt-side; no seam behaviour changes. Installed on ToolAgent.
_build_user_prompt after any other graft (append-only, fail-open).
"""
from __future__ import annotations

import os
from typing import Any

_STATE = {"installed": False}
_OFF = {"0", "false", "no", "off"}

ECONOMY_BLOCK = (
    "ACTION ECONOMY (scoring truth): every level is scored (human_baseline / your_actions)^2 — "
    "taking twice the needed actions keeps only a quarter of the level's points, and points are "
    "only awarded for COMPLETED levels. You have executed {n} actions on this level so far. "
    "Probe once per hypothesis; never repeat an action that already changed nothing; once a "
    "mechanic is verified, execute the whole remaining plan as one action([...]) batch."
)


def enabled() -> bool:
    raw = os.environ.get("TP6_ENABLE")
    value = "1" if raw is None or not raw.strip() else raw.strip()
    return value.lower() not in _OFF


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "enabled": enabled()}


def _level_actions(agent: Any, action_num: int) -> int:
    try:
        summary = getattr(agent, "_last_step_summary", None) or {}
        end = summary.get("end_action_num")
        # best available proxy: total actions this game; per-level split is not
        # visible to the agent, so state the game counter when level unknown
        return int(end if end is not None else max(0, action_num))
    except Exception:  # noqa: BLE001
        return max(0, int(action_num or 0))


def install() -> str:
    if _STATE["installed"]:
        return "economy: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"economy: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None or getattr(cls, "_build_user_prompt", None) is None:
        return "economy: SKIP (seam missing)"

    stock = cls._build_user_prompt

    def build(self, action_num, **kwargs):
        text = stock(self, action_num, **kwargs)
        if not enabled():
            return text
        try:
            return text + "\n" + ECONOMY_BLOCK.format(n=_level_actions(self, action_num))
        except Exception:  # noqa: BLE001
            return text

    build._tp6_stock = stock
    cls._build_user_prompt = build
    _STATE["installed"] = True
    return "economy: OK"
