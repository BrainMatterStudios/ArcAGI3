"""Tests for the two-tier knowledge ledger.

Run:  .venv/bin/python -m pytest submission/_duck_fixes/test_ledger_fixes.py -q
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
for p in (str(TAAF_INFERENCE), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ledger_fixes  # noqa: E402


KNOWLEDGE = {
    "world_model": "CONFIRMED: walls block movement\nthe red tile might be lava",
    "goal_model": "reach the flag maybe",
    "action_model": "CONFIRMED: ACTION7_SEMANTICS = undo last move",
    "recent_findings": "probed row 3",
    "open_questions": "what does the bar mean",
    "current_plan": "walk onto the red tile",
    "cross_level_notes": "keys persist",
}


def fresh_agent_cls(monkeypatch):
    from inference.agent import tool_agent
    ta = importlib.reload(tool_agent)
    monkeypatch.delenv("LEDGER_TWO_TIER", raising=False)
    return ta


def make_agent(ta, summary):
    agent = object.__new__(ta.ToolAgent)
    agent._last_step_summary = summary
    agent._summarized_knowledge = dict(KNOWLEDGE)
    return agent


# ---------- pure transform ----------

def test_split_ledger():
    confirmed, demoted = ledger_fixes.split_ledger(KNOWLEDGE["world_model"])
    assert confirmed == "CONFIRMED: walls block movement"
    assert demoted == ["the red tile might be lava"]


def test_death_transform_keeps_confirmed_demotes_rest():
    k = dict(KNOWLEDGE)
    ledger_fixes.apply_death_transform(k)
    assert k["world_model"] == "CONFIRMED: walls block movement"
    assert k["goal_model"] == ""
    assert "ACTION7_SEMANTICS" in k["action_model"]
    assert "red tile might be lava" in k["open_questions"]
    assert "reach the flag maybe" in k["open_questions"]
    assert "what does the bar mean" in k["open_questions"]  # original kept


def test_death_transform_fallback_without_confirmed_lines():
    k = {key: v.replace("CONFIRMED: ", "") for key, v in KNOWLEDGE.items()}
    before = dict(k)
    ledger_fixes.apply_death_transform(k)
    assert k == before  # FIX A behavior: keep everything


def test_demoted_note_is_capped():
    k = dict(KNOWLEDGE)
    k["goal_model"] = "\n".join(f"hypothesis number {i} about the mechanics" for i in range(60))
    ledger_fixes.apply_death_transform(k)
    added = k["open_questions"].split("Unverified at death", 1)[1]
    assert len(added) <= ledger_fixes.DEMOTED_CAP_CHARS


# ---------- patched method on the real class ----------

def test_disabled_delegates_verbatim(monkeypatch):
    ta = fresh_agent_cls(monkeypatch)
    ledger_fixes = importlib.reload(sys.modules["ledger_fixes"])
    assert "ledger: OK" in ledger_fixes.patch_two_tier_ledger()
    agent = make_agent(ta, {"game_over": True})
    agent._update_summarized_knowledge_from_step_summary()
    # env off -> upstream behavior = full wipe of the six keys
    assert agent._summarized_knowledge["world_model"] == ""
    assert agent._summarized_knowledge["cross_level_notes"] == "keys persist"


def test_enabled_death_keeps_confirmed(monkeypatch):
    ta = fresh_agent_cls(monkeypatch)
    ledger_fixes = importlib.reload(sys.modules["ledger_fixes"])
    ledger_fixes.patch_two_tier_ledger()
    monkeypatch.setenv("LEDGER_TWO_TIER", "1")
    agent = make_agent(ta, {"game_over": True})
    agent._update_summarized_knowledge_from_step_summary()
    k = agent._summarized_knowledge
    assert k["world_model"] == "CONFIRMED: walls block movement"
    assert k["current_plan"] == ""
    assert "GAME_OVER just occurred" in k["recent_findings"]
    assert "red tile might be lava" in k["open_questions"]


def test_enabled_level_transition_still_clears(monkeypatch):
    ta = fresh_agent_cls(monkeypatch)
    ledger_fixes = importlib.reload(sys.modules["ledger_fixes"])
    ledger_fixes.patch_two_tier_ledger()
    monkeypatch.setenv("LEDGER_TWO_TIER", "1")
    agent = make_agent(ta, {"level_transition": True})
    agent._update_summarized_knowledge_from_step_summary()
    assert agent._summarized_knowledge["world_model"] == ""


def test_prompt_line_env_gated(monkeypatch):
    ta = fresh_agent_cls(monkeypatch)
    ledger_fixes = importlib.reload(sys.modules["ledger_fixes"])
    ledger_fixes.patch_two_tier_ledger()
    base = ta._build_system_prompt(tool_output_tokens=1000)
    assert "Knowledge ledger protocol" not in base
    monkeypatch.setenv("LEDGER_TWO_TIER", "1")
    on = ta._build_system_prompt(tool_output_tokens=1000)
    assert on.startswith(base) and "Knowledge ledger protocol" in on


def test_idempotent(monkeypatch):
    ta = fresh_agent_cls(monkeypatch)
    ledger_fixes = importlib.reload(sys.modules["ledger_fixes"])
    assert "ledger: OK" in ledger_fixes.patch_two_tier_ledger()
    assert "SKIP" in ledger_fixes.patch_two_tier_ledger()


def test_composes_with_fix_a(monkeypatch):
    """Ledger over FIX A: disabled -> FIX A's death-safe behavior shows through."""
    ta = fresh_agent_cls(monkeypatch)
    import duck_fixes
    importlib.reload(duck_fixes)
    duck_fixes.patch_gameover_wipe()
    ledger_fixes = importlib.reload(sys.modules["ledger_fixes"])
    ledger_fixes.patch_two_tier_ledger()
    agent = make_agent(ta, {"game_over": True})
    agent._update_summarized_knowledge_from_step_summary()
    k = agent._summarized_knowledge
    # env off -> FIX A: world model KEPT in full, plan cleared, death noted
    assert k["world_model"] == KNOWLEDGE["world_model"]
    assert k["current_plan"] == ""
    assert "GAME_OVER just occurred" in k["recent_findings"]
