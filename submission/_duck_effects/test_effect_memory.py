"""Tests for the grounded effect memory pack.

Run:  .venv/bin/python -m pytest submission/_duck_effects/test_effect_memory.py -q

Imports the SCORED reference tree (Law 11: build against the downloaded
scored dataset, never the drifted _adopt tree).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "scratchpad/taaf_scored_ref/src/ARC3-Inference"
for p in (str(TAAF_INFERENCE), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import effect_memory  # noqa: E402


def result(**kw):
    base = {"executed": True, "level": 1, "board_changed": False, "executed_actions": []}
    base.update(kw)
    return base


def fresh_agent(ta):
    agent = object.__new__(ta.ToolAgent)
    agent._summarized_knowledge = ta._empty_world_model()
    agent._last_step_summary = None
    return agent


def reload_all(monkeypatch):
    from inference.agent import tool_agent
    ta = importlib.reload(tool_agent)
    monkeypatch.delenv("EFFECT_MEMORY", raising=False)
    em = importlib.reload(sys.modules["effect_memory"])
    return ta, em


# ---------- pure recording ----------

def test_single_action_attribution():
    mem = effect_memory.fresh_memory()
    effect_memory.record_results(mem, [result(executed_actions=["UP"], board_changed=True)])
    effect_memory.record_results(mem, [result(executed_actions=["UP"], board_changed=False)])
    assert mem["actions"]["UP"] == (2, 1)


def test_click_attribution_parses_row_col():
    mem = effect_memory.fresh_memory()
    effect_memory.record_results(
        mem, [result(executed_actions=["MOUSE(row=3, col=41)"], board_changed=True)])
    assert mem["clicks"][(3, 41)] == (1, 1)


def test_unchanged_batch_marks_all_actions():
    mem = effect_memory.fresh_memory()
    effect_memory.record_results(
        mem, [result(executed_actions=["UP", "DOWN", "UP"], board_changed=False)])
    assert mem["actions"]["UP"] == (2, 0)
    assert mem["actions"]["DOWN"] == (1, 0)


def test_changed_batch_is_not_attributed():
    mem = effect_memory.fresh_memory()
    effect_memory.record_results(
        mem, [result(executed_actions=["UP", "DOWN"], board_changed=True)])
    assert mem["actions"] == {}


def test_level_change_resets_stats():
    mem = effect_memory.fresh_memory()
    effect_memory.record_results(mem, [result(executed_actions=["UP"], board_changed=True)])
    effect_memory.record_results(
        mem, [result(level=2, executed_actions=["DOWN"], board_changed=True)])
    assert "UP" not in mem["actions"]
    assert mem["actions"]["DOWN"] == (1, 1)
    assert mem["level"] == 2


def test_unexecuted_results_ignored():
    mem = effect_memory.fresh_memory()
    effect_memory.record_results(
        mem, [result(executed=False, executed_actions=["UP"], board_changed=True)])
    assert mem["actions"] == {}


# ---------- block rendering ----------

def test_block_empty_without_conclusive_evidence():
    mem = effect_memory.fresh_memory()
    # one no-effect try is below NOOP_MIN_TRIED and no productive clicks exist
    effect_memory.record_results(mem, [result(executed_actions=["UP"])])
    assert effect_memory.build_effect_block(mem) == ""


def test_block_reports_noops_and_clicks():
    mem = effect_memory.fresh_memory()
    for _ in range(2):
        effect_memory.record_results(mem, [result(executed_actions=["UP"])])
        effect_memory.record_results(mem, [result(executed_actions=["MOUSE(row=9, col=9)"])])
    effect_memory.record_results(
        mem, [result(executed_actions=["MOUSE(row=1, col=2)"], board_changed=True)])
    block = effect_memory.build_effect_block(mem)
    assert "UP (0/2)" in block
    assert "(1,2): 1/1" in block
    assert "(9,9) 0/2" in block


def test_block_caps_click_lists():
    mem = effect_memory.fresh_memory()
    for i in range(effect_memory.MAX_REPORT_ITEMS + 5):
        effect_memory.record_results(
            mem, [result(executed_actions=[f"MOUSE(row={i}, col=0)"], board_changed=True)])
    block = effect_memory.build_effect_block(mem)
    assert block.count("): 1/1") == effect_memory.MAX_REPORT_ITEMS


# ---------- patched class, env gating ----------

def test_disabled_is_verbatim(monkeypatch):
    ta, em = reload_all(monkeypatch)
    baseline_agent = fresh_agent(ta)
    base_prompt = baseline_agent._build_user_prompt(3, valid_actions=["UP", "DOWN"])
    base_system = ta._build_system_prompt(tool_output_tokens=1000)

    assert "effects: OK" in em.patch_effect_memory()
    agent = fresh_agent(ta)
    agent._summarize_step_sequence([result(executed_actions=["UP"], board_changed=True)])
    assert agent._build_user_prompt(3, valid_actions=["UP", "DOWN"]) == base_prompt
    assert ta._build_system_prompt(tool_output_tokens=1000) == base_system
    # disabled -> nothing recorded either
    assert getattr(agent, "_effect_memory", {"actions": {}})["actions"] == {}


def test_enabled_records_and_surfaces(monkeypatch):
    ta, em = reload_all(monkeypatch)
    em.patch_effect_memory()
    monkeypatch.setenv("EFFECT_MEMORY", "1")
    agent = fresh_agent(ta)
    for _ in range(2):
        agent._summarize_step_sequence([result(executed_actions=["UP"])])
    agent._summarize_step_sequence(
        [result(executed_actions=["MOUSE(row=5, col=6)"], board_changed=True)])
    prompt = agent._build_user_prompt(3, valid_actions=["UP", "MOUSE"])
    assert "Grounded effect memory" in prompt
    assert "UP (0/2)" in prompt
    assert "(5,6): 1/1" in prompt
    assert "Grounded effect memory" in ta._build_system_prompt(tool_output_tokens=1000)


def test_summary_return_value_unchanged(monkeypatch):
    ta, em = reload_all(monkeypatch)
    plain_agent = fresh_agent(ta)
    results = [result(executed_actions=["UP"], board_changed=True, action_num=7)]
    expected = plain_agent._summarize_step_sequence(results)
    em.patch_effect_memory()
    monkeypatch.setenv("EFFECT_MEMORY", "1")
    agent = fresh_agent(ta)
    assert agent._summarize_step_sequence(results) == expected


def test_idempotent(monkeypatch):
    ta, em = reload_all(monkeypatch)
    assert "effects: OK" in em.patch_effect_memory()
    assert "SKIP" in em.patch_effect_memory()


def test_composes_with_doctrine_and_ledger(monkeypatch):
    ta, em = reload_all(monkeypatch)
    sys.path.insert(0, str(REPO / "submission/_duck_doctrine"))
    sys.path.insert(0, str(REPO / "submission/_duck_fixes"))
    import duck_doctrine
    import ledger_fixes
    importlib.reload(duck_doctrine).patch_doctrine()
    importlib.reload(ledger_fixes).patch_two_tier_ledger()
    em.patch_effect_memory()
    for name in ("DOCTRINE_FIELDGUIDE", "LEDGER_TWO_TIER", "EFFECT_MEMORY"):
        monkeypatch.setenv(name, "1")
    system = ta._build_system_prompt(tool_output_tokens=1000)
    assert "Mechanic field guide" in system
    assert "Knowledge ledger protocol" in system
    assert "Grounded effect memory" in system
