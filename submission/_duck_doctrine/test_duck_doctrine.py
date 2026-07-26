"""Tests for the prompt doctrine pack.

Run:  .venv/bin/python -m pytest submission/_duck_doctrine/test_duck_doctrine.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
DUCK_PATCHED = REPO / "submission/_duck_patched"
for p in (str(TAAF_INFERENCE), str(Path(__file__).parent), str(DUCK_PATCHED)):
    if p not in sys.path:
        sys.path.insert(0, p)

import duck_doctrine  # noqa: E402


@pytest.fixture()
def fresh_tool_agent(monkeypatch):
    """Reload tool_agent so each test sees an unpatched _build_system_prompt."""
    from inference.agent import tool_agent
    ta = importlib.reload(tool_agent)
    for var in ("DOCTRINE_FIELDGUIDE", "DOCTRINE_PLAYBOOK", "DOCTRINE_ACTION7"):
        monkeypatch.delenv(var, raising=False)
    return ta


def build(ta):
    return ta._build_system_prompt(tool_output_tokens=1000)


def test_all_off_is_byte_identical(fresh_tool_agent):
    ta = fresh_tool_agent
    before = build(ta)
    assert "doctrine: OK" in duck_doctrine.patch_doctrine()
    assert build(ta) == before


def test_each_toggle_appends_its_block(fresh_tool_agent, monkeypatch):
    ta = fresh_tool_agent
    base = build(ta)
    monkeypatch.setenv("DOCTRINE_FIELDGUIDE", "1")
    monkeypatch.setenv("DOCTRINE_PLAYBOOK", "1")
    duck_doctrine.patch_doctrine()
    out = build(ta)
    assert out.startswith(base)
    assert "Mechanic field guide" in out
    assert "Playbook" in out and "never the goal" in out
    assert "ACTION7 protocol" not in out


def test_toggles_read_at_build_time_not_patch_time(fresh_tool_agent, monkeypatch):
    """The A/B driver flips env between arms without re-patching."""
    ta = fresh_tool_agent
    duck_doctrine.patch_doctrine()
    base = build(ta)
    monkeypatch.setenv("DOCTRINE_PLAYBOOK", "1")
    with_playbook = build(ta)
    assert with_playbook != base and "Playbook" in with_playbook
    monkeypatch.delenv("DOCTRINE_PLAYBOOK")
    assert build(ta) == base


def test_action7_toggle_requires_mapping_fix(fresh_tool_agent, monkeypatch):
    ta = fresh_tool_agent
    monkeypatch.setenv("DOCTRINE_ACTION7", "1")
    from inference.agent import action_names
    importlib.reload(action_names)  # pristine = shipped bug (no ACTION7 mapping)
    result = duck_doctrine.patch_doctrine()
    assert "FAIL" in result and "mapping fix" in result
    # prompt unchanged on refusal
    assert "ACTION7 protocol" not in build(ta)


def test_action7_toggle_with_mapping_fix(fresh_tool_agent, monkeypatch):
    ta = fresh_tool_agent
    monkeypatch.setenv("DOCTRINE_ACTION7", "1")
    import duck_patches
    duck_patches.patch_action7()
    assert "doctrine: OK" in duck_doctrine.patch_doctrine()
    out = build(ta)
    assert "ACTION7 protocol" in out and "AT MOST ONCE" in out


def test_action7_toctou_bypass_closed(fresh_tool_agent, monkeypatch):
    """Audit 2026-07-26: enabling DOCTRINE_ACTION7 AFTER patching (when the
    patch-time guard can no longer see it) must NOT ship the guidance if the
    mapping fix is absent — the guard now also runs at build time."""
    ta = fresh_tool_agent
    from inference.agent import action_names
    importlib.reload(action_names)  # pristine = no ACTION7 mapping
    assert "doctrine: OK" in duck_doctrine.patch_doctrine()  # env unset -> OK
    monkeypatch.setenv("DOCTRINE_ACTION7", "1")              # the bypass attempt
    assert "ACTION7 protocol" not in build(ta)
    # and with the fix applied, the same env now ships the block
    import duck_patches
    duck_patches.patch_action7()
    assert "ACTION7 protocol" in build(ta)


def test_enabled_is_case_insensitive(fresh_tool_agent, monkeypatch):
    ta = fresh_tool_agent
    duck_doctrine.patch_doctrine()
    monkeypatch.setenv("DOCTRINE_PLAYBOOK", "True")  # audit: 'True' was rejected
    assert "Playbook" in build(ta)


def test_field_guide_reset_cost_is_correct():
    """RESET costs 1 scored action (scorecard.py:701-704) — the guide must not
    call it free (audit 2026-07-26)."""
    assert "restarts free" not in duck_doctrine.FIELD_GUIDE
    assert "one scored action" in duck_doctrine.FIELD_GUIDE


def test_idempotent(fresh_tool_agent, monkeypatch):
    ta = fresh_tool_agent
    monkeypatch.setenv("DOCTRINE_PLAYBOOK", "1")
    duck_doctrine.patch_doctrine()
    once = build(ta)
    assert "SKIP" in duck_doctrine.patch_doctrine()
    assert build(ta) == once
    assert once.count("Playbook (behaviors") == 1


def test_block_token_budget():
    """Keep the pack lean: every enabled block costs prompt tokens on EVERY turn."""
    approx_tokens = len(duck_doctrine.FIELD_GUIDE + duck_doctrine.PLAYBOOK
                        + duck_doctrine.ACTION7_GUIDE) / 4
    assert approx_tokens < 800, f"doctrine pack too fat: ~{approx_tokens:.0f} tokens"
