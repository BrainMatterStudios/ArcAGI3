"""Tests for patch 23 — the token-estimator correction.

The lever rests on a measurement (1.44x overcount over 3,556 real requests), so
these tests check the two things that could silently break it: that the flag
actually gates behaviour, and that the new estimate never UNDERSHOOTS the real
token count (undershooting would overrun the served window and get requests
rejected mid-game, which is far worse than the overcount it replaces).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# MUST match the other test modules: pointing at a different ARC3-Inference tree
# imports a second, conflicting `inference` package and silently breaks whichever
# tests import after this one (it broke test_hud_mask when first written).
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches as dp  # noqa: E402


@pytest.fixture()
def tool_agent(monkeypatch):
    ta = importlib.import_module("inference.agent.tool_agent")
    original = ta._estimate_tokens
    yield ta
    ta._estimate_tokens = original


def _payload(n: int = 400) -> dict:
    return {"messages": [{"role": "user", "content": "x" * n}]}


def test_disabled_is_byte_identical_to_the_original(tool_agent, monkeypatch):
    monkeypatch.delenv("TAAF_TOKEN_EST", raising=False)
    before = tool_agent._estimate_tokens(_payload())
    # OK on a fresh module, SKIP if an earlier test in the session already applied
    # it (apply_all runs in test_duck_patches). Either way it must not FAIL, and
    # the behavioural property below is the real assertion.
    status = dp.patch_token_estimator()
    assert "FAIL" not in status, status
    after = tool_agent._estimate_tokens(_payload())
    assert after == before, "a dormant patch must not change any number"


def test_enabled_lowers_the_estimate_by_about_a_quarter(tool_agent, monkeypatch):
    monkeypatch.delenv("TAAF_TOKEN_EST", raising=False)
    dp.patch_token_estimator()
    old = tool_agent._estimate_tokens(_payload(4000))
    monkeypatch.setenv("TAAF_TOKEN_EST", "1")
    new = tool_agent._estimate_tokens(_payload(4000))
    assert new < old
    # chars/3 -> chars/4 is a 25% reduction in the estimate.
    assert 0.72 < new / old < 0.78


def test_never_undershoots_real_tokens_at_the_measured_ratio(tool_agent, monkeypatch):
    """The safety property. Real content measured ~4.3 chars/token; we divide by
    4, so the estimate must stay >= the real count."""
    dp.patch_token_estimator()
    monkeypatch.setenv("TAAF_TOKEN_EST", "1")
    payload = _payload(20000)
    rendered = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    est = tool_agent._estimate_tokens(payload)
    real_at_measured_ratio = len(rendered) / 4.3
    assert est >= real_at_measured_ratio, "estimate must err HIGH, never low"


def test_gate_is_read_at_call_time_not_patch_time(tool_agent, monkeypatch):
    """A rebuild must not be needed to disarm it."""
    monkeypatch.delenv("TAAF_TOKEN_EST", raising=False)
    dp.patch_token_estimator()
    off1 = tool_agent._estimate_tokens(_payload(3000))
    monkeypatch.setenv("TAAF_TOKEN_EST", "1")
    on = tool_agent._estimate_tokens(_payload(3000))
    monkeypatch.setenv("TAAF_TOKEN_EST", "0")
    off2 = tool_agent._estimate_tokens(_payload(3000))
    assert off1 == off2 and on < off1


def test_divisor_is_overridable_for_a_sweep(tool_agent, monkeypatch):
    dp.patch_token_estimator()
    monkeypatch.setenv("TAAF_TOKEN_EST", "1")
    monkeypatch.setenv("TAAF_TOKEN_EST_DIV", "5")
    five = tool_agent._estimate_tokens(_payload(5000))
    monkeypatch.setenv("TAAF_TOKEN_EST_DIV", "4")
    four = tool_agent._estimate_tokens(_payload(5000))
    assert five < four
    monkeypatch.setenv("TAAF_TOKEN_EST_DIV", "garbage")
    assert tool_agent._estimate_tokens(_payload(5000)) == four, "bad value falls back to 4"


def test_reapplying_is_a_skip(tool_agent):
    dp.patch_token_estimator()
    assert "SKIP" in dp.patch_token_estimator()


def test_diagnostics_record_both_estimates(tool_agent, monkeypatch):
    dp.TOKEN_EST_DIAGNOSTICS.update({"calls": 0, "chars": 0, "est_new": 0, "est_old": 0})
    dp.patch_token_estimator()
    monkeypatch.setenv("TAAF_TOKEN_EST", "1")
    tool_agent._estimate_tokens(_payload(9000))
    d = dp.TOKEN_EST_DIAGNOSTICS
    assert d["calls"] == 1 and d["chars"] > 0
    # the recorded ratio is what a wave will be judged on
    assert d["est_old"] > d["est_new"] > 0
