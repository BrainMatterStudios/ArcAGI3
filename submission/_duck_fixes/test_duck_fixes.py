"""Tests for the Layer-2 duck fixes.
Run: .venv/bin/python -m pytest submission/_duck_fixes/test_duck_fixes.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"))
sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("MULTIMODAL_CONTEXT", "current_grid")
os.environ.setdefault("MULTIMODAL_UPSCALE", "4")

import duck_fixes  # noqa: E402


# ---- FIX B: image-aware estimator ----

def _image_part():
    import numpy as np
    from inference.agent.runtime_state import Frame
    from inference.agent.vision_context import frame_to_png_data_url

    d = np.load(str(REPO / "scratchpad/killexp_data/tu93.npz"))
    grid = tuple(tuple(int(v) for v in row) for row in d["base"])
    url = frame_to_png_data_url(Frame(grid=grid, step=0, level=1))
    return {"type": "image_url", "image_url": {"url": url}}


def test_estimator_before_patch_overcounts():
    from inference.agent import tool_agent as ta
    if getattr(ta._estimate_tokens, "_image_aware", False):
        import pytest
        pytest.skip("patch already applied by earlier test run")
    est = ta._estimate_tokens(_image_part())
    assert est > 300, "shipped estimator should overcount the image (len/3 of base64)"


def test_estimator_after_patch_counts_vision_tokens():
    from inference.agent import tool_agent as ta

    status = duck_fixes.patch_estimator()
    assert "OK" in status or "SKIP" in status

    part = _image_part()
    est = ta._estimate_tokens(part)
    # 256x256 at 28px patches -> ceil(256/28)^2 + 2 = 102, plus the tiny json wrapper
    assert 90 <= est <= 140, f"expected ~102+wrapper, got {est}"


def test_estimator_text_unchanged():
    from inference.agent import tool_agent as ta

    duck_fixes.patch_estimator()
    text = {"role": "user", "content": "hello " * 100}
    # text-only estimate must match the original len/3 formula
    import json as _json
    expected = max(1, (len(_json.dumps(text, ensure_ascii=True, sort_keys=True, default=str)) + 2) // 3)
    assert ta._estimate_tokens(text) == expected


def test_estimator_mixed_message():
    from inference.agent import tool_agent as ta

    duck_fixes.patch_estimator()
    msg = {"role": "user", "content": [
        {"type": "text", "text": "what changed?"},
        _image_part(),
    ]}
    est = ta._estimate_tokens(msg)
    assert est < 250, f"mixed message should be ~110-160, got {est}"


def test_estimator_idempotent():
    s1 = duck_fixes.patch_estimator()
    s2 = duck_fixes.patch_estimator()
    assert "SKIP" in s2


# ---- FIX A: game_over wipe ----

class _AgentStub:
    def __init__(self, summary):
        self._last_step_summary = summary
        self._summarized_knowledge = {
            "world_model": "walls kill; key opens door",
            "goal_model": "reach the exit",
            "action_model": "ACTION1..4 move",
            "recent_findings": "blue tile teleports",
            "open_questions": "what does ACTION5 do?",
            "current_plan": "go left, take key",
            "cross_level_notes": "levels reuse the key mechanic",
        }


def _apply_wipe(stub):
    from inference.agent import tool_agent as ta
    ta.ToolAgent._update_summarized_knowledge_from_step_summary(stub)


def test_gameover_keeps_world_model():
    duck_fixes.patch_gameover_wipe()
    stub = _AgentStub({"game_over": True})
    _apply_wipe(stub)
    k = stub._summarized_knowledge
    assert k["world_model"] == "walls kill; key opens door", "world model must survive death"
    assert k["goal_model"] and k["action_model"]
    assert k["current_plan"] == "", "the plan that led to death must be dropped"
    assert "fatal" in k["recent_findings"], "death must be recorded as evidence"


def test_level_transition_still_wipes():
    duck_fixes.patch_gameover_wipe()
    stub = _AgentStub({"level_transition": True})
    _apply_wipe(stub)
    k = stub._summarized_knowledge
    assert k["world_model"] == "" and k["current_plan"] == "", "deliberate per-level fresh grounding preserved"
    assert k["cross_level_notes"] == "levels reuse the key mechanic", "carry-over channel untouched"


def test_run_complete_still_wipes():
    duck_fixes.patch_gameover_wipe()
    stub = _AgentStub({"run_complete": True})
    _apply_wipe(stub)
    assert stub._summarized_knowledge["world_model"] == ""


def test_no_summary_noop():
    duck_fixes.patch_gameover_wipe()
    stub = _AgentStub(None)
    _apply_wipe(stub)
    assert stub._summarized_knowledge["world_model"] == "walls kill; key opens door"


def test_death_note_not_duplicated():
    duck_fixes.patch_gameover_wipe()
    stub = _AgentStub({"game_over": True})
    _apply_wipe(stub)
    once = stub._summarized_knowledge["recent_findings"]
    stub._last_step_summary = {"game_over": True}
    _apply_wipe(stub)
    assert stub._summarized_knowledge["recent_findings"] == once


def test_apply_all_clean():
    lines = duck_fixes.apply_all(verbose=False)
    assert len(lines) == 2 and not any("FAIL" in l for l in lines), lines
