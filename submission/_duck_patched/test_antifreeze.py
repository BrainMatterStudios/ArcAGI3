"""Tests for patch 14 (world-model anti-freeze guard).

The tr87 pathology: the copy-forward world-model instruction let the model repeat
identical world-model text + plan for 10+ turns while the board stayed stuck
(bsm k021 — 6 actions in a whole run). The guard appends a one-line breaker to
the user prompt once the carried world model AND the (HUD-masked) board are both
unchanged for N consecutive prompt builds.

Layers: unit trigger/no-trigger on a REAL ToolAgent, env gating, HUD interplay,
clean decline on hostile bundles, and scored-bundle subprocess validation.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_antifreeze.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref/src/ARC3-Inference"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import _HUD_TLS, ANTIFREEZE_DIAGNOSTICS  # noqa: E402

BREAKER_MARKER = "ALERT: your world model text has been IDENTICAL"


def _apply() -> None:
    result = duck_patches.patch_antifreeze()
    assert "OK" in result or "SKIP" in result, result


def _agent(monkeypatch=None):
    from inference.agent.tool_agent import ToolAgent

    _apply()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/antifreeze-test")
    return agent


def _frame(grid):
    from inference.agent.runtime_state import Frame

    return Frame(grid=tuple(tuple(row) for row in grid), step=1, level=1)


_GRID_A = [[1, 2], [3, 4]]
_GRID_B = [[1, 2], [3, 9]]


def _set_model(agent, plan: str) -> None:
    agent._summarized_knowledge["world_model"] = "two boxes and a selector"
    agent._summarized_knowledge["current_plan"] = plan


def _build(agent, grid=_GRID_A):
    return agent._build_user_prompt(
        1, valid_actions=["ACTION1"], current_frame=_frame(grid)
    )


# ---------------------------------------------------------------------------------
# trigger / no-trigger
# ---------------------------------------------------------------------------------


def test_patch_applies_and_is_idempotent():
    first = duck_patches.patch_antifreeze()
    assert first in ("patch14 antifreeze: OK", "patch14 antifreeze: SKIP (already applied)")
    assert duck_patches.patch_antifreeze() == "patch14 antifreeze: SKIP (already applied)"


def test_breaker_fires_after_n_frozen_turns(monkeypatch):
    monkeypatch.delenv("TAAF_ANTIFREEZE", raising=False)
    monkeypatch.delenv("TAAF_ANTIFREEZE_TURNS", raising=False)
    agent = _agent()
    _set_model(agent, "probe RIGHT")
    before = ANTIFREEZE_DIAGNOSTICS["triggers"]

    prompts = [_build(agent) for _ in range(4)]
    # builds 1-3: baseline + streak 1, 2 — no breaker yet
    for prompt in prompts[:3]:
        assert BREAKER_MARKER not in prompt
    # build 4: streak 3 == default N — breaker
    assert BREAKER_MARKER in prompts[3]
    assert "IDENTICAL for 3 consecutive turns" in prompts[3]
    assert ANTIFREEZE_DIAGNOSTICS["triggers"] == before + 1
    # still frozen -> keeps injecting (and the streak keeps counting up)
    prompt5 = _build(agent)
    assert BREAKER_MARKER in prompt5 and "for 4 consecutive turns" in prompt5


def test_world_model_change_resets_the_streak():
    agent = _agent()
    _set_model(agent, "probe RIGHT")
    for _ in range(3):
        _build(agent)
    _set_model(agent, "probe LEFT — new hypothesis")  # model revised
    assert BREAKER_MARKER not in _build(agent)
    # ... and the clock restarts from the revision, not from zero history
    _build(agent)
    _build(agent)
    assert BREAKER_MARKER in _build(agent)


def test_board_change_resets_the_streak():
    agent = _agent()
    _set_model(agent, "probe RIGHT")
    for _ in range(3):
        _build(agent, _GRID_A)
    assert BREAKER_MARKER not in _build(agent, _GRID_B)  # board moved


def test_empty_world_model_never_triggers():
    """A fresh game/level (knowledge cleared) is not a freeze."""
    agent = _agent()
    for _ in range(6):
        prompt = _build(agent)
        assert BREAKER_MARKER not in prompt


def test_hud_tick_does_not_hide_a_freeze():
    """A budget bar ticking in a masked HUD cell must NOT count as board change."""
    agent = _agent()
    _set_model(agent, "probe RIGHT")
    _HUD_TLS.mask_cells = [[0, 0]]
    try:
        grids = [[[t, 2], [3, 4]] for t in range(5)]  # cell (0,0) ticks each turn
        prompts = [_build(agent, g) for g in grids]
        assert BREAKER_MARKER in prompts[3], (
            "HUD-only ticking must not reset the freeze streak"
        )
    finally:
        _HUD_TLS.mask_cells = []
    # Without the mask the same ticking reads as real change: no trigger.
    agent2 = _agent()
    _set_model(agent2, "probe RIGHT")
    for t in range(5):
        prompt = _build(agent2, [[t, 2], [3, 4]])
        assert BREAKER_MARKER not in prompt


def test_env_gating_and_custom_threshold(monkeypatch):
    monkeypatch.setenv("TAAF_ANTIFREEZE", "0")
    agent = _agent()
    _set_model(agent, "probe RIGHT")
    for _ in range(6):
        assert BREAKER_MARKER not in _build(agent)

    monkeypatch.setenv("TAAF_ANTIFREEZE", "1")
    monkeypatch.setenv("TAAF_ANTIFREEZE_TURNS", "2")
    agent2 = _agent()
    _set_model(agent2, "probe RIGHT")
    assert BREAKER_MARKER not in _build(agent2)
    assert BREAKER_MARKER not in _build(agent2)
    assert BREAKER_MARKER in _build(agent2)  # streak 2 >= N=2


def test_guard_is_exception_safe(monkeypatch):
    """A hostile frame object must not break prompt building."""
    agent = _agent()
    _set_model(agent, "probe RIGHT")

    class _EvilGrid:
        def __iter__(self):
            raise RuntimeError("boom")

    class _EvilFrame:
        grid = _EvilGrid()
        step = 1
        level = 1

    prompt = agent._build_user_prompt(
        1, valid_actions=["ACTION1"], current_frame=_EvilFrame()
    )
    assert "Valid actions right now" in prompt  # base prompt intact


def test_patch_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.agent import tool_agent

    monkeypatch.delattr(tool_agent.ToolAgent, "_summarized_knowledge_lines")
    msg = duck_patches.patch_antifreeze()
    assert msg.startswith("patch14 antifreeze: FAIL"), msg
    monkeypatch.undo()

    monkeypatch.setattr(tool_agent, "ToolAgent", None)
    msg = duck_patches.patch_antifreeze()
    assert msg.startswith("patch14 antifreeze: FAIL"), msg


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE validation (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------


def test_scored_bundle_has_every_symbol_patch14_touches():
    source = (SCORED_REF / "inference/agent/tool_agent.py").read_text()
    for symbol in (
        "def _build_user_prompt",
        "def _summarized_knowledge_lines",
        "_summarized_knowledge",
    ):
        assert symbol in source, f"scored bundle lost {symbol!r} — re-validate patch14"
    runtime_state = (SCORED_REF / "inference/agent/runtime_state.py").read_text()
    assert "class Frame" in runtime_state and "grid" in runtime_state


@pytest.mark.skipif(not SCORED_REF.is_dir(), reason="scored bundle bytes absent")
def test_patch14_applies_on_scored_bundle_bytes():
    """Apply patch 14 on the scored bytes in a clean subprocess; a frozen agent
    there must receive the breaker, a revising agent must not."""
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(SCORED_REF)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

r = duck_patches.patch_antifreeze()
assert r == "patch14 antifreeze: OK", r

from inference.agent.tool_agent import ToolAgent
from inference.agent.runtime_state import Frame
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/antifreeze-scored")
agent._summarized_knowledge["world_model"] = "frozen model"
agent._summarized_knowledge["current_plan"] = "probe RIGHT"
frame = Frame(grid=((1, 2), (3, 4)), step=1, level=1)
prompts = [
    agent._build_user_prompt(1, valid_actions=["ACTION1"], current_frame=frame)
    for _ in range(4)
]
assert all({BREAKER_MARKER!r} not in p for p in prompts[:3]), "fired early"
assert {BREAKER_MARKER!r} in prompts[3], "breaker missing on the scored bytes"
agent._summarized_knowledge["current_plan"] = "new plan"
p5 = agent._build_user_prompt(1, valid_actions=["ACTION1"], current_frame=frame)
assert {BREAKER_MARKER!r} not in p5, "revision must clear the breaker"
print("SCORED-ANTIFREEZE-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-ANTIFREEZE-OK" in proc.stdout
