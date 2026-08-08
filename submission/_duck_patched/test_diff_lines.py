"""Tests for patch 16 (action-locked structured change lines / Transient Map).

Rank 4a of the 2026-08-07 human-play idea sweep: the model must never have to
spot frame differences itself — a deterministic differ narrates every executed
action's HUD-masked pre/post diff as compact lines (translation / recolor /
no-change), banked per action and prepended ABOVE the grids in the analyzer
user prompt.

Layers: pure-differ unit cases (translation, recolor, no-change, HUD-only,
overflow, missing-mask caveat), capture bookkeeping, injection through a REAL
ToolAgent, env gating (OFF by default = zero behavior change), and scored-bundle
subprocess validation.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_diff_lines.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref/src/ARC3-Inference"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import (  # noqa: E402
    _DIFF_TLS,
    _diff_prompt_block,
    _diff_record_action,
    diff_change_lines,
)

HEADER_MARKER = "STRUCTURED CHANGE REPORT"


def _board(fill: int = 0, size: int = 16) -> list[list[int]]:
    return [[fill] * size for _ in range(size)]


def _paint(board: list[list[int]], cells, color: int) -> list[list[int]]:
    out = [list(row) for row in board]
    for y, x in cells:
        out[y][x] = color
    return out


def _block(y0: int, x0: int, h: int, w: int):
    return [(y, x) for y in range(y0, y0 + h) for x in range(x0, x0 + w)]


# ---------------------------------------------------------------------------------
# pure differ
# ---------------------------------------------------------------------------------


def test_no_change_board_identical():
    pre = _board()
    assert diff_change_lines(pre, pre, [[0, 0]]) == ["NO CHANGE (board identical)"]


def test_hud_only_tick_is_named_not_narrated():
    pre = _board()
    post = _paint(pre, [(0, 3)], 5)  # only the masked HUD cell ticked
    lines = diff_change_lines(pre, post, [[0, x] for x in range(16)])
    assert lines == ["NO CHANGE (HUD/step-counter tick only)"]


def test_translation_is_detected_with_shift_and_destination():
    pre = _paint(_board(), _block(5, 3, 4, 3), 9)
    post = _paint(_board(), _block(6, 3, 4, 3), 9)  # moved +1 row
    lines = diff_change_lines(pre, post, [[15, x] for x in range(16)])
    assert len(lines) == 1
    line = lines[0]
    assert "CHANGE #1:" in line and "MOVED" in line
    assert "4x3" in line
    assert "dr=+1" in line and "dc=+0" in line
    assert "rows 6-9" in line and "cols 3-5" in line
    # color letter code, not a numeric id ('b' = ARC_COLOR_CHARS[9])
    assert "(b)" in line


def test_long_jump_reported_as_two_changes_not_fake_motion():
    """A jump beyond the search radius splits into two honest change lines."""
    pre = _paint(_board(0, 40), _block(2, 2, 2, 2), 9)
    post = _paint(_board(0, 40), _block(30, 30, 2, 2), 9)
    lines = diff_change_lines(pre, post, [[39, x] for x in range(40)])
    body = "\n".join(lines)
    assert "CHANGE #1:" in body and "CHANGE #2:" in body
    assert "MOVED" not in body  # 28-cell jump > radius: never narrated as motion


def test_recolor_case_uses_board_letter_codes():
    pre = _board()
    post = _paint(pre, _block(10, 5, 2, 3), 15)
    lines = diff_change_lines(pre, post, [[0, 0]])
    assert len(lines) == 1
    assert lines[0] == "CHANGE #1: 6 cells at rows 10-11 cols 5-7 RECOLORED W->p"


def test_pure_recolor_never_reported_as_motion():
    """In-place recolor of a block must be RECOLORED, not MOVED."""
    pre = _paint(_board(), _block(4, 4, 3, 3), 7)
    post = _paint(_board(), _block(4, 4, 3, 3), 11)
    lines = diff_change_lines(pre, post, [[0, 0]])
    assert "RECOLORED" in lines[0] and "MOVED" not in lines[0]


def test_overflow_is_capped_and_summarized_honestly():
    pre = _board(0, 32)
    cells = [(2 * i, 2 * j) for i in range(8) for j in range(8)]  # 64 isolated cells
    post = _paint(pre, cells, 3)
    lines = diff_change_lines(pre, post, [[31, 0]], max_lines=5)
    assert sum(1 for line in lines if line.startswith("CHANGE #")) == 5
    assert lines[5] == "+59 more small changes (59 cells total)"


def test_missing_mask_adds_caveat_line():
    pre = _board()
    post = _paint(pre, _block(3, 3, 2, 2), 6)
    lines = diff_change_lines(pre, post, [])
    assert any("HUD not yet identified" in line for line in lines)
    # ... and a present mask suppresses the caveat
    lines_masked = diff_change_lines(pre, post, [[0, 0]])
    assert not any("HUD not yet identified" in line for line in lines_masked)


def test_reshape_reported_as_repaint():
    lines = diff_change_lines(_board(0, 8), _board(0, 16), [])
    assert "BOARD RESHAPED" in lines[0]


def test_line_cap_env_clamped(monkeypatch):
    monkeypatch.setenv("TAAF_DIFF_MAX_LINES", "50")
    pre = _board(0, 32)
    cells = [(2 * i, 2 * j) for i in range(8) for j in range(8)]
    post = _paint(pre, cells, 3)
    lines = diff_change_lines(pre, post, [[31, 0]])
    assert sum(1 for line in lines if line.startswith("CHANGE #")) == 10  # hard clamp


# ---------------------------------------------------------------------------------
# capture bookkeeping (host side)
# ---------------------------------------------------------------------------------


def _payload(**overrides):
    base = {
        "executed": True,
        "action_display": "ACTION3",
        "action_num": 42,
        "board_changed": True,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
    }
    base.update(overrides)
    return base


def _action(name: str = "ACTION3"):
    return SimpleNamespace(id=SimpleNamespace(name=name))


@pytest.fixture(autouse=True)
def _clean_tls():
    _DIFF_TLS.entries = []
    yield
    _DIFF_TLS.entries = []


def test_record_action_banks_lines_and_caps_history(monkeypatch):
    monkeypatch.setenv("TAAF_DIFF_MAX_ACTIONS", "3")
    pre = _paint(_board(), _block(5, 3, 4, 3), 9)
    post = _paint(_board(), _block(6, 3, 4, 3), 9)
    for step in range(5):
        _diff_record_action(_action(), _payload(action_num=step), pre, post)
    entries = _DIFF_TLS.entries
    assert len(entries) == 3  # capped, oldest dropped
    assert [e["action_num"] for e in entries] == [2, 3, 4]
    assert entries[-1]["action"] == "ACTION3"
    assert any("MOVED" in line for line in entries[-1]["lines"])


def test_record_action_skips_unexecuted_and_marks_boundaries():
    pre, post = _board(), _paint(_board(), [(1, 1)], 4)
    _diff_record_action(_action(), _payload(executed=False), pre, post)
    assert _DIFF_TLS.entries == []
    _diff_record_action(_action("RESET"), _payload(action_display="RESET"), pre, post)
    assert _DIFF_TLS.entries[-1]["lines"] == ["RESET — scene repainted, per-object diff skipped"]
    _diff_record_action(_action(), _payload(level_completed=True), pre, post)
    assert "LEVEL COMPLETED" in _DIFF_TLS.entries[-1]["lines"][0]


def test_prompt_block_renders_consumes_and_is_empty_when_idle():
    assert _diff_prompt_block() == ""
    pre = _paint(_board(), _block(10, 5, 2, 3), 0)
    post = _paint(pre, _block(10, 5, 2, 3), 15)
    _diff_record_action(_action(), _payload(), pre, post)
    block = _diff_prompt_block()
    assert block.startswith(HEADER_MARKER)
    assert "After ACTION3 (action #42):" in block
    assert "RECOLORED W->p" in block
    # consumed: a second build must not repeat stale lines
    assert _diff_prompt_block() == ""


def test_prompt_block_caps_total_lines(monkeypatch):
    monkeypatch.setenv("TAAF_DIFF_MAX_ACTIONS", "20")
    monkeypatch.setenv("TAAF_DIFF_MAX_BLOCK", "6")
    pre, post = _board(), _paint(_board(), [(1, 1)], 4)
    for step in range(12):
        _diff_record_action(_action(), _payload(action_num=step), pre, post)
    block = _diff_prompt_block()
    body = block.splitlines()
    assert body[1] == "(earlier actions omitted — block capped)"
    assert len(body) == 8  # header + omission note + 6 capped lines
    assert "(action #11)" in block  # the newest action survives the cap
    assert "(action #0)" not in block  # the oldest is what gets dropped


# ---------------------------------------------------------------------------------
# injection through a REAL ToolAgent + env gate
# ---------------------------------------------------------------------------------


def _apply() -> str:
    result = duck_patches.patch_diff_lines()
    assert "OK" in result or "SKIP" in result, result
    return result


def _agent():
    from inference.agent.tool_agent import ToolAgent

    _apply()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/diff-lines-test")
    return agent


def _frame(grid):
    from inference.agent.runtime_state import Frame

    return Frame(grid=tuple(tuple(row) for row in grid), step=1, level=1)


def _build(agent):
    return agent._build_user_prompt(
        1, valid_actions=["ACTION1"], current_frame=_frame([[1, 2], [3, 4]])
    )


def test_patch_applies_and_is_idempotent():
    first = _apply()
    assert first.startswith("patch16 diff-lines: OK") or "SKIP" in first
    assert duck_patches.patch_diff_lines() == "patch16 diff-lines: SKIP (already applied)"


def test_disabled_by_default_no_injection_even_with_banked_lines(monkeypatch):
    monkeypatch.delenv("TAAF_DIFF_LINES", raising=False)
    agent = _agent()
    _DIFF_TLS.entries = [{"action": "ACTION1", "action_num": 7, "lines": ["CHANGE #1: x"]}]
    prompt = _build(agent)
    assert HEADER_MARKER not in prompt
    assert _DIFF_TLS.entries  # untouched: the gate short-circuits before consumption


def test_enabled_injection_lands_above_everything_and_consumes(monkeypatch):
    monkeypatch.setenv("TAAF_DIFF_LINES", "1")
    agent = _agent()
    _DIFF_TLS.entries = [
        {
            "action": "ACTION6(col=3, row=4)",
            "action_num": 9,
            "lines": ["CHANGE #1: 6 cells at rows 10-11 cols 5-7 RECOLORED W->p"],
        }
    ]
    prompt = _build(agent)
    assert prompt.startswith(HEADER_MARKER)  # above ALL prompt text (and the grid image
    # appended after the text by _build_user_message)
    assert "RECOLORED W->p" in prompt
    assert prompt.index(HEADER_MARKER) < prompt.index("Current state:")
    assert "Valid actions right now" in prompt  # base prompt intact below the report
    assert _DIFF_TLS.entries == []  # consumed on injection
    # next turn with nothing banked: clean prompt again
    assert HEADER_MARKER not in _build(agent)


def test_enabled_capture_gate_delegates_when_off(monkeypatch):
    """The _execute_action wrapper must be a pure pass-through when disabled."""
    monkeypatch.delenv("TAAF_DIFF_LINES", raising=False)
    from inference.framework import solver

    _apply()
    wrapper = solver._HarnessGameSession._execute_action
    assert getattr(wrapper, "_diff_lines_patched", False)
    # Real-session capture is exercised in the scored-bundle subprocess below.
    # Here: disabled means the env gate returns before any grid work, so banked
    # state must stay empty across a full prompt-build cycle.
    _DIFF_TLS.entries = []
    agent = _agent()
    _build(agent)
    assert _DIFF_TLS.entries == []


def test_patch_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.agent import tool_agent
    from inference.framework import solver

    monkeypatch.setattr(solver, "_grid_from_state", None)
    msg = duck_patches.patch_diff_lines()
    assert msg.startswith("patch16 diff-lines: FAIL"), msg
    monkeypatch.undo()

    monkeypatch.setattr(tool_agent, "ToolAgent", None)
    msg = duck_patches.patch_diff_lines()
    assert msg.startswith("patch16 diff-lines: FAIL"), msg


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE validation (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------


def test_scored_bundle_has_every_symbol_patch16_touches():
    solver_src = (SCORED_REF / "inference/framework/solver.py").read_text()
    for symbol in ("def _grid_from_state", "class _HarnessGameSession", "def _execute_action"):
        assert symbol in solver_src, f"scored bundle lost {symbol!r} — re-validate patch16"
    agent_src = (SCORED_REF / "inference/agent/tool_agent.py").read_text()
    assert "def _build_user_prompt" in agent_src
    grid_utils = (SCORED_REF / "inference/utils/grid_utils.py").read_text()
    assert "ARC_COLOR_CHARS" in grid_utils, "letter-code palette moved — color names drift"


@pytest.mark.skipif(not SCORED_REF.is_dir(), reason="scored bundle bytes absent")
def test_patch16_applies_on_scored_bundle_bytes():
    """Apply patch 16 on the scored bytes in a clean subprocess: enabled, a banked
    change report must land above the prompt; disabled, the prompt is untouched."""
    script = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(SCORED_REF)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

r = duck_patches.patch_diff_lines()
assert r.startswith("patch16 diff-lines: OK"), r

from inference.agent.tool_agent import ToolAgent
from inference.agent.runtime_state import Frame
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/diff-lines-scored")
frame = Frame(grid=((1, 2), (3, 4)), step=1, level=1)

def build():
    return agent._build_user_prompt(1, valid_actions=["ACTION1"], current_frame=frame)

# disabled (default): banked lines must NOT leak into the scored-bundle prompt
duck_patches._DIFF_TLS.entries = [
    {{"action": "ACTION2", "action_num": 5, "lines": ["CHANGE #1: test-line"]}}
]
assert "STRUCTURED CHANGE REPORT" not in build(), "dormant patch leaked into prompt"

# enabled: the report must land at the very top and be consumed
os.environ["TAAF_DIFF_LINES"] = "1"
pre = [[0] * 16 for _ in range(16)]
post = [list(row) for row in pre]
for y in range(5, 9):
    for x in range(3, 6):
        pre[y][x] = 9
        post[y + 1][x] = 9
from types import SimpleNamespace
duck_patches._DIFF_TLS.entries = []
duck_patches._diff_record_action(
    SimpleNamespace(id=SimpleNamespace(name="ACTION3")),
    {{"executed": True, "action_display": "ACTION3", "action_num": 42}},
    pre,
    post,
)
prompt = build()
assert prompt.startswith("STRUCTURED CHANGE REPORT"), prompt[:200]
assert "MOVED" in prompt and "dr=+1" in prompt, prompt[:400]
assert duck_patches._DIFF_TLS.entries == [], "report not consumed"
assert "STRUCTURED CHANGE REPORT" not in build(), "stale report repeated"
print("SCORED-DIFF-LINES-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-DIFF-LINES-OK" in proc.stdout
