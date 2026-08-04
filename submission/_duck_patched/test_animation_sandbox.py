"""Tests for patch 15 (animation frames as the `last_animation` sandbox global).

Covers the four pre-registered claims:
  1. animation is CAPTURED on a game that actually animates (lf52 — the
     2026-08-04 review's example: animates on 30/30 actions) through the real
     harness session loop;
  2. `last_animation` is queryable inside a REAL sandbox subprocess (FrameView
     API: .ascii / .shape / ._grid), including refresh after `action(...)`;
  3. ZERO tokens added when unqueried: exactly one system-prompt line, and a
     sandbox run that never touches the global emits no animation content;
  4. scored-bundle discipline (patch-9 law): the patch applies on the scored
     bundle's bootstrap shape (verified against scratchpad/taaf_scored_ref
     bytes) and declines cleanly on a bootstrap that cannot support it.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_animation_sandbox.py -v
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
ENV_DIR = REPO / "environment_files"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import _ANIM_TLS  # noqa: E402

GAME = "lf52"  # animates on 30/30 actions per the 2026-08-04 review


def _apply() -> None:
    status = duck_patches.patch_animation_sandbox()
    assert "OK" in status or "already applied" in status, status


def _result(step_executed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        step_executed=step_executed,
        retryable_failure=False,
        yielded_control=False,
        reasoning="",
    )


class _ProbeAnalyzer:
    """Executes one valid action per turn and snapshots the host-side payload
    right after the action lands (same thread as the play loop, exactly the
    thread `run_sandboxed_python` would read the TLS from)."""

    generated_tokens = 0
    _timeout = None

    def __init__(self) -> None:
        self.captures: list[list] = []

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                **kwargs):
        name = (valid_actions or ["ACTION1"])[0]
        args = {"action": name}
        if name == "ACTION6":
            args.update({"row": 32, "col": 32})
        step_env(args)
        self.captures.append(duck_patches._animation_current_payload())
        return _result(step_executed=True)


def _make_session(tmp_path, analyzer, game_name=GAME):
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from inference.framework import solver as duck_solver

    run_session = RunSession(record_intermediate_states=False)
    game = GameAPI(env_name=game_name,
                   arcade_spec=ArcadeSpec(environments_dir=str(ENV_DIR)))
    game.start_game(run_session)

    solver = duck_solver.HarnessSolver(
        label="animation-test",
        model="stub",
        analyzer_timeout=5.0,
        concurrency=1,
    )
    solver.job_dir = tmp_path
    session = duck_solver._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "artifacts" / "runtime_state.json",
        transcript_path=tmp_path / "transcripts" / f"{game_name}.txt",
        analysis_html_relpath=f"solver_analysis/{game_name}.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "artifacts" / "viewer_data.json",
    )
    return game, session


# --- 1. capture on a real animating game -----------------------------------------


def test_animation_captured_on_lf52(tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "0")
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    monkeypatch.setenv("TAAF_ANIMATION", "1")
    _apply()

    analyzer = _ProbeAnalyzer()
    game, session = _make_session(tmp_path, analyzer)
    session.solver.max_actions_per_game = 4
    session.play()

    assert analyzer.captures, "no actions executed"
    non_empty = [c for c in analyzer.captures if c]
    assert non_empty, (
        "lf52 must produce intermediate animation frames "
        f"(captures: {[len(c) for c in analyzer.captures]})"
    )
    frame = non_empty[0][0]
    assert set(frame) == {"ascii", "step", "level", "shape", "grid"}, frame.keys()
    assert frame["shape"][0] > 0 and frame["shape"][1] > 0
    assert len(frame["grid"]) == frame["shape"][0]
    assert frame["ascii"].strip(), "ascii render must be non-empty"
    # The payload crosses the sandbox IPC as JSON — engine grids are numpy
    # int8 and must have been converted to plain ints at capture time.
    json.dumps(non_empty[0])


def test_animation_disabled_yields_empty_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_GRAPH", "0")
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    monkeypatch.setenv("TAAF_ANIMATION", "0")
    _apply()

    analyzer = _ProbeAnalyzer()
    game, session = _make_session(tmp_path, analyzer)
    session.solver.max_actions_per_game = 2
    session.play()
    assert analyzer.captures
    assert all(c == [] for c in analyzer.captures), "TAAF_ANIMATION=0 must empty the payload"


# --- 2. queryable in a real sandbox subprocess ------------------------------------


def _minimal_state(extra=None):
    state = {
        "current_frame": {"ascii": "", "step": 1, "level": 1,
                          "shape": [2, 2], "grid": [[0, 0], [0, 0]]},
        "history": [],
        "valid_actions": ["ACTION1"],
        "last_action_result": {},
    }
    if extra:
        state.update(extra)
    return state


def _anim_payload(tag: int):
    return {
        "ascii": f"frame{tag}",
        "step": tag,
        "level": 1,
        "shape": [2, 2],
        "grid": [[tag, 0], [0, tag]],
    }


def test_last_animation_queryable_in_real_sandbox(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.setenv("TAAF_ANIMATION", "1")
    _apply()
    assert "last_animation" in sandbox_mod._SANDBOX_BOOTSTRAP

    _ANIM_TLS.frames = [_anim_payload(1), _anim_payload(2)]
    try:
        out = sandbox_mod.run_sandboxed_python(
            code=(
                "result = {\n"
                "  'n': len(last_animation),\n"
                "  'shape': last_animation[0].shape,\n"
                "  'grid00': last_animation[0]._grid[0][0],\n"
                "  'grid11_last': last_animation[-1]._grid[1][1],\n"
                "  'has_ascii': bool(last_animation[0].ascii),\n"
                "}\n"
            ),
            timeout_seconds=30,
            initial_state=_minimal_state(),  # no last_animation key: TLS route
            action_handler=lambda actions: {},
        )
        assert not out.get("error"), out
        assert out["result"] == {
            "n": 2, "shape": [2, 2], "grid00": 1, "grid11_last": 2, "has_ascii": True,
        }, out["result"]
    finally:
        _ANIM_TLS.frames = []


def test_last_animation_refreshes_after_sandbox_action(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.setenv("TAAF_ANIMATION", "1")
    _apply()

    def handler(actions):
        # The real handler executes the action (updating the TLS via
        # _execute_action) then serializes state; simulate exactly that.
        _ANIM_TLS.frames = [_anim_payload(7), _anim_payload(8), _anim_payload(9)]
        return {"action_result": {"executed": True},
                "state": _minimal_state()}  # no last_animation: wrapper injects

    _ANIM_TLS.frames = [_anim_payload(1)]
    try:
        out = sandbox_mod.run_sandboxed_python(
            code=(
                "before = len(last_animation)\n"
                "action([{'action': 'ACTION1'}])\n"
                "result = {'before': before, 'after': len(last_animation),\n"
                "          'after00': last_animation[0]._grid[0][0]}\n"
            ),
            timeout_seconds=30,
            initial_state=_minimal_state(),
            action_handler=handler,
        )
        assert not out.get("error"), out
        assert out["result"] == {"before": 1, "after": 3, "after00": 7}, out["result"]
    finally:
        _ANIM_TLS.frames = []


# --- 3. zero token cost when unqueried --------------------------------------------


def test_prompt_gains_exactly_one_line(monkeypatch):
    from inference.agent import tool_agent

    _apply()
    monkeypatch.setenv("TAAF_ANIMATION", "1")
    prompt_on = tool_agent._build_system_prompt(tool_output_tokens=1000)
    monkeypatch.setenv("TAAF_ANIMATION", "0")
    prompt_off = tool_agent._build_system_prompt(tool_output_tokens=1000)

    assert prompt_on == prompt_off + "\n\n" + duck_patches._ANIMATION_PROMPT_LINE
    assert "\n" not in duck_patches._ANIMATION_PROMPT_LINE, "must be ONE line"
    assert prompt_on.count("last_animation") == 1


def test_no_animation_content_when_unqueried(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.setenv("TAAF_ANIMATION", "1")
    _apply()

    _ANIM_TLS.frames = [_anim_payload(5)]
    try:
        out = sandbox_mod.run_sandboxed_python(
            code="result = 1 + 1\n",
            timeout_seconds=30,
            initial_state=_minimal_state(),
            action_handler=lambda actions: {},
        )
        assert not out.get("error"), out
        # Everything the model would ever see from this tool call:
        rendered = json.dumps(out)
        assert "last_animation" not in rendered
        assert "frame5" not in rendered, "unqueried frames must not surface"
    finally:
        _ANIM_TLS.frames = []


# --- 4. scored-bundle apply-or-SKIP (patch-9 law) ---------------------------------


@pytest.mark.skipif(not SCORED_REF.exists(), reason="scored ref not present")
def test_scored_bundle_bytes_carry_the_required_seams():
    """The patch's presence gates must all be satisfiable by the SCORED bundle
    (validated against the actual scored bytes, not the drifted _adopt tree)."""
    sandbox_py = (SCORED_REF / "src/ARC3-Inference/inference/agent/"
                  "python_tool_sandbox.py").read_text()
    game_py = (SCORED_REF / "src/tufa-arc-agi-framework/src/taaf/"
               "game.py").read_text()
    anchor = duck_patches._HUD_REFRESH_ANCHOR.strip()
    assert anchor in sandbox_py, "refresh anchor missing from scored bundle"
    assert "_frame_from_payload" in sandbox_py
    assert "def animation_frames" in game_py
    assert "self.raw.frame[-1]" in game_py, "frame[-1] discard site moved"
    grid_utils = (SCORED_REF / "src/ARC3-Inference/inference/utils/"
                  "grid_utils.py")
    assert grid_utils.exists() and "def format_grid_ascii" in grid_utils.read_text()


def test_patch_declines_on_unsupported_bootstrap(monkeypatch):
    """A bootstrap without _frame_from_payload (or without the refresh anchor)
    must produce a clean SKIP, never a NameError-in-every-sandbox ship."""
    from inference.agent import python_tool_sandbox as sandbox_mod

    original = sandbox_mod._SANDBOX_BOOTSTRAP
    try:
        sandbox_mod._SANDBOX_BOOTSTRAP = (
            "import json\n" + duck_patches._HUD_REFRESH_ANCHOR + "\ndef main():\n    pass\n"
        )
        status = duck_patches.patch_animation_sandbox()
        assert status.startswith("patch15 animation: SKIP"), status
        assert "_frame_from_payload" in status or "refresh anchor" in status, status

        sandbox_mod._SANDBOX_BOOTSTRAP = "def _frame_from_payload(p):\n    pass\n"
        status = duck_patches.patch_animation_sandbox()
        assert status.startswith("patch15 animation: SKIP"), status
    finally:
        sandbox_mod._SANDBOX_BOOTSTRAP = original


def test_reapply_is_idempotent():
    _apply()
    from inference.agent import python_tool_sandbox as sandbox_mod

    status = duck_patches.patch_animation_sandbox()
    assert "SKIP (already applied)" in status, status
    assert sandbox_mod._SANDBOX_BOOTSTRAP.count('runtime_globals["last_animation"]') == 1
