"""Tests for the in-memory duck patches.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_duck_patches.py -v

Test 1 deliberately runs BEFORE the patch so it records the bug as it ships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402


def test_action7_is_broken_before_patch():
    """The bug, as it ships: ACTION7 reaches the model but cannot get back to the engine."""
    from inference.agent import action_names as an

    # Reload a pristine copy so this test is order-independent.
    import importlib

    an = importlib.reload(an)

    assert an.to_model_action("ACTION7") == "ACTION7", "model is shown ACTION7"
    assert an.to_engine_action("ACTION7") is None, (
        "expected the shipped bug: ACTION7 does not map back to an engine action"
    )


def test_action7_round_trips_after_patch():
    from inference.agent import action_names as an

    duck_patches.patch_action7()

    assert an.to_engine_action("ACTION7") == "ACTION7"
    assert an.to_model_action("ACTION7") == "ACTION7"
    assert an.to_engine_action(an.to_model_action("ACTION7")) == "ACTION7"


def test_patch_is_idempotent():
    first = duck_patches.patch_action7()
    second = duck_patches.patch_action7()
    assert "OK" in first or "SKIP" in first
    assert "SKIP" in second


def test_existing_actions_still_round_trip():
    """The patch must not disturb ACTION1-6 or RESET."""
    from inference.agent import action_names as an

    duck_patches.patch_action7()

    for engine, model in [
        ("ACTION1", "UP"),
        ("ACTION2", "DOWN"),
        ("ACTION3", "LEFT"),
        ("ACTION4", "RIGHT"),
        ("ACTION5", "SPACE"),
        ("ACTION6", "MOUSE"),
        ("RESET", "RESET"),
    ]:
        assert an.to_model_action(engine) == model
        assert an.to_engine_action(model) == engine
        assert an.to_engine_action(engine) == engine


def test_action7_survives_valid_action_normalization():
    """The list of legal actions shown to the model must retain ACTION7."""
    duck_patches.patch_action7()
    from inference.agent.tool_agent import _normalize_valid_actions

    names = _normalize_valid_actions(["ACTION1", "ACTION6", "ACTION7"])
    assert "ACTION7" in names, f"ACTION7 dropped from valid actions: {names}"
    assert "UP" in names and "MOUSE" in names


def test_action7_resolves_to_an_engine_action_id():
    """arcengine must accept the name we hand back to it."""
    arcengine = pytest.importorskip("arcengine")
    duck_patches.patch_action7()
    from inference.agent import action_names as an

    engine_name = an.to_engine_action("ACTION7")
    action = arcengine.GameAction.from_name(engine_name)
    assert action is not None
    assert action.name == "ACTION7"


# --- animation metadata ---------------------------------------------------


class _Frame:
    def __init__(self, data):
        self.data = data


class _State:
    def __init__(self, frames):
        self._frames = [_Frame(f) for f in frames]

    @property
    def frame(self):
        return self._frames[-1]

    @property
    def animation_frames(self):
        return self._frames[:-1]


def test_summarize_animation_empty_when_single_frame():
    state = _State([[[0, 0], [0, 0]]])
    assert duck_patches.summarize_animation(state) == {}


def test_summarize_animation_reports_transient_motion():
    """A cell lit up mid-animation and is dark in the final frame — the key signal."""
    mid = [[0, 3], [0, 0]]
    final = [[0, 0], [0, 0]]
    state = _State([mid, final])

    out = duck_patches.summarize_animation(state)

    assert out["animation_frame_count"] == 1
    assert out["animation_changed"] is True
    assert out["animation_only_changed"] is True
    assert out["animation_changed_cell_count"] == 1
    assert out["animation_changed_bbox"] == [1, 0, 1, 0]


def test_summarize_animation_quiet_when_animation_matches_final():
    frame = [[1, 1], [1, 1]]
    state = _State([frame, frame])

    out = duck_patches.summarize_animation(state)

    assert out["animation_changed"] is False
    assert out["animation_changed_cell_count"] == 0
    assert "animation_changed_bbox" not in out


def test_summarize_animation_never_raises_on_odd_input():
    class _Broken:
        @property
        def animation_frames(self):
            raise ValueError("no frames here")

    assert duck_patches.summarize_animation(_Broken()) == {}


def test_animation_producer_injects_keys_into_payload(monkeypatch):
    """The wrapped _execute_action must add the animation summary to the real payload.

    The upstream body needs a live engine, so we substitute a stub, apply the patch on
    top of it, and check the wrapper enriched what the stub returned.
    """
    from inference.framework import solver

    def _stub(self, *args, **kwargs):
        return {"executed": True, "action_num": 1}

    monkeypatch.setattr(solver._HarnessGameSession, "_execute_action", _stub, raising=True)

    status = duck_patches.patch_animation_producer()
    assert "OK" in status, status

    class _Game:
        # one animation frame showing a lit cell that is dark in the final frame
        current_state = _State([[[0, 3], [0, 0]], [[0, 0], [0, 0]]])

    class _Session:
        game = _Game()

    session = _Session()
    payload = solver._HarnessGameSession._execute_action(session)

    assert payload["executed"] is True, "must preserve the original payload"
    assert payload["animation_frame_count"] == 1
    assert payload["animation_only_changed"] is True
    assert payload["animation_changed_cell_count"] == 1
    assert payload["animation_changed_bbox"] == [1, 0, 1, 0]


def test_animation_producer_never_breaks_an_action(monkeypatch):
    """A broken game state must not stop the action from returning its payload."""
    from inference.framework import solver

    def _stub(self, *args, **kwargs):
        return {"executed": True}

    monkeypatch.setattr(solver._HarnessGameSession, "_execute_action", _stub, raising=True)
    duck_patches.patch_animation_producer()

    class _Session:
        @property
        def game(self):
            raise RuntimeError("engine gone")

    payload = solver._HarnessGameSession._execute_action(_Session())
    assert payload == {"executed": True}


def test_patch6_survives_notebook_exec_without_file():
    """patch6 exec'd the way the Kaggle hook cell runs (no __file__ in globals).

    The shipped v5 built its arcagi3 path list from Path(__file__) and died with a
    NameError -> 'patch6 ... FAIL' in every notebook context. It must now either
    apply or SKIP with an accurate message — never FAIL for this structural reason.
    """
    src = (Path(__file__).parent / "duck_patches.py").read_text()
    ns: dict = {}
    exec(compile(src, "<hook-cell>", "exec"), ns)  # noqa: S102 - mirrors Kaggle exactly
    assert "__file__" not in ns
    line = ns["patch_tool_agent_analyze"]()
    assert line.startswith("patch6"), line
    assert "FAIL" not in line, line
    assert "NameError" not in line, line


class _FakeFrame:
    def __init__(self, grid):
        self.grid = grid


def _grid_burner_fixture():
    """Pristine stock renderer + freshly-applied patch4 wrapper on the same module."""
    import importlib

    from inference.agent import vision_context

    vision_context = importlib.reload(vision_context)  # drop any previous wrapper
    stock = vision_context.frame_to_png_data_url
    status = duck_patches.patch_dynamic_grid_burner()
    assert "OK" in status, status
    return stock, vision_context.frame_to_png_data_url


def test_grid_burner_default_off_is_pixel_identical_to_stock(monkeypatch):
    """Unset env (the scored default): the wrapper must not touch a single pixel."""
    monkeypatch.delenv("TAAF_GRID_BURNER", raising=False)
    stock, patched = _grid_burner_fixture()
    assert patched is not stock
    frame = _FakeFrame([[1, 2, 3, 0], [3, 4, 0, 1], [0, 1, 2, 3], [2, 0, 1, 4]])
    # upscale=4 is the scored config's MULTIMODAL_UPSCALE — the exact setting the
    # 2026-08-03 diagnosis showed the burner defaces.
    assert patched(frame, upscale=4) == stock(frame, upscale=4)
    assert patched(frame, upscale=16) == stock(frame, upscale=16)


def test_grid_burner_explicit_zero_is_off(monkeypatch):
    monkeypatch.setenv("TAAF_GRID_BURNER", "0")
    stock, patched = _grid_burner_fixture()
    frame = _FakeFrame([[1, 2], [3, 4]])
    assert patched(frame, upscale=4) == stock(frame, upscale=4)


def test_grid_burner_opt_in_draws_the_overlay(monkeypatch):
    """TAAF_GRID_BURNER=1: grid lines/labels change the rendered image."""
    monkeypatch.setenv("TAAF_GRID_BURNER", "1")
    stock, patched = _grid_burner_fixture()
    frame = _FakeFrame([[1, 2, 3, 0], [3, 4, 0, 1], [0, 1, 2, 3], [2, 0, 1, 4]])
    assert patched(frame, upscale=16) != stock(frame, upscale=16)


def test_grid_burner_toggle_is_call_time_not_apply_time(monkeypatch):
    """The switch is read per call, like the other patches' env switches."""
    monkeypatch.setenv("TAAF_GRID_BURNER", "1")
    stock, patched = _grid_burner_fixture()
    frame = _FakeFrame([[1, 2], [3, 4]])
    burned = patched(frame, upscale=16)
    assert burned != stock(frame, upscale=16)
    monkeypatch.setenv("TAAF_GRID_BURNER", "0")
    assert patched(frame, upscale=16) == stock(frame, upscale=16)


def test_apply_all_reports_every_patch():
    lines = duck_patches.apply_all(verbose=False)
    assert len(lines) == 24  # 1-6 + verify + 7/8/9/10/11 + 12a + 12b + 13 + 14 + 15 + 17 + 16 + 18 + 19 + 20 + 21 + 22
    joined = " | ".join(lines)
    assert "FAIL" not in joined, joined
    assert "patch3 reset: NOT NEEDED" in joined, (
        "our bundle should already restrict RESET; if this fails, upstream changed"
    )
