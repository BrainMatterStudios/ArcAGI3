"""Tests for patch 19 (archetype-dispatch prompt scaffolds).

Rank 3 of the 2026-08-07 human-play idea sweep: patch 17 measures the genre
(GAME_MODE), patch 19 dispatches on it — a short mode-specific doctrine addendum
rides the system prompt while the verdict holds. Exactly TWO scaffolds exist
(AVATAR-NAV, CLICK-PUZZLE) per the sweep's scope warning; ARROW-MORPH and
UNCLEAR fall back to the neutral (current) prompts. The addendum is served
through a data-descriptor property on ToolAgent._system_prompt (the cached
`self._system_prompt` would otherwise go stale across re-probe verdict flips),
and a stall escape hatch drops the scaffold one-way per level after
TAAF_DISPATCH_STALL scaffold-active actions with zero effect novelty.

Layers: scaffold selection per mode (incl. both fallback modes and pre-battery
silence), system-prompt property freshness (swap-on-reprobe, base preservation,
agents constructed before the patch), escape-hatch trigger / novelty reset /
one-way-per-level / level-clear semantics, the _execute_action feed through the
REAL patched session stack, coexistence with patch 13/16/17/18 blocks, OFF-by-
default no-op, idempotency, hostile-bundle decline, and the scored-bundle
staleness gate.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_dispatch.py -v
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import (  # noqa: E402
    _DISPATCH_MARKER,
    _DISPATCH_TLS,
    _WIGGLE_TLS,
    DISPATCH_DIAGNOSTICS,
    WiggleState,
    _dispatch_observe,
    _dispatch_prompt_block,
    patch_archetype_dispatch,
)

LEGEND_MARKER = "CONTROLLABILITY LEGEND"
PLAYBOOK_MARKER = "Mechanic playbook"
PROBE_ADDENDUM_MARKER = "run_probe — probe battery"


def _apply() -> str:
    # patch 17 first: patch 19's escape hatch reads its per-action state, and
    # apply_all installs them in exactly this order.
    wiggle = duck_patches.patch_wiggle()
    assert "OK" in wiggle or "SKIP" in wiggle, wiggle
    result = patch_archetype_dispatch()
    assert "OK" in result or "SKIP" in result, result
    return result


@pytest.fixture(autouse=True)
def _clean_tls():
    _WIGGLE_TLS.state = None
    _DISPATCH_TLS.state = None
    yield
    _WIGGLE_TLS.state = None
    _DISPATCH_TLS.state = None


def _ws(mode: str = "AVATAR", level: int = 1) -> WiggleState:
    state = WiggleState()
    state.battery_done = True
    state.level = level
    state.mode = mode
    state.mode_reason = "test verdict"
    state.avatar_color = "b"
    state.avatar_cells = [(5, 3), (5, 4)]
    state.avatar_action = "ACTION2"
    state.avatar_shift = (1, 0)
    state.shape = (16, 16)
    state.changed_ever = {(5, 3), (5, 4)}
    return state


# ---------------------------------------------------------------------------------
# scaffold selection: two scaffolds + neutral fallback
# ---------------------------------------------------------------------------------


def test_avatar_scaffold_cites_measured_fields(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
    _WIGGLE_TLS.state = _ws("AVATAR")
    block = _dispatch_prompt_block()
    assert block.startswith(f"{_DISPATCH_MARKER}: AVATAR-NAV")
    assert "'b'" in block  # the measured avatar color, not a guess
    assert "WIGGLE_MASKS" in block and "GAME_MODE" in block
    assert LEGEND_MARKER in block  # points at the field the model actually receives
    assert "DEATH" in block  # avatar resets are deaths, not strategy
    assert "wall" in block


def test_click_scaffold_cites_measured_fields(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
    _WIGGLE_TLS.state = _ws("CLICK")
    block = _dispatch_prompt_block()
    assert block.startswith(f"{_DISPATCH_MARKER}: CLICK-PUZZLE")
    assert "NO avatar" in block
    assert "reactive_remote" in block  # remote-effect clicks (patch 17 tracks them)
    assert "SAME cell" in block  # repeat-clicking is mechanic-required in this genre
    assert "reconnaissance" in block  # reset-then-execute is a human strategy here


def test_scaffolds_stay_short(monkeypatch):
    """~15-line budget per the build spec — scaffolds are doctrine, not essays."""
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")  # longest variant
    for mode in ("AVATAR", "CLICK"):
        _WIGGLE_TLS.state = _ws(mode)
        assert len(_dispatch_prompt_block().splitlines()) <= 15, mode


def test_neutral_fallback_for_arrow_morph_and_unclear(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    for mode in ("ARROW-MORPH", "UNCLEAR"):
        _WIGGLE_TLS.state = _ws(mode)
        assert _dispatch_prompt_block() == "", mode


def test_block_empty_when_disabled_or_before_battery(monkeypatch):
    monkeypatch.delenv("TAAF_DISPATCH", raising=False)
    _WIGGLE_TLS.state = _ws("AVATAR")
    assert _dispatch_prompt_block() == ""
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    _WIGGLE_TLS.state = None  # no wiggle measurement -> nothing to dispatch on
    assert _dispatch_prompt_block() == ""
    state = WiggleState()  # battery not yet run
    _WIGGLE_TLS.state = state
    assert _dispatch_prompt_block() == ""


def test_run_probe_line_gated_on_probe_env(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    for mode in ("AVATAR", "CLICK"):
        _WIGGLE_TLS.state = _ws(mode)
        monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
        assert "run_probe" not in _dispatch_prompt_block(), mode
        monkeypatch.setenv("TAAF_RUN_PROBE", "1")
        assert "run_probe([...])" in _dispatch_prompt_block(), mode


# ---------------------------------------------------------------------------------
# system-prompt property: freshness, base preservation, swap-on-reprobe
# ---------------------------------------------------------------------------------


def _agent():
    from inference.agent.tool_agent import ToolAgent

    _apply()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/dispatch-test")
    return agent


def test_system_prompt_appends_scaffold_and_preserves_base(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    agent = _agent()
    base = agent._dispatch_base_system_prompt
    assert base  # __init__'s write went through the property setter
    _WIGGLE_TLS.state = _ws("AVATAR")
    prompt = agent._system_prompt
    assert prompt.startswith(base)  # ADDITIVE: nothing stripped from the base
    assert f"{_DISPATCH_MARKER}: AVATAR-NAV" in prompt
    assert prompt.index(_DISPATCH_MARKER) > len(base) - 1


def test_system_prompt_swaps_on_reprobe_verdict_flip(monkeypatch):
    """A per-level re-probe flips the verdict -> the NEXT read swaps the
    scaffold; no stale scaffold can ship (the seam that forced the property)."""
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    agent = _agent()
    state = _ws("AVATAR")
    _WIGGLE_TLS.state = state
    assert "AVATAR-NAV" in agent._system_prompt
    state.mode = "CLICK"  # what _wiggle_reprobe does on a click-only level
    prompt = agent._system_prompt
    assert "CLICK-PUZZLE" in prompt and "AVATAR-NAV" not in prompt
    state.mode = "UNCLEAR"  # verdict withdrawn -> neutral prompts
    assert _DISPATCH_MARKER not in agent._system_prompt


def test_off_by_default_system_prompt_is_exactly_the_base(monkeypatch):
    monkeypatch.delenv("TAAF_DISPATCH", raising=False)
    agent = _agent()
    _WIGGLE_TLS.state = _ws("AVATAR")
    assert agent._system_prompt == agent._dispatch_base_system_prompt
    assert _DISPATCH_MARKER not in agent._system_prompt


def test_agent_constructed_before_patch_keeps_its_prompt(monkeypatch):
    """An instance whose __dict__ carries _system_prompt (written before the
    property existed) must keep serving it — the getter falls back."""
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("TAAF_DISPATCH", "1")
    _apply()
    agent = object.__new__(ToolAgent)
    agent.__dict__["_system_prompt"] = "PRE-PATCH BASE"
    _WIGGLE_TLS.state = _ws("CLICK")
    prompt = ToolAgent._system_prompt.fget(agent)
    assert prompt.startswith("PRE-PATCH BASE")
    assert "CLICK-PUZZLE" in prompt


# ---------------------------------------------------------------------------------
# escape hatch: trigger, novelty reset, one-way per level
# ---------------------------------------------------------------------------------


def test_escape_fires_after_k_stalled_actions(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_DISPATCH_STALL", "3")
    state = _ws("AVATAR")
    _WIGGLE_TLS.state = state
    escapes_before = DISPATCH_DIAGNOSTICS["escapes"]
    _dispatch_observe(state)  # arms the scaffold + baseline novelty signature
    _dispatch_observe(state)  # stall 1
    _dispatch_observe(state)  # stall 2
    assert state.mode == "AVATAR"
    assert _dispatch_prompt_block() != ""
    _dispatch_observe(state)  # stall 3 == K -> escape
    disp = _DISPATCH_TLS.state
    assert disp.escaped and disp.escaped_mode == "AVATAR"
    assert DISPATCH_DIAGNOSTICS["escapes"] == escapes_before + 1
    # the verdict is marked UNCLEAR so legend + sandbox global + prompts agree
    assert state.mode == "UNCLEAR"
    assert "dispatch escape" in state.mode_reason
    assert state.mode_history[-1] == (1, "UNCLEAR")
    assert _dispatch_prompt_block() == ""  # neutral (current) prompts from here


def test_novelty_resets_the_stall_counter(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_DISPATCH_STALL", "3")
    state = _ws("AVATAR")
    _WIGGLE_TLS.state = state
    _dispatch_observe(state)
    _dispatch_observe(state)
    _dispatch_observe(state)  # stall 2
    state.changed_ever.add((9, 9))  # a genuinely new effect cell
    _dispatch_observe(state)  # novelty -> stall back to 0
    assert _DISPATCH_TLS.state.stall_actions == 0
    _dispatch_observe(state)
    _dispatch_observe(state)
    assert not _DISPATCH_TLS.state.escaped
    assert state.mode == "AVATAR"


def test_new_effect_class_row_counts_as_novelty(monkeypatch):
    """A new (action -> effect-class) row in patch 17's press stats resets the
    clock even when no new cell changed (e.g. a first morph on a known cell)."""
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_DISPATCH_STALL", "4")
    state = _ws("AVATAR")
    _WIGGLE_TLS.state = state
    _dispatch_observe(state)
    _dispatch_observe(state)  # stall 1
    state.press_stats["ACTION1"] = {"presses": 1, "noops": 0, "morphs": 1, "moves": []}
    _dispatch_observe(state)  # new row -> reset
    assert _DISPATCH_TLS.state.stall_actions == 0


def test_escape_is_one_way_within_the_level(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_DISPATCH_STALL", "2")
    state = _ws("CLICK")
    _WIGGLE_TLS.state = state
    for _ in range(3):
        _dispatch_observe(state)
    assert _DISPATCH_TLS.state.escaped
    # even if the verdict re-flips on the same level, the scaffold stays down
    state.mode = "CLICK"
    state.changed_ever.add((1, 1))
    for _ in range(3):
        _dispatch_observe(state)
    assert _DISPATCH_TLS.state.escaped
    assert _dispatch_prompt_block() == ""


def test_escape_clears_on_level_change(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_DISPATCH_STALL", "2")
    state = _ws("AVATAR")
    _WIGGLE_TLS.state = state
    for _ in range(3):
        _dispatch_observe(state)
    assert _DISPATCH_TLS.state.escaped
    # what patch 17 does at a level boundary: reset evidence, re-verdict
    state.level = 2
    state.mode = "AVATAR"
    _dispatch_observe(state)
    disp = _DISPATCH_TLS.state
    assert not disp.escaped and disp.level == 2
    assert _dispatch_prompt_block() != ""


def test_mode_swap_resets_stall_and_counts_a_swap(monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_DISPATCH_STALL", "10")
    state = _ws("AVATAR")
    _WIGGLE_TLS.state = state
    swaps_before = DISPATCH_DIAGNOSTICS["swaps"]
    _dispatch_observe(state)
    _dispatch_observe(state)  # stall 1
    state.mode = "CLICK"
    _dispatch_observe(state)  # fresh scaffold -> fresh stall window
    disp = _DISPATCH_TLS.state
    assert disp.active_mode == "CLICK" and disp.stall_actions == 0
    assert DISPATCH_DIAGNOSTICS["swaps"] == swaps_before + 1


# ---------------------------------------------------------------------------------
# real patched stack: the _execute_action wrap feeds the escape hatch
# ---------------------------------------------------------------------------------


def _board(fill: int = 0, size: int = 16) -> list[list[int]]:
    return [[fill] * size for _ in range(size)]


class _PlainGame:
    """Fake engine game: every action executes, nothing terminal happens."""

    def __init__(self, board):
        import arcengine

        self._arcengine = arcengine
        self.number_of_levels = 3
        self.game_run = SimpleNamespace(history=[], state="playing")
        self._board = board
        self.current_state = self._state()

    def _state(self):
        return SimpleNamespace(
            frame=SimpleNamespace(data=[list(r) for r in self._board]),
            available_actions=[1, 2, 3, 4, 6],
            levels_completed=0,
            won=False,
            just_won_level=False,
            raw=SimpleNamespace(state=self._arcengine.GameState.NOT_FINISHED),
            animation_frames=[],
        )

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        self.game_run.history.append(action.id.name)
        self.current_state = self._state()
        return self.current_state


def _real_session(tmp_path, game):
    from inference.framework import solver as duck_solver

    solver = duck_solver.HarnessSolver(label="t", model="stub", concurrency=1)
    solver.job_dir = None  # viewer writes no-op
    return duck_solver._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=SimpleNamespace(),
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "runtime_state.json",
        transcript_path=tmp_path / "t.txt",
        analysis_html_relpath="t.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "viewer_data.json",
    )


def _click_action():
    import arcengine

    return arcengine.ActionInput(
        id=arcengine.GameAction.ACTION6, data={"x": 3, "y": 3}
    )


def test_execute_action_wrap_feeds_the_escape_hatch(tmp_path, monkeypatch):
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    _apply()
    seen = []
    monkeypatch.setattr(duck_patches, "_dispatch_observe", lambda ws: seen.append(ws))
    session = _real_session(tmp_path, _PlainGame(_board()))
    state = _ws("CLICK")
    session._wiggle_state = state
    payload = session._execute_action(
        _click_action(), batch_index=1, batch_size=1, generated_tokens=0
    )
    assert payload["executed"] and not payload["level_completed"]
    assert seen == [state]


def test_execute_action_wrap_skips_probe_presses_and_off(tmp_path, monkeypatch):
    _apply()
    seen = []
    monkeypatch.setattr(duck_patches, "_dispatch_observe", lambda ws: seen.append(ws))
    session = _real_session(tmp_path, _PlainGame(_board()))
    state = _ws("CLICK")
    session._wiggle_state = state
    # disabled: never fed
    monkeypatch.delenv("TAAF_DISPATCH", raising=False)
    session._execute_action(_click_action(), batch_index=1, batch_size=1,
                            generated_tokens=0)
    assert seen == []
    # battery/reprobe presses are not scaffold turns
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    state.probing = True
    session._execute_action(_click_action(), batch_index=1, batch_size=1,
                            generated_tokens=0)
    assert seen == []


# ---------------------------------------------------------------------------------
# coexistence with patches 13 / 16 / 17 / 18
# ---------------------------------------------------------------------------------


def test_scaffold_lands_below_playbook_and_probe_addenda(monkeypatch):
    """System-prompt stack: base -> playbook (13) -> run_probe ad (18) ->
    scaffold (19). 13/18 append at build time, 19 appends at read time."""
    monkeypatch.setenv("TAAF_PLAYBOOK", "1")
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    duck_patches.patch_mechanic_playbook()
    duck_patches.patch_run_probe()
    agent = _agent()
    _WIGGLE_TLS.state = _ws("AVATAR")
    prompt = agent._system_prompt
    for marker in (PLAYBOOK_MARKER, PROBE_ADDENDUM_MARKER, _DISPATCH_MARKER):
        assert marker in prompt, marker
    assert (
        prompt.index(PLAYBOOK_MARKER)
        < prompt.index(PROBE_ADDENDUM_MARKER)
        < prompt.index(_DISPATCH_MARKER)
    )


def test_user_prompt_untouched_by_dispatch(monkeypatch):
    """Patch 19 rides the system prompt only: the analyzer user prompt keeps
    patch 17's legend and never carries the scaffold."""
    from inference.agent.runtime_state import Frame

    monkeypatch.setenv("TAAF_WIGGLE", "1")
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    agent = _agent()
    _WIGGLE_TLS.state = _ws("AVATAR")
    _WIGGLE_TLS.state.battery_presses = 4
    prompt = agent._build_user_prompt(
        1,
        valid_actions=["ACTION1"],
        current_frame=Frame(grid=((1, 2), (3, 4)), step=1, level=1),
    )
    assert LEGEND_MARKER in prompt
    assert _DISPATCH_MARKER not in prompt


# ---------------------------------------------------------------------------------
# patch mechanics
# ---------------------------------------------------------------------------------


def test_patch_applies_and_is_idempotent():
    first = _apply()
    assert first.startswith("patch19 dispatch: OK") or "SKIP" in first
    assert patch_archetype_dispatch() == "patch19 dispatch: SKIP (already applied)"


def test_patch_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.framework import solver

    monkeypatch.setattr(solver, "_HarnessGameSession", None)
    msg = patch_archetype_dispatch()
    assert msg.startswith("patch19 dispatch: FAIL"), msg


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE staleness gate (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------

SCORED_INFERENCE = SCORED_REF / "src/ARC3-Inference"


def test_scored_bundle_has_every_symbol_patch19_touches():
    solver_src = (SCORED_INFERENCE / "inference/framework/solver.py").read_text()
    for symbol in ("class _HarnessGameSession", "def _execute_action", "def play"):
        assert symbol in solver_src, f"scored bundle lost {symbol!r} — re-validate patch19"
    agent_src = (SCORED_INFERENCE / "inference/agent/tool_agent.py").read_text()
    for symbol in (
        "class ToolAgent",
        "def _build_system_prompt",
        # the cached-assignment seam the property routes; if this moves, the
        # base-slot write path must be re-validated
        "self._system_prompt = _build_system_prompt(",
    ):
        assert symbol in agent_src, f"scored bundle lost {symbol!r} — re-validate patch19"


@pytest.mark.skipif(not SCORED_INFERENCE.is_dir(), reason="scored bundle bytes absent")
def test_patch19_applies_on_scored_bundle_bytes():
    """Apply patch 19 on the scored bytes in a clean subprocess: dormant, the
    system prompt is untouched; enabled, the scaffold lands and swaps live."""
    script = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(SCORED_INFERENCE)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
os.environ.pop("TAAF_DISPATCH", None)
os.environ.pop("TAAF_RUN_PROBE", None)
import duck_patches

r = duck_patches.patch_archetype_dispatch()
assert r.startswith("patch19 dispatch: OK"), r

from inference.agent.tool_agent import ToolAgent
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/dispatch-scored")
base = agent._dispatch_base_system_prompt
assert base, "base system prompt did not route through the property setter"

state = duck_patches.WiggleState()
state.battery_done = True
state.level = 1
state.mode = "CLICK"
state.mode_reason = "all 4 directional presses were masked no-ops"
duck_patches._WIGGLE_TLS.state = state

# disabled (default): the scored-bundle system prompt must be byte-identical
assert agent._system_prompt == base, "dormant patch leaked into the system prompt"

os.environ["TAAF_DISPATCH"] = "1"
prompt = agent._system_prompt
assert prompt.startswith(base)
assert "ARCHETYPE SCAFFOLD: CLICK-PUZZLE" in prompt, prompt[-300:]
state.mode = "AVATAR"
state.avatar_color = "b"
assert "ARCHETYPE SCAFFOLD: AVATAR-NAV" in agent._system_prompt, "swap did not land"
print("SCORED-DISPATCH-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-DISPATCH-OK" in proc.stdout
