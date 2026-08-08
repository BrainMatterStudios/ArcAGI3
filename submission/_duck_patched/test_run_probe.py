"""Tests for patch 18 (run_probe — guarded probe-battery macro-action).

Rank 1 of the 2026-08-07 human-play idea sweep: the model gets a probe
macro-action in the python tool — `run_probe(actions)` executes up to
TAAF_PROBE_MAX_ACTIONS (default 20) real environment actions as one guarded
call via the sandbox's native `action()` IPC channel (a sentinel-tagged
action() call intercepted host-side), returning a per-action effect table
(patch 16's structured change lines, level/score, terminal flags) that is also
repeated as a PROBE REPORT in the next analyzer prompt.

Layers: host executor guards (per-call cap incl. env tuning, per-level call
budget, RESET rejection, availability respect, early stop on level/game_over,
effect-table correctness on synthetic handlers), sentinel interception through
a REAL sandbox subprocess, coexistence with patches 16/17 through the REAL
patched session stack (their observers must see probe actions), system-prompt
advertisement incl. the verbatim batching doctrine, next-prompt report
injection and consumption, OFF-by-default no-op, idempotency, hostile-bundle
decline, and the scored-bundle staleness gate.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_run_probe.py -v
"""
from __future__ import annotations

import json
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
    _PROBE_SENTINEL,
    _PROBE_TLS,
    _probe_prompt_block,
    _probe_tls_state,
    probe_execute,
)

REPORT_MARKER = "PROBE REPORT"
ADVERT_MARKER = "run_probe — probe battery"
DOCTRINE_SENTENCE = "short lists for new hypotheses, scale up for proven sequences"


def _apply() -> str:
    result = duck_patches.patch_run_probe()
    assert "OK" in result or "SKIP" in result, result
    return result


@pytest.fixture(autouse=True)
def _fresh_probe_tls():
    _PROBE_TLS.state = None
    yield
    _PROBE_TLS.state = None


# ---------------------------------------------------------------------------------
# synthetic handler: a scripted environment at the state-payload level
# ---------------------------------------------------------------------------------


def _grid(fill: int = 0, size: int = 8) -> list[list[int]]:
    return [[fill] * size for _ in range(size)]


def _paint(grid, cells, color):
    out = [list(row) for row in grid]
    for y, x in cells:
        out[y][x] = color
    return out


def _state(grid, level=1, valid=("UP", "DOWN", "LEFT", "RIGHT", "MOUSE"), step=1):
    return {
        "current_frame": {
            "ascii": "",
            "step": step,
            "level": level,
            "shape": [len(grid), len(grid[0])],
            "grid": [list(r) for r in grid],
        },
        "history": [],
        "valid_actions": list(valid),
        "last_action_result": {},
    }


class ScriptedHandler:
    """Fake `_handle_action`: applies scripted per-action outcomes in order."""

    def __init__(self, outcomes, start_grid=None, level=1, valid=None):
        self.outcomes = list(outcomes)
        self.calls: list[list[dict]] = []
        self.grid = [list(r) for r in (start_grid or _grid())]
        self.level = level
        self.valid = list(valid) if valid is not None else None
        self.step = 0

    def state(self):
        valid = self.valid if self.valid is not None else ["UP", "DOWN", "MOUSE"]
        return _state(self.grid, level=self.level, valid=valid, step=self.step)

    def __call__(self, actions):
        self.calls.append([dict(a) for a in actions])
        assert len(actions) == 1, "probe must execute ONE action per handler call"
        self.step += 1
        outcome = dict(self.outcomes.pop(0)) if self.outcomes else {}
        if "grid" in outcome:
            self.grid = outcome.pop("grid")
        if "level" in outcome:
            self.level = outcome["level"]
        result = {
            "executed": True,
            "action_num": self.step,
            "level": self.level,
            "score": self.level - 1,
            "reward": 0.0,
            "board_changed": True,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "action_display": str(actions[0].get("action", "?")),
        }
        result.update(outcome)
        return {"action_result": result, "state": self.state()}


def _enable(monkeypatch, **env):
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))


# ---------------------------------------------------------------------------------
# host executor: effect-table correctness
# ---------------------------------------------------------------------------------


def test_effect_table_records_per_action_change_lines(monkeypatch):
    _enable(monkeypatch)
    start = _paint(_grid(), [(2, 2), (2, 3)], 9)
    moved = _paint(_grid(), [(3, 2), (3, 3)], 9)
    handler = ScriptedHandler([{"grid": moved}, {"board_changed": False}], start)
    out = probe_execute(["DOWN", "DOWN"], handler, _state(start))
    table = out["action_result"]
    assert table["probe"] is True and table["executed"] is True
    assert table["requested"] == 2 and table["executed_count"] == 2
    assert not table["stopped_early"]
    step1, step2 = table["steps"]
    assert step1["action"] == "DOWN" and step1["executed"]
    assert any("MOVED" in line or "CHANGE" in line for line in step1["change_lines"])
    assert step2["change_lines"] == ["NO CHANGE (board identical)"]
    cost = table["cost"]
    assert cost["actions_spent_this_call"] == 2
    assert cost["probe_actions_total"] == 2
    assert cost["probe_calls_left_this_level"] == 2  # 3 - this call
    assert json.dumps(table)  # the whole table must be JSON-able
    # the refreshed state travels back so the sandbox variables stay current
    assert out["state"]["current_frame"]["grid"] == moved


def test_effect_table_is_hud_masked(monkeypatch):
    _enable(monkeypatch)
    start = _grid()
    ticked = _paint(_grid(), [(0, 1)], 5)  # only a HUD bar cell changes
    handler = ScriptedHandler([{"grid": ticked}], start)
    monkeypatch.setattr(duck_patches, "_hud_current_mask_cells", lambda: [[0, 1]])
    out = probe_execute(["UP"], handler, _state(start))
    lines = out["action_result"]["steps"][0]["change_lines"]
    assert lines == ["NO CHANGE (HUD/step-counter tick only)"]


def test_cost_accumulates_across_calls(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([{}, {}, {}])
    probe_execute(["UP", "UP"], handler, handler.state())
    out = probe_execute(["DOWN"], handler, handler.state())
    cost = out["action_result"]["cost"]
    assert cost["probe_actions_total"] == 3
    assert cost["probe_calls_left_this_level"] == 1


# ---------------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------------


def test_per_call_cap_truncates_and_reports(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([{} for _ in range(30)])
    out = probe_execute(["UP"] * 30, handler, handler.state())
    table = out["action_result"]
    assert table["requested"] == 30 and table["executed_count"] == 20
    assert table["truncated_to_cap"] == 20
    assert table["stop_reason"] == "per_call_cap"
    assert table["stopped_early"]
    assert len(handler.calls) == 20


def test_cap_is_env_tunable(monkeypatch):
    _enable(monkeypatch, TAAF_PROBE_MAX_ACTIONS=2)
    handler = ScriptedHandler([{}, {}, {}])
    out = probe_execute(["UP", "UP", "UP"], handler, handler.state())
    assert out["action_result"]["executed_count"] == 2
    assert out["action_result"]["truncated_to_cap"] == 2


def test_reset_refuses_the_whole_call_without_executing(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([{}, {}, {}])
    for spec in ("RESET", "reset", {"action": "RESET"}):
        out = probe_execute(["UP", spec, "UP"], handler, handler.state())
        table = out["action_result"]
        assert table["executed"] is False
        assert "RESET" in table["error"]
        assert table["steps"] == []
    assert handler.calls == []  # nothing was ever executed


def test_per_level_call_budget_refuses_then_resets_on_new_level(monkeypatch):
    _enable(monkeypatch, TAAF_PROBE_MAX_CALLS=2)
    handler = ScriptedHandler([{} for _ in range(10)])
    assert probe_execute(["UP"], handler, handler.state())["action_result"]["executed"]
    assert probe_execute(["UP"], handler, handler.state())["action_result"]["executed"]
    refused = probe_execute(["UP"], handler, handler.state())["action_result"]
    assert refused["executed"] is False and "budget exhausted" in refused["error"]
    assert len(handler.calls) == 2
    handler.level = 2  # fresh level -> fresh budget
    assert probe_execute(["UP"], handler, handler.state())["action_result"]["executed"]


def test_available_actions_respected_per_step(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([{}], valid=["MOUSE"])  # arrows not offered
    out = probe_execute(["UP", "UP"], handler, handler.state())
    table = out["action_result"]
    assert table["executed_count"] == 0 and table["steps"] == []
    assert "UP is not in valid_actions" in table["stop_reason"]
    assert handler.calls == []
    # availability is re-checked against the REFRESHED state after each action
    handler2 = ScriptedHandler([{}], valid=["UP"])
    handler2.outcomes = [{}]

    def flip_valid(actions, _orig=handler2.__call__):
        out = _orig(actions)
        out["state"]["valid_actions"] = ["MOUSE"]  # arrows vanish after step 1
        return out

    out2 = probe_execute(["UP", "UP"], flip_valid, handler2.state())
    assert out2["action_result"]["executed_count"] == 1
    assert "not in valid_actions" in out2["action_result"]["stop_reason"]


def test_early_stop_on_level_completed_and_game_over(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([{}, {"level_completed": True, "level": 2}, {}])
    out = probe_execute(["UP"] * 5, handler, handler.state())
    table = out["action_result"]
    assert table["executed_count"] == 2 and table["stopped_early"]
    assert table["stop_reason"] == "level_completed"
    assert table["steps"][1]["change_lines"] == [
        "LEVEL COMPLETED — scene repainted, per-object diff skipped"
    ]
    handler2 = ScriptedHandler([{"game_over": True}])
    out2 = probe_execute(["UP", "UP"], handler2, handler2.state())
    assert out2["action_result"]["stop_reason"] == "game_over"
    assert len(handler2.calls) == 1


def test_unexecuted_result_stops_the_probe(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([{}, {"executed": False, "error": "not valid right now"}])
    out = probe_execute(["UP", "DOWN", "UP"], handler, handler.state())
    table = out["action_result"]
    assert table["executed_count"] == 1
    assert table["steps"][1]["executed"] is False
    assert "not valid" in table["steps"][1]["error"]
    assert len(handler.calls) == 2  # the failed step consumed no further actions


def test_disabled_probe_refuses_without_executing(monkeypatch):
    monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
    handler = ScriptedHandler([{}])
    out = probe_execute(["UP"], handler, handler.state())
    table = out["action_result"]
    assert table["executed"] is False and "not enabled" in table["error"]
    assert handler.calls == []


# ---------------------------------------------------------------------------------
# sandbox integration: sentinel interception through the REAL subprocess
# ---------------------------------------------------------------------------------


def test_run_probe_global_round_trips_through_real_sandbox(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    _enable(monkeypatch)
    _apply()
    assert 'runtime_globals["run_probe"]' in sandbox_mod._SANDBOX_BOOTSTRAP
    start = _paint(_grid(), [(2, 2)], 9)
    moved = _paint(_grid(), [(3, 2)], 9)
    handler = ScriptedHandler([{"grid": moved}], start)
    out = sandbox_mod.run_sandboxed_python(
        code=(
            "table = run_probe(['DOWN'])\n"
            "result = {'executed_count': table['executed_count'],\n"
            "          'first_action': table['steps'][0]['action'],\n"
            "          'lines': table['steps'][0]['change_lines'],\n"
            "          'spent': table['cost']['actions_spent_this_call']}\n"
        ),
        timeout_seconds=30,
        initial_state=_state(start),
        action_handler=handler,
    )
    assert not out.get("error"), out
    assert out["result"]["executed_count"] == 1
    assert out["result"]["first_action"] == "DOWN"
    assert out["result"]["spent"] == 1
    assert any("CHANGE" in line or "MOVED" in line for line in out["result"]["lines"])
    # the probe executed exactly one real action through the original handler
    assert handler.calls == [[{"action": "DOWN"}]]


def test_plain_action_calls_pass_through_untouched(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}])
    out = sandbox_mod.run_sandboxed_python(
        code="result = action(['UP'])['executed']\n",
        timeout_seconds=30,
        initial_state=_state(_grid()),
        action_handler=handler,
    )
    assert not out.get("error"), out
    assert out["result"] is True
    assert handler.calls == [[{"action": "UP"}]]


def test_disabled_run_probe_in_sandbox_executes_nothing(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
    _apply()
    handler = ScriptedHandler([{}])
    out = sandbox_mod.run_sandboxed_python(
        code="result = run_probe(['UP'])['error']\n",
        timeout_seconds=30,
        initial_state=_state(_grid()),
        action_handler=handler,
    )
    assert not out.get("error"), out
    assert "not enabled" in out["result"]
    assert handler.calls == []  # zero real actions when dormant


def test_bootstrap_injection_is_idempotent():
    from inference.agent import python_tool_sandbox as sandbox_mod

    _apply()
    _apply()
    assert sandbox_mod._SANDBOX_BOOTSTRAP.count('runtime_globals["run_probe"]') == 1


# ---------------------------------------------------------------------------------
# coexistence: patch 16's differ and patch 17's observer see probe actions
# ---------------------------------------------------------------------------------


class _ShiftGame:
    """Fake engine game: ACTION2 (DOWN) shifts the color-9 block down one row."""

    def __init__(self, board):
        import arcengine

        self._arcengine = arcengine
        self.number_of_levels = 3
        self.game_run = SimpleNamespace(history=[], state="playing")
        self._board = [list(r) for r in board]
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
        if action.id.name == "ACTION2":
            cells = [
                (y, x)
                for y, row in enumerate(self._board)
                for x, v in enumerate(row)
                if v == 9
            ]
            for y, x in cells:
                self._board[y][x] = 0
            for y, x in cells:
                if y + 1 < len(self._board):
                    self._board[y + 1][x] = 9
        self.current_state = self._state()
        return self.current_state


def _real_session(tmp_path):
    from inference.framework import solver as duck_solver

    board = _paint(_grid(0, 16), [(5, 3), (5, 4)], 9)
    game = _ShiftGame(board)
    solver = duck_solver.HarnessSolver(label="probe-test", model="stub", concurrency=1)
    solver.job_dir = None
    session = duck_solver._HarnessGameSession(
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
    return game, session


def test_probe_actions_are_seen_by_patch16_and_patch17(tmp_path, monkeypatch):
    from inference.framework import solver as duck_solver

    _enable(monkeypatch)
    monkeypatch.setenv("TAAF_DIFF_LINES", "1")
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    duck_patches.patch_diff_lines()
    duck_patches.patch_wiggle()
    _apply()
    game, session = _real_session(tmp_path)
    wiggle_state = duck_patches.WiggleState()
    wiggle_state.battery_done = True
    session._wiggle_state = wiggle_state
    duck_patches._WIGGLE_TLS.state = wiggle_state
    duck_patches._DIFF_TLS.entries = []

    def handler(actions):
        payload = session.step_env({"actions": [dict(a) for a in actions]})
        grid = [list(r) for r in duck_solver._grid_from_state(game.current_state)]
        return {"action_result": payload, "state": _state(grid, valid=["DOWN"])}

    try:
        out = probe_execute(["DOWN", "DOWN"], handler, _state(_grid(0, 16), valid=["DOWN"]))
        table = out["action_result"]
        assert table["executed_count"] == 2
        assert game.game_run.history == ["ACTION2", "ACTION2"]
        # patch 16's host-side differ banked both probe actions
        assert len(duck_patches._DIFF_TLS.entries) == 2
        # patch 17's standing observer folded both presses into its evidence
        assert len(wiggle_state.press_stats["ACTION2"]["moves"]) == 2
    finally:
        duck_patches._DIFF_TLS.entries = []
        duck_patches._WIGGLE_TLS.state = None


# ---------------------------------------------------------------------------------
# prompt advertisement + PROBE REPORT injection
# ---------------------------------------------------------------------------------


def test_system_prompt_advertises_contract_and_doctrine(monkeypatch):
    from inference.agent import tool_agent

    _enable(monkeypatch)
    _apply()
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
    assert ADVERT_MARKER in prompt
    assert DOCTRINE_SENTENCE in prompt  # PRO-LONG batching doctrine, verbatim
    assert "20 actions per call" in prompt and "3 probe calls per level" in prompt
    assert "RESET is never allowed" in prompt
    assert "REAL and scored" in prompt


def test_system_prompt_untouched_when_disabled(monkeypatch):
    from inference.agent import tool_agent

    monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
    _apply()
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
    assert ADVERT_MARKER not in prompt
    assert DOCTRINE_SENTENCE not in prompt


def _agent():
    from inference.agent.tool_agent import ToolAgent

    _apply()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/probe-test")
    return agent


def _build(agent):
    from inference.agent.runtime_state import Frame

    return agent._build_user_prompt(
        1,
        valid_actions=["ACTION1"],
        current_frame=Frame(grid=((1, 2), (3, 4)), step=1, level=1),
    )


def test_probe_report_injected_then_consumed(monkeypatch):
    _enable(monkeypatch)
    agent = _agent()
    handler = ScriptedHandler([{}, {"level_completed": True, "level": 2}])
    probe_execute(["UP", "UP", "UP"], handler, handler.state())
    prompt = _build(agent)
    assert REPORT_MARKER in prompt
    assert prompt.index(REPORT_MARKER) < prompt.index("Current state:")
    assert "stopped: level_completed" in prompt
    assert "probe cost: 2 scored actions this call" in prompt
    # consumed: the report does not repeat on the following turn
    assert REPORT_MARKER not in _build(agent)


def test_probe_report_renders_refusals(monkeypatch):
    _enable(monkeypatch)
    handler = ScriptedHandler([])
    probe_execute(["RESET"], handler, handler.state())
    block = _probe_prompt_block()
    assert block.startswith(REPORT_MARKER)
    assert "probe call refused" in block and "RESET" in block


def test_no_report_when_disabled_even_with_banked_tables(monkeypatch):
    monkeypatch.delenv("TAAF_RUN_PROBE", raising=False)
    agent = _agent()
    _probe_tls_state()["reports"].append(
        {"executed": True, "requested": 1, "executed_count": 1, "steps": [], "cost": {}}
    )
    assert REPORT_MARKER not in _build(agent)


# ---------------------------------------------------------------------------------
# lifecycle: per-game reset, idempotency, hostile bundle
# ---------------------------------------------------------------------------------


def test_play_resets_probe_budgets(monkeypatch):
    from inference.framework import solver as duck_solver

    _enable(monkeypatch)
    _apply()
    assert getattr(
        duck_solver._HarnessGameSession.play, "_run_probe_play_patched", False
    ), "patch18's play wrapper (or its forwarded marker) is missing"
    tls = _probe_tls_state()
    tls["actions_total"] = 7
    tls["calls_by_level"][1] = 3
    # patch 18's play wrapper clears the TLS BEFORE delegating; a hostile stub
    # makes every downstream wrapper raise immediately after the reset ran.
    try:
        duck_solver._HarnessGameSession.play(SimpleNamespace())
    except Exception:
        pass
    fresh = _probe_tls_state()
    assert fresh["actions_total"] == 0 and fresh["calls_by_level"] == {}


def test_patch_applies_and_is_idempotent():
    first = _apply()
    assert first.startswith("patch18 run-probe: OK") or "SKIP" in first
    assert duck_patches.patch_run_probe() == "patch18 run-probe: SKIP (already applied)"


def test_patch_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "_SANDBOX_BOOTSTRAP", 42)
    msg = duck_patches.patch_run_probe()
    assert msg.startswith("patch18 run-probe: FAIL"), msg


def test_off_by_default_via_env_contract():
    import os

    assert os.environ.get("TAAF_RUN_PROBE") is None or True  # doc anchor
    assert duck_patches._probe_enabled() in (True, False)
    # the default, with the variable unset, must be OFF
    saved = os.environ.pop("TAAF_RUN_PROBE", None)
    try:
        assert duck_patches._probe_enabled() is False
    finally:
        if saved is not None:
            os.environ["TAAF_RUN_PROBE"] = saved


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE staleness gate (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------

SCORED_INFERENCE = SCORED_REF / "src/ARC3-Inference"


def test_scored_bundle_has_every_symbol_patch18_touches():
    sandbox_src = (SCORED_INFERENCE / "inference/agent/python_tool_sandbox.py").read_text()
    assert 'runtime_globals["action"] = action' in sandbox_src, (
        "scored bundle lost the action-registration anchor — run_probe would not ship"
    )
    assert "def run_sandboxed_python" in sandbox_src
    assert "action_handler" in sandbox_src
    agent_src = (SCORED_INFERENCE / "inference/agent/tool_agent.py").read_text()
    assert "def _build_system_prompt" in agent_src
    assert "def _build_user_prompt" in agent_src
    assert "def _handle_action" in agent_src
    names_src = (SCORED_INFERENCE / "inference/agent/action_names.py").read_text()
    assert "def to_engine_action" in names_src
    solver_src = (SCORED_INFERENCE / "inference/framework/solver.py").read_text()
    assert "class _HarnessGameSession" in solver_src
    assert "def play" in solver_src


@pytest.mark.skipif(not SCORED_INFERENCE.is_dir(), reason="scored bundle bytes absent")
def test_patch18_applies_on_scored_bundle_bytes():
    """Apply patch 18 on the scored bytes in a clean subprocess: enabled, the
    advertisement and sandbox global must land; disabled, prompts are untouched."""
    script = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(SCORED_INFERENCE)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

r = duck_patches.patch_run_probe()
assert r.startswith("patch18 run-probe: OK"), r
from inference.agent import python_tool_sandbox as sandbox_mod
assert 'runtime_globals["run_probe"]' in sandbox_mod._SANDBOX_BOOTSTRAP

from inference.agent import tool_agent

# disabled (default): the scored-bundle prompts must be untouched
assert "run_probe" not in tool_agent._build_system_prompt(tool_output_tokens=1000)

os.environ["TAAF_RUN_PROBE"] = "1"
prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
assert "run_probe" in prompt, prompt[-500:]
assert {DOCTRINE_SENTENCE!r} in prompt

from inference.agent.tool_agent import ToolAgent
from inference.agent.runtime_state import Frame
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/probe-scored")
handler_calls = []
out = duck_patches.probe_execute(
    ["UP"],
    lambda actions: handler_calls.append(actions) or {{
        "action_result": {{"executed": True, "level": 1, "action_display": "UP"}},
        "state": {{"current_frame": {{"grid": [[0]], "level": 1}}, "valid_actions": ["UP"]}},
    }},
    {{"current_frame": {{"grid": [[0]], "level": 1}}, "valid_actions": ["UP"]}},
)
assert out["action_result"]["executed_count"] == 1, out
user_prompt = agent._build_user_prompt(
    1, valid_actions=["ACTION1"], current_frame=Frame(grid=((1, 2),), step=1, level=1)
)
assert "PROBE REPORT" in user_prompt
print("SCORED-PROBE-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-PROBE-OK" in proc.stdout
