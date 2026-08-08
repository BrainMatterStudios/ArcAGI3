"""Tests for patches 21+22 (structural action channel + enforced brake/phase gates).

The 2026-08 package screen measured ADOPTION as the binding failure: the 27B
ignored every advertised mechanism (run_probe: 1 call in 28 games) but fully
engaged the one enforced mechanism (wiggle, 28/28). Patch 21 makes the
PRO-LONG plan contract STRUCTURAL: every plain sandbox `action()` call is a
plan submission of 1-20 actions routed through a per-step executor (lenient
whitelist parsing, RESET dedupe, score-change flush, per-step HUD-masked
change lines, PLAN REPORT + retry nudge next prompt, contract in the TOOL
SCHEMA / globals sentence / system prompt). Patch 22 enforces the A-not-B
brake ((masked-state-hash, action) pairs at >= 3 consecutive null effects are
stripped from menu and plans; hard only on HUD-mask-gated games) and the
human-budget phase gate (SCOUT plans capped at 5 until first progress or the
~20-click/~31-avatar budget is spent, then COMMIT).

Layers: executor semantics on scripted handlers, the adoption metric itself
(scripted-model e2e through the REAL prompts + REAL sandbox subprocess:
actions-per-turn > 3 when STRUCT=1 vs 1 baseline), brake and phase behavior,
prompt/schema surfaces, one-report-one-place (patch 16 differ suppression),
coexistence with patches 16-20 through the REAL patched session stack, the
2026-08-09 wiggle verdict repairs (CURSOR demotion, HERD, terrain-morph
rescue), OFF-by-default no-op, idempotency, hostile-bundle decline, and the
scored-bundle staleness gate.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_struct.py -v
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
SCORED_REF = REPO / "scratchpad/taaf_scored_ref"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import (  # noqa: E402
    _STRUCT_CHANNEL,
    _STRUCT_GATES,
    _STRUCT_TLS,
    _WIGGLE_TLS,
    STRUCT_DIAGNOSTICS,
    WiggleState,
    _struct_prompt_block,
    _struct_tls_state,
    plan_execute,
    wiggle_observe_transition,
    wiggle_verdict,
)

CONTRACT_MARKER = "PLAN CONTRACT"
REPORT_MARKER = "PLAN REPORT"
GATES_MARKER = "ENFORCED GATES"
NUDGE_MARKER = "did not produce a valid plan"
PROBE_ADVERT_MARKER = "run_probe — probe battery"


def _apply() -> None:
    r18 = duck_patches.patch_run_probe()
    assert "OK" in r18 or "SKIP" in r18, r18
    r21 = duck_patches.patch_struct_channel()
    assert "OK" in r21 or "SKIP" in r21, r21
    r22 = duck_patches.patch_struct_gates()
    assert "OK" in r22 or "SKIP" in r22, r22


def _enable(monkeypatch, **env):
    monkeypatch.setenv("TAAF_STRUCT", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))


@pytest.fixture(autouse=True)
def _fresh_struct_state():
    _STRUCT_TLS.state = None
    _STRUCT_TLS.in_plan = False
    for key in STRUCT_DIAGNOSTICS:
        STRUCT_DIAGNOSTICS[key] = {} if isinstance(STRUCT_DIAGNOSTICS[key], dict) else 0
    yield
    _STRUCT_TLS.state = None
    _STRUCT_TLS.in_plan = False
    _WIGGLE_TLS.state = None


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

    def __init__(self, outcomes, start_grid=None, level=1, valid=None, score=0):
        self.outcomes = list(outcomes)
        self.calls: list[list[dict]] = []
        self.grid = [list(r) for r in (start_grid or _grid())]
        self.level = level
        self.score = score
        self.valid = list(valid) if valid is not None else None
        self.step = 0

    def state(self):
        valid = self.valid if self.valid is not None else ["UP", "DOWN", "MOUSE"]
        return _state(self.grid, level=self.level, valid=valid, step=self.step)

    def __call__(self, actions):
        self.calls.append([dict(a) for a in actions])
        assert len(actions) == 1, "the plan executor must execute ONE action per call"
        self.step += 1
        outcome = dict(self.outcomes.pop(0)) if self.outcomes else {}
        if "grid" in outcome:
            self.grid = outcome.pop("grid")
        if "level" in outcome:
            self.level = outcome["level"]
        if "score" in outcome:
            self.score = outcome["score"]
        result = {
            "executed": True,
            "action_num": self.step,
            "level": self.level,
            "score": self.score,
            "reward": 0.0,
            "board_changed": True,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "action_display": str(actions[0].get("action", "?")),
        }
        result.update(outcome)
        return {"action_result": result, "state": self.state()}


class BatchHandler:
    """Native-shaped handler: accepts whole lists (the non-struct action path)."""

    def __init__(self):
        self.calls: list[list[dict]] = []
        self.grid = _grid()

    def state(self):
        return _state(self.grid, valid=["UP", "DOWN", "MOUSE"])

    def __call__(self, actions):
        self.calls.append([dict(a) for a in actions])
        return {
            "action_result": {
                "executed": True,
                "level": 1,
                "score": 0,
                "executed_count": len(actions),
                "action_display": str(actions[-1].get("action", "?")),
            },
            "state": self.state(),
        }


# ---------------------------------------------------------------------------------
# executor: plan semantics
# ---------------------------------------------------------------------------------


def test_multi_action_plan_executes_sequentially_with_change_lines(monkeypatch):
    _enable(monkeypatch)
    _apply()
    start = _paint(_grid(), [(2, 2), (2, 3)], 9)
    moved = _paint(_grid(), [(3, 2), (3, 3)], 9)
    handler = ScriptedHandler([{"grid": moved}, {"board_changed": False}], start)
    out = plan_execute(["DOWN", "DOWN"], handler, _state(start))
    merged = out["action_result"]
    assert merged["plan"] is True and merged["executed"] is True
    assert merged["requested_count"] == 2 and merged["executed_count"] == 2
    assert merged["level"] == 1 and merged["score"] == 0  # native keys survive
    assert merged["executed_actions"] == ["DOWN", "DOWN"]
    step1, step2 = merged["steps"]
    assert any("MOVED" in line or "CHANGE" in line for line in step1["change_lines"])
    assert step2["change_lines"] == ["NO CHANGE (board identical)"]
    assert [len(c) for c in handler.calls] == [1, 1]  # one real action per call
    assert json.dumps(merged)  # the whole result must be JSON-able
    assert out["state"]["current_frame"]["grid"] == moved


def test_single_action_auto_wraps_to_1_plan(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}])
    out = plan_execute(["UP"], handler, handler.state())
    assert out["action_result"]["executed_count"] == 1
    assert STRUCT_DIAGNOSTICS["wrapped_singles"] == 1
    assert STRUCT_DIAGNOSTICS["plan_lengths"] == {1: 1}


def test_lenient_parser_drops_invalid_entries_never_rejects(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}, {}])
    out = plan_execute(
        ["UP", "FLY_TO_MOON", {"action": "MOUSE"}, "DOWN"], handler, handler.state()
    )
    merged = out["action_result"]
    assert merged["executed_count"] == 2  # UP and DOWN ran; junk dropped silently
    assert STRUCT_DIAGNOSTICS["invalid_dropped"] == 2
    assert any("invalid" in note for note in merged["notes"])


def test_plan_with_no_valid_entries_reports_without_executing(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}])
    out = plan_execute(["NOPE", {"action": "MOUSE"}], handler, handler.state())
    merged = out["action_result"]
    assert merged["executed"] is False and "no valid actions" in merged["error"]
    assert handler.calls == []
    # an all-invalid submission does NOT count as a plan for nudge purposes
    assert _struct_tls_state()["plan_seen_since_prompt"] is None


def test_consecutive_reset_dedupe_within_and_across_plans(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{} for _ in range(6)])
    out = plan_execute(["RESET", "RESET", "UP"], handler, handler.state())
    assert out["action_result"]["executed_count"] == 2  # one RESET + UP
    assert STRUCT_DIAGNOSTICS["reset_deduped"] == 1
    # cross-call: a plan that ENDED on RESET dedupes a leading RESET next call
    plan_execute(["RESET"], handler, handler.state())
    out3 = plan_execute(["RESET", "UP"], handler, handler.state())
    assert out3["action_result"]["executed_actions"] == ["UP"]
    assert STRUCT_DIAGNOSTICS["reset_deduped"] == 2


def test_cap_slices_to_20_and_is_env_tunable(monkeypatch):
    _enable(monkeypatch)
    _apply()
    # force COMMIT so the scout cap does not mask the hard cap
    tls = _struct_tls_state()
    tls["phase"].update({"level": 1, "phase": "COMMIT", "reason": "test"})
    handler = ScriptedHandler([{} for _ in range(30)])
    out = plan_execute(["UP"] * 30, handler, handler.state())
    merged = out["action_result"]
    assert merged["executed_count"] == 20 and merged["requested_count"] == 30
    assert any("cap" in note for note in merged["notes"])
    assert STRUCT_DIAGNOSTICS["cap_truncations"] == 1

    monkeypatch.setenv("TAAF_STRUCT_MAX_PLAN", "3")
    handler2 = ScriptedHandler([{} for _ in range(5)])
    tls["phase"].update({"phase": "COMMIT"})
    out2 = plan_execute(["UP"] * 5, handler2, handler2.state())
    assert out2["action_result"]["executed_count"] == 3


def test_per_step_availability_check_stops_the_plan(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}], valid=["MOUSE"])  # arrows not offered
    out = plan_execute(["UP", "UP"], handler, handler.state())
    merged = out["action_result"]
    assert merged["executed_count"] == 0
    assert "not in valid_actions" in merged["stop_reason"]
    assert handler.calls == []


def test_unexecuted_step_result_stops_the_plan(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}, {"executed": False, "error": "not valid right now"}])
    out = plan_execute(["UP", "DOWN", "UP"], handler, handler.state())
    merged = out["action_result"]
    assert merged["executed_count"] == 1
    assert merged["steps"][1]["executed"] is False
    assert len(handler.calls) == 2


def test_early_stop_on_terminal_flags(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}, {"level_completed": True, "level": 2}, {}])
    out = plan_execute(["UP"] * 5, handler, handler.state())
    merged = out["action_result"]
    assert merged["executed_count"] == 2 and merged["stop_reason"] == "level_completed"
    assert merged["steps"][1]["change_lines"] == [
        "LEVEL COMPLETED — scene repainted, per-object diff skipped"
    ]
    handler2 = ScriptedHandler([{"game_over": True}])
    out2 = plan_execute(["UP", "UP"], handler2, handler2.state())
    assert out2["action_result"]["stop_reason"] == "game_over"


def test_score_change_flush_is_absolute(monkeypatch):
    """Any score change flushes the remaining plan — not just terminal flags."""
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{}, {"score": 1}, {}, {}])
    out = plan_execute(["UP", "UP", "UP", "UP"], handler, handler.state())
    merged = out["action_result"]
    assert merged["executed_count"] == 2
    assert merged["stop_reason"] == "score_changed"
    assert STRUCT_DIAGNOSTICS["score_flushes"] == 1
    assert len(handler.calls) == 2


# ---------------------------------------------------------------------------------
# patch 22: phase gate
# ---------------------------------------------------------------------------------


def test_scout_phase_caps_plans_at_5(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{} for _ in range(10)])
    out = plan_execute(["UP"] * 9, handler, handler.state())
    merged = out["action_result"]
    assert merged["executed_count"] == 5
    assert merged["phase"] == "SCOUT"
    assert any("SCOUT" in note for note in merged["notes"])
    assert STRUCT_DIAGNOSTICS["scout_truncations"] == 1


def test_commit_unlocks_after_first_progress(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler([{"reward": 0.2}] + [{} for _ in range(20)])
    plan_execute(["UP"], handler, handler.state())  # progress on step 1
    assert _struct_tls_state()["phase"]["phase"] == "COMMIT"
    assert STRUCT_DIAGNOSTICS["phase_transitions"] == 1
    out = plan_execute(["UP"] * 9, handler, handler.state())
    assert out["action_result"]["executed_count"] == 9  # cap lifted
    assert STRUCT_DIAGNOSTICS["scout_truncations"] == 0


def test_commit_unlocks_after_budget_exhaustion(monkeypatch):
    _enable(monkeypatch, TAAF_STRUCT_SCOUT_CLICK=3)
    _apply()
    # click-mode fallback: no directionals in the menu
    handler = ScriptedHandler([{} for _ in range(10)], valid=["MOUSE"])
    specs = [{"action": "MOUSE", "row": 1, "col": c} for c in range(3)]
    plan_execute(specs, handler, handler.state())
    ph = _struct_tls_state()["phase"]
    assert ph["phase"] == "COMMIT" and "budget" in ph["reason"]


def test_level_change_resets_phase_to_scout(monkeypatch):
    _enable(monkeypatch)
    _apply()
    handler = ScriptedHandler(
        [{"reward": 0.2}, {}, {"level": 2}, {}]
    )
    plan_execute(["UP"], handler, handler.state())  # -> COMMIT on level 1
    assert _struct_tls_state()["phase"]["phase"] == "COMMIT"
    plan_execute(["UP"], handler, handler.state())
    plan_execute(["UP"], handler, handler.state())  # lands on level 2
    ph = _struct_tls_state()["phase"]
    assert ph["level"] == 2 and ph["phase"] == "SCOUT"
    assert any("SCOUT reopened" in t for t in ph["transitions"])


def test_archetype_budget_wired_from_wiggle_game_mode(monkeypatch):
    _enable(monkeypatch)
    _apply()
    ws = WiggleState()
    ws.mode = "AVATAR"
    _WIGGLE_TLS.state = ws
    budget, mode = duck_patches._struct_scout_budget(["UP", "DOWN"])
    assert (budget, mode) == (31, "AVATAR")
    ws.mode = "CLICK"
    assert duck_patches._struct_scout_budget([])[0] == 20
    ws.mode = "CURSOR"  # selectors take the click budget
    assert duck_patches._struct_scout_budget([])[0] == 20
    ws.mode = "HERD"
    assert duck_patches._struct_scout_budget([])[0] == 31
    _WIGGLE_TLS.state = None
    # fallback without a battery: menu shape decides
    assert duck_patches._struct_scout_budget(["MOUSE"]) == (20, "CLICK")
    assert duck_patches._struct_scout_budget(["UP", "MOUSE"])[0] == 31


# ---------------------------------------------------------------------------------
# patch 22: A-not-B brake
# ---------------------------------------------------------------------------------


def _null_press(handler_grid):
    """Handler outcome for a masked-null press: board identical."""
    return {"board_changed": False, "grid": [list(r) for r in handler_grid]}


def test_brake_strips_pair_after_3_consecutive_nulls_hard(monkeypatch):
    _enable(monkeypatch)
    _apply()
    monkeypatch.setattr(duck_patches, "_hud_current_mask_cells", lambda: [[0, 7]])
    grid = _paint(_grid(), [(4, 4)], 9)
    handler = ScriptedHandler([_null_press(grid) for _ in range(10)], grid)
    for _ in range(3):
        plan_execute(["UP"], handler, _state(grid))
    assert len(handler.calls) == 3
    out = plan_execute(["UP", "DOWN"], handler, _state(grid))
    merged = out["action_result"]
    assert merged["steps"][0].get("stripped") is True
    assert "A-not-B brake" in merged["steps"][0]["change_lines"][0]
    assert len(handler.calls) == 4  # only DOWN executed on the 4th plan
    assert handler.calls[-1][0]["action"] == "DOWN"
    assert STRUCT_DIAGNOSTICS["brake_strips"] == 1


def test_brake_is_advisory_without_hud_mask(monkeypatch):
    _enable(monkeypatch)
    _apply()
    monkeypatch.setattr(duck_patches, "_hud_current_mask_cells", lambda: [])
    grid = _paint(_grid(), [(4, 4)], 9)
    handler = ScriptedHandler([_null_press(grid) for _ in range(10)], grid)
    for _ in range(3):
        plan_execute(["UP"], handler, _state(grid))
    out = plan_execute(["UP"], handler, _state(grid))
    step = out["action_result"]["steps"][0]
    assert step["executed"] is True  # soft: executed anyway
    assert any("BRAKE (advisory)" in line for line in step["change_lines"])
    assert STRUCT_DIAGNOSTICS["brake_strips"] == 0
    assert STRUCT_DIAGNOSTICS["brake_soft_flags"] == 1


def test_brake_counter_resets_on_a_real_effect(monkeypatch):
    _enable(monkeypatch)
    _apply()
    monkeypatch.setattr(duck_patches, "_hud_current_mask_cells", lambda: [[0, 7]])
    grid = _paint(_grid(), [(4, 4)], 9)
    moved = _paint(_grid(), [(3, 4)], 9)
    handler = ScriptedHandler(
        [_null_press(grid), _null_press(grid), {"grid": moved}]
        + [_null_press(moved) for _ in range(5)],
        grid,
    )
    for _ in range(3):
        plan_execute(["UP"], handler, _state(grid))
    tls = _struct_tls_state()
    h = duck_patches._struct_masked_hash(grid)
    assert (h, "UP") not in tls["brake"]  # the effect wiped the pair


def test_mouse_brake_is_per_coordinate(monkeypatch):
    _enable(monkeypatch)
    _apply()
    monkeypatch.setattr(duck_patches, "_hud_current_mask_cells", lambda: [[0, 7]])
    grid = _paint(_grid(), [(4, 4)], 9)
    handler = ScriptedHandler([_null_press(grid) for _ in range(10)], grid)
    dead = {"action": "MOUSE", "row": 2, "col": 2}
    for _ in range(3):
        plan_execute([dead], handler, _state(grid))
    out = plan_execute([dead, {"action": "MOUSE", "row": 5, "col": 5}], handler, _state(grid))
    steps = out["action_result"]["steps"]
    assert steps[0].get("stripped") is True  # (2,2) braked
    assert steps[1]["executed"] is True  # (5,5) untouched


def test_menu_strip_in_prompt_hard_and_never_to_empty(monkeypatch):
    from inference.agent.runtime_state import Frame

    _enable(monkeypatch)
    _apply()
    monkeypatch.setattr(duck_patches, "_hud_current_mask_cells", lambda: [[0, 1]])
    frame_grid = ((1, 2), (3, 4))
    h = duck_patches._struct_masked_hash(frame_grid)
    tls = _struct_tls_state()
    tls["brake"][(h, "UP")] = 3
    agent = _agent()
    prompt = agent._build_user_prompt(
        1,
        valid_actions=["UP", "DOWN"],
        current_frame=Frame(grid=frame_grid, step=1, level=1),
    )
    valid_line = next(l for l in prompt.splitlines() if "Valid actions right now" in l)
    assert "UP" not in valid_line.replace("UP,", "").replace(": UP", "") or "UP" not in valid_line
    assert "A-NOT-B BRAKE: UP stripped from the menu" in prompt
    assert STRUCT_DIAGNOSTICS["menu_strips"] == 1
    # never strip the menu to empty: UP alone stays, advisory line instead
    tls2 = _struct_tls_state()
    tls2["brake"][(h, "UP")] = 3
    prompt2 = agent._build_user_prompt(
        1,
        valid_actions=["UP"],
        current_frame=Frame(grid=frame_grid, step=1, level=1),
    )
    line2 = next(l for l in prompt2.splitlines() if "Valid actions right now" in l)
    assert "UP" in line2
    assert "BRAKE (advisory): UP" in prompt2


def test_phase_line_reported_in_every_prompt(monkeypatch):
    _enable(monkeypatch)
    _apply()
    agent = _agent()
    prompt = _build(agent)
    assert "PHASE: SCOUT" in prompt
    tls = _struct_tls_state()
    tls["phase"].update({"level": 3, "phase": "COMMIT", "reason": "first level progress"})
    prompt2 = _build(agent)
    assert "PHASE: COMMIT on level 3" in prompt2


# ---------------------------------------------------------------------------------
# prompt surfaces: report, nudge, contract, schema, framing
# ---------------------------------------------------------------------------------


def _agent():
    from inference.agent.tool_agent import ToolAgent

    _apply()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/struct-test")
    return agent


def _build(agent):
    from inference.agent.runtime_state import Frame

    return agent._build_user_prompt(
        1,
        valid_actions=["UP", "DOWN"],
        current_frame=Frame(grid=((1, 2), (3, 4)), step=1, level=1),
    )


def test_plan_report_injected_then_consumed(monkeypatch):
    _enable(monkeypatch)
    agent = _agent()
    handler = ScriptedHandler([{}, {"level_completed": True, "level": 2}])
    plan_execute(["UP", "UP", "UP"], handler, handler.state())
    prompt = _build(agent)
    assert REPORT_MARKER in prompt
    assert prompt.index(REPORT_MARKER) < prompt.index("Current state:")
    assert "stopped: level_completed" in prompt
    assert STRUCT_DIAGNOSTICS["reports_injected"] == 1
    follow_up = _build(agent)
    assert REPORT_MARKER not in follow_up  # consumed


def test_retry_nudge_appears_after_planless_turn_and_caps(monkeypatch):
    _enable(monkeypatch, TAAF_STRUCT_NUDGE_MAX=2)
    agent = _agent()
    assert NUDGE_MARKER not in _build(agent)  # first prompt: never nudged
    assert NUDGE_MARKER in _build(agent)  # no plan since -> nudge 1
    assert NUDGE_MARKER in _build(agent)  # nudge 2
    assert NUDGE_MARKER not in _build(agent)  # capped
    assert STRUCT_DIAGNOSTICS["nudges"] == 2
    assert STRUCT_DIAGNOSTICS["turns_without_plan"] == 3
    # a plan resets the streak
    handler = ScriptedHandler([{}])
    plan_execute(["UP"], handler, handler.state())
    assert NUDGE_MARKER not in _build(agent)
    assert NUDGE_MARKER in _build(agent)  # nudging works again


def test_system_prompt_carries_contract_and_replaces_framing(monkeypatch):
    from inference.agent import tool_agent

    _enable(monkeypatch)
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    _apply()
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
    assert CONTRACT_MARKER in prompt
    assert GATES_MARKER in prompt
    assert "short plans (1-5) to test hypotheses" in prompt.lower() or "1-5" in prompt
    # the single-step framing is REPLACED, not supplemented
    assert duck_patches._STRUCT_CYCLE_OLD not in prompt
    assert "observe-deliberate-plan cycle" in prompt
    # run_probe's advertisement is withdrawn under STRUCT even when enabled
    assert PROBE_ADVERT_MARKER not in prompt


def test_tool_schema_carries_the_contract(monkeypatch, tmp_path):
    _enable(monkeypatch)
    agent = _agent()
    tools = agent._tools(tmp_path / "runtime_state.json")
    desc = next(
        t["function"]["description"] for t in tools if t["function"]["name"] == "python"
    )
    assert "ordered plan of 1-20" in desc
    assert duck_patches._STRUCT_TOOL_DESC_OLD not in desc


def test_user_prompt_globals_sentence_rewritten(monkeypatch):
    _enable(monkeypatch)
    agent = _agent()
    prompt = _build(agent)
    assert duck_patches._STRUCT_USER_LINE_NEW in prompt
    assert duck_patches._STRUCT_USER_LINE_OLD not in prompt


def test_all_surfaces_untouched_when_disabled(monkeypatch, tmp_path):
    from inference.agent import tool_agent

    monkeypatch.delenv("TAAF_STRUCT", raising=False)
    agent = _agent()
    system_prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
    assert CONTRACT_MARKER not in system_prompt
    assert GATES_MARKER not in system_prompt
    assert duck_patches._STRUCT_CYCLE_OLD in system_prompt  # framing intact
    user_prompt = _build(agent)
    assert REPORT_MARKER not in user_prompt
    assert "PHASE:" not in user_prompt
    assert duck_patches._STRUCT_USER_LINE_OLD in user_prompt
    tools = agent._tools(tmp_path / "runtime_state.json")
    desc = next(
        t["function"]["description"] for t in tools if t["function"]["name"] == "python"
    )
    assert duck_patches._STRUCT_TOOL_DESC_OLD in desc


# ---------------------------------------------------------------------------------
# one report, one place: patch 16's differ must skip plan-executed steps
# ---------------------------------------------------------------------------------


def test_patch16_differ_skips_plan_steps(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("TAAF_DIFF_LINES", "1")
    _apply()
    duck_patches._DIFF_TLS.entries = []
    action = SimpleNamespace(id=SimpleNamespace(name="ACTION1"))
    payload = {"executed": True, "action_display": "UP", "action_num": 1}
    pre, post = _grid(), _paint(_grid(), [(1, 1)], 5)
    _STRUCT_TLS.in_plan = True
    try:
        duck_patches._diff_record_action(action, payload, pre, post)
        assert duck_patches._DIFF_TLS.entries == []  # suppressed inside a plan
    finally:
        _STRUCT_TLS.in_plan = False
    duck_patches._diff_record_action(action, payload, pre, post)
    assert len(duck_patches._DIFF_TLS.entries) == 1  # normal path still banks
    duck_patches._DIFF_TLS.entries = []


# ---------------------------------------------------------------------------------
# sandbox integration: the channel through the REAL subprocess
# ---------------------------------------------------------------------------------


def test_action_call_routes_through_plan_channel_in_real_sandbox(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    _enable(monkeypatch)
    _apply()
    start = _paint(_grid(), [(2, 2)], 9)
    moved = _paint(_grid(), [(3, 2)], 9)
    handler = ScriptedHandler([{"grid": moved}, {"board_changed": False}], start)
    out = sandbox_mod.run_sandboxed_python(
        code=(
            "r = action(['DOWN', 'DOWN'])\n"
            "result = {'plan': r.get('plan'), 'n': r['executed_count'],\n"
            "          'lines': r['steps'][0]['change_lines']}\n"
        ),
        timeout_seconds=30,
        initial_state=_state(start),
        action_handler=handler,
    )
    assert not out.get("error"), out
    assert out["result"]["plan"] is True and out["result"]["n"] == 2
    assert any("CHANGE" in l or "MOVED" in l for l in out["result"]["lines"])
    assert [len(c) for c in handler.calls] == [1, 1]


def test_run_probe_sentinel_still_routes_to_probe_under_struct(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    _enable(monkeypatch)
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    _apply()
    duck_patches._PROBE_TLS.state = None
    handler = ScriptedHandler([{}])
    out = sandbox_mod.run_sandboxed_python(
        code="result = run_probe(['UP'])['probe']\n",
        timeout_seconds=30,
        initial_state=_state(_grid()),
        action_handler=handler,
    )
    assert not out.get("error"), out
    assert out["result"] is True  # probe table, not a plan table
    assert STRUCT_DIAGNOSTICS["plans"] == 0  # the sentinel call is NOT a plan
    duck_patches._PROBE_TLS.state = None


def test_off_by_default_native_batch_passes_through_untouched(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.delenv("TAAF_STRUCT", raising=False)
    _apply()
    handler = BatchHandler()
    out = sandbox_mod.run_sandboxed_python(
        code="r = action(['UP', 'DOWN'])\nresult = r.get('plan') is None\n",
        timeout_seconds=30,
        initial_state=_state(_grid()),
        action_handler=handler,
    )
    assert not out.get("error"), out
    assert out["result"] is True
    # the native path receives the WHOLE batch in one handler call
    assert [len(c) for c in handler.calls] == [2]
    assert STRUCT_DIAGNOSTICS["plans"] == 0


# ---------------------------------------------------------------------------------
# THE ADOPTION METRIC: scripted-model e2e, actions-per-turn > 3 vs ~1 baseline
# ---------------------------------------------------------------------------------


def _scripted_model_turns(n_turns: int) -> float:
    """A scripted 27B stand-in: it reads the REAL prompts (system + user) and
    obeys whatever action contract they state — a plan when the plan contract
    is present, one action otherwise. Each turn runs through the REAL sandbox
    subprocess and the REAL (patched) interception seams. Returns realized
    actions per turn."""
    from inference.agent import python_tool_sandbox as sandbox_mod
    from inference.agent import tool_agent

    agent = _agent()
    total_actions = 0
    for _ in range(n_turns):
        system_prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
        user_prompt = _build(agent)
        assert "python" in user_prompt  # the model reads both surfaces
        if CONTRACT_MARKER in system_prompt:
            code = "result = action(['DOWN', 'UP', 'DOWN', 'UP', 'DOWN'])"
        else:
            code = "result = action(['DOWN'])"
        handler = ScriptedHandler([{} for _ in range(8)])
        out = sandbox_mod.run_sandboxed_python(
            code=code,
            timeout_seconds=30,
            initial_state=handler.state(),
            action_handler=(
                handler if CONTRACT_MARKER in system_prompt else BatchHandler()
            ),
        )
        assert not out.get("error"), out
        if CONTRACT_MARKER in system_prompt:
            total_actions += sum(len(c) for c in handler.calls)
        else:
            total_actions += 1
    return total_actions / n_turns


def test_adoption_metric_scripted_model_e2e(monkeypatch):
    monkeypatch.delenv("TAAF_STRUCT", raising=False)
    baseline = _scripted_model_turns(3)
    assert baseline == 1.0, baseline
    _STRUCT_TLS.state = None
    monkeypatch.setenv("TAAF_STRUCT", "1")
    structural = _scripted_model_turns(3)
    assert structural > 3.0, (
        f"adoption bar missed: {structural} actions/turn with STRUCT=1 "
        f"(baseline {baseline})"
    )
    assert STRUCT_DIAGNOSTICS["plans"] == 3
    assert STRUCT_DIAGNOSTICS["plan_lengths"] == {5: 3}


# ---------------------------------------------------------------------------------
# coexistence with patches 16-20 through the REAL patched session stack
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
    solver = duck_solver.HarnessSolver(label="struct-test", model="stub", concurrency=1)
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


def test_plan_steps_seen_by_patch17_observer_not_double_diffed(tmp_path, monkeypatch):
    from inference.framework import solver as duck_solver

    _enable(monkeypatch)
    monkeypatch.setenv("TAAF_DIFF_LINES", "1")
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    duck_patches.patch_diff_lines()
    duck_patches.patch_wiggle()
    _apply()
    game, session = _real_session(tmp_path)
    wiggle_state = WiggleState()
    wiggle_state.battery_done = True
    session._wiggle_state = wiggle_state
    _WIGGLE_TLS.state = wiggle_state
    duck_patches._DIFF_TLS.entries = []

    def handler(actions):
        payload = session.step_env({"actions": [dict(a) for a in actions]})
        grid = [list(r) for r in duck_solver._grid_from_state(game.current_state)]
        return {"action_result": payload, "state": _state(grid, valid=["DOWN"])}

    try:
        start = _paint(_grid(0, 16), [(5, 3), (5, 4)], 9)  # the game's real board
        out = plan_execute(["DOWN", "DOWN"], handler, _state(start, valid=["DOWN"]))
        merged = out["action_result"]
        assert merged["executed_count"] == 2
        assert game.game_run.history == ["ACTION2", "ACTION2"]
        # patch 17's standing observer folded both plan steps into its evidence
        assert len(wiggle_state.press_stats["ACTION2"]["moves"]) == 2
        # patch 16's banker skipped them (ONE report, ONE place: the PLAN REPORT)
        assert duck_patches._DIFF_TLS.entries == []
        # ... and the PLAN REPORT itself carries the per-step observations
        assert any("MOVED" in l for l in merged["steps"][0]["change_lines"])
    finally:
        duck_patches._DIFF_TLS.entries = []
        _WIGGLE_TLS.state = None


def test_probe_and_plan_channels_coexist(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    _apply()
    duck_patches._PROBE_TLS.state = None
    handler = ScriptedHandler([{} for _ in range(4)])
    plan_execute(["UP"], handler, handler.state())
    duck_patches.probe_execute(["UP", "UP"], handler, handler.state())
    assert STRUCT_DIAGNOSTICS["plans"] == 1
    assert duck_patches.RUN_PROBE_DIAGNOSTICS["calls"] >= 1
    duck_patches._PROBE_TLS.state = None


# ---------------------------------------------------------------------------------
# 2026-08-09 wiggle verdict repairs (consumed by patch 22's budgets/scaffolds)
# ---------------------------------------------------------------------------------


def _board16():
    return _grid(0, 16)


def test_cursor_demotion_on_jump_displacement():
    state = WiggleState()
    state.directionals_at_start = ["ACTION4"]
    marker = [(4, 2), (4, 3)]
    pre = _paint(_grid(0, 64), marker, 3)
    mid = _paint(_grid(0, 64), [(y, x + 28) for y, x in marker], 3)
    post = _paint(_grid(0, 64), [(y, x + 56) for y, x in marker], 3)
    wiggle_observe_transition(state, "ACTION4", {}, pre, mid)
    wiggle_observe_transition(state, "ACTION4", {}, mid, post)
    mode, reason = wiggle_verdict(state)
    assert mode == "CURSOR" and "displacement" in reason


def test_cursor_demotion_on_single_direction_only_evidence():
    state = WiggleState()
    state.directionals_at_start = ["ACTION1", "ACTION2"]
    board = _paint(_board16(), [(5, 3), (5, 4)], 9)
    moved = _paint(_board16(), [(6, 3), (6, 4)], 9)
    # ACTION2 moves it twice; ACTION1 pressed twice with NO effect
    wiggle_observe_transition(state, "ACTION2", {}, board, moved)
    wiggle_observe_transition(state, "ACTION1", {}, moved, moved)
    wiggle_observe_transition(state, "ACTION1", {}, moved, moved)
    board2 = _paint(_board16(), [(7, 3), (7, 4)], 9)
    wiggle_observe_transition(state, "ACTION2", {}, moved, board2)
    mode, reason = wiggle_verdict(state)
    assert mode == "CURSOR" and "single-direction" in reason


def test_no_demotion_while_other_directions_untested():
    """A mid-battery same-action lock stays AVATAR until others were tried."""
    state = WiggleState()
    state.directionals_at_start = ["ACTION2"]
    board = _paint(_board16(), [(5, 3), (5, 4)], 9)
    moved = _paint(_board16(), [(6, 3), (6, 4)], 9)
    moved2 = _paint(_board16(), [(7, 3), (7, 4)], 9)
    wiggle_observe_transition(state, "ACTION2", {}, board, moved)
    wiggle_observe_transition(state, "ACTION2", {}, moved, moved2)
    assert wiggle_verdict(state)[0] == "AVATAR"


def test_herd_verdict_on_coupled_motion():
    state = WiggleState()
    state.directionals_at_start = ["ACTION2"]
    flock = [(2, 2), (2, 3), (6, 8), (6, 9), (11, 12), (11, 13)]
    pre = _paint(_board16(), flock, 9)
    post = _paint(_board16(), [(y + 1, x) for y, x in flock], 9)
    post2 = _paint(_board16(), [(y + 2, x) for y, x in flock], 9)
    out = wiggle_observe_transition(state, "ACTION2", {}, pre, post)
    assert out["kind"] == "herd"
    wiggle_observe_transition(state, "ACTION2", {}, post, post2)
    mode, reason = wiggle_verdict(state)
    assert mode == "HERD" and "coupled" in reason.lower()


def test_terrain_morph_rescue_reads_as_avatar_move():
    """ka59/bp35 class: a conserved mover whose diff is fused with morphing
    terrain must be a MOVE (relaxed conserved-mover subtraction), and the
    verdict an AVATAR lock, not ARROW-MORPH."""
    state = WiggleState()
    state.directionals_at_start = ["ACTION4"]

    def scene(mover_col, terrain_color):
        board = _board16()
        board = _paint(board, [(5, c) for c in range(2, 12)], terrain_color)
        board = _paint(board, [(5, mover_col), (6, mover_col)], 9)
        return board

    # mover steps right while the terrain strip it touches recolors wholesale
    pre = scene(3, 4)
    post = scene(4, 6)
    out = wiggle_observe_transition(state, "ACTION4", {}, pre, post)
    assert out["kind"] == "move"
    assert state.press_stats["ACTION4"]["moves"][0]["terrain_morph"] is True
    pre2, post2 = scene(4, 6), scene(5, 4)
    wiggle_observe_transition(state, "ACTION4", {}, pre2, post2)
    mode, reason = wiggle_verdict(state)
    assert mode == "AVATAR" and "terrain" in reason


def test_click_verdict_untouched():
    state = WiggleState()
    state.directionals_at_start = []
    assert wiggle_verdict(state)[0] == "CLICK"


# ---------------------------------------------------------------------------------
# lifecycle: per-game reset, idempotency, hostile bundle, env contract
# ---------------------------------------------------------------------------------


def test_play_resets_struct_state(monkeypatch):
    from inference.framework import solver as duck_solver

    _enable(monkeypatch)
    _apply()
    assert getattr(
        duck_solver._HarnessGameSession.play, "_struct_play_patched", False
    ), "patch21's play wrapper (or its forwarded marker) is missing"
    tls = _struct_tls_state()
    tls["brake"][("h", "UP")] = 3
    tls["phase"]["phase"] = "COMMIT"
    try:
        duck_solver._HarnessGameSession.play(SimpleNamespace(game=SimpleNamespace()))
    except Exception:
        pass
    fresh = _struct_tls_state()
    assert fresh["brake"] == {} and fresh["phase"]["phase"] == "SCOUT"


def test_patches_apply_and_are_idempotent():
    _apply()
    assert duck_patches.patch_struct_channel() == "patch21 struct: SKIP (already applied)"
    assert duck_patches.patch_struct_gates() == "patch22 gates: SKIP (already applied)"


def test_patch21_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.agent import python_tool_sandbox as sandbox_mod

    monkeypatch.setattr(
        sandbox_mod, "run_sandboxed_python", lambda **kwargs: None
    )
    msg = duck_patches.patch_struct_channel()
    assert msg.startswith("patch21 struct: FAIL"), msg


def test_patch22_requires_patch21(monkeypatch):
    monkeypatch.setitem(_STRUCT_CHANNEL, "installed", False)
    msg = duck_patches.patch_struct_gates()
    assert msg.startswith("patch22 gates: FAIL (needs patch21"), msg


def test_off_by_default_via_env_contract():
    import os

    saved = os.environ.pop("TAAF_STRUCT", None)
    try:
        assert duck_patches._struct_enabled() is False
    finally:
        if saved is not None:
            os.environ["TAAF_STRUCT"] = saved


def test_one_flag_arms_the_whole_set(monkeypatch):
    _apply()
    monkeypatch.setenv("TAAF_STRUCT", "1")
    assert duck_patches._struct_enabled() is True
    assert duck_patches._struct_gates_active() is True
    monkeypatch.setenv("TAAF_STRUCT", "0")
    assert duck_patches._struct_enabled() is False
    assert duck_patches._struct_gates_active() is False


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE staleness gate (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------

SCORED_INFERENCE = SCORED_REF / "src/ARC3-Inference"


@pytest.mark.skipif(not SCORED_INFERENCE.is_dir(), reason="scored bundle bytes absent")
def test_scored_bundle_has_every_surface_patch21_rewrites():
    """The three exact sentences the channel rewrites must exist in the scored
    bytes — silent upstream drift would no-op the contract placement (the
    autopsy's root cause) without failing anything."""
    agent_src = (SCORED_INFERENCE / "inference/agent/tool_agent.py").read_text()
    assert duck_patches._STRUCT_TOOL_DESC_OLD in agent_src, (
        "tool-schema clause drifted — the contract would not land in the schema"
    )
    assert duck_patches._STRUCT_USER_LINE_OLD in agent_src
    assert "def _tools" in agent_src and "def _build_user_prompt" in agent_src
    prompts_src = (SCORED_INFERENCE / "inference/agent/prompts.py").read_text()
    assert duck_patches._STRUCT_CYCLE_OLD in prompts_src, (
        "observe-plan-act framing drifted — the replacement would silently no-op"
    )
    solver_src = (SCORED_INFERENCE / "inference/framework/solver.py").read_text()
    assert "class _HarnessGameSession" in solver_src and "def play" in solver_src


@pytest.mark.skipif(not SCORED_INFERENCE.is_dir(), reason="scored bundle bytes absent")
def test_patches_21_22_apply_on_scored_bundle_bytes():
    """Apply 18+21+22 on the scored bytes in a clean subprocess: enabled, every
    surface must carry the contract; disabled, all surfaces untouched."""
    import subprocess

    script = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(SCORED_INFERENCE)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

assert duck_patches.patch_run_probe().startswith("patch18 run-probe: OK")
assert duck_patches.patch_struct_channel().startswith("patch21 struct: OK")
assert duck_patches.patch_struct_gates().startswith("patch22 gates: OK")

from inference.agent import tool_agent

# disabled (default): the scored-bundle prompts must be untouched
prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
assert "PLAN CONTRACT" not in prompt
assert duck_patches._STRUCT_CYCLE_OLD in prompt

os.environ["TAAF_STRUCT"] = "1"
os.environ["TAAF_RUN_PROBE"] = "1"
prompt = tool_agent._build_system_prompt(tool_output_tokens=1000)
assert "PLAN CONTRACT" in prompt and "ENFORCED GATES" in prompt
assert duck_patches._STRUCT_CYCLE_OLD not in prompt
assert "observe-deliberate-plan cycle" in prompt
assert "run_probe — probe battery" not in prompt  # advert replaced

from inference.agent.tool_agent import ToolAgent
from inference.agent.runtime_state import Frame
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/struct-scored")
tools = agent._tools(Path("/tmp/struct-scored/runtime_state.json"))
desc = next(t["function"]["description"] for t in tools
            if t["function"]["name"] == "python")
assert "ordered plan of 1-20" in desc, desc

user_prompt = agent._build_user_prompt(
    1, valid_actions=["UP"], current_frame=Frame(grid=((1, 2),), step=1, level=1))
assert duck_patches._STRUCT_USER_LINE_NEW in user_prompt
assert "PHASE: SCOUT" in user_prompt

out = duck_patches.plan_execute(
    ["UP", "UP"],
    lambda actions: {{
        "action_result": {{"executed": True, "level": 1, "score": 0,
                           "action_display": "UP"}},
        "state": {{"current_frame": {{"grid": [[0]], "level": 1}},
                   "valid_actions": ["UP"]}},
    }},
    {{"current_frame": {{"grid": [[0]], "level": 1}}, "valid_actions": ["UP"]}},
)
assert out["action_result"]["executed_count"] == 2, out
print("SCORED-BUNDLE STRUCT OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-BUNDLE STRUCT OK" in proc.stdout
