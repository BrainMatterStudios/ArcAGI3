"""Tests for patch 20 (verifier-at-commit for large run_probe plans).

When the model commits to a run_probe plan of >= TAAF_VERIFY_MIN_ACTIONS
actions (default 5), a separate synchronous analyzer request — issued through
the harness's own `ToolAgent._chat_completion` client, reached via the
`_run_python_tool` TLS stash — asks an adversarial verifier for VETO(reason)
or PASS. Fail-open everywhere: timeout, HTTP error, missing agent, malformed
reply all mean PASS. A veto blocks execution once, injects "PLAN VETOED by
verifier: <reason>" into the PROBE REPORT slot, costs no probe-budget slot,
and the next run_probe call of any kind executes unconditionally (no loops).

Layers: trigger threshold (incl. env tuning), veto blocking + exact injection
text, single-revision no-loop rule, every fail-open path, guard ordering
(RESET/budget refusals never spend a verifier request), verifier prompt
contents, token accounting, agent stash + per-game reset, coexistence with
patch 18's executor and patch 19's _system_prompt descriptor, OFF-by-default
no-op, idempotency, hostile-bundle decline, scored-bundle staleness gate.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_verify.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
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
    _PROBE_TLS,
    _VERIFY_TLS,
    VERIFY_DIAGNOSTICS,
    _verify_maybe_veto,
    _verify_parse,
    _verify_tls_state,
    probe_execute,
)
from test_run_probe import ScriptedHandler, _grid, _state  # noqa: E402

VETO_MARKER = "PLAN VETOED by verifier:"
REPORT_MARKER = "PROBE REPORT"


def _apply() -> str:
    result = duck_patches.patch_verify_at_commit()
    assert "OK" in result or "SKIP" in result, result
    return result


@pytest.fixture(autouse=True)
def _fresh_verify_state():
    _PROBE_TLS.state = None
    _VERIFY_TLS.state = None
    _VERIFY_TLS.agent = None
    saved = dict(VERIFY_DIAGNOSTICS)
    for key in VERIFY_DIAGNOSTICS:
        VERIFY_DIAGNOSTICS[key] = 0
    yield
    _PROBE_TLS.state = None
    _VERIFY_TLS.state = None
    _VERIFY_TLS.agent = None
    VERIFY_DIAGNOSTICS.update(saved)


class FakeChatAgent:
    """Scripted `_chat_completion`: each reply is a content string or an Exception."""

    def __init__(self, replies, usage=None):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.usage = usage if usage is not None else {"total_tokens": 123}

    def _chat_completion(self, messages, *, tools=None, request_timeout_seconds=None):
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "timeout": request_timeout_seconds,
            }
        )
        item = self.replies.pop(0) if self.replies else "PASS"
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            message={"content": item}, finish_reason="stop", usage=dict(self.usage)
        )


def _enable(monkeypatch, agent=None, **env):
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    monkeypatch.setenv("TAAF_VERIFY", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    if agent is not None:
        _VERIFY_TLS.agent = agent


# ---------------------------------------------------------------------------------
# trigger threshold
# ---------------------------------------------------------------------------------


def test_small_plans_skip_the_verifier(monkeypatch):
    agent = FakeChatAgent(["VETO(should never be asked)"])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(4)])
    out = probe_execute(["UP"] * 4, handler, handler.state())  # 4 < K=5
    assert out["action_result"]["executed_count"] == 4
    assert agent.requests == []
    assert VERIFY_DIAGNOSTICS["calls"] == 0


def test_threshold_plan_is_verified_then_executes_on_pass(monkeypatch):
    agent = FakeChatAgent(["PASS"])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(5)])
    out = probe_execute(["UP"] * 5, handler, handler.state())
    assert out["action_result"]["executed_count"] == 5
    assert len(agent.requests) == 1
    assert VERIFY_DIAGNOSTICS == {
        "calls": 1,
        "passes": 1,
        "vetoes": 0,
        "timeouts": 0,
        "malformed": 0,
        "unavailable": 0,
        "tokens": 123,
    }


def test_threshold_is_env_tunable(monkeypatch):
    agent = FakeChatAgent(["PASS"])
    _enable(monkeypatch, agent, TAAF_VERIFY_MIN_ACTIONS=2)
    handler = ScriptedHandler([{}, {}])
    probe_execute(["UP", "UP"], handler, handler.state())
    assert len(agent.requests) == 1


# ---------------------------------------------------------------------------------
# veto: blocking, injection text, budget, single revision cycle
# ---------------------------------------------------------------------------------


def test_veto_blocks_execution_and_keeps_the_budget_slot(monkeypatch):
    agent = FakeChatAgent(["VETO(the third DOWN walks into the hazard row)"])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(5)])
    out = probe_execute(["DOWN"] * 5, handler, handler.state())
    table = out["action_result"]
    assert handler.calls == []  # nothing executed
    assert table["executed"] is False and table["vetoed"] is True
    assert table["error"] == (
        "PLAN VETOED by verifier: the third DOWN walks into the hazard row"
    )
    assert table["veto_reason"] == "the third DOWN walks into the hazard row"
    assert table["steps"] == []
    assert table["cost"]["actions_spent_this_call"] == 0
    assert table["cost"]["probe_calls_left_this_level"] == 3  # slot NOT consumed
    assert VERIFY_DIAGNOSTICS["vetoes"] == 1
    assert json.dumps(table)  # veto table must be JSON-able for the sandbox IPC


def test_single_revision_cycle_no_veto_loops(monkeypatch):
    agent = FakeChatAgent(["VETO(bad plan)", "VETO(would veto again)"])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(20)])
    first = probe_execute(["UP"] * 5, handler, handler.state())
    assert first["action_result"]["vetoed"] is True
    # the re-submitted plan — even identical, even >= K — executes unconditionally
    second = probe_execute(["UP"] * 5, handler, handler.state())
    assert second["action_result"]["executed_count"] == 5
    assert len(agent.requests) == 1  # no second verifier request
    # the cycle is over: the NEXT big plan is verified again
    third = probe_execute(["UP"] * 5, handler, handler.state())
    assert third["action_result"]["vetoed"] is True
    assert len(agent.requests) == 2


def test_skip_flag_is_consumed_by_any_next_call_even_small(monkeypatch):
    agent = FakeChatAgent(["VETO(bad)", "PASS"])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(20)])
    probe_execute(["UP"] * 5, handler, handler.state())  # veto -> skip_next set
    probe_execute(["UP"], handler, handler.state())  # small call consumes the flag
    assert _verify_tls_state()["skip_next"] is False
    probe_execute(["UP"] * 5, handler, handler.state())  # verified again
    assert len(agent.requests) == 2


def test_veto_text_lands_in_the_next_prompt_report(monkeypatch):
    from inference.agent.runtime_state import Frame
    from inference.agent.tool_agent import ToolAgent

    agent = FakeChatAgent(["VETO(all five clicks target dead cells)"])
    _enable(monkeypatch, agent)
    duck_patches.patch_run_probe()
    _apply()
    handler = ScriptedHandler([])
    probe_execute([{"action": "MOUSE", "row": 1, "col": 1}] * 5, handler, handler.state())
    tool_agent_obj = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    tool_agent_obj._session_runtime_dir = Path("/tmp/verify-test")
    prompt = tool_agent_obj._build_user_prompt(
        1,
        valid_actions=["ACTION6"],
        current_frame=Frame(grid=((1, 2), (3, 4)), step=1, level=1),
    )
    assert REPORT_MARKER in prompt
    assert f"{VETO_MARKER} all five clicks target dead cells" in prompt
    assert "next run_probe call executes without verification" in prompt


# ---------------------------------------------------------------------------------
# fail-open paths (the verifier must never block progress)
# ---------------------------------------------------------------------------------


def test_fail_open_on_timeout(monkeypatch):
    import requests

    agent = FakeChatAgent([requests.exceptions.Timeout("verifier too slow")])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(5)])
    out = probe_execute(["UP"] * 5, handler, handler.state())
    assert out["action_result"]["executed_count"] == 5  # plan ran anyway
    assert VERIFY_DIAGNOSTICS["timeouts"] == 1 and VERIFY_DIAGNOSTICS["vetoes"] == 0


def test_fail_open_on_malformed_reply(monkeypatch):
    agent = FakeChatAgent(["hmm, it could work, hard to say"])
    _enable(monkeypatch, agent)
    handler = ScriptedHandler([{} for _ in range(5)])
    out = probe_execute(["UP"] * 5, handler, handler.state())
    assert out["action_result"]["executed_count"] == 5
    assert VERIFY_DIAGNOSTICS["malformed"] == 1


def test_fail_open_when_no_agent_is_stashed(monkeypatch):
    _enable(monkeypatch, agent=None)
    handler = ScriptedHandler([{} for _ in range(5)])
    out = probe_execute(["UP"] * 5, handler, handler.state())
    assert out["action_result"]["executed_count"] == 5
    assert VERIFY_DIAGNOSTICS["unavailable"] == 1


def test_hard_timeout_is_passed_to_the_client(monkeypatch):
    agent = FakeChatAgent(["PASS"])
    _enable(monkeypatch, agent, TAAF_VERIFY_TIMEOUT_S=7)
    handler = ScriptedHandler([{} for _ in range(5)])
    probe_execute(["UP"] * 5, handler, handler.state())
    assert agent.requests[0]["timeout"] == 7.0
    assert agent.requests[0]["tools"] is None


# ---------------------------------------------------------------------------------
# parsing + guard ordering + prompt contents
# ---------------------------------------------------------------------------------


def test_parse_veto_wins_over_pass_and_normalizes():
    assert _verify_parse("I want to say PASS but VETO( route is\n  blocked )") == (
        "veto",
        "route is blocked",
    )
    assert _verify_parse("...analysis...\nPASS") == ("pass", "")
    assert _verify_parse("veto(lowercase works)") == ("veto", "lowercase works")
    assert _verify_parse("VETO()") == ("veto", "unspecified risk")
    assert _verify_parse(None) == ("malformed", "")
    assert _verify_parse([{"type": "text", "text": "PASS"}]) == ("pass", "")


def test_hard_guards_never_spend_a_verifier_request(monkeypatch):
    agent = FakeChatAgent(["VETO(should never be asked)"])
    _enable(monkeypatch, agent, TAAF_PROBE_MAX_CALLS=1)
    handler = ScriptedHandler([{} for _ in range(10)])
    # RESET refusal happens before the verifier
    refused = probe_execute(["UP"] * 4 + ["RESET"], handler, handler.state())
    assert refused["action_result"]["executed"] is False
    assert agent.requests == []
    # budget refusal happens before the verifier
    probe_execute(["UP"], handler, handler.state())  # consumes the only slot
    refused2 = probe_execute(["UP"] * 5, handler, handler.state())
    assert "budget exhausted" in refused2["action_result"]["error"]
    assert agent.requests == []


def test_verifier_prompt_carries_plan_legend_tables_and_instruction(monkeypatch):
    agent = FakeChatAgent(["PASS", "PASS"])
    _enable(monkeypatch, agent)
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    ws = duck_patches.WiggleState()
    ws.battery_done = True
    ws.battery_presses = 4
    ws.mode = "CLICK"
    ws.mode_reason = "all 4 directional presses were masked no-ops"
    duck_patches._WIGGLE_TLS.state = ws
    handler = ScriptedHandler([{} for _ in range(10)])
    try:
        probe_execute(["UP", "UP"], handler, handler.state())  # banks an effect table
        probe_execute(["DOWN"] * 5, handler, handler.state())
        request = agent.requests[-1]
        system, user = request["messages"]
        assert system["role"] == "system" and "adversarial" in system["content"]
        assert "VETO(<one short reason>) or PASS" in system["content"]
        body = user["content"]
        assert "PROPOSED PLAN (5 actions)" in body and '"DOWN"' in body
        assert "GAME MODE: CLICK" in body  # patch 17's legend, un-consumed
        assert "RECENT EFFECT TABLES:" in body and '"executed_count": 2' in body
        assert (
            "Find the single strongest reason this plan fails; "
            "answer VETO(reason) or PASS." in body
        )
        assert "VALID ACTIONS NOW:" in body
    finally:
        duck_patches._WIGGLE_TLS.state = None


# ---------------------------------------------------------------------------------
# agent stash + per-game reset + coexistence
# ---------------------------------------------------------------------------------


def test_run_python_tool_stashes_and_clears_the_agent():
    from inference.agent.tool_agent import ToolAgent

    _apply()
    seen: list = []

    class Trap:
        @property
        def _ensure_session(self):
            seen.append(getattr(_VERIFY_TLS, "agent", None))
            raise RuntimeError("stop after the stash")

    trap = Trap()
    with pytest.raises(RuntimeError, match="stop after the stash"):
        ToolAgent._run_python_tool(trap, Path("/tmp/x"), {"code": "1"})
    assert seen == [trap]  # the live agent was stashed while the tool ran
    assert getattr(_VERIFY_TLS, "agent", None) is None  # and cleared after


def test_play_resets_the_revision_flag(monkeypatch):
    from inference.framework import solver as duck_solver

    _apply()
    assert getattr(
        duck_solver._HarnessGameSession.play, "_verify_play_patched", False
    )
    _verify_tls_state()["skip_next"] = True
    try:
        duck_solver._HarnessGameSession.play(SimpleNamespace())
    except Exception:
        pass  # hostile stub: downstream wrappers raise after the reset ran
    assert _verify_tls_state()["skip_next"] is False


def test_coexists_with_patch19_descriptor(monkeypatch):
    from inference.agent.tool_agent import ToolAgent

    duck_patches.patch_archetype_dispatch()
    _apply()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    # patch 19's data descriptor still routes writes/reads after patch 20
    agent._system_prompt = "BASE-PROMPT"
    monkeypatch.delenv("TAAF_DISPATCH", raising=False)
    assert agent._system_prompt == "BASE-PROMPT"
    monkeypatch.setenv("TAAF_DISPATCH", "1")
    monkeypatch.setenv("TAAF_WIGGLE", "1")
    ws = duck_patches.WiggleState()
    ws.battery_done = True
    ws.mode = "CLICK"
    ws.mode_reason = "test"
    duck_patches._WIGGLE_TLS.state = ws
    try:
        assert "ARCHETYPE SCAFFOLD" in agent._system_prompt
        assert agent._system_prompt.startswith("BASE-PROMPT")
    finally:
        duck_patches._WIGGLE_TLS.state = None
        duck_patches._DISPATCH_TLS.state = None


# ---------------------------------------------------------------------------------
# OFF-by-default, idempotency, hostile bundle
# ---------------------------------------------------------------------------------


def test_off_by_default_never_calls_the_verifier(monkeypatch):
    monkeypatch.setenv("TAAF_RUN_PROBE", "1")
    monkeypatch.delenv("TAAF_VERIFY", raising=False)
    agent = FakeChatAgent(["VETO(should never be asked)"])
    _VERIFY_TLS.agent = agent
    handler = ScriptedHandler([{} for _ in range(20)])
    out = probe_execute(["UP"] * 20, handler, handler.state())
    assert out["action_result"]["executed_count"] == 20
    assert agent.requests == []
    assert all(v == 0 for v in VERIFY_DIAGNOSTICS.values())


def test_patch_applies_and_is_idempotent():
    first = _apply()
    assert first.startswith("patch20 verify: OK") or "SKIP" in first
    assert (
        duck_patches.patch_verify_at_commit()
        == "patch20 verify: SKIP (already applied)"
    )


def test_patch_declines_cleanly_on_hostile_bundle(monkeypatch):
    from inference.agent import tool_agent

    def bare(self, *args, **kwargs):  # no _verify_patched marker
        raise AssertionError("never called")

    monkeypatch.setattr(tool_agent.ToolAgent, "_run_python_tool", bare)
    monkeypatch.setattr(tool_agent.ToolAgent, "_chat_completion", None)
    msg = duck_patches.patch_verify_at_commit()
    assert msg.startswith("patch20 verify: FAIL"), msg


# ---------------------------------------------------------------------------------
# SCORED-BUNDLE staleness gate (apply-or-clean-SKIP on the eval bytes)
# ---------------------------------------------------------------------------------

SCORED_INFERENCE = SCORED_REF / "src/ARC3-Inference"


def test_scored_bundle_has_every_symbol_patch20_touches():
    agent_src = (SCORED_INFERENCE / "inference/agent/tool_agent.py").read_text()
    for symbol in (
        "def _run_python_tool",
        "def _chat_completion",
        "request_timeout_seconds",
        "def _build_user_prompt",
    ):
        assert symbol in agent_src, f"scored bundle lost {symbol!r} — re-validate patch20"
    solver_src = (SCORED_INFERENCE / "inference/framework/solver.py").read_text()
    assert "class _HarnessGameSession" in solver_src
    assert "def play" in solver_src


@pytest.mark.skipif(not SCORED_INFERENCE.is_dir(), reason="scored bundle bytes absent")
def test_patch20_applies_on_scored_bundle_bytes():
    """Apply patches 18+20 on the scored bytes in a clean subprocess and drive a
    veto end-to-end: blocked plan, exact injection text, single revision cycle."""
    script = f"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {str(SCORED_INFERENCE)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import duck_patches

assert duck_patches.patch_run_probe().startswith("patch18 run-probe: OK")
r = duck_patches.patch_verify_at_commit()
assert r.startswith("patch20 verify: OK"), r

os.environ["TAAF_RUN_PROBE"] = "1"
os.environ["TAAF_VERIFY"] = "1"

class FakeAgent:
    def __init__(self):
        self.n = 0
    def _chat_completion(self, messages, *, tools=None, request_timeout_seconds=None):
        self.n += 1
        return SimpleNamespace(
            message={{"content": "VETO(the plan clicks a dead region)"}},
            finish_reason="stop", usage={{"total_tokens": 50}})

fake = FakeAgent()
duck_patches._VERIFY_TLS.agent = fake
executed = []
def handler(actions):
    executed.append(actions)
    return {{"action_result": {{"executed": True, "level": 1, "action_display": "UP"}},
             "state": {{"current_frame": {{"grid": [[0]], "level": 1}}, "valid_actions": ["UP"]}}}}

state = {{"current_frame": {{"grid": [[0]], "level": 1}}, "valid_actions": ["UP"]}}
out = duck_patches.probe_execute(["UP"] * 5, handler, state)
assert out["action_result"]["vetoed"] is True, out
assert out["action_result"]["error"] == "PLAN VETOED by verifier: the plan clicks a dead region"
assert executed == [] and fake.n == 1

# single revision cycle on the scored bytes too
out2 = duck_patches.probe_execute(["UP"] * 5, handler, state)
assert out2["action_result"]["executed_count"] == 5 and fake.n == 1

from inference.agent.tool_agent import ToolAgent
from inference.agent.runtime_state import Frame
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/verify-scored")
prompt = agent._build_user_prompt(
    1, valid_actions=["ACTION1"], current_frame=Frame(grid=((1, 2),), step=1, level=1)
)
assert "PLAN VETOED by verifier: the plan clicks a dead region" in prompt
print("SCORED-VERIFY-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-VERIFY-OK" in proc.stdout
