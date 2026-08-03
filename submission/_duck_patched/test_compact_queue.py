"""Tests for patch 12 (compaction-on-evict + LLM-free plan queue).

No real LLM anywhere: the analyzer endpoint is stubbed at `_chat_completion`.
Three layers:

  1. UNIT — compaction cadence/caps/fallback/failed-hypotheses merge; queue
     parsing, validation, precondition checks, abort-and-report, length cap.
  2. E2E — the stub-brain harness driver pattern from test_watchdog.py: a real
     ``taaf.GameAPI`` + ``_HarnessGameSession.play`` loop driven by a REAL
     ``ToolAgent`` whose only stub is ``_chat_completion``. A canned response
     emits a plan_queue; the loop must drain it with zero further LLM calls and
     correct diagnostics; an induced violation must abort cleanly and surface
     the report in the next prompt the stub receives.
  3. SCORED-BUNDLE — patch 12 must APPLY (or decline with a clean FAIL string,
     never crash) on the scored bundle bytes at scratchpad/taaf_scored_ref,
     which is what actually runs at eval (law: the drifted _adopt tree is NOT
     the eval bytes; see commit 7edec38).

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_compact_queue.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref/src/ARC3-Inference"
ENV_DIR = REPO / "environment_files"
if str(TAAF_INFERENCE) not in sys.path:
    sys.path.insert(0, str(TAAF_INFERENCE))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import duck_patches  # noqa: E402
from duck_patches import (  # noqa: E402
    COMPACT_DIAGNOSTICS,
    _abort_plan_queue,
    _capture_plan_queue,
    _compact_state,
    _drain_plan_queue,
    _find_plan_queue_json,
    _merge_compact_knowledge,
    _normalize_plan_queue,
    _note_evictions,
    _parse_compaction_reply,
    _render_compaction_prompt,
)

GAME = "ft09"


def _apply_patch12() -> None:
    for result in (duck_patches.patch_compaction(), duck_patches.patch_plan_queue()):
        assert "OK" in result or "SKIP" in result, result


def _fresh_agent(monkeypatch=None):
    """A real ToolAgent, constructed offline; no request leaves the process."""
    from inference.agent.tool_agent import ToolAgent

    _apply_patch12()
    agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
    agent._session_runtime_dir = Path("/tmp/compact-test")  # skip _ensure_session reset
    return agent


def _diag_snapshot() -> dict[str, int]:
    return dict(COMPACT_DIAGNOSTICS)


def _diag_delta(before: dict[str, int]) -> dict[str, int]:
    return {key: COMPACT_DIAGNOSTICS[key] - before[key] for key in COMPACT_DIAGNOSTICS}


class _StubChat:
    """Records calls; returns canned _ChatCompletionResult-shaped objects."""

    def __init__(self, replies=None, error=None):
        self.calls: list[dict] = []
        self.replies = list(replies or [])
        self.error = error

    def __call__(self, messages, *, tools=None, request_timeout_seconds=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "timeout": request_timeout_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        content = self.replies.pop(0) if self.replies else "{}"
        return SimpleNamespace(
            message={"content": content},
            finish_reason="stop",
            usage={"total_tokens": 123},
        )


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


# =================================================================================
# UNIT — compaction
# =================================================================================


def test_eviction_capture_and_cadence(monkeypatch):
    """No compaction below the cadence threshold; exactly one at/after it."""
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_COMPACT_EVERY", "4")
    agent = _fresh_agent()
    stub = _StubChat(replies=[json.dumps({"facts": ["the door is red"]})])
    agent._chat_completion = stub
    before = _diag_snapshot()

    # 2 evictions: below cadence -> buffered, no call.
    agent._history_messages = [_user("old finding A"), _assistant("old reply A"), _user("keep")]
    _note_evictions(agent, agent._history_messages, [_user("keep")])
    assert len(stub.calls) == 0
    state = _compact_state(agent)
    assert state["evictions_since"] == 2
    assert len(state["evict_buffer"]) == 2

    # 2 more evictions: cadence hit -> exactly one compaction call.
    _note_evictions(agent, [_user("old B"), _assistant("old reply B")], [])
    assert len(stub.calls) == 1
    state = _compact_state(agent)
    assert state["evictions_since"] == 0
    assert state["evict_buffer"] == []
    assert state["knowledge"]["facts"] == ["the door is red"]

    delta = _diag_delta(before)
    assert delta["evictions_seen"] == 4
    assert delta["compactions_done"] == 1
    assert delta["compaction_tokens"] == 123
    assert delta["compaction_failures"] == 0


def test_compaction_prompt_and_response_token_caps(monkeypatch):
    """The compaction prompt is char-capped and the reply is token-capped."""
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_COMPACT_EVERY", "1")
    monkeypatch.setenv("TAAF_COMPACT_PROMPT_CHARS", "2000")
    monkeypatch.setenv("TAAF_COMPACT_RESPONSE_TOKENS", "128")
    monkeypatch.setenv("TAAF_COMPACT_TIMEOUT_S", "7")
    agent = _fresh_agent()
    agent._max_output_tokens = 4096
    observed: dict = {}

    def stub(messages, *, tools=None, request_timeout_seconds=None):
        observed["prompt"] = messages[0]["content"]
        observed["max_output_tokens_during_call"] = agent._max_output_tokens
        observed["timeout"] = request_timeout_seconds
        return SimpleNamespace(message={"content": "{}"}, finish_reason="stop", usage=None)

    agent._chat_completion = stub
    huge = [_assistant("X" * 3000), _assistant("Y" * 3000)]
    _note_evictions(agent, huge, [])

    assert "prompt" in observed, "compaction call did not fire"
    assert len(observed["prompt"]) <= 2000 + 100  # fixed preamble is protected
    assert observed["max_output_tokens_during_call"] == 128
    assert observed["timeout"] == 7.0
    assert agent._max_output_tokens == 4096, "cap must be restored after the call"


def test_compaction_fallback_on_failure(monkeypatch):
    """A failing/timing-out compaction call degrades to the stock silent drop."""
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_COMPACT_EVERY", "1")
    agent = _fresh_agent()
    agent._chat_completion = _StubChat(error=RuntimeError("endpoint down"))
    before = _diag_snapshot()

    _note_evictions(agent, [_assistant("finding that will be lost")], [])

    state = _compact_state(agent)
    assert state["evict_buffer"] == [], "buffer must be discarded on failure"
    assert state["knowledge"]["facts"] == [], "knowledge must be untouched on failure"
    delta = _diag_delta(before)
    assert delta["compaction_failures"] == 1
    assert delta["compactions_done"] == 0

    # A malformed (non-JSON) reply is the same failure class.
    agent._chat_completion = _StubChat(replies=["sorry, I cannot do that"])
    _note_evictions(agent, [_assistant("another finding")], [])
    assert _diag_delta(before)["compaction_failures"] == 2


def test_compaction_max_calls_cap(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_COMPACT_EVERY", "1")
    monkeypatch.setenv("TAAF_COMPACT_MAX_CALLS", "2")
    agent = _fresh_agent()
    stub = _StubChat(replies=["{}"] * 10)
    agent._chat_completion = stub
    for index in range(5):
        _note_evictions(agent, [_assistant(f"finding {index}")], [])
    assert len(stub.calls) == 2, "per-game compaction call cap must bind"


def test_failed_hypotheses_merge_is_mechanical():
    """Old failed hypotheses survive even when the reply omits them."""
    state = {"knowledge": {"facts": ["f0"], "action_effects": [], "failed_hypotheses": ["clicking the sun does nothing"], "open_questions": ["q0"]}}
    _merge_compact_knowledge(
        state,
        {
            "facts": ["f1"],
            "action_effects": ["UP moves the block"],
            "failed_hypotheses": ["walls are not passable"],
            "open_questions": [],
        },
    )
    knowledge = state["knowledge"]
    assert knowledge["failed_hypotheses"] == [
        "clicking the sun does nothing",
        "walls are not passable",
    ], "failed_hypotheses must be old ∪ new"
    assert knowledge["facts"] == ["f1"], "other keys trust the model's merge"
    assert knowledge["open_questions"] == ["q0"], "empty reply keys keep the old value"

    # Cap: newest entries win once over the item cap.
    state = {"knowledge": {"facts": [], "action_effects": [], "failed_hypotheses": [f"h{i}" for i in range(10)], "open_questions": []}}
    _merge_compact_knowledge(state, {"failed_hypotheses": ["h-new"]})
    merged = state["knowledge"]["failed_hypotheses"]
    assert len(merged) == 10 and merged[-1] == "h-new" and "h0" not in merged


def test_compaction_reply_parser_tolerates_prose_and_fences():
    reply = "Sure! Here is the store:\n```json\n{\"facts\": [\"a\"], \"failed_hypotheses\": [\"b\"]}\n```"
    parsed = _parse_compaction_reply(reply)
    assert parsed is not None
    assert parsed["facts"] == ["a"] and parsed["failed_hypotheses"] == ["b"]
    assert parsed["action_effects"] == [] and parsed["open_questions"] == []
    assert _parse_compaction_reply("no json here") is None
    long_item = "x" * 1000
    parsed = _parse_compaction_reply(json.dumps({"facts": [long_item] * 30}))
    assert len(parsed["facts"]) == 10 and len(parsed["facts"][0]) == 240


def test_knowledge_injected_into_user_prompt(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()
    state = _compact_state(agent)
    state["knowledge"]["facts"] = ["keys open matching doors"]
    state["knowledge"]["failed_hypotheses"] = ["pushing walls does nothing"]
    prompt = agent._build_user_prompt(1, valid_actions=["ACTION1"])
    assert "Compacted memory from evicted earlier turns" in prompt
    assert "keys open matching doors" in prompt
    assert "FAILED HYPOTHESES (do not retry these)" in prompt
    assert "pushing walls does nothing" in prompt


def test_compact_disabled_env_is_inert(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "0")
    agent = _fresh_agent()
    stub = _StubChat()
    agent._chat_completion = stub
    before = _diag_snapshot()

    # The wrapped _persistent_history_messages must not capture anything.
    agent._history_messages = [_user("u1"), _assistant("a1")]
    agent._context_budget_tokens = 1  # force the trimmer to evict everything
    survivors = agent._persistent_history_messages(
        [{"role": "system", "content": "s"}, _user("u1"), _assistant("a1"), _user("u2")]
    )
    assert isinstance(survivors, list)
    assert _diag_delta(before)["evictions_seen"] == 0
    assert len(stub.calls) == 0

    # Queue capture and prompt injection are also off.
    agent2 = _fresh_agent()
    agent2._update_summarized_knowledge_from_assistant('{"plan_queue": ["UP"]}')
    assert not _compact_state(agent2)["queue"], "wrapped capture path must be gated off"
    state = _compact_state(agent)
    state["knowledge"]["facts"] = ["should not appear"]
    prompt = agent._build_user_prompt(1, valid_actions=["ACTION1"])
    assert "Compacted memory" not in prompt


def test_eviction_capture_via_real_trimmer(monkeypatch):
    """End-to-end through the REAL _persistent_history_messages under a tiny budget."""
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_COMPACT_EVERY", "50")  # buffer only, no call
    agent = _fresh_agent()
    agent._chat_completion = _StubChat()
    before = _diag_snapshot()

    old_history = [_user("ancient finding ONE"), _assistant("ancient reply TWO"), _user("recent")]
    agent._history_messages = list(old_history)
    agent._context_budget_tokens = 60  # tokens; forces dropping the oldest blocks
    messages = [{"role": "system", "content": "sys"}, *old_history, _user("newest question")]
    survivors = agent._persistent_history_messages(messages)

    evicted_count = _diag_delta(before)["evictions_seen"]
    assert evicted_count >= 1, "the tiny budget must evict at least one old block"
    state = _compact_state(agent)
    joined = "\n".join(state["evict_buffer"])
    assert "ancient" in joined
    for message in survivors:
        assert message in messages


# =================================================================================
# UNIT — plan queue
# =================================================================================


def test_queue_json_extraction_variants():
    fenced = 'Plan below.\n```json\n{"plan_queue": [{"action": "UP"}]}\n```'
    bare = 'World model: ok. {"plan_queue": ["UP", "DOWN"]} end.'
    nested = 'x {"note": "outer"} y {"plan_queue": [{"action": "MOUSE", "row": 1, "col": 2, "expect": {"board_changed": true}}]}'
    assert _find_plan_queue_json(fenced) is not None
    assert _find_plan_queue_json(bare)[1] == ["UP", "DOWN"]
    assert len(_find_plan_queue_json(nested)[1]) == 1
    assert _find_plan_queue_json("no queue here") is None
    assert _find_plan_queue_json('{"plan_queue": "not-a-list"}') is None


def test_queue_normalization_and_length_cap():
    steps, rejection = _normalize_plan_queue(
        ["UP", {"action": "ACTION2"}, {"action": "MOUSE", "row": 70, "col": -3, "note": "probe"}], 10
    )
    assert rejection is None
    assert [s["action"] for s in steps] == ["ACTION1", "ACTION2", "ACTION6"]
    assert steps[2]["row"] == 63 and steps[2]["col"] == 0  # clamped like the solver
    assert steps[2]["note"] == "probe"

    steps, rejection = _normalize_plan_queue(["UP"] * 25, 10)
    assert rejection is None and len(steps) == 10, "length cap must truncate"

    for bad, why in (
        (["FLY"], "unknown"),
        (["RESET"], "RESET"),
        ([{"action": "MOUSE"}], "row and col"),
        ([], "empty"),
        ([42], "not an action object"),
    ):
        steps, rejection = _normalize_plan_queue(bad, 10)
        assert steps == [] and rejection is not None and why in rejection


def test_queue_capture_and_rejection_reporting(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()
    before = _diag_snapshot()
    agent._last_action_result = {"level": 2}

    _capture_plan_queue(agent, 'Plan: {"plan_queue": ["UP", "DOWN"]}')
    state = _compact_state(agent)
    assert len(state["queue"]) == 2 and state["queue_level"] == 2
    assert _diag_delta(before)["queue_plans"] == 1

    # Re-emitting the identical block (retry loop) is not a second plan.
    _capture_plan_queue(agent, 'again {"plan_queue": ["UP", "DOWN"]}')
    assert _diag_delta(before)["queue_plans"] == 1

    # A rejected plan clears the queue and surfaces a report.
    _capture_plan_queue(agent, '{"plan_queue": ["RESET"]}')
    state = _compact_state(agent)
    assert state["queue"] == []
    assert "PLAN QUEUE REJECTED" in state["queue_report"]
    assert _diag_delta(before)["queue_plans_rejected"] == 1
    prompt = agent._build_user_prompt(1, valid_actions=["ACTION1"])
    assert "PLAN QUEUE REJECTED" in prompt
    prompt2 = agent._build_user_prompt(1, valid_actions=["ACTION1"])
    assert "PLAN QUEUE REJECTED" not in prompt2, "reports show exactly once"


def _drain(agent, valid=("ACTION1", "ACTION2", "ACTION6"), step_env=None, should_stop=None):
    return _drain_plan_queue(
        agent, Path("/tmp/compact-test/state.json"), list(valid), step_env, should_stop
    )


def _step_payload(**overrides):
    payload = {
        "executed": True,
        "action_num": 5,
        "level": 1,
        "score": 0,
        "state": "NOT_FINISHED",
        "board_changed": True,
        "done": False,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
        "action_display": "UP",
    }
    payload.update(overrides)
    return payload


def test_queue_drain_executes_without_llm(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["UP", "DOWN"]}')
    before = _diag_snapshot()
    calls: list[dict] = []

    def step_env(arguments):
        calls.append(arguments)
        return _step_payload()

    result = _drain(agent, step_env=step_env)
    assert result is not None and result.step_executed
    assert calls == [{"action": "ACTION1"}]
    assert len(_compact_state(agent)["queue"]) == 1
    # Book-keeping mirrors a model-driven step.
    assert agent._last_action_result["executed"] is True
    assert agent._last_step_summary is not None

    result = _drain(agent, step_env=step_env)
    assert result is not None and calls[-1] == {"action": "ACTION2"}
    assert _compact_state(agent)["queue"] == []
    delta = _diag_delta(before)
    assert delta["queue_steps_executed"] == 2
    assert delta["llm_calls_saved"] == 2
    assert delta["queue_aborts"] == 0

    assert _drain(agent, step_env=step_env) is None, "empty queue -> fall through to LLM"


def test_queue_abort_on_board_expectation(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(
        agent,
        '{"plan_queue": [{"action": "UP", "expect": {"board_changed": true}}, {"action": "DOWN"}]}',
    )
    before = _diag_snapshot()

    result = _drain(agent, step_env=lambda args: _step_payload(board_changed=False))
    assert result is not None and result.step_executed, "the violating step DID execute"
    state = _compact_state(agent)
    assert state["queue"] == [], "remainder must be dropped on first violation"
    assert "board_changed=False" in state["queue_report"]
    assert "DOWN" in state["queue_report"], "report lists the dropped remainder"
    delta = _diag_delta(before)
    assert delta["queue_aborts"] == 1 and delta["queue_steps_executed"] == 1
    prompt = agent._build_user_prompt(1, valid_actions=["ACTION1"])
    assert "PLAN QUEUE ABORTED after 1/2 steps" in prompt


def test_queue_abort_on_game_over_and_level_change(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()

    # GAME_OVER on the queued step.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["UP", "DOWN"]}')
    _drain(agent, step_env=lambda args: _step_payload(game_over=True, state="GAME_OVER"))
    state = _compact_state(agent)
    assert state["queue"] == [] and "GAME_OVER" in state["queue_report"]

    # Unexpected level completion aborts the (now stale) remainder.
    # (A distinct plan: an aborted plan's verbatim JSON is never re-accepted.)
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["DOWN", "UP"]}')
    _drain(agent, step_env=lambda args: _step_payload(level_completed=True, level=2))
    state = _compact_state(agent)
    assert state["queue"] == [] and "not planned" in state["queue_report"]

    # Expected level completion continues and re-stamps the level guard.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(
        agent,
        '{"plan_queue": [{"action": "UP", "expect": {"level_completed": true}}, {"action": "DOWN"}]}',
    )
    _drain(agent, step_env=lambda args: _step_payload(level_completed=True, level=2))
    state = _compact_state(agent)
    assert len(state["queue"]) == 1 and state["queue_level"] == 2

    # ...and an expected-but-missing completion aborts.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(
        agent, '{"plan_queue": [{"action": "UP", "expect": {"level_completed": true}}]}'
    )
    _drain(agent, step_env=lambda args: _step_payload())
    assert "did not complete the level" in _compact_state(agent)["queue_report"]


def test_queue_stale_and_invalid_guards(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()
    # Each stage uses a distinct plan: aborted plans are never re-accepted verbatim.

    # Level moved between capture and drain -> dropped before firing.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["UP"]}')
    agent._last_action_result = {"level": 2}
    fired: list = []
    assert _drain(agent, step_env=lambda args: fired.append(args)) is None
    assert fired == [] and "level changed" in _compact_state(agent)["queue_report"]

    # The just-aborted plan's identical JSON is ignored on re-emission.
    before = _diag_snapshot()
    _capture_plan_queue(agent, '{"plan_queue": ["UP"]}')
    assert _diag_delta(before)["queue_plans"] == 0
    assert _compact_state(agent)["queue"] == []

    # Terminal state before the queued step -> dropped before firing.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["DOWN"]}')
    agent._last_action_result = {"level": 1, "game_over": True}
    assert _drain(agent, step_env=lambda args: fired.append(args)) is None
    assert fired == [] and "terminal state" in _compact_state(agent)["queue_report"]

    # Queued action not valid right now -> dropped before firing.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["UP", "UP"]}')
    assert _drain(agent, valid=("ACTION6",), step_env=lambda args: fired.append(args)) is None
    assert fired == [] and "not a valid action" in _compact_state(agent)["queue_report"]

    # step_env not executed -> abort, no analyzer result (LLM handles the turn).
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["DOWN", "DOWN"]}')
    result = _drain(agent, step_env=lambda args: {"executed": False, "error": "nope"})
    assert result is None and "was not executed" in _compact_state(agent)["queue_report"]

    # should_stop -> leave the queue alone, no action fired.
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["LEFT"]}')
    assert _drain(agent, step_env=lambda args: fired.append(args), should_stop=lambda: True) is None
    assert fired == [] and len(_compact_state(agent)["queue"]) == 1


def test_queue_session_change_clears_state(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()
    agent._last_action_result = {"level": 1}
    _capture_plan_queue(agent, '{"plan_queue": ["UP"]}')
    assert _compact_state(agent)["queue"]
    # A different runtime dir (new game) must start from a clean slate.
    state = _compact_state(agent, runtime_dir=Path("/tmp/other-game"))
    assert state["queue"] == [] and state["knowledge"]["facts"] == []


def test_system_prompt_carries_queue_guidance(monkeypatch):
    monkeypatch.setenv("TAAF_COMPACT", "1")
    agent = _fresh_agent()  # constructed AFTER the patch wrapped _build_system_prompt
    assert "PLAN QUEUE" in agent._system_prompt
    assert '"plan_queue"' in agent._system_prompt


# =================================================================================
# E2E — stub-brain harness driver (pattern from test_watchdog.py)
# =================================================================================


def _make_session(tmp_path: Path, analyzer):
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from inference.framework import solver as duck_solver

    run_session = RunSession(record_intermediate_states=False)
    game = GameAPI(env_name=GAME, arcade_spec=ArcadeSpec(environments_dir=str(ENV_DIR)))
    game.start_game(run_session)

    solver = duck_solver.HarnessSolver(
        label="compact-test",
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
        transcript_path=tmp_path / "transcripts" / f"{GAME}.txt",
        analysis_html_relpath=f"solver_analysis/{GAME}.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "artifacts" / "viewer_data.json",
    )
    return game, session


def _stub_brain(replies: list[str]):
    """A REAL ToolAgent whose only stub is the network call. Each LLM 'reply' is
    a content-only message; the last reply repeats if the loop asks for more."""
    agent = _fresh_agent()
    agent._tool_steps = 1  # one stub LLM iteration per analyzer turn
    calls: list[list[dict]] = []
    pending = list(replies)

    def stub(messages, *, tools=None, request_timeout_seconds=None):
        calls.append(json.loads(json.dumps(messages)))
        content = pending.pop(0) if len(pending) > 1 else pending[0]
        time.sleep(0.02)  # bound the spin rate of no-action turns
        return SimpleNamespace(
            message={"content": content},
            finish_reason="stop",
            usage={"total_tokens": 10},
        )

    agent._chat_completion = stub
    return agent, calls


def _queue_entries(game, count: int) -> list[dict]:
    """Plan steps built from actions the game actually offers right now."""
    from inference.framework.solver import _engine_action_names

    names = _engine_action_names(game)
    assert names, "game offers no actions"
    name = names[0]
    if name == "ACTION6":
        # Distinct click targets so each step is a fresh, executable action.
        return [{"action": "MOUSE", "row": 0, "col": index} for index in range(count)]
    return [{"action": name} for index in range(count)]


def test_e2e_stub_brain_queue_drained_llm_free(tmp_path, monkeypatch):
    """A valid queue is drained by the real play loop with zero extra LLM calls."""
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    monkeypatch.setenv("TAAF_WIN_REPLAY", "0")
    _apply_patch12()
    before = _diag_snapshot()

    placeholder, _ = _stub_brain(["placeholder"])
    game, session = _make_session(tmp_path, placeholder)
    # The plan can only be written once the game exists (it uses the game's own
    # valid actions), so the real stub brain is bound after session creation.
    plan = json.dumps({"plan_queue": _queue_entries(game, 4)})
    agent, llm_calls = _stub_brain([f"Recent findings: probing.\n{plan}"])
    session.analyzer = agent
    session.solver.max_actions_per_game = 4
    session.solver.max_runtime_s_per_game = 60.0  # hard test bound, never reached

    session.play()

    delta = _diag_delta(before)
    assert delta["queue_steps_executed"] >= 3, delta
    assert delta["llm_calls_saved"] >= 3, delta
    assert delta["queue_plans"] >= 1, delta
    executed = sum(1 for rec in game.game_run.history if rec.action.id.name != "RESET")
    assert executed >= 4, "the queue must actually act on the engine"
    # The whole run needed at most a couple of stub calls; the queue did the rest.
    assert len(llm_calls) <= 2, f"queue was not drained LLM-free ({len(llm_calls)} LLM calls)"
    assert game.game_run.final_score is not None, "run must be banked"


def test_e2e_stub_brain_violation_aborts_and_reports(tmp_path, monkeypatch):
    """An induced violation aborts the queue and the report reaches the model."""
    monkeypatch.setenv("TAAF_COMPACT", "1")
    monkeypatch.setenv("TAAF_WATCHDOG", "0")
    monkeypatch.setenv("TAAF_WIN_REPLAY", "0")
    _apply_patch12()
    before = _diag_snapshot()

    agent, _ = _stub_brain(["placeholder"])
    game, session = _make_session(tmp_path, agent)
    steps = _queue_entries(game, 2)
    # Step 1 claims it completes the level — it will not; the queue must abort
    # there and drop step 2.
    steps[0]["expect"] = {"level_completed": True}
    plan = json.dumps({"plan_queue": steps})
    # Reply 1 carries the plan; later replies are plain text (no re-plan), so
    # the run ends via the runtime bound after the report round-trip.
    agent2, llm_calls = _stub_brain([f"Plan below.\n{plan}", "Observing the report."])
    session.analyzer = agent2
    session.solver.max_actions_per_game = 2
    session.solver.max_runtime_s_per_game = 5.0  # bounds the no-action tail

    session.play()

    delta = _diag_delta(before)
    assert delta["queue_aborts"] >= 1, delta
    assert delta["queue_steps_executed"] >= 1, delta
    # The violation report must surface in a prompt the stub brain received.
    reported = any(
        "PLAN QUEUE ABORTED" in json.dumps(call) for call in llm_calls
    )
    assert reported, "violation report never reached the model"
    assert game.game_run.final_score is not None


# =================================================================================
# SCORED-BUNDLE validation (law: _adopt is NOT the eval bytes — commit 7edec38)
# =================================================================================


def test_scored_bundle_has_every_symbol_patch12_wraps():
    """Static check on the scored bytes for each wrapped/read symbol."""
    source = (SCORED_REF / "inference/agent/tool_agent.py").read_text()
    for symbol in (
        "def _persistent_history_messages",
        "def _drop_oldest_history_block",
        "def _keep_recent_history_turns",
        "def _build_user_prompt",
        "def _update_summarized_knowledge_from_assistant",
        "def _update_summarized_knowledge_from_step_summary",
        "def _chat_completion",
        "def _build_system_prompt",
        "def analyze",
        "class AnalyzerTurnResult",
        "def _compact_action_result",
        "def _summarize_step_sequence",
        "request_timeout_seconds",
        "_last_action_result",
        "_max_output_tokens",
    ):
        assert symbol in source, f"scored bundle lost {symbol!r} — re-validate patch12"
    action_names = (SCORED_REF / "inference/agent/action_names.py").read_text()
    assert "def to_engine_action" in action_names


def test_patch12_applies_on_scored_bundle_bytes():
    """Import the SCORED bundle in a subprocess and apply patch 12 there.

    This is the strongest form of the law: the patch must return OK against the
    exact bytes that run at eval, in a process where the drifted tree was never
    imported. Also proves the plan-queue capture/drain path works on those bytes.
    """
    script = f"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {str(SCORED_REF)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import os
os.environ["TAAF_COMPACT"] = "1"
import duck_patches

r1 = duck_patches.patch_compaction()
r2 = duck_patches.patch_plan_queue()
assert r1 == "patch12a compaction: OK", r1
assert r2 == "patch12b plan-queue: OK", r2

from inference.agent.tool_agent import ToolAgent
agent = ToolAgent(model="stub-model", base_url="http://127.0.0.1:9/v1")
agent._session_runtime_dir = Path("/tmp/scored-check")
assert "PLAN QUEUE" in agent._system_prompt

# capture + drain one step on the scored bytes
agent._last_action_result = {{"level": 1}}
agent._update_summarized_knowledge_from_assistant('{{"plan_queue": ["UP"]}}')
result = duck_patches._drain_plan_queue(
    agent, Path("/tmp/scored-check/state.json"), ["ACTION1"],
    lambda args: {{"executed": True, "level": 1, "board_changed": True,
                  "level_completed": False, "game_over": False,
                  "run_complete": False, "state": "NOT_FINISHED"}},
)
assert result is not None and result.step_executed

# compaction on the scored bytes with a stubbed endpoint
agent._chat_completion = lambda messages, tools=None, request_timeout_seconds=None: SimpleNamespace(
    message={{"content": json.dumps({{"facts": ["scored ok"]}})}},
    finish_reason="stop", usage={{"total_tokens": 5}})
os.environ["TAAF_COMPACT_EVERY"] = "1"
duck_patches._note_evictions(agent, [{{"role": "assistant", "content": "old"}}], [])
assert agent._compact12_state["knowledge"]["facts"] == ["scored ok"]
prompt = agent._build_user_prompt(1, valid_actions=["ACTION1"])
assert "scored ok" in prompt
print("SCORED-BUNDLE-OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SCORED-BUNDLE-OK" in proc.stdout


def test_patch12_declines_cleanly_when_bundle_shape_differs(monkeypatch):
    """Missing upstream symbols -> loud FAIL string, no exception, no mutation."""
    from inference.agent import tool_agent

    monkeypatch.delattr(tool_agent.ToolAgent, "_persistent_history_messages")
    msg = duck_patches.patch_compaction()
    assert msg.startswith("patch12a compaction: FAIL"), msg
    monkeypatch.undo()

    monkeypatch.delattr(tool_agent.ToolAgent, "analyze")
    msg = duck_patches.patch_plan_queue()
    assert msg.startswith("patch12b plan-queue: FAIL"), msg
    monkeypatch.undo()

    # apply_all must survive a hostile bundle without raising.
    monkeypatch.setattr(tool_agent, "ToolAgent", None)
    results = [duck_patches.patch_compaction(), duck_patches.patch_plan_queue()]
    assert all("FAIL" in line for line in results), results
