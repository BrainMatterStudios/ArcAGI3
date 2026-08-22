"""Tests for the bugfix-pack graft (submission/_bugfix_pack/graft_bugfix.py).

Because install() mutates module state, every scenario runs in a fresh
subprocess (`python test_bugfix_pack.py --scenario NAME`) with an explicit
environment; the pytest layer just launches scenarios and surfaces output.

Target tree resolution (June stock harness):
1. env BUGFIX_TARGET_TREE
2. the wave-2 srcpull copy of thtennant/taaf-kaggle-source-share-fork
   (byte-identical to the original June source share)
3. the repo-local scratchpad/taaf_scored_ref copy (June stock + our additive
   geodesic postpass; every patched seam is identical)
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

_TREE_CANDIDATES = [
    os.environ.get("BUGFIX_TARGET_TREE", "").strip(),
    (
        "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
        "9a35ebe3-1f42-4aac-97ed-d1de24bf2346/scratchpad/srcpull/"
        "thtennant_taaf-kaggle-source-share-fork/src/ARC3-Inference"
    ),
    str(_REPO / "scratchpad/taaf_scored_ref/src/ARC3-Inference"),
]


def _resolve_tree() -> Path:
    for candidate in _TREE_CANDIDATES:
        if candidate and (Path(candidate) / "inference/agent/tool_agent.py").is_file():
            return Path(candidate)
    raise RuntimeError(f"No June-stock target tree found in: {_TREE_CANDIDATES}")


# ---------------------------------------------------------------------------
# scenario plumbing
# ---------------------------------------------------------------------------

def _bootstrap():
    """Import the target tree + graft inside a scenario subprocess."""
    tree = _resolve_tree()
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(tree))
    import graft_bugfix  # noqa: PLC0415

    return graft_bugfix


def _mk_history(n: int, rows: int = 8, cols: int = 8):
    from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: PLC0415

    entries = []
    for i in range(n):
        grid = tuple(tuple((i + r + c) % 16 for c in range(cols)) for r in range(rows))
        entries.append(HistoryEntry(action=f"a{i}", frame=Frame(grid=grid, step=i, level=1)))
    return entries


def _run_sandbox(code: str):
    from inference.agent.python_tool_sandbox import run_sandboxed_python  # noqa: PLC0415

    return run_sandboxed_python(
        code=code,
        timeout_seconds=15,
        initial_state={
            "current_frame": None,
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
        },
        action_handler=lambda actions: {"action_result": {}, "state": {}},
    )


_IMG_URL = "data:image/png;base64," + "A" * 100_000


def _image_messages():
    return [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current state: step 12, level 1.\n\nCurrent grid image:"},
                {"type": "image_url", "image_url": {"url": _IMG_URL}},
            ],
        },
        {"role": "assistant", "content": "thinking"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current state: step 13, level 1.\n\nCurrent grid image:"},
                {"type": "image_url", "image_url": {"url": _IMG_URL}},
            ],
        },
        {"role": "assistant", "content": "more"},
    ]


def _fake_agent_instance():
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

    inst = object.__new__(agent_mod.ToolAgent)
    inst._context_budget_tokens = 10**9
    return inst


# ---------------------------------------------------------------------------
# scenarios (each asserts and exits non-zero on failure)
# ---------------------------------------------------------------------------

def scenario_install_all() -> None:
    g = _bootstrap()
    from inference.agent import python_tool_sandbox as sandbox_mod  # noqa: PLC0415
    from inference.agent import runtime_state as state_mod  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.agent import prompts as prompts_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415

    status = g.install()
    print(status)
    for name in (
        "sandbox_stderr",
        "sandbox_imports",
        "runtime_state_cap",
        "analyzer_timeout_cap",
        "estimator_images",
        "history_image_strip",
        "click_range_reject",
        "request_error_preserve",
    ):
        assert f"{name}: OK" in status, f"{name} not OK in install status"
    assert "animation_doc: SKIP (seam absent" in status, "animation_doc should skip on June stock"

    # patch 2: real stderr surfaces
    out = sandbox_mod._sanitize_host_error_text("Traceback: KeyError boom\n")
    assert "KeyError boom" in out and "Sandbox process exited unexpectedly." in out, out
    assert sandbox_mod._sanitize_host_error_text("") == "Sandbox process exited unexpectedly."

    # patch 3: difflib importable end-to-end in the real sandbox
    result = _run_sandbox(
        "import difflib\nprint(round(difflib.SequenceMatcher(None, 'abc', 'abd').ratio(), 2))"
    )
    assert not result.get("error"), f"sandbox error: {result!r}"
    assert "0.67" in result.get("stdout", ""), result
    # sys stays blocked
    result = _run_sandbox("import sys")
    assert "not allowed" in result.get("error", ""), result
    # prompt list advertises difflib
    assert "copy, difflib, fractions" in prompts_mod.PYTHON_ADDENDUM
    assert "copy, difflib, fractions" in agent_mod.PYTHON_ADDENDUM

    # patch 4: capped write + payload, no indent, solver namespace rebound
    history = _mk_history(120)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        state_mod.write_runtime_state(path, current_frame=history[-1].frame, history=history)
        payload = json.loads(path.read_text())
        assert len(payload["history"]) == 50, len(payload["history"])
        assert payload["history"][-1]["action"] == "a119"
        assert '\n  "' not in path.read_text(), "indent=2 still present"
    assert solver_mod.write_runtime_state is state_mod.write_runtime_state
    view = agent_mod._ascii_history_view_payload(history)
    assert len(view) == 50, len(view)

    # patch 6: image parts priced at ~64 tokens
    est = agent_mod._estimate_tokens({"messages": _image_messages()})
    assert est < 1200, f"image-aware estimate too big: {est}"
    text_only = {"messages": [{"role": "user", "content": "hello world"}]}
    # identical to stock for text-only payloads
    stock = max(1, (len(json.dumps(text_only, ensure_ascii=True, sort_keys=True, default=str)) + 2) // 3)
    assert agent_mod._estimate_tokens(text_only) == stock

    # patch 7: prior user turns lose images, placeholder names the step
    inst = _fake_agent_instance()
    history_out = agent_mod.ToolAgent._persistent_history_messages(
        inst, _image_messages(), tools=None
    )
    rendered = json.dumps(history_out)
    assert "data:image/" not in rendered, "image survived history strip"
    assert "[frame image for step 12 omitted]" in rendered, rendered[:400]
    assert "[frame image for step 13 omitted]" in rendered

    # patch 8: out-of-range MOUSE rejected with zero-cost error
    dummy = object.__new__(solver_mod._HarnessGameSession)
    actions, error = solver_mod._HarnessGameSession._normalize_actions(
        dummy, {"actions": [{"action": "MOUSE", "row": 99, "col": 3}]}
    )
    assert actions is None and error and "0-63" in error, (actions, error)
    # single-action form too
    actions, error = solver_mod._HarnessGameSession._normalize_actions(
        dummy, {"action": "MOUSE", "row": 2, "col": -5}
    )
    assert actions is None and error and "col=-5" in error, (actions, error)
    # in-range passes through to stock behavior
    actions, error = solver_mod._HarnessGameSession._normalize_actions(
        dummy, {"actions": [{"action": "MOUSE", "row": 5, "col": 7}]}
    )
    assert error is None and len(actions) == 1, (actions, error)
    assert actions[0].data == {"x": 7, "y": 5}
    # prompt sentence present in both namespaces
    for mod in (prompts_mod, agent_mod):
        assert "GRID units 0-63" in mod.MULTIMODAL_CONTEXT_ADDENDUM, mod

    # idempotency: second install reports already applied, not double-wraps
    status2 = g.install()
    assert "already applied" in status2, status2
    print("scenario_install_all: PASS")


def scenario_flags_off() -> None:
    g = _bootstrap()
    from inference.agent import python_tool_sandbox as sandbox_mod  # noqa: PLC0415
    from inference.agent import runtime_state as state_mod  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415

    status = g.install()
    print(status)
    assert status.count("SKIP (flag off)") == 9, status

    # stock behavior fully intact
    assert sandbox_mod._sanitize_host_error_text("boom") == "Sandbox process exited unexpectedly."
    result = _run_sandbox("import difflib")
    assert "not allowed" in result.get("error", ""), result

    history = _mk_history(120)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        state_mod.write_runtime_state(path, current_frame=history[-1].frame, history=history)
        payload = json.loads(path.read_text())
        assert len(payload["history"]) == 120
        assert '\n  "' in path.read_text(), "stock indent=2 missing"
    assert len(agent_mod._ascii_history_view_payload(history)) == 120

    est = agent_mod._estimate_tokens({"messages": _image_messages()})
    assert est > 30_000, f"stock estimator should overcharge images: {est}"

    inst = _fake_agent_instance()
    rendered = json.dumps(
        agent_mod.ToolAgent._persistent_history_messages(inst, _image_messages(), tools=None)
    )
    assert "data:image/" in rendered, "stock history should keep images"

    dummy = object.__new__(solver_mod._HarnessGameSession)
    actions, error = solver_mod._HarnessGameSession._normalize_actions(
        dummy, {"actions": [{"action": "MOUSE", "row": 99, "col": 3}]}
    )
    assert error is None and actions[0].data == {"x": 3, "y": 63}, "stock clamp missing"
    print("scenario_flags_off: PASS")


def scenario_pack_off() -> None:
    g = _bootstrap()
    status = g.install()
    print(status)
    assert status == "bugfix_pack: SKIP (BUGFIX_PACK=0)", status
    print("scenario_pack_off: PASS")


def scenario_call_time_toggle() -> None:
    """Install with everything on, then turn flags off via env: the shipped
    wrappers must become pure pass-throughs (stock behavior)."""
    g = _bootstrap()
    from inference.agent import python_tool_sandbox as sandbox_mod  # noqa: PLC0415
    from inference.agent import runtime_state as state_mod  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415

    status = g.install()
    assert "sandbox_stderr: OK" in status, status
    for name in (
        "SANDBOX_STDERR",
        "RUNTIME_STATE_CAP",
        "ESTIMATOR_IMAGES",
        "HISTORY_IMAGE_STRIP",
        "CLICK_RANGE_REJECT",
    ):
        os.environ[f"BUGFIX_{name}"] = "0"

    assert sandbox_mod._sanitize_host_error_text("boom") == "Sandbox process exited unexpectedly."
    history = _mk_history(80)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        state_mod.write_runtime_state(path, current_frame=history[-1].frame, history=history)
        assert len(json.loads(path.read_text())["history"]) == 80
    assert len(agent_mod._ascii_history_view_payload(history)) == 80
    assert agent_mod._estimate_tokens({"messages": _image_messages()}) > 30_000
    inst = _fake_agent_instance()
    rendered = json.dumps(
        agent_mod.ToolAgent._persistent_history_messages(inst, _image_messages(), tools=None)
    )
    assert "data:image/" in rendered
    dummy = object.__new__(solver_mod._HarnessGameSession)
    actions, error = solver_mod._HarnessGameSession._normalize_actions(
        dummy, {"actions": [{"action": "MOUSE", "row": 99, "col": 3}]}
    )
    assert error is None and actions[0].data == {"x": 3, "y": 63}
    print("scenario_call_time_toggle: PASS")


def scenario_failopen() -> None:
    """Rename/deform every patch target: install() must still succeed and
    report each deformed patch as SKIP, never raise."""
    g = _bootstrap()
    from inference.agent import python_tool_sandbox as sandbox_mod  # noqa: PLC0415
    from inference.agent import runtime_state as state_mod  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.framework import solver as solver_mod  # noqa: PLC0415

    del sandbox_mod._sanitize_host_error_text
    sandbox_mod._SANDBOX_BOOTSTRAP = "print('gutted')"
    del state_mod.write_runtime_state
    del agent_mod.ToolAgent._chat_completion
    del agent_mod._estimate_tokens
    del agent_mod.ToolAgent._persistent_history_messages
    del solver_mod._HarnessGameSession._normalize_actions

    status = g.install()
    print(status)
    expected_skips = (
        "sandbox_stderr: SKIP (missing _sanitize_host_error_text)",
        "sandbox_imports: SKIP (SAFE_MODULES marker absent",
        "runtime_state_cap: SKIP (missing write_runtime_state)",
        "analyzer_timeout_cap: SKIP (missing ToolAgent._chat_completion)",
        "estimator_images: SKIP (missing _estimate_tokens)",
        "history_image_strip: SKIP (missing ToolAgent._persistent_history_messages)",
        "click_range_reject: SKIP (missing _normalize_actions/to_engine_action)",
        "animation_doc: SKIP (seam absent",
        "request_error_preserve: SKIP (missing ToolAgent._chat_completion)",
    )
    for line in expected_skips:
        assert line in status, f"missing: {line}\n{status}"
    assert "FAIL" not in status, status
    print("scenario_failopen: PASS")


def scenario_animation_doc() -> None:
    """June stock lacks the seam; inject the recovered anim-bundle bullet into
    both namespaces and verify the patch rewrites it truthfully."""
    g = _bootstrap()
    from inference.agent import prompts as prompts_mod  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

    # 1) seam absent -> reported skip, nothing changed
    line = g.patch_animation_doc()
    assert "SKIP (seam absent" in line, line

    # 2) simulate the anim-lineage tree
    prompts_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM += g._ANIM_DOC_OLD
    agent_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = (
        prompts_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM
    )
    line = g.patch_animation_doc()
    assert line.startswith("animation_doc: OK"), line
    for mod in (prompts_mod, agent_mod):
        text = mod.STRUCTURED_RUNTIME_STATE_ADDENDUM
        assert g._ANIM_DOC_OLD not in text
        assert "'changes': ['W>R @ (2,3) (2,4)']" in text, "example line missing"
        assert "never call `.get(...)`" in text
    # 3) idempotent
    line = g.patch_animation_doc()
    assert "already applied" in line, line
    print("scenario_animation_doc: PASS")


class _ModelsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.endswith("/models"):
            body = b'{"data": []}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # noqa: D102
        pass


def scenario_analyzer_timeout() -> None:
    g = _bootstrap()
    import requests  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415

    recorded: list = []
    script: list = []  # each entry: "ok" or "timeout"

    def fake_chat(self, messages, *, tools, request_timeout_seconds=None):
        recorded.append(request_timeout_seconds)
        if script and script.pop(0) == "timeout":
            raise requests.ReadTimeout("Read timed out.")
        return "ok"

    agent_mod.ToolAgent._chat_completion = fake_chat
    status = g.patch_analyzer_timeout_cap()
    assert status.startswith("analyzer_timeout_cap: OK"), status

    # dead port for the probe
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]

    inst = object.__new__(agent_mod.ToolAgent)
    inst._model = type("M", (), {"base_url": f"http://127.0.0.1:{dead_port}/v1"})()
    inst._headers = lambda: {}

    # healthy call: timeout passes through untouched, no probe state
    script[:] = ["ok"]
    agent_mod.ToolAgent._chat_completion(inst, [], tools=None, request_timeout_seconds=250.0)
    assert recorded[-1] == 250.0

    # a timeout arms the gate
    script[:] = ["timeout"]
    try:
        agent_mod.ToolAgent._chat_completion(inst, [], tools=None, request_timeout_seconds=250.0)
        raise AssertionError("expected ReadTimeout")
    except requests.ReadTimeout:
        pass
    assert recorded[-1] == 250.0, "first failing attempt keeps its full budget"
    assert inst._bugfix_last_request_timed_out is True

    # retry with the server DEAD: capped at 10s
    script[:] = ["timeout"]
    try:
        agent_mod.ToolAgent._chat_completion(inst, [], tools=None, request_timeout_seconds=250.0)
        raise AssertionError("expected ReadTimeout")
    except requests.ReadTimeout:
        pass
    assert recorded[-1] == 10.0, f"dead-server retry not capped: {recorded[-1]}"

    # retry with the server ALIVE: full budget again (long decodes allowed)
    server = http.server.HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        inst._model = type("M", (), {"base_url": f"http://127.0.0.1:{server.server_port}/v1"})()
        script[:] = ["ok"]
        agent_mod.ToolAgent._chat_completion(inst, [], tools=None, request_timeout_seconds=250.0)
        assert recorded[-1] == 250.0, f"alive-server retry wrongly capped: {recorded[-1]}"
        assert inst._bugfix_last_request_timed_out is False, "success must clear the gate"

        # cleared gate -> next call passes through without probing
        script[:] = ["ok"]
        agent_mod.ToolAgent._chat_completion(inst, [], tools=None, request_timeout_seconds=99.0)
        assert recorded[-1] == 99.0
    finally:
        server.shutdown()

    # flag off at call time: pass-through even when armed
    inst._bugfix_last_request_timed_out = True
    os.environ["BUGFIX_ANALYZER_TIMEOUT_CAP"] = "0"
    script[:] = ["ok"]
    agent_mod.ToolAgent._chat_completion(inst, [], tools=None, request_timeout_seconds=250.0)
    assert recorded[-1] == 250.0
    print("scenario_analyzer_timeout: PASS")


def _pb_tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }


def scenario_request_error_preserve() -> None:
    """Patch 9 end-to-end on the real analyze loop: a RequestException
    mid-turn keeps the completed tool exchanges (and drops a dangling
    assistant tool_call) instead of the stock wholesale revert."""
    g = _bootstrap()
    import requests  # noqa: PLC0415
    from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
    from inference.agent.runtime_state import write_runtime_state  # noqa: PLC0415

    # ---- unit checks on the truncation helper ---------------------------
    sysm = {"role": "system", "content": "sys"}
    user = {"role": "user", "content": "turn prompt"}
    asst = {"role": "assistant", "content": None, "tool_calls": [_pb_tool_call("a", "x")]}
    tool = {"role": "tool", "tool_call_id": "a", "content": "res-a"}
    dangling = {"role": "assistant", "content": None, "tool_calls": [_pb_tool_call("b", "y")]}
    # completed exchange kept, dangling assistant dropped
    out = g._truncate_to_completed_exchanges([sysm, user, asst, tool, dangling])
    assert out == [sysm, user, asst, tool], out
    # partial batch: assistant(a,b) answered only for a -> whole batch dropped,
    # and the then-trailing user prompt goes too (no preserved progress left,
    # so the wrapper falls back to the stock revert)
    batch = {"role": "assistant", "content": None,
             "tool_calls": [_pb_tool_call("a", "x"), _pb_tool_call("b", "y")]}
    out = g._truncate_to_completed_exchanges([sysm, user, batch, tool])
    assert out == [sysm], out
    # trailing user (no progress) dropped down to the system message
    out = g._truncate_to_completed_exchanges([sysm, user])
    assert out == [sysm], out
    # assistant content-only tail is progress and is kept
    content_only = {"role": "assistant", "content": "world model note"}
    followup = {"role": "user", "content": "You have not acted yet."}
    out = g._truncate_to_completed_exchanges([sysm, user, content_only, followup])
    assert out == [sysm, user, content_only], out

    # ---- integration: scripted class-level chat, real analyze/sandbox ---
    script: list = []

    def fake_chat(self, messages, *, tools=None, request_timeout_seconds=None):
        step = script.pop(0)
        if step["kind"] == "raise":
            if step.get("dangling"):
                # simulate a dangling assistant tool_call left in the live
                # turn list at failure time (defensive-lineage case)
                messages.append(
                    {"role": "assistant", "content": None,
                     "tool_calls": [_pb_tool_call("dangling-call", "print('never-ran')")]}
                )
            raise requests.ConnectionError("boom")
        return agent_mod._ChatCompletionResult(
            message=step["message"], finish_reason="tool_calls", usage=None
        )

    agent_mod.ToolAgent._chat_completion = fake_chat
    status = g.patch_request_error_preserve()
    assert status.startswith("request_error_preserve: OK"), status
    assert "already applied" in g.patch_request_error_preserve()

    def make_agent():
        return agent_mod.ToolAgent(
            model="test-model", base_url="http://127.0.0.1:9/v1", provider="vllm"
        )

    def make_state(tmp: str):
        entries = _mk_history(1)
        path = Path(tmp) / "tool_runtime_state.json"
        write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
        return path

    def run_turn(agent, path):
        return agent.analyze(
            path,
            0,
            valid_actions=["ACTION1"],
            step_env=None,
            analysis_step=1,
            request_timeout_seconds=5.0,
            should_stop=lambda: False,
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = make_state(tmp)

        # A) mid-turn failure after one COMPLETED exchange (+ simulated
        #    dangling assistant): completed exchange preserved, dangler gone
        agent = make_agent()
        script[:] = [
            {"kind": "ok", "message": {
                "reasoning": "r1",
                "tool_calls": [_pb_tool_call("c1", "print('probe-result-alpha')")],
            }},
            {"kind": "raise", "dangling": True},
        ]
        result = run_turn(agent, path)
        assert result is not None and result.retryable_failure, result
        hist = agent._history_messages
        rendered = json.dumps(hist)
        assert "probe-result-alpha" in rendered, "completed tool result was not preserved"
        assert '"c1"' in rendered, "assistant tool_call of the completed exchange missing"
        assert "dangling-call" not in rendered, "dangling assistant tool_call survived"
        assert str(hist[0].get("role")) == "user", hist[0]
        roles = [str(m.get("role")) for m in hist]
        assert "assistant" in roles and "tool" in roles, roles

        # B) failure on the FIRST request (no progress): stock revert stands
        agent = make_agent()
        script[:] = [{"kind": "raise"}]
        result = run_turn(agent, path)
        assert result is not None and result.retryable_failure, result
        assert agent._history_messages == [], agent._history_messages

        # C) flag off at call time: pure pass-through, stock revert stands
        os.environ["BUGFIX_REQUEST_ERROR_PRESERVE"] = "0"
        try:
            agent = make_agent()
            script[:] = [
                {"kind": "ok", "message": {
                    "reasoning": "r1",
                    "tool_calls": [_pb_tool_call("c1", "print('probe-off')")],
                }},
                {"kind": "raise"},
            ]
            result = run_turn(agent, path)
            assert result is not None and result.retryable_failure, result
            assert agent._history_messages == [], agent._history_messages
        finally:
            os.environ.pop("BUGFIX_REQUEST_ERROR_PRESERVE", None)
    print("scenario_request_error_preserve: PASS")


_SCENARIOS = {
    "install_all": scenario_install_all,
    "flags_off": scenario_flags_off,
    "pack_off": scenario_pack_off,
    "call_time_toggle": scenario_call_time_toggle,
    "failopen": scenario_failopen,
    "animation_doc": scenario_animation_doc,
    "analyzer_timeout": scenario_analyzer_timeout,
    "request_error_preserve": scenario_request_error_preserve,
}


# ---------------------------------------------------------------------------
# pytest layer — each test launches one scenario subprocess
# ---------------------------------------------------------------------------

def _launch(name: str, env_overrides: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("BUGFIX_"):
            env.pop(key)
    env.update(env_overrides or {})
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--scenario", name],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"scenario {name} failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    assert f"scenario_{name}: PASS" in proc.stdout, proc.stdout


def test_install_all_patched_behavior():
    _launch("install_all")


def test_flags_off_leaves_stock_intact():
    _launch(
        "flags_off",
        {f"BUGFIX_{n.upper()}": "0" for n in (
            "animation_doc",
            "sandbox_stderr",
            "sandbox_imports",
            "runtime_state_cap",
            "analyzer_timeout_cap",
            "estimator_images",
            "history_image_strip",
            "click_range_reject",
            "request_error_preserve",
        )},
    )


def test_master_kill_switch():
    _launch("pack_off", {"BUGFIX_PACK": "0"})


def test_call_time_flag_toggle():
    _launch("call_time_toggle")


def test_failopen_on_renamed_targets():
    _launch("failopen")


def test_animation_doc_rewrite():
    _launch("animation_doc")


def test_analyzer_timeout_gate():
    _launch("analyzer_timeout")


def test_request_error_preserve():
    _launch("request_error_preserve")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        _SCENARIOS[sys.argv[2]]()
        sys.exit(0)
    print("usage: test_bugfix_pack.py --scenario NAME | run under pytest")
    sys.exit(2)
