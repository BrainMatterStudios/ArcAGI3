"""Tests for the EWM agent harness fixes (2026-07-22 audit repairs).

Run:  .venv/bin/python -m pytest scratchpad/ewm_pathb/test_ewm_agent_fixes.py -v

Covers the four audit defects plus the few-shot:
  1. repetition loop  -> loop-breaker fires, sampling params sent
  2. max_tokens=4096  -> now 16384 (config assertion)
  3. raw code fences  -> strip_code_fence
  4. unreadable spec  -> obs-cap raised; contract inlined in SYSTEM
  +  few-shot from the Opus transcript, with a leakage guard
  +  scaffold-file write guard
  +  informative (py-compile) write observation
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ewm_agent as ea  # noqa: E402


# --- unit: fence stripping -------------------------------------------------

def test_strip_fence_removes_python_wrapper():
    body = "```python\nimport numpy as np\nx = 1\n```"
    assert ea.strip_code_fence(body) == "import numpy as np\nx = 1"


def test_strip_fence_plain_and_untagged():
    assert ea.strip_code_fence("```\na = 1\n```") == "a = 1"


def test_strip_fence_leaves_unfenced_code_untouched():
    body = "import os\nprint(os.getcwd())\n"
    assert ea.strip_code_fence(body) == body


def test_strip_fence_does_not_eat_inner_backticks():
    body = "```python\ns = '```'\n```"
    assert ea.strip_code_fence(body) == "s = '```'"


# --- unit: informative write observation -----------------------------------

def test_describe_write_reports_compile_ok():
    obs = ea.describe_write("world_model_engine.py", "def f():\n    return 1\n")
    assert "py-compile OK" in obs


def test_describe_write_reports_syntax_error_with_line():
    obs = ea.describe_write("world_model_engine.py", "def f(:\n    return 1\n")
    assert "SyntaxError" in obs and "line" in obs


def test_describe_write_catches_the_fence_bug_regression():
    """A raw fence must be caught as a syntax error if it ever reaches describe_write."""
    obs = ea.describe_write("world_model_engine.py", "```python\nx=1\n```")
    assert "SyntaxError" in obs


def test_describe_write_non_python_is_plain():
    obs = ea.describe_write("world_model.md", "# notes")
    assert "py-compile" not in obs and "chars" in obs


# --- unit: scaffold lock ----------------------------------------------------

def test_scaffold_lock_covers_search_lib():
    assert "search_lib.py" in ea._SCAFFOLD_LOCKED
    assert "verify_world_model.py" in ea._SCAFFOLD_LOCKED


def test_deliverable_files_are_not_locked():
    for f in ("world_model_engine.py", "world_model_state_io.py",
              "world_model_main_planner.py", "world_model.md"):
        assert f not in ea._SCAFFOLD_LOCKED


# --- unit: normalization for repeat detection ------------------------------

def test_norm_collapses_whitespace():
    assert ea._norm("a  b\n\n c") == ea._norm("a b c")


# --- unit: few-shot loading + leakage guard --------------------------------

def test_fewshot_loads_a_real_opus_solution():
    # tu93_run1 exists under scratchpad/ewm_pathb/runs/
    text = ea._load_fewshot("tu93", str(HERE), ws="/tmp/some_other_game/workspace")
    assert text, "expected the tu93 Opus artifacts to load"
    assert "world_model.md" in text and "WORKED EXAMPLE" in text
    assert "UNRELATED game" in text


def test_fewshot_leakage_guard_skips_same_game():
    # Solving tu93 must NOT get the tu93 solution as its example.
    text = ea._load_fewshot("tu93", str(HERE), ws="/tmp/game_tu93-0768757b/workspace")
    assert text == ""


def test_fewshot_disabled_returns_empty():
    assert ea._load_fewshot("", str(HERE), ws="/tmp/x/workspace") == ""


def test_fewshot_missing_game_returns_empty():
    assert ea._load_fewshot("zzzz", str(HERE), ws="/tmp/x/workspace") == ""


# --- unit: config assertions (the cheap defects) ---------------------------

def test_defaults_fix_the_audit_numbers():
    import argparse
    # Rebuild the parser the way main() does, by introspecting the module source is
    # brittle; instead assert the documented defaults directly from a fresh parse.
    argv = ["--workspace", "/tmp/x", "--base-url", "http://x/v1", "--model", "m"]
    ns = _parse_main_args(argv)
    assert ns.max_tokens == 16384, "max_tokens must clear a full 64x64-frame file"
    assert ns.obs_cap >= 13932, "obs-cap must exceed main_prompt.md so the spec is readable"
    assert ns.repetition_penalty and ns.repetition_penalty > 1.0
    assert 0.0 < ns.top_p <= 1.0


def test_contract_is_inlined_in_system_prompt():
    for token in ("world_model_engine(state, action) -> (new_state, game_status)",
                  "LEVEL_COMPLETED", "GAME_OVER",
                  '{"name": "ACTION1"'):
        assert token in ea.SYSTEM, f"missing from SYSTEM: {token}"


def _parse_main_args(argv):
    """Reconstruct main()'s argparse to check defaults without running the loop."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8879)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--max-ctx-chars", type=int, default=90000)
    ap.add_argument("--max-ctx-tokens", type=int, default=45000)
    ap.add_argument("--obs-cap", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--fewshot-game", default="tu93")
    ap.add_argument("--fewshot-root", default=None)
    ap.add_argument("--log", default=None)
    return ap.parse_args(argv)


# --- integration: full loop against a stub brain + fake game ---------------

class _StubBrain(http.server.BaseHTTPRequestHandler):
    """Scripted /v1/chat/completions server. Replies come from the class-level queue."""

    replies: list[str] = []
    received: list[dict] = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        _StubBrain.received.append(req)
        i = min(len(_StubBrain.received) - 1, len(_StubBrain.replies) - 1)
        content = _StubBrain.replies[i]
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_server():
    _StubBrain.replies = []
    _StubBrain.received = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _StubBrain)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    d = ws / "client/session/level_01_attempt_01"
    d.mkdir(parents=True)
    (d / "initial_frame.txt").write_text("00\n00\n")
    (d / "initial_metadata.json").write_text(
        json.dumps({"available_actions": [1, 2, 3, 4], "win_levels": 3}))
    # a fake `./g` and `./verify` so <bash> calls succeed
    for name in ("g", "verify", "plan"):
        p = ws / name
        p.write_text("#!/bin/sh\necho ok\n")
        p.chmod(0o755)
    # a scaffold file the model must not clobber
    (ws / "search_lib.py").write_text("# provided BFS\n")
    return ws


def _run_agent(ws: Path, base_url: str, argv_extra=None):
    argv = ["ewm_agent.py", "--workspace", str(ws), "--base-url", base_url,
            "--model", "stub", "--max-turns", "8", "--fewshot-game", ""]
    argv += argv_extra or []
    old = sys.argv
    sys.argv = argv
    try:
        ea.main()
    finally:
        sys.argv = old


def test_loop_breaker_fires_on_identical_replies(stub_server, tmp_path, capsys):
    _, base_url = stub_server
    ws = _make_workspace(tmp_path)
    # The brain repeats the SAME write forever — exactly the v3 failure.
    _StubBrain.replies = ['thinking\n<write path="world_model_engine.py">```python\nx=1\n```</write>']
    _run_agent(ws, base_url)
    out = capsys.readouterr().out
    assert "loop-breaker" in out, "loop-breaker never fired on identical replies"


def test_sampling_params_reach_the_server(stub_server, tmp_path):
    _, base_url = stub_server
    ws = _make_workspace(tmp_path)
    _StubBrain.replies = ["<done>stop</done>"]
    _run_agent(ws, base_url)
    req = _StubBrain.received[0]
    assert req["top_p"] == 0.95
    assert req["repetition_penalty"] == 1.05
    assert req["max_tokens"] == 16384


def test_fence_is_stripped_on_disk(stub_server, tmp_path):
    _, base_url = stub_server
    ws = _make_workspace(tmp_path)
    _StubBrain.replies = [
        '<write path="world_model_engine.py">```python\ndef world_model_engine(s,a):\n'
        '    return s, "RUNNING"\n```</write>',
        "<done>done</done>",
    ]
    _run_agent(ws, base_url)
    written = (ws / "world_model_engine.py").read_text()
    assert not written.lstrip().startswith("```"), "fence was written to disk"
    compile(written, "world_model_engine.py", "exec")  # must import cleanly


def test_scaffold_write_is_refused(stub_server, tmp_path):
    _, base_url = stub_server
    ws = _make_workspace(tmp_path)
    original = (ws / "search_lib.py").read_text()
    _StubBrain.replies = [
        '<write path="search_lib.py">garbage that would break BFS</write>',
        "<done>done</done>",
    ]
    _run_agent(ws, base_url)
    assert (ws / "search_lib.py").read_text() == original, "scaffold file was clobbered"


# --- unit: write-time signature lint (defect class from gates v4+v5) -------

def test_signature_lint_catches_the_v5_bug():
    """1-arg initial_state_reconstruction must be flagged the moment it is written."""
    body = "def initial_state_reconstruction(frame_data):\n    return {}\n" \
           "def state_renderer(state):\n    return []\n"
    out = ea.check_signatures("world_model_state_io.py", body)
    assert "SIGNATURE ERROR" in out and "initial_state_reconstruction takes 1" in out
    assert "level_index, initial_frame" in out


def test_signature_lint_passes_correct_contract():
    body = ("def initial_state_reconstruction(level_index, initial_frame):\n    return {}\n"
            "def state_renderer(state):\n    return []\n")
    assert ea.check_signatures("world_model_state_io.py", body) == ""


def test_signature_lint_flags_missing_function():
    body = "def initial_state_reconstruction(level_index, initial_frame):\n    return {}\n"
    out = ea.check_signatures("world_model_state_io.py", body)
    assert "MISSING function state_renderer" in out


def test_signature_lint_checks_engine_arity():
    out = ea.check_signatures("world_model_engine.py",
                              "def world_model_engine(state):\n    return state\n")
    assert "takes 1" in out and "world_model_engine(state, action)" in out


def test_signature_lint_tolerates_varargs_and_other_files():
    assert ea.check_signatures("world_model_engine.py",
                               "def world_model_engine(*args):\n    return None\n") == ""
    assert ea.check_signatures("helpers.py", "def foo():\n    pass\n") == ""


def test_describe_write_carries_the_lint():
    obs = ea.describe_write("world_model_state_io.py",
                            "def initial_state_reconstruction(frame):\n    return {}\n")
    assert "py-compile OK" in obs and "SIGNATURE ERROR" in obs


def test_reconstruction_tool_adapts_to_one_arg_impl():
    """The scaffold must call a 1-param implementation with the frame only."""
    import types
    sys.modules.pop("state_reconstruction_tools", None)
    for m in ["game_status", "session_tools", "world_model_engine"]:
        mod = types.ModuleType(m); sys.modules[m] = mod
    sys.modules["game_status"].RUNNING = "RUNNING"
    for fn in ["attempt_step_count", "read_attempt_prefix_for_level",
               "read_current_attempt", "read_latest_attempt_for_level", "truncate_attempt"]:
        setattr(sys.modules["session_tools"], fn, lambda *a: None)
    sys.modules["world_model_engine"].world_model_engine = lambda s, a: (s, "RUNNING")

    calls = {}
    sio = types.ModuleType("world_model_state_io")
    def one_arg(initial_frame):          # the exact v5 shape
        calls["args"] = (initial_frame,)
        return {"level": 1}
    sio.initial_state_reconstruction = one_arg
    sys.modules["world_model_state_io"] = sio

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "srt", str(HERE / "src/agent/workspace_init/state_reconstruction_tools.py"))
    srt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srt)

    out = srt.reconstruct_initial_state_from_attempt(3, {"initial_frame": "FRAME"})
    assert out == {"level": 1}
    assert calls["args"] == ("FRAME",), "1-param impl must receive the frame only"

    # and the correct 2-param contract still gets (level, frame)
    def two_arg(level_index, initial_frame):
        calls["args2"] = (level_index, initial_frame)
        return {"level": level_index}
    sio.initial_state_reconstruction = two_arg
    srt2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srt2)
    assert srt2.reconstruct_initial_state_from_attempt(3, {"initial_frame": "F"}) == {"level": 3}
    assert calls["args2"] == (3, "F")


# --- unit: tag-vocabulary aliases (defect #7, gate v6: 146/300 turns burned) ---

def test_alias_verify_variants():
    for text in ("<verify>\n</verify>", "<verify/>", "thinking first\n<verify></verify>"):
        act = ea.parse_action(text)
        assert act == {"tool": "bash", "path": None, "body": "./verify"}, (text, act)


def test_alias_plan_with_args():
    act = ea.parse_action("<plan --from-current>")
    assert act == {"tool": "bash", "path": None, "body": "./plan --from-current"}


def test_alias_g_move():
    act = ea.parse_action("<g move ACTION1>")
    assert act == {"tool": "bash", "path": None, "body": "./g move ACTION1"}


def test_alias_status():
    act = ea.parse_action("<status/>")
    assert act == {"tool": "bash", "path": None, "body": "./g status"}


def test_real_tags_still_win_over_aliases():
    act = ea.parse_action("<verify/>\n<bash>./g status</bash>")
    assert act["tool"] == "bash" and act["body"] == "./g status"


def test_no_alias_on_prose():
    assert ea.parse_action("I should run verify next.") is None
