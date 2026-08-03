"""Pre-push e2e gate: exec the SHIPPED notebook's hook cell the way Kaggle does.

Adapted from the 2026-08-03 zero-submission diagnosis repro (repro_v5_full.py, now
archived at docs/test-artifacts-2026-08-02/zero_diag/). Three properties, verified
against the EXACT artifact we push (duck-patched.ipynb), not the source module:

  1. STALENESS — the notebook's hook cell contains the current duck_patches.py
     bytes verbatim. A forgotten rebuild after editing duck_patches.py shipped the
     stale-notebook class of failure once already (Gate 0, 2026-07-31).
  2. STRUCTURAL — the hook cell execs in a namespace WITHOUT __file__ (notebook
     semantics), on the scored bundle bytes (scratchpad/taaf_scored_ref), in
     submission mode, and NO patch reports FAIL. This is the gate that would have
     caught the shipped patch6 NameError and the patch9 sandbox-shape class.
  3. BEHAVIOURAL — with the full patch set applied, the python-tool sandbox is
     alive and a real Benchmark over a competition-mode arcade drives a mock brain
     to EXECUTE actions in every game (a zero-action run is the fingerprint of the
     never-played 0.00 submission).

The heavy part runs in a subprocess so the scored-bundle imports never collide
with the _adopt tree the rest of the suite imports.

Games: tu93 + ls20 — both proven to execute under the scripted mock brain. ft09
is deliberately NOT used: its mock run banks 0 actions in patched AND control
(mock artifact, see the diagnosis notes), which would fake a red gate.

Run:  .venv/bin/python -m pytest submission/_duck_patched/test_notebook_hook_e2e.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
NOTEBOOK = HERE / "duck-patched.ipynb"
PATCHES = HERE / "duck_patches.py"
SCORED_REF = REPO / "scratchpad/taaf_scored_ref"
ENV_DIR = REPO / "environment_files"

HOOK_SENTINEL = "In-memory harness patches. Inlined from"


def _hook_cell_source() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    hooks = [
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == "code" and HOOK_SENTINEL in "".join(cell.get("source", []))
    ]
    assert len(hooks) == 1, f"expected exactly one hook cell, found {len(hooks)}"
    return hooks[0]


def test_hook_cell_parses_and_is_not_stale():
    """The shipped cell compiles and carries the CURRENT duck_patches.py bytes."""
    src = _hook_cell_source()
    compile(src, "<hook-cell>", "exec")  # SyntaxError -> fail
    body = PATCHES.read_text()
    assert body in src, (
        "duck-patched.ipynb hook cell does not contain the current duck_patches.py "
        "verbatim — rebuild with build_duck_patched.py before pushing"
    )
    assert "_patch_results = apply_all()" in src


@pytest.mark.filterwarnings("ignore")
def test_hook_cell_e2e_submission_mode_mock_game():
    """Full Kaggle-shaped run: exec hook cell (no __file__) on scored-ref bytes,
    submission mode, competition arcade, mock 27B — actions must execute and no
    patch may report FAIL."""
    if not SCORED_REF.is_dir():
        pytest.skip(f"scored bundle bytes absent at {SCORED_REF} (investigate: they "
                    "should exist in the main checkout)")
    if not ENV_DIR.is_dir():
        pytest.skip(f"environment_files absent at {ENV_DIR} (investigate: they "
                    "should exist in the main checkout)")

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-e2e"],
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=str(REPO),
    )
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    assert proc.returncode == 0, (
        f"e2e subprocess failed (rc={proc.returncode})\n--- stdout tail ---\n{tail}"
        f"\n--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
    assert "E2E-GATE-PASS" in proc.stdout, tail


# --------------------------------------------------------------------------------
# Subprocess body (python test_notebook_hook_e2e.py --run-e2e). Kept in this file
# so the gate is one self-contained artifact.
# --------------------------------------------------------------------------------

_MOCK_REPLY = {
    "id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
    "choices": [{
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "Plan: probe with UP.",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "python",
                             "arguments": json.dumps({"code": "action('UP')"})},
            }],
        },
        "logprobs": {"content": [
            {"token": "Plan", "logprob": -0.02, "top_logprobs": []},
        ]},
    }],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
}


def _e2e_main() -> int:
    import asyncio
    import http.server
    import os
    import tempfile
    import threading
    from types import SimpleNamespace

    workdir = Path(tempfile.mkdtemp(prefix="hook_e2e_"))

    class MockBrain(http.server.BaseHTTPRequestHandler):
        n_posts = 0
        system_prompts: list[str] = []

        def log_message(self, *a):  # noqa: D102
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._send({"object": "list", "data": [{"id": "mock-27b"}]})
            else:
                self._send({})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n)
            MockBrain.n_posts += 1
            try:
                payload = json.loads(raw or b"{}")
                for message in payload.get("messages", []):
                    if message.get("role") == "system":
                        MockBrain.system_prompts.append(str(message.get("content", "")))
            except Exception:
                pass  # prompt capture must never break the mock
            self._send(_MOCK_REPLY)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[e2e] mock brain at {base}", flush=True)

    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base,
        "OPENAI_BASE_URL": base,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b",
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true",
        "TAAF_RUN_AS_SUBMISSION": "1",
        "TAAF_MINIMAL_DIAGNOSTICS": "1",
        "ARC_ENVIRONMENTS_DIR": str(ENV_DIR),
        "RECORDINGS_DIR": str(workdir / "server_recording"),
    })

    # The scored bundle bytes — the exact tree sys.path holds at eval.
    for p in (SCORED_REF / "src/ARC3-Inference",
              SCORED_REF / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))

    # --- exec the SHIPPED hook cell with notebook semantics (no __file__) --------
    src = _hook_cell_source()
    cellns: dict = {"__name__": "__main__"}
    exec(compile(src, "<hook-cell>", "exec"), cellns)  # noqa: S102 - mirrors Kaggle
    results = cellns.get("_patch_results")
    assert results, "hook cell did not produce _patch_results"
    fails = [r for r in results if "FAIL" in r]
    print(f"[e2e] patch results: {len(results)} lines", flush=True)
    assert not fails, f"structural patch failures on the shipped artifact: {fails}"

    # --- sandbox liveness after the FULL patch set (the patch9 class) ------------
    from inference.agent import python_tool_sandbox as ptx
    sbx = ptx.run_sandboxed_python(
        code="print('sandbox-alive')", timeout_seconds=30,
        initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
        action_handler=lambda actions: {"result": [], "state": {}})
    assert "sandbox-alive" in str(sbx.get("stdout", "")), (
        f"python-tool sandbox dead after apply_all: {sbx.get('stderr')!r}")
    print("[e2e] sandbox alive after full apply_all", flush=True)

    # --- vision path (must not raise regardless of TAAF_GRID_BURNER) -------------
    from inference.agent import vision_context
    frame = SimpleNamespace(grid=[[(r + c) % 10 for c in range(64)] for r in range(64)])
    url = vision_context.frame_to_png_data_url(frame)
    assert url.startswith("data:image/png;base64,")
    print(f"[e2e] frame render OK ({len(url)} chars)", flush=True)

    # --- drive a real Benchmark over a competition-mode arcade -------------------
    import arc_agi
    import taaf.benchmark
    import taaf.competition_arcade as _ca
    import taaf.game_api
    from inference.framework import solver as duck_solver

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    by_stem = {e.game_id.split("-")[0]: e.game_id for e in arcade.available_environments}
    chosen = [by_stem[g] for g in ("tu93", "ls20") if g in by_stem]
    assert len(chosen) == 2, f"expected tu93+ls20 in {sorted(by_stem)}"
    csrv = _ca.CompetitionArcadeServer(
        game_ids=tuple(chosen), total_runs=len(chosen),
        environments_dir=str(ENV_DIR)).start()
    try:
        games = [taaf.game_api.GameAPI(env_name=c, arcade_spec=csrv.arcade_spec)
                 for c in csrv.exposed_game_ids]
        solver = duck_solver.HarnessSolver(
            label="hook-e2e", model="mock-27b", analyzer_timeout=30,
            max_actions_per_game=8, max_runtime_s_per_game=90.0, concurrency=2)
        bm = taaf.benchmark.Benchmark(
            label="hook_e2e", games=games, solver=solver, n_passes=1)
        asyncio.run(bm.run(soft_end_time=None, runtime_environment=None,
                           minimal_diagnostics=True))
        rows = [
            {
                "game": getattr(gr, "game_id", None),
                "levels": int(getattr(gr, "levels_completed", 0) or 0),
                "actions": len(getattr(gr, "history", []) or []),
                "state": str(getattr(gr, "state", None)),
            }
            for gr in (getattr(bm, "game_runs", None) or [])
        ]
        print(f"[e2e] game rows: {json.dumps(rows)}", flush=True)
        print(f"[e2e] mock-brain calls: {MockBrain.n_posts}", flush=True)
        assert len(rows) == 2, f"expected 2 game runs, got {rows}"
        zero = [r for r in rows if r["actions"] == 0]
        assert not zero, (
            f"zero-action game runs {zero} — the never-played fingerprint of the "
            "0.00 submission; the shipped artifact must not do this")
        assert MockBrain.n_posts > 0, "the benchmark never called the model"

        # --- patch13 prompt plumbing, through the REAL request path ---------------
        # The shipped arm pins TAAF_PLAYBOOK=0 (EXPERIMENT_ENV), so the system
        # prompts the model actually received must NOT carry the playbook...
        assert MockBrain.system_prompts, "no system prompt reached the mock brain"
        assert all(
            "Mechanic playbook" not in p for p in MockBrain.system_prompts
        ), "pinned TAAF_PLAYBOOK=0 arm leaked the playbook into a live prompt"
        # ...while flipping the switch on the SAME scored bytes must inject it
        # (call-time gating through the patched builder chain).
        from inference.agent import tool_agent as _ta
        os.environ["TAAF_PLAYBOOK"] = "1"
        try:
            enabled_prompt = _ta._build_system_prompt(tool_output_tokens=1000)
        finally:
            os.environ["TAAF_PLAYBOOK"] = "0"
        assert "PRECONDITION HUNT" in enabled_prompt, (
            "playbook did not land in the scored-bundle system prompt when enabled"
        )
        overhead = len(enabled_prompt) - len(_ta._build_system_prompt(tool_output_tokens=1000))
        assert overhead <= 1500, f"playbook overhead {overhead} chars (> ~350 tokens)"
        print(f"[e2e] playbook plumbing OK (pin honored; overhead {overhead} chars)",
              flush=True)
    finally:
        csrv.stop()

    print("E2E-GATE-PASS", flush=True)
    return 0


if __name__ == "__main__":
    if "--run-e2e" in sys.argv:
        raise SystemExit(_e2e_main())
    raise SystemExit("usage: pytest this file, or --run-e2e for the subprocess body")
