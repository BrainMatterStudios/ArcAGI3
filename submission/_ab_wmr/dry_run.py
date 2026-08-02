#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the ab-wmr kernel logic.

Mock brain (fixed `python` tool call -> action('UP'), plus /models and
logprobs so the serving assert's endpoint checks execute for real) <- REAL
duck harness (scratchpad/taaf_scored_ref bundle, per the 2026-07-26 audit
law) <- REAL CompetitionArcadeServer over repo environment_files <- the REAL
ab_wave_driver (imported from this directory, same file the builder inlines).

Proves before any GPU minute is spent: curated patch application + hard gate,
per-arm env toggling + verification, session registry, serving assert path,
wave loop with fresh competition server per wave, row collection incl.
watchdog/hud/replay diagnostics, in-run scoring, ab_result.json writing, and
the summary table.

Run:  .venv/bin/python submission/_ab_wmr/dry_run.py
"""
from __future__ import annotations

import asyncio
import http.server
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
WORKDIR = Path(os.environ.get("AB_DRY_WORKDIR",
                              str(REPO / "scratchpad" / "ab_dry_run"))) / time.strftime("%H%M%S")

MOCK_REPLY = {
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
            {"token": ":", "logprob": -0.11, "top_logprobs": []},
        ]},
    }],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
}


class MockBrain(http.server.BaseHTTPRequestHandler):
    n_posts = 0

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": "mock-27b"}]})
        else:
            self._send({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        _ = self.rfile.read(n)
        MockBrain.n_posts += 1
        self._send(MOCK_REPLY)


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[dry] mock brain at {base}")

    # env BEFORE importing the duck (tool_agent reads env at import time)
    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base,
        "OPENAI_BASE_URL": base,
        "LOCAL_ANALYZER_MODEL_ID": "mock-27b",
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "ONLY_RESET_LEVELS": "true",
        "TAAF_RUN_AS_SUBMISSION": "0",
        "ARC_ENVIRONMENTS_DIR": str(REPO / "environment_files"),
        "RECORDINGS_DIR": str(WORKDIR / "server_recording"),
        "AB_DRY_RUN": "1",
        "AB_GAMES": "tu93,ls20",
        "AB_WAVES": "A,B",
        "AB_BUDGET": "60",
        "AB_DEADLINE_S": "3000",
    })

    for p in (TAAF_ROOT / "src/ARC3-Inference", TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))
    sys.path.insert(0, str(REPO / "submission/_duck_patched"))
    sys.path.insert(0, str(REPO / "submission/_rig"))

    # --- curated patch application, exactly as the hook cell does -------------
    # patch_hud_sandbox EXCLUDED: measured defect — on this bundle its helper
    # block NameErrors the sandbox bootstrap (no state_hash there) and the duck
    # executes 0 actions. See build_ab_wmr.py APPLY_BLOCK comment.
    import duck_patches as dp
    fns = [dp.patch_action7, dp.patch_animation_producer, dp.patch_animation_metadata,
           dp.verify_reset_already_handled, dp.patch_watchdog,
           dp.patch_hud_board_identity, dp.patch_win_replay]
    results = {}
    for fn in fns:
        line = fn()
        results[fn.__name__] = line
        print(f"[dry][ab-patch] {line}")
    bad = {k: v for k, v in results.items() if "FAIL" in v or "REVIEW" in v}
    assert not bad, f"patch layer did not fully apply: {bad}"

    # sandbox liveness — the check the hook cell also performs
    from inference.agent import python_tool_sandbox as ptx
    sbx = ptx.run_sandboxed_python(
        code="print('sandbox-alive')", timeout_seconds=20,
        initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
        action_handler=lambda actions: {"result": [], "state": {}})
    assert "sandbox-alive" in str(sbx.get("stdout", "")), f"sandbox DEAD: {sbx}"
    print("[dry] sandbox liveness: OK")

    import behav_probe
    assert behav_probe.install(), "probe failed to install"

    def behav_raw():
        return {stem: {k: v for k, v in s.items() if not k.startswith("_")}
                for stem, s in behav_probe._G.items()}

    # --- the real driver (same file the builder inlines) ----------------------
    spec = importlib.util.spec_from_file_location("ab_wave_driver", HERE / "ab_wave_driver.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    from inference.framework import solver as duck_solver
    solver = duck_solver.HarnessSolver(
        label="ab-dry", model="mock-27b", analyzer_timeout=30,
        max_actions_per_game=6, max_runtime_s_per_game=60.0, concurrency=2)
    bm = SimpleNamespace(solver=solver)

    result = asyncio.run(drv.ab_main(
        bm=bm, target=None, working_dir=WORKDIR,
        behav_report=behav_probe.report, behav_raw=behav_raw))

    # --- verify the artifact ---------------------------------------------------
    out = json.loads((WORKDIR / "ab_result.json").read_text())
    assert out["error"] is None, f"driver recorded an error:\n{out['error']}"
    assert out["stage"] == "done", f"stage={out['stage']}"
    sa = {c["check"]: c["ok"] for c in out["serving_assert"]["checks"]}
    assert all(sa.values()) and len(sa) == 4, f"serving assert incomplete: {sa}"
    pp = out["patch_proof"]
    assert pp["watchdog_should_stop_patched"] and pp["play_patched"], pp
    waves = out["waves"]
    assert [w["arm"] for w in waves] == ["A", "B"], waves
    for w in waves:
        exp = w["arm"] == "B"
        assert all(v == exp for v in w["toggles_verified"].values()), w["toggles_verified"]
        assert len(w["rows"]) == 2, f"wave {w['wave']}: {len(w['rows'])} rows"
        for r in w["rows"]:
            assert r["source_game"] is not None and r["actions_total"] > 0, r
            assert r["score"] is not None, r
            assert "watchdog" in r and "hud" in r and "trace_len" in r, sorted(r)
            rp = (r.get("replay") or {}).get("status")
            if w["arm"] == "A":
                assert rp == "disabled", f"arm A replay status {rp!r}"
            else:
                assert rp in ("skipped", "replayed", "aborted"), f"arm B replay status {rp!r}"
        assert "behav_cumulative" in w and "behav_raw_cumulative" in w
    assert MockBrain.n_posts > 0
    print(f"\n[dry] PASS — {MockBrain.n_posts} mock-brain calls, "
          f"artifact at {WORKDIR / 'ab_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
