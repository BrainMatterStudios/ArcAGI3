#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the ab-wmr ROUND 2 kernel logic.

Mock brain (fixed `python` tool call -> action('UP'), a plan_queue block in the
assistant text to exercise patch12b, usage counts so the token accounting is
provable, plus /models and logprobs so the serving assert's endpoint checks
execute for real) <- REAL duck harness (scratchpad/taaf_scored_ref bundle, per
the 2026-07-26 audit law) <- REAL CompetitionArcadeServer over repo
environment_files <- the REAL ab_wave_driver (imported from this directory,
same file the builder inlines).

duck_patches.py is loaded EXACTLY as the kernel loads it: source exec'd into a
module namespace with NO __file__ (the builder inlines it into a notebook
cell), so patch6 declines for want of /kaggle/input/arcagi3-agent here too and
the dry run exercises the kernel's true patch surface (patch9 SKIP included).

Proves before any GPU minute is spent:
  * full v6 apply_all() + the expected-SKIP hard gate (patch9, patch6);
  * per-arm env toggling B/C/D + verification against AB_ARM_ENV;
  * patch11 activation ONLY in C waves (graph diagnostics rows: nodes/edges/
    veto/grinder counters), patch12 activation ONLY in D waves
    (COMPACT_DIAGNOSTICS wave delta: plan accepted + steps drained LLM-free);
  * NONZERO gen_tokens per row (round-1 defect fixed: tokens are summed from
    run.history records; the analyzer's usage-fed session counter is captured
    as a cross-check);
  * session registry, serving assert path, fresh competition server per wave,
    watchdog/hud/replay diagnostics, in-run scoring, incremental
    ab_result.json writes, and the summary table.

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
import types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAAF_ROOT = REPO / "scratchpad/taaf_scored_ref"
WORKDIR = Path(os.environ.get("AB_DRY_WORKDIR",
                              str(REPO / "scratchpad" / "ab_dry_run"))) / time.strftime("%H%M%S")

# The assistant text carries a plan_queue block so patch12b's capture/drain
# path runs for real in D waves (and is provably inert in B/C waves).
MOCK_CONTENT = (
    "Plan: probe with UP.\n"
    '{"plan_queue": [{"action": "UP"}, {"action": "UP"}]}'
)

MOCK_REPLY = {
    "id": "cmpl-mock", "object": "chat.completion", "model": "mock-27b",
    "choices": [{
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": MOCK_CONTENT,
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


def load_duck_patches_like_the_kernel() -> types.ModuleType:
    """Exec duck_patches.py source with NO __file__, as the inlined cell does.

    This makes patch6's path discovery behave exactly as on Kaggle (no repo-src
    fallback), so it SKIPs here for the same reason it SKIPs there.
    """
    source = (REPO / "submission/_duck_patched/duck_patches.py").read_text()
    mod = types.ModuleType("duck_patches")
    mod.__dict__["__name__"] = "duck_patches"
    assert "__file__" not in mod.__dict__
    exec(compile(source, "<inlined duck_patches>", "exec"), mod.__dict__)
    sys.modules["duck_patches"] = mod
    return mod


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
        "AB_WAVES": "B,C,D",
        "AB_BUDGET": "60",
        "AB_DEADLINE_S": "3000",
    })

    for p in (TAAF_ROOT / "src/ARC3-Inference", TAAF_ROOT / "src/tufa-arc-agi-framework/src"):
        assert p.is_dir(), f"missing bundle path {p}"
        sys.path.insert(0, str(p))
    sys.path.insert(0, str(REPO / "submission/_rig"))

    # --- full v6 patch application + the EXACT gate the hook cell uses --------
    dp = load_duck_patches_like_the_kernel()
    results = dp.apply_all()
    expected_skips = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
    bad = [line for line in results
           if "FAIL" in line or "REVIEW" in line
           or ("SKIP" in line and not line.startswith(expected_skips))]
    assert not bad, f"patch layer did not fully apply: {bad}"
    for prefix in expected_skips:
        line = next((l for l in results if l.startswith(prefix)), "")
        assert "SKIP" in line, f"expected {prefix} SKIP on this bundle, got {line!r}"
    print("[dry] patch layer = v6 apply_all(); expected SKIPs verified")

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
    for key in ("watchdog_should_stop_patched", "graph_should_stop_patched",
                "graph_step_env_patched", "hud_or_outer_execute_patched",
                "play_patched", "plan_queue_analyze_patched",
                "compaction_history_patched", "compact_prompt_injector_patched"):
        assert pp.get(key) is True, f"patch proof {key}: {pp}"

    waves = out["waves"]
    assert [w["arm"] for w in waves] == ["B", "C", "D"], waves
    for w in waves:
        arm = w["arm"]
        expected = {k: v == "1" for k, v in drv.AB_ARM_ENV[arm].items()}
        assert w["toggles_verified"] == expected, (arm, w["toggles_verified"])
        wp = w["patch_proof_wave"]
        assert all(wp.get(k) is True for k in wp), (arm, wp)
        assert len(w["rows"]) == 2, f"wave {w['wave']}: {len(w['rows'])} rows"
        assert "compact_diag_delta" in w, sorted(w)
        delta = w["compact_diag_delta"]
        assert set(delta) == set(drv._COMPACT_DIAG_KEYS), delta
        for r in w["rows"]:
            assert r["source_game"] is not None and r["actions_total"] > 0, r
            assert r["score"] is not None, r
            assert "watchdog" in r and "hud" in r and "trace_len" in r, sorted(r)
            # round-1 defect fixed: nonzero token accounting must flow to rows
            assert r["gen_tokens"] > 0, f"gen_tokens still zero: {r}"
            assert r["analyzer_tokens"]["generated"] > 0, r["analyzer_tokens"]
            assert "tokens=" in r["solver_note"], r["solver_note"]
            rp = (r.get("replay") or {}).get("status")
            assert rp in ("skipped", "replayed", "aborted"), f"arm {arm} replay {rp!r}"
            if arm == "C":
                g = r.get("graph")
                assert isinstance(g, dict) and "error" not in g, f"C-wave graph diag: {r}"
                for key in ("nodes", "edges", "vetoes_issued", "vetoes_capped",
                            "grinder_engagements", "grinder_actions",
                            "levels_unlocked_by_grinder"):
                    assert key in g, (key, g)
                assert g["nodes"] >= 1, f"graph recorded no nodes in C wave: {g}"
            else:
                assert "graph" not in r, f"graph state leaked into arm {arm}: {r}"
            if arm == "D":
                assert "compact" in r and "error" not in r["compact"], r.get("compact")
        if arm == "D":
            assert delta["queue_plans"] >= 1, f"plan_queue never captured: {delta}"
            assert delta["queue_steps_executed"] + delta["queue_aborts"] >= 1, delta
            assert delta["llm_calls_saved"] == delta["queue_steps_executed"], delta
        else:
            assert not any(delta.values()), f"compact counters moved in arm {arm}: {delta}"
        assert "behav_cumulative" in w and "behav_raw_cumulative" in w
    assert MockBrain.n_posts > 0
    print(f"\n[dry] PASS — {MockBrain.n_posts} mock-brain calls, "
          f"artifact at {WORKDIR / 'ab_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
