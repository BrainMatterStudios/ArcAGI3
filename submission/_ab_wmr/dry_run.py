#!/usr/bin/env python3
"""GPU-free end-to-end dry run of the ab-wmr ROUND 5 (K3 run-8 adapter eval)
logic.

Mock brain (fixed `python` tool call -> action('UP'), a fixed "World model:"
scientist note, usage counts so the token accounting is provable, /models and
logprobs so the serving assert's endpoint checks execute for real — and the
LOGPROB VALUES keyed off AB_MOCK_MODEL, the env var the driver's dry-run serve
stub publishes, so the M and B fingerprints genuinely differ and the
weights-identity gate is exercised end-to-end) <- REAL duck harness
(scratchpad/taaf_scored_ref bundle, per the 2026-07-26 audit law) <- REAL
CompetitionArcadeServer over repo environment_files <- the REAL ab_wave_driver
(imported from this directory, same file the builder inlines).

duck_patches.py is loaded EXACTLY as the kernel loads it: source exec'd into a
module namespace with NO __file__ (the builder inlines it into a notebook
cell), so patch6 declines for want of /kaggle/input/arcagi3-agent here too and
the dry run exercises the kernel's true patch surface (patch9 SKIP included).

The merge itself cannot run locally (no snapshot/GPU); its script logic is
validated at build time (ast.parse) and its PATH RESOLUTION block
(build_ab_wmr.ADAPTER_SELECT) is exec'd here verbatim against a mock
/kaggle/input reproducing the REAL arc3-sft-k3-ckpts file listing (fetched
via the kaggle CLI 2026-08-04; trainer_state.json content is the actual
downloaded run-8 record), plus tampered trees proving every abort path:
duplicate checkpoint-8, wrong-dataset checkpoint-8, non-run-8 trainer_state,
wrong adapter byte size, missing corpus.

Proves before any GPU minute is spent:
  * full v6 apply_all() + the expected-SKIP hard gate (patch9, patch6);
  * ARM SCHEDULING with per-wave serve records: M and B waves resolve to
    DIFFERENT model paths, the serve stub publishes them, and the per-arm
    serving assert runs per wave (cmdline check relaxed only for dry run);
  * WEIGHTS-IDENTITY GATE: M and B fingerprints differ in-run, identical
    fingerprints raise (direct negative test of the gate function);
  * both arms mechanically IDENTICAL in config: same env pins verified, the
    playbook text ABSENT from real sampled system prompts in BOTH arms, graph
    state absent, compact counters frozen (GRAPH=0/COMPACT=0/PLAYBOOK=0);
  * patch14 antifreeze diagnostics plumbing (per-wave delta + per-row
    counters + deterministic direct trigger exercise);
  * NONZERO gen_tokens per row, HUD mask stats per row (m0r0 on the dry
    panel), session registry, fresh competition server per wave,
    watchdog/replay diagnostics, in-run scoring, incremental ab_result.json
    writes (with the pre-registered QUESTION header), and the summary table.

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

# The assistant text carries a fixed "World model:" scientist note so the
# agent's knowledge carry is non-empty and constant — the input patch14's
# freeze detector hashes. (Round 3 pins COMPACT=0 in both arms, so no
# plan_queue block: compact counters must stay frozen everywhere.)
MOCK_CONTENT = (
    "World model: mock static world for the dry run.\n"
    "Plan: probe with UP."
)

def mock_reply():
    """Reply with logprobs keyed off AB_MOCK_MODEL (set by the driver's
    dry-run serve stub): 'merged' weights answer with different numbers than
    base weights, so the weights-identity gate sees genuinely distinct
    fingerprints — exactly what a real adapter-vs-base serve would produce."""
    offset = 0.0 if "merged" in os.environ.get("AB_MOCK_MODEL", "") else -0.5
    return {
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
                {"token": "Plan", "logprob": -0.02 + offset, "top_logprobs": []},
                {"token": ":", "logprob": -0.11 + offset, "top_logprobs": []},
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
        self._send(mock_reply())


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
        "AB_GAMES": "m0r0,tu93",   # round-4 panel members; m0r0 = the fixed HUD class
        "AB_WAVES": "M,B",
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
    assert out["experiment"] == "ab_round5_k3_adapter", out["experiment"]
    prr = out.get("pre_registered_reading")
    assert isinstance(prr, dict) and set(prr) == {
        "question", "primary", "secondary", "not_a_readout",
        "adapter_provenance"}, prr
    assert "run-8" in prr["question"], prr["question"]
    assert "gen_tokens" in prr["primary"], prr["primary"]
    assert "NLL is NOT a criterion" in prr["not_a_readout"], prr["not_a_readout"]
    assert "checkpoint-8" in prr["adapter_provenance"], prr["adapter_provenance"]
    assert out["target_games"] == list(drv._AB_TARGET_GAMES), out.get("target_games")
    assert set(out["arm_models"]) == {"M", "B"}, out.get("arm_models")
    assert out.get("antifreeze_counter_installed") is True, out

    pp = out["patch_proof"]
    for key in ("watchdog_should_stop_patched", "graph_should_stop_patched",
                "graph_step_env_patched", "hud_or_outer_execute_patched",
                "play_patched", "plan_queue_analyze_patched",
                "compaction_history_patched", "compact_prompt_injector_patched",
                "playbook_system_prompt_patched", "antifreeze_user_prompt_patched"):
        assert pp.get(key) is True, f"patch proof {key}: {pp}"

    waves = out["waves"]
    assert [w["arm"] for w in waves] == ["M", "B"], waves
    fps = {}
    for w in waves:
        arm = w["arm"]
        expected = {k: v == "1" for k, v in drv.AB_ARM_ENV[arm].items()}
        assert set(expected) >= {"TAAF_PLAYBOOK", "TAAF_ANTIFREEZE"}, expected
        assert w["toggles_verified"] == expected, (arm, w["toggles_verified"])
        wp = w["patch_proof_wave"]
        assert all(wp.get(k) is True for k in wp), (arm, wp)
        assert {"playbook_system_prompt_patched",
                "antifreeze_user_prompt_patched"} <= set(wp), (arm, sorted(wp))

        # -- round 4: per-wave serve record + per-arm serving assert ----------
        srv_rec = w["serve"]
        assert srv_rec["dry_run"] is True and srv_rec["restarted"] is False, srv_rec
        want_model = "merged_sft" if arm == "M" else "qwen3-6-27b-fp8"
        assert want_model in srv_rec["desired_model"], (arm, srv_rec)
        assert w["served_model"] == srv_rec["desired_model"], (arm, w["served_model"])
        sa = {c["check"]: c["ok"] for c in w["serving_assert"]["checks"]}
        assert all(sa.values()) and len(sa) == 4, f"arm {arm} serving assert: {sa}"
        fp = w["weights_fingerprint"]
        assert fp and fp["tokens"] and fp["logprobs"], (arm, fp)
        fps[arm] = fp

        # -- playbook ABSENCE proof: OFF in BOTH arms this round --------------
        pb = w["playbook_proof"]
        assert pb["expected_present"] is False, (arm, pb)
        assert pb["n_prompts_sampled"] >= 1 and pb["n_mismatches"] == 0, (arm, pb)
        assert pb["probe_before_wave"]["present"] is False, (arm, pb)
        assert pb["sample_sha256"], (arm, pb)

        # -- round 3: antifreeze diagnostics plumbing --------------------------
        assert "antifreeze_diag_delta" in w, sorted(w)
        assert isinstance(w["antifreeze_diag_delta"]["triggers"], int), w
        assert w["antifreeze_diag_delta"]["triggers"] >= 0, w

        assert len(w["rows"]) == 2, f"wave {w['wave']}: {len(w['rows'])} rows"
        delta = w["compact_diag_delta"]
        assert set(delta) == set(drv._COMPACT_DIAG_KEYS), delta
        # COMPACT=0 and GRAPH=0 in BOTH arms: counters frozen, no graph state.
        assert not any(delta.values()), f"compact counters moved in arm {arm}: {delta}"
        for r in w["rows"]:
            assert r["source_game"] is not None and r["actions_total"] > 0, r
            assert r["score"] is not None, r
            assert "watchdog" in r and "hud" in r and "trace_len" in r, sorted(r)
            assert "graph" not in r, f"graph state leaked into arm {arm}: {r}"
            # HUD mask stats must be collected per row (m0r0 mask readout)
            hud = r["hud"]
            assert "error" not in hud, hud
            assert isinstance(hud.get("mask_cells"), int), hud
            assert isinstance(hud.get("confirmed_lines"), int), hud
            # per-row antifreeze trigger counter
            af = r.get("antifreeze")
            assert isinstance(af, dict) and "error" not in af, (arm, r)
            assert isinstance(af["triggers"], int) and af["triggers"] >= 0, af
            # round-1 defect fixed: nonzero token accounting must flow to rows
            assert r["gen_tokens"] > 0, f"gen_tokens still zero: {r}"
            assert r["analyzer_tokens"]["generated"] > 0, r["analyzer_tokens"]
            assert "tokens=" in r["solver_note"], r["solver_note"]
            rp = (r.get("replay") or {}).get("status")
            assert rp in ("skipped", "replayed", "aborted"), f"arm {arm} replay {rp!r}"
        assert "behav_cumulative" in w and "behav_raw_cumulative" in w

    # --- round 4: weights-identity gate ---------------------------------------
    # In-run: the M and B fingerprints must genuinely differ (the mock keys its
    # logprobs off the served model path, as real weights would).
    assert not drv._ab_fingerprints_identical(fps["M"], fps["B"]), \
        f"M and B fingerprints identical in dry run: {fps}"
    # Direct negative test: identical fingerprints across arms must ABORT.
    fa = {"tokens": ["x", "y"], "logprobs": [-0.1, -0.2]}
    assert drv._ab_fingerprints_identical(fa, json.loads(json.dumps(fa)))
    saved = dict(drv._AB_FINGERPRINTS)
    drv._AB_FINGERPRINTS.clear()
    drv._AB_FINGERPRINTS["B"] = fa
    try:
        drv._ab_fingerprint_gate("M", dict(fa))
        raise AssertionError("identity gate FAILED to raise on identical fingerprints")
    except RuntimeError as e:
        assert "WEIGHTS-IDENTITY GATE" in str(e), e
    drv._AB_FINGERPRINTS.clear()
    drv._AB_FINGERPRINTS.update(saved)
    print("[dry] weights-identity gate: fingerprints differ in-run, "
          "identical fingerprints raise: OK")

    # --- round 4: serving-assert ABORT path ------------------------------------
    # A dead endpoint must raise (mis-served wave aborts before burning its
    # hour), not degrade to a warning.
    real_url = os.environ["LOCAL_ANALYZER_BASE_URL"]
    os.environ["LOCAL_ANALYZER_BASE_URL"] = "http://127.0.0.1:9/v1"  # nothing listens
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
    try:
        drv.ab_serving_assert(WORKDIR, "M")
        raise AssertionError("serving assert FAILED to raise on a dead endpoint")
    except RuntimeError as e:
        assert "serving assert FAILED" in str(e), e
    finally:
        os.environ["LOCAL_ANALYZER_BASE_URL"] = real_url
        os.environ["OPENAI_BASE_URL"] = real_url
    print("[dry] serving-assert abort on dead endpoint: OK")

    # --- round 4: deadline guard skips TRAILING waves ---------------------------
    # With an exhausted budget every wave is skipped as 'deadline' (the guard
    # drops from the tail because elapsed only grows — M,B,B,M degrades to
    # M,B,B, never to an unpaired design).
    guard_dir = WORKDIR / "guard"
    guard_dir.mkdir(exist_ok=True)
    guard = asyncio.run(drv.ab_main(
        bm=bm, target=None, working_dir=guard_dir,
        notebook_start=time.time() - (drv.AB_DEADLINE_S + 60),
        behav_report=behav_probe.report, behav_raw=behav_raw))
    assert guard["error"] is None, guard["error"]
    skipped = [(w.get("wave"), w.get("arm"), w.get("skipped")) for w in guard["waves"]]
    assert all(s == "deadline" for _, _, s in skipped) and len(skipped) == 2, skipped
    print("[dry] deadline guard skip path: OK", skipped)

    # --- deterministic exercise of the antifreeze trigger + counting wrapper ---
    # (in-wave firing depends on the game board actually freezing; this proves
    # the trigger path and the driver's per-agent attribution deterministically)
    mod = drv._ab_patch_module()
    assert mod is not None and getattr(mod._antifreeze_note, "_ab_counted", False), \
        "driver's antifreeze counting wrapper is not installed on the patch module"
    before = mod.ANTIFREEZE_DIAGNOSTICS["triggers"]

    class _FakeAgent:
        pass

    fake = _FakeAgent()
    fake._summarized_knowledge = {"world_model": "frozen hypothesis"}
    notes = [mod._antifreeze_note(fake, None) for _ in range(4)]
    assert notes[:3] == ["", "", ""] and notes[3], notes
    assert mod.ANTIFREEZE_DIAGNOSTICS["triggers"] == before + 1, \
        (before, mod.ANTIFREEZE_DIAGNOSTICS)
    assert fake.__dict__.get("_ab_antifreeze_triggers") == 1, fake.__dict__
    print("[dry] antifreeze trigger + per-agent attribution: OK")

    # --- round 5: ADAPTER PATH RESOLUTION against the real dataset listing -----
    # Exec build_ab_wmr.ADAPTER_SELECT (the exact block the merge subprocess
    # runs) against a mock /kaggle/input reproducing the REAL
    # arc3-sft-k3-ckpts + arc3-sft-k3-corpus listings (kaggle CLI,
    # 2026-08-04) in the nested mount layout the run-8 kernel log showed
    # (/kaggle/input/datasets/<owner>/<slug>/...), then against tampered
    # trees to prove every abort path.
    import glob as glob_mod
    spec_b = importlib.util.spec_from_file_location("build_ab_wmr", HERE / "build_ab_wmr.py")
    bld = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(bld)

    RUN8_LOSS8 = 0.7335078716278076   # downloaded trainer_state, 2026-08-04
    RUN8_BYTES = 467062560            # live listing, 2026-08-04

    def mk_tree(root: Path, *, corpus=True, k3_ckpt=True, synth_ckpt8=False,
                loss8=RUN8_LOSS8, adapter_bytes=RUN8_BYTES) -> Path:
        snap = root / "datasets/driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
        if corpus:
            corp = root / "datasets/ahmedmobasher86/arc3-sft-k3-corpus"
            (corp / "tokenizer_bundle").mkdir(parents=True)
            (corp / "train.jsonl").write_text("{}\n")
            (corp / "sft_common.py").write_text("# mock\n")
            # present in the real listing; must NOT be picked as MODEL
            (corp / "tokenizer_bundle/config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}))
        if k3_ckpt:
            ck = root / "datasets/ahmedmobasher86/arc3-sft-k3-ckpts/sft_out/checkpoint-8"
            ck.mkdir(parents=True)
            (ck / "trainer_state.json").write_text(json.dumps(
                {"global_step": 8, "max_steps": 15,
                 "log_history": [{"step": 8, "loss": loss8}]}))
            with open(ck / "adapter_model.safetensors", "wb") as f:
                f.truncate(adapter_bytes)   # sparse file with the real size
            # sibling step-15 artifact from the real listing; must NOT be selected
            sa = ck.parents[1] / "sft_adapter"
            sa.mkdir(exist_ok=True)
            (sa / "adapter_config.json").write_text("{}")
        if synth_ckpt8:  # a second sft_out tree, as if sft-synth were still mounted
            ck = root / "notebooks/ahmedmobasher86/arc-agi-3-sft-synth-v1/sft_out/checkpoint-8"
            ck.mkdir(parents=True)
            (ck / "trainer_state.json").write_text(json.dumps({"global_step": 8}))
        return root

    def run_select(root: Path) -> dict:
        ns = {"glob": glob_mod, "json": json, "os": os}
        exec(bld.ADAPTER_SELECT.replace("/kaggle/input", str(root)), ns)
        return ns

    sel_root = WORKDIR / "kmock"
    ns = run_select(mk_tree(sel_root / "ok"))
    assert ns["CKPT"].endswith("arc3-sft-k3-ckpts/sft_out/checkpoint-8"), ns["CKPT"]
    assert ns["CORPUS"].endswith("arc3-sft-k3-corpus"), ns["CORPUS"]
    assert ns["MODEL"].endswith("vrfai-qwen3-6-27b-fp8-hf-snapshot"), ns["MODEL"]
    print("[dry] adapter select: run-8 checkpoint-8 resolved from the real "
          "listing shape (sft_adapter + tokenizer_bundle correctly ignored): OK")

    def expect_abort(root: Path, needle: str):
        try:
            run_select(root)
        except AssertionError as e:
            assert needle in str(e), (needle, repr(e))
            return
        raise AssertionError(f"ADAPTER_SELECT failed to abort ({needle!r}) for {root}")

    expect_abort(mk_tree(sel_root / "dup", synth_ckpt8=True),
                 "expected exactly one sft_out/checkpoint-8")
    expect_abort(mk_tree(sel_root / "wrongds", k3_ckpt=False, synth_ckpt8=True),
                 "not from the arc3-sft-k3-ckpts dataset")
    expect_abort(mk_tree(sel_root / "wrongloss", loss8=0.5551234),
                 "does not match the verified run-8 record")
    expect_abort(mk_tree(sel_root / "wrongsize", adapter_bytes=123),
                 "run-8 recorded 467062560")
    expect_abort(mk_tree(sel_root / "nocorpus", corpus=False),
                 "expected exactly one k3-corpus train.jsonl")
    print("[dry] adapter select abort paths: duplicate ckpt-8, wrong-dataset "
          "ckpt-8, non-run-8 trainer_state, wrong byte size, missing corpus: OK")

    assert MockBrain.n_posts > 0
    print(f"\n[dry] PASS — {MockBrain.n_posts} mock-brain calls, "
          f"artifact at {WORKDIR / 'ab_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
