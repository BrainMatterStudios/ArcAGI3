#!/usr/bin/env python3
"""Host-only tests for offkaggle/run_regime_wave.py. No GPU, no Modal, no
internet (the end-to-end test drives the loopback mock only).

What is checked:
  1. arm envs: `keith` == KEITH_REGIME.md §5 (parsed, not re-typed) and ==
     keith's persisted taaf_setup_env.json when the session scratchpad holds
     it; `flight` == the flight notebook's cell-9 duck_env dict (ast-parsed,
     the >=32768-ctx branch); the two arms differ on exactly the two window keys.
  2. geometry + game list == keith notebook cells 13 and 15 (parsed).
  3. stock pins: june_stock agent tree, framework tree and the two solver
     pkls hash to the pinned values; and == keith's dataset copy when present.
  4. telemetry extractor reproduces the judge's numbers on keith's commit-run
     transcripts (55 calls/game, reasoning mean 3,406 / median 2,206, 53
     turns/game) and behaves on a synthetic transcript with known counts.
  5. /metrics parsing on keith's vllm-metrics-final.prom (1,371 requests,
     1,366 stop + 5 length, e2e mean 142 s, MTP acceptance 60 %, 57 preemptions).
  6. the analyzer_factory hook builds a ToolAgent identical to the stock
     HarnessSolver._make_analyzer (subprocess, real imports, both arms).
  7. an end-to-end dry run (2 games, 8 s cap) writes the results layout,
     exercises 303 redirect legs with the bearer preserved, and leaks no token.

Run:  .venv/bin/python offkaggle/test_run_regime_wave.py     (or pytest)
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_regime_wave as rw  # noqa: E402

PYTHON = str(REPO / ".venv/bin/python") if (REPO / ".venv/bin/python").exists() else sys.executable
SCRATCH = Path("/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad")
KEITH_KOUT = SCRATCH / "search/judge/kout/keithtyser"
KEITH_BUNDLE = SCRATCH / "search/field-delta/keith_smoke_v1"
KEITH_NB = REPO / "submission/_keith_copy/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
FLIGHT_NB = REPO / "submission/_flashnext_flight/arc3-flashnext-flight.ipynb"
REGIME_MD = HERE / "KEITH_REGIME.md"

ANALYZER_PREFIXES = ("LOCAL_ANALYZER_", "INFERENCE_ANALYZER_", "MULTIMODAL_", "OPENAI_PROVIDER")
RUNTIME_KEYS = set(rw.RUNTIME_ENV_KEYS) | {"OPENAI_API_KEY"}


def _skip(msg: str) -> None:
    print(f"  (skipped: {msg})")


def _nb_cell(path: Path, index: int) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "".join(nb["cells"][index]["source"])


# --- 1. arm envs ------------------------------------------------------------


def test_keith_arm_env_matches_regime_doc():
    text = REGIME_MD.read_text(encoding="utf-8")
    sec = text.split("## 5. Harness-side analyzer settings", 1)[1].split("## 6.", 1)[0]
    pairs = dict(re.findall(r"`([A-Z_]+)=([^`]+)`", sec))
    assert len(pairs) >= 14, pairs
    for k, v in pairs.items():
        assert rw.KEITH_ANALYZER_ENV.get(k) == v, (k, v, rw.KEITH_ANALYZER_ENV.get(k))
    # keys we carry that §5 does not spell out come from keith's persisted env
    extra = set(rw.KEITH_ANALYZER_ENV) - set(pairs)
    assert extra == {"OPENAI_PROVIDER", "LOCAL_ANALYZER_MODEL_ID", "LOCAL_ANALYZER_APP_NAME"}, extra
    assert "concurrency 28, 7920 s/game, analyzer_timeout 900" in sec


def test_keith_arm_env_matches_persisted_setup_env():
    path = KEITH_KOUT / "taaf_setup_env.json"
    if not path.is_file():
        return _skip(f"{path} not present")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    for k, v in rw.KEITH_ANALYZER_ENV.items():
        assert persisted.get(k) == v, (k, v, persisted.get(k))
    theirs = {k for k in persisted if k.startswith(ANALYZER_PREFIXES) or k == "OPENAI_PROVIDER"}
    missing = theirs - set(rw.KEITH_ANALYZER_ENV) - RUNTIME_KEYS
    assert not missing, f"persisted analyzer keys we do not set: {missing}"


def _flight_duck_env() -> dict:
    src = _nb_cell(FLIGHT_NB, 9)
    assert 'if _booted_ctx >= 32768:\n    _ctx_window, _max_out = "24576", "4096"' in src, "flight branch moved"
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "duck_env" for t in n.targets))
    names = {"VLLM_API": "http://127.0.0.1:1234/v1", "QWEN_SERVED_MODEL_NAME": rw.SERVED_MODEL_NAME,
             "_ctx_window": "24576", "_max_out": "4096"}
    return eval(compile(ast.Expression(node.value), "<duck_env>", "eval"), {}, names)  # noqa: S307


def test_flight_arm_env_matches_flight_notebook():
    duck_env = _flight_duck_env()
    assert duck_env["INFERENCE_ANALYZER_MODEL"] == rw.SERVED_MODEL_NAME
    theirs = {k: v for k, v in duck_env.items()
              if (k.startswith(ANALYZER_PREFIXES) or k == "OPENAI_PROVIDER") and k not in RUNTIME_KEYS}
    assert theirs == rw.FLIGHT_ANALYZER_ENV, {
        k: (theirs.get(k), rw.FLIGHT_ANALYZER_ENV.get(k))
        for k in set(theirs) | set(rw.FLIGHT_ANALYZER_ENV) if theirs.get(k) != rw.FLIGHT_ANALYZER_ENV.get(k)}
    # the notebook's extra keys are TF/torchvision import guards — inert for the harness
    assert set(duck_env) - set(theirs) - RUNTIME_KEYS == {
        "USE_TF", "TRANSFORMERS_NO_TF", "TRANSFORMERS_NO_TORCHVISION", "VLLM_NO_USAGE_STATS"}


def test_arms_differ_on_exactly_the_window_keys():
    diff = {k for k in set(rw.KEITH_ANALYZER_ENV) | set(rw.FLIGHT_ANALYZER_ENV)
            if rw.KEITH_ANALYZER_ENV.get(k) != rw.FLIGHT_ANALYZER_ENV.get(k)}
    assert diff == {"LOCAL_ANALYZER_CONTEXT_WINDOW", "LOCAL_ANALYZER_MAX_OUTPUT"}, diff
    assert (rw.KEITH_ANALYZER_ENV["LOCAL_ANALYZER_CONTEXT_WINDOW"], rw.KEITH_ANALYZER_ENV["LOCAL_ANALYZER_MAX_OUTPUT"]) == ("32768", "0")
    assert (rw.FLIGHT_ANALYZER_ENV["LOCAL_ANALYZER_CONTEXT_WINDOW"], rw.FLIGHT_ANALYZER_ENV["LOCAL_ANALYZER_MAX_OUTPUT"]) == ("24576", "4096")
    assert rw.ARMS == ("keith", "flight")


# --- 2. geometry + game list == the keith notebook -------------------------


def test_geometry_and_games_match_keith_notebook():
    cell13 = _nb_cell(KEITH_NB, 13)
    got = {}
    for key in ("max_runtime_s_per_game", "analyzer_timeout", "concurrency", "max_actions_per_game", "save_request_logs"):
        m = re.search(rf"^bm\.solver\.{key} = (.+)$", cell13, re.M)
        assert m, key
        got[key] = ast.literal_eval(m.group(1))
    assert got == rw.GEOMETRY, (got, rw.GEOMETRY)
    cell15 = _nb_cell(KEITH_NB, 15)
    tree = ast.parse(cell15)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC_GAME_IDS" for t in n.targets))
    value = node.value
    if isinstance(value, ast.Call) and getattr(value.func, 'id', None) == 'tuple':
        value = value.args[0]
    ids = ast.literal_eval(value)
    assert tuple(ids) == rw.PUBLIC_GAME_IDS
    assert "bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)" in cell15
    cell3 = _nb_cell(KEITH_NB, 3)
    for k, v in rw.PROCESS_ENV.items():
        if k == "TAAF_RUN_AS_SUBMISSION":
            assert 'os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"' in cell3
        else:
            assert f'os.environ["{k}"] = "{v}"' in cell3, k


def test_resolve_game_ids():
    assert rw.resolve_game_ids("all") == list(rw.PUBLIC_GAME_IDS)
    assert rw.resolve_game_ids("tu93, ft09-0d8bbf25,tu93") == ["tu93-0768757b", "ft09-0d8bbf25"]
    try:
        rw.resolve_game_ids("zzzz")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown game accepted")


# --- 3. stock pins ---------------------------------------------------------


def test_stock_pins():
    sha, n = rw.tree_sha256(rw.STOCK_AGENT_DIR)
    assert sha == rw.STOCK_AGENT_TREE_SHA256 and n == 42, (sha, n)
    fw, n_fw = rw.tree_sha256(rw.FRAMEWORK_TREE_DIR)
    assert fw == rw.FRAMEWORK_TREE_SHA256 and n_fw == 23, (fw, n_fw)
    assert rw.file_sha256(rw.BUNDLE_PKL_DIR / "benchmark_initial.pkl") == rw.BENCHMARK_PKL_SHA256
    assert rw.file_sha256(rw.BUNDLE_PKL_DIR / "deploy_target.pkl") == rw.DEPLOY_TARGET_PKL_SHA256
    info = rw.assert_stock_tree()
    assert info["agent_tree_sha256"] == sha
    if not KEITH_BUNDLE.is_dir():
        return _skip(f"{KEITH_BUNDLE} not present")
    assert rw.tree_sha256(KEITH_BUNDLE / "src/ARC3-Inference")[0] == sha, "keith's agent copy differs from june_stock"
    assert rw.tree_sha256(KEITH_BUNDLE / "src/tufa-arc-agi-framework")[0] == fw
    assert rw.file_sha256(KEITH_BUNDLE / "benchmark_initial.pkl") == rw.BENCHMARK_PKL_SHA256
    assert rw.file_sha256(KEITH_BUNDLE / "deploy_target.pkl") == rw.DEPLOY_TARGET_PKL_SHA256


# --- 4. telemetry extractor ------------------------------------------------


def test_extractor_reproduces_judge_on_keith_transcripts():
    files = sorted(glob.glob(str(KEITH_KOUT / "transcripts__*_p0.txt")))
    if len(files) != 25:
        return _skip(f"{len(files)} keith transcripts under {KEITH_KOUT}")
    per_game = {}
    pooled = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="replace")
        gid = Path(f).name[len("transcripts__"):-len("_p0.txt")]
        per_game[gid] = rw.telemetry_for_game(text)
        pooled.extend(c["reasoning_chars"] for c in rw.parse_transcript(text)["calls"])
    agg = rw.aggregate_telemetry(per_game)
    assert agg["calls_total"] == 1371, agg["calls_total"]
    assert round(agg["calls_per_game"]) == 55, agg["calls_per_game"]
    assert round(agg["turns_per_game"]) == 53, agg["turns_per_game"]
    assert abs(statistics.fmean(pooled) - 3406) < 1.0, statistics.fmean(pooled)
    assert statistics.median(pooled) == 2206, statistics.median(pooled)
    assert agg["finish_reasons"] == {"tool_calls": 1339, "stop": 27, "length": 5}, agg["finish_reasons"]
    assert round(agg["no_tool_call_share"] * 1371) == 32
    assert agg["turn_outcomes"]["step_executed"] == 659 and agg["turn_outcomes"]["yielded"] == 648, agg["turn_outcomes"]
    assert 27 < agg["distinct_steps_per_game"] < 28
    tu93 = per_game["tu93-0768757b"]
    assert tu93["calls"] == 56 and tu93["turns"] == 55
    assert tu93["analyzer_status_config"] == {"max_output_tokens": "server default",
                                              "context_budget_tokens": "31744", "yield_seconds": "60.0"}


SYNTHETIC_TRANSCRIPT = """
--- analysis_step=1 | action=1 | 10:00:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
hello [not a label]
[MODEL RESPONSE META]
finish_reason: stop
tool_call_count: 0
content_chars: 5
reasoning_chars: 7
tool_call_markup_in_text: no
[THINKING]
abcdefg
[ASSISTANT]
hello
[USER PROMPT]
You have not acted yet.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 3
[THINKING]
xyz

[TOOL CALL: python]
{"code": "action(['UP'])"}
[TOOL RESULT: python]
ok
[ANALYZER STATUS]
model: m
max_output_tokens: 4096
context_budget_tokens: 19968
yield_seconds: 60.0
step_executed: True
message: Step executed.

--- analysis_step=2 | action=2 | 10:01:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
next
[MODEL RESPONSE META]
finish_reason: length
tool_call_count: 0
content_chars: 0
reasoning_chars: 0
[ANALYZER STATUS]
model: m
max_output_tokens: 4096
context_budget_tokens: 19968
yield_seconds: 60.0
step_executed: False
message: Yielded control to solver: turn_time_budget.

--- analysis_step=2 | action=2 | 10:02:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
again
[ANALYZER STATUS]
request_error: HTTPConnectionPool read timed out
"""


def test_extractor_on_synthetic_transcript():
    tel = rw.telemetry_for_game(SYNTHETIC_TRANSCRIPT, actions_total=4, wallclock_s=180.0,
                                shim_records=[{"status": 200, "elapsed_s": 2.0, "redirects": 1, "prompt_tokens": 10,
                                               "completion_tokens": 5, "timeout": 900.0, "max_tokens": 4096},
                                              {"status": 200, "elapsed_s": 4.0, "redirects": 0, "prompt_tokens": 20,
                                               "completion_tokens": 7, "timeout": 900.0, "max_tokens": 4096},
                                              {"error": "ReadTimeout", "elapsed_s": 0.1, "timeout": 0.1}])
    assert tel["calls"] == 3 and tel["turns"] == 3 and tel["distinct_analysis_steps"] == 2
    assert [c["reasoning_chars"] for c in rw.parse_transcript(SYNTHETIC_TRANSCRIPT)["calls"]] == [7, 3, 0]
    assert tel["reasoning_chars"]["mean"] == statistics.fmean([7, 3, 0])
    assert tel["no_tool_call_calls"] == 2 and abs(tel["no_tool_call_share"] - 2 / 3) < 1e-9
    assert tel["finish_reasons"] == {"stop": 1, "tool_calls": 1, "length": 1}
    assert abs(tel["length_finish_share"] - 1 / 3) < 1e-9
    assert tel["turn_outcomes"] == {"step_executed": 1, "yielded": 1, "request_error": 1}, tel["turn_outcomes"]
    assert tel["actions_per_call"] == 4 / 3
    assert tel["analyzer_status_config"] == {"max_output_tokens": "4096", "context_budget_tokens": "19968",
                                             "yield_seconds": "60.0"}
    assert tel["client"]["e2e_s"]["mean"] == 3.0 and tel["client"]["redirected_posts"] == 1
    assert tel["client"]["errors"] == 1 and tel["client"]["max_tokens_sent"] == [4096, None]
    agg = rw.aggregate_telemetry({"g": tel})
    assert agg["calls_per_game"] == 3 and agg["yielded_turn_share"] == 1 / 3


# --- 5. /metrics parsing ---------------------------------------------------


def test_prom_parser_on_keith_metrics():
    path = KEITH_KOUT / "vllm-metrics-final.prom"
    if not path.is_file():
        return _skip(f"{path} not present")
    summ = rw.summarize_metrics(path.read_text(encoding="utf-8"))
    assert summ["request_success_by_reason"]["stop"] == 1366 and summ["request_success_by_reason"]["length"] == 5
    assert summ["e2e_count"] == 1371 and summ["preemptions"] == 57
    assert summ["model_names"] == [rw.SERVED_MODEL_NAME]
    zero = {k: 0.0 for k in rw._METRIC_SUMS}
    zero["request_success_by_reason"] = {}
    d = rw.metrics_delta(zero, summ, wall_s=7921.0)
    assert d["requests"] == 1371 and abs(d["e2e_mean_s"] - 142.2) < 0.5, d["e2e_mean_s"]
    assert abs(d["queue_mean_s"] - 124) < 1.0, d["queue_mean_s"]
    assert abs(d["mtp_acceptance_rate"] - 0.598) < 0.005, d["mtp_acceptance_rate"]
    assert abs(d["gen_tokens_per_s"] - 249) < 1.0, d["gen_tokens_per_s"]
    assert d["prefix_cache_hit_rate"] is None or d["prefix_queries"] == 0.0  # prefix caching OFF


def test_metrics_delta_synthetic():
    lab = 'engine="0",model_name="m"'
    a = rw.summarize_metrics(f'vllm:e2e_request_latency_seconds_sum{{{lab}}} 10\n'
                             f'vllm:e2e_request_latency_seconds_count{{{lab}}} 2\n'
                             f'vllm:request_success_total{{{lab},finished_reason="stop"}} 2\n'
                             f'vllm:generation_tokens_total{{{lab}}} 100\n')
    b = rw.summarize_metrics(f'vllm:e2e_request_latency_seconds_sum{{{lab}}} 40\n'
                             f'vllm:e2e_request_latency_seconds_count{{{lab}}} 5\n'
                             f'vllm:request_success_total{{{lab},finished_reason="stop"}} 4\n'
                             f'vllm:request_success_total{{{lab},finished_reason="length"}} 1\n'
                             f'vllm:generation_tokens_total{{{lab}}} 400\n')
    d = rw.metrics_delta(a, b, wall_s=100.0)
    assert d["requests"] == 3 and d["e2e_mean_s"] == 10.0 and d["gen_tokens_per_s"] == 3.0
    assert d["request_success_by_reason"] == {"length": 1.0, "stop": 2.0}
    assert d["mtp_acceptance_rate"] is None and rw.metrics_delta(None, b, 1.0) is None


# --- 6. analyzer factory == stock _make_analyzer ----------------------------

_FACTORY_PROBE = r'''
import json, os, sys
sys.path.insert(0, __HERE__)
import run_regime_wave as rw
from pathlib import Path
out = Path(__TMP__)
env = rw.install_env(__ARM__, "http://127.0.0.1:9/v1", "probe-token", out)
rw.install_paths()
rw.verify_imports()
bm, target = rw.load_bundle(out)
geo = rw.apply_geometry(bm, per_game_s=7920.0, concurrency=28)
solver = bm.solver
assert solver.analyzer_factory is None
stock = solver._make_analyzer(None, 0)          # the stock path (no factory, no local server)
factory = rw.make_tagging_analyzer_factory(solver)
class _G:  # a started game stand-in: only game_run.game_id is read
    class game_run: game_id = "tu93-0768757b"
ours = factory(_G(), 0)
print(json.dumps({"stock": rw.analyzer_config_fingerprint(stock), "ours": rw.analyzer_config_fingerprint(ours),
                  "tag": rw.current_game_tag(), "geo": geo, "types": [type(stock).__name__, type(ours).__name__]}))
'''


def test_analyzer_factory_equals_stock_make_analyzer():
    for arm, want_ctx_budget, want_max_out in (("keith", 31744, None), ("flight", 19968, 4096)):
        with tempfile.TemporaryDirectory() as tmp:
            code = (_FACTORY_PROBE.replace("__HERE__", repr(str(HERE))).replace("__TMP__", repr(tmp))
                    .replace("__ARM__", repr(arm)))
            r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=300)
            assert r.returncode == 0, r.stderr[-2000:]
            probe = json.loads(r.stdout.strip().splitlines()[-1])
        assert probe["types"] == ["ToolAgent", "ToolAgent"]
        assert probe["stock"] == probe["ours"], (arm, probe)
        assert probe["stock"]["timeout"] == 900.0 and probe["stock"]["model_id"] == rw.SERVED_MODEL_NAME
        assert probe["stock"]["base_url"] == "http://127.0.0.1:9/v1" and probe["stock"]["provider"] == "vllm"
        assert probe["stock"]["context_budget_tokens"] == want_ctx_budget, (arm, probe["stock"])
        assert probe["stock"]["max_output_tokens"] == want_max_out, (arm, probe["stock"])
        assert probe["stock"]["yield_seconds"] == 60.0 and probe["stock"]["tool_steps"] is None
        assert probe["tag"] == "tu93-0768757b"
        assert probe["geo"]["matches_public25"] is True and probe["geo"]["analyzer_timeout"] == 900.0
        assert probe["geo"]["max_actions_per_game"] is None and probe["geo"]["save_request_logs"] is False


# --- 7. end-to-end dry run (mock vLLM, two games, ~15 s) --------------------


def test_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "flight",
                            "--games", "tu93,ft09", "--per-game-s", "8", "--wave-cap-s", "40",
                            "--progress-every", "60", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=300)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        runs = list(Path(tmp).glob("*-regime-flight-dry"))
        assert len(runs) == 1, runs
        out = runs[0]
        for name in ("results.json", "telemetry.json", "summary.txt", "arm_env.json", "metrics_before.prom",
                     "metrics_after.prom", "requests_shim.jsonl", "benchmark.json", "score.json"):
            assert (out / name).is_file(), name
        assert sorted(p.name for p in (out / "transcripts").glob("*.txt")) == ["ft09-0d8bbf25_p0.txt", "tu93-0768757b_p0.txt"]
        res = json.loads((out / "results.json").read_text())
        tel = json.loads((out / "telemetry.json").read_text())
        assert res["status"] == "done" and res["dry_run"] is True and res["arm"] == "flight"
        assert res["stock"]["agent_tree_sha256"] == rw.STOCK_AGENT_TREE_SHA256
        assert res["geometry"]["analyzer_timeout"] == 900.0 and res["geometry"]["concurrency"] == 28
        assert res["geometry"]["max_actions_per_game"] is None and res["geometry"]["matches_public25"] is False
        assert [g["game_id"] for g in res["games"]] == ["tu93-0768757b", "ft09-0d8bbf25"]
        assert res["totals"]["games"] == 2 and res["totals"]["score_source"] == "score.json"
        assert res["analyzer_env"]["LOCAL_ANALYZER_API_KEY"] == "<redacted>"
        assert res["analyzer_env"]["LOCAL_ANALYZER_MAX_OUTPUT"] == "4096"
        # the harness itself attests the arm: flight budget = 24576 - 4096 - 512
        cfg = tel["aggregate"]["analyzer_status_config_first"]
        assert cfg == {"max_output_tokens": "4096", "context_budget_tokens": "19968", "yield_seconds": "60.0"}, cfg
        agg = tel["aggregate"]
        assert agg["games"] == 2 and agg["calls_total"] >= 10 and agg["reasoning_chars_pooled"]["n"] == agg["calls_total"]
        assert agg["no_tool_call_share"] > 0                     # the text-only branch was exercised
        assert agg["client_redirected_posts"] > 0                # 303 legs happened...
        ms = res["mock_state"]
        assert ms["poll_with_auth"] == ms["redirected"] > 0     # ...and the stock client followed them with the bearer
        assert ms["poll_without_auth"] == 0 and ms["unauthorized"] == 0
        shim = [json.loads(l) for l in (out / "requests_shim.jsonl").read_text().splitlines() if l.strip()]
        assert all(rec["max_tokens"] == 4096 for rec in shim), "flight arm must send max_tokens=4096"
        assert all(rec["game_id"] in ("tu93-0768757b", "ft09-0d8bbf25") for rec in shim)
        assert any(rec.get("redirect_codes") == [303] for rec in shim)
        ok = [rec for rec in shim if rec.get("status") == 200]
        assert len(ok) == agg["calls_total"], (len(ok), agg["calls_total"])   # shim == transcript call count
        # /metrics counts every request the server received: >= transcript calls (a
        # client-side timeout at the per-game cap leaves no META) and <= shim posts;
        # the preflight completion is excluded by the before-snapshot.
        assert agg["calls_total"] <= res["metrics"]["delta"]["requests"] <= len(shim), (
            agg["calls_total"], res["metrics"]["delta"]["requests"], len(shim))
        # no secret anywhere in the artifacts
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path
        assert "dry-run-token" not in r.stdout and "dry-run-token" not in r.stderr
        summary = (out / "summary.txt").read_text()
        assert "REGIME WAVE  arm=flight  status=done" in summary and "CONTEXT_WINDOW=24576" in summary
        assert summary.count("\n") < 45, "summary must fit one screen"


# --- runner -----------------------------------------------------------------


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
