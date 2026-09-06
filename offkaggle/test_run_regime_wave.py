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
    assert rw.ARMS == ("keith", "flight", "keith_yield180", "keith_retry", "keith_evid", "keith_hypo", "keith_up8")
    # the original single-knob arm differs from the keith base on exactly the yield key
    d2 = {k for k in set(rw.KEITH_ANALYZER_ENV) | set(rw.KEITH_YIELD180_ENV)
          if rw.KEITH_ANALYZER_ENV.get(k) != rw.KEITH_YIELD180_ENV.get(k)}
    assert d2 == {"LOCAL_ANALYZER_YIELD_SECONDS"}, d2
    assert rw.KEITH_YIELD180_ENV["LOCAL_ANALYZER_YIELD_SECONDS"] == "180"
    # the retry arm differs from the keith base on exactly the graft's RETRY_* flags
    d3 = {k for k in set(rw.KEITH_ANALYZER_ENV) | set(rw.KEITH_RETRY_ENV)
          if rw.KEITH_ANALYZER_ENV.get(k) != rw.KEITH_RETRY_ENV.get(k)}
    assert d3 == set(rw.RETRY_ENV_KEYS) == {"RETRY_ENABLE", "RETRY_K", "RETRY_ABS", "RETRY_COOLDOWN", "RETRY_MAX"}, d3
    assert {k: rw.KEITH_RETRY_ENV[k] for k in rw.RETRY_ENV_KEYS} == {
        "RETRY_ENABLE": "1", "RETRY_K": "3", "RETRY_ABS": "200", "RETRY_COOLDOWN": "150", "RETRY_MAX": "2"}
    assert rw.ARM_GRAFTS == {"keith_retry": ("graft_retry",), "keith_evid": ("graft_evidence",),
                             "keith_hypo": ("graft_hypo",)}
    # 09-06 arms: each differs from the keith base by exactly its own keys
    def _diff(env):
        return {k for k in set(rw.KEITH_ANALYZER_ENV) | set(env) if rw.KEITH_ANALYZER_ENV.get(k) != env.get(k)}
    assert _diff(rw.KEITH_EVID_ENV) == set(rw.EVID_ENV_KEYS) == {"EVID_ENABLE", "EVID_MAX_ENTRIES", "EVID_MAX_CHARS", "EVID_TRACE"}
    assert {k: rw.KEITH_EVID_ENV[k] for k in rw.EVID_ENV_KEYS} == {"EVID_ENABLE": "1", "EVID_MAX_ENTRIES": "40",
                                                                    "EVID_MAX_CHARS": "1500", "EVID_TRACE": "1"}
    assert _diff(rw.KEITH_HYPO_ENV) == set(rw.HYPO_ENV_KEYS) == {"HYPO_ENABLE"} and rw.KEITH_HYPO_ENV["HYPO_ENABLE"] == "1"
    assert _diff(rw.KEITH_UP8_ENV) == {"MULTIMODAL_UPSCALE"} and rw.KEITH_UP8_ENV["MULTIMODAL_UPSCALE"] == "8"
    assert rw.KEITH_ANALYZER_ENV["MULTIMODAL_UPSCALE"] == "4" and rw.KEITH_UP8_ENV["MULTIMODAL_CONTEXT"] == "current_grid"
    for arm in ("keith", "flight", "keith_yield180", "keith_up8"):
        assert not any(k.startswith(rw.GRAFT_ENV_PREFIXES) for k in rw.ARM_ENV[arm]), arm   # stock arms carry no graft flag
    for arm, prefix in (("keith_retry", "RETRY_"), ("keith_evid", "EVID_"), ("keith_hypo", "HYPO_")):
        assert all(k.startswith(prefix) for k in rw.ARM_ENV[arm] if k.startswith(rw.GRAFT_ENV_PREFIXES)), arm


def test_install_env_never_leaks_graft_flags_into_a_stock_arm():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RETRY_ENABLE"] = "1"          # stale shell state
        try:
            rec = rw.install_env("keith", "http://127.0.0.1:9/v1", "t", Path(tmp))
            assert "RETRY_ENABLE" not in os.environ and "RETRY_ENABLE" not in rec
            rec = rw.install_env("keith_retry", "http://127.0.0.1:9/v1", "t", Path(tmp))
            assert os.environ["RETRY_ENABLE"] == "1" and rec["RETRY_K"] == "3"
            rec = rw.install_env("keith_retry", "http://127.0.0.1:9/v1", "t", Path(tmp), {"RETRY_ABS": "7"})
            assert os.environ["RETRY_ABS"] == "7" and rec["RETRY_ABS"] == "7" and rec["RETRY_K"] == "3"
            # a stale EVID_/HYPO_ flag never leaks into a stock arm or into another graft's arm
            os.environ["EVID_ENABLE"] = "1"
            os.environ["HYPO_ENABLE"] = "1"
            rec = rw.install_env("keith_up8", "http://127.0.0.1:9/v1", "t", Path(tmp))
            assert "EVID_ENABLE" not in os.environ and "HYPO_ENABLE" not in os.environ and "RETRY_ENABLE" not in os.environ
            assert rec["MULTIMODAL_UPSCALE"] == "8" and not any(k.startswith(rw.GRAFT_ENV_PREFIXES) for k in rec)
            os.environ["HYPO_ENABLE"] = "1"
            rec = rw.install_env("keith_evid", "http://127.0.0.1:9/v1", "t", Path(tmp))
            assert "HYPO_ENABLE" not in os.environ and os.environ["EVID_ENABLE"] == "1" and rec["EVID_MAX_CHARS"] == "1500"
            assert rec["LOCAL_ANALYZER_API_KEY"] == "<redacted>" and "t" != os.environ["LOCAL_ANALYZER_API_KEY"][:0]
        finally:
            for k in [k for k in os.environ if k.startswith(rw.GRAFT_ENV_PREFIXES)]:
                os.environ.pop(k, None)
    assert rw.parse_knobs(["A=1", "B = x=y"]) == {"A": "1", "B": "x=y"}
    try:
        rw.parse_knobs(["novalue"])
    except ValueError:
        pass
    else:
        raise AssertionError("bad knob accepted")


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
    assert tel["retries_fired"] == 0 and tel["retry_clears"] == 0
    assert agg["retries_total"] == 0 and agg["games_with_retry"] == 0 and agg["retry_clears_total"] == 0


RETRY_TRANSCRIPT = SYNTHETIC_TRANSCRIPT + """
[HARNESS RETRY]
[RETRY] game=tu93-0768757b level=1 actions=57 baseline=19 threshold=57 retry=1/2 action_num=58

--- analysis_step=3 | action=58 | 10:03:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
next
FRESH MIND (harness level retry 1/2): ...
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 2
[THINKING]
ab
[ANALYZER STATUS]
model: m
step_executed: True
message: Step executed.

[HARNESS RETRY]
[RETRY-CLEAR] game=tu93-0768757b level=1 actions=70 retries=1

"""


def test_extractor_counts_retry_markers():
    tel = rw.telemetry_for_game(RETRY_TRANSCRIPT, actions_total=70)
    assert tel["retries_fired"] == 1 and tel["retry_clears"] == 1
    assert tel["calls"] == 4 and tel["turns"] == 4          # the marker sections do not disturb turn/call parsing
    assert tel["turn_outcomes"]["step_executed"] == 2
    agg = rw.aggregate_telemetry({"a": tel, "b": rw.telemetry_for_game(SYNTHETIC_TRANSCRIPT)})
    assert agg["retries_total"] == 1 and agg["retries_per_game"] == 0.5
    assert agg["retry_clears_total"] == 1 and agg["games_with_retry"] == 1
    # a "[RETRY]" mention inside model text is not a marker (anchored at line start with the field shape)
    assert rw.telemetry_for_game("hello [RETRY] game=x\n[RETRY] not a marker\n")["retries_fired"] == 0


# --- 4b. serving identity gate ---------------------------------------------

_RECORDED_IDENTITY = {   # shape of GET /arc3/identity as recorded in offkaggle/results/*/results.json
    "app": "arc3-flashnext", "profile": "kv5-bf16-mtp3-c8-cg32", "gpu": "RTX-PRO-6000",
    "host": {"cpu_count": 24, "gpu_rows": ["0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.95.05, 12.0"]},
}


def test_identity_gate():
    ok = rw.check_identity(_RECORDED_IDENTITY, rw.DEFAULT_EXPECT_PROFILE)
    assert ok["profile_ok"] is True and ok["gpu_ok"] is True and ok["profile"] == "kv5-bf16-mtp3-c8-cg32"
    assert rw.DEFAULT_EXPECT_PROFILE == "kv5-bf16-mtp3-c8-cg32" and rw.EXPECT_GPU_SUBSTRING == "RTX PRO 6000"
    # the 09-03 kv10 override identity must be refused by default ...
    kv10 = {**_RECORDED_IDENTITY, "profile": "kv10-bf16-mtp3-c8-cg32-OVERRIDE"}
    try:
        rw.check_identity(kv10, rw.DEFAULT_EXPECT_PROFILE)
    except RuntimeError as e:
        assert "SERVING PROFILE MISMATCH" in str(e) and "kv10-bf16-mtp3-c8-cg32-OVERRIDE" in str(e) and "--expect-profile" in str(e)
    else:
        raise AssertionError("kv10 override profile accepted")
    # ... and accepted only when asked for explicitly
    assert rw.check_identity(kv10, "kv10-bf16-mtp3-c8-cg32-OVERRIDE")["profile_ok"] is True
    # wrong / missing GPU rows
    for rows in (["0, NVIDIA A100-SXM4-80GB, 81920 MiB, 550.0, 8.0"], [], ["nvidia-smi unavailable: FileNotFoundError"]):
        bad = {**_RECORDED_IDENTITY, "host": {"gpu_rows": rows}}
        try:
            rw.check_identity(bad, rw.DEFAULT_EXPECT_PROFILE)
        except RuntimeError as e:
            assert "GPU MISMATCH" in str(e), str(e)
        else:
            raise AssertionError(f"gpu rows {rows} accepted")
    # no identity payload at all -> refuse (never silent)
    for payload in (None, {}):
        try:
            rw.check_identity(payload, rw.DEFAULT_EXPECT_PROFILE)
        except RuntimeError as e:
            assert "IDENTITY UNAVAILABLE" in str(e)
        else:
            raise AssertionError("missing identity accepted")


def test_dry_run_refuses_wrong_serving_profile():
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith",
                            "--games", "tu93", "--per-game-s", "5", "--wave-cap-s", "20", "--out", tmp,
                            "--mock-profile", "kv10-bf16-mtp3-c8-cg32-OVERRIDE"],
                           capture_output=True, text=True, cwd=str(REPO), timeout=300)
        assert r.returncode != 0
        assert "SERVING PROFILE MISMATCH" in (r.stderr + r.stdout)
        assert "kv10-bf16-mtp3-c8-cg32-OVERRIDE" in (r.stderr + r.stdout)
        runs = list(Path(tmp).glob("*-regime-keith-dry"))
        assert runs and not (runs[0] / "results.json").exists()      # refused before the wave started


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


_GRAFT_PROBE = r'''
import json, os, sys
sys.path.insert(0, __HERE__)
import run_regime_wave as rw
from pathlib import Path
out = Path(__TMP__)
env = rw.install_env("keith_retry", "http://127.0.0.1:9/v1", "probe-token", out)
rw.install_paths()
rw.verify_imports()
from inference.agent import tool_agent as ta
before = (ta.ToolAgent.analyze, ta.ToolAgent._build_user_prompt)
env = rw.install_env("keith_retry", "http://127.0.0.1:9/v1", "probe-token", out, {"RETRY_K": "2.5"})  # the launch shape
grafts = rw.install_grafts("keith_retry")
after = (ta.ToolAgent.analyze, ta.ToolAgent._build_user_prompt)
knob_status = rw.graft_status("keith_retry")["graft_retry"]
stock = rw.assert_stock_tree()                       # bytes untouched after the in-memory install
bm, target = rw.load_bundle(out)
factory = rw.make_tagging_analyzer_factory(bm.solver)
class _G:
    class game_run: game_id = "tu93-0768757b"
agent = factory(_G(), 0)
print(json.dumps({"grafts": grafts, "rebound": [a is not b for a, b in zip(before, after)],
                  "stock_attr": [hasattr(after[0], "_retry_stock"), hasattr(after[1], "_retry_stock")],
                  "chain": [after[0]._retry_stock is before[0], after[1]._retry_stock is before[1]],
                  "sha": stock["agent_tree_sha256"], "status": rw.graft_status("keith_retry"),
                  "knob": {"k": knob_status["k"], "env_k": os.environ.get("RETRY_K"), "rec_k": env.get("RETRY_K"),
                           "abs": knob_status["abs"]},
                  "env": {k: os.environ.get(k) for k in rw.RETRY_ENV_KEYS},
                  "fingerprint": rw.analyzer_config_fingerprint(agent),
                  "keith_grafts": rw.install_grafts("keith")}))
'''


def test_retry_arm_installs_graft_in_memory_and_keeps_stock_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        code = _GRAFT_PROBE.replace("__HERE__", repr(str(HERE))).replace("__TMP__", repr(tmp))
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        probe = json.loads(r.stdout.strip().splitlines()[-1])
    assert probe["grafts"] == {"graft_retry": "retry: OK"}
    assert probe["rebound"] == [True, True] and probe["stock_attr"] == [True, True] and probe["chain"] == [True, True]
    assert probe["sha"] == rw.STOCK_AGENT_TREE_SHA256          # the graft patches in memory; the tree sha holds
    st = probe["status"]["graft_retry"]
    assert st["installed"] and st["enabled"] and (st["abs"], st["cooldown"], st["max"]) == (200, 150, 2)
    assert st["retries_fired"] == 0 and st["retry_log"] == []
    # --knob RETRY_K=2.5 applied by install_env BEFORE install_grafts, recorded, and read by the graft
    assert probe["knob"] == {"k": 2.5, "env_k": "2.5", "rec_k": "2.5", "abs": 200}, probe["knob"]
    assert probe["env"] == {**{k: rw.KEITH_RETRY_ENV[k] for k in rw.RETRY_ENV_KEYS}, "RETRY_K": "2.5"}
    # the analyzer the factory builds is still the keith-configured ToolAgent
    fp = probe["fingerprint"]
    assert fp["context_budget_tokens"] == 31744 and fp["max_output_tokens"] is None and fp["yield_seconds"] == 60.0
    assert probe["keith_grafts"] == {}                         # stock arms install nothing


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
        head = summary.split("\n  run ")[0]
        assert "RETRY" not in head and "AID" not in head and "DRAWS" not in head   # stock arm, 1 draw: no graft/draw lines
        assert "UPSCALE=4 ([256, 256] px, ~64 vision tok/img derived)" in summary
        assert res["vision"]["png_px"] == [256, 256] and res["vision"]["vision_tokens_derived"] == 64
        assert res["draws"] == 1 and all(g["draw"] == 0 and g["run_stem"] == g["game_id"] + "_p0" for g in res["games"])
        assert set(tel["per_game"]) == {"tu93-0768757b_p0", "ft09-0d8bbf25_p0"}
        assert all(g["evid_markers"] == 0 and g["hypo_markers"] == 0 for g in tel["per_game"].values())
        assert all(rec["run_stem"] == rec["game_id"] + "_p0" for rec in shim)
        ic = res["endpoint"]["identity_check"]
        assert ic["profile"] == rw.DEFAULT_EXPECT_PROFILE and ic["profile_ok"] is True and ic["gpu_ok"] is True
        assert "RTX PRO 6000" in ic["gpu_rows"][0] and res["expect_profile"] == rw.DEFAULT_EXPECT_PROFILE
        assert "IDENTITY profile 'kv5-bf16-mtp3-c8-cg32'" in summary and "ok=True" in summary
        assert res["grafts"] == {"installed": {}, "status": {}}
        assert tel["aggregate"]["retries_total"] == 0 and tel["grafts"]["installed"] == {}


def test_dry_run_keith_retry_arm_end_to_end():
    """keith_retry through the runner on the loopback mock: the graft installs
    in memory, the stock sha still holds, and with shrunk knobs the retry
    fires on the real engine and reaches telemetry.json / summary.txt."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_retry",
                            "--games", "tu93,ft09", "--per-game-s", "14", "--wave-cap-s", "60",
                            "--progress-every", "60", "--out", tmp,
                            "--knob", "RETRY_K=0.2", "--knob", "RETRY_ABS=4", "--knob", "RETRY_COOLDOWN=3"],
                           capture_output=True, text=True, cwd=str(REPO), timeout=300)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        runs = list(Path(tmp).glob("*-regime-keith_retry-dry"))
        assert len(runs) == 1, runs
        out = runs[0]
        res = json.loads((out / "results.json").read_text())
        tel = json.loads((out / "telemetry.json").read_text())
        assert res["status"] == "done" and res["arm"] == "keith_retry"
        assert res["stock"]["agent_tree_sha256"] == rw.STOCK_AGENT_TREE_SHA256
        assert res["grafts"]["installed"] == {"graft_retry": "retry: OK"}
        assert res["knob_overrides"] == {"RETRY_K": "0.2", "RETRY_ABS": "4", "RETRY_COOLDOWN": "3"}
        assert res["analyzer_env"]["RETRY_ENABLE"] == "1" and res["analyzer_env"]["RETRY_MAX"] == "2"
        assert res["analyzer_env"]["RETRY_K"] == "0.2" and res["analyzer_env"]["RETRY_ABS"] == "4"   # knobs in the recorded env
        arm_env = json.loads((out / "arm_env.json").read_text())
        assert arm_env["RETRY_K"] == "0.2"
        assert res["grafts"]["status"]["graft_retry"]["k"] == 0.2                                   # ...and seen by the graft
        assert "'RETRY_K': '0.2'" in (out / "summary.txt").read_text()
        assert res["analyzer_env"]["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768"      # still the keith window
        cfg = tel["aggregate"]["analyzer_status_config_first"]
        assert cfg == {"max_output_tokens": "server default", "context_budget_tokens": "31744", "yield_seconds": "60.0"}, cfg
        st = res["grafts"]["status"]["graft_retry"]
        agg = tel["aggregate"]
        assert st["retries_fired"] >= 1, st
        assert agg["retries_total"] == st["retries_fired"] == len(st["retry_log"])   # transcript markers == graft counters
        assert agg["games_with_retry"] >= 1
        assert all(rec["baseline"] is not None for rec in st["retry_log"])          # offline engine exposes baselines
        assert all(rec["retry"] <= 2 for rec in st["retry_log"])
        per = tel["per_game"]
        assert sum(g["retries_fired"] for g in per.values()) == agg["retries_total"]
        for gid in ("tu93-0768757b", "ft09-0d8bbf25"):
            text = (out / "transcripts" / f"{gid}_p0.txt").read_text()
            assert text.count("FRESH MIND (harness level retry") == per[f"{gid}_p0"]["retries_fired"]   # per_game keyed by run stem
            assert text.count("[HARNESS RETRY]\n[RETRY] game=") == per[f"{gid}_p0"]["retries_fired"]
        # the RESETs are in the harness's own action record (benchmark.json history)
        bench = json.loads((out / "benchmark.json").read_text())
        hist_resets = 0
        for run in bench.get("game_runs", []):
            for rec in run.get("history", []):
                act = rec.get("action") if isinstance(rec, dict) else None
                name = (act or {}).get("id") if isinstance(act, dict) else act
                if str(name).upper().endswith("RESET") or name == 0:
                    hist_resets += 1
        assert hist_resets >= st["retries_fired"], (hist_resets, st["retries_fired"])
        summary = (out / "summary.txt").read_text()
        assert "REGIME WAVE  arm=keith_retry" in summary and "RETRY  fired" in summary and "KNOB OVERRIDES" in summary
        assert summary.count("\n") < 45
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path


# --- 8. 09-06 aids: extractor, upscale facts, in-memory installs, draws dry run ----

AID_TRANSCRIPT = """
--- analysis_step=1 | action=1 | 10:00:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
No previous sequence has been executed yet.
Current state: step 1, level 1.
Valid actions right now: UP, DOWN.
[HYPO] Hypothesis discipline for this uncleared level (harness rule, every turn):
1. List >=3 candidate mechanics.
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
p

[EVID] harness object diff for action 1 (1 executed in this call); coords are (row,col), 0-based
diff, before action 1 -> after action 1: 8 cells changed, 1 object changes
  MOVED b/blue size 4: (5,1)-(6,2) -> (5,3)-(6,4) (d row +0, col +2)
[ANALYZER STATUS]
step_executed: True
message: Step executed.

--- analysis_step=2 | action=2 | 10:01:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
The code executed 1 action in the previous sequence.
Current state: step 2, level 1.
[HYPO] Hypothesis discipline for this uncleared level (harness rule, every turn):
[MODEL RESPONSE META]
finish_reason: stop
tool_call_count: 0
content_chars: 5
reasoning_chars: 0
[ASSISTANT]
thinking about [EVID] and [HYPO] in prose is not a marker
[USER PROMPT]
You have not acted yet. Investigate first.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "action(['UP','SPACE','LEFT'])"}
[TOOL RESULT: python]
[EVID] harness object diff for actions 2-4 (3 executed in this call); coords are (row,col), 0-based
LEVEL CLEARED after action 2 (SPACE) — the frames after it belong to the NEXT level (level 2); do not diff them against this level.
level 1 diff, before action 1 -> after action 1: 8 cells changed, 1 object changes
level 2 start frame (after action 2): 3 non-background objects
TRACE per action: 1 UP: mover b/blue size 4 -> (5,3)-(6,4) | 2 SPACE: LEVEL CLEARED (frame now level 2) | 3 LEFT: 2 cells changed
[ANALYZER STATUS]
step_executed: True
message: Step executed.

--- analysis_step=3 | action=5 | 10:02:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
You have progressed to a new level!
Current state: step 5, level 2.
[HYPO] Hypothesis discipline for this uncleared level (harness rule, every turn):
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "print(1)"}
[TOOL RESULT: python]
1
[ANALYZER STATUS]
step_executed: False
message: Yielded control to solver: turn_time_budget.

--- analysis_step=3 | action=5 | 10:03:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
Current state: step 5, level 2.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "action(['UP'])"}
[TOOL RESULT: python]
[EVID] harness object diff for action 5 (1 executed in this call); coords are (row,col), 0-based
diff, before action 1 -> after action 1: no cell changed
[ANALYZER STATUS]
step_executed: True
message: Step executed.
"""


def test_extractor_counts_aid_markers_and_engagement():
    parsed = rw.parse_transcript(AID_TRANSCRIPT)
    assert [t["level"] for t in parsed["turns"]] == [1, 1, 2, 2]
    assert [t["hypo"] for t in parsed["turns"]] == [1, 1, 1, 0]        # the follow-up prompt does not repeat it
    assert [t["evid"] for t in parsed["turns"]] == [1, 1, 0, 1]
    assert [t["evid_level_flags"] for t in parsed["turns"]] == [0, 1, 0, 0]
    assert parsed["aid_markers"] == {"evid_markers": 3, "evid_level_flags": 1, "hypo_markers": 3}
    tel = rw.telemetry_for_game(AID_TRANSCRIPT, actions_total=5, levels_completed=1, number_of_levels=4)
    assert (tel["evid_markers"], tel["evid_level_flags"], tel["hypo_markers"]) == (3, 1, 3)
    assert tel["turn_levels"] == [1, 1, 2, 2] and tel["level_reached"] == 2 and tel["wall_level"] == 2
    e = tel["engagement"]
    assert e["evid"] == {"turns": 3, "of": 3, "share": 1.0}            # 3 step-executed turns, all carried the aid
    assert e["evid_wall"] == {"turns": 1, "of": 1, "share": 1.0}       # wall level 2: one executed turn, aided
    assert e["hypo"] == {"turns": 3, "of": 4, "share": 0.75}
    assert e["hypo_wall"] == {"turns": 1, "of": 2, "share": 0.5} and e["wall_turns"] == 2
    # a won game has no wall
    won = rw.telemetry_for_game(AID_TRANSCRIPT, levels_completed=4, number_of_levels=4)
    assert won["wall_level"] is None and won["engagement"]["evid_wall"] == {"turns": 0, "of": 0, "share": None}
    # markers inside model prose are not counted; a stock transcript has none
    assert rw.telemetry_for_game(SYNTHETIC_TRANSCRIPT)["evid_markers"] == 0
    assert rw.telemetry_for_game("x [EVID] harness object diff for action 1\n")["evid_markers"] == 0
    agg = rw.aggregate_telemetry({"a_p0": tel, "a_p1": won})
    assert agg["evid_markers_total"] == 6 and agg["hypo_markers_total"] == 6 and agg["evid_level_flags_total"] == 2
    assert agg["engagement"]["evid_wall"] == {"turns": 1, "of": 1, "share": 1.0}
    assert agg["engagement"]["hypo"] == {"turns": 6, "of": 8, "share": 0.75}
    assert rw.run_stem_for("cd82-fb555c5d", 4, 3) == "cd82-fb555c5d_p1" and rw.run_stem_for("x", 7, None) == "x_p0"


_VISION_PROBE = r'''
import json, os, sys
sys.path.insert(0, __HERE__)
import run_regime_wave as rw
from pathlib import Path
out = Path(__TMP__)
env = rw.install_env(__ARM__, "http://127.0.0.1:9/v1", "probe-token", out)
rw.install_paths()
rw.verify_imports()
from inference.agent import vision_context as vc
from inference.agent import tool_agent as ta
facts = rw.vision_image_facts()
agent = ta.ToolAgent(model=rw.SERVED_MODEL_NAME, timeout=900.0)
from inference.agent.runtime_state import Frame
frame = Frame(grid=tuple(tuple(0 for _ in range(64)) for _ in range(64)), step=0, level=1)
msg = agent._build_user_message("prompt", frame)
print(json.dumps({"arm": __ARM__, "upscale": vc.current_grid_image_upscale(), "facts": facts,
                  "env_upscale": env.get("MULTIMODAL_UPSCALE"),
                  "image_attached": isinstance(msg["content"], list) and msg["content"][1]["type"] == "image_url",
                  "system_prompt_multimodal": "Multimodal context" in agent._system_prompt}))
'''


def test_upscale_arm_vision_facts():
    """keith_up8 differs from keith only in MULTIMODAL_UPSCALE; the stock vision_context reads it at
    call time: 64x64 -> 512x512 px PNG (vs 256) => 256 derived vision tokens (vs 64, the measured value)."""
    got = {}
    for arm in ("keith", "keith_up8"):
        with tempfile.TemporaryDirectory() as tmp:
            code = (_VISION_PROBE.replace("__HERE__", repr(str(HERE))).replace("__TMP__", repr(tmp))
                    .replace("__ARM__", repr(arm)))
            r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=300)
            assert r.returncode == 0, r.stderr[-2000:]
            got[arm] = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["keith"]["upscale"] == 4 and got["keith"]["facts"]["png_px"] == [256, 256]
    assert got["keith"]["facts"]["vision_tokens_derived"] == 64
    assert got["keith_up8"]["upscale"] == 8 and got["keith_up8"]["env_upscale"] == "8"
    assert got["keith_up8"]["facts"]["png_px"] == [512, 512] and got["keith_up8"]["facts"]["vision_tokens_derived"] == 256
    assert got["keith_up8"]["facts"]["png_bytes"] > got["keith"]["facts"]["png_bytes"]
    for arm in got:
        assert got[arm]["image_attached"] and got[arm]["system_prompt_multimodal"], arm


_AID_GRAFT_PROBE = r'''
import json, os, sys
sys.path.insert(0, __HERE__)
import run_regime_wave as rw
from pathlib import Path
out = Path(__TMP__)
env = rw.install_env(__ARM__, "http://127.0.0.1:9/v1", "probe-token", out)
rw.install_paths()
rw.verify_imports()
from inference.agent import tool_agent as ta
before = {"run": ta.ToolAgent._run_python_tool, "prompt": ta.ToolAgent._build_user_prompt, "analyze": ta.ToolAgent.analyze}
grafts = rw.install_grafts(__ARM__)
after = {"run": ta.ToolAgent._run_python_tool, "prompt": ta.ToolAgent._build_user_prompt, "analyze": ta.ToolAgent.analyze}
stock = rw.assert_stock_tree()
bm, target = rw.load_bundle(out)
factory = rw.make_tagging_analyzer_factory(bm.solver, n_games=2)
class _G:
    class game_run: game_id = "cd82-fb555c5d"
agent = factory(_G(), 3)
print(json.dumps({"grafts": grafts, "rebound": {k: after[k] is not before[k] for k in before},
                  "sha": stock["agent_tree_sha256"], "status": rw.graft_status(__ARM__),
                  "env": {k: os.environ.get(k) for k in rw.GRAFT_FLAG_KEYS if os.environ.get(k) is not None},
                  "tag": rw.current_game_tag(), "stem": rw.current_run_stem(),
                  "fingerprint": rw.analyzer_config_fingerprint(agent)}))
'''


def test_aid_arms_install_grafts_in_memory():
    for arm, graft, method in (("keith_evid", "graft_evidence", "run"), ("keith_hypo", "graft_hypo", "prompt")):
        with tempfile.TemporaryDirectory() as tmp:
            code = (_AID_GRAFT_PROBE.replace("__HERE__", repr(str(HERE))).replace("__TMP__", repr(tmp))
                    .replace("__ARM__", repr(arm)))
            r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=300)
            assert r.returncode == 0, r.stderr[-2000:]
            probe = json.loads(r.stdout.strip().splitlines()[-1])
        assert probe["grafts"] == {graft: f"{graft.split('_', 1)[1]}: OK"}, probe["grafts"]
        assert probe["rebound"] == {k: (k == method) for k in ("run", "prompt", "analyze")}, (arm, probe["rebound"])
        assert probe["sha"] == rw.STOCK_AGENT_TREE_SHA256
        st = probe["status"][graft]
        assert st["installed"] and st["enabled"] and st["errors"] == 0
        if graft == "graft_evidence":
            assert (st["max_entries"], st["max_chars"], st["trace"], st["diffs_emitted"]) == (40, 1500, True, 0)
            assert probe["env"] == {"EVID_ENABLE": "1", "EVID_MAX_ENTRIES": "40", "EVID_MAX_CHARS": "1500", "EVID_TRACE": "1"}
        else:
            assert st["block_chars"] <= 900 and st["blocks_injected"] == 0 and probe["env"] == {"HYPO_ENABLE": "1"}
        assert probe["tag"] == "cd82-fb555c5d" and probe["stem"] == "cd82-fb555c5d_p1"    # index 3 of 2 games = draw 1
        fp = probe["fingerprint"]
        assert fp["context_budget_tokens"] == 31744 and fp["max_output_tokens"] is None and fp["yield_seconds"] == 60.0


def test_dry_run_keith_evid_draws_end_to_end():
    """keith_evid with --draws 2 through the runner on the loopback mock + real engine: two independent
    runs per game (<gid>_p0/_p1), the [EVID] block on every executed-action tool result, engagement
    telemetry, per-(game, draw) levels, run-stem-tagged shim records, stock sha intact."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_evid",
                            "--games", "cd82,lf52", "--draws", "2", "--per-game-s", "12", "--wave-cap-s", "60",
                            "--progress-every", "60", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        runs = list(Path(tmp).glob("*-regime-keith_evid-dry"))
        assert len(runs) == 1, runs
        out = runs[0]
        res = json.loads((out / "results.json").read_text())
        tel = json.loads((out / "telemetry.json").read_text())
        stems = ["cd82-fb555c5d_p0", "cd82-fb555c5d_p1", "lf52-271a04aa_p0", "lf52-271a04aa_p1"]
        assert sorted(p.name[:-4] for p in (out / "transcripts").glob("*.txt")) == stems
        assert res["status"] == "done" and res["arm"] == "keith_evid" and res["draws"] == 2
        assert res["stock"]["agent_tree_sha256"] == rw.STOCK_AGENT_TREE_SHA256
        assert res["grafts"]["installed"] == {"graft_evidence": "evidence: OK"}
        assert res["analyzer_env"]["EVID_ENABLE"] == "1" and res["analyzer_env"]["MULTIMODAL_UPSCALE"] == "4"
        assert res["geometry"]["max_runtime_s_per_game"] == 12.0
        rows = res["games"]
        assert [(g["game_id"], g["draw"], g["run_stem"]) for g in rows] == [
            ("cd82-fb555c5d", 0, stems[0]), ("lf52-271a04aa", 0, stems[2]),
            ("cd82-fb555c5d", 1, stems[1]), ("lf52-271a04aa", 1, stems[3])]      # taaf plays pass 0 then pass 1
        assert res["totals"] == {**res["totals"], "games": 4, "distinct_games": 2, "draws": 2}
        assert set(res["states"]) == set(stems)
        assert sorted(tel["per_game"]) == stems and tel["aggregate"]["draws"] == 2
        for stem, g in tel["per_game"].items():
            assert g["game_id"] == stem[:-3] and g["draw"] == int(stem[-1])
            executed = g["turn_outcomes"].get("step_executed", 0)
            assert executed >= 3, (stem, g["turn_outcomes"])
            assert g["evid_markers"] == executed, (stem, g["evid_markers"], executed)   # one block per executed turn
            assert g["engagement"]["evid"] == {"turns": executed, "of": executed, "share": 1.0}
            assert g["wall_level"] == g["levels_completed"] + 1 and g["level_reached"] == g["wall_level"]
            assert g["engagement"]["evid_wall"]["share"] == 1.0 and g["hypo_markers"] == 0
        pgd = tel["per_game_draw"]
        assert set(pgd) == {"cd82-fb555c5d", "lf52-271a04aa"} and set(pgd["cd82-fb555c5d"]) == {"0", "1"}
        assert all("levels_completed" in d and "evid_wall_share" in d for v in pgd.values() for d in v.values())
        agg = tel["aggregate"]
        assert agg["evid_markers_total"] == sum(g["evid_markers"] for g in tel["per_game"].values()) > 0
        assert agg["engagement"]["evid_wall"]["share"] == 1.0
        st = res["grafts"]["status"]["graft_evidence"]
        assert st["diffs_emitted"] == agg["evid_markers_total"] and st["errors"] == 0   # transcript markers == graft counter
        assert st["chars_added"] > 0 and set(st["per_game"]) == {"cd82-fb555c5d", "lf52-271a04aa"}
        # the block rides the tool result the model sees: in the transcript's [TOOL RESULT: python] and in the prompt log
        text = (out / "transcripts" / f"{stems[1]}.txt").read_text()
        assert "[TOOL RESULT: python]\n" in text and re.search(r"\[TOOL RESULT: python\]\n(?:.*\n)*?\[EVID\] harness object diff for action", text)
        assert "coords are (row,col), 0-based: row = line index of `.ascii` from the top" in text
        assert (out / "prompts" / f"{stems[1]}.log").read_text().count("[EVID] harness object diff") >= 1
        shim = [json.loads(l) for l in (out / "requests_shim.jsonl").read_text().splitlines() if l.strip()]
        assert {rec["run_stem"] for rec in shim} == set(stems) and all(rec["game_id"] == rec["run_stem"][:-3] for rec in shim)
        summary = (out / "summary.txt").read_text()
        assert "AID    [EVID]" in summary and "DRAWS 2 | levels per (game, draw): {'cd82-fb555c5d': [" in summary
        assert "engagement wall 100.0%" in summary and all(stem in summary for stem in stems)
        assert summary.count("\n") < 45
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path


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
