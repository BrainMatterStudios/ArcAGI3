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
    assert rw.ARMS == ("keith", "flight", "keith_yield180", "keith_yield900", "keith_retry", "keith_evid", "keith_hypo", "keith_up8",
                       "keith_probe", "keith_carry", "keith_ws", "keith_wsd", "keith_fx")
    # the original single-knob arm differs from the keith base on exactly the yield key
    d2 = {k for k in set(rw.KEITH_ANALYZER_ENV) | set(rw.KEITH_YIELD180_ENV)
          if rw.KEITH_ANALYZER_ENV.get(k) != rw.KEITH_YIELD180_ENV.get(k)}
    assert d2 == {"LOCAL_ANALYZER_YIELD_SECONDS"}, d2
    assert rw.KEITH_YIELD180_ENV["LOCAL_ANALYZER_YIELD_SECONDS"] == "180"
    # the effects arm differs from the keith base on exactly the graft's EFFECTS_* flags
    dfx = {k for k in set(rw.KEITH_ANALYZER_ENV) | set(rw.KEITH_FX_ENV)
           if rw.KEITH_ANALYZER_ENV.get(k) != rw.KEITH_FX_ENV.get(k)}
    assert dfx == set(rw.EFFECTS_ENV_KEYS), dfx
    assert rw.ARM_GRAFTS["keith_fx"] == ("graft_effects",)
    assert "EFFECTS_" in rw.GRAFT_ENV_PREFIXES and set(rw.EFFECTS_ENV_KEYS) <= set(rw.GRAFT_FLAG_KEYS)
    # the retry arm differs from the keith base on exactly the graft's RETRY_* flags
    d3 = {k for k in set(rw.KEITH_ANALYZER_ENV) | set(rw.KEITH_RETRY_ENV)
          if rw.KEITH_ANALYZER_ENV.get(k) != rw.KEITH_RETRY_ENV.get(k)}
    assert d3 == set(rw.RETRY_ENV_KEYS) == {"RETRY_ENABLE", "RETRY_K", "RETRY_ABS", "RETRY_COOLDOWN", "RETRY_MAX"}, d3
    assert {k: rw.KEITH_RETRY_ENV[k] for k in rw.RETRY_ENV_KEYS} == {
        "RETRY_ENABLE": "1", "RETRY_K": "3", "RETRY_ABS": "200", "RETRY_COOLDOWN": "150", "RETRY_MAX": "2"}
    assert rw.ARM_GRAFTS == {"keith_retry": ("graft_retry",), "keith_evid": ("graft_evidence",),
                             "keith_hypo": ("graft_hypo",), "keith_probe": ("graft_probe",),
                             "keith_carry": ("graft_carry",), "keith_ws": ("graft_workspace",),
                             "keith_wsd": ("graft_workspace",), "keith_fx": ("graft_effects",)}
    # 09-08 Track A1: the carry arm differs from keith_yield900 (its base) by exactly the graft's CARRY_* flags
    d5 = {k for k in set(rw.KEITH_YIELD900_ENV) | set(rw.KEITH_CARRY_ENV)
          if rw.KEITH_YIELD900_ENV.get(k) != rw.KEITH_CARRY_ENV.get(k)}
    assert d5 == set(rw.CARRY_ENV_KEYS) == {"CARRY_ENABLE", "CARRY_TARGET_FRACTION", "CARRY_SUMMARY_CHARS", "CARRY_INPUT_CHARS",
                                            "CARRY_COMPACT_MAX_TOKENS", "CARRY_COMPACT_THINKING", "CARRY_MIN_DROP_MSGS"}, d5
    assert {k: rw.KEITH_CARRY_ENV[k] for k in rw.CARRY_ENV_KEYS} == {
        "CARRY_ENABLE": "1", "CARRY_TARGET_FRACTION": "0.5", "CARRY_SUMMARY_CHARS": "4800", "CARRY_INPUT_CHARS": "48000",
        "CARRY_COMPACT_MAX_TOKENS": "1500", "CARRY_COMPACT_THINKING": "0", "CARRY_MIN_DROP_MSGS": "2"}
    assert rw.KEITH_CARRY_ENV["LOCAL_ANALYZER_YIELD_SECONDS"] == "900" and rw.KEITH_CARRY_ENV["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768"
    assert "CARRY_" in rw.GRAFT_ENV_PREFIXES and set(rw.CARRY_ENV_KEYS) <= set(rw.GRAFT_FLAG_KEYS)
    # 09-09 Track A2: the ws arm differs from keith_yield900 by exactly the graft's WS_* flags
    d6 = {k for k in set(rw.KEITH_YIELD900_ENV) | set(rw.KEITH_WS_ENV)
          if rw.KEITH_YIELD900_ENV.get(k) != rw.KEITH_WS_ENV.get(k)}
    assert d6 == set(rw.WS_ENV_KEYS), d6
    assert rw.KEITH_WS_ENV["WS_ENABLE"] == "1" and rw.KEITH_WS_ENV["LOCAL_ANALYZER_YIELD_SECONDS"] == "900"
    assert rw.ARM_GRAFTS["keith_ws"] == ("graft_workspace",)
    assert "WS_" in rw.GRAFT_ENV_PREFIXES and set(rw.WS_ENV_KEYS) <= set(rw.GRAFT_FLAG_KEYS)
    assert rw.ARMS[-3:] == ("keith_ws", "keith_wsd", "keith_fx")
    d7 = {k for k in set(rw.KEITH_WS_ENV) | set(rw.KEITH_WSD_ENV) if rw.KEITH_WS_ENV.get(k) != rw.KEITH_WSD_ENV.get(k)}
    assert d7 == {"WS_DIRECT_ENABLE", "WS_DIRECT_AFTER_ACTIONS", "WS_DIRECT_MAX_CALLS", "WS_DIRECT_MAX_PER_GAME",
                  "WS_DIRECT_MAX_TRANSITIONS", "WS_DIRECT_MAX_CELLS"}, d7
    assert rw.KEITH_WSD_ENV["WS_DIRECT_MAX_TRANSITIONS"] == "12" and rw.KEITH_WSD_ENV["WS_DIRECT_MAX_CELLS"] == "60"
    assert rw.KEITH_WSD_ENV["WS_DIRECT_ENABLE"] == "1" and "WS_DIRECT_ENABLE" not in rw.KEITH_WS_ENV
    sys.path.insert(0, str(rw.GRAFT_DIR))
    import graft_carry  # noqa: PLC0415
    assert graft_carry.COMPACT_SYSTEM_HEAD == rw.CARRY_COMPACT_HEAD                      # the mock recognises compaction requests
    assert graft_carry.DEFAULT_WINDOW_TOKENS == rw.CARRY_WINDOW_TOKENS
    assert graft_carry._PER_GAME_KEYS == rw._CARRY_GRAFT_KEYS
    # 09-08: the probe arm differs from keith_yield900 (its base) by exactly the graft's PROBE_* flags
    d4 = {k for k in set(rw.KEITH_YIELD900_ENV) | set(rw.KEITH_PROBE_ENV)
          if rw.KEITH_YIELD900_ENV.get(k) != rw.KEITH_PROBE_ENV.get(k)}
    assert d4 == set(rw.PROBE_ENV_KEYS) == {"PROBE_ENABLE", "PROBE_MAX_ANALYSIS", "PROBE_MAX_PROBE", "PROBE_MAX_REFUSALS",
                                            "PROBE_NOTE_LINES"}, d4
    assert {k: rw.KEITH_PROBE_ENV[k] for k in rw.PROBE_ENV_KEYS} == {
        "PROBE_ENABLE": "1", "PROBE_MAX_ANALYSIS": "2", "PROBE_MAX_PROBE": "5", "PROBE_MAX_REFUSALS": "4", "PROBE_NOTE_LINES": "3"}
    assert set(rw.NEVER6_WALLS) == {"bp35", "dc22", "g50t", "lf52", "lp85", "ls20", "r11l", "sb26", "sp80", "tn36", "vc33", "wa30"}
    assert rw.LEDGER3_REFERENCE["base_levels_mean"] == 39.33 and rw.LEDGER3_REFERENCE["base_levels_sd"] == 2.34
    assert abs(statistics.mean(rw.LEDGER3_REFERENCE["base_levels_six_draws"]) - 39.33) < 0.01
    assert abs(statistics.stdev(rw.LEDGER3_REFERENCE["base_levels_six_draws"]) - 2.34) < 0.01
    assert rw.KEITH_PROBE_ENV["LOCAL_ANALYZER_YIELD_SECONDS"] == "900" and rw.KEITH_YIELD900_ENV["LOCAL_ANALYZER_YIELD_SECONDS"] == "900"
    assert "PROBE_" in rw.GRAFT_ENV_PREFIXES and set(rw.PROBE_ENV_KEYS) <= set(rw.GRAFT_FLAG_KEYS)
    # 09-06 arms: each differs from the keith base by exactly its own keys
    def _diff(env):
        return {k for k in set(rw.KEITH_ANALYZER_ENV) | set(env) if rw.KEITH_ANALYZER_ENV.get(k) != env.get(k)}
    assert _diff(rw.KEITH_EVID_ENV) == set(rw.EVID_ENV_KEYS) == {"EVID_ENABLE", "EVID_MAX_ENTRIES", "EVID_MAX_CHARS", "EVID_TRACE"}
    assert {k: rw.KEITH_EVID_ENV[k] for k in rw.EVID_ENV_KEYS} == {"EVID_ENABLE": "1", "EVID_MAX_ENTRIES": "40",
                                                                    "EVID_MAX_CHARS": "1500", "EVID_TRACE": "1"}
    assert _diff(rw.KEITH_HYPO_ENV) == set(rw.HYPO_ENV_KEYS) == {"HYPO_ENABLE"} and rw.KEITH_HYPO_ENV["HYPO_ENABLE"] == "1"
    assert _diff(rw.KEITH_UP8_ENV) == {"MULTIMODAL_UPSCALE"} and rw.KEITH_UP8_ENV["MULTIMODAL_UPSCALE"] == "8"
    assert rw.KEITH_ANALYZER_ENV["MULTIMODAL_UPSCALE"] == "4" and rw.KEITH_UP8_ENV["MULTIMODAL_CONTEXT"] == "current_grid"
    for arm in ("keith", "flight", "keith_yield180", "keith_yield900", "keith_up8"):
        assert not any(k.startswith(rw.GRAFT_ENV_PREFIXES) for k in rw.ARM_ENV[arm]), arm   # stock arms carry no graft flag
    for arm, prefix in (("keith_retry", "RETRY_"), ("keith_evid", "EVID_"), ("keith_hypo", "HYPO_"), ("keith_probe", "PROBE_"),
                        ("keith_carry", "CARRY_"), ("keith_ws", "WS_")):
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
            # a stale PROBE_* flag never reaches the yield900 base arm; the probe arm sets exactly its keys
            os.environ["PROBE_ENABLE"] = "1"
            os.environ["PROBE_MAX_ANALYSIS"] = "0"
            rec = rw.install_env("keith_yield900", "http://127.0.0.1:9/v1", "t", Path(tmp))
            assert not any(k.startswith("PROBE_") for k in os.environ) and not any(k.startswith("PROBE_") for k in rec)
            assert rec["LOCAL_ANALYZER_YIELD_SECONDS"] == "900"
            os.environ["EVID_ENABLE"] = "1"
            rec = rw.install_env("keith_probe", "http://127.0.0.1:9/v1", "t", Path(tmp))
            assert "EVID_ENABLE" not in os.environ and os.environ["PROBE_MAX_ANALYSIS"] == "2" and rec["PROBE_MAX_REFUSALS"] == "4"
            assert rec["LOCAL_ANALYZER_YIELD_SECONDS"] == "900" and {k for k in rec if k.startswith("PROBE_")} == set(rw.PROBE_ENV_KEYS)
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
reasoning_chars: 40
[THINKING]
The harness diff says MOVED b/blue ... -> (5,3)-(6,4); use that.
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
    # judge 09-06 wall reads: quotes, calls@L2, attempt, void, pass
    assert [t["quotes"] for t in parsed["turns"]] == [0, 3, 0, 0]      # prose "[EVID]"/"[HYPO]" + the coordinate cite in turn 2
    w = tel["wall"]
    assert w["calls_total"] == 6 and w["calls_at_l2"] == 4 and w["calls_budget"] == 6 and w["calls_remaining_at_l2"] == 2
    assert w["reached_l2"] and not w["attempt"] and not w["passed"] and not w["void"] and w["void_reasons"] == []
    assert w["uptake"] == {"turns": 1, "of": 4, "share": 0.25} and w["uptake_wall"] == {"turns": 0, "of": 2, "share": 0.0}
    w60 = rw.telemetry_for_game(AID_TRANSCRIPT, levels_completed=1, number_of_levels=4, calls_budget=60)["wall"]
    assert w60["calls_remaining_at_l2"] == 56 and w60["attempt"] is True
    wp = rw.telemetry_for_game(AID_TRANSCRIPT, levels_completed=2, number_of_levels=4, calls_budget=60, wave_preemptions=3)["wall"]
    assert wp["passed"] is True and wp["void"] is True and wp["void_reasons"] == ["preemptions=3"] and wp["attempt"] is False
    err = rw.telemetry_for_game(AID_TRANSCRIPT + "\n[ANALYZER STATUS]\nrequest_error: boom\n", levels_completed=1,
                                number_of_levels=4, calls_budget=60)["wall"]
    assert err["void"] and err["void_reasons"] == ["request_errors=1"]
    long_ = rw.telemetry_for_game(AID_TRANSCRIPT.replace("finish_reason: stop", "finish_reason: length"), levels_completed=1,
                                  number_of_levels=4, calls_budget=60)["wall"]
    assert long_["void"] and long_["void_reasons"] == ["length_finish_share=0.167"]     # 1 of 6 calls > 1 %
    zero = rw.telemetry_for_game(SYNTHETIC_TRANSCRIPT, levels_completed=0, number_of_levels=4, calls_budget=60)["wall"]
    assert zero["calls_at_l2"] is None and not zero["reached_l2"] and not zero["attempt"]
    shim = [{"n_messages": 2, "status": 200, "prompt_tokens": 4242, "elapsed_s": 1.0}, {"n_messages": 5, "status": 200, "prompt_tokens": 9000, "elapsed_s": 1.0}]
    assert rw.telemetry_for_game(AID_TRANSCRIPT, shim_records=shim)["first_call_prompt_tokens"] == 4242
    agg2 = rw.aggregate_telemetry({"a_p0": tel, "a_p1": rw.telemetry_for_game(AID_TRANSCRIPT, shim_records=shim, levels_completed=2,
                                                                               number_of_levels=4, calls_budget=60)})
    assert agg2["wall"] == {**agg2["wall"], "runs": 2, "void": 0, "attempts": 1, "passes": 1, "reached_l2": 2}
    # run 2 (levels_completed=2 of 4) sits on wall level 3, where the transcript has no turn: pooled wall denominator 2
    assert agg2["wall"]["uptake_wall"] == {"turns": 0, "of": 2, "share": 0.0} and agg2["first_call_prompt_tokens"]["n"] == 1


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


def test_probe_arm_installs_graft_in_memory():
    """keith_probe: graft_probe rebinds analyze, _run_python_tool AND _build_user_prompt; the stock tree
    sha holds; the factory-built ToolAgent carries the yield900 regime (900 s yield, keith window)."""
    with tempfile.TemporaryDirectory() as tmp:
        code = (_AID_GRAFT_PROBE.replace("__HERE__", repr(str(HERE))).replace("__TMP__", repr(tmp))
                .replace("__ARM__", repr("keith_probe")))
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        probe = json.loads(r.stdout.strip().splitlines()[-1])
    assert probe["grafts"] == {"graft_probe": "probe: OK"}
    assert probe["rebound"] == {"run": True, "prompt": True, "analyze": True}, probe["rebound"]
    assert probe["sha"] == rw.STOCK_AGENT_TREE_SHA256
    st = probe["status"]["graft_probe"]
    assert st["installed"] and st["enabled"] and st["errors"] == 0
    assert (st["max_analysis"], st["max_probe"], st["max_refusals"], st["note_lines"]) == (2, 5, 4, 3)
    assert st["refusals"] == 0 and st["turns_total"] == 0 and st["per_game"] == {}
    assert probe["env"] == {"PROBE_ENABLE": "1", "PROBE_MAX_ANALYSIS": "2", "PROBE_MAX_PROBE": "5", "PROBE_MAX_REFUSALS": "4",
                            "PROBE_NOTE_LINES": "3"}
    fp = probe["fingerprint"]
    assert fp["context_budget_tokens"] == 31744 and fp["max_output_tokens"] is None and fp["yield_seconds"] == 900.0


PROBE_TRANSCRIPT = """
--- analysis_step=1 | action=0 | 10:00:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
No previous sequence has been executed yet.
Current state: step 1, level 1.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 3
[THINKING]
abc
[TOOL CALL: python]
<tool_call>
<function=python>
<parameter=code>
print(len(history))
</parameter>
</function>
</tool_call>
[TOOL RESULT: python]
0
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "x = 1 / 0"}
[TOOL RESULT: python]
Traceback (most recent call last):
  File "<python_tool>", line 1, in <module>
ZeroDivisionError: division by zero
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
<tool_call>
<function=python>
<parameter=code>
print('third analysis')
</parameter>
</function>
</tool_call>
[HARNESS PROBE]
[PROBE-REFUSE] game=tu93-0768757b_p0 turn=1 analysis_calls=2 refusal=1/2

[TOOL RESULT: python]
Analysis budget for this turn is spent (2 analysis-only calls). Only a snippet that executes a game action is accepted now: run a <=5-action test of your leading hypothesis with action([...]) and read the result. Untested hypotheses in your notes: (none recorded - state one now and test it)
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
<tool_call>
<function=python>
<parameter=code>
r = action(['UP'])
print(r)
</parameter>
</function>
</tool_call>
[TOOL RESULT: python]
{'executed': True}
[ANALYZER STATUS]
step_executed: True
message: Step executed.

--- analysis_step=2 | action=1 | 10:01:00 | tool-agent ---
[USER PROMPT]
The code executed 1 action in the previous sequence.
Current state: step 2, level 1.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "print(1)"}
[TOOL RESULT: python]
1
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "print(2)"}
[TOOL RESULT: python]
2
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "print(3)"}
[HARNESS PROBE]
[PROBE-REFUSE] game=tu93-0768757b_p0 turn=2 analysis_calls=2 refusal=1/2

[TOOL RESULT: python]
Analysis budget for this turn is spent (2 analysis-only calls). Only a snippet that executes a game action is accepted now: run a <=5-action test of your leading hypothesis with action([...]) and read the result. Untested hypotheses in your notes: (none recorded - state one now and test it)
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "print(4)"}
[HARNESS PROBE]
[PROBE-REFUSE] game=tu93-0768757b_p0 turn=2 analysis_calls=2 refusal=2/2

[TOOL RESULT: python]
Analysis budget for this turn is spent (2 analysis-only calls). Only a snippet that executes a game action is accepted now: run a <=5-action test of your leading hypothesis with action([...]) and read the result. Untested hypotheses in your notes: (none recorded - state one now and test it)
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "print(5)"}
[TOOL RESULT: python]
5
[ANALYZER STATUS]
step_executed: False
message: Yielded control to solver: turn_time_budget.
[HARNESS PROBE]
[PROBE-NOACT] game=tu93-0768757b_p0 turn=2 analysis_calls=3 refusals=2 reason=yield

--- analysis_step=2 | action=1 | 10:16:00 | tool-agent ---
[USER PROMPT]
Previous turn executed no action after 3 analysis calls (2 refused).
The code executed 1 action in the previous sequence.
Current state: step 2, level 1.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "action(['DOWN'])"}
[TOOL RESULT: python]
{'executed': True}
[ANALYZER STATUS]
step_executed: True
message: Step executed.
"""


def test_game_overs_from_events_counts_compact_json(tmp_path=None):
    """The harness writes events.jsonl with json.dumps(separators=(",", ":")) — no space after the colon.
    The 09-08 probe wave read 0 GAME_OVERs because the pre-filter looked for '"game_over": true'.
    (tmp_path is pytest's fixture; the script runner passes nothing and gets a TemporaryDirectory.)"""
    import json as _json
    if tmp_path is None:
        _tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(_tmp.name)
    f = tmp_path / "x_events.jsonl"
    rows = [{"type": "initial", "game_over": False},
            {"type": "action", "game_over": True, "action_display": "ACTION1"},
            {"type": "action", "game_over": False},
            {"type": "action", "game_over": True},
            {"type": "level", "game_over": True}]          # not an action row -> not counted
    compact = "\n".join(_json.dumps(r, separators=(",", ":")) for r in rows)
    spaced = "\n".join(_json.dumps(r) for r in rows)
    f.write_text(compact + "\n" + spaced, encoding="utf-8")
    assert rw.game_overs_from_events(f) == 4
    assert rw.game_overs_from_events(tmp_path / "missing_events.jsonl") is None


def test_extractor_reads_probe_markers_and_ledger_call_types():
    parsed = rw.parse_transcript(PROBE_TRANSCRIPT)
    turns, calls = parsed["turns"], parsed["calls"]
    assert [t["call_types"] for t in turns] == ["AERX", "AARRA", "X"]
    assert [t["probe_refusals"] for t in turns] == [1, 2, 0]
    assert [t["probe_noact"] for t in turns] == [0, 1, 0]
    assert [t["probe_noact_notice"] for t in turns] == [0, 0, 1]
    assert [t["analysis_before_act"] for t in turns] == [1, 3, 0]          # A-class only (E, R excluded)
    assert [t["nonacting_before_act"] for t in turns] == [3, 5, 0]         # strict: everything before the first X
    assert [t["yield_turn_time_budget"] for t in turns] == [False, True, False]
    assert [(t["calls_after_refusal"], t["acting_after_refusal"]) for t in turns] == [(1, 1), (2, 0), (0, 0)]
    assert [(t["first_refusal_followups"], t["acted_after_first_refusal"]) for t in turns] == [(1, 1), (1, 0), (0, 0)]
    assert [c["refused"] for c in calls] == [0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    assert [c["acts_code"] for c in calls] == [False, False, False, True, False, False, False, False, False, True]
    assert parsed["probe_markers"] == {"refusals": 3, "noact_turns": 1, "noact_notices": 1}
    # a refusal that is the turn's last call is a non-acting follow-up (transcript read): cut turn 1 after its refusal
    head = PROBE_TRANSCRIPT.split("[MODEL RESPONSE META]\nfinish_reason: tool_calls\ntool_call_count: 1\ncontent_chars: 0\nreasoning_chars: 0\n[TOOL CALL: python]\n<tool_call>\n<function=python>\n<parameter=code>\nr = action(['UP'])", 1)[0]
    ending = rw.parse_transcript(head + "[ANALYZER STATUS]\nstep_executed: False\nmessage: Yielded control to solver: turn_time_budget.\n")
    assert (ending["turns"][0]["first_refusal_followups"], ending["turns"][0]["acted_after_first_refusal"]) == (1, 0)
    assert (ending["turns"][0]["calls_after_refusal"], ending["turns"][0]["acting_after_refusal"]) == (1, 0)
    graft = {"refusals": 3, "calls_after_refusal": 3, "acting_calls_after_refusal": 1, "first_refusal_followups": 2,
             "acted_after_first_refusal": 1, "refusal_turn_ending": 1, "carried_turns": 1, "noact_turns": 1, "turns_total": 3,
             "turns_ge3_analysis": 1, "leak_cap_lifted": 1, "leak_dead_branch": 0, "leak_unparsable": 0,
             "analysis_calls_total": 5, "acting_calls_total": 2, "turns_with_refusal": 2}
    tel = rw.telemetry_for_game(PROBE_TRANSCRIPT, actions_total=2, levels_completed=0, number_of_levels=9,
                                actions_per_level=[30, 0, 0], baselines=[19, 16, 34], graft_counters=graft, game_overs=2)
    p = tel["probe"]
    assert (p["refusals"], p["noact_turns"], p["noact_notices"], p["turns_with_refusal"]) == (3, 1, 1, 2)
    assert p["turns_ge3_analysis"] == {"turns": 1, "of": 3, "share": 1 / 3}
    assert p["turns_ge3_nonacting"] == {"turns": 2, "of": 3, "share": 2 / 3}
    assert p["acting_after_refusal"] == {"acted": 1, "of": 3, "share": 1 / 3}
    assert p["acted_after_first_refusal"] == {"acted": 1, "of": 2, "share": 0.5}
    assert p["call_types"] == {"A": 4, "E": 1, "R": 3, "X": 2} and abs(p["analysis_call_share"] - 0.4) < 1e-9
    assert p["yields_turn_time_budget"] == 1
    assert (p["wall_actions"], p["wall_baseline"]) == (30, 19) and abs(p["wall_actions_ratio"] - 30 / 19) < 1e-9
    assert p["graft"]["acting_after_refusal_share"] == 1 / 3 and p["graft"]["refusals"] == 3
    assert p["graft"]["acted_after_first_refusal_share"] == 0.5 and p["graft"]["leak_cap_lifted"] == 1
    assert p["game_overs"] == 2 and p["live_cap_score"] == 0.0            # 0 levels -> 0 live-cap score
    # live-cap score = the ledger's score(): level 1 cleared in 30 actions vs baseline 19 -> (19/30)^2 * 1 / 45 * 100
    assert abs(rw.live_cap_score([19, 16, 34, 42, 123, 80, 14, 23, 111], [30, 5], 1) - 100 * (19 / 30) ** 2 / 45) < 1e-9
    assert abs(rw.live_cap_score([19, 16], [5, 100], 2) - 100 * (1.15 + 2 * (16 / 100) ** 2) / 3) < 1e-9   # cap 1.15 on level 1
    assert rw.live_cap_score(None, [1], 1) is None and rw.live_cap_score([], [1], 1) is None
    # a won run has no wall ratio; missing baselines -> None
    assert rw.telemetry_for_game(PROBE_TRANSCRIPT, levels_completed=9, number_of_levels=9, actions_per_level=[1] * 9,
                                 baselines=[1] * 9)["probe"]["wall_actions_ratio"] is None
    assert rw.telemetry_for_game(PROBE_TRANSCRIPT, levels_completed=0, number_of_levels=9, actions_per_level=[30],
                                 baselines=None)["probe"]["wall_actions_ratio"] is None
    # stock transcripts: no markers, ratio still computed, gate NOT engaged
    stock = rw.telemetry_for_game(SYNTHETIC_TRANSCRIPT, levels_completed=1, number_of_levels=4, actions_per_level=[10, 40, 0, 0],
                                  baselines=[20, 20, 20, 20])
    assert stock["probe"]["refusals"] == 0 and stock["probe"]["wall_actions_ratio"] == 2.0
    tel2 = rw.telemetry_for_game(PROBE_TRANSCRIPT, levels_completed=2, number_of_levels=9, actions_per_level=[5, 8, 40],
                                 baselines=[19, 16, 34], game_overs=0)
    tel2["draw"] = 1
    for t_, gid, lv in ((tel, "vc33-5430563c", 0), (tel2, "vc33-5430563c", 2)):
        t_["game_id"], t_["levels_completed"] = gid, lv
    agg = rw.aggregate_telemetry({"vc33_p0": tel, "vc33_p1": tel2})
    pb = agg["probe"]
    assert pb["refusals_total"] == 6 and pb["refusals_per_game"] == 3.0 and pb["games_with_refusal"] == 2
    assert pb["noact_turns_total"] == 2 and pb["noact_notices_total"] == 2
    assert pb["turns_ge3_analysis"] == {"turns": 2, "of": 6, "share": 1 / 3}
    assert pb["acting_after_refusal"] == {"acted": 2, "of": 6, "share": 1 / 3}
    assert pb["acted_after_first_refusal"] == {"acted": 2, "of": 4, "share": 0.5}
    assert pb["graft"] == {**pb["graft"], "runs": 1, "calls_after_refusal": 3, "acting_calls_after_refusal": 1,
                           "acting_after_refusal_share": 1 / 3, "refusals": 3, "first_refusal_followups": 2,
                           "acted_after_first_refusal": 1, "acted_after_first_refusal_share": 0.5, "refusal_turn_ending": 1,
                           "carried_turns": 1, "turns_ge3_analysis": 1, "turns_ge3_analysis_share": 1 / 3,
                           "leak_cap_lifted": 1, "leak_dead_branch": 0, "leak_unparsable": 0}
    assert pb["call_types"] == {"A": 8, "E": 2, "R": 6, "X": 4}
    assert pb["draws"] == 2 and pb["yields_turn_time_budget_total"] == 2 and pb["yields_per_draw"] == 1.0
    assert pb["wall_actions_ratio"]["n"] == 2 and pb["wall_actions_ratio"]["under_1x"] == 0     # run 2 sits on level 3: 40/34
    assert abs(pb["wall_actions_ratio"]["median"] - (30 / 19 + 40 / 34) / 2) < 1e-9
    pri, saf = pb["primary"], pb["safety"]
    assert (pri["levels_total"], pri["draws"], pri["levels_per_draw"]) == (2, 2, 1.0)
    assert abs(pri["delta_vs_base"] - (1.0 - 39.33)) < 1e-9 and pri["base_levels_sd"] == 2.34
    assert pri["walls_present"] == ["vc33"] and pri["walls_passed"] == [] and pri["walls_passed_n"] == 0 and pri["walls_total"] == 12
    assert (saf["game_overs_total"], saf["game_overs_per_run"], saf["game_overs_runs"]) == (2, 1.0, 2)
    assert saf["base_game_overs_per_run"] == 0.87 and saf["base_live_cap_score_per_game"] == 8.42
    # run 2: level 1 capped at 1.15 (5 vs 19), level 2 capped at 1.15 (8 vs 16 -> 4.0), weights 1..3 -> 100*(1.15+2.3)/6
    assert saf["live_cap_runs"] == 2 and abs(saf["live_cap_score_per_game"] - (0.0 + 100 * (1.15 + 2 * 1.15) / 6) / 2) < 1e-9
    g = pb["gate"]
    assert g["refusals_per_game"]["ok"] and g["acted_after_first_refusal"]["ok"] and g["wall_actions_ratio"]["ok"]
    assert not g["turns_ge3_analysis"]["ok"] and g["turns_ge3_analysis"]["secondary"] and g["yields_per_draw"]["ok"]
    assert g["engaged"] is True and abs(g["turns_ge3_analysis"]["value"] - 1 / 3) < 1e-9      # graft read (1 of 3 turns)
    # a wall passed: vc33 reaching level 5 (>= its never-passed wall L4)
    tel3 = dict(tel2)
    tel3["levels_completed"] = 4
    tel3["draw"] = 0
    pri3 = rw.aggregate_telemetry({"vc33_p0": tel3})["probe"]["primary"]
    assert pri3["walls_passed"] == ["vc33"] and pri3["walls_passed_n"] == 1
    # gate fails on the wall ratio alone (compliant but under-exploring)
    tel4 = rw.telemetry_for_game(PROBE_TRANSCRIPT, levels_completed=0, number_of_levels=9, actions_per_level=[5, 0, 0],
                                 baselines=[19, 16, 34], graft_counters=graft)
    g4 = rw.aggregate_telemetry({"a_p0": tel4})["probe"]["gate"]
    assert g4["refusals_per_game"]["ok"] and g4["acted_after_first_refusal"]["ok"] and not g4["wall_actions_ratio"]["ok"]
    assert g4["engaged"] is False
    # pooled from a stock run only: nothing refused -> not engaged, no crash
    stock["game_id"] = "tu93-0768757b"
    agg0 = rw.aggregate_telemetry({"a_p0": stock})
    assert agg0["probe"]["refusals_total"] == 0 and agg0["probe"]["gate"]["engaged"] is False
    assert agg0["probe"]["acting_after_refusal"]["share"] is None and agg0["probe"]["primary"]["walls_present"] == []
    assert agg0["probe"]["safety"]["game_overs_per_run"] is None
    # a "[PROBE-REFUSE]" mention in model prose is not a marker
    assert rw.parse_transcript("hello [PROBE-REFUSE] game=x turn=1 analysis_calls=2\n")["probe_markers"]["refusals"] == 0


def test_dry_run_keith_probe_arm_end_to_end():
    """The pre-registered launch shape on the loopback mock (+ the real engine): the mock answers the first
    3 python calls of every turn with analysis-only snippets, the graft refuses the 3rd (the sentinel never
    runs), the next call acts, and the PROBE / PROBE-WALL summary lines + telemetry carry the reads."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_probe",
                            "--games", "tu93,ft09,cd82", "--concurrency", "3", "--max-calls", "12",
                            "--per-game-s", "20", "--wave-cap-s", "90", "--progress-every", "60", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        runs = list(Path(tmp).glob("*-regime-keith_probe-dry"))
        assert len(runs) == 1, runs
        out = runs[0]
        res = json.loads((out / "results.json").read_text())
        tel = json.loads((out / "telemetry.json").read_text())
        assert res["status"] == "done" and res["arm"] == "keith_probe" and res["mock_analysis_calls"] == 3
        assert res["stock"]["agent_tree_sha256"] == rw.STOCK_AGENT_TREE_SHA256
        assert res["grafts"]["installed"] == {"graft_probe": "probe: OK"}
        assert res["analyzer_env"]["LOCAL_ANALYZER_YIELD_SECONDS"] == "900" and res["analyzer_env"]["PROBE_ENABLE"] == "1"
        assert res["analyzer_env"]["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768" and "knob_overrides" not in res
        cfg = tel["aggregate"]["analyzer_status_config_first"]
        assert cfg == {"max_output_tokens": "server default", "context_budget_tokens": "31744", "yield_seconds": "900.0"}, cfg
        st = res["grafts"]["status"]["graft_probe"]
        pb = tel["aggregate"]["probe"]
        assert st["refusals"] >= 3 and st["errors"] == 0
        assert pb["refusals_total"] == st["refusals"]                                  # transcript markers == graft counter
        assert pb["games_with_refusal"] == 3 and pb["graft"]["refusals"] == st["refusals"]
        # the mock complies with every refusal that is followed by a call; the run's last call may be a refusal
        # (--max-calls stop), which is settled as a non-acting turn-ending refusal
        assert st["acting_calls_after_refusal"] == st["calls_after_refusal"] - st["refusal_turn_ending"] >= 3
        assert st["acted_after_first_refusal"] == st["first_refusal_followups"] - st["refusal_turn_ending"]
        assert pb["graft"]["acted_after_first_refusal_share"] >= 0.5 and pb["acted_after_first_refusal"]["share"] >= 0.5
        assert pb["graft"]["acted_after_first_refusal"] == st["acted_after_first_refusal"]
        assert pb["turns_ge3_analysis"]["turns"] == 0 and st["turns_ge3_analysis"] == 0    # never 3 executed analysis calls
        assert (st["leak_cap_lifted"], st["leak_dead_branch"], st["leak_unparsable"], st["carried_turns"]) == (0, 0, 0, 0)
        assert pb["call_types"].get("R", 0) == st["refusals"] and pb["call_types"].get("X", 0) >= 3
        gate = pb["gate"]
        assert gate["refusals_per_game"]["ok"] and gate["acted_after_first_refusal"]["ok"]
        assert gate["wall_actions_ratio"]["ok"] is False and gate["engaged"] is False       # 2 actions vs baseline: under-explored
        assert gate["turns_ge3_analysis"]["ok"] and gate["turns_ge3_analysis"]["secondary"]
        assert pb["wall_actions_ratio"]["n"] == 3 and all(
            g["probe"]["wall_baseline"] and g["probe"]["wall_actions_ratio"] is not None for g in tel["per_game"].values())
        pri, saf = pb["primary"], pb["safety"]
        assert pri["games"] == 3 and pri["draws"] == 1 and pri["walls_total"] == 12 and pri["walls_present"] == []   # none of the 3 games is a never-passed wall
        assert abs(pri["delta_vs_base"] - (pri["levels_total"] - 39.33)) < 1e-9
        assert saf["game_overs_runs"] == 3 and saf["game_overs_per_run"] is not None and saf["live_cap_runs"] == 3
        assert all(g["probe"]["game_overs"] is not None and g["probe"]["live_cap_score"] is not None for g in tel["per_game"].values())
        assert all(g["baselines"] for g in res["games"])                                  # offline engine exposes baselines
        per = tel["per_game"]
        assert set(per) == {"tu93-0768757b_p0", "ft09-0d8bbf25_p0", "cd82-fb555c5d_p0"}
        for stem, g in per.items():
            text = (out / "transcripts" / f"{stem}.txt").read_text()
            assert text.count("[HARNESS PROBE]\n[PROBE-REFUSE] game=" + stem) == g["probe"]["refusals"] == st["per_game"][stem]["refusals"]
            assert "[TOOL RESULT: python]\n" + rw.MOCK_SENTINEL not in text               # the refused snippet never ran
            assert text.count("[TOOL RESULT: python]\nAnalysis budget for this turn is spent (2 analysis-only calls)") == g["probe"]["refusals"]
            assert g["probe"]["graft"]["refusals"] == g["probe"]["refusals"]
        summary = (out / "summary.txt").read_text()
        assert "REGIME WAVE  arm=keith_probe" in summary and "PROBE  refusals" in summary
        for line in ("PROBE-2ND spans >=3", "PROBE-PRIMARY levels", "vs pooled six-draw base 39.33 (sd 2.34)", "never-passed walls 0/12",
                     "PROBE-SAFETY GAME_OVERs", "yield900 base 0.87", "yield900 base 8.42/game", "ENGAGED = NO {refusals_per_game+, "
                     "acted_after_first_refusal+, wall_actions_ratio-}"):
            assert line in summary, line
        assert "'PROBE_MAX_ANALYSIS': '2'" in summary and "'PROBE_MAX_REFUSALS': '4'" in summary and "yield 900 s" in summary
        assert summary.count("\n") < 45
        ms = res["mock_state"]
        assert ms["sentinel_calls"] >= 3 and ms["analysis_calls"] >= 9
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path
    # a stock arm with the same mock knob RUNS the sentinel (the refusal is the graft, not the mock)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_yield900",
                            "--games", "tu93", "--concurrency", "1", "--max-calls", "6", "--mock-analysis-calls", "3",
                            "--per-game-s", "12", "--wave-cap-s", "60", "--progress-every", "60", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        out = list(Path(tmp).glob("*-regime-keith_yield900-dry"))[0]
        text = (out / "transcripts" / "tu93-0768757b_p0.txt").read_text()
        assert "[TOOL RESULT: python]\n" + rw.MOCK_SENTINEL in text and "[PROBE-REFUSE]" not in text
        tel = json.loads((out / "telemetry.json").read_text())
        assert tel["aggregate"]["probe"]["refusals_total"] == 0 and tel["aggregate"]["probe"]["turns_ge3_analysis"]["turns"] >= 1
        summary = (out / "summary.txt").read_text()
        assert "PROBE  refusals" not in summary.split("\n  run ")[0]          # stock arm: no PROBE lines
        assert json.loads((out / "results.json").read_text())["mock_analysis_calls"] == 3


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
                            "--games", "cd82,lf52", "--draws", "2", "--per-game-s", "40", "--wave-cap-s", "120",
                            "--concurrency", "3", "--max-calls", "8", "--progress-every", "60", "--out", tmp],
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
        assert res["geometry"]["max_runtime_s_per_game"] == 40.0 and res["geometry"]["concurrency"] == 3
        assert res["concurrency_override"] is True and res["max_calls"] == 8
        # --max-calls: every run made exactly 8 analyzer calls, then ended through its own runtime cap (gave_up)
        assert all(g["calls"] == 8 for g in tel["per_game"].values()), {k: g["calls"] for k, g in tel["per_game"].items()}
        assert all(g["state"] == "gave_up" for g in res["games"]), res["states"]
        assert sorted(res["max_calls_stops"]) == stems and all(v == 8 for v in res["max_calls_stops"].values())
        assert all(g["wallclock_s"] < 40 for g in res["games"])                # the stop came from max-calls, not the cap
        for g in tel["per_game"].values():
            w = g["wall"]
            assert w["calls_budget"] == 8 and w["calls_total"] == 8 and w["void"] is False and w["attempt"] is False
            assert w["uptake"]["of"] == g["turns"] and g["first_call_prompt_tokens"] is not None
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
        assert "WALL   attempts 0/4 (L2 reached 0; attempt = L2 with >= 30 calls left) | passes 0" in summary
        assert "FIRST-CALL prompt_tokens (n_messages==2) mean" in summary and "max_calls 8" in summary
        assert "engagement wall 100.0%" in summary and all(stem in summary for stem in stems)
        assert summary.count("\n") < 45
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path


# --- 09-08 Track A1: keith_carry (graft_carry, compaction instead of eviction) ----------------------

_CARRY_INSTALL_CODE = r'''
import json, os, sys
sys.path.insert(0, __HERE__)
import run_regime_wave as rw
from pathlib import Path
out = Path(__TMP__)
env = rw.install_env("keith_carry", "http://127.0.0.1:9/v1", "carry-token", out)
rw.install_paths()
rw.verify_imports()
from inference.agent import tool_agent as ta
before = {"analyze": ta.ToolAgent.analyze, "trim": ta.ToolAgent._trim_messages_for_context, "chat": ta.ToolAgent._chat_completion,
          "run": ta.ToolAgent._run_python_tool}
grafts = rw.install_grafts("keith_carry")
after = {"analyze": ta.ToolAgent.analyze, "trim": ta.ToolAgent._trim_messages_for_context, "chat": ta.ToolAgent._chat_completion,
         "run": ta.ToolAgent._run_python_tool}
stock = rw.assert_stock_tree()
bm, target = rw.load_bundle(out)
factory = rw.make_tagging_analyzer_factory(bm.solver, n_games=2)
class _G:
    class game_run: game_id = "cd82-fb555c5d"
agent = factory(_G(), 3)
print(json.dumps({"grafts": grafts, "rebound": {k: after[k] is not before[k] for k in before},
                  "sha": stock["agent_tree_sha256"], "status": rw.graft_status("keith_carry"),
                  "env": {k: os.environ.get(k) for k in rw.GRAFT_FLAG_KEYS if os.environ.get(k) is not None},
                  "fingerprint": rw.analyzer_config_fingerprint(agent)}))
'''


def test_carry_arm_installs_graft_in_memory():
    """keith_carry: graft_carry rebinds analyze, _trim_messages_for_context and _chat_completion (NOT the python
    tool); the stock tree sha holds; the factory-built ToolAgent carries the yield900 regime."""
    with tempfile.TemporaryDirectory() as tmp:
        code = _CARRY_INSTALL_CODE.replace("__HERE__", repr(str(HERE))).replace("__TMP__", repr(tmp))
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        probe = json.loads(r.stdout.strip().splitlines()[-1])
    assert probe["grafts"] == {"graft_carry": "carry: OK"}
    assert probe["rebound"] == {"analyze": True, "trim": True, "chat": True, "run": False}, probe["rebound"]
    assert probe["sha"] == rw.STOCK_AGENT_TREE_SHA256
    st = probe["status"]["graft_carry"]
    assert st["installed"] and st["enabled"] and st["errors"] == 0
    assert (st["target_fraction"], st["summary_chars_cap"], st["input_chars_cap"], st["compact_max_tokens"], st["compact_thinking"],
            st["min_drop_msgs"], st["window_tokens"]) == (0.5, 4800, 48000, 1500, False, 2, 32768)
    assert st["compactions"] == 0 and st["calls_total"] == 0 and st["per_game"] == {}
    assert probe["env"] == {k: rw.KEITH_CARRY_ENV[k] for k in rw.CARRY_ENV_KEYS}
    fp = probe["fingerprint"]
    assert fp["context_budget_tokens"] == 31744 and fp["max_output_tokens"] is None and fp["yield_seconds"] == 900.0


CARRY_TRANSCRIPT = """
--- analysis_step=1 | action=0 | 10:00:00 | tool-agent ---
[SYSTEM PROMPT]
sys
[USER PROMPT]
No previous sequence has been executed yet.
Current state: step 1, level 1.
[HARNESS CARRY]
[CARRY-CALL] game=g_p0 turn=1 req=1 msgs=2 reasoning_msgs=0 reasoning_chars=0 summary_chars=0 prompt_tokens=4000 completion_tokens=100
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 3
[THINKING]
abc
[TOOL CALL: python]
{"code": "r = action(['UP'])"}
[TOOL RESULT: python]
ok
[ANALYZER STATUS]
step_executed: True
message: Step executed.

--- analysis_step=2 | action=1 | 10:01:00 | tool-agent ---
[USER PROMPT]
Current state: step 2, level 1.
[HARNESS CARRY]
[CARRY-COMPACT] game=g_p0 turn=2 dropped_msgs=6 input_chars=9000 summary_chars=800 prompt_tokens=2600 completion_tokens=210 e2e_s=12.5 ok=1
SUMMARY:
MECHANICS VERIFIED
- UP moves the block.
[HARNESS CARRY]
[CARRY-CALL] game=g_p0 turn=2 req=1 msgs=6 reasoning_msgs=2 reasoning_chars=5000 summary_chars=800 prompt_tokens=33000 completion_tokens=150
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 3
[THINKING]
def
[TOOL CALL: python]
{"code": "print(1)"}
[TOOL RESULT: python]
1
[HARNESS CARRY]
[CARRY-COMPACT] game=g_p0 turn=2 dropped_msgs=2 input_chars=3000 summary_chars=0 prompt_tokens=None completion_tokens=None e2e_s=0.4 ok=0 err=ConnectionError
[HARNESS CARRY]
[CARRY-CALL] game=g_p0 turn=2 req=2 msgs=8 reasoning_msgs=3 reasoning_chars=6000 summary_chars=800 prompt_tokens=30000 completion_tokens=90
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: python]
{"code": "r = action(['DOWN'])"}
[TOOL RESULT: python]
ok
[ANALYZER STATUS]
step_executed: True
message: Step executed.
"""


def test_extractor_reads_carry_markers():
    parsed = rw.parse_transcript(CARRY_TRANSCRIPT)
    turns, calls = parsed["turns"], parsed["calls"]
    assert len(turns) == 2 and len(calls) == 3
    assert [c["reasoning_chars"] for c in calls] == [3, 3, 0]                  # META -> THINKING adjacency survives the sections
    assert [t["call_types"] for t in turns] == ["X", "AX"]
    assert [t["carry_compactions"] for t in turns] == [0, 2]
    cm = parsed["carry_markers"]
    assert [c["prompt_tokens"] for c in cm["calls"]] == [4000, 33000, 30000]
    assert [c["reasoning_msgs"] for c in cm["calls"]] == [0, 2, 3] and [c["summary_chars"] for c in cm["calls"]] == [0, 800, 800]
    assert [c["ok"] for c in cm["compactions"]] == [True, False]
    assert cm["compactions"][0] == {"turn_index": 1, "ok": True, "dropped_msgs": 6, "input_chars": 9000, "summary_chars": 800,
                                    "prompt_tokens": 2600, "completion_tokens": 210, "e2e_s": 12.5}
    assert cm["compactions"][1]["prompt_tokens"] is None and cm["compactions"][1]["e2e_s"] == 0.4
    rd = rw.carry_reads(cm)
    assert (rd["compactions"], rd["compaction_failures"], rd["compaction_attempts"], rd["dropped_msgs"], rd["input_chars"]) == (1, 1, 2, 8, 12000)
    assert rd["summary_chars"]["mean"] == 800 and rd["compaction_e2e_s"]["n"] == 2 and rd["compaction_prompt_tokens"]["n"] == 1
    assert (rd["calls_marked"], rd["reasoning_msgs_total"], rd["reasoning_chars_total"]) == (3, 5, 11000)
    assert rd["calls_with_summary"] == 2 and abs(rd["calls_with_summary_share"] - 2 / 3) < 1e-9 and rd["first_call_with_block"] == 2
    assert rd["prompt_tokens"]["max"] == 33000 and rd["prompt_over_window"] == 1
    assert "graft" not in rd
    rd2 = rw.carry_reads(cm, {"compactions": 1, "compaction_failures": 1, "calls_total": 3, "prompt_tokens_max": 33000, "prompt_over_window": 1})
    assert rd2["graft"]["compactions"] == 1 and rd2["graft"]["prompt_over_window"] == 1 and set(rd2["graft"]) == set(rw._CARRY_GRAFT_KEYS)
    tel = rw.telemetry_for_game(CARRY_TRANSCRIPT, levels_completed=1, number_of_levels=6,
                                carry_counters={"compactions": 1, "compaction_failures": 1, "calls_total": 3})
    assert tel["carry"]["compactions"] == 1 and tel["carry"]["graft"]["calls_total"] == 3
    tel["game_id"], tel["draw"] = "g", 0
    tel2 = dict(tel)
    tel2["draw"] = 1
    agg = rw.aggregate_telemetry({"g_p0": tel, "g_p1": tel2})["carry"]
    assert (agg["compactions_total"], agg["compaction_failures_total"], agg["compaction_attempts"], agg["compactions_per_game"]) == (2, 2, 4, 1.0)
    assert agg["failure_share"] == 0.5 and agg["games_with_compaction"] == 2 and agg["dropped_msgs_per_compaction"] == 4.0
    assert agg["summary_chars_mean"] == 800 and abs(agg["compaction_e2e_s_mean"] - (12.5 + 0.4) / 2) < 1e-9
    assert (agg["calls_marked"], agg["calls_with_summary"]) == (6, 4) and abs(agg["calls_with_summary_share"] - 2 / 3) < 1e-9
    assert agg["prompt_tokens_max"] == 33000 and agg["prompt_over_window"] == 2 and agg["first_call_with_block"]["median"] == 2
    assert abs(agg["reasoning_msgs_per_call"] - 10 / 6) < 1e-9 and agg["graft"]["runs"] == 2 and agg["graft"]["compactions"] == 2
    g = agg["gate"]
    assert g["compactions_per_game"]["ok"] and g["calls_with_summary_share"]["ok"]
    assert not g["failure_share"]["ok"] and not g["prompt_over_window"]["ok"] and g["engaged"] is False
    # a clean run: engaged
    clean = CARRY_TRANSCRIPT.replace("prompt_tokens=33000", "prompt_tokens=23000").replace("prompt_tokens=30000", "prompt_tokens=25000")
    clean = clean.split("[HARNESS CARRY]\n[CARRY-COMPACT] game=g_p0 turn=2 dropped_msgs=2", 1)[0] + "[HARNESS CARRY]\n" + \
        clean.split("[HARNESS CARRY]\n[CARRY-COMPACT] game=g_p0 turn=2 dropped_msgs=2", 1)[1].split("[HARNESS CARRY]\n", 1)[1]
    telc = rw.telemetry_for_game(clean)
    telc["game_id"], telc["draw"] = "g", 0
    aggc = rw.aggregate_telemetry({"g_p0": telc})["carry"]
    assert aggc["compaction_failures_total"] == 0 and aggc["prompt_over_window"] == 0 and aggc["gate"]["engaged"] is True
    # a stock transcript: no markers, nothing engaged, no crash
    agg0 = rw.aggregate_telemetry({"a_p0": {**rw.telemetry_for_game(PROBE_TRANSCRIPT), "game_id": "tu93", "draw": 0}})["carry"]
    assert agg0["compactions_total"] == 0 and agg0["calls_marked"] == 0 and agg0["gate"]["engaged"] is False
    # a marker mentioned in prose is not a marker
    assert rw.parse_transcript("hello [CARRY-COMPACT] game=x turn=1 dropped_msgs=1 input_chars=1 summary_chars=1 prompt_tokens=1 "
                               "completion_tokens=1 e2e_s=1 ok=1\n")["carry_markers"]["compactions"] == []


def test_dry_run_keith_carry_arm_end_to_end():
    """The pre-registered launch shape on the loopback mock (+ the real engine) with the window shrunk to
    9,000 tokens so the trimmer must evict inside 14 calls (and a 120 s game cap: the graft skips compaction
    in the last 30 s of a game, so the dry-run default of 25 s/game would skip every one): compactions fire in every run, the mock answers
    them with a summary block, the block rides the system message of later requests, the window is never
    exceeded, and the CARRY / CARRY-WINDOW / CARRY-PRIMARY / CARRY-SAFETY summary lines carry the reads."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_carry",
                            "--games", "tu93,ft09,cd82", "--concurrency", "3", "--max-calls", "14",
                            "--per-game-s", "120", "--wave-cap-s", "200", "--progress-every", "60",
                            "--knob", "LOCAL_ANALYZER_CONTEXT_WINDOW=9000", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        runs = list(Path(tmp).glob("*-regime-keith_carry-dry"))
        assert len(runs) == 1, runs
        out = runs[0]
        res = json.loads((out / "results.json").read_text())
        tel = json.loads((out / "telemetry.json").read_text())
        assert res["status"] == "done" and res["arm"] == "keith_carry" and res["knob_overrides"] == {"LOCAL_ANALYZER_CONTEXT_WINDOW": "9000"}
        assert res["stock"]["agent_tree_sha256"] == rw.STOCK_AGENT_TREE_SHA256
        assert res["grafts"]["installed"] == {"graft_carry": "carry: OK"}
        assert res["analyzer_env"]["LOCAL_ANALYZER_YIELD_SECONDS"] == "900" and res["analyzer_env"]["CARRY_ENABLE"] == "1"
        cfg = tel["aggregate"]["analyzer_status_config_first"]
        assert cfg == {"max_output_tokens": "server default", "context_budget_tokens": "7976", "yield_seconds": "900.0"}, cfg
        st = res["grafts"]["status"]["graft_carry"]
        ca = tel["aggregate"]["carry"]
        assert st["errors"] == 0 and st["compaction_failures"] == 0
        assert ca["compactions_total"] == st["compactions"] == res["mock_state"]["compactions"] >= 3   # markers == graft == mock
        assert ca["games_with_compaction"] == 3 and ca["compaction_failures_total"] == 0
        assert ca["prompt_over_window"] == 0 and st["prompt_over_window"] == 0
        assert ca["calls_marked"] == st["calls_total"] == tel["aggregate"]["calls_total"]                # one CARRY-CALL per model call
        assert ca["graft"]["compactions"] == st["compactions"] and ca["graft"]["runs"] == 3
        assert ca["summary_chars_mean"] and ca["summary_chars_mean"] <= 4800 and ca["dropped_msgs_per_compaction"] >= 2
        assert ca["reasoning_msgs_per_call"] > 0                                                        # the stock carries reasoning
        gate = ca["gate"]
        assert gate["compactions_per_game"]["ok"] and gate["failure_share"]["ok"] and gate["prompt_over_window"]["ok"]
        assert gate["calls_with_summary_share"]["ok"] and gate["engaged"] is True, gate
        per = tel["per_game"]
        assert set(per) == {"tu93-0768757b_p0", "ft09-0d8bbf25_p0", "cd82-fb555c5d_p0"}
        for stem, g in per.items():
            text = (out / "transcripts" / f"{stem}.txt").read_text()
            c = g["carry"]
            assert c["compactions"] >= 1 and c["compactions"] == st["per_game"][stem]["compactions"]
            assert text.count("[HARNESS CARRY]\n[CARRY-COMPACT] game=" + stem) == c["compactions"]
            assert text.count("SUMMARY:\nMECHANICS VERIFIED\n- " + rw.MOCK_COMPACT_SUMMARY) == c["compactions"]
            assert text.count("[HARNESS CARRY]\n[CARRY-CALL] game=" + stem) == c["calls_marked"] == g["calls"]
            assert c["prompt_tokens"]["max"] <= rw.CARRY_WINDOW_TOKENS and c["calls_with_summary"] >= 1
            # the block reaches the model: the latest prompt-log snapshot renders it inside the system message
            log = (out / "prompts" / f"{stem}.log").read_text()
            assert "# Compacted knowledge from your earlier turns" in log and rw.MOCK_COMPACT_SUMMARY in log
        summary = (out / "summary.txt").read_text()
        assert "REGIME WAVE  arm=keith_carry" in summary
        for line in ("CARRY  compactions", "ENGAGED = YES {compactions_per_game+, failure_share+, prompt_over_window+, calls_with_summary_share+}",
                     "CARRY-WINDOW prompt tok/call", "over 32768: 0 (gate 0)", "reasoning carried per request:",
                     "CARRY-PRIMARY levels", "vs pooled six-draw base 39.33 (sd 2.34)", ">= 48 step candidate, 45-47 redraw, <= 44 dead",
                     "CARRY-SAFETY GAME_OVERs", "fit-the-clock: calls/game", "KNOB OVERRIDES (not the pinned arm env)"):
            assert line in summary, line
        assert "'CARRY_TARGET_FRACTION': '0.5'" in summary and "yield 900 s" in summary
        assert summary.count("\n") < 45
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path
    # the stock yield900 arm under the same knob: the trimmer EVICTS (no compaction request, no CARRY lines)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_yield900",
                            "--games", "tu93", "--concurrency", "1", "--max-calls", "10", "--per-game-s", "20",
                            "--wave-cap-s", "90", "--progress-every", "60", "--knob", "LOCAL_ANALYZER_CONTEXT_WINDOW=9000", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        out = list(Path(tmp).glob("*-regime-keith_yield900-dry"))[0]
        res = json.loads((out / "results.json").read_text())
        assert res["mock_state"]["compactions"] == 0 and res["grafts"]["installed"] == {}
        text = (out / "transcripts" / "tu93-0768757b_p0.txt").read_text()
        assert "[HARNESS CARRY]" not in text
        summary = (out / "summary.txt").read_text()
        assert "CARRY  compactions" not in summary.split("\n  run ")[0]


WS_TRANSCRIPT = """
--- analysis_step=1 | action=0 | 10:00:00 | tool-agent ---
[USER PROMPT]
Current state: step 1, level 1.
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 3
[THINKING]
abc
[TOOL CALL: backtest]
{"code": "..."}
[HARNESS WS]
[WS-BACKTEST] game=dc22-x_p0 level=2 matched=14/20 green=0 code_chars=3100 ms=420
[TOOL RESULT: backtest]
{"matched": 14}
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
content_chars: 0
reasoning_chars: 0
[TOOL CALL: backtest]
{"code": "..."}
[HARNESS WS]
[WS-BACKTEST] game=dc22-x_p0 level=2 matched=20/20 green=1 code_chars=4200 ms=515
[HARNESS WS]
[WS-SAVE] game=dc22-x_p0 name=model.py chars=4200 files=1
[TOOL RESULT: backtest]
{"green": true}
[ANALYZER STATUS]
step_executed: True
message: Step executed.
"""


def test_extractor_reads_ws_markers_and_conversion():
    parsed = rw.parse_transcript(WS_TRANSCRIPT)
    m = parsed["ws_markers"]
    assert [b["matched"] for b in m["backtests"]] == [14, 20]
    assert [b["green"] for b in m["backtests"]] == [False, True]
    assert m["backtests"][1] == {"turn_index": 0, "level": 2, "matched": 20, "total": 20, "green": True,
                                 "code_chars": 4200, "ms": 515}
    assert m["saves"] == [{"turn_index": 0, "name": "model.py", "chars": 4200, "files": 1}]
    assert [c["reasoning_chars"] for c in parsed["calls"]] == [3, 0]      # META->THINKING adjacency survives
    # the conversion read: green on level 2, and the run did reach level 2 => converted
    r = rw.ws_reads(m, levels_completed=2)
    assert (r["backtests"], r["backtests_green"], r["saves"]) == (2, 1, 1)
    assert r["green_levels"] == [2] and r["cleared_after_green"] == [2] and r["conversion"] == 1.0
    assert r["best_by_level"]["2"] == {"matched": 20, "total": 20, "green": True}
    assert r["best_match_share"] == 1.0 and r["backtest_ms_total"] == 935
    # a run that verified a model and still never cleared that level => conversion 0
    r0 = rw.ws_reads(m, levels_completed=1)
    assert r0["green_levels"] == [2] and r0["cleared_after_green"] == [] and r0["conversion"] == 0.0
    # pooled + gate
    tel = rw.telemetry_for_game(WS_TRANSCRIPT, levels_completed=2, number_of_levels=6)
    tel["game_id"], tel["draw"] = "dc22", 0
    tel2 = dict(tel); tel2["draw"] = 1
    agg = rw.aggregate_telemetry({"dc22_p0": tel, "dc22_p1": tel2})["ws"]
    assert (agg["backtests_total"], agg["backtests_green_total"], agg["backtests_per_game"]) == (4, 2, 2.0)
    assert agg["runs_with_backtest_share"] == 1.0 and agg["green_share"] == 0.5
    assert agg["green_levels_total"] == 2 and agg["cleared_after_green_total"] == 2 and agg["conversion"] == 1.0
    assert agg["gate"]["engaged"] is True
    # a stock transcript: nothing engaged, no crash
    agg0 = rw.aggregate_telemetry({"a_p0": {**rw.telemetry_for_game(PROBE_TRANSCRIPT), "game_id": "t", "draw": 0}})["ws"]
    assert agg0["backtests_total"] == 0 and agg0["gate"]["engaged"] is False and agg0["conversion"] is None
    # prose mentioning a marker is not a marker
    assert rw.parse_transcript("see [WS-BACKTEST] game=x level=1 matched=1/1 green=1 code_chars=1 ms=1\n")["ws_markers"]["backtests"] == []


def test_dry_run_keith_ws_arm_end_to_end():
    """The mock answers with real `backtest` and `workspace` tool calls whenever the arm offers them, so the
    whole path is exercised: tools advertised -> dispatched host-side -> markers in the transcript -> telemetry."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_ws",
                            "--games", "tu93,ft09,cd82", "--concurrency", "3", "--max-calls", "16",
                            "--per-game-s", "60", "--wave-cap-s", "150", "--progress-every", "60", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, (r.returncode, r.stderr[-3000:], r.stdout[-3000:])
        out = list(Path(tmp).glob("*-regime-keith_ws-dry"))[0]
        res = json.loads((out / "results.json").read_text())
        tel = json.loads((out / "telemetry.json").read_text())
        assert res["grafts"]["installed"] == {"graft_workspace": "workspace: OK"}
        assert res["analyzer_env"]["WS_ENABLE"] == "1" and res["analyzer_env"]["LOCAL_ANALYZER_YIELD_SECONDS"] == "900"
        st = res["grafts"]["status"]["graft_workspace"]
        wsx = tel["aggregate"]["ws"]
        assert st["errors"] == 0 and st["backtests"] > 0 and st["saves"] > 0
        assert st["transitions_logged"] > 0                       # the log captured real executed actions
        assert wsx["backtests_total"] == st["backtests"] == res["mock_state"]["ws_backtests"]
        assert wsx["saves_total"] == st["saves"] == res["mock_state"]["ws_saves"]
        assert wsx["runs_with_backtest"] == 3 and wsx["gate"]["engaged"] is True
        # per-run counters are keyed by RUN STEM on both sides, else they orphan
        assert set(st["per_game"]) == set(tel["per_game"]) == {"tu93-0768757b_p0", "ft09-0d8bbf25_p0", "cd82-fb555c5d_p0"}
        for stem, g in tel["per_game"].items():
            text = (out / "transcripts" / f"{stem}.txt").read_text()
            assert text.count("[HARNESS WS]\n[WS-BACKTEST] game=" + stem) == g["ws"]["backtests"]
            assert g["ws"]["graft"] is not None
        summary = (out / "summary.txt").read_text()
        for line in ("WS     backtests", "ENGAGED = YES {backtests_per_game+, runs_with_backtest_share+}",
                     "WS-CONVERT green world models on", "WS-PRIMARY levels", "WS-SAFETY GAME_OVERs",
                     "+ verifier"):
            assert line in summary, line
        assert "'WS_ENABLE': '1'" in summary and summary.count("\n") < 50
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".txt", ".prom", ".jsonl", ".log"):
                assert "dry-run-token" not in path.read_text(encoding="utf-8", errors="replace"), path
    # a stock arm never advertises the tools, so the mock never calls them and no WS lines appear
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PYTHON, str(HERE / "run_regime_wave.py"), "--dry-run", "--arm", "keith_yield900",
                            "--games", "tu93", "--concurrency", "1", "--max-calls", "6", "--per-game-s", "20",
                            "--wave-cap-s", "60", "--progress-every", "60", "--out", tmp],
                           capture_output=True, text=True, cwd=str(REPO), timeout=400)
        assert r.returncode == 0, r.stderr[-2000:]
        out = list(Path(tmp).glob("*-regime-keith_yield900-dry"))[0]
        res = json.loads((out / "results.json").read_text())
        assert res["mock_state"]["ws_backtests"] == 0 and res["grafts"]["installed"] == {}
        assert "WS     backtests" not in (out / "summary.txt").read_text().split("\n  run ")[0]


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
