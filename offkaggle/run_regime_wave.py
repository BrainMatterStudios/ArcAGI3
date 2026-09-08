#!/usr/bin/env python3
"""Off-Kaggle REGIME wave: the UNMODIFIED June-stock duck harness against a
remote vLLM serving Qwen3.8-Flash-Next, 25 public games at the live geometry,
with per-call TELEMETRY so two analyzer configurations can be compared by
mechanism (calls, reasoning length, truncation, yields, latency) rather than
by a 25-game score that varies 1.28-1.80 between identical-bytes draws.

What runs (mirrors the public keithtyser V14 notebook cell by cell):
  * agent bytes  = scratchpad/bundles/june_stock/src/ARC3-Inference (tree sha
                   pinned; byte-identical to keith's dataset copy)
  * framework    = scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework
  * solver pkls  = scratchpad/taaf_scored_ref/{benchmark_initial,deploy_target}.pkl
                   (sha-pinned; byte-identical to keith's dataset copy)
  * notebook cell 3 process env, cell 13 settings (7920 s/game, analyzer
    timeout 900, concurrency 28, max_actions None, request logs off), cell 15
    offline game list (the 25 public ids in that order, OFFLINE arcade over
    environment_files), bm.run(minimal_diagnostics=True) + bm._save_json() +
    the frozen scorer.
  * ONE knob differs between the arms — the analyzer env:
      keith  : LOCAL_ANALYZER_CONTEXT_WINDOW=32768, LOCAL_ANALYZER_MAX_OUTPUT=0
      flight : LOCAL_ANALYZER_CONTEXT_WINDOW=24576, LOCAL_ANALYZER_MAX_OUTPUT=4096
      keith_yield180 : keith + LOCAL_ANALYZER_YIELD_SECONDS=180 (original single-knob arm)
      keith_retry    : keith + graft_retry (submission/_throughput_v1/graft_retry.py,
                       the fresh-mind level retry) installed IN MEMORY at wave
                       start, flags RETRY_ENABLE=1 K=3 ABS=200 COOLDOWN=150 MAX=2
      keith_evid     : keith + graft_evidence (evidence-integrity aid: object diff +
                       per-action trace + LEVEL CLEARED flag appended to every
                       executed-action tool result), flags EVID_ENABLE=1
                       EVID_MAX_ENTRIES=40 EVID_MAX_CHARS=1500 EVID_TRACE=1
      keith_hypo     : keith + graft_hypo (hypothesis-enumeration + probe rule
                       appended to every analyzer prompt), flag HYPO_ENABLE=1
      keith_up8      : keith + MULTIMODAL_UPSCALE=8 (the current-grid PNG 512 px
                       instead of 256 px: 256 vision tokens per image instead of 64)
      keith_probe    : keith_yield900 + graft_probe (harness-enforced probe
                       discipline: after PROBE_MAX_ANALYSIS=2 analysis-only python
                       calls in a span the next analysis-only snippet is refused
                       with a "run a <=5-action test" tool result, at most
                       PROBE_MAX_REFUSALS=4 refusals per span — a span carries
                       across no-action turns on the same level; one-line notice
                       on the turn after a no-action turn), flags PROBE_ENABLE=1
                       PROBE_MAX_ANALYSIS=2 PROBE_MAX_PROBE=5 PROBE_MAX_REFUSALS=4
                       PROBE_NOTE_LINES=3
      keith_carry    : keith_yield900 + graft_carry (Track A1, 09-08: compaction
                       instead of eviction — when the trimmer must drop history it
                       drops a chunk down to CARRY_TARGET_FRACTION of the budget and
                       asks the model, in one extra no-tools call, to fold the
                       dropped turns into a compacted-knowledge block kept in the
                       system message; prior-turn reasoning is already carried by
                       the stock and is MEASURED per call), flags CARRY_ENABLE=1
                       CARRY_TARGET_FRACTION=0.5 CARRY_SUMMARY_CHARS=4800
                       CARRY_INPUT_CHARS=48000 CARRY_COMPACT_MAX_TOKENS=1500
                       CARRY_COMPACT_THINKING=0 CARRY_MIN_DROP_MSGS=2
  --draws N plays the selected games N times as independent runs in one wave
  (taaf Benchmark.n_passes; run stems <gid>_p0, <gid>_p1, ...); --per-game-s
  caps each run (default 7920 = public geometry).
    (everything else in the analyzer env is identical: sampling 0.6/0.95/20,
    thinking on, 60 s yield, tool steps unlimited, multimodal current_grid x4).

Nothing in the agent's BYTES is edited (the stock tree sha is asserted for every
arm); a graft arm rebinds ToolAgent methods in memory after the import. Two
hooks live OUTSIDE the agent for every arm:
  1. `HarnessSolver.analyzer_factory` (a documented solver field) builds the
     stock ToolAgent with exactly `_make_analyzer`'s arguments and tags the
     game thread so requests can be attributed to a game.
  2. a `requests.post` client shim (pure observer) records per-call wall time,
     status, redirect legs, usage tokens and finish reason to
     <out>/requests_shim.jsonl. It changes no request parameter.

Run (arm = keith | flight):
    .venv/bin/python offkaggle/run_regime_wave.py --arm keith \
        --base-url https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1 \
        --games all --out offkaggle/results
    (token: --token-file, default ~/.config/arc3/vllm_token, or $ARC3_VLLM_TOKEN;
     it is read into memory only and never written or printed)

Dry run (no network; a loopback mock vLLM, ~1-2 min):
    .venv/bin/python offkaggle/run_regime_wave.py --dry-run --arm keith

Outputs under <out>/<ts>-regime-<arm>/:
    results.json, telemetry.json, summary.txt, arm_env.json,
    metrics_before.prom / metrics_after.prom, requests_shim.jsonl,
    benchmark.json + score.json (the harness's own), transcripts/, prompts/,
    artifacts/ (the harness's own per-game files).
Ctrl-C once = graceful stop (games marked cancelled, partial results written);
twice = hard exit.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pickle
import re
import signal
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
STOCK_AGENT_DIR = REPO / "scratchpad/bundles/june_stock/src/ARC3-Inference"
FRAMEWORK_DIR = REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework/src"
FRAMEWORK_TREE_DIR = REPO / "scratchpad/taaf_scored_ref/src/tufa-arc-agi-framework"
TOOLKIT_DIR = REPO / "reference/arc-agi-toolkit"
BUNDLE_PKL_DIR = REPO / "scratchpad/taaf_scored_ref"
ENV_FILES_DIR = REPO / "environment_files"
DEFAULT_OUT = REPO / "offkaggle/results"
TOKEN_FILE = Path.home() / ".config/arc3/vllm_token"

SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"
DEFAULT_BASE_URL = "https://a-m-mobasher--arc3-flashnext-serve.modal.run/v1"
# The serving profile every scored arm must run on (modal_flashnext_serve.PUBLIC25_VLLM_PROFILE_NAME)
# and the GPU it must run on. Asserted against GET /arc3/identity in preflight — the two
# 09-03 runs on this endpoint ran on the kv10 override; that must never again be silent.
DEFAULT_EXPECT_PROFILE = "kv5-bf16-mtp3-c8-cg32"
EXPECT_GPU_SUBSTRING = "RTX PRO 6000"

# Pins (computed 2026-09-02; test_run_regime_wave.py re-derives them from the
# trees and, when the session scratchpad is present, from keith's dataset copy).
STOCK_AGENT_TREE_SHA256 = "74ab691052406c22c46cf8420ee96d59a877a4277d2b9e357de7f7412e6754ae"
FRAMEWORK_TREE_SHA256 = "f68b6850b242010d3af0d2059e7cbd3c375700488316b59bf5f5e2ae27ada43e"
BENCHMARK_PKL_SHA256 = "7f619ac0831bfc4d138365681fc46de2f7107b54e2d62bbb3194b65d9aec1d67"
DEPLOY_TARGET_PKL_SHA256 = "f0dc4b59f390862326b0e40a6ff0f8910d9e6487ae244efb1ae9ab495f79ba19"

# keith V14 notebook cell 13 — "Exact public-25 and competition settings."
GEOMETRY = {
    "max_runtime_s_per_game": 7920.0,
    "analyzer_timeout": 900.0,
    "concurrency": 28,
    "max_actions_per_game": None,
    "save_request_logs": False,
}
WAVE_CAP_S = 2.5 * 3600.0          # whole-wave hard cap (soft deadline in bm.run)
DRY_RUN_PER_GAME_S = 25.0
DRY_RUN_WAVE_CAP_S = 100.0

# keith V14 notebook cell 15 — PUBLIC_GAME_IDS, same order.
PUBLIC_GAME_IDS = (
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940",
    "wa30-ee6fef47", "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b",
    "sp80-589a99af", "m0r0-492f87ba", "vc33-5430563c", "ar25-0c556536",
    "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
    "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab",
    "tr87-cd924810",
)

# keith V14 notebook cell 3 — harness-side process env (TRUE_SUBMISSION=False).
PROCESS_ENV = {
    "MPLBACKEND": "Agg",
    "TAAF_RUN_AS_SUBMISSION": "0",
    "TAAF_MINIMAL_DIAGNOSTICS": "1",
    "ONLY_RESET_LEVELS": "true",
}

# The public notebook's analyzer env — keith's persisted taaf_setup_env.json /
# KEITH_REGIME.md §5 (base URL + api key are runtime; see RUNTIME_ENV_KEYS).
KEITH_ANALYZER_ENV = {
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}
# Our Flash-Next flight (submission/_flashnext_flight/arc3-flashnext-flight.ipynb
# cell 9, the >=32768-ctx branch): identical except the two window keys.
FLIGHT_ANALYZER_ENV = {
    **KEITH_ANALYZER_ENV,
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "24576",
    "LOCAL_ANALYZER_MAX_OUTPUT": "4096",
}
# 09-03 original single-knob arm on the keith base: the 60 s turn yield (which cuts ~43% of the
# base's turns) raised to 180 s; everything else identical to `keith`.
KEITH_YIELD180_ENV = {**KEITH_ANALYZER_ENV, "LOCAL_ANALYZER_YIELD_SECONDS": "180"}
KEITH_YIELD900_ENV = {**KEITH_ANALYZER_ENV, "LOCAL_ANALYZER_YIELD_SECONDS": "900"}  # 09-06: within-turn multi-step at the live cadence
# 09-03 fresh-mind level retry arm: the keith base + graft_retry installed in memory at
# wave start (ARM_GRAFTS). The RETRY_* keys are the graft's own flags, read at call time:
# a level RESET + "FRESH MIND" prompt block once a level's action bucket reaches
# K x its human baseline (ABS actions when the engine hides baselines), at most MAX
# per level, COOLDOWN actions apart. Everything else identical to `keith`.
RETRY_ENV_KEYS = ("RETRY_ENABLE", "RETRY_K", "RETRY_ABS", "RETRY_COOLDOWN", "RETRY_MAX")
KEITH_RETRY_ENV = {**KEITH_ANALYZER_ENV, "RETRY_ENABLE": "1", "RETRY_K": "3", "RETRY_ABS": "200",
                   "RETRY_COOLDOWN": "150", "RETRY_MAX": "2"}
# 09-06 judge program item 1 (docs/research-2026-09-06/J-judge-0906.md): three single-lever arms
# on the keith base for the 3-wall turn-capped instrument (cd82/dc22/lf52, --draws 2, --per-game-s 1500).
#  (a) evidence-integrity aid — graft_evidence, flags read at call time
EVID_ENV_KEYS = ("EVID_ENABLE", "EVID_MAX_ENTRIES", "EVID_MAX_CHARS", "EVID_TRACE")
KEITH_EVID_ENV = {**KEITH_ANALYZER_ENV, "EVID_ENABLE": "1", "EVID_MAX_ENTRIES": "40", "EVID_MAX_CHARS": "1500",
                  "EVID_TRACE": "1"}
#  (b) hypothesis-enumeration + probe rule — graft_hypo
HYPO_ENV_KEYS = ("HYPO_ENABLE",)
KEITH_HYPO_ENV = {**KEITH_ANALYZER_ENV, "HYPO_ENABLE": "1"}
#  (c) MULTIMODAL_UPSCALE 8 — no graft; vision_context.current_grid_image_upscale() reads the key at
#      call time (64x64 grid -> 512x512 px PNG instead of 256x256; 16-px patches x 2 merge = 32 px per
#      vision token -> 256 tokens/image instead of 64)
KEITH_UP8_ENV = {**KEITH_ANALYZER_ENV, "MULTIMODAL_UPSCALE": "8"}
# 09-08 loss-ledger-3 #1: harness-enforced probe discipline on the 900 s regime — graft_probe
# (submission/_throughput_v1/graft_probe.py) installed in memory; flags read at call time. Differs
# from keith_yield900 by exactly the PROBE_* keys. Pre-registration: offkaggle/REGIME_WAVE_STATUS.md.
PROBE_ENV_KEYS = ("PROBE_ENABLE", "PROBE_MAX_ANALYSIS", "PROBE_MAX_PROBE", "PROBE_MAX_REFUSALS", "PROBE_NOTE_LINES")
KEITH_PROBE_ENV = {**KEITH_YIELD900_ENV, "PROBE_ENABLE": "1", "PROBE_MAX_ANALYSIS": "2", "PROBE_MAX_PROBE": "5",
                   "PROBE_MAX_REFUSALS": "4", "PROBE_NOTE_LINES": "3"}
# 09-08 Track A1 (docs/PLAN-2026-09-08-revised-plan-to-7plus.md §6): compaction instead of eviction — graft_carry
# (submission/_throughput_v1/graft_carry.py) on the yield900 base; flags read at call time. Differs from
# keith_yield900 by exactly the CARRY_* keys. Pre-registration: offkaggle/REGIME_WAVE_STATUS.md.
CARRY_ENV_KEYS = ("CARRY_ENABLE", "CARRY_TARGET_FRACTION", "CARRY_SUMMARY_CHARS", "CARRY_INPUT_CHARS",
                  "CARRY_COMPACT_MAX_TOKENS", "CARRY_COMPACT_THINKING", "CARRY_MIN_DROP_MSGS")
KEITH_CARRY_ENV = {**KEITH_YIELD900_ENV, "CARRY_ENABLE": "1", "CARRY_TARGET_FRACTION": "0.5", "CARRY_SUMMARY_CHARS": "4800",
                   "CARRY_INPUT_CHARS": "48000", "CARRY_COMPACT_MAX_TOKENS": "1500", "CARRY_COMPACT_THINKING": "0",
                   "CARRY_MIN_DROP_MSGS": "2"}
ARM_ENV = {"keith": KEITH_ANALYZER_ENV, "flight": FLIGHT_ANALYZER_ENV, "keith_yield180": KEITH_YIELD180_ENV, "keith_yield900": KEITH_YIELD900_ENV,
           "keith_retry": KEITH_RETRY_ENV, "keith_evid": KEITH_EVID_ENV, "keith_hypo": KEITH_HYPO_ENV,
           "keith_up8": KEITH_UP8_ENV, "keith_probe": KEITH_PROBE_ENV, "keith_carry": KEITH_CARRY_ENV}
ARMS = tuple(ARM_ENV)
# grafts (submission/_throughput_v1/<name>.py, install() -> "<name>: OK") an arm installs in memory
ARM_GRAFTS = {"keith_retry": ("graft_retry",), "keith_evid": ("graft_evidence",), "keith_hypo": ("graft_hypo",),
              "keith_probe": ("graft_probe",), "keith_carry": ("graft_carry",)}
GRAFT_ENV_PREFIXES = ("RETRY_", "EVID_", "HYPO_", "PROBE_", "CARRY_")     # every graft flag; scrubbed from the shell for every arm
GRAFT_FLAG_KEYS = RETRY_ENV_KEYS + EVID_ENV_KEYS + HYPO_ENV_KEYS + PROBE_ENV_KEYS + CARRY_ENV_KEYS
# loss-ledger-3 reference reads for the PROBE gate (docs/research-2026-09-08/R-loss-ledger-3.md, yield900 regime)
LEDGER3_REFERENCE = {"turns_ge3_analysis_share": 0.15, "wall_actions_ratio_median": 0.72, "yields_per_draw": "27-30",
                     "analysis_call_share": 0.49,
                     # PRIMARY comparator: the six 25-game draws of the two regimes pooled (Y1 41, Y2 40, YK 37, M1 36,
                     # M4 40, K 42; loss-ledger-3 NOTES Q1) — levels are a null between the regimes
                     "base_levels_six_draws": [41, 40, 37, 36, 40, 42], "base_levels_mean": 39.33, "base_levels_sd": 2.34,
                     # SAFETY comparators, yield900 regime (the arm's base): GAME_OVERs/run and live-cap score/game
                     "game_overs_per_run": 0.87, "live_cap_score_per_game": 8.42}
# the 12 games whose modal wall was never passed in any of the six draws (loss-ledger-3 NOTES Q1/q7): the co-primary
# read is how many of these walls the arm passes (levels_completed >= wall level in any draw)
NEVER6_WALLS = {"bp35": 2, "dc22": 2, "g50t": 2, "lf52": 2, "lp85": 6, "ls20": 2, "r11l": 3, "sb26": 2, "sp80": 2,
                "tn36": 3, "vc33": 4, "wa30": 2}
# pre-registered ENGAGEMENT gate (judge 09-08): refusals >= 1/game AND the very next call after the FIRST refusal
# of a span acted >= 50 % (turn-ending refusals count as non-acting) AND wall actions/baseline median >= 0.9;
# secondary: turns with >= 3 executed analysis-only calls (leak split) and turn_time_budget yields per draw
PROBE_GATE = {"refusals_per_game_min": 1.0, "acted_after_first_refusal_min": 0.5, "wall_actions_ratio_min": 0.9,
              "turns_ge3_analysis_max": 0.05, "yields_per_draw_max": 20}
# pre-registered ENGAGEMENT gate for keith_carry (09-08): compactions >= 1 per game AND compaction failures <= 10 % of
# attempts AND no request over the 32,768-token window AND >= 40 % of model calls carry the compacted block
CARRY_GATE = {"compactions_per_game_min": 1.0, "failure_share_max": 0.10, "prompt_over_window_max": 0,
              "calls_with_summary_share_min": 0.40}
CARRY_WINDOW_TOKENS = 32768                          # vLLM --max-model-len (KEITH_REGIME.md); usage.prompt_tokens must stay under
CARRY_COMPACT_HEAD = "You are the same agent that played the turns below"   # == graft_carry.COMPACT_SYSTEM_HEAD (asserted in tests)
MOCK_COMPACT_SUMMARY = "MOCK-COMPACT-SUMMARY"
LIVE_CAP = 1.15


def live_cap_score(baselines: list | None, actions_per_level: list | None, levels: int, cap: float = LIVE_CAP) -> float | None:
    """loss-ledger-3 common.py score(): 100 * sum_{l<levels} (l+1) * min(cap, (b/a)^2) / sum(1..n) — the live
    formula recomputed from the per-level action buckets; None when baselines are unavailable (submission mode)."""
    if not baselines or actions_per_level is None:
        return None
    n = len(baselines)
    if n == 0:
        return None
    total = sum(range(1, n + 1))
    s = 0.0
    for lv in range(min(int(levels or 0), n, len(actions_per_level))):
        a, b = actions_per_level[lv], baselines[lv]
        s += (lv + 1) * min(cap, (b / a) ** 2 if a else 0.0)
    return 100.0 * s / total


def game_overs_from_events(path: Path) -> int | None:
    """GAME_OVERs of a run = action rows with game_over true in the harness's <stem>_events.jsonl (loss-ledger-3 q2)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    n = 0
    for line in text.splitlines():
        if '"game_over"' not in line:      # the harness writes compact JSON ("game_over":true) — never pre-filter on spacing
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") == "action" and rec.get("game_over") is True:
            n += 1
    return n
GRAFT_DIR = REPO / "submission/_throughput_v1"
RUNTIME_ENV_KEYS = ("LOCAL_ANALYZER_BASE_URL", "OPENAI_BASE_URL", "LOCAL_ANALYZER_API_KEY")

# The judge's reference read of keith's V14 commit run (docs/research-2026-09-02/J-judge.md F4).
KEITH_COMMIT_REFERENCE = {
    "calls_per_game": 55, "reasoning_chars_mean": 3406, "reasoning_chars_median": 2206,
    "turns_per_game": 53, "vllm_e2e_mean_s": 142, "vllm_queue_mean_s": 124,
    "mtp_acceptance": 0.60, "preemptions": 57, "levels_per_game": 1.44, "score": 6.76,
}

TELEMETRY_DEFINITIONS = {
    "calls": "HTTP chat completions the harness made for the game = count of "
             "'[MODEL RESPONSE META]' transcript sections (one per successful request).",
    "turns": "analyze() invocations = count of '--- analysis_step=N | action=M | ... ---' "
             "headers. This is the judge's 'analysis steps per game' (53 for keith).",
    "distinct_analysis_steps": "max analysis_step number (a step repeats its header after a "
                               "60 s yield or a retryable failure).",
    "reasoning_chars": "length of the '[THINKING]' section that follows each META (stripped); "
                       "0 when the response carried no reasoning. Pooled over calls "
                       "(the judge's mean 3,406 / median 2,206).",
    "no_tool_call_share": "calls whose META says tool_call_count: 0 (text-only replies that "
                          "trigger the 'You have not acted yet' follow-up) / calls.",
    "length_finish_share": "calls with finish_reason: length (output truncated by max_tokens "
                           "or the context) / calls.",
    "turn_outcomes": "per-turn '[ANALYZER STATUS] message:' — Step executed / Yielded control "
                     "(turn_time_budget|stop_requested) / No action(...) captured / request_error / error.",
    "actions_per_call": "len(game_run.history) / calls.",
    "e2e_s (client)": "requests.post wall time incl. redirect legs, from the client shim.",
    "e2e_s (vllm)": "vllm:e2e_request_latency_seconds_sum/count delta from /metrics.",
    "retries_fired": "graft_retry level retries = count of '[RETRY] game=..' marker lines the graft "
                     "wrote into the transcript (one per harness-issued level RESET + FRESH MIND turn).",
    "retry_clears": "count of '[RETRY-CLEAR] game=..' marker lines = retried levels that later cleared.",
    "evid_markers": "graft_evidence blocks = '[EVID] harness object diff ...' lines inside [TOOL RESULT: python] "
                    "sections (one per tool call that executed actions).",
    "evid_level_flags": "'LEVEL CLEARED after action k ...' lines the aid wrote (a level clear inside a batch).",
    "hypo_markers": "graft_hypo blocks = '[HYPO] Hypothesis discipline ...' lines inside [USER PROMPT] sections.",
    "turn_levels": "per turn, the level in the first [USER PROMPT] 'Current state: step N, level L' line.",
    "level_reached": "max turn level = the level the run ended on (levels_completed + 1 unless won).",
    "wall_level": "the uncleared level the run ended on (None when the game was won).",
    "engagement": "share of turns carrying the aid: evid = turns with >= 1 [EVID] / turns that executed a step; "
                  "hypo = turns whose prompt carries [HYPO] / turns; '_wall' restricts both to turns on wall_level "
                  "(the judge's engagement gate: >= 80 % of wall turns).",
    "wall": "judge 09-06 instrument reads per run: void (request errors > 0, vLLM preemptions delta > 0 for the wave, "
            "or length-finish share > 1 %), calls_at_l2 (calls made before the first level-2 turn), attempt (L2 reached "
            "with >= 30 calls left of the budget = --max-calls or the run's own total), passed (level 3 reached), "
            "uptake / uptake_wall (share of turns whose THINKING or ASSISTANT text quotes the aid: '[EVID]', "
            "'TRACE per action', '[HYPO]', or MOVED/APPEARED cited with a coordinate).",
    "first_call_prompt_tokens": "prompt_tokens of the run's first request (n_messages == 2); keith baseline ~4,050; "
                                "the up8 arm expects ≈ +192 (256 - 64 vision tokens).",
    "max_calls": "--max-calls N: in-memory stop after N analyzer calls per run (results.json:max_calls_stops).",
    "draws": "--draws N = taaf n_passes: each game played N times as independent runs (<gid>_p<draw>); "
             "per_game is keyed by run stem and carries game_id + draw.",
    "call_types": "loss-ledger-3 call classes from the transcript (scratchpad loss-ledger-3/q3.py ctype): N = no tool "
                  "call; R = refused by graft_probe ([PROBE-REFUSE] marker inside the [TOOL CALL] section); E = python "
                  "error (Traceback|Error: in the result) without action( in the code; X = code contains action( ; "
                  "A = analysis-only (tool call, no action( in the code, no error). The ledger reads the CODE, not the "
                  "payload: an action() that failed is X here but analysis-only for the graft.",
    "probe": "graft_probe reads per run: refusals / noact_turns = '[PROBE-REFUSE]' / '[PROBE-NOACT]' marker lines; "
             "turns_ge3_analysis = turns with >= 3 A-class calls before the first X (the ledger's 15 %; refused calls "
             "excluded because they were not executed); turns_ge3_nonacting = >= 3 of any non-X class before the first X "
             "(refusals, errors and no-tool replies included: the strict read); acting_after_refusal = the next python "
             "call after a refusal in the same turn is X (transcript, code-based) — the graft's own payload-based count "
             "is in probe.graft (per run) and aggregate.probe.graft; yields_turn_time_budget = turns whose status says "
             "turn_time_budget; wall_actions_ratio = actions_per_level[wall] / base_actions_per_level[wall] "
             "(loss-ledger-3 q7: 'wall-level actions/baseline', median 0.72 on yield900).",
    "probe_gate": "ENGAGEMENT (judge 09-08): refusals >= 1/game AND acted_after_first_refusal share >= 50 % (graft "
                  "counters: the very next call after the FIRST refusal of a span acted; a refusal that ended the turn is "
                  "a non-acting follow-up) AND wall actions/baseline median >= 0.9; SECONDARY: turns_ge3_analysis (graft "
                  "leak split: leak_cap_lifted = the leaking snippet had no action() and only the refusal cap let it run; "
                  "leak_dead_branch = it contained an action() call but executed none; leak_unparsable) and "
                  "turn_time_budget yields per draw. PRIMARY: levels vs the pooled six-draw base 39.3 (sd 2.34) and the "
                  "co-primary 'walls passed among the 12 six-draw-never-passed walls' (NEVER6_WALLS). SAFETY: GAME_OVERs "
                  "per run vs 0.87 (events.jsonl action rows with game_over) and live-cap score per game vs 8.42 "
                  "(live_cap_score: min(1.15, (b/a)^2) level-weighted).",
    "probe_span": "graft_probe counters are per SPAN: a turn that executed nothing carries its analysis/refusal counts "
                  "into the next turn on the same level (carried_turns); the span resets after an acting turn, a level "
                  "change or a new game.",
    "carry": "graft_carry reads per run from '[HARNESS CARRY]' sections: one '[CARRY-CALL]' per model call (msgs in the "
             "request, reasoning_msgs / reasoning_chars = assistant messages carrying a `reasoning` field — the stock "
             "already carries them and the template renders them, MEASURED here —, summary_chars = size of the compacted "
             "block in the system message, usage prompt/completion tokens) and one '[CARRY-COMPACT]' per compaction "
             "(dropped_msgs, input_chars fed to the compactor, summary_chars, tokens, e2e_s, ok=1|0 + err). carry.graft = "
             "the graft's own counters for the run. prompt_over_window = calls whose usage.prompt_tokens > 32768.",
    "carry_gate": "ENGAGEMENT (pre-registered 09-08): compactions >= 1 per game AND compaction failures <= 10 % of attempts "
                  "AND no call over the 32,768-token window AND >= 40 % of calls carry the block. PRIMARY / co-primary / "
                  "SAFETY are the probe arm's (levels vs 39.33 sd 2.34 with >= 48 step candidate / 45-47 redraw / <= 44 dead; "
                  "walls passed among NEVER6_WALLS >= 3; GAME_OVERs/run vs 0.87; live-cap vs 8.42) plus fit-the-clock "
                  "(calls/game x e2e + compaction calls <= 7,920 s).",
}

# ---------------------------------------------------------------------------
# helpers: hashing, games, env
# ---------------------------------------------------------------------------


def tree_sha256(root: Path) -> tuple[str, int]:
    """sha256 over (relative path, bytes) of every file under root, sorted,
    skipping __pycache__ and .pyc. Returns (hexdigest, file_count)."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name.endswith(".pyc"):
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
        n += 1
    return h.hexdigest(), n


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_game_ids(spec: str) -> list[str]:
    """'all' -> the 25 public ids in notebook order; else a comma list of
    full ids or 4-char stems (tu93). Order = the caller's."""
    if spec.strip().lower() == "all":
        return list(PUBLIC_GAME_IDS)
    by_stem = {g.split("-")[0]: g for g in PUBLIC_GAME_IDS}
    out: list[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok in PUBLIC_GAME_IDS:
            gid = tok
        elif tok in by_stem:
            gid = by_stem[tok]
        else:
            raise ValueError(f"unknown game {tok!r}; known stems: {sorted(by_stem)}")
        if gid not in out:
            out.append(gid)
    if not out:
        raise ValueError("--games resolved to an empty list")
    return out


def arm_analyzer_env(arm: str) -> dict[str, str]:
    if arm not in ARM_ENV:
        raise ValueError(f"unknown arm {arm!r} (choose from {ARMS})")
    return dict(ARM_ENV[arm])


def parse_knobs(items: list[str] | None) -> dict[str, str]:
    """--knob KEY=VALUE overrides (recorded in results.json; meant for dry runs
    and pre-registered sweeps, never silently)."""
    out: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--knob expects KEY=VALUE, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def install_env(arm: str, base_url: str, token: str, out_dir: Path,
                knobs: dict[str, str] | None = None) -> dict[str, str]:
    """Process env BEFORE any harness import (tool_agent reads its
    _LOCAL_ANALYZER_* constants at import time; ONLY_RESET_LEVELS must precede
    arcengine). Returns the recorded (token-free) env."""
    for key in [k for k in os.environ if k.startswith("TAAF_")]:
        del os.environ[key]              # clean slate; keith's TAAF_VLLM_* are serving-side
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(key, None)        # tool_agent._headers fallback chain — only ours
    for key in [k for k in os.environ if k.startswith(GRAFT_ENV_PREFIXES)]:
        os.environ.pop(key, None)        # a graft flag inherited from the shell must never leak into a stock arm
    os.environ.update(PROCESS_ENV)
    env = arm_analyzer_env(arm)
    env.update(knobs or {})
    os.environ.update(env)
    os.environ.update({
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_API_KEY": token,
        "RECORDINGS_DIR": str(out_dir / "server_recording"),
    })
    recorded = {**PROCESS_ENV, **env, "LOCAL_ANALYZER_BASE_URL": base_url,
                "OPENAI_BASE_URL": base_url, "LOCAL_ANALYZER_API_KEY": "<redacted>"}
    return recorded


def install_paths() -> None:
    for p in (TOOLKIT_DIR, STOCK_AGENT_DIR, FRAMEWORK_DIR):
        assert p.is_dir(), f"missing {p}"
    for p in (FRAMEWORK_DIR, STOCK_AGENT_DIR, TOOLKIT_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def assert_stock_tree() -> dict[str, str]:
    agent_sha, n_agent = tree_sha256(STOCK_AGENT_DIR)
    if agent_sha != STOCK_AGENT_TREE_SHA256:
        raise RuntimeError(
            f"stock agent tree {STOCK_AGENT_DIR} hashes to {agent_sha[:16]}…, "
            f"expected june_stock pin {STOCK_AGENT_TREE_SHA256[:16]}… — the bytes moved")
    fw_sha, n_fw = tree_sha256(FRAMEWORK_TREE_DIR)
    if fw_sha != FRAMEWORK_TREE_SHA256:
        raise RuntimeError(f"framework tree hashes to {fw_sha[:16]}…, expected "
                           f"{FRAMEWORK_TREE_SHA256[:16]}…")
    bm_sha = file_sha256(BUNDLE_PKL_DIR / "benchmark_initial.pkl")
    dt_sha = file_sha256(BUNDLE_PKL_DIR / "deploy_target.pkl")
    if bm_sha != BENCHMARK_PKL_SHA256 or dt_sha != DEPLOY_TARGET_PKL_SHA256:
        raise RuntimeError("solver pickles differ from the pinned (keith-identical) bytes")
    return {"agent_dir": str(STOCK_AGENT_DIR), "agent_tree_sha256": agent_sha,
            "agent_files": n_agent, "framework_dir": str(FRAMEWORK_TREE_DIR),
            "framework_tree_sha256": fw_sha, "framework_files": n_fw,
            "benchmark_pkl_sha256": bm_sha, "deploy_target_pkl_sha256": dt_sha}


def verify_imports() -> None:
    """The imported modules must come from the pinned trees, not a stray install."""
    import inference  # noqa: PLC0415
    import taaf  # noqa: PLC0415
    from inference.agent import tool_agent as ta  # noqa: PLC0415
    for mod, root in ((inference, STOCK_AGENT_DIR), (taaf, FRAMEWORK_DIR)):
        f = Path(getattr(mod, "__file__", "") or "").resolve()
        assert str(f).startswith(str(root.resolve())), f"{mod.__name__} imported from {f}, not {root}"
    want_ctx = int(os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"])
    want_out = int(os.environ["LOCAL_ANALYZER_MAX_OUTPUT"])
    assert ta._LOCAL_ANALYZER_CONTEXT_WINDOW == want_ctx, ta._LOCAL_ANALYZER_CONTEXT_WINDOW
    assert ta._LOCAL_ANALYZER_MAX_OUTPUT == want_out, ta._LOCAL_ANALYZER_MAX_OUTPUT
    assert ta._LOCAL_ANALYZER_YIELD_SECONDS == float(os.environ["LOCAL_ANALYZER_YIELD_SECONDS"]), ta._LOCAL_ANALYZER_YIELD_SECONDS
    assert ta._LOCAL_ANALYZER_TOOL_STEPS == 0
    assert (ta._LOCAL_ANALYZER_TEMPERATURE, ta._LOCAL_ANALYZER_TOP_P, ta._LOCAL_ANALYZER_TOP_K) == (0.6, 0.95, 20)
    assert ta._LOCAL_ANALYZER_ENABLE_THINKING is True
    from inference.agent import vision_context as vc  # noqa: PLC0415
    assert vc.current_grid_image_enabled(), os.environ.get("MULTIMODAL_CONTEXT")
    assert vc.current_grid_image_upscale() == int(os.environ["MULTIMODAL_UPSCALE"]), vc.current_grid_image_upscale()


def vision_image_facts(grid_size: int = 64, patch_px: int = 16, merge: int = 2) -> dict:
    """What the arm's MULTIMODAL_UPSCALE makes of a 64x64 frame: the PNG the stock
    harness attaches (rendered through vision_context.frame_to_png_data_url) and
    the vision-token count DERIVED from Qwen3-VL geometry (patch 16 px, 2x2 merge
    -> one token per 32x32 px). 64 tokens at upscale 4 was matched against vLLM's
    prompt_tokens on 25 real calls (judge recon_tokens.py, median ratio 0.986);
    the value at other upscales is the same formula, not a measurement."""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from inference.agent import vision_context as vc  # noqa: PLC0415
    from inference.agent.runtime_state import Frame  # noqa: PLC0415
    up = vc.current_grid_image_upscale()
    frame = Frame(grid=tuple(tuple((r * 3 + c) % 16 for c in range(grid_size)) for r in range(grid_size)), step=0, level=1)
    url = vc.frame_to_png_data_url(frame)
    png = base64.b64decode(url.split(",", 1)[1])
    w, h = Image.open(io.BytesIO(png)).size
    per_side = max(1, (w // (patch_px * merge)))
    return {"upscale": up, "png_px": [w, h], "png_bytes": len(png),
            "vision_tokens_derived": per_side * (max(1, h // (patch_px * merge))),
            "derivation": f"({w}/{patch_px * merge})^2, patch {patch_px} px x merge {merge}; measured only at upscale 4 (64)"}


def install_grafts(arm: str) -> dict[str, str]:
    """Install the arm's in-memory grafts AFTER the stock imports are verified.
    Each graft's install() rebinds ToolAgent methods (no file is touched, so
    assert_stock_tree still holds). Returns {graft: install status}."""
    names = ARM_GRAFTS.get(arm, ())
    if not names:
        return {}
    if str(GRAFT_DIR) not in sys.path:
        sys.path.insert(0, str(GRAFT_DIR))
    statuses: dict[str, str] = {}
    for name in names:
        mod = __import__(name)
        status = mod.install()
        if not status.endswith(": OK"):
            raise RuntimeError(f"graft {name} did not install: {status}")
        statuses[name] = status
    return statuses


KEITH_FIRST_CALL_PROMPT_TOKENS = 4050       # n_messages==2 prompt_tokens on the real keith run (4042-4059)
UP8_EXPECTED_FIRST_CALL_DELTA = 192         # 256 - 64 vision tokens per image (derived, §UP8)
WALL_MIN_CALLS_LEFT_AT_L2 = 30              # judge: an attempt counts only if L2 is reached with >= 30 calls left
VOID_LENGTH_SHARE = 0.01

_MAX_CALLS: dict = {"n": None, "installed": False, "stops": {}}


def install_max_calls(n: int | None) -> dict:
    """--max-calls N: in-memory per-run stop after N analyzer calls (successful chat completions).
    Wraps ToolAgent._chat_completion (one ToolAgent per run, built by the factory). After the Nth
    completion returns, the run's OWN cap is triggered through the session's runtime_limit_reached()
    (started_at is moved back by max_runtime_s_per_game): should_stop() turns True, the in-flight turn
    yields after acting on that Nth reply, play() exits and the run finishes exactly like a run that
    hit --per-game-s (state gave_up). The session's stop_event is deliberately NOT used: it is the
    solver-wide event shared by every run (a set flag stops all of them and _finish_if_needed marks
    them 'cancelled')."""
    _MAX_CALLS.update({"n": n, "stops": {}})
    if n is None or _MAX_CALLS["installed"]:
        return dict(_MAX_CALLS)
    from inference.agent import tool_agent as ta  # noqa: PLC0415
    stock = ta.ToolAgent._chat_completion

    def _chat_completion(self, messages, **kwargs):
        result = stock(self, messages, **kwargs)
        count = int(getattr(self, "_rw_calls", 0)) + 1
        self._rw_calls = count
        limit = _MAX_CALLS["n"]
        if limit is not None and count >= int(limit):
            sess = getattr(getattr(self, "_step_env_callback", None), "__self__", None)
            if sess is not None and not getattr(sess, "_rw_max_calls_hit", False):
                cap = getattr(getattr(sess, "solver", None), "max_runtime_s_per_game", None)
                if cap is not None:
                    sess.started_at = time.monotonic() - float(cap) - 1.0
                    sess._rw_max_calls_hit = True
                    _MAX_CALLS["stops"][current_run_stem() or current_game_tag() or "?"] = count
        return result

    _chat_completion._rw_stock = stock
    ta.ToolAgent._chat_completion = _chat_completion
    _MAX_CALLS["installed"] = True
    return dict(_MAX_CALLS)


def graft_status(arm: str) -> dict[str, dict]:
    """Each installed graft's status() counters (telemetry)."""
    out: dict[str, dict] = {}
    for name in ARM_GRAFTS.get(arm, ()):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "status"):
            try:
                out[name] = mod.status()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ---------------------------------------------------------------------------
# hook 1: analyzer factory (stock ToolAgent, exactly _make_analyzer's args) + game tag
# ---------------------------------------------------------------------------

_GAME_TAG = threading.local()


def current_game_tag() -> str | None:
    return getattr(_GAME_TAG, "game_id", None)


def current_run_stem() -> str | None:
    return getattr(_GAME_TAG, "run_stem", None)


def run_stem_for(game_id: str, index: int, n_games: int | None) -> str:
    """The harness's own run stem (solver._run_stem): <artifact_stem(game_id)>_p<draw>.
    taaf plays passes in order (pass 0 = the first n_games entries of to_play, ...)
    and the solver assigns pass_index per game_id in that same order, so
    draw = index // n_games."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", game_id)
    draw = (index // n_games) if n_games else 0
    return f"{stem}_p{draw}"


def make_tagging_analyzer_factory(solver, n_games: int | None = None):
    """HarnessSolver.analyzer_factory hook: returns the stock ToolAgent built
    with the same arguments HarnessSolver._make_analyzer uses when no local
    server is started (model=self.model, timeout=self.analyzer_timeout,
    save_request_logs=self.save_request_logs, api_key=None, base_url=None,
    provider=None) and tags the calling (game) thread with the game id and
    the run stem (<gid>_p<draw>)."""
    from inference.agent.tool_agent import ToolAgent  # noqa: PLC0415

    def factory(game, index):
        run = getattr(game, "game_run", None)
        gid = getattr(run, "game_id", None) or f"index-{index}"
        _GAME_TAG.game_id = gid
        _GAME_TAG.run_stem = run_stem_for(gid, index, n_games)
        return ToolAgent(
            model=solver.model,
            timeout=solver.analyzer_timeout,
            save_request_logs=solver.save_request_logs,
            api_key=None,
            base_url=None,
            provider=None,
        )
    return factory


def analyzer_config_fingerprint(agent) -> dict:
    """The fields that decide what the agent sends (for the factory-equivalence test)."""
    m = agent._model
    return {"provider": m.provider, "base_url": m.base_url, "model_id": m.model_id,
            "timeout": agent._timeout, "save_request_logs": agent._save_request_logs,
            "api_key_set": bool(agent._api_key), "max_output_tokens": agent._max_output_tokens,
            "context_budget_tokens": agent._context_budget_tokens,
            "yield_seconds": agent._yield_seconds, "tool_steps": agent._tool_steps}


# ---------------------------------------------------------------------------
# hook 2: requests.post observer shim (per-call telemetry; changes nothing)
# ---------------------------------------------------------------------------


class RequestShim:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.count = 0
        self.errors = 0
        self.redirected = 0
        self._orig = None
        self._fh = None

    def install(self) -> None:
        import requests  # noqa: PLC0415
        self._orig = requests.post
        self._fh = open(self.path, "a", encoding="utf-8")
        orig = self._orig
        shim = self

        def post(url, data=None, json=None, **kwargs):  # noqa: A002 - requests' signature
            if not str(url).rstrip("/").endswith("/chat/completions"):
                return orig(url, data=data, json=json, **kwargs)
            rec = {"t": time.time(), "game_id": current_game_tag(), "run_stem": current_run_stem(),
                   "timeout": kwargs.get("timeout"),
                   "allow_redirects": kwargs.get("allow_redirects", True)}
            if isinstance(json, dict):
                rec["n_messages"] = len(json.get("messages") or [])
                rec["max_tokens"] = json.get("max_tokens")
                rec["has_tools"] = bool(json.get("tools"))
            t0 = time.monotonic()
            try:
                resp = orig(url, data=data, json=json, **kwargs)
            except Exception as exc:  # noqa: BLE001 - observe, then re-raise unchanged
                rec["elapsed_s"] = round(time.monotonic() - t0, 3)
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                shim._write(rec, error=True)
                raise
            rec["elapsed_s"] = round(time.monotonic() - t0, 3)
            rec["status"] = resp.status_code
            rec["redirects"] = len(resp.history)
            rec["redirect_codes"] = [h.status_code for h in resp.history]
            rec["final_path"] = urlparse(resp.url).path
            try:
                body = resp.json()
                usage = body.get("usage") or {}
                choice = (body.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                reasoning = msg.get("reasoning")
                if reasoning in (None, ""):
                    reasoning = msg.get("reasoning_content") or ""
                rec.update({
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                    "content_chars": len(msg.get("content") or "") if isinstance(msg.get("content"), str) else 0,
                    "tool_calls": len(msg.get("tool_calls") or []),
                })
            except Exception:  # noqa: BLE001 - non-JSON / error bodies: keep the status only
                pass
            shim._write(rec, error=resp.status_code >= 400)
            return resp

        requests.post = post

    def _write(self, rec: dict, *, error: bool) -> None:
        with self.lock:
            self.count += 1
            if error:
                self.errors += 1
            if rec.get("redirects"):
                self.redirected += 1
            if self._fh is not None:
                self._fh.write(json.dumps(rec, sort_keys=True) + "\n")
                self._fh.flush()

    def uninstall(self) -> None:
        import requests  # noqa: PLC0415
        if self._orig is not None:
            requests.post = self._orig
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def load_shim_records(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# endpoint: preflight, /metrics
# ---------------------------------------------------------------------------


def _root_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def check_identity(identity, expected_profile: str, gpu_substring: str = EXPECT_GPU_SUBSTRING) -> dict:
    """Hard gate on the endpoint's /arc3/identity: the serving profile and the
    GPU rows must be what the arm was pre-registered on. Raises RuntimeError
    with the observed values otherwise; returns the recorded check."""
    if not isinstance(identity, dict) or not identity:
        raise RuntimeError(
            "ENDPOINT IDENTITY UNAVAILABLE: GET /arc3/identity returned no payload, so the serving "
            f"profile cannot be attested (expected {expected_profile!r} on {gpu_substring!r}). Refusing to run.")
    profile = identity.get("profile")
    rows = list(((identity.get("host") or {}).get("gpu_rows")) or [])
    gpu_ok = any(gpu_substring in str(r) for r in rows)
    check = {"profile": profile, "expected_profile": expected_profile, "profile_ok": profile == expected_profile,
             "gpu_rows": rows, "expected_gpu_substring": gpu_substring, "gpu_ok": gpu_ok}
    if not check["profile_ok"]:
        raise RuntimeError(
            f"SERVING PROFILE MISMATCH: endpoint reports profile {profile!r}, this run expects "
            f"{expected_profile!r} (the 09-03 keith/kv10 runs were silently on 'kv10-bf16-mtp3-c8-cg32-OVERRIDE'). "
            f"Redeploy the endpoint on the expected profile, or pass --expect-profile {profile!r} to run on it "
            f"DELIBERATELY (it is recorded in results.json).")
    if not gpu_ok:
        raise RuntimeError(
            f"GPU MISMATCH: endpoint host gpu_rows={rows!r} do not contain {gpu_substring!r}; "
            "a run on the fallback GPU is not comparable to the pre-registered arms. Refusing to run.")
    return check


def preflight(base_url: str, token: str, deadline_s: float,
              expected_profile: str = DEFAULT_EXPECT_PROFILE) -> dict:
    """/v1/models unauthenticated (the exempt route; also wakes a scaled-to-zero
    container), /metrics with the bearer (proves auth + the telemetry route),
    one tiny non-thinking completion (proves the chat route end to end), then
    GET /arc3/identity and ASSERT the serving profile + GPU (check_identity)."""
    import requests  # noqa: PLC0415
    base = base_url.rstrip("/")
    deadline = time.monotonic() + deadline_s
    last = None
    print(f"[regime] preflight: GET {base}/models (cold start can take 10-20 min)", flush=True)
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base}/models", timeout=600)
            r.raise_for_status()
            models = r.json()
            ids = [m.get("id") for m in models.get("data", [])]
            if SERVED_MODEL_NAME not in ids:
                raise RuntimeError(f"endpoint serves {ids}, expected {SERVED_MODEL_NAME!r}")
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[regime] endpoint not ready ({type(exc).__name__}: {str(exc)[:120]}); retry in 15 s",
                  flush=True)
            time.sleep(15)
    else:
        raise TimeoutError(f"endpoint never became ready: {last!r}")
    print("[regime] /models OK", flush=True)
    hdr = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{_root_url(base)}/metrics", headers=hdr, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"GET /metrics with bearer -> {r.status_code}: {r.text[:200]}")
    print(f"[regime] /metrics OK ({len(r.text)} bytes)", flush=True)
    r = requests.post(f"{base}/chat/completions", headers={**hdr, "Content-Type": "application/json"},
                      json={"model": SERVED_MODEL_NAME,
                            "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                            "temperature": 0.0, "max_tokens": 8,
                            "chat_template_kwargs": {"enable_thinking": False}},
                      timeout=900)
    r.raise_for_status()
    content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    print(f"[regime] authenticated completion OK: {content[:40]!r} "
          f"(redirect legs: {len(r.history)})", flush=True)
    identity = None
    identity_error = None
    try:
        ri = requests.get(f"{_root_url(base)}/arc3/identity", headers=hdr, timeout=120)
        if ri.status_code == 200:
            identity = ri.json()
        else:
            identity_error = f"HTTP {ri.status_code}"
    except Exception as exc:  # noqa: BLE001
        identity = None
        identity_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    if identity is None:
        print(f"[regime] /arc3/identity unavailable ({identity_error})", flush=True)
    identity_check = check_identity(identity, expected_profile)     # raises on mismatch — never silent
    print(f"[regime] identity OK: profile={identity_check['profile']!r} gpu_rows={identity_check['gpu_rows']}",
          flush=True)
    return {"models": models, "identity": identity, "identity_check": identity_check,
            "preflight_redirect_legs": len(r.history)}


def fetch_metrics(base_url: str, token: str) -> str | None:
    import requests  # noqa: PLC0415
    try:
        r = requests.get(f"{_root_url(base_url)}/metrics",
                         headers={"Authorization": f"Bearer {token}"}, timeout=120)
        if r.status_code == 200:
            return r.text
        print(f"[regime] WARN /metrics -> {r.status_code}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[regime] WARN /metrics failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    return None


_PROM_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+(-?[0-9.eE+-]+|NaN|[+-]?Inf)\s*$")
_PROM_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom(text: str) -> dict[str, list[tuple[dict, float]]]:
    out: dict[str, list[tuple[dict, float]]] = {}
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(value)
        except ValueError:
            continue
        out.setdefault(name, []).append((dict(_PROM_LABEL.findall(labels)), v))
    return out


_METRIC_SUMS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "e2e_sum": "vllm:e2e_request_latency_seconds_sum",
    "e2e_count": "vllm:e2e_request_latency_seconds_count",
    "queue_sum": "vllm:request_queue_time_seconds_sum",
    "queue_count": "vllm:request_queue_time_seconds_count",
    "inference_sum": "vllm:request_inference_time_seconds_sum",
    "inference_count": "vllm:request_inference_time_seconds_count",
    "prefill_sum": "vllm:request_prefill_time_seconds_sum",
    "prefill_count": "vllm:request_prefill_time_seconds_count",
    "decode_sum": "vllm:request_decode_time_seconds_sum",
    "decode_count": "vllm:request_decode_time_seconds_count",
    "ttft_sum": "vllm:time_to_first_token_seconds_sum",
    "ttft_count": "vllm:time_to_first_token_seconds_count",
    "tpot_sum": "vllm:request_time_per_output_token_seconds_sum",
    "tpot_count": "vllm:request_time_per_output_token_seconds_count",
    "req_gen_tokens_sum": "vllm:request_generation_tokens_sum",
    "req_prompt_tokens_sum": "vllm:request_prompt_tokens_sum",
    "preemptions": "vllm:num_preemptions_total",
    "spec_drafts": "vllm:spec_decode_num_drafts_total",
    "spec_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "spec_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "prefix_queries": "vllm:prefix_cache_queries_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
}
_METRIC_GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}


def summarize_metrics(text: str | None) -> dict | None:
    """Flatten the vLLM Prometheus text into the counters the comparison uses
    (summed over labels, except request_success which keeps finished_reason)."""
    if not text:
        return None
    prom = parse_prom(text)
    out: dict = {"present": True}
    for key, name in _METRIC_SUMS.items():
        series = prom.get(name)
        out[key] = sum(v for _, v in series) if series else None
    for key, name in _METRIC_GAUGES.items():
        series = prom.get(name)
        out[key] = sum(v for _, v in series) if series else None
    by_reason: dict[str, float] = {}
    for labels, v in prom.get("vllm:request_success_total", []):
        reason = labels.get("finished_reason", "?")
        by_reason[reason] = by_reason.get(reason, 0.0) + v
    out["request_success_by_reason"] = by_reason
    out["request_success"] = sum(by_reason.values()) if by_reason else None
    out["model_names"] = sorted({lab.get("model_name") for series in prom.values()
                                 for lab, _ in series if lab.get("model_name")})
    return out


def _sub(a, b):
    return None if a is None or b is None else a - b


def _div(a, b):
    return None if a in (None, 0) or b in (None, 0) else a / b


def metrics_delta(before: dict | None, after: dict | None, wall_s: float | None) -> dict | None:
    if not before or not after:
        return None
    d: dict = {}
    for key in _METRIC_SUMS:
        d[key] = _sub(after.get(key), before.get(key))
    reasons = set(before.get("request_success_by_reason", {})) | set(after.get("request_success_by_reason", {}))
    d["request_success_by_reason"] = {
        r: _sub(after.get("request_success_by_reason", {}).get(r, 0.0),
                before.get("request_success_by_reason", {}).get(r, 0.0)) for r in sorted(reasons)}
    d["requests"] = d["e2e_count"]
    d["e2e_mean_s"] = _div(d["e2e_sum"], d["e2e_count"])
    d["queue_mean_s"] = _div(d["queue_sum"], d["queue_count"])
    d["inference_mean_s"] = _div(d["inference_sum"], d["inference_count"])
    d["prefill_mean_s"] = _div(d["prefill_sum"], d["prefill_count"])
    d["decode_mean_s"] = _div(d["decode_sum"], d["decode_count"])
    d["ttft_mean_s"] = _div(d["ttft_sum"], d["ttft_count"])
    d["tpot_mean_s"] = _div(d["tpot_sum"], d["tpot_count"])
    d["gen_tokens_per_request"] = _div(d["generation_tokens"], d["requests"])
    d["prompt_tokens_per_request"] = _div(d["prompt_tokens"], d["requests"])
    d["gen_tokens_per_s"] = _div(d["generation_tokens"], wall_s)
    d["prompt_tokens_per_s"] = _div(d["prompt_tokens"], wall_s)
    d["mtp_acceptance_rate"] = _div(d["spec_accepted_tokens"], d["spec_draft_tokens"])
    d["mtp_accepted_per_draft"] = _div(d["spec_accepted_tokens"], d["spec_drafts"])
    d["prefix_cache_hit_rate"] = _div(d["prefix_hits"], d["prefix_queries"])
    d["after_gauges"] = {k: after.get(k) for k in _METRIC_GAUGES}
    return d


# ---------------------------------------------------------------------------
# telemetry from the harness's own transcripts
# ---------------------------------------------------------------------------

_TURN_HEADER = r"--- analysis_step=(?P<step>\d+) \| action=(?P<action>\d+) \| (?P<time>\d\d:\d\d:\d\d) \| tool-agent ---"
_SECTION = (r"\[(?P<label>SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|ASSISTANT|"
            r"ANALYZER STATUS|HARNESS CARRY|TOOL CALL: [^\]\n]+|TOOL RESULT: [^\]\n]+)\]")
# graft_carry markers (one [HARNESS CARRY] section each; written BEFORE the call's [MODEL RESPONSE META])
_CARRY_CALL_RE = re.compile(r"^\[CARRY-CALL\] game=\S+ turn=(?P<turn>\d+) req=(?P<req>\d+) msgs=(?P<msgs>\d+) "
                            r"reasoning_msgs=(?P<rmsgs>\d+) reasoning_chars=(?P<rchars>\d+) summary_chars=(?P<schars>\d+) "
                            r"prompt_tokens=(?P<pt>\S+) completion_tokens=(?P<ct>\S+)$", re.M)
_CARRY_COMPACT_RE = re.compile(r"^\[CARRY-COMPACT\] game=\S+ turn=(?P<turn>\d+) dropped_msgs=(?P<dropped>\d+) "
                               r"input_chars=(?P<ichars>\d+) summary_chars=(?P<schars>\d+) prompt_tokens=(?P<pt>\S+) "
                               r"completion_tokens=(?P<ct>\S+) e2e_s=(?P<e2e>\S+) ok=(?P<ok>[01])", re.M)


def _num(v: str):
    try:
        return float(v) if "." in v else int(v)
    except (TypeError, ValueError):
        return None
_EVENT_RE = re.compile(rf"^(?:{_TURN_HEADER}|{_SECTION})$", re.M)


def parse_transcript(text: str) -> dict:
    """Split a duck transcript into turns and model calls.

    turns: [{step, action, time, calls, status_messages, step_executed, outcome}]
    calls: [{turn_index, finish_reason, tool_call_count, content_chars,
             reasoning_chars_meta, reasoning_chars}]  (reasoning_chars = the
             stripped [THINKING] section that follows the META, else 0)
    """
    events = []
    for m in _EVENT_RE.finditer(text):
        events.append((m.start(), m.end(), m))
    turns: list[dict] = []
    calls: list[dict] = []
    carry_calls: list[dict] = []
    carry_compactions: list[dict] = []
    status_cfg: dict = {}
    for i, (start, end, m) in enumerate(events):
        body_end = events[i + 1][0] if i + 1 < len(events) else len(text)
        body = text[end:body_end]
        if m.group("step") is not None:
            turns.append({"step": int(m.group("step")), "action": int(m.group("action")),
                          "time": m.group("time"), "calls": 0, "status_messages": [],
                          "step_executed": False, "outcome": "incomplete",
                          "level": None, "hypo": 0, "evid": 0, "evid_level_flags": 0, "quotes": 0,
                          "probe_refusals": 0, "probe_noact": 0, "probe_noact_notice": 0, "carry_compactions": 0})
            continue
        label = m.group("label")
        if label == "HARNESS CARRY":
            for cm in _CARRY_CALL_RE.finditer(body):
                carry_calls.append({"turn_index": len(turns) - 1, "msgs": int(cm.group("msgs")),
                                    "reasoning_msgs": int(cm.group("rmsgs")), "reasoning_chars": int(cm.group("rchars")),
                                    "summary_chars": int(cm.group("schars")), "prompt_tokens": _num(cm.group("pt")),
                                    "completion_tokens": _num(cm.group("ct"))})
            for cm in _CARRY_COMPACT_RE.finditer(body):
                carry_compactions.append({"turn_index": len(turns) - 1, "ok": cm.group("ok") == "1",
                                          "dropped_msgs": int(cm.group("dropped")), "input_chars": int(cm.group("ichars")),
                                          "summary_chars": int(cm.group("schars")), "prompt_tokens": _num(cm.group("pt")),
                                          "completion_tokens": _num(cm.group("ct")), "e2e_s": _num(cm.group("e2e"))})
                if turns:
                    turns[-1]["carry_compactions"] += 1
            continue
        if label in ("THINKING", "ASSISTANT") and turns:
            turns[-1]["quotes"] += len(_QUOTE_RE.findall(body))
        if label == "USER PROMPT" and turns:
            t = turns[-1]
            if t["level"] is None:
                lm = _PROMPT_LEVEL_RE.search(body)
                if lm:
                    t["level"] = int(lm.group(1))
            t["hypo"] += len(_HYPO_MARK.findall(body))
            t["probe_noact_notice"] += len(_PROBE_NOTICE_RE.findall(body))
        elif label.startswith("TOOL CALL: ") and calls:
            c = calls[-1]
            if c["turn_index"] == len(turns) - 1 and label == "TOOL CALL: python":
                c["python_calls"] += 1
                code = _tool_call_code(body)
                c["code_chars"] += len(code)
                c["acts_code"] = c["acts_code"] or bool(_ACTION_CALL_RE.search(code))
                refused = len(_PROBE_REFUSE_MARK.findall(body))
                c["refused"] += refused
                if turns:
                    turns[-1]["probe_refusals"] += refused
        elif label.startswith("TOOL RESULT: ") and turns:
            turns[-1]["evid"] += len(_EVID_MARK.findall(body))
            turns[-1]["evid_level_flags"] += len(_EVID_LEVEL_MARK.findall(body))
            if calls and calls[-1]["turn_index"] == len(turns) - 1 and label == "TOOL RESULT: python":
                calls[-1]["pyerr"] = calls[-1]["pyerr"] or bool(_PYERR_RE.search(body))
        if label == "MODEL RESPONSE META":
            call = {"turn_index": len(turns) - 1, "finish_reason": None, "tool_call_count": 0,
                    "content_chars": 0, "reasoning_chars_meta": 0, "reasoning_chars": 0,
                    "python_calls": 0, "code_chars": 0, "acts_code": False, "refused": 0, "pyerr": False}
            for line in body.splitlines():
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "finish_reason":
                    call["finish_reason"] = v
                elif k == "tool_call_count":
                    call["tool_call_count"] = int(v or 0)
                elif k == "content_chars":
                    call["content_chars"] = int(v or 0)
                elif k == "reasoning_chars":
                    call["reasoning_chars_meta"] = int(v or 0)
            nxt = events[i + 1][2] if i + 1 < len(events) else None
            if nxt is not None and nxt.group("label") == "THINKING":
                nb_end = events[i + 2][0] if i + 2 < len(events) else len(text)
                call["reasoning_chars"] = len(text[events[i + 1][1]:nb_end].strip())
            calls.append(call)
            if turns:
                turns[-1]["calls"] += 1
        elif label == "ANALYZER STATUS" and turns:
            t = turns[-1]
            for line in body.strip().splitlines():
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "message":
                    t["status_messages"].append(v)
                elif k == "step_executed":
                    t["step_executed"] = (v == "True")
                elif k in ("context_budget_tokens", "max_output_tokens", "yield_seconds") and k not in status_cfg:
                    status_cfg[k] = v
            first = body.strip().split("\n", 1)[0] if body.strip() else ""
            if first.startswith("request_error"):
                t["status_messages"].append("request_error")
            elif first.startswith("error:"):
                t["status_messages"].append("error")
            t["probe_noact"] += len(_PROBE_NOACT_MARK.findall(body))   # written right after the turn's status
    for c in calls:
        c["ledger_type"] = _ledger_call_type(c)
    for ti, t in enumerate(turns):
        seq = [c for c in calls if c["turn_index"] == ti]
        t["call_types"] = "".join(c["ledger_type"] for c in seq)
        before = t["call_types"].split("X", 1)[0]
        t["analysis_before_act"] = before.count("A")
        t["nonacting_before_act"] = len(before)
        t["yield_turn_time_budget"] = any("turn_time_budget" in m for m in t["status_messages"])
        # the next python call after a refusal, inside the turn: did it act (code-based)? A refusal that was the
        # turn's last call is a non-acting follow-up. first_* = the FIRST refusal of the turn only.
        after, acted = 0, 0
        pending = False
        first_pending, first_of, first_acted = False, 0, 0
        for c in seq:
            if c["tool_call_count"] > 0:
                if pending:
                    after += 1
                    acted += bool(c["acts_code"] and not c["refused"])
                    pending = False
                if first_pending:
                    first_of += 1
                    first_acted += bool(c["acts_code"] and not c["refused"])
                    first_pending = False
            if c["refused"]:
                if first_of == 0 and not first_pending:
                    first_pending = True
                pending = True
        if pending:
            after += 1                       # turn-ending refusal: settled as non-acting
        if first_pending:
            first_of += 1
        t["calls_after_refusal"], t["acting_after_refusal"] = after, acted
        t["first_refusal_followups"], t["acted_after_first_refusal"] = first_of, first_acted
    for t in turns:
        msgs = " | ".join(t["status_messages"])
        if "Step executed" in msgs:
            t["outcome"] = "step_executed"
        elif "Yielded control" in msgs:
            t["outcome"] = "yielded"
        elif "No action(...) call was captured" in msgs:
            t["outcome"] = "no_capture"
        elif "request_error" in msgs:
            t["outcome"] = "request_error"
        elif "error" in msgs:
            t["outcome"] = "error"
    retry_markers = {"retries_fired": len(_RETRY_MARK.findall(text)),
                     "retry_clears": len(_RETRY_CLEAR_MARK.findall(text))}
    request_errors = sum(1 for t in turns if "request_error" in t["status_messages"])
    aid_markers = {"evid_markers": sum(t["evid"] for t in turns),
                   "evid_level_flags": sum(t["evid_level_flags"] for t in turns),
                   "hypo_markers": sum(t["hypo"] for t in turns)}
    probe_markers = {"refusals": sum(t["probe_refusals"] for t in turns),
                     "noact_turns": sum(t["probe_noact"] for t in turns),
                     "noact_notices": sum(t["probe_noact_notice"] for t in turns)}
    return {"turns": turns, "calls": calls, "analyzer_status_config": status_cfg, "retry_markers": retry_markers,
            "aid_markers": aid_markers, "probe_markers": probe_markers, "request_errors": request_errors,
            "carry_markers": {"calls": carry_calls, "compactions": carry_compactions}}


def _tool_call_code(body: str) -> str:
    """The snippet inside a [TOOL CALL: python] section: the stock markup rendering
    (<parameter=code>...</parameter>) or the JSON fallback ({"code": ...})."""
    m = _CODE_PARAM_RE.search(body)
    if m:
        return m.group(1)
    try:
        obj = json.loads(body.strip().split("\n[HARNESS PROBE]", 1)[0])
        if isinstance(obj, dict) and isinstance(obj.get("code"), str):
            return obj["code"]
    except (ValueError, TypeError):
        pass
    return ""


def _ledger_call_type(c: dict) -> str:
    """loss-ledger-3 q3.py ctype, plus R for a call graft_probe refused."""
    if c["tool_call_count"] == 0:
        return "N"
    if c["refused"]:
        return "R"
    if c["pyerr"] and not c["acts_code"]:
        return "E"
    return "X" if c["acts_code"] else "A"


# graft_retry marker lines (written under a "[HARNESS RETRY]" transcript section)
_RETRY_MARK = re.compile(r"^\[RETRY\] game=\S+ level=\d+ actions=\d+", re.M)
_RETRY_CLEAR_MARK = re.compile(r"^\[RETRY-CLEAR\] game=\S+ level=\d+ actions=", re.M)
# graft_probe: refusal marker (inside the refused call's [TOOL CALL] section), NOACT marker (after the turn's
# [ANALYZER STATUS]), the one-line notice at the top of the next [USER PROMPT]
_PROBE_REFUSE_MARK = re.compile(r"^\[PROBE-REFUSE\] game=\S+ turn=\d+ analysis_calls=\d+", re.M)
_PROBE_NOACT_MARK = re.compile(r"^\[PROBE-NOACT\] game=\S+ turn=\d+ analysis_calls=\d+", re.M)
_PROBE_NOTICE_RE = re.compile(r"^Previous turn executed no action after \d+ analysis calls", re.M)
# loss-ledger-3 load.py: code from the tool-call markup, act = r'\baction\(' on the code, pyerr on the result
_CODE_PARAM_RE = re.compile(r"<parameter=code>\n?([\s\S]*?)</parameter>")
_ACTION_CALL_RE = re.compile(r"\baction\(")
_PYERR_RE = re.compile(r"Traceback|Error:")
# graft_evidence / graft_hypo blocks (line-anchored on the block's fixed header, inside the section they ride)
_EVID_MARK = re.compile(r"^\[EVID\] harness object diff for ", re.M)
_EVID_LEVEL_MARK = re.compile(r"^LEVEL CLEARED after action \d+ ", re.M)
_HYPO_MARK = re.compile(r"^\[HYPO\] Hypothesis discipline for ", re.M)
_PROMPT_LEVEL_RE = re.compile(r"^Current state: step \d+, level (\d+)", re.M)
# the model quoting the aid in its own thinking/text (judge's UPTAKE read): the markers, the trace
# line, or a MOVED/APPEARED entry cited with a coordinate
_QUOTE_RE = re.compile(r"\[EVID\]|TRACE per action|\[HYPO\]|\b(?:MOVED|APPEARED)\b[^\n]{0,80}?\(\d+,\s?\d+\)")


def wall_reads(turns: list[dict], *, wall_level: int | None, levels_completed: int | None, calls_budget: int | None,
               request_errors: int, length_share: float | None, wave_preemptions: float | None) -> dict:
    """Judge 09-06 instrument rules per run: VOID (errors / preemptions / truncation > 1 %), calls at the
    first level-2 turn, ATTEMPT (L2 reached with >= WALL_MIN_CALLS_LEFT_AT_L2 calls left), PASS (level 3
    reached = levels_completed >= 2), and UPTAKE (share of turns whose thinking/text quotes the aid)."""
    void_reasons = []
    if request_errors > 0:
        void_reasons.append(f"request_errors={request_errors}")
    if wave_preemptions:
        void_reasons.append(f"preemptions={wave_preemptions:g}")
    if length_share is not None and length_share > VOID_LENGTH_SHARE:
        void_reasons.append(f"length_finish_share={length_share:.3f}")
    total_calls = sum(t["calls"] for t in turns)
    calls_at_l2 = None
    seen = 0
    for t in turns:
        if t.get("level") is not None and t["level"] >= 2:
            calls_at_l2 = seen
            break
        seen += t["calls"]
    budget = calls_budget if calls_budget is not None else total_calls
    remaining = (budget - calls_at_l2) if calls_at_l2 is not None else None
    reached_l2 = calls_at_l2 is not None or (levels_completed or 0) >= 1
    attempt = bool(reached_l2 and remaining is not None and remaining >= WALL_MIN_CALLS_LEFT_AT_L2 and not void_reasons)
    passed = (levels_completed or 0) >= 2
    wall = [t for t in turns if wall_level is not None and t.get("level") == wall_level]
    quoting = [t for t in turns if t.get("quotes")]
    return {"void": bool(void_reasons), "void_reasons": void_reasons, "calls_total": total_calls,
            "calls_budget": budget, "calls_at_l2": calls_at_l2, "calls_remaining_at_l2": remaining,
            "reached_l2": bool(reached_l2), "attempt": attempt, "passed": passed,
            "uptake": {"turns": len(quoting), "of": len(turns), "share": _share(len(quoting), len(turns))},
            "uptake_wall": {"turns": sum(1 for t in wall if t.get("quotes")), "of": len(wall),
                            "share": _share(sum(1 for t in wall if t.get("quotes")), len(wall))}}


def _share(a, b):
    """a / b as a share; None only when there is nothing to divide by (0/0), never for 0/n."""
    return None if not b else a / b


def engagement(turns: list[dict], wall_level: int | None) -> dict:
    """Share of turns carrying each aid, overall and on the wall level."""
    def share(sel: list[dict], key: str, denom_key: str | None) -> dict:
        denom = [t for t in sel if (t.get(denom_key) if denom_key else True)]
        num = [t for t in denom if t.get(key)]
        return {"turns": len(num), "of": len(denom), "share": _share(len(num), len(denom))}
    wall = [t for t in turns if wall_level is not None and t.get("level") == wall_level]
    return {"evid": share(turns, "evid", "step_executed"), "evid_wall": share(wall, "evid", "step_executed"),
            "hypo": share(turns, "hypo", None), "hypo_wall": share(wall, "hypo", None),
            "wall_level": wall_level, "wall_turns": len(wall)}


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "sum": 0.0}
    vs = sorted(values)
    return {"n": len(vs), "mean": statistics.fmean(vs), "median": statistics.median(vs),
            "p90": vs[min(len(vs) - 1, int(round(0.9 * (len(vs) - 1))))], "max": vs[-1], "sum": sum(vs)}


_GRAFT_KEYS = ("refusals", "turns_with_refusal", "noact_turns", "carried_turns", "analysis_calls_total", "acting_calls_total",
               "calls_after_refusal", "acting_calls_after_refusal", "first_refusal_followups", "acted_after_first_refusal",
               "refusal_turn_ending", "turns_total", "turns_ge3_analysis", "leak_cap_lifted", "leak_dead_branch",
               "leak_unparsable")


def probe_reads(turns: list[dict], calls: list[dict], probe_markers: dict, *, actions_per_level: list | None,
                baselines: list | None, levels_completed: int | None, number_of_levels: int | None,
                graft_counters: dict | None = None, game_overs: int | None = None) -> dict:
    """graft_probe mechanism reads for one run (TELEMETRY_DEFINITIONS['probe'])."""
    n_turns = len(turns)
    ge3 = sum(1 for t in turns if t.get("analysis_before_act", 0) >= 3)
    ge3_strict = sum(1 for t in turns if t.get("nonacting_before_act", 0) >= 3)
    after = sum(t.get("calls_after_refusal", 0) for t in turns)
    acted = sum(t.get("acting_after_refusal", 0) for t in turns)
    first_of = sum(t.get("first_refusal_followups", 0) for t in turns)
    first_acted = sum(t.get("acted_after_first_refusal", 0) for t in turns)
    types: dict[str, int] = {}
    for c in calls:
        types[c.get("ledger_type", "?")] = types.get(c.get("ledger_type", "?"), 0) + 1
    ratio = wall_actions = wall_baseline = None
    if (levels_completed is not None and number_of_levels and levels_completed < number_of_levels
            and actions_per_level and baselines and levels_completed < len(actions_per_level)
            and levels_completed < len(baselines)):
        wall_actions = actions_per_level[levels_completed]
        wall_baseline = baselines[levels_completed]
        if wall_baseline:
            ratio = wall_actions / wall_baseline
    out = {
        "refusals": probe_markers.get("refusals", 0),
        "noact_turns": probe_markers.get("noact_turns", 0),
        "noact_notices": probe_markers.get("noact_notices", 0),
        "turns_with_refusal": sum(1 for t in turns if t.get("probe_refusals")),
        "turns_ge3_analysis": {"turns": ge3, "of": n_turns, "share": _share(ge3, n_turns)},
        "turns_ge3_nonacting": {"turns": ge3_strict, "of": n_turns, "share": _share(ge3_strict, n_turns)},
        "acting_after_refusal": {"acted": acted, "of": after, "share": _share(acted, after)},
        "acted_after_first_refusal": {"acted": first_acted, "of": first_of, "share": _share(first_acted, first_of)},
        "call_types": types,
        "analysis_call_share": _share(types.get("A", 0), len(calls)),
        "yields_turn_time_budget": sum(1 for t in turns if t.get("yield_turn_time_budget")),
        "wall_actions": wall_actions, "wall_baseline": wall_baseline, "wall_actions_ratio": ratio,
        "game_overs": game_overs,
        "live_cap_score": live_cap_score(baselines, actions_per_level, levels_completed or 0),
    }
    if graft_counters:
        g = graft_counters
        out["graft"] = {k: g.get(k) for k in _GRAFT_KEYS}
        out["graft"]["acting_after_refusal_share"] = _share(g.get("acting_calls_after_refusal", 0) or 0, g.get("calls_after_refusal", 0) or 0)
        out["graft"]["acted_after_first_refusal_share"] = _share(g.get("acted_after_first_refusal", 0) or 0, g.get("first_refusal_followups", 0) or 0)
    return out


_CARRY_GRAFT_KEYS = ("calls_total", "calls_with_summary", "reasoning_msgs_total", "reasoning_chars_total", "prompt_tokens_total",
                     "prompt_tokens_max", "prompt_over_window", "compactions", "compaction_failures", "dropped_msgs_total",
                     "dropped_chars_total", "summary_chars_total", "compaction_prompt_tokens", "compaction_completion_tokens",
                     "compaction_e2e_s", "turns_total")


def carry_reads(carry_markers: dict, graft_counters: dict | None = None) -> dict:
    """graft_carry mechanism reads for one run (TELEMETRY_DEFINITIONS['carry']) from the transcript markers."""
    calls = list((carry_markers or {}).get("calls") or [])
    comps = list((carry_markers or {}).get("compactions") or [])
    ok = [c for c in comps if c.get("ok")]
    bad = [c for c in comps if not c.get("ok")]
    pt = [c["prompt_tokens"] for c in calls if isinstance(c.get("prompt_tokens"), (int, float))]
    first_with_block = next((i + 1 for i, c in enumerate(calls) if c.get("summary_chars", 0) > 0), None)
    out = {
        "compactions": len(ok), "compaction_failures": len(bad), "compaction_attempts": len(comps),
        "dropped_msgs": sum(c.get("dropped_msgs", 0) for c in comps),
        "input_chars": sum(c.get("input_chars", 0) for c in comps),
        "summary_chars": _stats([c["summary_chars"] for c in ok]),
        "compaction_e2e_s": _stats([c["e2e_s"] for c in comps if isinstance(c.get("e2e_s"), (int, float))]),
        "compaction_prompt_tokens": _stats([c["prompt_tokens"] for c in comps if isinstance(c.get("prompt_tokens"), (int, float))]),
        "compaction_completion_tokens": _stats([c["completion_tokens"] for c in comps if isinstance(c.get("completion_tokens"), (int, float))]),
        "calls_marked": len(calls),
        "reasoning_msgs_total": sum(c.get("reasoning_msgs", 0) for c in calls),
        "reasoning_chars_total": sum(c.get("reasoning_chars", 0) for c in calls),
        "reasoning_msgs_per_call": _div(sum(c.get("reasoning_msgs", 0) for c in calls), len(calls)),
        "reasoning_chars_per_call": _div(sum(c.get("reasoning_chars", 0) for c in calls), len(calls)),
        "calls_with_summary": sum(1 for c in calls if c.get("summary_chars", 0) > 0),
        "calls_with_summary_share": _share(sum(1 for c in calls if c.get("summary_chars", 0) > 0), len(calls)),
        "first_call_with_block": first_with_block,
        "prompt_tokens": _stats(pt),
        "prompt_over_window": sum(1 for p in pt if p > CARRY_WINDOW_TOKENS),
    }
    if graft_counters:
        out["graft"] = {k: graft_counters.get(k) for k in _CARRY_GRAFT_KEYS}
    return out


def telemetry_for_game(transcript_text: str, *, actions_total: int | None = None,
                       shim_records: list[dict] | None = None,
                       wallclock_s: float | None = None,
                       levels_completed: int | None = None, number_of_levels: int | None = None,
                       calls_budget: int | None = None, wave_preemptions: float | None = None,
                       actions_per_level: list | None = None, baselines: list | None = None,
                       graft_counters: dict | None = None, game_overs: int | None = None,
                       carry_counters: dict | None = None) -> dict:
    parsed = parse_transcript(transcript_text)
    calls, turns = parsed["calls"], parsed["turns"]
    turn_levels = [t["level"] for t in turns]
    level_reached = max((lv for lv in turn_levels if lv is not None), default=None)
    if levels_completed is not None:
        won = number_of_levels is not None and levels_completed >= number_of_levels
        wall_level = None if won else levels_completed + 1
    else:
        wall_level = level_reached
    n_calls, n_turns = len(calls), len(turns)
    finish: dict[str, int] = {}
    for c in calls:
        finish[c["finish_reason"] or "(empty)"] = finish.get(c["finish_reason"] or "(empty)", 0) + 1
    outcomes: dict[str, int] = {}
    for t in turns:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
    reasoning = [c["reasoning_chars"] for c in calls]
    tel: dict = {
        "calls": n_calls,
        "turns": n_turns,
        "distinct_analysis_steps": max((t["step"] for t in turns), default=0),
        "calls_per_turn": _div(n_calls, n_turns),
        "reasoning_chars": _stats(reasoning),
        "reasoning_chars_nonzero": _stats([r for r in reasoning if r > 0]),
        "content_chars_mean": _div(sum(c["content_chars"] for c in calls), n_calls),
        "no_tool_call_calls": sum(1 for c in calls if c["tool_call_count"] == 0),
        "no_tool_call_share": _div(sum(1 for c in calls if c["tool_call_count"] == 0), n_calls),
        "finish_reasons": finish,
        "length_finish_share": _div(finish.get("length", 0), n_calls),
        "turn_outcomes": outcomes,
        "step_executed_turn_share": _div(outcomes.get("step_executed", 0), n_turns),
        "yielded_turn_share": _div(outcomes.get("yielded", 0), n_turns),
        "actions_total": actions_total,
        "actions_per_call": _div(actions_total, n_calls),
        "actions_per_turn": _div(actions_total, n_turns),
        "wallclock_s": wallclock_s,
        "calls_per_hour": _div(n_calls, (wallclock_s or 0) / 3600.0) if wallclock_s else None,
        "analyzer_status_config": parsed["analyzer_status_config"],
        "retries_fired": parsed["retry_markers"]["retries_fired"],
        "retry_clears": parsed["retry_markers"]["retry_clears"],
        "evid_markers": parsed["aid_markers"]["evid_markers"],
        "evid_level_flags": parsed["aid_markers"]["evid_level_flags"],
        "hypo_markers": parsed["aid_markers"]["hypo_markers"],
        "turn_levels": turn_levels,
        "level_reached": level_reached,
        "wall_level": wall_level,
        "engagement": engagement(turns, wall_level),
        "probe": probe_reads(turns, calls, parsed["probe_markers"], actions_per_level=actions_per_level,
                             baselines=baselines, levels_completed=levels_completed, number_of_levels=number_of_levels,
                             graft_counters=graft_counters, game_overs=game_overs),
        "carry": carry_reads(parsed["carry_markers"], carry_counters),
    }
    client_errors = sum(1 for r in (shim_records or []) if r.get("error") or (r.get("status") or 0) >= 400)
    tel["wall"] = wall_reads(turns, wall_level=wall_level, levels_completed=levels_completed, calls_budget=calls_budget,
                             request_errors=parsed["request_errors"] + client_errors,
                             length_share=tel["length_finish_share"], wave_preemptions=wave_preemptions)
    first = [r["prompt_tokens"] for r in (shim_records or []) if r.get("n_messages") == 2 and r.get("status") == 200
             and r.get("prompt_tokens") is not None]
    tel["first_call_prompt_tokens"] = first[0] if first else None
    if shim_records:
        ok = [r for r in shim_records if r.get("status") == 200]
        tel["client"] = {
            "posts": len(shim_records),
            "ok": len(ok),
            "errors": sum(1 for r in shim_records if r.get("error") or (r.get("status") or 0) >= 400),
            "e2e_s": _stats([r["elapsed_s"] for r in ok if "elapsed_s" in r]),
            "prompt_tokens": _stats([r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]),
            "completion_tokens": _stats([r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]),
            "redirected_posts": sum(1 for r in shim_records if r.get("redirects")),
            "redirect_legs": sum(r.get("redirects") or 0 for r in shim_records),
            "max_tokens_sent": sorted({r.get("max_tokens") for r in shim_records}, key=lambda x: (x is None, x)),
            "timeouts_sent": _stats([r["timeout"] for r in shim_records if isinstance(r.get("timeout"), (int, float))]),
            "model_time_share_of_wall": _div(sum(r.get("elapsed_s") or 0 for r in ok), wallclock_s),
        }
    return tel


def aggregate_telemetry(per_game: dict[str, dict]) -> dict:
    games = list(per_game.values())
    n = len(games)
    if n == 0:
        return {"games": 0}
    finish: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    tot_calls = sum(g["calls"] for g in games)
    tot_turns = sum(g["turns"] for g in games)
    tot_actions = sum(g["actions_total"] or 0 for g in games)
    no_tool = sum(g["no_tool_call_calls"] for g in games)
    for g in games:
        for k, v in g["finish_reasons"].items():
            finish[k] = finish.get(k, 0) + v
        for k, v in g["turn_outcomes"].items():
            outcomes[k] = outcomes.get(k, 0) + v
    return {
        "games": n,
        "calls_total": tot_calls,
        "calls_per_game": tot_calls / n,
        "turns_per_game": tot_turns / n,
        "distinct_steps_per_game": statistics.fmean(g["distinct_analysis_steps"] for g in games),
        "calls_per_turn": _div(tot_calls, tot_turns),
        "no_tool_call_share": _div(no_tool, tot_calls),
        "length_finish_share": _div(finish.get("length", 0), tot_calls),
        "finish_reasons": finish,
        "turn_outcomes": outcomes,
        "step_executed_turn_share": _div(outcomes.get("step_executed", 0), tot_turns),
        "yielded_turn_share": _div(outcomes.get("yielded", 0), tot_turns),
        "actions_total": tot_actions,
        "actions_per_game": tot_actions / n,
        "actions_per_call": _div(tot_actions, tot_calls),
        "retries_total": sum(g.get("retries_fired") or 0 for g in games),
        "retries_per_game": sum(g.get("retries_fired") or 0 for g in games) / n,
        "retry_clears_total": sum(g.get("retry_clears") or 0 for g in games),
        "games_with_retry": sum(1 for g in games if (g.get("retries_fired") or 0) > 0),
        "evid_markers_total": sum(g.get("evid_markers") or 0 for g in games),
        "evid_markers_per_game": sum(g.get("evid_markers") or 0 for g in games) / n,
        "evid_level_flags_total": sum(g.get("evid_level_flags") or 0 for g in games),
        "hypo_markers_total": sum(g.get("hypo_markers") or 0 for g in games),
        "hypo_markers_per_game": sum(g.get("hypo_markers") or 0 for g in games) / n,
        "engagement": _pooled_engagement(games),
        "wall": _pooled_wall(games),
        "probe": _pooled_probe(games),
        "carry": _pooled_carry(games),
        "first_call_prompt_tokens": _stats([g["first_call_prompt_tokens"] for g in games
                                            if g.get("first_call_prompt_tokens") is not None]),
    }


def _pooled_carry(games: list[dict]) -> dict:
    """Pooled graft_carry reads + the pre-registered ENGAGEMENT gate (CARRY_GATE)."""
    cs = [g.get("carry") or {} for g in games]
    n = len(cs)
    comps = sum(c.get("compactions", 0) for c in cs)
    fails = sum(c.get("compaction_failures", 0) for c in cs)
    attempts = comps + fails
    calls = sum(c.get("calls_marked", 0) for c in cs)
    with_block = sum(c.get("calls_with_summary", 0) for c in cs)
    over = sum(c.get("prompt_over_window", 0) for c in cs)
    pt_max = max([((c.get("prompt_tokens") or {}).get("max") or 0) for c in cs] + [0])
    pt_means = [((c.get("prompt_tokens") or {}).get("mean"), c.get("calls_marked", 0)) for c in cs]
    pt_mean = _div(sum(m * k for m, k in pt_means if m is not None), sum(k for m, k in pt_means if m is not None))
    grafts = [c["graft"] for c in cs if c.get("graft")]

    def gsum(key: str) -> float:
        return sum((g.get(key) or 0) for g in grafts)

    def gmax(key: str) -> float:
        return max([(g.get(key) or 0) for g in grafts] + [0])

    out = {
        "compactions_total": comps, "compactions_per_game": (comps / n) if n else None,
        "compaction_failures_total": fails, "compaction_attempts": attempts,
        "failure_share": _share(fails, attempts),
        "games_with_compaction": sum(1 for c in cs if c.get("compactions", 0) > 0),
        "dropped_msgs_per_compaction": _div(sum(c.get("dropped_msgs", 0) for c in cs), attempts),
        "input_chars_per_compaction": _div(sum(c.get("input_chars", 0) for c in cs), attempts),
        "summary_chars_mean": _div(sum(((c.get("summary_chars") or {}).get("mean") or 0) * c.get("compactions", 0) for c in cs), comps),
        "compaction_e2e_s_mean": _div(sum(((c.get("compaction_e2e_s") or {}).get("mean") or 0) * ((c.get("compaction_e2e_s") or {}).get("n") or 0) for c in cs),
                                      sum(((c.get("compaction_e2e_s") or {}).get("n") or 0) for c in cs)),
        "compaction_completion_tokens_mean": _div(sum(((c.get("compaction_completion_tokens") or {}).get("mean") or 0) * ((c.get("compaction_completion_tokens") or {}).get("n") or 0) for c in cs),
                                                  sum(((c.get("compaction_completion_tokens") or {}).get("n") or 0) for c in cs)),
        "compaction_prompt_tokens_mean": _div(sum(((c.get("compaction_prompt_tokens") or {}).get("mean") or 0) * ((c.get("compaction_prompt_tokens") or {}).get("n") or 0) for c in cs),
                                              sum(((c.get("compaction_prompt_tokens") or {}).get("n") or 0) for c in cs)),
        "calls_marked": calls, "calls_with_summary": with_block, "calls_with_summary_share": _share(with_block, calls),
        "first_call_with_block": _stats([c["first_call_with_block"] for c in cs if c.get("first_call_with_block") is not None]),
        "reasoning_msgs_per_call": _div(sum(c.get("reasoning_msgs_total", 0) for c in cs), calls),
        "reasoning_chars_per_call": _div(sum(c.get("reasoning_chars_total", 0) for c in cs), calls),
        "prompt_tokens_mean": pt_mean, "prompt_tokens_max": pt_max, "prompt_over_window": over,
        "graft": {"runs": len(grafts), "compactions": gsum("compactions"), "compaction_failures": gsum("compaction_failures"),
                  "calls_total": gsum("calls_total"), "calls_with_summary": gsum("calls_with_summary"),
                  "prompt_tokens_max": gmax("prompt_tokens_max"), "prompt_over_window": gsum("prompt_over_window"),
                  "reasoning_msgs_per_call": _div(gsum("reasoning_msgs_total"), gsum("calls_total")),
                  "compaction_e2e_s": gsum("compaction_e2e_s"), "dropped_msgs_total": gsum("dropped_msgs_total"),
                  "turns_total": gsum("turns_total")},
        "per_run_compactions": {},
    }
    gate = {
        "compactions_per_game": {"value": out["compactions_per_game"], "min": CARRY_GATE["compactions_per_game_min"],
                                 "ok": out["compactions_per_game"] is not None and out["compactions_per_game"] >= CARRY_GATE["compactions_per_game_min"]},
        "failure_share": {"value": out["failure_share"], "max": CARRY_GATE["failure_share_max"],
                          "ok": attempts > 0 and (fails / attempts) <= CARRY_GATE["failure_share_max"]},
        "prompt_over_window": {"value": over, "max": CARRY_GATE["prompt_over_window_max"], "ok": over <= CARRY_GATE["prompt_over_window_max"]},
        "calls_with_summary_share": {"value": out["calls_with_summary_share"], "min": CARRY_GATE["calls_with_summary_share_min"],
                                     "ok": out["calls_with_summary_share"] is not None
                                     and out["calls_with_summary_share"] >= CARRY_GATE["calls_with_summary_share_min"]},
    }
    gate["engaged"] = bool(all(v["ok"] for v in gate.values() if isinstance(v, dict)))
    out["gate"] = gate
    return out


def _pooled_probe(games: list[dict]) -> dict:
    ps = [g.get("probe") or {} for g in games]
    n = len(ps)
    draws = max((int(g.get("draw") or 0) for g in games), default=0) + 1

    def pooled(key: str, num: str, den: str) -> dict:
        a = sum(((p.get(key) or {}).get(num, 0)) for p in ps)
        b = sum(((p.get(key) or {}).get(den, 0)) for p in ps)
        return {num: a, den: b, "share": _share(a, b)}

    types: dict[str, int] = {}
    for p in ps:
        for k, v in (p.get("call_types") or {}).items():
            types[k] = types.get(k, 0) + v
    ratios = [p["wall_actions_ratio"] for p in ps if p.get("wall_actions_ratio") is not None]
    refusals = sum(p.get("refusals", 0) for p in ps)
    yields = sum(p.get("yields_turn_time_budget", 0) for p in ps)
    grafts = [p["graft"] for p in ps if p.get("graft")]

    def gsum(key: str) -> int:
        return sum(int(g.get(key) or 0) for g in grafts)

    g_after, g_acted = gsum("calls_after_refusal"), gsum("acting_calls_after_refusal")
    g_first_of, g_first_acted = gsum("first_refusal_followups"), gsum("acted_after_first_refusal")
    g_ge3 = gsum("turns_ge3_analysis")
    g_turns = gsum("turns_total")
    gos = [p["game_overs"] for p in ps if p.get("game_overs") is not None]
    scores = [p["live_cap_score"] for p in ps if p.get("live_cap_score") is not None]
    levels_total = sum(int(g.get("levels_completed") or 0) for g in games)
    # co-primary: the 12 six-draw-never-passed walls, passed = levels_completed >= wall level in any draw of that game
    best: dict[str, int] = {}
    for g in games:
        short = str(g.get("game_id") or "").split("-", 1)[0]
        if short in NEVER6_WALLS:
            best[short] = max(best.get(short, 0), int(g.get("levels_completed") or 0))
    walls_passed = sorted(s for s, lv in best.items() if lv >= NEVER6_WALLS[s])
    out = {
        "refusals_total": refusals, "refusals_per_game": refusals / n,       # n >= 1 here (0 refusals is a real read, not None)
        "games_with_refusal": sum(1 for p in ps if p.get("refusals", 0) > 0),
        "noact_turns_total": sum(p.get("noact_turns", 0) for p in ps),
        "noact_notices_total": sum(p.get("noact_notices", 0) for p in ps),
        "turns_ge3_analysis": pooled("turns_ge3_analysis", "turns", "of"),
        "turns_ge3_nonacting": pooled("turns_ge3_nonacting", "turns", "of"),
        "acting_after_refusal": pooled("acting_after_refusal", "acted", "of"),
        "acted_after_first_refusal": pooled("acted_after_first_refusal", "acted", "of"),
        "graft": {"runs": len(grafts), "calls_after_refusal": g_after, "acting_calls_after_refusal": g_acted,
                  "acting_after_refusal_share": _share(g_acted, g_after),
                  "first_refusal_followups": g_first_of, "acted_after_first_refusal": g_first_acted,
                  "acted_after_first_refusal_share": _share(g_first_acted, g_first_of),
                  "refusal_turn_ending": gsum("refusal_turn_ending"), "carried_turns": gsum("carried_turns"),
                  "refusals": gsum("refusals"), "turns_total": g_turns,
                  "turns_ge3_analysis": g_ge3, "turns_ge3_analysis_share": _share(g_ge3, g_turns),
                  "leak_cap_lifted": gsum("leak_cap_lifted"), "leak_dead_branch": gsum("leak_dead_branch"),
                  "leak_unparsable": gsum("leak_unparsable"),
                  "analysis_calls_total": gsum("analysis_calls_total"), "acting_calls_total": gsum("acting_calls_total")},
        "call_types": types,
        "analysis_call_share": _share(types.get("A", 0), sum(types.values())),
        "yields_turn_time_budget_total": yields, "draws": draws, "yields_per_draw": yields / draws,
        "wall_actions_ratio": {"n": len(ratios), "median": statistics.median(ratios) if ratios else None,
                               "under_1x": sum(1 for r in ratios if r < 1.0)},
        "primary": {"levels_total": levels_total, "levels_per_draw": (levels_total / draws) if n else None, "games": n, "draws": draws,
                    "base_levels_mean": LEDGER3_REFERENCE["base_levels_mean"], "base_levels_sd": LEDGER3_REFERENCE["base_levels_sd"],
                    "delta_vs_base": (levels_total / draws - LEDGER3_REFERENCE["base_levels_mean"]) if n else None,
                    "walls_passed": walls_passed, "walls_passed_n": len(walls_passed), "walls_present": sorted(best),
                    "walls_total": len(NEVER6_WALLS)},
        "safety": {"game_overs_total": sum(gos), "game_overs_per_run": (sum(gos) / len(gos)) if gos else None,
                   "game_overs_runs": len(gos), "base_game_overs_per_run": LEDGER3_REFERENCE["game_overs_per_run"],
                   "live_cap_score_total": sum(scores), "live_cap_score_per_game": (sum(scores) / len(scores)) if scores else None,
                   "live_cap_runs": len(scores), "base_live_cap_score_per_game": LEDGER3_REFERENCE["live_cap_score_per_game"]},
    }
    first_share = out["graft"]["acted_after_first_refusal_share"]
    if first_share is None:
        first_share = out["acted_after_first_refusal"]["share"]
    ge3 = out["graft"]["turns_ge3_analysis_share"] if grafts else out["turns_ge3_analysis"]["share"]
    gate = {
        "refusals_per_game": {"value": out["refusals_per_game"], "min": PROBE_GATE["refusals_per_game_min"],
                              "ok": out["refusals_per_game"] is not None and out["refusals_per_game"] >= PROBE_GATE["refusals_per_game_min"]},
        "acted_after_first_refusal": {"value": first_share, "min": PROBE_GATE["acted_after_first_refusal_min"],
                                      "ok": first_share is not None and first_share >= PROBE_GATE["acted_after_first_refusal_min"]},
        "wall_actions_ratio": {"value": out["wall_actions_ratio"]["median"], "min": PROBE_GATE["wall_actions_ratio_min"],
                               "ok": (out["wall_actions_ratio"]["median"] is not None
                                      and out["wall_actions_ratio"]["median"] >= PROBE_GATE["wall_actions_ratio_min"])},
        "turns_ge3_analysis": {"value": ge3, "max": PROBE_GATE["turns_ge3_analysis_max"], "secondary": True,
                               "ok": ge3 is not None and ge3 < PROBE_GATE["turns_ge3_analysis_max"]},
        "yields_per_draw": {"value": out["yields_per_draw"], "max": PROBE_GATE["yields_per_draw_max"], "secondary": True,
                            "ok": out["yields_per_draw"] is not None and out["yields_per_draw"] < PROBE_GATE["yields_per_draw_max"]},
    }
    gate["engaged"] = bool(gate["refusals_per_game"]["ok"] and gate["acted_after_first_refusal"]["ok"]
                           and gate["wall_actions_ratio"]["ok"])
    out["gate"] = gate
    return out


def _pooled_wall(games: list[dict]) -> dict:
    ws = [g.get("wall") or {} for g in games]
    up = sum(((w.get("uptake") or {}).get("turns", 0)) for w in ws)
    up_of = sum(((w.get("uptake") or {}).get("of", 0)) for w in ws)
    upw = sum(((w.get("uptake_wall") or {}).get("turns", 0)) for w in ws)
    upw_of = sum(((w.get("uptake_wall") or {}).get("of", 0)) for w in ws)
    return {"runs": len(ws), "void": sum(1 for w in ws if w.get("void")),
            "attempts": sum(1 for w in ws if w.get("attempt")), "passes": sum(1 for w in ws if w.get("passed")),
            "reached_l2": sum(1 for w in ws if w.get("reached_l2")),
            "uptake": {"turns": up, "of": up_of, "share": _share(up, up_of)},
            "uptake_wall": {"turns": upw, "of": upw_of, "share": _share(upw, upw_of)}}


def _pooled_engagement(games: list[dict]) -> dict:
    out: dict = {}
    for key in ("evid", "evid_wall", "hypo", "hypo_wall"):
        num = sum(((g.get("engagement") or {}).get(key) or {}).get("turns", 0) for g in games)
        den = sum(((g.get("engagement") or {}).get(key) or {}).get("of", 0) for g in games)
        out[key] = {"turns": num, "of": den, "share": _share(num, den)}
    return out


_STEM_RE = re.compile(r"^(?P<gid>.+)_p(?P<draw>\d+)\.txt$")


def build_telemetry(transcripts_dir: Path, game_rows: list[dict], shim_records: list[dict], *,
                    calls_budget: int | None = None, wave_preemptions: float | None = None,
                    graft_per_game: dict | None = None, carry_per_game: dict | None = None) -> dict:
    """Per-run + pooled telemetry from <out>/transcripts/<gid>_p<draw>.txt, the
    benchmark rows (actions, wallclock, levels) and the client shim records.
    per_game is keyed by run stem (<gid>_p<draw>; one entry per game per draw).
    graft_per_game: graft_probe.status()['per_game'] (keyed by run stem) when installed."""
    by_stem_shim: dict[str, list[dict]] = {}
    for r in shim_records:
        key = r.get("run_stem") or f"{r.get('game_id')}_p0"
        by_stem_shim.setdefault(str(key), []).append(r)
    per_game: dict[str, dict] = {}
    pooled_reasoning: list[float] = []
    pooled_e2e: list[float] = []
    pooled_prompt: list[float] = []
    pooled_completion: list[float] = []
    rows_by_stem = {r.get("run_stem") or f"{r['game_id']}_p0": r for r in game_rows}
    for path in sorted(Path(transcripts_dir).glob("*_p*.txt")):
        m = _STEM_RE.match(path.name)
        if not m:
            continue
        stem = path.name[:-len(".txt")]
        gid, draw = m.group("gid"), int(m.group("draw"))
        row = rows_by_stem.get(stem, {})
        text = path.read_text(encoding="utf-8", errors="replace")
        recs = by_stem_shim.get(stem, [])
        events_path = Path(transcripts_dir).parent / "artifacts" / f"{stem}_events.jsonl"
        tel = telemetry_for_game(text, actions_total=row.get("actions"),
                                 shim_records=recs, wallclock_s=row.get("wallclock_s"),
                                 levels_completed=row.get("levels_completed"),
                                 number_of_levels=row.get("number_of_levels"),
                                 calls_budget=calls_budget, wave_preemptions=wave_preemptions,
                                 actions_per_level=row.get("actions_per_level"), baselines=row.get("baselines"),
                                 graft_counters=(graft_per_game or {}).get(stem),
                                 game_overs=game_overs_from_events(events_path) if events_path.is_file() else None,
                                 carry_counters=(carry_per_game or {}).get(stem))
        tel["game_id"] = gid
        tel["draw"] = draw
        tel["levels_completed"] = row.get("levels_completed")
        tel["number_of_levels"] = row.get("number_of_levels")
        tel["score"] = row.get("score")
        per_game[stem] = tel
        pooled_reasoning.extend(c["reasoning_chars"] for c in parse_transcript(text)["calls"])
        pooled_e2e.extend(r["elapsed_s"] for r in recs if r.get("status") == 200 and "elapsed_s" in r)
        pooled_prompt.extend(r["prompt_tokens"] for r in recs if r.get("prompt_tokens") is not None)
        pooled_completion.extend(r["completion_tokens"] for r in recs if r.get("completion_tokens") is not None)
    agg = aggregate_telemetry(per_game)
    agg["reasoning_chars_pooled"] = _stats(pooled_reasoning)
    agg["reasoning_chars_pooled_nonzero"] = _stats([r for r in pooled_reasoning if r > 0])
    agg["client_e2e_s_pooled"] = _stats(pooled_e2e)
    agg["client_prompt_tokens_pooled"] = _stats(pooled_prompt)
    agg["client_completion_tokens_pooled"] = _stats(pooled_completion)
    agg["client_redirected_posts"] = sum(1 for r in shim_records if r.get("redirects"))
    agg["client_post_errors"] = sum(1 for r in shim_records if r.get("error") or (r.get("status") or 0) >= 400)
    agg["levels_total"] = sum((g.get("levels_completed") or 0) for g in per_game.values())
    agg["levels_per_game"] = _div(agg["levels_total"], len(per_game)) if per_game else None
    # per-(game, draw) level reached, grouped by game id
    by_gid: dict[str, dict] = {}
    for stem, g in sorted(per_game.items()):
        by_gid.setdefault(g["game_id"], {})[str(g["draw"])] = {
            "levels_completed": g.get("levels_completed"), "level_reached": g.get("level_reached"),
            "wall_level": g.get("wall_level"), "turns": g.get("turns"),
            "evid_wall_share": (g.get("engagement") or {}).get("evid_wall", {}).get("share"),
            "hypo_wall_share": (g.get("engagement") or {}).get("hypo_wall", {}).get("share"),
            "calls_at_l2": (g.get("wall") or {}).get("calls_at_l2"), "attempt": (g.get("wall") or {}).get("attempt"),
            "void": (g.get("wall") or {}).get("void"), "passed": (g.get("wall") or {}).get("passed"),
            "uptake_wall_share": ((g.get("wall") or {}).get("uptake_wall") or {}).get("share")}
    agg["draws"] = max((g["draw"] for g in per_game.values()), default=-1) + 1
    return {"definitions": TELEMETRY_DEFINITIONS, "reference_keith_commit_run": KEITH_COMMIT_REFERENCE,
            "aggregate": agg, "per_game": per_game, "per_game_draw": by_gid}


# ---------------------------------------------------------------------------
# dry run: a loopback mock vLLM (OpenAI-compatible; emulates Modal's 303 legs)
# ---------------------------------------------------------------------------

MOCK_TOOL_CODE = (
    "acts = [a for a in valid_actions if a != 'RESET' and not a.startswith('ACTION')]\n"
    "acts = acts or [a for a in valid_actions if a != 'RESET']\n"
    "a = acts[(current_frame.step + len(history)) % len(acts)] if acts else 'UP'\n"
    "if a == 'MOUSE':\n"
    "    a = {'action': 'MOUSE', 'row': (current_frame.step * 7) % 64, 'col': (current_frame.step * 11) % 64}\n"
    "r = action([a])\n"
    "print('mock-step', a, r.get('board_changed'), r.get('level_completed'))\n"
)


MOCK_ANALYSIS_CODE = "seg = current_frame.segmentation\nprint('mock-analysis', len(history), len(seg) if seg else 0)\n"
MOCK_SENTINEL = "MOCK-SENTINEL-ANALYSIS"
MOCK_SENTINEL_CODE = f"print('{MOCK_SENTINEL}', len(history))\n"


def mock_turn_position(messages: list) -> int:
    """Python calls already made in the turn in flight = tool messages after the last user
    message that carries the turn's own prompt ('Current state: step ...'; the inline
    'You have not acted yet' follow-ups do not)."""
    def text(m):
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
        return ""
    last = None
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user" and "Current state: step" in text(m):
            last = i
    if last is None:
        return 0
    return sum(1 for m in messages[last + 1:] if isinstance(m, dict) and m.get("role") == "tool")


class MockVLLM:
    """Loopback OpenAI-compatible server for --dry-run. Serves /v1/models,
    /health, /metrics (vLLM-shaped counters), /arc3/identity and
    /v1/chat/completions. Every 3rd completion answers 303 -> /v1/_poll/<id>
    (Modal's long-request continuation) so the STOCK client's redirect
    behaviour is exercised; every 7th reply is text-only (no tool call) to
    exercise the harness's follow-up path; the rest call `python` with a
    real `action(...)`."""

    def __init__(self, token: str, served_model: str = SERVED_MODEL_NAME, latency_s: float = 0.25,
                 profile: str = DEFAULT_EXPECT_PROFILE, analysis_calls_per_turn: int = 0) -> None:
        import http.server  # noqa: PLC0415
        self.token = token
        self.served_model = served_model
        self.profile = profile          # what the mock's /arc3/identity reports (the preflight gate reads it)
        self.latency_s = latency_s
        # --mock-analysis-calls N: the first N python calls of every turn are analysis-only (no action());
        # the Nth prints MOCK_SENTINEL, so a probe arm must refuse it and a stock arm runs it
        self.analysis_calls_per_turn = max(0, int(analysis_calls_per_turn or 0))
        self.lock = threading.Lock()
        self.state = {"calls": 0, "redirected": 0, "poll_with_auth": 0, "poll_without_auth": 0,
                      "prompt_tokens": 0, "generation_tokens": 0, "e2e_sum": 0.0,
                      "unauthorized": 0, "stop": 0, "tool_calls": 0, "analysis_calls": 0, "sentinel_calls": 0,
                      "compactions": 0}
        self.pending: dict[str, bytes] = {}
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # quiet
                pass

            def _reply(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                return self.headers.get("Authorization", "") == f"Bearer {mock.token}"

            def do_GET(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/v1/models":
                    self._reply(200, json.dumps({"object": "list", "data": [
                        {"id": mock.served_model, "object": "model", "max_model_len": 32768}]}).encode())
                    return
                if path == "/health":
                    self._reply(200, b"ok", "text/plain")
                    return
                if path.startswith("/v1/_poll/"):
                    pid = path.rsplit("/", 1)[-1]
                    with mock.lock:
                        body = mock.pending.pop(pid, None)
                        if self._authed():
                            mock.state["poll_with_auth"] += 1
                        else:
                            mock.state["poll_without_auth"] += 1
                    if body is None:
                        self._reply(404, b'{"error":"unknown poll id"}')
                        return
                    self._reply(200, body)
                    return
                if not self._authed():
                    with mock.lock:
                        mock.state["unauthorized"] += 1
                    self._reply(401, b'{"error":"unauthorized"}')
                    return
                if path == "/metrics":
                    self._reply(200, mock.metrics_text().encode(), "text/plain; version=0.0.4")
                    return
                if path == "/arc3/identity":
                    self._reply(200, json.dumps({
                        "mock": True, "served_model_name": mock.served_model, "profile": mock.profile,
                        "host": {"gpu_rows": [f"0, NVIDIA RTX PRO 6000 Blackwell Server Edition (mock), "
                                              f"97887 MiB, 580.95.05, 12.0"]}}).encode())
                    return
                self._reply(404, b'{"error":"not found"}')

            def do_POST(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if not self._authed():
                    with mock.lock:
                        mock.state["unauthorized"] += 1
                    self._reply(401, b'{"error":"unauthorized"}')
                    return
                if path != "/v1/chat/completions":
                    self._reply(404, b'{"error":"not found"}')
                    return
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._reply(400, b'{"error":"bad json"}')
                    return
                t0 = time.monotonic()
                with mock.lock:
                    mock.state["calls"] += 1
                    k = mock.state["calls"]
                time.sleep(mock.latency_s)
                body = mock.completion(k, payload)
                with mock.lock:
                    mock.state["e2e_sum"] += time.monotonic() - t0
                if k % 3 == 0:
                    pid = f"p{k}"
                    with mock.lock:
                        mock.pending[pid] = body
                        mock.state["redirected"] += 1
                    host = self.headers.get("Host") or f"127.0.0.1:{mock.port}"
                    self._reply(303, b"", "text/plain", {"Location": f"http://{host}/v1/_poll/{pid}"})
                    return
                self._reply(200, body)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="mock-vllm")

    def start(self) -> "MockVLLM":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()

    def completion(self, k: int, payload: dict) -> bytes:
        # deterministic, skewed reasoning length so mean != median
        frac = (k * 0.6180339887) % 1.0
        reasoning = ("Mock reasoning about the board; step %d. " % k) * max(1, int(1 + 90 * frac * frac))
        prompt_tokens = max(1, len(json.dumps(payload.get("messages") or [])) // 4)
        msgs = payload.get("messages") or []
        if (not payload.get("tools") and msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system"
                and str(msgs[0].get("content", "")).startswith(CARRY_COMPACT_HEAD)):
            # a graft_carry compaction request: no tools, the compaction system prompt -> a summary block
            with self.lock:
                self.state["compactions"] += 1
                c = self.state["compactions"]
                self.state["prompt_tokens"] += prompt_tokens
                self.state["generation_tokens"] += 40
            text = (f"MECHANICS VERIFIED\n- {MOCK_COMPACT_SUMMARY} #{c}: the chosen action moved a block one cell.\n"
                    "HYPOTHESES REFUTED\n- clicking the HUD bar did nothing.\nCURRENT PLAN\n- keep cycling the valid actions.")
            body = {"id": f"chatcmpl-mock-{k}", "object": "chat.completion", "model": self.served_model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 40, "total_tokens": prompt_tokens + 40}}
            return json.dumps(body).encode()
        if k % 7 == 3:
            content = "World model: still exploring. Plan: inspect the segmentation next."
            message = {"role": "assistant", "content": content, "reasoning": reasoning}
            finish = "stop"
            with self.lock:
                self.state["stop"] += 1
        else:
            code = MOCK_TOOL_CODE
            pos = mock_turn_position(payload.get("messages") or []) if self.analysis_calls_per_turn else 0
            if self.analysis_calls_per_turn and pos < self.analysis_calls_per_turn:
                code = MOCK_SENTINEL_CODE if pos == self.analysis_calls_per_turn - 1 else MOCK_ANALYSIS_CODE
                with self.lock:
                    self.state["analysis_calls"] += 1
                    self.state["sentinel_calls"] += code is MOCK_SENTINEL_CODE
            message = {"role": "assistant", "content": "Open questions: what does SPACE do; is the bar a timer.",
                       "reasoning": reasoning,
                       "tool_calls": [{"id": f"call-{k}", "type": "function",
                                       "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}
            finish = "tool_calls"
            with self.lock:
                self.state["tool_calls"] += 1
        completion_tokens = len(reasoning) // 4 + 60
        with self.lock:
            self.state["prompt_tokens"] += prompt_tokens
            self.state["generation_tokens"] += completion_tokens
        body = {"id": f"chatcmpl-mock-{k}", "object": "chat.completion", "model": self.served_model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}}
        return json.dumps(body).encode()

    def metrics_text(self) -> str:
        s = dict(self.state)
        lab = f'engine="0",model_name="{self.served_model}"'
        calls = s["calls"]
        lines = [
            f'vllm:request_success_total{{{lab},finished_reason="stop"}} {calls}.0',
            f'vllm:request_success_total{{{lab},finished_reason="length"}} 0.0',
            f'vllm:prompt_tokens_total{{{lab}}} {s["prompt_tokens"]}.0',
            f'vllm:generation_tokens_total{{{lab}}} {s["generation_tokens"]}.0',
            f'vllm:e2e_request_latency_seconds_sum{{{lab}}} {s["e2e_sum"]}',
            f'vllm:e2e_request_latency_seconds_count{{{lab}}} {calls}.0',
            f'vllm:request_queue_time_seconds_sum{{{lab}}} {0.01 * calls}',
            f'vllm:request_queue_time_seconds_count{{{lab}}} {calls}.0',
            f'vllm:request_inference_time_seconds_sum{{{lab}}} {s["e2e_sum"]}',
            f'vllm:request_inference_time_seconds_count{{{lab}}} {calls}.0',
            f'vllm:num_preemptions_total{{{lab}}} 0.0',
            f'vllm:spec_decode_num_drafts_total{{{lab}}} {calls * 100}.0',
            f'vllm:spec_decode_num_draft_tokens_total{{{lab}}} {calls * 300}.0',
            f'vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {calls * 180}.0',
            f'vllm:prefix_cache_queries_total{{{lab}}} 0.0',
            f'vllm:prefix_cache_hits_total{{{lab}}} 0.0',
            f'vllm:num_requests_running{{{lab}}} 0.0',
            f'vllm:num_requests_waiting{{{lab}}} 0.0',
            f'vllm:kv_cache_usage_perc{{{lab}}} 0.0',
            f'mock:redirected_total {s["redirected"]}.0',
            f'mock:poll_with_auth_total {s["poll_with_auth"]}.0',
            f'mock:poll_without_auth_total {s["poll_without_auth"]}.0',
            f'mock:unauthorized_total {s["unauthorized"]}.0',
            f'mock:text_only_replies_total {s["stop"]}.0',
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# the wave
# ---------------------------------------------------------------------------


def load_bundle(out_dir: Path) -> tuple:
    """keith V14 cells 11 + 13: unpickle the deploy target + benchmark, stamp
    the non-submission state, point outputs at out_dir, apply the exact
    public-25 settings."""
    with open(BUNDLE_PKL_DIR / "deploy_target.pkl", "rb") as f:
        target = pickle.load(f)
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    with open(BUNDLE_PKL_DIR / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = out_dir
    if float(getattr(target, "max_runtime_s", 0.0) or 0.0) != 32400.0:
        raise RuntimeError(f"expected the 32400-second notebook budget, got {target.max_runtime_s!r}")
    return bm, target


def apply_geometry(bm, *, per_game_s: float, concurrency: int) -> dict:
    s = bm.solver
    s.max_runtime_s_per_game = float(per_game_s)
    s.analyzer_timeout = float(GEOMETRY["analyzer_timeout"])
    s.concurrency = int(concurrency)
    s.max_actions_per_game = GEOMETRY["max_actions_per_game"]
    s.save_request_logs = GEOMETRY["save_request_logs"]
    return {"max_runtime_s_per_game": s.max_runtime_s_per_game, "analyzer_timeout": s.analyzer_timeout,
            "concurrency": s.concurrency, "max_actions_per_game": s.max_actions_per_game,
            "save_request_logs": s.save_request_logs,
            "matches_public25": (s.max_runtime_s_per_game == GEOMETRY["max_runtime_s_per_game"]
                                 and s.concurrency == GEOMETRY["concurrency"])}


def build_games(game_ids: list[str], env_dir: Path) -> list:
    """keith V14 cell 15 (non-submission branch): OFFLINE arcade over the
    competition environment files, the public ids in order."""
    import arc_agi  # noqa: PLC0415
    import taaf.game_api  # noqa: PLC0415
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                                    environments_dir=str(env_dir))
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
    available = {e.game_id for e in arcade.available_environments}
    missing = sorted(set(PUBLIC_GAME_IDS) - available)
    extra = sorted(available - set(PUBLIC_GAME_IDS))
    if missing or extra:
        raise RuntimeError(f"offline public game set changed; missing={missing}, extra={extra}")
    return [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in game_ids]


def game_rows(bm) -> list[dict]:
    rows = []
    n_games = len(getattr(bm, "games", None) or []) or None
    for index, gr in enumerate(list(getattr(bm, "game_runs", None) or [])):
        hist = getattr(gr, "history", None) or []
        gid = getattr(gr, "game_id", None)
        rows.append({
            "game_id": gid,
            "draw": (index // n_games) if n_games else 0,
            "run_stem": run_stem_for(str(gid), index, n_games),
            "levels_completed": int(getattr(gr, "levels_completed", 0) or 0),
            "number_of_levels": int(getattr(gr, "number_of_levels", 0) or 0),
            "actions": len(hist),
            "actions_per_level": list(getattr(gr, "actions_per_level", None) or []),
            # per-level HUMAN baselines (taaf game_api copies arcengine baseline_actions offline; None in submission mode)
            "baselines": (list(getattr(gr, "base_actions_per_level", None) or []) or None),
            "state": str(getattr(gr, "state", None)),
            "score": getattr(gr, "final_score", None),
            "wallclock_s": getattr(gr, "final_wallclock_seconds", None),
            "solver_note": str(getattr(gr, "solver_note", "") or "")[:200],
            "started_at": str(getattr(gr, "started_at", "") or ""),
        })
    return rows


def score_run(out_dir: Path) -> dict | None:
    """The frozen scorer (keith cell 15): evaluate_runs + save_score_file."""
    try:
        from inference.tools.eval import evaluate_runs, save_score_file  # noqa: PLC0415
        summary = evaluate_runs([out_dir])
        path = save_score_file(summary, run_dirs=[out_dir], output_path=out_dir / "score.json")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[regime] WARN frozen scorer failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return None


def _fmt(v, nd=1, suffix=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def _pct(v):
    return "-" if v is None else f"{100.0 * v:.1f}%"


def render_summary(result: dict, telemetry: dict) -> str:
    agg = telemetry.get("aggregate", {})
    md = (result.get("metrics") or {}).get("delta") or {}
    env = result.get("analyzer_env", {})
    tot = result.get("totals", {})
    rc = agg.get("reasoning_chars_pooled") or {}
    ce = agg.get("client_e2e_s_pooled") or {}
    ct = agg.get("client_completion_tokens_pooled") or {}
    cp = agg.get("client_prompt_tokens_pooled") or {}
    geo = result.get("geometry", {})
    ic = (result.get("endpoint") or {}).get("identity_check") or {"profile": None, "expected_profile": result.get("expect_profile"),
                                                                  "profile_ok": "SKIPPED", "gpu_ok": "SKIPPED", "gpu_rows": None}
    tpot_ms = (md.get("tpot_mean_s") * 1000.0) if md.get("tpot_mean_s") is not None else None
    lines = [
        f"REGIME WAVE  arm={result['arm']}  status={result['status']}  dry_run={result['dry_run']}  "
        f"games={tot.get('games')}  wall={_fmt((result.get('wall_s') or 0) / 3600.0, 2)} h",
        f"  endpoint {result.get('base_url')}  model {result.get('served_model')}",
        f"  IDENTITY profile {ic.get('profile')!r} (expected {ic.get('expected_profile')!r}, ok={ic.get('profile_ok')}) | "
        f"gpu ok={ic.get('gpu_ok')} {ic.get('gpu_rows')}",
        f"  geometry conc {geo.get('concurrency')} | {_fmt(geo.get('max_runtime_s_per_game'), 0)} s/game | "
        f"analyzer_timeout {_fmt(geo.get('analyzer_timeout'), 0)} | max_actions {geo.get('max_actions_per_game')} | "
        f"max_calls {result.get('max_calls')} | wave cap {_fmt(result.get('wave_cap_s'), 0)} s | "
        f"public25 geometry: {geo.get('matches_public25')}",
        f"  stock agent sha {result['stock']['agent_tree_sha256'][:12]} (== june_stock pin) | "
        f"framework {result['stock']['framework_tree_sha256'][:12]} | pkls pinned",
        f"  ARM KNOB  CONTEXT_WINDOW={env.get('LOCAL_ANALYZER_CONTEXT_WINDOW')}  "
        f"MAX_OUTPUT={env.get('LOCAL_ANALYZER_MAX_OUTPUT')}  UPSCALE={env.get('MULTIMODAL_UPSCALE')}"
        f"{' (' + str((result.get('vision') or {}).get('png_px')) + ' px, ~' + str((result.get('vision') or {}).get('vision_tokens_derived')) + ' vision tok/img derived)' if result.get('vision') else ''}"
        f"  (yield {env.get('LOCAL_ANALYZER_YIELD_SECONDS')} s, "
        f"temp {env.get('LOCAL_ANALYZER_TEMPERATURE')}/{env.get('LOCAL_ANALYZER_TOP_P')}/{env.get('LOCAL_ANALYZER_TOP_K')}, "
        f"thinking {env.get('LOCAL_ANALYZER_ENABLE_THINKING')}); harness-reported "
        f"{agg.get('analyzer_status_config_first') or ''}",
        f"  SCORE {_fmt(tot.get('score'), 2)} total | levels {tot.get('levels')} ({_fmt(agg.get('levels_per_game'), 2)}/game) | "
        f"actions {tot.get('actions')} ({_fmt(agg.get('actions_per_game'), 0)}/game) | "
        f"zero-level games {tot.get('zero_level_games')} | states {tot.get('states')}",
        f"  CADENCE calls/game {_fmt(agg.get('calls_per_game'))} | turns/game {_fmt(agg.get('turns_per_game'))} | "
        f"distinct steps/game {_fmt(agg.get('distinct_steps_per_game'))} | actions/call {_fmt(agg.get('actions_per_call'), 2)} | "
        f"calls/turn {_fmt(agg.get('calls_per_turn'), 2)}",
        f"  REASONING chars/call mean {_fmt(rc.get('mean'), 0)} median {_fmt(rc.get('median'), 0)} p90 {_fmt(rc.get('p90'), 0)} "
        f"(n={rc.get('n')}) | no-tool-call share {_pct(agg.get('no_tool_call_share'))} | "
        f"length-finish share {_pct(agg.get('length_finish_share'))} | finish {agg.get('finish_reasons')}",
        f"  TURNS step-executed {_pct(agg.get('step_executed_turn_share'))} | yielded {_pct(agg.get('yielded_turn_share'))} | "
        f"outcomes {agg.get('turn_outcomes')}",
        f"  CLIENT e2e/call mean {_fmt(ce.get('mean'))} s median {_fmt(ce.get('median'))} s p90 {_fmt(ce.get('p90'))} s | "
        f"prompt tok/call {_fmt(cp.get('mean'), 0)} | completion tok/call {_fmt(ct.get('mean'), 0)} | "
        f"redirected posts {agg.get('client_redirected_posts')} | post errors {agg.get('client_post_errors')}",
        f"  VLLM   requests {_fmt(md.get('requests'), 0)} | e2e mean {_fmt(md.get('e2e_mean_s'))} s | queue {_fmt(md.get('queue_mean_s'))} s | "
        f"inference {_fmt(md.get('inference_mean_s'))} s | TPOT {_fmt(tpot_ms)} ms | "
        f"gen tok/req {_fmt(md.get('gen_tokens_per_request'), 0)} | gen tok/s {_fmt(md.get('gen_tokens_per_s'), 0)} | "
        f"MTP accept {_pct(md.get('mtp_acceptance_rate'))} | preemptions {_fmt(md.get('preemptions'), 0)} | "
        f"prefix hit {_pct(md.get('prefix_cache_hit_rate'))} | finished {md.get('request_success_by_reason')}",
        f"  REF keith V14 commit: 55 calls/game, reasoning 3406/2206, 53 turns/game, e2e 142 s, queue 124 s, "
        f"MTP 60%, 1.44 lv/game, 6.76 pts",
    ]
    grafts = result.get("grafts") or {}
    installed = grafts.get("installed") or {}
    if "graft_retry" in installed or agg.get("retries_total"):
        knobs = {k: env.get(k) for k in RETRY_ENV_KEYS if env.get(k) is not None}
        lines.append(
            f"  RETRY  fired {agg.get('retries_total')} ({_fmt(agg.get('retries_per_game'), 2)}/game) | "
            f"levels cleared after retry {agg.get('retry_clears_total')} | games with retry {agg.get('games_with_retry')} | "
            f"grafts {installed} | flags {knobs}")
    eng = agg.get("engagement") or {}
    if "graft_evidence" in installed or "graft_hypo" in installed or agg.get("evid_markers_total") or agg.get("hypo_markers_total"):
        knobs = {k: env.get(k) for k in EVID_ENV_KEYS + HYPO_ENV_KEYS if env.get(k) is not None}
        lines.append(
            f"  AID    [EVID] {agg.get('evid_markers_total')} ({_fmt(agg.get('evid_markers_per_game'), 1)}/run) "
            f"level flags {agg.get('evid_level_flags_total')} | engagement wall {_pct((eng.get('evid_wall') or {}).get('share'))} "
            f"({(eng.get('evid_wall') or {}).get('turns')}/{(eng.get('evid_wall') or {}).get('of')}) all {_pct((eng.get('evid') or {}).get('share'))} || "
            f"[HYPO] {agg.get('hypo_markers_total')} ({_fmt(agg.get('hypo_markers_per_game'), 1)}/run) | "
            f"engagement wall {_pct((eng.get('hypo_wall') or {}).get('share'))} "
            f"({(eng.get('hypo_wall') or {}).get('turns')}/{(eng.get('hypo_wall') or {}).get('of')}) all {_pct((eng.get('hypo') or {}).get('share'))} | "
            f"grafts {installed} | flags {knobs}")
    pb = agg.get("probe") or {}
    if "graft_probe" in installed or pb.get("refusals_total"):
        knobs = {k: env.get(k) for k in PROBE_ENV_KEYS if env.get(k) is not None}
        gate = pb.get("gate") or {}
        ge3, ge3s, aar, afr, gaar = (pb.get("turns_ge3_analysis") or {}), (pb.get("turns_ge3_nonacting") or {}), \
            (pb.get("acting_after_refusal") or {}), (pb.get("acted_after_first_refusal") or {}), (pb.get("graft") or {})
        war = pb.get("wall_actions_ratio") or {}
        pri = pb.get("primary") or {}
        saf = pb.get("safety") or {}
        lines.append(
            f"  PROBE  refusals {pb.get('refusals_total')} ({_fmt(pb.get('refusals_per_game'), 2)}/game; gate >= {PROBE_GATE['refusals_per_game_min']:g}) "
            f"games {pb.get('games_with_refusal')} | acted-after-FIRST-refusal graft {gaar.get('acted_after_first_refusal')}/{gaar.get('first_refusal_followups')} "
            f"({_pct(gaar.get('acted_after_first_refusal_share'))}; gate >= {_pct(PROBE_GATE['acted_after_first_refusal_min'])}; turn-ending refusals "
            f"{gaar.get('refusal_turn_ending')} count as non-acting) transcript {afr.get('acted')}/{afr.get('of')} ({_pct(afr.get('share'))}) | "
            f"after ANY refusal graft {gaar.get('acting_calls_after_refusal')}/{gaar.get('calls_after_refusal')} ({_pct(gaar.get('acting_after_refusal_share'))}) "
            f"transcript {aar.get('acted')}/{aar.get('of')} | wall actions/baseline median {_fmt(war.get('median'), 2)} (n={war.get('n')}, under 1x {war.get('under_1x')}; "
            f"gate >= {PROBE_GATE['wall_actions_ratio_min']:g}, ledger-3 {LEDGER3_REFERENCE['wall_actions_ratio_median']}) | "
            f"ENGAGED = {'YES' if gate.get('engaged') else 'NO'} "
            f"{{{', '.join(k + ('+' if v.get('ok') else '-') for k, v in gate.items() if isinstance(v, dict) and not v.get('secondary'))}}} | "
            f"grafts {installed} | flags {knobs}")
        lines.append(
            f"  PROBE-2ND spans >=3 executed analysis-only (graft) {gaar.get('turns_ge3_analysis')}/{gaar.get('turns_total')} "
            f"({_pct(gaar.get('turns_ge3_analysis_share'))}; target < {_pct(PROBE_GATE['turns_ge3_analysis_max'])}, ledger-3 "
            f"{_pct(LEDGER3_REFERENCE['turns_ge3_analysis_share'])}) leak: cap_lifted {gaar.get('leak_cap_lifted')} dead_branch "
            f"{gaar.get('leak_dead_branch')} unparsable {gaar.get('leak_unparsable')} | transcript turns >=3 A-class {ge3.get('turns')}/{ge3.get('of')} "
            f"({_pct(ge3.get('share'))}) strict incl. refused/errors {_pct(ge3s.get('share'))} | carried turns {gaar.get('carried_turns')} | "
            f"turn_time_budget yields {pb.get('yields_turn_time_budget_total')} ({_fmt(pb.get('yields_per_draw'), 1)}/draw; ledger-3 "
            f"{LEDGER3_REFERENCE['yields_per_draw']}, target < {PROBE_GATE['yields_per_draw_max']}) | NOACT turns {pb.get('noact_turns_total')} "
            f"(notices {pb.get('noact_notices_total')}) | call mix {pb.get('call_types')} analysis share {_pct(pb.get('analysis_call_share'))} "
            f"(ledger-3 {_pct(LEDGER3_REFERENCE['analysis_call_share'])})")
        delta = pri.get("delta_vs_base")
        lines.append(
            f"  PROBE-PRIMARY levels {pri.get('levels_total')} over {pri.get('games')} runs / {pri.get('draws')} draw(s) = "
            f"{_fmt(pri.get('levels_per_draw'), 1)}/draw vs pooled six-draw base {pri.get('base_levels_mean')} (sd {pri.get('base_levels_sd')}) "
            f"-> delta {('%+.1f' % delta) if delta is not None else '-'} "
            f"({('%+.2f' % (delta / pri['base_levels_sd'])) if delta is not None else '-'} sd; 25-game waves only; >= 52 step candidate, 45-51 redraw, < 45 dead) | "
            f"never-passed walls {pri.get('walls_passed_n')}/{pri.get('walls_total')} passed {pri.get('walls_passed')} "
            f"(present in wave: {len(pri.get('walls_present') or [])}; co-primary target >= 3)")
        lines.append(
            f"  PROBE-SAFETY GAME_OVERs {saf.get('game_overs_total')} ({_fmt(saf.get('game_overs_per_run'), 2)}/run over {saf.get('game_overs_runs')} runs; "
            f"yield900 base {saf.get('base_game_overs_per_run')}) | live-cap score {_fmt(saf.get('live_cap_score_total'), 2)} "
            f"({_fmt(saf.get('live_cap_score_per_game'), 2)}/game over {saf.get('live_cap_runs')} runs; yield900 base {saf.get('base_live_cap_score_per_game')}/game)")
    ca = agg.get("carry") or {}
    if "graft_carry" in installed or ca.get("compactions_total") or ca.get("calls_marked"):
        knobs = {k: env.get(k) for k in CARRY_ENV_KEYS if env.get(k) is not None}
        gate = ca.get("gate") or {}
        pri = (agg.get("probe") or {}).get("primary") or {}
        saf = (agg.get("probe") or {}).get("safety") or {}
        fcb = ca.get("first_call_with_block") or {}
        per_run = {stem: f"{(g.get('carry') or {}).get('compactions', 0)}/{(g.get('carry') or {}).get('compaction_failures', 0)}"
                   for stem, g in sorted(telemetry.get("per_game", {}).items())}
        lines.append(
            f"  CARRY  compactions {ca.get('compactions_total')} ({_fmt(ca.get('compactions_per_game'), 2)}/game; gate >= "
            f"{CARRY_GATE['compactions_per_game_min']:g}) in {ca.get('games_with_compaction')} runs | failures {ca.get('compaction_failures_total')} "
            f"({_pct(ca.get('failure_share'))} of {ca.get('compaction_attempts')} attempts; gate <= {_pct(CARRY_GATE['failure_share_max'])}) | "
            f"per compaction: dropped msgs {_fmt(ca.get('dropped_msgs_per_compaction'), 1)}, input chars {_fmt(ca.get('input_chars_per_compaction'), 0)}, "
            f"e2e {_fmt(ca.get('compaction_e2e_s_mean'))} s, prompt tok {_fmt(ca.get('compaction_prompt_tokens_mean'), 0)}, "
            f"completion tok {_fmt(ca.get('compaction_completion_tokens_mean'), 0)} | block chars mean {_fmt(ca.get('summary_chars_mean'), 0)} "
            f"(cap {knobs.get('CARRY_SUMMARY_CHARS')}) | ENGAGED = {'YES' if gate.get('engaged') else 'NO'} "
            f"{{{', '.join(k + ('+' if v.get('ok') else '-') for k, v in gate.items() if isinstance(v, dict))}}} | "
            f"grafts {installed} | flags {knobs}")
        lines.append(
            f"  CARRY-WINDOW prompt tok/call (server usage via markers) mean {_fmt(ca.get('prompt_tokens_mean'), 0)} max {ca.get('prompt_tokens_max')} "
            f"| over {CARRY_WINDOW_TOKENS}: {ca.get('prompt_over_window')} (gate 0) | calls carrying the block {ca.get('calls_with_summary')}/{ca.get('calls_marked')} "
            f"({_pct(ca.get('calls_with_summary_share'))}; gate >= {_pct(CARRY_GATE['calls_with_summary_share_min'])}) | first block at call "
            f"#{_fmt(fcb.get('median'), 0)} (median over {fcb.get('n')} runs) | reasoning carried per request: {_fmt(ca.get('reasoning_msgs_per_call'), 2)} msgs, "
            f"{_fmt(ca.get('reasoning_chars_per_call'), 0)} chars (stock behaviour, measured) | compactions/failures per run {per_run}")
        delta = pri.get("delta_vs_base")
        lines.append(
            f"  CARRY-PRIMARY levels {pri.get('levels_total')} over {pri.get('games')} runs / {pri.get('draws')} draw(s) = "
            f"{_fmt(pri.get('levels_per_draw'), 1)}/draw vs pooled six-draw base {pri.get('base_levels_mean')} (sd {pri.get('base_levels_sd')}) "
            f"-> delta {('%+.1f' % delta) if delta is not None else '-'} "
            f"({('%+.2f' % (delta / pri['base_levels_sd'])) if delta is not None else '-'} sd; 25-game waves only; >= 48 step candidate, 45-47 redraw, <= 44 dead) | "
            f"never-passed walls {pri.get('walls_passed_n')}/{pri.get('walls_total')} passed {pri.get('walls_passed')} "
            f"(present in wave: {len(pri.get('walls_present') or [])}; co-primary target >= 3)")
        lines.append(
            f"  CARRY-SAFETY GAME_OVERs {saf.get('game_overs_total')} ({_fmt(saf.get('game_overs_per_run'), 2)}/run over {saf.get('game_overs_runs')} runs; "
            f"yield900 base {saf.get('base_game_overs_per_run')}) | live-cap score {_fmt(saf.get('live_cap_score_total'), 2)} "
            f"({_fmt(saf.get('live_cap_score_per_game'), 2)}/game over {saf.get('live_cap_runs')} runs; yield900 base {saf.get('base_live_cap_score_per_game')}/game) | "
            f"fit-the-clock: calls/game {_fmt(agg.get('calls_per_game'))} x e2e {_fmt(ce.get('mean'))} s = "
            f"{_fmt((agg.get('calls_per_game') or 0) * (ce.get('mean') or 0), 0)} s (+ compactions) vs 7920 s")
    w = agg.get("wall") or {}
    if w:
        per_l2 = {stem: (g.get("wall") or {}).get("calls_at_l2") for stem, g in sorted(telemetry.get("per_game", {}).items())}
        voids = {stem: (g.get("wall") or {}).get("void_reasons") for stem, g in sorted(telemetry.get("per_game", {}).items())
                 if (g.get("wall") or {}).get("void")}
        lines.append(
            f"  WALL   attempts {w.get('attempts')}/{w.get('runs')} (L2 reached {w.get('reached_l2')}; attempt = L2 with "
            f">= {WALL_MIN_CALLS_LEFT_AT_L2} calls left) | passes {w.get('passes')} (level 3 reached) | void {w.get('void')} {voids} | "
            f"calls@L2 {per_l2} | UPTAKE wall {_pct((w.get('uptake_wall') or {}).get('share'))} "
            f"({(w.get('uptake_wall') or {}).get('turns')}/{(w.get('uptake_wall') or {}).get('of')}) all {_pct((w.get('uptake') or {}).get('share'))}")
    fc = agg.get("first_call_prompt_tokens") or {}
    if fc.get("n"):
        delta = fc["mean"] - KEITH_FIRST_CALL_PROMPT_TOKENS
        lines.append(
            f"  FIRST-CALL prompt_tokens (n_messages==2) mean {_fmt(fc.get('mean'), 0)} n={fc.get('n')} "
            f"[{_fmt(min(fc.get('mean'), fc.get('median')), 0)}..{_fmt(fc.get('max'), 0)}] | keith baseline ~{KEITH_FIRST_CALL_PROMPT_TOKENS} | "
            f"delta {delta:+.0f} | UPSCALE={env.get('MULTIMODAL_UPSCALE')}: up8 expects ≈ +{UP8_EXPECTED_FIRST_CALL_DELTA} "
            f"(+0 => processor downscaled = no-op; >> => geometry differs)")
    if (agg.get("draws") or 1) > 1:
        per_draw = {gid: [d.get("levels_completed") for _, d in sorted(v.items(), key=lambda kv: int(kv[0]))]
                    for gid, v in (telemetry.get("per_game_draw") or {}).items()}
        lines.append(f"  DRAWS {agg.get('draws')} | levels per (game, draw): {per_draw}")
    if result.get("knob_overrides"):
        lines.append(f"  KNOB OVERRIDES (not the pinned arm env): {result['knob_overrides']}")
    lines.append("  run               lv/n    act  calls turns reas_mean  len%  notool%  e2e_s   score  retry probe wall/b  evid  hypo  wall%  upt%  c@L2 att void  state")
    for stem, g in sorted(telemetry.get("per_game", {}).items()):
        r = g.get("reasoning_chars") or {}
        c = (g.get("client") or {}).get("e2e_s") or {}
        e = g.get("engagement") or {}
        p = g.get("probe") or {}
        retry_col = f"{g.get('retries_fired') or 0}/{g.get('retry_clears') or 0}"
        probe_col = f"{p.get('refusals') or 0}/{p.get('noact_turns') or 0}"
        wall_share = (e.get("evid_wall") or {}).get("share") if g.get("evid_markers") else (e.get("hypo_wall") or {}).get("share")
        lines.append(
            f"  {stem:17s} {str(g.get('levels_completed')) + '/' + str(g.get('number_of_levels')):>5} "
            f"{_fmt(g.get('actions_total')):>6} {g.get('calls'):>5} {g.get('turns'):>5} "
            f"{_fmt(r.get('mean'), 0):>9} {_pct(g.get('length_finish_share')):>6} "
            f"{_pct(g.get('no_tool_call_share')):>7} {_fmt(c.get('mean')):>6} {_fmt(g.get('score'), 2):>7} "
            f"{retry_col:>6} {probe_col:>5} {_fmt(p.get('wall_actions_ratio'), 2):>6} "
            f"{g.get('evid_markers') or 0:>5} {g.get('hypo_markers') or 0:>5} {_pct(wall_share):>6} "
            f"{_pct(((g.get('wall') or {}).get('uptake_wall') or {}).get('share')):>5} "
            f"{str((g.get('wall') or {}).get('calls_at_l2', '-')):>5} {'Y' if (g.get('wall') or {}).get('attempt') else '-':>3} "
            f"{'VOID' if (g.get('wall') or {}).get('void') else '-':>4}  "
            f"{(result.get('states') or {}).get(stem, '')}")
    return "\n".join(lines) + "\n"


class Wave:
    def __init__(self, *, arm: str, base_url: str, token: str, out_dir: Path, game_ids: list[str],
                 per_game_s: float, concurrency: int, wave_cap_s: float, dry_run: bool,
                 progress_every_s: float, draws: int = 1, max_calls: int | None = None) -> None:
        self.arm, self.base_url, self.token, self.out_dir = arm, base_url, token, out_dir
        self.game_ids, self.per_game_s, self.concurrency = game_ids, per_game_s, concurrency
        self.wave_cap_s, self.dry_run, self.progress_every_s = wave_cap_s, dry_run, progress_every_s
        self.draws = max(1, int(draws))
        self.max_calls = max_calls
        self.result: dict = {"schema_version": 3, "max_calls": max_calls, "arm": arm, "dry_run": dry_run, "status": "init",
                             "base_url": base_url, "served_model": SERVED_MODEL_NAME,
                             "wave_cap_s": wave_cap_s, "game_ids": list(game_ids), "draws": self.draws,
                             "grafts": {"installed": {}, "status": {}}}
        self.bm = None
        self.target = None
        self.shim = RequestShim(out_dir / "requests_shim.jsonl")
        self._sigints = 0
        self._stop_progress = threading.Event()
        self.t0 = time.time()

    # -- setup --------------------------------------------------------------
    def setup(self, recorded_env: dict, stock: dict, endpoint: dict | None) -> None:
        self.result.update({"analyzer_env": recorded_env, "stock": stock, "endpoint": endpoint,
                            "started_at": datetime.now().isoformat(timespec="seconds")})
        bm, target = load_bundle(self.out_dir)
        geometry = apply_geometry(bm, per_game_s=self.per_game_s, concurrency=self.concurrency)
        bm.games = build_games(self.game_ids, ENV_FILES_DIR)
        bm.n_passes = self.draws                       # --draws: N independent runs per game (<gid>_p<draw>)
        bm.game_weights = None
        bm.solver.analyzer_factory = make_tagging_analyzer_factory(bm.solver, n_games=len(bm.games))
        self.bm, self.target = bm, target
        self.result["geometry"] = geometry
        self.result["solver_label"] = bm.solver.label
        try:
            self.result["vision"] = vision_image_facts()
        except Exception as exc:  # noqa: BLE001
            self.result["vision"] = {"error": f"{type(exc).__name__}: {exc}"}
        (self.out_dir / "arm_env.json").write_text(json.dumps(recorded_env, indent=1, sort_keys=True) + "\n")
        self._dump()
        print(f"[regime] arm={self.arm} games={len(self.game_ids)} draws={self.draws} geometry={geometry} "
              f"vision={self.result['vision']}", flush=True)

    def _dump(self) -> None:
        self.result["elapsed_s"] = round(time.time() - self.t0, 1)
        (self.out_dir / "results.json").write_text(json.dumps(self.result, indent=1, default=str, sort_keys=True) + "\n")

    # -- progress -----------------------------------------------------------
    def _progress_loop(self) -> None:
        while not self._stop_progress.wait(self.progress_every_s):
            rows = game_rows(self.bm) if self.bm is not None else []
            done = sum(1 for r in rows if r["score"] is not None)
            lv = sum(r["levels_completed"] for r in rows)
            acts = sum(r["actions"] for r in rows)
            print(f"[regime] +{(time.time() - self.t0) / 60:.1f} min: calls={self.shim.count} "
                  f"errors={self.shim.errors} redirected={self.shim.redirected} games_done={done}/{len(rows)} "
                  f"levels={lv} actions={acts}", flush=True)

    # -- run ----------------------------------------------------------------
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()

        def on_sigint():
            self._sigints += 1
            if self._sigints == 1:
                print("\n[regime] Ctrl-C: graceful stop requested (games -> cancelled, partial results "
                      "will be written; press again to hard-exit)", flush=True)
                self.result["status"] = "cancelled"
                self.bm.request_stop()
            else:
                print("\n[regime] second Ctrl-C: hard exit", flush=True)
                self._finish(status="cancelled")
                os._exit(130)
        try:
            loop.add_signal_handler(signal.SIGINT, on_sigint)
        except (NotImplementedError, RuntimeError):
            pass
        soft_end = datetime.now() + timedelta(seconds=self.wave_cap_s)
        self.result["soft_end_time"] = soft_end.isoformat(timespec="seconds")
        self.result["status"] = "running"
        self._dump()
        try:
            await self.bm.run(soft_end_time=soft_end, runtime_environment=self.target, minimal_diagnostics=True)
            if getattr(self.bm, "_deadline_fired", False):
                self.result["status"] = "deadline"
            elif self.result["status"] == "running":
                self.result["status"] = "done"
        except asyncio.CancelledError:
            self.result["status"] = "cancelled"

    def run(self) -> str:
        self.shim.install()
        before = fetch_metrics(self.base_url, self.token)
        (self.out_dir / "metrics_before.prom").write_text(before or "")
        self.result["metrics"] = {"before": summarize_metrics(before)}
        progress = threading.Thread(target=self._progress_loop, daemon=True, name="regime-progress")
        progress.start()
        t_wave = time.time()
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            self.result["status"] = "cancelled"
        except Exception as exc:  # noqa: BLE001
            self.result["status"] = "error"
            self.result["error"] = f"{type(exc).__name__}: {exc}"
            import traceback  # noqa: PLC0415
            traceback.print_exc()
        finally:
            self.result["wave_s"] = round(time.time() - t_wave, 1)
            self._stop_progress.set()
            self._finish(status=self.result["status"])
        return self.result["status"]

    def _finish(self, *, status: str) -> None:
        if self.result.get("finished"):
            return
        self.result["finished"] = True
        self.result["status"] = status
        self.result["ended_at"] = datetime.now().isoformat(timespec="seconds")
        self.result["wall_s"] = round(time.time() - self.t0, 1)
        try:
            self.bm._save_json()          # keith cell 15: explicit save after a minimal-diagnostics run
        except Exception as exc:  # noqa: BLE001
            print(f"[regime] WARN benchmark.json save failed: {exc!r}", flush=True)
        after = fetch_metrics(self.base_url, self.token)
        (self.out_dir / "metrics_after.prom").write_text(after or "")
        self.shim.uninstall()
        rows = game_rows(self.bm)
        score = score_run(self.out_dir)
        if score is not None:
            per = score.get("games", {})
            for r in rows:
                entry = per.get(r["game_id"])
                if not isinstance(entry, dict):
                    continue
                trials = entry.get("trial_scores") or {}
                if self.draws > 1:
                    # the frozen scorer names passes "<run>/pass-<k>"; take this draw's own score
                    match = [v for k, v in trials.items() if str(k).endswith(f"/pass-{r['draw']}")]
                    if match:
                        r["score"] = match[0]
                else:
                    r["score"] = entry.get("score", r["score"])
        self.result["games"] = rows
        self.result["states"] = {r["run_stem"]: r["state"] for r in rows}
        self.result["states_by_game"] = {r["game_id"]: r["state"] for r in rows if r["draw"] == 0}
        self.result["totals"] = {
            "games": len(rows),
            "distinct_games": len({r["game_id"] for r in rows}),
            "draws": self.draws,
            "score": (score or {}).get("score", sum((r["score"] or 0) for r in rows)),
            "score_source": "score.json" if score else "game_run.final_score",
            "levels": sum(r["levels_completed"] for r in rows),
            "actions": sum(r["actions"] for r in rows),
            "zero_level_games": sum(1 for r in rows if r["levels_completed"] == 0),
            "states": {s: sum(1 for r in rows if r["state"] == s) for s in sorted({r["state"] for r in rows})},
        }
        self.result["metrics"]["after"] = summarize_metrics(after)
        self.result["metrics"]["delta"] = metrics_delta(self.result["metrics"].get("before"),
                                                        self.result["metrics"]["after"],
                                                        self.result.get("wave_s"))
        self.result["shim"] = {"posts": self.shim.count, "errors": self.shim.errors,
                               "redirected": self.shim.redirected, "path": str(self.shim.path)}
        shim_records = load_shim_records(self.shim.path)
        md = self.result["metrics"].get("delta") or {}
        self.result["grafts"]["status"] = graft_status(self.arm)   # the graft's own counters (retry_log, skips, ...)
        probe_per_game = (self.result["grafts"]["status"].get("graft_probe") or {}).get("per_game")
        carry_per_game = (self.result["grafts"]["status"].get("graft_carry") or {}).get("per_game")
        telemetry = build_telemetry(self.out_dir / "transcripts", rows, shim_records,
                                    calls_budget=self.max_calls, wave_preemptions=md.get("preemptions"),
                                    graft_per_game=probe_per_game, carry_per_game=carry_per_game)
        self.result["max_calls_stops"] = dict(_MAX_CALLS.get("stops") or {})
        self.result["concurrency_override"] = self.concurrency != GEOMETRY["concurrency"]
        telemetry["arm"] = self.arm
        telemetry["dry_run"] = self.dry_run
        telemetry["grafts"] = self.result["grafts"]
        cfgs = [g.get("analyzer_status_config") for g in telemetry["per_game"].values() if g.get("analyzer_status_config")]
        telemetry["aggregate"]["analyzer_status_config_first"] = cfgs[0] if cfgs else None
        (self.out_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=1, default=str) + "\n")
        self._dump()
        summary = render_summary(self.result, telemetry)
        (self.out_dir / "summary.txt").write_text(summary)
        print("\n" + summary, flush=True)
        print(f"[regime] artifacts: {self.out_dir}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def read_token(args) -> str:
    if args.dry_run:
        return "dry-run-token"
    tok = (args.token or os.environ.get("ARC3_VLLM_TOKEN", "") or "").strip()
    if not tok and Path(args.token_file).expanduser().is_file():
        tok = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit(f"no bearer token: pass --token, set $ARC3_VLLM_TOKEN, or create {args.token_file}")
    return tok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, default="keith")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL incl. /v1")
    p.add_argument("--games", default="all", help="all | comma list of ids or stems (tu93,ft09)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results root; a <ts>-regime-<arm> dir is created")
    p.add_argument("--token", default=None, help="bearer token value (prefer --token-file / $ARC3_VLLM_TOKEN)")
    p.add_argument("--token-file", default=str(TOKEN_FILE))
    p.add_argument("--per-game-s", type=float, default=None,
                   help=f"per-game runtime cap in seconds (default {GEOMETRY['max_runtime_s_per_game']:.0f} = the public "
                        f"geometry; dry-run {DRY_RUN_PER_GAME_S:.0f}); e.g. 1500 for the 3-wall turn-capped instrument")
    p.add_argument("--draws", type=int, default=1,
                   help="play the selected games N times as independent runs in this wave (taaf n_passes; "
                        "run stems <gid>_p0.._p<N-1>; default 1)")
    p.add_argument("--concurrency", type=int, default=GEOMETRY["concurrency"],
                   help="solver concurrency override (public geometry 28; the 3-wall instrument uses 3); recorded "
                        "in results.json:geometry and concurrency_override")
    p.add_argument("--max-calls", type=int, default=None,
                   help="stop each run after N analyzer calls (in-memory wrapper on ToolAgent._chat_completion; "
                        "the run then ends through its own runtime cap, state gave_up); recorded in results.json")
    p.add_argument("--wave-cap-s", type=float, default=None,
                   help=f"whole-wave cap (default {WAVE_CAP_S:.0f}; dry-run {DRY_RUN_WAVE_CAP_S:.0f})")
    p.add_argument("--dry-run", action="store_true", help="loopback mock vLLM; no network")
    p.add_argument("--mock-latency-s", type=float, default=0.25)
    p.add_argument("--skip-preflight", action="store_true",
                   help="also skips the serving-profile/GPU identity gate (recorded as skipped)")
    p.add_argument("--expect-profile", default=DEFAULT_EXPECT_PROFILE,
                   help=f"serving profile /arc3/identity must report (default {DEFAULT_EXPECT_PROFILE}); "
                        "the run refuses to start on any other profile")
    p.add_argument("--mock-profile", default=None,
                   help="dry-run only: the profile the mock identity reports (to exercise the gate)")
    p.add_argument("--mock-analysis-calls", type=int, default=None,
                   help="dry-run only: the mock answers the first N python calls of every turn with analysis-only "
                        "snippets (the Nth prints a sentinel) so graft_probe's refusal is exercised; default 3 for an "
                        "arm that installs graft_probe, else 0 (recorded in results.json:mock_analysis_calls)")
    p.add_argument("--preflight-timeout", type=float, default=2400.0)
    p.add_argument("--progress-every", type=float, default=120.0)
    p.add_argument("--knob", action="append", default=None, metavar="KEY=VALUE",
                   help="override one analyzer/graft env key (repeatable; recorded in results.json as "
                        "knob_overrides — the run is then NOT the pinned arm env)")
    args = p.parse_args(argv)
    knobs = parse_knobs(args.knob)
    if args.draws < 1:
        p.error("--draws must be >= 1")
    if args.max_calls is not None and args.max_calls < 1:
        p.error("--max-calls must be >= 1")

    per_game_s = args.per_game_s if args.per_game_s is not None else (
        DRY_RUN_PER_GAME_S if args.dry_run else GEOMETRY["max_runtime_s_per_game"])
    wave_cap_s = args.wave_cap_s if args.wave_cap_s is not None else (
        DRY_RUN_WAVE_CAP_S if args.dry_run else WAVE_CAP_S)
    game_ids = resolve_game_ids(args.games)
    token = read_token(args)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / f"{ts}-regime-{args.arm}{'-dry' if args.dry_run else ''}"
    out_dir.mkdir(parents=True, exist_ok=False)

    mock = None
    mock_analysis_calls = 0
    if args.dry_run:
        mock_analysis_calls = args.mock_analysis_calls if args.mock_analysis_calls is not None else (
            3 if "graft_probe" in ARM_GRAFTS.get(args.arm, ()) else 0)
        mock = MockVLLM(token, latency_s=args.mock_latency_s,
                        profile=args.mock_profile or args.expect_profile,
                        analysis_calls_per_turn=mock_analysis_calls).start()
        base_url = mock.base_url
        print(f"[regime] DRY RUN: mock vLLM on {base_url} (303 legs every 3rd call, text-only every 7th"
              f"{', first ' + str(mock_analysis_calls) + ' python calls per turn analysis-only' if mock_analysis_calls else ''})",
              flush=True)
    else:
        base_url = args.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            p.error(f"--base-url must end in /v1, got {args.base_url!r}")

    stock = assert_stock_tree()
    print(f"[regime] stock agent tree sha {stock['agent_tree_sha256'][:16]}… == june_stock pin "
          f"({stock['agent_files']} files); framework {stock['framework_tree_sha256'][:16]}…", flush=True)
    recorded_env = install_env(args.arm, base_url, token, out_dir, knobs)
    install_paths()
    verify_imports()
    print(f"[regime] arm {args.arm}: CONTEXT_WINDOW={recorded_env['LOCAL_ANALYZER_CONTEXT_WINDOW']} "
          f"MAX_OUTPUT={recorded_env['LOCAL_ANALYZER_MAX_OUTPUT']} (import-time constants verified)", flush=True)
    if knobs:
        print(f"[regime] KNOB OVERRIDES {knobs} — this run is not the pinned {args.arm} env", flush=True)
    grafts = install_grafts(args.arm)          # in memory only; the stock tree sha above still holds
    if args.max_calls is not None:
        install_max_calls(args.max_calls)      # in memory only
        print(f"[regime] --max-calls {args.max_calls}: each run stops after {args.max_calls} analyzer calls", flush=True)
    if grafts:
        flags = {k: recorded_env.get(k) for k in GRAFT_FLAG_KEYS if recorded_env.get(k) is not None}
        print(f"[regime] grafts installed: {grafts} (flags {flags})", flush=True)

    endpoint = None
    if not args.skip_preflight:
        endpoint = preflight(base_url, token, args.preflight_timeout, expected_profile=args.expect_profile)
    else:
        print(f"[regime] WARNING --skip-preflight: serving profile NOT attested (expected {args.expect_profile!r})",
              flush=True)
        endpoint = {"identity": None, "identity_check": {"skipped": True, "expected_profile": args.expect_profile,
                                                          "profile": None, "profile_ok": "SKIPPED", "gpu_ok": "SKIPPED",
                                                          "gpu_rows": None}}

    wave = Wave(arm=args.arm, base_url=base_url, token=token, out_dir=out_dir, game_ids=game_ids,
                per_game_s=per_game_s, concurrency=args.concurrency, wave_cap_s=wave_cap_s,
                dry_run=args.dry_run, progress_every_s=args.progress_every, draws=args.draws,
                max_calls=args.max_calls)
    wave.result["grafts"]["installed"] = grafts
    wave.result["expect_profile"] = args.expect_profile
    if args.dry_run:
        wave.result["mock_analysis_calls"] = mock_analysis_calls
    if knobs:
        wave.result["knob_overrides"] = knobs
    wave.setup(recorded_env, stock, endpoint)
    status = wave.run()
    if mock is not None:
        wave.result["mock_state"] = dict(mock.state)
        wave._dump()
        print(f"[regime] mock state: {mock.state}", flush=True)
        mock.stop()
    rc = 0 if status in ("done", "deadline") else (2 if status == "error" else 130)
    sys.stdout.flush()
    sys.stderr.flush()
    # Worker threads may still be blocked in an in-flight request (up to the
    # 900 s analyzer timeout) after a cancel; everything is written, so exit now.
    alive = [t for t in threading.enumerate() if t is not threading.current_thread() and not t.daemon]
    if alive:
        os._exit(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
