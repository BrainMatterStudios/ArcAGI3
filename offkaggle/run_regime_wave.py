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
    (everything else in the analyzer env is identical: sampling 0.6/0.95/20,
    thinking on, 60 s yield, tool steps unlimited, multimodal current_grid x4).

Nothing in the agent is edited or monkey-patched. Two hooks live OUTSIDE it:
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
ARM_ENV = {"keith": KEITH_ANALYZER_ENV, "flight": FLIGHT_ANALYZER_ENV, "keith_yield180": KEITH_YIELD180_ENV}
ARMS = tuple(ARM_ENV)
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


def install_env(arm: str, base_url: str, token: str, out_dir: Path) -> dict[str, str]:
    """Process env BEFORE any harness import (tool_agent reads its
    _LOCAL_ANALYZER_* constants at import time; ONLY_RESET_LEVELS must precede
    arcengine). Returns the recorded (token-free) env."""
    for key in [k for k in os.environ if k.startswith("TAAF_")]:
        del os.environ[key]              # clean slate; keith's TAAF_VLLM_* are serving-side
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(key, None)        # tool_agent._headers fallback chain — only ours
    os.environ.update(PROCESS_ENV)
    env = arm_analyzer_env(arm)
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


# ---------------------------------------------------------------------------
# hook 1: analyzer factory (stock ToolAgent, exactly _make_analyzer's args) + game tag
# ---------------------------------------------------------------------------

_GAME_TAG = threading.local()


def current_game_tag() -> str | None:
    return getattr(_GAME_TAG, "game_id", None)


def make_tagging_analyzer_factory(solver):
    """HarnessSolver.analyzer_factory hook: returns the stock ToolAgent built
    with the same arguments HarnessSolver._make_analyzer uses when no local
    server is started (model=self.model, timeout=self.analyzer_timeout,
    save_request_logs=self.save_request_logs, api_key=None, base_url=None,
    provider=None) and tags the calling (game) thread with the game id."""
    from inference.agent.tool_agent import ToolAgent  # noqa: PLC0415

    def factory(game, index):
        run = getattr(game, "game_run", None)
        _GAME_TAG.game_id = getattr(run, "game_id", None) or f"index-{index}"
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
            rec = {"t": time.time(), "game_id": current_game_tag(),
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


def preflight(base_url: str, token: str, deadline_s: float) -> dict:
    """/v1/models unauthenticated (the exempt route; also wakes a scaled-to-zero
    container), /metrics with the bearer (proves auth + the telemetry route),
    one tiny non-thinking completion (proves the chat route end to end)."""
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
    try:
        ri = requests.get(f"{_root_url(base)}/arc3/identity", headers=hdr, timeout=120)
        if ri.status_code == 200:
            identity = ri.json()
    except Exception:  # noqa: BLE001
        identity = None
    return {"models": models, "identity": identity, "preflight_redirect_legs": len(r.history)}


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
            r"ANALYZER STATUS|TOOL CALL: [^\]\n]+|TOOL RESULT: [^\]\n]+)\]")
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
    status_cfg: dict = {}
    for i, (start, end, m) in enumerate(events):
        body_end = events[i + 1][0] if i + 1 < len(events) else len(text)
        body = text[end:body_end]
        if m.group("step") is not None:
            turns.append({"step": int(m.group("step")), "action": int(m.group("action")),
                          "time": m.group("time"), "calls": 0, "status_messages": [],
                          "step_executed": False, "outcome": "incomplete"})
            continue
        label = m.group("label")
        if label == "MODEL RESPONSE META":
            call = {"turn_index": len(turns) - 1, "finish_reason": None, "tool_call_count": 0,
                    "content_chars": 0, "reasoning_chars_meta": 0, "reasoning_chars": 0}
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
    return {"turns": turns, "calls": calls, "analyzer_status_config": status_cfg}


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "sum": 0.0}
    vs = sorted(values)
    return {"n": len(vs), "mean": statistics.fmean(vs), "median": statistics.median(vs),
            "p90": vs[min(len(vs) - 1, int(round(0.9 * (len(vs) - 1))))], "max": vs[-1], "sum": sum(vs)}


def telemetry_for_game(transcript_text: str, *, actions_total: int | None = None,
                       shim_records: list[dict] | None = None,
                       wallclock_s: float | None = None) -> dict:
    parsed = parse_transcript(transcript_text)
    calls, turns = parsed["calls"], parsed["turns"]
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
    }
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
    }


def build_telemetry(transcripts_dir: Path, game_rows: list[dict], shim_records: list[dict]) -> dict:
    """Per-game + pooled telemetry from <out>/transcripts/<gid>_p0.txt, the
    benchmark rows (actions, wallclock) and the client shim records."""
    by_game_shim: dict[str, list[dict]] = {}
    for r in shim_records:
        by_game_shim.setdefault(str(r.get("game_id")), []).append(r)
    per_game: dict[str, dict] = {}
    pooled_reasoning: list[float] = []
    pooled_e2e: list[float] = []
    pooled_prompt: list[float] = []
    pooled_completion: list[float] = []
    rows_by_id = {r["game_id"]: r for r in game_rows}
    for path in sorted(Path(transcripts_dir).glob("*_p0.txt")):
        gid = path.name[:-len("_p0.txt")]
        row = rows_by_id.get(gid, {})
        text = path.read_text(encoding="utf-8", errors="replace")
        recs = by_game_shim.get(gid, [])
        tel = telemetry_for_game(text, actions_total=row.get("actions"),
                                 shim_records=recs, wallclock_s=row.get("wallclock_s"))
        tel["levels_completed"] = row.get("levels_completed")
        tel["number_of_levels"] = row.get("number_of_levels")
        tel["score"] = row.get("score")
        per_game[gid] = tel
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
    return {"definitions": TELEMETRY_DEFINITIONS, "reference_keith_commit_run": KEITH_COMMIT_REFERENCE,
            "aggregate": agg, "per_game": per_game}


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


class MockVLLM:
    """Loopback OpenAI-compatible server for --dry-run. Serves /v1/models,
    /health, /metrics (vLLM-shaped counters), /arc3/identity and
    /v1/chat/completions. Every 3rd completion answers 303 -> /v1/_poll/<id>
    (Modal's long-request continuation) so the STOCK client's redirect
    behaviour is exercised; every 7th reply is text-only (no tool call) to
    exercise the harness's follow-up path; the rest call `python` with a
    real `action(...)`."""

    def __init__(self, token: str, served_model: str = SERVED_MODEL_NAME, latency_s: float = 0.25) -> None:
        import http.server  # noqa: PLC0415
        self.token = token
        self.served_model = served_model
        self.latency_s = latency_s
        self.lock = threading.Lock()
        self.state = {"calls": 0, "redirected": 0, "poll_with_auth": 0, "poll_without_auth": 0,
                      "prompt_tokens": 0, "generation_tokens": 0, "e2e_sum": 0.0,
                      "unauthorized": 0, "stop": 0, "tool_calls": 0}
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
                    self._reply(200, json.dumps({"mock": True, "served_model_name": mock.served_model}).encode())
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
        if k % 7 == 3:
            content = "World model: still exploring. Plan: inspect the segmentation next."
            message = {"role": "assistant", "content": content, "reasoning": reasoning}
            finish = "stop"
            with self.lock:
                self.state["stop"] += 1
        else:
            message = {"role": "assistant", "content": None, "reasoning": reasoning,
                       "tool_calls": [{"id": f"call-{k}", "type": "function",
                                       "function": {"name": "python", "arguments": json.dumps({"code": MOCK_TOOL_CODE})}}]}
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
    for gr in list(getattr(bm, "game_runs", None) or []):
        hist = getattr(gr, "history", None) or []
        rows.append({
            "game_id": getattr(gr, "game_id", None),
            "levels_completed": int(getattr(gr, "levels_completed", 0) or 0),
            "number_of_levels": int(getattr(gr, "number_of_levels", 0) or 0),
            "actions": len(hist),
            "actions_per_level": list(getattr(gr, "actions_per_level", None) or []),
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
    tpot_ms = (md.get("tpot_mean_s") * 1000.0) if md.get("tpot_mean_s") is not None else None
    lines = [
        f"REGIME WAVE  arm={result['arm']}  status={result['status']}  dry_run={result['dry_run']}  "
        f"games={tot.get('games')}  wall={_fmt((result.get('wall_s') or 0) / 3600.0, 2)} h",
        f"  endpoint {result.get('base_url')}  model {result.get('served_model')}",
        f"  geometry conc {geo.get('concurrency')} | {_fmt(geo.get('max_runtime_s_per_game'), 0)} s/game | "
        f"analyzer_timeout {_fmt(geo.get('analyzer_timeout'), 0)} | max_actions {geo.get('max_actions_per_game')} | "
        f"wave cap {_fmt(result.get('wave_cap_s'), 0)} s | public25 geometry: {geo.get('matches_public25')}",
        f"  stock agent sha {result['stock']['agent_tree_sha256'][:12]} (== june_stock pin) | "
        f"framework {result['stock']['framework_tree_sha256'][:12]} | pkls pinned",
        f"  ARM KNOB  CONTEXT_WINDOW={env.get('LOCAL_ANALYZER_CONTEXT_WINDOW')}  "
        f"MAX_OUTPUT={env.get('LOCAL_ANALYZER_MAX_OUTPUT')}  (yield {env.get('LOCAL_ANALYZER_YIELD_SECONDS')} s, "
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
        "  game            lv/n    act  calls turns reas_mean  len%  notool%  e2e_s   score  state",
    ]
    for gid, g in sorted(telemetry.get("per_game", {}).items()):
        r = g.get("reasoning_chars") or {}
        c = (g.get("client") or {}).get("e2e_s") or {}
        lines.append(
            f"  {gid:14s} {str(g.get('levels_completed')) + '/' + str(g.get('number_of_levels')):>5} "
            f"{_fmt(g.get('actions_total')):>6} {g.get('calls'):>5} {g.get('turns'):>5} "
            f"{_fmt(r.get('mean'), 0):>9} {_pct(g.get('length_finish_share')):>6} "
            f"{_pct(g.get('no_tool_call_share')):>7} {_fmt(c.get('mean')):>6} {_fmt(g.get('score'), 2):>7}  "
            f"{(result.get('states') or {}).get(gid, '')}")
    return "\n".join(lines) + "\n"


class Wave:
    def __init__(self, *, arm: str, base_url: str, token: str, out_dir: Path, game_ids: list[str],
                 per_game_s: float, concurrency: int, wave_cap_s: float, dry_run: bool,
                 progress_every_s: float) -> None:
        self.arm, self.base_url, self.token, self.out_dir = arm, base_url, token, out_dir
        self.game_ids, self.per_game_s, self.concurrency = game_ids, per_game_s, concurrency
        self.wave_cap_s, self.dry_run, self.progress_every_s = wave_cap_s, dry_run, progress_every_s
        self.result: dict = {"schema_version": 1, "arm": arm, "dry_run": dry_run, "status": "init",
                             "base_url": base_url, "served_model": SERVED_MODEL_NAME,
                             "wave_cap_s": wave_cap_s, "game_ids": list(game_ids)}
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
        bm.n_passes = 1
        bm.game_weights = None
        bm.solver.analyzer_factory = make_tagging_analyzer_factory(bm.solver)
        self.bm, self.target = bm, target
        self.result["geometry"] = geometry
        self.result["solver_label"] = bm.solver.label
        (self.out_dir / "arm_env.json").write_text(json.dumps(recorded_env, indent=1, sort_keys=True) + "\n")
        self._dump()
        print(f"[regime] arm={self.arm} games={len(self.game_ids)} geometry={geometry}", flush=True)

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
                if r["game_id"] in per and isinstance(per[r["game_id"]], dict):
                    r["score"] = per[r["game_id"]].get("score", r["score"])
        self.result["games"] = rows
        self.result["states"] = {r["game_id"]: r["state"] for r in rows}
        self.result["totals"] = {
            "games": len(rows),
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
        telemetry = build_telemetry(self.out_dir / "transcripts", rows, shim_records)
        telemetry["arm"] = self.arm
        telemetry["dry_run"] = self.dry_run
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
                   help=f"per-game cap (default {GEOMETRY['max_runtime_s_per_game']:.0f}; dry-run {DRY_RUN_PER_GAME_S:.0f})")
    p.add_argument("--concurrency", type=int, default=GEOMETRY["concurrency"])
    p.add_argument("--wave-cap-s", type=float, default=None,
                   help=f"whole-wave cap (default {WAVE_CAP_S:.0f}; dry-run {DRY_RUN_WAVE_CAP_S:.0f})")
    p.add_argument("--dry-run", action="store_true", help="loopback mock vLLM; no network")
    p.add_argument("--mock-latency-s", type=float, default=0.25)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--preflight-timeout", type=float, default=2400.0)
    p.add_argument("--progress-every", type=float, default=120.0)
    args = p.parse_args(argv)

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
    if args.dry_run:
        mock = MockVLLM(token, latency_s=args.mock_latency_s).start()
        base_url = mock.base_url
        print(f"[regime] DRY RUN: mock vLLM on {base_url} (303 legs every 3rd call, text-only every 7th)", flush=True)
    else:
        base_url = args.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            p.error(f"--base-url must end in /v1, got {args.base_url!r}")

    stock = assert_stock_tree()
    print(f"[regime] stock agent tree sha {stock['agent_tree_sha256'][:16]}… == june_stock pin "
          f"({stock['agent_files']} files); framework {stock['framework_tree_sha256'][:16]}…", flush=True)
    recorded_env = install_env(args.arm, base_url, token, out_dir)
    install_paths()
    verify_imports()
    print(f"[regime] arm {args.arm}: CONTEXT_WINDOW={recorded_env['LOCAL_ANALYZER_CONTEXT_WINDOW']} "
          f"MAX_OUTPUT={recorded_env['LOCAL_ANALYZER_MAX_OUTPUT']} (import-time constants verified)", flush=True)

    endpoint = None
    if not args.skip_preflight:
        endpoint = preflight(base_url, token, args.preflight_timeout)

    wave = Wave(arm=args.arm, base_url=base_url, token=token, out_dir=out_dir, game_ids=game_ids,
                per_game_s=per_game_s, concurrency=args.concurrency, wave_cap_s=wave_cap_s,
                dry_run=args.dry_run, progress_every_s=args.progress_every)
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
