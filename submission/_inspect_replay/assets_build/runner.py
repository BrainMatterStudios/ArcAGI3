"""runner.py — Kaggle-side battery for the INSPECTION-PROMPT REPLAY falsifier.

Replays each recorded inspection-call context through the locally served
small Qwen3.5 model, executes the emitted snippet in the rehydrated sandbox,
and grades it against the 27B's recorded output. Results JSON is rewritten
atomically after every sample.

Called from the notebook as:
    run_battery(model_tag, base_url, served_name, deadline_epoch)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ASSETS = Path(os.environ.get("INSPECT_ASSETS_DIR", "/kaggle/input/arc3-inspect-replay-assets"))
WORKING = Path(os.environ.get("INSPECT_WORKING_DIR", "/kaggle/working"))
RESULTS_PATH = WORKING / "inspect_replay_results.json"

sys.path.insert(0, str(ASSETS))

import replay_lib  # noqa: E402

TA = replay_lib.bundle_setup()
from replay_lib import (  # noqa: E402
    StateRebuilder,
    grade_fact_recovery,
    run_snippet,
)

import gzip  # noqa: E402

_SAMPLES = None
_PACKS = None
_REBUILDERS: dict = {}


def _load_json_maybe_gz(stem: str):
    plain = ASSETS / f"{stem}.json"
    if plain.exists():  # Kaggle auto-decompresses .gz on ingestion
        return json.loads(plain.read_text(encoding="utf-8"))
    with gzip.open(ASSETS / f"{stem}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def load_battery():
    global _SAMPLES, _PACKS
    if _SAMPLES is None:
        _SAMPLES = _load_json_maybe_gz("samples")
        _PACKS = _load_json_maybe_gz("gamepacks")
    return _SAMPLES, _PACKS


def _rebuilder(pack_key: str) -> StateRebuilder:
    if pack_key not in _REBUILDERS:
        _REBUILDERS[pack_key] = StateRebuilder(_PACKS[pack_key], TA)
    return _REBUILDERS[pack_key]


def _post_chat(base_url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def _extract_code(message: dict) -> tuple[str | None, str]:
    """Return (code, how). Mirrors production: prefer parsed tool_calls,
    fall back to <tool_call> markup recovery from reasoning/content."""
    tool_calls = message.get("tool_calls") or []
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    if not tool_calls and TA._contains_tool_call_markup(str(reasoning), str(content)):
        tool_calls = TA._recover_tool_calls_from_markup(str(reasoning), str(content))
        if tool_calls:
            how = "markup_recovered"
        else:
            return None, "markup_unparsed"
    else:
        how = "parsed"
    if not tool_calls:
        return None, "none"
    fn = (tool_calls[0] or {}).get("function", {})
    if str(fn.get("name", "")).strip() != "python":
        return None, "wrong_tool"
    raw_args = fn.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "bad_arguments"
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return None, "empty_code"
    return code, how


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
    return ""


def run_battery(model_tag: str, base_url: str, served_name: str, deadline_epoch: float,
                max_tokens: int = 16384, request_timeout: float = 900.0) -> dict:
    samples, packs = load_battery()
    results = {}
    if RESULTS_PATH.exists():
        try:
            results = json.loads(RESULTS_PATH.read_text())
        except json.JSONDecodeError:
            results = {}
    arm = results.setdefault(model_tag, {"samples": {}, "meta": {
        "served_name": served_name, "started": time.time()}})
    done = arm["samples"]

    for sample in samples:
        sid = sample["sample_id"]
        if sid in done:
            continue
        if time.time() > deadline_epoch:
            arm["meta"]["stopped"] = "deadline"
            break
        row: dict = {"game_id": sample["game_id"], "corpus": sample["corpus"],
                     "position": sample["position"],
                     "recorded_error": bool(sample["recorded_error"])}
        try:
            payload = {
                "model": served_name,
                "messages": sample["messages"],
                "tools": sample["tools"],
                "tool_choice": "auto",
                "stream": False,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
            }
            t0 = time.time()
            resp = _post_chat(base_url, payload, request_timeout)
            row["decode_seconds"] = round(time.time() - t0, 2)
            choice = (resp.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = resp.get("usage") or {}
            row["finish_reason"] = choice.get("finish_reason")
            row["completion_tokens"] = usage.get("completion_tokens")
            row["prompt_tokens"] = usage.get("prompt_tokens")
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            row["reasoning_chars"] = len(str(reasoning))
            code, how = _extract_code(message)
            row["tool_call"] = how
            if code is None:
                row["outcome"] = "no_tool_call"
            else:
                row["code_chars"] = len(code)
                rb = _rebuilder(sample["pack_key"])
                res = run_snippet(
                    TA, rb, sample["state_idx"], code,
                    sample.get("last_action_result"), sample.get("valid_actions"),
                    WORKING / "sandbox" / f"{model_tag}_{sid}",
                )
                row["action_attempted"] = res["action_attempted"]
                row["exec_error"] = bool(res["error"]) and not res["action_attempted"]
                row["stdout_chars"] = len(res["stdout"])
                if res["action_attempted"]:
                    row["outcome"] = "action_attempted"
                elif row["exec_error"]:
                    row["outcome"] = "exec_error"
                    row["error_head"] = res["error"][:300]
                else:
                    row["outcome"] = "ok"
                ref_stdout = sample.get("recorded_stdout") or ""
                if ref_stdout and not sample["recorded_error"]:
                    row["grade"] = grade_fact_recovery(
                        ref_stdout, res["stdout"], _last_user_text(sample["messages"])
                    )
        except Exception as exc:  # noqa: BLE001 — per-sample isolation
            row["outcome"] = "harness_error"
            row["harness_error"] = f"{type(exc).__name__}: {exc}"[:400]
        done[sid] = row
        _atomic_write(RESULTS_PATH, results)
        print(f"[{model_tag}] {sid} {sample['game_id']} -> {row.get('outcome')} "
              f"({row.get('decode_seconds', '?')}s, {row.get('completion_tokens', '?')} tok)",
              flush=True)

    arm["meta"]["finished"] = time.time()
    arm["meta"]["n_done"] = len(done)
    _atomic_write(RESULTS_PATH, results)
    return results


def verdict(results: dict) -> dict:
    """Pre-registered kill criteria vs the 27B's recorded behavior."""
    samples, _ = load_battery()
    by_id = {s["sample_id"]: s for s in samples}
    out = {}
    for tag, arm in results.items():
        if tag.startswith("_"):
            continue
        rows = arm.get("samples", {})
        attempted = {sid: r for sid, r in rows.items() if r.get("outcome") != "harness_error"}
        n = len(attempted)
        if n == 0:
            continue
        # (a) execution-error rate on completed samples vs 27B recorded on same
        exec_errors = sum(1 for r in attempted.values() if r.get("outcome") == "exec_error")
        no_tool = sum(1 for r in attempted.values() if r.get("outcome") == "no_tool_call")
        actioned = sum(1 for r in attempted.values() if r.get("outcome") == "action_attempted")
        # dead completion = automatic fail for that sample (counts as error)
        small_fail = exec_errors + no_tool
        rec_err = sum(1 for sid in attempted if by_id[sid]["recorded_error"])
        small_rate = small_fail / n
        rec_rate = rec_err / n
        ratio = (small_rate / rec_rate) if rec_rate > 0 else float("inf") if small_rate > 0 else 0.0
        # (b) key-fact recovery on graded samples
        grades = [r["grade"] for r in attempted.values() if isinstance(r.get("grade"), dict)
                  and r["grade"].get("recovery") is not None]
        recovery = sum(g["recovery"] for g in grades) / len(grades) if grades else None
        novel = [g["novel_recovery"] for g in grades if g.get("novel_recovery") is not None]
        novel_recovery = sum(novel) / len(novel) if novel else None
        decode = [r["decode_seconds"] for r in attempted.values() if r.get("decode_seconds")]
        mean_decode = sum(decode) / len(decode) if decode else None
        kill_error = ratio > 1.5
        kill_recovery = (recovery is not None and recovery < 0.70)
        out[tag] = {
            "n": n,
            "exec_error_rate": round(small_rate, 4),
            "exec_errors": exec_errors,
            "no_tool_call": no_tool,
            "action_attempted": actioned,
            "recorded_27b_error_rate": round(rec_rate, 4),
            "error_ratio_vs_27b": round(ratio, 3) if ratio != float("inf") else "inf",
            "fact_recovery_mean": round(recovery, 4) if recovery is not None else None,
            "novel_fact_recovery_mean": round(novel_recovery, 4) if novel_recovery is not None else None,
            "graded_n": len(grades),
            "mean_decode_seconds": round(mean_decode, 1) if mean_decode else None,
            "speed_multiple_vs_50s": round(50.0 / mean_decode, 2) if mean_decode else None,
            "KILL_error_criterion(>1.5x)": kill_error,
            "KILL_recovery_criterion(<0.70)": kill_recovery,
            "VERDICT": "CASCADE DEAD (this model)" if (kill_error or kill_recovery) else "PASS (this model)",
        }
    return out
