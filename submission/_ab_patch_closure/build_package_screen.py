#!/usr/bin/env python3
"""Build the probe-infrastructure PACKAGE SCREEN kernel (approved 2026-08-09).

One candidate-only commit kernel on the patch-closure machinery: duck-base +
the CURRENT duck_patches.py (patches 1-19), arm env = closure BASE_ENV (v7
pins) + the four package flags (TAAF_DIFF_LINES / TAAF_WIGGLE /
TAAF_RUN_PROBE / TAAF_DISPATCH = 1), closure geometry (28 clones, 7920s,
concurrency 28), same contract embedding, docker pin, dual-mount install cell
and visible pip stderr. The pre-registered reading contract
(package_screen_config.PACKAGE_READING) is baked into the run cell and lands
verbatim in patch_closure_result.json as `pre_registered_reading`.

Read against the BANKED closure base pair (11 / 12 levels excl-ft09) with
package_screen_config.classify_screen — see classify_package.py.

Usage:
    .venv/bin/python submission/_ab_patch_closure/build_package_screen.py
    kaggle kernels push -p submission/_ab_patch_closure/package --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_patch_closure import build_kernel, notebook_contract  # noqa: E402
from package_screen_config import (  # noqa: E402
    PACKAGE_ENV,
    PACKAGE_HYPOTHESIS,
    PACKAGE_READING,
)

PACKAGE_SLUG = "arc-agi-3-package-screen"

# ---------------------------------------------------------------------------
# PARALLEL LOAD PROBE — post-benchmark cell (runs AFTER the run cell, i.e.
# after patch_closure_result.json is written, while vLLM is still up; the run
# cell does not execute duck-base's teardown commands). Purpose: quantify
# whether a second parallel thought-stream per game rides free capacity — the
# verifier-at-commit patch20 premise. The interesting number is phase-B
# aggregate gen tokens/s vs 2x phase-A per-stream latency. Guards: total wall
# hard-capped at 5 minutes; fully wrapped so no failure can fail the kernel.
# ---------------------------------------------------------------------------
PARALLEL_LOAD_CELL = '''\
# ============================================================================
# PARALLEL LOAD PROBE (post-benchmark; patch_closure_result.json is already on
# disk, so this can never contaminate the screen result). Two measured phases
# against the still-running vLLM endpoint:
#   A: 28 concurrent analyzer-shaped streams (~2-4k prompt tokens, 256 out,
#      thinking off) for ~90s;  B: 56 concurrent streams, same shape.
# Emits /kaggle/working/parallel_load.json. Read: phase-B aggregate tokens/s
# vs 2x phase-A per-stream latency (verifier-at-commit patch20 premise).
# Fully wrapped: any failure logs and continues to the end of the notebook.
# ============================================================================
import json as _pl_json
import os as _pl_os
import statistics as _pl_stat
import threading as _pl_thr
import time as _pl_time
import traceback as _pl_tb
import urllib.request as _pl_url


def _pl_run_probe():
    HARD_CAP_S = 300.0     # total probe wall cap (both phases + slack)
    PHASE_WALL_S = 90.0
    REQ_TIMEOUT_S = 85.0
    t_start = _pl_time.monotonic()
    base_url = (_pl_os.environ.get("LOCAL_ANALYZER_BASE_URL")
                or _pl_os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    model = _pl_os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
    if not base_url:
        raise RuntimeError("no analyzer endpoint configured")

    # Analyzer-shaped prompt: a large SHARED prefix (realistic — real analyzer
    # prompts share system text and legend blocks, so prefix caching applies)
    # plus a unique per-request tail (realistic too — every game state
    # differs; defeats full-prompt caching).
    para = ("The board is a 64x64 grid of colored cells; rows are numbered "
            "from the top and columns from the left. Track object persistence, "
            "movement contingency, and the measured effect of every action "
            "before committing to a plan. ") * 55
    shape = {
        "shared_prefix_chars": len(para),
        "approx_prompt_tokens": int(len(para) / 4),
        "max_tokens": 256,
        "temperature": 0.7,
        "thinking": False,
        "unique_tail_per_request": True,
        "request_timeout_s": REQ_TIMEOUT_S,
        "phase_wall_s": PHASE_WALL_S,
    }
    out = {
        "purpose": ("quantify whether a second parallel thought-stream per game "
                    "rides free vLLM capacity (verifier-at-commit patch20 premise)"),
        "read": ("phase-B aggregate gen tokens/s vs 2x phase-A per-stream "
                 "latency: near-2x aggregate at similar latency = free capacity"),
        "endpoint": base_url,
        "model": model,
        "request_shape": shape,
        "phases": [],
    }

    def one_request(stream_i, phase, step):
        tail = (f"\\n\\n[probe stream {stream_i} phase {phase} step {step}] "
                "Reply with a numbered 3-step plan for exploring an unknown grid game.")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": para + tail}],
            "temperature": 0.7,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = _pl_url.Request(base_url + "/chat/completions",
                              data=_pl_json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
        t0 = _pl_time.monotonic()
        with _pl_url.urlopen(req, timeout=REQ_TIMEOUT_S) as r:
            resp = _pl_json.loads(r.read().decode())
        latency = _pl_time.monotonic() - t0
        gen = int(((resp.get("usage") or {}).get("completion_tokens")) or 0)
        return latency, gen

    for label, conc in (("A", 28), ("B", 56)):
        remaining = HARD_CAP_S - (_pl_time.monotonic() - t_start)
        if remaining < 30.0:
            out["phases"].append({"label": label, "concurrency": conc,
                                  "skipped": "hard cap reached"})
            break
        wall = min(PHASE_WALL_S, remaining - 10.0)
        deadline = _pl_time.monotonic() + wall
        lat, toks, errs = [], [], []
        lock = _pl_thr.Lock()

        def worker(stream_i, _deadline=deadline, _label=label):
            step = 0
            while _pl_time.monotonic() < _deadline:
                step += 1
                try:
                    latency, gen = one_request(stream_i, _label, step)
                    with lock:
                        lat.append(latency)
                        toks.append(gen)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errs.append(f"{type(exc).__name__}: {exc}"[:200])

        t_phase = _pl_time.monotonic()
        threads = [_pl_thr.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(conc)]
        for t in threads:
            t.start()
        join_by = deadline + REQ_TIMEOUT_S + 10.0
        for t in threads:
            t.join(timeout=max(0.0, join_by - _pl_time.monotonic()))
        phase_wall = _pl_time.monotonic() - t_phase
        lat_sorted = sorted(lat)
        timeouts = sum(1 for e in errs if "timed out" in e.lower())
        phase_rec = {
            "label": label,
            "concurrency": conc,
            "wall_s": round(phase_wall, 1),
            "requests_completed": len(lat),
            "errors": len(errs) - timeouts,
            "timeouts": timeouts,
            "error_samples": errs[:5],
            "gen_tokens_total": sum(toks),
            "agg_gen_tokens_per_s": round(sum(toks) / max(phase_wall, 1e-6), 1),
            "per_stream_gen_tokens_per_s": round(
                sum(toks) / max(phase_wall, 1e-6) / conc, 2),
            "latency_s": {
                "median": round(_pl_stat.median(lat_sorted), 2) if lat_sorted else None,
                "p90": (round(lat_sorted[max(0, int(len(lat_sorted) * 0.9) - 1)], 2)
                        if lat_sorted else None),
                "mean": round(_pl_stat.mean(lat_sorted), 2) if lat_sorted else None,
                "n": len(lat_sorted),
            },
        }
        out["phases"].append(phase_rec)
        print(f"[parallel-load] phase {label}: {phase_rec}", flush=True)

    out["total_wall_s"] = round(_pl_time.monotonic() - t_start, 1)
    target_dir = str(globals().get("WORKING_DIR") or "/kaggle/working")
    path = _pl_os.path.join(target_dir, "parallel_load.json")
    with open(path, "w") as fh:
        _pl_json.dump(out, fh, indent=1)
    print(f"[parallel-load] wrote {path} in {out['total_wall_s']}s", flush=True)


try:
    _pl_run_probe()
except Exception:  # noqa: BLE001 - the probe must NEVER fail the kernel
    print(f"[parallel-load] probe failed (kernel continues):\\n{_pl_tb.format_exc()}",
          flush=True)
'''


def build_package(output_root: Path | None = None) -> Path:
    """Build the package-screen kernel; returns the notebook path."""
    return build_kernel(
        arm="package",
        slug=PACKAGE_SLUG,
        arm_env=PACKAGE_ENV,
        hypothesis=PACKAGE_HYPOTHESIS,
        reading=PACKAGE_READING,
        code_stem="package-screen",
        output_root=output_root,
        post_run_cell=PARALLEL_LOAD_CELL,
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    nb_path = build_package()
    contract = notebook_contract(nb_path)
    print(f"wrote {nb_path}")
    print(f"  base_sha256={contract['source_base_sha256'][:12]}... "
          f"patch_sha256={contract['patch_sha256'][:12]}... arm={contract['arm']}")
    print(f"  push with: kaggle kernels push -p {nb_path.parent} "
          f"--accelerator NvidiaRtxPro6000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
