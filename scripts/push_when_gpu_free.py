#!/usr/bin/env python3
"""Push a kernel as soon as a Kaggle batch-GPU session frees up.

Kaggle caps concurrent batch GPU sessions at 2 per account, ACROSS ALL PROJECTS.
On 2026-08-09 an unrelated `rsna-knee-phase0-baseline-v21` run held one slot, so
the second ft09-ablation arm could not launch. Rather than serialise by hand or
kill someone else's job, this polls and pushes the instant capacity appears.

It only ever pushes the kernel it was told to push, once, and stops on success.

Usage:
    python3 scripts/push_when_gpu_free.py --path submission/_ab_patch_closure/ft09_struct \
        [--interval 300] [--max-hours 10]
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Both are transient-capacity conditions worth polling through: concurrent
# session cap (frees in minutes-hours) and the weekly quota cap (frees at the
# weekly reset — poll hourly with a multi-day --max-hours). Discovered
# 2026-08-10: the 45h weekly quota can be exhausted by OTHER projects on the
# account, so a rig launch must be able to wait for the reset unattended.
RETRYABLE_MARKERS = (
    "Maximum batch GPU session count",
    "Maximum weekly GPU quota",
)


def log(msg: str) -> None:
    print(f"[push-when-free {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def try_push(path: Path) -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["python3", "-m", "kaggle", "kernels", "push", "-p", str(path),
             "--accelerator", "NvidiaRtxPro6000"],
            capture_output=True, text=True, timeout=300, cwd=REPO,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"transport failure: {exc!r}"
    text = (out.stdout + out.stderr).strip()
    return ("successfully pushed" in text), text.splitlines()[-1] if text else "(no output)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="kernel directory to push")
    ap.add_argument("--interval", type=int, default=300, help="seconds between attempts")
    ap.add_argument("--max-hours", type=float, default=10.0)
    args = ap.parse_args()

    path = (REPO / args.path).resolve()
    if not (path / "kernel-metadata.json").is_file():
        raise SystemExit(f"ABORT: no kernel-metadata.json under {path}")

    deadline = time.time() + args.max_hours * 3600
    attempt = 0
    log(f"waiting for a free GPU session to push {path.name}")
    while time.time() < deadline:
        attempt += 1
        ok, msg = try_push(path)
        if ok:
            log(f"PUSHED on attempt {attempt}: {msg}")
            return 0
        if any(marker in msg for marker in RETRYABLE_MARKERS):
            log(f"attempt {attempt}: GPU capacity/quota unavailable — retry in {args.interval}s")
        else:
            # Anything that is NOT the capacity error is a real problem: a bad
            # notebook, an auth failure, a metadata error. Retrying would just
            # repeat it, so surface it and stop.
            log(f"attempt {attempt}: NON-CAPACITY failure, aborting: {msg}")
            return 1
        time.sleep(args.interval)
    log(f"gave up after {args.max_hours}h without a free session")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
