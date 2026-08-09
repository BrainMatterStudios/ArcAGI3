#!/usr/bin/env python3
"""auto_submit_on_reset.py — Automated Submission Scheduler for Daily Kaggle Quota Reset.

Calculates remaining time until 00:01:00 UTC (daily Kaggle submission allowance reset),
sleeps until the reset window opens, and automatically executes `submit_gated.py` for Version 7 (duck-sparse 35B swap).
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="[auto-submit %(asctime)s] %(message)s")
logger = logging.getLogger("auto_submit")


def seconds_until_utc_reset(target_hour: int = 0, target_minute: int = 1) -> float:
    """Calculate remaining seconds until target UTC time (00:01:00 UTC)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    if target <= now:
        target += datetime.timedelta(days=1)
        
    diff = (target - now).total_seconds()
    return diff


def main() -> int:
    delay = seconds_until_utc_reset(target_hour=0, target_minute=1)
    target_utc_str = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=delay)).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    logger.info(f"Target reset time: {target_utc_str}")
    logger.info(f"Sleeping for {delay:.0f} seconds ({delay/3600:.2f} hours) until daily Kaggle submission reset...")
    
    time.sleep(delay)
    
    logger.info("Daily Kaggle quota reset window reached! Executing submit_gated.py for Version 7 (duck-sparse 35B swap)...")
    
    cmd = [
        sys.executable,
        "scripts/submit_gated.py",
        "--kernel", "ahmedmobasher86/arc-agi-3-duck-sparse",
        "--version", "7",
        "--notebook", "submission/_duck_sparse/duck-sparse.ipynb",
        "--message", "v7: Qwen3.6-35B-A3B-FP8 swap via arcagi3-bundle-35b + cmechevalier snapshot; sparse sandbox + tool_agent prompt advertisement",
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"submit_gated exit code: {proc.returncode}")
    logger.info(f"Output:\n{proc.stdout}")
    
    if proc.returncode != 0:
        logger.error(f"Error output:\n{proc.stderr}")
        return proc.returncode

    logger.info("SUCCESS: Version 7 submission automatically submitted and recorded!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
