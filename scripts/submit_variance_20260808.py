#!/usr/bin/env python3
"""submit_variance_20260808.py — one-shot armed runner for the 2026-08-08 00:01 UTC slot.

Submits the FROZEN sampling-variance arm (arc-agi-3-duck-variance v1) through the
sanctioned gated path, per the 2026-08-05 pre-reset amendment (arm displaced from
Friday by the v8 auto-submit; re-approved by Ahmed 2026-08-07 for the Saturday slot).

Hard guards, in order, all fatal on failure:
  1. ONE-SHOT WINDOW — refuses to run outside 2026-08-08 00:01–02:00 UTC (after the
     sleep), so a stale relaunch can never fire on a later day.
  2. IDENTITY RE-ATTEST — remote kernel must read version==1, status COMPLETE, and
     the pulled code-cell hash (canonical method: "\n".join(code cells) + "\n")
     must equal the frozen 71c25dc2… attestation; remote must equal the tracked
     local notebook byte-for-byte.
  3. SLOT RACE — aborts if any submission already exists in the new UTC day.
  4. Everything else is submit_gated.py's job (builder honesty, COMPLETE+settle,
     never-played watch).

--mock: target = now + 90 s, submit_gated runs with --dry-run, window/race guards
report but do not abort. Used to test the full chain without submitting.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNEL = "ahmedmobasher86/arc-agi-3-duck-variance"
EXPECTED_VERSION = 1
EXPECTED_HASH = "71c25dc27bd9d619d6ec7a765402a4c5b77d64eff69ea6a0d121440b20510c1b"
LOCAL_NOTEBOOK = REPO / "submission/_duck_variance/duck-variance.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 8, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 8, 2, 0, 0, tzinfo=timezone.utc)
MESSAGE = (
    "Frozen sampling-variance exploratory arm [variance-v1-71c25dc2]: plain base duck "
    "with only analyzer sampling changed from temp/top_k/top_p 0.6/20/0.95 to "
    "0.9/50/0.98; unseeded. Model, prompt, serving, concurrency, budgets and game "
    "list unchanged. PURPOSE: large-effect and rank-upside screen, not a one-draw "
    "estimate of mean or variance. Only a score outside 0.69-1.27 is individually "
    "actionable."
)


def log(msg: str) -> None:
    print(f"[variance-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def auth_header() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (user and key):
        creds = json.loads((Path.home() / ".kaggle/kaggle.json").read_text())
        user, key = creds["username"], creds["key"]
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def api_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"Authorization": auth_header()})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def canonical_code_hash(nb: dict) -> str:
    src = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
    ) + "\n"
    return hashlib.sha256(src.encode()).hexdigest()


def reattest() -> None:
    owner, slug = KERNEL.split("/")
    pulled = api_json(f"https://www.kaggle.com/api/v1/kernels/pull?user_name={owner}&kernel_slug={slug}")
    version = pulled["metadata"].get("currentVersionNumber")
    if version != EXPECTED_VERSION:
        raise SystemExit(f"ABORT: remote version {version} != expected {EXPECTED_VERSION}")
    remote_nb = json.loads(pulled["blob"]["source"])
    remote_hash = canonical_code_hash(remote_nb)
    if remote_hash != EXPECTED_HASH:
        raise SystemExit(f"ABORT: remote code hash {remote_hash[:16]}… != attested {EXPECTED_HASH[:16]}…")
    local_nb = json.loads(LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != EXPECTED_HASH:
        raise SystemExit("ABORT: local tracked notebook no longer matches the attested hash")
    status = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True, timeout=120
    )
    combined = (status.stdout + status.stderr).upper()
    if "COMPLETE" not in combined:
        raise SystemExit(f"ABORT: kernel status not COMPLETE: {combined.strip()[:200]}")
    log(f"re-attest OK: version 1, hash {remote_hash[:16]}…, remote==local, COMPLETE")


def slot_already_used(day_start: datetime) -> bool:
    rows = api_json(
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMPETITION}?page=1"
    )
    for row in rows if isinstance(rows, list) else []:
        raw = str(row.get("date") or "")
        try:
            made = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if made >= day_start:
            log(f"found submission {row.get('ref')} at {raw} in the new UTC day")
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="target now+90s, pass --dry-run to submit_gated, guards report only")
    args = parser.parse_args()

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        log("WARNING: certifi unavailable — post-submit watch may be SSL-blind")

    target = datetime.now(timezone.utc) + timedelta(seconds=90) if args.mock else TARGET_UTC
    delay = (target - datetime.now(timezone.utc)).total_seconds()
    log(f"target {target:%Y-%m-%d %H:%M:%S}Z — sleeping {delay/3600:.2f} h")
    if delay > 0:
        time.sleep(delay)

    now = datetime.now(timezone.utc)
    if not args.mock and not (TARGET_UTC <= now <= WINDOW_END_UTC):
        raise SystemExit(f"ABORT: outside one-shot window (now {now:%Y-%m-%d %H:%M}Z)")

    reattest()

    day_start = target.replace(hour=0, minute=0, second=0, microsecond=0)
    if slot_already_used(day_start):
        if args.mock:
            log("mock: race guard would abort here (expected during daytime test)")
        else:
            raise SystemExit("ABORT: today's slot already consumed by another submission")

    cmd = [
        sys.executable, str(REPO / "scripts/submit_gated.py"),
        "--kernel", KERNEL, "--version", str(EXPECTED_VERSION),
        "--notebook", str(LOCAL_NOTEBOOK), "--message", MESSAGE,
    ]
    if args.mock:
        cmd.append("--dry-run")
    log("invoking: " + " ".join(cmd[:-1] if args.mock else cmd[:8]) + " …")
    proc = subprocess.run(cmd, cwd=REPO)
    log(f"submit_gated exit code: {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
