#!/usr/bin/env python3
"""submit_base_20260809.py — one-shot armed runner for the 2026-08-09 00:01 UTC slot.

Submits the PINNED duck-base config — kernel version 2 of
ahmedmobasher86/arc-agi-3-duck-base — the exact bytes behind the n=8 base
distribution (mean 0.929 sd 0.195), through the sanctioned gated path.

IDENTITY EVIDENCE (established 2026-08-08, this repo):
  * Kaggle's own submission rows bind all 8 farm draws to scriptVersionId
    336252059 (urlNullable on /api/v1/competitions/submissions/list).
  * /api/v1/kernels/output?version_label=v2 returns output files whose storage
    URLs embed /kf/336252059/ — Kaggle's own record that VERSION 2 IS
    scriptVersionId 336252059 (v1 -> kf/336229761 = the 07-18 ERROR sub's
    binding, an independent anchor; v3 -> kf/336323574, never submitted).
  * /api/v1/kernels/pull?...&version_label=v2 returns code cells whose
    canonical hash ("\n".join(code-cell sources), the submission-ledger method)
    is 886dbc8a58ec79a0... == the tracked local notebook, byte-for-byte.
  * NOTE: docs/submission-ledger.json records 468aa314e61ead74 as v2's hash —
    that is WRONG; it is v3's content (gpu-probe cell). The 08-04
    reconstruction pulled "latest" believing it was v2, but v3 (336323574,
    created ~07-19) already existed. Version attribution (v2) was correct;
    the content hash was v3's. Verified via per-version session logs
    ([gpu-probe] output appears ONLY in v3's log) and the kf ids above.

Hard guards, in order, all fatal on failure:
  1. ONE-SHOT WINDOW — refuses to run outside 2026-08-09 00:01–02:00 UTC (after
     the sleep), so a stale relaunch can never fire on a later day.
  2. IDENTITY RE-ATTEST — version_label=v2 pull must hash to the attested
     886dbc8a... under the ledger canonical method AND report version 2; the
     v2 output listing must still bind /kf/336252059/; the tracked local
     notebook must hash identically; kernel status must read COMPLETE.
  3. SLOT RACE — aborts if any submission already exists in the new UTC day.
  4. Everything else is submit_gated.py's job (builder honesty, COMPLETE+settle,
     never-played watch). Note gate 2 of submit_gated reads the kernel's LATEST
     session status (v3's) — acceptable: it is a liveness check, and the
     version actually submitted is pinned by -v 2 server-side.

--mock: target = now + 90 s, submit_gated runs with --dry-run, window/race
guards report but do not abort. Used to test the full chain without submitting.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNEL = "ahmedmobasher86/arc-agi-3-duck-base"
EXPECTED_VERSION = 2
EXPECTED_SCRIPT_VERSION_ID = "336252059"
# sha256 over "\n".join(code-cell sources) — the submission-ledger canonical method.
EXPECTED_HASH = "886dbc8a58ec79a090085481d616385d742fefa30cbb3403076cf9f788e58a5a"
LOCAL_NOTEBOOK = REPO / "submission/_duck_base/duck-base.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 9, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 9, 2, 0, 0, tzinfo=timezone.utc)
MESSAGE = (
    "Idle-day byte-identical base draw [base-v2-886dbc8a]: pinned duck-base v2 "
    "(scriptVersionId 336252059), the exact bytes of the n=8 yardstick "
    "distribution (mean 0.929), no changes. CONTEXT: variance v1 drew 0.85 "
    "IN_BAND; Track A closure NO_GO on wave-2 replication (+1); Track B 35B "
    "gate NO_GO -> per the approved slot framework this is opportunistic rank "
    "banking on the best-mean config, zero evidentiary weight, no hypothesis "
    "claimed."
)


def log(msg: str) -> None:
    print(f"[base-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


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
    """Submission-ledger canonical method: sha256 over newline-joined code cells."""
    src = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
    )
    return hashlib.sha256(src.encode()).hexdigest()


def reattest() -> None:
    owner, slug = KERNEL.split("/")
    base = f"https://www.kaggle.com/api/v1/kernels"

    # 1. Version-2 bytes: version_label pull must hash to the attestation and
    #    self-report version 2 (label pulls verified faithful against the
    #    ledger-proven duck-patched v7 hash on 2026-08-08).
    pulled = api_json(
        f"{base}/pull?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    version = pulled["metadata"].get("currentVersionNumber")
    if version != EXPECTED_VERSION:
        raise SystemExit(f"ABORT: version_label pull reports version {version} != {EXPECTED_VERSION}")
    remote_nb = json.loads(pulled["blob"]["source"])
    remote_hash = canonical_code_hash(remote_nb)
    if remote_hash != EXPECTED_HASH:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} code hash {remote_hash[:16]}… != attested {EXPECTED_HASH[:16]}…")

    # 2. Version number <-> scriptVersionId: the v2 output listing's storage
    #    URLs must still embed /kf/336252059/ (Kaggle's own binding; the same
    #    id every one of the 8 scored farm draws bound).
    output = api_json(
        f"{base}/output?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    kf_ids = sorted(set(re.findall(r"/kf/(\d+)/", json.dumps(output))))
    if kf_ids != [EXPECTED_SCRIPT_VERSION_ID]:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} output binds kf ids {kf_ids} != [{EXPECTED_SCRIPT_VERSION_ID}]")

    # 3. Local tracked notebook is those same bytes.
    local_nb = json.loads(LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != EXPECTED_HASH:
        raise SystemExit("ABORT: local tracked notebook no longer matches the attested hash")

    # 4. Kernel liveness (latest session; the submitted version is pinned by -v).
    status = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True, timeout=120
    )
    combined = (status.stdout + status.stderr).upper()
    if "COMPLETE" not in combined:
        raise SystemExit(f"ABORT: kernel status not COMPLETE: {combined.strip()[:200]}")
    log(
        f"re-attest OK: v{EXPECTED_VERSION} == scriptVersionId {EXPECTED_SCRIPT_VERSION_ID}, "
        f"hash {remote_hash[:16]}…, remote==local, COMPLETE"
    )


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
