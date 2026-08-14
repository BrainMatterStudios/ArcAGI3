#!/usr/bin/env python3
"""submit_mem_20260815.py — one-shot armed runner for the 2026-08-15 00:01 UTC slot.

Submits duck-mem v4 for its DRAW #2. Context at arm time (2026-08-14): debut
draw 0.83 (in-band, individually silent per the pre-registration); same-stack
rig A/B mem 1.5018 vs shipped 1.3467 (+0.155, below the one-wave 0.2712 bar,
directionally positive, stack proven ACTIVE: 1045 trims, 30 guard stops);
a second rig wave is queued to pool toward a ~0.19 bar. The live series
accumulates draws under the fixed reading rule while the rig converges.

Guards mirror the proven 08-11/08-14 runners: one-shot window, marker,
fire-time re-attestation (version pull hash + required markers + svid binding
+ local-notebook identity), slot race, submit_gated.
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
KERNEL = "ahmedmobasher86/arc-agi-3-duck-mem"
EXPECTED_VERSION = 4
EXPECTED_SCRIPT_VERSION_ID = "342201393"
EXPECTED_HASH = "401f551f6f6914df3edb3ced6ee756d05a6acaacd4a27935f62a36db7b23255f"
LOCAL_NOTEBOOK = REPO / "submission/_duck_mem/duck-mem.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 15, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 15, 2, 0, 0, tzinfo=timezone.utc)
MARKER = REPO / "logs/duck_mem_v4_20260815.marker"

REQUIRED_MARKERS = (
    "P1 estimator //4",
    "P2 middle-drop trimmer",
    "P3 prompt own-goal neutralized (both namespaces)",
    "P4 repeated-no-effect guard",
    'os.environ.get("DUCK_MEM_P5", "0")',
    'os.environ.get("DUCK_MEM_P6", "0")',
)

MESSAGE = (
    "duck-mem v4 draw #2 [duck-mem-v4-401f551f]: identical bytes to the debut "
    "(sub 55488796, 0.83 in-band). Accumulating series: same-stack rig A/B "
    "measured +0.155 true-objective delta over the unpatched config (below the "
    "one-wave 0.2712 bar; stack ACTIVE: 1045 middle-drop trims, 30 "
    "repeated-no-effect stops; mem arm used HALF the actions for a higher "
    "score). READING RULE, pre-registered and unchanged: base band 0.69-1.30 "
    "(n=10 mean 0.9650 sd 0.2082); only a draw OUTSIDE the band is "
    "individually actionable; live n accumulates while the offline pooled bar "
    "drops to ~0.19 at two rig waves. SECONDARY non-binding: totalBytes vs "
    "null 3676.8+/-21.8 (debut drew 3645, -1.5 sigma)."
)


def log(msg: str) -> None:
    print(f"[mem-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def auth_header() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (user and key):
        creds = json.loads((Path.home() / ".kaggle/kaggle.json").read_text())
        user, key = creds["username"], creds["key"]
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def api_json(url: str, attempts: int = 5) -> dict | list:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"Authorization": auth_header()})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts:
                backoff = min(30, 3 * attempt)
                log(f"api attempt {attempt}/{attempts} failed ({exc!r}) — retry in {backoff}s")
                time.sleep(backoff)
    raise SystemExit(f"ABORT: API unreachable after {attempts} attempts: {last!r}")


def canonical_code_hash(nb: dict) -> str:
    src = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
    )
    return hashlib.sha256(src.encode()).hexdigest()


def kernel_status_positive(attempts: int = 5) -> str | None:
    pattern = re.compile(r'has status\s+"([^"]+)"', re.IGNORECASE)
    for attempt in range(1, attempts + 1):
        try:
            out = subprocess.run(
                ["python3", "-m", "kaggle", "kernels", "status", KERNEL],
                capture_output=True, text=True, timeout=120,
            )
            match = pattern.search(out.stdout + out.stderr)
            if match:
                return match.group(1).strip()
        except (subprocess.TimeoutExpired, OSError) as exc:
            log(f"status attempt {attempt}/{attempts} transport failure: {exc!r}")
        if attempt < attempts:
            time.sleep(min(30, 3 * attempt))
    return None


def reattest() -> None:
    owner, slug = KERNEL.split("/")
    base = "https://www.kaggle.com/api/v1/kernels"
    pulled = api_json(
        f"{base}/pull?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    version = pulled["metadata"].get("currentVersionNumber")
    if version != EXPECTED_VERSION:
        raise SystemExit(f"ABORT: pull reports version {version} != {EXPECTED_VERSION}")
    remote_nb = json.loads(pulled["blob"]["source"])
    remote_hash = canonical_code_hash(remote_nb)
    if remote_hash != EXPECTED_HASH:
        raise SystemExit(
            f"ABORT: v{EXPECTED_VERSION} hash {remote_hash[:16]}… != attested {EXPECTED_HASH[:16]}…")
    remote_src = "\n".join(
        "".join(c.get("source", [])) for c in remote_nb["cells"] if c.get("cell_type") == "code"
    )
    missing = [m for m in REQUIRED_MARKERS if m not in remote_src]
    if missing:
        raise SystemExit(f"ABORT: remote bytes missing markers: {missing}")
    output = api_json(
        f"{base}/output?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    kf_ids = sorted(set(re.findall(r"/kf/(\d+)/", json.dumps(output))))
    if kf_ids != [EXPECTED_SCRIPT_VERSION_ID]:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} binds kf ids {kf_ids} != [{EXPECTED_SCRIPT_VERSION_ID}]")
    local_nb = json.loads(LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != EXPECTED_HASH:
        raise SystemExit("ABORT: local tracked notebook no longer matches the attested hash")
    status = kernel_status_positive()
    if status is None:
        log("WARNING: kernel status unreadable — deferring to submit_gated's gate")
    elif "COMPLETE" not in status.upper():
        raise SystemExit(f"ABORT: kernel status not COMPLETE: {status}")
    log(f"re-attest OK: v{EXPECTED_VERSION} == svid {EXPECTED_SCRIPT_VERSION_ID}, "
        f"hash {remote_hash[:16]}…, {len(REQUIRED_MARKERS)} markers, remote==local")


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
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        log("WARNING: certifi unavailable")

    if MARKER.exists() and not args.mock:
        raise SystemExit(f"ABORT: marker present ({MARKER}) — this slot already fired")

    reattest()

    target = datetime.now(timezone.utc) + timedelta(seconds=90) if args.mock else TARGET_UTC
    log(f"target {target:%Y-%m-%d %H:%M:%S}Z — "
        f"{(target - datetime.now(timezone.utc)).total_seconds()/3600:.2f} h to go")
    while True:
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(60.0, remaining))

    now = datetime.now(timezone.utc)
    if not args.mock and not (TARGET_UTC - timedelta(seconds=60) <= now <= WINDOW_END_UTC):
        raise SystemExit(f"ABORT: outside one-shot window (now {now:%Y-%m-%d %H:%M}Z)")

    reattest()

    day_start = target.replace(hour=0, minute=0, second=0, microsecond=0)
    if slot_already_used(day_start):
        if args.mock:
            log("mock: race guard would abort here (expected during daytime test)")
        else:
            raise SystemExit("ABORT: today's slot already consumed")

    if not args.mock:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(f"claimed {datetime.now(timezone.utc).isoformat()} pid={os.getpid()}\n")

    cmd = [
        sys.executable, str(REPO / "scripts/submit_gated.py"),
        "--kernel", KERNEL, "--version", str(EXPECTED_VERSION),
        "--notebook", str(LOCAL_NOTEBOOK), "--message", MESSAGE,
        "--not-after", WINDOW_END_UTC.isoformat(),
    ]
    if args.mock:
        cmd.append("--dry-run")
    proc = subprocess.run(cmd, cwd=REPO)
    log(f"submit_gated exit code: {proc.returncode}")
    # NOTE (08-12 lesson): submit_gated returns nonzero AFTER a successful
    # submit if its watch sees ERROR — release the marker ONLY when the
    # failure is pre-submit (no submission row exists for today).
    if proc.returncode != 0 and not args.mock:
        if not slot_already_used(day_start):
            MARKER.unlink(missing_ok=True)
            log("pre-submit failure — marker released for manual retry in-window")
        else:
            log("post-submit nonzero rc (watch alarm) — marker kept; submission exists")
    return proc.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        log(f"UNHANDLED {exc!r}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
