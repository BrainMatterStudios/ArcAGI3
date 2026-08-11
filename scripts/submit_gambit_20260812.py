#!/usr/bin/env python3
"""submit_gambit_20260812.py — two-phase armed runner for the 2026-08-12 00:01 UTC slot.

PHASE 1 — PLATFORM PROBE (duck-mem v1, CPU-committed):
  Submits the CPU-only duck-mem version. PRE-REGISTERED HYPOTHESIS: the scored
  rerun's hardware follows the submitted version's accelerator, so a
  CPU-committed version ERRORs at the serve's GPU assert within minutes, and —
  per the audited 07-30 precedent (ERROR 55102674 07:19Z, scored sub 55121691
  23:59Z same UTC day) — an ERROR does NOT consume the day's slot.
  If it instead SCORES, rerun hardware is competition-fixed and every candidate
  this week can be pushed CPU and fired without GPU quota (week-unlocking).

PHASE 2 — FALLBACK (struct-v9 draw #2) fires ONLY IF phase 1 reaches status
  "error" before ERROR_DEADLINE_UTC. If phase 1 is still pending at the
  deadline, we let it ride (it cannot be cancelled; a ride is either a score —
  jackpot — or a late error that forfeits the day, the accepted tail risk).

Quota context (2026-08-11): 45h weekly GPU quota exhausted by sibling projects;
pushing ANY version with enable_gpu or machine_shape set is rejected. The CPU
push (enable_gpu:false, machine_shape REMOVED) succeeded — machine_shape alone
triggers the quota gate, a new platform fact.

Guards mirror submit_struct_20260811.py: one-shot window, marker idempotency,
identity re-attestation of BOTH kernels at fire time, slot-race check, and
submit_gated for the actual submissions.

--mock: everything runs (incl. both attestations) with --dry-run submits and a
90s target; used to test the full chain in daylight.
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
COMPETITION = "arc-prize-2026-arc-agi-3"

# --- phase 1: duck-mem v1 (CPU platform probe) ------------------------------
MEM_KERNEL = "ahmedmobasher86/arc-agi-3-duck-mem"
MEM_EXPECTED_VERSION = 2  # v1's commit ERRORed on the CPU machine's wheel-mount form
MEM_EXPECTED_HASH = "3d890a4692dc66c11853f98bf308ed99ac09017ed783776c6fb99e02f32dd2e0"
MEM_LOCAL_NOTEBOOK = REPO / "submission/_duck_mem/duck-mem.ipynb"
# Pins that MUST be in the remote bytes: the four default-armed duck-mem
# patches, and the two gates that must be ABSENT/OFF.
MEM_REQUIRED_MARKERS = (
    "P1 estimator //4",
    "P2 middle-drop trimmer",
    "P3 prompt own-goal neutralized (both namespaces)",
    "P4 repeated-no-effect guard",
    'os.environ.get("DUCK_MEM_P5", "0")',
    'os.environ.get("DUCK_MEM_P6", "0")',
)

MEM_MESSAGE = (
    "PLATFORM PROBE + duck-mem debut [duck-mem-v2-cpu-3d890a46]: CPU-committed "
    "version of duck-mem (shipped base v2 + P1-P4 anti-waste stack: estimator "
    "chars/4, middle-drop trimmer, minimize-actions prompt neutralized, "
    "repeated-no-effect guard; P5/P6 gated OFF). PRE-REGISTERED: primary "
    "hypothesis is about the PLATFORM — scored-rerun hardware follows the "
    "version's accelerator, so this CPU version should ERROR at the GPU assert "
    "with no slot cost (07-30 precedent); an in-window fallback then fires "
    "struct-v9 draw #2. IF this scores instead: rerun hardware is "
    "competition-fixed (week-unlocking fact) and the score reads against the "
    "base band 0.69-1.30 (n=10 mean 0.9650 sd 0.2082), mid-band ambiguous, "
    "4-variable stack so no per-lever attribution."
)

# --- phase 2: struct-v9 fallback (identical to the 08-11 runner) ------------
V9_KERNEL = "ahmedmobasher86/arc-agi-3-duck-patched"
V9_EXPECTED_VERSION = 9
V9_EXPECTED_SCRIPT_VERSION_ID = "341311881"
V9_EXPECTED_HASH = "b48f64c9f47bc0e4d8da0934a21000ff8683914a2dcda8fc2e009b42d68c3112"
V9_GIT_COMMIT = "8c27e430c403d10e07288013da7b2e82d4581acb"
V9_LOCAL_NOTEBOOK = REPO / "logs/duck-patched-v9-8c27e430.ipynb"
V9_REQUIRED_PINS = {
    "TAAF_STRUCT": "1",
    "TAAF_DIFF_LINES": "1",
    "TAAF_WIGGLE": "1",
    "TAAF_DISPATCH": "1",
    "TAAF_ANIMATION": "0",
    "TAAF_WATCHDOG_STALL_S": "900",
}
V9_MESSAGE = (
    "Structural plan channel draw #2 [struct-v9-b48f64c9] — FALLBACK after the "
    "pre-registered CPU platform probe ERRORed as predicted (no slot cost, "
    "07-30 precedent). Same bytes as sub 55418633 (scored 1.03 in-band on "
    "08-11). READING RULE unchanged: base band 0.69-1.30 (n=10 mean 0.9650 sd "
    "0.2082); only a draw OUTSIDE the band is individually actionable; n=2 "
    "transfer evidence accumulates toward the 4-6 wave certification."
)

TARGET_UTC = datetime(2026, 8, 12, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 12, 2, 30, 0, tzinfo=timezone.utc)
# Phase 1 must reach "error" by this time for the fallback to fire; a pending
# probe past this rides to its natural end.
ERROR_DEADLINE_UTC = datetime(2026, 8, 12, 1, 15, 0, tzinfo=timezone.utc)
POLL_S = 120

MARKER_P1 = REPO / "logs/gambit_p1_20260812.marker"
MARKER_P2 = REPO / "logs/gambit_p2_20260812.marker"


def log(msg: str) -> None:
    print(f"[gambit-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


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


def code_source(nb: dict) -> str:
    return "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
    )


def kernel_status_positive(kernel: str, attempts: int = 5) -> str | None:
    pattern = re.compile(r'has status\s+"([^"]+)"', re.IGNORECASE)
    for attempt in range(1, attempts + 1):
        try:
            out = subprocess.run(
                ["python3", "-m", "kaggle", "kernels", "status", kernel],
                capture_output=True, text=True, timeout=120,
            )
            match = pattern.search(out.stdout + out.stderr)
            if match:
                return match.group(1).strip()
            log(f"status attempt {attempt}/{attempts}: no positive status line")
        except (subprocess.TimeoutExpired, OSError) as exc:
            log(f"status attempt {attempt}/{attempts} transport failure: {exc!r}")
        if attempt < attempts:
            time.sleep(min(30, 3 * attempt))
    return None


def extract_v9_notebook() -> None:
    blob = subprocess.check_output(
        ["git", "-C", str(REPO), "show",
         f"{V9_GIT_COMMIT}:submission/_duck_patched/duck-patched.ipynb"])
    nb = json.loads(blob)
    got = canonical_code_hash(nb)
    if got != V9_EXPECTED_HASH:
        raise SystemExit(
            f"ABORT: git blob {V9_GIT_COMMIT[:12]} hashes to {got[:16]}… "
            f"!= attested {V9_EXPECTED_HASH[:16]}…")
    V9_LOCAL_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    V9_LOCAL_NOTEBOOK.write_bytes(blob)
    log(f"v9 notebook extracted from {V9_GIT_COMMIT[:12]} (hash verified)")


def reattest_mem() -> None:
    owner, slug = MEM_KERNEL.split("/")
    base = "https://www.kaggle.com/api/v1/kernels"
    pulled = api_json(
        f"{base}/pull?user_name={owner}&kernel_slug={slug}&version_label=v{MEM_EXPECTED_VERSION}"
    )
    version = pulled["metadata"].get("currentVersionNumber")
    if version != MEM_EXPECTED_VERSION:
        raise SystemExit(f"ABORT: duck-mem pull reports version {version} != {MEM_EXPECTED_VERSION}")
    remote_nb = json.loads(pulled["blob"]["source"])
    remote_hash = canonical_code_hash(remote_nb)
    if remote_hash != MEM_EXPECTED_HASH:
        raise SystemExit(
            f"ABORT: duck-mem v{MEM_EXPECTED_VERSION} hash {remote_hash[:16]}… "
            f"!= attested {MEM_EXPECTED_HASH[:16]}…")
    remote_src = code_source(remote_nb)
    missing = [m for m in MEM_REQUIRED_MARKERS if m not in remote_src]
    if missing:
        raise SystemExit(f"ABORT: duck-mem remote bytes missing markers: {missing}")
    local_nb = json.loads(MEM_LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != MEM_EXPECTED_HASH:
        raise SystemExit("ABORT: local duck-mem notebook no longer matches the attested hash")
    status = kernel_status_positive(MEM_KERNEL)
    if status is None:
        log("WARNING: duck-mem status unreadable — deferring to submit_gated's gate")
    elif "COMPLETE" not in status.upper():
        raise SystemExit(f"ABORT: duck-mem status not COMPLETE: {status}")
    log(f"duck-mem re-attest OK: v{MEM_EXPECTED_VERSION}, hash {remote_hash[:16]}…, "
        f"{len(MEM_REQUIRED_MARKERS)} markers present, remote==local")


def reattest_v9() -> None:
    owner, slug = V9_KERNEL.split("/")
    base = "https://www.kaggle.com/api/v1/kernels"
    pulled = api_json(
        f"{base}/pull?user_name={owner}&kernel_slug={slug}&version_label=v{V9_EXPECTED_VERSION}"
    )
    version = pulled["metadata"].get("currentVersionNumber")
    if version != V9_EXPECTED_VERSION:
        raise SystemExit(f"ABORT: v9 pull reports version {version} != {V9_EXPECTED_VERSION}")
    remote_nb = json.loads(pulled["blob"]["source"])
    remote_hash = canonical_code_hash(remote_nb)
    if remote_hash != V9_EXPECTED_HASH:
        raise SystemExit(f"ABORT: v9 hash {remote_hash[:16]}… != attested {V9_EXPECTED_HASH[:16]}…")
    remote_src = code_source(remote_nb)
    disarmed = [
        f"{k}={v}" for k, v in sorted(V9_REQUIRED_PINS.items())
        if f'_os.environ["{k}"] = "{v}"' not in remote_src
    ]
    if disarmed:
        raise SystemExit(f"ABORT: v9 missing armed pins: {disarmed}")
    output = api_json(
        f"{base}/output?user_name={owner}&kernel_slug={slug}&version_label=v{V9_EXPECTED_VERSION}"
    )
    kf_ids = sorted(set(re.findall(r"/kf/(\d+)/", json.dumps(output))))
    if kf_ids != [V9_EXPECTED_SCRIPT_VERSION_ID]:
        raise SystemExit(f"ABORT: v9 output binds kf ids {kf_ids} != [{V9_EXPECTED_SCRIPT_VERSION_ID}]")
    local_nb = json.loads(V9_LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != V9_EXPECTED_HASH:
        raise SystemExit("ABORT: extracted v9 notebook no longer matches the attested hash")
    log(f"v9 re-attest OK: v{V9_EXPECTED_VERSION} == svid {V9_EXPECTED_SCRIPT_VERSION_ID}, "
        f"hash {remote_hash[:16]}…, {len(V9_REQUIRED_PINS)} pins armed")


def day_submissions(day_start: datetime) -> list[dict]:
    rows = api_json(
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMPETITION}?page=1"
    )
    out = []
    for row in rows if isinstance(rows, list) else []:
        raw = str(row.get("date") or "")
        try:
            made = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if made >= day_start:
            out.append(row)
    return out


def run_submit_gated(kernel: str, version: int, notebook: Path, message: str,
                     mock: bool) -> int:
    cmd = [
        sys.executable, str(REPO / "scripts/submit_gated.py"),
        "--kernel", kernel, "--version", str(version),
        "--notebook", str(notebook), "--message", message,
        "--not-after", WINDOW_END_UTC.isoformat(),
    ]
    if mock:
        cmd.append("--dry-run")
    log("invoking submit_gated for " + kernel + " …")
    proc = subprocess.run(cmd, cwd=REPO)
    log(f"submit_gated exit code: {proc.returncode}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        log("WARNING: certifi unavailable — API calls may be SSL-blind")

    if MARKER_P1.exists() and not args.mock:
        raise SystemExit(f"ABORT: marker present ({MARKER_P1}) — phase 1 already fired")

    # Fail-fast at launch: both identities must attest BEFORE the wait.
    extract_v9_notebook()
    reattest_mem()
    reattest_v9()

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

    # Re-attest at fire time (bytes could have moved during the wait).
    reattest_mem()
    reattest_v9()

    day_start = (target if not args.mock else now).replace(
        hour=0, minute=0, second=0, microsecond=0)
    existing = day_submissions(day_start)
    if existing and not args.mock:
        raise SystemExit(f"ABORT: today's slot already has submissions: "
                         f"{[r.get('ref') for r in existing]}")
    if existing and args.mock:
        log("mock: race guard would abort here (expected during daytime test)")

    if not args.mock:
        MARKER_P1.parent.mkdir(parents=True, exist_ok=True)
        MARKER_P1.write_text(f"claimed {datetime.now(timezone.utc).isoformat()} pid={os.getpid()}\n")

    rc = run_submit_gated(MEM_KERNEL, MEM_EXPECTED_VERSION, MEM_LOCAL_NOTEBOOK,
                          MEM_MESSAGE, args.mock)
    if rc != 0:
        if not args.mock:
            MARKER_P1.unlink(missing_ok=True)
        raise SystemExit(f"phase 1 submit_gated failed (rc={rc}) — marker released")

    if args.mock:
        log("mock: skipping error-watch; phase 2 dry-run follows immediately")
    else:
        # Watch for the pre-registered fast ERROR.
        log(f"phase 1 submitted — polling for status until "
            f"{ERROR_DEADLINE_UTC:%H:%M}Z (poll {POLL_S}s)")
        probe_status = "pending"
        while datetime.now(timezone.utc) < ERROR_DEADLINE_UTC:
            rows = day_submissions(day_start)
            ours = [r for r in rows if str(r.get("status", "")).lower() in
                    ("error", "complete", "pending")]
            if ours:
                probe_status = str(ours[0].get("status", "")).lower()
                log(f"probe status: {probe_status}")
                if probe_status in ("error", "complete"):
                    break
            time.sleep(POLL_S)
        if probe_status == "complete":
            log("PROBE SCORED?! — rerun hardware is competition-fixed. "
                "Week unlocked; no fallback needed.")
            return 0
        if probe_status != "error":
            log("probe still pending at deadline — RIDING it; no fallback "
                "(pre-registered tail risk accepted)")
            return 0
        log("probe ERRORed as predicted — slot preserved per 07-30 precedent; "
            "firing struct-v9 fallback")

    if MARKER_P2.exists() and not args.mock:
        raise SystemExit(f"ABORT: marker present ({MARKER_P2}) — phase 2 already fired")
    if not args.mock:
        MARKER_P2.write_text(f"claimed {datetime.now(timezone.utc).isoformat()} pid={os.getpid()}\n")
    rc = run_submit_gated(V9_KERNEL, V9_EXPECTED_VERSION, V9_LOCAL_NOTEBOOK,
                          V9_MESSAGE, args.mock)
    if rc != 0 and not args.mock:
        MARKER_P2.unlink(missing_ok=True)
        log("phase 2 submit_gated failed — marker released for manual retry in-window")
    return rc


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
