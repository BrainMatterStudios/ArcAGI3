#!/usr/bin/env python3
"""submit_struct_20260810.py — one-shot armed runner for the 2026-08-10 00:01 UTC slot.

Submits duck-patched v9 — the FIRST LIVE PROBE of the structural plan channel
(patches 21+22) — through the sanctioned gated path.

WHAT v9 IS
  Built 2026-08-09 from branch winning/duck-patched @ 8c27e43 (patches on
  f151409). Its EXPERIMENT_ENV is a LITERAL copy of ARM_ENV from the struct
  screen kernel (submission/_ab_patch_closure/struct/struct-screen.ipynb cell
  12) — the arm that produced the replicated 2.5x plan adoption (2.01/2.59/2.57
  plan-actions per deliberation vs base 1.0) and the uncertified +2 levels/wave.
  Copied literally so this is a faithful transfer test, not a hybrid.

  Armed:   TAAF_STRUCT=1 (patches 21+22), DIFF_LINES=1, WIGGLE=1, DISPATCH=1
  Also differs from the settled patched arm v7/v8 by two knobs the screen set:
           TAAF_ANIMATION 1->0, TAAF_WATCHDOG_STALL_S 600->900
  Dormant: RUN_PROBE, VERIFY, GRAPH, COMPACT, PLAYBOOK, GRID_BURNER (as in the screen)

IDENTITY EVIDENCE (established 2026-08-09, this repo)
  * Commit run of version 9 reached COMPLETE at 20:01Z and its log printed
    "[duck-patch] experiment pins: [('TAAF_ANIMATION', '0'), ...]" plus
    "patch21 struct: OK (TAAF_STRUCT=1 - plan channel armed)" and
    "patch22 gates: OK" — wiring proven. Per the standing law, a COMPLETE
    commit proves wiring and NEVER serving; serving happens only in the
    scored rerun.
  * /api/v1/kernels/pull?...&version_label=v9 returns code cells whose
    canonical hash ("\n".join(code-cell sources), the submission-ledger
    method, NO trailing newline) is b48f64c9f47bc0e4... == the tracked local
    notebook, byte-for-byte.
  * /api/v1/kernels/output?version_label=v9 storage URLs embed /kf/341311881/
    — Kaggle's own record that VERSION 9 IS scriptVersionId 341311881.

READING RULE (pre-registered, binding — write this down before the score lands)
  The base band is 0.69-1.30 (identical bytes, n=9, mean 0.970 sd 0.220). A
  single draw INSIDE that band is NOT evidence in either direction. Only a
  result outside 0.69-1.30 is individually actionable.
  CAVEAT that must travel with the reading: v9 is a patched-family kernel, and
  the patched family's own five draws were 0.76/0.78/0.80/0.93/0.67 (mean
  0.788) — BELOW the base mean. So "in the base band" is not the same as "no
  better than the patched baseline", and a mid-band draw is genuinely
  ambiguous between the two references. This probe cannot resolve that at n=1;
  it is a cheap transfer check, not the certification A/B.

Hard guards, in order, all fatal on failure:
  1. ONE-SHOT WINDOW — refuses to run outside 2026-08-10 00:01-02:00 UTC (after
     the sleep), so a stale relaunch can never fire on a later day.
  2. IDENTITY RE-ATTEST — version_label=v9 pull must hash to the attested
     b48f64c9... under the ledger canonical method AND report version 9; the
     v9 output listing must still bind /kf/341311881/; the tracked local
     notebook must hash identically; the armed pins must still be present in
     the remote bytes; kernel status must read COMPLETE.
  3. SLOT RACE — aborts if any submission already exists in the new UTC day.
  4. Everything else is submit_gated.py's job (builder honesty, COMPLETE+settle,
     never-played watch).

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
KERNEL = "ahmedmobasher86/arc-agi-3-duck-patched"
EXPECTED_VERSION = 9
EXPECTED_SCRIPT_VERSION_ID = "341311881"
# sha256 over "\n".join(code-cell sources) — the submission-ledger canonical
# method (NO trailing newline; session attestations use the trailing-newline
# variant, which for these bytes is b97b9f06bc61cfe6...).
EXPECTED_HASH = "b48f64c9f47bc0e4d8da0934a21000ff8683914a2dcda8fc2e009b42d68c3112"
LOCAL_NOTEBOOK = REPO / "submission/_duck_patched/duck-patched.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 10, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 10, 2, 0, 0, tzinfo=timezone.utc)

# Pins that MUST still be present in the remote v9 bytes at fire time. If a
# rebuild ever silently disarms the experiment, the hash check already catches
# it; this makes the failure legible instead of a bare hash mismatch.
REQUIRED_PINS = {
    "TAAF_STRUCT": "1",
    "TAAF_DIFF_LINES": "1",
    "TAAF_WIGGLE": "1",
    "TAAF_DISPATCH": "1",
    "TAAF_ANIMATION": "0",
    "TAAF_WATCHDOG_STALL_S": "900",
}

MESSAGE = (
    "Structural plan channel, first live probe [struct-v9-b48f64c9]: duck-patched "
    "v9 (scriptVersionId 341311881) with TAAF_STRUCT=1 (patch21 plan channel + "
    "patch22 A-not-B brake and scout/commit phase gate), plus diff-lines, wiggle "
    "and dispatch — EXPERIMENT_ENV copied literally from the struct screen arm. "
    "HYPOTHESIS: the offline struct effect transfers to the scored run — replicated "
    "2.5x plan adoption (2.01/2.59/2.57 vs base 1.0 across 3 waves) and a "
    "directional, UNCERTIFIED +2 levels/wave (struct 17/12/12 vs base 11/12). "
    "READING RULE, pre-registered: base band is 0.69-1.30 (identical bytes, n=9, "
    "mean 0.970 sd 0.220) — only a draw OUTSIDE that band is individually "
    "actionable; the patched family's own draws averaged 0.788, so a mid-band "
    "result is ambiguous between the two references and certifies nothing. n=1 "
    "transfer check, not the 4-6 wave certification A/B."
)


def log(msg: str) -> None:
    print(f"[struct-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


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


def code_source(nb: dict) -> str:
    return "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
    )


def reattest() -> None:
    owner, slug = KERNEL.split("/")
    base = "https://www.kaggle.com/api/v1/kernels"

    # 1. Version-9 bytes: version_label pull must hash to the attestation and
    #    self-report version 9.
    pulled = api_json(
        f"{base}/pull?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    version = pulled["metadata"].get("currentVersionNumber")
    if version != EXPECTED_VERSION:
        raise SystemExit(f"ABORT: version_label pull reports version {version} != {EXPECTED_VERSION}")
    remote_nb = json.loads(pulled["blob"]["source"])
    remote_hash = canonical_code_hash(remote_nb)
    if remote_hash != EXPECTED_HASH:
        raise SystemExit(
            f"ABORT: v{EXPECTED_VERSION} code hash {remote_hash[:16]}… != attested {EXPECTED_HASH[:16]}…"
        )

    # 2. The experiment is actually armed in the bytes we are about to submit.
    remote_src = code_source(remote_nb)
    disarmed = [
        f"{k}={v}" for k, v in sorted(REQUIRED_PINS.items())
        if f'_os.environ["{k}"] = "{v}"' not in remote_src
    ]
    if disarmed:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} is missing armed pins: {disarmed}")

    # 3. Version number <-> scriptVersionId: the v9 output listing's storage
    #    URLs must still embed /kf/341311881/ (Kaggle's own binding).
    output = api_json(
        f"{base}/output?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    kf_ids = sorted(set(re.findall(r"/kf/(\d+)/", json.dumps(output))))
    if kf_ids != [EXPECTED_SCRIPT_VERSION_ID]:
        raise SystemExit(
            f"ABORT: v{EXPECTED_VERSION} output binds kf ids {kf_ids} != [{EXPECTED_SCRIPT_VERSION_ID}]"
        )

    # 4. Local tracked notebook is those same bytes.
    local_nb = json.loads(LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != EXPECTED_HASH:
        raise SystemExit("ABORT: local tracked notebook no longer matches the attested hash")

    # 5. Kernel liveness (latest session; the submitted version is pinned by -v).
    status = subprocess.run(
        ["python3", "-m", "kaggle", "kernels", "status", KERNEL],
        capture_output=True, text=True, timeout=120,
    )
    combined = (status.stdout + status.stderr).upper()
    if "COMPLETE" not in combined:
        raise SystemExit(f"ABORT: kernel status not COMPLETE: {combined.strip()[:200]}")
    log(
        f"re-attest OK: v{EXPECTED_VERSION} == scriptVersionId {EXPECTED_SCRIPT_VERSION_ID}, "
        f"hash {remote_hash[:16]}…, {len(REQUIRED_PINS)} pins armed, remote==local, COMPLETE"
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
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        log("WARNING: certifi unavailable — API re-attest and post-submit watch may be SSL-blind")

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
    log("invoking: " + " ".join(cmd[:8]) + " …")
    proc = subprocess.run(cmd, cwd=REPO)
    log(f"submit_gated exit code: {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
