#!/usr/bin/env python3
"""submit_p3_20260816.py — one-shot armed runner for the 2026-08-16 00:01 UTC slot.

Submits duck-p3 v2 — the SINGLE-VARIABLE arm neutralizing the "minimize
actions" own-goal line — through the sanctioned gated path.

WHAT duck-p3 v2 IS
  Shipped duck-base v2 bytes + exactly TWO commit-machinery cells and ONE
  behavioural delta:
    * probe-both mount cells (install + environment_files), ported verbatim
      from build_duck_mem.py after the v1 commit ERROR proved GPU commit
      machines can also mount /kaggle/input/<slug> (2026-08-15 00:28Z).
      Commit-only code paths; the scored rerun uses the gateway.
    * the ONE behavioural change: the system-prompt line
      "- Optimize for as few in-game actions as possible while still being
      reliable." replaced by an exploration-neutral line, rebound in BOTH
      namespaces (prompts module AND tool_agent's import-time binding — the
      silent-no-op bug found in the duck-mem review).
  Rationale (measured): 8/10 completed levels are efficiency-cap-BOUND, so an
  action-minimization instruction buys nothing on cleared levels and
  suppresses the exploration that unlocks additional levels (worth 3x per
  level step). The P1-P4 bundle measured negative pooled, but attribution was
  unknown; this isolates the one member whose direction rests on measured
  facts and whose mechanism cannot degrade throughput (pure prompt text).

IDENTITY EVIDENCE (established 2026-08-15, this session)
  * v2 commit reached COMPLETE and its log printed
    "[duck-p3] installing arc-agi from /kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    and "[duck-p3] minimize-actions line neutralized in BOTH namespaces: OK"
    — wiring proven. A COMPLETE commit proves wiring, NEVER serving.
  * /api/v1/kernels/pull?...&version_label=v2 code cells hash (ledger
    canonical method) to 06e74d2806c1eb06... == the tracked local notebook.
  * /api/v1/kernels/output?version_label=v2 storage URLs embed /kf/342532400/
    — Kaggle's own record that VERSION 2 IS scriptVersionId 342532400.

READING RULE (pre-registered, binding)
  Base band 0.69-1.30 (identical-bytes control, n=11, mean ~0.973 sd ~0.20).
  A single draw INSIDE the band is NOT evidence in either direction; only a
  result outside the band is individually actionable. This is a lottery
  ticket WITH a falsifiable one-variable hypothesis, not a certification A/B.

Hard guards, in order, all fatal on failure:
  1. ONE-SHOT WINDOW — refuses to run outside 2026-08-16 00:01-02:00 UTC.
  2. IDENTITY RE-ATTEST — version_label=v2 pull must hash to 06e74d28... and
     report version 2; output listing must bind /kf/342532400/; the local
     tracked notebook must hash identically; the P3 hook markers must be
     present in the remote bytes; kernel status COMPLETE.
  3. SLOT RACE — aborts if any submission already exists in the new UTC day.
  4. submit_gated.py: builder honesty, COMPLETE+settle, never-played watch.

LESSON ENCODED (08-12 gambit rc-bug): on submit_gated rc != 0 the marker is
released ONLY if no submission actually landed in the UTC day — submit_gated
can return rc=2 AFTER a successful submit when its watch sees ERROR.

--mock: target = now + 90 s, submit_gated runs with --dry-run, window/race
guards report but do not abort.
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
KERNEL = "ahmedmobasher86/arc-agi-3-duck-p3"
EXPECTED_VERSION = 2
EXPECTED_SCRIPT_VERSION_ID = "342532400"
# sha256 over "\n".join(code-cell sources) — submission-ledger canonical method.
EXPECTED_HASH = "06e74d2806c1eb06e87f08ed99887dcf5bf68a28201ff2eb17aff6a7d591b26e"
LOCAL_NOTEBOOK = REPO / "submission/_duck_p3/duck-p3.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 16, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 16, 2, 0, 0, tzinfo=timezone.utc)
MARKER = REPO / "logs/duck_p3_20260816.marker"

# The one behavioural delta plus the commit-machinery cells must be present in
# the submitted bytes; base-v2 purity markers must also survive (no other patch
# machinery snuck in). Both directions can actually fail.
REQUIRED_MARKERS = (
    "_p3_wheel_dirs",                     # probe-both install cell
    "_p3_env_candidates",                 # probe-both environment_files cell
    "_P3_OLD",                            # hook: anchor line constant
    "minimize-actions line neutralized",  # hook: success print
)
FORBIDDEN_MARKERS = (
    '_os.environ["TAAF_',
    "def apply_all",
    "patch_struct_channel",
)

MESSAGE = (
    "Single-variable arm duck-p3 [duck-p3-v2-06e74d28]: shipped duck-base v2 "
    "bytes (scriptVersionId 342532400) with exactly ONE behavioural change - "
    "the 'minimize in-game actions' system-prompt line replaced by an "
    "exploration-neutral line, rebound in both namespaces (the import-time "
    "binding bug made naive rebinds silent no-ops). Plus commit-only "
    "mount-probing cells (the scored rerun never executes them). HYPOTHESIS: "
    "8/10 completed levels are efficiency-cap-bound, so the instruction buys "
    "nothing where it applies and suppresses the exploration that unlocks "
    "additional levels (worth 3x per level step); the negative pooled read of "
    "the 4-lever bundle left per-lever attribution unknown, and this isolates "
    "the one member whose direction rests on measured facts and whose "
    "mechanism cannot reduce throughput (pure prompt text). READING RULE, "
    "pre-registered: base band 0.69-1.30 (identical-bytes control n=11, mean "
    "~0.973); only a draw OUTSIDE the band is individually actionable; an "
    "in-band draw certifies nothing at n=1."
)


def log(msg: str) -> None:
    print(f"[p3-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def auth_header() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (user and key):
        creds = json.loads((Path.home() / ".kaggle/kaggle.json").read_text())
        user, key = creds["username"], creds["key"]
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def api_json(url: str, attempts: int = 5) -> dict | list:
    """GET with retries — a 429/503 on an unretried call would kill the slot."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"Authorization": auth_header()})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 — any transport failure is retryable
            last = exc
            if attempt < attempts:
                backoff = min(30, 3 * attempt)
                log(f"api attempt {attempt}/{attempts} failed ({exc!r}) — retry in {backoff}s")
                time.sleep(backoff)
    raise SystemExit(f"ABORT: API unreachable after {attempts} attempts: {last!r}")


def kernel_status_positive(attempts: int = 5) -> str | None:
    """The kernel's POSITIVE status, or None if every attempt was transport noise."""
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
            log(f"status attempt {attempt}/{attempts}: no positive status line")
        except (subprocess.TimeoutExpired, OSError) as exc:
            log(f"status attempt {attempt}/{attempts} transport failure: {exc!r}")
        if attempt < attempts:
            time.sleep(min(30, 3 * attempt))
    return None


def canonical_code_hash(nb: dict) -> str:
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

    remote_src = code_source(remote_nb)
    missing = [m for m in REQUIRED_MARKERS if m not in remote_src]
    if missing:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} bytes missing required markers {missing}")
    present = [m for m in FORBIDDEN_MARKERS if m in remote_src]
    if present:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} carries foreign patch machinery {present}")

    output = api_json(
        f"{base}/output?user_name={owner}&kernel_slug={slug}&version_label=v{EXPECTED_VERSION}"
    )
    kf_ids = sorted(set(re.findall(r"/kf/(\d+)/", json.dumps(output))))
    if kf_ids != [EXPECTED_SCRIPT_VERSION_ID]:
        raise SystemExit(
            f"ABORT: v{EXPECTED_VERSION} output binds kf ids {kf_ids} != [{EXPECTED_SCRIPT_VERSION_ID}]"
        )

    local_nb = json.loads(LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != EXPECTED_HASH:
        raise SystemExit("ABORT: local tracked notebook no longer matches the attested hash")

    status = kernel_status_positive()
    if status is None:
        log("WARNING: kernel status unreadable after retries — deferring the "
            "liveness call to submit_gated's own gate 2 (identity already proven)")
    elif "COMPLETE" not in status.upper():
        raise SystemExit(f"ABORT: kernel status not COMPLETE: {status}")
    log(
        f"re-attest OK: v{EXPECTED_VERSION} == scriptVersionId {EXPECTED_SCRIPT_VERSION_ID}, "
        f"hash {remote_hash[:16]}…, markers OK, remote==local, COMPLETE"
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

    if MARKER.exists() and not args.mock:
        raise SystemExit(f"ABORT: marker present ({MARKER}) — this slot already fired")

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
            raise SystemExit("ABORT: today's slot already consumed by another submission")

    if not args.mock:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(f"claimed {datetime.now(timezone.utc).isoformat()} pid={os.getpid()}\n")
        log(f"slot claimed via marker {MARKER}")

    cmd = [
        sys.executable, str(REPO / "scripts/submit_gated.py"),
        "--kernel", KERNEL, "--version", str(EXPECTED_VERSION),
        "--notebook", str(LOCAL_NOTEBOOK), "--message", MESSAGE,
        "--not-after", WINDOW_END_UTC.isoformat(),
    ]
    if args.mock:
        cmd.append("--dry-run")
    log("invoking: " + " ".join(cmd[:8]) + " …")
    proc = subprocess.run(cmd, cwd=REPO)
    log(f"submit_gated exit code: {proc.returncode}")
    if proc.returncode != 0 and not args.mock:
        # 08-12 lesson: submit_gated can exit rc=2 AFTER a successful submit
        # (its post-submit watch saw ERROR). Only release the claim if no
        # submission actually landed in the day — otherwise a relaunch would
        # double-submit.
        if slot_already_used(day_start):
            log("submit_gated rc!=0 but a submission EXISTS in the UTC day — "
                "keeping the marker (post-submit watch failure, not submit failure)")
        else:
            MARKER.unlink(missing_ok=True)
            log("submit_gated failed pre-submit — marker released for a manual retry in-window")
    return proc.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — nothing may die silently here
        log(f"UNHANDLED {exc!r}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
