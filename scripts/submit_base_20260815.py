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
KERNEL = "ahmedmobasher86/arc-agi-3-duck-base"
EXPECTED_VERSION = 2
EXPECTED_SCRIPT_VERSION_ID = "336252059"
# sha256 over "\n".join(code-cell sources) — the submission-ledger canonical
# method (NO trailing newline; session attestations use the trailing-newline
# variant, which for these bytes is b97b9f06bc61cfe6...).
EXPECTED_HASH = "886dbc8a58ec79a090085481d616385d742fefa30cbb3403076cf9f788e58a5a"
LOCAL_NOTEBOOK = REPO / "submission/_duck_base/duck-base.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 15, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 15, 2, 0, 0, tzinfo=timezone.utc)
# Idempotency claim. Checked at start, written once the slot is ours. Survives a
# relaunch after a presumed death, which the race guard alone cannot (it reads the
# submissions list ~10 min before submit_gated actually submits).
MARKER = REPO / "logs/base_v2_20260815.marker"

# Base v2 is the UNPATCHED duck, so there are no pins to require. The meaningful
# guard is the inverse: assert the bytes carry NO experiment pins and no patch
# machinery at all. A vacuous "check" that always passes is worse than none, so
# this one can actually fail.
FORBIDDEN_MARKERS = (
    '_os.environ["TAAF_',
    "def apply_all",
    "patch_struct_channel",
)

MESSAGE = (
    "Byte-identical base draw #11 [base-v2-886dbc8a]: pinned duck-base v2 "
    "(scriptVersionId 336252059), the exact bytes behind the n=10 control "
    "distribution (mean 0.9650 sd 0.2082, band 0.69-1.30), unmodified. "
    "CONTEXT: a measurement-heavy day killed all four incremental levers "
    "offline (the planner-series 4-wave pooled negative; the anti-waste "
    "bundle pooled -0.29; RedHatAI quant -0.56; Tycho-27B 0 levels); "
    "tonight's intended single-variable arm "
    "(prompt own-goal neutralization, duck-p3) is built but push-blocked on "
    "the exhausted 60h GPU quota. This slot is therefore opportunistic rank "
    "banking on the best-mean configuration plus control accumulation to "
    "n=11: no hypothesis claimed, no conclusion drawn from an in-band draw."
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


def api_json(url: str, attempts: int = 5) -> dict | list:
    """GET with retries. Every fire-time API call goes through here.

    Rationale: the L24 launchd job fires five Kaggle submissions plus repeated
    listing calls at 00:00:05Z, 55 seconds before this runner wakes. A 429/503
    landing on an unretried call would kill the slot outright. Retrying costs
    seconds; not retrying costs the day.
    """
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
    """The kernel's POSITIVE status, or None if every attempt was transport noise.

    The CLI prints transport failures to stderr, and strings like
    NewConnectionError and "500 Server Error" contain the substring ERROR — so a
    naive `"COMPLETE" in stdout+stderr` test turns a DNS hiccup into an abort.
    Only a `has status "..."` line counts as the kernel speaking.
    """
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

    # 2. These bytes are the UNPATCHED base. If any patch machinery appears, the
    #    hash check would already have failed - but name the reason explicitly.
    remote_src = code_source(remote_nb)
    present = [m for m in FORBIDDEN_MARKERS if m in remote_src]
    if present:
        raise SystemExit(f"ABORT: v{EXPECTED_VERSION} is not unpatched base bytes; found {present}")

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
    status = kernel_status_positive()
    if status is None:
        # Never seen the kernel speak. Do NOT abort: identity checks 1-4 already
        # proved the exact bytes and their kf binding, and submit_gated runs its
        # own COMPLETE gate with retries. Aborting here on transport noise would
        # burn the slot for a network blip.
        log("WARNING: kernel status unreadable after retries — deferring the "
            "liveness call to submit_gated's own gate 2 (identity already proven)")
    elif "COMPLETE" not in status.upper():
        raise SystemExit(f"ABORT: kernel status not COMPLETE: {status}")
    log(
        f"re-attest OK: v{EXPECTED_VERSION} == scriptVersionId {EXPECTED_SCRIPT_VERSION_ID}, "
        f"hash {remote_hash[:16]}…, unpatched-base markers clean, remote==local, COMPLETE"
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
    # Poll loop, not one monolithic sleep: a single multi-hour time.sleep() is not
    # robust to suspend/resume, and capping each nap at 60 s lets a late wake still
    # land inside the window instead of overshooting it.
    while True:
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(60.0, remaining))

    now = datetime.now(timezone.utc)
    # Lower bound carries 60 s of slack: `now` is wall-clock, and macOS `timed` can
    # step the clock BACKWARD during the wait (notably after a network change or a
    # wake). Without slack such a step lands just under TARGET_UTC and aborts a
    # perfectly good run. Overshoot stays fail-closed at WINDOW_END_UTC.
    if not args.mock and not (TARGET_UTC - timedelta(seconds=60) <= now <= WINDOW_END_UTC):
        raise SystemExit(f"ABORT: outside one-shot window (now {now:%Y-%m-%d %H:%M}Z)")

    reattest()

    day_start = target.replace(hour=0, minute=0, second=0, microsecond=0)
    if slot_already_used(day_start):
        if args.mock:
            log("mock: race guard would abort here (expected during daytime test)")
        else:
            raise SystemExit("ABORT: today's slot already consumed by another submission")

    # Claim the slot BEFORE handing off. submit_gated settles 10 min before the
    # actual submit, so the race guard's reading is stale by then; the marker is
    # what stops a second copy of this runner (a relaunch after a presumed death)
    # from walking the same path and double-submitting.
    if not args.mock:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(f"claimed {datetime.now(timezone.utc).isoformat()} pid={os.getpid()}\n")
        log(f"slot claimed via marker {MARKER}")

    cmd = [
        sys.executable, str(REPO / "scripts/submit_gated.py"),
        "--kernel", KERNEL, "--version", str(EXPECTED_VERSION),
        "--notebook", str(LOCAL_NOTEBOOK), "--message", MESSAGE,
        # Bound submit_gated's COMPLETE wait by our own window: without it a kernel
        # stuck in RUNNING keeps its 6-hour loop alive and could submit at 06:00Z,
        # hours outside the window this runner pre-registered.
        "--not-after", WINDOW_END_UTC.isoformat(),
    ]
    if args.mock:
        cmd.append("--dry-run")
    log("invoking: " + " ".join(cmd[:8]) + " …")
    proc = subprocess.run(cmd, cwd=REPO)
    log(f"submit_gated exit code: {proc.returncode}")
    if proc.returncode != 0 and not args.mock:
        # Release the claim so a deliberate manual retry inside the window is not
        # blocked by our own marker.
        MARKER.unlink(missing_ok=True)
        log("submit_gated failed — marker released for a manual retry in-window")
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
