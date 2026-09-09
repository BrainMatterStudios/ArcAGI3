#!/usr/bin/env python3
"""submit_keithcopy_20260910.py — one-shot armed runner for the 2026-09-10 00:01 UTC slot.

WHAT FLIES
  arc3-keith-copy v4 — the byte-copy of keithtyser V14 (stock June duck agent
  code, Flash-Next NVFP4, serving profile kv5-bf16-mtp3-c8-cg32) with the
  both-mount install cell. NO CODE CHANGE. Byte-identical to subs 56042273
  (3.25) and 56058136 (2.58).

WHY A BASE DRAW AND NOT A CANDIDATE (the honest reason)
  Nothing passed a rig gate. As of 2026-09-09 every lever in Track A is dead:
    * A1 carry+compact      41 lv vs pooled base 39.33 (+0.71 sd)      -> dead
    * A2 workspace+verifier 0 verifier calls in 358                    -> closed
    * A3 NOOA               violates the clock contract                -> dead
    * A4 Polyphony          one verified model = 79% of a game's decode -> dead
    * CADENCE conc 6/1767 s ENGAGED every gate, read 34 lv = -2.28 sd  -> dead
  Plus the KV10 elasticity (+47% calls -> +19% levels) closes the whole
  throughput family as a source of a step. There is no candidate to fly.

  This slot therefore does the ONE thing a base draw legitimately does: it
  tightens the live null that every future candidate is read against. The base
  family is n=5 (3.25, 2.58, 4.31, 2.45, 4.01), mean 3.32, sd 0.83, SE 0.37.
  A sixth draw takes SE to ~0.34. That is a small gain and it is the whole
  claim -- this draw is NOT evidence about any lever.

READING RULE (pre-registered, binding, written before the draw)
  2.0-4.7 (mean +/- 1.65 sd) = IN BAND: pool into the base family, update the
  mean, no verdict on anything. >4.7 or <2.0 = pull the commit log and the
  vLLM log before reading; a single draw outside the band on IDENTICAL bytes is
  an instrument question, not a result. CV on identical bytes is ~0.25, so no
  single draw can move any decision by itself.

Hard guards (unchanged from the 09-07 runner, all fatal):
  1. ONE-SHOT WINDOW  2026-09-10 00:01-23:55 UTC
  2. IDENTITY RE-ATTEST  v4 pull hashes to the attested constant, binds
     scriptVersionId 347562879, required markers present / forbidden absent,
     remote == local tracked notebook, kernel COMPLETE
  3. SLOT RACE  aborts if any submission already exists in the new UTC day
  4. submit_gated.py  builder honesty, COMPLETE+settle, never-played watch
  5. MARKER  logs/keithcopy_20260910.marker, released only if no submission landed

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
KERNEL = "ahmedmobasher86/arc3-keith-copy"
EXPECTED_VERSION = 4
EXPECTED_SCRIPT_VERSION_ID = "347562879"  # 09-05: accepted from the completed commit (exactly one kf id), logged + written to logs/keithcopy_20260910.svid
# sha256 over "\n".join(code-cell sources) — submission-ledger canonical method.
# Attested 2026-08-31 ~15:35Z: remote v1 pull == local notebook == this hash.
EXPECTED_HASH = "90efdebbf8dcbe204f54f286cf8ed0733b400219b0550c5bf9b3694b82bdab59"
LOCAL_NOTEBOOK = REPO / "submission/_keith_copy/push_bothmounts/arc3-keith-copy.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 9, 10, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 9, 10, 23, 55, 0, tzinfo=timezone.utc)
MARKER = REPO / "logs/keithcopy_20260910.marker"

# Both directions can actually fail: the V31 serving block must be present AND
# the bytes must still be the public-lane copy (no grafts, no Flash-Next).
REQUIRED_MARKERS = (
    'kv5-bf16-mtp3-c8-cg32',
    'TAAF_VLLM_MTP_TOKENS',
    'ONLY_RESET_LEVELS',
    'KAGGLE_IS_COMPETITION_RERUN',
    '_resolve_comp_root',
)
FORBIDDEN_MARKERS = (
    '_tpmod.install()',
    'TP9_ENABLE',
    'RETRY_ENABLE',
    'ARC3_KV_CACHE_MEMORY_BYTES',
)

MESSAGE = (
    "BASE DRAW #4 (family n=5 -> 6) arc3-keith-copy v4 [svid 347562879, hash 90efdebb, BYTE-IDENTICAL to subs 56042273 (3.25) and 56058136 (2.58)]: byte-copy of keithtyser V14, stock June duck agent code, Flash-Next NVFP4, profile kv5-bf16-mtp3-c8-cg32, both-mount install cell. NO CODE CHANGE. WHY A BASE DRAW: nothing passed a rig gate - A1 carry 41 lv (+0.71 sd), A2 workspace 0 uptake in 358 calls, NOOA and Polyphony gated out on clock and compute, and the 09-09 CADENCE arm (conc 6 / 1767 s) ENGAGED every pre-registered gate (e2e 26.6 s, calls/turn 1.60, calls/game 56.8) and still read 34 lv = -2.28 sd. With the KV10 elasticity (+47 pct calls -> +19 pct levels) the throughput family is closed as a step. So this slot only tightens the live null: family n=5 (3.25/2.58/4.31/2.45/4.01) mean 3.32 sd 0.83 SE 0.37 -> SE 0.34. READING RULE (pre-registered): 2.0-4.7 = in band, pool and update the mean, NO verdict on any lever; outside = pull the commit and vLLM logs, it is an instrument question. CV on identical bytes 0.25."
)


def log(msg: str) -> None:
    print(f"[keithcopy-20260910 {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


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

    # 09-05: the commit may still be running at TARGET; wait for COMPLETE (leave 12 min for the
    # settle + submit before WINDOW_END) instead of aborting.
    deadline = WINDOW_END_UTC - timedelta(minutes=12)
    while True:
        st = kernel_status_positive()
        if st and "COMPLETE" in st.upper():
            break
        if st and any(k in st.upper() for k in ("ERROR", "CANCEL")):
            raise SystemExit(f"ABORT: kernel ended {st} — nothing to submit")
        if datetime.now(timezone.utc) >= deadline:
            raise SystemExit(f"ABORT: kernel still {st} at {datetime.now(timezone.utc):%H:%M}Z — no time left in the slot")
        log(f"kernel status {st} — waiting for COMPLETE (deadline {deadline:%H:%M}Z)")
        time.sleep(60)

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
    global EXPECTED_SCRIPT_VERSION_ID
    if EXPECTED_SCRIPT_VERSION_ID == "AUTO":
        if len(kf_ids) != 1:
            raise SystemExit(f"ABORT: expected exactly one kf id in v{EXPECTED_VERSION} output, got {kf_ids}")
        EXPECTED_SCRIPT_VERSION_ID = kf_ids[0]
        (REPO / "logs/keithcopy_20260910.svid").write_text(EXPECTED_SCRIPT_VERSION_ID + "\n")
        log(f"AUTO-attested scriptVersionId {EXPECTED_SCRIPT_VERSION_ID} from the completed commit output")
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

    if "FILL_AFTER_COMMIT" in (EXPECTED_HASH,) \
            or "__PENDING__" in (EXPECTED_HASH, EXPECTED_SCRIPT_VERSION_ID):
        raise SystemExit("ABORT: runner not attested — EXPECTED_HASH / "
                         "EXPECTED_SCRIPT_VERSION_ID still placeholders")

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
