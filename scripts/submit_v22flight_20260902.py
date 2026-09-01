#!/usr/bin/env python3
"""submit_v8harvest_20260830.py — one-shot armed runner for the 2026-08-30 00:01 UTC
slot (duck38-v8: the crack-or-nothing search arm, first flight).

WHAT duck38-v8 IS
  The duck38-v12 base — Aug-07 community anim bundle + Qwen3.8-27B-FP8
  repacked, boot-attested, the bytes that flew 1.55 — plus exactly ONE
  inserted cell before the run cell: the explorer-v8 six-file SearchCore
  bundle embedded inline and install() behind a HARD assert on all five
  halves of the verdict (explorer / v8 / early / prescreen /
  stop-after-crack). effort_medium is ABSENT and purged (reverted as harmful
  live, pooled 1.12 vs 1.55).

  Flight config, every flag set explicitly:
    EXPLORER_V8=1  EXPLORER_V8_EARLY=1  EXPLORER_V8_EARLY_ENGAGE=1
    EXPLORER_V8_PRESCREEN=1  EXPLORER_V8_ENGAGE_SPECIALISTS=ft09_gf2
    EXPLORER_V8_SPECIALIST_MAX_ACTIONS=4000  EXPLORER_V8_STALL_GRIND=0
    EXPLORER_V8_BANK=1  EXPLORER_V8_STOP_AFTER_CRACK=1

WHY IT IS SHAPED THIS WAY (all measured)
  * A zero-action frame-0 PRE-SCREEN decides, from the observation the harness
    already holds at game start, whether a game is even plausibly of an
    engageable class. On the 25 dev fixtures: 1 true positive (ft09), 0 false
    negatives, 0 false positives, 0 engine actions. Mean probe cost per game
    falls 125.4 -> 3.6 actions; 24 of 25 games pay ZERO.
  * It engages ONLY the ft09_gf2 class, the one measured to CRACK its game.
    Detection is not a crack predictor, and a failed engagement's actions are
    billed into the same play the LLM keeps using (only a post-WIN reset
    escapes the competition guard, api.py:316-334).
  * A crack is BANKED by replaying the win on a fresh play; the engine scores
    the MAX over plays.
  * STALL_GRIND is OFF: partial grinding is measured net-negative (smoke #2
    spent 292 687 / 292 635 engine actions on dc22 / sk48 and cracked
    nothing).
  * EV +3.43 points/game at p = 4 %, with the downside floored at 0.00 by
    measurement — a declined game spends nothing at all.

IDENTITY EVIDENCE
  * arc3-v8-smoke version 4 (RTX Pro 6000, COMPLETE, 2026-08-27) ran this
    exact configuration and cleared all five pre-registered bars on the ENGINE
    SCORECARD: pre-screen declined vc33/dc22/sk48 at exactly 0 probe actions;
    ft09 detected in 89 probe actions, cracked 6/6 in 1233 engine actions,
    banked a 75-action replay, engine score 100.0 with plays [3.512, 100.0];
    every envelope guard inside cap; no crash.
  * arc3-duck38-v8 version 1 committed COMPLETE on RTX Pro 6000. A COMPLETE
    commit proves wiring, never scored serving (TRUE_SUBMISSION is false
    during commits).
  * /api/v1/kernels/pull?...&version_label=v1 code cells hash (ledger
    canonical method) to the constant below == the tracked local notebook.
  * /api/v1/kernels/output?version_label=v1 storage URLs embed the
    scriptVersionId below — Kaggle's own record of which version this is.

READING RULE (pre-registered, binding)
  THIS IS A LOTTERY TICKET WITH A ZERO FLOOR. The arm can only add score on a
  game of the ft09_gf2 class; the public half of the hidden set may contain
  none. A NULL DRAW IN THE BASE BAND (0.69-1.30) IS THE EXPECTED OUTCOME IF
  NO SUCH GAME IS THERE, and is NOT evidence against the mechanism — it is
  evidence about the hidden set's composition. Above 1.74 re-banks the LB
  best and proves the class is present and bankable live. Below the band would
  mean the probe is costing more than the measurement says and is the only
  result that falsifies the shape.

Hard guards, in order, all fatal on failure:
  1. ONE-SHOT WINDOW — refuses to run outside 2026-09-01 00:01-23:30 UTC.
     (WIDENED: the 02:00 end was left at 08-28 when the target moved to
     08-29, so the runner woke on time and aborted itself. --mock skips
     this check, which is exactly why the mock did not catch it.)
  2. IDENTITY RE-ATTEST — version_label=v1 pull must hash to the attested
     constant and report version 1; output listing must bind the attested
     scriptVersionId; the local tracked notebook must hash identically; the
     required markers must be present and the forbidden ones absent; kernel
     status COMPLETE.
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
KERNEL = "ahmedmobasher86/arc3-v22-flight-tp9"
EXPECTED_VERSION = 1
EXPECTED_SCRIPT_VERSION_ID = "346574220"
# sha256 over "\n".join(code-cell sources) — submission-ledger canonical method.
# Local build hash 2026-09-01; re-attested against the remote v1 pull by main().
EXPECTED_HASH = "26b5db7847c388db5bc4156209d2f247d80e0a7bfd237aaa64824140287aedcb"
LOCAL_NOTEBOOK = REPO / "submission/_v22_flight/arc3-v22-flight-tp9.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 9, 2, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 9, 2, 23, 30, 0, tzinfo=timezone.utc)
MARKER = REPO / "logs/v22flight_20260902.marker"

# Both directions can actually fail: the V31 serving block must be present AND
# the bytes must still be the public-lane copy (no grafts, no Flash-Next).
REQUIRED_MARKERS = (
    'num_speculative_tokens',
    '--kv-cache-dtype',
    '_v31_start_watchdog',
    'KAGGLE_IS_COMPETITION_RERUN',
    '_competition_games',
    "{'TP9_ENABLE': '1'}",
    'grafts installed',
    'graft_pipeline',
)
FORBIDDEN_MARKERS = (
    'os.environ["EXPLORER_V8"] = "1"',
    'Flash-Next',
    "{'TP10_ENABLE': '1'}",
)

MESSAGE = (
    "HARNESS LEVER FLIGHT arc3-v22-flight-tp9: the V22 serving stack (fp8 KV + MTP-3 + async, official "
    "Qwen3.8-27B-FP8, stock June agent code) + exactly ONE adversarially-verified graft: TP9 turn-pipeline "
    "repair (resume truncated turns, livelock breaker, budget-aware retry; two independent judge SHIP "
    "verdicts). Same-boot 25v25 A/B on this exact stack (kernel arc3-v22-ab-tp9 v3): stock 4.27 score / "
    "1.04 levels / 8 zero-level vs TP9 6.42 / 1.56 / 3 zero-level - +50%% levels, 11 games gained (sc25 +3, "
    "sb26 +3), 2 lost 1. READING RULE (pre-registered): baseline = the same stack's live draw 1.71 "
    "(sub 55927189); >=2.2 = lever transfers live, new floor and the lever ladder continues; 1.7-2.2 = "
    "inconclusive single draw (CV 0.17-0.20), redraw before judging the lever; <1.4 = lever hurts live or "
    "boot/serving failure - pull the log before any conclusion."
)


def log(msg: str) -> None:
    print(f"[v8-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


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

    if "FILL_AFTER_COMMIT" in (EXPECTED_HASH, EXPECTED_SCRIPT_VERSION_ID) \
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
