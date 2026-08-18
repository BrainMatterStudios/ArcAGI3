#!/usr/bin/env python3
"""submit_gated.py — the ONLY sanctioned Kaggle submission path.

Born from the 2026-08-03 zero diagnosis (docs/DIAGNOSIS-2026-08-03-zero-submission.md):
duck-patched v5 was pushed and submitted ~90-150 s later with no COMPLETE gate; the
gateway scored a 3411-byte never-played parquet as 0.00 inside 16 minutes. Every
guard below maps to a fact from that diagnosis.

Gates, in order:
  1. BUILDER HONESTY — if the submission message names a pack (depth, wmr/watchdog,
     hud/mask, replay, graph, compact, ...), the pack's marker strings must be
     present in the notebook's hook cell. v5's message claimed "on depth-pack v1"
     while the notebook contained no depth code; this gate refuses to let a
     description lie about its own artifact.
  2. KERNEL COMPLETE + SETTLE — `kaggle kernels status` must positively read
     COMPLETE, then a >=10 minute settle is enforced (re-checked after), before
     any submit. Transient empty status replies are retried, never treated as
     failure (the 2026-08-01 lesson), but a positive ERROR/CANCEL aborts.
  3. POST-SUBMIT WATCH — polls the submissions API for up to ~90 minutes and
     ALARMS (exit 2, loud banner) if the submission's total_bytes <= 3500 (the
     3411-byte never-played fingerprint; genuinely-played duck runs are
     3642-3710 bytes) or its status reaches COMPLETE in under 1 hour (a real run
     takes ~9 h; a fast COMPLETE means the games were never played).

Dependency-free: python stdlib + the `kaggle` CLI on PATH (credentials are read
from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json for the byte-size query,
which the CLI's CSV output does not expose).

Usage:
  python scripts/submit_gated.py \
      --kernel ahmedmobasher86/arc-agi-3-duck-patched --version 6 \
      --notebook submission/_duck_patched/duck-patched.ipynb \
      --message "WMR+graph+compact pack ..." \
      [--competition arc-prize-2026-arc-agi-3] [--file submission.parquet]
      [--dry-run]   # run every pre-submit gate, stop before submitting

Exit codes: 0 = submitted and watch window passed clean (or dry-run OK);
            1 = a pre-submit gate refused; 2 = POST-SUBMIT ALARM (investigate NOW).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_COMPETITION = "arc-prize-2026-arc-agi-3"

# The never-played fingerprint. Three independent 0.00 submissions (55197631,
# 54847434, 54344660) all scored a 3411-byte gateway parquet; genuinely-played
# duck runs measure 3642-3710 bytes.
NEVER_PLAYED_MAX_BYTES = 3500

SETTLE_SECONDS = 10 * 60          # gate 2: settle after COMPLETE
WATCH_SECONDS = 90 * 60           # gate 3: total watch window
WATCH_POLL_SECONDS = 5 * 60
FAST_COMPLETE_SECONDS = 60 * 60   # COMPLETE under this = never played

# Builder-honesty markers: message keyword (case-insensitive) -> strings, ANY of
# which must appear in the notebook's source. Keyed on how packs are actually
# named in submission messages.
PACK_MARKERS: dict[str, list[str]] = {
    "depth": ["cross_level_notes", "depth_pack"],
    "watchdog": ["patch_watchdog", "TAAF_WATCHDOG"],
    "wmr": ["patch_watchdog", "patch_hud_board_identity", "patch_win_replay"],
    "hud": ["patch_hud_board_identity", "TAAF_HUD_MASK"],
    "mask": ["patch_hud_board_identity", "TAAF_HUD_MASK"],
    # 2026-08-18: the banking graft implements win-then-replay natively;
    # its install banner is the marker proving the mechanism is in the bytes.
    "replay": ["patch_win_replay", "TAAF_WIN_REPLAY", "[banking] armed"],
    "banking": ["[banking] armed"],
    "graph": ["patch_frontier_graph", "TAAF_GRAPH"],
    "frontier": ["patch_frontier_graph", "TAAF_GRAPH"],
    "grinder": ["patch_frontier_graph", "TAAF_GRAPH"],
    "compact": ["patch_compaction", "TAAF_COMPACT"],
    "compaction": ["patch_compaction", "TAAF_COMPACT"],
    "plan-queue": ["patch_plan_queue", "TAAF_COMPACT"],
    "plan queue": ["patch_plan_queue", "TAAF_COMPACT"],
    "action7": ["patch_action7"],
    "burner": ["patch_dynamic_grid_burner", "TAAF_GRID_BURNER"],
    # patches 16-22 (added 2026-08-09 with the struct v9 probe): the gate only
    # covered packs through patch15, so a message naming any of these went
    # unchecked. Keywords are how these packs are actually named in messages.
    "diff-lines": ["patch_diff_lines", "TAAF_DIFF_LINES"],
    "wiggle": ["patch_wiggle", "TAAF_WIGGLE"],
    "run_probe": ["patch_run_probe", "TAAF_RUN_PROBE"],
    "dispatch": ["patch_archetype_dispatch", "TAAF_DISPATCH"],
    "verifier": ["patch_verify_at_commit", "TAAF_VERIFY"],
    "struct": ["patch_struct_channel", "TAAF_STRUCT"],
    "plan channel": ["patch_struct_channel", "TAAF_STRUCT"],
    "brake": ["patch_struct_gates", "TAAF_STRUCT"],
    "phase gate": ["patch_struct_gates", "TAAF_STRUCT"],
}


def log(msg: str) -> None:
    print(f"[submit-gated {datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def alarm(msg: str) -> None:
    banner = "!" * 78
    print(f"\n{banner}\n!!! SUBMISSION ALARM\n!!! {msg}\n{banner}\n",
          file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------- gates


def notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def check_builder_honesty(message: str, notebook: Path) -> list[str]:
    """Packs named in the message whose markers are absent from the notebook."""
    src = notebook_source(notebook)
    lowered = message.lower()
    missing = []
    for keyword, markers in PACK_MARKERS.items():
        # WHOLE-WORD match, not substring. A bare `keyword in lowered` made short
        # keys into landmines: 'struct' matched "construction", "structural",
        # "instructions", "infrastructure". In a campaign literally named the
        # structural campaign, the next duck-base draw whose message said
        # "structural campaign context" would have tripped this gate against a
        # notebook with no TAAF_STRUCT and REFUSED — burning the slot on a
        # description word. Boundaries make 'struct-v9' match and 'structural' not.
        if not re.search(rf"\b{re.escape(keyword)}\b", lowered):
            continue
        if not any(marker in src for marker in markers):
            missing.append(f"{keyword!r} (needs one of {markers})")
    return missing


# A POSITIVE status reply looks like:  <slug> has status "KernelWorkerStatus.COMPLETE"
# Anything else — a proxy error, a 500, a rate-limit body — is TRANSPORT NOISE and
# must never be read as a kernel verdict. This matters because the CLI prints
# transport failures to stderr, and strings like NewConnectionError / "500 Server
# Error" contain the substring ERROR; the previous `"ERROR" in status` test turned a
# DNS hiccup into "REFUSED: kernel reports terminal failure" and burned the day.
_STATUS_RE = re.compile(r'has status\s+"([^"]+)"', re.IGNORECASE)


def kernel_status(slug: str) -> str | None:
    """The kernel's POSITIVE status, or None if the reply was transport noise."""
    try:
        out = subprocess.run(
            ["python3", "-m", "kaggle", "kernels", "status", slug],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f"kernel status: transport failure ({exc!r}) — treating as unreadable")
        return None
    combined = (out.stdout + out.stderr).strip()
    match = _STATUS_RE.search(combined)
    if not match:
        log(f"kernel status: no positive status in reply — treating as unreadable: {combined[:160]}")
        return None
    return match.group(1).strip()


def wait_for_complete(slug: str, max_wait_s: int = 6 * 3600,
                      not_after: datetime | None = None) -> None:
    """Block until the kernel positively reads COMPLETE; abort on ERROR/CANCEL.

    Only a POSITIVE status line can end this loop. Transport noise is retried, so a
    network blip at fire time costs a minute, not the slot.

    `not_after` bounds the wait against a caller's pre-registered window: without it
    a kernel stuck in RUNNING could keep this loop alive for six hours and submit
    long outside the window the caller promised to fire in.
    """
    deadline = time.time() + max_wait_s
    attempt = 0
    while time.time() < deadline:
        if not_after is not None and datetime.now(timezone.utc) >= not_after:
            raise SystemExit(
                f"REFUSED: kernel not COMPLETE before the caller's window closed at "
                f"{not_after:%Y-%m-%d %H:%M}Z"
            )
        attempt += 1
        status = kernel_status(slug)
        log(f"kernel status (attempt {attempt}): {status or '<unreadable>'}")
        if status is not None:
            upper = status.upper()
            if "COMPLETE" in upper:
                return
            if "ERROR" in upper or "CANCEL" in upper:
                raise SystemExit(f"REFUSED: kernel reports terminal failure: {status}")
        # unreadable or still running -> retry; an unknown is never a failure
        time.sleep(60)
    raise SystemExit(f"REFUSED: kernel never read COMPLETE within {max_wait_s}s")


# ------------------------------------------------------------------- submissions api


def _kaggle_auth_header() -> str:
    import os

    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (user and key):
        creds = json.loads((Path.home() / ".kaggle/kaggle.json").read_text())
        user, key = creds["username"], creds["key"]
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def _ssl_context() -> "ssl.SSLContext":
    """SSL context that can actually verify api.kaggle.com on macOS framework pythons."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the interpreter default
        return ssl.create_default_context()


def list_submissions(competition: str) -> list[dict]:
    url = f"https://www.kaggle.com/api/v1/competitions/submissions/list/{competition}?page=1"
    req = urllib.request.Request(url, headers={"Authorization": _kaggle_auth_header()})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        rows = json.load(resp)
    return rows if isinstance(rows, list) else []


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def find_our_submission(competition: str, message: str, not_before: datetime) -> dict | None:
    """Newest submission whose description matches `message`, made after we started."""
    key = message.strip()[:120]
    candidates = []
    for row in list_submissions(competition):
        desc = str(row.get("description") or "")
        made = _parse_date(str(row.get("date") or ""))
        if desc.strip()[:120] == key and made and made >= not_before:
            candidates.append((made, row))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def watch_submission(competition: str, message: str, submitted_at: datetime) -> int:
    """Gate 3. Returns exit code: 0 clean, 2 alarm."""
    deadline = time.time() + WATCH_SECONDS
    row = None
    while time.time() < deadline:
        try:
            row = find_our_submission(competition, message, submitted_at)
        except Exception as exc:  # noqa: BLE001 - transient API noise must not kill the watch
            log(f"submissions API unreadable ({exc!r}) — retrying")
            row = row  # keep last good reading
        if row is None:
            log("submission not visible in the list yet")
        else:
            ref = row.get("ref")
            status = str(row.get("status") or "").lower()
            nbytes = row.get("totalBytes")
            score = row.get("publicScore")
            age_s = (datetime.now(timezone.utc) - submitted_at).total_seconds()
            log(f"ref={ref} status={status} totalBytes={nbytes} "
                f"publicScore={score} age={age_s/60:.0f}min")

            try:
                ledger_path = Path(__file__).resolve().parent.parent / "docs/submission-ledger.json"
                if ledger_path.is_file():
                    data = json.loads(ledger_path.read_text())
                    if not any(r.get("submission_id") == ref for r in data.get("submissions", [])):
                        entry = {
                            "submission_id": ref,
                            "date_utc": str(row.get("date")),
                            "status": str(row.get("status")),
                            "public_score": score,
                            "total_bytes": nbytes,
                            "description": message,
                            "notes": "auto-appended by submit_gated.py",
                        }
                        data.setdefault("submissions", []).insert(0, entry)
                        ledger_path.write_text(json.dumps(data, indent=2) + "\n")
                        log(f"ledger row appended for submission {ref}")
            except Exception as exc:  # noqa: BLE001
                log(f"ledger append non-fatal warning: {exc!r}")

            if isinstance(nbytes, int) and 0 < nbytes <= NEVER_PLAYED_MAX_BYTES:
                alarm(
                    f"submission {ref} scored file is {nbytes} bytes — the "
                    f"never-played fingerprint (<= {NEVER_PLAYED_MAX_BYTES}; real "
                    "duck runs are 3642-3710). The hidden games were NOT played. "
                    "The slot is burned; diagnose before submitting anything else."
                )
                return 2
            if "complete" in status and age_s < FAST_COMPLETE_SECONDS:
                alarm(
                    f"submission {ref} reached COMPLETE after only {age_s/60:.0f} "
                    "minutes — a real run takes ~9 h. This is the push->submit race "
                    "signature (games never played)."
                )
                return 2
            if "error" in status:
                alarm(f"submission {ref} status ERROR — investigate the kernel log.")
                return 2
            if "complete" in status:
                log("COMPLETE after the 1-hour bar with a plausible byte size — "
                    "unusual (a full run takes ~9 h) but not the never-played "
                    "signature; verify the score manually.")
                return 0
        time.sleep(WATCH_POLL_SECONDS)
    log(f"watch window ({WATCH_SECONDS//60} min) passed with no alarm — the run is "
        "still in flight, which is exactly what a genuine ~9 h run looks like.")
    return 0


# ------------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True,
                        help="kernel slug, e.g. ahmedmobasher86/arc-agi-3-duck-patched")
    parser.add_argument("--version", required=True, type=int,
                        help="kernel version to submit (must be the kernel's latest; "
                             "`kaggle kernels status` reports the latest run)")
    parser.add_argument("--message", required=True, help="submission description")
    parser.add_argument("--notebook", required=True, type=Path,
                        help="local .ipynb the kernel version was pushed from "
                             "(builder-honesty marker check)")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--file", default="submission.parquet")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every pre-submit gate, then stop WITHOUT submitting")
    parser.add_argument("--not-after", default=None,
                        help="ISO-8601 UTC instant; refuse to keep waiting for COMPLETE "
                             "past it. One-shot runners pass their window end so a kernel "
                             "stuck in RUNNING cannot submit hours outside the window.")
    args = parser.parse_args()

    not_after = None
    if args.not_after:
        not_after = datetime.fromisoformat(args.not_after.replace("Z", "+00:00"))
        if not_after.tzinfo is None:
            not_after = not_after.replace(tzinfo=timezone.utc)

    if not args.notebook.is_file():
        raise SystemExit(f"REFUSED: notebook not found: {args.notebook}")

    # Gate 1 — builder honesty.
    missing = check_builder_honesty(args.message, args.notebook)
    if missing:
        alarm("message names packs whose markers are ABSENT from the notebook "
              "(the v5 'depth-pack' lie): " + "; ".join(missing))
        return 1
    log("gate 1 OK: every pack named in the message has markers in the notebook")

    # Gate 2 — kernel COMPLETE + settle.
    wait_for_complete(args.kernel, not_after=not_after)
    log(f"kernel COMPLETE — settling {SETTLE_SECONDS//60} min before submitting "
        "(a submit racing the version publish scores a never-played parquet)")
    if args.dry_run:
        log("dry-run: skipping the settle sleep")
    else:
        time.sleep(SETTLE_SECONDS)
        recheck = kernel_status(args.kernel)
        # None == transport noise, NOT a regression. Only a positive status that
        # fails to say COMPLETE is a real regression worth refusing on.
        if recheck is None:
            log("post-settle status unreadable (transport) — proceeding on the "
                "pre-settle COMPLETE rather than burning the slot on a blip")
        elif "COMPLETE" not in recheck.upper():
            raise SystemExit(f"REFUSED: status regressed during settle: {recheck}")
    log("gate 2 OK: COMPLETE and settled")

    if args.dry_run:
        log("dry-run: all pre-submit gates passed; NOT submitting")
        return 0

    # Submit.
    submitted_at = datetime.now(timezone.utc)
    cmd = [
        "kaggle", "competitions", "submit", args.competition,
        "-k", args.kernel, "-v", str(args.version),
        "-f", args.file, "-m", args.message,
    ]
    log("submitting: " + " ".join(cmd[:-1]) + " -m <message>")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    log(f"submit rc={proc.returncode}: {(proc.stdout + proc.stderr).strip()[:300]}")
    if proc.returncode != 0:
        alarm(f"kaggle submit failed rc={proc.returncode}")
        return 1

    # Gate 3 — watch for the never-played fingerprint.
    return watch_submission(args.competition, args.message, submitted_at)


if __name__ == "__main__":
    sys.exit(main())
