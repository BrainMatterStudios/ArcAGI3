#!/usr/bin/env python3
"""submit_struct_20260811.py — one-shot armed runner for the 2026-08-11 00:01 UTC slot.

Submits duck-patched v9 — the FIRST LIVE PROBE of the structural plan channel
(patches 21+22) — through the sanctioned gated path.

DEFERRED FROM 2026-08-10: the 08-10 slot went to base draw #10 (sub 55392192,
0.92) to complete the n=10 base distribution. v9 remains attested and unfired;
this runner re-targets the same probe at the 08-11 slot. Two deltas from the
08-10 runner:
  * The reading rule below is updated to the n=10 band (mean 0.9650 sd 0.2082).
  * The tracked notebook advanced past v9 on 2026-08-10 (commit 5e434df added
    patch23 bytes, env-gated OFF), so the local-honesty check (gate 2, step 4)
    now pins the v9 blob from git history — commit 8c27e430c403 — instead of
    the working-tree file. The REMOTE bytes submitted are unchanged: the
    version_label=v9 pull must still hash to b48f64c9... and bind
    scriptVersionId 341311881, exactly as attested on 08-09.

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
  The base band is 0.69-1.30 (identical bytes, n=10, mean 0.9650 sd 0.2082). A
  single draw INSIDE that band is NOT evidence in either direction. Only a
  result outside 0.69-1.30 is individually actionable.
  CAVEAT that must travel with the reading: v9 is a patched-family kernel, and
  the patched family's own five draws were 0.76/0.78/0.80/0.93/0.67 (mean
  0.788) — BELOW the base mean. So "in the base band" is not the same as "no
  better than the patched baseline", and a mid-band draw is genuinely
  ambiguous between the two references. This probe cannot resolve that at n=1;
  it is a cheap transfer check, not the certification A/B.

Hard guards, in order, all fatal on failure:
  1. ONE-SHOT WINDOW — refuses to run outside 2026-08-11 00:01-02:00 UTC (after
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
# Extracted from V9_GIT_COMMIT at launch (see extract_v9_notebook) — NOT the
# working tree, which moved on to patch23 bytes after the v9 attestation.
LOCAL_NOTEBOOK = REPO / "logs/duck-patched-v9-8c27e430.ipynb"
COMPETITION = "arc-prize-2026-arc-agi-3"
TARGET_UTC = datetime(2026, 8, 11, 0, 1, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2026, 8, 11, 2, 0, 0, tzinfo=timezone.utc)
# Idempotency claim. Checked at start, written once the slot is ours. Survives a
# relaunch after a presumed death, which the race guard alone cannot (it reads the
# submissions list ~10 min before submit_gated actually submits).
MARKER = REPO / "logs/struct_v9_20260811.marker"

# The tracked notebook advanced past v9 on 2026-08-10 (patch23, env-gated OFF,
# commit 5e434df) — its hash is now f05260f8..., NOT the attested v9 hash. The
# local-honesty check therefore pins the v9 bytes by their COMMIT, not by the
# working-tree file. Full sha, verified 2026-08-10: this blob's canonical code
# hash equals EXPECTED_HASH (recomputed in this session).
V9_GIT_COMMIT = "8c27e430c403d10e07288013da7b2e82d4581acb"

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
    "READING RULE, pre-registered: base band is 0.69-1.30 (identical bytes, n=10, "
    "mean 0.9650 sd 0.2082) — only a draw OUTSIDE that band is individually "
    "actionable; the patched family's own draws averaged 0.788 (n=5, heterogeneous "
    "configs), so a mid-band result is ambiguous between the two references and "
    "certifies nothing. SECONDARY, non-binding: totalBytes vs the base null "
    "3676.8+/-21.8. n=1 transfer check, not the 4-6 wave certification A/B. "
    "Deferred from the 08-10 slot (which completed base n=10)."
)


def log(msg: str) -> None:
    print(f"[struct-runner {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def extract_v9_notebook() -> None:
    """Materialize the attested v9 notebook bytes from git history.

    Runs at LAUNCH, before the sleep: a bad git state must abort at arm time,
    not at 00:01 UTC. The extracted file is what gate 2 step 4 and
    submit_gated's builder-honesty gate read.
    """
    blob = subprocess.check_output(
        ["git", "-C", str(REPO), "show",
         f"{V9_GIT_COMMIT}:submission/_duck_patched/duck-patched.ipynb"])
    nb = json.loads(blob)
    got = canonical_code_hash(nb)
    if got != EXPECTED_HASH:
        raise SystemExit(
            f"ABORT: git blob {V9_GIT_COMMIT[:12]} hashes to {got[:16]}… "
            f"!= attested {EXPECTED_HASH[:16]}…")
    LOCAL_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_NOTEBOOK.write_bytes(blob)
    log(f"v9 notebook extracted from {V9_GIT_COMMIT[:12]} -> {LOCAL_NOTEBOOK} "
        f"(hash {got[:16]}… verified)")


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

    # 4. The git-pinned v9 notebook (extracted at launch) is those same bytes.
    local_nb = json.loads(LOCAL_NOTEBOOK.read_text())
    if canonical_code_hash(local_nb) != EXPECTED_HASH:
        raise SystemExit("ABORT: extracted v9 notebook no longer matches the attested hash")

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

    if MARKER.exists() and not args.mock:
        raise SystemExit(f"ABORT: marker present ({MARKER}) — this slot already fired")

    # Fail-fast identity source: materialize the attested v9 bytes from git
    # BEFORE the multi-hour wait.
    extract_v9_notebook()

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
