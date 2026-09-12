#!/usr/bin/env python3
"""release_watch.py — daily $0 diff of public ARC-AGI-3 artifacts (plan 2026-09-12, step 1).

Snapshots what the Kaggle API lists for the competition and for the search terms the
top teams' releases would carry, and prints what is NEW since the stored baseline.
Public-artifact sources only; no browser, no scraping.

  python3 scripts/release_watch.py --baseline     # store today's snapshot as the baseline
  python3 scripts/release_watch.py                # diff against the baseline, print new rows
  python3 scripts/release_watch.py --update       # diff, then roll the baseline forward

State: scratchpad/release_watch/baseline.json (untracked). Add Hugging Face / GitHub checks
by hand (network policy): huggingface.co/api/models?search=arc-agi-3 ; github Tufa / NVIDIA /
sonpham-org/arc-3 — the API here covers Kaggle kernels, datasets and models.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "scratchpad" / "release_watch" / "baseline.json"
COMP = "arc-prize-2026-arc-agi-3"
TERMS = ["arc-agi-3", "arc agi 3", "arc3", "flash-next", "flash next", "nvfp4", "duck", "tufa", "nvarc"]


def _run(args: list[str]) -> list[dict]:
    try:
        out = subprocess.run(["kaggle", *args, "-v"], capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"[watch] {' '.join(args)}: {exc!r}", file=sys.stderr)
        return []
    if not out.strip() or out.lstrip().startswith("Next Page Token"):
        out = "\n".join(l for l in out.splitlines() if not l.startswith("Next Page Token"))
    try:
        return list(csv.DictReader(io.StringIO(out)))
    except Exception:  # noqa: BLE001
        return []


def snapshot() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for r in _run(["kernels", "list", "--competition", COMP, "--sort-by", "dateRun", "--page-size", "100"]):
        rows[f"kernel:{r.get('ref')}"] = {"kind": "kernel", "ref": r.get("ref"), "title": r.get("title"),
                                          "author": r.get("author"), "lastRun": r.get("lastRunTime"), "votes": r.get("totalVotes")}
    for term in TERMS:
        for r in _run(["kernels", "list", "-s", term, "--sort-by", "dateRun", "--page-size", "50"]):
            rows.setdefault(f"kernel:{r.get('ref')}", {"kind": "kernel", "ref": r.get("ref"), "title": r.get("title"),
                                                       "author": r.get("author"), "lastRun": r.get("lastRunTime"), "votes": r.get("totalVotes")})
        for r in _run(["datasets", "list", "-s", term, "--sort-by", "updated", "--page-size", "50"]):
            rows.setdefault(f"dataset:{r.get('ref')}", {"kind": "dataset", "ref": r.get("ref"), "title": r.get("title"),
                                                        "size": r.get("size"), "lastUpdated": r.get("lastUpdated")})
        for r in _run(["models", "list", "-s", term, "--page-size", "50"]):
            rows.setdefault(f"model:{r.get('ref') or r.get('id')}", {"kind": "model", "ref": r.get("ref") or r.get("id"),
                                                                     "title": r.get("title"), "updated": r.get("lastUpdated") or r.get("updated")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = snapshot()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if a.baseline or not STATE.exists():
        STATE.write_text(json.dumps({"taken": now, "rows": cur}, indent=1))
        print(f"[watch] baseline stored: {len(cur)} rows at {now} -> {STATE}")
        return 0
    base = json.loads(STATE.read_text())
    new = {k: v for k, v in cur.items() if k not in base["rows"]}
    changed = {k: v for k, v in cur.items() if k in base["rows"] and v != base["rows"][k]}
    print(f"[watch] {now}: {len(cur)} rows; baseline {base['taken']} had {len(base['rows'])}; NEW {len(new)}, CHANGED {len(changed)}")
    for k, v in sorted(new.items()):
        print("  NEW    ", k, json.dumps(v))
    for k, v in sorted(changed.items()):
        print("  CHANGED", k, json.dumps(v))
    if a.update:
        STATE.write_text(json.dumps({"taken": now, "rows": cur}, indent=1))
        print("[watch] baseline rolled forward")
    return 0


if __name__ == "__main__":
    sys.exit(main())
