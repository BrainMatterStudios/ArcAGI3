#!/usr/bin/env python3
"""fetch_results.py — targeted REST pull of the v8-smoke kernel's outputs.

The `kaggle kernels output` CLI hangs on big outputs (stdout.log is large), so
this lists the output files through the REST API and downloads only the ones
named on the command line (default: the two JSON reads).

Usage:
  python3 submission/_v8_smoke/fetch_results.py [--list] [file ...]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from base64 import b64encode
from pathlib import Path

KERNEL_USER = "ahmedmobasher86"
KERNEL_SLUG = os.environ.get("V8_KERNEL_SLUG", "arc3-v8-smoke")
OUT_DIR = Path(__file__).parent / "results"

creds = json.loads(Path(os.path.expanduser("~/.kaggle/kaggle.json")).read_text())
AUTH = "Basic " + b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()


def api(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": AUTH})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def listing() -> dict:
    url = ("https://www.kaggle.com/api/v1/kernels/output"
           f"?user_name={KERNEL_USER}&kernel_slug={KERNEL_SLUG}")
    return json.loads(api(url))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data = listing()
    files = data.get("files") or []
    print(f"{len(files)} output files:")
    for f in files:
        print(f"  {f.get('fileName')}  {f.get('fileSize')} bytes")
    if "--list" in sys.argv:
        return
    wanted = args or ["v8_smoke_results.json", "v8_smoke_telemetry.json"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f.get("fileName")
        if name not in wanted:
            continue
        blob = api(f["url"])
        dest = OUT_DIR / name
        dest.write_bytes(blob)
        print(f"wrote {dest} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
