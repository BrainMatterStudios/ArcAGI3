"""CLI orchestrator: sample -> render -> validate -> manifest.

Usage (from repo root):
    .venv/bin/python tools/synthgen/gen.py --n-nav 20 --n-click 20 --n-push 10 \
        --seed 20260803 --out tools/synthgen/out/games

Writes one environment_files-style dir per game plus a manifest.jsonl at the
output root summarizing validation for every game. Exits non-zero if any game
fails validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from families import FAMILIES, sample_game  # noqa: E402
from render_game import write_env_dir  # noqa: E402
from validate import validate_game  # noqa: E402


def generate_batch(
    counts: dict[str, int], seed: int, out_root: Path, progress: bool = True
) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for family in FAMILIES:
        for index in range(counts.get(family, 0)):
            t0 = time.time()
            spec = sample_game(family, index, seed)
            write_env_dir(spec, out_root)
            report = validate_game(out_root, spec)
            report["gen_seconds"] = round(time.time() - t0, 2)
            reports.append(report)
            if progress:
                status = "OK " if report["ok"] else "FAIL"
                print(
                    f"[{status}] {spec['game_id']} family={family} levels={len(spec['levels'])} "
                    f"trace={report['trace_len']} ({report['gen_seconds']}s)",
                    flush=True,
                )
    manifest = out_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(r) + "\n")
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-nav", type=int, default=20)
    ap.add_argument("--n-click", type=int, default=20)
    ap.add_argument("--n-push", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", type=Path, default=_HERE / "out" / "games")
    args = ap.parse_args()

    counts = {"nav": args.n_nav, "click": args.n_click, "push": args.n_push}
    reports = generate_batch(counts, args.seed, args.out)
    n_ok = sum(r["ok"] for r in reports)
    print(f"\n{n_ok}/{len(reports)} games valid -> {args.out}")
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
