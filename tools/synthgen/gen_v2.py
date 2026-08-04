"""v2 CLI orchestrator: sample -> render -> validate (clean + episode) -> manifest.

Usage (from repo root):
    .venv/bin/python tools/synthgen/gen_v2.py --seed 20260804 \
        --out tools/synthgen/out/games_v2

Per game, TWO acceptance tests must pass before it enters the manifest:
  * the v1 suite on the clean reference solution (load/solve/determinism), and
  * the scripted-teacher EPISODE replay (missteps included) reaching WIN with
    exact level boundaries and no deaths (validate_v2.validate_episode).
Exits non-zero if any game fails.
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

from episodes import build_episode  # noqa: E402
from families_v2 import ALL_FAMILIES, sample_game_v2  # noqa: E402
from render_v2 import write_env_dir_v2  # noqa: E402
from validate import validate_game  # noqa: E402
from validate_v2 import validate_episode  # noqa: E402

DEFAULT_COUNTS = {
    "nav": 12, "click": 12, "push": 10,
    "replay": 14, "carry": 14, "mirror": 14, "rules": 14,
}


def generate_batch_v2(
    counts: dict[str, int], seed: int, out_root: Path, ep_seed: int = 0,
    progress: bool = True,
) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for family in ALL_FAMILIES:
        for index in range(counts.get(family, 0)):
            t0 = time.time()
            spec = sample_game_v2(family, index, seed)
            write_env_dir_v2(spec, out_root)
            report = validate_game(out_root, spec)
            episode = build_episode(spec, ep_seed)
            report.update(validate_episode(out_root, spec, episode))
            report["clean_episode"] = episode["clean"]
            report["ok"] = bool(report["ok"] and report["episode_ok"])
            report["gen_seconds"] = round(time.time() - t0, 2)
            reports.append(report)
            if progress:
                status = "OK " if report["ok"] else "FAIL"
                kind = "clean" if episode["clean"] else "revise"
                print(
                    f"[{status}] {spec['game_id']} family={family} "
                    f"levels={len(spec['levels'])} trace={report['trace_len']} "
                    f"episode={report['episode_len']} ({kind}, "
                    f"{report['gen_seconds']}s)",
                    flush=True,
                )
    manifest = out_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(r) + "\n")
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for fam, n in DEFAULT_COUNTS.items():
        ap.add_argument(f"--n-{fam}", type=int, default=n)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--ep-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=_HERE / "out" / "games_v2")
    args = ap.parse_args()

    counts = {fam: getattr(args, f"n_{fam}") for fam in DEFAULT_COUNTS}
    reports = generate_batch_v2(counts, args.seed, args.out, args.ep_seed)
    n_ok = sum(r["ok"] for r in reports)
    print(f"\n{n_ok}/{len(reports)} games valid -> {args.out}")
    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
