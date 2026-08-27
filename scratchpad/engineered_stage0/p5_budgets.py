"""Stage-0 Pillar 5 — human-replay budget extractor (rebuilt equivalent).

The 2026-08-08 extractor (extract_budgets.py + budgets.json/budgets.md) is NOT
in the repo tree — it lived in a session scratchpad. This script re-derives the
per-game budget table directly from scratchpad/human_replays/extracted/
public_games-dataset/<game>/<uuid>.recording.jsonl (one line per action;
data.levels_completed tracks level-ups).

Metrics per game:
  * actions/completed-level: pooled per-level action counts (actions between
    consecutive levels_completed increments), median + IQR
  * decode budget: actions before the FIRST level-up, replays that completed
    >= 1 level, median + p90

Parity targets (memory arcagi3-human-replay-dataset): median actions/completed
level ft09 = 24, vc33 = 44, ls20 = 91.5.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "scratchpad/human_replays/extracted/public_games-dataset"
OUT = Path(__file__).resolve().parent
FRAME_RE = re.compile(r'"frame":\s*\[.*?\]\s*,\s*"state"', re.DOTALL)

PARITY = {"ft09": 24, "vc33": 44, "ls20": 91.5}


def parse_line(line: str):
    """levels_completed without paying for the frame payload."""
    slim = FRAME_RE.sub('"frame": 0, "state"', line, count=1)
    d = json.loads(slim)["data"]
    return int(d.get("levels_completed") or 0)


def replay_stats(path: Path):
    levels_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                levels_seq.append(parse_line(line))
            except Exception:
                d = json.loads(line)["data"]
                levels_seq.append(int(d.get("levels_completed") or 0))
    if not levels_seq:
        return None
    per_level, decode = [], None
    last_lv, last_idx = 0, 0
    for i, lv in enumerate(levels_seq):
        if lv > last_lv:
            per_level.append(i - last_idx + 1)
            if decode is None:
                decode = i + 1
            last_lv, last_idx = lv, i + 1
    return {"actions": len(levels_seq), "levels": last_lv,
            "per_level": per_level, "decode": decode}


def q(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def main() -> int:
    t0 = time.time()
    table = {}
    games = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    print(f"{'game':6} {'replays':>7} {'lvl_med':>7} {'apl_med':>7} {'apl_iqr':>11} {'decode_med':>10} {'decode_p90':>10}")
    for game in games:
        pooled, decodes, n = [], [], 0
        for rec in sorted((ROOT / game).glob("*.recording.jsonl")):
            r = replay_stats(rec)
            if r is None:
                continue
            n += 1
            pooled.extend(r["per_level"])
            if r["decode"] is not None:
                decodes.append(r["decode"])
        med = st.median(pooled) if pooled else None
        table[game] = {
            "replays": n,
            "apl_median": med, "apl_q25": q(pooled, .25), "apl_q75": q(pooled, .75),
            "decode_median": st.median(decodes) if decodes else None,
            "decode_p90": q(decodes, .90),
            "n_level_completions": len(pooled),
        }
        print(f"{game:6} {n:>7} {len(pooled):>7} {str(med):>7} "
              f"[{table[game]['apl_q25']}-{table[game]['apl_q75']}]".ljust(12)
              + f" {str(table[game]['decode_median']):>10} {str(table[game]['decode_p90']):>10}",
              flush=True)

    ok, checks = True, {}
    for g, ref in PARITY.items():
        got = table.get(g, {}).get("apl_median")
        # parity: within 25% of the recorded median
        hit = got is not None and abs(got - ref) <= 0.25 * ref
        checks[g] = {"ref": ref, "got": got, "ok": hit}
        ok &= hit
    print("\nparity vs memory (apl median):", checks)
    print(f"P5 VERDICT: {'PASS' if ok else 'FAIL'}  elapsed {time.time()-t0:.0f}s")
    json.dump({"table": table, "parity": checks},
              open(OUT / "p5_budgets.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
