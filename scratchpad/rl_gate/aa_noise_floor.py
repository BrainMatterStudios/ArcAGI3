#!/usr/bin/env python3
"""aa_noise_floor.py — measure the A/A replicate noise floor from existing episodes.

Pre-registered thresholds are meaningless until the variance they must exceed is
known. This reads same-config replicate pairs already on disk (no GPU, no new
rollouts), reports per-game RMS in levels and score, and propagates it to the
false-positive rate of each registered gate.

Re-run it whenever new replicate pairs land — n=4 gives a 95% CI on the RMS of
roughly 0.4x to 2.6x, so the estimate tightens cheaply with more pairs.

Usage: .venv/bin/python scratchpad/rl_gate/aa_noise_floor.py [--panel 13] [--rollouts 2]
"""
import argparse, collections, json, math, re, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EPISODES = REPO / "scratchpad/rl_gate/episodes"
PAIR_RE = re.compile(r"^(?P<stem>.+?)_(?P<arm>a|a2|b|b2)$")


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def collect_pairs():
    by = collections.defaultdict(dict)
    for d in sorted(EPISODES.iterdir()):
        man = d / "manifest.json"
        if not man.is_file():
            continue
        m = PAIR_RE.match(d.name)
        if not m:
            continue
        try:
            j = json.loads(man.read_text())
        except Exception:
            continue
        arm = m.group("arm")
        by[(m.group("stem"), arm.rstrip("2"))][2 if arm.endswith("2") else 1] = j
    return {k: v for k, v in by.items() if len(v) == 2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=int, default=13, help="games in the A/B panel")
    ap.add_argument("--rollouts", type=int, default=2, help="paired rollouts per game")
    args = ap.parse_args()

    pairs = collect_pairs()
    if not pairs:
        print("no same-config replicate pairs found")
        return 1

    dl, ds = [], []
    print(f"{'pair':<28}{'Δlevels':>9}{'Δscore':>10}")
    for (stem, arm), v in sorted(pairs.items()):
        a, b = v[1], v[2]
        d_l = (b.get("levels_completed") or 0) - (a.get("levels_completed") or 0)
        d_s = (b.get("final_score") or 0.0) - (a.get("final_score") or 0.0)
        dl.append(d_l); ds.append(d_s)
        print(f"{stem+'_'+arm:<28}{d_l:>9}{d_s:>10.2f}")

    rms = lambda x: (sum(v * v for v in x) / len(x)) ** 0.5
    rl, rs = rms(dl), rms(ds)
    print(f"\nn pairs = {len(dl)}")
    print(f"levels : mean {st.mean(dl):+.3f}  RMS {rl:.3f}")
    print(f"score  : mean {st.mean(ds):+.3f}  RMS {rs:.3f}")
    if abs(st.mean(dl)) > rl / math.sqrt(len(dl)):
        print("  WARNING: A/A mean is not centred on zero — check for an order effect\n"
              "  (replicate 2 always ran second). Randomise arm order before trusting a paired design.")

    sd_total = rl * math.sqrt(args.panel) / math.sqrt(args.rollouts)
    print(f"\npanel={args.panel} games, {args.rollouts} paired rollouts/game")
    print(f"SD of panel total = {sd_total:.2f} levels")
    print(f"{'threshold':>10}{'z':>7}{'1-sided FP':>12}")
    for th in (2, 3, 4, 5):
        z = th / sd_total
        print(f"{'+'+str(th):>10}{z:>7.2f}{norm_sf(z):>11.1%}")
    print(f"\nminimum detectable effect at 2 sigma: +{2*sd_total:.1f} total levels")
    need = (2 * rl * math.sqrt(args.panel) / 2) ** 2
    print(f"rollouts/game for a +2 threshold to reach 2 sigma: {need:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
