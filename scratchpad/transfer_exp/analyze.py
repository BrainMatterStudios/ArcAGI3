"""Verdict on the cross-level transfer bet, against the criterion registered in
docs/superpowers/specs/2026-08-01-sacrificial-recon-spine-design.md §3.

PRE-REGISTERED: the carried-knowledge (WARM) arm must beat the wipe-at-boundary
(COLD) arm on TOTAL OFFICIAL SCORE across games -- not on levels reached. Gating on
depth is how the earlier level-reset kill experiment reached a wrong verdict.

Paired by (game, seed): both arms run the same explorer with the same seed and are
identical through level 1, diverging only at the boundary. So the paired difference
is attributable to the carry.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/transfer_exp/transfer_full.json")
rows = json.load(open(path))
n = len(rows)

d = [r["warm_score"] - r["cold_score"] for r in rows]
mean_d = sum(d) / n
sd_d = math.sqrt(sum((x - mean_d) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
se = sd_d / math.sqrt(n) if n > 1 else 0.0
t = mean_d / se if se > 0 else 0.0

wins = sum(1 for x in d if x > 1e-9)
losses = sum(1 for x in d if x < -1e-9)
ties = n - wins - losses

cold_lv = sum(len(r["cold"]["levels_cleared"]) for r in rows)
warm_lv = sum(len(r["warm"]["levels_cleared"]) for r in rows)
cold_sc = sum(r["cold_score"] for r in rows) / n
warm_sc = sum(r["warm_score"] for r in rows) / n

print(f"paired runs: {n}\n")
print(f"{'metric':34} {'COLD':>10} {'WARM':>10} {'delta':>10}")
print(f"{'levels cleared (total)':34} {cold_lv:>10} {warm_lv:>10} {warm_lv-cold_lv:>+10}")
print(f"{'mean official score':34} {cold_sc:>10.3f} {warm_sc:>10.3f} {warm_sc-cold_sc:>+10.3f}")
print(f"\npaired difference (warm - cold): mean {mean_d:+.4f}  sd {sd_d:.4f}  se {se:.4f}  t {t:+.2f}")
print(f"per-run: warm better {wins}, cold better {losses}, tied {ties}")

# Sign test, exact two-sided, on discordant pairs only.
m = wins + losses
if m:
    k = min(wins, losses)
    p = 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)
    print(f"sign test on {m} discordant pairs: p = {min(1.0, p):.4f}")

# Does level-2 search actually get cheaper? The mechanism the bet rests on.
print("\n--- a(2): actions spent on level 2, where both arms cleared level 1 ---")
pairs = [(r["game"], r["seed"], r["cold"]["level_actions"].get("2"), r["warm"]["level_actions"].get("2"))
         for r in rows
         if 1 in r["cold"]["levels_cleared"] and 1 in r["warm"]["levels_cleared"]]
both = [(g, s, c, w) for g, s, c, w in pairs if c and w]
if both:
    ratios = [w / c for _, _, c, w in both]
    better = sum(1 for x in ratios if x < 1)
    print(f"{len(both)} comparable pairs; warm cheaper on {better}, "
          f"median warm/cold ratio {sorted(ratios)[len(ratios)//2]:.2f}")
else:
    print("no comparable pairs -- level 1 rarely cleared in both arms")

# Economic viability, independent of which arm wins.
print("\n--- economic viability: a(l) vs human baseline h(l) ---")
mult = []
for r in rows:
    base = r["baselines"]
    for arm in ("cold", "warm"):
        for lv in r[arm]["levels_cleared"]:
            h = base[lv - 1] if lv - 1 < len(base) else None
            a = r[arm]["level_actions"].get(str(lv)) or r[arm]["level_actions"].get(lv)
            if h and a:
                mult.append(a / h)
if mult:
    mult.sort()
    med = mult[len(mult) // 2]
    print(f"{len(mult)} cleared levels; a/h median {med:.1f}x, "
          f"min {mult[0]:.1f}x, max {mult[-1]:.1f}x")
    print(f"levels cleared at <=3x baseline (the economic zone): "
          f"{sum(1 for x in mult if x <= 3)}/{len(mult)}")
    print(f"score at median: {min(115.0, 100.0/med**2):.2f} per level "
          f"(field leader's whole-run score is 1.86)")

print("\n--- per-game (mean over seeds) ---")
byg = defaultdict(lambda: [0.0, 0.0, 0])
for r in rows:
    e = byg[r["game"]]
    e[0] += r["cold_score"]; e[1] += r["warm_score"]; e[2] += 1
print(f"{'game':6} {'cold':>8} {'warm':>8} {'delta':>8}")
for g in sorted(byg):
    c, w, k = byg[g]
    print(f"{g:6} {c/k:>8.2f} {w/k:>8.2f} {(w-c)/k:>+8.2f}")

print("\n" + "=" * 62)
verdict = "PASS — carry beats wipe on total score" if mean_d > 0 else "FAIL — carry does not beat wipe on total score"
print(f"PRE-REGISTERED CRITERION: {verdict}")
print("=" * 62)
