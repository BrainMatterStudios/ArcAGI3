"""G0 — estimate rho(public, private). Written BEFORE the data exists.

THE QUESTION. Each scored submission plays 55 semi-private games (which become the
public leaderboard) and 55 fully private games (which decide the prize). They are
DIFFERENT games. The whole max-over-draws programme assumes that picking the
highest-scoring public draw also picks a good private score. If run-to-run noise is
predominantly per-GAME rather than per-RUN, that is false: rho ~ 0, selection returns
a random private draw, and best-of-N yields the private MEAN rather than the max.

THE ESTIMATOR, and why it is not a correlation of half-means.

With only a handful of repeats, correlating half-means gives a 4-point correlation --
almost no information. A variance decomposition uses every cell instead. Model each
per-game score in repeat r, game g as

    x[r, g] = mu + run[r] + game[g] + noise[r, g]

`game[g]` is game difficulty. Crucially it is CONSTANT across repeats, so it does not
contribute to run-to-run variance at all -- and because the public and private sets are
disjoint, it never correlates them either. What correlates the two halves is `run[r]`
alone: a run that happened to get more throughput, or a luckier sampling seed, lifts
every game it played.

So for two disjoint halves of 55 games each, across runs:

    Var(half mean) = s2_run + s2_noise / 55
    Cov(halves)    = s2_run
    rho            = s2_run / (s2_run + s2_noise / 55)

Both components come from a two-way decomposition of the repeat x game table.

READ:
  rho >= 0.5   selection works; farming and the variance arm both keep their rationale
  rho ~ 0      selection buys nothing on private; stop spending effort on draw strategy
               and move it all to Tier 0 capability work
  in between   compute the attenuated expected gain and decide on the number

POWER, measured by self-test against known truth (400 sims per cell) BEFORE spending
any GPU. The estimator is approximately unbiased but noisy at low repeat counts:

    rho_true   R=4 mean/p05    R=8 mean/p05    R=16 mean/p05
      0.982    0.944 / 0.790   0.976 / 0.946   0.979 / 0.960
      0.910    0.770 / 0.000   0.850 / 0.570   0.895 / 0.789
      0.462    0.345 / 0.000   0.330 / 0.000   0.384 / 0.000

So a small R resolves only the EXTREMES. Quote the bootstrap interval, never the point
estimate, and treat a middling result as "insufficient power" rather than as evidence
of a middling rho.

HONEST LIMITS, stated up front rather than discovered later:
  * Few repeats means a wide interval on s2_run. This is a first estimate, not a
    settled number, and the bootstrap interval below is the thing to quote.
  * These are the 25 public games cloned, not the real private set. If per-game
    difficulty variance differs there, s2_noise transfers imperfectly.
  * A shortened per-game box understates the achievable score, but rho is a ratio of
    variances and is far less sensitive to that than the mean is.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np


def load(path: Path):
    d = json.load(open(path))
    reps = d.get("repeats") or []
    table = []
    for rows in reps:
        by_game = {}
        for r in rows:
            s = r.get("score")
            if isinstance(s, (int, float)):
                by_game[r.get("game_id")] = float(s)
        table.append(by_game)
    return d, table


def components(table):
    """Two-way decomposition -> (s2_run, s2_noise, matrix, games)."""
    games = sorted(set.intersection(*[set(t) for t in table if t])) if table else []
    if len(table) < 2 or len(games) < 4:
        return None
    M = np.array([[t[g] for g in games] for t in table], dtype=float)  # repeats x games
    grand = M.mean()
    run_eff = M.mean(axis=1) - grand
    game_eff = M.mean(axis=0) - grand
    resid = M - grand - run_eff[:, None] - game_eff[None, :]
    R, G = M.shape
    # Unbiased-ish: subtract the residual contribution baked into the run means.
    s2_noise = float((resid ** 2).sum() / max((R - 1) * (G - 1), 1))
    s2_run = float(max(0.0, (run_eff ** 2).sum() / max(R - 1, 1) - s2_noise / G))
    return s2_run, s2_noise, M, games


def rho_from(s2_run, s2_noise, k=55):
    denom = s2_run + s2_noise / k
    return (s2_run / denom) if denom > 0 else float("nan")


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
                "066c8134-787a-41a1-8dcb-dd4815d4a1d1/scratchpad/g0/rig_result.json")
    d, table = load(path)
    print(f"stage={d.get('stage')} elapsed={d.get('elapsed_s')}s repeats={len(table)}")
    for i, t in enumerate(table):
        v = list(t.values())
        if v:
            print(f"  repeat {i}: n={len(v)} mean={np.mean(v):.4f} sd={np.std(v, ddof=1) if len(v)>1 else 0:.4f}")
    out = components(table)
    if out is None:
        print("\nINSUFFICIENT DATA — need >=2 repeats and >=4 shared games. No verdict.")
        return
    s2_run, s2_noise, M, games = out
    R, G = M.shape
    print(f"\nrepeats={R} games={G}")
    print(f"s2_run   (run-level, correlates the halves) = {s2_run:.6f}")
    print(f"s2_noise (per game-run, does not)           = {s2_noise:.6f}")
    print(f"run-level share of variance                 = {s2_run/(s2_run+s2_noise):.3f}"
          if (s2_run + s2_noise) > 0 else "")

    rho = rho_from(s2_run, s2_noise, 55)
    print(f"\nrho(public55, private55) = {rho:.3f}")

    # Bootstrap over games, which is the dimension we have enough of.
    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        idx = [rng.randrange(G) for _ in range(G)]
        sub = [{games[i]: M[r, i] for i in set(idx)} for r in range(R)]
        o = components(sub)
        if o:
            boots.append(rho_from(o[0], o[1], 55))
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"bootstrap 95% CI over games: [{lo:.3f}, {hi:.3f}]  (n={len(boots)})")

    print("\n--- what this implies for best-of-N ---")
    mu, sd_pub = 0.9288, 0.1947   # measured base distribution, public
    for n in (8, 40, 90):
        from statistics import NormalDist
        z = NormalDist().inv_cdf((n - 0.375) / (n + 0.25))
        print(f"  N={n:>3}: E[max public]={mu + sd_pub*z:.2f}   "
              f"E[private | selected on public]={mu + rho*sd_pub*z:.2f}")
    print("\n(the second column is what actually scores; if rho ~ 0 it stays at the mean")
    print(" no matter how many draws are taken, and farming is free rather than valuable)")

    verdict = ("SELECTION WORKS — farming and the variance arm keep their rationale" if rho >= 0.5
               else "SELECTION BUYS LITTLE — move effort to Tier 0 capability work" if rho < 0.2
               else "PARTIAL — quote the attenuated gain, do not assume the public figure")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
