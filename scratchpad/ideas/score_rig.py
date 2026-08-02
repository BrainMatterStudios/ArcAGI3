"""Score a rig run offline, and compare two arms.

WHY OFFLINE. The competition arcade hides per-level baselines exactly as a real
submission does, so `GameRun._compute_final_score()` returns 0.0 by construction
(taaf/game.py:391). The harness can never score itself in competition mode. We hold
the baselines locally in environment_files/*/metadata.json — verified current, dated
after the 2026-04-14 scoring change, and cross-checked against values reconstructed
independently from 340 human sessions — so we apply the formula ourselves.

THE FORMULA, mirroring arc_agi.scorecard and taaf/game.py:381-413 exactly:
    per level l (0-indexed):  weight = l+1
                              score  = min(115, (baseline/actions)^2 * 100)
                                       if completed AND actions > 0, else 0
    env score = min( sum(score*weight)/sum(weight),
                     sum(weight over levels that scored > 0)/sum(weight) * 100 )

EVERY JOIN IS ASSERTED. A silent mis-join would score games against the wrong
baselines and produce a confident, wrong answer — the exact failure mode this
campaign has a law about. Nothing here degrades quietly: it raises.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_baselines(root: str = "environment_files") -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for meta in glob.glob(f"{root}/*/*/metadata.json"):
        d = json.loads(Path(meta).read_text())
        gid = d.get("game_id")
        b = d.get("baseline_actions")
        if gid and b:
            out[gid] = list(b)
    if not out:
        raise RuntimeError(f"no baselines found under {root}")
    return out


def env_score(levels_completed: int, actions_per_level: list[int],
              n_levels: int, baselines: list[int]) -> float:
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")
    if len(baselines) < n_levels:
        raise ValueError(f"baselines has {len(baselines)} entries, need {n_levels}")
    total_w = 0
    total_score = 0.0
    max_w = 0
    for i in range(n_levels):
        w = i + 1
        total_w += w
        a = actions_per_level[i] if i < len(actions_per_level) else 0
        completed = i < levels_completed
        s = min(115.0, (baselines[i] / a) ** 2 * 100.0) if (completed and a > 0) else 0.0
        if s > 0:
            max_w += w
        total_score += s * w
    return min(total_score / total_w, max_w / total_w * 100.0)


def score_run(path: Path, baselines: dict[str, list[int]], strict: bool = True):
    d = json.loads(path.read_text())
    if d.get("error"):
        raise RuntimeError(f"{path.name}: run recorded an error at stage={d.get('stage')}:\n{d['error']}")
    reps = d.get("repeats") or []
    if not reps:
        raise RuntimeError(f"{path.name}: no repeats recorded")

    seen_counts = {len(r) for r in reps}
    if strict and len(seen_counts) != 1:
        raise RuntimeError(f"{path.name}: repeats have differing game counts {seen_counts} "
                           "— a corrupted repeat would bias the comparison")

    per_repeat = []
    for ri, rows in enumerate(reps):
        scores = {}
        for r in rows:
            src = r.get("source_game")
            if not src:
                raise RuntimeError(f"{path.name} repeat {ri}: row {r.get('clone_id')} has no source_game; "
                                   "the clone->official join is unavailable, refusing to guess")
            if src not in baselines:
                raise RuntimeError(f"{path.name}: no local baselines for source game {src!r}")
            n = r.get("levels_total")
            if not n:
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} has no levels_total")
            apl = r.get("actions_per_level") or []
            if strict and not r.get("apl_sum_matches_history", True):
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} violates "
                                   "sum(actions_per_level)==len(history); the per-level split is untrustworthy")
            scores[r["clone_id"]] = env_score(int(r.get("levels_completed") or 0), apl, int(n), baselines[src])
        per_repeat.append(scores)
    return d, per_repeat


def summarise(label: str, per_repeat):
    means = [float(np.mean(list(s.values()))) for s in per_repeat if s]
    print(f"{label:10} repeats={len(per_repeat)}  per-repeat mean="
          f"{[round(m, 3) for m in means]}")
    allv = [v for s in per_repeat for v in s.values()]
    print(f"{'':10} pooled mean={np.mean(allv):.4f} sd={np.std(allv, ddof=1):.4f} "
          f"nonzero={sum(1 for v in allv if v > 0)}/{len(allv)}")
    return means, allv


def compare(a_path: Path, b_path: Path, root: str = "environment_files"):
    base = load_baselines(root)
    da, ra = score_run(a_path, base)
    db, rb = score_run(b_path, base)

    for d, nm in ((da, a_path.name), (db, b_path.name)):
        cfg = d.get("config") or {}
        print(f"config[{nm}]: {cfg}")
    ca, cb = (da.get("config") or {}), (db.get("config") or {})
    drift = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    if drift:
        print(f"\n!! CONFIG DIFFERS BETWEEN ARMS: {drift}")
        print("   Anything other than the intended pack difference invalidates the comparison.")
    print()

    ma, va = summarise(f"A:{da.get('label')}", ra)
    mb, vb = summarise(f"B:{db.get('label')}", rb)

    # Paired by game where both arms cover the same clone set.
    common = set.intersection(*[set(s) for s in ra + rb])
    if not common:
        raise RuntimeError("no common games between arms — cannot pair")
    pa = np.array([np.mean([s[g] for s in ra]) for g in sorted(common)])
    pb = np.array([np.mean([s[g] for s in rb]) for g in sorted(common)])
    d_ = pb - pa
    n = len(d_)
    se = d_.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    t = d_.mean() / se if se and se > 0 else float("nan")
    wins = int((d_ > 1e-9).sum())
    losses = int((d_ < -1e-9).sum())
    print(f"\npaired over {n} games (mean over repeats):")
    print(f"  A mean {pa.mean():.4f}   B mean {pb.mean():.4f}   delta {d_.mean():+.4f}")
    print(f"  paired t={t:+.2f}  B better on {wins}, A better on {losses}, tied {n-wins-losses}")
    if wins + losses:
        k = min(wins, losses)
        p = 2 * sum(math.comb(wins + losses, i) for i in range(k + 1)) / 2 ** (wins + losses)
        print(f"  sign test on {wins+losses} discordant games: p={min(1.0,p):.4f}")
    print("\nNOTE: within-arm repeats give the noise floor. A delta smaller than the")
    print("spread of per-repeat means is not a result, whatever the sign.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        b = load_baselines()
        d, r = score_run(Path(sys.argv[1]), b)
        print(f"config: {d.get('config')}")
        summarise(d.get("label", "run"), r)
    elif len(sys.argv) >= 3:
        compare(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        print("usage: score_rig.py <rig_result.json> [<other_rig_result.json>]")
