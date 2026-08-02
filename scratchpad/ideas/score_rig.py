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
    if len(baselines) != n_levels:
        raise ValueError(f"baselines has {len(baselines)} entries, need exactly {n_levels}")
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
    src_of, acts_of = {}, {}
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
            # A game that errored, gave up, or never got a turn has levels_completed=0
            # and would score a clean 0.0 -- indistinguishable from a game that was
            # played and lost. Three dead games in 28 is a ~10% arm-level deficit with
            # no warning, so refuse rather than average them in.
            st = str(r.get("state") or "")
            if strict and st and not any(k in st.lower() for k in ("win", "not_finished", "game_over", "none")):
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} ended in state {st!r} — "
                                   "refusing to score a dead game as 0.0")
            if strict and (r.get("actions_total") or 0) == 0:
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} took 0 actions — "
                                   "never played; refusing to average it in as 0.0")
            if "actions_per_level" not in r:
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} has no actions_per_level; "
                                   "cannot score without the per-level split")
            apl = r.get("actions_per_level") or []
            if strict and "apl_sum_matches_history" not in r:
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} predates the integrity "
                                   "field apl_sum_matches_history — refusing to trust it")
            if strict and not r.get("apl_sum_matches_history", True):
                raise RuntimeError(f"{path.name}: {r.get('clone_id')} violates "
                                   "sum(actions_per_level)==len(history); the per-level split is untrustworthy")
            src_of[r["clone_id"]] = src
            acts_of.setdefault(r["clone_id"], []).append(r.get("actions_total") or 0)
            scores[r["clone_id"]] = env_score(int(r.get("levels_completed") or 0), apl, int(n), baselines[src])
        per_repeat.append(scores)
    d["_src_of"], d["_acts_of"] = src_of, acts_of
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

    ca, cb = (da.get("config") or {}), (db.get("config") or {})
    print(f"config[A]: {ca}\nconfig[B]: {cb}\n")
    drift = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    if drift:
        raise RuntimeError(
            f"CONFIG DIFFERS BETWEEN ARMS: {drift}. Anything other than the intended pack "
            "difference invalidates the comparison — a budget mismatch alone produced a "
            "spurious p<0.0001 'catastrophic regression' in testing.")
    if len(ra) != len(rb):
        raise RuntimeError(f"unequal repeat counts: A={len(ra)} B={len(rb)}. Averaging over "
                           "different repeat counts biases the noisier arm.")

    ma, va = summarise(f"A:{da.get('label')}", ra)
    mb, vb = summarise(f"B:{db.get('label')}", rb)
    print(f"\nwithin-arm repeat spread (the noise floor): A {np.ptp(ma):.4f}  B {np.ptp(mb):.4f}")

    # THROUGHPUT GUARD. Budgets are wall-clock, so an arm that generates more tokens
    # completes fewer actions and scores lower for reasons unrelated to decision
    # quality. This is a first-order confound, not a second-order one.
    aa = float(np.mean([v for vs in (da.get("_acts_of") or {}).values() for v in vs] or [0]))
    ab = float(np.mean([v for vs in (db.get("_acts_of") or {}).values() for v in vs] or [0]))
    rel = abs(ab - aa) / max(aa, 1e-9)
    print(f"mean actions/game: A {aa:.1f}  B {ab:.1f}  relative difference {rel*100:.1f}%")
    if rel > 0.05:
        print("!! THROUGHPUT CONFOUND: the arms did not get comparable action budgets.")
        print("   Any score delta is partly a speed difference. Treat as UNINTERPRETABLE")
        print("   until re-run with an action-based budget.")

    # Clones of the same source game are NOT independent (k000 and k025 are both ar25).
    src = {**(da.get("_src_of") or {}), **(db.get("_src_of") or {})}
    common = sorted(set.intersection(*[set(s_) for s_ in ra + rb]))
    if not common:
        raise RuntimeError("no common games between arms — cannot pair")
    by_src = {}
    for g in common:
        by_src.setdefault(src.get(g, g), []).append(g)
    pa, pb = [], []
    for _s, gs in sorted(by_src.items()):
        pa.append(np.mean([np.mean([r[g] for r in ra]) for g in gs]))
        pb.append(np.mean([np.mean([r[g] for r in rb]) for g in gs]))
    pa, pb = np.array(pa), np.array(pb)
    d_ = pb - pa
    n = len(d_)
    nz = int((np.abs(d_) > 1e-9).sum())
    print(f"\npaired over {n} INDEPENDENT games ({len(common)} clones deduped by source):")
    print(f"  A {pa.mean():.4f}   B {pb.mean():.4f}   delta {d_.mean():+.4f}   nonzero diffs {nz}/{n}")
    if nz == 0:
        print("  identical on every game — the arms are behaviourally the same. STOP.")
        return
    se = d_.std(ddof=1) / math.sqrt(n)
    t = d_.mean() / se if se > 0 else float("nan")
    # exact two-sided permutation test on signs; no normality assumption
    rng = np.random.default_rng(0)
    perm = np.abs(rng.choice([-1.0, 1.0], size=(20000, n)) @ d_ / n)
    p_perm = float((perm >= abs(d_.mean()) - 1e-15).mean())
    wins = int((d_ > 1e-9).sum()); losses = int((d_ < -1e-9).sum())
    k = min(wins, losses)
    p_sign = min(1.0, 2 * sum(math.comb(wins + losses, i) for i in range(k + 1)) / 2 ** (wins + losses))
    print(f"  paired t={t:+.2f} (df={n-1})   permutation p={p_perm:.4f}   sign p={p_sign:.4f}")
    print(f"  B better on {wins}, A better on {losses}, tied {n-wins-losses}")
    if nz <= 4:
        print(f"  !! only {nz} games differ — a t of this size is achievable from "
              "run-to-run nondeterminism alone. Prefer the permutation/sign p.")
    if abs(d_.mean()) < max(np.ptp(ma), np.ptp(mb)):
        print("  !! delta is SMALLER than the within-arm repeat spread. Not a result.")


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
