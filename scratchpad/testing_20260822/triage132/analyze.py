#!/usr/bin/env python3
"""Triage-at-55 false-negative curves on competition-geometry (7920s) sessions.

FN definition (replicated from triage_study.py rules): a session killed at minute T
because it has zero completed levels at T, that WOULD have completed its first level
after T (within the 132-min box) = false negative. FN rate at T = FN / (sessions with
zero levels at T).
"""
import json, os
from statistics import median

OUT = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(OUT, "sessions.json")))
PLAYBOOK = json.load(open(os.path.join(os.path.dirname(OUT), "playbook.json")))

ARCH = {}
for a in ("AVATAR", "CLICK", "MIXED"):
    for g in PLAYBOOK[a]["games"]:
        ARCH[g] = a

BOX = 132.0
THRESH = [40, 45, 50, 55, 60, 70, 80, 90]

def prep(s):
    s = dict(s)
    s["arch"] = ARCH.get(s["game"], "?")
    lo, hi = s["first_lo"], s["first_hi"]
    s["first"] = None if lo is None else (lo + (hi or lo)) / 2 if hi is not None else lo
    # competition-geometry truncation: completions after the 132-min box don't exist
    if s["first"] is not None and s["first"] > BOX:
        s["first"] = None
    return s

def fn_table(rows, label):
    print(f"\n== {label} (n={len(rows)}) ==")
    n_zero_final = sum(1 for r in rows if r["first"] is None)
    print(f"   final zero-first-level sessions: {n_zero_final}/{len(rows)}")
    print(f"   {'T':>3} {'zero@T':>6} {'FN':>3} {'FN%':>6}  (sensitivity lo/hi)")
    out = {}
    for T in THRESH:
        zero = [r for r in rows if r["first"] is None or r["first"] > T]
        fn = [r for r in zero if r["first"] is not None]
        # sensitivity: classify by lo (earliest possible) and hi (latest possible)
        def cnt(key):
            z = [r for r in rows if r[key] is None or r[key] > T or (r["first"] is None)]
            # a session whose completion bound is None but first is not None: keep simple
            zz = [r for r in rows if (r["first"] is None) or ((r[key] if r[key] is not None else r["first"]) > T)]
            f = [r for r in zz if r["first"] is not None]
            return len(zz), len(f)
        zlo, flo = cnt("first_lo")
        zhi, fhi = cnt("first_hi")
        rate = 100 * len(fn) / len(zero) if zero else float("nan")
        out[T] = (len(zero), len(fn), rate)
        print(f"   {T:>3} {len(zero):>6} {len(fn):>3} {rate:6.1f}%  [lo {flo}/{zlo}, hi {fhi}/{zhi}]")
    late = sorted((r["first"], r["game"], r["corpus"]) for r in rows if r["first"] is not None and r["first"] > 40)
    if late:
        print("   completions after minute 40:", [(f"{t:.1f}m", g, c.split(':')[0]) for t, g, c in late])
    return out

sess = [prep(s) for s in S]

w38 = [s for s in sess if s["corpus"] == "w38_shipped"]
ft09 = [s for s in sess if s["corpus"].startswith("w36_ft09")]
smoke = [s for s in sess if s["corpus"].startswith("smoke") and s["box_min"] >= 90]

# integrity check: every levels>0 session must have a first estimate (within box)
for s in sess:
    if s["levels"] > 0 and s["first_lo"] is None and not s["corpus"].startswith("smoke"):
        print("WARN: completed session without timing:", s["corpus"], s["clone"], s["game"], s["notes"])

fn_table(w38, "Qwen3.8 shipped wave 20260815-135107 (DEPLOYMENT-RELEVANT, 132-min boxes)")
fn_table(ft09, "Qwen3.6 ft09-ablation waves (132-min boxes, single game ft09/CLICK, patched arms)")
fn_table(w38 + ft09, "POOLED competition-geometry (w38 + ft09 waves)")
fn_table(smoke, "SUPPLEMENT: duck38-v12 smoke corpus, 4h-class boxes truncated at 132 (different bundle)")
fn_table(w38 + ft09 + smoke, "POOLED ALL (incl. smoke supplement)")

# ---- archetype-conditioned, deployment corpus (+supplements marked) ----
print("\n\n==== ARCHETYPE-CONDITIONED ====")
for arch in ("AVATAR", "CLICK", "MIXED"):
    fn_table([s for s in w38 if s["arch"] == arch], f"w38 {arch}")
for arch in ("AVATAR", "CLICK", "MIXED"):
    pool = [s for s in w38 + ft09 + smoke if s["arch"] == arch]
    fn_table(pool, f"pooled-all {arch}")

# ---- first-completion distribution per archetype (w38 + supplements) ----
print("\n\n==== first-completion minute distributions ====")
for arch in ("AVATAR", "CLICK", "MIXED"):
    for name, pool in (("w38", w38), ("all", w38 + ft09 + smoke)):
        fs = sorted(s["first"] for s in pool if s["arch"] == arch and s["first"] is not None)
        if fs:
            q = lambda p: fs[min(len(fs) - 1, int(p * (len(fs) - 1) + 0.5))]
            print(f"{arch:6s} {name:4s} n={len(fs):2d} min={fs[0]:5.1f} med={median(fs):5.1f} p90={q(0.9):5.1f} max={fs[-1]:5.1f}  all={[round(f,1) for f in fs]}")

# ---- trigger table + reclaimed minutes ----
print("\n\n==== TRIGGER CANDIDATES: reclaimed worker-minutes per 110-game run ====")
# archetype shares from the 25-game public dev set (playbook membership)
share = {"AVATAR": 15 / 25, "CLICK": 9 / 25, "MIXED": 1 / 25}
N_GAMES = 110

def eval_rule(rule, corpus, label):
    """rule: dict arch->T (None = never kill). Returns (FN_count_expected, reclaimed_worker_min)."""
    tot_fn = 0.0
    tot_zero = 0.0
    reclaimed = 0.0
    detail = []
    for arch, T in rule.items():
        pool = [s for s in corpus if s["arch"] == arch]
        if not pool or T is None:
            detail.append(f"{arch}: no-kill")
            continue
        n_games_arch = N_GAMES * share[arch]
        zero = [r for r in pool if r["first"] is None or r["first"] > T]
        fn = [r for r in zero if r["first"] is not None]
        p_zero = len(zero) / len(pool)
        p_fn = len(fn) / len(pool)
        exp_kills = n_games_arch * p_zero
        exp_fn = n_games_arch * p_fn
        reclaimed += exp_kills * (BOX - T)
        tot_fn += exp_fn
        tot_zero += exp_kills
        detail.append(f"{arch}@{T}: P(zero@T)={p_zero:.2f} P(FN)={p_fn:.2f} kills={exp_kills:.1f} FN={exp_fn:.2f}")
    fn_rate = 100 * tot_fn / tot_zero if tot_zero else 0.0
    print(f"{label}: reclaimed={reclaimed:.0f} worker-min ({reclaimed/60:.1f} h; {reclaimed/60/28:.2f} h/worker) "
          f"expected kills={tot_zero:.1f} expected FN={tot_fn:.2f} (FN rate among kills {fn_rate:.1f}%)")
    for d in detail:
        print("   ", d)

flat55 = {a: 55 for a in share}
flat60 = {a: 60 for a in share}
flat70 = {a: 70 for a in share}
flat90 = {a: 90 for a in share}
arche = {"AVATAR": 60, "CLICK": 90, "MIXED": 60}
for corpus, cname in ((w38, "w38 only"), (w38 + ft09 + smoke, "pooled-all")):
    print(f"\n-- calibrated on {cname} --")
    eval_rule(flat55, corpus, "FLAT kill@55")
    eval_rule(flat60, corpus, "FLAT kill@60")
    eval_rule(flat70, corpus, "FLAT kill@70")
    eval_rule(flat90, corpus, "FLAT kill@90")
    eval_rule(arche, corpus, "ARCHETYPE AVATAR@60 CLICK@90 MIXED@60")

# score lost by w38 FNs at 55
print("\nw38 sessions that a flat-55 kill would lose (score from rows):")
raw = json.load(open("/Users/ahmed/Documents/ArcAGI3/offkaggle/results/20260815-135107-shipped/patch_closure_result.json"))
scores = {r["clone_id"]: r.get("score") for r in raw["rows"]}
for s in w38:
    if s["first"] is not None and s["first"] > 55:
        print(f"   {s['clone']} {s['game']} ({s['arch']}) first@{s['first']:.1f}m levels={s['levels']} score={scores.get(s['clone'])}")
