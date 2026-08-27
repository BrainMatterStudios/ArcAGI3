"""Four decisive, purely-observational tests. No agent, no model, no GPU.

Each returns a single number that rules an idea in or out. They share one data
collection pass because they all need the same thing: random rollouts with frames,
actions and level-completion events.

  T1 AGENCY          Is there an object whose displacement is near-deterministic in
                     the action? If yes, the avatar is identifiable in ~tens of
                     actions and everything downstream can plan in body coordinates.
                     Measured as normalised mutual information I(dx ; a) / H(dx).

  T2 HIDDEN STATE    In a deterministic world, the same (frame, action) yielding two
                     different successors is a PROOF that latent state exists. Count
                     violations per game. This also tells us which games are
                     frame-Markov, which decides whether frame-hash search is sound
                     at all.

  T3 AFFORDANCE      Is "will this click do anything" predictable from local
                     appearance? If AUC is high, 4096 independent click arms collapse
                     to a handful of object-type arms and exploration gets cheap.
                     If AUC ~ 0.5, appearance-based pruning is simply unavailable.

  T4 PRAGNANZ        Do goal states look "tidier"? Human-authored puzzles may end in
                     regular configurations (symmetric, compressible, few regions).
                     If so the goal is partly readable from a single frame with no
                     interaction at all. Compares regularity of frames immediately
                     before a level completion against matched frames from the same
                     level.

All four are falsifiable in the honest direction: a null result kills the idea and
costs minutes.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
logging.disable(logging.CRITICAL)

import numpy as np
from scipy import ndimage

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

STEPS = int(os.environ.get("STEPS", 4000))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
GAMES = [g for g in os.environ.get("GAMES", "").split(",") if g]
_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def frame_np(o):
    a = np.asarray(o.frame)
    return a[-1] if a.ndim == 3 else a


def _reach_game(env):
    """The underlying ARCBaseGame, which carries the ground-truth oracles."""
    for obj in (env, getattr(env, "_wrapper", None), getattr(env, "wrapper", None)):
        if obj is None:
            continue
        for attr in ("_game", "game", "_env"):
            g = getattr(obj, attr, None)
            if g is not None and hasattr(g, "_get_hidden_state"):
                return g
    return None


def _hidden_of(game):
    if game is None:
        return None
    try:
        return np.asarray(game._get_hidden_state()).tobytes()
    except Exception:
        return None


def collect(game_id, arcade, seed, steps=STEPS):
    """One random rollout. Records everything the four tests need."""
    rng = random.Random(seed)
    env = arcade.make(game_id=game_id, scorecard_id=f"probe-{seed}-{game_id[:8]}")
    o = env.reset()
    prev = frame_np(o).copy()
    # Ground-truth oracle. ARCBaseGame exposes the real hidden state, so T2 can be
    # VALIDATED rather than merely asserted: we can ask whether determinism
    # violations actually coincide with hidden state that really moved.
    game = _reach_game(env)
    rec = {"frames": [], "acts": [], "changed": [], "levels": [], "clicks": [], "hidden": []}
    for _ in range(steps):
        if o.state == GameState.WIN:
            break
        if o.state == GameState.GAME_OVER:
            o = env.step(GameAction.RESET)
            prev = frame_np(o).copy()
            continue
        avail = [a for a in (o.available_actions or []) if a != 0]
        if not avail:
            break
        aid = rng.choice(avail)
        if aid == 6:
            y, x = rng.randrange(64), rng.randrange(64)
            o = env.step(GameAction.ACTION6, data={"x": x, "y": y})
            akey = ("C", y, x)
        else:
            o = env.step(GameAction.from_id(aid))
            akey = ("S", aid)
        cur = frame_np(o)
        changed = not np.array_equal(cur, prev)
        rec["frames"].append(cur.copy())
        rec["acts"].append(akey)
        rec["changed"].append(changed)
        rec["levels"].append(int(o.levels_completed))
        rec["hidden"].append(_hidden_of(game))
        if akey[0] == "C":
            rec["clicks"].append((akey[1], akey[2], int(prev[akey[1], akey[2]]), changed))
        prev = cur.copy()
    return rec


# ---------------------------------------------------------------- T1 agency
def t1_agency(rec):
    """Normalised MI between simple-action id and the displacement of each object.

    Objects are tracked crudely by (colour, area) identity between consecutive
    frames -- enough to see whether SOME object moves deterministically with the
    action, which is the question.
    """
    per_obj = defaultdict(lambda: defaultdict(list))
    for i in range(1, len(rec["frames"])):
        if rec["acts"][i][0] != "S":
            continue
        a = rec["acts"][i][1]
        f0, f1 = rec["frames"][i - 1], rec["frames"][i]
        if np.array_equal(f0, f1):
            per_obj[("NULL",)][a].append((0, 0))
            continue
        for col in np.unique(f0):
            if col == 0:
                continue
            l0, n0 = ndimage.label(f0 == col, structure=_S4)
            l1, n1 = ndimage.label(f1 == col, structure=_S4)
            if n0 != n1 or n0 == 0 or n0 > 12:
                continue
            c0 = ndimage.center_of_mass(f0 == col, l0, range(1, n0 + 1))
            c1 = ndimage.center_of_mass(f1 == col, l1, range(1, n1 + 1))
            for k, (p0, p1) in enumerate(zip(c0, c1)):
                d = (round(p1[0] - p0[0]), round(p1[1] - p0[1]))
                per_obj[(int(col), k)][a].append(d)

    best = (0.0, None, 0)
    for obj, bya in per_obj.items():
        if obj == ("NULL",):
            continue
        total = sum(len(v) for v in bya.values())
        if total < 20:
            continue
        # H(d) and H(d|a) over the empirical distribution
        alld = [d for v in bya.values() for d in v]
        hd = _entropy([alld.count(x) for x in set(alld)])
        hda = 0.0
        for a, v in bya.items():
            if not v:
                continue
            hda += (len(v) / total) * _entropy([v.count(x) for x in set(v)])
        nmi = (hd - hda) / hd if hd > 1e-9 else 0.0
        if nmi > best[0]:
            best = (nmi, str(obj), total)
    return {"best_nmi": round(best[0], 3), "object": best[1], "n": best[2]}


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * np.log2(c / tot) for c in counts if c > 0)


# ------------------------------------------------------- T2 hidden state
def t2_violations(rec):
    """Same (frame, action) -> two different successors proves latent state.

    Validated against the engine's own `_get_hidden_state()`, so we can separate
    three cases: hidden state exists and we detect it; hidden state exists and the
    frame hash misses it; no hidden state and we hallucinate one.
    """
    seen = {}
    viol = pairs = 0
    for i in range(1, len(rec["frames"])):
        key = (hash(rec["frames"][i - 1].tobytes()), rec["acts"][i])
        out = hash(rec["frames"][i].tobytes())
        if key in seen:
            pairs += 1
            if seen[key] != out:
                viol += 1
        else:
            seen[key] = out

    hid = [h for h in rec["hidden"] if h is not None]
    truth = {"hidden_distinct": len(set(hid)) if hid else None,
             "hidden_varies": (len(set(hid)) > 1) if hid else None}

    # Silent moves: the frame is byte-identical across a step but hidden state moved.
    silent = 0
    for i in range(1, len(rec["frames"])):
        if rec["hidden"][i] is None or rec["hidden"][i - 1] is None:
            continue
        if np.array_equal(rec["frames"][i], rec["frames"][i - 1]) and rec["hidden"][i] != rec["hidden"][i - 1]:
            silent += 1
    return {"repeat_pairs": pairs, "violations": viol,
            "viol_rate": round(viol / pairs, 4) if pairs else None,
            "silent_hidden_moves": silent, **truth}


# --------------------------------------------------------- T3 affordance
def t3_affordance(rec):
    """AUC for predicting 'this click changes something' from the clicked colour.

    Colour alone is the cheapest possible feature. If even that carries signal, the
    richer feature set the idea calls for will carry more.
    """
    clicks = rec["clicks"]
    if len(clicks) < 100:
        return {"auc": None, "n": len(clicks)}
    by_col = defaultdict(lambda: [0, 0])
    for _, _, col, ch in clicks:
        by_col[col][1] += 1
        if ch:
            by_col[col][0] += 1
    rate = {c: h / n for c, (h, n) in by_col.items() if n >= 5}
    if len(rate) < 2:
        return {"auc": None, "n": len(clicks)}
    pos = [rate.get(c, 0.5) for _, _, c, ch in clicks if ch]
    neg = [rate.get(c, 0.5) for _, _, c, ch in clicks if not ch]
    if not pos or not neg:
        return {"auc": None, "n": len(clicks)}
    wins = sum(1 for p in pos for q in neg if p > q)
    ties = sum(1 for p in pos for q in neg if p == q)
    return {"auc": round((wins + 0.5 * ties) / (len(pos) * len(neg)), 3),
            "n": len(clicks), "colour_rates": {int(k): round(v, 2) for k, v in sorted(rate.items())}}


# ----------------------------------------------------------- T4 pragnanz
def _regularity(f):
    b = np.ascontiguousarray(f, dtype=np.int8).tobytes()
    comp = len(zlib.compress(b, 6))
    sym_h = np.mean(f == f[:, ::-1])
    sym_v = np.mean(f == f[::-1, :])
    n_regions = 0
    for col in np.unique(f):
        _, n = ndimage.label(f == col, structure=_S4)
        n_regions += n
    return {"neg_bytes": -comp, "sym_h": float(sym_h), "sym_v": float(sym_v),
            "neg_regions": -n_regions}


def t4_pragnanz(rec, k=3, sample=60):
    """Regularity just before a level completion vs matched frames from that level."""
    lv = rec["levels"]
    trans = [i for i in range(1, len(lv)) if lv[i] > lv[i - 1]]
    if not trans:
        return {"transitions": 0}
    rng = random.Random(0)
    pre, ctrl = [], []
    for t in trans:
        lo = max(0, t - 1)
        pre.append(_regularity(rec["frames"][lo]))
        # matched controls: same level, same rollout, not adjacent to the transition
        cand = [i for i in range(len(lv)) if lv[i] == lv[t - 1] and abs(i - t) > k]
        for i in rng.sample(cand, min(sample, len(cand))):
            ctrl.append(_regularity(rec["frames"][i]))
    if not ctrl:
        return {"transitions": len(trans)}
    out = {"transitions": len(trans), "n_ctrl": len(ctrl)}
    for key in ("neg_bytes", "sym_h", "sym_v", "neg_regions"):
        p = np.mean([d[key] for d in pre])
        c = np.mean([d[key] for d in ctrl])
        s = np.std([d[key] for d in ctrl]) or 1.0
        out[key] = {"pre": round(float(p), 2), "ctrl": round(float(c), 2),
                    "z": round(float((p - c) / s), 2)}
    return out


def main():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    envs = sorted(arcade.get_environments(), key=lambda e: e.game_id)
    if GAMES:
        envs = [e for e in envs if e.game_id.split("-")[0][:4] in GAMES]
    out = {}
    t0 = time.time()
    print(f"steps={STEPS} seeds={SEEDS} games={len(envs)}", flush=True)
    print(f"{'game':6} {'agencyNMI':>10} {'violRate':>9} {'clickAUC':>9} {'lvTrans':>8}", flush=True)
    for e in envs:
        stem = e.game_id.split("-")[0][:4]
        agg = {"t1": [], "t2": [], "t3": [], "t4": []}
        for s in SEEDS:
            rec = collect(e.game_id, arcade, s)
            agg["t1"].append(t1_agency(rec))
            agg["t2"].append(t2_violations(rec))
            agg["t3"].append(t3_affordance(rec))
            agg["t4"].append(t4_pragnanz(rec))
        out[stem] = agg
        nmi = max((a["best_nmi"] for a in agg["t1"]), default=0)
        vr = [a["viol_rate"] for a in agg["t2"] if a["viol_rate"] is not None]
        auc = [a["auc"] for a in agg["t3"] if a["auc"] is not None]
        tr = sum(a.get("transitions", 0) for a in agg["t4"])
        print(f"{stem:6} {nmi:>10.3f} "
              f"{(str(round(np.mean(vr),4)) if vr else '-'):>9} "
              f"{(str(round(np.mean(auc),3)) if auc else '-'):>9} {tr:>8}", flush=True)
        json.dump(out, open("scratchpad/engineered_stage0/p3b_probe_core.json", "w"), indent=1, default=str)
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
