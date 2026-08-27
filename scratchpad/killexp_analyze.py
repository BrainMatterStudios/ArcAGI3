"""Kill-experiment analysis: (1) no-op click predictor, (2) affordance-language oracle."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.chdir("/Users/ahmed/Documents/ArcAGI3")
DATA = "scratchpad/killexp_data"
GRID = 64


def load(prefix):
    z = np.load(f"{DATA}/{prefix}.npz")
    meta = json.load(open(f"{DATA}/{prefix}.json"))
    return z["base"], z["lab"], z["volatile"], z["rows"], meta


def effect_class(rows):
    """0=no-op 1=local 2=global 3=level-up"""
    nch, far, lv, over, win = rows[:, 2], rows[:, 3], rows[:, 5], rows[:, 6], rows[:, 7]
    cls = np.zeros(len(rows), dtype=np.int32)
    cls[(nch > 0) & (far > 3)] = 2
    cls[(nch > 0) & (far <= 3)] = 1
    cls[(lv > 0) | (win > 0)] = 3
    cls[(nch == 0) & (over > 0)] = 2  # state change with no frame change
    return cls


def features(base, lab, meta, k=5):
    """Per-cell feature matrix + names. Cells in row-major (y,x) order."""
    comps = meta["comps"]
    rank = {int(k_): v for k_, v in meta["rank"].items()}
    csize = np.array([c["size"] for c in comps])
    ccr = np.array([c["cr"] for c in comps])
    ccc = np.array([c["cc"] for c in comps])
    cw = np.array([c["c1"] - c["c0"] + 1 for c in comps])
    ch = np.array([c["r1"] - c["r0"] + 1 for c in comps])
    avatar = meta["avatar"]
    pad = np.full((GRID + 4, GRID + 4), -1, dtype=np.int16)
    pad[2:-2, 2:-2] = base
    rows, names = [], []
    for y in range(GRID):
        for x in range(GRID):
            li = int(lab[y, x])
            f = [int(base[y, x])]
            n3 = pad[y + 1:y + 4, x + 1:x + 4].ravel()
            n5 = pad[y:y + 5, x:x + 5].ravel()
            f += list(n3)
            if k == 5:
                f += list(n5)
            f += [
                int(csize[li]),
                rank.get(int(base[y, x]), 15),
                int(abs(ccr[li] - y) <= 0.5 and abs(ccc[li] - x) <= 0.5),
                int(cw[li]), int(ch[li]),
                int(len(set(int(v) for v in n5 if v >= 0))),
                y, x, min(y, x, GRID - 1 - y, GRID - 1 - x),
                int(abs(y - ccr[li])), int(abs(x - ccc[li])),
            ]
            if avatar:
                f += [int(max(abs(y - avatar[0]), abs(x - avatar[1])))]
            else:
                f += [-1]
            rows.append(f)
    names = ["color"] + [f"n3_{i}" for i in range(9)]
    if k == 5:
        names += [f"n5_{i}" for i in range(25)]
    names += ["comp_size", "color_rank", "is_centroid", "comp_w", "comp_h",
              "n_colors_k5", "y", "x", "border_dist", "dy_cent", "dx_cent", "avatar_dist"]
    return np.array(rows, dtype=np.int32), names


def pr_curve(scores, y_noop):
    """Precision/recall for 'predicted no-op' at every threshold (scores = P(no-op))."""
    order = np.argsort(-scores)
    yl = y_noop[order]
    tp = np.cumsum(yl)
    k = np.arange(1, len(yl) + 1)
    prec = tp / k
    rec = tp / max(y_noop.sum(), 1)
    return prec, rec, scores[order]


def prec_at_recall(prec, rec, target):
    m = rec >= target
    return float(prec[m].max()) if m.any() else float("nan")


def recall_at_prec(prec, rec, target):
    m = prec >= target
    return float(rec[m].max()) if m.any() else 0.0


# ---------------------------------------------------------------- experiment 1
POS = {"y", "x", "border_dist"}


def exp1(prefixes, k=5, holdout=False, drop_pos=False, ntrain=None):
    from sklearn.ensemble import RandomForestClassifier
    X, Y, G = {}, {}, {}
    for p in prefixes:
        base, lab, vol, rows, meta = load(p)
        cls = effect_class(rows)
        # cells are probed in row-major order, same as features()
        f, names = features(base, lab, meta, k=k)
        if drop_pos:
            keep = [i for i, nm in enumerate(names) if nm not in POS]
            f = f[:, keep]
        X[p], Y[p] = f, (cls == 0).astype(np.int32)
        G[p] = cls
    out = {}
    if not holdout:
        for p in prefixes:
            n = len(X[p])
            cut = int(n * 0.6) if ntrain is None else int(ntrain)
            idx = np.random.RandomState(0).permutation(n)
            tr, te = idx[:cut], idx[int(n * 0.6):]
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                         random_state=0, n_jobs=-1)
            clf.fit(X[p][tr], Y[p][tr])
            s = clf.predict_proba(X[p][te])[:, list(clf.classes_).index(1)] \
                if 1 in clf.classes_ else np.ones(len(te))
            prec, rec, _ = pr_curve(s, Y[p][te])
            out[p] = dict(base=float(Y[p][te].mean()), n_test=len(te),
                          live=int((1 - Y[p][te]).sum()),
                          p50=prec_at_recall(prec, rec, 0.5),
                          p80=prec_at_recall(prec, rec, 0.8),
                          r99=recall_at_prec(prec, rec, 0.99),
                          r999=recall_at_prec(prec, rec, 0.999),
                          r100=recall_at_prec(prec, rec, 1.0))
    else:
        for p in prefixes:
            tr_p = [q for q in prefixes if q != p]
            Xtr = np.vstack([X[q] for q in tr_p])
            Ytr = np.concatenate([Y[q] for q in tr_p])
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                         random_state=0, n_jobs=-1)
            clf.fit(Xtr, Ytr)
            s = clf.predict_proba(X[p])[:, list(clf.classes_).index(1)]
            prec, rec, _ = pr_curve(s, Y[p])
            out[p] = dict(base=float(Y[p].mean()), n_test=len(Y[p]),
                          live=int((1 - Y[p]).sum()),
                          p50=prec_at_recall(prec, rec, 0.5),
                          p80=prec_at_recall(prec, rec, 0.8),
                          r99=recall_at_prec(prec, rec, 0.99),
                          r999=recall_at_prec(prec, rec, 0.999),
                          r100=recall_at_prec(prec, rec, 1.0))
    return out


# ---------------------------------------------------------------- experiment 2
def atoms(base, lab, meta):
    """Boolean masks over the 64x64 grid, named. The hypothesis language."""
    comps = meta["comps"]
    rank = {int(k_): v for k_, v in meta["rank"].items()}
    csize = np.array([c["size"] for c in comps])
    ccr = np.array([c["cr"] for c in comps])
    ccc = np.array([c["cc"] for c in comps])
    A = {}
    for c in range(16):
        m = base == c
        if m.any():
            A[f"color=={c}"] = m
    # colour rarity rank <= r
    for r in range(0, 8):
        m = np.zeros_like(base, dtype=bool)
        for c, rk in rank.items():
            if rk <= r:
                m |= (base == c)
        if m.any() and not m.all():
            A[f"rarity_rank<={r}"] = m
    # component size predicates
    size_of = csize[lab]
    for s in sorted(set(int(v) for v in np.unique(size_of)))[:40]:
        A[f"comp_size=={s}"] = size_of == s
    for s in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        m = size_of <= s
        if m.any() and not m.all():
            A[f"comp_size<={s}"] = m
    # centroid of its component
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    A["is_centroid"] = (np.abs(ccr[lab] - yy) <= 0.5) & (np.abs(ccc[lab] - xx) <= 0.5)
    # adjacency to avatar component
    av = meta["avatar"]
    if av:
        ay, ax = int(round(av[0])), int(round(av[1]))
        A["cheb_dist_avatar<=1"] = (np.abs(yy - ay) <= 1) & (np.abs(xx - ax) <= 1)
        A["cheb_dist_avatar<=3"] = (np.abs(yy - ay) <= 3) & (np.abs(xx - ax) <= 3)
        A["same_row_as_avatar"] = yy == ay
        A["same_col_as_avatar"] = xx == ax
    # in row/col occupied by a non-background object
    bg = int(np.bincount(base.ravel() + 1).argmax() - 1)
    obj = base != bg
    A["non_background"] = obj
    A["row_has_object"] = np.repeat(obj.any(axis=1)[:, None], GRID, axis=1)
    A["col_has_object"] = np.repeat(obj.any(axis=0)[None, :], GRID, axis=0)
    return A


def f1(pred, true):
    tp = int((pred & true).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / int(pred.sum())
    rec = tp / int(true.sum())
    return 2 * prec * rec / (prec + rec), prec, rec


def exp2(prefixes):
    res = {}
    for p in prefixes:
        base, lab, vol, rows, meta = load(p)
        cls = effect_class(rows)
        true = np.zeros((GRID, GRID), dtype=bool)
        for i, (y, x) in enumerate(rows[:, :2]):
            true[y, x] = cls[i] != 0
        A = atoms(base, lab, meta)
        keys = list(A)
        best = ("<none>", 0.0, 0.0, 0.0)
        # depth 1
        for kname in keys:
            s = f1(A[kname], true)
            if s[0] > best[1]:
                best = (kname, *s)
        # depth 2 conjunctions
        for i in range(len(keys)):
            ai = A[keys[i]]
            if not ai.any():
                continue
            for j in range(i + 1, len(keys)):
                m = ai & A[keys[j]]
                if not m.any():
                    continue
                s = f1(m, true)
                if s[0] > best[1]:
                    best = (f"{keys[i]} AND {keys[j]}", *s)
        res[p] = dict(n_true=int(true.sum()), rule=best[0], f1=best[1],
                      prec=best[2], rec=best[3], n_atoms=len(keys))
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    prefixes = sys.argv[2:]
    if mode == "exp1":
        r = exp1(prefixes, k=5)
        print(json.dumps(r, indent=1))
    elif mode == "exp1k3":
        print(json.dumps(exp1(prefixes, k=3), indent=1))
    elif mode == "exp1nopos":
        print(json.dumps(exp1(prefixes, k=5, drop_pos=True), indent=1))
    elif mode == "exp1k3nopos":
        print(json.dumps(exp1(prefixes, k=3, drop_pos=True), indent=1))
    elif mode == "exp1lo":
        print(json.dumps(exp1(prefixes, k=5, holdout=True), indent=1))
    elif mode == "exp1lonopos":
        print(json.dumps(exp1(prefixes, k=5, holdout=True, drop_pos=True), indent=1))
    elif mode == "curve":
        res = {}
        for nt in (20, 50, 100, 200, 500, 1000):
            res[nt] = exp1(prefixes, k=5, drop_pos=True, ntrain=nt)
        print(json.dumps(res, indent=1))
    elif mode == "curvepos":
        res = {}
        for nt in (20, 50, 100, 200, 500, 1000):
            res[nt] = exp1(prefixes, k=5, drop_pos=False, ntrain=nt)
        print(json.dumps(res, indent=1))
    elif mode == "exp2":
        print(json.dumps(exp2(prefixes), indent=1))


def exp2b(prefixes):
    """Fairer test: AND, OR, greedy 3-term DNF, + structural diagnosis of the true set."""
    out = {}
    for p in prefixes:
        base, lab, vol, rows, meta = load(p)
        cls = effect_class(rows)
        true = np.zeros((GRID, GRID), dtype=bool)
        for i, (y, x) in enumerate(rows[:, :2]):
            true[y, x] = cls[i] != 0
        A = atoms(base, lab, meta)
        keys = [k for k in A if A[k].any()]
        best_and = ("<none>", 0.0)
        best_or = ("<none>", 0.0)
        for i in range(len(keys)):
            ai = A[keys[i]]
            s = f1(ai, true)
            if s[0] > best_and[1]:
                best_and = (keys[i], s[0])
            for j in range(i + 1, len(keys)):
                sa = f1(ai & A[keys[j]], true)
                if sa[0] > best_and[1]:
                    best_and = (f"{keys[i]} AND {keys[j]}", sa[0])
                so = f1(ai | A[keys[j]], true)
                if so[0] > best_or[1]:
                    best_or = (f"{keys[i]} OR {keys[j]}", so[0])
        # greedy 3-term DNF over AND-pairs (terms restricted to single atoms for cost)
        cur = np.zeros_like(true)
        terms = []
        for _ in range(3):
            bk, bf = None, f1(cur, true)[0]
            for kk in keys:
                s = f1(cur | A[kk], true)[0]
                if s > bf:
                    bf, bk = s, kk
            if bk is None:
                break
            cur = cur | A[bk]
            terms.append(bk)
        dnf3 = (" OR ".join(terms) or "<none>", f1(cur, true)[0])

        # structural diagnosis
        comps = meta["comps"]
        bg = int(np.bincount(base.ravel() + 1).argmax() - 1)
        ncomp_touched, ncomp_full, ncomp_partial = 0, 0, 0
        for ci in range(len(comps)):
            m = lab == ci
            t = int((true & m).sum())
            if t == 0:
                continue
            ncomp_touched += 1
            if t == int(m.sum()):
                ncomp_full += 1
            else:
                ncomp_partial += 1
        diag = dict(
            n_true=int(true.sum()),
            frac_true_nonbg=round(float((true & (base != bg)).sum()) / max(int(true.sum()), 1), 3),
            comps_touched=ncomp_touched, comps_full=ncomp_full, comps_partial=ncomp_partial,
            union_of_components=(ncomp_partial == 0),
            colors_of_true=sorted({int(c) for c in np.unique(base[true])}) if true.any() else [],
            n_true_colors=len({int(c) for c in np.unique(base[true])}) if true.any() else 0,
        )
        out[p] = dict(best_and=best_and, best_or=best_or, dnf3=dnf3, diag=diag)
    return out


def exp1_group(prefixes, k=5, drop_pos=True, holdout=False):
    """Honest within-game test: split by CONNECTED COMPONENT, not by cell.

    Cells of one object are near-duplicates; a random cell split lets the model
    interpolate inside an already-probed object (which needs no model at all).
    Here train components and test components are disjoint -> measures whether an
    UNPROBED object's clickability is predictable.
    """
    from sklearn.ensemble import RandomForestClassifier
    out = {}
    store = {}
    for p in prefixes:
        base, lab, vol, rows, meta = load(p)
        cls = effect_class(rows)
        f, names = features(base, lab, meta, k=k)
        if drop_pos:
            keep = [i for i, nm in enumerate(names) if nm not in POS]
            f = f[:, keep]
        g = lab.ravel()  # row-major, matches features()
        store[p] = (f, (cls == 0).astype(np.int32), g)
    for p in prefixes:
        f, y, g = store[p]
        if holdout:
            Xtr = np.vstack([store[q][0] for q in prefixes if q != p])
            Ytr = np.concatenate([store[q][1] for q in prefixes if q != p])
            Xte, Yte = f, y
        else:
            comps = np.unique(g)
            rs = np.random.RandomState(0)
            perm = rs.permutation(comps)
            tr_c = set(perm[:int(len(perm) * 0.6)].tolist())
            m = np.array([c in tr_c for c in g])
            if m.all() or (~m).all() or len(np.unique(y[m])) < 2:
                out[p] = dict(base=float(y[~m].mean()) if (~m).any() else 1.0,
                              live=int((1 - y[~m]).sum()) if (~m).any() else 0,
                              p50=float("nan"), r99=0.0, r999=0.0, degenerate=True)
                continue
            Xtr, Ytr, Xte, Yte = f[m], y[m], f[~m], y[~m]
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                     random_state=0, n_jobs=-1)
        clf.fit(Xtr, Ytr)
        s = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        prec, rec, _ = pr_curve(s, Yte)
        out[p] = dict(base=float(Yte.mean()), live=int((1 - Yte).sum()),
                      n_test=int(len(Yte)),
                      p50=prec_at_recall(prec, rec, 0.5),
                      p80=prec_at_recall(prec, rec, 0.8),
                      r99=recall_at_prec(prec, rec, 0.99),
                      r999=recall_at_prec(prec, rec, 0.999),
                      r100=recall_at_prec(prec, rec, 1.0), degenerate=False)
    return out


def prune_curve(prefixes, k=5, drop_pos=True, holdout=False, group=True):
    """Decision-relevant metric: prune_frac (share of click space skipped) vs
    live_lost (share of TRUE affordances thrown away). Precision is misleading at
    these base rates: 99% precision over ~3900 pruned cells can discard every one of
    a game's ~40 live cells."""
    from sklearn.ensemble import RandomForestClassifier
    out, store = {}, {}
    for p in prefixes:
        base, lab, vol, rows, meta = load(p)
        cls = effect_class(rows)
        f, names = features(base, lab, meta, k=k)
        if drop_pos:
            keep = [i for i, nm in enumerate(names) if nm not in POS]
            f = f[:, keep]
        store[p] = (f, (cls == 0).astype(np.int32), lab.ravel())
    for p in prefixes:
        f, y, g = store[p]
        if holdout:
            Xtr = np.vstack([store[q][0] for q in prefixes if q != p])
            Ytr = np.concatenate([store[q][1] for q in prefixes if q != p])
            Xte, Yte = f, y
        else:
            comps = np.unique(g)
            perm = np.random.RandomState(0).permutation(comps)
            tr_c = set(perm[:int(len(perm) * 0.6)].tolist()) if group else None
            m = (np.array([c in tr_c for c in g]) if group else
                 np.zeros(len(y), bool))
            if len(np.unique(y[m])) < 2:
                out[p] = dict(degenerate=True, live=int((1 - y[~m]).sum()))
                continue
            Xtr, Ytr, Xte, Yte = f[m], y[m], f[~m], y[~m]
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                     random_state=0, n_jobs=-1)
        clf.fit(Xtr, Ytr)
        s = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        live = (Yte == 0)
        nlive = int(live.sum())
        order = np.argsort(-s)
        lost = np.cumsum(live[order]) / max(nlive, 1)
        frac = np.arange(1, len(order) + 1) / len(order)
        res = {}
        for tol in (0.0, 0.01, 0.05, 0.10):
            ok = lost <= tol
            res[f"prune@lost<={tol:.2f}"] = float(frac[ok].max()) if ok.any() else 0.0
        res["live"] = nlive
        res["n_test"] = int(len(Yte))
        res["degenerate"] = False
        out[p] = res
    return out
