"""Direction #1 — real-frame teacher distillation, leave-one-GAME-out CV.

Exp 52 failed on the synthetic->real gap. This trains the genre student on REAL teacher-labelled
frames instead, and tests generalization honestly: hold out each game, train on the other 24,
predict the held-out game's genre. Two conditions:
  real    — train on 24 real games only (the clean #1 bet)
  hybrid  — 24 real games + synthetic (synthetic covers the rare genres SYMMETRY/PUSH/COLLECT,
            which have <=1 real exemplar so leave-one-out can't otherwise see them)
Color-permutation augmentation (relabel the 16 indices per sample — label-preserving on one-hot)
prevents palette overfit.

Honest baselines reported: the 25 public games are genre-imbalanced (MATCH=11/25), so an
always-guess-MATCH baseline already scores ~44% under accept-sets — the model must beat that AND
get non-MATCH genres right.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/genre_distill_logo.py [epochs]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import genre_gate as G
from arcagi3.synthgen import GENRES, GENRE_IDX, generate_dataset

CACHE = Path(__file__).resolve().parent / ".cache" / "all25_frames.npz"

GAME_GENRE = {  # teacher (strong-VLM) labels — primary, for training targets
    "ls20": "NAVIGATE", "su15": "AIM", "m0r0": "SYMMETRY", "sk48": "MATCH", "wa30": "PUSH",
    "re86": "AIM", "tn36": "CLICK", "tr87": "MATCH", "vc33": "CLICK", "cd82": "MATCH",
    "sc25": "MATCH", "lp85": "CLICK", "lf52": "MATCH", "tu93": "NAVIGATE", "ar25": "MATCH",
    "sp80": "NAVIGATE", "bp35": "NAVIGATE", "cn04": "MATCH", "dc22": "MATCH", "ft09": "MATCH",
    "g50t": "NAVIGATE", "ka59": "MATCH", "r11l": "AIM", "s5i5": "AIM", "sb26": "MATCH",
}
ACCEPT = {  # generous accept-sets for genuinely ambiguous / out-of-taxonomy games
    "re86": {"AIM", "MATCH"}, "sk48": {"MATCH", "AIM"}, "tr87": {"MATCH", "CLICK"},
    "cd82": {"MATCH", "PUSH"}, "sc25": {"MATCH", "PUSH"}, "ar25": {"MATCH", "PUSH"},
    "lp85": {"CLICK", "MATCH"}, "lf52": {"CLICK", "MATCH", "PUSH"}, "tu93": {"NAVIGATE", "COLLECT"},
    "sp80": {"NAVIGATE", "AIM"}, "bp35": {"NAVIGATE", "COLLECT", "PUSH"}, "cn04": {"MATCH", "PUSH"},
    "dc22": {"MATCH", "CLICK"}, "ka59": {"MATCH", "PUSH", "NAVIGATE"}, "s5i5": {"AIM", "COLLECT", "PUSH"},
}


def accept_of(g):
    return ACCEPT.get(g, {GAME_GENRE[g]})


def color_aug(X, rng):
    """Relabel the 16 palette indices by a random permutation per sample (one-hot => label-safe)."""
    out = np.empty_like(X)
    for i in range(len(X)):
        perm = rng.permutation(16).astype(np.int8)
        out[i] = perm[np.clip(X[i], 0, 15)]
    return out


def train_aug(model, X, y, dev, epochs, rng):
    yt = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(y); bs = 128
    model.train(True)
    for _ in range(epochs):
        Xa = color_aug(X, rng)            # fresh recolor each epoch
        Xo = G.onehot(Xa)
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(Xo[idx].to(dev)), yt[idx].to(dev))
            loss.backward(); opt.step()


def run(mode, epochs, dev, Xr, gr, syn=None):
    rng = np.random.default_rng(0)
    games = sorted(set(gr))
    hits = 0; rows = []
    for held in games:
        tr_mask = gr != held
        Xtr = Xr[tr_mask]
        ytr = np.array([GENRE_IDX[GAME_GENRE[g]] for g in gr[tr_mask]])
        if mode == "hybrid" and syn is not None:
            Xs, ys = syn
            Xtr = np.concatenate([Xtr, Xs]); ytr = np.concatenate([ytr, ys])
        model = G.GenreCNN(len(GENRES)).to(dev)
        train_aug(model, Xtr, ytr, dev, epochs, rng)
        preds = G.predict(model, Xr[gr == held], dev)
        names = [GENRES[i] for i in preds]
        maj = max(set(names), key=names.count)
        ok = maj in accept_of(held); hits += ok
        rows.append((held, GAME_GENRE[held], maj, ok))
    return hits, rows


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    dev = G.device()
    d = np.load(CACHE)
    Xr = d["X"]; gr = np.array([str(x) for x in d["games"]])
    games = sorted(set(gr))
    base = Counter(GAME_GENRE[g] for g in games)
    print(f"=== Direction #1 — real-frame distillation LOGO-CV (epochs={epochs}, device={dev}) ===")
    print(f"genre balance: {dict(base)}")
    maj_genre = base.most_common(1)[0][0]
    maj_base = sum(maj_genre in accept_of(g) for g in games) / len(games)
    print(f"always-'{maj_genre}' baseline (accept-sets): {maj_base:.0%}\n")

    Xs, ys = generate_dataset(n_per_genre=400, seed=1)
    for mode in ("real", "hybrid"):
        hits, rows = run(mode, epochs, dev, Xr, gr, syn=(Xs, ys))
        print(f"--- {mode.upper()} ---")
        for held, truth, maj, ok in rows:
            print(f"  {held:7}{truth:10}-> {maj:10}{'HIT' if ok else 'miss'}")
        print(f"  LOGO acc: {hits}/{len(rows)} = {hits/len(rows):.0%}   (baseline {maj_base:.0%})\n")


if __name__ == "__main__":
    main()
