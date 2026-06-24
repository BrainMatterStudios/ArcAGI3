"""Phase-1 GATE — does genre perception learned from SYNTHETIC games transfer to real HOLDOUT games?

The make-or-break test of the priors-from-simulation thesis (spec
docs/superpowers/specs/2026-06-23-genre-perceiver-program-design.md §5.2):

  1. generate a synthetic multi-genre dataset (arcagi3.synthgen)
  2. train a small CNN genre-classifier on synthetic ONLY (the honest transfer test)
  3. collect real frames for the 8 HOLDOUT public games (live API, cached)
  4. report HOLDOUT genre accuracy — per frame and per game (majority vote)

GO/NO-GO: HOLDOUT per-game top-1 >= 75% (>=6/8). Below that, simulated priors don't transfer
(the StochasticGoose overfit trap) and the program stops here, cheaply.

Real frames are used only to TEST (never to train), so this is a clean transfer measurement.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/genre_gate.py [n_per_genre] [epochs]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv; load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcagi3.synthgen import GENRES, generate_dataset

logging.basicConfig(level=logging.ERROR)
CACHE = Path(__file__).resolve().parent / ".cache" / "holdout_frames.npz"
N_COLORS = 16

# HOLDOUT genre ground truth (Exp 48/49/51 + frame inspection). ACCEPT allows a visually-defensible
# alternative for the genuinely ambiguous ones (re86 has crosshairs AND colour-match markers).
HOLDOUT_GENRE = {
    "ls20": "NAVIGATE", "su15": "AIM", "m0r0": "SYMMETRY", "sk48": "MATCH",
    "wa30": "PUSH", "re86": "AIM", "tn36": "CLICK", "tr87": "MATCH",
}
ACCEPT = {"re86": {"AIM", "MATCH"}, "sk48": {"MATCH", "AIM"}, "tr87": {"MATCH", "CLICK"}}


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def onehot(X):  # (N,64,64) int -> (N,16,64,64) float
    t = torch.from_numpy(np.clip(X, 0, 15).astype(np.int64))
    return F.one_hot(t, N_COLORS).permute(0, 3, 1, 2).float().contiguous()


class GenreCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.c1 = nn.Conv2d(N_COLORS, 32, 3, padding=1); self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1); self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1); self.b3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        x = F.relu(self.b3(self.c3(x)))
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return self.fc(x)


def train(model, X, y, dev, epochs):
    Xo = onehot(X); yt = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(y); bs = 128
    model.train(True)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xo[idx].to(dev); yb = yt[idx].to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:2d}  loss {tot/n:.3f}", flush=True)


@torch.no_grad()
def predict(model, X, dev):
    model.train(False)
    out = model(onehot(X).to(dev))
    return out.argmax(1).cpu().numpy()


def collect_holdout_frames(per_game=10):
    if CACHE.exists():
        d = np.load(CACHE)
        print(f"  loaded cached HOLDOUT frames from {CACHE}", flush=True)
        return d["X"], [str(g) for g in d["games"]]
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from arcagi3 import perception as P
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("g"))
    envs = {e.game_id: e for e in client.get_environments()}
    Xs, games = [], []
    rng = np.random.default_rng(0)
    for prefix in HOLDOUT_GENRE:
        gid = next((g for g in envs if g.startswith(prefix)), None)
        if gid is None:
            print(f"  [{prefix}] not live — skipped", flush=True); continue
        env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["gate"]))
        obs = env.reset()
        for _ in range(per_game):
            if obs is None:
                break
            Xs.append(P.to_grid(obs.frame)); games.append(prefix)
            avail = list(obs.available_actions or [1])
            a = int(rng.choice(avail))
            try:
                if a == 6:
                    obs = env.step(GameAction.ACTION6, data={"x": int(rng.integers(0, 64)), "y": int(rng.integers(0, 64))})
                else:
                    obs = env.step(GameAction.from_id(a))
            except Exception:
                break
        print(f"  [{prefix}] collected", flush=True)
    X = np.stack(Xs).astype(np.int8)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, games=np.array(games))
    return X, games


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    dev = device()
    print(f"=== Phase-1 genre gate — synthetic n/genre={n_per}, epochs={epochs}, device={dev} ===\n", flush=True)

    print("[1] generating synthetic dataset...", flush=True)
    X, y = generate_dataset(n_per_genre=n_per, seed=0)
    ntr = int(len(y) * 0.9)
    print(f"    {len(y)} samples ({len(GENRES)} genres)", flush=True)

    print("[2] training GenreCNN on synthetic only...", flush=True)
    model = GenreCNN(len(GENRES)).to(dev)
    train(model, X[:ntr], y[:ntr], dev, epochs)
    val_acc = (predict(model, X[ntr:], dev) == y[ntr:]).mean()
    print(f"    synthetic val acc: {val_acc:.1%}", flush=True)

    print("[3] collecting real HOLDOUT frames (live, cached)...", flush=True)
    Xh, games = collect_holdout_frames()

    print("[4] HOLDOUT genre transfer:\n", flush=True)
    preds = predict(model, Xh, dev)
    games = np.array(games)
    per_game_hits = 0; per_game_total = 0; frame_hits = 0
    print(f"    {'game':7}{'truth':10}{'pred(majority)':16}{'frame-acc':10}result")
    for prefix in HOLDOUT_GENRE:
        mask = games == prefix
        if not mask.any():
            continue
        truth = HOLDOUT_GENRE[prefix]; accept = ACCEPT.get(prefix, {truth})
        gp = preds[mask]
        names = [GENRES[i] for i in gp]
        maj = max(set(names), key=names.count)
        fa = np.mean([n in accept for n in names])
        frame_hits += sum(n in accept for n in names)
        ok = maj in accept
        per_game_hits += ok; per_game_total += 1
        print(f"    {prefix:7}{truth:10}{maj:16}{fa:<10.0%}{'HIT' if ok else 'miss'}", flush=True)

    print("\n" + "=" * 60)
    pg = per_game_hits / per_game_total if per_game_total else 0
    ff = frame_hits / len(preds)
    print(f"HOLDOUT per-game genre acc: {per_game_hits}/{per_game_total} = {pg:.0%}", flush=True)
    print(f"HOLDOUT per-frame genre acc: {ff:.0%}  ({len(preds)} frames)", flush=True)
    print(f"GATE (>=75% per-game): {'PASS — proceed to Phase 2' if pg >= 0.75 else 'FAIL — priors do not transfer; STOP'}", flush=True)


if __name__ == "__main__":
    main()
