"""Raw-pixel CNN prune-probe — the one untested residual from Phase 0a'.

The prune oracle showed hand-engineered features + MLP cannot distinguish productive
(on-shortest-path-to-reward) states from wasted ones at decision time (leave-one-game-out
AUC ~= chance). The residual caveat: a RAW-PIXEL CNN might extract features hand-coding
missed. This tests exactly that — same labels, same leave-one-game-out protocol, but a small
conv net on the 16-channel one-hot 64x64 frame instead of scalar features.

If CNN LOGO-AUC ~= chance too -> the goal-signal wall is closed for learned models as well
(productive states are observationally indistinguishable, full stop). If meaningfully >0.65
-> a real signal hand-features missed, worth a full build.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/cnn_prune_probe.py [budget]
"""
from __future__ import annotations
import logging, sys
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3 import prune_analysis as A
from arcagi3.salience_explorer import SalienceExplorer
import torch, torch.nn as nn

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("cnn"))
GAMES = ["tu93", "vc33", "m0r0", "ls20", "lp85", "cd82"]
torch.manual_seed(0); np.random.seed(0)


def capture(prefix, budget):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["cnn"]))
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset(); steps, grid_by_key = [], {}; n = 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN: break
        grid = P.to_grid(obs.frame); lv = int(obs.levels_completed or 0)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         lv, list(obs.available_actions or []))
        fk = pol.prev_key
        if tok == ("reset",): obs = env.reset()
        elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
        else: obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        la = int(obs.levels_completed or 0)
        if fk is not None:
            steps.append(A.Step(n, lv, fk, tok, float(la - lv), 0))
            grid_by_key.setdefault(fk, grid)
        n += 1
    return steps, grid_by_key


def labels_for_game(steps):
    """Return {key: 1 if on a shortest path to a reward, else 0} over all seen states."""
    edges, first_seen = A.build_edges(steps)
    segs = A.segment_levels(steps, first_seen)          # dict {level: LevelSeg}
    pos = set()
    for lvl, seg in segs.items():
        entries = A.level_entries(steps, seg)
        path, entry = A.best_path(edges, entries, seg.target_key, seg.member_keys)
        if path: pos |= set(path)
    allkeys = set(first_seen)
    return {k: (1 if k in pos else 0) for k in allkeys}


def make_xy(grid_by_key, lab):
    X, y = [], []
    for k, l in lab.items():
        if k not in grid_by_key: continue
        oh = P.encode_onehot(grid_by_key[k])               # already (16,64,64) = (C,H,W)
        X.append(oh.astype(np.float32))
        y.append(l)
    return np.array(X), np.array(y, dtype=np.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1)).squeeze(1)


def fit_and_score(train, test):
    Xtr = np.concatenate([t[0] for t in train]); ytr = np.concatenate([t[1] for t in train])
    Xte, yte = test
    if ytr.sum() == 0 or yte.sum() == 0 or yte.sum() == len(yte): return None
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32)
    m = CNN(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    m.train()
    for _ in range(8):
        idx = np.random.permutation(len(Xtr_t))
        for i in range(0, len(idx), 64):
            b = idx[i:i+64]
            opt.zero_grad(); loss = loss_fn(m(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        sc = torch.sigmoid(m(torch.tensor(Xte))).numpy()
    return A.roc_auc(list(sc), list(yte))


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    import os
    data = {}
    for g in GAMES:
        cache = f"/tmp/cnn_{g}_{budget}.npz"
        if os.path.exists(cache):
            d = np.load(cache); X, y = d["X"], d["y"]
        else:
            steps, grids = capture(g, budget)
            X, y = make_xy(grids, labels_for_game(steps))
            np.savez(cache, X=X, y=y)
        npos = int(y.sum())
        print(f"  {g}: states={len(y)} positives={npos}", flush=True)
        if len(y) > 20 and 0 < npos < len(y):
            data[g] = (X, y)
    print(f"\nleave-one-game-out CNN AUC ({len(data)} games):", flush=True)
    aucs = []
    for held in data:
        train = [data[g] for g in data if g != held]
        a = fit_and_score(train, data[held])
        if a is not None:
            aucs.append(a); print(f"  held={held}: AUC={a:.3f}", flush=True)
    if aucs:
        print(f"\n== mean LOGO CNN AUC = {np.mean(aucs):.3f}  (chance=0.5; hand-feature MLP was ~chance)", flush=True)


if __name__ == "__main__":
    main()
