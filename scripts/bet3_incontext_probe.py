"""BET 3 Stage 0 — held-out-family in-context inference probe (the W1 weak-vs-strong kill-test).

Trains a small in-context model (frame-CNN -> GRU over (frame, action, next-frame) context -> head)
to predict the rewarding-action affordance for a query state, LEAVE-ONE-MECHANIC-FAMILY-OUT. The
context is RANDOM exploration and the label is env-truth, so held-out-family skill = INFERENCE of an
unseen mechanic from interaction history (not imitation).

Pre-registered decision (accuracy units; chance = 1/5 = 0.20):
- KILL (W1-strong confirmed): mean held-out accuracy <= ~0.30 (within noise of chance) OR the
  in-context model does no better than the NO-CONTEXT (query-only) baseline -> the interaction history
  carries no cross-family inference signal -> meta-RL hypothesis class closed.
- PROCEED to RL2 / leave-one-REAL-game-out: mean held-out accuracy >= ~0.55 AND clearly above the
  no-context baseline on >=3/4 folds, including the GATE and PUSH folds (the affordance-novel ones).
- GREY otherwise: report, do not spend on Stage 1 without a second look.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/bet3_incontext_probe.py [n_per_family] [epochs] [T]
       ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/bet3_incontext_probe.py --smoke
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn

from arcagi3.synthgen.world import FAMILIES
from arcagi3.synthgen.probe import make_dataset

PAL = 16
HW = 16


class InContextNet(nn.Module):
    """Spatial conv over [query frame, last context frame, motion map]. The motion map (how often each
    cell changed across the context) is the interaction-history signal that lets the model identify the
    controllable/dynamic cells under role randomization — colour identity carries none. A conv preserves
    the avatar<->goal spatial relation the direction affordance needs. use_context=False zeroes the
    history channels (query-only baseline) -> should sit at chance, since one frame is role-ambiguous."""
    def __init__(self, use_context=True, width=32, head=64):
        super().__init__()
        self.use_context = use_context
        self.emb = nn.Embedding(PAL, 6)
        in_ch = 6 + 6 + 1   # query emb + last-frame emb + motion map
        self.c1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width // 2, 3, padding=1)
        self.head = nn.Sequential(nn.Linear((width // 2) * HW * HW, head), nn.ReLU(),
                                  nn.Linear(head, 5))

    def forward(self, cf, cn, ca, q):
        B, T = cf.shape[:2]
        qe = self.emb(q).permute(0, 3, 1, 2)               # (B,6,H,W)
        if self.use_context:
            motion = (cf != cn).float().sum(dim=1, keepdim=True)        # (B,1,H,W)
            last = self.emb(cn[:, -1]).permute(0, 3, 1, 2)              # (B,6,H,W)
        else:
            motion = torch.zeros(B, 1, HW, HW)
            last = torch.zeros(B, 6, HW, HW)
        x = torch.cat([qe, last, motion], dim=1)
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return self.head(x.flatten(1))


def to_tensors(ds):
    cf = torch.tensor(np.stack([e.ctx_frames for e in ds]), dtype=torch.long)
    cn = torch.tensor(np.stack([e.ctx_next for e in ds]), dtype=torch.long)
    ca = torch.tensor(np.stack([e.ctx_actions for e in ds]), dtype=torch.long)
    q = torch.tensor(np.stack([e.query for e in ds]), dtype=torch.long)
    y = torch.tensor(np.array([e.label for e in ds]), dtype=torch.long)
    return cf, cn, ca, q, y


def fit(train_ds, epochs, use_context, seed=0, bs=128, lr=2e-3, width=32, head=64):
    torch.manual_seed(seed)
    net = InContextNet(use_context=use_context, width=width, head=head)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    cf, cn, ca, q, y = to_tensors(train_ds)
    n = len(y)
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(cf[idx], cn[idx], ca[idx], q[idx]), y[idx])
            loss.backward()
            opt.step()
    return net


def score(net, ds):
    with torch.no_grad():
        cf, cn, ca, q, y = to_tensors(ds)
        pred = net(cf, cn, ca, q).argmax(1)
        acc = (pred == y).float().mean().item()
        maj = max((y == c).float().mean().item() for c in range(5))
    return acc, maj


def score_detail(net, ds):
    with torch.no_grad():
        cf, cn, ca, q, y = to_tensors(ds)
        pred = net(cf, cn, ca, q).argmax(1)
        acc = (pred == y).float().mean().item()
        maj = max((y == c).float().mean().item() for c in range(5))
        move = y < 4
        click = y == 4
        move_acc = (pred[move] == y[move]).float().mean().item() if move.any() else float("nan")
        click_acc = (pred[click] == y[click]).float().mean().item() if click.any() else float("nan")
        click_share = click.float().mean().item()          # fraction of queries whose label is CLICK
        pred_click_rate = (pred == 4).float().mean().item()  # how often the model ever outputs CLICK
    return dict(acc=acc, maj=maj, move_acc=move_acc, click_acc=click_acc,
                click_share=click_share, pred_click_rate=pred_click_rate)


def main():
    if "--smoke" in sys.argv:
        n_per, epochs, T, seeds = 30, 4, 8, 1
    else:
        n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 600
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        T = int(sys.argv[3]) if len(sys.argv) > 3 else 12
        seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    print(f"BET 3 Stage 0 in-context probe | n_per_family={n_per} epochs={epochs} T={T} "
          f"seeds={seeds} chance=0.20\n")
    per_seed = []  # (mean_in, mean_held, mean_noctx)
    fold_acc = {f: [] for f in FAMILIES}
    detail_rows = []
    print(f"  {'seed':>4}  {'held-out':>8}  {'in-dist':>7}  {'held':>6}  {'no-ctx':>6}  "
          f"{'maj':>5}  {'move_acc':>8}  {'click_acc':>9}  {'click%':>6}")
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ins, helds, noctxs = [], [], []
        for held in FAMILIES:
            train_fams = [f for f in FAMILIES if f != held]
            train_ds = make_dataset(train_fams, n_per, rng, T=T)
            val_in = make_dataset(train_fams, max(n_per // 4, 30), rng, T=T)
            test_ds = make_dataset([held], max(n_per // 2, 50), rng, T=T)
            net = fit(train_ds, epochs, use_context=True, seed=seed)
            acc_in, _ = score(net, val_in)
            d = score_detail(net, test_ds)
            net0 = fit(train_ds, epochs, use_context=False, seed=seed)
            acc_noctx, _ = score(net0, test_ds)
            ins.append(acc_in); helds.append(d["acc"]); noctxs.append(acc_noctx)
            fold_acc[held].append(d["acc"])
            detail_rows.append((seed, held, acc_in, d, acc_noctx))
            ca = f"{d['click_acc']:.3f}" if d["click_acc"] == d["click_acc"] else "  n/a"
            print(f"  {seed:>4}  {held:>8}  {acc_in:>7.3f}  {d['acc']:>6.3f}  {acc_noctx:>6.3f}  "
                  f"{d['maj']:>5.3f}  {d['move_acc']:>8.3f}  {ca:>9}  {d['click_share']:>6.2f}")
        per_seed.append((float(np.mean(ins)), float(np.mean(helds)), float(np.mean(noctxs))))
    mean_in = float(np.mean([p[0] for p in per_seed]))
    mean_held = float(np.mean([p[1] for p in per_seed]))
    mean_noctx = float(np.mean([p[2] for p in per_seed]))
    sd_held = float(np.std([p[1] for p in per_seed]))
    lift = mean_held - mean_noctx
    gate = float(np.mean(fold_acc["GATE"]))
    push = float(np.mean(fold_acc["PUSH"]))
    # GATE's CLICK affordance is never in training (move-only families) -> the novel-affordance probe.
    gate_click = [d["click_acc"] for (_s, h, _i, d, _n) in detail_rows if h == "GATE"]
    gate_click = float(np.nanmean(gate_click)) if gate_click else float("nan")
    gate_move = float(np.nanmean([d["move_acc"] for (_s, h, _i, d, _n) in detail_rows if h == "GATE"]))
    print(f"\n  mean IN-DIST val acc = {mean_in:.3f}  (must be >> 0.20 for the probe to be valid)")
    print(f"  mean HELD-OUT acc = {mean_held:.3f} ±{sd_held:.3f}  | no-context = {mean_noctx:.3f}  "
          f"| context lift = {lift:+.3f}")
    print(f"  held-out folds: GATE={gate:.3f}  PUSH={push:.3f}")
    print(f"  GATE novel-affordance split: move-labeled acc={gate_move:.3f}  "
          f"CLICK-labeled acc={gate_click:.3f}  (CLICK is unseen in training)")
    if mean_in < 0.45:
        verdict = ("INCONCLUSIVE — model did not learn the affordance even IN-distribution "
                   f"(val={mean_in:.3f}); enlarge model/data/epochs before reading held-out")
    elif mean_held <= 0.30 or lift <= 0.03:
        verdict = ("KILL — W1-strong confirmed: model learns in-distribution but CANNOT infer a "
                   "held-out mechanic family from interaction history (no cross-family transfer)")
    elif mean_held >= 0.55 and lift > 0.03 and gate >= 0.40 and push >= 0.40:
        verdict = "PROCEED — build Stage 1 RL2 + leave-one-REAL-game-out"
    else:
        verdict = "GREY — partial signal; re-examine before any Stage 1 spend"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
