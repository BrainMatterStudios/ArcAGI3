"""ALIASING DIAGNOSTIC (2026-06-28) — why does in-distribution accuracy COLLAPSE at 6-8 families?

Hypothesis: the collapse is NOT capacity — it's that different mechanic families produce OBSERVATIONALLY
ALIASED contexts under short random exploration (e.g. AVOID vs REACH: avatar wanders near one salient
cell, role-randomized colours, but the rewarding affordance is OPPOSITE — flee vs approach). A single
network then faces a contradictory same-input->opposite-output mapping and averages to chance.

Decisive test: append the TRUE family-id one-hot to the head. If in-dist accuracy JUMPS, the families
are aliased (the mechanic is not in the observable interaction history = W1) and scaling families makes it
WORSE, not better -> convex-hull CLOSED for a fundamental reason. If in-dist stays low even WITH family-id,
it's capacity/optimization instead.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/bet3_aliasing_diag.py [n_per] [epochs] [T]
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn

from arcagi3.synthgen.world import FAMILIES
from arcagi3.synthgen.probe import make_dataset

PAL, HW = 16, 16
FID = {f: i for i, f in enumerate(FAMILIES)}


class Net(nn.Module):
    def __init__(self, use_family_id, width=64):
        super().__init__()
        self.use_fid = use_family_id
        self.emb = nn.Embedding(PAL, 6)
        self.c1 = nn.Conv2d(13, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width // 2, 3, padding=1)
        extra = len(FAMILIES) if use_family_id else 0
        self.head = nn.Sequential(nn.Linear((width // 2) * HW * HW + extra, 128), nn.ReLU(),
                                  nn.Linear(128, 5))

    def forward(self, cf, cn, q, fid):
        B, T = cf.shape[:2]
        qe = self.emb(q).permute(0, 3, 1, 2)
        motion = (cf != cn).float().sum(dim=1, keepdim=True)
        last = self.emb(cn[:, -1]).permute(0, 3, 1, 2)
        x = torch.cat([qe, last, motion], dim=1)
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x)).flatten(1)
        if self.use_fid:
            x = torch.cat([x, fid], dim=1)
        return self.head(x)


def tensors(ds):
    cf = torch.tensor(np.stack([e.ctx_frames for e in ds]), dtype=torch.long)
    cn = torch.tensor(np.stack([e.ctx_next for e in ds]), dtype=torch.long)
    q = torch.tensor(np.stack([e.query for e in ds]), dtype=torch.long)
    y = torch.tensor(np.array([e.label for e in ds]), dtype=torch.long)
    fid = torch.zeros(len(ds), len(FAMILIES))
    for i, e in enumerate(ds):
        fid[i, FID[e.family]] = 1.0
    return cf, cn, q, y, fid


def fit_score(train, val, epochs, use_fid, seed=0, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    net = Net(use_fid)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    cf, cn, q, y, fid = tensors(train)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(cf[idx], cn[idx], q[idx], fid[idx]), y[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        cf, cn, q, y, fid = tensors(val)
        acc = (net(cf, cn, q, fid).argmax(1) == y).float().mean().item()
    return acc


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rng = np.random.default_rng(0)
    train = make_dataset(FAMILIES, n_per, rng, T=T)
    val = make_dataset(FAMILIES, max(n_per // 4, 60), rng, T=T)
    print(f"ALIASING DIAGNOSTIC | all {len(FAMILIES)} families | n_per={n_per} epochs={epochs} chance=0.20\n")
    acc_no = fit_score(train, val, epochs, use_fid=False, seed=0)
    acc_yes = fit_score(train, val, epochs, use_fid=True, seed=0)
    print(f"  in-dist acc WITHOUT family-id = {acc_no:.3f}")
    print(f"  in-dist acc WITH    family-id = {acc_yes:.3f}")
    print(f"  family-id lift = {acc_yes - acc_no:+.3f}\n")
    if acc_yes > acc_no + 0.15 and acc_yes > 0.45:
        print("  => COLLAPSE IS ALIASING: the mechanic is NOT recoverable from the interaction context")
        print("     (different families produce the same context). This IS W1, surfacing as in-dist")
        print("     un-learnability. Scaling families makes aliasing WORSE -> convex-hull CLOSED.")
    elif acc_yes <= acc_no + 0.15:
        print("  => NOT aliasing — family-id doesn't help; the bottleneck is capacity/optimization.")
    else:
        print("  => PARTIAL — some aliasing; re-examine.")


if __name__ == "__main__":
    main()
