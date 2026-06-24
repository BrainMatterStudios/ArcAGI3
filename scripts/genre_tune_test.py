"""Clean generalization test — run the v2 genre model on the TUNE games it was NEVER designed against.

The v2 generator's prototypes were tuned by inspecting HOLDOUT frames, so HOLDOUT accuracy is
contaminated (teaching-to-the-test). The generator has never seen TUNE frames, so this is a clean
transfer test of the priors-from-simulation thesis. TUNE genre labels assigned by inspection
(labeling the test set is fine; the contamination concern is tuning the GENERATOR to it).

Several TUNE games are visibly OUTSIDE the 7-genre taxonomy (tu93 snake, lf52 abstract grid),
so generous accept-sets are used and reported honestly.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/genre_tune_test.py [n] [epochs]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import genre_gate as G  # reuse GenreCNN / train / predict / generate_dataset
from arcagi3.synthgen import GENRES

CACHE = Path(__file__).resolve().parent / ".cache" / "tune_frames.npz"

# Best-guess genre labels for TUNE (inspected once, post-hoc). Accept-sets for the ambiguous /
# out-of-taxonomy games. vc33/lp85 are click-only per prior knowledge.
TUNE_ACCEPT = {
    "vc33": {"CLICK"},
    "cd82": {"MATCH", "PUSH"},
    "sc25": {"MATCH", "PUSH"},
    "lp85": {"CLICK", "MATCH"},
    "lf52": {"CLICK", "MATCH", "PUSH"},
    "tu93": {"NAVIGATE", "COLLECT"},
    "ar25": {"MATCH", "PUSH"},
    "sp80": {"NAVIGATE", "AIM"},
}


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    dev = G.device()
    print(f"=== Clean TUNE generalization test — device={dev} ===\n", flush=True)

    X, y = G.generate_dataset(n_per_genre=n_per, seed=0)
    model = G.GenreCNN(len(GENRES)).to(dev)
    print("[train v2 model on synthetic]", flush=True)
    G.train(model, X, y, dev, epochs)

    d = np.load(CACHE)
    Xt = d["X"]; games = np.array([str(g) for g in d["games"]])
    preds = G.predict(model, Xt, dev)

    print("\n    game   accept-set            pred(majority)  frame-acc result")
    hits = 0; total = 0; fh = 0
    for pre in TUNE_ACCEPT:
        mask = games == pre
        if not mask.any():
            continue
        accept = TUNE_ACCEPT[pre]
        names = [GENRES[i] for i in preds[mask]]
        maj = max(set(names), key=names.count)
        fa = np.mean([nm in accept for nm in names]); fh += sum(nm in accept for nm in names)
        ok = maj in accept; hits += ok; total += 1
        print(f"    {pre:7}{'/'.join(sorted(accept)):22}{maj:16}{fa:<10.0%}{'HIT' if ok else 'miss'}", flush=True)

    print("\n" + "=" * 60)
    print(f"TUNE per-game genre acc (clean, generous accept): {hits}/{total} = {hits/total:.0%}", flush=True)
    print(f"TUNE per-frame acc: {fh/len(preds):.0%}  ({len(preds)} frames)", flush=True)
    print("NOTE: several TUNE games are outside the 7-genre taxonomy (tu93 snake, lf52 abstract),", flush=True)
    print("      so even a perfect perceiver would miss them — interpret with that caveat.", flush=True)


if __name__ == "__main__":
    main()
