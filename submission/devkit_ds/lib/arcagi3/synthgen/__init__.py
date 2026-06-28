"""Synthetic multi-genre ARC-AGI-3 frame generator (Phase 1 of the Genre-Perceiver program).

Produces 64x64 palette-index frames labelled by game GENRE, to test whether genre perception
learned from broad *simulated* experience transfers to unseen real games (the priors-from-
simulation thesis). See docs/superpowers/specs/2026-06-23-genre-perceiver-program-design.md.

    from arcagi3.synthgen import generate_dataset, GENRES
    X, y = generate_dataset(n_per_genre=700, seed=0)   # X: (N,64,64) int8, y: (N,) genre idx
"""
from __future__ import annotations

import numpy as np

from .genres import GENRES, generate_one

GENRE_IDX = {g: i for i, g in enumerate(GENRES)}


def generate_dataset(n_per_genre: int = 700, seed: int = 0):
    """Balanced dataset across genres. Returns (X int8 [N,64,64], y int64 [N])."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for g in GENRES:
        for _ in range(n_per_genre):
            grid, genre, _apos, _goal = generate_one(rng, g)
            xs.append(grid)
            ys.append(GENRE_IDX[genre])
    X = np.stack(xs).astype(np.int8)
    y = np.array(ys, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


__all__ = ["generate_dataset", "generate_one", "GENRES", "GENRE_IDX"]
