"""BET 3 Stage 0 — in-context inference probe DATA pipeline.

An EXAMPLE = (context of (frame, action, next-frame) from random exploration on one episode) +
(query frame) + (affordance label = the rewarding-action CLASS for the query state). The context is
RANDOM exploration (not an oracle demo) and the label is env-truth (the mechanic), so a model that
predicts the held-out-family label from the context is doing INFERENCE (reading the mechanic from
frame-changes), not imitation.

Label space (fixed 5-way): UP/DOWN/LEFT/RIGHT/CLICK. CLICK = the GATE switch affordance; the move
families use the 4 directions. Chance = 1/5 = 0.20.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import (
    MOVES, UP, DOWN, LEFT, RIGHT, build, candidate_actions, rewarding_action,
)

ACTION5 = [UP, DOWN, LEFT, RIGHT, "CLICK"]


def label5(w) -> int:
    a = rewarding_action(w)
    return 4 if a[0] == "C" else ACTION5.index(a)


def _act_idx(a) -> int:
    return 4 if a[0] == "C" else MOVES.index(a)


@dataclass
class Example:
    family: str
    ctx_frames: np.ndarray   # (T,H,W) int8
    ctx_next: np.ndarray     # (T,H,W) int8
    ctx_actions: np.ndarray  # (T,) int 0..4
    query: np.ndarray        # (H,W) int8
    label: int               # 0..4
    world: object = None


def make_example(family, rng, T=12, max_retries=200, return_world=False) -> Example:
    """Random-exploration context of exactly T non-terminal transitions + an unsolved query state."""
    for _ in range(max_retries):
        w = build(family, rng)
        fs, ns, acts = [], [], []
        cur = w.frame()
        ok = True
        for _t in range(T):
            cands = candidate_actions(w)
            a = cands[int(rng.integers(len(cands)))]
            _r, done = w.step(a)
            if done:
                ok = False   # solved mid-context -> query would be degenerate; resample
                break
            nxt = w.frame()
            fs.append(cur)
            ns.append(nxt)
            acts.append(_act_idx(a))
            cur = nxt
        if not ok:
            continue
        if rewarding_action(w) is None:   # PUSH deadlock (block cornered) -> no valid label, resample
            continue
        return Example(family, np.stack(fs), np.stack(ns), np.array(acts, dtype=np.int64),
                       cur.copy(), label5(w), w if return_world else None)
    raise RuntimeError(f"could not build a {T}-step unsolved context for {family}")


def make_dataset(families, n_per_family, rng, T=12):
    return [make_example(f, rng, T=T) for f in families for _ in range(n_per_family)]
