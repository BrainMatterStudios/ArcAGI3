# src/arcagi3/prune_analysis.py
"""Pure, API-free analysis for the Phase 0a prune-capability oracle probe.

Consumes a captured banked-v6 trajectory (list[Step]) and answers, per completed level:
  Tier-1 (ceilings): how much exploration was OFF the shortest path to the level-up edge.
  Tier-2 (logo_auc): are on-path vs off-path states separable by decision-time features.
numpy only; no live API, no torch. See docs/superpowers/specs/2026-06-21-prune-oracle-probe-design.md
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from . import perception as P

RESET = ("reset",)


@dataclass
class Step:
    idx: int          # action index
    level: int        # levels_completed BEFORE this action
    from_key: bytes   # state acted from (pol.prev_key after decide)
    action: tuple     # ("S", id) | ("C", x, y) | ("reset",)
    reward: float     # levels_after - levels_before for this action
    tier: int         # salience tier of `action` at from_key


def build_edges(steps):
    """Global directed edges from consecutive steps, EXCLUDING reset actions and
    level-up (reward > 0) transitions (those cross into the next level/board).

    Returns (edges, first_seen):
      edges: dict[from_key] -> list[(action, to_key)]   (to_key = next step's from_key)
      first_seen: dict[key] -> idx of first step where it appears as from_key.
    """
    edges = defaultdict(list)
    first_seen = {}
    for i, s in enumerate(steps):
        if s.from_key not in first_seen:
            first_seen[s.from_key] = s.idx
        if i + 1 < len(steps) and s.action != RESET and s.reward <= 0:
            edges[s.from_key].append((s.action, steps[i + 1].from_key))
    return dict(edges), first_seen


@dataclass
class LevelSeg:
    level: int
    start_key: bytes
    target_key: bytes      # from_key of the reward (level-up) step
    member_keys: set       # keys first-seen within [start_idx, end_idx]
    start_idx: int
    end_idx: int           # the reward step idx

    @property
    def actual_actions(self):
        return self.end_idx - self.start_idx


def segment_levels(steps, first_seen):
    """One LevelSeg per level that ENDS in a reward step (a completed level)."""
    level_first_idx = {}
    reward_step = {}
    for s in steps:
        if s.level not in level_first_idx:
            level_first_idx[s.level] = s.idx
        if s.reward > 0 and s.level not in reward_step:
            reward_step[s.level] = s
    by_idx = {s.idx: s for s in steps}
    segs = {}
    for lvl, rs in reward_step.items():
        if lvl not in level_first_idx:
            continue
        start_idx, end_idx = level_first_idx[lvl], rs.idx
        members = {k for k, fi in first_seen.items() if start_idx <= fi <= end_idx}
        segs[lvl] = LevelSeg(lvl, by_idx[start_idx].from_key, rs.from_key,
                             members, start_idx, end_idx)
    return segs


def shortest_path(edges, start, target, allowed):
    """BFS over `edges` restricted to nodes in `allowed`. Returns [start..target] or None."""
    if start == target:
        return [start]
    if start not in allowed or target not in allowed:
        return None
    seen = {start}
    q = deque([(start, [start])])
    while q:
        k, path = q.popleft()
        for _a, nk in edges.get(k, []):
            if nk in seen or nk not in allowed:
                continue
            seen.add(nk)
            if nk == target:
                return path + [nk]
            q.append((nk, path + [nk]))
    return None


def ceilings(seg, path):
    """Tier-1 numbers for one level. path is None when target is unreachable in-level."""
    disc = len(seg.member_keys)
    if path is None:
        return {"discovered_states": disc, "path_states": None, "ceiling_states": None,
                "actual_actions": seg.actual_actions, "path_actions": None,
                "ceiling_actions": None, "reachable": False}
    ps, pa = len(path), len(path) - 1
    return {"discovered_states": disc, "path_states": ps,
            "ceiling_states": round(1 - ps / max(disc, 1), 3),
            "actual_actions": seg.actual_actions, "path_actions": pa,
            "ceiling_actions": round(1 - pa / max(seg.actual_actions, 1), 3),
            "reachable": True}


FEATURE_ORDER = ["n_objects", "n_small", "median_obj_size", "max_obj_size",
                 "n_distinct_colors", "board_fill", "discovery_tier", "n_new_colors"]


def state_features(grid, background, discovery_tier, parent_colors):
    """Features computable ONLY from what is observable at the state's discovery time."""
    objs = P.connected_components(grid, background=background)
    sizes = np.array([o.size for o in objs]) if objs else np.array([0])
    colors = {int(c) for c in np.unique(grid)} - {int(background)}
    n_new = len(colors - set(parent_colors)) if parent_colors is not None else 0
    return {
        "n_objects": float(len(objs)),
        "n_small": float(sum(1 for o in objs if o.size <= 4)),
        "median_obj_size": float(np.median(sizes)),
        "max_obj_size": float(sizes.max()),
        "n_distinct_colors": float(len(colors)),
        "board_fill": float((grid != background).mean()),
        "discovery_tier": float(discovery_tier),
        "n_new_colors": float(n_new),
    }


def feature_vector(f):
    return np.array([f[k] for k in FEATURE_ORDER], dtype=float)


def roc_auc(scores, labels):
    """AUC via the rank (Mann-Whitney U) statistic, with average ranks for ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def logistic_fit(X, y, iters=800, lr=0.2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        w -= lr * Xs.T @ (p - y) / len(y)
    return w, mu, sd


def logistic_score(X, w, mu, sd):
    Xs = np.hstack([(np.asarray(X, dtype=float) - mu) / sd, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Xs @ w))


def logo_auc(per_game):
    """Leave-one-GAME-out: fit on all-but-one game, score held-out, pool, AUC.

    per_game: dict[game] -> (X [n,F] float array, y [n] {0,1} array). Measures TRANSFER
    (does the on-path signal generalize to an unseen game), not per-game memorization.
    """
    games = list(per_game)
    pooled_s, pooled_y = [], []
    for held in games:
        tr = [g for g in games if g != held]
        if not tr:
            continue
        Xtr = np.vstack([per_game[g][0] for g in tr])
        ytr = np.concatenate([per_game[g][1] for g in tr])
        if ytr.sum() == 0 or (ytr == 0).sum() == 0:
            continue
        w, mu, sd = logistic_fit(Xtr, ytr)
        pooled_s.append(logistic_score(per_game[held][0], w, mu, sd))
        pooled_y.append(per_game[held][1])
    if not pooled_s:
        return float("nan")
    return roc_auc(np.concatenate(pooled_s), np.concatenate(pooled_y))


def shuffle_auc(per_game, seed=0):
    """Label-permutation control: same features, labels shuffled WITHIN each game."""
    rng = np.random.default_rng(seed)
    shuffled = {g: (X, rng.permutation(y)) for g, (X, y) in per_game.items()}
    return logo_auc(shuffled)
