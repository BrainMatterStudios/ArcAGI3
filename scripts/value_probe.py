"""Offline replay/value separability probe for Phase B RVH No-T."""

from __future__ import annotations

from collections import deque
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.value_model import ValueModel
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from scripts.discovery_bakeoff import make_game


def _nodes_of(graph_like):
    return getattr(graph_like, "nodes", graph_like)


def _untried_count(node) -> int:
    if hasattr(node, "untried"):
        return len(node.untried())
    if hasattr(node, "untried_le"):
        return len(node.untried_le(9))
    return 0


def encode_graph_state(node, depth: int, reward_distance: int | None, reward_seen: float) -> np.ndarray:
    reward_known = 0.0 if reward_distance is None else 1.0
    reward_inv_dist = 0.0 if reward_distance is None or reward_distance < 0 else 1.0 / max(1.0, float(reward_distance))
    return np.array(
        [
            float(getattr(node, "visits", 0)),
            float(len(getattr(node, "edges", {}))),
            float(_untried_count(node)),
            float(depth),
            float(reward_seen),
            reward_known,
            reward_inv_dist,
        ],
        dtype=float,
    )


def build_labeled_rows(graph_like, root: bytes, max_positive_distance: int | None = None):
    nodes = _nodes_of(graph_like)
    reverse = {key: [] for key in nodes}
    reward_nodes = set()
    for src, node in nodes.items():
        for _action, (dst, reward) in node.edges.items():
            reverse.setdefault(dst, []).append(src)
            if reward > 0:
                reward_nodes.add(dst)

    depths = {root: 0}
    q = deque([root])
    while q:
        key = q.popleft()
        node = nodes.get(key)
        if node is None:
            continue
        for _action, (dst, _reward) in node.edges.items():
            if dst not in depths:
                depths[dst] = depths[key] + 1
                q.append(dst)

    reward_distance = {}
    q = deque([(key, 0) for key in reward_nodes])
    for key in reward_nodes:
        reward_distance[key] = 0
    while q:
        key, dist = q.popleft()
        for prev in reverse.get(key, []):
            if prev not in reward_distance:
                reward_distance[prev] = dist + 1
                q.append((prev, dist + 1))

    rows = []
    for key, node in nodes.items():
        depth = depths.get(key, -1)
        dist = reward_distance.get(key)
        label = 1.0 if dist is not None and (
            max_positive_distance is None or dist <= max_positive_distance
        ) else 0.0
        reward_seen = 1.0 if any(reward > 0 for _action, (_dst, reward) in node.edges.items()) else 0.0
        rows.append((key, encode_graph_state(node, depth=depth, reward_distance=dist, reward_seen=reward_seen), label))
    return rows


def probe_game(game_id: str, budget: int, seed: int = 0):
    eng = TransferExplorer(seed=seed)
    eng.reset_all()
    env = make_game(game_id)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = eng.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level,
            available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
            actions += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
            actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    return eng


def fit_and_report(rows):
    if not rows:
        return None
    pos = [r for r in rows if r[2] > 0.5]
    neg = [r for r in rows if r[2] <= 0.5]
    if not pos or not neg:
        return {
            "positives": len(pos),
            "negatives": len(neg),
            "mean_positive": None,
            "mean_negative": None,
            "separable": False,
        }
    X = np.stack([features for _key, features, _label in rows])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    norm_rows = [((features - mu) / sigma, label) for _key, features, label in rows]
    model = ValueModel(input_dim=len(rows[0][1]))
    model.fit(norm_rows, epochs=50)
    pos_scores = [model.score((features - mu) / sigma) for _key, features, label in pos]
    neg_scores = [model.score((features - mu) / sigma) for _key, features, label in neg]
    return {
        "positives": len(pos),
        "negatives": len(neg),
        "mean_positive": float(np.mean(pos_scores)),
        "mean_negative": float(np.mean(neg_scores)),
        "separable": float(np.mean(pos_scores)) > float(np.mean(neg_scores)),
    }


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["tu93", "vc33"]
    for game in games:
        eng = probe_game(game, budget)
        rows = build_labeled_rows(eng, eng.root_key, max_positive_distance=8)
        report = fit_and_report(rows)
        print(
            f"[{game:8}] rows={len(rows)} positives={report['positives']} negatives={report['negatives']} "
            f"mean_pos={report['mean_positive']} mean_neg={report['mean_negative']} separable={report['separable']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
