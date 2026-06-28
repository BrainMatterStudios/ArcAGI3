from __future__ import annotations

from collections import deque

import numpy as np

from .transfer_explorer import TransferExplorer
from .value_model import ValueModel


class ValueGuidedExplorer(TransferExplorer):
    def __init__(self, *args, enable_value_guidance: bool = True, max_positive_distance: int = 8,
                 train_epochs: int = 50, **kwargs) -> None:
        self.enable_value_guidance = bool(enable_value_guidance)
        self.max_positive_distance = int(max_positive_distance)
        self.train_epochs = int(train_epochs)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.value_model: ValueModel | None = None
        self._mu = None
        self._sigma = None
        self._value_cache: dict[bytes, float | None] = {}

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if (self.enable_value_guidance and self.prev_action is not None
                and not gstate_terminal and not gstate_notplayed
                and levels > self.prev_levels and self.prev_key is not None):
            self._train_from_graph()
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _encode_key(self, key: bytes, depths: dict[bytes, int], reward_distance: dict[bytes, int]):
        node = self.nodes[key]
        dist = reward_distance.get(key)
        reward_seen = 1.0 if any(reward > 0 for _action, (_dst, reward) in node.edges.items()) else 0.0
        reward_known = 0.0 if dist is None else 1.0
        reward_inv_dist = 0.0 if dist is None else 1.0 / max(1.0, float(dist))
        return np.array(
            [
                float(getattr(node, "visits", 0)),
                float(len(getattr(node, "edges", {}))),
                float(len(node.untried_le(9))),
                float(depths.get(key, -1)),
                float(reward_seen),
                reward_known,
                reward_inv_dist,
            ],
            dtype=float,
        )

    def _build_rows(self):
        if self.root_key is None:
            return []
        reverse = {key: [] for key in self.nodes}
        reward_nodes = set()
        for src, node in self.nodes.items():
            for _action, (dst, reward) in node.edges.items():
                reverse.setdefault(dst, []).append(src)
                if reward > 0:
                    reward_nodes.add(dst)

        depths = {self.root_key: 0}
        q = deque([self.root_key])
        while q:
            key = q.popleft()
            node = self.nodes.get(key)
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
        for key in self.nodes:
            dist = reward_distance.get(key)
            label = 1.0 if dist is not None and dist <= self.max_positive_distance else 0.0
            rows.append((key, self._encode_key(key, depths, reward_distance), label))
        return rows

    def _train_from_graph(self):
        rows = self._build_rows()
        pos = [r for r in rows if r[2] > 0.5]
        neg = [r for r in rows if r[2] <= 0.5]
        if not pos or not neg:
            return
        X = np.stack([features for _key, features, _label in rows])
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0.0] = 1.0
        norm_rows = [((features - self._mu) / self._sigma, label) for _key, features, label in rows]
        self.value_model = ValueModel(input_dim=X.shape[1])
        self.value_model.fit(norm_rows, epochs=self.train_epochs)
        self._value_cache = {}

    def _score_key(self, key: bytes):
        if not self.enable_value_guidance or self.value_model is None or self._mu is None or self._sigma is None:
            return None
        if key in self._value_cache:
            return self._value_cache[key]
        rows = self._build_rows()
        row_map = {k: features for k, features, _label in rows}
        features = row_map.get(key)
        if features is None:
            return None
        score = self.value_model.score((features - self._mu) / self._sigma)
        self._value_cache[key] = score
        return score

    def _path_to_frontier(self, start, p):
        base = super()._path_to_frontier(start, p)
        if not self.enable_value_guidance or self.value_model is None or not base:
            return base
        if start not in self.nodes or self.nodes[start].has_untried_le(p):
            return base
        seen = {start}
        q = deque([(start, [])])
        best = base
        best_depth = len(base)
        best_score = -1e18
        while q:
            key, path = q.popleft()
            if len(path) > best_depth:
                break
            node = self.nodes.get(key)
            if node is None:
                continue
            for action, (next_key, _reward) in node.edges.items():
                if next_key in seen:
                    continue
                seen.add(next_key)
                next_path = path + [action]
                next_node = self.nodes.get(next_key)
                if next_node is not None and next_node.has_untried_le(p):
                    score = self._score_key(next_key)
                    if score is not None and len(next_path) == best_depth and score > best_score:
                        best = next_path
                        best_score = score
                else:
                    q.append((next_key, next_path))
        return best
